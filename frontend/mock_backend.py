"""Persistent mock task service. No model, shell, media generation or real HTTP calls."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from pathlib import Path
from uuid import uuid4

from backend_client import BackendError, LABELS, REVIEW_STAGES, STAGE_IDS
from fixtures import CORPUS, demo_translation, new_segments
from routing_explanation import explain_route

# Reuse the exact delivered pure policy. Support the local stage tree and the
# verified GitHub layout (frontend/ next to the existing root policy), without
# silently selecting a different version or creating a source __pycache__.
POLICY_SHA256 = "a1ad84218cd028e5e7c3bf337e1fbff23e6caa2489601b56cffc62e25b9388f7"
_policy_root = Path(__file__).resolve().parent.parent
_policy_paths = (_policy_root / "source/autodub_v2_scheme3_lighttts_final/scheme3_policy.py",
                 _policy_root / "scheme3_policy.py")
_policy_path = next((path for path in _policy_paths if path.is_file()), None)
if _policy_path is None:
    raise FileNotFoundError("缺少 scheme3_policy.py：请把 frontend/ 放在仓库根目录现有规则文件旁。")
_policy_bytes = _policy_path.read_bytes()
# Git on Windows may check out LF as CRLF. Accept only that newline conversion;
# every other content change still requires an explicit new policy version.
if hashlib.sha256(_policy_bytes.replace(b"\r\n", b"\n")).hexdigest() != POLICY_SHA256:
    raise RuntimeError("scheme3_policy.py 版本不匹配：需要已核对的 scheme3_snapshot_20261008，不能静默切换规则。")
_spec = importlib.util.spec_from_file_location("autodub_mock_policy", _policy_path)
assert _spec and _spec.loader
_policy = importlib.util.module_from_spec(_spec)
import sys
sys.modules[_spec.name] = _policy
# Compile the read-only file directly so Python cannot add __pycache__ to the source snapshot.
exec(compile(_policy_bytes.decode("utf-8"), str(_policy_path), "exec"), _policy.__dict__)


def now() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


class MockBackendClient:
    def __init__(self, db_path: str | Path, start_worker: bool = True, interval: float = 1.5):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.worker_error: str | None = None
        self.interval = interval
        with self._connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, data TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS requests (scope TEXT, id TEXT, body TEXT, result TEXT, PRIMARY KEY(scope,id))")
        self.worker = None
        if start_worker:
            self.worker = threading.Thread(target=self._work, daemon=True, name="autodub-mock-worker")
            self.worker.start()

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        try:
            with db:
                yield db
        finally:
            db.close()

    def close(self):
        self.stop_event.set()
        if self.worker:
            self.worker.join(timeout=3)

    def _read(self, db, job_id):
        row = db.execute("SELECT data FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not row:
            raise BackendError("NOT_FOUND", "找不到此模拟任务。", 404)
        job = json.loads(row[0])
        for capability in ["edit_emotion", "trial_tts_models", "adopt_tts_trial"]:
            job["capabilities"].setdefault(capability, True)
        job.setdefault("tts_engines", self._engine_catalog())
        if "emotion" not in job["capabilities"]["rework_stages"]:
            job["capabilities"]["rework_stages"].append("emotion")
        # Compatibility for older mock snapshots: an unstarted result cannot be stale.
        for record in job["stages"]:
            if record["status"] == "stale" and record["attempt"] == 0:
                record["status"] = "pending"
        for segment in job["_segments"]:
            segment.setdefault("tts_trials", [])
            emotion = segment["emotion"]
            emotion.setdefault("origin", "model")
            emotion.setdefault("recognized_label", emotion["label"])
            emotion.setdefault("status", "success" if self._stage(job, "emotion")["attempt"] else "pending")
            tr, tts = segment["translation"], segment["tts"]
            if tr["status"] == "stale" and not tr["text"] and not tr["candidates"]:
                tr["status"] = "pending"
            if tts["status"] == "stale" and not tts.get("planned_text") and not tts.get("audio_artifact_id"):
                tts["status"] = "pending"
            if segment.get("light_tts_status") == "stale" and self._stage(job, "light_tts")["attempt"] == 0:
                segment["light_tts_status"] = "pending"
        if job["status"] == "waiting_review" and not job["active_operation"] and not job.get("pending_edit"):
            edits = [e for e in job["_events"] if e["type"] == "edit.applied"]
            starts = [e["sequence"] for e in job["_events"] if e["type"] == "operation.started"]
            reviews = [e for e in job["_events"] if e["type"] == "stage.review_required"]
            if edits and reviews and edits[-1]["sequence"] > max(starts, default=0):
                job["current_stage"] = reviews[-1]["stage"]
                job["pending_edit"] = {"from_stage": job["resume_from"], "reason": "人工修改"}
        return job

    @staticmethod
    def _engine_catalog():
        return [{"engine": engine, "available": True, "languages": ["zh", "en", "ja"] if engine == "omnivoice" else ["zh", "en"],
                 "mock": True} for engine in ["f5tts", "indextts2", "cosyvoice3", "confucius4", "omnivoice"]]

    @staticmethod
    def _tts_fingerprint(job, segment):
        # Mock dependencies include all speaker reference text, matching source-edit invalidation.
        inputs = {"text": segment["translation"]["text"], "emotion": segment["emotion"]["label"],
                  "language": job["target_language"], "speaker": segment["speaker"],
                  "start": segment["start"], "end": segment["end"],
                  "references": [(s["segment_id"], s["source"]["effective_text"]) for s in job["_segments"]]}
        return hashlib.sha256(json.dumps(inputs, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def _write(self, db, job):
        db.execute("INSERT OR REPLACE INTO jobs VALUES (?,?)",
                   (job["job_id"], json.dumps(job, ensure_ascii=False)))

    def _event(self, job, event_type, message, stage=None, segment_id=None, data=None, level="info"):
        job["snapshot_sequence"] += 1
        job["_events"].append({
            "sequence": job["snapshot_sequence"], "timestamp": now(), "job_id": job["job_id"],
            "operation_id": job["active_operation"], "revision": job["revision"],
            "stage": stage, "segment_id": segment_id, "type": event_type,
            "level": level, "message": "[模拟] " + message, "data": data or {},
        })

    def _request(self, scope, payload, mutation):
        request_id = payload.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            raise BackendError("BAD_REQUEST", "request_id 必填。", 400)
        body = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        with self.lock, self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute("SELECT body,result FROM requests WHERE scope=? AND id=?", (scope, request_id)).fetchone()
            if old:
                if old[0] != body:
                    raise BackendError("IDEMPOTENCY_CONFLICT", "同一个 request_id 对应不同请求。")
                return json.loads(old[1])
            result = mutation(db)
            db.execute("INSERT INTO requests VALUES (?,?,?,?)",
                       (scope, request_id, body, json.dumps(result, ensure_ascii=False)))
            return copy.deepcopy(result)

    def create_job(self, video, target_language, mode, request_id):
        payload = {"video": video, "target_language": target_language, "mode": mode, "request_id": request_id}
        if target_language not in {"zh", "en", "ja"} or mode not in {"review", "auto"}:
            raise BackendError("BAD_REQUEST", "模拟数据支持 zh/en/ja 和 review/auto。", 400)

        def create(db):
            job_id = "job_" + uuid4().hex[:12]
            job = {"job_id": job_id, "revision": 1, "snapshot_sequence": 0, "mode": mode,
                   "status": "created", "current_stage": None, "resume_from": "separation",
                   "source_language": "en", "target_language": target_language,
                   "active_operation": None, "mock": True,
                   "video_name": Path(video).name if video else "内置三句模拟样例（无视频）",
                   "stages": [{"id": stage, "status": "pending", "attempt": 0,
                               "progress": None, "review_required": stage in REVIEW_STAGES,
                               "model": {"separation": "Demucs htdemucs two-stems", "asr": "Qwen3-ASR",
                                         "ocr": "HunyuanOCR / frozen_b", "emotion": "emotion2vec_plus_base",
                                         "routing": "Scheme3 router", "tts": "Scheme3"}.get(stage, "mock"), "artifact_ids": []}
                              for stage in STAGE_IDS],
                   "capabilities": {"edit_source_text": True, "edit_translation_text": True,
                                    "edit_emotion": True, "trial_tts_models": True, "adopt_tts_trial": True,
                                    "rerun_tts_segments": True, "rerun_asr_segments": False,
                                    "edit_timing": False, "freeform_instruction": False,
                                    "rework_stages": ["emotion", "translation", "light_tts", "tts"]},
                   "tts_engines": self._engine_catalog(),
                   "artifacts": [], "_segments": [], "_events": [], "_operations": {},
                   "_pause": False, "_affected": None, "_fail_next_tts": None}
            self._event(job, "job.created", "已创建隔离任务；上传内容仅作预览，执行使用内置模拟样例。")
            self._write(db, job)
            return self._public(job)
        return self._request("create", payload, create)

    @staticmethod
    def _public(job):
        return {key: copy.deepcopy(value) for key, value in job.items() if not key.startswith("_")}

    def get_job(self, job_id):
        with self.lock, self._connect() as db:
            job = self._public(self._read(db, job_id))
            job["mock_worker_error"] = self.worker_error
            return job

    def list_segments(self, job_id, offset=0, limit=50):
        with self.lock, self._connect() as db:
            job = self._read(db, job_id)
            rows = sorted(job["_segments"], key=lambda s: (s["start"], s["segment_id"]))
            return {"revision": job["revision"], "items": rows[offset:offset + limit], "total": len(rows)}

    def list_events(self, job_id, after_sequence=0, limit=200):
        with self.lock, self._connect() as db:
            job = self._read(db, job_id)
            rows = [e for e in job["_events"] if e["sequence"] > after_sequence]
            page = rows[:limit]
            return {"items": page, "next_sequence": page[-1]["sequence"] if page else after_sequence,
                    "has_more": len(rows) > len(page)}

    def get_operation(self, job_id, operation_id):
        with self.lock, self._connect() as db:
            operation = self._read(db, job_id)["_operations"].get(operation_id)
            if not operation:
                raise BackendError("NOT_FOUND", "操作不存在。", 404)
            return {k: copy.deepcopy(v) for k, v in operation.items() if not k.startswith("_")}

    def get_artifact_url(self, job_id, artifact_id):
        artifacts = self.get_job(job_id)["artifacts"]
        artifact = next((a for a in artifacts if a["artifact_id"] == artifact_id), None)
        if not artifact:
            raise BackendError("NOT_FOUND", "结果不存在。", 404)
        return artifact["url"]  # Always None for mock placeholders.

    def _check_revision(self, job, payload, allow_busy=False):
        if payload.get("expected_revision") != job["revision"]:
            raise BackendError("REVISION_CONFLICT", "任务版本已变化。未保存输入已保留，请重新载入句子后保存。",
                               current_revision=job["revision"])
        if job["active_operation"] and not allow_busy:
            raise BackendError("JOB_BUSY", "任务执行中；请等检查点或阶段边界暂停后编辑。")

    @staticmethod
    def _stage(job, stage):
        return next(s for s in job["stages"] if s["id"] == stage)

    @staticmethod
    def _segment(job, segment_id):
        segment = next((s for s in job["_segments"] if s["segment_id"] == segment_id), None)
        if not segment:
            raise BackendError("NOT_FOUND", "句子不存在。", 404)
        return segment

    def _invalidate(self, job, from_stage, ids, keep_checkpoint=False):
        stages = STAGE_IDS[STAGE_IDS.index(from_stage):]
        for stage in stages:
            record = self._stage(job, stage)
            if record["status"] != "pending":
                record.update(status="stale", progress=None)
        for segment in job["_segments"]:
            if segment["segment_id"] not in ids:
                continue
            segment["revision"] = job["revision"]
            if "translation" in stages:
                if segment["translation"]["status"] != "pending":
                    segment["translation"]["status"] = "stale"
                for candidate in segment["translation"]["candidates"]:
                    candidate["stale"] = True
            if "light_tts" in stages:
                if segment.get("light_tts_status", "pending") != "pending":
                    segment["light_tts_status"] = "stale"
            if "routing" in stages and segment["routing"]:
                segment["routing"]["status"] = "stale"
            if "tts" in stages:
                if segment["tts"]["status"] != "pending":
                    segment["tts"]["status"] = "stale"
                for trial in segment["tts_trials"]:
                    trial["stale"] = True
        for artifact in job["artifacts"]:
            if artifact["kind"] == "final_video" or (artifact.get("segment_id") in ids and artifact["stage"] in stages):
                artifact["stale"] = True
        # Multiple saved edits must accumulate dependencies; never lose an earlier dirty sentence.
        previous_ids = job.get("_affected") or []
        if job.get("resume_from") and job["status"] != "created":
            from_stage = min(from_stage, job["resume_from"], key=STAGE_IDS.index)
        current = job["current_stage"] if keep_checkpoint else from_stage
        job.update(status="waiting_review", current_stage=current, resume_from=from_stage,
                   _affected=sorted(set(previous_ids + ids)))
        stale_stages = [s for s in stages if self._stage(job, s)["status"] == "stale"]
        affected = [{"stage": s, **({"scope": "job"} if s == "mixing" else {"segment_ids": ids})} for s in stages]
        invalidated = [item for item in affected if item["stage"] in stale_stages]
        self._event(job, "results.invalidated", "后续处理起点：" + LABELS[from_stage] +
                    ("；已有旧结果待更新：" + "、".join(LABELS[s] for s in stale_stages) if stale_stages else "；未生成的步骤保持未开始"),
                    from_stage, data={"invalidated": invalidated, "affected": affected})
        return invalidated

    def edit_segment(self, job_id, segment_id, payload):
        request = {**payload, "segment_id": segment_id, "endpoint": "edit"}

        def edit(db):
            job = self._read(db, job_id)
            self._check_revision(job, payload)
            segment = self._segment(job, segment_id)
            field, value = payload.get("field"), payload.get("value")
            if field not in {"source_text", "translation_text", "candidate_id", "emotion_label"}:
                raise BackendError("UNSUPPORTED_SCOPE", "不支持此编辑字段。", 422)
            if not isinstance(value, str) or not value.strip():
                raise BackendError("EMPTY_TEXT", "文本不能为空。", 422)
            value = value.strip()
            if field == "emotion_label":
                if value not in _policy.CANONICAL_EMOTIONS:
                    raise BackendError("INVALID_EMOTION", "请选择七种规范情绪之一。", 422)
                if segment["emotion"].get("status") != "success":
                    raise BackendError("MISSING_PREREQUISITE", "请先完成情绪识别。", 422)
            if field in {"translation_text", "candidate_id"} and segment["translation"]["status"] == "pending":
                raise BackendError("MISSING_PREREQUISITE", "请先完成模拟翻译。", 422)
            if field == "candidate_id":
                candidate = next((c for c in segment["translation"]["candidates"] if c["candidate_id"] == value), None)
                if not candidate or candidate.get("stale"):
                    raise BackendError("STALE_CANDIDATE", "候选已过期，请重新运行翻译检查。")
                value = candidate["text"]
            if ((field == "source_text" and value == segment["source"]["effective_text"])
                    or (field == "translation_text" and value == segment["translation"]["text"]
                        and segment["translation"]["human_pinned"])
                    or (field == "emotion_label" and value == segment["emotion"]["label"] and segment["emotion"]["origin"] == "human")):
                return {"job_id": job_id, "revision": job["revision"], "edited_segment_ids": [],
                        "invalidated": [], "affected": [], "resume_from": job["resume_from"],
                        "requires_review": job["status"] == "waiting_review", "unchanged": True}
            before = copy.deepcopy(segment)
            job["revision"] += 1
            segment["history"].append({"revision": before["revision"], "source": before["source"],
                                       "translation": before["translation"], "tts": before["tts"],
                                       "emotion": before["emotion"], "routing": before["routing"],
                                       "reason": payload.get("reason", "人工修改"), "timestamp": now()})
            if field == "source_text":
                segment["source"].update(effective_text=value, origin="human", word_alignment_status="stale")
                # Explicit mock dependency policy: all three context/reference sentences affected.
                ids = [s["segment_id"] for s in job["_segments"]]
                start = "diarization"
            elif field == "emotion_label":
                segment["emotion"].update(label=value, origin="human", human_pinned=True)
                ids, start = [segment_id], "routing"
            else:
                segment["translation"].update(text=value, origin="human", human_pinned=True,
                                               status="success", selected_candidate_id=payload["value"] if field == "candidate_id" else None)
                for candidate in segment["translation"]["candidates"]:
                    candidate["stale"] = True
                ids, start = [segment_id], "light_tts"
            invalidated = self._invalidate(job, start, ids, keep_checkpoint=True)
            job["pending_edit"] = {"from_stage": job["resume_from"], "reason": "人工修改",
                                   "affected_segment_ids": job["_affected"]}
            self._event(job, "edit.applied", f"{segment_id} 已保存{field}；人工译文锁定会在返工时保留。",
                        start, segment_id, {"field": field, "value": value})
            self._write(db, job)
            return {"job_id": job_id, "revision": job["revision"], "edited_segment_ids": [segment_id],
                    "invalidated": invalidated,
                    "affected": [{"stage": s, **({"scope": "job"} if s == "mixing" else {"segment_ids": job["_affected"]})}
                                 for s in STAGE_IDS[STAGE_IDS.index(job["resume_from"]):]],
                    "resume_from": job["resume_from"], "requires_review": True}
        return self._request(job_id, request, edit)

    def _prerequisite(self, job, stage, ids):
        if stage not in {"separation", "asr"} and not job["_segments"]:
            raise BackendError("MISSING_PREREQUISITE", "请先完成 ASR/OCR。", 422)
        segments = [self._segment(job, sid) for sid in ids] if ids else job["_segments"]
        if stage in {"light_tts", "routing", "tts", "mixing"} and any(s["translation"]["status"] != "success" for s in segments):
            raise BackendError("MISSING_PREREQUISITE", "译文缺失或已过期，请先恢复翻译。", 422)
        if stage == "tts" and any(s["routing"].get("status") != "success" for s in segments):
            raise BackendError("MISSING_PREREQUISITE", "路由结果已过期，请先完成轻量 TTS/路由检查。", 422)
        if stage == "mixing" and any(s["tts"]["status"] != "success" for s in job["_segments"]):
            raise BackendError("MISSING_PREREQUISITE", "存在缺失、失败或过期 TTS，不能混音。", 422)

    def perform_action(self, job_id, payload):
        def act(db):
            job = self._read(db, job_id)
            action = payload.get("action")
            self._check_revision(job, payload, allow_busy=action == "pause_after_stage")
            if payload.get("instruction") is not None:
                raise BackendError("UNSUPPORTED_SCOPE", "定向自然语言指令尚未启用。", 422)
            if action in {"trial_tts", "adopt_tts_trial"}:
                result = self._trial_action(job, payload)
                self._write(db, job)
                return result
            if action == "pause_after_stage":
                if not job["active_operation"]:
                    raise BackendError("MISSING_PREREQUISITE", "当前没有运行中的操作。", 422)
                job["_pause"] = True
                job["status"] = "pause_requested"
                op = job["_operations"][job["active_operation"]]
                if op.get("kind") == "tts_trial":
                    raise BackendError("UNSUPPORTED_SCOPE", "本句试合成会自行结束，暂不支持阶段暂停。", 422)
                op["pause_boundary"] = job["current_stage"]
                self._event(job, "job.pause_requested", "已登记阶段边界暂停请求；当前阶段完成后暂停。", job["current_stage"])
            else:
                ids = [s["segment_id"] for s in job["_segments"]]
                if action == "run":
                    if job["status"] != "created":
                        raise BackendError("MISSING_PREREQUISITE", "已运行任务请使用继续或返工。", 422)
                    stage = "separation"
                elif action == "resume":
                    stage = job["resume_from"]
                    if stage is None:
                        raise BackendError("MISSING_PREREQUISITE", "任务已经完成，请选择返工。", 422)
                    ids = job["_affected"] or ids
                    # Edits to one sentence must not skip other ungenerated inputs.
                    missing = {"translation": lambda s: s["translation"]["status"] == "pending",
                               "light_tts": lambda s: s.get("light_tts_status", "pending") == "pending",
                               "routing": lambda s: s.get("routing", {}).get("status", "pending") == "pending"}
                    if stage in missing:
                        ids = sorted(set(ids + [s["segment_id"] for s in job["_segments"] if missing[stage](s)]))
                    self._prerequisite(job, stage, ids)
                elif action == "rework":
                    stage = payload.get("from_stage")
                    scope = payload.get("scope")
                    if stage not in job["capabilities"]["rework_stages"] or scope not in {"segments", "job"}:
                        raise BackendError("UNSUPPORTED_SCOPE", "此阶段或局部范围尚不支持返工。", 422)
                    if scope == "segments":
                        ids = payload.get("segment_ids") or []
                        if not ids or len(ids) != len(set(ids)):
                            raise BackendError("BAD_REQUEST", "请传入非空且不重复的稳定 segment_id。", 400)
                    self._prerequisite(job, stage, ids)
                    job["revision"] += 1
                    self._invalidate(job, stage, ids)
                    stage = job["resume_from"]
                    ids = job["_affected"]
                    self._event(job, "rework.requested", f"回到{LABELS[stage]}，范围：{', '.join(ids)}；保留人工锁定。",
                                stage, data={"segment_ids": ids, "force": payload.get("force", False)})
                else:
                    raise BackendError("BAD_REQUEST", "未知 action。", 400)
                for record in job["stages"]:
                    if record["status"] == "waiting_review":
                        record["status"] = "succeeded"
                op_id = "op_" + uuid4().hex[:12]
                op = {"operation_id": op_id, "job_id": job_id, "revision": job["revision"], "status": "queued",
                      "from_stage": stage, "current_stage": stage, "affected_segment_ids": ids,
                      "pause_boundary": None, "failure": None, "_stage_index": STAGE_IDS.index(stage), "_phase": "start"}
                job["_operations"][op_id] = op
                job.update(active_operation=op_id, current_stage=stage, status="queued", _pause=False, _affected=ids,
                           pending_edit=None)
                self._stage(job, stage)["status"] = "queued"
            self._write(db, job)
            return {k: v for k, v in op.items() if not k.startswith("_")}
        return self._request(job_id, {**payload, "endpoint": "action"}, act)

    def _trial_action(self, job, payload):
        action = payload["action"]
        if job["status"] not in {"waiting_review", "completed", "partial_failed"} or job["current_stage"] not in {"tts", "mixing"}:
            raise BackendError("REVIEW_LOCKED", "请先到配音审阅步骤，再试合成或采用结果。", 422)
        ids = payload.get("segment_ids")
        if payload.get("scope") != "segments" or not isinstance(ids, list) or len(ids) != 1:
            raise BackendError("BAD_REQUEST", "试合成/采用一次仅支持一个稳定句子 ID。", 400)
        segment = self._segment(job, ids[0])
        self._prerequisite(job, "tts", ids)
        if action == "trial_tts":
            engine = payload.get("engine")
            available = next((e for e in job["tts_engines"] if e["engine"] == engine and e["available"]
                              and job["target_language"] in e["languages"]), None)
            if not job["capabilities"].get("trial_tts_models") or not available:
                raise BackendError("UNSUPPORTED_MODEL", "该模型不可用或不支持当前目标语言。", 422)
            job["revision"] += 1
            segment["revision"] = job["revision"]
            trial_id, op_id = "trial_" + uuid4().hex[:12], "op_" + uuid4().hex[:12]
            trial = {"trial_id": trial_id, "segment_id": segment["segment_id"], "revision": job["revision"],
                     "engine": engine, "actual_engine": None, "status": "queued", "stale": False,
                     "input_fingerprint": self._tts_fingerprint(job, segment), "text": segment["translation"]["text"],
                     "emotion": segment["emotion"]["label"], "audio_artifact_id": None, "actual_duration": None,
                     "duration_error": None, "duration_quality": "mock_unmeasured", "inference_seconds": None,
                     "rtf": None, "error": None, "mock": True, "created_at": now()}
            segment["tts_trials"].append(trial)
            op = {"operation_id": op_id, "job_id": job["job_id"], "revision": job["revision"],
                  "status": "queued", "kind": "tts_trial", "from_stage": "tts", "current_stage": "tts",
                  "affected_segment_ids": ids, "trial_id": trial_id, "engine": engine,
                  "pause_boundary": None, "failure": None, "_phase": "start",
                  "_return_state": {k: copy.deepcopy(job[k]) for k in ["status", "current_stage", "resume_from"]}}
            job["_operations"][op_id] = op
            job.update(active_operation=op_id, active_operation_kind="tts_trial", trial_engine=engine,
                       status="queued", current_stage="tts")
            self._event(job, "tts.trial_requested", f"{ids[0]} 请求用 {engine} 试合成；原正式配音保留。", "tts", ids[0], trial)
        else:
            if not job["capabilities"].get("adopt_tts_trial"):
                raise BackendError("UNSUPPORTED_SCOPE", "当前服务不支持采用试合成。", 422)
            trial = next((t for t in segment["tts_trials"] if t["trial_id"] == payload.get("trial_id")), None)
            if not trial:
                raise BackendError("NOT_FOUND", "找不到当前句的试合成结果。", 404)
            if trial["stale"] or trial["input_fingerprint"] != self._tts_fingerprint(job, segment):
                raise BackendError("STALE_TRIAL", "试合成依赖已变化，请用当前文案和情绪重新试合成。")
            if trial["status"] != "success":
                raise BackendError("MISSING_PREREQUISITE", "只能采用成功的试合成结果。", 422)
            artifact = next((a for a in job["artifacts"] if a["artifact_id"] == trial["audio_artifact_id"] and not a["stale"]), None)
            if not artifact:
                raise BackendError("STALE_TRIAL", "试合成媒体已失效。")
            job["revision"] += 1
            segment["history"].append({"revision": segment["revision"], "tts": copy.deepcopy(segment["tts"]),
                                       "reason": "人工采用其他模型试合成", "timestamp": now()})
            old_audio = segment["tts"].get("audio_artifact_id")
            for item in job["artifacts"]:
                if item["kind"] == "final_video" or (item["artifact_id"] == old_audio and old_audio != trial["audio_artifact_id"]
                                                     and item["kind"] != "tts_trial_audio"):
                    item["stale"] = True
            segment["revision"] = job["revision"]
            segment["tts"] = {"status": "success", "planned_text": trial["text"], "synthesized_text": trial["text"],
                              "planned_engine": trial["engine"], "actual_engine": trial["actual_engine"],
                              "model_origin": "human_trial", "selected_trial_id": trial["trial_id"],
                              "fallback_count": 0, "fallback_reason": None,
                              "target_duration": segment["end"] - segment["start"], "error": None,
                              **{k: trial[k] for k in ["audio_artifact_id", "actual_duration", "duration_error", "duration_quality", "inference_seconds", "rtf"]}}
            all_success = all(s["tts"]["status"] == "success" for s in job["_segments"])
            job.update(status="waiting_review" if all_success else "partial_failed", current_stage="tts",
                       resume_from="mixing" if all_success else "tts", pending_edit=None,
                       _affected=None if all_success else [s["segment_id"] for s in job["_segments"] if s["tts"]["status"] != "success"])
            self._stage(job, "mixing").update(status="stale" if self._stage(job, "mixing")["attempt"] else "pending", progress=None)
            self._stage(job, "tts")["status"] = "waiting_review" if all_success else "failed"
            op_id = "op_" + uuid4().hex[:12]
            op = {"operation_id": op_id, "job_id": job["job_id"], "revision": job["revision"], "kind": "adopt_tts_trial",
                  "status": "succeeded", "from_stage": "tts", "affected_segment_ids": ids, "trial_id": trial["trial_id"],
                  "result_revision": job["revision"], "failure": None}
            job["_operations"][op_id] = op
            job["active_operation"] = op_id
            self._event(job, "tts.trial_adopted", f"{ids[0]} 已采用 {trial['actual_engine']} 试合成作为正式配音；成品需重新合成。",
                        "tts", ids[0], {"trial_id": trial["trial_id"], "actual_engine": trial["actual_engine"]})
        self._event(job, "segment.updated", "本句试合成/配音状态已更新。", "tts", ids[0])
        if action == "adopt_tts_trial":
            self._event(job, "operation.finished", "已完成本句配音替换。", "tts", ids[0], {"kind": "adopt_tts_trial"})
            job["active_operation"] = None
        return {k: v for k, v in op.items() if not k.startswith("_")}

    def _tick_trial(self, job, op):
        segment = self._segment(job, op["affected_segment_ids"][0])
        trial = next(t for t in segment["tts_trials"] if t["trial_id"] == op["trial_id"])
        if op["_phase"] == "start":
            op.update(status="running", _phase="complete")
            job["status"] = "running"
            trial["status"] = "running"
            self._event(job, "operation.started", "开始本句模型试合成。", "tts", segment["segment_id"], {"kind": "tts_trial"})
            self._event(job, "tts.trial_started", f"正在用 {trial['engine']} 试合成本句，保留原正式配音。", "tts", segment["segment_id"], trial)
        else:
            failed = job["_fail_next_tts"] == segment["segment_id"]
            if failed:
                job["_fail_next_tts"] = None
            trial.update(status="failed" if failed else "success", actual_engine=None if failed else trial["engine"],
                         error={"code": "MOCK_TTS_FAILURE", "message": "模拟试合成失败，原配音保留。"} if failed else None)
            if not failed:
                artifact_id = trial["trial_id"] + "_audio"
                trial["audio_artifact_id"] = artifact_id
                artifact = {"artifact_id": artifact_id, "kind": "tts_trial_audio", "segment_id": segment["segment_id"],
                            "stage": "tts", "revision": trial["revision"], "url": None, "stale": False,
                            "mock": True, "placeholder_reason": "模拟试合成，无真实媒体"}
                job["artifacts"].append(artifact)
                self._event(job, "artifact.created", "已生成试合成占位记录，无真实媒体。", "tts", segment["segment_id"], artifact)
            op.update(status="failed" if failed else "succeeded", failure=trial["error"], result_revision=job["revision"])
            job.update(**op["_return_state"])
            self._event(job, "tts.trial_completed", "试合成失败，原配音保留。" if failed else "试合成已完成；试听比较后可采用。",
                        "tts", segment["segment_id"], trial, level="error" if failed else "info")
            self._event(job, "operation.failed" if failed else "operation.finished", "本句试合成操作已结束。",
                        "tts", segment["segment_id"], {"kind": "tts_trial", "failure": trial["error"]}, level="error" if failed else "info")
        self._event(job, "segment.updated", "试合成状态已更新。", "tts", segment["segment_id"])
        if op["status"] in {"succeeded", "failed"}:
            job.update(active_operation=None, active_operation_kind=None, trial_engine=None)

    def inject_tts_failure(self, job_id, segment_id):
        """Demo-only control, intentionally outside the proposed BackendClient contract."""
        with self.lock, self._connect() as db:
            job = self._read(db, job_id)
            if job["active_operation"]:
                raise BackendError("JOB_BUSY", "请在检查点设置失败样例。")
            self._segment(job, segment_id)
            job["_fail_next_tts"] = segment_id
            self._event(job, "log.output", f"演示控制：下一次 {segment_id} TTS 将模拟失败。", "tts")
            self._write(db, job)

    def _artifact(self, job, segment, stage, kind, text=None):
        attempt = self._stage(job, stage)["attempt"]
        artifact_id = f"{segment or 'job'}_{stage}_r{job['revision']}_a{attempt}"
        artifact = {"artifact_id": artifact_id, "kind": kind, "segment_id": segment,
                    "stage": stage, "url": None, "revision": job["revision"], "stale": False,
                    "mock": True, "text": text, "placeholder_reason": "模拟结果，无真实媒体"}
        job["artifacts"].append(artifact)
        self._stage(job, stage)["artifact_ids"].append(artifact_id)
        self._event(job, "artifact.created", f"已生成 {kind} 占位记录，无真实媒体。", stage, segment, artifact)
        return artifact_id

    def _complete_stage(self, job, stage, ids):
        if stage == "asr":
            job["_segments"] = new_segments()
        segments = [s for s in job["_segments"] if not ids or s["segment_id"] in ids]
        for s in segments:
            sid = s["segment_id"]
            s["revision"] = job["revision"]
            if stage == "ocr":
                corrected = CORPUS[int(sid.split("_")[1]) - 1][1]
                s["source"]["ocr_corrected_text"] = corrected
                if s["source"]["origin"] != "human":
                    s["source"].update(effective_text=corrected, origin="ocr")
                s["ocr"].update(available=True, guard_reason="mock_fixture")
            elif stage == "emotion":
                s["emotion"]["status"] = "success"
            elif stage == "translation":
                tr = s["translation"]
                if not tr["human_pinned"]:
                    tr.update(text=demo_translation(s, job["target_language"]), origin="model", selected_candidate_id=None)
                tr.update(status="success", error=None)
                tr["candidates"] = [{"candidate_id": f"{sid}_r{job['revision']}_a{self._stage(job, stage)['attempt']}_c1",
                                      "text": tr["text"], "light_tts_duration": None,
                                      "duration_error": None, "duration_source": "text_estimate",
                                      "audio_artifact_id": None, "stale": False}]
                self._event(job, "log.output", f"{sid} 使用有效源文：{s['source']['effective_text']}；人工译文锁定={tr['human_pinned']}", stage, sid)
            elif stage == "light_tts":
                s["light_tts_status"] = "success"
                if s["translation"]["human_pinned"]:
                    s["translation"]["candidates"] = [{
                        "candidate_id": f"{sid}_human_r{job['revision']}_a{self._stage(job, stage)['attempt']}",
                        "text": s["translation"]["text"], "light_tts_duration": None,
                        "duration_error": None, "duration_source": "text_estimate", "audio_artifact_id": None, "stale": False}]
                # No duration measurement or audio is fabricated.
                audio_id = self._artifact(job, sid, stage, "light_tts_placeholder", s["translation"]["text"])
                for candidate in s["translation"]["candidates"]:
                    if not candidate.get("stale"):
                        candidate["audio_artifact_id"] = audio_id
            elif stage == "routing":
                cfg = _policy.ComplexityConfig()  # snapshot defaults, independent of model environment
                text, emotion, language = s["translation"]["text"], s["emotion"]["label"], job["target_language"]
                complexity = _policy.analyze_text_complexity(text, language, cfg)
                engine = _policy.select_scheme3_tts_engine(language, emotion, text, cfg)
                code = {"omnivoice": "other_language", "confucius4": "long_or_complex",
                        "indextts2": "zh_en_strong_emotion", "cosyvoice3": "positive_emotion", "f5tts": "neutral_short"}[engine]
                s["routing"] = {"status": "success", "planned_engine": engine, "rule_code": code,
                                "rule_version": "scheme3_snapshot_20261008_mock",
                                "inputs": {"target_language": language, "emotion": emotion, "tts_complexity": complexity,
                                           "thresholds": {"zh_long_chars": 28, "en_long_words": 20, "complex_clause_count": 3}}}
                s["routing"]["reason"] = explain_route(code, s["routing"]["inputs"], s["emotion"].get("origin"))
                self._event(job, "routing.decided", f"{sid} 计划 {engine}；{code}。", stage, sid, s["routing"])
            elif stage == "tts":
                old = copy.deepcopy(s["tts"])
                if old.get("planned_text"):
                    s["history"].append({"revision": s["revision"], "tts": old, "reason": "模拟 TTS 重做", "timestamp": now()})
                engine = s["routing"]["planned_engine"]
                failed = job["_fail_next_tts"] == sid
                fallback = not failed and sid == "seg_0001" and engine == "indextts2"
                s["tts"] = {"status": "failed" if failed else "success", "planned_text": s["translation"]["text"],
                            "synthesized_text": None if failed else s["translation"]["text"], "planned_engine": engine,
                            "actual_engine": None if failed else ("f5tts" if fallback else engine),
                            "fallback_count": 1 if fallback else 0, "fallback_reason": "模拟原引擎时长门禁失败" if fallback else None,
                            "target_duration": s["end"] - s["start"], "actual_duration": None,
                            "duration_error": None, "duration_quality": "mock_unmeasured", "inference_seconds": None,
                            "rtf": None, "audio_artifact_id": None,
                            "error": {"code": "MOCK_TTS_FAILURE", "message": "模拟合成失败；可仅重做此句 TTS。"} if failed else None}
                if fallback:
                    self._event(job, "tts.fallback", f"{sid} 模拟 {engine} → f5tts；模拟门禁失败。", stage, sid, s["tts"])
                if not failed:
                    s["tts"]["audio_artifact_id"] = self._artifact(job, sid, stage, "tts_placeholder", s["translation"]["text"])
            if stage in {"asr", "ocr", "emotion", "translation", "light_tts", "routing", "tts"}:
                self._event(job, "segment.updated", f"{sid} {LABELS[stage]}模拟结果已更新。", stage, sid)
        if stage == "tts":
            job["_fail_next_tts"] = None
        if stage == "mixing":
            for artifact in job["artifacts"]:
                if artifact["kind"] == "final_video":
                    artifact["stale"] = True
            self._artifact(job, None, stage, "final_video")

    def tick(self, job_id=None):
        """One mock worker step. Tests drive this deterministically; UI polling never drives it."""
        with self.lock, self._connect() as db:
            rows = db.execute("SELECT id FROM jobs").fetchall() if job_id is None else [(job_id,)]
            for (jid,) in rows:
                job = self._read(db, jid)
                if not job["active_operation"]:
                    continue
                op = job["_operations"][job["active_operation"]]
                if op.get("kind") == "tts_trial":
                    self._tick_trial(job, op)
                    self._write(db, job)
                    continue
                stage = STAGE_IDS[op["_stage_index"]]
                record = self._stage(job, stage)
                if op["_phase"] == "start":
                    if op["status"] == "queued":
                        op["status"] = "running"
                        self._event(job, "operation.started", f"从{LABELS[stage]}开始模拟执行。", stage)
                    job.update(status="pause_requested" if job["_pause"] else "running", current_stage=stage)
                    op["current_stage"] = stage
                    record.update(status="running", attempt=record["attempt"] + 1, started_at=now(), progress=None)
                    self._event(job, "stage.started", f"{LABELS[stage]}运行中；模型字段仅表示模拟调用计划。", stage,
                                data={"model": record["model"], "attempt": record["attempt"]})
                    op["_phase"] = "complete"
                else:
                    ids = op["affected_segment_ids"]
                    self._prerequisite(job, stage, ids) if stage in {"tts", "mixing"} else None
                    self._complete_stage(job, stage, ids)
                    failed = stage == "tts" and any(s["tts"]["status"] != "success" for s in job["_segments"])
                    record.update(status="failed" if failed else "succeeded", finished_at=now())
                    if stage in {"translation", "light_tts", "routing", "tts"}:
                        count = len(ids) or len(job["_segments"])
                        done = sum(s["tts"]["status"] == "success" for s in job["_segments"] if not ids or s["segment_id"] in ids) if stage == "tts" else count
                        record["progress"] = {"completed": done, "total": count, "ratio": done / count if count else None}
                    self._event(job, "stage.failed" if failed else "stage.completed", f"{LABELS[stage]}模拟{'存在失败句' if failed else '完成'}。", stage,
                                data={"model": record["model"], "attempt": record["attempt"], "progress": record["progress"]})
                    next_stage = STAGE_IDS[op["_stage_index"] + 1] if stage != "mixing" else None
                    review = job["mode"] == "review" and stage in REVIEW_STAGES
                    if failed or review or job["_pause"] or not next_stage:
                        op.update(status="partial_failed" if failed else "succeeded", result_revision=job["revision"])
                        if failed:
                            op["failure"] = {"code": "MOCK_TTS_FAILURE", "message": "有失败或过期句，未执行混音。"}
                            job.update(status="partial_failed", resume_from="tts")
                            job["_affected"] = [s["segment_id"] for s in job["_segments"] if s["tts"]["status"] != "success"]
                            self._event(job, "job.partial_failed", "请重做失败句 TTS；成功句可复用。", stage, level="error")
                        elif not next_stage:
                            job.update(status="completed", resume_from=None, _affected=None)
                            self._event(job, "job.completed", "模拟闭环完成；成品为占位记录，无视频输出。", stage)
                        elif job["_pause"]:
                            job.update(status="paused", resume_from=next_stage, _pause=False)
                            self._event(job, "job.paused", "已在阶段边界暂停。", stage)
                        else:
                            job.update(status="waiting_review", resume_from=next_stage)
                            if stage in {"ocr", "tts"}:
                                job["_affected"] = None
                            record["status"] = "waiting_review"
                            self._event(job, "stage.review_required", "等待人工确认，可编辑后继续。", stage)
                        self._event(job, "operation.finished", "模拟操作结束。", stage, data={"status": op["status"]})
                        job["active_operation"] = None
                    else:
                        op["_stage_index"] += 1
                        op["_phase"] = "start"
                self._write(db, job)

    def _work(self):
        while not self.stop_event.wait(self.interval):
            try:
                self.tick()
            except Exception as error:
                # Stop rather than spin silently; surface public, bounded diagnostic in UI.
                self.worker_error = f"模拟 worker 停止：{type(error).__name__}: {error}"
                return
