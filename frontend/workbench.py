"""UI-independent controller and safe presentation functions."""
from __future__ import annotations

import copy
import html
import json
from uuid import uuid4

from backend_client import BackendClient, BackendError, LABELS, STAGES

STATUS = {"pending": "未开始", "queued": "排队中", "running": "运行中", "succeeded": "已完成",
          "success": "已完成", "failed": "失败", "waiting_review": "待人工确认", "stale": "待更新",
          "paused": "已暂停", "created": "已创建", "completed": "已完成", "partial_failed": "部分失败",
          "pause_requested": "阶段结束后暂停", "skipped": "已跳过", "cancelled": "已取消"}


def escape(value):
    return html.escape(str(value if value is not None else "—"), quote=True)


def empty_session():
    return {"job_id": None, "job": None, "segments": [], "events": [],
            "sequence": 0, "selected_id": None, "editor_loaded_id": None}


def changed_outputs(state, values):
    """Mark only changed view fields; idle polling must not rewrite every component."""
    previous = state.get("_view_values")
    changed = [previous is None or i >= len(previous) or old != previous[i]
               for i, old in enumerate(values)]
    state["_view_values"] = copy.deepcopy(values)
    return changed


def review_permissions(job, segment):
    job = job or {}
    caps = job.get("capabilities", {})
    idle = not job.get("active_operation") and not job.get("mock_worker_error")
    review = idle and job.get("status") == "waiting_review"
    current = job.get("current_stage")
    tr = (segment or {}).get("translation", {})
    tts = (segment or {}).get("tts", {})
    return {
        "completed": job.get("status") == "completed",
        "source": bool(segment and review and current == "ocr" and caps.get("edit_source_text")),
        "translation": bool(segment and review and current == "light_tts" and tr.get("status") == "success"
                            and caps.get("edit_translation_text")),
        "emotion": bool(segment and review and current == "emotion" and caps.get("edit_emotion")),
        "routing": bool(segment and review and current == "routing"),
        "tts": bool(segment and idle and ((review and current == "tts")
                    or job.get("status") == "partial_failed")),
        "source_visible": bool(segment and segment["ocr"]["available"]),
        "translation_visible": bool(tr.get("text") and current in {"light_tts", "routing", "tts", "mixing"}),
        "emotion_visible": bool(segment and current in {"emotion", "translation", "light_tts", "routing", "tts", "mixing"}
                                and segment.get("emotion", {}).get("status") == "success"),
        "routing_visible": bool(segment and current in {"routing", "tts", "mixing"} and segment.get("routing", {}).get("status") in {"success", "stale"}),
        "tts_visible": bool(segment and current in {"tts", "mixing"}),
        "trial_visible": bool(segment and (bool(segment.get("tts_trials")) if job.get("status") == "completed" else
                              (current == "tts" or job.get("status") == "partial_failed")
                              and (caps.get("trial_tts_models") or segment.get("tts_trials")))),
    }


def review_key(state):
    job = state.get("job") or {}
    if not job.get("active_operation") and job.get("status") in {"waiting_review", "partial_failed", "completed"}:
        return (state["job_id"], job["status"], job["current_stage"])
    return None


def review_section_header(kind, label, active):
    if active:
        return f'<div class="review-disclosure active"><strong>{escape(label)}</strong><span>当前步骤</span></div>'
    return (f'<label class="review-disclosure"><input type="checkbox" id="review-{kind}-toggle">'
            f'<strong>{escape(label)}（只读）</strong><span class="review-arrow">▸</span></label>')


def review_policy_html(permissions, context, tts_open):
    flags = {"source-visible": permissions["source_visible"], "source-active": permissions["source"],
             "translation-visible": permissions["translation_visible"], "translation-active": permissions["translation"],
             "emotion-visible": permissions["emotion_visible"], "emotion-active": permissions["emotion"],
             "tts-visible": permissions["tts_visible"], "tts-active": tts_open,
             "routing-visible": permissions["routing_visible"], "routing-active": permissions["routing"],
             "trial-visible": permissions["trial_visible"], "completed": permissions["completed"]}
    attrs = " ".join(f'data-{k}="{str(bool(v)).lower()}"' for k, v in flags.items())
    return f'<div id="review-policy" {attrs}>{escape(context.replace("**", ""))}</div>'


class Workbench:
    def __init__(self, client: BackendClient):
        self.client = client

    def create(self, video, language, mode):
        job = self.client.create_job(video, language, mode, uuid4().hex)
        return self.restore(job["job_id"])

    def restore(self, job_id):
        state = empty_session()
        if not job_id:
            return state
        state["job_id"] = job_id
        return self.refresh(state)

    def refresh(self, state):
        state = copy.deepcopy(state or empty_session())
        if not state["job_id"]:
            return state
        job = self.client.get_job(state["job_id"])
        new_events = []
        while True:
            page = self.client.list_events(state["job_id"], state["sequence"])
            new_events.extend(page["items"])
            state["sequence"] = page["next_sequence"]
            if not page["has_more"]:
                break
        known = {e["sequence"] for e in state["events"]}
        state["events"].extend(e for e in new_events if e["sequence"] not in known)
        if state["sequence"] > job["snapshot_sequence"]:
            # Events can cross a stage boundary after the first snapshot read.
            job = self.client.get_job(state["job_id"])
        # Logs are plain text/history. Old revision events never write current job or segments.
        state["events"] = state["events"][-600:]
        if (not state["job"] or job["revision"] != state["job"]["revision"]
                or any(e["type"] == "segment.updated" for e in new_events)):
            offset, segments = 0, []
            while True:
                page = self.client.list_segments(state["job_id"], offset)
                segments.extend(page["items"])
                if len(segments) >= page["total"]:
                    break
                offset = len(segments)
            state["segments"] = segments
        state["job"] = job
        if not state["selected_id"] and state["segments"]:
            state["selected_id"] = state["segments"][0]["segment_id"]
        return state

    def select(self, state, segment_id):
        state = self.refresh(state)
        segment = next((s for s in state["segments"] if s["segment_id"] == segment_id), None)
        if not segment:
            raise BackendError("NOT_FOUND", "尚无可编辑句子，请先运行到 OCR 检查点。", 404)
        state["selected_id"] = segment_id
        state["editor_loaded_id"] = segment_id
        state["candidate_preview_id"] = None
        state["trial_preview_id"] = None
        state["emotion_draft"] = segment["emotion"]["label"]
        state["trial_model_choice"] = None
        state["auto_review_key"] = review_key(state)
        editor = {"job_id": state["job_id"], "segment_id": segment_id,
                  "expected_revision": state["job"]["revision"]}
        return state, editor, segment

    def edit(self, state, editor, field, value):
        self.ensure_open(state)
        if not editor or editor["job_id"] != state["job_id"] or editor["segment_id"] != state["selected_id"]:
            raise BackendError("MISSING_PREREQUISITE", "请先载入当前句子。", 422)
        current = self.refresh(state)
        if editor["expected_revision"] != current["job"]["revision"]:
            raise BackendError("REVISION_CONFLICT", "审阅版本已更新，请载入新版本后保存。")
        segment = next(s for s in current["segments"] if s["segment_id"] == editor["segment_id"])
        permission = {"source_text": "source", "emotion_label": "emotion"}.get(field, "translation")
        if not review_permissions(current["job"], segment)[permission]:
            raise BackendError("REVIEW_LOCKED", "请在对应审核步骤修改；已确认内容只读。", 422)
        result = self.client.edit_segment(state["job_id"], editor["segment_id"], {
            "request_id": uuid4().hex, "expected_revision": editor["expected_revision"],
            "field": field, "value": value, "reason": "工作台人工确认",
        })
        return self.refresh(state), result

    def action(self, state, action, stage=None):
        self.ensure_open(state)
        if not state.get("job"):
            raise BackendError("MISSING_PREREQUISITE", "请先创建任务。", 422)
        payload = {"request_id": uuid4().hex, "expected_revision": state["job"]["revision"],
                   "action": action, "instruction": None}
        if stage:
            current = self.refresh(state)
            segment = next((s for s in current["segments"] if s["segment_id"] == state["selected_id"]), None)
            if not review_permissions(current["job"], segment)["tts"]:
                raise BackendError("REVIEW_LOCKED", "请先到配音审核步骤，再发起返工。", 422)
            if not state["selected_id"]:
                raise BackendError("MISSING_PREREQUISITE", "请选择句子。", 422)
            payload.update(from_stage=stage, scope="segments", segment_ids=[state["selected_id"]], force=True)
        self.client.perform_action(state["job_id"], payload)
        state["auto_review_key"] = None
        return self.refresh(state)

    def tts_trial_action(self, state, editor, action, value):
        self.ensure_open(state)
        if not editor or editor["job_id"] != state["job_id"] or editor["segment_id"] != state["selected_id"]:
            raise BackendError("MISSING_PREREQUISITE", "请先载入当前句。", 422)
        payload = {"request_id": uuid4().hex, "expected_revision": editor["expected_revision"],
                   "action": action, "scope": "segments", "segment_ids": [editor["segment_id"]],
                   "engine" if action == "trial_tts" else "trial_id": value}
        self.client.perform_action(state["job_id"], payload)
        state["auto_review_key"] = None
        state["trial_preview_id"] = None if action == "trial_tts" else value
        return self.refresh(state)

    def ensure_open(self, state):
        # Check fresh service state too: an old tab must not bypass completion locks.
        if state.get("job_id") and self.client.get_job(state["job_id"])["status"] == "completed":
            raise BackendError("JOB_COMPLETED", "任务已完成，结果只读；当前工作台已关闭编辑与返工。", 422)

    def confirm_review(self, state, editor, source_text, translation_text, emotion_label=None):
        state = self.refresh(state)
        segment = next((s for s in state["segments"] if s["segment_id"] == state["selected_id"]), None)
        permissions = review_permissions(state["job"], segment)
        if any(permissions[k] for k in ["source", "translation", "emotion", "routing"]):
            if not editor or editor["segment_id"] != state["selected_id"] or editor["job_id"] != state["job_id"]:
                raise BackendError("MISSING_PREREQUISITE", "请先载入当前句并审阅。", 422)
            if editor["expected_revision"] != state["job"]["revision"]:
                raise BackendError("REVISION_CONFLICT", "审阅版本已更新，请载入新版本后确认。")
            comparisons = []
            if permissions["source"]:
                comparisons.append((source_text, segment["source"]["effective_text"]))
            if permissions["translation"]:
                comparisons.append((translation_text, segment["translation"]["text"]))
            if any((draft or "").strip() != saved for draft, saved in comparisons):
                raise BackendError("UNSAVED_CHANGES", "当前句有未保存修改。请先保存，再确认并继续。", 422)
            if permissions["emotion"] and emotion_label is not None and emotion_label != segment["emotion"]["label"]:
                raise BackendError("UNSAVED_CHANGES", "当前情绪选择尚未保存，请先保存情绪。", 422)
        return self.action(state, "resume")


def stage_html(job):
    records = {s["id"]: s for s in job["stages"]} if job else {}
    cards = []
    for index, (stage, label) in enumerate(STAGES, 1):
        record = records.get(stage, {"status": "pending", "attempt": 0})
        status = record["status"]
        current = job and job["status"] != "completed" and job["current_stage"] == stage
        classes = "stage " + escape(status) + (" current" if current else "")
        progress = record.get("progress")
        counter = f" · {progress['completed']}/{progress['total']}句" if progress else ""
        state_label = STATUS.get(status, status)
        model = record.get("model", "尚未调用")
        facts = [("状态", state_label + counter), ("模块 / 模型", model),
                 ("尝试次数", record["attempt"]), ("开始", record.get("started_at")),
                 ("结束", record.get("finished_at"))]
        cells = "".join(f'<dt>{escape(k)}</dt><dd>{escape(v)}</dd>' for k, v in facts)
        cards.append(f'<div class="{classes}" data-stage="{escape(stage)}">'
                     f'<div class="stage-label" tabindex="0" aria-current="{str(bool(current)).lower()}" '
                     f'aria-describedby="stage-info-{escape(stage)}" '
                     f'aria-label="{escape(label)}，{escape(state_label)}，悬停查看详情">'
                     f'<span class="stage-number">{index:02d}</span><strong>{escape(label)}</strong></div>'
                     f'<div class="stage-info" id="stage-info-{escape(stage)}" role="tooltip">'
                     f'<strong>{escape(label)} · 模拟</strong><dl>{cells}</dl></div></div>')
    return '<div class="stage-strip">' + "".join(cards) + "</div>"


ENGINE_NAMES = {"f5tts": "F5-TTS", "indextts2": "IndexTTS2", "cosyvoice3": "CosyVoice3",
                "confucius4": "Confucius4", "omnivoice": "OmniVoice"}
STAGE_TASKS = {"separation": "分离人声与背景音", "asr": "识别逐句原文", "ocr": "核对字幕与识别文本",
               "diarization": "匹配句子与说话人", "emotion": "分析逐句情绪", "translation": "生成并校验译文",
               "light_tts": "检查候选译文与轻量合成结果", "routing": "选择语音合成模型",
               "tts": "合成逐句配音", "mixing": "混音并合成成品"}


def module_name(stage, model=None):
    if stage == "routing":
        return "Scheme3 路由工具" if model in {None, "mock", "Scheme3", "Scheme3 router"} else str(model)
    if model and model != "mock":
        return f"{model} 模块"
    return f"{LABELS.get(stage, stage or '任务')}模块"


def activity_html(job, events):
    """Agent-style execution trace from explicit service events, never invented reasoning."""
    records = {s["id"]: s for s in job["stages"]} if job else {}
    rows = []
    supported = {"stage.completed", "stage.failed", "routing.decided", "tts.fallback",
                 "edit.applied", "rework.requested", "tts.trial_requested", "tts.trial_completed", "tts.trial_adopted"}
    for event in [e for e in events if e["type"] in supported][-12:]:
        stage, data, kind = event.get("stage"), event.get("data") or {}, event["type"]
        module = module_name(stage, data.get("model", records.get(stage, {}).get("model")))
        if kind == "stage.completed":
            text = "已调用 " + module + "完成模型选择" if stage == "routing" else f"已使用 {module}{STAGE_TASKS.get(stage, LABELS.get(stage, '处理'))}"
        elif kind == "routing.decided":
            engine = ENGINE_NAMES.get(data.get("planned_engine"), data.get("planned_engine") or "待确认")
            text = f"已为 {event.get('segment_id') or '当前句'} 选择 {engine} 模型（计划使用）"
            if data.get("reason"):
                text += " · " + data["reason"]
        elif kind == "tts.fallback":
            planned = ENGINE_NAMES.get(data.get("planned_engine"), data.get("planned_engine") or "原模型")
            actual = ENGINE_NAMES.get(data.get("actual_engine"), data.get("actual_engine") or "备用模型")
            text = f"已从 {planned} 切换到 {actual} 模型"
        elif kind == "stage.failed":
            text = f"{module}执行失败，请查看错误记录"
        elif kind == "edit.applied":
            field = {"source_text": "有效源文", "translation_text": "译文", "candidate_id": "译文候选", "emotion_label": "情绪"}.get(data.get("field"), "文本")
            text = f"已保存 {event.get('segment_id') or '当前句'} 的人工{field}"
        elif kind.startswith("tts.trial_"):
            engine = ENGINE_NAMES.get(data.get("actual_engine") or data.get("engine"), "所选模型")
            text = (f"已采用 {engine} 试合成替换本句正式配音" if kind == "tts.trial_adopted" else
                    f"{engine} 试合成失败，原配音保留" if data.get("status") == "failed" else
                    f"{engine} 试合成完成，等待试听比较" if kind == "tts.trial_completed" else
                    f"已请求用 {engine} 试合成本句，原配音保留")
        else:
            text = f"已请求从{LABELS.get(stage, '指定阶段')}重新处理"
        metadata = f'#{event["sequence"]} · r{event["revision"]} · {LABELS.get(stage, "任务")}'
        detail = event["message"] + ("\n" + json.dumps(data, ensure_ascii=False, indent=2) if data else "")
        tone = "error" if kind == "stage.failed" or data.get("status") == "failed" else "done"
        icon = "!" if tone == "error" else "✓"
        rows.append(f'<details class="activity-row {tone}"><summary><span class="activity-icon">{icon}</span>'
                    f'<span>{escape(text)}</span><small>{escape(metadata)}</small></summary>'
                    f'<pre>{escape(detail)}</pre></details>')
    status = job["status"] if job else "created"
    stage = job.get("current_stage") if job else None
    module = module_name(stage, records.get(stage, {}).get("model"))
    task = STAGE_TASKS.get(stage, "处理任务")
    if job and job.get("active_operation_kind") == "tts_trial":
        module = ENGINE_NAMES.get(job.get("trial_engine"), "所选 TTS 模型") + " 模块"
        task = "试合成当前句，原正式配音保留"
    if status in {"running", "pause_requested"}:
        if records.get(stage, {}).get("status") == "succeeded":
            current, tone, icon = f"已完成 {LABELS.get(stage, '当前阶段')}，正在准备下一阶段", "queued", "◷"
        else:
            current = f"正在调用 {module}{task}" if stage == "routing" else f"正在使用 {module}{task}"
            tone, icon = "running", "◌"
    elif status == "queued":
        current, tone, icon = f"等待调用 {module}{task}", "queued", "◷"
    elif status == "waiting_review" and job.get("pending_edit"):
        current = f"人工修改已保存 · 确认后从{LABELS.get(job['resume_from'], '后续步骤')}开始处理"
        tone, icon = "review", "○"
    elif status == "waiting_review":
        current, tone, icon = f"等待人工确认 · {LABELS.get(stage, '当前阶段')}", "review", "○"
    elif status == "completed":
        current, tone, icon = "已完成全部模拟阶段", "done", "✓"
    elif status in {"failed", "partial_failed"}:
        current, tone, icon = "存在失败结果，等待修正或重做", "error", "!"
    elif status == "paused":
        current, tone, icon = f"已在 {LABELS.get(stage, '当前阶段')} 结束后暂停", "review", "○"
    else:
        current, tone, icon = "等待开始执行", "queued", "○"
    if job and job.get("mock_worker_error"):
        current, tone, icon = job["mock_worker_error"], "error", "!"
    processing = status in {"queued", "running", "pause_requested"} and tone != "error"
    waiting = status == "waiting_review" and tone != "error"
    motion = " activity-motion" if processing or waiting else ""
    visual_icon = '<span class="activity-spinner"></span>' if processing else icon
    current_row = (f'<div class="activity-current {tone}{motion}" role="status">'
                   f'<span class="activity-icon" aria-hidden="true">{visual_icon}</span>'
                   f'<span>{escape(current)}</span></div>')
    recent = "".join(rows[-6:])
    history = ('<details class="activity-history"><summary>查看较早执行记录</summary>'
               + "".join(rows[:-6]) + '</details>') if len(rows) > 6 else ""
    return ('<section class="agent-activity" aria-label="模块执行动态"><div class="activity-heading">'
            '<strong>执行动态</strong><small>模拟事件 · 点击记录查看依据</small></div>' + recent + current_row + history + '</section>')


def summary_html(job):
    if not job:
        return '<div class="job-summary">先创建模拟任务，再开始执行。</div>'
    stage = LABELS.get(job["current_stage"], "尚未开始")
    resume = LABELS.get(job["resume_from"], "无")
    error = f'<p class="error">{escape(job["mock_worker_error"])}</p>' if job.get("mock_worker_error") else ""
    pending = '<small>修改已保存；仅已有旧结果需要更新。点击“确认并继续”后执行后续步骤。</small>' if job.get("pending_edit") else ""
    stage_label = "最后阶段" if job["status"] == "completed" else "当前"
    return (f'<div class="job-summary"><strong>{escape(job["job_id"])} · r{job["revision"]}</strong>'
            f'<span>{escape(STATUS.get(job["status"], job["status"]))} · {stage_label}：{escape(stage)}</span>'
            f'<small>{escape(job["video_name"])} · {escape(job.get("source_language", "待识别"))} → {escape(job["target_language"])} · '
            f'继续起点：{escape(resume)}</small>{pending}{error}</div>')


def edit_notice(result):
    if result.get("unchanged"):
        return "内容未变化；没有增加版本，也没有使结果失效。当前仍停留在原审阅位置。"
    return (f"已保存为 r{result['revision']}，尚未运行。当前审阅位置保留；"
            f"点击“确认并继续”后从{LABELS.get(result['resume_from'], result['resume_from'])}开始处理。"
            "未生成的步骤保持未开始；已有旧结果标为待更新。")


def candidate_label(candidate):
    duration = candidate.get("light_tts_duration")
    prefix = "估算" if candidate.get("duration_source") == "text_estimate" else "时长"
    measured = f"{prefix} {duration:.2f}s" if duration is not None else "时长未测量"
    error = candidate.get("duration_error")
    if error is not None:
        measured += f" · 误差 {error:+.2f}s"
    return candidate["text"] + " [" + ("待更新" if candidate.get("stale") else measured) + "]"


def ocr_evidence_html(job, segment, get_url):
    if not segment:
        return '<div class="media-note">尚无 OCR 审阅结果。</div>'
    ocr = segment["ocr"]
    artifact_id = ocr.get("evidence_artifact_id")
    artifact = next((a for a in job["artifacts"] if a["artifact_id"] == artifact_id), None)
    url = None
    if artifact and not artifact.get("stale") and not artifact.get("mock"):
        url = artifact.get("url") or get_url(job["job_id"], artifact_id)
    link = f'<a href="{escape(url)}" target="_blank" rel="noopener">查看 OCR 审计依据</a>' if url and str(url).startswith(("https://", "http://", "/api/")) else "审计文件链接待真实后端提供"
    return f'<div class="media-note">OCR 校验依据：{escape(ocr.get("guard_reason", "待校验"))} · {link}</div>'


def playback_values(job, segment, get_url, candidate_id=None):
    """Resolve task-owned media slots; never feed obsolete/mock outputs to players."""
    artifacts = {a["artifact_id"]: a for a in (job or {}).get("artifacts", [])}

    def resolve(artifact_id):
        artifact = artifacts.get(artifact_id)
        if not artifact or artifact.get("stale") or artifact.get("mock"):
            return None
        return artifact.get("url") or get_url(job["job_id"], artifact_id)

    def latest(kind):
        return next((resolve(a["artifact_id"]) for a in reversed(list(artifacts.values()))
                     if a["kind"] == kind and not a.get("stale")), None)

    candidates = (segment or {}).get("translation", {}).get("candidates", [])
    selected_id = candidate_id or (segment or {}).get("translation", {}).get("selected_candidate_id")
    candidate = next((c for c in candidates if c["candidate_id"] == selected_id and not c.get("stale")), None)
    candidate = candidate or ({} if candidate_id else next((c for c in candidates if not c.get("stale")), {}))
    tts = (segment or {}).get("tts", {})
    return [latest("final_video"), latest("dialogue_audio"), latest("background_audio"),
            resolve((segment or {}).get("source", {}).get("audio_artifact_id")),
            resolve(candidate.get("audio_artifact_id")),
            resolve(tts.get("audio_artifact_id")) if tts.get("status") == "success" else None]


def segment_rows(segments):
    return [[s["segment_id"], f'{s["start"]:.1f}–{s["end"]:.1f}',
             s["source"]["effective_text"][:48] + ("…" if len(s["source"]["effective_text"]) > 48 else ""),
             STATUS.get(s["tts"]["status"], s["tts"]["status"])] for s in segments]


def details_html(segment):
    if not segment:
        return '<div class="detail">OCR 后选择句子，载入编辑器。</div>'
    s, route, tts = segment, segment["routing"], segment["tts"]
    facts = [("稳定 ID / 版本", f"{s['segment_id']} / r{s['revision']}"),
             ("时间 / 说话人", f"{s['start']:.1f}–{s['end']:.1f}s / {s['speaker']}"),
             ("情绪 / 可靠性", f"{s['emotion']['label']} / {s['emotion'].get('reliable')}"),
             ("OCR 依据", s['ocr'].get('guard_reason', '—')),
             ("当前有效源文", s["source"]["effective_text"]),
             ("源文来源 / 逐词对齐", f"{s['source']['origin']} / {s['source']['word_alignment_status']}"),
             ("译文 / 人工锁定", f"{s['translation']['status']} / {s['translation']['human_pinned']}"),
             ("路由 / TTS 状态", f"{route.get('status', 'pending')} / {tts['status']}"),
             ("计划模型 → 实际模型", f"{route.get('planned_engine', '—')} → {tts.get('actual_engine') or '—'}"),
             ("路由依据（模拟输入）", route.get("reason", "—")),
             ("路由输入 / 阈值", route.get("inputs", "—")),
             ("fallback", tts.get("fallback_reason") or "无"),
             ("回退次数", tts.get("fallback_count", 0)),
             ("TTS 读取文案" + ("（已过期）" if tts["status"] == "stale" else ""), tts.get("synthesized_text") or tts.get("planned_text") or "—"),
             ("TTS 错误", (tts.get("error") or {}).get("message", "无")),
             ("目标 / 实测时长", f"{s['end'] - s['start']:.1f}s / {tts.get('actual_duration') or '未测量'}"),
             ("时长误差 / 质量", f"{tts.get('duration_error') if tts.get('duration_error') is not None else '未测量'} / {tts.get('duration_quality', '待合成')}"),
             ("推理耗时 / RTF", f"{tts.get('inference_seconds') if tts.get('inference_seconds') is not None else '未测量'} / {tts.get('rtf') if tts.get('rtf') is not None else '未测量'}"),
             ("历史记录", f"{len(s['history'])} 条；原 ASR 保留")]
    cells = "".join(f'<dt>{escape(key)}</dt><dd>{escape(value)}</dd>' for key, value in facts)
    history = "".join(f'<li>r{h["revision"]} · {escape(h["reason"])} · '
                      f'{escape((h.get("translation") or {}).get("text", (h.get("tts") or {}).get("planned_text", "—")))}</li>'
                      for h in s["history"][-5:])
    return f'<div class="detail"><dl>{cells}</dl><details><summary>最近版本记录</summary><ul>{history}</ul></details></div>'


def event_html(events):
    business = [e for e in events if e["type"] not in {"log.output", "segment.updated", "artifact.created"}]
    cards = "".join(f'<div class="event {escape(e["level"])}"><small>#{e["sequence"]} · r{e["revision"]} · '
                    f'{escape(e["type"])} · {escape(e["timestamp"])}</small><p>{escape(e["message"])}</p></div>'
                    for e in reversed(business[-12:]))
    return '<div class="events">' + (cards or "暂无事件。") + "</div>"


def log_text(events):
    return "\n".join(f'#{e["sequence"]} [{e["timestamp"]}] r{e["revision"]} {e["type"]} {e["message"]}'
                     for e in events) or "等待模拟任务。"


def media_html(job, segment):
    if not job:
        return '<div class="media-note">无媒体。真实模型及视频处理在服务器运行。</div>'
    if not segment:
        return '<div class="media-note">选择一句后查看轻量试听、正式配音和返工结果。</div>'
    route, tts = segment["routing"], segment["tts"]
    planned = ENGINE_NAMES.get(route.get("planned_engine"), route.get("planned_engine") or "待路由")
    actual = ENGINE_NAMES.get(tts.get("actual_engine"), tts.get("actual_engine") or "尚未合成")
    if route.get("status") == "stale":
        planned += "（旧计划，待更新）"
    if tts["status"] == "stale":
        actual += "（旧结果，待更新）"
    read_text = tts.get("synthesized_text") or tts.get("planned_text") or segment["translation"]["text"] or "待确认译文"
    text_label = "上次合成文案（待更新）" if tts["status"] == "stale" else ("实际合成文案" if tts.get("synthesized_text") else "已保存文案（待合成）")
    facts = [("轻量试听", STATUS.get(segment.get("light_tts_status", "pending"), "未开始")),
             ("正式配音", STATUS.get(tts["status"], tts["status"])),
             ("计划模型", planned), ("实际模型", actual)]
    facts.append(("模型选择来源", "人工采用试合成" if tts.get("model_origin") == "human_trial" else "自动路由 / 回退"))
    cells = "".join(f'<dt>{escape(k)}</dt><dd>{escape(v)}</dd>' for k, v in facts)
    issue = tts.get("fallback_reason") or ""
    if tts.get("error"):
        issue = tts["error"]["message"]
    relevant = [a for a in job["artifacts"] if a["kind"] == "final_video" or a.get("segment_id") == segment["segment_id"]]
    rows = "".join(f'<li>{escape(a["kind"])} · r{a["revision"]} · '
                   f'<strong>{"旧结果待更新" if a["stale"] else "模拟占位" if a.get("mock") else "可用媒体"}</strong></li>' for a in relevant[-8:])
    media_note = "模拟结果尚无真实音频；播放器位置与媒体接口已预留。" if job.get("mock") else "选择候选可切换轻量试听；正式配音使用已保存文案。"
    duration_note = f"合成时长 {tts['actual_duration']:.2f}s。" if tts.get("actual_duration") is not None else "合成时长尚未测量。"
    return ('<div class="media-note"><dl class="tts-facts">' + cells + '</dl>'
            f'<p><strong>{text_label}：</strong>{escape(read_text)}</p>'
            + (f'<p><strong>自动路由依据：</strong>{escape(route["reason"])}'
               + ('（旧依据，待重新路由）' if route.get("status") == "stale" else '') + '</p>' if route.get("reason") else '')
            + (f'<p class="error">{escape(issue)}</p>' if tts.get("error") else f'<p>{escape(issue)}</p>' if issue else '')
            + f'<p>{media_note}原文时间段为 {segment["end"] - segment["start"]:.1f}s，{duration_note}</p>'
            f'<details><summary>查看结果版本与过期状态</summary><ul>{rows}</ul></details></div>')


def routing_html(segment):
    route = (segment or {}).get("routing", {})
    if not route.get("planned_engine"):
        return "尚未计算模型路由。"
    return ('<div class="media-note"><p>计划模型：<strong>'
            + escape(ENGINE_NAMES.get(route["planned_engine"], route["planned_engine"]))
            + '</strong></p><p>路由依据：' + escape(route.get("reason"))
            + '</p><p>结果状态：' + escape(STATUS.get(route.get("status"), route.get("status")))
            + '</p><details><summary>查看规则及输入</summary><pre>'
            + escape(json.dumps({"rule_code": route.get("rule_code"), "rule_version": route.get("rule_version"),
                                 "inputs": route.get("inputs")}, ensure_ascii=False, indent=2)) + '</pre></details></div>')


def emotion_html(segment):
    emotion = (segment or {}).get("emotion", {})
    if emotion.get("status") != "success":
        return "情绪识别完成后显示。"
    origin = "人工确认" if emotion.get("origin") == "human" else "模型识别 / 可靠性回退"
    return (f'<div class="media-note"><p>原识别：<strong>{escape(emotion.get("raw_label"))}</strong> · '
            f'当前采用：<strong>{escape(emotion.get("label"))}</strong> · {origin}</p>'
            f'<p>模型置信度：{escape(emotion.get("score") if emotion.get("score") is not None else "未提供")} · '
            f'识别可靠性：{escape(emotion.get("reliable"))}。人工修正保留原识别记录。</p></div>')


def available_tts_engines(job):
    return [(ENGINE_NAMES.get(e["engine"], e["engine"]), e["engine"]) for e in (job or {}).get("tts_engines", [])
            if e.get("available") and (job or {}).get("target_language") in e.get("languages", [])]


def trial_view(job, segment, trial_id, get_url):
    trials = (segment or {}).get("tts_trials", [])
    trial = next((t for t in trials if t["trial_id"] == trial_id), None) if trial_id else (trials[-1] if trials else None)
    if not trial:
        note = "请从本句试合成记录选择一条结果，再试听比较或采用。" if trials else "选择其他模型试合成后，在这里与原正式配音比较。采用前不会替换原配音。"
        if (job or {}).get("status") == "completed":
            note = "任务已完成；历史试合成仅供查看和试听。"
        return f'<div class="media-note">{note}</div>', None, False
    model = ENGINE_NAMES.get(trial["engine"], trial["engine"])
    state = "待更新" if trial.get("stale") else STATUS.get(trial["status"], trial["status"])
    selected = (segment or {}).get("tts", {}).get("selected_trial_id") == trial["trial_id"]
    audio = None
    artifact = next((a for a in (job or {}).get("artifacts", []) if a["artifact_id"] == trial.get("audio_artifact_id") and not a.get("stale")), None)
    valid = trial["status"] == "success" and not trial.get("stale") and artifact is not None
    if valid and not artifact.get("mock"):
        audio = artifact.get("url") or get_url(job["job_id"], artifact["artifact_id"])
    note = "模拟试合成，无真实音频，时长尚未测量。" if trial.get("mock") else "试听后可明确采用；不会自动判断哪一个更好。"
    if (job or {}).get("status") == "completed":
        note = "任务已完成，此记录仅供查看和试听。" + ("模拟记录无真实音频。" if trial.get("mock") else "")
    error = (trial.get("error") or {}).get("message")
    measured = trial.get("actual_duration")
    duration = f"{measured:.2f}s" if measured is not None else "未测量"
    delta = trial.get("duration_error")
    delta_text = f"{delta:+.2f}s" if delta is not None else "未测量"
    facts = [("实际模型", ENGINE_NAMES.get(trial.get("actual_engine"), trial.get("actual_engine") or "尚未完成")),
             ("时长质量", trial.get("duration_quality", "未提供")),
             ("推理耗时 / RTF", f"{trial.get('inference_seconds') if trial.get('inference_seconds') is not None else '未测量'} / {trial.get('rtf') if trial.get('rtf') is not None else '未测量'}")]
    cells = "".join(f'<dt>{escape(k)}</dt><dd>{escape(v)}</dd>' for k, v in facts)
    rendered = (f'<div class="media-note"><p><strong>{escape(model)}</strong> · {escape(state)} · r{trial["revision"]}'
                + (' · 已采用为正式配音' if selected else '') + '</p>'
                + f'<p>试合成文案：{escape(trial["text"])} · 情绪：{escape(trial["emotion"])}</p>'
                + f'<p>合成时长：{escape(duration)} · 时长误差：{escape(delta_text)}</p>'
                + f'<p>{escape(error or note)}</p><details><summary>试合成质量与耗时详情</summary><dl class="tts-facts">{cells}</dl></details></div>')
    return rendered, audio, bool(valid and not selected and review_permissions(job, segment)["tts"]
                                 and (job or {}).get("capabilities", {}).get("adopt_tts_trial"))
