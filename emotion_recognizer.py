#!/usr/bin/env python3
"""emotion2vec bridge for AutoDub.

The public API can recognize one file, while the pipeline path batches all
segments through one isolated Conda worker so that the model is loaded once.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List

from scheme3_policy import EMOTION_LABEL_MAP, normalize_emotion as _normalize_emotion


DEFAULT_MODEL_DIR = "/data0/goldenseed/qgd/models/emotion2vec_plus_base"
DEFAULT_MODEL_ID = "iic/emotion2vec_plus_base"
DEFAULT_ENV = "emotion2vec"


def apply_emotion_reliability(segments, *, model=DEFAULT_MODEL_DIR):
    """Retain raw predictions; smooth only unreliable same-speaker neighbors."""
    rows = [dict(s) for s in segments]
    minimum_duration = float(os.getenv("AUTODUB_EMOTION_MIN_DURATION", "0.8"))
    for row in rows:
        raw = row.get("raw_emotion", row.get("emotion_raw", row.get("emotion", "unknown")))
        try:
            score = float(row.get("emotion_raw_score", row.get("emotion_score", 0.0)))
        except (TypeError, ValueError):
            score = 0.0
        label = normalize_emotion(raw)
        threshold = float(os.getenv(f"AUTODUB_EMOTION_MIN_SCORE_{label.upper()}",
                                    os.getenv("AUTODUB_EMOTION_MIN_SCORE", "0.5")))
        duration = float(row["end"])-float(row["start"])
        failure = row.get("emotion_status") in {"failed", "pending"}
        if failure:
            status, reason = "failed", "worker_failed:" + str(row.get("emotion_error", "unknown"))
        elif duration < minimum_duration:
            status, reason = "too_short", "shorter_than_min_duration"
        elif not math.isfinite(score) or not 0 <= score <= 1 or score < threshold:
            status, reason = "low_confidence", "invalid_or_below_class_threshold"
        elif str(raw).lower() in {"unknown", "other"} or not any(
                part.strip() in EMOTION_LABEL_MAP for part in str(raw).lower().split("/")):
            status, reason = "unknown_label", "unknown_label"
        else:
            status, reason = "success", ""
        row.update(emotion_raw_label=str(raw), emotion_raw_score=score if math.isfinite(score) else None,
                   emotion=label if status=="success" else "neutral",
                   normalized_emotion=label if status=="success" else "neutral",
                   emotion_score=score if status=="success" else 0.0,
                   emotion_status=status, emotion_reliable=status=="success",
                   emotion_fallback_reason=reason, emotion_model=model,
                   emotion_unsmoothed=label, emotion_reliability_source="model" if status=="success" else "fallback")
    anchors = [dict(row) for row in rows]
    gap = float(os.getenv("AUTODUB_EMOTION_CONTEXT_GAP", "1.5"))
    for i, row in enumerate(rows):
        if row["emotion_reliable"] or row["emotion_status"] not in {"too_short", "low_confidence"}:
            continue
        adjacent = []
        for j in (i-1, i+1):
            if not 0 <= j < len(rows):
                continue
            anchor = anchors[j]
            separation = abs(float(row["start"])-float(anchor["end"])) if j<i else abs(float(anchor["start"])-float(row["end"]))
            if (row.get("speaker") and row.get("speaker") == anchor.get("speaker") and
                    anchor["emotion_reliable"] and separation <= gap):
                adjacent.append(anchor)
        if adjacent and len({s["emotion"] for s in adjacent}) == 1:
            anchor = max(adjacent, key=lambda s: s["emotion_score"])
            row.update(emotion=anchor["emotion"], normalized_emotion=anchor["emotion"],
                       emotion_score=anchor["emotion_score"], emotion_reliable=True,
                       emotion_reliability_source="same_speaker_neighbor",
                       emotion_context_segment_ids=[s.get("id") for s in adjacent])
    distribution = {}
    for row in rows:
        values = distribution.setdefault(row["emotion"], [])
        if row["emotion_raw_score"] is not None:
            values.append(row["emotion_raw_score"])
    print("Emotion summary:", {k: {"count":len(v), "min":min(v), "max":max(v),
                                 "mean":sum(v)/len(v)} for k,v in distribution.items() if v})
    scores = [r["emotion_raw_score"] for r in rows if r["emotion_raw_score"] is not None]
    if scores and sum(s>=0.999 for s in scores)/len(scores) >= 0.8:
        print("⚠️ Emotion scores saturated; raw model scores are NOT calibrated probabilities.")
    return rows

def normalize_emotion(
    raw_emotion: Any,
    *,
    score: Any = None,
    duration: Any = None,
) -> str:
    """Backwards-compatible public wrapper around the scheme-three policy."""
    return _normalize_emotion(raw_emotion, score=score, duration=duration)


def _atomic_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _load_results(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return {str(item["segment_id"]): item for item in payload}
    return {str(key): value for key, value in payload.items()}


def _has_emotion(segment: Dict[str, Any]) -> bool:
    return (
        segment.get("emotion_status") == "success"
        and bool(segment.get("emotion"))
        and segment.get("emotion_score") is not None
    )


def _worker_command(plan: Path, result: Path, model_dir: str, model_id: str) -> List[str]:
    worker = Path(__file__).with_name("emotion_worker.py")
    env_name = os.getenv("AUTODUB_EMOTION_ENV", DEFAULT_ENV)
    device = os.getenv("AUTODUB_EMOTION_DEVICE", "cuda:0")
    conda = shutil.which("conda") or "/opt/anaconda3/bin/conda"
    return [
        conda, "run", "--no-capture-output", "-n", env_name,
        "python", str(worker), "--plan", str(plan), "--result", str(result),
        "--model-dir", model_dir, "--model-id", model_id, "--device", device,
    ]


def recognize_emotion(
    audio_path: str,
    *,
    model_dir: str = DEFAULT_MODEL_DIR,
    model_id: str = DEFAULT_MODEL_ID,
) -> Dict[str, Any]:
    """Recognize one audio file through the isolated worker."""
    source = Path(audio_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    with tempfile.TemporaryDirectory(prefix="autodub_emotion_") as directory:
        root = Path(directory)
        plan = root / "plan.json"
        result = root / "results.json"
        _atomic_json(plan, {"items": [{"segment_id": "0", "audio_path": str(source)}]})
        environment = os.environ.copy()
        environment["PYTHONNOUSERSITE"] = "1"
        subprocess.run(
            _worker_command(plan, result, model_dir, model_id),
            check=True,
            env=environment,
        )
        item = _load_results(result).get("0")
        if not item or item.get("status") != "success":
            raise RuntimeError((item or {}).get("error", "emotion worker returned no result"))
        return {
            "emotion": item["normalized_emotion"],
            "score": float(item["emotion_score"]),
            "raw_emotion": item["raw_emotion"],
        }


def recognize_segments(
    segments: Iterable[Dict[str, Any]],
    vocal_path: str,
    temp_dir: str,
    *,
    force: bool = False,
    model_dir: str = DEFAULT_MODEL_DIR,
    model_id: str = DEFAULT_MODEL_ID,
) -> List[Dict[str, Any]]:
    """Recognize every pending segment, preserving completed checkpoint data."""
    from pydub import AudioSegment

    updated = [dict(segment) for segment in segments]
    root = Path(temp_dir) / "emotion2vec"
    clips_dir = root / "segments"
    plan_path = root / "plan.json"
    result_path = root / "results.json"
    clips_dir.mkdir(parents=True, exist_ok=True)
    results = _load_results(result_path)
    full_audio = AudioSegment.from_wav(vocal_path)
    pending: List[Dict[str, str]] = []

    for index, segment in enumerate(updated):
        segment_id = str(segment.get("id", index))
        previous = results.get(segment_id, {})
        duration_seconds = max(0.0, float(segment["end"]) - float(segment["start"]))
        if not force and _has_emotion(segment):
            raw = segment.get("raw_emotion", segment.get("emotion_raw", segment.get("emotion")))
            normalized = normalize_emotion(
                raw,
                score=segment.get("emotion_score"),
                duration=duration_seconds,
            )
            segment["emotion_raw"] = str(raw or "unknown")
            segment["raw_emotion"] = str(raw or "unknown")
            segment["emotion"] = normalized
            segment["normalized_emotion"] = normalized
            continue
        if not force and previous.get("status") == "success":
            normalized = normalize_emotion(
                previous["raw_emotion"],
                score=previous.get("emotion_score"),
                duration=duration_seconds,
            )
            previous["normalized_emotion"] = normalized
            previous["duration_seconds"] = duration_seconds
            results[segment_id] = previous
            segment["emotion_raw"] = previous["raw_emotion"]
            segment["raw_emotion"] = previous["raw_emotion"]
            segment["emotion"] = normalized
            segment["normalized_emotion"] = normalized
            segment["emotion_score"] = float(previous["emotion_score"])
            segment["emotion_raw_score"] = float(previous["emotion_score"])
            segment["emotion_status"] = "success"
            continue

        start_ms = max(0, int(round(float(segment["start"]) * 1000)))
        end_ms = min(len(full_audio), int(round(float(segment["end"]) * 1000)))
        clip_path = clips_dir / f"segment_{segment_id}.wav"
        if force or not clip_path.exists() or clip_path.stat().st_size == 0:
            full_audio[start_ms:end_ms].export(clip_path, format="wav")
        pending.append({
            "segment_id": segment_id,
            "audio_path": str(clip_path),
            "duration_seconds": duration_seconds,
        })
        results[segment_id] = {
            **previous,
            "segment_id": segment_id,
            "audio_path": str(clip_path),
            "status": "pending",
        }

    _atomic_json(result_path, results)
    if pending:
        _atomic_json(plan_path, {"items": pending})
        environment = os.environ.copy()
        environment["PYTHONNOUSERSITE"] = "1"
        print(f"--- [Step 2.5] emotion2vec: {len(pending)} 个待处理 segment ---")
        completed = subprocess.run(
            _worker_command(plan_path, result_path, model_dir, model_id),
            env=environment,
            check=False,
        )
        if completed.returncode != 0:
            print(f"⚠️ emotion2vec worker 退出码: {completed.returncode}")
            results = _load_results(result_path)
            for item in pending:
                segment_id = item["segment_id"]
                if results.get(segment_id, {}).get("status") == "pending":
                    results[segment_id] = {
                        **results.get(segment_id, {}),
                        "status": "failed",
                        "error": f"emotion worker exited with {completed.returncode}",
                    }
            _atomic_json(result_path, results)

    results = _load_results(result_path)
    for index, segment in enumerate(updated):
        segment_id = str(segment.get("id", index))
        item = results.get(segment_id, {})
        if item.get("status") == "success":
            duration_seconds = max(0.0, float(segment["end"]) - float(segment["start"]))
            normalized = normalize_emotion(
                item["raw_emotion"],
                score=item.get("emotion_score"),
                duration=duration_seconds,
            )
            item["normalized_emotion"] = normalized
            item["duration_seconds"] = duration_seconds
            results[segment_id] = item
            segment["emotion_raw"] = item["raw_emotion"]
            segment["raw_emotion"] = item["raw_emotion"]
            segment["emotion"] = normalized
            segment["normalized_emotion"] = normalized
            segment["emotion_score"] = float(item["emotion_score"])
            segment["emotion_raw_score"] = float(item["emotion_score"])
            segment["emotion_status"] = "success"
        elif item:
            segment["emotion_status"] = item.get("status", "failed")
            segment["emotion_error"] = item.get("error", "")
    _atomic_json(result_path, results)
    return apply_emotion_reliability(updated, model=model_dir)
