"""Normalize AutoDub pipeline state for the frontend contract.

This module is intentionally independent from model and GPU dependencies.  It
does not alter the pipeline's checkpoint format; callers pass real persisted
state and receive JSON-serializable response objects with stable field names.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional


STAGE_IDS = (
    "separation",
    "asr",
    "ocr",
    "diarization",
    "emotion",
    "translation",
    "light_tts",
    "routing",
    "tts",
    "mixing",
)

CAPABILITY_KEYS = (
    "edit_source_text",
    "edit_translation_text",
    "edit_emotion",
    "rerun_tts_segments",
    "trial_tts_models",
    "adopt_tts_trial",
    "rerun_asr_segments",
    "edit_timing",
    "freeform_instruction",
)

ArtifactResolver = Callable[[Any], Optional[str]]


def _string(value: Any, default: str = "") -> str:
    return default if value is None else str(value)


def _optional_string(value: Any) -> Optional[str]:
    if value is None or value == "":
        return None
    return str(value)


def _optional_number(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _integer(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _error(value: Any, default_code: str) -> Optional[dict[str, Any]]:
    if value in (None, "", {}):
        return None
    if isinstance(value, Mapping):
        result = dict(value)
        result.setdefault("code", default_code)
        result.setdefault("message", _string(result.get("error") or result.get("detail")))
        return result
    return {"code": default_code, "message": str(value)}


def _artifact_id(value: Any, resolver: Optional[ArtifactResolver]) -> Optional[str]:
    """Resolve an internal media reference without exposing a server path."""
    if value in (None, ""):
        return None
    if resolver is None:
        return None
    return _optional_string(resolver(value))


def _stable_id(prefix: str, *parts: Any) -> str:
    payload = json.dumps(parts, ensure_ascii=False, separators=(",", ":"), default=str)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    return f"{prefix}_{digest}"


def serialize_candidate(
    candidate: Mapping[str, Any],
    *,
    segment_id: str,
    index: int,
    artifact_resolver: Optional[ArtifactResolver] = None,
) -> dict[str, Any]:
    text = _string(candidate.get("text")).strip()
    candidate_id = _optional_string(candidate.get("candidate_id")) or _stable_id(
        "candidate", segment_id, text
    )
    measured = candidate.get("light_tts_duration", candidate.get("actual_duration"))
    if "duration_source" in candidate:
        duration_source = _string(candidate.get("duration_source"), "unmeasured")
    elif candidate.get("estimated"):
        duration_source = "text_estimate"
    elif measured is not None:
        duration_source = "measured_audio"
    else:
        duration_source = "unmeasured"
    media = candidate.get("audio_artifact_id")
    if media is None:
        media = candidate.get("preview_audio", candidate.get("audio_file"))
    return {
        "candidate_id": candidate_id,
        "text": text,
        "light_tts_duration": _optional_number(measured),
        "duration_error": _optional_number(candidate.get("duration_error")),
        "duration_source": duration_source,
        "audio_artifact_id": (
            _optional_string(media)
            if candidate.get("audio_artifact_id") is not None
            else _artifact_id(media, artifact_resolver)
        ),
        "stale": bool(candidate.get("stale", False)),
    }


def serialize_trial(
    trial: Mapping[str, Any],
    *,
    artifact_resolver: Optional[ArtifactResolver] = None,
) -> dict[str, Any]:
    media = trial.get("audio_artifact_id")
    if media is None:
        media = trial.get("file", trial.get("audio_file"))
    return {
        "trial_id": _string(trial.get("trial_id")),
        "segment_id": _string(trial.get("segment_id")),
        "revision": _integer(trial.get("revision")),
        "engine": _string(trial.get("engine")),
        "actual_engine": _optional_string(trial.get("actual_engine")),
        "status": _string(trial.get("status"), "queued"),
        "stale": bool(trial.get("stale", False)),
        "input_fingerprint": _string(
            trial.get("input_fingerprint", trial.get("request_fingerprint"))
        ),
        "text": _string(trial.get("text")),
        "emotion": _string(trial.get("emotion"), "neutral"),
        "audio_artifact_id": (
            _optional_string(media)
            if trial.get("audio_artifact_id") is not None
            else _artifact_id(media, artifact_resolver)
        ),
        "actual_duration": _optional_number(trial.get("actual_duration")),
        "duration_error": _optional_number(trial.get("duration_error")),
        "duration_quality": _string(trial.get("duration_quality"), "unmeasured"),
        "inference_seconds": _optional_number(trial.get("inference_seconds")),
        "rtf": _optional_number(trial.get("rtf")),
        "error": _error(trial.get("error"), "TTS_TRIAL_FAILED"),
        "mock": bool(trial.get("mock", False)),
        "created_at": _string(trial.get("created_at")),
    }


def serialize_segment(
    segment: Mapping[str, Any],
    *,
    revision: int,
    raw_source: Optional[Mapping[str, Any]] = None,
    ocr_record: Optional[Mapping[str, Any]] = None,
    tts_result: Optional[Mapping[str, Any]] = None,
    artifact_resolver: Optional[ArtifactResolver] = None,
) -> dict[str, Any]:
    raw_source = raw_source or {}
    ocr_record = ocr_record or {}
    tts_result = tts_result or {}
    segment_id = _string(segment.get("segment_id", segment.get("id")))

    raw_text = _string(
        segment.get("raw_asr_text", raw_source.get("text", segment.get("original_text", "")))
    )
    corrected = segment.get("ocr_corrected_text", segment.get("corrected_text"))
    if corrected is None and raw_text and segment.get("original_text") is None:
        current_source = _string(segment.get("text"))
        if current_source and current_source != raw_text and not segment.get("translation_status"):
            corrected = current_source
    effective_source = _string(
        segment.get("effective_source_text", segment.get("original_text", corrected or raw_text))
    )
    source_origin = _string(segment.get("source_origin"))
    if source_origin not in {"asr", "ocr", "human"}:
        source_origin = "ocr" if corrected is not None and corrected != raw_text else "asr"

    emotion_status = _string(segment.get("emotion_status"), "pending")
    if emotion_status not in {"pending", "success"}:
        emotion_status = "success" if segment.get("emotion") is not None else "pending"
    raw_emotion = segment.get(
        "emotion_raw_label", segment.get("raw_emotion", segment.get("emotion_raw"))
    )

    candidates = [
        serialize_candidate(
            item,
            segment_id=segment_id,
            index=index,
            artifact_resolver=artifact_resolver,
        )
        for index, item in enumerate(segment.get("translations_candidates") or [])
        if isinstance(item, Mapping)
    ]
    selected_candidate_id = _optional_string(segment.get("selected_candidate_id"))
    if selected_candidate_id is None:
        for index, item in enumerate(segment.get("translations_candidates") or []):
            if isinstance(item, Mapping) and item.get("selected") and index < len(candidates):
                selected_candidate_id = candidates[index]["candidate_id"]
                break

    translation_status = _string(segment.get("translation_status"), "pending")
    if "translation_text" in segment:
        translation_text = _string(segment.get("translation_text"))
    elif "translation_status" in segment or "original_text" in segment:
        translation_text = _string(segment.get("text"))
    else:
        translation_text = ""
    translation_error = _error(segment.get("translation_error"), "TRANSLATION_FAILED")

    complexity = segment.get("tts_complexity")
    planned_engine = segment.get("planned_engine", segment.get("tts_engine"))
    routing: dict[str, Any] = {}
    if planned_engine or complexity or segment.get("routing_status"):
        routing = {
            "status": _string(segment.get("routing_status"), "success"),
            "planned_engine": _string(planned_engine),
            "rule_code": _string(segment.get("routing_rule_code")),
            "rule_version": _string(segment.get("routing_rule_version")),
            "reason": _string(segment.get("routing_reason")),
            "inputs": dict(segment.get("routing_inputs") or {
                "target_language": segment.get("target_lang"),
                "emotion": segment.get("emotion"),
                "tts_complexity": complexity or {},
            }),
        }

    tts_status = _string(
        segment.get("tts_status", tts_result.get("status", "pending")), "pending"
    )
    tts_media = segment.get("tts_audio_artifact_id", tts_result.get("audio_artifact_id"))
    if tts_media is None:
        tts_media = tts_result.get("file")
    tts = {
        "status": tts_status,
        "audio_artifact_id": (
            _optional_string(tts_media)
            if segment.get("tts_audio_artifact_id") is not None
            or tts_result.get("audio_artifact_id") is not None
            else _artifact_id(tts_media, artifact_resolver)
        ),
        "planned_text": _optional_string(
            segment.get("tts_planned_text", tts_result.get("planned_text"))
        ),
        "synthesized_text": _optional_string(
            segment.get("tts_synthesized_text", tts_result.get("tts_text"))
        ),
        "planned_engine": _optional_string(
            segment.get("planned_engine", tts_result.get("planned_engine"))
        ),
        "actual_engine": _optional_string(
            segment.get("actual_engine", tts_result.get("actual_engine"))
        ),
        "fallback_count": _integer(
            segment.get("fallback_count", tts_result.get("fallback_count"))
        ),
        "fallback_reason": _optional_string(
            segment.get("fallback_reason", tts_result.get("fallback_reason"))
        ),
        "target_duration": _optional_number(
            segment.get("tts_target_duration", tts_result.get("target_duration"))
        ),
        "actual_duration": _optional_number(
            segment.get("tts_actual_duration", tts_result.get("actual_duration"))
        ),
        "duration_error": _optional_number(
            segment.get("tts_duration_error", tts_result.get("duration_error"))
        ),
        "duration_quality": _optional_string(
            segment.get("tts_duration_quality", tts_result.get("duration_quality"))
        ),
        "inference_seconds": _optional_number(
            segment.get("tts_inference_seconds", tts_result.get("inference_seconds"))
        ),
        "rtf": _optional_number(segment.get("tts_rtf", tts_result.get("rtf"))),
        "error": _error(
            segment.get("tts_error", tts_result.get("error")), "TTS_FAILED"
        ),
    }
    model_origin = segment.get("tts_model_origin", tts_result.get("model_origin"))
    if model_origin is not None:
        tts["model_origin"] = _string(model_origin)
    selected_trial = segment.get("selected_trial_id", tts_result.get("selected_trial_id"))
    if selected_trial is not None:
        tts["selected_trial_id"] = _string(selected_trial)

    source_media = segment.get("source_audio_artifact_id")
    if source_media is None:
        source_media = segment.get("source_audio", segment.get("audio_file"))
    evidence_media = segment.get("ocr_evidence_artifact_id")
    return {
        "segment_id": segment_id,
        "parent_segment_id": _optional_string(segment.get("parent_segment_id")),
        "revision": _integer(segment.get("revision"), revision),
        "start": float(segment.get("start", 0.0)),
        "end": float(segment.get("end", 0.0)),
        "speaker": _string(segment.get("speaker"), "Unknown"),
        "source": {
            "raw_asr_text": raw_text,
            "ocr_corrected_text": _optional_string(corrected),
            "effective_text": effective_source,
            "origin": source_origin,
            "word_alignment_status": _string(
                segment.get("word_alignment_status"), "valid"
            ),
            "audio_artifact_id": (
                _optional_string(source_media)
                if segment.get("source_audio_artifact_id") is not None
                else _artifact_id(source_media, artifact_resolver)
            ),
        },
        "ocr": {
            "available": bool(
                segment.get("ocr_available", bool(ocr_record) or corrected is not None)
            ),
            "evidence_artifact_id": (
                _optional_string(evidence_media)
                if evidence_media is not None
                else _artifact_id(ocr_record.get("evidence"), artifact_resolver)
            ),
            "guard_reason": _optional_string(
                segment.get("ocr_guard_reason", ocr_record.get("reason"))
            ),
        },
        "history": list(segment.get("history") or []),
        "emotion": {
            "status": emotion_status,
            "label": _string(segment.get("emotion"), "neutral"),
            "raw_label": _optional_string(raw_emotion),
            "recognized_label": _optional_string(
                segment.get("emotion_recognized_label", segment.get("normalized_emotion"))
            ),
            "score": _optional_number(segment.get("emotion_score")),
            "reliable": (
                None
                if segment.get("emotion_reliable") is None
                else bool(segment.get("emotion_reliable"))
            ),
            "fallback_reason": _optional_string(
                segment.get("emotion_fallback_reason")
            ),
            "origin": _string(segment.get("emotion_origin"), "model"),
            **(
                {"human_pinned": bool(segment.get("emotion_human_pinned"))}
                if "emotion_human_pinned" in segment
                else {}
            ),
        },
        "translation": {
            "text": translation_text,
            "status": translation_status,
            "origin": _string(segment.get("translation_origin"), "model"),
            "human_pinned": bool(segment.get("translation_human_pinned", False)),
            "selected_candidate_id": selected_candidate_id,
            "candidates": candidates,
            "error": translation_error,
        },
        "light_tts_status": _string(segment.get("light_tts_status"), "pending"),
        "tts_trials": [
            serialize_trial(item, artifact_resolver=artifact_resolver)
            for item in segment.get("tts_trials") or []
            if isinstance(item, Mapping)
        ],
        "routing": routing,
        "tts": tts,
    }


def serialize_segment_page(
    segments: Iterable[Mapping[str, Any]],
    *,
    revision: int,
    raw_sources: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ocr_records: Optional[Mapping[str, Mapping[str, Any]]] = None,
    tts_results: Optional[Mapping[str, Mapping[str, Any]]] = None,
    artifact_resolver: Optional[ArtifactResolver] = None,
) -> dict[str, Any]:
    raw_sources = raw_sources or {}
    ocr_records = ocr_records or {}
    tts_results = tts_results or {}
    items = []
    for index, segment in enumerate(segments):
        segment_id = _string(segment.get("segment_id", segment.get("id", index)))
        items.append(
            serialize_segment(
                segment,
                revision=revision,
                raw_source=raw_sources.get(segment_id),
                ocr_record=ocr_records.get(segment_id),
                tts_result=tts_results.get(segment_id),
                artifact_resolver=artifact_resolver,
            )
        )
    return {"revision": _integer(revision), "items": items, "total": len(items)}


def serialize_artifact(artifact: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "artifact_id": _string(artifact.get("artifact_id")),
        "kind": _string(artifact.get("kind")),
        "stage": _string(artifact.get("stage")),
        "segment_id": _optional_string(artifact.get("segment_id")),
        "revision": _integer(artifact.get("revision")),
        "url": _optional_string(artifact.get("url")),
        "stale": bool(artifact.get("stale", False)),
        "mock": bool(artifact.get("mock", False)),
        **(
            {"placeholder_reason": _string(artifact.get("placeholder_reason"))}
            if artifact.get("placeholder_reason") is not None
            else {}
        ),
        **(
            {"text": _optional_string(artifact.get("text"))}
            if "text" in artifact
            else {}
        ),
    }


def serialize_stage(stage: Mapping[str, Any]) -> dict[str, Any]:
    progress = stage.get("progress")
    normalized_progress = None
    if isinstance(progress, Mapping):
        completed = _integer(progress.get("completed"))
        total = _integer(progress.get("total"))
        normalized_progress = {
            "completed": completed,
            "total": total,
            "ratio": _optional_number(
                progress.get("ratio", completed / total if total else None)
            ),
        }
    result = {
        "id": _string(stage.get("id")),
        "status": _string(stage.get("status"), "pending"),
        "attempt": _integer(stage.get("attempt")),
        "progress": normalized_progress,
        "review_required": bool(stage.get("review_required", False)),
        "model": _optional_string(stage.get("model")),
        "artifact_ids": [str(item) for item in stage.get("artifact_ids") or []],
    }
    if "started_at" in stage:
        result["started_at"] = _optional_string(stage.get("started_at"))
    if "finished_at" in stage:
        result["finished_at"] = _optional_string(stage.get("finished_at"))
    return result


def _capabilities(value: Any) -> dict[str, Any]:
    source = value if isinstance(value, Mapping) else {}
    result = {key: bool(source.get(key, False)) for key in CAPABILITY_KEYS}
    result["rework_stages"] = [str(item) for item in source.get("rework_stages") or []]
    return result


def serialize_job(state: Mapping[str, Any]) -> dict[str, Any]:
    """Serialize persisted service state; a real job_id is mandatory."""
    job_id = _string(state.get("job_id")).strip()
    if not job_id:
        raise ValueError("job_id is required to serialize a frontend Job")
    input_file = _optional_string(state.get("input_file"))
    raw_stages = state.get("stages") or []
    stage_by_id = {
        _string(item.get("id")): item
        for item in raw_stages
        if isinstance(item, Mapping)
    }
    stages = [
        serialize_stage(stage_by_id.get(stage_id, {"id": stage_id, "status": "pending"}))
        for stage_id in STAGE_IDS
    ]
    result = {
        "job_id": job_id,
        "revision": _integer(state.get("revision")),
        "snapshot_sequence": _integer(state.get("snapshot_sequence")),
        "mode": _string(state.get("mode"), "review"),
        "status": _string(state.get("status"), "created"),
        "current_stage": _optional_string(state.get("current_stage")),
        "resume_from": _optional_string(state.get("resume_from")),
        "active_operation": _optional_string(state.get("active_operation")),
        "source_language": _optional_string(
            state.get("source_language", state.get("src_lang"))
        ),
        "target_language": _string(
            state.get("target_language", state.get("target_lang"))
        ),
        "video_name": _string(
            state.get("video_name"), Path(input_file).name if input_file else ""
        ),
        "stages": stages,
        "capabilities": _capabilities(state.get("capabilities")),
        "artifacts": [
            serialize_artifact(item)
            for item in state.get("artifacts") or []
            if isinstance(item, Mapping)
        ],
    }
    for field in (
        "tts_engines",
        "mock",
        "pending_edit",
        "mock_worker_error",
        "active_operation_kind",
        "trial_engine",
    ):
        if field in state:
            result[field] = state[field]
    return result


def serialize_event(event: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "sequence": _integer(event.get("sequence")),
        "timestamp": _string(event.get("timestamp")),
        "job_id": _string(event.get("job_id")),
        "operation_id": _optional_string(event.get("operation_id")),
        "revision": _integer(event.get("revision")),
        "stage": _optional_string(event.get("stage")),
        "segment_id": _optional_string(event.get("segment_id")),
        "type": _string(event.get("type")),
        "level": _string(event.get("level"), "info"),
        "message": _string(event.get("message")),
        "data": dict(event.get("data") or {}),
    }


def serialize_event_page(
    events: Iterable[Mapping[str, Any]], *, next_sequence: int, has_more: bool
) -> dict[str, Any]:
    return {
        "items": [serialize_event(item) for item in events],
        "next_sequence": _integer(next_sequence),
        "has_more": bool(has_more),
    }


def serialize_operation(operation: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        "operation_id": _string(operation.get("operation_id")),
        "job_id": _string(operation.get("job_id")),
        "revision": _integer(operation.get("revision")),
        "status": _string(operation.get("status")),
        "from_stage": _string(operation.get("from_stage")),
        "affected_segment_ids": [
            str(item) for item in operation.get("affected_segment_ids") or []
        ],
        "failure": _error(operation.get("failure"), "OPERATION_FAILED"),
    }
    for field in (
        "current_stage",
        "pause_boundary",
        "result_revision",
        "kind",
        "trial_id",
        "engine",
    ):
        if field in operation:
            result[field] = operation[field]
    return result


def serialize_edit_result(result: Mapping[str, Any]) -> dict[str, Any]:
    normalized = {
        "job_id": _string(result.get("job_id")),
        "revision": _integer(result.get("revision")),
        "edited_segment_ids": [
            str(item) for item in result.get("edited_segment_ids") or []
        ],
        "invalidated": list(result.get("invalidated") or []),
        "affected": list(result.get("affected") or []),
        "resume_from": _optional_string(result.get("resume_from")),
        "requires_review": bool(result.get("requires_review", False)),
    }
    if "unchanged" in result:
        normalized["unchanged"] = bool(result.get("unchanged"))
    return normalized


def segments_from_state(
    state: Mapping[str, Any],
    *,
    artifact_resolver: Optional[ArtifactResolver] = None,
) -> dict[str, Any]:
    """Build a SegmentPage from the latest real pipeline checkpoint rows."""
    rows = (
        state.get("segments_step3")
        or state.get("segments_translation_candidates")
        or state.get("segments_emotion")
        or state.get("segments_step2")
        or state.get("segments_step1")
        or []
    )
    raw_rows = state.get("segments_qwen3_raw") or state.get("segments_step1_asr") or []
    raw_sources = {
        _string(item.get("segment_id", item.get("id", index))): item
        for index, item in enumerate(raw_rows)
        if isinstance(item, Mapping)
    }
    ocr_records = state.get("ocr_texts_by_segment") or {}
    if not isinstance(ocr_records, Mapping):
        ocr_records = {}
    tts_results = {
        _string(item.get("segment_id")): item
        for item in state.get("tts_router_results") or []
        if isinstance(item, Mapping)
    }
    return serialize_segment_page(
        rows,
        revision=_integer(state.get("revision")),
        raw_sources=raw_sources,
        ocr_records=ocr_records,
        tts_results=tts_results,
        artifact_resolver=artifact_resolver,
    )
