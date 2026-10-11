"""Per-segment translation validation, retry and resumable checkpoints."""
from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any, Callable

RULE_VERSION = "translation-v2.2.1"
LANGUAGE_VERSION = "target-script-v2"
PROMPT_VERSION = "context-candidates-v2"

# Translation is fail-closed by default: one segment that cannot be validated
# stops the stage instead of letting an untranslated line reach the mixer.
# AUTODUB_TRANSLATION_FAILURE_POLICY=skip_failed is the explicit opt-out for
# long videos -- blocking segments are marked as skipped, left undubbed, and
# their original vocals are restored in Step 5.
TRANSLATION_POLICIES = ("fail_fast", "skip_failed")
DEFAULT_TRANSLATION_POLICY = "fail_fast"
SKIPPED_STATUS = "skipped"


def resolve_translation_policy(explicit: Any = None) -> str:
    """Resolve the failure policy; unknown values fall back to the safe default."""
    value = explicit if explicit is not None else os.getenv("AUTODUB_TRANSLATION_FAILURE_POLICY", "")
    value = str(value or "").strip().lower()
    if not value:
        return DEFAULT_TRANSLATION_POLICY
    if value not in TRANSLATION_POLICIES:
        print(f"⚠️ 未知的翻译失败策略 {value!r}，回退到 {DEFAULT_TRANSLATION_POLICY}")
        return DEFAULT_TRANSLATION_POLICY
    return value


def language_code(value: Any) -> str:
    value = str(value or "").strip().lower().replace("_", "-")
    value = {"english": "en", "chinese": "zh", "mandarin": "zh", "cmn": "zh",
             "japanese": "ja", "korean": "ko", "french": "fr", "german": "de",
             "spanish": "es", "russian": "ru"}.get(value, value)
    return value.split("-", 1)[0]


def validate_target_text(text: Any, target: str, original: str = "", source: str = "",
                         *, allow_identity: bool = False) -> str:
    if not isinstance(text, str) or not text.strip() or not any(c.isalnum() for c in text):
        raise ValueError("empty_or_nonlinguistic_text")
    text = text.strip()
    lang = language_code(target)
    han = len(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    if lang == "zh" and (not han or latin > max(12, han * 4)):
        raise ValueError("target_zh_missing_or_not_mainly_chinese")
    if lang == "en" and (not latin or han):
        raise ValueError("target_en_contains_source_script_or_no_english")
    scripts = {"ja": r"[\u3040-\u30ff\u3400-\u9fff]", "ko": r"[\uac00-\ud7af]",
               "ru": r"[\u0400-\u04ff]", "ar": r"[\u0600-\u06ff]"}
    if lang in scripts and not re.search(scripts[lang], text):
        raise ValueError(f"target_{lang}_missing_script")
    if lang in {"fr", "de", "es", "it", "pt"} and (not latin or han):
        raise ValueError(f"target_{lang}_invalid_script")
    canonical = lambda s: re.sub(r"[\W_]+", "", s.casefold())
    if (original and canonical(original) == canonical(text) and
            language_code(source) != lang and not allow_identity):
        raise ValueError("unchanged_source_text")
    return text


def is_tiny(text: str) -> bool:
    han = len(re.findall(r"[\u3400-\u9fff]", text))
    words = re.findall(r"\w+", text, re.UNICODE)
    return (0 < han <= 4) or (not han and len(words) <= 2)


def validate_translation(row: dict, original: dict, source: str, target: str) -> dict:
    from light_tts_selector import normalize_translation_candidates
    if not isinstance(row, dict):
        raise ValueError("missing_translation_id")
    allow_identity = bool(original.get("translation_allow_identity", False))
    primary = validate_target_text(row.get("text"), target, original.get("text", ""), source,
                                   allow_identity=allow_identity)
    raw = row.get("translations_candidates")
    if not isinstance(raw, list) or not raw:
        raise ValueError("missing_candidates")
    # Validate every supplied candidate BEFORE normalization/truncation.
    for candidate in raw:
        validate_target_text(candidate.get("text") if isinstance(candidate, dict) else candidate,
                             target, original.get("text", ""), source, allow_identity=allow_identity)
    candidates = normalize_translation_candidates(raw, "")
    minimum = 1 if is_tiny(str(original.get("text", ""))) else 3
    if not minimum <= len(candidates) <= 5 or len(raw) > 5:
        raise ValueError(f"candidate_count:{len(candidates)} required:{minimum}-5")
    warnings = []
    if len({len(c["text"]) for c in candidates}) < min(3, len(candidates)):
        warnings.append("candidate_lengths_similar")
    return {"text": primary, "translations_candidates": candidates,
            "translation_status": "success", "translation_error": "",
            "translation_warnings": warnings}


def segment_fingerprint(segment: dict, previous: str, following: str, source: str,
                        target: str, model: str) -> str:
    payload = {"rules": RULE_VERSION, "language_check": LANGUAGE_VERSION, "prompt": PROMPT_VERSION,
               "source": language_code(source), "target": language_code(target), "model": model,
               "segment": {k: segment.get(k) for k in
                           ("id", "text", "start", "end", "speaker", "translation_allow_identity")},
               "previous_text": previous, "next_text": following}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def translate_segments(segments: list[dict], source: str, target: str, model: str,
                       request: Callable | None, *, batch_size: int = 10, retries: int = 2,
                       cached: list[dict] | None = None, force_ids: set[str] | None = None,
                       checkpoint: Callable | None = None) -> list[dict]:
    ids = [str(s.get("id", i)) for i, s in enumerate(segments)]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate segment IDs")
    previous = {str(s.get("id", i)): s for i, s in enumerate(cached or [])}
    rows, pending, inputs = [], [], {}
    for i, segment in enumerate(segments):
        sid = ids[i]
        before = str(segments[i-1].get("text", "")) if i else ""
        after = str(segments[i+1].get("text", "")) if i+1 < len(segments) else ""
        fp = segment_fingerprint(segment, before, after, source, target, model)
        row = dict(segment, original_text=segment.get("text", ""), text="",
                   translations_candidates=[], translation_status="failed",
                   translation_source_language=language_code(source),
                   translation_target_language=language_code(target),
                   translation_request_fingerprint=fp, translation_attempts=0)
        row["translation_merge_suggestion"] = bool(is_tiny(str(segment.get("text", ""))) and
            i+1 < len(segments) and segment.get("speaker") == segments[i+1].get("speaker"))
        known = previous.get(sid, {})
        try:
            if sid in (force_ids or set()) or known.get("translation_status") != "success" or known.get("translation_request_fingerprint") != fp:
                raise ValueError("cache_miss")
            if known.get("translation_same_language") and language_code(source) == language_code(target):
                text = validate_target_text(known.get("text"), target)
                row.update(text=text, translations_candidates=[{"text": text}],
                           translation_status="success", translation_error="", translation_same_language=True)
            else:
                row.update(validate_translation(known, segment, source, target))
            row["translation_attempts"] = known.get("translation_attempts", 0)
            row["translation_cache_hit"] = True
        except ValueError:
            pending.append(i)
        rows.append(row)
        inputs[i] = {"id": sid, "speaker": segment.get("speaker", "Unknown"),
                     "text_asr_corrected": segment.get("text", ""),
                     "previous_text": before, "next_text": after,
                     "duration": float(segment["end"])-float(segment["start"]),
                     "minimum_candidates": 1 if is_tiny(str(segment.get("text", ""))) else 3}

    def save():
        if checkpoint:
            checkpoint(rows)

    def attempt(indices):
        response = {}
        error = "missing_api_key"
        try:
            if request is not None:
                payload = request([inputs[i] for i in indices])
                items = payload.get("translations")
                if not isinstance(items, list):
                    raise ValueError("translations_not_list")
                duplicates = set()
                for item in items:
                    if not isinstance(item, dict) or "id" not in item:
                        continue
                    sid = str(item["id"])
                    if sid in response:
                        duplicates.add(sid)
                    response[sid] = item
                for sid in duplicates:
                    response.pop(sid)
                error = "missing_or_duplicate_id"
        except Exception as exc:
            error = f"{type(exc).__name__}:{exc}"
        for i in indices:
            rows[i]["translation_attempts"] += 1
            try:
                if ids[i] not in response:
                    raise ValueError(error)
                valid = validate_translation(response[ids[i]], segments[i], source, target)
                rows[i].update(valid)
                for warning in valid["translation_warnings"]:
                    print(f"⚠️ translation segment={ids[i]} {warning}")
            except ValueError as exc:
                rows[i]["translation_error"] = str(exc)
                print(f"⚠️ translation segment={ids[i]} attempt={rows[i]['translation_attempts']} failed: {exc}")
        save()

    if language_code(source) == language_code(target):
        for i in pending:
            identity = {"text": str(segments[i].get("text", "")),
                        "translations_candidates": [{"text": str(segments[i].get("text", ""))}]}
            # Same-language passthrough is explicit; no invented variants needed.
            try:
                text = validate_target_text(identity["text"], target)
                rows[i].update(identity, text=text, translation_status="success", translation_error="",
                               translation_same_language=True)
            except ValueError as exc:
                rows[i]["translation_error"] = str(exc)
        save()
        return rows
    for offset in range(0, len(pending), max(1, batch_size)):
        attempt(pending[offset:offset+max(1, batch_size)])
    if request is not None:
        for i in pending:
            for _ in range(max(0, min(3, retries))):
                if rows[i]["translation_status"] == "success":
                    break
                attempt([i])
    save()
    print(f"Translation: {sum(r['translation_status']=='success' for r in rows)}/{len(rows)} success")
    return rows


def is_skipped(segment) -> bool:
    """True when a segment was explicitly degraded by the skip_failed policy."""
    return segment.get("translation_status") == SKIPPED_STATUS


def segment_translation_error(segment: dict, target: str, source: str) -> str:
    """Return the blocking reason for one segment, or '' when it is usable."""
    try:
        if segment.get("translation_target_language") and segment["translation_target_language"] != language_code(target):
            raise ValueError("translation_target_language_changed")
        if segment.get("translation_status") == "failed":
            raise ValueError(segment.get("translation_error", "translation_failed"))
        original = segment.get("original_text", "")
        source_language = target if segment.get("translation_same_language") else segment.get(
            "translation_source_language", segment.get("source_lang", source))
        validate_target_text(segment.get("text"), target, original,
                             source_language,
                             allow_identity=bool(segment.get("translation_allow_identity")))
        for candidate in segment.get("translations_candidates", []):
            validate_target_text(candidate.get("text"), target, original,
                                 source_language,
                                 allow_identity=bool(segment.get("translation_allow_identity")))
    except ValueError as exc:
        return str(exc)
    return ""


def collect_translation_failures(segments, target: str, source: str = "") -> list[dict]:
    """Rows that would block the pipeline.

    Segments already marked as skipped are not re-reported: they were degraded
    by an explicit policy decision, not by a new failure.
    """
    failures = []
    for index, segment in enumerate(segments):
        if is_skipped(segment):
            continue
        reason = segment_translation_error(segment, target, source)
        if reason:
            failures.append({"index": index, "id": str(segment.get("id", index)), "reason": reason})
    return failures


def describe_translation_failures(failures) -> str:
    return "; ".join(f"{row['id']}:{row['reason']}" for row in failures)


def skip_failed_segments(segments, failures) -> list[dict]:
    """Mark blocking segments so routing and TTS leave them undubbed."""
    skipped = []
    for row in failures:
        segment = segments[row["index"]]
        segment["translation_status"] = SKIPPED_STATUS
        segment["translation_skip_reason"] = row["reason"]
        segment["dubbing_enabled"] = False
        skipped.append(segment)
    return skipped


def require_translated(segments, target: str, source: str = "", *, policy: Any = None) -> list[dict]:
    """Gate the run on validated translations, with an explicit degradation path.

    ``fail_fast`` (the default) raises on the first blocking segment, exactly as
    before.  ``skip_failed`` marks those segments as skipped and returns them so
    the caller can continue; the caller is then responsible for reporting them.
    """
    segments = list(segments)
    failures = collect_translation_failures(segments, target, source)
    if not failures:
        return []
    detail = describe_translation_failures(failures)
    if resolve_translation_policy(policy) != "skip_failed":
        raise RuntimeError("Translation blocked; segment IDs/reasons: " + detail)
    skipped = skip_failed_segments(segments, failures)
    print(
        f"⚠️ 翻译降级(skip_failed)：{len(skipped)}/{len(segments)} 句未通过校验，"
        f"这些句子本轮不配音，Step 5 将保留其原声 -> {detail}"
    )
    return skipped
