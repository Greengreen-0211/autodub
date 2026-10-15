"""Per-segment translation validation, retry and resumable checkpoints."""
from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from typing import Any, Callable

RULE_VERSION = "translation-v2.12.0"
LANGUAGE_VERSION = "target-script-v2"
PROMPT_VERSION = "neutral-dubbing-ocr-v12"

OCR_HARD_CONFLICTS = {
    "wrong_entity",
    "wrong_number",
    "wrong_negation",
    "contradiction",
    "missing_independent_proposition",
    "invented_proposition",
    "source_mismatch",
    "glossary_violation",
    "corrupt_text",
}

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


def _canonical(value: Any) -> str:
    return re.sub(r"[\W_]+", "", str(value or "").casefold())


def _source_contains_form(text: Any, form: Any) -> bool:
    text, form = str(text or ""), str(form or "").strip()
    if not form:
        return False
    if re.search(r"[A-Za-z0-9]", form):
        pattern = r"(?<![A-Za-z0-9])" + re.escape(form) + r"(?![A-Za-z0-9])"
        return re.search(pattern, text, flags=re.IGNORECASE) is not None
    return _canonical(form) in _canonical(text)


def applicable_glossary_entries(segment: dict, glossary: list[dict] | None) -> list[dict]:
    """Return validated video-level entities that occur in this source turn."""
    source_parts = [str(segment.get("text", ""))]
    source_parts.extend(
        str(reference.get("source_text", ""))
        for reference in segment.get("_ocr_translation_references", [])
        if isinstance(reference, dict)
    )
    source_text = " ".join(source_parts)
    output = []
    for entry in glossary or []:
        if not isinstance(entry, dict):
            continue
        target_form = str(entry.get("target_form", "")).strip()
        source_forms = [str(value).strip() for value in entry.get("source_forms", []) if str(value).strip()]
        if target_form and source_forms and any(
            _source_contains_form(source_text, form) for form in source_forms
        ):
            output.append({"source_forms": source_forms, "target_form": target_form})
    return output


def validate_entity_glossary(text: Any, required: list[dict]) -> None:
    """Fail closed when a translation drops a required localized entity."""
    normalized = _canonical(text)
    missing = [
        entry["target_form"] for entry in required
        if _canonical(entry["target_form"]) not in normalized
    ]
    if missing:
        raise ValueError("entity_glossary_mismatch:" + ",".join(missing))


def validate_ocr_text_quality(text: Any) -> None:
    """Reject deterministic OCR corruption before semantic model auditing."""
    value = str(text or "")
    if any(character == "\ufffd" or 0xE000 <= ord(character) <= 0xF8FF for character in value):
        raise ValueError("ocr_corrupt_unicode")
    if re.search(r"<\/?(?:ref|quad)>|\(\s*\d+\s*,\s*\d+\s*\)", value, re.I):
        raise ValueError("ocr_markup_leaked")
    if any(ord(character) < 32 and character not in "\t\r\n" for character in value):
        raise ValueError("ocr_control_character")


def _ocr_audit_acceptance(response_row: dict, variant: str) -> tuple[bool, str]:
    audit = response_row.get("_ocr_candidate_audit")
    if not isinstance(audit, dict):
        return False, "audit_missing"
    verdict = audit.get(variant)
    if not isinstance(verdict, dict):
        return False, f"{variant}_audit_missing"
    hard = verdict.get("hard_conflicts")
    if isinstance(hard, list):
        hard = [str(item).strip() for item in hard if str(item).strip()]
        unknown = [item for item in hard if item not in OCR_HARD_CONFLICTS]
        if unknown:
            return False, f"{variant}_audit_invalid_hard_conflict:{','.join(unknown[:3])}"
        if hard:
            return False, f"{variant}_audit_hard_conflict:{','.join(hard[:3])}"
        if verdict.get("semantic_equivalent") is not True:
            return False, f"{variant}_audit_not_semantically_equivalent"
        if variant == "hybrid" and verdict.get("only_uncovered_translated") is not True:
            return False, f"{variant}_audit_changed_trusted_cards"
        soft = verdict.get("soft_issues") if isinstance(verdict.get("soft_issues"), list) else []
        detail = ",".join(str(item) for item in soft[:3] if str(item).strip())
        return True, "audit_passed" + (f":soft={detail}" if detail else "")

    # Backward compatibility for an old cached/API response. New requests use
    # the structured hard/soft schema and are not rejected merely for idiom,
    # register, or repairable punctuation.
    required = ("accepted", "semantic_complete", "faithful", "no_added_meaning")
    failed = [field for field in required if verdict.get(field) is not True]
    if variant == "hybrid" and verdict.get("only_uncovered_translated") is not True:
        failed.append("only_uncovered_translated")
    if failed:
        return False, f"{variant}_legacy_audit_failed:{','.join(failed)}"
    return True, "legacy_audit_passed"


def _ocr_audit_verdict(response_row: dict, variant: str) -> dict:
    audit = response_row.get("_ocr_candidate_audit")
    if not isinstance(audit, dict):
        return {}
    verdict = audit.get(variant)
    return verdict if isinstance(verdict, dict) else {}


def _punctuation_lexical_signature(text: Any) -> str:
    """Return text content with punctuation/spacing removed, preserving symbols."""
    output = []
    value = str(text or "").casefold()
    for index, character in enumerate(value):
        category = unicodedata.category(character)
        previous_is_alnum = index > 0 and value[index - 1].isalnum()
        next_is_alnum = index + 1 < len(value) and value[index + 1].isalnum()
        if previous_is_alnum and next_is_alnum:
            if character in {"'", "’", "ʼ"}:
                output.append("'")
                continue
            if character in {"-", "‐", "‑", "–"}:
                output.append("-")
                continue
            if character in {".", "．", ",", "，"} and (
                value[index - 1].isdigit() and value[index + 1].isdigit()
            ):
                output.append(character.translate(str.maketrans({"．": ".", "，": ","})))
                continue
        if character.isspace() or category.startswith("P") or category.startswith("Z"):
            continue
        output.append(character)
    return "".join(output)


def _source_sentence_count(text: Any) -> int:
    parts = [
        value for value in re.split(r"(?<=[.!?。！？])\s+|[\r\n]+", str(text or ""))
        if value.strip()
    ]
    return max(1, len(parts))


def _target_terminal(source_text: Any, target: Any) -> str:
    source = str(source_text or "").rstrip()
    lang = language_code(target)
    if source.endswith(("?", "？")):
        return "？" if lang in {"zh", "ja"} else "?"
    if source.endswith(("!", "！")):
        return "！" if lang in {"zh", "ja"} else "!"
    return "。" if lang in {"zh", "ja"} else "."


def _punctuation_ready_text(
    raw_text: Any,
    source_text: Any,
    target: Any,
    audit_verdict: Any,
    required_entities: list[dict] | None = None,
) -> tuple[str, bool, str]:
    """Accept only punctuation/spacing edits and verify TTS-ready boundaries."""
    raw = str(raw_text or "").strip()
    verdict = audit_verdict if isinstance(audit_verdict, dict) else {}
    suggested = str(verdict.get("tts_ready_text", "") or "").strip()
    status = "original_punctuation"
    value = raw
    if suggested:
        if _punctuation_lexical_signature(suggested) != _punctuation_lexical_signature(raw):
            return raw, False, "punctuation_changed_words"
        value = suggested
        status = "audited_punctuation"

    # Normalize spacing around punctuation without changing lexical content.
    value = re.sub(r"\s+([，。！？；：、,.!?;:])", r"\1", value)
    value = re.sub(r"[ \t]{2,}", " ", value).strip()
    if not re.search(r"[。！？.!?][\"'”’）\]】》]*$", value):
        value += _target_terminal(source_text, target)
        status = "audited_plus_terminal" if suggested else "deterministic_terminal"

    if _punctuation_lexical_signature(value) != _punctuation_lexical_signature(raw):
        return raw, False, "punctuation_signature_mismatch"

    # Multi-sentence source rendered as one unbroken target phrase is unsafe
    # for both prosody and clause-based routing.
    body = re.sub(r"[。！？.!?][\"'”’）\]】》]*$", "", value).strip()
    if _source_sentence_count(source_text) >= 2 and not re.search(
        r"[，。！？；：,.!?;:]", body
    ):
        return value, False, "missing_internal_boundary"

    # A genuinely short address before the first comma is commonly a vocative.
    # Never treat a completed sentence followed by a later comma as one.
    source = str(source_text or "").strip()
    first_comma = source.find(",")
    vocative_prefix = source[:first_comma].strip() if first_comma >= 0 else ""
    vocative_words = re.findall(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?", vocative_prefix)
    is_vocative = bool(
        first_comma > 0
        and not re.search(r"[.!?。！？;；:]", vocative_prefix)
        and 1 <= len(vocative_words) <= 3
        and len(vocative_prefix) <= 24
    )
    if is_vocative:
        # If the source address maps to a validated video-level entity and the
        # OCR target begins with that entity, inserting a comma is deterministic
        # punctuation restoration rather than translation.
        for entry in required_entities or []:
            target_form = str(entry.get("target_form", "")).strip()
            source_forms = entry.get("source_forms", [])
            if not target_form or not value.startswith(target_form):
                continue
            if not any(_source_contains_form(vocative_prefix, form) for form in source_forms):
                continue
            boundary = len(target_form)
            if boundary < len(value) and value[boundary] not in "，,：:。！？.!?":
                value = value[:boundary] + ("，" if language_code(target) in {"zh", "ja"} else ", ") + value[boundary:]
                status = "deterministic_vocative" if not suggested else "audited_plus_vocative"
            break
        prefix = value[: max(4, min(10, len(value)))]
        if not re.search(r"[，,：:]", prefix):
            return value, False, "missing_vocative_pause"

    pairs = (("（", "）"), ("(", ")"), ("【", "】"), ("[", "]"), ("《", "》"))
    if any(value.count(left) != value.count(right) for left, right in pairs):
        return value, False, "unbalanced_punctuation"
    return value, True, status


def _ocr_repair_distance(reference: Any, repaired: Any) -> float:
    """Return lexical edit fraction; punctuation-only changes score zero."""
    from difflib import SequenceMatcher
    left = _punctuation_lexical_signature(reference)
    right = _punctuation_lexical_signature(repaired)
    if not left or not right:
        return 1.0
    return 1.0 - SequenceMatcher(None, left, right).ratio()


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
    canonical = lambda value: re.sub(r"[\W_]+", "", str(value).casefold())
    primary_key = canonical(primary)
    primary_index = next(
        (index for index, candidate in enumerate(candidates)
         if canonical(candidate["text"]) == primary_key),
        None,
    )
    if primary_index is None:
        candidates = ([{"text": primary}] + candidates)[:5]
        primary_index = 0
    # Candidate metadata returned by the model is untrusted. Provenance is
    # assigned only by this module after validation.
    for candidate in candidates:
        candidate["_selection_priority"] = "standard"
    trusted_ocr = any(
        isinstance(reference, dict)
        and str(reference.get("confidence", "")).lower() == "high"
        and str(reference.get("reference_policy", "hint")) in {
            "direct", "composed", "repair_or_compress"
        }
        for reference in original.get("_ocr_translation_references", [])
    )
    # Trusted human subtitles already contribute a candidate after independent
    # auditing. Do not retry merely to invent three machine alternatives.
    minimum = 1 if is_tiny(str(original.get("text", ""))) or trusted_ocr else 2
    if not minimum <= len(candidates) <= 5 or len(raw) > 5:
        raise ValueError(f"candidate_count:{len(candidates)} required:{minimum}-5")
    warnings = []
    if len({len(c["text"]) for c in candidates}) < min(3, len(candidates)):
        warnings.append("candidate_lengths_similar")
    return {"text": primary, "translations_candidates": candidates,
            "translation_status": "success", "translation_error": "",
            "translation_warnings": warnings}


def _restore_backend_candidate_priorities(validated: dict, cached: dict) -> None:
    """Restore only provenance previously assigned by this backend.

    Model-supplied candidate metadata is discarded by ``validate_translation``.
    A resumable checkpoint, however, already contains backend-validated OCR
    candidates and must retain their selection priority on a cache hit.
    """
    allowed = {"ocr_direct", "ocr_repaired", "ocr_hybrid", "ocr_reference", "standard"}
    priorities = {
        _canonical(candidate.get("text")): candidate.get("_selection_priority")
        for candidate in cached.get("translations_candidates", [])
        if isinstance(candidate, dict)
        and candidate.get("_selection_priority") in allowed
    }
    for candidate in validated.get("translations_candidates", []):
        priority = priorities.get(_canonical(candidate.get("text")))
        if priority in allowed:
            candidate["_selection_priority"] = priority


def segment_fingerprint(segment: dict, previous: str, following: str, source: str,
                        target: str, model: str,
                        entity_glossary: list[dict] | None = None) -> str:
    payload = {"rules": RULE_VERSION, "language_check": LANGUAGE_VERSION, "prompt": PROMPT_VERSION,
               "source": language_code(source), "target": language_code(target), "model": model,
               "segment": {k: segment.get(k) for k in
                           ("id", "text", "start", "end", "speaker", "translation_allow_identity")},
               "previous_text": previous, "next_text": following,
               "entity_glossary": entity_glossary or [],
               "ocr_translation_references": segment.get("_ocr_translation_references", []),
               "ocr_source_spelling_hints": segment.get("_ocr_source_spelling_hints", [])}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def translate_segments(segments: list[dict], source: str, target: str, model: str,
                       request: Callable | None, *, batch_size: int = 10, retries: int = 2,
                       cached: list[dict] | None = None, force_ids: set[str] | None = None,
                       checkpoint: Callable | None = None,
                       entity_glossary: list[dict] | None = None) -> list[dict]:
    ids = [str(s.get("id", i)) for i, s in enumerate(segments)]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate segment IDs")
    previous = {str(s.get("id", i)): s for i, s in enumerate(cached or [])}
    rows, pending, inputs = [], [], {}
    for i, segment in enumerate(segments):
        sid = ids[i]
        before = str(segments[i-1].get("text", "")) if i else ""
        after = str(segments[i+1].get("text", "")) if i+1 < len(segments) else ""
        required_entities = applicable_glossary_entries(segment, entity_glossary)
        fp = segment_fingerprint(
            segment, before, after, source, target, model, required_entities
        )
        # Prompt-only keys are intentionally not copied into persisted/public rows.
        public_segment = {
            key: value for key, value in segment.items()
            if not str(key).startswith("_")
        }
        row = dict(public_segment, original_text=segment.get("text", ""), text="",
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
                validated = validate_translation(known, segment, source, target)
                _restore_backend_candidate_priorities(validated, known)
                row.update(validated)
            row["translation_attempts"] = known.get("translation_attempts", 0)
            row["translation_cache_hit"] = True
        except ValueError:
            pending.append(i)
        rows.append(row)
        references = []
        for reference in segment.get("_ocr_translation_references", [])[:3]:
            if not isinstance(reference, dict):
                continue
            target_text = str(reference.get("target_text", "")).strip()[:240]
            source_text = str(reference.get("source_text", "")).strip()[:240]
            if target_text and source_text:
                references.append({
                    "source_subtitle": source_text,
                    "target_subtitle_reference": target_text,
                    "confidence": str(reference.get("confidence", "medium")),
                    "reference_tier": str(reference.get("reference_tier", "trusted")),
                    "reference_policy": str(reference.get("reference_policy", "hint")),
                    "source_coverage": reference.get("source_coverage"),
                    "missing_rate": reference.get("missing_rate"),
                    "extra_rate": reference.get("extra_rate"),
                    "composed": bool(reference.get("composed", False)),
                    "subtitle_cards": reference.get("subtitle_cards", [])[:8],
                    "source_clauses": reference.get("source_clauses", [])[:12],
                    "covered_clause_indices": reference.get("covered_clause_indices", [])[:12],
                    "uncovered_source_clauses": reference.get("uncovered_source_clauses", [])[:12],
                    "usage_policy": (
                        "preferred_if_complete"
                        if (
                            str(reference.get("confidence", "")).lower() == "high"
                            and str(reference.get("reference_policy", "hint")) != "hint"
                        )
                        else "terminology_hint"
                    ),
                })
        source_spelling_hints = []
        for hint in segment.get("_ocr_source_spelling_hints", [])[:5]:
            if not isinstance(hint, dict):
                continue
            text = str(hint.get("source_text", "")).strip()[:240]
            if text:
                source_spelling_hints.append({
                    "source_ocr_text": text,
                    "source_similarity": hint.get("source_similarity"),
                    "spelling_similarity": hint.get("spelling_similarity"),
                    "usage_policy": "source_spelling_only",
                })
        trusted_ocr = any(
            reference.get("usage_policy") == "preferred_if_complete"
            for reference in references
        )
        inputs[i] = {"id": sid, "speaker": segment.get("speaker", "Unknown"),
                     "text_asr_corrected": segment.get("text", ""),
                     "previous_text": before, "next_text": after,
                     "duration": float(segment["end"])-float(segment["start"]),
                     "minimum_candidates": (
                         1 if is_tiny(str(segment.get("text", ""))) or trusted_ocr else 2
                     ),
                     "entity_glossary": required_entities,
                     "ocr_translation_references": references,
                     "ocr_source_spelling_hints": source_spelling_hints}

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
                response_row = response[ids[i]]
                valid = validate_translation(response_row, segments[i], source, target)
                required_entities = inputs[i]["entity_glossary"]
                validate_entity_glossary(valid["text"], required_entities)
                for candidate in valid["translations_candidates"]:
                    validate_entity_glossary(candidate["text"], required_entities)
                preferred = next(
                    (reference for reference in inputs[i]["ocr_translation_references"]
                     if reference.get("usage_policy") == "preferred_if_complete"),
                    None,
                )
                model_ocr_action = str(
                    response_row.get("ocr_reference_action", "semantic_hint")
                )
                ocr_action = model_ocr_action
                admitted_ocr: list[tuple[str, str, str]] = []
                rejected_ocr: list[str] = []
                reference_policy = str(
                    preferred.get("reference_policy", "hint") if preferred else "hint"
                )
                # Source-exact subtitles always remain visible as evidence, but
                # receive trusted priority only after an independent audit.
                if preferred and reference_policy == "direct":
                    raw_trusted, raw_audit_reason = _ocr_audit_acceptance(
                        response_row, "raw"
                    )
                    raw_text, raw_punctuation_ready, raw_punctuation_reason = (
                        _punctuation_ready_text(
                            preferred["target_subtitle_reference"],
                            segments[i].get("text", ""),
                            target,
                            _ocr_audit_verdict(response_row, "raw") if raw_trusted else {},
                            required_entities,
                        )
                    )
                    trusted_direct = raw_trusted and raw_punctuation_ready
                    admitted_ocr.append((
                        raw_text,
                        "ocr_direct" if trusted_direct else "ocr_reference",
                        (
                            f"exact_source_boundary:{raw_punctuation_reason}"
                            if trusted_direct
                            else (
                                raw_audit_reason
                                if not raw_trusted
                                else f"punctuation_not_ready:{raw_punctuation_reason}"
                            )
                        ),
                    ))
                    ocr_action = (
                        "backend_direct" if trusted_direct else "backend_reference_only"
                    )
                elif preferred and reference_policy == "repair_or_compress":
                    # Boundary evidence is strong but not source-exact. Let the
                    # independent semantic audit decide whether the human line
                    # is already complete; do not depend on the translator's
                    # self-reported action alone.
                    raw_trusted, raw_audit_reason = _ocr_audit_acceptance(
                        response_row, "raw"
                    )
                    raw_text, raw_punctuation_ready, raw_punctuation_reason = (
                        _punctuation_ready_text(
                            preferred["target_subtitle_reference"],
                            segments[i].get("text", ""),
                            target,
                            _ocr_audit_verdict(response_row, "raw") if raw_trusted else {},
                            required_entities,
                        )
                    )
                    trusted_raw = raw_trusted and raw_punctuation_ready
                    admitted_ocr.append((
                        raw_text,
                        "ocr_repaired" if trusted_raw else "ocr_reference",
                        (
                            f"trusted_semantic_boundary:{raw_punctuation_reason}"
                            if trusted_raw
                            else (
                                raw_audit_reason
                                if not raw_trusted
                                else f"punctuation_not_ready:{raw_punctuation_reason}"
                            )
                        ),
                    ))
                    ocr_action = (
                        "backend_trusted_raw" if trusted_raw else "backend_reference_only"
                    )

                repaired = str(response_row.get("ocr_repaired_text", "") or "").strip()
                hybrid = str(response_row.get("ocr_hybrid_text", "") or "").strip()
                if (
                    preferred
                    and reference_policy == "composed"
                    and (preferred.get("uncovered_source_clauses") or [])
                    and not hybrid
                ):
                    rejected_ocr.append("hybrid_missing_for_uncovered_source")
                if preferred and reference_policy == "composed" and hybrid:
                    hybrid_trusted, hybrid_audit_reason = _ocr_audit_acceptance(
                        response_row, "hybrid"
                    )
                    if hybrid_trusted:
                        hybrid_text, punctuation_ready, punctuation_reason = (
                            _punctuation_ready_text(
                                hybrid,
                                segments[i].get("text", ""),
                                target,
                                _ocr_audit_verdict(response_row, "hybrid"),
                                required_entities,
                            )
                        )
                        if punctuation_ready:
                            admitted_ocr.append((
                                hybrid_text,
                                "ocr_hybrid",
                                f"clause_aligned_hybrid:{punctuation_reason}",
                            ))
                        else:
                            rejected_ocr.append(
                                f"hybrid={hybrid!r}:punctuation_not_ready:{punctuation_reason}"
                            )
                    else:
                        rejected_ocr.append(f"hybrid={hybrid!r}:{hybrid_audit_reason}")
                elif preferred and model_ocr_action == "minimal_repair" and repaired:
                    repaired_trusted, repaired_audit_reason = _ocr_audit_acceptance(
                        response_row, "repaired"
                    )
                    repair_distance = _ocr_repair_distance(
                        preferred["target_subtitle_reference"], repaired
                    )
                    if repair_distance > 0.18:
                        rejected_ocr.append(
                            f"repaired={repaired!r}:repair_too_extensive:{repair_distance:.3f}"
                        )
                    elif repaired_trusted:
                        repaired_text, punctuation_ready, punctuation_reason = (
                            _punctuation_ready_text(
                                repaired,
                                segments[i].get("text", ""),
                                target,
                                _ocr_audit_verdict(response_row, "repaired"),
                                required_entities,
                            )
                        )
                        if punctuation_ready:
                            admitted_ocr.append((
                                repaired_text,
                                "ocr_hybrid" if reference_policy == "composed" else "ocr_repaired",
                                f"minimal_repair:{punctuation_reason}:edit={repair_distance:.3f}",
                            ))
                        else:
                            rejected_ocr.append(
                                f"repaired={repaired!r}:punctuation_not_ready:{punctuation_reason}"
                            )
                    else:
                        rejected_ocr.append(f"repaired={repaired!r}:{repaired_audit_reason}")
                elif preferred and reference_policy == "composed" and repaired:
                    repaired_trusted, repaired_audit_reason = _ocr_audit_acceptance(
                        response_row, "repaired"
                    )
                    repair_distance = _ocr_repair_distance(
                        preferred["target_subtitle_reference"], repaired
                    )
                    if repair_distance > 0.18:
                        rejected_ocr.append(
                            f"repaired={repaired!r}:repair_too_extensive:{repair_distance:.3f}"
                        )
                    elif repaired_trusted:
                        repaired_text, punctuation_ready, punctuation_reason = (
                            _punctuation_ready_text(
                                repaired,
                                segments[i].get("text", ""),
                                target,
                                _ocr_audit_verdict(response_row, "repaired"),
                                required_entities,
                            )
                        )
                        if punctuation_ready:
                            admitted_ocr.append((
                                repaired_text,
                                "ocr_hybrid",
                                f"composed_repair:{punctuation_reason}:edit={repair_distance:.3f}",
                            ))
                        else:
                            rejected_ocr.append(
                                f"repaired={repaired!r}:punctuation_not_ready:{punctuation_reason}"
                            )
                    else:
                        rejected_ocr.append(f"repaired={repaired!r}:{repaired_audit_reason}")

                # A fully covered composed card sequence may also stand on its
                # own. Partial sequences require the model-generated hybrid.
                if preferred and reference_policy == "composed":
                    uncovered = preferred.get("uncovered_source_clauses") or []
                    coverage = float(preferred.get("source_coverage") or 0.0)
                    if not uncovered and coverage >= 0.95:
                        raw_trusted, raw_audit_reason = _ocr_audit_acceptance(
                            response_row, "raw"
                        )
                        raw_text, punctuation_ready, punctuation_reason = (
                            _punctuation_ready_text(
                                preferred["target_subtitle_reference"],
                                segments[i].get("text", ""),
                                target,
                                _ocr_audit_verdict(response_row, "raw") if raw_trusted else {},
                                required_entities,
                            )
                        )
                        trusted_composed = raw_trusted and punctuation_ready
                        admitted_ocr.append((
                            raw_text,
                            "ocr_hybrid" if trusted_composed else "ocr_reference",
                            (
                                f"complete_composed_cards:{punctuation_reason}"
                                if trusted_composed
                                else (
                                    raw_audit_reason
                                    if not raw_trusted
                                    else f"punctuation_not_ready:{punctuation_reason}"
                                )
                            ),
                        ))

                accepted_candidates: list[dict[str, str]] = []
                admission_notes: list[str] = []
                seen_ocr = set()
                for raw_candidate, priority, admission_reason in admitted_ocr:
                    try:
                        ocr_candidate = validate_target_text(
                            raw_candidate,
                            target,
                            segments[i].get("text", ""),
                            source,
                            allow_identity=bool(segments[i].get("translation_allow_identity")),
                        )
                        validate_ocr_text_quality(ocr_candidate)
                        validate_entity_glossary(ocr_candidate, required_entities)
                    except ValueError as exc:
                        rejected_ocr.append(
                            f"candidate={str(raw_candidate)[:120]!r}:{admission_reason}:{exc}"
                        )
                    else:
                        key = _canonical(ocr_candidate)
                        if key and key not in seen_ocr:
                            accepted_candidates.append({
                                "text": ocr_candidate,
                                "_selection_priority": priority,
                            })
                            admission_notes.append(
                                f"{priority}={ocr_candidate[:120]!r}:{admission_reason}"
                            )
                            seen_ocr.add(key)
                if accepted_candidates:
                    candidates = list(accepted_candidates)
                    seen = set(seen_ocr)
                    for candidate in valid["translations_candidates"]:
                        key = _canonical(candidate["text"])
                        if key not in seen:
                            candidates.append(candidate)
                            seen.add(key)
                    trusted_primary = next(
                        (
                            candidate for candidate in accepted_candidates
                            if candidate.get("_selection_priority") != "ocr_reference"
                        ),
                        None,
                    )
                    if trusted_primary:
                        valid["text"] = trusted_primary["text"]
                    valid["translations_candidates"] = candidates[:5]
                elif preferred and response_row.get("ocr_reference_complete") is True:
                    valid["translation_warnings"].append(
                        "ocr_complete_reference_used_as_semantic_hint"
                    )
                for reason in rejected_ocr:
                    valid["translation_warnings"].append(f"ocr_candidate_rejected:{reason}")
                rows[i].update(valid)
                for warning in valid["translation_warnings"]:
                    if warning != "candidate_lengths_similar":
                        print(f"⚠️ translation segment={ids[i]} {warning}")
                if preferred:
                    trusted_count = sum(
                        candidate.get("_selection_priority") != "ocr_reference"
                        for candidate in accepted_candidates
                    )
                    reference_count = sum(
                        candidate.get("_selection_priority") == "ocr_reference"
                        for candidate in accepted_candidates
                    )
                    print(
                        f"[Translation {ids[i]}] OCR={preferred.get('reference_tier', 'trusted')} "
                        f"policy={reference_policy} action={ocr_action} "
                        f"coverage={preferred.get('source_coverage', 'n/a')} "
                        f"trusted_selectable={trusted_count} reference_only={reference_count} "
                        f"admission={','.join(admission_notes) if admission_notes else 'none'} "
                        f"rejected={','.join(rejected_ocr) if rejected_ocr else 'none'} "
                        f"entity={'consistent' if required_entities else 'not_applicable'}"
                    )
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
