"""C3 structured contrast adapter. Design-stage code; no efficacy claim.

The public decision input is limited to ASR, selected OCR lines, and one C3
assessment. Reference text, audit labels, opportunity groups, and scores are not
accepted. Eligible contrasts are projected to the frozen P2/C1R path.
"""
from __future__ import annotations

import copy
import json
import math
from typing import Any, Iterable, Mapping


TOP_FIELDS = {"assessments"}
ASSESSMENT_FIELDS = {"id", "contrasts"}
CONTRAST_FIELDS = {
    "original_span", "replacement", "difference_types", "correspondence",
    "evidence", "error_evidence", "confidence",
}
EVIDENCE_FIELDS = {"line_id", "quote"}
ERROR_EVIDENCE_FIELDS = {"basis", "asr_alternative", "observation"}
DIFFERENCE_TYPES = {
    "surface_form", "subtitle_rewrite", "content_alternative",
    "lexical_repair", "unresolved",
}
CORRESPONDENCE = {"local", "conflicting", "unlocated"}
ERROR_BASES = {"broken_fragment", "context_disambiguated", "subtitle_difference_only", "none"}
ASR_ALTERNATIVES = {"ruled_out", "plausible", "unknown"}
TYPE_VETOES = {"surface_form", "subtitle_rewrite", "unresolved"}
ELIGIBLE_BASES = {"broken_fragment", "context_disambiguated"}


class DuplicateKeyError(ValueError):
    """Raised when a JSON object contains duplicate keys."""


def _unique_object(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise DuplicateKeyError(f"duplicate_json_key:{key}")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise ValueError(f"invalid_json_constant:{value}")


def strict_json_loads(text: str) -> Any:
    """Parse JSON while rejecting duplicate keys and NaN/Infinity."""
    if not isinstance(text, str):
        raise TypeError("response_text_must_be_string")
    return json.loads(text, object_pairs_hook=_unique_object, parse_constant=_reject_constant)


def parse_batch_response(text: str, requested_ids: Iterable[str]) -> dict[str, dict[str, Any]]:
    """Parse one complete C3 response and require exact requested-ID coverage."""
    payload = strict_json_loads(text)
    if not isinstance(payload, dict) or set(payload) != TOP_FIELDS:
        raise ValueError("invalid_top_level_fields")
    assessments = payload["assessments"]
    if not isinstance(assessments, list):
        raise ValueError("assessments_must_be_list")
    expected = list(requested_ids)
    if any(not isinstance(value, str) or not value for value in expected) or len(expected) != len(set(expected)):
        raise ValueError("invalid_or_duplicate_requested_ids")
    returned: dict[str, dict[str, Any]] = {}
    for assessment in assessments:
        if not isinstance(assessment, dict) or set(assessment) != ASSESSMENT_FIELDS:
            raise ValueError("invalid_assessment_fields")
        sample_id = assessment.get("id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("invalid_assessment_id")
        if sample_id in returned:
            raise ValueError(f"duplicate_assessment_id:{sample_id}")
        returned[sample_id] = assessment
    unexpected = sorted(set(returned) - set(expected))
    missing = sorted(set(expected) - set(returned))
    if unexpected:
        raise ValueError(f"unrequested_assessment_ids:{unexpected}")
    if missing:
        raise ValueError(f"missing_assessment_ids:{missing}")
    return returned


def occurrences(text: str, needle: str) -> list[int]:
    """Return overlapping exact occurrences."""
    if not needle:
        return []
    result: list[int] = []
    start = 0
    while True:
        index = text.find(needle, start)
        if index < 0:
            return result
        result.append(index)
        start = index + 1


def _validate_inputs(qwen: Any, selected_lines: Any) -> tuple[list[dict[str, str]] | None, str | None]:
    if not isinstance(qwen, str) or not qwen:
        return None, "invalid_asr"
    if not isinstance(selected_lines, list):
        return None, "invalid_ocr"
    lines: list[dict[str, str]] = []
    seen: set[str] = set()
    for line in selected_lines:
        if not isinstance(line, dict) or set(line) != {"line_id", "text"}:
            return None, "invalid_ocr_line_fields"
        line_id, text = line["line_id"], line["text"]
        if not isinstance(line_id, str) or not line_id or not isinstance(text, str):
            return None, "invalid_ocr_line"
        if line_id in seen:
            return None, "duplicate_ocr_line_id"
        seen.add(line_id)
        lines.append({"line_id": line_id, "text": text})
    return lines, None


def _structural_reject(qwen: Any, reason: str, contrasts: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "accepted": False,
        "corrected_text": qwen if isinstance(qwen, str) else "",
        "reason": reason,
        "structural_error": reason,
        "contrasts": contrasts or [],
        "eligible_indices": [],
        "p2_projection": None,
        "downstream_decision": None,
    }


def _ineligibility_reasons(contrast: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    if contrast["original_span"] == contrast["replacement"]:
        reasons.append("no_text_change")
    if contrast["confidence"] < 0.85:
        reasons.append("confidence_below_0.85")
    if contrast["correspondence"] != "local":
        reasons.append(f"correspondence_{contrast['correspondence']}")
    vetoes = sorted(set(contrast["difference_types"]) & TYPE_VETOES)
    if vetoes:
        reasons.append("typed_veto:" + ",".join(vetoes))
    error = contrast["error_evidence"]
    if error["basis"] not in ELIGIBLE_BASES:
        reasons.append(f"error_basis_{error['basis']}")
    if error["asr_alternative"] != "ruled_out":
        reasons.append(f"asr_alternative_{error['asr_alternative']}")
    return reasons


def decide(qwen: Any, selected_lines: Any, assessment: Any, *, c1r, core, a1, p3, p4) -> dict[str, Any]:
    """Validate, type-gate, project, and run the existing C1R/P3/P4 path."""
    lines, input_error = _validate_inputs(qwen, selected_lines)
    if input_error:
        return _structural_reject(qwen, input_error)
    assert lines is not None
    line_map = {line["line_id"]: line["text"] for line in lines}

    if not isinstance(assessment, dict) or set(assessment) != ASSESSMENT_FIELDS:
        return _structural_reject(qwen, "invalid_assessment_fields")
    if not isinstance(assessment["id"], str) or not assessment["id"]:
        return _structural_reject(qwen, "invalid_assessment_id")
    raw_contrasts = assessment["contrasts"]
    if not isinstance(raw_contrasts, list):
        return _structural_reject(qwen, "contrasts_must_be_list")

    parsed: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_contrasts):
        prefix = f"contrast_{index}"
        if not isinstance(raw, dict) or set(raw) != CONTRAST_FIELDS:
            return _structural_reject(qwen, f"{prefix}:invalid_fields", parsed)
        if not isinstance(raw["original_span"], str) or not raw["original_span"]:
            return _structural_reject(qwen, f"{prefix}:invalid_original_span", parsed)
        if not isinstance(raw["replacement"], str):
            return _structural_reject(qwen, f"{prefix}:invalid_replacement", parsed)
        positions = occurrences(qwen, raw["original_span"])
        if len(positions) != 1:
            return _structural_reject(qwen, f"{prefix}:nonunique_original_span", parsed)

        difference_types = raw["difference_types"]
        if (not isinstance(difference_types, list) or not difference_types or
                any(not isinstance(value, str) or value not in DIFFERENCE_TYPES for value in difference_types) or
                len(difference_types) != len(set(difference_types))):
            return _structural_reject(qwen, f"{prefix}:invalid_difference_types", parsed)
        if raw["correspondence"] not in CORRESPONDENCE:
            return _structural_reject(qwen, f"{prefix}:invalid_correspondence", parsed)

        evidence = raw["evidence"]
        if not isinstance(evidence, list):
            return _structural_reject(qwen, f"{prefix}:evidence_must_be_list", parsed)
        evidence_seen: set[tuple[str, str]] = set()
        for item in evidence:
            if not isinstance(item, dict) or set(item) != EVIDENCE_FIELDS:
                return _structural_reject(qwen, f"{prefix}:invalid_evidence_fields", parsed)
            line_id, quote = item["line_id"], item["quote"]
            if not isinstance(line_id, str) or line_id not in line_map or not isinstance(quote, str) or not quote:
                return _structural_reject(qwen, f"{prefix}:invalid_evidence_reference", parsed)
            if quote not in line_map[line_id]:
                return _structural_reject(qwen, f"{prefix}:quote_not_in_line", parsed)
            key = (line_id, quote)
            if key in evidence_seen:
                return _structural_reject(qwen, f"{prefix}:duplicate_evidence", parsed)
            evidence_seen.add(key)
        if raw["correspondence"] == "local" and not evidence:
            return _structural_reject(qwen, f"{prefix}:local_without_evidence", parsed)

        error = raw["error_evidence"]
        if not isinstance(error, dict) or set(error) != ERROR_EVIDENCE_FIELDS:
            return _structural_reject(qwen, f"{prefix}:invalid_error_evidence_fields", parsed)
        if error["basis"] not in ERROR_BASES or error["asr_alternative"] not in ASR_ALTERNATIVES:
            return _structural_reject(qwen, f"{prefix}:invalid_error_evidence_enum", parsed)
        observation = error["observation"]
        if not isinstance(observation, str) or not observation or len(observation) > 240:
            return _structural_reject(qwen, f"{prefix}:invalid_observation", parsed)
        confidence = raw["confidence"]
        if (isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or
                not math.isfinite(confidence) or not 0 <= confidence <= 1):
            return _structural_reject(qwen, f"{prefix}:invalid_confidence", parsed)

        start = positions[0]
        item = copy.deepcopy(raw)
        reasons = _ineligibility_reasons(item)
        parsed.append({
            "contrast_index": index,
            "start": start,
            "end": start + len(item["original_span"]),
            "assessment": item,
            "eligible": not reasons,
            "eligibility_reasons": reasons,
            "projected_edit_index": None,
            "downstream": None,
        })

    ordered = sorted(parsed, key=lambda value: value["start"])
    if any(left["end"] > right["start"] for left, right in zip(ordered, ordered[1:])):
        return _structural_reject(qwen, "overlapping_contrasts", parsed)

    projected_edits: list[dict[str, Any]] = []
    for item in parsed:
        if not item["eligible"]:
            continue
        contrast = item["assessment"]
        line_ids = list(dict.fromkeys(value["line_id"] for value in contrast["evidence"]))
        item["projected_edit_index"] = len(projected_edits)
        projected_edits.append({
            "original_span": contrast["original_span"],
            "replacement": contrast["replacement"],
            "evidence_line_ids": line_ids,
            "confidence": contrast["confidence"],
            "reason": contrast["error_evidence"]["observation"],
        })
    projection = {
        "id": assessment["id"],
        "action": "CORRECT" if projected_edits else "KEEP",
        "reason": "C3 structured projection",
        "edits": projected_edits,
    }
    eligible_indices = [item["contrast_index"] for item in parsed if item["eligible"]]
    if not projected_edits:
        return {
            "accepted": False,
            "corrected_text": qwen,
            "reason": "no_eligible_contrasts",
            "structural_error": None,
            "contrasts": parsed,
            "eligible_indices": eligible_indices,
            "p2_projection": projection,
            "downstream_decision": None,
        }

    downstream = c1r.decide(qwen, lines, projection, core=core, a1=a1, p3=p3, p4=p4)
    audit_by_projection = {item.get("edit_index"): item for item in downstream.get("edits", [])}
    for item in parsed:
        projection_index = item["projected_edit_index"]
        if projection_index is not None:
            item["downstream"] = copy.deepcopy(audit_by_projection.get(projection_index))
    return {
        "accepted": bool(downstream["accepted"]),
        "corrected_text": downstream["corrected_text"],
        "reason": "downstream:" + str(downstream["reason"]),
        "structural_error": None,
        "contrasts": parsed,
        "eligible_indices": eligible_indices,
        "p2_projection": projection,
        "downstream_decision": copy.deepcopy(downstream),
    }
