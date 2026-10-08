"""Pure C8 model-request construction and validation.

The public functions accept only a sanitized request and its deterministic C4
r4 inventory. No human label, reference, acoustic decision, or parent model
response is accepted at this boundary.
"""
from __future__ import annotations

import json
from typing import Any, Mapping

REQUEST_FIELDS = {"id", "source_language", "qwen_asr", "ocr_candidates", "contrast_candidates"}
CANDIDATE_FIELDS = {
    "candidate_id", "original_span", "replacement", "alignment_status", "evidence",
    "asr_start", "asr_end", "applied_text",
}
EVIDENCE_FIELDS = {"line_id", "quote"}


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def exact_application(text: str, start: int, end: int, original: str, replacement: str) -> str:
    if not isinstance(text, str) or not text:
        raise ValueError("invalid_asr")
    if not _integer(start) or not _integer(end):
        raise ValueError("offsets_must_be_codepoint_integers")
    if not 0 <= start < end <= len(text):
        raise ValueError("offset_out_of_range_or_empty_original")
    if not isinstance(original, str) or not original or text[start:end] != original:
        raise ValueError("original_span_does_not_match_offset")
    if not isinstance(replacement, str) or not replacement:
        raise ValueError("empty_or_invalid_replacement")
    return text[:start] + replacement + text[end:]


def build_model_item(request: Mapping[str, Any], inventory: Mapping[str, Any], *, inventory_impl) -> dict[str, Any]:
    inventory_impl.validate_request(request)
    inventory_impl.validate_inventory(request, inventory)
    if canonical(inventory_impl.build_inventory(request)) != canonical(inventory):
        raise ValueError("inventory_differs_from_deterministic_regeneration")
    candidates = []
    for item in inventory["candidates"]:
        if item["proposal_status"] != "classifiable":
            continue
        applied = exact_application(
            request["qwen_asr"], item["asr_start"], item["asr_end"],
            item["original_span"], item["replacement"],
        )
        candidates.append({
            "candidate_id": item["candidate_id"],
            "original_span": item["original_span"],
            "replacement": item["replacement"],
            "alignment_status": item["alignment_status"],
            "evidence": [
                {"line_id": evidence["line_id"], "quote": evidence["quote"]}
                for evidence in item["evidence"]
            ],
            "asr_start": item["asr_start"],
            "asr_end": item["asr_end"],
            "applied_text": applied,
        })
    result = {key: request[key] for key in ("id", "source_language", "qwen_asr", "ocr_candidates")}
    result["contrast_candidates"] = candidates
    validate_model_item(result, inventory=inventory, inventory_impl=inventory_impl)
    return result


def validate_model_item(item: Any, *, inventory: Mapping[str, Any], inventory_impl, require_candidates: bool = False) -> None:
    if not isinstance(item, Mapping) or set(item) != REQUEST_FIELDS:
        raise ValueError("model_item_fields_not_exact")
    request = {key: item[key] for key in ("id", "source_language", "qwen_asr", "ocr_candidates")}
    inventory_impl.validate_request(request)
    inventory_impl.validate_inventory(request, inventory)
    candidates = item["contrast_candidates"]
    if not isinstance(candidates, list) or (require_candidates and not candidates):
        raise ValueError("missing_contrast_candidates")
    expected = [row for row in inventory["candidates"] if row["proposal_status"] == "classifiable"]
    if len(candidates) != len(expected):
        raise ValueError("candidate_coverage_mismatch")
    for candidate, source in zip(candidates, expected):
        if not isinstance(candidate, Mapping) or set(candidate) != CANDIDATE_FIELDS:
            raise ValueError("model_candidate_fields_not_exact")
        if candidate["candidate_id"] != source["candidate_id"]:
            raise ValueError("candidate_order_or_identity_mismatch")
        if candidate["alignment_status"] not in {"one_sided", "two_sided"}:
            raise ValueError("invalid_model_alignment")
        evidence = candidate["evidence"]
        expected_evidence = [{"line_id": x["line_id"], "quote": x["quote"]} for x in source["evidence"]]
        if (not isinstance(evidence, list) or any(not isinstance(x, Mapping) or set(x) != EVIDENCE_FIELDS for x in evidence)
                or evidence != expected_evidence):
            raise ValueError("invalid_or_changed_model_evidence")
        for key in ("original_span", "replacement", "asr_start", "asr_end"):
            if candidate[key] != source[key] or type(candidate[key]) is not type(source[key]):
                raise ValueError("candidate_value_differs_from_inventory")
        expected_applied = exact_application(
            item["qwen_asr"], source["asr_start"], source["asr_end"],
            source["original_span"], source["replacement"],
        )
        if candidate["applied_text"] != expected_applied:
            raise ValueError("applied_text_not_exact")


def user_message(item: Mapping[str, Any], *, inventory: Mapping[str, Any], inventory_impl) -> str:
    validate_model_item(item, inventory=inventory, inventory_impl=inventory_impl, require_candidates=True)
    return json.dumps({"items": [item]}, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
