"""C4 r2: predeclared precision filter for r1 one-sided candidates."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Mapping


R1_PATH = Path(__file__).with_name("v12_c4_candidate_inventory.py")
SPEC = importlib.util.spec_from_file_location("c4_inventory_r1_for_r2", R1_PATH)
r1 = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = r1
SPEC.loader.exec_module(r1)

ALLOWED_REQUEST_FIELDS = r1.ALLOWED_REQUEST_FIELDS
MIN_ONE_SIDED_CONTENT_ANCHOR = 3


def canonical_bytes(value: Any) -> bytes:
    return r1.canonical_bytes(value)


def tokenize(text: str):
    return r1.tokenize(text)


def _adjacent_matches(left: list[dict[str, Any]], right: list[dict[str, Any]], *, reverse: bool) -> int:
    if reverse:
        left = list(reversed(left))
        right = list(reversed(right))
    count = 0
    for one, two in zip(left, right):
        if one["key"] != two["key"]:
            break
        if one["kind"] != "punct":
            count += 1
    return count


def _one_sided_support(asr: str, ocr: str, candidate: Mapping[str, Any], evidence: Mapping[str, Any]) -> int:
    asr_tokens = r1.tokenize(asr)
    ocr_tokens = r1.tokenize(ocr)
    asr_left = [token for token in asr_tokens if token["end"] <= candidate["asr_start"]]
    asr_right = [token for token in asr_tokens if token["start"] >= candidate["asr_end"]]
    ocr_left = [token for token in ocr_tokens if token["end"] <= evidence["ocr_start"]]
    ocr_right = [token for token in ocr_tokens if token["start"] >= evidence["ocr_end"]]
    return max(
        _adjacent_matches(asr_left, ocr_left, reverse=True),
        _adjacent_matches(asr_right, ocr_right, reverse=False),
    )


def _relocates_existing_text(asr: str, candidate: Mapping[str, Any]) -> bool:
    replacement = candidate["replacement"]
    if not replacement:
        return False
    haystack, needle = asr.lower(), replacement.lower()
    start = 0
    while True:
        index = haystack.find(needle, start)
        if index < 0:
            return False
        if index + len(needle) <= candidate["asr_start"] or index >= candidate["asr_end"]:
            return True
        start = index + 1


def build_inventory(request: Mapping[str, Any]) -> dict[str, Any]:
    result = r1.build_inventory(request)
    lines = {str(line["line_id"]): str(line["text"]) for line in request["ocr_candidates"]}
    filtered = 0
    for candidate in result["candidates"]:
        if candidate["proposal_status"] != "classifiable" or candidate["alignment_status"] != "one_sided":
            continue
        support = max(
            _one_sided_support(request["qwen_asr"], lines[evidence["line_id"]], candidate, evidence)
            for evidence in candidate["evidence"]
        )
        candidate["one_sided_content_anchor"] = support
        reasons = []
        if support < MIN_ONE_SIDED_CONTENT_ANCHOR:
            reasons.append("weak_one_sided_anchor")
        if _relocates_existing_text(request["qwen_asr"], candidate):
            reasons.append("relocated_existing_text")
        if reasons:
            candidate["proposal_status"] = "diagnostic_only"
            candidate["diagnostic_reasons"] = sorted(set(candidate["diagnostic_reasons"] + reasons))
            filtered += 1
    classifiable = [item for item in result["candidates"] if item["proposal_status"] == "classifiable"]
    payload = dict(request)
    payload["contrast_candidates"] = classifiable
    request_bytes = len(r1.canonical_bytes(payload))
    capacity_reasons = []
    if len(classifiable) > r1.MAX_CLASSIFIABLE:
        capacity_reasons.append("classifiable_candidate_limit")
    if request_bytes > r1.MAX_REQUEST_BYTES:
        capacity_reasons.append("request_byte_limit")
    result["capacity_blocked"] = bool(capacity_reasons)
    result["capacity_reasons"] = capacity_reasons
    result["request_bytes_with_classifiable_candidates"] = request_bytes
    result["diagnostics"]["r2_one_sided_filtered"] = filtered
    result["diagnostics"]["classifiable_candidate_count"] = len(classifiable)
    result["diagnostics"]["diagnostic_only_candidate_count"] = len(result["candidates"]) - len(classifiable)
    result["r2_policy"] = {
        "minimum_one_sided_content_anchor": MIN_ONE_SIDED_CONTENT_ANCHOR,
        "reject_relocated_existing_text": True,
        "two_sided_rules_changed": False,
    }
    r1.validate_inventory(request, result)
    return result


validate_inventory = r1.validate_inventory
expand_unique_context = r1.expand_unique_context
parse_classification_response = r1.parse_classification_response
judgment_eligible = r1.judgment_eligible
