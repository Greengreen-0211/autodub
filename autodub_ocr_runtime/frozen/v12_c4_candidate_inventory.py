"""C4 deterministic ASR/OCR candidate inventory and response contract.

This module is deliberately blind to references, audit labels, opportunity
groups, model proposals, and scores. It performs no network or API calls.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import math
from collections import defaultdict
from typing import Any, Iterable, Mapping


ALLOWED_REQUEST_FIELDS = {"id", "source_language", "qwen_asr", "ocr_candidates"}
MAX_TOKENS_PER_SIDE = 12
MAX_CODEPOINTS_PER_SIDE = 80
MAX_CLASSIFIABLE = 24
MAX_REQUEST_BYTES = 24_000
MIN_ANCHOR_TOKENS = 2
MAX_EDGE_TOKENS = 4

DIFFERENCE_TYPES = {
    "surface_form", "subtitle_rewrite", "content_alternative",
    "lexical_repair", "unresolved",
}
CORRESPONDENCE = {"local", "conflicting", "unlocated"}
ERROR_BASES = {"broken_fragment", "context_disambiguated", "subtitle_difference_only", "none"}
ASR_ALTERNATIVES = {"ruled_out", "plausible", "unknown"}
INDEPENDENCE = {"independent", "requires_other_change", "unknown"}
TYPE_VETOES = {"surface_form", "subtitle_rewrite", "unresolved"}
ELIGIBLE_BASES = {"broken_fragment", "context_disambiguated"}


class DuplicateKeyError(ValueError):
    pass


def _unique_object(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateKeyError(f"duplicate_json_key:{key}")
        result[key] = value
    return result


def strict_json_loads(text: str) -> Any:
    if not isinstance(text, str):
        raise TypeError("response_text_must_be_string")
    return json.loads(
        text,
        object_pairs_hook=_unique_object,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"invalid_json_constant:{value}")),
    )


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _is_cjk(char: str) -> bool:
    code = ord(char)
    return (
        0x3400 <= code <= 0x4DBF or 0x4E00 <= code <= 0x9FFF
        or 0x20000 <= code <= 0x2EBEF or 0x30000 <= code <= 0x323AF
    )


def _is_ascii_letter(char: str) -> bool:
    return "A" <= char <= "Z" or "a" <= char <= "z"


def tokenize(text: str) -> list[dict[str, Any]]:
    """Tokenize while preserving Python Unicode code-point offsets."""
    if not isinstance(text, str):
        raise TypeError("text_must_be_string")
    result: list[dict[str, Any]] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char.isspace():
            index += 1
            continue
        start = index
        if _is_cjk(char):
            index += 1
            kind = "cjk"
        elif _is_ascii_letter(char):
            index += 1
            while index < len(text):
                current = text[index]
                if _is_ascii_letter(current):
                    index += 1
                    continue
                if current in {"'", "\u2019"} and index + 1 < len(text) and _is_ascii_letter(text[index + 1]):
                    index += 2
                    while index < len(text) and _is_ascii_letter(text[index]):
                        index += 1
                    continue
                break
            kind = "word"
        elif char.isascii() and char.isdigit():
            index += 1
            while index < len(text) and text[index].isascii() and text[index].isdigit():
                index += 1
            kind = "number"
        else:
            index += 1
            kind = "punct"
        raw = text[start:index]
        result.append({
            "raw": raw,
            "key": raw.lower() if kind == "word" else raw,
            "kind": kind,
            "start": start,
            "end": index,
        })
    return result


def _subsequence_count(keys: list[str], needle: list[str]) -> int:
    if not needle or len(needle) > len(keys):
        return 0
    return sum(keys[index:index + len(needle)] == needle for index in range(len(keys) - len(needle) + 1))


def _content_count(tokens: list[dict[str, Any]]) -> int:
    return sum(token["kind"] != "punct" for token in tokens)


def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def occurrences(text: str, needle: str) -> list[int]:
    if not needle:
        return []
    result: list[int] = []
    start = 0
    while True:
        found = text.find(needle, start)
        if found < 0:
            return result
        result.append(found)
        start = found + 1


def _surface_key(text: str) -> str:
    return "".join(char.lower() for char in text if char.isalnum() or _is_cjk(char))


def _raw_candidate(
    asr: str,
    ocr: str,
    *,
    asr_start: int,
    asr_end: int,
    ocr_start: int,
    ocr_end: int,
    line_id: str,
    alignment_status: str,
) -> dict[str, Any] | None:
    asr_start, asr_end = _trim_span(asr, asr_start, asr_end)
    ocr_start, ocr_end = _trim_span(ocr, ocr_start, ocr_end)
    original = asr[asr_start:asr_end]
    replacement = ocr[ocr_start:ocr_end]
    if original == replacement:
        return None
    asr_tokens = tokenize(original)
    ocr_tokens = tokenize(replacement)
    asr_content = _content_count(asr_tokens)
    ocr_content = _content_count(ocr_tokens)
    operation = "replace" if original and replacement else "insert" if replacement else "delete"
    reasons: list[str] = []
    if operation != "replace":
        reasons.append(operation + "_diagnostic")
    if asr_content == 0 and ocr_content == 0 or _surface_key(original) == _surface_key(replacement):
        reasons.append("surface_only")
    if len(asr_tokens) > MAX_TOKENS_PER_SIDE or len(ocr_tokens) > MAX_TOKENS_PER_SIDE or len(original) > MAX_CODEPOINTS_PER_SIDE or len(replacement) > MAX_CODEPOINTS_PER_SIDE:
        reasons.append("long_span")
    if original and len(occurrences(asr, original)) != 1:
        reasons.append("nonunique_original_span")
    return {
        "asr_start": asr_start,
        "asr_end": asr_end,
        "original_span": original,
        "replacement": replacement,
        "operation": operation,
        "alignment_status": alignment_status,
        "proposal_status": "diagnostic_only" if reasons else "classifiable",
        "diagnostic_reasons": sorted(set(reasons)),
        "evidence": [{
            "line_id": line_id,
            "ocr_start": ocr_start,
            "ocr_end": ocr_end,
            "quote": replacement,
        }],
        "asr_token_count": len(asr_tokens),
        "ocr_token_count": len(ocr_tokens),
    }


def _unique_anchors(asr_tokens: list[dict[str, Any]], ocr_tokens: list[dict[str, Any]]) -> tuple[list[dict[str, int]], int]:
    asr_keys = [token["key"] for token in asr_tokens]
    ocr_keys = [token["key"] for token in ocr_tokens]
    matcher = difflib.SequenceMatcher(None, asr_keys, ocr_keys, autojunk=False)
    anchors: list[dict[str, int]] = []
    ambiguous = 0
    for block in matcher.get_matching_blocks():
        if block.size < MIN_ANCHOR_TOKENS:
            continue
        block_tokens = asr_tokens[block.a:block.a + block.size]
        if _content_count(block_tokens) == 0:
            continue
        keys = asr_keys[block.a:block.a + block.size]
        if _subsequence_count(asr_keys, keys) != 1 or _subsequence_count(ocr_keys, keys) != 1:
            ambiguous += 1
            continue
        anchors.append({"a": block.a, "b": block.b, "size": block.size})
    return anchors, ambiguous


def _enumerate_line(asr: str, line_id: str, ocr: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    asr_tokens = tokenize(asr)
    ocr_tokens = tokenize(ocr)
    anchors, ambiguous = _unique_anchors(asr_tokens, ocr_tokens)
    diagnostics: dict[str, Any] = {
        "line_id": line_id,
        "anchor_count": len(anchors),
        "ambiguous_anchor_blocks": ambiguous,
        "uncovered": [],
    }
    if [token["key"] for token in asr_tokens] == [token["key"] for token in ocr_tokens] and asr != ocr:
        diagnostics["uncovered"].append("folded_surface_difference")
    if not anchors:
        diagnostics["uncovered"].append("no_unique_anchor")
        return [], diagnostics
    candidates: list[dict[str, Any]] = []
    for left, right in zip(anchors, anchors[1:]):
        a0 = left["a"] + left["size"]
        a1 = right["a"]
        b0 = left["b"] + left["size"]
        b1 = right["b"]
        asr_start = asr_tokens[a0]["start"] if a0 < len(asr_tokens) else asr_tokens[-1]["end"]
        asr_end = asr_tokens[a1]["start"] if a1 < len(asr_tokens) else len(asr)
        ocr_start = ocr_tokens[b0]["start"] if b0 < len(ocr_tokens) else ocr_tokens[-1]["end"]
        ocr_end = ocr_tokens[b1]["start"] if b1 < len(ocr_tokens) else len(ocr)
        item = _raw_candidate(
            asr, ocr, asr_start=asr_start, asr_end=asr_end,
            ocr_start=ocr_start, ocr_end=ocr_end, line_id=line_id,
            alignment_status="two_sided",
        )
        if item:
            candidates.append(item)

    first = anchors[0]
    prefix_ocr = ocr_tokens[:first["b"]]
    prefix_n = _content_count(prefix_ocr)
    if prefix_n:
        content_indices = [i for i, token in enumerate(prefix_ocr) if token["kind"] != "punct"]
        if prefix_n <= MAX_EDGE_TOKENS and first["a"] >= prefix_n:
            a0 = first["a"] - prefix_n
            item = _raw_candidate(
                asr, ocr, asr_start=asr_tokens[a0]["start"], asr_end=asr_tokens[first["a"]]["start"],
                ocr_start=prefix_ocr[content_indices[0]]["start"], ocr_end=ocr_tokens[first["b"]]["start"],
                line_id=line_id, alignment_status="one_sided",
            )
            if item:
                candidates.append(item)
        else:
            diagnostics["uncovered"].append("long_boundary_uncovered" if prefix_n > MAX_EDGE_TOKENS else "boundary_uncovered")

    last = anchors[-1]
    a_after = last["a"] + last["size"]
    b_after = last["b"] + last["size"]
    suffix_ocr = ocr_tokens[b_after:]
    suffix_n = _content_count(suffix_ocr)
    if suffix_n:
        content_indices = [i for i, token in enumerate(suffix_ocr) if token["kind"] != "punct"]
        if suffix_n <= MAX_EDGE_TOKENS and len(asr_tokens) - a_after >= suffix_n:
            a1 = a_after + suffix_n
            item = _raw_candidate(
                asr, ocr, asr_start=asr_tokens[a_after]["start"], asr_end=asr_tokens[a1 - 1]["end"],
                ocr_start=ocr_tokens[b_after]["start"], ocr_end=suffix_ocr[content_indices[-1]]["end"],
                line_id=line_id, alignment_status="one_sided",
            )
            if item:
                candidates.append(item)
        else:
            diagnostics["uncovered"].append("long_boundary_uncovered" if suffix_n > MAX_EDGE_TOKENS else "boundary_uncovered")
    return candidates, diagnostics


def _candidate_id(request_hash: str, candidate: Mapping[str, Any]) -> str:
    identity = {
        "request_sha256": request_hash,
        "asr_start": candidate["asr_start"],
        "asr_end": candidate["asr_end"],
        "replacement": candidate["replacement"],
        "evidence": sorted(
            ({key: item[key] for key in ("line_id", "ocr_start", "ocr_end", "quote")} for item in candidate["evidence"]),
            key=lambda item: (item["line_id"], item["ocr_start"], item["ocr_end"], item["quote"]),
        ),
    }
    return "c4_" + sha256(identity)


def validate_inventory(request: Mapping[str, Any], inventory: Mapping[str, Any]) -> None:
    if inventory.get("request_sha256") != sha256(request):
        raise ValueError("request_hash_mismatch")
    asr = request["qwen_asr"]
    lines = {str(line["line_id"]): str(line["text"]) for line in request["ocr_candidates"]}
    ids: set[str] = set()
    for candidate in inventory.get("candidates", []):
        cid = candidate.get("candidate_id")
        if not isinstance(cid, str) or cid in ids:
            raise ValueError("invalid_or_duplicate_candidate_id")
        ids.add(cid)
        start, end = candidate["asr_start"], candidate["asr_end"]
        if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start <= end <= len(asr):
            raise ValueError(f"invalid_asr_offset:{cid}")
        if asr[start:end] != candidate["original_span"]:
            raise ValueError(f"asr_slice_mismatch:{cid}")
        for evidence in candidate["evidence"]:
            text = lines.get(evidence["line_id"])
            if text is None or text[evidence["ocr_start"]:evidence["ocr_end"]] != evidence["quote"]:
                raise ValueError(f"ocr_slice_mismatch:{cid}")
        if cid != _candidate_id(inventory["request_sha256"], candidate):
            raise ValueError(f"candidate_hash_mismatch:{cid}")


def build_inventory(request: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(request, Mapping) or set(request) != ALLOWED_REQUEST_FIELDS:
        raise ValueError("request_fields_must_be_exact_whitelist")
    sample_id = request["id"]
    language = request["source_language"]
    asr = request["qwen_asr"]
    lines = request["ocr_candidates"]
    if not isinstance(sample_id, str) or not sample_id or not isinstance(language, str) or not isinstance(asr, str) or not isinstance(lines, list):
        raise ValueError("invalid_request_values")
    seen_lines: set[str] = set()
    raw: list[dict[str, Any]] = []
    line_diagnostics: list[dict[str, Any]] = []
    for line in lines:
        if not isinstance(line, Mapping) or "line_id" not in line or "text" not in line:
            raise ValueError("invalid_ocr_candidate")
        line_id, text = line["line_id"], line["text"]
        if not isinstance(line_id, str) or not line_id or line_id in seen_lines or not isinstance(text, str):
            raise ValueError("invalid_or_duplicate_ocr_line")
        seen_lines.add(line_id)
        found, diagnostic = _enumerate_line(asr, line_id, text)
        raw.extend(found)
        line_diagnostics.append(diagnostic)

    merged: dict[tuple[int, int, str], dict[str, Any]] = {}
    for item in raw:
        key = (item["asr_start"], item["asr_end"], item["replacement"])
        if key not in merged:
            merged[key] = item
        else:
            merged[key]["evidence"].extend(item["evidence"])
            if item["alignment_status"] != merged[key]["alignment_status"]:
                merged[key]["alignment_status"] = "ambiguous"
            merged[key]["diagnostic_reasons"] = sorted(set(merged[key]["diagnostic_reasons"] + item["diagnostic_reasons"]))

    candidates = list(merged.values())
    for item in candidates:
        item["evidence"] = sorted(
            {tuple(evidence[key] for key in ("line_id", "ocr_start", "ocr_end", "quote")): evidence for evidence in item["evidence"]}.values(),
            key=lambda evidence: (evidence["line_id"], evidence["ocr_start"], evidence["ocr_end"], evidence["quote"]),
        )
    by_span: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for item in candidates:
        by_span[(item["asr_start"], item["asr_end"])].append(item)
    conflict_counter = 0
    for group in by_span.values():
        if len({item["replacement"] for item in group}) > 1:
            conflict_counter += 1
            group_id = f"conflict_{conflict_counter:03d}"
            for item in group:
                item["conflict_group"] = group_id
                item["proposal_status"] = "diagnostic_only"
                item["diagnostic_reasons"] = sorted(set(item["diagnostic_reasons"] + ["conflicting_candidates"]))
        else:
            for item in group:
                item["conflict_group"] = None

    request_hash = sha256(dict(request))
    candidates.sort(key=lambda item: (
        item["asr_start"], item["asr_end"], item["replacement"],
        item["evidence"][0]["line_id"], item["evidence"][0]["ocr_start"], item["evidence"][0]["ocr_end"],
    ))
    for item in candidates:
        item["candidate_id"] = _candidate_id(request_hash, item)

    classifiable = [item for item in candidates if item["proposal_status"] == "classifiable"]
    request_payload = dict(request)
    request_payload["contrast_candidates"] = classifiable
    request_bytes = len(canonical_bytes(request_payload))
    capacity_reasons = []
    if len(classifiable) > MAX_CLASSIFIABLE:
        capacity_reasons.append("classifiable_candidate_limit")
    if request_bytes > MAX_REQUEST_BYTES:
        capacity_reasons.append("request_byte_limit")
    result = {
        "id": sample_id,
        "request_sha256": request_hash,
        "capacity_blocked": bool(capacity_reasons),
        "capacity_reasons": capacity_reasons,
        "request_bytes_with_classifiable_candidates": request_bytes,
        "candidates": candidates,
        "diagnostics": {
            "line_diagnostics": line_diagnostics,
            "raw_candidate_count": len(raw),
            "deduplicated_candidate_count": len(candidates),
            "classifiable_candidate_count": len(classifiable),
            "diagnostic_only_candidate_count": len(candidates) - len(classifiable),
            "conflict_group_count": conflict_counter,
        },
    }
    validate_inventory(request, result)
    return result


def expand_unique_context(asr: str, start: int, end: int, replacement: str) -> dict[str, Any] | None:
    """Expand a located span to the smallest exact unique token-boundary window."""
    if not isinstance(asr, str) or not isinstance(start, int) or not isinstance(end, int) or not isinstance(replacement, str):
        raise TypeError("invalid_bridge_input")
    if not 0 <= start < end <= len(asr):
        return None
    tokens = tokenize(asr)
    inside = [index for index, token in enumerate(tokens) if token["start"] >= start and token["end"] <= end]
    if not inside:
        return None
    first, last = min(inside), max(inside)
    windows: list[tuple[int, int, int, int]] = []
    for left in range(first, -1, -1):
        for right in range(last, len(tokens)):
            raw_start, raw_end = tokens[left]["start"], tokens[right]["end"]
            if raw_start > start or raw_end < end:
                continue
            original = asr[raw_start:raw_end]
            if len(occurrences(asr, original)) == 1:
                extra = (first - left) + (right - last)
                windows.append((extra, raw_end - raw_start, raw_start, raw_end))
    if not windows:
        return None
    _, _, raw_start, raw_end = min(windows)
    return {
        "original_span": asr[raw_start:raw_end],
        "replacement": asr[raw_start:start] + replacement + asr[end:raw_end],
        "start": raw_start,
        "end": raw_end,
    }


def parse_classification_response(text: str, submitted: Mapping[str, Iterable[str]]) -> dict[str, dict[str, Any]]:
    payload = strict_json_loads(text)
    if not isinstance(payload, dict) or set(payload) != {"assessments"} or not isinstance(payload["assessments"], list):
        raise ValueError("invalid_top_level")
    expected = {str(key): list(value) for key, value in submitted.items()}
    returned: dict[str, dict[str, Any]] = {}
    for assessment in payload["assessments"]:
        if not isinstance(assessment, dict) or set(assessment) != {"id", "judgments"}:
            raise ValueError("invalid_assessment_fields")
        sample_id = assessment["id"]
        if not isinstance(sample_id, str) or sample_id in returned or sample_id not in expected or not isinstance(assessment["judgments"], list):
            raise ValueError("invalid_duplicate_or_unknown_assessment")
        seen: set[str] = set()
        for judgment in assessment["judgments"]:
            if not isinstance(judgment, dict) or set(judgment) != {"candidate_id", "difference_types", "correspondence", "error_evidence", "independence", "confidence"}:
                raise ValueError("invalid_judgment_fields")
            candidate_id = judgment["candidate_id"]
            if not isinstance(candidate_id, str) or candidate_id in seen or candidate_id not in expected[sample_id]:
                raise ValueError("invalid_duplicate_or_unknown_candidate")
            seen.add(candidate_id)
            types = judgment["difference_types"]
            error = judgment["error_evidence"]
            confidence = judgment["confidence"]
            if not isinstance(types, list) or not types or len(types) != len(set(types)) or any(value not in DIFFERENCE_TYPES for value in types):
                raise ValueError("invalid_difference_types")
            if judgment["correspondence"] not in CORRESPONDENCE or judgment["independence"] not in INDEPENDENCE:
                raise ValueError("invalid_judgment_enum")
            if not isinstance(error, dict) or set(error) != {"basis", "asr_alternative", "observation"}:
                raise ValueError("invalid_error_evidence")
            if error["basis"] not in ERROR_BASES or error["asr_alternative"] not in ASR_ALTERNATIVES or not isinstance(error["observation"], str) or not 1 <= len(error["observation"]) <= 240:
                raise ValueError("invalid_error_evidence_value")
            if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError("invalid_confidence")
        if set(seen) != set(expected[sample_id]):
            raise ValueError("missing_candidate_judgments")
        returned[sample_id] = assessment
    if set(returned) != set(expected):
        raise ValueError("missing_assessments")
    return returned


def judgment_eligible(judgment: Mapping[str, Any]) -> tuple[bool, list[str]]:
    reasons = []
    vetoes = sorted(set(judgment["difference_types"]) & TYPE_VETOES)
    if vetoes:
        reasons.append("typed_veto:" + ",".join(vetoes))
    if judgment["correspondence"] != "local":
        reasons.append("correspondence_" + judgment["correspondence"])
    error = judgment["error_evidence"]
    if error["basis"] not in ELIGIBLE_BASES:
        reasons.append("error_basis_" + error["basis"])
    if error["asr_alternative"] != "ruled_out":
        reasons.append("asr_alternative_" + error["asr_alternative"])
    if judgment["independence"] != "independent":
        reasons.append("independence_" + judgment["independence"])
    if judgment["confidence"] < 0.85:
        reasons.append("confidence_below_0.85")
    return not reasons, reasons
