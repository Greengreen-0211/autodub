"""C4 r4 inventory: content-count-consistent one-sided boundary windows.

Pure implementation. It reads no files and makes no API calls.
"""
from __future__ import annotations

import copy
import difflib
import importlib.util
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping


HERE = Path(__file__).resolve().parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = _load("c4_r1_private_for_r4", HERE / "v12_c4_candidate_inventory.py")
r2_rules = _load("c4_r2_private_for_r4", HERE / "v12_c4_candidate_inventory_r2.py")

ALLOWED_REQUEST_FIELDS = base.ALLOWED_REQUEST_FIELDS
REQUIRED_LINE_FIELDS = {"line_id", "text"}
OPTIONAL_LINE_FIELDS = {"lexical_score", "subtitle_score", "selection"}
ALLOWED_LINE_FIELDS = REQUIRED_LINE_FIELDS | OPTIONAL_LINE_FIELDS
MIN_ANCHOR_CONTENT_TOKENS = 2
MIN_ONE_SIDED_CONTENT_ANCHOR = 3
MAX_EDGE_CONTENT_TOKENS = 4
MAX_TOKENS_PER_SIDE = base.MAX_TOKENS_PER_SIDE
MAX_CODEPOINTS_PER_SIDE = base.MAX_CODEPOINTS_PER_SIDE
MAX_CLASSIFIABLE = base.MAX_CLASSIFIABLE
MAX_REQUEST_BYTES = base.MAX_REQUEST_BYTES
CANDIDATE_ID_RE = re.compile(r"^c4_[0-9a-f]{64}$")

strict_json_loads = base.strict_json_loads
canonical_bytes = base.canonical_bytes
sha256 = base.sha256
tokenize = base.tokenize
occurrences = base.occurrences
expand_unique_context = base.expand_unique_context
parse_classification_response = base.parse_classification_response
judgment_eligible = base.judgment_eligible


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _content_keys(text: str) -> list[str]:
    return [token["key"] for token in tokenize(text) if token["kind"] != "punct"]


def _content_count(tokens: list[dict[str, Any]]) -> int:
    return sum(token["kind"] != "punct" for token in tokens)


def validate_request(request: Mapping[str, Any]) -> None:
    if not isinstance(request, Mapping) or set(request) != ALLOWED_REQUEST_FIELDS:
        raise ValueError("request_fields_must_be_exact_whitelist")
    if not isinstance(request["id"], str) or not request["id"]:
        raise ValueError("invalid_request_id")
    if not isinstance(request["source_language"], str) or not request["source_language"]:
        raise ValueError("invalid_source_language")
    if not isinstance(request["qwen_asr"], str):
        raise ValueError("invalid_qwen_asr")
    lines = request["ocr_candidates"]
    if not isinstance(lines, list):
        raise ValueError("ocr_candidates_must_be_list")
    seen = set()
    for line in lines:
        if not isinstance(line, Mapping) or not REQUIRED_LINE_FIELDS <= set(line) or not set(line) <= ALLOWED_LINE_FIELDS:
            raise ValueError("invalid_ocr_candidate_fields")
        line_id, text = line["line_id"], line["text"]
        if not isinstance(line_id, str) or not line_id or line_id in seen or not isinstance(text, str):
            raise ValueError("invalid_or_duplicate_ocr_line")
        seen.add(line_id)
        for field in ("lexical_score", "subtitle_score"):
            if field in line:
                value = line[field]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f"invalid_{field}")
        if "selection" in line and (not isinstance(line["selection"], str) or not line["selection"]):
            raise ValueError("invalid_selection")


def _unique_anchors(asr_tokens: list[dict[str, Any]], ocr_tokens: list[dict[str, Any]]):
    asr_keys = [token["key"] for token in asr_tokens]
    ocr_keys = [token["key"] for token in ocr_tokens]
    matcher = difflib.SequenceMatcher(None, asr_keys, ocr_keys, autojunk=False)
    anchors, ambiguous = [], 0
    for block in matcher.get_matching_blocks():
        block_tokens = asr_tokens[block.a:block.a + block.size]
        if _content_count(block_tokens) < MIN_ANCHOR_CONTENT_TOKENS:
            continue
        keys = asr_keys[block.a:block.a + block.size]
        if base._subsequence_count(asr_keys, keys) != 1 or base._subsequence_count(ocr_keys, keys) != 1:
            ambiguous += 1
            continue
        anchors.append({"a": block.a, "b": block.b, "size": block.size})
    return anchors, ambiguous


def _anchor_record(asr_tokens, ocr_tokens, block):
    a_tokens = asr_tokens[block["a"]:block["a"] + block["size"]]
    o_tokens = ocr_tokens[block["b"]:block["b"] + block["size"]]
    keys = [token["key"] for token in a_tokens]
    return {
        "asr_start": a_tokens[0]["start"], "asr_end": a_tokens[-1]["end"],
        "ocr_start": o_tokens[0]["start"], "ocr_end": o_tokens[-1]["end"],
        "content_token_count": _content_count(a_tokens), "token_count": len(a_tokens),
        "keys": keys,
    }


def _edge_window(tokens: list[dict[str, Any]], boundary: int, content_count: int, side: str):
    """Return a continuous token window with exactly N content tokens.

    Punctuation adjacent to the N selected content tokens stays inside the
    window but never consumes the content quota.
    """
    if side not in {"prefix", "suffix"} or not _is_int(content_count) or content_count < 0:
        raise ValueError("invalid_edge_window_args")
    if not _is_int(boundary) or not 0 <= boundary <= len(tokens):
        raise ValueError("invalid_edge_boundary")
    step = -1 if side == "prefix" else 1
    index = boundary - 1 if side == "prefix" else boundary
    selected, found = [], 0
    while 0 <= index < len(tokens):
        token = tokens[index]
        if found == content_count and token["kind"] != "punct":
            break
        selected.append(index)
        found += token["kind"] != "punct"
        index += step
    if found != content_count:
        return None
    if not selected:
        offset = tokens[boundary]["start"] if boundary < len(tokens) else (tokens[-1]["end"] if tokens else 0)
        return {"start": offset, "end": offset, "content_token_count": 0, "token_count": 0}
    chosen = tokens[min(selected):max(selected) + 1]
    return {
        "start": chosen[0]["start"], "end": chosen[-1]["end"],
        "content_token_count": _content_count(chosen), "token_count": len(chosen),
    }


def _trim(text: str, start: int, end: int):
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _raw_candidate(asr: str, ocr: str, *, asr_start: int, asr_end: int,
                   ocr_start: int, ocr_end: int, line_id: str,
                   alignment_status: str, left_anchor, right_anchor):
    asr_start, asr_end = _trim(asr, asr_start, asr_end)
    ocr_start, ocr_end = _trim(ocr, ocr_start, ocr_end)
    original, replacement = asr[asr_start:asr_end], ocr[ocr_start:ocr_end]
    if original == replacement:
        return None
    asr_tokens, ocr_tokens = tokenize(original), tokenize(replacement)
    asr_content, ocr_content = _content_count(asr_tokens), _content_count(ocr_tokens)
    if asr_content and ocr_content:
        operation = "replace"
    elif ocr_content:
        operation = "insert"
    elif asr_content:
        operation = "delete"
    else:
        operation = "replace"  # compatible enum; always surface diagnostic
    reasons = []
    if operation != "replace":
        reasons.append(operation + "_diagnostic")
    if _content_keys(original) == _content_keys(replacement):
        reasons.append("surface_only")
    if (len(asr_tokens) > MAX_TOKENS_PER_SIDE or len(ocr_tokens) > MAX_TOKENS_PER_SIDE
            or len(original) > MAX_CODEPOINTS_PER_SIDE or len(replacement) > MAX_CODEPOINTS_PER_SIDE):
        reasons.append("long_span")
    evidence = {"line_id": line_id, "ocr_start": ocr_start, "ocr_end": ocr_end, "quote": replacement}
    source = {
        "line_id": line_id, "alignment_status": alignment_status,
        "asr_window": {"start": asr_start, "end": asr_end, "content_token_count": asr_content, "token_count": len(asr_tokens)},
        "ocr_window": {"start": ocr_start, "end": ocr_end, "content_token_count": ocr_content, "token_count": len(ocr_tokens)},
        "left_anchor": copy.deepcopy(left_anchor), "right_anchor": copy.deepcopy(right_anchor),
    }
    return {
        "asr_start": asr_start, "asr_end": asr_end, "original_span": original,
        "replacement": replacement, "operation": operation,
        "alignment_status": alignment_status,
        "proposal_status": "diagnostic_only" if reasons else "classifiable",
        "diagnostic_reasons": sorted(set(reasons)), "evidence": [evidence],
        "source_alignments": [source],
        "anchor_audit": [{"line_id": line_id, "left": copy.deepcopy(left_anchor), "right": copy.deepcopy(right_anchor)}],
        "asr_token_count": len(asr_tokens), "ocr_token_count": len(ocr_tokens),
    }


def _enumerate_line(asr: str, line_id: str, ocr: str):
    asr_tokens, ocr_tokens = tokenize(asr), tokenize(ocr)
    anchors, ambiguous = _unique_anchors(asr_tokens, ocr_tokens)
    diagnostics = {"line_id": line_id, "anchor_count": len(anchors),
                   "ambiguous_anchor_blocks": ambiguous, "uncovered": []}
    if [x["key"] for x in asr_tokens] == [x["key"] for x in ocr_tokens] and asr != ocr:
        diagnostics["uncovered"].append("folded_surface_difference")
    if not anchors:
        diagnostics["uncovered"].append("no_unique_anchor")
        return [], diagnostics
    found = []
    for left, right in zip(anchors, anchors[1:]):
        a0, a1 = left["a"] + left["size"], right["a"]
        b0, b1 = left["b"] + left["size"], right["b"]
        item = _raw_candidate(
            asr, ocr,
            asr_start=asr_tokens[a0]["start"] if a0 < len(asr_tokens) else len(asr),
            asr_end=asr_tokens[a1]["start"] if a1 < len(asr_tokens) else len(asr),
            ocr_start=ocr_tokens[b0]["start"] if b0 < len(ocr_tokens) else len(ocr),
            ocr_end=ocr_tokens[b1]["start"] if b1 < len(ocr_tokens) else len(ocr),
            line_id=line_id, alignment_status="two_sided",
            left_anchor=_anchor_record(asr_tokens, ocr_tokens, left),
            right_anchor=_anchor_record(asr_tokens, ocr_tokens, right),
        )
        if item:
            found.append(item)

    first = anchors[0]
    prefix = ocr_tokens[:first["b"]]
    if prefix:
        n = _content_count(prefix)
        if n > MAX_EDGE_CONTENT_TOKENS:
            diagnostics["uncovered"].append("long_boundary_uncovered")
        else:
            window = _edge_window(asr_tokens, first["a"], n, "prefix")
            if window is None:
                diagnostics["uncovered"].append("boundary_uncovered")
            else:
                item = _raw_candidate(
                    asr, ocr, asr_start=window["start"], asr_end=window["end"],
                    ocr_start=prefix[0]["start"], ocr_end=prefix[-1]["end"],
                    line_id=line_id, alignment_status="one_sided", left_anchor=None,
                    right_anchor=_anchor_record(asr_tokens, ocr_tokens, first),
                )
                if item:
                    found.append(item)

    last = anchors[-1]
    suffix = ocr_tokens[last["b"] + last["size"]:]
    if suffix:
        n = _content_count(suffix)
        if n > MAX_EDGE_CONTENT_TOKENS:
            diagnostics["uncovered"].append("long_boundary_uncovered")
        else:
            window = _edge_window(asr_tokens, last["a"] + last["size"], n, "suffix")
            if window is None:
                diagnostics["uncovered"].append("boundary_uncovered")
            else:
                item = _raw_candidate(
                    asr, ocr, asr_start=window["start"], asr_end=window["end"],
                    ocr_start=suffix[0]["start"], ocr_end=suffix[-1]["end"],
                    line_id=line_id, alignment_status="one_sided",
                    left_anchor=_anchor_record(asr_tokens, ocr_tokens, last), right_anchor=None,
                )
                if item:
                    found.append(item)
    return found, diagnostics


def _overlap(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if left["asr_start"] == left["asr_end"] == right["asr_start"] == right["asr_end"]:
        return True
    return left["asr_start"] < right["asr_end"] and right["asr_start"] < left["asr_end"]


def _mark_conflicts(candidates: list[dict[str, Any]]) -> int:
    graph = defaultdict(set)
    for index, left in enumerate(candidates):
        for other in range(index + 1, len(candidates)):
            if _overlap(left, candidates[other]):
                graph[index].add(other); graph[other].add(index)
    seen, groups = set(), 0
    for index in sorted(graph):
        if index in seen:
            continue
        stack, component = [index], []
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current); component.append(current)
            stack.extend(sorted(graph[current] - seen, reverse=True))
        if len(component) < 2:
            continue
        groups += 1
        group_id = f"r4_conflict_{groups:03d}"
        for current in component:
            item = candidates[current]
            item["conflict_group"] = group_id
            item["proposal_status"] = "diagnostic_only"
            item["diagnostic_reasons"] = sorted(set(item["diagnostic_reasons"] + ["overlapping_candidates"]))
    return groups


def build_inventory(request: Mapping[str, Any]) -> dict[str, Any]:
    validate_request(request)
    raw, line_diagnostics = [], []
    for line in request["ocr_candidates"]:
        values, diagnostics = _enumerate_line(request["qwen_asr"], line["line_id"], line["text"])
        raw.extend(values); line_diagnostics.append(diagnostics)
    merged = {}
    for item in raw:
        key = (item["asr_start"], item["asr_end"], item["replacement"])
        if key not in merged:
            merged[key] = item
        else:
            target = merged[key]
            target["evidence"].extend(item["evidence"])
            target["source_alignments"].extend(item["source_alignments"])
            target["anchor_audit"].extend(item["anchor_audit"])
            target["diagnostic_reasons"] = sorted(set(target["diagnostic_reasons"] + item["diagnostic_reasons"]))
    candidates = list(merged.values())
    for item in candidates:
        statuses = {source["alignment_status"] for source in item["source_alignments"]}
        if len(statuses) > 1:
            item["alignment_status"] = "ambiguous"
            item["proposal_status"] = "diagnostic_only"
            item["diagnostic_reasons"] = sorted(set(item["diagnostic_reasons"] + ["mixed_alignment_provenance"]))
        else:
            item["alignment_status"] = next(iter(statuses))
        if item["proposal_status"] == "classifiable" and item["alignment_status"] == "one_sided":
            supports = []
            for source in item["source_alignments"]:
                anchor = source["left_anchor"] or source["right_anchor"]
                supports.append(anchor["content_token_count"] if anchor else 0)
            support = min(supports)
            item["one_sided_content_anchor"] = support
            reasons = []
            if support < MIN_ONE_SIDED_CONTENT_ANCHOR:
                reasons.append("weak_one_sided_anchor")
            if r2_rules._relocates_existing_text(request["qwen_asr"], item):
                reasons.append("relocated_existing_text")
            if reasons:
                item["proposal_status"] = "diagnostic_only"
                item["diagnostic_reasons"] = sorted(set(item["diagnostic_reasons"] + reasons))
        item["evidence"] = sorted(item["evidence"], key=lambda x: (x["line_id"], x["ocr_start"], x["ocr_end"], x["quote"]))
        item["source_alignments"] = sorted(item["source_alignments"], key=lambda x: (x["line_id"], x["ocr_window"]["start"], x["alignment_status"]))
        item["anchor_audit"] = sorted(item["anchor_audit"], key=lambda x: (x["line_id"], str(x["left"]), str(x["right"])))

    conflict_groups = _mark_conflicts(candidates)
    request_hash = sha256(dict(request))
    candidates.sort(key=lambda item: (item["asr_start"], item["asr_end"], item["replacement"], item["evidence"][0]["line_id"]))
    for item in candidates:
        item.setdefault("conflict_group", None)
        item["candidate_id"] = base._candidate_id(request_hash, item)
    classifiable = [item for item in candidates if item["proposal_status"] == "classifiable"]
    payload = dict(request); payload["contrast_candidates"] = classifiable
    request_bytes = len(canonical_bytes(payload))
    capacity_reasons = []
    if len(classifiable) > MAX_CLASSIFIABLE:
        capacity_reasons.append("classifiable_candidate_limit")
    if request_bytes > MAX_REQUEST_BYTES:
        capacity_reasons.append("request_byte_limit")
    result = {
        "id": request["id"], "request_sha256": request_hash,
        "capacity_blocked": bool(capacity_reasons), "capacity_reasons": capacity_reasons,
        "request_bytes_with_classifiable_candidates": request_bytes,
        "candidates": candidates,
        "diagnostics": {
            "line_diagnostics": line_diagnostics, "raw_candidate_count": len(raw),
            "deduplicated_candidate_count": len(candidates),
            "classifiable_candidate_count": len(classifiable),
            "diagnostic_only_candidate_count": len(candidates) - len(classifiable),
            "conflict_group_count": conflict_groups,
            "mixed_alignment_candidate_count": sum("mixed_alignment_provenance" in x["diagnostic_reasons"] for x in candidates),
        },
        "r4_policy": {
            "minimum_anchor_content_tokens": MIN_ANCHOR_CONTENT_TOKENS,
            "minimum_one_sided_content_anchor": MIN_ONE_SIDED_CONTENT_ANCHOR,
            "maximum_edge_content_tokens": MAX_EDGE_CONTENT_TOKENS,
            "edge_windows_count_content_not_punctuation": True,
            "zero_content_side_is_not_classifiable_replace": True,
            "mixed_alignment_provenance_is_diagnostic": True,
            "word_boundary_and_apostrophe_differences_are_not_auto_surface": True,
            "all_candidate_overlap_conflicts_before_model": True,
        },
    }
    validate_inventory(request, result)
    return result


def _validate_anchor(anchor, asr: str, ocr: str, cid: str):
    if anchor is None:
        return
    fields = {"asr_start", "asr_end", "ocr_start", "ocr_end", "content_token_count", "token_count", "keys"}
    if not isinstance(anchor, Mapping) or set(anchor) != fields:
        raise ValueError(f"invalid_anchor_fields:{cid}")
    for key in fields - {"keys"}:
        if not _is_int(anchor[key]):
            raise ValueError(f"invalid_anchor_integer:{cid}:{key}")
    if not 0 <= anchor["asr_start"] <= anchor["asr_end"] <= len(asr) or not 0 <= anchor["ocr_start"] <= anchor["ocr_end"] <= len(ocr):
        raise ValueError(f"anchor_out_of_range:{cid}")
    asr_tokens, ocr_tokens = tokenize(asr[anchor["asr_start"]:anchor["asr_end"]]), tokenize(ocr[anchor["ocr_start"]:anchor["ocr_end"]])
    if ([x["key"] for x in asr_tokens] != anchor["keys"] or [x["key"] for x in ocr_tokens] != anchor["keys"]
            or _content_count(asr_tokens) != anchor["content_token_count"] or len(asr_tokens) != anchor["token_count"]):
        raise ValueError(f"anchor_content_mismatch:{cid}")


def validate_inventory(request: Mapping[str, Any], inventory: Mapping[str, Any]) -> None:
    validate_request(request)
    top = {"id", "request_sha256", "capacity_blocked", "capacity_reasons", "request_bytes_with_classifiable_candidates", "candidates", "diagnostics", "r4_policy"}
    if not isinstance(inventory, Mapping) or set(inventory) != top:
        raise ValueError("invalid_inventory_fields")
    if inventory["id"] != request["id"] or inventory["request_sha256"] != sha256(dict(request)):
        raise ValueError("inventory_request_binding_mismatch")
    if not isinstance(inventory["capacity_blocked"], bool) or not isinstance(inventory["capacity_reasons"], list) or not _is_int(inventory["request_bytes_with_classifiable_candidates"]):
        raise ValueError("invalid_capacity_fields")
    if not isinstance(inventory["candidates"], list) or not isinstance(inventory["diagnostics"], Mapping):
        raise ValueError("invalid_inventory_collections")
    asr = request["qwen_asr"]
    lines = {line["line_id"]: line["text"] for line in request["ocr_candidates"]}
    ids = set()
    required = {"asr_start", "asr_end", "original_span", "replacement", "operation", "alignment_status", "proposal_status", "diagnostic_reasons", "evidence", "source_alignments", "anchor_audit", "asr_token_count", "ocr_token_count", "conflict_group", "candidate_id"}
    for candidate in inventory["candidates"]:
        if not isinstance(candidate, Mapping) or not required <= set(candidate) or not set(candidate) <= required | {"one_sided_content_anchor"}:
            raise ValueError("invalid_candidate_fields")
        cid = candidate["candidate_id"]
        if not isinstance(cid, str) or not CANDIDATE_ID_RE.fullmatch(cid) or cid in ids:
            raise ValueError("invalid_or_duplicate_candidate_id")
        ids.add(cid)
        start, end = candidate["asr_start"], candidate["asr_end"]
        if not _is_int(start) or not _is_int(end) or not 0 <= start <= end <= len(asr) or asr[start:end] != candidate["original_span"]:
            raise ValueError(f"invalid_asr_slice:{cid}")
        if candidate["operation"] not in {"replace", "insert", "delete"} or candidate["alignment_status"] not in {"two_sided", "one_sided", "ambiguous"}:
            raise ValueError(f"invalid_candidate_enum:{cid}")
        if candidate["proposal_status"] not in {"classifiable", "diagnostic_only"} or not isinstance(candidate["diagnostic_reasons"], list):
            raise ValueError(f"invalid_candidate_status:{cid}")
        asr_content, ocr_content = len(_content_keys(candidate["original_span"])), len(_content_keys(candidate["replacement"]))
        expected_operation = "replace" if asr_content and ocr_content else "insert" if ocr_content else "delete" if asr_content else "replace"
        if candidate["operation"] != expected_operation:
            raise ValueError(f"operation_content_mismatch:{cid}")
        if (not asr_content or not ocr_content) and candidate["proposal_status"] == "classifiable":
            raise ValueError(f"zero_content_classifiable:{cid}")
        evidence_map = {}
        if not isinstance(candidate["evidence"], list) or not candidate["evidence"]:
            raise ValueError(f"invalid_evidence:{cid}")
        for evidence in candidate["evidence"]:
            if not isinstance(evidence, Mapping) or set(evidence) != {"line_id", "ocr_start", "ocr_end", "quote"}:
                raise ValueError(f"invalid_evidence_fields:{cid}")
            line_id = evidence["line_id"]
            if line_id not in lines or not _is_int(evidence["ocr_start"]) or not _is_int(evidence["ocr_end"]):
                raise ValueError(f"invalid_evidence_reference:{cid}")
            a, b = evidence["ocr_start"], evidence["ocr_end"]
            if not 0 <= a <= b <= len(lines[line_id]) or lines[line_id][a:b] != evidence["quote"] or evidence["quote"] != candidate["replacement"]:
                raise ValueError(f"invalid_evidence_slice:{cid}")
            evidence_map[(line_id, a, b)] = evidence
        if len(evidence_map) != len(candidate["evidence"]):
            raise ValueError(f"duplicate_evidence:{cid}")
        if not isinstance(candidate["source_alignments"], list) or len(candidate["source_alignments"]) != len(candidate["evidence"]):
            raise ValueError(f"invalid_source_alignment_count:{cid}")
        for source in candidate["source_alignments"]:
            fields = {"line_id", "alignment_status", "asr_window", "ocr_window", "left_anchor", "right_anchor"}
            if not isinstance(source, Mapping) or set(source) != fields or source["line_id"] not in lines or source["alignment_status"] not in {"one_sided", "two_sided"}:
                raise ValueError(f"invalid_source_alignment:{cid}")
            for name, text in (("asr_window", asr), ("ocr_window", lines[source["line_id"]])):
                window = source[name]
                if not isinstance(window, Mapping) or set(window) != {"start", "end", "content_token_count", "token_count"} or any(not _is_int(window[k]) for k in window):
                    raise ValueError(f"invalid_source_window:{cid}")
                if not 0 <= window["start"] <= window["end"] <= len(text):
                    raise ValueError(f"source_window_out_of_range:{cid}")
                tokens = tokenize(text[window["start"]:window["end"]])
                if len(tokens) != window["token_count"] or _content_count(tokens) != window["content_token_count"]:
                    raise ValueError(f"source_window_count_mismatch:{cid}")
            aw, ow = source["asr_window"], source["ocr_window"]
            if (aw["start"], aw["end"]) != (start, end) or (source["line_id"], ow["start"], ow["end"]) not in evidence_map:
                raise ValueError(f"source_window_candidate_mismatch:{cid}")
            _validate_anchor(source["left_anchor"], asr, lines[source["line_id"]], cid)
            _validate_anchor(source["right_anchor"], asr, lines[source["line_id"]], cid)
        if not isinstance(candidate["anchor_audit"], list) or len(candidate["anchor_audit"]) != len(candidate["source_alignments"]):
            raise ValueError(f"invalid_anchor_audit:{cid}")
        statuses = {source["alignment_status"] for source in candidate["source_alignments"]}
        expected_status = next(iter(statuses)) if len(statuses) == 1 else "ambiguous"
        if candidate["alignment_status"] != expected_status or (len(statuses) > 1) != ("mixed_alignment_provenance" in candidate["diagnostic_reasons"]):
            raise ValueError(f"alignment_provenance_mismatch:{cid}")
        if cid != base._candidate_id(inventory["request_sha256"], candidate):
            raise ValueError(f"candidate_hash_mismatch:{cid}")
    classifiable = [x for x in inventory["candidates"] if x["proposal_status"] == "classifiable"]
    for i, left in enumerate(classifiable):
        for right in classifiable[i + 1:]:
            if _overlap(left, right):
                raise ValueError("classifiable_candidates_overlap")
    payload = dict(request); payload["contrast_candidates"] = classifiable
    request_bytes = len(canonical_bytes(payload))
    reasons = []
    if len(classifiable) > MAX_CLASSIFIABLE: reasons.append("classifiable_candidate_limit")
    if request_bytes > MAX_REQUEST_BYTES: reasons.append("request_byte_limit")
    if request_bytes != inventory["request_bytes_with_classifiable_candidates"] or reasons != inventory["capacity_reasons"] or bool(reasons) != inventory["capacity_blocked"]:
        raise ValueError("capacity_recalculation_mismatch")


def inventory_matches_request(request: Mapping[str, Any], inventory: Mapping[str, Any]) -> bool:
    try:
        validate_inventory(request, inventory)
        return canonical_bytes(build_inventory(request)) == canonical_bytes(inventory)
    except Exception:
        return False

