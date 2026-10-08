"""Pure decision adapter for the predeclared v1.2 A1 deletion experiment."""
from __future__ import annotations

import copy
import json
from typing import Any, Mapping

CANDIDATES = ("V12_A1_r1_anchor3", "V12_A1_r2_anchor3to8")
ROW_KEYS = ("qwen_raw", "ocr_selected_json", "proposed_changed", "guard_reason")
EDIT_KEYS = ("original_span", "replacement", "evidence_line_ids", "confidence", "reason")


def decision_input(row: Mapping[str, Any], proposal: Mapping[str, Any]):
    # Explicitly exclude ids, references, labels, metrics, and owner annotations.
    clean_row = {key: copy.deepcopy(row[key]) for key in ROW_KEYS}
    lines = json.loads(clean_row["ocr_selected_json"])
    clean_row["ocr_selected_json"] = json.dumps([
        {"line_id": line["line_id"], "text": line["text"]} for line in lines
    ], ensure_ascii=False)
    clean_proposal = {
        "action": proposal.get("action", "KEEP"),
        "reason": proposal.get("reason", ""),
        "edits": [{key: copy.deepcopy(edit.get(key)) for key in EDIT_KEYS}
                  for edit in proposal.get("edits", [])],
    }
    return clean_row, clean_proposal


def occurrences(text, needle):
    result, start = [], 0
    while needle:
        pos = text.find(needle, start)
        if pos < 0:
            break
        result.append(pos)
        start = pos + 1
    return result


def aligned_deletion(row, edit, width, p3):
    """Require a unique local replacement window and no original window in own cited lines."""
    qwen = row["qwen_raw"]
    original, replacement = str(edit["original_span"]), str(edit["replacement"])
    if not original or qwen.count(original) != 1:
        return False, {"reason": "span_not_unique"}
    start = qwen.index(original)
    left_all = p3.normalize(qwen[:start])
    right_all = p3.normalize(qwen[start + len(original):])
    left, right = left_all[-width:], right_all[:width]
    old = left + p3.normalize(original) + right
    new = left + p3.normalize(replacement) + right
    if not new or new == old or (not left and not right):
        return False, {"reason": "insufficient_local_anchors"}
    selected = {str(line["line_id"]): str(line["text"])
                for line in json.loads(row["ocr_selected_json"])}
    ids = edit.get("evidence_line_ids") or []
    if not ids or any(str(line_id) not in selected for line_id in ids):
        return False, {"reason": "missing_own_cited_line"}
    # Single-line evidence only in this round: no speculative cross-line joins.
    evidence = {}
    for line_id in ids:
        evidence.setdefault(p3.normalize(selected[str(line_id)]), []).append(str(line_id))
    hits, old_hits = [], []
    for line, line_ids in evidence.items():
        for pos in occurrences(line, old):
            old_hits.append({"line_ids": line_ids, "start": pos})
        for pos in occurrences(line, new):
            # If a side reaches an ASR boundary, require the same evidence boundary.
            if len(left_all) <= width and pos != 0:
                continue
            if len(right_all) <= width and pos + len(new) != len(line):
                continue
            hits.append({"line_ids": line_ids, "start": pos, "end": pos + len(new)})
    reason = ("original_window_supported" if old_hits else
              "unique_local_replacement" if len(hits) == 1 else
              "ambiguous_local_alignment" if len(hits) > 1 else "local_alignment_not_found")
    return reason == "unique_local_replacement", {
        "reason": reason, "width": width, "left_anchor": left, "right_anchor": right,
        "original_window": old, "replacement_window": new,
        "original_hits": old_hits, "replacement_hits": hits,
    }


def decide(clean_row, clean_proposal, candidate, p3, p4):
    if candidate not in ("parent", *CANDIDATES):
        raise ValueError(candidate)
    # Both baseline and candidates use the same sanitized inputs.
    accepted, reason = p3.guard_decision("P0P2P3_r3_20260902", clean_row, clean_proposal)
    trace = []
    if candidate != "parent" and reason.startswith("p3_deleted_content_present_in_cited_evidence:"):
        # Keep replacement grounding and every other legacy rule identical.
        grounded, replacement_reason = p3.grounding_check(clean_row, clean_proposal, check_deletions=False)
        if not grounded:
            accepted, reason = False, replacement_reason
        else:
            combined = p3.normalize("".join(p3.cited_texts(clean_row, clean_proposal)))
            recovered = True
            for index, edit in enumerate(clean_proposal["edits"]):
                suspect = [segment for segment in p3.deleted_segments(edit["original_span"], edit["replacement"])
                           if p3.normalize(segment) in combined]
                if not suspect:
                    continue
                widths = (3,) if candidate == CANDIDATES[0] else range(3, 9)
                ok, attempts = False, []
                for width in widths:
                    ok, attempt = aligned_deletion(clean_row, edit, width, p3)
                    attempts.append(attempt)
                    if ok or attempt["reason"] in {"original_window_supported", "span_not_unique", "missing_own_cited_line"}:
                        break
                trace.append({"edit_index": index, "suspect_segments": suspect, "accepted": ok, "attempts": attempts})
                if not ok:
                    recovered = False
            accepted = recovered
            reason = "v12_local_deletion_recovered" if recovered else "v12_local_deletion_unresolved"
            # The original early deletion return preceded this frozen risk check.
            if recovered and p3.technical_risk(clean_proposal):
                accepted, reason = False, "p3_self_identified_technical_term_risk"
    final, p4_reason = p4.boundary_decision("V11_P4_r2_task_boundary_20260903", accepted, clean_proposal)
    return {"p3_accepted": accepted, "p3_reason": reason,
            "final_accepted": final, "p4_reason": p4_reason, "alignment": trace}
