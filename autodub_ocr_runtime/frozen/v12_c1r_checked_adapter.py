"""C1R engineering repair, P2 schema retained. Not an approved efficacy candidate.

Dependencies are injected: core = frozen Stage-4 base guard, a1/p3/p4 = frozen
guards. The public decision input has no cached guard flag or scoring data.
"""
from __future__ import annotations

import copy
import json
import math

EDIT_FIELDS = {"original_span", "replacement", "evidence_line_ids", "confidence", "reason"}
PROPOSAL_FIELDS = {"id", "action", "reason", "edits"}


def decide(qwen, selected_lines, proposal, *, core, a1, p3, p4):
    def reject(reason, audit=None):
        return {"accepted": False, "corrected_text": qwen, "reason": reason,
                "edits": audit or [], "final_base_reason": None}

    # Validate original types and required fields before projection or coercion.
    if not isinstance(qwen, str) or not qwen:
        return reject("invalid_asr")
    if not isinstance(selected_lines, list):
        return reject("invalid_ocr")
    lines, seen = [], set()
    for line in selected_lines:
        if not isinstance(line, dict) or not isinstance(line.get("line_id"), str) or not line["line_id"]:
            return reject("invalid_ocr_line")
        if not isinstance(line.get("text"), str) or line["line_id"] in seen:
            return reject("invalid_or_duplicate_ocr_line")
        seen.add(line["line_id"])
        lines.append({"line_id": line["line_id"], "text": line["text"]})
    if not isinstance(proposal, dict) or set(proposal) - PROPOSAL_FIELDS:
        return reject("invalid_proposal_fields")
    if not {"action", "reason", "edits"} <= set(proposal):
        return reject("missing_proposal_fields")
    if not isinstance(proposal["reason"], str) or proposal["action"] not in ("KEEP", "CORRECT"):
        return reject("invalid_action_or_reason")
    edits = proposal["edits"]
    if not isinstance(edits, list) or ((proposal["action"] == "KEEP") != (len(edits) == 0)):
        return reject("action_edits_mismatch")
    if not edits:
        return reject("model_keep")
    parsed = []
    for index, raw in enumerate(edits):
        if not isinstance(raw, dict) or set(raw) != EDIT_FIELDS:
            return reject("invalid_edit_fields")
        if any(not isinstance(raw[k], str) for k in ("original_span", "replacement", "reason")):
            return reject("invalid_edit_text_type")
        confidence = raw["confidence"]
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            return reject("invalid_confidence_type")
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            return reject("invalid_confidence_range")
        ids = raw["evidence_line_ids"]
        if not isinstance(ids, list) or any(not isinstance(v, str) or v not in seen for v in ids):
            return reject("invalid_evidence_ids")
        if len(ids) != len(set(ids)):
            return reject("duplicate_evidence_ids")
        positions = a1.occurrences(qwen, raw["original_span"])
        if len(positions) != 1:
            return reject("nonunique_original_span")
        parsed.append({"index": index, "start": positions[0],
                       "end": positions[0] + len(raw["original_span"]), "edit": copy.deepcopy(raw)})
    ordered = sorted(parsed, key=lambda e: e["start"])
    if any(left["end"] > right["start"] for left, right in zip(ordered, ordered[1:])):
        return reject("overlapping_edits")

    def basic(text, subset):
        ids = list(dict.fromkeys(v for x in subset for v in x["edit"]["evidence_line_ids"]))
        packed = {"text": text, "changed": text != qwen,
                  "confidence": min(x["edit"]["confidence"] for x in subset),
                  "evidence_role": "transcription", "evidence_line_ids": ids}
        return core.validate_proposal(qwen, packed, lines, 0.85, 0.45)

    audits, accepted = [], []
    for item in parsed:
        e, start, end = item["edit"], item["start"], item["end"]
        standalone = qwen[:start] + e["replacement"] + qwen[end:]
        _, _, base_reason = basic(standalone, [item])
        singleton = {"action": "CORRECT", "reason": proposal["reason"] if len(parsed) == 1 else e["reason"],
                     "edits": [e]}
        row = {"qwen_raw": qwen, "ocr_selected_json": json.dumps(lines, ensure_ascii=False),
               "guard_reason": base_reason, "proposed_changed": standalone != qwen}
        # A1's exception parses a rounded reason. Enforce the exact inherited cap.
        ratio = core.normalized_edit_ratio(qwen, standalone)
        full_exception_eligible = len(parsed) == 1 and e["original_span"].strip() == qwen.strip()
        if base_reason.startswith("edit_ratio_") and (not full_exception_eligible or ratio > 0.65):
            row["guard_reason"] = "c1r_ratio_no_global_exception"
        d = a1.decide(row, singleton, a1.CANDIDATES[0], p3, p4)
        audit = {"edit_index": item["index"], "start": start, "end": end,
                 "original_span": e["original_span"], "replacement": e["replacement"],
                 "standalone_text": standalone, "base_reason": base_reason, "edit_ratio": ratio,
                 "gate_accepted": d["final_accepted"], "applied": False,
                 "p3_reason": d["p3_reason"], "p4_reason": d["p4_reason"], "alignment": d["alignment"]}
        audits.append(audit)
        if d["final_accepted"]:
            accepted.append(item)
    if not accepted:
        return reject("all_edits_rejected", audits)
    corrected = qwen
    for item in sorted(accepted, key=lambda e: e["start"], reverse=True):
        corrected = corrected[:item["start"]] + item["edit"]["replacement"] + corrected[item["end"]:]
    _, combined_ok, combined_reason = basic(corrected, accepted)
    single_global = (len(parsed) == len(accepted) == 1 and
                     audits[0]["p3_reason"] == "p3_controlled_global_multiline_exception")
    if not combined_ok and not (single_global and combined_reason.startswith("edit_ratio_")):
        result = reject("combined_base_rejected", audits)
        result["final_base_reason"] = combined_reason
        return result
    applied = {item["index"] for item in accepted}
    for audit in audits:
        audit["applied"] = audit["edit_index"] in applied
    return {"accepted": corrected != qwen, "corrected_text": corrected,
            "reason": "partial_accepted" if len(accepted) < len(parsed) else "all_accepted",
            "edits": audits, "final_base_reason": combined_reason}
