"""C4 candidate-bound classifier adapter into the frozen C3/C1R path."""
from __future__ import annotations

import copy
import json
from typing import Any, Mapping

import v12_c3_structured_adapter_r2 as c3
import v12_c4_candidate_inventory_r2 as inventory_impl


def _overlap(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return left["asr_start"] < right["asr_end"] and right["asr_start"] < left["asr_end"]


def decide(qwen: Any, selected_lines: Any, inventory: Any, assessment: Any, *, c1r, core, a1, p3, p4) -> dict[str, Any]:
    if not isinstance(qwen, str) or not isinstance(selected_lines, list) or not isinstance(inventory, dict):
        return {"accepted": False, "corrected_text": qwen if isinstance(qwen, str) else "", "reason": "invalid_c4_inputs", "candidate_audit": []}
    classifiable = [item for item in inventory.get("candidates", []) if item.get("proposal_status") == "classifiable"]
    submitted = {str(inventory.get("id", "")): [item["candidate_id"] for item in classifiable]}
    try:
        parsed = inventory_impl.parse_classification_response(
            json.dumps({"assessments": [assessment]}, ensure_ascii=False, separators=(",", ":")), submitted
        )
    except Exception as exc:
        return {"accepted": False, "corrected_text": qwen, "reason": f"classification_structural_error:{type(exc).__name__}:{exc}", "candidate_audit": []}
    sample_id = next(iter(submitted))
    judgments = {item["candidate_id"]: item for item in parsed[sample_id]["judgments"]}
    candidate_audit = []
    for candidate in classifiable:
        judgment = judgments[candidate["candidate_id"]]
        eligible, reasons = inventory_impl.judgment_eligible(judgment)
        candidate_audit.append({
            "candidate_id": candidate["candidate_id"],
            "candidate": copy.deepcopy(candidate),
            "judgment": copy.deepcopy(judgment),
            "typed_eligible": eligible,
            "reasons": list(reasons),
            "bridge": None,
            "c3_contrast_index": None,
            "downstream": None,
        })

    typed = [item for item in candidate_audit if item["typed_eligible"]]
    for index, left in enumerate(typed):
        for right in typed[index + 1:]:
            if _overlap(left["candidate"], right["candidate"]):
                left["reasons"].append("typed_candidate_overlap")
                right["reasons"].append("typed_candidate_overlap")
    for item in typed:
        if item["reasons"]:
            continue
        candidate = item["candidate"]
        bridge = inventory_impl.expand_unique_context(
            qwen, candidate["asr_start"], candidate["asr_end"], candidate["replacement"]
        )
        if bridge is None:
            item["reasons"].append("bridge_unique_context_failed")
        else:
            item["bridge"] = bridge
    bridged = [item for item in typed if not item["reasons"] and item["bridge"]]
    for index, left in enumerate(bridged):
        for right in bridged[index + 1:]:
            if left["bridge"]["start"] < right["bridge"]["end"] and right["bridge"]["start"] < left["bridge"]["end"]:
                left["reasons"].append("bridge_context_overlap")
                right["reasons"].append("bridge_context_overlap")

    contrasts = []
    projected_audits = []
    for item in candidate_audit:
        if item["reasons"] or item["bridge"] is None:
            continue
        candidate, judgment, bridge = item["candidate"], item["judgment"], item["bridge"]
        item["c3_contrast_index"] = len(contrasts)
        projected_audits.append(item)
        contrasts.append({
            "original_span": bridge["original_span"],
            "replacement": bridge["replacement"],
            "difference_types": judgment["difference_types"],
            "correspondence": judgment["correspondence"],
            "evidence": [{"line_id": evidence["line_id"], "quote": evidence["quote"]} for evidence in candidate["evidence"]],
            "error_evidence": judgment["error_evidence"],
            "confidence": judgment["confidence"],
        })
    if not contrasts:
        return {
            "accepted": False,
            "corrected_text": qwen,
            "reason": "no_c4_candidates_reached_legacy_checks",
            "candidate_audit": candidate_audit,
            "c3_decision": None,
        }
    result = c3.decide(
        qwen, selected_lines, {"id": sample_id, "contrasts": contrasts},
        c1r=c1r, core=core, a1=a1, p3=p3, p4=p4,
    )
    for item in projected_audits:
        index = item["c3_contrast_index"]
        if index < len(result.get("contrasts", [])):
            item["downstream"] = copy.deepcopy(result["contrasts"][index])
    return {
        "accepted": bool(result["accepted"]),
        "corrected_text": result["corrected_text"],
        "reason": "c3:" + str(result["reason"]),
        "candidate_audit": candidate_audit,
        "c3_decision": copy.deepcopy(result),
    }
