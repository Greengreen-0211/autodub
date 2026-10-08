"""Fail-closed r4 adapter bound to the full request and inventory."""
from __future__ import annotations

import copy
from typing import Any, Mapping

import v12_c4_candidate_adapter as legacy_adapter
import v12_c4_candidate_inventory_r4 as inventory_impl


def _keep(text: str, reason: str):
    return {"accepted": False, "corrected_text": text, "reason": reason,
            "candidate_audit": [], "c3_decision": None}


def decide(request: Any, inventory: Any, assessment: Any, *, c1r, core, a1, p3, p4):
    qwen = request.get("qwen_asr", "") if isinstance(request, Mapping) else ""
    if not isinstance(qwen, str): qwen = ""
    try:
        inventory_impl.validate_request(request)
        inventory_impl.validate_inventory(request, inventory)
        regenerated = inventory_impl.build_inventory(request)
        if inventory_impl.canonical_bytes(regenerated) != inventory_impl.canonical_bytes(inventory):
            raise ValueError("inventory_differs_from_deterministic_regeneration")
    except Exception as exc:
        return _keep(qwen, f"c4_integrity_error:{type(exc).__name__}:{exc}")
    if inventory["capacity_blocked"]:
        return _keep(qwen, "c4_capacity_blocked")
    result = legacy_adapter.decide(
        request["qwen_asr"], request["ocr_candidates"], copy.deepcopy(inventory),
        copy.deepcopy(assessment), c1r=c1r, core=core, a1=a1, p3=p3, p4=p4,
    )
    result["c4_r4_request_sha256"] = inventory["request_sha256"]
    result["c4_r4_inventory_verified"] = True
    return result
