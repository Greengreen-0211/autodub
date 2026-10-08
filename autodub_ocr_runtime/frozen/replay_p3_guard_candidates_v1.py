#!/usr/bin/env python3
"""Offline P3 guard replay over frozen P0+P2 pilot proposals; no API calls."""

from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import importlib.util
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping


SCRIPT_DIR = Path(__file__).resolve().parent
STAGE_DIR = SCRIPT_DIR.parent
MODEL = "deepseek-v4-pro"
P2_EXPERIMENT = "P0P2_r1_20260902"
CANDIDATES = (
    "P0P2P3_r1_20260902",
    "P0P2P3_r2_20260902",
    "P0P2P3_r3_20260902",
)
TECHNICAL_RISK = re.compile(r"technical\s+term|specialist\s+term|terminolog|domain[- ]specific", re.I)
EDIT_RATIO_REASON = re.compile(r"^edit_ratio_([0-9]+(?:\.[0-9]+)?)_above_")
GLOBAL_EXCEPTION_MAX_EDIT_RATIO = 0.65


def import_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: list[str], rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize(text: Any) -> str:
    value = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return "".join(char for char in value if char.isalnum())


def truthy(value: Any) -> bool:
    return str(value).strip().casefold() in {"1", "true", "yes"}


def raw_proposals(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    proposals: dict[str, dict[str, Any]] = {}
    for record in records:
        if record.get("api_error"):
            raise RuntimeError(f"P2 audit contains API error: {record['api_error']}")
        raw = record.get("response") or {}
        payload = json.loads(raw["choices"][0]["message"]["content"])
        for proposal in payload.get("corrections", []):
            proposals[str(proposal["id"])] = proposal
    return proposals


def cited_texts(row: Mapping[str, Any], proposal: Mapping[str, Any]) -> list[str]:
    selected = json.loads(str(row.get("ocr_selected_json") or "[]"))
    by_id = {str(item.get("line_id")): str(item.get("text", "")) for item in selected}
    ids: list[str] = []
    for edit in proposal.get("edits") or []:
        for line_id in edit.get("evidence_line_ids") or []:
            value = str(line_id)
            if value not in ids:
                ids.append(value)
    return [by_id[line_id] for line_id in ids if line_id in by_id]


def deleted_segments(original: str, replacement: str) -> list[str]:
    matcher = difflib.SequenceMatcher(a=original, b=replacement, autojunk=False)
    segments: list[str] = []
    for tag, left_start, left_end, _, _ in matcher.get_opcodes():
        if tag in {"delete", "replace"}:
            segment = original[left_start:left_end]
            if normalize(segment):
                segments.append(segment)
    return segments


def grounding_check(
    row: Mapping[str, Any],
    proposal: Mapping[str, Any],
    *,
    check_deletions: bool = True,
) -> tuple[bool, str]:
    evidence = cited_texts(row, proposal)
    combined = normalize("".join(evidence))
    if not evidence:
        return False, "p3_missing_cited_evidence"
    for edit in proposal.get("edits") or []:
        replacement = str(edit.get("replacement", ""))
        replacement_norm = normalize(replacement)
        if replacement_norm and replacement_norm not in combined:
            return False, "p3_replacement_not_verbatim_in_cited_evidence"
        if check_deletions:
            for segment in deleted_segments(str(edit.get("original_span", "")), replacement):
                segment_norm = normalize(segment)
                if segment_norm and segment_norm in combined:
                    return False, f"p3_deleted_content_present_in_cited_evidence:{segment}"
    return True, "p3_grounding_passed"


def technical_risk(proposal: Mapping[str, Any]) -> bool:
    texts = [str(proposal.get("reason", ""))]
    texts.extend(str(edit.get("reason", "")) for edit in proposal.get("edits") or [])
    return bool(TECHNICAL_RISK.search(" ".join(texts)))


def edit_ratio_from_guard_reason(row: Mapping[str, Any]) -> float | None:
    match = EDIT_RATIO_REASON.match(str(row.get("guard_reason", "")))
    if not match:
        return None
    return float(match.group(1))


def global_exception(row: Mapping[str, Any], proposal: Mapping[str, Any]) -> bool:
    edit_ratio = edit_ratio_from_guard_reason(row)
    if edit_ratio is None or edit_ratio > GLOBAL_EXCEPTION_MAX_EDIT_RATIO:
        return False
    edits = proposal.get("edits") or []
    if len(edits) != 1:
        return False
    edit = edits[0]
    if str(edit.get("original_span", "")).strip() != str(row.get("qwen_raw", "")).strip():
        return False
    try:
        confidence = float(edit.get("confidence", 0) or 0)
    except (TypeError, ValueError):
        return False
    ids = [str(value) for value in edit.get("evidence_line_ids") or []]
    if confidence < 0.90 or len(set(ids)) < 2:
        return False
    evidence = normalize("".join(cited_texts(row, proposal)))
    replacement = normalize(edit.get("replacement", ""))
    return bool(replacement and (replacement in evidence or evidence in replacement))


def guard_decision(candidate: str, row: Mapping[str, Any], proposal: Mapping[str, Any]) -> tuple[bool, str]:
    base_accepted = str(row.get("guard_reason", "")) == "accepted" and truthy(row.get("proposed_changed"))
    allow_global = candidate == "P0P2P3_r3_20260902" and str(row.get("guard_reason", "")).startswith("edit_ratio_")
    global_override = allow_global and global_exception(row, proposal)
    if not base_accepted and not global_override:
        return False, f"p3_preserve_base_rejection:{row.get('guard_reason', '')}"
    grounded, reason = grounding_check(row, proposal, check_deletions=not global_override)
    if not grounded:
        return False, reason
    if candidate in {"P0P2P3_r2_20260902", "P0P2P3_r3_20260902"} and technical_risk(proposal):
        return False, "p3_self_identified_technical_term_risk"
    if global_override and not base_accepted:
        return True, "p3_controlled_global_multiline_exception"
    return True, "p3_accepted"


def prepare(args: argparse.Namespace) -> tuple[list[dict[str, str]], dict[str, dict[str, Any]], Path, Path]:
    details = args.p2_result_dir / "details.csv"
    audit = args.p2_result_dir / "condition" / MODEL / "api_audit.jsonl"
    for path in (details, audit):
        if not path.is_file():
            raise FileNotFoundError(path)
    rows = read_csv(details)
    proposals = raw_proposals(read_jsonl(audit))
    ids = {row["audio_id"] for row in rows}
    if len(rows) != 19 or len(ids) != 19 or set(proposals) != ids:
        raise RuntimeError("P3 replay requires 19 matching P2 rows and raw proposals")
    return rows, proposals, details, audit


def run(args: argparse.Namespace) -> None:
    rows, proposals, details_path, audit_path = prepare(args)
    fixed = import_file("stage8_fixed_for_p3", SCRIPT_DIR / "run_fixed_baselines_v1.py")
    recalc = fixed.import_file("stage8_recalc_for_p3", fixed.RECALCULATE_PATH)
    converter = fixed.load_recalc_converter(recalc)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "protocol": "OCR_PROMPT_EXPERIMENT_PROTOCOL_V1.0",
        "component": "P3 deterministic guard",
        "parent": P2_EXPERIMENT,
        "mode": "offline replay; no API call",
        "samples": 19,
        "input_hashes": {"p2_details": sha256(details_path), "p2_api_audit": sha256(audit_path)},
        "candidates": {
            CANDIDATES[0]: "verbatim replacement grounding + reject deletion of cited OCR-supported content",
            CANDIDATES[1]: "r1 + reject self-identified technical/specialist terminology risk",
            CANDIDATES[2]: "r2 + controlled full-utterance exception for edit ratio<=0.65, >=2 cited lines, confidence>=0.90",
        },
        "r3_global_exception_max_edit_ratio": GLOBAL_EXCEPTION_MAX_EDIT_RATIO,
        "guard_inputs": "ASR, P2 raw edits/reasons/confidence/evidence ids, frozen selected OCR lines, prior guard reason",
        "forbidden_guard_inputs": "reference text and opportunity/audit labels are scoring-only",
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    comparison: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        candidate_rows: list[dict[str, Any]] = []
        for source in rows:
            row = dict(source)
            sample_id = row["audio_id"]
            accepted, reason = guard_decision(candidate, row, proposals[sample_id])
            qwen = row["qwen_raw"]
            proposed = row["proposed_text"]
            if accepted:
                corrected = proposed
                wer, cer = fixed.rescore(recalc, converter, row["language"], row["reference"], corrected)
                baseline_wer = float(row["baseline_wer"])
                baseline_cer = float(row["baseline_cer"])
                row.update({
                    "condition": candidate,
                    "corrected_text": corrected,
                    "accepted_changed": True,
                    "corrected_wer": round(wer, 8),
                    "corrected_cer": round(cer, 8),
                    "wer_delta": round(wer - baseline_wer, 8),
                    "cer_delta": round(cer - baseline_cer, 8),
                    "guard_reason": reason,
                })
            else:
                row.update({
                    "condition": candidate,
                    "corrected_text": qwen,
                    "accepted_changed": False,
                    "corrected_wer": row["baseline_wer"],
                    "corrected_cer": row["baseline_cer"],
                    "wer_delta": 0.0,
                    "cer_delta": 0.0,
                    "guard_reason": reason,
                })
            candidate_rows.append(row)
        candidate_dir = args.output_dir / candidate
        fixed.write_rows(candidate_dir / "details.csv", fixed.DETAIL_FIELDS, candidate_rows)
        overall = fixed.summarize_rows(candidate, "pilot", candidate_rows)
        fixed.write_rows(candidate_dir / "summary.csv", fixed.SUMMARY_FIELDS, [overall])
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in candidate_rows:
            groups[row["opportunity"]].append(row)
        by_group = [fixed.summarize_rows(candidate, group, groups[group]) for group in sorted(groups)]
        fixed.write_rows(candidate_dir / "summary_by_opportunity.csv", fixed.SUMMARY_FIELDS, by_group)
        comparison.append(overall)
        print(
            f"{candidate}: CER={float(overall['baseline_cer']):.4f}->{float(overall['corrected_cer']):.4f} "
            f"delta={float(overall['cer_delta']):+.4f} accepted={overall['accepted_changes']} "
            f"improved={overall['improved']} worsened={overall['worsened']} "
            f"false_corrections={overall['false_corrections']}"
        )
    fixed.write_rows(args.output_dir / "comparison.csv", fixed.SUMMARY_FIELDS, comparison)
    print("P3_OFFLINE_REPLAY=COMPLETED api_calls=0 samples=19 candidates=3")


def validate(args: argparse.Namespace) -> None:
    rows, proposals, details_path, audit_path = prepare(args)
    config_path = args.output_dir / "config.json"
    comparison_path = args.output_dir / "comparison.csv"
    for path in (config_path, comparison_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("mode") != "offline replay; no API call" or config.get("samples") != 19:
        raise RuntimeError("Unexpected P3 replay config")
    if config.get("r3_global_exception_max_edit_ratio") != GLOBAL_EXCEPTION_MAX_EDIT_RATIO:
        raise RuntimeError("P3 r3 global-exception edit-ratio cap changed")
    if config["input_hashes"] != {"p2_details": sha256(details_path), "p2_api_audit": sha256(audit_path)}:
        raise RuntimeError("P3 replay input hashes changed")
    comparison = read_csv(comparison_path)
    if [row["condition"] for row in comparison] != list(CANDIDATES):
        raise RuntimeError("P3 comparison candidate order mismatch")
    for candidate in CANDIDATES:
        details = read_csv(args.output_dir / candidate / "details.csv")
        if len(details) != 19 or len({row["audio_id"] for row in details}) != 19:
            raise RuntimeError(f"{candidate} does not contain 19 unique rows")
    print("STATUS=READY")
    print("experiment=P3_offline_guard_replay_v1")
    print("pilot_samples=19 candidates=3 api_calls=0")
    print("guard_reference_or_opportunity_input=0")
    print("holdout_exposed=0")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["preflight", "run", "validate"], required=True)
    parser.add_argument("--p2-result-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.p2_result_dir = args.p2_result_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    return args


def main() -> None:
    args = parse_args()
    rows, proposals, _, _ = prepare(args)
    if args.stage == "preflight":
        if edit_ratio_from_guard_reason({"guard_reason": "edit_ratio_0.600_above_0.450"}) != 0.6:
            raise RuntimeError("P3 edit-ratio guard parser self-check failed")
        if edit_ratio_from_guard_reason({"guard_reason": "edit_ratio_0.700_above_0.450"}) <= GLOBAL_EXCEPTION_MAX_EDIT_RATIO:
            raise RuntimeError("P3 edit-ratio cap self-check failed")
        print("P3_OFFLINE_PREFLIGHT=PASSED samples=19 raw_proposals=19 api_calls=0")
        print("candidates=" + ",".join(CANDIDATES))
        print(f"r3_global_exception_max_edit_ratio={GLOBAL_EXCEPTION_MAX_EDIT_RATIO}")
        print("Holdout is not exposed by this runner.")
        return
    if args.stage == "run":
        run(args)
        return
    validate(args)


if __name__ == "__main__":
    main()
