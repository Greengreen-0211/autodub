#!/usr/bin/env python3
"""Offline v1.1 P4 task-boundary replay over frozen full-Dev P2/P3 outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


SCRIPT_DIR = Path(__file__).resolve().parent
STAGE_DIR = SCRIPT_DIR.parent
MODEL = "deepseek-v4-pro"
PARENT = "P0P2P3_r3_full_dev_20260902"
CANDIDATES = (
    "V11_P4_r1_code_switch_20260903",
    "V11_P4_r2_task_boundary_20260903",
)
EXPECTED_PARENT_DETAILS_HASH = "B4B62C9FEF05BBFDC69D2860F63EAFA7B99275B8B1BB89769424458FA5078983"
EXPECTED_PARENT_AUDIT_HASH = "AA3A8B2E72070605B3390BDFD99D0106ECFA631FE53289088F892D53CB51C3C1"
EXPECTED_FOLDS_HASH = "8139FD37EB32A730EF7E04898813CA383CEDBA257F4707115595A5AA2F31CE4C"
NEW_OCR_V0_CER = 0.1910178722352941
LATIN = re.compile(r"[A-Za-z]")
TASK_BOUNDARY_RISK = re.compile(
    r"\btranslat(?:e|ed|ion|ing)?\b|\bequivalent\b|same\s+meaning|"
    r"orthograph(?:ic|y)|simplified.{0,48}traditional|traditional.{0,48}simplified|"
    r"paraphras(?:e|ed|ing)?|stylistic",
    re.I | re.S,
)


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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def truthy(value: Any) -> bool:
    return str(value).strip().casefold() in {"1", "true", "yes"}


def response_objects(raw: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    if raw.get("choices"):
        yield raw
    if isinstance(raw.get("initial"), Mapping):
        yield from response_objects(raw["initial"])
    retries = raw.get("missing_id_retries")
    for retry in retries if isinstance(retries, list) else []:
        response = retry.get("response") if isinstance(retry, Mapping) else None
        if isinstance(response, Mapping):
            yield from response_objects(response)


def raw_proposals(path: Path) -> dict[str, dict[str, Any]]:
    proposals: dict[str, dict[str, Any]] = {}
    request_ids: list[str] = []
    for record in read_jsonl(path):
        if record.get("api_error"):
            raise RuntimeError(f"Parent audit contains API error: {record['api_error']}")
        request_ids.extend(str(item["id"]) for item in record.get("request_items", []))
        for raw in response_objects(record.get("response") or {}):
            payload = json.loads(raw["choices"][0]["message"]["content"])
            for proposal in payload.get("corrections", []):
                sample_id = str(proposal["id"])
                if sample_id in proposals:
                    raise RuntimeError(f"Duplicate raw proposal: {sample_id}")
                proposals[sample_id] = proposal
    if len(request_ids) != 85 or len(set(request_ids)) != 85 or len(proposals) != 85:
        raise RuntimeError("Parent audit/proposal coverage must be exactly 85 unique Dev ids")
    return proposals


def removes_latin_script(proposal: Mapping[str, Any]) -> bool:
    for edit in proposal.get("edits") or []:
        original = str(edit.get("original_span", ""))
        replacement = str(edit.get("replacement", ""))
        if LATIN.search(original) and not LATIN.search(replacement):
            return True
    return False


def self_declared_boundary_risk(proposal: Mapping[str, Any]) -> bool:
    texts = [str(proposal.get("reason", ""))]
    texts.extend(str(edit.get("reason", "")) for edit in proposal.get("edits") or [])
    return bool(TASK_BOUNDARY_RISK.search(" ".join(texts)))


def boundary_decision(candidate: str, parent_accepted: bool, proposal: Mapping[str, Any]) -> tuple[bool, str]:
    if not parent_accepted:
        return False, "v11_preserve_parent_rejection"
    if removes_latin_script(proposal):
        return False, "v11_code_switch_latin_script_removed"
    if candidate == CANDIDATES[1] and self_declared_boundary_risk(proposal):
        return False, "v11_self_declared_translation_or_style_rewrite"
    return True, "v11_p4_accepted"


def apply_decision(row: Mapping[str, Any], accepted: bool, reason: str, candidate: str) -> dict[str, Any]:
    result = dict(row)
    result["condition"] = candidate
    result["guard_reason"] = reason
    if accepted:
        result["accepted_changed"] = True
        return result
    result.update({
        "corrected_text": row["qwen_raw"],
        "accepted_changed": False,
        "corrected_wer": row["baseline_wer"],
        "corrected_cer": row["baseline_cer"],
        "wer_delta": 0.0,
        "cer_delta": 0.0,
    })
    return result


def group_summaries(rows: list[dict[str, Any]], key: str, condition: str, fixed) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    return [fixed.summarize_rows(condition, group, groups[group]) for group in sorted(groups)]


def candidate_gates(rows: list[dict[str, Any]], folds: Mapping[str, str], fixed) -> dict[str, Any]:
    overall = fixed.summarize_rows("candidate", "dev", rows)
    by_opp: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_fold: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_opp[row["opportunity"]].append(row)
        by_fold[folds[row["audio_id"]]].append(row)
    strong = fixed.summarize_rows("candidate", "Strong Opportunity", by_opp["Strong Opportunity"])
    safety = fixed.summarize_rows("candidate", "Safety Negative", by_opp["Safety Negative"])
    no_opp = fixed.summarize_rows("candidate", "No Opportunity", by_opp["No Opportunity"])
    negative_false = int(safety["false_corrections"]) + int(no_opp["false_corrections"])
    fold_delta = {
        fold: float(fixed.summarize_rows("candidate", f"fold_{fold}", by_fold[fold])["cer_delta"])
        for fold in ("1", "2", "3", "4")
    }
    gates = {
        "overall_cer_not_worse_than_new_ocr_v0": float(overall["corrected_cer"]) <= NEW_OCR_V0_CER + 1e-12,
        "strong_opportunity_positive": float(strong["cer_delta"]) < -1e-12,
        "safety_negative_false_corrections_zero": int(safety["false_corrections"]) == 0,
        "no_opportunity_false_corrections_zero": int(no_opp["false_corrections"]) == 0,
        "combined_negative_false_corrections_zero": negative_false == 0,
        "worsened_le_half_improved": int(overall["worsened"]) <= int(overall["improved"]) / 2,
        "at_least_three_improving_folds": sum(value < -1e-12 for value in fold_delta.values()) >= 3,
    }
    return {
        "pass": all(gates.values()),
        "gates": gates,
        "overall": overall,
        "strong": strong,
        "safety_negative": safety,
        "no_opportunity": no_opp,
        "combined_negative_false_corrections": negative_false,
        "fold_cer_delta": fold_delta,
    }


def prepare(args: argparse.Namespace):
    details_path = args.parent_dir / "details.csv"
    audit_path = args.parent_dir / "p2_condition" / MODEL / "api_audit.jsonl"
    for path in (details_path, audit_path, args.fold_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    actual = {
        "parent_details": sha256(details_path),
        "parent_api_audit": sha256(audit_path),
        "fold_manifest": sha256(args.fold_manifest),
    }
    expected = {
        "parent_details": EXPECTED_PARENT_DETAILS_HASH,
        "parent_api_audit": EXPECTED_PARENT_AUDIT_HASH,
        "fold_manifest": EXPECTED_FOLDS_HASH,
    }
    if actual != expected:
        raise RuntimeError(f"Frozen v1.1 input hash mismatch: actual={actual} expected={expected}")
    rows = read_csv(details_path)
    if len(rows) != 85 or len({row["audio_id"] for row in rows}) != 85 or {row["split"] for row in rows} != {"dev"}:
        raise RuntimeError("Parent details must contain 85 unique Dev rows")
    proposals = raw_proposals(audit_path)
    if set(proposals) != {row["audio_id"] for row in rows}:
        raise RuntimeError("Parent details and raw proposals differ")
    fold_rows = read_csv(args.fold_manifest)
    folds = {row["audio_id"]: row["fold"] for row in fold_rows}
    if set(folds) != set(proposals) or Counter(folds.values()) != Counter({"1": 22, "2": 21, "3": 21, "4": 21}):
        raise RuntimeError("Frozen Dev folds mismatch")
    return rows, proposals, folds, actual, details_path, audit_path


def run(args: argparse.Namespace) -> None:
    rows, proposals, folds, hashes, details_path, audit_path = prepare(args)
    fixed = import_file("stage8_fixed_v11_p4", SCRIPT_DIR / "run_fixed_baselines_v1.py")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "protocol": "OCR_PROMPT_EXPERIMENT_PROTOCOL_V1.1",
        "experiment": "V11_P4_round_20260903",
        "parent": PARENT,
        "mode": "offline deterministic replay; api_calls=0",
        "samples": 85,
        "candidates": list(CANDIDATES),
        "rules": {
            CANDIDATES[0]: "P3 r3 + reject removal of all Latin script from a Latin-containing original span",
            CANDIDATES[1]: "r1 + reject self-declared translation/equivalent/same-meaning/orthographic/paraphrase/stylistic rewrite",
        },
        "guard_inputs": "parent accepted flag and P2 raw action/edits/reasons only",
        "forbidden_guard_inputs": "reference, Opportunity, final ASR/OCR audit labels, owner notes",
        "input_hashes": hashes,
        "guard_source_hash": sha256(Path(__file__)),
        "holdout_exposed": False,
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    comparison: list[dict[str, Any]] = []
    selection: dict[str, Any] = {}
    for candidate in CANDIDATES:
        candidate_rows: list[dict[str, Any]] = []
        for row in rows:
            sample_id = row["audio_id"]
            accepted, reason = boundary_decision(candidate, truthy(row["accepted_changed"]), proposals[sample_id])
            candidate_rows.append(apply_decision(row, accepted, reason, candidate))
        candidate_dir = args.output_dir / candidate
        fixed.write_rows(candidate_dir / "details.csv", fixed.DETAIL_FIELDS, candidate_rows)
        overall = fixed.summarize_rows(candidate, "dev", candidate_rows)
        fixed.write_rows(candidate_dir / "summary.csv", fixed.SUMMARY_FIELDS, [overall])
        fixed.write_rows(
            candidate_dir / "summary_by_opportunity.csv", fixed.SUMMARY_FIELDS,
            group_summaries(candidate_rows, "opportunity", candidate, fixed),
        )
        fixed.write_rows(
            candidate_dir / "summary_by_language.csv", fixed.SUMMARY_FIELDS,
            group_summaries(candidate_rows, "language", candidate, fixed),
        )
        fold_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in candidate_rows:
            fold_groups[folds[row["audio_id"]]].append(row)
        fold_summary = [
            fixed.summarize_rows(candidate, f"fold_{fold}", fold_groups[fold])
            for fold in ("1", "2", "3", "4")
        ]
        fixed.write_rows(candidate_dir / "summary_by_fold.csv", fixed.SUMMARY_FIELDS, fold_summary)
        comparison.append(overall)
        gates = candidate_gates(candidate_rows, folds, fixed)
        selection[candidate] = gates
        print(
            f"{candidate}: CER={float(overall['baseline_cer']):.4f}->{float(overall['corrected_cer']):.4f} "
            f"delta={float(overall['cer_delta']):+.4f} accepted={overall['accepted_changes']} "
            f"improved={overall['improved']} worsened={overall['worsened']} "
            f"false_corrections={overall['false_corrections']} gates_pass={gates['pass']}"
        )
    fixed.write_rows(args.output_dir / "comparison.csv", fixed.SUMMARY_FIELDS, comparison)

    passing = [candidate for candidate in CANDIDATES if selection[candidate]["pass"]]
    selected = passing[0] if passing else None
    frozen = {
        "protocol": "OCR_PROMPT_EXPERIMENT_PROTOCOL_V1.1",
        "experiment": "V11_P4_round_20260903",
        "selected_candidate": selected,
        "candidate_frozen": bool(selected),
        "selection_policy": "first predeclared candidate in order that passes every v1.1 gate",
        "candidate_results": selection,
        "parent_input_hashes": hashes,
        "guard_source_hash": sha256(Path(__file__)),
        "selected_details_hash": sha256(args.output_dir / selected / "details.csv") if selected else None,
        "holdout_exposed": False,
    }
    (args.output_dir / "frozen_candidate.json").write_text(json.dumps(frozen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"V11_P4_OFFLINE_REPLAY=COMPLETED api_calls=0 selected={selected or 'NONE'}")


def validate(args: argparse.Namespace) -> None:
    rows, proposals, folds, hashes, _, _ = prepare(args)
    config_path = args.output_dir / "config.json"
    frozen_path = args.output_dir / "frozen_candidate.json"
    comparison_path = args.output_dir / "comparison.csv"
    for path in (config_path, frozen_path, comparison_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if config.get("input_hashes") != hashes or config.get("holdout_exposed"):
        raise RuntimeError("v1.1 config input hashes or holdout status changed")
    if config.get("guard_source_hash") != sha256(Path(__file__)):
        raise RuntimeError("v1.1 P4 guard source changed after replay")
    if frozen.get("selected_candidate") != CANDIDATES[1] or not frozen.get("candidate_frozen"):
        raise RuntimeError(f"Expected predeclared r2 to freeze, got {frozen.get('selected_candidate')}")
    if frozen.get("guard_source_hash") != sha256(Path(__file__)) or frozen.get("holdout_exposed"):
        raise RuntimeError("Frozen candidate manifest changed")
    for candidate in CANDIDATES:
        details_path = args.output_dir / candidate / "details.csv"
        details = read_csv(details_path)
        if len(details) != 85 or {row["audio_id"] for row in details} != set(proposals):
            raise RuntimeError(f"{candidate} does not cover 85 Dev rows")
        for row in details:
            expected, reason = boundary_decision(
                candidate,
                truthy(next(source["accepted_changed"] for source in rows if source["audio_id"] == row["audio_id"])),
                proposals[row["audio_id"]],
            )
            if truthy(row["accepted_changed"]) != expected or row["guard_reason"] != reason:
                raise RuntimeError(f"Decision mismatch for {candidate}/{row['audio_id']}")
    selected_details = args.output_dir / CANDIDATES[1] / "details.csv"
    if frozen.get("selected_details_hash") != sha256(selected_details):
        raise RuntimeError("Frozen selected candidate details hash mismatch")
    print("STATUS=READY")
    print("protocol=OCR_PROMPT_EXPERIMENT_PROTOCOL_V1.1")
    print("experiment=V11_P4_round_20260903 split=dev samples=85 api_calls=0")
    print(f"selected_candidate={CANDIDATES[1]}")
    print("guard_reference_or_opportunity_input=0")
    print("holdout_exposed=0")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["preflight", "run", "validate"], required=True)
    parser.add_argument("--parent-dir", type=Path, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.parent_dir = args.parent_dir.resolve()
    args.fold_manifest = args.fold_manifest.resolve()
    args.output_dir = args.output_dir.resolve()
    return args


def main() -> None:
    args = parse_args()
    rows, proposals, _, _, _, _ = prepare(args)
    if args.stage == "preflight":
        accepted = [row for row in rows if truthy(row["accepted_changed"])]
        r1_hits = sum(removes_latin_script(proposals[row["audio_id"]]) for row in accepted)
        r2_extra = sum(
            self_declared_boundary_risk(proposals[row["audio_id"]])
            and not removes_latin_script(proposals[row["audio_id"]])
            for row in accepted
        )
        print(f"V11_P4_PREFLIGHT=PASSED samples=85 parent_accepted={len(accepted)} api_calls=0")
        print(f"predeclared_hits=r1_code_switch:{r1_hits},r2_additional_boundary:{r2_extra}")
        print("guard_inputs=parent_accept_plus_raw_P2_proposal_only")
        print("Holdout is not exposed by this runner.")
        return
    if args.stage == "run":
        run(args)
        return
    validate(args)


if __name__ == "__main__":
    main()
