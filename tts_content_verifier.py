"""Batch ASR verification for synthesized speech completeness."""
from __future__ import annotations

import argparse
import difflib
import json
import math
import os
import re
from pathlib import Path
from typing import Any


def _language_code(value: Any) -> str:
    text = str(value or "").strip().lower().replace("_", "-")
    return {"chinese": "zh", "mandarin": "zh", "english": "en"}.get(
        text, text
    ).split("-", 1)[0]


def normalize_content_text(text: Any, language: Any) -> str:
    value = str(text or "").strip().lower().replace("’", "'")
    if _language_code(language) == "zh":
        return "".join(char for char in value if "\u4e00" <= char <= "\u9fff")
    return " ".join(re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", value))


def _latin_units(text: Any) -> list[str]:
    return re.findall(
        r"[a-z0-9]+(?:'[a-z0-9]+)?",
        str(text or "").lower().replace("’", "'"),
    )


def _cjk_units(text: Any) -> list[str]:
    return [
        char for char in str(text or "")
        if "\u4e00" <= char <= "\u9fff"
    ]


def _subtract_units(values: list[str], allowed: list[str]) -> list[str]:
    """Remove target-text units before looking for reference-only speech."""
    remaining = list(allowed)
    output = []
    for value in values:
        if value in remaining:
            remaining.remove(value)
        else:
            output.append(value)
    return output


def _sequence_coverage(reference: list[str], recognized: list[str]) -> tuple[int, float]:
    if not reference or not recognized:
        return 0, 0.0
    matcher = difflib.SequenceMatcher(None, reference, recognized)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    return matched, matched / len(reference)


def reference_leakage_quality(
    expected_text: Any,
    recognized_text: Any,
    reference_text: Any,
    language: Any,
) -> dict[str, Any]:
    """Detect source/reference words reproduced by the synthesized speech.

    Completeness normalization intentionally ignores the opposite script. That
    is correct for checking a translation, but unsafe for voice cloning: a
    Chinese target followed by leaked English reference speech could otherwise
    pass. This audit therefore examines reference-only evidence separately.
    """
    expected_raw = str(expected_text or "").strip()
    recognized_raw = str(recognized_text or "").strip()
    reference_raw = str(reference_text or "").strip()
    if not reference_raw:
        return {
            "reference_leakage_quality": "passed",
            "reference_leakage_detected": False,
            "reference_leakage_similarity": 0.0,
            "reference_leakage_reason": "no_reference_text",
            "reference_text": "",
        }

    target_language = _language_code(language)
    if target_language == "zh":
        reference_units = _subtract_units(
            _latin_units(reference_raw), _latin_units(expected_raw)
        )
        recognized_units = _subtract_units(
            _latin_units(recognized_raw), _latin_units(expected_raw)
        )
        meaningful = [unit for unit in reference_units if len(unit) >= 3]
        minimum_match = 1 if meaningful else 2
        evidence_kind = "reference_latin_in_chinese_target"
    else:
        reference_units = _subtract_units(
            _cjk_units(reference_raw), _cjk_units(expected_raw)
        )
        recognized_units = _subtract_units(
            _cjk_units(recognized_raw), _cjk_units(expected_raw)
        )
        minimum_match = 2
        evidence_kind = "reference_cjk_in_english_target"

    matched, coverage = _sequence_coverage(reference_units, recognized_units)
    cross_script_detected = bool(
        reference_units
        and matched >= minimum_match
        and coverage >= float(os.getenv(
            "AUTODUB_TTS_REFERENCE_LEAK_MIN_COVERAGE", "0.50"
        ))
    )

    # Same-language dubbing has no opposite-script signal. Keep a conservative
    # phrase-level check so a whole reference utterance appended to the target
    # is still blocked without rejecting an ordinary shared name or keyword.
    expected_same = normalize_content_text(expected_raw, target_language)
    recognized_same = normalize_content_text(recognized_raw, target_language)
    reference_same = normalize_content_text(reference_raw, target_language)
    same_units = (
        len(reference_same)
        if target_language == "zh"
        else len(reference_same.split())
    )
    same_similarity = (
        difflib.SequenceMatcher(None, reference_same, recognized_same).ratio()
        if reference_same and recognized_same else 0.0
    )
    expected_similarity = (
        difflib.SequenceMatcher(None, expected_same, recognized_same).ratio()
        if expected_same and recognized_same else 0.0
    )
    same_contained = bool(
        reference_same
        and reference_same != expected_same
        and reference_same in recognized_same
        and reference_same not in expected_same
    )
    same_language_detected = bool(
        same_units >= (4 if target_language == "zh" else 3)
        and (
            same_contained
            or (
                same_similarity >= float(os.getenv(
                    "AUTODUB_TTS_REFERENCE_LEAK_SAME_LANG_SIMILARITY", "0.82"
                ))
                and same_similarity >= expected_similarity + 0.12
            )
        )
    )
    detected = cross_script_detected or same_language_detected
    similarity = max(coverage, same_similarity if same_language_detected else 0.0)
    reason = ""
    if cross_script_detected:
        reason = f"{evidence_kind}:matched={matched},coverage={coverage:.3f}"
    elif same_language_detected:
        reason = (
            "reference_phrase_reproduced:"
            f"similarity={same_similarity:.3f},target_similarity={expected_similarity:.3f}"
        )
    return {
        "reference_leakage_quality": "failed" if detected else "passed",
        "reference_leakage_detected": detected,
        "reference_leakage_similarity": round(similarity, 6),
        "reference_leakage_reason": reason,
        "reference_text": reference_raw,
    }


def content_completeness_quality(
    expected_text: Any, recognized_text: Any, language: Any
) -> dict[str, Any]:
    expected = normalize_content_text(expected_text, language)
    recognized = normalize_content_text(recognized_text, language)
    ratio = difflib.SequenceMatcher(None, expected, recognized).ratio() if expected else 0.0
    threshold = float(os.getenv(
        "AUTODUB_TTS_CONTENT_ASR_MIN_SIMILARITY",
        "0.72" if _language_code(language) == "zh" else "0.68",
    ))
    # Short lines have little redundancy: one dropped Chinese character or
    # English word is meaningful, so require a slightly stronger match.
    units = len(expected) if _language_code(language) == "zh" else len(expected.split())
    if units <= 5:
        threshold = max(threshold, float(os.getenv(
            "AUTODUB_TTS_CONTENT_ASR_SHORT_MIN_SIMILARITY", "0.90"
        )))
    match_rule = "similarity"
    warnings = []
    passed = bool(expected and recognized and ratio >= threshold)

    # This verifier is a *completeness* gate, not a second translator.  Chinese
    # ASR commonly restores a short grammatical particle that the translation
    # omitted (for example expected ``手给我`` vs recognized ``把手给我``).
    # Rejecting that as swallowed speech discards a complete, natural clip and
    # needlessly sends it to a slower fallback model.  Permit only a small,
    # explicit set of harmless boundary particles; missing expected characters,
    # negation and arbitrary extra content remain blocking.
    safe_extra = "把请啊呀吧呢了的嘛哦嗯哈哎"
    if not passed and _language_code(language) == "zh" and expected:
        position = recognized.find(expected)
        if position >= 0:
            extra = recognized[:position] + recognized[position + len(expected):]
            max_extra = max(1, min(2, math.ceil(len(expected) * float(
                os.getenv("AUTODUB_TTS_CONTENT_ASR_MAX_SAFE_EXTRA_RATIO", "0.25")
            ))))
            if (
                extra
                and len(extra) <= max_extra
                and all(char in safe_extra for char in extra)
            ):
                passed = True
                match_rule = "expected_contained_with_safe_particles"
                warnings.append(f"tts_asr_safe_extra={extra}")
    return {
        "content_completeness_quality": "passed" if passed else "failed",
        "content_completeness_similarity": round(ratio, 6),
        "content_completeness_threshold": threshold,
        "content_expected_normalized": expected,
        "content_recognized_normalized": recognized,
        "content_asr_text": str(recognized_text or "").strip(),
        "content_completeness_match_rule": match_rule,
        "content_completeness_warnings": warnings,
        "content_completeness_reason": "" if passed else (
            f"tts_asr_similarity={ratio:.3f}<{threshold:.3f}"
        ),
    }


def synthesized_content_quality(
    expected_text: Any,
    recognized_text: Any,
    reference_text: Any,
    language: Any,
) -> dict[str, Any]:
    """Combine completeness with the non-negotiable reference-leak gate."""
    audit = content_completeness_quality(
        expected_text, recognized_text, language
    )
    leakage = reference_leakage_quality(
        expected_text, recognized_text, reference_text, language
    )
    audit.update(leakage)
    if leakage["reference_leakage_detected"]:
        audit.update(
            content_completeness_quality="failed",
            content_completeness_reason=(
                "reference_audio_text_leakage:"
                + str(leakage.get("reference_leakage_reason") or "detected")
            ),
            content_completeness_match_rule="reference_leakage_blocked",
        )
    return audit


def _result_text(result: Any) -> str:
    if isinstance(result, dict):
        return str(result.get("text") or "").strip()
    return str(getattr(result, "text", "") or "").strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    items = list(plan.get("items") or [])
    if not items:
        Path(args.output).write_text("{}", encoding="utf-8")
        return

    import torch
    from qwen_asr import Qwen3ASRModel

    model_dir = os.getenv("QWEN3_ASR_MODEL_DIR", "").strip()
    if not model_dir or not Path(model_dir).is_dir():
        raise FileNotFoundError(f"QWEN3_ASR_MODEL_DIR unavailable: {model_dir}")
    dtype_name = os.getenv("QWEN3_ASR_DTYPE", "bfloat16").lower()
    dtype = torch.bfloat16 if dtype_name in {"bf16", "bfloat16"} else (
        torch.float16 if dtype_name in {"fp16", "float16", "half"} else torch.float32
    )
    kwargs = {
        "dtype": dtype,
        "device_map": os.getenv("QWEN3_ASR_DEVICE_MAP", "cuda:0"),
        "max_inference_batch_size": int(os.getenv("QWEN3_MAX_INFERENCE_BATCH_SIZE", "4")),
        "max_new_tokens": int(os.getenv("QWEN3_MAX_NEW_TOKENS", "256")),
    }
    attention = os.getenv("QWEN3_ATTN_IMPLEMENTATION", "").strip()
    if attention:
        kwargs["attn_implementation"] = attention
    model = Qwen3ASRModel.from_pretrained(model_dir, **kwargs)
    paths = [str(item["file"]) for item in items]
    try:
        results = model.transcribe(
            audio=paths if len(paths) > 1 else paths[0], language=None
        )
        if not isinstance(results, list):
            results = [results]
        if len(results) != len(items):
            raise RuntimeError(
                f"TTS ASR returned {len(results)} rows for {len(items)} files"
            )
    except Exception as batch_error:
        print(f"TTS content ASR batch failed; retrying individually: {batch_error}")
        results = []
        for path in paths:
            try:
                result = model.transcribe(audio=path, language=None)
                if isinstance(result, list):
                    result = result[0]
                results.append(result)
            except Exception as item_error:
                print(f"TTS content ASR failed for {path}: {item_error}")
                results.append({"text": ""})
    payload = {}
    for item, result in zip(items, results):
        recognized = _result_text(result)
        payload[str(item["segment_id"])] = synthesized_content_quality(
            item.get("text"), recognized, item.get("reference_text"),
            item.get("language"),
        )
    Path(args.output).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
