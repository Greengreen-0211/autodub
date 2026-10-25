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
        payload[str(item["segment_id"])] = content_completeness_quality(
            item.get("text"), _result_text(result), item.get("language")
        )
    Path(args.output).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
