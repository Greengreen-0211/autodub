#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate OCR-assisted correction of existing Qwen3-ASR results.

The reference transcript is used only for scoring. It is never sent to the
language model. This keeps the experiment honest and makes false corrections
visible instead of silently turning the test set into an answer key.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import math
import os
import re
import subprocess
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_METADATA = REPO_ROOT / "project_log" / "第4阶段" / "test_set" / "metadata.csv"
DEFAULT_ASR_RESULTS = REPO_ROOT / "project_log" / "第4阶段" / "results" / "results_lqz_router.csv"
DEFAULT_AUDIO_DIR = REPO_ROOT / "project_log" / "第4阶段" / "test_set" / "audio"
DEFAULT_VIDEO_DIR = REPO_ROOT / "project_log" / "第5阶段" / "test_set" / "video"
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results"

ASR_FIELDNAMES = [
    "audio_id", "asr_result", "wer", "cer", "time_cost", "success", "model", "language",
]

OCR_FIELDNAMES = [
    "video_id", "ocr_result", "frame_results_json", "success",
    "frames_requested", "frames_succeeded", "time_cost", "model",
]

DETAIL_FIELDNAMES = [
    "audio_id", "language", "tags", "model", "api_success", "api_error",
    "qwen_raw", "ocr_raw", "ocr_selected_json", "ocr_excluded_json",
    "proposed_text", "corrected_text", "proposed_changed", "accepted_changed",
    "confidence", "evidence_role", "evidence_line_ids", "reason", "guard_reason",
    "reference", "baseline_wer", "baseline_cer", "corrected_wer", "corrected_cer",
    "wer_delta", "cer_delta", "prompt_tokens", "completion_tokens", "total_tokens",
]

SUMMARY_FIELDNAMES = [
    "model", "scope", "samples", "api_failures", "accepted_changes",
    "improved", "worsened", "unchanged", "false_corrections",
    "baseline_wer", "corrected_wer", "wer_delta",
    "baseline_cer", "corrected_cer", "cer_delta",
    "prompt_tokens", "completion_tokens", "total_tokens",
]


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def append_jsonl(path: Path, item: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def first_existing(row: Mapping[str, Any], names: Sequence[str], default: str = "") -> str:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return default


def sample_sort_key(value: str) -> Tuple[str, int, str]:
    match = re.search(r"(\d+)$", value)
    return re.sub(r"\d+$", "", value), int(match.group(1)) if match else -1, value


def load_metadata(path: Path) -> Dict[str, Dict[str, str]]:
    rows = read_csv_rows(path)
    result: Dict[str, Dict[str, str]] = {}
    for row in rows:
        sample_id = first_existing(row, ["音频编号", "视频编号", "audio_id", "video_id", "id"])
        if not sample_id:
            continue
        result[sample_id] = {
            "language": first_existing(row, ["语言（zh/en……）", "语言", "language", "lang"]),
            "reference": first_existing(row, ["正确文本", "reference", "ref_text", "text"]),
            "tags": first_existing(row, ["标签（统一用英文,可有多个标签,逗号隔开）", "标签", "tags"]),
        }
    return result


def load_text_map(path: Path, text_fields: Sequence[str]) -> Dict[str, Dict[str, str]]:
    result: Dict[str, Dict[str, str]] = {}
    for row in read_csv_rows(path):
        sample_id = first_existing(row, ["audio_id", "video_id", "sample_id", "id"])
        if sample_id:
            result[sample_id] = {
                "text": first_existing(row, text_fields),
                "success": first_existing(row, ["success"], "True"),
            }
    return result


def normalize_metric_text(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text or "")).casefold()
    text = "".join(" " if unicodedata.category(ch).startswith("P") else ch for ch in text)
    return re.sub(r"\s+", " ", text).strip()


def is_cjk(char: str) -> bool:
    code = ord(char)
    return 0x3400 <= code <= 0x4DBF or 0x4E00 <= code <= 0x9FFF or 0xF900 <= code <= 0xFAFF


def wer_tokens(text: str) -> List[str]:
    text = normalize_metric_text(text)
    tokens: List[str] = []
    latin: List[str] = []
    for char in text:
        if is_cjk(char):
            if latin:
                tokens.append("".join(latin))
                latin = []
            tokens.append(char)
        elif char.isspace():
            if latin:
                tokens.append("".join(latin))
                latin = []
        else:
            latin.append(char)
    if latin:
        tokens.append("".join(latin))
    return tokens


def edit_distance(reference: Sequence[Any], hypothesis: Sequence[Any]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, ref_item in enumerate(reference, start=1):
        current = [i]
        for j, hyp_item in enumerate(hypothesis, start=1):
            current.append(min(
                current[-1] + 1,
                previous[j] + 1,
                previous[j - 1] + (ref_item != hyp_item),
            ))
        previous = current
    return previous[-1]


def compute_metrics(reference: str, hypothesis: str) -> Tuple[float, float]:
    ref_words = wer_tokens(reference)
    hyp_words = wer_tokens(hypothesis)
    ref_chars = list(normalize_metric_text(reference).replace(" ", ""))
    hyp_chars = list(normalize_metric_text(hypothesis).replace(" ", ""))
    wer = edit_distance(ref_words, hyp_words) / max(1, len(ref_words))
    cer = edit_distance(ref_chars, hyp_chars) / max(1, len(ref_chars))
    return wer, cer


def normalize_match_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return "".join(ch for ch in text if ch.isalnum() or is_cjk(ch))


def ngrams(text: str, size: int = 2) -> set:
    if not text:
        return set()
    if len(text) < size:
        return {text}
    return {text[index:index + size] for index in range(len(text) - size + 1)}


def lexical_similarity(left: str, right: str) -> float:
    left_norm = normalize_match_text(left)
    right_norm = normalize_match_text(right)
    if not left_norm or not right_norm:
        return 0.0
    sequence = difflib.SequenceMatcher(None, left_norm, right_norm).ratio()
    left_grams = ngrams(left_norm)
    right_grams = ngrams(right_norm)
    union = left_grams | right_grams
    jaccard = len(left_grams & right_grams) / len(union) if union else 0.0
    containment = min(
        len(left_grams & right_grams) / max(1, len(left_grams)),
        len(left_grams & right_grams) / max(1, len(right_grams)),
    )
    return 0.50 * sequence + 0.35 * jaccard + 0.15 * containment


def structural_noise_reason(line: str) -> str:
    compact = re.sub(r"\s+", "", line)
    if not compact:
        return "empty"
    if re.search(r"(?:https?://|www\.)\S+", line, flags=re.IGNORECASE):
        return "url"
    if re.fullmatch(r"[\[(]?\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?[\])]?", compact):
        return "timecode"
    if re.fullmatch(r"[\d０-９]+(?:[.,，]\d+)?", compact):
        return "pure_number"
    if all(not ch.isalnum() and not is_cjk(ch) for ch in compact):
        return "pure_symbol"
    return ""


def load_external_patterns(path: Optional[Path]) -> List[re.Pattern[str]]:
    if not path:
        return []
    patterns: List[re.Pattern[str]] = []
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            patterns.append(re.compile(line, flags=re.IGNORECASE))
    return patterns


def subtitle_likeness(line: str) -> float:
    compact = re.sub(r"\s+", "", line)
    if not compact:
        return 0.0
    meaningful = sum(ch.isalnum() or is_cjk(ch) for ch in compact) / len(compact)
    length = len(compact)
    length_score = 1.0 if 4 <= length <= 80 else 0.65 if 2 <= length <= 140 else 0.25
    sentence_bonus = 0.10 if re.search(r"[。！？!?.,，]", line) else 0.0
    return min(1.0, 0.65 * meaningful + 0.35 * length_score + sentence_bonus)


def split_ocr_lines(text: str) -> List[str]:
    text = str(text or "").replace("```", "\n")
    raw_lines = re.split(r"[\r\n]+", text)
    return [re.sub(r"\s+", " ", line).strip(" \t|-") for line in raw_lines if line.strip()]


def select_ocr_evidence(
    raw_ocr: str,
    asr_text: str,
    max_lines: int,
    lexical_quota: int,
    external_patterns: Sequence[re.Pattern[str]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    candidates: List[Dict[str, Any]] = []
    excluded: List[Dict[str, Any]] = []
    seen = set()
    for raw_line in split_ocr_lines(raw_ocr):
        reason = structural_noise_reason(raw_line)
        if reason:
            excluded.append({"text": raw_line, "reason": reason})
            continue
        normalized = normalize_match_text(raw_line)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        if any(pattern.search(raw_line) for pattern in external_patterns):
            reason = "external_watermark_pattern"
        if reason:
            excluded.append({"text": raw_line, "reason": reason})
            continue
        candidates.append({
            "text": raw_line,
            "lexical_score": round(lexical_similarity(raw_line, asr_text), 6),
            "subtitle_score": round(subtitle_likeness(raw_line), 6),
        })

    lexical_ranked = sorted(candidates, key=lambda item: (item["lexical_score"], item["subtitle_score"]), reverse=True)
    selected: List[Dict[str, Any]] = []
    for item in lexical_ranked:
        if len(selected) >= min(lexical_quota, max_lines):
            break
        if item["lexical_score"] >= 0.05:
            selected.append(item)

    selected_texts = {item["text"] for item in selected}
    subtitle_ranked = sorted(candidates, key=lambda item: (item["subtitle_score"], item["lexical_score"]), reverse=True)
    for item in subtitle_ranked:
        if len(selected) >= max_lines:
            break
        if item["text"] not in selected_texts:
            selected.append(item)
            selected_texts.add(item["text"])

    for index, item in enumerate(selected, start=1):
        item["line_id"] = f"L{index:02d}"
        item["selection"] = "lexical" if item in lexical_ranked[:lexical_quota] and item["lexical_score"] >= 0.05 else "subtitle_candidate"
    return selected, excluded


def extract_json_object(text: str) -> Dict[str, Any]:
    content = str(text or "").strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?", "", content).strip()
        content = re.sub(r"```$", "", content).strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start >= 0 and end > start:
            return json.loads(content[start:end + 1])
        raise


def digit_tokens(text: str) -> set:
    return set(re.findall(r"\d+(?:[.:：/\-]\d+)*", str(text or "")))


def normalized_edit_ratio(left: str, right: str) -> float:
    left_norm = normalize_match_text(left)
    right_norm = normalize_match_text(right)
    return edit_distance(list(left_norm), list(right_norm)) / max(1, len(left_norm))


def validate_proposal(
    qwen_text: str,
    proposal: Mapping[str, Any],
    selected_lines: Sequence[Mapping[str, Any]],
    confidence_threshold: float,
    max_edit_ratio: float,
) -> Tuple[str, bool, str]:
    proposed = str(proposal.get("text") or "").strip()
    changed = bool(proposal.get("changed", False))
    try:
        confidence = float(proposal.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    role = str(proposal.get("evidence_role") or "uncertain").strip()
    raw_ids = proposal.get("evidence_line_ids") or []
    evidence_ids = [str(value) for value in raw_ids] if isinstance(raw_ids, list) else []
    valid_ids = {str(item.get("line_id")) for item in selected_lines}

    if not changed:
        return qwen_text, False, "model_changed_false"
    if not proposed:
        return qwen_text, False, "empty_proposal"
    if confidence < confidence_threshold:
        return qwen_text, False, f"confidence_below_{confidence_threshold:.2f}"
    if not evidence_ids or not set(evidence_ids).issubset(valid_ids):
        return qwen_text, False, "missing_or_invalid_evidence"
    if role in {"ui_or_watermark", "uncertain"}:
        return qwen_text, False, f"unsafe_evidence_role:{role}"
    new_digits = digit_tokens(proposed) - digit_tokens(qwen_text)
    if new_digits:
        return qwen_text, False, f"introduced_digits:{sorted(new_digits)}"
    if normalize_match_text(proposed) == normalize_match_text(qwen_text):
        return qwen_text, False, "punctuation_or_case_only"
    if len(qwen_text) >= 20 and len(proposed) > len(qwen_text) * 1.45:
        return qwen_text, False, "over_expansion"
    ratio = normalized_edit_ratio(qwen_text, proposed)
    if role == "translated_subtitle":
        role_limit = min(max_edit_ratio, 0.25)
    elif role == "proper_noun_or_title" and confidence >= 0.95:
        role_limit = max(max_edit_ratio, 0.60)
    else:
        role_limit = max_edit_ratio
    if ratio > role_limit:
        return qwen_text, False, f"edit_ratio_{ratio:.3f}_above_{role_limit:.3f}"
    return proposed, proposed != qwen_text, "accepted"


SYSTEM_PROMPT = """
You are a conservative ASR verifier for a multilingual video-dubbing pipeline.
The Qwen3-ASR transcript is the primary acoustic hypothesis. OCR lines are noisy
visual evidence and may contain subtitles, translated subtitles, titles, UI,
watermarks, comments, advertisements, or unrelated scene text.

For every item:
1. Infer the spoken/source language from source_language and qwen_asr.
2. Classify the useful evidence as one of: same_language_subtitle,
   translated_subtitle, bilingual_subtitle, proper_noun_or_title,
   ui_or_watermark, uncertain.
3. Self-check qwen_asr for a concrete local error. Consider semantic context,
   phonetic plausibility, code-switching, names and places. A translated subtitle
   may confirm meaning or a uniquely recoverable proper noun, but must never turn
   the source transcript into a translation.
4. Default to unchanged. Correct only a small, strongly supported span. Never
   polish style, add facts, complete missing sentences, copy UI text, or use the
   reference transcript (it is not provided).
5. confidence is evidence confidence, from 0 to 1. Set changed=false when below
   0.85 or when evidence is ambiguous.

Return strict JSON only:
{"corrections":[{"id":"clip_001","text":"source-language transcript","changed":false,"confidence":0.0,"evidence_role":"uncertain","evidence_line_ids":[],"reason":"short audit reason"}]}
""".strip()


def deepseek_request(
    args: argparse.Namespace,
    model: str,
    items: Sequence[Mapping[str, Any]],
    require_all: bool = True,
    system_prompt: Optional[str] = None,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    import requests

    api_key = args.deepseek_api_key or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not configured")

    session = requests.Session()
    # An explicit proxy must not be overridden by stale HTTP(S)_PROXY values.
    session.trust_env = args.trust_env and not bool(args.proxy_url)
    if args.proxy_url:
        session.proxies.update({"http": args.proxy_url, "https": args.proxy_url})

    body: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt or SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps({"items": items}, ensure_ascii=False)},
        ],
        "thinking": {"type": args.thinking},
        "response_format": {"type": "json_object"},
        "max_tokens": args.max_tokens,
        "stream": False,
    }
    response = session.post(
        args.deepseek_base_url.rstrip("/") + "/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=body,
        timeout=args.deepseek_timeout,
    )
    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:500]}")
    raw = response.json()
    payload = extract_json_object(raw["choices"][0]["message"].get("content", ""))
    corrections = {
        str(item.get("id", "")).strip(): item
        for item in payload.get("corrections", [])
        if str(item.get("id", "")).strip()
    }
    expected_ids = {str(item.get("id", "")).strip() for item in items}
    missing_ids = sorted(expected_ids - set(corrections), key=sample_sort_key)
    if require_all and missing_ids:
        raise RuntimeError(f"model response omitted sample ids: {missing_ids}")
    return corrections, raw.get("usage") or {}, raw


def add_usage(total: Dict[str, Any], extra: Mapping[str, Any]) -> None:
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        total[key] = float(total.get(key, 0) or 0) + float(extra.get(key, 0) or 0)


def retryable_api_error(exc: Exception) -> bool:
    try:
        import requests

        if isinstance(exc, requests.exceptions.RequestException):
            return True
    except ImportError:
        pass
    message = str(exc)
    return any(
        marker in message
        for marker in (
            "HTTP 429",
            "HTTP 500",
            "HTTP 502",
            "HTTP 503",
            "HTTP 504",
        )
    )


def deepseek_request_with_retry(
    args: argparse.Namespace,
    model: str,
    items: Sequence[Mapping[str, Any]],
    require_all: bool,
    system_prompt: Optional[str] = None,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    attempts = max(1, int(args.api_retries))
    for attempt in range(1, attempts + 1):
        try:
            return deepseek_request(
                args,
                model,
                items,
                require_all=require_all,
                system_prompt=system_prompt,
            )
        except Exception as exc:
            if attempt >= attempts or not retryable_api_error(exc):
                raise
            delay = max(0.0, float(args.api_retry_backoff)) * (2 ** (attempt - 1))
            print(
                f"[{model}] transient API error on attempt {attempt}/{attempts}: "
                f"{type(exc).__name__}: {exc}; retrying in {delay:.1f}s"
            )
            time.sleep(delay)
    raise AssertionError("unreachable")


def check_deepseek_api(args: argparse.Namespace) -> None:
    import requests

    api_key = args.deepseek_api_key or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not configured")
    session = requests.Session()
    # An explicit proxy must not be overridden by stale HTTP(S)_PROXY values.
    session.trust_env = args.trust_env and not bool(args.proxy_url)
    if args.proxy_url:
        session.proxies.update({"http": args.proxy_url, "https": args.proxy_url})
    response = session.get(
        args.deepseek_base_url.rstrip("/") + "/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=min(args.deepseek_timeout, 30.0),
    )
    if response.status_code != 200:
        raise RuntimeError(f"DeepSeek model-list check failed HTTP {response.status_code}: {response.text[:500]}")
    available = sorted(
        str(item.get("id", ""))
        for item in response.json().get("data", [])
        if item.get("id")
    )
    missing = [model for model in args.models if model not in available]
    print(f"DeepSeek API reachable. Available models: {available}")
    if missing:
        raise RuntimeError(f"Requested models are unavailable: {missing}")
    print("DeepSeek API model check passed.")


def parse_fractions(value: str) -> List[float]:
    fractions = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not fractions or any(not 0.0 <= item <= 1.0 for item in fractions):
        raise argparse.ArgumentTypeError("OCR fractions must be comma-separated values in [0, 1]")
    return fractions


def torch_dtype_from_name(name: str):
    import torch

    lowered = str(name).strip().casefold()
    if lowered in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if lowered in {"fp16", "float16", "half"}:
        return torch.float16
    return torch.float32


def load_qwen_model(args: argparse.Namespace):
    try:
        from qwen_asr import Qwen3ASRModel
    except ImportError as exc:
        raise ImportError(
            "qwen_asr is unavailable in this environment. Run the ASR stage in autodub_server."
        ) from exc

    kwargs: Dict[str, Any] = {
        "dtype": torch_dtype_from_name(args.qwen_dtype),
        "device_map": args.qwen_device_map,
        "max_inference_batch_size": args.asr_batch_size,
        "max_new_tokens": args.qwen_max_new_tokens,
    }
    if args.qwen_attn_implementation:
        kwargs["attn_implementation"] = args.qwen_attn_implementation
    print(
        f"Loading Qwen3-ASR: {args.qwen_model_dir}; device_map={args.qwen_device_map}, "
        f"dtype={args.qwen_dtype}, batch={args.asr_batch_size}"
    )
    return Qwen3ASRModel.from_pretrained(str(args.qwen_model_dir), **kwargs)


def parse_qwen_result(result: Any) -> Tuple[str, str]:
    if isinstance(result, Mapping):
        return str(result.get("text", "")).strip(), str(result.get("language", "") or "").strip()
    return str(getattr(result, "text", "")).strip(), str(getattr(result, "language", "") or "").strip()


def transcribe_qwen_batch(model: Any, audio_paths: Sequence[Path]) -> Tuple[List[str], List[str], List[bool], List[float]]:
    started = time.monotonic()
    try:
        audio_arg: Any = [str(path) for path in audio_paths] if len(audio_paths) > 1 else str(audio_paths[0])
        results = model.transcribe(audio=audio_arg, language=None)
        if not isinstance(results, list):
            results = [results]
        if len(results) != len(audio_paths):
            raise RuntimeError(f"Qwen3 returned {len(results)} results for {len(audio_paths)} inputs")
        parsed = [parse_qwen_result(item) for item in results]
        per_item = (time.monotonic() - started) / max(1, len(audio_paths))
        return (
            [item[0] for item in parsed],
            [item[1] for item in parsed],
            [True] * len(audio_paths),
            [per_item] * len(audio_paths),
        )
    except Exception as batch_error:
        print(f"Qwen3 batch failed; retrying one by one: {batch_error}")

    texts: List[str] = []
    languages: List[str] = []
    successes: List[bool] = []
    costs: List[float] = []
    for path in audio_paths:
        item_started = time.monotonic()
        try:
            result = model.transcribe(audio=str(path), language=None)
            if isinstance(result, list):
                result = result[0]
            text, language = parse_qwen_result(result)
            texts.append(text)
            languages.append(language)
            successes.append(True)
        except Exception as exc:
            print(f"Qwen3 failed for {path.name}: {type(exc).__name__}: {exc}")
            texts.append("")
            languages.append("")
            successes.append(False)
        costs.append(time.monotonic() - item_started)
    return texts, languages, successes, costs


def find_audio_file(audio_dir: Path, sample_id: str) -> Optional[Path]:
    for extension in (".mp3", ".wav", ".flac", ".m4a", ".ogg"):
        candidate = audio_dir / f"{sample_id}{extension}"
        if candidate.is_file():
            return candidate
    return None


def run_asr_stage(
    args: argparse.Namespace,
    metadata: Mapping[str, Mapping[str, str]],
    sample_ids: Sequence[str],
) -> Path:
    existing: Dict[str, Dict[str, str]] = {}
    if args.asr_results.exists() and not args.overwrite:
        existing = {row["audio_id"]: row for row in read_csv_rows(args.asr_results) if row.get("audio_id")}
        print(f"ASR resume: {len(existing)} existing rows")

    audio_paths = {sample_id: find_audio_file(args.audio_dir, sample_id) for sample_id in sample_ids}
    missing = [sample_id for sample_id, path in audio_paths.items() if path is None]
    if missing:
        raise FileNotFoundError(f"Missing audio for {len(missing)} samples; examples: {missing[:5]}")
    pending = [sample_id for sample_id in sample_ids if sample_id not in existing]
    rows: List[Dict[str, Any]] = list(existing.values())
    print(f"Qwen3-ASR pending: {len(pending)}/{len(sample_ids)}")
    if pending:
        model = load_qwen_model(args)
        for offset in range(0, len(pending), args.asr_batch_size):
            batch_ids = pending[offset:offset + args.asr_batch_size]
            batch_paths = [audio_paths[sample_id] for sample_id in batch_ids]
            texts, languages, successes, costs = transcribe_qwen_batch(model, batch_paths)  # type: ignore[arg-type]
            for sample_id, text, language, success, cost in zip(batch_ids, texts, languages, successes, costs):
                reference = metadata[sample_id].get("reference", "")
                wer, cer = compute_metrics(reference, text) if success else (1.0, 1.0)
                rows.append({
                    "audio_id": sample_id,
                    "asr_result": text,
                    "wer": round(wer, 8),
                    "cer": round(cer, 8),
                    "time_cost": round(cost, 3),
                    "success": success,
                    "model": "Qwen3-ASR-1.7B",
                    "language": language,
                })
            rows = sorted(rows, key=lambda row: sample_sort_key(str(row["audio_id"])))
            write_csv_rows(args.asr_results, ASR_FIELDNAMES, rows)
            print(f"[ASR {offset + len(batch_ids):03d}/{len(pending):03d}] saved")
    return args.asr_results


def probe_duration(video_path: Path, ffprobe: str) -> float:
    result = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(video_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def ocr_row_complete(row: Mapping[str, str]) -> bool:
    try:
        frame_results = json.loads(row.get("frame_results_json", ""))
    except (TypeError, ValueError, json.JSONDecodeError):
        return False
    return bool(frame_results) and all(
        isinstance(item, dict) and not item.get("error")
        for item in frame_results
    )


def run_ocr_stage(args: argparse.Namespace, sample_ids: Sequence[str]) -> Path:
    import requests

    output = args.ocr_results
    existing: Dict[str, Dict[str, str]] = {}
    if output.exists() and not args.overwrite:
        loaded_rows = [row for row in read_csv_rows(output) if row.get("video_id")]
        existing = {
            row["video_id"]: row
            for row in loaded_rows
            if ocr_row_complete(row)
        }
        print(f"OCR resume: {len(existing)} existing rows")
        failed_ids = [row["video_id"] for row in loaded_rows if not ocr_row_complete(row)]
        if failed_ids:
            print(f"OCR retrying incomplete rows: {failed_ids}")

    rows = list(existing.values())
    session = requests.Session()
    session.trust_env = False
    health_url = args.ocr_url.rsplit("/", 1)[0] + "/health"
    health = session.get(health_url, timeout=10)
    health.raise_for_status()
    print(f"OCR service: {health_url} status={health.json().get('status')}")

    pending = [sample_id for sample_id in sample_ids if sample_id not in existing]
    for index, sample_id in enumerate(pending, start=1):
        video_path = args.video_dir / f"{sample_id}.mp4"
        started = time.monotonic()
        frame_results: List[Dict[str, Any]] = []
        if not video_path.is_file():
            frame_results.append({"error": f"missing video: {video_path}"})
        else:
            try:
                duration = probe_duration(video_path, args.ffprobe)
                timestamps = sorted({round(max(0.0, min(duration - 0.05, duration * fraction)), 3) for fraction in args.ocr_fractions})
                for frame_number, timestamp in enumerate(timestamps, start=1):
                    request_id = f"{sample_id}-{frame_number}"
                    try:
                        response = session.post(
                            args.ocr_url,
                            json={
                                "video_path": str(video_path.resolve()),
                                "timestamp_seconds": timestamp,
                                "request_id": request_id,
                            },
                            timeout=args.ocr_timeout,
                        )
                        response.raise_for_status()
                        data = response.json()
                        frame_results.append({
                            "timestamp": timestamp,
                            "text": str(data.get("text") or "").strip(),
                            "timings": data.get("timings") or {},
                            "request_success": True,
                        })
                    except Exception as exc:
                        frame_results.append({"timestamp": timestamp, "error": f"{type(exc).__name__}: {exc}"})
            except Exception as exc:
                frame_results.append({"error": f"{type(exc).__name__}: {exc}"})

        successful_requests = [item for item in frame_results if not item.get("error")]
        text_results = [item for item in successful_requests if item.get("text")]
        combined_lines: List[str] = []
        seen = set()
        for item in text_results:
            for line in split_ocr_lines(str(item.get("text", ""))):
                key = normalize_match_text(line)
                if key and key not in seen:
                    seen.add(key)
                    combined_lines.append(line)
        rows.append({
            "video_id": sample_id,
            "ocr_result": "\n".join(combined_lines),
            "frame_results_json": json.dumps(frame_results, ensure_ascii=False),
            "success": bool(successful_requests) and len(successful_requests) == len(frame_results),
            "frames_requested": len(frame_results),
            "frames_succeeded": len(successful_requests),
            "time_cost": round(time.monotonic() - started, 3),
            "model": "HunyuanOCR-multiframe",
        })
        rows = sorted(rows, key=lambda row: sample_sort_key(str(row["video_id"])))
        write_csv_rows(output, OCR_FIELDNAMES, rows)
        print(
            f"[OCR {index:03d}/{len(pending):03d}] {sample_id}: "
            f"requests={len(successful_requests)}/{len(frame_results)}, "
            f"text_frames={len(text_results)}, lines={len(combined_lines)}"
        )
    return output


def model_slug(model: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", model)


def run_model_evaluation(
    args: argparse.Namespace,
    model: str,
    sample_ids: Sequence[str],
    metadata: Mapping[str, Mapping[str, str]],
    asr_map: Mapping[str, Mapping[str, str]],
    ocr_map: Mapping[str, Mapping[str, str]],
) -> Path:
    output_dir = args.results_dir / model_slug(model)
    details_path = output_dir / "details.csv"
    audit_path = output_dir / "api_audit.jsonl"
    existing: Dict[str, Dict[str, str]] = {}
    if details_path.exists() and not args.overwrite:
        existing = {row["audio_id"]: row for row in read_csv_rows(details_path) if row.get("audio_id")}
        print(f"[{model}] resume: {len(existing)} existing rows")
    elif args.overwrite and audit_path.exists():
        audit_path.unlink()

    rows: List[Dict[str, Any]] = list(existing.values())
    pending = [sample_id for sample_id in sample_ids if sample_id not in existing]
    patterns = load_external_patterns(args.noise_patterns)

    for offset in range(0, len(pending), args.batch_size):
        batch_ids = pending[offset:offset + args.batch_size]
        api_items: List[Dict[str, Any]] = []
        evidence_by_id: Dict[str, Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]] = {}
        for sample_id in batch_ids:
            qwen_text = asr_map[sample_id]["text"]
            raw_ocr = ocr_map.get(sample_id, {}).get("text", "")
            selected, excluded = select_ocr_evidence(
                raw_ocr,
                qwen_text,
                args.max_ocr_lines,
                args.lexical_quota,
                patterns,
            )
            evidence_by_id[sample_id] = selected, excluded
            api_items.append({
                "id": sample_id,
                "source_language": metadata[sample_id].get("language", ""),
                "qwen_asr": qwen_text,
                "ocr_candidates": selected,
            })

        corrections: Dict[str, Dict[str, Any]] = {}
        usage: Dict[str, Any] = {}
        api_error = ""
        raw_response: Dict[str, Any] = {}
        try:
            corrections, usage, initial_response = deepseek_request_with_retry(
                args,
                model,
                api_items,
                require_all=False,
            )
            usage = dict(usage)
            raw_response = initial_response
            expected_ids = {str(item["id"]) for item in api_items}
            missing_ids = sorted(expected_ids - set(corrections), key=sample_sort_key)
            retry_audit: List[Dict[str, Any]] = []
            if missing_ids:
                print(
                    f"[{model}] batch {offset + 1}-{offset + len(batch_ids)} "
                    f"omitted {missing_ids}; retrying each missing id"
                )
            items_by_id = {str(item["id"]): item for item in api_items}
            for missing_id in missing_ids:
                resolved = False
                last_error = ""
                for attempt in range(1, args.missing_id_retries + 1):
                    try:
                        retry_corrections, retry_usage, retry_response = deepseek_request_with_retry(
                            args,
                            model,
                            [items_by_id[missing_id]],
                            require_all=True,
                        )
                        corrections.update(retry_corrections)
                        add_usage(usage, retry_usage)
                        retry_audit.append({
                            "id": missing_id,
                            "attempt": attempt,
                            "success": True,
                            "response": retry_response,
                        })
                        resolved = True
                        break
                    except Exception as exc:
                        last_error = f"{type(exc).__name__}: {exc}"
                        retry_audit.append({
                            "id": missing_id,
                            "attempt": attempt,
                            "success": False,
                            "error": last_error,
                        })
                if not resolved:
                    raise RuntimeError(
                        f"model response still omitted {missing_id} after "
                        f"{args.missing_id_retries} retries: {last_error}"
                    )
            if retry_audit:
                raw_response = {
                    "initial": initial_response,
                    "missing_id_retries": retry_audit,
                }
        except Exception as exc:
            api_error = f"{type(exc).__name__}: {exc}"
            print(f"[{model}] batch {offset + 1}-{offset + len(batch_ids)} failed: {api_error}")

        append_jsonl(audit_path, {
            "model": model,
            "sample_ids": batch_ids,
            "thinking": args.thinking,
            "request_items": api_items,
            "usage": usage,
            "api_error": api_error,
            "response": raw_response,
        })
        if api_error and not args.continue_on_api_error:
            raise RuntimeError(
                f"Stopping after the first API failure. Inspect {audit_path}: {api_error}"
            )

        batch_size = max(1, len(batch_ids))
        for sample_id in batch_ids:
            qwen_text = asr_map[sample_id]["text"]
            raw_ocr = ocr_map.get(sample_id, {}).get("text", "")
            selected, excluded = evidence_by_id[sample_id]
            proposal = corrections.get(sample_id, {})
            corrected, accepted, guard_reason = validate_proposal(
                qwen_text,
                proposal,
                selected,
                args.confidence_threshold,
                args.max_edit_ratio,
            )
            reference = metadata[sample_id].get("reference", "")
            baseline_wer, baseline_cer = compute_metrics(reference, qwen_text)
            corrected_wer, corrected_cer = compute_metrics(reference, corrected)
            evidence_ids = proposal.get("evidence_line_ids") or []
            rows.append({
                "audio_id": sample_id,
                "language": metadata[sample_id].get("language", ""),
                "tags": metadata[sample_id].get("tags", ""),
                "model": model,
                "api_success": not bool(api_error),
                "api_error": api_error,
                "qwen_raw": qwen_text,
                "ocr_raw": raw_ocr,
                "ocr_selected_json": json.dumps(selected, ensure_ascii=False),
                "ocr_excluded_json": json.dumps(excluded, ensure_ascii=False),
                "proposed_text": str(proposal.get("text") or ""),
                "corrected_text": corrected,
                "proposed_changed": bool(proposal.get("changed", False)),
                "accepted_changed": accepted,
                "confidence": proposal.get("confidence", ""),
                "evidence_role": proposal.get("evidence_role", ""),
                "evidence_line_ids": json.dumps(evidence_ids, ensure_ascii=False),
                "reason": proposal.get("reason", ""),
                "guard_reason": guard_reason if not api_error else "api_failure_fallback",
                "reference": reference,
                "baseline_wer": round(baseline_wer, 8),
                "baseline_cer": round(baseline_cer, 8),
                "corrected_wer": round(corrected_wer, 8),
                "corrected_cer": round(corrected_cer, 8),
                "wer_delta": round(corrected_wer - baseline_wer, 8),
                "cer_delta": round(corrected_cer - baseline_cer, 8),
                "prompt_tokens": round(float(usage.get("prompt_tokens", 0)) / batch_size, 3),
                "completion_tokens": round(float(usage.get("completion_tokens", 0)) / batch_size, 3),
                "total_tokens": round(float(usage.get("total_tokens", 0)) / batch_size, 3),
            })

        rows = sorted(rows, key=lambda row: sample_sort_key(str(row["audio_id"])))
        write_csv_rows(details_path, DETAIL_FIELDNAMES, rows)
        print(f"[{model}] {offset + len(batch_ids):03d}/{len(pending):03d} saved")
    return details_path


def as_float(row: Mapping[str, Any], key: str) -> float:
    try:
        return float(row.get(key, 0) or 0)
    except (TypeError, ValueError):
        return 0.0


def bool_value(value: Any) -> bool:
    return str(value).strip().casefold() in {"1", "true", "yes"}


def summarize_rows(model: str, scope: str, rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    count = len(rows)
    baseline_wer = sum(as_float(row, "baseline_wer") for row in rows) / max(1, count)
    corrected_wer = sum(as_float(row, "corrected_wer") for row in rows) / max(1, count)
    baseline_cer = sum(as_float(row, "baseline_cer") for row in rows) / max(1, count)
    corrected_cer = sum(as_float(row, "corrected_cer") for row in rows) / max(1, count)
    improved = sum(as_float(row, "cer_delta") < -1e-9 for row in rows)
    worsened = sum(as_float(row, "cer_delta") > 1e-9 for row in rows)
    false_corrections = sum(
        bool_value(row.get("accepted_changed"))
        and as_float(row, "baseline_cer") <= 1e-9
        and as_float(row, "corrected_cer") > 1e-9
        for row in rows
    )
    return {
        "model": model,
        "scope": scope,
        "samples": count,
        "api_failures": sum(not bool_value(row.get("api_success")) for row in rows),
        "accepted_changes": sum(bool_value(row.get("accepted_changed")) for row in rows),
        "improved": improved,
        "worsened": worsened,
        "unchanged": count - improved - worsened,
        "false_corrections": false_corrections,
        "baseline_wer": round(baseline_wer, 8),
        "corrected_wer": round(corrected_wer, 8),
        "wer_delta": round(corrected_wer - baseline_wer, 8),
        "baseline_cer": round(baseline_cer, 8),
        "corrected_cer": round(corrected_cer, 8),
        "cer_delta": round(corrected_cer - baseline_cer, 8),
        "prompt_tokens": round(sum(as_float(row, "prompt_tokens") for row in rows), 3),
        "completion_tokens": round(sum(as_float(row, "completion_tokens") for row in rows), 3),
        "total_tokens": round(sum(as_float(row, "total_tokens") for row in rows), 3),
    }


def write_summaries(args: argparse.Namespace) -> None:
    overall: List[Dict[str, Any]] = []
    by_language: List[Dict[str, Any]] = []
    for model in args.models:
        details_path = args.results_dir / model_slug(model) / "details.csv"
        if not details_path.exists():
            continue
        rows = read_csv_rows(details_path)
        overall.append(summarize_rows(model, "all", rows))
        languages = sorted({str(row.get("language", "") or "unknown") for row in rows})
        for language in languages:
            language_rows = [row for row in rows if str(row.get("language", "") or "unknown") == language]
            by_language.append(summarize_rows(model, language, language_rows))
    write_csv_rows(args.results_dir / "summary.csv", SUMMARY_FIELDNAMES, overall)
    write_csv_rows(args.results_dir / "summary_by_language.csv", SUMMARY_FIELDNAMES, by_language)
    print(f"Summary: {args.results_dir / 'summary.csv'}")
    for row in overall:
        print(
            f"{row['model']}: CER {row['baseline_cer']:.4f} -> {row['corrected_cer']:.4f} "
            f"(delta={row['cer_delta']:+.4f}), improved={row['improved']}, "
            f"worsened={row['worsened']}, false_corrections={row['false_corrections']}"
        )


def select_sample_ids(
    metadata: Mapping[str, Any],
    asr_map: Mapping[str, Any],
    limit: int,
) -> List[str]:
    sample_ids = sorted(set(metadata) & set(asr_map), key=sample_sort_key)
    return sample_ids[:limit] if limit > 0 else sample_ids


def preflight(
    args: argparse.Namespace,
    require_asr: bool,
    require_ocr: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any], List[str]]:
    missing = [path for path in [args.metadata] if not path.is_file()]
    if require_asr and not args.asr_results.is_file():
        missing.append(args.asr_results)
    if missing:
        raise FileNotFoundError("Missing required paths:\n" + "\n".join(map(str, missing)))
    metadata = load_metadata(args.metadata)
    asr_map = (
        load_text_map(args.asr_results, ["asr_result", "qwen_raw", "text"])
        if args.asr_results.is_file()
        else {}
    )
    sample_ids = (
        select_sample_ids(metadata, asr_map, args.limit)
        if require_asr
        else sorted(metadata, key=sample_sort_key)[:args.limit or None]
    )
    if not sample_ids:
        raise RuntimeError("No matching metadata/ASR identities")
    print(f"metadata={len(metadata)}, asr={len(asr_map)}, selected={len(sample_ids)}")
    print(f"models={args.models}, thinking={args.thinking}")
    print(f"video_dir={args.video_dir}")
    missing_audio = [sample_id for sample_id in sample_ids if find_audio_file(args.audio_dir, sample_id) is None]
    missing_video = [sample_id for sample_id in sample_ids if not (args.video_dir / f"{sample_id}.mp4").is_file()]
    audio_matches = len(sample_ids) - len(missing_audio)
    video_matches = len(sample_ids) - len(missing_video)
    print(f"matched audio={audio_matches}/{len(sample_ids)}")
    print(f"matched videos={video_matches}/{len(sample_ids)}")
    if missing_audio or missing_video:
        problems = []
        if missing_audio:
            problems.append(f"missing audio ({len(missing_audio)}): {missing_audio[:10]}")
        if missing_video:
            problems.append(f"missing video ({len(missing_video)}): {missing_video[:10]}")
        raise FileNotFoundError("Dataset identity check failed:\n" + "\n".join(problems))
    if require_ocr:
        if not args.ocr_results.is_file():
            raise FileNotFoundError(f"OCR results not found: {args.ocr_results}")
        ocr_map = load_text_map(args.ocr_results, ["ocr_result", "ocr_text", "text"])
        nonempty = sum(bool(ocr_map.get(sample_id, {}).get("text")) for sample_id in sample_ids)
        print(f"OCR rows={len(ocr_map)}, nonempty matched={nonempty}/{len(sample_ids)}")
    print("Preflight passed.")
    return metadata, asr_map, sample_ids


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OCR-assisted Qwen3-ASR correction ablation")
    parser.add_argument(
        "--stage",
        choices=["preflight", "api-check", "asr", "ocr", "evaluate", "all", "summarize"],
        default="preflight",
    )
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--asr-results", type=Path, default=DEFAULT_ASR_RESULTS)
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--ocr-results", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0, help="0 means all matched samples")
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--ocr-url", default=os.getenv("AUTODUB_OCR_URL", "http://127.0.0.1:8000/extract_text"))
    parser.add_argument("--ocr-fractions", type=parse_fractions, default=parse_fractions("0.25,0.5,0.75"))
    parser.add_argument("--ocr-timeout", type=float, default=300.0)
    parser.add_argument("--ffprobe", default="ffprobe")

    parser.add_argument(
        "--qwen-model-dir",
        type=Path,
        default=Path(os.getenv("QWEN3_ASR_MODEL_DIR", "/data0/goldenseed/models/Qwen3-ASR-1.7B")),
    )
    parser.add_argument("--qwen-device-map", default=os.getenv("QWEN3_ASR_DEVICE_MAP", "cuda:0"))
    parser.add_argument("--qwen-dtype", default=os.getenv("QWEN3_ASR_DTYPE", "bfloat16"))
    parser.add_argument("--qwen-attn-implementation", default=os.getenv("QWEN3_ATTN_IMPLEMENTATION", ""))
    parser.add_argument("--qwen-max-new-tokens", type=int, default=256)
    parser.add_argument("--asr-batch-size", type=int, default=4)

    parser.add_argument("--models", nargs="+", default=["deepseek-v4-flash", "deepseek-v4-pro"])
    parser.add_argument("--thinking", choices=["enabled", "disabled"], default="disabled")
    parser.add_argument("--deepseek-api-key", default=os.getenv("DEEPSEEK_API_KEY", ""))
    parser.add_argument("--deepseek-base-url", default=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--proxy-url", default=os.getenv("DEEPSEEK_PROXY_URL", ""))
    parser.add_argument("--trust-env", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--deepseek-timeout", type=float, default=180.0)
    parser.add_argument("--continue-on-api-error", action="store_true")
    parser.add_argument("--api-retries", type=int, default=3)
    parser.add_argument("--api-retry-backoff", type=float, default=2.0)
    parser.add_argument("--missing-id-retries", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-ocr-lines", type=int, default=8)
    parser.add_argument("--lexical-quota", type=int, default=4)
    parser.add_argument("--noise-patterns", type=Path, default=None)
    parser.add_argument("--confidence-threshold", type=float, default=0.85)
    parser.add_argument("--max-edit-ratio", type=float, default=0.45)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.results_dir = args.results_dir.resolve()
    if args.ocr_results is None:
        args.ocr_results = args.results_dir / "ocr_multiframe.csv"
    else:
        args.ocr_results = args.ocr_results.resolve()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    if args.stage == "summarize":
        write_summaries(args)
        return
    if args.stage == "api-check":
        check_deepseek_api(args)
        return

    require_asr = args.stage in {"evaluate"}
    require_ocr = args.stage in {"evaluate"}
    metadata, asr_map, sample_ids = preflight(
        args,
        require_asr=require_asr,
        require_ocr=require_ocr,
    )
    if args.stage == "preflight":
        return
    if args.stage in {"asr", "all"}:
        run_asr_stage(args, metadata, sample_ids)
    if args.stage in {"ocr", "all"}:
        run_ocr_stage(args, sample_ids)
    if args.stage in {"evaluate", "all"}:
        asr_map = load_text_map(args.asr_results, ["asr_result", "qwen_raw", "text"])
        ocr_map = load_text_map(args.ocr_results, ["ocr_result", "ocr_text", "text"])
        for model in args.models:
            print("=" * 80)
            print(f"MODEL: {model}")
            run_model_evaluation(args, model, sample_ids, metadata, asr_map, ocr_map)
        write_summaries(args)


if __name__ == "__main__":
    main()
