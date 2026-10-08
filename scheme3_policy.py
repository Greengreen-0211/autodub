#!/usr/bin/env python3
"""Pure routing policy for AutoDub scheme three.

This module intentionally has no ML/audio dependencies.  Step 3, the TTS
router, and local unit tests all import the same normalization and routing
rules so that generated JSON cannot drift from runtime behavior.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Union


CANONICAL_LANGUAGES = frozenset({"zh", "en", "other"})
CANONICAL_EMOTIONS = frozenset(
    {"neutral", "angry", "sad", "fearful", "disgusted", "happy", "surprised"}
)
STRONG_EMOTIONS = frozenset({"angry", "sad", "fearful", "disgusted"})
POSITIVE_EMOTIONS = frozenset({"happy", "surprised"})
SCHEME3_ENGINES = frozenset(
    {"f5tts", "indextts2", "cosyvoice3", "confucius4", "omnivoice"}
)
REQUIRED_SEGMENT_FIELDS = frozenset(
    {"text", "start", "end", "source_lang", "target_lang", "emotion", "speaker", "tts_engine"}
)

LANGUAGE_ALIASES = {
    "chinese": "zh",
    "cmn": "zh",
    "mandarin": "zh",
    "zh-cn": "zh",
    "zh-hans": "zh",
    "english": "en",
    "eng": "en",
    "en-us": "en",
    "en-gb": "en",
}

# All recognized aliases terminate in exactly one of the seven canonical
# emotions.  Unknown labels are retained separately by callers and normalize
# to neutral; scheme three never invents an eighth "other" emotion.
EMOTION_LABEL_MAP = {
    "anger": "angry",
    "angry": "angry",
    "disgust": "disgusted",
    "disgusted": "disgusted",
    "fear": "fearful",
    "fearful": "fearful",
    "happiness": "happy",
    "happy": "happy",
    "neutral": "neutral",
    "sadness": "sad",
    "sad": "sad",
    "surprise": "surprised",
    "surprised": "surprised",
    "other": "neutral",
    "unknown": "neutral",
}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    return int(value) if value else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name, "").strip()
    return float(value) if value else default


@dataclass(frozen=True)
class ComplexityConfig:
    """Project defaults; the normative MD does not prescribe these values."""

    zh_long_chars: int = 28
    en_long_words: int = 20
    complex_clause_count: int = 3

    @classmethod
    def from_env(cls) -> "ComplexityConfig":
        return cls(
            zh_long_chars=_env_int("AUTODUB_SCHEME3_ZH_LONG_CHARS", 28),
            en_long_words=_env_int("AUTODUB_SCHEME3_EN_LONG_WORDS", 20),
            complex_clause_count=_env_int("AUTODUB_SCHEME3_COMPLEX_CLAUSES", 3),
        )


def normalize_language(language: Any) -> str:
    value = str(language or "").strip().lower().replace("_", "-")
    value = LANGUAGE_ALIASES.get(value, value)
    primary = value.split("-", 1)[0]
    if primary == "zh":
        return "zh"
    if primary == "en":
        return "en"
    return "other"


def normalize_emotion(
    raw_emotion: Any,
    *,
    score: Any = None,
    duration: Any = None,
    min_score: Optional[float] = None,
    min_duration: Optional[float] = None,
) -> str:
    raw = str(raw_emotion or "unknown").strip().lower()
    normalized = "neutral"
    for candidate in reversed([part.strip() for part in raw.replace("｜", "/").split("/")]):
        if candidate in EMOTION_LABEL_MAP:
            normalized = EMOTION_LABEL_MAP[candidate]
            break

    score_threshold = (
        _env_float("AUTODUB_EMOTION_MIN_SCORE", 0.5) if min_score is None else min_score
    )
    duration_threshold = (
        _env_float("AUTODUB_EMOTION_MIN_DURATION", 0.8)
        if min_duration is None
        else min_duration
    )
    if score is not None and float(score) < score_threshold:
        return "neutral"
    if duration is not None and float(duration) < duration_threshold:
        return "neutral"
    return normalized if normalized in CANONICAL_EMOTIONS else "neutral"


def analyze_text_complexity(
    text: Any,
    target_lang: Any,
    config: Optional[ComplexityConfig] = None,
) -> dict[str, Any]:
    cfg = config or ComplexityConfig.from_env()
    value = str(text or "").strip()
    lang = normalize_language(target_lang)
    cjk_chars = len(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]", value))
    words = len(re.findall(r"[A-Za-z]+(?:['’-][A-Za-z]+)*|\d+(?:\.\d+)?", value))
    # Clause count is language-agnostic and intentionally conservative: at
    # least three clauses are required before punctuation alone marks a sentence
    # complex.  Terminal punctuation is not treated as a separator.
    clause_separators = len(re.findall(r"[,;:，；：、]", value))
    clause_count = clause_separators + 1 if value else 0
    if lang == "zh":
        is_long = cjk_chars >= cfg.zh_long_chars
    elif lang == "en":
        is_long = words >= cfg.en_long_words
    else:
        is_long = False
    is_complex = clause_count >= cfg.complex_clause_count
    return {
        "language": lang,
        "cjk_chars": cjk_chars,
        "words": words,
        "clause_count": clause_count,
        "is_long": is_long,
        "is_complex": is_complex,
        "is_long_or_complex": is_long or is_complex,
    }


def is_long_or_complex(
    text: Any,
    target_lang: Any,
    config: Optional[ComplexityConfig] = None,
) -> bool:
    return bool(analyze_text_complexity(text, target_lang, config)["is_long_or_complex"])


def select_scheme3_tts_engine(
    target_lang: Any,
    emotion: Any,
    text: Any,
    config: Optional[ComplexityConfig] = None,
) -> str:
    lang = normalize_language(target_lang)
    normalized_emotion = normalize_emotion(emotion)
    if lang not in {"zh", "en"}:
        return "omnivoice"
    if is_long_or_complex(text, lang, config):
        return "confucius4"
    if normalized_emotion in STRONG_EMOTIONS:
        return "indextts2"
    if normalized_emotion in POSITIVE_EMOTIONS:
        return "cosyvoice3"
    return "f5tts"


def build_scheme3_segments(
    segments: Iterable[Mapping[str, Any]],
    *,
    source_lang: Any,
    target_lang: Any,
    config: Optional[ComplexityConfig] = None,
) -> list[dict[str, Any]]:
    canonical_target = normalize_language(target_lang)
    target_detail = str(target_lang or "").strip().lower().replace("_", "-") or "other"
    output: list[dict[str, Any]] = []
    for index, source in enumerate(segments):
        segment = dict(source)
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", start))
        if end <= start:
            raise ValueError(f"segment {segment.get('id', index)!r} has invalid range: {start}..{end}")
        duration = max(0.0, end - start)
        raw = segment.get("raw_emotion", segment.get("emotion_raw", segment.get("emotion")))
        score = segment.get("emotion_score")
        if "emotion_reliable" in segment:
            emotion = normalize_emotion(segment.get("emotion")) if segment["emotion_reliable"] else "neutral"
        else:
            emotion = normalize_emotion(raw, score=score, duration=duration)
        text = str(segment.get("text", "")).strip()
        per_segment_source = segment.get("source_lang", source_lang)
        complexity = analyze_text_complexity(text, canonical_target, config)
        engine = select_scheme3_tts_engine(canonical_target, emotion, text, config)
        segment.update(
            {
                "id": segment.get("id", index),
                "text": text,
                "start": start,
                "end": end,
                "source_lang": normalize_language(per_segment_source),
                "target_lang": canonical_target,
                "emotion": emotion,
                "normalized_emotion": emotion,
                "speaker": str(segment.get("speaker", "SPEAKER_00")),
                "tts_engine": engine,
                "tts_complexity": complexity,
            }
        )
        if canonical_target == "other":
            segment["target_lang_raw"] = target_detail
        output.append(segment)
    return output


def validate_scheme3_segments(segments: Iterable[Mapping[str, Any]]) -> None:
    for index, row in enumerate(segments):
        missing = REQUIRED_SEGMENT_FIELDS.difference(row)
        if missing:
            raise ValueError(f"segment {index} is missing fields: {sorted(missing)}")
        if float(row["end"]) <= float(row["start"]):
            raise ValueError(f"segment {index} has end <= start")
        if row["source_lang"] not in CANONICAL_LANGUAGES:
            raise ValueError(f"segment {index} has invalid source_lang: {row['source_lang']!r}")
        if row["target_lang"] not in CANONICAL_LANGUAGES:
            raise ValueError(f"segment {index} has invalid target_lang: {row['target_lang']!r}")
        if row["emotion"] not in CANONICAL_EMOTIONS:
            raise ValueError(f"segment {index} has invalid emotion: {row['emotion']!r}")
        if row["tts_engine"] not in SCHEME3_ENGINES:
            raise ValueError(f"segment {index} has invalid tts_engine: {row['tts_engine']!r}")


def write_segments_json(path: Union[str, Path], segments: Iterable[Mapping[str, Any]]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = [dict(item) for item in segments]
    validate_scheme3_segments(rows)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, destination)
    return destination
