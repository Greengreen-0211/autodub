"""Evidence-driven helpers for recovering ASR text without trusting bad timing.

The functions in this module deliberately keep *what was said* separate from
*when every word was said*.  This lets AutoDub retain text supported by two
independent acoustic views while preventing broken forced-alignment output from
creating speaker or sentence boundaries.
"""

from __future__ import annotations

import difflib
import html
import math
import re
import unicodedata
from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def normalize_text(text: str) -> str:
    return "".join(
        char
        for char in unicodedata.normalize("NFKC", str(text or "")).casefold()
        if char.isalnum()
    )


def text_similarity(left: str, right: str) -> float:
    left_key, right_key = normalize_text(left), normalize_text(right)
    if not left_key or not right_key:
        return 0.0
    return difflib.SequenceMatcher(None, left_key, right_key).ratio()


def _alignment_rank(segment: Mapping[str, Any]) -> Tuple[int, float, int, int]:
    status = str(segment.get("timestamp_alignment_status") or "")
    return (
        {"valid": 2, "degraded": 1}.get(status, 0),
        float(segment.get("timestamp_coverage_ratio") or 0.0),
        -len(segment.get("timestamp_warning_codes") or []),
        len(normalize_text(str(segment.get("text") or ""))),
    )


def coarse_segment(
    candidate: Mapping[str, Any],
    *,
    start: float,
    end: float,
    origin: str,
    text_confidence: float,
    reason: str,
) -> Dict[str, Any]:
    """Return a safe segment whose outer speech bounds may be used, but not words."""
    result = dict(candidate)
    result.update(
        {
            "start": round(float(start), 3),
            "end": round(float(end), 3),
            "speech_start": round(float(start), 3),
            "speech_end": round(float(end), 3),
            "asr_origin": origin,
            "asr_recovery_status": "accepted_coarse",
            "timestamp_precision": "coarse",
            "speaker_boundary_eligible": False,
            "asr_text_confidence": round(float(text_confidence), 4),
            "asr_alignment_confidence": 0.0,
            "timestamp_alignment_status": "coarse",
            "timestamp_coverage_ratio": 0.0,
            "timestamp_warning_codes": sorted(
                set(result.get("timestamp_warning_codes") or [])
                | {"coarse_recovery_bounds"}
            ),
            "qwen3_time_stamps": [],
            "qwen3_sentence_candidates": [],
            "asr_recovery_reason": reason,
        }
    )
    return result


def choose_consensus_candidate(
    candidates: Sequence[Mapping[str, Any]],
    *,
    threshold: float,
    start: float,
    end: float,
    precise_origin: str,
    coarse_origin: str,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """Choose a candidate from independent sources after text-only consensus.

    Importantly, candidates with unusable timestamps remain eligible to vote on
    text.  If no candidate has usable timing but two texts agree, a coarse
    segment is returned instead of discarding the transcript.
    """
    usable = [item for item in candidates if normalize_text(str(item.get("text") or ""))]
    best_pair: Optional[Tuple[Mapping[str, Any], Mapping[str, Any]]] = None
    best_similarity = 0.0
    for index, left in enumerate(usable):
        for right in usable[index + 1 :]:
            if left.get("asr_repair_source") == right.get("asr_repair_source"):
                continue
            similarity = text_similarity(str(left.get("text") or ""), str(right.get("text") or ""))
            if similarity > best_similarity:
                best_similarity = similarity
                best_pair = (left, right)
    decision = {
        "candidate_count": len(usable),
        "consensus_similarity": round(best_similarity, 4),
        "threshold": round(float(threshold), 4),
    }
    effective_threshold = float(threshold)
    if best_pair is not None:
        pair_lengths = [len(normalize_text(str(item.get("text") or ""))) for item in best_pair]
        # A single content-word disagreement can still produce a deceptively
        # high character score in a short utterance.
        # Short phrases therefore need near-exact independent agreement; OCR
        # may resolve the remaining ambiguity later.
        if max(pair_lengths, default=0) <= 24:
            effective_threshold = max(effective_threshold, 0.94)
    decision["effective_threshold"] = round(effective_threshold, 4)
    if best_pair is None or best_similarity < effective_threshold:
        decision["reason"] = "no_independent_text_consensus"
        return None, decision

    selected = dict(max(best_pair, key=_alignment_rank))
    selected["asr_text_confidence"] = round(best_similarity, 4)
    selected["asr_recovery_sources"] = [
        str(item.get("asr_repair_source") or "unknown") for item in best_pair
    ]
    if _alignment_rank(selected)[0] > 0:
        selected.update(
            {
                "asr_origin": precise_origin,
                "asr_recovery_status": "accepted_precise",
                "timestamp_precision": "word",
                "speaker_boundary_eligible": True,
                "asr_alignment_confidence": (
                    0.95
                    if selected.get("timestamp_alignment_status") == "valid"
                    else 0.60
                ),
            }
        )
        decision["reason"] = "text_consensus_with_usable_alignment"
        decision["precision"] = "word"
        return selected, decision

    decision["reason"] = "text_consensus_without_usable_alignment"
    decision["precision"] = "coarse"
    return coarse_segment(
        selected,
        start=start,
        end=end,
        origin=coarse_origin,
        text_confidence=best_similarity,
        reason="dual_source_text_consensus_without_safe_word_timestamps",
    ), decision


def speech_micro_ranges(
    speech_ranges: Iterable[Tuple[float, float]],
    *,
    outer_start: float,
    outer_end: float,
    min_seconds: float = 0.65,
    max_seconds: float = 4.2,
) -> List[Tuple[float, float]]:
    """Clip independently detected speech to an empty ASR block and bound its size."""
    result: List[Tuple[float, float]] = []
    for raw_start, raw_end in speech_ranges:
        start = max(float(outer_start), float(raw_start))
        end = min(float(outer_end), float(raw_end))
        if end - start < min_seconds:
            continue
        parts = max(1, int(math.ceil((end - start) / max_seconds)))
        width = (end - start) / parts
        for index in range(parts):
            part_start = start + width * index
            part_end = end if index == parts - 1 else start + width * (index + 1)
            if part_end - part_start >= min_seconds:
                result.append((round(part_start, 3), round(part_end, 3)))
    return result


_COORDINATE_PREFIX = re.compile(
    r"^\s*(?:\[?\s*\d{1,4}\s*[,;:]\s*\d{1,4}[^\]]*\]?\s*)+"
)
_SPOTTING_RECORD = re.compile(
    r"<ref>(.*?)</ref>\s*(?:<quad>.*?</quad>)?", re.I | re.S
)
_PLAIN_COORD_RECORD = re.compile(
    r"(.*?)(\(\s*\d+\s*,\s*\d+\s*\)\s*,\s*"
    r"\(\s*\d+\s*,\s*\d+\s*\))"
)
_URL_OR_UI = re.compile(r"https?://|www\.|bilibili|coding|字幕组|@[A-Za-z0-9_]", re.I)
_CJK = re.compile(r"[\u3400-\u9fff]")
_LATIN = re.compile(r"[A-Za-z]")


def _ocr_lines(frame_text: str) -> List[str]:
    cleaned = str(frame_text or "").replace("```", "\n")
    references = [
        re.sub(r"\s+", " ", html.unescape(match.group(1))).strip(" \t|-_")
        for match in _SPOTTING_RECORD.finditer(cleaned)
    ]
    if references:
        return [line for line in references if line and not _URL_OR_UI.search(line)]
    coordinate_lines = []
    for physical_line in re.split(r"[\r\n]+", cleaned):
        for match in _PLAIN_COORD_RECORD.finditer(physical_line):
            line = re.sub(r"\s+", " ", match.group(1)).strip(" \t|-_")
            if line and not _URL_OR_UI.search(line):
                coordinate_lines.append(line)
    if coordinate_lines:
        return coordinate_lines
    lines: List[str] = []
    for raw in re.split(r"[\r\n]+", cleaned):
        line = _COORDINATE_PREFIX.sub("", raw).strip(" \t|-_")
        if line and not _URL_OR_UI.search(line):
            lines.append(line)
    return lines


def _language_matches(line: str, language: str) -> bool:
    cjk, latin = len(_CJK.findall(line)), len(_LATIN.findall(line))
    language = str(language or "").lower()
    language = {
        "english": "en",
        "chinese": "zh",
        "mandarin": "zh",
        "japanese": "ja",
        "korean": "ko",
    }.get(language, language)
    if language.startswith(("zh", "ja", "ko")):
        return cjk >= 2 and cjk >= latin
    if language.startswith(("en", "fr", "de", "es", "it", "pt")):
        return latin >= 3 and latin >= cjk * 2
    return cjk + latin >= 3


def stable_ocr_source_line(
    frame_texts: Sequence[str], source_language: str
) -> Tuple[Optional[str], Dict[str, Any]]:
    """Select a source-language subtitle repeated in at least two frames."""
    by_key: Dict[str, List[Tuple[int, str]]] = {}
    for frame_index, frame in enumerate(frame_texts):
        seen = set()
        for line in _ocr_lines(frame):
            key = normalize_text(line)
            if (
                key in seen
                or len(key) < 2
                or len(key) > 140
                or not _language_matches(line, source_language)
            ):
                continue
            seen.add(key)
            by_key.setdefault(key, []).append((frame_index, line))
    ranked = []
    for key, observations in by_key.items():
        frame_count = len({item[0] for item in observations})
        if frame_count < 2:
            continue
        line_counts = Counter(item[1] for item in observations)
        line = line_counts.most_common(1)[0][0]
        word_count = len(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", line))
        subtitle_shape = int(bool(re.search(r"[.!?。！？]$", line))) + int(word_count >= 2)
        ranked.append((frame_count, subtitle_shape, len(key), line))
    if not ranked:
        return None, {"reason": "no_stable_source_language_subtitle", "frame_count": len(frame_texts)}
    frame_count, subtitle_shape, _, line = max(ranked)
    if subtitle_shape <= 0:
        return None, {"reason": "stable_text_not_subtitle_like", "frame_count": frame_count}
    confidence = min(0.92, 0.70 + 0.08 * (frame_count - 2) + 0.04 * subtitle_shape)
    return line, {
        "reason": "stable_source_language_subtitle",
        "frame_count": frame_count,
        "confidence": round(confidence, 4),
    }
