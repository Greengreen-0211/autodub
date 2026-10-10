"""Shared timestamp normalization and RTTM-to-text alignment.

Silence-detected ASR chunks are transport units, not final sentences.  This
module keeps Qwen's word/character alignment as boundary evidence and lets
RTTM speaker turns snap to safe text gaps before producing final segments.
It has no model, GPU, or audio-library dependencies.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


STRONG_PUNCTUATION = re.compile(r'[.!?。！？]["\'\u2019\u201d)\]]*$')
WEAK_PUNCTUATION = re.compile(r'[,;:，；：、]["\'\u2019\u201d)\]]*$')
LEXICAL = re.compile(r"[\w\u3400-\u9fff]", re.UNICODE)


@dataclass(frozen=True)
class AlignmentConfig:
    weak_gap_seconds: float = 0.35
    strong_gap_seconds: float = 0.60
    minimum_sentence_seconds: float = 0.80
    target_sentence_seconds: float = 8.0
    maximum_sentence_seconds: float = 12.0
    speaker_snap_radius_seconds: float = 0.80
    minimum_speaker_turn_seconds: float = 0.75
    island_max_seconds: float = 0.50
    island_max_gap_seconds: float = 0.15


def normalize_speaker(value: Any) -> str:
    text = str(value or "SPEAKER_00")
    match = re.fullmatch(r"(?:spk|speaker)[_-]?(\d+)", text, flags=re.IGNORECASE)
    if match:
        return f"SPEAKER_{int(match.group(1)):02d}"
    if text.isdigit():
        return f"SPEAKER_{int(text):02d}"
    return text


def _number(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _lexical_count(text: str) -> int:
    return len(LEXICAL.findall(text))


def _find_text_span(transcript: str, token: str, cursor: int) -> Optional[Tuple[int, int]]:
    """Find a token after cursor without guessing proportional character cuts."""
    if not token:
        return None
    position = transcript.find(token, cursor)
    if position >= 0:
        return position, position + len(token)
    stripped = token.strip()
    if stripped and stripped != token:
        position = transcript.find(stripped, cursor)
        if position >= 0:
            return position, position + len(stripped)
    # Some aligners normalize runs of spaces.  Permit only whitespace-flexible
    # matching; lexical substitutions are not guessed.
    parts = [re.escape(part) for part in re.split(r"\s+", stripped) if part]
    if parts:
        match = re.search(r"\s*".join(parts), transcript[cursor:])
        if match:
            return cursor + match.start(), cursor + match.end()
    return None


def normalize_timestamp_items(
    raw_items: Iterable[Mapping[str, Any]],
    *,
    transcript: str,
    chunk_start: float,
    chunk_end: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Validate absolute Qwen timestamps and attach exact transcript spans."""
    normalized: List[Dict[str, Any]] = []
    warnings: List[str] = []
    cursor = 0
    mapped_lexical = 0
    invalid_count = 0
    previous_start = chunk_start

    for raw in raw_items:
        text = str(raw.get("text", ""))
        start = _number(raw.get("start"))
        end = _number(raw.get("end"))
        if not text.strip() or start is None or end is None:
            invalid_count += 1
            continue
        start = min(chunk_end, max(chunk_start, start))
        end = min(chunk_end, max(chunk_start, end))
        if start < previous_start:
            warnings.append("non_monotonic_timestamp")
            start = previous_start
        if end < start:
            warnings.append("negative_timestamp_duration")
            end = start
        span = _find_text_span(transcript, text, cursor)
        item: Dict[str, Any] = {
            "text": text,
            "start": round(start, 3),
            "end": round(end, 3),
        }
        if span is not None:
            item["char_start"], item["char_end"] = span
            cursor = span[1]
            mapped_lexical += _lexical_count(transcript[span[0]:span[1]])
        else:
            warnings.append("timestamp_text_unmapped")
        normalized.append(item)
        previous_start = start

    transcript_lexical = _lexical_count(transcript)
    coverage = (
        min(1.0, mapped_lexical / transcript_lexical)
        if transcript_lexical
        else (1.0 if normalized else 0.0)
    )
    total_raw = len(normalized) + invalid_count
    valid_ratio = len(normalized) / total_raw if total_raw else 0.0
    if normalized and coverage >= 0.80 and valid_ratio >= 0.90:
        status = "valid"
    elif normalized and coverage >= 0.50:
        status = "degraded"
    else:
        status = "unusable"
    if invalid_count:
        warnings.append("invalid_timestamp_items")
    metadata = {
        "status": status,
        "coverage_ratio": round(coverage, 4),
        "valid_item_ratio": round(valid_ratio, 4),
        "warning_codes": sorted(set(warnings)),
    }
    return normalized, metadata


def _gap_seconds(items: Sequence[Mapping[str, Any]], index: int) -> float:
    return max(
        0.0,
        float(items[index].get("start", 0.0))
        - float(items[index - 1].get("end", 0.0)),
    )


def sentence_boundary_indices(
    items: Sequence[Mapping[str, Any]],
    *,
    config: AlignmentConfig = AlignmentConfig(),
) -> List[int]:
    """Return token-gap indices that form usable sentence candidates."""
    if len(items) < 2:
        return []
    boundaries: List[int] = []
    group_start = float(items[0].get("start", 0.0))
    last_boundary = 0
    for index in range(1, len(items)):
        left = str(items[index - 1].get("text", "")).strip()
        gap = _gap_seconds(items, index)
        elapsed = float(items[index - 1].get("end", group_start)) - group_start
        strong = bool(STRONG_PUNCTUATION.search(left))
        weak = bool(WEAK_PUNCTUATION.search(left))
        should_split = False
        if strong and elapsed >= config.minimum_sentence_seconds:
            should_split = True
        elif gap >= config.strong_gap_seconds and elapsed >= config.minimum_sentence_seconds:
            should_split = True
        elif (
            weak
            and gap >= config.weak_gap_seconds
            and elapsed >= config.minimum_sentence_seconds
        ):
            should_split = True
        elif weak and elapsed >= config.target_sentence_seconds:
            should_split = True
        elif elapsed >= config.maximum_sentence_seconds:
            should_split = True
        if should_split and index > last_boundary:
            boundaries.append(index)
            last_boundary = index
            group_start = float(items[index].get("start", group_start))
    return boundaries


def sentence_candidates(
    items: Sequence[Mapping[str, Any]],
    *,
    config: AlignmentConfig = AlignmentConfig(),
) -> List[Dict[str, Any]]:
    boundaries = [0, *sentence_boundary_indices(items, config=config), len(items)]
    result = []
    for start_index, end_index in zip(boundaries, boundaries[1:]):
        group = items[start_index:end_index]
        if not group:
            continue
        candidate = {
            "start": float(group[0].get("start", 0.0)),
            "end": float(group[-1].get("end", 0.0)),
            "token_start": start_index,
            "token_end": end_index,
        }
        if "char_start" in group[0] and "char_end" in group[-1]:
            candidate["char_start"] = int(group[0]["char_start"])
            candidate["char_end"] = int(group[-1]["char_end"])
        result.append(candidate)
    return result


def speaker_overlaps(
    start: float, end: float, spans: Sequence[Mapping[str, Any]]
) -> Dict[str, float]:
    overlaps: Dict[str, float] = {}
    for span in spans:
        overlap = max(
            0.0,
            min(end, float(span["end"])) - max(start, float(span["start"])),
        )
        if overlap > 0:
            speaker = normalize_speaker(span.get("speaker"))
            overlaps[speaker] = overlaps.get(speaker, 0.0) + overlap
    return overlaps


def smooth_rttm_spans(
    spans: Iterable[Mapping[str, Any]],
    *,
    config: AlignmentConfig = AlignmentConfig(),
) -> List[Dict[str, Any]]:
    """Merge adjacent labels and remove only local, short A-B-A islands."""
    ordered = sorted(
        (
            {
                "start": float(item["start"]),
                "end": float(item["end"]),
                "speaker": normalize_speaker(item.get("speaker")),
            }
            for item in spans
            if float(item["end"]) > float(item["start"])
        ),
        key=lambda item: item["start"],
    )
    merged: List[Dict[str, Any]] = []
    for span in ordered:
        if (
            merged
            and merged[-1]["speaker"] == span["speaker"]
            and span["start"] - merged[-1]["end"] <= config.island_max_gap_seconds
        ):
            merged[-1]["end"] = max(merged[-1]["end"], span["end"])
        else:
            merged.append(dict(span))
    index = 1
    while index < len(merged) - 1:
        left, middle, right = merged[index - 1:index + 2]
        duration = middle["end"] - middle["start"]
        if (
            left["speaker"] == right["speaker"] != middle["speaker"]
            and duration <= config.island_max_seconds
            and middle["start"] - left["end"] <= config.island_max_gap_seconds
            and right["start"] - middle["end"] <= config.island_max_gap_seconds
        ):
            merged[index - 1] = {
                "start": left["start"],
                "end": right["end"],
                "speaker": left["speaker"],
            }
            del merged[index:index + 2]
            index = max(1, index - 1)
        else:
            index += 1
    return merged


def _speaker_change_points(
    spans: Sequence[Mapping[str, Any]], config: AlignmentConfig
) -> List[Dict[str, Any]]:
    changes = []
    for left, right in zip(spans, spans[1:]):
        if normalize_speaker(left.get("speaker")) == normalize_speaker(right.get("speaker")):
            continue
        left_duration = float(left["end"]) - float(left["start"])
        right_duration = float(right["end"]) - float(right["start"])
        if min(left_duration, right_duration) < config.minimum_speaker_turn_seconds:
            continue
        changes.append(
            {
                "time": (float(left["end"]) + float(right["start"])) / 2.0,
                "left_speaker": normalize_speaker(left.get("speaker")),
                "right_speaker": normalize_speaker(right.get("speaker")),
            }
        )
    return changes


def _boundary_score(
    items: Sequence[Mapping[str, Any]], index: int, change_time: float, radius: float
) -> float:
    left = items[index - 1]
    right = items[index]
    gap = _gap_seconds(items, index)
    gap_time = (
        float(left.get("end", change_time)) + float(right.get("start", change_time))
    ) / 2.0
    distance_score = max(0.0, 1.0 - abs(gap_time - change_time) / max(radius, 0.001))
    punctuation_score = 1.5 if STRONG_PUNCTUATION.search(str(left.get("text", "")).strip()) else 0.0
    if not punctuation_score and WEAK_PUNCTUATION.search(str(left.get("text", "")).strip()):
        punctuation_score = 0.6
    silence_score = min(1.0, gap / 0.6)
    return distance_score + punctuation_score + silence_score


def _snap_change_to_gap(
    items: Sequence[Mapping[str, Any]],
    change_time: float,
    *,
    config: AlignmentConfig,
) -> Optional[int]:
    candidates = []
    for index in range(1, len(items)):
        left_end = float(items[index - 1].get("end", change_time))
        right_start = float(items[index].get("start", change_time))
        gap_time = (left_end + right_start) / 2.0
        if abs(gap_time - change_time) <= config.speaker_snap_radius_seconds:
            candidates.append(
                (
                    _boundary_score(
                        items, index, change_time, config.speaker_snap_radius_seconds
                    ),
                    -abs(gap_time - change_time),
                    index,
                )
            )
    return max(candidates)[2] if candidates else None


def _text_for_group(
    transcript: str,
    items: Sequence[Mapping[str, Any]],
    start_index: int,
    end_index: int,
) -> Optional[str]:
    group = items[start_index:end_index]
    if not group:
        return None
    if "char_start" not in group[0] or "char_end" not in group[-1]:
        return None
    return transcript[int(group[0]["char_start"]):int(group[-1]["char_end"])].strip()


def align_segments_with_rttm(
    segments: Iterable[Mapping[str, Any]],
    rttm_spans: Iterable[Mapping[str, Any]],
    *,
    config: AlignmentConfig = AlignmentConfig(),
) -> List[Dict[str, Any]]:
    """Produce final sentence-level, speaker-consistent segments."""
    spans = smooth_rttm_spans(rttm_spans, config=config)
    changes = _speaker_change_points(spans, config)
    output: List[Dict[str, Any]] = []
    for source in segments:
        segment = dict(source)
        segment_start = float(segment["start"])
        segment_end = float(segment["end"])
        transcript = str(segment.get("text", ""))
        saved_items = list(segment.get("qwen3_time_stamps") or [])
        items, alignment = normalize_timestamp_items(
            saved_items,
            transcript=transcript,
            chunk_start=segment_start,
            chunk_end=segment_end,
        )
        segment["qwen3_time_stamps"] = items
        segment["timestamp_alignment_status"] = alignment["status"]
        segment["timestamp_coverage_ratio"] = alignment["coverage_ratio"]
        segment["timestamp_warning_codes"] = alignment["warning_codes"]
        segment["qwen3_sentence_candidates"] = sentence_candidates(items, config=config)
        status = alignment["status"]
        overlaps = speaker_overlaps(segment_start, segment_end, spans)
        fallback = max(overlaps, key=overlaps.get) if overlaps else "SPEAKER_00"
        segment_changes = [
            item for item in changes if segment_start < float(item["time"]) < segment_end
        ]
        if not items or status == "unusable":
            segment["speaker"] = normalize_speaker(fallback)
            if len(overlaps) > 1:
                segment["speaker_boundary_ambiguous"] = True
            output.append(segment)
            continue

        sentence_splits = set(sentence_boundary_indices(items, config=config))
        speaker_splits: Dict[int, Dict[str, Any]] = {}
        for change in segment_changes:
            boundary = _snap_change_to_gap(items, float(change["time"]), config=config)
            if boundary is not None:
                speaker_splits[boundary] = change
        split_indices = sorted(sentence_splits | set(speaker_splits))

        # Exact transcript spans are mandatory for splitting.  A partial or
        # mismatched alignment is retained for audit, never used to guess text.
        boundaries = [0, *split_indices, len(items)]
        text_parts = [
            _text_for_group(transcript, items, left, right)
            for left, right in zip(boundaries, boundaries[1:])
        ]
        if any(part is None or not part for part in text_parts):
            segment["speaker"] = normalize_speaker(fallback)
            segment["speaker_boundary_ambiguous"] = bool(segment_changes)
            output.append(segment)
            continue

        time_boundaries = [segment_start]
        for index in split_indices:
            left_end = float(items[index - 1].get("end", segment_start))
            right_start = float(items[index].get("start", segment_end))
            time_boundaries.append((left_end + right_start) / 2.0)
        time_boundaries.append(segment_end)

        for group_index, ((left, right), text_part) in enumerate(
            zip(zip(boundaries, boundaries[1:]), text_parts)
        ):
            start = time_boundaries[group_index]
            end = time_boundaries[group_index + 1]
            if end <= start:
                continue
            group_overlaps = speaker_overlaps(start, end, spans)
            speaker = max(group_overlaps, key=group_overlaps.get) if group_overlaps else fallback
            result = dict(segment)
            result["parent_segment_id"] = segment.get("id")
            result["start"] = round(start, 3)
            result["end"] = round(end, 3)
            result["text"] = text_part
            result["speaker"] = normalize_speaker(speaker)
            result["speaker_overlap_seconds"] = {
                key: round(value, 3) for key, value in group_overlaps.items()
            }
            result["qwen3_time_stamps"] = items[left:right]
            result["segment_split_reasons"] = sorted(
                {
                    *( ["sentence_boundary"] if right in sentence_splits else [] ),
                    *( ["speaker_boundary"] if right in speaker_splits else [] ),
                }
            )
            output.append(result)

    for index, segment in enumerate(output):
        segment["id"] = index
    return output
