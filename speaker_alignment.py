"""Shared timestamp normalization and RTTM-to-text alignment.

Silence-detected ASR chunks are transport units, not final sentences.  This
module keeps Qwen's word/character alignment as boundary evidence and lets
RTTM speaker turns snap to safe text gaps before producing final segments.
It has no model, GPU, or audio-library dependencies.
"""

from __future__ import annotations

import difflib
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
    minimum_speaker_turn_seconds: float = 0.45
    island_max_seconds: float = 0.30
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
    chunk_duration = max(0.0, chunk_end - chunk_start)
    timestamp_durations = [
        max(0.0, float(item["end"]) - float(item["start"]))
        for item in normalized
    ]
    zero_duration_count = sum(duration <= 0.02 for duration in timestamp_durations)
    zero_duration_ratio = (
        zero_duration_count / len(normalized) if normalized else 0.0
    )
    longest_item_duration = max(timestamp_durations, default=0.0)
    endpoint_tolerance = min(0.08, max(0.02, chunk_duration * 0.01))
    endpoint_pileup_count = sum(
        abs(float(item["start"]) - chunk_end) <= endpoint_tolerance
        and abs(float(item["end"]) - chunk_end) <= endpoint_tolerance
        for item in normalized
    )
    endpoint_pileup_ratio = (
        endpoint_pileup_count / len(normalized) if normalized else 0.0
    )

    # Forced aligners sometimes return superficially well-mapped text while
    # collapsing most words onto the chunk endpoint (or stretching one word
    # across several seconds).  Such timestamps must never drive sentence or
    # speaker boundaries.
    severe_temporal_failure = bool(
        len(normalized) >= 4
        and (
            (zero_duration_count >= 3 and zero_duration_ratio >= 0.45)
            or (endpoint_pileup_count >= 3 and endpoint_pileup_ratio >= 0.40)
            or longest_item_duration >= max(2.5, chunk_duration * 0.55)
        )
    )
    temporal_degradation = bool(
        len(normalized) >= 3
        and (
            zero_duration_ratio >= 0.25
            or endpoint_pileup_ratio >= 0.25
            or longest_item_duration >= max(2.0, chunk_duration * 0.40)
        )
    )
    if zero_duration_ratio >= 0.25:
        warnings.append("excessive_zero_duration_timestamps")
    if endpoint_pileup_ratio >= 0.25:
        warnings.append("timestamp_endpoint_pileup")
    if longest_item_duration >= max(2.0, chunk_duration * 0.40):
        warnings.append("implausibly_long_timestamp_item")
    for index, item in enumerate(normalized):
        char_end = item.get("char_end")
        if char_end is None:
            item["punctuation_after"] = "none"
            continue
        if index + 1 < len(normalized):
            next_start = normalized[index + 1].get("char_start")
            separator_end = int(next_start) if next_start is not None else int(char_end)
        else:
            separator_end = len(transcript)
        separator = transcript[int(char_end):separator_end].strip()
        if STRONG_PUNCTUATION.search(separator):
            item["punctuation_after"] = "strong"
        elif WEAK_PUNCTUATION.search(separator):
            item["punctuation_after"] = "weak"
        else:
            item["punctuation_after"] = "none"

    coverage = (
        min(1.0, mapped_lexical / transcript_lexical)
        if transcript_lexical
        else (1.0 if normalized else 0.0)
    )
    total_raw = len(normalized) + invalid_count
    valid_ratio = len(normalized) / total_raw if total_raw else 0.0
    if severe_temporal_failure:
        status = "unusable"
    elif normalized and coverage >= 0.80 and valid_ratio >= 0.90 and not temporal_degradation:
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
        "zero_duration_ratio": round(zero_duration_ratio, 4),
        "endpoint_pileup_ratio": round(endpoint_pileup_ratio, 4),
        "longest_item_duration": round(longest_item_duration, 4),
        "severe_temporal_failure": severe_temporal_failure,
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
        punctuation = str(items[index - 1].get("punctuation_after", "none"))
        strong = punctuation == "strong" or bool(STRONG_PUNCTUATION.search(left))
        weak = punctuation == "weak" or bool(WEAK_PUNCTUATION.search(left))
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


def long_turn_boundary_indices(
    items: Sequence[Mapping[str, Any]],
    sentence_boundaries: Sequence[int],
    *,
    config: AlignmentConfig = AlignmentConfig(),
) -> List[int]:
    """Keep only safe sentence boundaries needed to cap a long speaker turn."""
    if not items or not sentence_boundaries:
        return []
    result: List[int] = []
    group_start = float(items[0].get("start", 0.0))
    candidates: List[int] = []
    for index in sentence_boundaries:
        boundary_end = float(items[index - 1].get("end", group_start))
        if boundary_end - group_start < config.maximum_sentence_seconds:
            candidates.append(index)
            continue
        # Prefer the latest safe textual boundary before the turn becomes too
        # long.  Ordinary punctuation does not split a coherent speaker turn.
        chosen = candidates[-1] if candidates else index
        result.append(chosen)
        group_start = float(items[chosen].get("start", boundary_end))
        candidates = [] if chosen == index else [index]
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
    """Merge adjacent equal labels without deleting a distinct short voice."""
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
    # A-B-A removal used to rely only on the duration of B.  That can erase a
    # real short reply or interjection.  Acoustic identity reconciliation now
    # handles jitter upstream; this text-alignment layer therefore performs no
    # destructive label-only island smoothing.
    return merged


def _speaker_change_points(
    spans: Sequence[Mapping[str, Any]],
    config: AlignmentConfig,
    boundary_evidence: Optional[Sequence[Mapping[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    changes = []
    evidence_items = list(boundary_evidence or [])
    for left, right in zip(spans, spans[1:]):
        if normalize_speaker(left.get("speaker")) == normalize_speaker(right.get("speaker")):
            continue
        left_duration = float(left["end"]) - float(left["start"])
        right_duration = float(right["end"]) - float(right["start"])
        if min(left_duration, right_duration) < config.minimum_speaker_turn_seconds:
            continue
        change_time = (float(left["end"]) + float(right["start"])) / 2.0
        matched = None
        if evidence_items:
            nearby = [
                item for item in evidence_items
                if abs(float(item.get("time", -9999.0)) - change_time)
                <= max(1.25, config.speaker_snap_radius_seconds)
            ]
            if nearby:
                matched = min(
                    nearby,
                    key=lambda item: abs(float(item.get("time")) - change_time),
                )
        changes.append(
            {
                "time": change_time,
                "left_speaker": normalize_speaker(left.get("speaker")),
                "right_speaker": normalize_speaker(right.get("speaker")),
                "text_split_allowed": bool(
                    matched.get("text_split_allowed", True)
                    if matched is not None else not evidence_items
                ),
                "boundary_role": (
                    matched.get("boundary_role", "speaker_and_text")
                    if matched is not None else "unmatched_rttm_change"
                ),
                "boundary_source": (
                    matched.get("candidate_source", "word_timestamp")
                    if matched is not None else "unmatched_rttm_change"
                ),
                "boundary_score": (
                    _number(matched.get("score")) if matched is not None else None
                ),
                "boundary_evidence_time": (
                    _number(matched.get("time")) if matched is not None else None
                ),
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
    punctuation = str(left.get("punctuation_after", "none"))
    punctuation_score = 1.5 if (
        punctuation == "strong"
        or STRONG_PUNCTUATION.search(str(left.get("text", "")).strip())
    ) else 0.0
    if not punctuation_score and (
        punctuation == "weak"
        or WEAK_PUNCTUATION.search(str(left.get("text", "")).strip())
    ):
        punctuation_score = 0.6
    silence_score = min(1.0, gap / 0.6)
    return distance_score + punctuation_score + silence_score


def _snap_change_to_gap(
    items: Sequence[Mapping[str, Any]],
    change_time: float,
    *,
    config: AlignmentConfig,
    minimum_score: float = 0.0,
    minimum_gap_seconds: float = 0.0,
) -> Optional[int]:
    candidates = []
    for index in range(1, len(items)):
        left_end = float(items[index - 1].get("end", change_time))
        right_start = float(items[index].get("start", change_time))
        if _gap_seconds(items, index) < minimum_gap_seconds:
            continue
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
    if not candidates:
        return None
    best = max(candidates)
    return best[2] if best[0] >= minimum_score else None


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
    char_start = int(group[0]["char_start"])
    if end_index < len(items) and "char_start" in items[end_index]:
        # The next token begins after the separator, so this keeps terminal
        # punctuation on the left group while strip() removes only whitespace.
        char_end = int(items[end_index]["char_start"])
    else:
        char_end = len(transcript)
    return transcript[char_start:char_end].strip()


def _map_text_offset(source: str, target: str, offset: int) -> int:
    """Map a character boundary through a conservative OCR text rewrite."""
    offset = max(0, min(len(source), int(offset)))
    matcher = difflib.SequenceMatcher(None, source, target, autojunk=False)
    blocks = matcher.get_matching_blocks()
    previous_a = previous_b = 0
    for block in blocks:
        if block.a <= offset <= block.a + block.size:
            return max(0, min(len(target), block.b + offset - block.a))
        if offset < block.a:
            source_span = max(1, block.a - previous_a)
            ratio = (offset - previous_a) / source_span
            mapped = previous_b + ratio * (block.b - previous_b)
            return max(0, min(len(target), int(round(mapped))))
        previous_a = block.a + block.size
        previous_b = block.b + block.size
    return len(target)


def _text_parts_from_timing_skeleton(
    corrected_text: str,
    timing_text: str,
    items: Sequence[Mapping[str, Any]],
    boundaries: Sequence[int],
) -> Optional[List[str]]:
    """Transfer only clause boundaries, never old token labels, to OCR text."""
    if difflib.SequenceMatcher(
        None, timing_text, corrected_text, autojunk=False
    ).ratio() < 0.55:
        return None
    old_offsets = [0]
    for index in boundaries[1:-1]:
        if index >= len(items) or "char_start" not in items[index]:
            return None
        old_offsets.append(int(items[index]["char_start"]))
    old_offsets.append(len(timing_text))
    new_offsets = [
        _map_text_offset(timing_text, corrected_text, offset)
        for offset in old_offsets
    ]
    for index in range(1, len(new_offsets)):
        new_offsets[index] = max(new_offsets[index], new_offsets[index - 1])
    parts = [
        corrected_text[left:right].strip()
        for left, right in zip(new_offsets, new_offsets[1:])
    ]
    return parts if all(parts) else None


def align_segments_with_rttm(
    segments: Iterable[Mapping[str, Any]],
    rttm_spans: Iterable[Mapping[str, Any]],
    *,
    config: AlignmentConfig = AlignmentConfig(),
    speaker_boundary_evidence: Optional[Sequence[Mapping[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Produce final sentence-level, speaker-consistent segments."""
    spans = smooth_rttm_spans(rttm_spans, config=config)
    changes = _speaker_change_points(
        spans, config, boundary_evidence=speaker_boundary_evidence
    )
    output: List[Dict[str, Any]] = []
    for source in segments:
        segment = dict(source)
        segment_start = float(segment["start"])
        segment_end = float(segment["end"])
        transcript = str(segment.get("text", ""))
        use_timing_skeleton = bool(
            segment.get("speaker_timing_skeleton_valid")
            and segment.get("speaker_timing_time_stamps")
            and segment.get("speaker_timing_text") is not None
        )
        timing_transcript = str(
            segment.get("speaker_timing_text")
            if use_timing_skeleton else transcript
        )
        saved_items = list(
            segment.get("speaker_timing_time_stamps")
            if use_timing_skeleton else segment.get("qwen3_time_stamps") or []
        )
        if segment.get("speaker_boundary_eligible") is False:
            items = []
            alignment = {
                "status": "coarse",
                "coverage_ratio": 0.0,
                "warning_codes": ["coarse_recovery_bounds"],
            }
        else:
            items, alignment = normalize_timestamp_items(
                saved_items,
                transcript=timing_transcript,
                chunk_start=segment_start,
                chunk_end=segment_end,
            )
        segment["qwen3_time_stamps"] = [] if use_timing_skeleton else items
        if use_timing_skeleton:
            segment["speaker_timing_time_stamps"] = items
            segment["speaker_timing_alignment_status"] = alignment["status"]
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
        identity_only_changes = [
            item
            for item in segment_changes
            if not item.get("text_split_allowed", True)
            and item.get("boundary_role") == "identity_separation_only"
        ]
        if not items or status in {"unusable", "coarse"}:
            segment["speaker"] = normalize_speaker(fallback)
            overlap_total = sum(overlaps.values())
            dominant_overlap = max(overlaps.values(), default=0.0)
            dominance = (
                dominant_overlap / overlap_total if overlap_total > 0.0 else 0.0
            )
            segment["speaker_overlap_dominance"] = round(dominance, 4)
            segment["speaker_overlap_candidates"] = [
                {
                    "speaker": normalize_speaker(speaker),
                    "seconds": round(seconds, 3),
                    "ratio": round(
                        seconds / overlap_total if overlap_total > 0.0 else 0.0,
                        4,
                    ),
                }
                for speaker, seconds in sorted(
                    overlaps.items(), key=lambda item: item[1], reverse=True
                )
            ]
            segment["speaker_assignment_confidence"] = round(
                dominance * (0.70 if status == "coarse" else 0.80), 4
            )
            if identity_only_changes:
                segment["speaker_boundary_ambiguous"] = True
                segment["speaker_assignment_status"] = (
                    "multi_speaker_text_unsplittable"
                )
                segment["speaker_assignment_confidence"] = min(
                    segment["speaker_assignment_confidence"], 0.65
                )
                segment["unsplit_identity_boundaries"] = [
                    round(float(item["time"]), 3)
                    for item in identity_only_changes
                ]
            elif len(overlaps) > 1 and dominance < 0.80:
                segment["speaker_boundary_ambiguous"] = True
                segment["speaker_assignment_status"] = (
                    "ambiguous_coarse_overlap"
                    if status == "coarse"
                    else "ambiguous_unusable_timestamps"
                )
            else:
                segment["speaker_assignment_status"] = (
                    "coarse_dominant_overlap"
                    if status == "coarse"
                    else "dominant_overlap_fallback"
                )
            output.append(segment)
            continue

        sentence_candidates_for_turn = sentence_boundary_indices(items, config=config)
        sentence_splits = set(
            long_turn_boundary_indices(
                items,
                sentence_candidates_for_turn,
                config=config,
            )
        )
        speaker_splits: Dict[int, Dict[str, Any]] = {}
        for change in segment_changes:
            if not change.get("text_split_allowed", True):
                continue
            boundary = _snap_change_to_gap(
                items,
                float(change["time"]),
                config=config,
                # Distance alone scores at most 1.0. Requiring additional
                # punctuation or silence prevents cuts such as "the / rail".
                minimum_score=1.10,
                minimum_gap_seconds=(
                    0.12
                    if change.get("boundary_source") == "acoustic_scan"
                    else 0.0
                ),
            )
            if boundary is not None:
                speaker_splits[boundary] = change
        split_indices = sorted(sentence_splits | set(speaker_splits))

        # Exact transcript spans are mandatory for splitting.  A partial or
        # mismatched alignment is retained for audit, never used to guess text.
        boundaries = [0, *split_indices, len(items)]
        if use_timing_skeleton:
            text_parts = _text_parts_from_timing_skeleton(
                transcript, timing_transcript, items, boundaries
            )
            if text_parts is None:
                text_parts = []
        else:
            text_parts = [
                _text_for_group(transcript, items, left, right)
                for left, right in zip(boundaries, boundaries[1:])
            ]
        if not text_parts or any(part is None or not part for part in text_parts):
            segment["speaker"] = normalize_speaker(fallback)
            segment["speaker_boundary_ambiguous"] = bool(segment_changes)
            if identity_only_changes:
                segment["speaker_assignment_status"] = (
                    "multi_speaker_text_unsplittable"
                )
                segment["speaker_assignment_confidence"] = min(
                    float(segment.get("speaker_assignment_confidence", 1.0)),
                    0.65,
                )
                segment["unsplit_identity_boundaries"] = [
                    round(float(item["time"]), 3)
                    for item in identity_only_changes
                ]
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
            overlap_total = sum(group_overlaps.values())
            dominant_overlap = max(group_overlaps.values(), default=0.0)
            dominance = dominant_overlap / overlap_total if overlap_total > 0 else 0.0
            result["speaker_overlap_dominance"] = round(dominance, 4)
            result["speaker_assignment_confidence"] = round(
                dominance * (1.0 if status == "valid" else 0.85), 4
            )
            result["speaker_assignment_status"] = (
                "word_aligned_dominant_overlap"
                if dominance >= 0.80 else "ambiguous_word_aligned_overlap"
            )
            if len(group_overlaps) > 1 and dominance < 0.80:
                result["speaker_boundary_ambiguous"] = True
            blocked_group_changes = [
                item
                for item in identity_only_changes
                if start < float(item["time"]) < end
            ]
            if blocked_group_changes:
                result["speaker_boundary_ambiguous"] = True
                result["speaker_assignment_status"] = (
                    "multi_speaker_text_unsplittable"
                )
                result["speaker_assignment_confidence"] = min(
                    result["speaker_assignment_confidence"], 0.65
                )
                result["unsplit_identity_boundaries"] = [
                    round(float(item["time"]), 3)
                    for item in blocked_group_changes
                ]
            result["qwen3_time_stamps"] = (
                [] if use_timing_skeleton else items[left:right]
            )
            if use_timing_skeleton:
                result["speaker_timing_time_stamps"] = items[left:right]
            group_items = items[left:right]
            if group_items:
                result["speech_start"] = round(
                    max(start, float(group_items[0].get("start", start)) - 0.10), 3
                )
                result["speech_end"] = round(
                    min(end, float(group_items[-1].get("end", end)) + 0.08), 3
                )
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


def _join_transcript(left: str, right: str) -> str:
    left = str(left or "").rstrip()
    right = str(right or "").lstrip()
    if not left:
        return right
    if not right:
        return left
    if right[0] in ",.;:!?，。；：！？、)]}\u2019\u201d":
        return left + right
    return left + " " + right


def merge_adjacent_speaker_turns(
    segments: Iterable[Mapping[str, Any]],
    *,
    max_gap_seconds: float = 0.80,
    maximum_turn_seconds: float = 12.0,
) -> List[Dict[str, Any]]:
    """Reassemble transport/sentence fragments belonging to one speaker."""
    def reference_component(item: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            key: item.get(key)
            for key in (
                "id", "start", "end", "text", "original_text",
                "qwen3_raw_text", "timestamp_precision",
                "asr_recovery_status", "speaker_boundary_ambiguous",
                "speaker_assignment_confidence", "speaker_raw", "speaker",
            )
        }

    merged: List[Dict[str, Any]] = []
    for source in segments:
        segment = dict(source)
        if not merged:
            merged.append(segment)
            continue
        left = merged[-1]
        gap = float(segment["start"]) - float(left["end"])
        combined_duration = float(segment["end"]) - float(left["start"])
        left_is_coarse = (
            left.get("asr_recovery_status") == "accepted_coarse"
            and left.get("timestamp_precision") == "coarse"
        )
        right_is_coarse = (
            segment.get("asr_recovery_status") == "accepted_coarse"
            and segment.get("timestamp_precision") == "coarse"
        )
        left_text = str(left.get("text") or "").strip()
        distinct_parent = (
            left.get("parent_segment_id") is not None
            and segment.get("parent_segment_id") is not None
            and left.get("parent_segment_id") != segment.get("parent_segment_id")
        )
        protected_sentence_boundary = bool(
            STRONG_PUNCTUATION.search(left_text)
            and (
                distinct_parent
                or "sentence_boundary"
                in (left.get("segment_split_reasons") or [])
            )
        )
        if (
            normalize_speaker(left.get("speaker"))
            == normalize_speaker(segment.get("speaker"))
            and gap <= max_gap_seconds
            and combined_duration <= maximum_turn_seconds
            # A coarse recovered sentence owns an independent acoustic TTS
            # placement window. Merging it with a neighbour would retain only
            # one window and make the combined text start too early or late.
            and not left_is_coarse
            and not right_is_coarse
            and not left.get("speaker_boundary_ambiguous")
            and not segment.get("speaker_boundary_ambiguous")
            and not protected_sentence_boundary
        ):
            left_reference_end = float(left["end"])
            parent_ids = list(left.get("parent_segment_ids") or [])
            if not parent_ids and left.get("parent_segment_id") is not None:
                parent_ids.append(left.get("parent_segment_id"))
            next_parent = segment.get("parent_segment_id")
            if next_parent is not None and next_parent not in parent_ids:
                parent_ids.append(next_parent)
            left["parent_segment_ids"] = parent_ids
            left["end"] = segment["end"]
            if segment.get("speech_end") is not None:
                left["speech_end"] = segment["speech_end"]
            left["alignment_confidence"] = min(
                float(left.get("alignment_confidence", 0.35)),
                float(segment.get("alignment_confidence", 0.35)),
            )
            left["speaker_assignment_confidence"] = min(
                float(left.get("speaker_assignment_confidence", 1.0)),
                float(segment.get("speaker_assignment_confidence", 1.0)),
            )
            raw_speakers = set(left.get("speaker_raws") or [])
            raw_speakers.add(str(left.get("speaker_raw", left.get("speaker"))))
            raw_speakers.update(segment.get("speaker_raws") or [])
            raw_speakers.add(str(segment.get("speaker_raw", segment.get("speaker"))))
            left["speaker_raws"] = sorted(raw_speakers)
            if len(raw_speakers) > 1:
                # The global identity may be correct, but this joined span is
                # no longer a pure single-observation cloning reference.
                left["speaker_reference_mixed_raw"] = True
                components = list(left.get("speaker_reference_components") or [])
                if not components:
                    # Capture the left side before its end/text are extended.
                    left_component = reference_component(left)
                    left_component["end"] = left_reference_end
                    components.append(left_component)
                components.extend(
                    segment.get("speaker_reference_components")
                    or [reference_component(segment)]
                )
                left["speaker_reference_components"] = components
            left["text"] = _join_transcript(left.get("text", ""), segment.get("text", ""))
            left["qwen3_time_stamps"] = list(left.get("qwen3_time_stamps") or []) + list(
                segment.get("qwen3_time_stamps") or []
            )
            left["speaker_turn_merged"] = True
            left["segment_split_reasons"] = sorted(
                set(left.get("segment_split_reasons") or [])
                | set(segment.get("segment_split_reasons") or [])
            )
        else:
            merged.append(segment)
    for index, segment in enumerate(merged):
        segment["id"] = index
    return merged
