"""Shared speech-activity detection and interval helpers.

The detector is deliberately independent from ASR text.  Step 1 uses it to
audit recognition coverage while the WeSpeaker bridge uses it to build its
SAD mask.  It never creates sentence boundaries by itself.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


TimeRange = Tuple[float, float]


def merge_ranges(ranges: Iterable[TimeRange], max_gap: float = 0.0) -> List[TimeRange]:
    merged: List[List[float]] = []
    for start, end in sorted((float(s), float(e)) for s, e in ranges):
        if end <= start:
            continue
        if merged and start - merged[-1][1] <= max_gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def subtract_ranges(
    source: Sequence[TimeRange],
    covered: Sequence[TimeRange],
    *,
    min_duration: float = 0.0,
) -> List[TimeRange]:
    """Return portions of ``source`` not contained in ``covered``."""
    masks = merge_ranges(covered)
    result: List[TimeRange] = []
    for source_start, source_end in merge_ranges(source):
        cursor = source_start
        for mask_start, mask_end in masks:
            if mask_end <= cursor:
                continue
            if mask_start >= source_end:
                break
            if mask_start > cursor and mask_start - cursor >= min_duration:
                result.append((cursor, min(mask_start, source_end)))
            cursor = max(cursor, mask_end)
            if cursor >= source_end:
                break
        if source_end - cursor >= min_duration:
            result.append((cursor, source_end))
    return result


def intersect_ranges(
    left: Sequence[TimeRange], right: Sequence[TimeRange]
) -> List[TimeRange]:
    """Return intersections between two interval collections."""
    left_ranges = merge_ranges(left)
    right_ranges = merge_ranges(right)
    intersections: List[TimeRange] = []
    left_index = 0
    right_index = 0
    while left_index < len(left_ranges) and right_index < len(right_ranges):
        left_start, left_end = left_ranges[left_index]
        right_start, right_end = right_ranges[right_index]
        start = max(left_start, right_start)
        end = min(left_end, right_end)
        if end > start:
            intersections.append((start, end))
        if left_end <= right_end:
            left_index += 1
        else:
            right_index += 1
    return intersections


def attach_coarse_tts_placement(
    segments: Sequence[Mapping[str, Any]],
    speech_ranges: Sequence[TimeRange],
    *,
    pre_roll: float = 0.06,
    post_roll: float = 0.08,
    subtitle_tolerance: float = 0.70,
    min_activity: float = 0.16,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Attach acoustic TTS windows only to coarse recovered dialogue.

    Existing ``start``/``end`` values are immutable. OCR timing is merely a
    soft ordering hint; final placement always starts on detected vocal
    activity. Precise ASR segments never receive these internal fields.
    """
    active = merge_ranges(speech_ranges, max_gap=0.10)
    output: List[Dict[str, Any]] = []
    audit: List[Dict[str, Any]] = []
    for source in segments:
        segment = dict(source)
        if not (
            str(segment.get("asr_recovery_status") or "") == "accepted_coarse"
            and str(segment.get("timestamp_precision") or "") == "coarse"
        ):
            output.append(segment)
            continue

        outer_start = float(segment.get("start", 0.0))
        outer_end = float(segment.get("end", outer_start))
        overlaps = intersect_ranges(active, [(outer_start, outer_end)])
        record: Dict[str, Any] = {
            "segment_id": str(segment.get("id")),
            "outer_start": round(outer_start, 3),
            "outer_end": round(outer_end, 3),
            "speech_candidates": [
                [round(start, 3), round(end, 3)] for start, end in overlaps
            ],
        }
        if not overlaps:
            record.update(accepted=False, reason="no_acoustic_activity_in_outer_window")
            audit.append(record)
            output.append(segment)
            continue

        hint_value = segment.get("ocr_subtitle_first_seen")
        try:
            hint = float(hint_value) if hint_value is not None else None
        except (TypeError, ValueError):
            hint = None

        first_index = 0
        if hint is not None and len(overlaps) > 1:
            # Prefer speech that contains or follows the subtitle observation.
            # Subtitle text commonly appears shortly before the voice; choosing
            # a nearby *preceding* speech island would recreate the early-start
            # bug this function is meant to prevent.
            eligible = [
                index
                for index, (_, end) in enumerate(overlaps)
                if end >= hint
            ]
            if not eligible:
                eligible = [
                    index
                    for index, (_, end) in enumerate(overlaps)
                    if end >= hint - float(subtitle_tolerance)
                ]
            if eligible:
                first_index = min(
                    eligible,
                    key=lambda index: (
                        0.0
                        if overlaps[index][0] <= hint <= overlaps[index][1]
                        else min(
                            abs(hint - overlaps[index][0]),
                            abs(hint - overlaps[index][1]),
                        ),
                        index,
                    ),
                )

        selected = overlaps[first_index:]
        active_duration = sum(end - start for start, end in selected)
        if active_duration < float(min_activity):
            record.update(
                accepted=False,
                reason="insufficient_acoustic_activity",
                active_duration=round(active_duration, 3),
            )
            audit.append(record)
            output.append(segment)
            continue

        acoustic_start = float(selected[0][0])
        acoustic_end = float(selected[-1][1])
        placement_start = max(outer_start, acoustic_start - float(pre_roll))
        placement_end = min(outer_end, acoustic_end + float(post_roll))
        if placement_end - placement_start < float(min_activity):
            record.update(accepted=False, reason="invalid_refined_window")
            audit.append(record)
            output.append(segment)
            continue

        outer_duration = max(0.001, outer_end - outer_start)
        activity_ratio = min(1.0, active_duration / outer_duration)
        confidence = min(0.95, 0.62 + 0.25 * activity_ratio)
        segment.update(
            {
                "recovery_outer_start": round(outer_start, 3),
                "recovery_outer_end": round(outer_end, 3),
                "speech_anchor_start": round(acoustic_start, 3),
                "speech_anchor_end": round(acoustic_end, 3),
                "tts_placement_start": round(placement_start, 3),
                "tts_placement_end": round(placement_end, 3),
                "tts_available_end": round(outer_end, 3),
                "recovery_timing_method": "vocal_activity_with_ocr_soft_hint",
                "recovery_timing_confidence": round(confidence, 4),
            }
        )
        record.update(
            accepted=True,
            reason="acoustic_activity_window",
            subtitle_first_seen=(round(hint, 3) if hint is not None else None),
            selected_candidate_index=first_index,
            placement_start=round(placement_start, 3),
            placement_end=round(placement_end, 3),
            available_end=round(outer_end, 3),
            active_duration=round(active_duration, 3),
            confidence=round(confidence, 4),
        )
        audit.append(record)
        output.append(segment)
    return output, audit


def detect_speech_regions(
    audio_path: str,
    *,
    min_silence_ms: int = 180,
    keep_ms: int = 80,
    merge_gap_ms: int = 120,
    min_region_ms: int = 220,
    db_offset: float = 18.0,
    floor_dbfs: float = -50.0,
) -> Tuple[List[TimeRange], Dict[str, float]]:
    """Detect speech-bearing ranges using adaptive short-time energy.

    The return value includes diagnostics so callers can record exactly which
    detector settings produced an audit.  Empty audio returns no ranges rather
    than treating the entire file as speech.
    """
    try:
        import numpy as np
        import soundfile as sf
    except ImportError:
        # The main pipeline already depends on pydub.  Keep coverage auditing
        # available even in environments where the WeSpeaker soundfile stack
        # is not installed; this fallback is still independent from ASR text.
        from pydub import AudioSegment, silence

        audio = AudioSegment.from_file(audio_path)
        duration_ms = float(len(audio))
        reference_dbfs = float(audio.dBFS)
        if reference_dbfs == float("-inf"):
            return [], {
                "duration": round(duration_ms / 1000.0, 3),
                "threshold_dbfs": float(floor_dbfs),
                "reference_dbfs": reference_dbfs,
                "speech_duration": 0.0,
                "backend": "pydub",
            }
        threshold = max(float(floor_dbfs), reference_dbfs - float(db_offset))
        raw = silence.detect_nonsilent(
            audio,
            min_silence_len=int(min_silence_ms),
            silence_thresh=threshold,
            seek_step=10,
        )
        padded = [
            (max(0.0, start - keep_ms), min(duration_ms, end + keep_ms))
            for start, end in raw
        ]
        regions_ms = merge_ranges(padded, max_gap=merge_gap_ms)
        regions = [
            (round(start / 1000.0, 3), round(end / 1000.0, 3))
            for start, end in regions_ms
            if end - start >= min_region_ms
        ]
        return regions, {
            "duration": round(duration_ms / 1000.0, 3),
            "threshold_dbfs": round(threshold, 3),
            "reference_dbfs": round(reference_dbfs, 3),
            "speech_duration": round(sum(end - start for start, end in regions), 3),
            "backend": "pydub",
        }

    samples, sample_rate = sf.read(audio_path, dtype="float32")
    if getattr(samples, "ndim", 1) > 1:
        samples = samples.mean(axis=1)
    if len(samples) <= 0 or sample_rate <= 0:
        return [], {"duration": 0.0, "threshold_dbfs": floor_dbfs}

    frame_ms = 30.0
    hop_ms = 10.0
    frame_samples = max(1, round(sample_rate * frame_ms / 1000.0))
    hop_samples = max(1, round(sample_rate * hop_ms / 1000.0))
    starts = np.arange(0, len(samples), hop_samples, dtype=int)
    rms = np.asarray(
        [
            math.sqrt(
                float(np.mean(np.square(samples[start : start + frame_samples])))
                + 1e-12
            )
            for start in starts
        ]
    )
    frame_dbfs = 20.0 * np.log10(np.maximum(rms, 1e-12))
    reference_rms = math.sqrt(float(np.mean(np.square(samples))) + 1e-12)
    reference_dbfs = 20.0 * math.log10(max(reference_rms, 1e-12))
    threshold = max(float(floor_dbfs), reference_dbfs - float(db_offset))
    active = np.flatnonzero(frame_dbfs >= threshold)
    duration_ms = len(samples) / sample_rate * 1000.0

    raw: List[TimeRange] = []
    if len(active):
        run_start = int(active[0])
        previous = run_start
        max_inactive_frames = max(1, round(min_silence_ms / hop_ms))
        for frame_index in map(int, active[1:]):
            if frame_index - previous > max_inactive_frames:
                raw.append((run_start * hop_ms, previous * hop_ms + frame_ms))
                run_start = frame_index
            previous = frame_index
        raw.append((run_start * hop_ms, previous * hop_ms + frame_ms))

    padded = [
        (max(0.0, start - keep_ms), min(duration_ms, end + keep_ms))
        for start, end in raw
    ]
    regions_ms = merge_ranges(padded, max_gap=merge_gap_ms)
    regions = [
        (round(start / 1000.0, 3), round(end / 1000.0, 3))
        for start, end in regions_ms
        if end - start >= min_region_ms
    ]
    diagnostics = {
        "duration": round(duration_ms / 1000.0, 3),
        "threshold_dbfs": round(threshold, 3),
        "reference_dbfs": round(reference_dbfs, 3),
        "speech_duration": round(sum(end - start for start, end in regions), 3),
        "backend": "numpy_soundfile",
    }
    return regions, diagnostics
