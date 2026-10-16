"""Shared speech-activity detection and interval helpers.

The detector is deliberately independent from ASR text.  Step 1 uses it to
audit recognition coverage while the WeSpeaker bridge uses it to build its
SAD mask.  It never creates sentence boundaries by itself.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Tuple


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
