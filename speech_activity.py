"""Shared speech-activity detection and interval helpers.

The detector is deliberately independent from ASR text.  Step 1 uses it to
audit recognition coverage while the WeSpeaker bridge uses it to build its
SAD mask.  It never creates sentence boundaries by itself.
"""

from __future__ import annotations

import math
import os
import re
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


TimeRange = Tuple[float, float]


def _spoken_duration_prior(text: Any) -> Tuple[float, float]:
    """Return a soft centre/ceiling for detecting impossible word spans.

    This never creates a timestamp on its own.  It only prevents an ambiguous
    five-word sentence from inheriting several seconds of unrelated activity.
    """
    value = str(text or "")
    cjk = len(re.findall(r"[\u3400-\u9fff]", value))
    words = len(re.findall(r"[A-Za-z]+(?:['’-][A-Za-z]+)?", value))
    if cjk:
        centre = cjk / 4.2
    elif words:
        centre = words / 2.5
    else:
        centre = max(0.35, len(value.strip()) / 8.0)
    centre = max(0.35, centre)
    ceiling = max(0.75, centre * 1.65 + 0.15)
    return centre, ceiling


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


def fuse_speech_regions(
    primary: Sequence[TimeRange],
    secondary: Sequence[TimeRange],
    *,
    edge_extension: float = 0.18,
) -> List[TimeRange]:
    """Use the source track to restore clipped vocal edges, not add music.

    Demucs vocals are the authoritative activity detector.  A source-track
    region may only extend an already detected vocal island by a small amount;
    an unrelated source-only region can therefore never create an early dub.
    """
    fused: List[TimeRange] = []
    secondary_ranges = merge_ranges(secondary, max_gap=0.08)
    for start, end in merge_ranges(primary, max_gap=0.08):
        refined_start, refined_end = start, end
        for other_start, other_end in secondary_ranges:
            if other_end < start - edge_extension or other_start > end + edge_extension:
                continue
            refined_start = min(refined_start, max(start - edge_extension, other_start))
            refined_end = max(refined_end, min(end + edge_extension, other_end))
        fused.append((refined_start, refined_end))
    return merge_ranges(fused, max_gap=0.10)


def attach_precise_tts_placement(
    segments: Sequence[Mapping[str, Any]],
    speech_ranges: Sequence[TimeRange],
    *,
    word_gap_seconds: float = 0.42,
    pre_roll: float = 0.06,
    post_roll: float = 0.08,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Attach an ideal speech duration and a separate collision boundary.

    Public sentence bounds remain immutable.  Long pauses between aligned
    words are retained as diagnostics but are not counted as speech that a
    translation must fill.  Acoustic activity only refines a word-aligned
    onset when the two sources agree locally.
    """
    active = merge_ranges(speech_ranges, max_gap=0.10)
    output: List[Dict[str, Any]] = []
    audit: List[Dict[str, Any]] = []
    for source in segments:
        segment = dict(source)
        if str(segment.get("timestamp_precision") or "").lower() != "word":
            output.append(segment)
            continue
        items = []
        for collection in (
            segment.get("qwen3_time_stamps") or [],
            segment.get("speaker_timing_time_stamps") or [],
        ):
            for item in collection:
                if not isinstance(item, Mapping):
                    continue
                try:
                    start = float(item.get("start"))
                    end = float(item.get("end", start))
                except (TypeError, ValueError):
                    continue
                if end > start and str(item.get("text", "")).strip():
                    items.append((start, end))
        if not items:
            output.append(segment)
            continue
        outer_start = float(segment.get("start", 0.0))
        outer_end = float(segment.get("end", outer_start))
        word_ranges = merge_ranges(
            [
                (max(outer_start, start), min(outer_end, end))
                for start, end in items
                if end > outer_start and start < outer_end
            ],
            max_gap=float(word_gap_seconds),
        )
        if not word_ranges:
            output.append(segment)
            continue
        internal_pause_threshold = max(
            float(word_gap_seconds),
            float(os.getenv(
                "AUTODUB_TTS_DOMINANT_ISLAND_MIN_PAUSE_SECONDS", "1.00"
            )),
        )
        internal_pauses = [
            [round(left[1], 3), round(right[0], 3)]
            for left, right in zip(word_ranges, word_ranges[1:])
            if right[0] - left[1] > float(word_gap_seconds)
        ]
        long_internal_pauses = [
            (left, right)
            for left, right in zip(word_ranges, word_ranges[1:])
            if right[0] - left[1] >= internal_pause_threshold
        ]
        word_start = word_ranges[0][0]
        word_end = word_ranges[-1][1]
        acoustic = intersect_ranges(active, [(word_start - 0.25, word_end + 0.25)])
        onset = max(outer_start, word_start - float(pre_roll))
        # ``aligned_end`` is evidence for how long the source actually spoke;
        # it is not the physical end of the dubbing slot.  Keeping those two
        # concepts in one variable used to collapse a short interjection to a
        # 0.15 s hard window even when the sentence owned trailing silence.
        # The router will still clip the physical window at the next dub onset,
        # so exposing the remainder of this sentence cannot create a new
        # collision.
        aligned_end = min(outer_end, word_end + float(post_roll))
        hard_end = aligned_end
        if acoustic:
            first_acoustic = acoustic[0][0]
            if abs(first_acoustic - word_start) <= 0.25:
                # Energy VAD frequently starts after an unvoiced consonant.
                # It may refine a loose pre-roll but must never push dubbing
                # past the first aligned word onset and make a short line late.
                onset = min(word_start, max(onset, first_acoustic - 0.03))
        ambiguous = bool(segment.get("speaker_boundary_ambiguous"))
        duration_centre, duration_ceiling = _spoken_duration_prior(
            segment.get("text", "")
        )
        original_onset = onset
        leading_slack_used = 0.0
        dominant_word_island = None
        dominant_island_mode = bool(ambiguous and long_internal_pauses)
        if dominant_island_mode:
            # The enclosing word span is not spoken continuously.  Keep the
            # complete semantic sentence, but anchor its single TTS clip to
            # the largest lexical island instead of centring it inside a long
            # silent gap.  Effective duration below still counts every word
            # island, never the silence between them.
            dominant_index, dominant_range = max(
                enumerate(word_ranges),
                key=lambda indexed: (
                    indexed[1][1] - indexed[1][0],
                    indexed[0],
                ),
            )
            dominant_start, dominant_end = dominant_range
            onset = max(outer_start, dominant_start - float(pre_roll))
            aligned_end = min(outer_end, dominant_end + float(post_roll))
            hard_end = aligned_end
            dominant_word_island = {
                "index": dominant_index,
                "start": round(dominant_start, 3),
                "end": round(dominant_end, 3),
                "duration": round(dominant_end - dominant_start, 3),
            }
        # A complete short sentence can own useful silence before its first
        # aligned word.  When the remaining slot is shorter than a conservative
        # pronunciation prior, borrow only the amount required, stay inside
        # the immutable sentence boundary, and cap the advance.  Ambiguous
        # speaker boundaries never receive this adjustment.
        available_after_onset = max(0.0, outer_end - onset)
        needed_leading_slack = max(
            0.0,
            min(duration_centre, outer_end - outer_start)
            - available_after_onset,
        )
        if not ambiguous and needed_leading_slack > 0.0:
            maximum_leading_slack = max(
                0.0,
                float(os.getenv(
                    "AUTODUB_TTS_MAX_LEADING_SLACK_SECONDS", "0.35"
                )),
            )
            leading_slack_used = min(
                needed_leading_slack,
                maximum_leading_slack,
                max(0.0, onset - outer_start),
            )
            onset -= leading_slack_used
        speech_islands = intersect_ranges(active, [(onset, aligned_end)]) or (
            [tuple(word_ranges[dominant_word_island["index"]])]
            if dominant_word_island is not None else word_ranges
        )
        timing_reason = (
            "ambiguous_long_pause_dominant_island"
            if dominant_island_mode else "word_vad_consensus"
        )
        ratio_guard = None
        word_span = max(0.0, word_end - word_start)
        # Speaker ambiguity means the inherited word skeleton may still be
        # structurally valid while owning activity from a neighbouring turn.
        # When its span is implausible for the sentence, retain the median word
        # anchor but limit it to a local acoustic window.  Public sentence
        # bounds remain untouched.
        if ambiguous and word_span > duration_ceiling and not dominant_island_mode:
            midpoints = sorted((start + end) / 2.0 for start, end in items)
            word_centre = midpoints[len(midpoints) // 2]
            guarded_duration = min(
                duration_ceiling,
                max(0.45, duration_centre * 1.25),
            )
            guarded_start = max(
                outer_start,
                word_centre - guarded_duration / 2.0,
            )
            guarded_end = min(
                outer_end,
                guarded_start + guarded_duration,
            )
            if guarded_end - guarded_start < guarded_duration:
                guarded_start = max(outer_start, guarded_end - guarded_duration)
            guarded_activity = intersect_ranges(
                speech_islands, [(guarded_start, guarded_end)]
            )
            if guarded_activity:
                onset = max(
                    outer_start,
                    guarded_activity[0][0] - float(pre_roll),
                )
                hard_end = min(
                    outer_end,
                    guarded_activity[-1][1] + float(post_roll),
                )
                speech_islands = guarded_activity
            else:
                onset, hard_end = guarded_start, guarded_end
                speech_islands = [(guarded_start, guarded_end)]
            timing_reason = "ambiguous_word_span_ratio_guard"
            aligned_end = hard_end
            ratio_guard = {
                "word_span": round(word_span, 3),
                "text_duration_centre": round(duration_centre, 3),
                "text_duration_ceiling": round(duration_ceiling, 3),
                "guarded_start": round(onset, 3),
                "guarded_end": round(hard_end, 3),
            }
        effective = sum(max(0.0, end - start) for start, end in speech_islands)
        word_effective = sum(max(0.0, end - start) for start, end in word_ranges)
        # Energy VAD can clip unvoiced consonants.  Never budget less than the
        # aligned word evidence, while still removing long internal silence.
        if timing_reason == "ambiguous_word_span_ratio_guard":
            # The word span is the evidence being rejected; do not re-expand
            # the target duration from that same span.
            effective = min(
                max(effective, min(duration_centre, hard_end - onset)),
                hard_end - onset,
            )
        else:
            effective = max(effective, word_effective)
        # A word/VAD span may be shorter than a humanly pronounceable version
        # of the complete sentence (especially unvoiced consonants and one-word
        # turns).  It remains a natural-duration target, not permission to
        # overlap the next turn.
        minimum_basis = (
            word_effective + float(pre_roll) + float(post_roll)
            if dominant_island_mode else duration_centre
        )
        minimum_pronounceable = min(
            max(0.1, minimum_basis),
            max(0.1, outer_end - onset),
        )
        effective = max(effective + pre_roll + post_roll, minimum_pronounceable)
        effective = min(max(0.1, effective), max(0.1, outer_end - onset))
        # Make trailing silence owned by the sentence available for fitting.
        # The chronological collision pass in TTSRouter remains authoritative
        # and will reduce this to the next onset when necessary.
        hard_end = outer_end
        segment.update({
            "tts_placement_start": round(onset, 3),
            "tts_hard_end": round(hard_end, 3),
            "tts_target_duration": round(effective, 3),
            "tts_aligned_speech_end": round(aligned_end, 3),
            "tts_minimum_pronounceable_duration": round(
                minimum_pronounceable, 3
            ),
            "tts_original_placement_start": round(original_onset, 3),
            "tts_leading_slack_used": round(leading_slack_used, 3),
            "tts_leading_slack_policy": (
                "bounded_sentence_internal_leading_slack"
                if leading_slack_used > 0.0 else "not_needed"
            ),
            "tts_dominant_word_island": dominant_word_island,
            "tts_speech_islands": [
                [round(start, 3), round(end, 3)] for start, end in speech_islands
            ],
            "tts_internal_pauses": internal_pauses,
            "tts_timing_precision": "word_vad_consensus",
            "tts_timing_confidence": round(
                (
                    0.58
                    if timing_reason == "ambiguous_word_span_ratio_guard"
                    else 0.68
                    if timing_reason == "ambiguous_long_pause_dominant_island"
                    else (
                        0.72 if ambiguous
                        else min(
                            0.98,
                            0.82
                            + (0.12 if acoustic else 0.0)
                            - (0.06 if leading_slack_used > 0.0 else 0.0),
                        )
                    )
                ), 4
            ),
            "tts_timing_reason": timing_reason,
            "tts_timing_ratio_guard": ratio_guard,
        })
        audit.append({
            "segment_id": str(segment.get("id")),
            "precision": "word_vad_consensus",
            "placement_start": round(onset, 3),
            "hard_end": round(hard_end, 3),
            "target_speech_duration": round(effective, 3),
            "original_placement_start": round(original_onset, 3),
            "leading_slack_used": round(leading_slack_used, 3),
            "dominant_word_island": dominant_word_island,
            "speech_islands": segment["tts_speech_islands"],
            "internal_pauses": internal_pauses,
            "timing_reason": timing_reason,
            "ratio_guard": ratio_guard,
        })
        output.append(segment)
    return output, audit


def attach_coarse_tts_placement(
    segments: Sequence[Mapping[str, Any]],
    speech_ranges: Sequence[TimeRange],
    *,
    pre_roll: float = 0.06,
    post_roll: float = 0.08,
    subtitle_tolerance: float = 0.70,
    min_activity: float = 0.16,
    hard_silence_seconds: float = 0.75,
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
        precision = str(segment.get("timestamp_precision") or "").lower()
        recovery_coarse = (
            str(segment.get("asr_recovery_status") or "") == "accepted_coarse"
            and precision == "coarse"
        )
        if precision not in {"coarse", "unusable"}:
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

        if recovery_coarse:
            # OCR recovery may cover only one dialogue island inside a wider
            # unresolved interval. Keep the contiguous island group nearest
            # its subtitle hint and expose the remainder for local recovery.
            selected = [overlaps[first_index]]
            for candidate in overlaps[first_index + 1:]:
                if candidate[0] - selected[-1][1] >= float(hard_silence_seconds):
                    break
                selected.append(candidate)
            excluded = overlaps[first_index + len(selected):]
        else:
            # A primary ASR sentence without word timestamps already owns its
            # full outer interval. Use all acoustic islands for its speech
            # budget, while retaining long pauses as non-speaking time.
            selected = overlaps
            excluded = []
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
                "tts_hard_end": round(placement_end, 3),
                "tts_target_duration": round(
                    min(
                        placement_end - placement_start,
                        max(
                            float(min_activity),
                            active_duration + float(pre_roll) + float(post_roll),
                        ),
                    ), 3
                ),
                "tts_available_end": round(outer_end, 3),
                "tts_speech_islands": [
                    [round(start, 3), round(end, 3)] for start, end in selected
                ],
                "tts_timing_precision": (
                    "coarse_recovery_vad" if recovery_coarse
                    else "coarse_primary_vad"
                ),
                "recovery_timing_method": "vocal_activity_with_ocr_soft_hint",
                "recovery_timing_confidence": round(confidence, 4),
                "coarse_timing_excluded_islands": [
                    [round(start, 3), round(end, 3)] for start, end in excluded
                ],
                "coarse_timing_requires_local_recovery": bool(excluded),
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
            excluded_islands=[
                [round(start, 3), round(end, 3)] for start, end in excluded
            ],
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
