"""Bounded TTS extension using source speech activity and the mix timeline."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from typing import Any

from mix_timing import plan_complete_mix_schedule, plan_mix_windows
from speech_activity import TimeRange, subtract_ranges


_EPSILON = 1e-9


@dataclass(frozen=True)
class ElasticTimingSettings:
    enabled: bool = False
    max_extension_seconds: float = 0.40
    safety_gap_seconds: float = 0.15

    def __post_init__(self) -> None:
        if not math.isfinite(self.max_extension_seconds) or self.max_extension_seconds < 0.0:
            raise ValueError("AUTODUB_TTS_ELASTIC_MAX_EXTENSION_SECONDS must be finite and non-negative")
        if not math.isfinite(self.safety_gap_seconds) or self.safety_gap_seconds < 0.15:
            raise ValueError("AUTODUB_TTS_ELASTIC_SAFETY_GAP_SECONDS must be finite and at least 0.15")

    @classmethod
    def from_config(cls, config: Any) -> ElasticTimingSettings:
        if not getattr(config, "enable_elastic_timing", False):
            return cls()
        return cls(
            enabled=True,
            max_extension_seconds=float(os.getenv("AUTODUB_TTS_ELASTIC_MAX_EXTENSION_SECONDS", "0.40")),
            safety_gap_seconds=float(os.getenv("AUTODUB_TTS_ELASTIC_SAFETY_GAP_SECONDS", "0.15")),
        )


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _time_range(start: Any, end: Any) -> TimeRange | None:
    left, right = _finite(start), _finite(end)
    if left is None or right is None or not 0.0 <= left < right:
        return None
    return left, right


def _confirmed_silence(state: Mapping[str, Any]) -> list[TimeRange]:
    report = state.get("tts_timing_report")
    if not isinstance(report, Mapping) or report.get("status") != "ok":
        return []
    diagnostics = report.get("primary_diagnostics")
    if not isinstance(diagnostics, Mapping):
        return []
    measured_duration = _finite(diagnostics.get("duration"))
    speech_duration = _finite(diagnostics.get("speech_duration"))
    video_duration = _finite(state.get("duration"))
    if not measured_duration or not speech_duration or not video_duration:
        return []
    covered_end = min(video_duration, measured_duration)
    if covered_end <= 0.0 or not 0.0 < speech_duration <= measured_duration + 0.001:
        return []

    # Use Step 3's padded, fused vocal activity only within measured audio.
    # An empty/invalid detector result does not establish a silent recording.
    activity = report.get("fused_speech_regions")
    if not isinstance(activity, (list, tuple)) or not activity:
        return []
    blocked = []
    has_observed_activity = False
    for pair in activity:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            return []
        span = _time_range(*pair)
        if span is None:
            return []
        blocked.append(span)
        has_observed_activity |= span[0] < covered_end
    if not has_observed_activity:
        return []

    # Local ASR recovery can flag dialogue even where the energy detector
    # failed to establish a usable speech region.
    for key in ("asr_unresolved_dialogue_regions", "asr_acoustic_uncertain_regions"):
        rows = state.get(key) or []
        if not isinstance(rows, (list, tuple)):
            return []
        for row in rows:
            if not isinstance(row, Mapping):
                return []
            span = _time_range(row.get("start"), row.get("end"))
            if span is None:
                return []
            blocked.append(span)
    return subtract_ranges([(0.0, covered_end)], blocked)


def plan_elastic_windows(
    state: Mapping[str, Any], settings: ElasticTimingSettings,
) -> dict[str, dict[str, Any]]:
    """Bound each eligible source interval before the formal TTS is measured."""
    if not settings.enabled:
        return {}
    rows = []
    seen = set()
    # Include translation-skipped and synthesis-failed source lines as blockers.
    for segment in state.get("segments_step3") or []:
        span = _time_range(segment.get("start"), segment.get("end"))
        sid = str(segment.get("id", "")) if segment.get("id") is not None else ""
        if span is None or not sid or sid in seen:
            return {}
        seen.add(sid)
        rows.append((*span, sid, bool(segment.get("speaker_boundary_ambiguous"))))
    rows.sort()

    silence = _confirmed_silence(state)
    silence_starts = [start for start, _ in silence]
    windows = {}
    preceding_end = 0.0
    for index, (start, end, sid, ambiguous) in enumerate(rows):
        next_start = rows[index + 1][0] if index + 1 < len(rows) else None
        overlap = preceding_end > start or (next_start is not None and next_start < end)
        preceding_end = max(preceding_end, end)
        allowed = end
        if overlap:
            reason = "source_overlap"
        elif ambiguous:
            reason = "ambiguous_speaker_boundary"
        elif settings.max_extension_seconds <= 0.0:
            reason = "extension_disabled_by_limit"
        elif next_start is not None and next_start - end <= settings.safety_gap_seconds + _EPSILON:
            reason = "next_sentence_safety_gap"
        else:
            silence_index = bisect_right(silence_starts, end + _EPSILON) - 1
            silence_end = silence[silence_index][1] if silence_index >= 0 else end
            # The silence must touch the source end. A later silent island
            # cannot be reached by playing across intervening speech.
            allowed = min(end + settings.max_extension_seconds, silence_end)
            if next_start is not None:
                allowed = min(allowed, next_start - settings.safety_gap_seconds)
            reason = "confirmed_trailing_silence" if allowed > end + _EPSILON else "no_confirmed_trailing_silence"
        windows[sid] = {
            "source_start": start,
            "source_end": end,
            "next_source_start": next_start,
            "maximum_end": max(end, allowed),
            "safety_gap_seconds": settings.safety_gap_seconds,
            "reason": reason,
        }
    return windows


def elastic_stage_fingerprint(
    state: Mapping[str, Any], settings: ElasticTimingSettings,
) -> str:
    """Bind the optional timing policy without changing legacy WAV fingerprints."""
    payload = {
        "version": 1,
        "max_extension_seconds": settings.max_extension_seconds,
        "safety_gap_seconds": settings.safety_gap_seconds,
        "windows": plan_elastic_windows(state, settings),
        "duration": state.get("duration"),
        "maximum_early_shift_seconds": os.getenv(
            "AUTODUB_TTS_MAX_COMPLETE_CLIP_EARLY_SHIFT_SECONDS", "0.35"
        ),
    }
    return sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def decide_elastic_timing(window: Mapping[str, Any], tts_duration: float) -> dict[str, Any]:
    """Use the minimum extension needed for the complete, unwarped WAV."""
    source_start, source_end = window["source_start"], window["source_end"]
    start, legacy_end = window["placement_start"], window["legacy_end"]
    original_duration = source_end - source_start
    requested_end = start + tts_duration
    maximum_end = window["maximum_end"]
    applied = False
    if tts_duration <= original_duration + _EPSILON:
        reason = "fits_original_window"
    elif maximum_end <= source_end + _EPSILON:
        reason = window["reason"]
    elif not source_start - _EPSILON <= start < legacy_end or abs(legacy_end - source_end) > _EPSILON:
        # Keep the hard stop of narrower VAD/recovery windows authoritative.
        reason = "narrower_legacy_window"
    elif requested_end > maximum_end + _EPSILON:
        reason = "insufficient_elastic_window"
    else:
        reason = "extended_in_confirmed_silence"
        applied = True
    allowed_end = min(requested_end, maximum_end) if applied else legacy_end
    return {
        **window,
        "tts_duration": tts_duration,
        "original_window_seconds": original_duration,
        "available_extension_seconds": max(0.0, maximum_end - source_end),
        "allowed_end": allowed_end,
        "extension_seconds": max(0.0, allowed_end - source_end) if applied else 0.0,
        "needs_speed_adjustment": not applied and requested_end > legacy_end + _EPSILON,
        "applied": applied,
        "reason": reason,
    }


def conflicting_elastic_clips(
    clips: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
    video_duration: float,
    *,
    maximum_early_shift_seconds: float,
) -> set[str]:
    """Recheck the silence reservation against Step 5's complete WAV schedule."""
    by_id = {str(row["segment_id"]): row for row in results}
    applied = {
        str(clip["segment_id"]): timing
        for clip in clips
        if (timing := by_id.get(str(clip["segment_id"]), {}).get("elastic_timing"))
        and timing["applied"]
    }
    if not applied:
        return set()

    measured = []
    for clip in clips:
        length = _finite(by_id.get(str(clip["segment_id"]), {}).get("actual_duration"))
        if length is None or length <= 0.0:
            return set(applied)
        # Mixer.run measures len(AudioSegment) and places it on whole ms.
        measured.append({**clip, "audio_duration_seconds": round(length * 1000) / 1000})
    scheduled = plan_complete_mix_schedule(
        plan_mix_windows(measured, video_duration), video_duration,
        maximum_early_shift_seconds=maximum_early_shift_seconds,
    )
    starts = [round(clip["mix_start"] * 1000) / 1000 for clip in scheduled]
    following_starts = [math.inf] * len(scheduled)
    next_start = math.inf
    for index in range(len(scheduled) - 1, -1, -1):
        following_starts[index] = next_start
        next_start = min(next_start, starts[index])

    conflicts = set()
    preceding_end = 0.0
    for index, clip in enumerate(scheduled):
        sid = str(clip["segment_id"])
        end = starts[index] + clip["audio_duration_seconds"]
        timing = applied.get(sid)
        if timing and (
            abs(clip["timing_shift_seconds"]) > _EPSILON
            or end > timing["maximum_end"] + _EPSILON
            or end + timing["safety_gap_seconds"] > following_starts[index] + _EPSILON
            or preceding_end > timing["source_end"] + _EPSILON
        ):
            conflicts.add(sid)
        preceding_end = max(preceding_end, end)
    return conflicts
