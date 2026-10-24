"""Pure helpers for collision-safe dubbing placement."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping


def plan_mix_windows(
    clips: Iterable[Mapping[str, Any]], video_duration: float
) -> List[Dict[str, Any]]:
    """Return sorted clips with an exclusive hard end and available length.

    Unless source overlap is explicitly preserved with ``allow_overlap``, a
    clip can never extend past the next dub onset.  Its own planned ``end`` and
    the video duration remain hard upper bounds as well.
    """
    ordered = sorted(
        (dict(item) for item in clips),
        key=lambda item: float(item.get("start", 0.0)),
    )
    duration = max(0.0, float(video_duration))
    for index, item in enumerate(ordered):
        start = max(0.0, float(item.get("start", 0.0)))
        hard_end = min(duration, float(item.get("end", duration)))
        if index + 1 < len(ordered) and not item.get("allow_overlap"):
            hard_end = min(
                hard_end,
                max(0.0, float(ordered[index + 1].get("start", hard_end))),
            )
        item["start"] = start
        item["hard_end"] = max(start, hard_end)
        item["available_ms"] = max(
            0, int(round((item["hard_end"] - start) * 1000))
        )
    return ordered
