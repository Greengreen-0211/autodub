"""Pure helpers for collision-safe dubbing placement."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping


def plan_mix_windows(
    clips: Iterable[Mapping[str, Any]], video_duration: float
) -> List[Dict[str, Any]]:
    """Return sorted clips with an exclusive hard end and available length.

    A clip may extend past the next onset only by its measured
    ``allowed_overlap_seconds`` budget.  A legacy boolean cannot grant
    unbounded overlap.  The clip's planned ``end`` and video duration remain
    hard upper bounds as well.
    """
    ordered = sorted(
        (dict(item) for item in clips),
        key=lambda item: float(item.get("start", 0.0)),
    )
    duration = max(0.0, float(video_duration))
    for index, item in enumerate(ordered):
        start = max(0.0, float(item.get("start", 0.0)))
        hard_end = min(duration, float(item.get("end", duration)))
        if index + 1 < len(ordered):
            next_start = max(
                0.0, float(ordered[index + 1].get("start", hard_end))
            )
            try:
                allowed_overlap = max(
                    0.0, float(item.get("allowed_overlap_seconds", 0.0))
                )
            except (TypeError, ValueError):
                allowed_overlap = 0.0
            hard_end = min(hard_end, next_start + allowed_overlap)
            item["allowed_overlap_seconds"] = allowed_overlap
            item["allow_overlap"] = allowed_overlap > 0.0
        item["start"] = start
        item["hard_end"] = max(start, hard_end)
        item["available_ms"] = max(
            0, int(round((item["hard_end"] - start) * 1000))
        )
    return ordered


def validate_mix_audit(
    expected_segment_ids: Iterable[Any],
    audit_rows: Iterable[Mapping[str, Any]],
    *,
    maximum_trim_seconds: float = 0.03,
) -> None:
    """Fail when a verified TTS clip did not actually reach the mix stem.

    Step 4 already owns duration adaptation. Step 5 must not silently repair a
    substantially overlong clip by cutting off its final word, nor may it omit
    a clip because its file/window disappeared between validation and mixing.
    """
    expected = [str(value) for value in expected_segment_ids]
    rows = list(audit_rows)
    actual = [str(row.get("segment_id")) for row in rows]
    if len(actual) != len(set(actual)):
        raise RuntimeError("Step5 mix audit found duplicate segment IDs")
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    if missing or unexpected:
        raise RuntimeError(
            "Step5 mix audit coverage mismatch: "
            f"missing={missing or 'none'} unexpected={unexpected or 'none'}"
        )
    failures = []
    for row in rows:
        sid = str(row.get("segment_id"))
        if float(row.get("audible_duration") or 0.0) <= 0.0:
            failures.append(f"{sid}:zero audible duration")
        if int(row.get("clip_rms") or 0) <= 0:
            failures.append(f"{sid}:silent clip reached mixer")
        if not bool(row.get("mix_verified", False)):
            failures.append(f"{sid}:clip is absent from rendered TTS stem")
        trimmed = max(
            0.0,
            float(row.get("original_wav_duration") or 0.0)
            - float(row.get("audible_duration") or 0.0),
        )
        if trimmed > float(maximum_trim_seconds) + 1e-6:
            failures.append(
                f"{sid}:would trim {trimmed:.3f}s at the sentence tail"
            )
    if failures:
        raise RuntimeError("Step5 mix audit failed: " + "; ".join(failures))
