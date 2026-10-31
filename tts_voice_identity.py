"""Conservative TTS-only voice identity reconciliation.

Step 2 speaker labels remain authoritative for text, timing, and diagnostics.
This module only decides whether two over-split labels may reuse one cloning
reference and one TTS engine.  It consumes WeSpeaker embeddings exported by
Step 2 and deliberately refuses transitive, low-margin, or overlapping merges.
"""

from __future__ import annotations

import math
import os
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return float(default)


def _unit(values: Iterable[Any]) -> List[float]:
    vector = [float(value) for value in values]
    norm = math.sqrt(sum(value * value for value in vector))
    if not vector or norm <= 1e-12:
        return []
    return [value / norm for value in vector]


def cosine_similarity(left: Sequence[Any], right: Sequence[Any]) -> float | None:
    a = _unit(left)
    b = _unit(right)
    if not a or len(a) != len(b):
        return None
    return float(sum(x * y for x, y in zip(a, b)))


def build_voice_profiles(
    segments: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[Tuple[float, float]]]]:
    """Pool reliable per-segment WeSpeaker vectors by immutable Step 2 label."""
    grouped: Dict[str, List[Tuple[List[float], float, str, float]]] = defaultdict(list)
    intervals: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for index, segment in enumerate(segments):
        speaker = str(segment.get("speaker", "SPEAKER_00"))
        try:
            start = float(segment.get("start", 0.0))
            end = float(segment.get("end", start))
        except (TypeError, ValueError):
            continue
        if end > start:
            intervals[speaker].append((start, end))
        vector = segment.get("wespeaker_tts_embedding") or []
        if not isinstance(vector, (list, tuple)) or not vector:
            continue
        try:
            normalized = _unit(vector)
            confidence = float(
                segment.get("speaker_assignment_confidence", 1.0) or 0.0
            )
            cohesion = float(
                segment.get("wespeaker_tts_embedding_cohesion", 1.0) or 0.0
            )
            window_count = max(
                1, int(segment.get("wespeaker_tts_embedding_windows", 1) or 1)
            )
        except (TypeError, ValueError):
            continue
        if not normalized:
            continue
        # Ambiguous boundaries remain useful as weak evidence only.  They can
        # never outweigh a clean, high-confidence utterance.
        ambiguity_weight = 0.35 if segment.get("speaker_boundary_ambiguous") else 1.0
        weight = (
            max(0.10, confidence)
            * max(0.20, cohesion)
            * math.sqrt(window_count)
            * ambiguity_weight
        )
        grouped[speaker].append((
            normalized,
            weight,
            str(segment.get("id", index)),
            max(0.0, end - start),
        ))

    profiles: Dict[str, Dict[str, Any]] = {}
    for speaker, rows in grouped.items():
        dimension = len(rows[0][0])
        usable = [row for row in rows if len(row[0]) == dimension]
        if not usable:
            continue
        total_weight = sum(row[1] for row in usable)
        pooled = _unit(
            sum(row[0][axis] * row[1] for row in usable) / total_weight
            for axis in range(dimension)
        )
        if not pooled:
            continue
        similarities = [
            cosine_similarity(row[0], pooled) or 0.0 for row in usable
        ]
        profiles[speaker] = {
            "embedding": pooled,
            "support_segment_ids": [row[2] for row in usable],
            "support_segment_count": len(usable),
            "support_duration": round(sum(row[3] for row in usable), 4),
            "cohesion": round(
                sum(value * row[1] for value, row in zip(similarities, usable))
                / total_weight,
                6,
            ),
        }
    return profiles, dict(intervals)


def _has_temporal_overlap(
    left: Sequence[Tuple[float, float]],
    right: Sequence[Tuple[float, float]],
    tolerance: float,
) -> bool:
    for left_start, left_end in left:
        for right_start, right_end in right:
            if min(left_end, right_end) - max(left_start, right_start) > tolerance:
                return True
    return False


def _reference_rank(reference: Mapping[str, Any], profile: Mapping[str, Any]) -> tuple:
    reliability = int(str(reference.get("reference_reliability", "")) == "reliable")
    try:
        purity = float(reference.get("purity_score") or 0.0)
    except (TypeError, ValueError):
        purity = 0.0
    try:
        duration = float(reference.get("duration") or 0.0)
    except (TypeError, ValueError):
        duration = 0.0
    return (
        reliability,
        purity,
        float(profile.get("cohesion") or 0.0),
        float(profile.get("support_duration") or 0.0),
        min(duration, 8.0),
    )


def conservative_identity_map(
    profiles: Mapping[str, Mapping[str, Any]],
    intervals: Mapping[str, Sequence[Tuple[float, float]]],
    references: Mapping[str, Mapping[str, Any]],
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """Return a non-transitive, mutual-nearest TTS reference reuse map."""
    threshold = _env_float("AUTODUB_TTS_WESPEAKER_IDENTITY_SIMILARITY", 0.78)
    margin = _env_float("AUTODUB_TTS_WESPEAKER_IDENTITY_MARGIN", 0.06)
    overlap_tolerance = _env_float(
        "AUTODUB_TTS_IDENTITY_OVERLAP_TOLERANCE_SECONDS", 0.08
    )
    names = sorted(name for name in profiles if name in references)
    similarities: List[Dict[str, Any]] = []
    allowed: Dict[str, List[Tuple[float, str]]] = defaultdict(list)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            similarity = cosine_similarity(
                profiles[left].get("embedding") or [],
                profiles[right].get("embedding") or [],
            )
            if similarity is None:
                continue
            overlap = _has_temporal_overlap(
                intervals.get(left, []), intervals.get(right, []), overlap_tolerance
            )
            row = {
                "left": left,
                "right": right,
                "similarity": round(similarity, 6),
                "temporal_overlap": overlap,
                "mutual_nearest": False,
                "margin_passed": False,
                "proposed_merge": False,
            }
            similarities.append(row)
            if not overlap:
                allowed[left].append((similarity, right))
                allowed[right].append((similarity, left))

    nearest: Dict[str, Tuple[float, str, float]] = {}
    for name in names:
        ranked = sorted(allowed.get(name, []), reverse=True)
        if not ranked:
            continue
        best_similarity, best_name = ranked[0]
        second_similarity = ranked[1][0] if len(ranked) > 1 else -1.0
        nearest[name] = (best_similarity, best_name, best_similarity - second_similarity)

    identity_map = {name: name for name in names}
    merged_names = set()
    decisions = []
    for left in names:
        if left in merged_names or left not in nearest:
            continue
        similarity, right, left_margin = nearest[left]
        if right in merged_names or right not in nearest:
            continue
        right_similarity, right_best, right_margin = nearest[right]
        mutual = right_best == left
        margin_passed = left_margin >= margin and right_margin >= margin
        eligible = mutual and margin_passed and similarity >= threshold
        for row in similarities:
            if {row["left"], row["right"]} == {left, right}:
                row["mutual_nearest"] = mutual
                row["margin_passed"] = margin_passed
                row["proposed_merge"] = eligible
                break
        if not eligible:
            continue
        canonical = max(
            (left, right),
            key=lambda name: _reference_rank(references[name], profiles[name]),
        )
        fragment = right if canonical == left else left
        identity_map[fragment] = canonical
        merged_names.update((left, right))
        decisions.append({
            "fragment": fragment,
            "canonical": canonical,
            "similarity": round(similarity, 6),
            "fragment_margin": round(
                left_margin if fragment == left else right_margin, 6
            ),
            "canonical_margin": round(
                right_margin if fragment == left else left_margin, 6
            ),
            "reason": "mutual_nearest_wespeaker_no_temporal_overlap",
        })

    diagnostics = {
        "backend": "wespeaker_precomputed_segment_embeddings",
        "similarity_threshold": threshold,
        "minimum_margin": margin,
        "overlap_tolerance_seconds": overlap_tolerance,
        "profile_count": len(profiles),
        "profiles": {
            name: {
                key: value for key, value in profile.items() if key != "embedding"
            }
            for name, profile in profiles.items()
        },
        "similarities": similarities,
        "proposed_identity_map": identity_map,
        "merge_decisions": decisions,
    }
    return identity_map, diagnostics

