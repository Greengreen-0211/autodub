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


def _segment_embedding(segment: Mapping[str, Any]) -> List[float]:
    values = segment.get("wespeaker_tts_embedding") or []
    if not isinstance(values, (list, tuple)):
        return []
    try:
        return _unit(values)
    except (TypeError, ValueError):
        return []


def _segment_interval(segment: Mapping[str, Any]) -> Tuple[float, float] | None:
    try:
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", start))
    except (TypeError, ValueError):
        return None
    return (start, end) if end > start else None


def _precise_reference_candidate(segment: Mapping[str, Any]) -> bool:
    return not (
        str(segment.get("timestamp_precision", "")).lower() == "coarse"
        or str(segment.get("asr_recovery_status", "")).lower()
        == "accepted_coarse"
        or bool(segment.get("speaker_boundary_ambiguous"))
        or bool(segment.get("speaker_reference_mixed_raw"))
    )


def build_local_continuity_groups(
    segments: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """Link only adjacent, strongly matching utterances for TTS continuity.

    This never changes diarization labels or sentence timestamps.  It gives the
    TTS layer a local identity key so an over-split speaker does not change
    reference voice and engine in the middle of one continuous turn.
    """
    cross_label_threshold = _env_float(
        "AUTODUB_TTS_LOCAL_CONTINUITY_SIMILARITY", 0.80
    )
    same_label_threshold = _env_float(
        "AUTODUB_TTS_LOCAL_CONTINUITY_SAME_LABEL_SIMILARITY", 0.70
    )
    maximum_gap = _env_float("AUTODUB_TTS_LOCAL_CONTINUITY_MAX_GAP", 0.45)
    ordered = sorted(
        [segment for segment in segments if _segment_interval(segment)],
        key=lambda row: _segment_interval(row) or (float("inf"), float("inf")),
    )
    parent: Dict[str, str] = {
        str(segment.get("id", index)): str(segment.get("id", index))
        for index, segment in enumerate(ordered)
    }

    def find(name: str) -> str:
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    edges: List[Dict[str, Any]] = []
    for left, right in zip(ordered, ordered[1:]):
        left_id = str(left.get("id"))
        right_id = str(right.get("id"))
        left_interval = _segment_interval(left)
        right_interval = _segment_interval(right)
        if left_interval is None or right_interval is None:
            continue
        gap = right_interval[0] - left_interval[1]
        if gap < -0.08 or gap > maximum_gap:
            continue
        similarity = cosine_similarity(
            _segment_embedding(left), _segment_embedding(right)
        )
        same_label = str(left.get("speaker")) == str(right.get("speaker"))
        threshold = same_label_threshold if same_label else cross_label_threshold
        accepted = similarity is not None and similarity >= threshold
        edges.append({
            "left": left_id,
            "right": right_id,
            "gap": round(gap, 6),
            "similarity": round(similarity, 6) if similarity is not None else None,
            "same_step2_label": same_label,
            "threshold": threshold,
            "accepted": accepted,
        })
        if accepted:
            union(left_id, right_id)

    members: Dict[str, List[str]] = defaultdict(list)
    for segment_id in parent:
        members[find(segment_id)].append(segment_id)
    identity_map: Dict[str, str] = {}
    groups = []
    for member_ids in members.values():
        if len(member_ids) < 2:
            continue
        ordered_ids = sorted(
            member_ids,
            key=lambda value: next(
                index for index, row in enumerate(ordered)
                if str(row.get("id")) == value
            ),
        )
        group_id = "local_continuity:" + ordered_ids[0]
        for segment_id in ordered_ids:
            identity_map[segment_id] = group_id
        groups.append({"group_id": group_id, "segment_ids": ordered_ids})
    return identity_map, {
        "policy": "adjacent_strong_wespeaker_similarity",
        "cross_label_similarity_threshold": cross_label_threshold,
        "same_label_similarity_threshold": same_label_threshold,
        "maximum_gap_seconds": maximum_gap,
        "edges": edges,
        "groups": groups,
    }


def contaminated_reference_rescue_pairs(
    segments: Sequence[Mapping[str, Any]],
    reference_source_ids: Mapping[str, Any],
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """Pair mutually matching outliers that were put in a contaminated label.

    A segment is rescued only when it disagrees with its cluster's selected
    reference and has a high-margin mutual match in another precise utterance.
    The caller can then cross-reference the pair, avoiding both the polluted
    cluster reference and same-sentence reference leakage.
    """
    contamination_threshold = _env_float(
        "AUTODUB_TTS_REFERENCE_CONTAMINATION_SIMILARITY", 0.62
    )
    rescue_threshold = _env_float(
        "AUTODUB_TTS_REFERENCE_RESCUE_SIMILARITY", 0.78
    )
    minimum_margin = _env_float("AUTODUB_TTS_REFERENCE_RESCUE_MARGIN", 0.03)
    rows = {
        str(segment.get("id", index)): segment
        for index, segment in enumerate(segments)
        if (
            _segment_embedding(segment)
            and _segment_interval(segment)
            and _precise_reference_candidate(segment)
        )
    }
    contaminated: Dict[str, Dict[str, Any]] = {}
    for segment_id, segment in rows.items():
        source_id = reference_source_ids.get(str(segment.get("speaker")))
        source = rows.get(str(source_id))
        similarity = (
            cosine_similarity(
                _segment_embedding(segment), _segment_embedding(source)
            )
            if source is not None else None
        )
        if similarity is not None and similarity < contamination_threshold:
            contaminated[segment_id] = {
                "speaker": str(segment.get("speaker")),
                "selected_reference_segment_id": str(source_id),
                "reference_similarity": round(similarity, 6),
            }

    nearest: Dict[str, Tuple[float, str, float]] = {}
    for segment_id in contaminated:
        segment = rows[segment_id]
        interval = _segment_interval(segment)
        ranked = []
        for other_id, other in rows.items():
            if other_id == segment_id:
                continue
            other_interval = _segment_interval(other)
            if interval is None or other_interval is None:
                continue
            if min(interval[1], other_interval[1]) - max(
                interval[0], other_interval[0]
            ) > 0.08:
                continue
            similarity = cosine_similarity(
                _segment_embedding(segment), _segment_embedding(other)
            )
            if similarity is not None:
                ranked.append((similarity, other_id))
        ranked.sort(reverse=True)
        if not ranked:
            continue
        best_similarity, best_id = ranked[0]
        second_similarity = ranked[1][0] if len(ranked) > 1 else -1.0
        nearest[segment_id] = (
            best_similarity, best_id, best_similarity - second_similarity
        )

    rescue: Dict[str, str] = {}
    decisions = []
    for segment_id, (similarity, partner_id, margin) in nearest.items():
        partner = nearest.get(partner_id)
        mutual = partner is not None and partner[1] == segment_id
        partner_margin = partner[2] if partner is not None else -1.0
        accepted = (
            partner_id in contaminated
            and mutual
            and similarity >= rescue_threshold
            and margin >= minimum_margin
            and partner_margin >= minimum_margin
        )
        decisions.append({
            "segment_id": segment_id,
            "partner_segment_id": partner_id,
            "similarity": round(similarity, 6),
            "margin": round(margin, 6),
            "mutual_nearest": mutual,
            "accepted": accepted,
        })
        if accepted:
            rescue[segment_id] = partner_id
    return rescue, {
        "policy": "cluster_reference_outlier_mutual_segment_rescue",
        "contamination_similarity_threshold": contamination_threshold,
        "rescue_similarity_threshold": rescue_threshold,
        "minimum_margin": minimum_margin,
        "contaminated_segments": contaminated,
        "decisions": decisions,
        "rescue_partner_ids": rescue,
    }


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
