"""Render target-language subtitles without covering subtitles in the source video.

This module is deliberately independent from ASR, diarization, translation,
TTS, and mixing.  It consumes their final artifacts after Step 5 and always
keeps the original, non-subtitled delivery intact.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
from typing import Any, Iterable, Mapping, Sequence


_OCR_LINE_RE = re.compile(
    r"<ref>(.*?)</ref>\s*(?:<quad>(.*?)</quad>)?",
    flags=re.IGNORECASE | re.DOTALL,
)
_OCR_PLAIN_COORD_RE = re.compile(
    r"(.*?)"
    r"(\(\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*\)\s*,\s*"
    r"\(\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*\))",
    flags=re.DOTALL,
)


def _number(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _segment_id(value: Any) -> str:
    return str(value if value is not None else "").strip()


def _normalized_box(raw: str, width: int, height: int) -> list[float] | None:
    numbers = [_number(item) for item in re.findall(r"-?\d+(?:\.\d+)?", raw or "")]
    if len(numbers) < 4:
        return None
    xs, ys = numbers[0::2], numbers[1::2]
    left, right, top, bottom = min(xs), max(xs), min(ys), max(ys)
    if right <= left or bottom <= top:
        return None
    maximum = max(abs(value) for value in (left, right, top, bottom))
    if maximum <= 1.5:
        scale_x = scale_y = 1.0
    elif maximum <= 1100.0:
        scale_x = scale_y = 1000.0
    else:
        scale_x, scale_y = max(1, width), max(1, height)
    box = [
        max(0.0, min(1.0, left / scale_x)),
        max(0.0, min(1.0, top / scale_y)),
        max(0.0, min(1.0, right / scale_x)),
        max(0.0, min(1.0, bottom / scale_y)),
    ]
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box


def _subtitle_candidate_reason(text: str, box: Sequence[float]) -> str:
    compact = re.sub(r"\s+", "", text or "")
    if not compact or len(compact) > 180:
        return "empty_or_too_long"
    if re.search(r"(?:https?://|www\.)", compact, flags=re.IGNORECASE):
        return "url"
    if re.fullmatch(r"[\W_]+", compact, flags=re.UNICODE):
        return "symbols_only"
    left, top, right, bottom = (float(value) for value in box)
    center_x = (left + right) / 2.0
    center_y = (top + bottom) / 2.0
    box_width, box_height = right - left, bottom - top
    if center_y < 0.70 or bottom < 0.74:
        return "outside_bottom_subtitle_band"
    if not 0.22 <= center_x <= 0.78:
        return "not_horizontally_centered"
    if not 0.025 <= box_width <= 0.98:
        return "implausible_width"
    if not 0.008 <= box_height <= 0.18:
        return "implausible_height"
    return ""


def _looks_like_subtitle(text: str, box: Sequence[float]) -> bool:
    return not _subtitle_candidate_reason(text, box)


def _ocr_records(raw: str) -> list[tuple[str, str]]:
    """Return text/coordinate pairs from both supported OCR wire formats.

    HunyuanOCR deployments in this project have returned both tagged spotting
    output (``<ref>...<quad>...``) and compact output such as
    ``dialogue(312,888),(685,947)``.  Compact responses may concatenate
    several detections without newlines, so ``finditer`` must consume every
    coordinate pair independently.
    """
    tagged = [
        (match.group(1), match.group(2) or "")
        for match in _OCR_LINE_RE.finditer(raw or "")
        if match.group(2)
    ]
    if tagged:
        return tagged
    return [
        (match.group(1), match.group(2))
        for match in _OCR_PLAIN_COORD_RE.finditer(raw or "")
    ]


def read_ocr_subtitle_observations(
    cache_path: str | Path,
    width: int,
    height: int,
    diagnostics: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Extract likely on-screen subtitle regions from the existing OCR cache."""
    path = Path(cache_path)
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    rows = payload.values() if isinstance(payload, dict) else payload
    filter_audit = diagnostics if diagnostics is not None else {}
    filter_audit.update({
        "raw_coordinate_record_count": 0,
        "accepted_line_count": 0,
        "rejected_line_count": 0,
        "rejected_lines": [],
    })
    frames: dict[float, list[dict[str, Any]]] = {}
    for row in rows or []:
        if not isinstance(row, Mapping) or row.get("status") != "ok":
            continue
        timestamp = round(_number(row.get("timestamp"), -1.0), 3)
        if timestamp < 0:
            continue
        raw = str(row.get("text") or "")
        for raw_text, raw_box in _ocr_records(raw):
            text = re.sub(r"\s+", " ", raw_text).strip(" \t|-:")
            box = _normalized_box(raw_box, width, height)
            filter_audit["raw_coordinate_record_count"] += 1
            reason = "invalid_coordinates" if not box else _subtitle_candidate_reason(text, box)
            if box and not reason:
                filter_audit["accepted_line_count"] += 1
                frames.setdefault(timestamp, []).append({"text": text, "bbox": box})
            else:
                filter_audit["rejected_line_count"] += 1
                if len(filter_audit["rejected_lines"]) < 100:
                    filter_audit["rejected_lines"].append({
                        "timestamp": timestamp,
                        "text": text,
                        "bbox": box,
                        "reason": reason,
                    })

    observations: list[dict[str, Any]] = []
    for timestamp, lines in sorted(frames.items()):
        # Keep only the bottom-most vertically adjacent group.  This preserves
        # two-line/bilingual subtitles without letting an unrelated price tag
        # higher in the frame pull the union toward a person's face.
        bottom_anchor = max(line["bbox"][3] for line in lines)
        lines = [
            line for line in lines
            if bottom_anchor - float(line["bbox"][3]) <= 0.10
        ]
        left = min(line["bbox"][0] for line in lines)
        top = min(line["bbox"][1] for line in lines)
        right = max(line["bbox"][2] for line in lines)
        bottom = max(line["bbox"][3] for line in lines)
        observations.append({
            "timestamp": timestamp,
            "bbox": [left, top, right, bottom],
            "texts": [line["text"] for line in lines],
        })
    filter_audit["accepted_frame_count"] = len(observations)
    return observations


def _percentile(values: Sequence[float], fraction: float, default: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return float(default)
    position = max(0.0, min(1.0, float(fraction))) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def select_stable_subtitle_track(
    observations: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Choose one persistent bottom subtitle track and reject spatial outliers."""
    candidates = [dict(row) for row in observations if row.get("bbox")]
    if not candidates:
        return [], {
            "status": "missing",
            "input_observation_count": len(observations),
            "selected_observation_count": 0,
            "rejected_observation_count": len(observations),
            "confidence": 0.0,
        }
    bottoms = [float(row["bbox"][3]) for row in candidates]
    best_anchor = max(
        bottoms,
        key=lambda anchor: (
            sum(abs(value - anchor) <= 0.055 for value in bottoms),
            anchor,
        ),
    )
    initial = [
        row for row in candidates
        if abs(float(row["bbox"][3]) - best_anchor) <= 0.055
    ]
    median_bottom = _median(
        [float(row["bbox"][3]) for row in initial], best_anchor
    )
    deviations = [
        abs(float(row["bbox"][3]) - median_bottom) for row in initial
    ]
    mad = _median(deviations, 0.0)
    tolerance = max(0.025, min(0.06, 3.5 * mad if mad > 0 else 0.035))
    selected = [
        row for row in candidates
        if abs(float(row["bbox"][3]) - median_bottom) <= tolerance
    ]
    tops = [float(row["bbox"][1]) for row in selected]
    selected_bottoms = [float(row["bbox"][3]) for row in selected]
    stable_top = _percentile(tops, 0.15, 0.80)
    stable_bottom = _median(selected_bottoms, median_bottom)
    top_spread = (
        _percentile(tops, 0.90, stable_top)
        - _percentile(tops, 0.10, stable_top)
    )
    support = len(selected) / max(1, len(candidates))
    stability = max(0.0, 1.0 - top_spread / 0.18)
    confidence = max(0.0, min(1.0, support * 0.7 + stability * 0.3))
    rejected = [
        row for row in candidates
        if abs(float(row["bbox"][3]) - median_bottom) > tolerance
    ]
    return selected, {
        "status": "selected",
        "input_observation_count": len(observations),
        "selected_observation_count": len(selected),
        "rejected_observation_count": len(rejected),
        "stable_top": round(stable_top, 6),
        "stable_bottom": round(stable_bottom, 6),
        "bottom_tolerance": round(tolerance, 6),
        "top_spread": round(top_spread, 6),
        "confidence": round(confidence, 6),
        "rejected_observations": rejected[:100],
    }


def _result_map(results: Any) -> dict[str, Mapping[str, Any]]:
    if isinstance(results, Mapping):
        iterable = results.items()
    else:
        iterable = (
            (_segment_id(row.get("segment_id", row.get("id"))), row)
            for row in (results or []) if isinstance(row, Mapping)
        )
    return {
        _segment_id(key): value for key, value in iterable
        if _segment_id(key) and isinstance(value, Mapping)
    }


def build_subtitle_events(
    segments: Iterable[Mapping[str, Any]],
    tts_results: Any,
    mix_audit: Iterable[Mapping[str, Any]],
    video_duration: float,
) -> list[dict[str, Any]]:
    """Bind subtitle text to the audio that was actually placed in Step 5."""
    results = _result_map(tts_results)
    audit = {
        _segment_id(row.get("segment_id")): row
        for row in (mix_audit or []) if isinstance(row, Mapping)
    }
    duration = max(0.0, _number(video_duration))
    events: list[dict[str, Any]] = []
    for index, segment in enumerate(segments or []):
        sid = _segment_id(segment.get("id", segment.get("segment_id", index)))
        timing = audit.get(sid)
        # No mixed audio means no target subtitle.  This prevents a translation
        # failure from being presented as if the line had been dubbed.
        if timing is None:
            continue
        result = results.get(sid, {})
        if result and str(result.get("status") or "success") != "success":
            continue
        text = str(result.get("tts_text") or segment.get("text") or "").strip()
        if not text:
            continue
        start = max(0.0, _number(timing.get("start"), _number(segment.get("start"))))
        end = _number(timing.get("audible_end"), _number(segment.get("end"), start))
        end = min(duration or end, max(start + 0.45, end))
        if end <= start:
            continue
        events.append({
            "segment_id": sid,
            "start": round(start, 3),
            "end": round(end, 3),
            "text": text,
            "text_source": "tts_result" if result.get("tts_text") else "step3",
            "timing_source": "step5_mix_audit",
            # These fields describe overlap in the source/mix.  The subtitle
            # timeline uses them to distinguish a real two-person overlap
            # from an accidental overlap caused by subtitle display windows.
            "source_overlap_evidence": timing.get(
                "source_overlap_evidence", "none"
            ),
            "allowed_overlap_seconds": round(
                max(0.0, _number(timing.get("allowed_overlap_seconds"))), 3
            ),
            "mixed_overlap_seconds": round(
                max(0.0, _number(timing.get("mixed_overlap_seconds"))), 3
            ),
        })
    return sorted(events, key=lambda row: (row["start"], row["end"]))


def _median(values: Sequence[float], default: float) -> float:
    return float(statistics.median(values)) if values else float(default)


def _uses_cjk(text: str, target_language: str) -> bool:
    return target_language.lower().startswith(("zh", "ja", "ko")) or bool(
        re.search(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]", text)
    )


def _text_weight(text: str) -> int:
    return max(1, len(re.sub(r"[\s\W_]+", "", text, flags=re.UNICODE)))


def _natural_subtitle_chunks(text: str, target_language: str) -> list[str]:
    """Split a long line at linguistic boundaries before timing is assigned."""
    clean = re.sub(r"\s+", " ", str(text or "")).strip()
    if not clean:
        return []
    chunks: list[str] = []
    if _uses_cjk(clean, target_language):
        compact = clean.replace(" ", "")
        buffer = ""
        for char in compact:
            buffer += char
            natural_break = char in "，。！？；：、,.!?;:" and len(buffer) >= 8
            if natural_break or len(buffer) >= 18:
                chunks.append(buffer)
                buffer = ""
        if buffer:
            if chunks and len(buffer) < 5:
                chunks[-1] += buffer
            else:
                chunks.append(buffer)
    else:
        # Prefer complete clauses.  Only split inside a clause when it cannot
        # fit comfortably as one card; this avoids breaks such as
        # "the sky was still | bright" immediately before a comma.
        clauses = [
            match.group(0).strip()
            for match in re.finditer(
                r".+?(?:[,;:.!?]+[\"'’”)]*(?=\s|$)|$)", clean
            )
            if match.group(0).strip()
        ]
        for clause in clauses or [clean]:
            words = clause.split()
            if len(words) <= 10 and len(clause) <= 58:
                chunks.append(clause)
                continue
            cursor = 0
            while cursor < len(words):
                remaining = len(words) - cursor
                take = remaining if remaining <= 10 else min(9, math.ceil(remaining / 2))
                chunks.append(" ".join(words[cursor:cursor + take]))
                cursor += take
        if len(chunks) > 1 and len(chunks[-1].split()) < 3:
            chunks[-2] += " " + chunks[-1]
            chunks.pop()
    return chunks or [clean]


def _merge_chunks_to_limit(chunks: Sequence[str], limit: int) -> list[str]:
    merged = [str(chunk).strip() for chunk in chunks if str(chunk).strip()]
    limit = max(1, int(limit))
    while len(merged) > limit:
        index = min(
            range(len(merged) - 1),
            key=lambda item: _text_weight(merged[item]) + _text_weight(merged[item + 1]),
        )
        separator = "" if _uses_cjk(merged[index] + merged[index + 1], "") else " "
        merged[index:index + 2] = [merged[index] + separator + merged[index + 1]]
    return merged


def split_long_subtitle_events(
    events: Sequence[Mapping[str, Any]],
    target_language: str,
    *,
    minimum_card_seconds: float = 0.85,
) -> list[dict[str, Any]]:
    """Turn long utterances into readable, consecutive subtitle cards."""
    output: list[dict[str, Any]] = []
    minimum = max(0.5, float(minimum_card_seconds))
    for event in events:
        start, end = float(event["start"]), float(event["end"])
        total = max(0.0, end - start)
        chunks = _natural_subtitle_chunks(str(event.get("text") or ""), target_language)
        maximum_cards = max(1, int(total / minimum + 1e-9))
        chunks = _merge_chunks_to_limit(chunks, maximum_cards)
        if len(chunks) <= 1:
            output.append({
                **dict(event),
                "subtitle_chunk_index": 1,
                "subtitle_chunk_count": 1,
            })
            continue
        floor = min(minimum, total / len(chunks))
        remaining = max(0.0, total - floor * len(chunks))
        weights = [_text_weight(chunk) for chunk in chunks]
        weight_total = sum(weights)
        cursor = start
        for index, (chunk, weight) in enumerate(zip(chunks, weights), start=1):
            chunk_end = (
                end if index == len(chunks)
                else cursor + floor + remaining * weight / weight_total
            )
            output.append({
                **dict(event),
                "start": round(cursor, 3),
                "end": round(chunk_end, 3),
                "text": chunk,
                "source_full_text": str(event.get("text") or ""),
                "subtitle_chunk_index": index,
                "subtitle_chunk_count": len(chunks),
            })
            cursor = chunk_end
    return output


def _has_source_overlap(event: Mapping[str, Any]) -> bool:
    if _number(event.get("allowed_overlap_seconds")) > 0.03:
        return True
    evidence = str(event.get("source_overlap_evidence") or "").strip().lower()
    return evidence not in {"", "none", "false", "0", "unknown", "not_applicable"}


def resolve_subtitle_timeline(
    events: Sequence[Mapping[str, Any]],
    *,
    minimum_display_seconds: float = 0.35,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Create a non-overlapping display timeline without changing audio.

    Accidental overlaps hand off at one shared boundary.  Overlaps backed by
    source/mix evidence are represented as one multi-line card during the
    shared interval, so two independent ASS events can never cover each other.
    """
    rows = sorted(
        (dict(row) for row in events if float(row["end"]) > float(row["start"])),
        key=lambda row: (float(row["start"]), float(row["end"])),
    )
    minimum = max(0.1, float(minimum_display_seconds))
    adjustments: list[dict[str, Any]] = []
    artificial_count = 0
    for index in range(1, len(rows)):
        previous, current = rows[index - 1], rows[index]
        if float(previous["end"]) <= float(current["start"]):
            continue
        # The mix audit stores the overlap budget on the earlier clip, where
        # it describes that clip's relationship with its following clip.
        # The current clip's value belongs to the next pair and must not make
        # this boundary look like a genuine source overlap.
        if _has_source_overlap(previous):
            continue
        old_previous_end = float(previous["end"])
        old_current_start = float(current["start"])
        boundary = (old_previous_end + old_current_start) / 2.0
        lower = float(previous["start"]) + minimum
        upper = float(current["end"]) - minimum
        if lower <= upper:
            boundary = max(lower, min(upper, boundary))
        else:
            boundary = max(float(previous["start"]), min(float(current["end"]), boundary))
        previous["end"] = round(boundary, 3)
        current["start"] = round(boundary, 3)
        previous["timeline_resolution"] = "artificial_overlap_handoff"
        current["timeline_resolution"] = "artificial_overlap_handoff"
        artificial_count += 1
        adjustments.append({
            "kind": "artificial_overlap_handoff",
            "previous_segment_id": previous.get("segment_id"),
            "current_segment_id": current.get("segment_id"),
            "previous_end_before": round(old_previous_end, 3),
            "current_start_before": round(old_current_start, 3),
            "handoff": round(boundary, 3),
        })

    boundaries = sorted({
        round(float(row[key]), 6)
        for row in rows for key in ("start", "end")
    })
    resolved: list[dict[str, Any]] = []
    genuine_intervals = 0
    for start, end in zip(boundaries, boundaries[1:]):
        if end <= start:
            continue
        midpoint = (start + end) / 2.0
        active = [
            row for row in rows
            if float(row["start"]) <= midpoint < float(row["end"])
        ]
        if not active:
            continue
        if len(active) == 1:
            card = dict(active[0])
            signature = (str(card.get("segment_id")), str(card.get("text") or ""))
        else:
            texts: list[str] = []
            segment_ids: list[str] = []
            for row in active:
                text = str(row.get("text") or "").strip()
                sid = str(row.get("segment_id") or "")
                if text and text not in texts:
                    texts.append(text)
                if sid and sid not in segment_ids:
                    segment_ids.append(sid)
            card = dict(active[0])
            card.update({
                "segment_id": "+".join(segment_ids),
                "combined_segment_ids": segment_ids,
                "text": "\n".join(texts),
                "timeline_resolution": "combined_source_overlap",
            })
            signature = (tuple(segment_ids), tuple(texts))
            genuine_intervals += 1
        card["start"] = round(start, 3)
        card["end"] = round(end, 3)
        card["_timeline_signature"] = signature
        if (
            resolved
            and resolved[-1].get("_timeline_signature") == signature
            and abs(float(resolved[-1]["end"]) - start) < 0.002
        ):
            resolved[-1]["end"] = round(end, 3)
        else:
            resolved.append(card)
    for row in resolved:
        row.pop("_timeline_signature", None)
    return resolved, {
        "input_event_count": len(events),
        "output_event_count": len(resolved),
        "artificial_overlap_handoff_count": artificial_count,
        "source_overlap_combined_interval_count": genuine_intervals,
        "adjustments": adjustments,
        "events_are_disjoint": all(
            float(resolved[index - 1]["end"]) <= float(resolved[index]["start"]) + 0.001
            for index in range(1, len(resolved))
        ),
    }


def _wrap_two_lines(text: str, target_language: str) -> str:
    clean = re.sub(r"\s+", " ", text).strip()
    if not clean:
        return ""
    is_cjk = _uses_cjk(clean, target_language)
    if is_cjk:
        compact = clean.replace(" ", "")
        if len(compact) <= 24:
            return compact
        midpoint = len(compact) // 2
        punctuation = "，。！？；：、,.!?;:"
        candidates = [
            index + 1 for index, char in enumerate(compact)
            if char in punctuation and 5 <= index <= len(compact) - 6
        ]
        split_at = min(candidates, key=lambda value: abs(value - midpoint)) if candidates else midpoint
        return compact[:split_at] + "\n" + compact[split_at:]
    words = clean.split()
    if len(clean) <= 42 or len(words) <= 5:
        return clean
    best = min(
        range(1, len(words)),
        key=lambda index: abs(
            len(" ".join(words[:index])) - len(" ".join(words[index:]))
        ),
    )
    return " ".join(words[:best]) + "\n" + " ".join(words[best:])


def plan_subtitle_layout(
    events: Sequence[Mapping[str, Any]],
    observations: Sequence[Mapping[str, Any]],
    width: int,
    height: int,
    *,
    target_language: str = "zh",
    preferred_font_size: int = 0,
) -> dict[str, Any]:
    """Plan a stable lower-screen placement above the existing subtitle."""
    width, height = max(1, int(width)), max(1, int(height))
    requested_font = int(preferred_font_size or 0)
    default_font = (
        requested_font if requested_font > 0
        else max(30, int(round(height * 0.063)))
    )
    minimum_font = max(24, int(round(default_font * 0.78)))
    default_gap = max(8, int(round(height * 0.018)))
    minimum_gap = max(5, int(round(height * 0.008)))
    safe_top = max(12, int(round(height * 0.035)))
    stable_track, track_audit = select_stable_subtitle_track(observations)
    stable_top = float(track_audit.get("stable_top", 0.76))
    paginated_events = split_long_subtitle_events(events, target_language)
    display_events, timeline_audit = resolve_subtitle_timeline(paginated_events)
    candidates: list[dict[str, Any]] = []
    for event in display_events:
        source = "stable_global_ocr_track" if stable_track else "missing_ocr_geometry"
        if event.get("timeline_resolution") == "combined_source_overlap":
            # Each source utterance gets its own line in the single shared
            # card; do not collapse the explicit line break during wrapping.
            render_text = str(event.get("text") or "").strip()
        else:
            render_text = _wrap_two_lines(
                str(event.get("text") or ""), target_language
            )
        candidates.append({
            **dict(event),
            "original_subtitle_top": round(stable_top, 6),
            "placement_source": source,
            "render_text": render_text,
            "line_count": max(1, render_text.count("\n") + 1),
        })

    # Use one global baseline.  Per-event OCR placement made subtitles jump
    # when OCR briefly detected a sign, price label, or face-adjacent text.
    font_size, gap = default_font, default_gap
    def fits(candidate: Mapping[str, Any], font: int, spacing: int) -> bool:
        bottom = float(candidate["original_subtitle_top"]) * height - spacing
        block = int(candidate["line_count"]) * font * 1.28
        return bottom - block >= safe_top

    if any(not fits(row, font_size, gap) for row in candidates):
        font_size, gap = minimum_font, minimum_gap
    expanded = (
        bool(candidates) and not stable_track
    ) or any(not fits(row, font_size, gap) for row in candidates)
    band_height = 0
    output_height = height
    if expanded:
        band_height = max(int(round(height * 0.16)), int(font_size * 2.9))
        output_height = height + band_height

    planned: list[dict[str, Any]] = []
    fixed_boundary = stable_top * height - gap
    for candidate in candidates:
        if expanded:
            baseline = height + int(round(band_height * 0.78))
            placement = "expanded_bottom_canvas"
            collision_boundary = None
        else:
            collision_boundary = fixed_boundary
            baseline = fixed_boundary
            placement = candidate["placement_source"]
        row = dict(candidate)
        row.update({
            "x": round(width / 2.0, 2),
            "y": round(baseline, 2),
            "font_size": font_size,
            "gap_pixels": 0 if expanded else gap,
            "placement_source": placement,
            "collision_boundary_y": (
                round(collision_boundary, 2) if collision_boundary is not None else None
            ),
            "overlaps_existing_subtitle": False,
        })
        planned.append(row)
    return {
        "mode": "expanded_bottom_canvas" if expanded else "above_existing_subtitle",
        "input_width": width,
        "input_height": height,
        "output_width": width,
        "output_height": output_height,
        "bottom_canvas_extension": band_height,
        "font_size": font_size,
        "events": planned,
        "source_event_count": len(events),
        "display_event_count": len(planned),
        "ocr_observation_count": len(observations),
        "selected_track_observation_count": len(stable_track),
        "stable_target_baseline_y": (
            round(planned[0]["y"], 2) if planned else None
        ),
        "position_is_globally_fixed": True,
        "subtitle_track": track_audit,
        "timeline_audit": timeline_audit,
        "spatial_no_overlap_guaranteed": bool(stable_track) or expanded,
        "timeline_no_overlap_guaranteed": bool(
            timeline_audit.get("events_are_disjoint")
        ),
        "no_overlap_guaranteed": (
            (bool(stable_track) or expanded)
            and bool(timeline_audit.get("events_are_disjoint"))
        ),
    }


def _ass_time(seconds: float) -> str:
    centiseconds = max(0, int(round(float(seconds) * 100)))
    hours, remainder = divmod(centiseconds, 360000)
    minutes, remainder = divmod(remainder, 6000)
    secs, cs = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{cs:02d}"


def _ass_text(text: str) -> str:
    clean = str(text or "").replace("\\", "\\\\")
    clean = clean.replace("{", "\\{").replace("}", "\\}")
    return re.sub(r"\s*\n\s*", r"\\N", clean.strip())


def write_ass(
    path: str | Path,
    layout: Mapping[str, Any],
    *,
    font_name: str = "Noto Sans CJK SC",
) -> None:
    width = int(layout["output_width"])
    height = int(layout["output_height"])
    font_size = int(layout["font_size"])
    outline = round(max(2.8, font_size * 0.052), 1)
    shadow = round(max(0.8, font_size * 0.015), 1)
    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        f"PlayResX: {width}",
        f"PlayResY: {height}",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Target,{font_name},{font_size},&H00FFFFFF,&H000000FF,"
        f"&H00000000,&H70000000,0,0,0,0,100,100,0,0,1,{outline},{shadow},2,24,24,12,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    for event in layout.get("events", []):
        override = (
            "{\\an2\\q2"
            f"\\pos({float(event['x']):.2f},{float(event['y']):.2f})"
            f"\\fs{int(event['font_size'])}" + "}"
        )
        lines.append(
            "Dialogue: 0,"
            f"{_ass_time(event['start'])},{_ass_time(event['end'])},"
            f"Target,,0,0,0,,{override}{_ass_text(event.get('render_text') or event['text'])}"
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def probe_video_size(video_path: str | Path) -> tuple[int, int]:
    completed = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", "-of", "json",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    streams = json.loads(completed.stdout).get("streams") or []
    if not streams:
        raise RuntimeError("ffprobe did not return a video stream")
    return int(streams[0]["width"]), int(streams[0]["height"])


def render_target_subtitles(
    *,
    base_video: str | Path,
    output_dir: str | Path,
    project_name: str,
    temp_dir: str | Path,
    target_language: str,
    segments: Sequence[Mapping[str, Any]],
    tts_results: Any,
    mix_audit: Sequence[Mapping[str, Any]],
    video_duration: float,
    font_name: str = "Noto Sans CJK SC",
    preferred_font_size: int = 0,
) -> dict[str, Any]:
    """Create a second, subtitled delivery and a machine-readable audit."""
    base = Path(base_video)
    if not base.exists():
        raise FileNotFoundError(f"base video does not exist: {base}")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    width, height = probe_video_size(base)
    cache_path = Path(temp_dir) / "ocr_optimized" / "ocr_frames.json"
    ocr_filter_diagnostics: dict[str, Any] = {}
    observations = read_ocr_subtitle_observations(
        cache_path,
        width,
        height,
        diagnostics=ocr_filter_diagnostics,
    )
    events = build_subtitle_events(segments, tts_results, mix_audit, video_duration)
    if not events:
        raise RuntimeError("no actually mixed target-language lines to subtitle")
    layout = plan_subtitle_layout(
        events,
        observations,
        width,
        height,
        target_language=target_language,
        preferred_font_size=preferred_font_size,
    )
    layout["ocr_filter_diagnostics"] = ocr_filter_diagnostics
    digest = hashlib.sha1(project_name.encode("utf-8")).hexdigest()[:8]
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", project_name).strip("_") or "project"
    stem = f"{safe_name}_{digest}_target_subtitles"
    ass_path = output / f"{stem}.ass"
    audit_path = output / f"{stem}_layout_audit.json"
    video_path = output / f"{safe_name}_final_video_subtitled.mp4"
    write_ass(ass_path, layout, font_name=font_name)
    filter_parts = []
    extension = int(layout["bottom_canvas_extension"])
    if extension:
        filter_parts.append(f"pad=iw:ih+{extension}:0:0:black")
    # Run from the output directory so the ASS filter never receives a Windows
    # drive letter or a path containing filter-special characters.
    filter_parts.append(f"ass={ass_path.name}")
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(base.resolve()),
            "-vf", ",".join(filter_parts),
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-c:a", "copy", video_path.name,
        ],
        check=True,
        cwd=str(output),
    )
    audit = {
        **layout,
        "status": "success",
        "base_output_video": str(base),
        "subtitle_output_video": str(video_path),
        "subtitle_ass_path": str(ass_path),
        "ocr_cache_path": str(cache_path),
        "font_name": font_name,
    }
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "status": "success",
        "output_video": str(video_path),
        "base_output_video": str(base),
        "subtitle_ass_path": str(ass_path),
        "subtitle_layout_audit_path": str(audit_path),
        "subtitle_layout_mode": layout["mode"],
        "subtitle_event_count": len(layout.get("events") or []),
        "subtitle_source_event_count": len(events),
        "ocr_observation_count": len(observations),
    }
