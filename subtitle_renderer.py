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


def _looks_like_subtitle(text: str, box: Sequence[float]) -> bool:
    compact = re.sub(r"\s+", "", text or "")
    if not compact or len(compact) > 180:
        return False
    if re.search(r"(?:https?://|www\.)", compact, flags=re.IGNORECASE):
        return False
    if re.fullmatch(r"[\W_]+", compact, flags=re.UNICODE):
        return False
    left, top, right, bottom = (float(value) for value in box)
    center_x = (left + right) / 2.0
    center_y = (top + bottom) / 2.0
    box_width, box_height = right - left, bottom - top
    return (
        center_y >= 0.45
        and 0.18 <= center_x <= 0.82
        and 0.035 <= box_width <= 0.98
        and 0.008 <= box_height <= 0.18
    )


def read_ocr_subtitle_observations(
    cache_path: str | Path,
    width: int,
    height: int,
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
    frames: dict[float, list[dict[str, Any]]] = {}
    for row in rows or []:
        if not isinstance(row, Mapping) or row.get("status") != "ok":
            continue
        timestamp = round(_number(row.get("timestamp"), -1.0), 3)
        if timestamp < 0:
            continue
        raw = str(row.get("text") or "")
        for match in _OCR_LINE_RE.finditer(raw):
            text = re.sub(r"\s+", " ", match.group(1)).strip()
            box = _normalized_box(match.group(2) or "", width, height)
            if box and _looks_like_subtitle(text, box):
                frames.setdefault(timestamp, []).append({"text": text, "bbox": box})

    observations: list[dict[str, Any]] = []
    for timestamp, lines in sorted(frames.items()):
        # A bilingual subtitle may occupy two adjacent rows.  Avoid the whole
        # card, not merely one line, by taking their union.
        left = min(line["bbox"][0] for line in lines)
        top = min(line["bbox"][1] for line in lines)
        right = max(line["bbox"][2] for line in lines)
        bottom = max(line["bbox"][3] for line in lines)
        observations.append({
            "timestamp": timestamp,
            "bbox": [left, top, right, bottom],
            "texts": [line["text"] for line in lines],
        })
    return observations


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
        })
    return sorted(events, key=lambda row: (row["start"], row["end"]))


def _median(values: Sequence[float], default: float) -> float:
    return float(statistics.median(values)) if values else float(default)


def _wrap_two_lines(text: str, target_language: str) -> str:
    clean = re.sub(r"\s+", " ", text).strip()
    if not clean:
        return ""
    is_cjk = target_language.lower().startswith(("zh", "ja", "ko")) or bool(
        re.search(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]", clean)
    )
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
) -> dict[str, Any]:
    """Plan a stable lower-screen placement above the existing subtitle."""
    width, height = max(1, int(width)), max(1, int(height))
    default_font = max(24, int(round(height * 0.045)))
    minimum_font = max(20, int(round(height * 0.034)))
    default_gap = max(8, int(round(height * 0.018)))
    minimum_gap = max(5, int(round(height * 0.008)))
    safe_top = max(12, int(round(height * 0.035)))
    global_top = _median(
        [float(row["bbox"][1]) for row in observations if row.get("bbox")],
        0.80,
    )
    candidates: list[dict[str, Any]] = []
    for event in events:
        start, end = float(event["start"]), float(event["end"])
        nearby = [
            row for row in observations
            if start - 0.45 <= _number(row.get("timestamp")) <= end + 0.45
            and row.get("bbox")
        ]
        source = "overlapping_ocr"
        if not nearby and observations:
            midpoint = (start + end) / 2.0
            nearest = min(
                observations,
                key=lambda row: abs(_number(row.get("timestamp")) - midpoint),
            )
            if abs(_number(nearest.get("timestamp")) - midpoint) <= 4.0:
                nearby = [nearest]
                source = "nearby_ocr"
        if nearby:
            original_top = _median([float(row["bbox"][1]) for row in nearby], global_top)
        elif observations:
            original_top = global_top
            source = "global_ocr_median"
        else:
            # This is only a provisional boundary.  With no OCR geometry at
            # all, the final planner uses a separate bottom canvas rather than
            # pretending it can prove a collision-free in-frame placement.
            original_top = 0.76
            source = "missing_ocr_geometry"
        render_text = _wrap_two_lines(
            str(event.get("text") or ""), target_language
        )
        candidates.append({
            **dict(event),
            "original_subtitle_top": round(original_top, 6),
            "placement_source": source,
            "render_text": render_text,
            "line_count": 2 if "\n" in render_text else 1,
        })

    # Smooth OCR jitter, but never move a subtitle down across the current
    # frame's own collision boundary.
    font_size, gap = default_font, default_gap
    def fits(candidate: Mapping[str, Any], font: int, spacing: int) -> bool:
        bottom = float(candidate["original_subtitle_top"]) * height - spacing
        block = int(candidate["line_count"]) * font * 1.28
        return bottom - block >= safe_top

    if any(not fits(row, font_size, gap) for row in candidates):
        font_size, gap = minimum_font, minimum_gap
    expanded = (
        bool(candidates) and not observations
    ) or any(not fits(row, font_size, gap) for row in candidates)
    band_height = 0
    output_height = height
    if expanded:
        band_height = max(int(round(height * 0.16)), int(font_size * 2.9))
        output_height = height + band_height

    planned: list[dict[str, Any]] = []
    raw_boundaries = [row["original_subtitle_top"] * height - gap for row in candidates]
    for index, candidate in enumerate(candidates):
        if expanded:
            baseline = height + int(round(band_height * 0.78))
            placement = "expanded_bottom_canvas"
            collision_boundary = None
        else:
            neighborhood = raw_boundaries[max(0, index - 1):index + 2]
            smoothed = _median(neighborhood, raw_boundaries[index])
            collision_boundary = raw_boundaries[index]
            baseline = min(smoothed, collision_boundary)
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
        "ocr_observation_count": len(observations),
        "no_overlap_guaranteed": bool(observations) or expanded,
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
        "&H00000000,&H70000000,0,0,0,0,100,100,0,0,1,2.4,0.8,2,24,24,12,1",
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
) -> dict[str, Any]:
    """Create a second, subtitled delivery and a machine-readable audit."""
    base = Path(base_video)
    if not base.exists():
        raise FileNotFoundError(f"base video does not exist: {base}")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    width, height = probe_video_size(base)
    cache_path = Path(temp_dir) / "ocr_optimized" / "ocr_frames.json"
    observations = read_ocr_subtitle_observations(cache_path, width, height)
    events = build_subtitle_events(segments, tts_results, mix_audit, video_duration)
    if not events:
        raise RuntimeError("no actually mixed target-language lines to subtitle")
    layout = plan_subtitle_layout(
        events, observations, width, height, target_language=target_language
    )
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
        "subtitle_event_count": len(events),
        "ocr_observation_count": len(observations),
    }
