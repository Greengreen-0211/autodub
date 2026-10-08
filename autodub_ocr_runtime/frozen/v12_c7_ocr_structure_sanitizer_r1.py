"""C7 OCR structural sanitizer.

Pure implementation: separates OCR region text from trailing bounding-box
annotations.  It performs no ASR/reference similarity selection and makes no
API calls.
"""
from __future__ import annotations

import re
from typing import Any, Mapping


BBOX_RE = re.compile(
    r"\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*,\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)"
)
ASCII_LETTER_RE = re.compile(r"[A-Za-z]")
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
CONTENT_RE = re.compile(r"[A-Za-z0-9\u3400-\u4dbf\u4e00-\u9fff]")


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def _script_allowed(text: str, source_language: str) -> bool:
    if source_language == "en":
        return bool(ASCII_LETTER_RE.search(text))
    if source_language == "zh":
        return bool(CJK_RE.search(text))
    raise ValueError(f"unsupported_source_language:{source_language}")


def split_ocr_line(text: str) -> list[dict[str, Any]]:
    """Split `text(x1,y1),(x2,y2)...` into ordered regions.

    Coordinates remain in the returned private provenance object.  Region text
    is normalized only for Unicode whitespace.
    """
    if not isinstance(text, str):
        raise TypeError("ocr_line_text_must_be_string")
    regions: list[dict[str, Any]] = []
    cursor = 0
    for match in BBOX_RE.finditer(text):
        raw_text = text[cursor:match.start()]
        regions.append({
            "raw_text": raw_text,
            "text": normalize_whitespace(raw_text),
            "bbox": [int(match.group(i)) for i in range(1, 5)],
            "source_start": cursor,
            "source_end": match.start(),
        })
        cursor = match.end()
    tail = text[cursor:]
    if tail or not regions:
        regions.append({
            "raw_text": tail,
            "text": normalize_whitespace(tail),
            "bbox": None,
            "source_start": cursor,
            "source_end": len(text),
        })
    return regions


def sanitize_request(request: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a C4-compatible request plus a private structural audit."""
    required = {"id", "source_language", "qwen_asr", "ocr_candidates"}
    if not isinstance(request, Mapping) or set(request) != required:
        raise ValueError("request_fields_must_be_exact")
    language = request["source_language"]
    seen: set[str] = set()
    sanitized_lines: list[dict[str, Any]] = []
    audit_lines: list[dict[str, Any]] = []
    for line in request["ocr_candidates"]:
        regions = split_ocr_line(line["text"])
        region_audit = []
        for index, region in enumerate(regions, start=1):
            value = region["text"]
            if not value or not CONTENT_RE.search(value):
                status = "dropped_non_content"
            elif not _script_allowed(value, language):
                status = "dropped_wrong_script"
            elif value in seen:
                status = "dropped_exact_duplicate"
            else:
                status = "kept"
                seen.add(value)
                clean = {
                    "line_id": f"{line['line_id']}S{index:02d}",
                    "text": value,
                }
                for field in ("lexical_score", "subtitle_score", "selection"):
                    if field in line:
                        clean[field] = line[field]
                sanitized_lines.append(clean)
            region_audit.append({**region, "region_index": index, "status": status})
        audit_lines.append({
            "line_id": line["line_id"],
            "raw_text": line["text"],
            "regions": region_audit,
        })
    sanitized = {
        "id": request["id"],
        "source_language": language,
        "qwen_asr": request["qwen_asr"],
        "ocr_candidates": sanitized_lines,
    }
    audit = {
        "id": request["id"],
        "source_language": language,
        "input_line_count": len(request["ocr_candidates"]),
        "output_region_count": len(sanitized_lines),
        "lines": audit_lines,
    }
    return sanitized, audit


def contains_bbox(text: str) -> bool:
    return bool(BBOX_RE.search(text))
