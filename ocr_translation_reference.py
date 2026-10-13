"""Build conservative, internal OCR translation references for Step 3.

ASR text remains authoritative.  This module only exposes a reference when a
source-language subtitle line can be matched to the final Step 2 segment and a
target-language line is paired with it in the same OCR frame.
"""
from __future__ import annotations

from collections import defaultdict
from difflib import SequenceMatcher
import hashlib
import html
import json
import math
from pathlib import Path
import re
from typing import Any


REFERENCE_VERSION = "ocr-translation-reference-v3"
_REF_RE = re.compile(
    r"<ref>(.*?)</ref>\s*(?:<quad>(.*?)</quad>)?",
    flags=re.IGNORECASE | re.DOTALL,
)
_PLAIN_COORD_RE = re.compile(
    r"(.*?)(\(\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*\)\s*,\s*"
    r"\(\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*\))"
)
_LANGUAGE_ALIASES = {
    "english": "en", "chinese": "zh", "mandarin": "zh", "cmn": "zh",
    "japanese": "ja", "korean": "ko", "russian": "ru", "arabic": "ar",
    "french": "fr", "german": "de", "spanish": "es", "italian": "it",
    "portuguese": "pt",
}
_LATIN_LANGUAGES = {"en", "fr", "de", "es", "it", "pt"}


def _language_code(value: Any) -> str:
    value = str(value or "").strip().lower().replace("_", "-")
    value = _LANGUAGE_ALIASES.get(value, value)
    return value.split("-", 1)[0]


def _script_family(language: Any) -> str:
    language = _language_code(language)
    if language == "zh":
        return "han"
    if language == "ja":
        return "japanese"
    if language == "ko":
        return "hangul"
    if language == "ru":
        return "cyrillic"
    if language == "ar":
        return "arabic"
    if language in _LATIN_LANGUAGES:
        return "latin"
    return "unknown"


def _line_matches_script(text: str, family: str) -> bool:
    counts = {
        "han": len(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]", text)),
        "kana": len(re.findall(r"[\u3040-\u30ff]", text)),
        "hangul": len(re.findall(r"[\uac00-\ud7af]", text)),
        "cyrillic": len(re.findall(r"[\u0400-\u04ff]", text)),
        "arabic": len(re.findall(r"[\u0600-\u06ff]", text)),
        "latin": len(re.findall(r"[A-Za-z]", text)),
    }
    if family == "han":
        return counts["han"] > 0 and counts["kana"] == 0
    if family == "japanese":
        return counts["kana"] > 0 or counts["han"] > 0
    if family in counts:
        return counts[family] > 0
    return False


def _noise_reason(text: str) -> str:
    compact = re.sub(r"\s+", "", text)
    if not compact:
        return "empty"
    if re.search(r"(?:https?://|www\.)\S+", text, flags=re.IGNORECASE):
        return "url"
    if re.fullmatch(r"(?:\[|\()?\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?(?:\]|\))?", compact):
        return "timecode"
    if re.fullmatch(r"[\d０-９]+(?:[.,，]\d+)?", compact):
        return "pure_number"
    if all(not char.isalnum() and not "\u3400" <= char <= "\u9fff" for char in compact):
        return "pure_symbol"
    if len(compact) > 180:
        return "too_long"
    return ""


def _bbox(value: str | None) -> list[float] | None:
    numbers = [float(number) for number in re.findall(r"-?\d+(?:\.\d+)?", value or "")]
    if len(numbers) < 4:
        return None
    xs, ys = numbers[0::2], numbers[1::2]
    if not xs or not ys:
        return None
    left, right, top, bottom = min(xs), max(xs), min(ys), max(ys)
    if right <= left or bottom <= top:
        return None
    return [round(left, 3), round(top, 3), round(right, 3), round(bottom, 3)]


def _parse_lines(raw: Any) -> list[dict[str, Any]]:
    text = str(raw or "").replace("```", "\n")
    records = []
    for match in _REF_RE.finditer(text):
        records.append((html.unescape(match.group(1)), _bbox(match.group(2))))
    if not records:
        for physical_line in re.split(r"[\r\n]+", text):
            for match in _PLAIN_COORD_RE.finditer(physical_line):
                records.append((match.group(1), _bbox(match.group(2))))
    if not records:
        records = [(line, None) for line in re.split(r"[\r\n]+", text)]
    output, seen = [], set()
    for raw_text, box in records:
        clean = re.sub(r"\s+", " ", raw_text).strip(" \t|-:")
        normalized = _normalize(clean)
        reason = _noise_reason(clean)
        if not normalized or normalized in seen or reason:
            continue
        seen.add(normalized)
        output.append({"text": clean, "normalized": normalized, "bbox": box})
    return output


def _normalize(text: Any) -> str:
    return "".join(
        char.casefold() for char in str(text or "")
        if char.isalnum() or "\u3400" <= char <= "\u9fff"
    )


def _tokens(text: Any) -> set[str]:
    value = str(text or "").casefold()
    words = set(re.findall(r"[a-z0-9]+", value))
    cjk = "".join(re.findall(r"[\u3400-\u9fff]", value))
    words.update(cjk[index:index + 2] for index in range(max(0, len(cjk) - 1)))
    if cjk and len(cjk) == 1:
        words.add(cjk)
    return words


def _source_similarity(left: Any, right: Any) -> float:
    left_norm, right_norm = _normalize(left), _normalize(right)
    if not left_norm or not right_norm:
        return 0.0
    sequence = SequenceMatcher(None, left_norm, right_norm).ratio()
    left_tokens, right_tokens = _tokens(left), _tokens(right)
    union = left_tokens | right_tokens
    jaccard = len(left_tokens & right_tokens) / len(union) if union else 0.0
    containment = min(len(left_norm), len(right_norm)) / max(len(left_norm), len(right_norm))
    if left_norm in right_norm or right_norm in left_norm:
        containment = min(1.0, containment + 0.15)
    return min(1.0, 0.68 * sequence + 0.20 * jaccard + 0.12 * containment)


def _source_spelling_similarity(left: Any, right: Any) -> float:
    """Best token-level spelling match for noisy names and acronyms."""
    left_tokens = re.findall(r"[A-Za-z0-9]{2,}", str(left or "").casefold())
    right_tokens = re.findall(r"[A-Za-z0-9]{2,}", str(right or "").casefold())
    if not left_tokens or not right_tokens:
        return 0.0
    return max(
        SequenceMatcher(None, left_token, right_token).ratio()
        for left_token in left_tokens
        for right_token in right_tokens
    )


def _spatial_score(left: list[float] | None, right: list[float] | None) -> float:
    if not left or not right:
        return 0.55
    lw, rw = left[2] - left[0], right[2] - right[0]
    overlap = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    overlap_ratio = overlap / max(1.0, min(lw, rw))
    center_distance = abs((left[0] + left[2]) / 2 - (right[0] + right[2]) / 2)
    center_score = max(0.0, 1.0 - center_distance / max(1.0, max(lw, rw)))
    lh, rh = left[3] - left[1], right[3] - right[1]
    vertical_gap = max(0.0, max(left[1], right[1]) - min(left[3], right[3]))
    vertical_score = max(0.0, 1.0 - vertical_gap / max(1.0, 5.0 * max(lh, rh)))
    return min(1.0, 0.45 * overlap_ratio + 0.35 * center_score + 0.20 * vertical_score)


def _time_score(timestamp: float, start: float, end: float) -> float:
    if start <= timestamp <= end:
        return 1.0
    distance = start - timestamp if timestamp < start else timestamp - end
    return max(0.0, 1.0 - distance / 0.8)


def _join_subtitle_lines(values: list[str]) -> str:
    """Join consecutive subtitle cards without altering their wording."""
    output = ""
    for value in values:
        value = re.sub(r"\s+", " ", str(value or "")).strip()
        if not value:
            continue
        if output and not re.search(r"\s$", output):
            output += " "
        output += value
    return output.strip()


def _composed_groups(
    pairs: list[dict[str, Any]], start: float, end: float, asr: str,
) -> list[dict[str, Any]]:
    """Build contiguous multi-card subtitle references for one final sentence."""
    nearby = [
        pair for pair in pairs
        if start - 0.8 <= float(pair["timestamp"]) <= end + 0.8
    ]
    by_time: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for pair in nearby:
        by_time[round(float(pair["timestamp"]), 3)].append(pair)
    events = []
    for timestamp in sorted(by_time):
        # Normally there is one bilingual pair per subtitle card.  If OCR found
        # several, spatial alignment is the safest deterministic representative.
        best = max(
            by_time[timestamp],
            key=lambda item: (
                _source_similarity(asr, item["source_text"]),
                float(item["spatial_score"]),
            ),
        )
        key = (_normalize(best["source_text"]), _normalize(best["target_text"]))
        if events and events[-1]["key"] == key:
            events[-1]["timestamps"].append(timestamp)
            continue
        events.append({"key": key, "pair": best, "timestamps": [timestamp]})

    groups = []
    for left in range(len(events)):
        for right in range(left + 1, min(len(events), left + 6)):
            selected = events[left:right + 1]
            first = selected[0]["timestamps"][0]
            last = selected[-1]["timestamps"][-1]
            if last - first > max(4.0, (end - start) + 1.6):
                break
            source_text = _join_subtitle_lines([item["pair"]["source_text"] for item in selected])
            target_text = _join_subtitle_lines([item["pair"]["target_text"] for item in selected])
            similarity = _source_similarity(asr, source_text)
            if similarity < 0.55:
                continue
            timestamps = sorted({value for item in selected for value in item["timestamps"]})
            spatial = sum(float(item["pair"]["spatial_score"]) for item in selected) / len(selected)
            midpoint = (first + last) / 2.0
            score = 0.72 * similarity + 0.18 * _time_score(midpoint, start, end) + 0.10 * spatial
            groups.append({
                "source_text": source_text,
                "target_text": target_text,
                "scores": [score],
                "source_similarities": [similarity],
                "timestamps": timestamps,
                "composed": True,
            })
    return groups


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _empty_result(directory: Path, reason: str, source: Any, target: Any) -> dict[str, Any]:
    index = {
        "version": REFERENCE_VERSION, "status": "disabled", "reason": reason,
        "source_language": _language_code(source), "target_language": _language_code(target),
        "frames": [], "pairs": [],
    }
    matches = {"version": REFERENCE_VERSION, "status": "disabled", "reason": reason,
               "segments": []}
    _atomic_json(directory / "translation_index.json", index)
    _atomic_json(directory / "translation_matches.json", matches)
    return {"references_by_segment": {}, "manifest": _manifest(index, matches)}


def _manifest(index: dict[str, Any], matches: dict[str, Any]) -> dict[str, Any]:
    digest = hashlib.sha256(
        json.dumps({"index": index, "matches": matches}, ensure_ascii=False,
                   sort_keys=True).encode("utf-8")
    ).hexdigest()
    accepted = sum(row.get("status") == "accepted" for row in matches.get("segments", []))
    return {"version": REFERENCE_VERSION, "fingerprint": digest,
            "accepted_segments": accepted,
            "index_path": "ocr_optimized/translation_index.json",
            "matches_path": "ocr_optimized/translation_matches.json"}


def prepare_ocr_translation_references(
    temp_dir: str | Path,
    segments: list[dict[str, Any]],
    source_language: Any,
    target_language: Any,
) -> dict[str, Any]:
    """Return prompt-safe references and write detailed internal audit files."""
    directory = Path(temp_dir) / "ocr_optimized"
    source_family = _script_family(source_language)
    target_family = _script_family(target_language)
    if source_family == "unknown" or target_family == "unknown":
        return _empty_result(directory, "unsupported_language_script", source_language, target_language)
    if source_family == target_family:
        return _empty_result(directory, "source_target_scripts_not_distinguishable", source_language, target_language)
    cache_path = directory / "ocr_frames.json"
    if not cache_path.is_file():
        return _empty_result(directory, "ocr_frame_cache_missing", source_language, target_language)
    try:
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return _empty_result(directory, f"ocr_frame_cache_invalid:{type(exc).__name__}",
                             source_language, target_language)

    values = cache.values() if isinstance(cache, dict) else cache if isinstance(cache, list) else []
    frames = []
    for record in values:
        if not isinstance(record, dict) or record.get("status") != "ok":
            continue
        try:
            timestamp = float(record["timestamp"])
        except (KeyError, TypeError, ValueError):
            continue
        lines = _parse_lines(record.get("text"))
        for line in lines:
            source_match = _line_matches_script(line["text"], source_family)
            target_match = _line_matches_script(line["text"], target_family)
            # A translated subtitle may retain Latin names/acronyms.  When a
            # line contains both scripts, the non-Latin script determines its
            # role instead of allowing one brand name to reclassify the line.
            if source_match and target_match and source_family == "latin":
                line["role"] = "target"
            elif source_match and target_match and target_family == "latin":
                line["role"] = "source"
            elif source_match:
                line["role"] = "source"
            elif target_match:
                line["role"] = "target"
            else:
                line["role"] = "other"
        frames.append({"timestamp": timestamp, "segment_id": str(record.get("segment_id", "")),
                       "lines": lines})
    frames.sort(key=lambda frame: frame["timestamp"])

    # Repeated text across distant frames is usually UI, a channel mark or a watermark.
    occurrences: dict[str, list[float]] = defaultdict(list)
    for frame in frames:
        for line in frame["lines"]:
            occurrences[line["normalized"]].append(frame["timestamp"])
    watermark_threshold = max(3, math.ceil(len(frames) * 0.4))
    watermarks = {
        normalized for normalized, times in occurrences.items()
        if len(times) >= watermark_threshold and max(times) - min(times) >= 8.0
    }

    pairs = []
    for frame_index, frame in enumerate(frames):
        sources = [line for line in frame["lines"]
                   if line["role"] == "source" and line["normalized"] not in watermarks]
        targets = [line for line in frame["lines"]
                   if line["role"] == "target" and line["normalized"] not in watermarks]
        for source in sources:
            for target in targets:
                spatial = _spatial_score(source["bbox"], target["bbox"])
                if source["bbox"] and target["bbox"] and spatial < 0.28:
                    continue
                pairs.append({
                    "frame_index": frame_index, "timestamp": frame["timestamp"],
                    "segment_id": frame["segment_id"], "source_text": source["text"],
                    "target_text": target["text"], "spatial_score": round(spatial, 6),
                })

    references_by_segment: dict[str, list[dict[str, Any]]] = {}
    source_hints_by_segment: dict[str, list[dict[str, Any]]] = {}
    match_rows = []
    for index, segment in enumerate(segments):
        sid = str(segment.get("id", index))
        start, end = float(segment.get("start", 0.0)), float(segment.get("end", 0.0))
        asr = str(segment.get("text", ""))
        grouped: dict[tuple[str, str], dict[str, Any]] = {}
        for pair in pairs:
            if pair["timestamp"] < start - 0.8 or pair["timestamp"] > end + 0.8:
                continue
            source_similarity = _source_similarity(asr, pair["source_text"])
            if source_similarity < 0.42:
                continue
            time_score = _time_score(pair["timestamp"], start, end)
            score = (0.72 * source_similarity + 0.18 * time_score
                     + 0.10 * float(pair["spatial_score"]))
            key = (_normalize(pair["source_text"]), _normalize(pair["target_text"]))
            group = grouped.setdefault(key, {
                "source_text": pair["source_text"], "target_text": pair["target_text"],
                "scores": [], "source_similarities": [], "timestamps": [],
            })
            group["scores"].append(score)
            group["source_similarities"].append(source_similarity)
            group["timestamps"].append(pair["timestamp"])
        for group in _composed_groups(pairs, start, end, asr):
            key = (_normalize(group["source_text"]), _normalize(group["target_text"]))
            existing = grouped.get(key)
            if existing:
                existing["scores"].extend(group["scores"])
                existing["source_similarities"].extend(group["source_similarities"])
                existing["timestamps"].extend(group["timestamps"])
                existing["composed"] = existing.get("composed", False) or group["composed"]
            else:
                grouped[key] = group
        candidates = []
        for group in grouped.values():
            support = len(set(group["timestamps"]))
            score = min(1.0, max(group["scores"]) + min(0.06, 0.03 * (support - 1)))
            source_similarity = max(group["source_similarities"])
            confidence = (
                "high" if score >= 0.78 and source_similarity >= 0.72 and
                (support >= 2 or source_similarity >= 0.88)
                else "medium" if score >= 0.70 and source_similarity >= 0.62 and support >= 2
                else "low"
            )
            candidates.append({
                "source_text": group["source_text"], "target_text": group["target_text"],
                "confidence": confidence, "score": round(score, 6),
                "source_similarity": round(source_similarity, 6), "frame_support": support,
                "composed": bool(group.get("composed", False)),
                "timestamps": sorted(set(round(value, 3) for value in group["timestamps"])),
            })
        candidates.sort(key=lambda item: (item["score"], item["frame_support"]), reverse=True)
        source_hints = []
        seen_sources = set()
        for candidate in candidates:
            normalized_source = _normalize(candidate["source_text"])
            if normalized_source in seen_sources or candidate["source_similarity"] < 0.42:
                continue
            seen_sources.add(normalized_source)
            source_hints.append({
                "source_text": candidate["source_text"],
                "source_similarity": candidate["source_similarity"],
                "frame_support": candidate["frame_support"],
            })
            if len(source_hints) >= 5:
                break
        # A name can occupy its own subtitle card just before the rest of the
        # sentence (e.g. "Elsa!" then "Wake up...").  Such a short line may
        # score poorly against the whole ASR sentence, but is still useful as
        # low-authority source spelling evidence.  Never expose its translation.
        nearby_sources: dict[str, dict[str, Any]] = {}
        for pair in pairs:
            timestamp = float(pair["timestamp"])
            if timestamp < start - 1.6 or timestamp > end + 0.8:
                continue
            normalized_source = _normalize(pair["source_text"])
            if not normalized_source:
                continue
            record = nearby_sources.setdefault(normalized_source, {
                "source_text": pair["source_text"],
                "source_similarity": 0.0,
                "spelling_similarity": 0.0,
                "timestamps": set(),
            })
            record["source_similarity"] = max(
                float(record["source_similarity"]),
                _source_similarity(asr, pair["source_text"]),
            )
            record["spelling_similarity"] = max(
                float(record["spelling_similarity"]),
                _source_spelling_similarity(asr, pair["source_text"]),
            )
            record["timestamps"].add(round(timestamp, 3))
        ranked_nearby = sorted(
            nearby_sources.values(),
            key=lambda item: (
                max(item["source_similarity"], item["spelling_similarity"]),
                len(item["timestamps"]),
            ),
            reverse=True,
        )
        for hint in ranked_nearby:
            normalized_source = _normalize(hint["source_text"])
            if normalized_source in seen_sources or (
                hint["source_similarity"] < 0.18 and hint["spelling_similarity"] < 0.72
            ):
                continue
            seen_sources.add(normalized_source)
            source_hints.append({
                "source_text": hint["source_text"],
                "source_similarity": round(float(hint["source_similarity"]), 6),
                "spelling_similarity": round(float(hint["spelling_similarity"]), 6),
                "frame_support": len(hint["timestamps"]),
            })
            if len(source_hints) >= 5:
                break
        if source_hints:
            source_hints_by_segment[sid] = source_hints
        eligible = [candidate for candidate in candidates if candidate["confidence"] in {"high", "medium"}]
        status, reason, accepted = "rejected", "no_trusted_reference", []
        if eligible:
            best = eligible[0]
            competing = next((item for item in eligible[1:]
                              if _normalize(item["target_text"]) != _normalize(best["target_text"])), None)
            margin = best["score"] - competing["score"] if competing else 1.0
            if competing and margin < 0.06:
                reason = "ambiguous_target_reference"
            else:
                status, reason = "accepted", "trusted_source_aligned_pair"
                accepted = [{key: best[key] for key in
                             ("target_text", "source_text", "confidence", "score", "frame_support")}]
                references_by_segment[sid] = accepted
        match_rows.append({
            "segment_id": sid, "start": start, "end": end, "asr_text": asr,
            "status": status, "reason": reason, "accepted_references": accepted,
            "candidates": candidates[:5],
        })

    index = {
        "version": REFERENCE_VERSION, "status": "ok",
        "source_language": _language_code(source_language),
        "target_language": _language_code(target_language),
        "watermark_lines": sorted(watermarks), "frames": frames, "pairs": pairs,
    }
    matches = {"version": REFERENCE_VERSION, "status": "ok", "segments": match_rows}
    _atomic_json(directory / "translation_index.json", index)
    _atomic_json(directory / "translation_matches.json", matches)
    return {"references_by_segment": references_by_segment,
            "source_hints_by_segment": source_hints_by_segment,
            "manifest": _manifest(index, matches)}
