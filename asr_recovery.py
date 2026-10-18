"""Evidence-driven helpers for recovering ASR text without trusting bad timing.

The functions in this module deliberately keep *what was said* separate from
*when every word was said*.  This lets AutoDub retain text supported by two
independent acoustic views while preventing broken forced-alignment output from
creating speaker or sentence boundaries.
"""

from __future__ import annotations

import difflib
import html
import math
import re
import unicodedata
from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


_WORD_TOKEN_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", flags=re.UNICODE)
_TERMINATED_CLAUSE_RE = re.compile(
    r"([^.!?。！？]+?)([.!?。！？]+)(?=\s|$)",
    flags=re.UNICODE,
)


def normalize_text(text: str) -> str:
    return "".join(
        char
        for char in unicodedata.normalize("NFKC", str(text or "")).casefold()
        if char.isalnum()
    )


def text_similarity(left: str, right: str) -> float:
    left_key, right_key = normalize_text(left), normalize_text(right)
    if not left_key or not right_key:
        return 0.0
    return difflib.SequenceMatcher(None, left_key, right_key).ratio()


def _alignment_rank(segment: Mapping[str, Any]) -> Tuple[int, float, int, int]:
    status = str(segment.get("timestamp_alignment_status") or "")
    return (
        {"valid": 2, "degraded": 1}.get(status, 0),
        float(segment.get("timestamp_coverage_ratio") or 0.0),
        -len(segment.get("timestamp_warning_codes") or []),
        len(normalize_text(str(segment.get("text") or ""))),
    )


def coarse_segment(
    candidate: Mapping[str, Any],
    *,
    start: float,
    end: float,
    origin: str,
    text_confidence: float,
    reason: str,
) -> Dict[str, Any]:
    """Return a safe segment whose outer speech bounds may be used, but not words."""
    result = dict(candidate)
    result.update(
        {
            "start": round(float(start), 3),
            "end": round(float(end), 3),
            "speech_start": round(float(start), 3),
            "speech_end": round(float(end), 3),
            "asr_origin": origin,
            "asr_recovery_status": "accepted_coarse",
            "timestamp_precision": "coarse",
            "speaker_boundary_eligible": False,
            "asr_text_confidence": round(float(text_confidence), 4),
            "asr_alignment_confidence": 0.0,
            "timestamp_alignment_status": "coarse",
            "timestamp_coverage_ratio": 0.0,
            "timestamp_warning_codes": sorted(
                set(result.get("timestamp_warning_codes") or [])
                | {"coarse_recovery_bounds"}
            ),
            "qwen3_time_stamps": [],
            "qwen3_sentence_candidates": [],
            "asr_recovery_reason": reason,
        }
    )
    return result


def _word_tokens_with_spans(text: str) -> List[Tuple[str, int, int]]:
    return [
        (match.group(0).casefold(), match.start(), match.end())
        for match in _WORD_TOKEN_RE.finditer(str(text or ""))
    ]


def _repair_repeated_clause_boundary(
    left_text: str,
    right_text: str,
) -> Optional[Tuple[str, str, Dict[str, Any]]]:
    """Move a repeated clause suffix across one broken coarse boundary.

    Recovery/OCR timelines can split ``Do the magic! Do the`` immediately
    before ``magic Ready?``.  This helper is deliberately evidence-bound: the
    unfinished left fragment must be an exact prefix of the preceding complete
    clause, and the right side must begin with the exact missing suffix.  It
    therefore repairs repeated dialogue without asking a language model to
    invent, delete, or paraphrase words.
    """
    left_text = str(left_text or "").strip()
    right_text = str(right_text or "").strip()
    completed = list(_TERMINATED_CLAUSE_RE.finditer(left_text))
    if not completed or not right_text:
        return None
    previous = completed[-1]
    trailing_text = left_text[previous.end() :].strip()
    if not trailing_text:
        return None

    clause_tokens = _word_tokens_with_spans(previous.group(1))
    trailing_tokens = _word_tokens_with_spans(trailing_text)
    right_tokens = _word_tokens_with_spans(right_text)
    if not clause_tokens or not trailing_tokens or not right_tokens:
        return None
    clause_words = [item[0] for item in clause_tokens]
    trailing_words = [item[0] for item in trailing_tokens]
    if (
        len(trailing_words) >= len(clause_words)
        or clause_words[: len(trailing_words)] != trailing_words
    ):
        return None
    missing_words = clause_words[len(trailing_words) :]
    if [item[0] for item in right_tokens[: len(missing_words)]] != missing_words:
        return None

    suffix_end = right_tokens[len(missing_words) - 1][2]
    suffix_text = right_text[:suffix_end].strip()
    remaining = right_text[suffix_end:]
    punctuation_match = re.match(r"^([.!?。！？]+)", remaining)
    if punctuation_match:
        punctuation = punctuation_match.group(1)
        suffix_text += punctuation
        remaining = remaining[punctuation_match.end() :]
    else:
        punctuation = previous.group(2)
        suffix_text += punctuation
    remaining = remaining.lstrip()
    # Do not erase a segment.  A fully consumed right side needs a higher-level
    # timing merge rather than a text-only boundary repair.
    if not remaining:
        return None

    repaired_left = f"{left_text.rstrip()} {suffix_text}".strip()
    before_tokens = [
        item[0]
        for item in _word_tokens_with_spans(f"{left_text} {right_text}")
    ]
    after_tokens = [
        item[0]
        for item in _word_tokens_with_spans(f"{repaired_left} {remaining}")
    ]
    if before_tokens != after_tokens:
        raise RuntimeError("coarse boundary repair changed dialogue token order")
    return repaired_left, remaining, {
        "reason": "repeated_clause_prefix_completed_from_right",
        "moved_tokens": missing_words,
        "reference_clause": previous.group(0).strip(),
        "punctuation": punctuation,
    }


def reconcile_coarse_recovery_boundaries(
    segments: Sequence[Mapping[str, Any]],
    *,
    max_gap: float = 0.12,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Repair provably broken text boundaries between adjacent coarse turns.

    Timing is intentionally left untouched.  Speaker clustering and subtitle
    timing may refine it later, while translation immediately benefits from
    complete clauses.  All edits are audited and preserve the exact token
    sequence across the pair.
    """
    repaired = [dict(item) for item in segments]
    audit: List[Dict[str, Any]] = []
    for index in range(len(repaired) - 1):
        left, right = repaired[index], repaired[index + 1]
        if (
            left.get("timestamp_precision") != "coarse"
            or right.get("timestamp_precision") != "coarse"
            or left.get("asr_recovery_status") != "accepted_coarse"
            or right.get("asr_recovery_status") != "accepted_coarse"
        ):
            continue
        gap = float(right.get("start", 0.0)) - float(left.get("end", 0.0))
        if gap < -0.02 or gap > float(max_gap):
            continue
        result = _repair_repeated_clause_boundary(
            str(left.get("text") or ""),
            str(right.get("text") or ""),
        )
        if result is None:
            continue
        left_text, right_text, evidence = result
        before_left, before_right = left.get("text", ""), right.get("text", "")
        left["text"], right["text"] = left_text, right_text
        # Coarse recovery must never regain authority from stale word stamps.
        left["qwen3_time_stamps"] = []
        right["qwen3_time_stamps"] = []
        left["qwen3_sentence_candidates"] = []
        right["qwen3_sentence_candidates"] = []
        for item in (left, right):
            item["boundary_text_repaired"] = True
            item["boundary_text_repair_reason"] = evidence["reason"]
        audit.append({
            "left_index": index,
            "right_index": index + 1,
            "left_before": before_left,
            "right_before": before_right,
            "left_after": left_text,
            "right_after": right_text,
            "gap": round(gap, 4),
            **evidence,
        })
    return repaired, audit


def choose_consensus_candidate(
    candidates: Sequence[Mapping[str, Any]],
    *,
    threshold: float,
    start: float,
    end: float,
    precise_origin: str,
    coarse_origin: str,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """Choose a candidate from independent sources after text-only consensus.

    Importantly, candidates with unusable timestamps remain eligible to vote on
    text.  If no candidate has usable timing but two texts agree, a coarse
    segment is returned instead of discarding the transcript.
    """
    usable = [item for item in candidates if normalize_text(str(item.get("text") or ""))]
    best_pair: Optional[Tuple[Mapping[str, Any], Mapping[str, Any]]] = None
    best_similarity = 0.0
    for index, left in enumerate(usable):
        for right in usable[index + 1 :]:
            if left.get("asr_repair_source") == right.get("asr_repair_source"):
                continue
            similarity = text_similarity(str(left.get("text") or ""), str(right.get("text") or ""))
            if similarity > best_similarity:
                best_similarity = similarity
                best_pair = (left, right)
    decision = {
        "candidate_count": len(usable),
        "consensus_similarity": round(best_similarity, 4),
        "threshold": round(float(threshold), 4),
    }
    effective_threshold = float(threshold)
    if best_pair is not None:
        pair_lengths = [len(normalize_text(str(item.get("text") or ""))) for item in best_pair]
        # A single content-word disagreement can still produce a deceptively
        # high character score in a short utterance.
        # Short phrases therefore need near-exact independent agreement; OCR
        # may resolve the remaining ambiguity later.
        if max(pair_lengths, default=0) <= 24:
            effective_threshold = max(effective_threshold, 0.94)
    decision["effective_threshold"] = round(effective_threshold, 4)
    if best_pair is None or best_similarity < effective_threshold:
        decision["reason"] = "no_independent_text_consensus"
        return None, decision

    selected = dict(max(best_pair, key=_alignment_rank))
    selected["asr_text_confidence"] = round(best_similarity, 4)
    selected["asr_recovery_sources"] = [
        str(item.get("asr_repair_source") or "unknown") for item in best_pair
    ]
    if _alignment_rank(selected)[0] > 0:
        selected.update(
            {
                "asr_origin": precise_origin,
                "asr_recovery_status": "accepted_precise",
                "timestamp_precision": "word",
                "speaker_boundary_eligible": True,
                "asr_alignment_confidence": (
                    0.95
                    if selected.get("timestamp_alignment_status") == "valid"
                    else 0.60
                ),
            }
        )
        decision["reason"] = "text_consensus_with_usable_alignment"
        decision["precision"] = "word"
        return selected, decision

    decision["reason"] = "text_consensus_without_usable_alignment"
    decision["precision"] = "coarse"
    return coarse_segment(
        selected,
        start=start,
        end=end,
        origin=coarse_origin,
        text_confidence=best_similarity,
        reason="dual_source_text_consensus_without_safe_word_timestamps",
    ), decision


def speech_micro_ranges(
    speech_ranges: Iterable[Tuple[float, float]],
    *,
    outer_start: float,
    outer_end: float,
    min_seconds: float = 0.65,
    max_seconds: float = 4.2,
) -> List[Tuple[float, float]]:
    """Clip independently detected speech to an empty ASR block and bound its size."""
    result: List[Tuple[float, float]] = []
    for raw_start, raw_end in speech_ranges:
        start = max(float(outer_start), float(raw_start))
        end = min(float(outer_end), float(raw_end))
        if end - start < min_seconds:
            continue
        parts = max(1, int(math.ceil((end - start) / max_seconds)))
        width = (end - start) / parts
        for index in range(parts):
            part_start = start + width * index
            part_end = end if index == parts - 1 else start + width * (index + 1)
            if part_end - part_start >= min_seconds:
                result.append((round(part_start, 3), round(part_end, 3)))
    return result


def build_recovery_groups(
    regions: Sequence[Mapping[str, Any]],
    *,
    max_gap: float = 0.18,
    context: float = 0.8,
    media_duration: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Group adjacent unresolved atoms for inference without losing provenance.

    The returned ``start``/``end`` values remain the acoustic core.  Expanded
    ``probe_start``/``probe_end`` values are only for ASR/OCR observation and
    must never become final sentence boundaries.
    """
    ordered = sorted(
        (dict(item) for item in regions),
        key=lambda item: (float(item["start"]), float(item["end"])),
    )
    groups: List[List[Dict[str, Any]]] = []
    for item in ordered:
        if not groups or float(item["start"]) - float(groups[-1][-1]["end"]) > max_gap:
            groups.append([item])
        else:
            groups[-1].append(item)

    result: List[Dict[str, Any]] = []
    for index, members in enumerate(groups):
        start = min(float(item["start"]) for item in members)
        end = max(float(item["end"]) for item in members)
        probe_start = max(0.0, start - max(0.0, context))
        probe_end = end + max(0.0, context)
        if media_duration is not None:
            probe_end = min(float(media_duration), probe_end)
        result.append(
            {
                "id": f"recovery-group-{index:04d}",
                "start": round(start, 3),
                "end": round(end, 3),
                "probe_start": round(probe_start, 3),
                "probe_end": round(probe_end, 3),
                "ocr_recovery_probe": True,
                "members": members,
                "member_ids": [str(item.get("id") or "") for item in members],
                "candidate_kind": "grouped_deferred_region",
                "dialogue_evidence": any(
                    bool(item.get("dialogue_evidence")) for item in members
                ),
                "acoustic_candidate": any(
                    bool(item.get("acoustic_candidate")) for item in members
                ),
                "acoustic_dialogue_score": max(
                    (float(item.get("acoustic_dialogue_score") or 0.0) for item in members),
                    default=0.0,
                ),
            }
        )
    return result


def acoustic_dialogue_score(
    *,
    duration: float,
    vocal_dbfs: float,
    source_dbfs: Optional[float],
    acoustic_text_observed: bool,
    primary_speech_confirmed: bool = False,
) -> Tuple[float, Dict[str, Any]]:
    """Return a conservative, auditable dialogue-likelihood score.

    Energy alone never proves dialogue.  It only upgrades an already isolated
    vocal region to a blocking candidate when the vocal stem is both audible
    and not dramatically weaker than the source mixture.
    """
    score = 0.0
    reasons: List[str] = []
    if acoustic_text_observed:
        score += 0.65
        reasons.append("asr_text_observed")
    if primary_speech_confirmed:
        score += 0.15
        reasons.append("primary_speech_chunk")
    if duration >= 0.65:
        score += 0.10
        reasons.append("sustained_region")
    if math.isfinite(vocal_dbfs) and vocal_dbfs >= -42.0:
        score += 0.20
        reasons.append("audible_vocal_stem")
    vocal_delta: Optional[float] = None
    if source_dbfs is not None and math.isfinite(source_dbfs) and math.isfinite(vocal_dbfs):
        vocal_delta = vocal_dbfs - source_dbfs
        if vocal_delta >= -12.0:
            score += 0.15
            reasons.append("vocal_not_buried_in_mix")
    score = min(1.0, score)
    return round(score, 4), {
        "duration": round(float(duration), 3),
        "vocal_dbfs": round(float(vocal_dbfs), 3) if math.isfinite(vocal_dbfs) else None,
        "source_dbfs": (
            round(float(source_dbfs), 3)
            if source_dbfs is not None and math.isfinite(source_dbfs)
            else None
        ),
        "vocal_source_delta_db": (
            round(float(vocal_delta), 3) if vocal_delta is not None else None
        ),
        "reasons": reasons,
    }


def classify_recovery_evidence(
    *,
    acoustic_text_observed: bool,
    acoustic_score: float,
    acoustic_candidate_threshold: float = 0.40,
) -> Dict[str, Any]:
    """Keep acoustic activity separate from confirmed dialogue evidence.

    Energy, duration, and vocal-stem dominance may identify a region worth
    auditing, but they cannot prove that translatable dialogue exists. Only
    observed text may mark the ASR-stage region as confirmed dialogue. OCR can
    independently upgrade it later when source-language subtitles are seen.
    """
    text_evidence = bool(acoustic_text_observed)
    acoustic_candidate = (
        not text_evidence
        and float(acoustic_score) >= float(acoustic_candidate_threshold)
    )
    if text_evidence:
        status = "confirmed_dialogue_text_observed"
    elif acoustic_candidate:
        status = "acoustic_candidate_without_text_evidence"
    else:
        status = "weak_activity_without_text_evidence"
    return {
        "dialogue_evidence": text_evidence,
        "acoustic_candidate": acoustic_candidate,
        "evidence_status": status,
    }


_COORDINATE_PREFIX = re.compile(
    r"^\s*(?:\[?\s*\d{1,4}\s*[,;:]\s*\d{1,4}[^\]]*\]?\s*)+"
)
_SPOTTING_RECORD = re.compile(
    r"<ref>(.*?)</ref>\s*(?:<quad>.*?</quad>)?", re.I | re.S
)
_PLAIN_COORD_RECORD = re.compile(
    r"(.*?)(\(\s*\d+\s*,\s*\d+\s*\)\s*,\s*"
    r"\(\s*\d+\s*,\s*\d+\s*\))"
)
_URL_OR_UI = re.compile(r"https?://|www\.|bilibili|coding|字幕组|@[A-Za-z0-9_]", re.I)
_CJK = re.compile(r"[\u3400-\u9fff]")
_LATIN = re.compile(r"[A-Za-z]")


def _ocr_lines(frame_text: str) -> List[str]:
    cleaned = str(frame_text or "").replace("```", "\n")
    references = [
        re.sub(r"\s+", " ", html.unescape(match.group(1))).strip(" \t|-_")
        for match in _SPOTTING_RECORD.finditer(cleaned)
    ]
    if references:
        return [line for line in references if line and not _URL_OR_UI.search(line)]
    coordinate_lines = []
    for physical_line in re.split(r"[\r\n]+", cleaned):
        for match in _PLAIN_COORD_RECORD.finditer(physical_line):
            line = re.sub(r"\s+", " ", match.group(1)).strip(" \t|-_")
            if line and not _URL_OR_UI.search(line):
                coordinate_lines.append(line)
    if coordinate_lines:
        return coordinate_lines
    lines: List[str] = []
    for raw in re.split(r"[\r\n]+", cleaned):
        line = _COORDINATE_PREFIX.sub("", raw).strip(" \t|-_")
        if line and not _URL_OR_UI.search(line):
            lines.append(line)
    return lines


def _language_matches(line: str, language: str) -> bool:
    cjk, latin = len(_CJK.findall(line)), len(_LATIN.findall(line))
    language = str(language or "").lower()
    language = {
        "english": "en",
        "chinese": "zh",
        "mandarin": "zh",
        "japanese": "ja",
        "korean": "ko",
    }.get(language, language)
    if language.startswith(("zh", "ja", "ko")):
        return cjk >= 2 and cjk >= latin
    if language.startswith(("en", "fr", "de", "es", "it", "pt")):
        return latin >= 3 and latin >= cjk * 2
    return cjk + latin >= 3


def stable_ocr_source_line(
    frame_texts: Sequence[str], source_language: str
) -> Tuple[Optional[str], Dict[str, Any]]:
    """Select a source-language subtitle repeated in at least two frames."""
    by_key: Dict[str, List[Tuple[int, str]]] = {}
    for frame_index, frame in enumerate(frame_texts):
        seen = set()
        for line in _ocr_lines(frame):
            key = normalize_text(line)
            if (
                key in seen
                or len(key) < 2
                or len(key) > 140
                or not _language_matches(line, source_language)
            ):
                continue
            seen.add(key)
            by_key.setdefault(key, []).append((frame_index, line))
    ranked = []
    for key, observations in by_key.items():
        frame_count = len({item[0] for item in observations})
        if frame_count < 2:
            continue
        line_counts = Counter(item[1] for item in observations)
        line = line_counts.most_common(1)[0][0]
        word_count = len(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", line))
        subtitle_shape = int(bool(re.search(r"[.!?。！？]$", line))) + int(word_count >= 2)
        ranked.append((frame_count, subtitle_shape, len(key), line))
    if not ranked:
        return None, {"reason": "no_stable_source_language_subtitle", "frame_count": len(frame_texts)}
    frame_count, subtitle_shape, _, line = max(ranked)
    if subtitle_shape <= 0:
        return None, {"reason": "stable_text_not_subtitle_like", "frame_count": frame_count}
    confidence = min(0.92, 0.70 + 0.08 * (frame_count - 2) + 0.04 * subtitle_shape)
    return line, {
        "reason": "stable_source_language_subtitle",
        "frame_count": frame_count,
        "confidence": round(confidence, 4),
    }


def _short_text_compatible(left: str, right: str) -> bool:
    """Match OCR variants without conflating different short utterances."""
    left_words = re.findall(r"[\w']+", unicodedata.normalize("NFKC", left).casefold())
    right_words = re.findall(r"[\w']+", unicodedata.normalize("NFKC", right).casefold())
    if max(len(left_words), len(right_words)) <= 4:
        return left_words == right_words
    return text_similarity(left, right) >= 0.88


def ocr_source_events(
    frame_records: Sequence[Mapping[str, Any]],
    source_language: str,
    *,
    region_start: float,
    region_end: float,
    context_tolerance: float = 0.2,
    exclude_texts: Sequence[str] = (),
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Build ordered subtitle events from dense OCR observations.

    Identical text separated by a real disappearance is kept as two events so
    repeated short calls are not collapsed.  Event bounds partition the parent
    speech region: OCR supplies text/order while the acoustic region remains the
    conservative outer timing boundary.
    """
    observations: List[Dict[str, Any]] = []
    for frame_index, record in enumerate(frame_records):
        timestamp = float(record.get("timestamp", 0.0))
        seen: set[str] = set()
        for line in _ocr_lines(str(record.get("text") or "")):
            key = normalize_text(line)
            if (
                key in seen
                or len(key) < 2
                or len(key) > 140
                or not _language_matches(line, source_language)
            ):
                continue
            seen.add(key)
            observations.append(
                {
                    "frame_index": frame_index,
                    "timestamp": timestamp,
                    "text": line,
                    "key": key,
                }
            )

    if not observations:
        return [], {
            "reason": "no_source_language_ocr_evidence",
            "frame_count": len(frame_records),
            "dialogue_evidence": False,
        }

    timestamps = sorted({float(item["timestamp"]) for item in observations})
    steps = [right - left for left, right in zip(timestamps, timestamps[1:]) if right > left]
    cadence = sorted(steps)[len(steps) // 2] if steps else 0.35
    episode_gap = max(0.9, cadence * 2.6)

    clusters: List[List[Dict[str, Any]]] = []
    for observation in sorted(observations, key=lambda item: (item["timestamp"], item["key"])):
        best_index: Optional[int] = None
        best_score = 0.0
        for index, cluster in enumerate(clusters):
            if observation["timestamp"] - float(cluster[-1]["timestamp"]) > episode_gap:
                continue
            representative = str(cluster[0]["text"])
            score = text_similarity(representative, str(observation["text"]))
            if _short_text_compatible(representative, str(observation["text"])) and score > best_score:
                best_index, best_score = index, score
        if best_index is None:
            clusters.append([observation])
        else:
            clusters[best_index].append(observation)

    candidates: List[Dict[str, Any]] = []
    for cluster in clusters:
        frame_count = len({int(item["frame_index"]) for item in cluster})
        if frame_count < 2:
            continue
        line_counts = Counter(str(item["text"]) for item in cluster)
        text = max(line_counts, key=lambda line: (line_counts[line], len(normalize_text(line))))
        first_seen = min(float(item["timestamp"]) for item in cluster)
        last_seen = max(float(item["timestamp"]) for item in cluster)
        candidates.append(
            {
                "text": text,
                "first_seen": first_seen,
                "last_seen": last_seen,
                "frame_count": frame_count,
                "confidence": round(min(0.96, 0.74 + 0.06 * (frame_count - 2)), 4),
            }
        )

    if not candidates:
        return [], {
            "reason": "source_language_ocr_seen_but_not_stable",
            "frame_count": len(frame_records),
            "observation_count": len(observations),
            "dialogue_evidence": True,
        }

    # Frames may be sampled outside the acoustic core so subtitles appearing
    # slightly early/late remain visible.  Do not steal a subtitle that exists
    # wholly outside the core plus a small synchronization tolerance.
    candidates = [
        item
        for item in candidates
        if float(item["last_seen"]) >= float(region_start) - context_tolerance
        and float(item["first_seen"]) <= float(region_end) + context_tolerance
    ]
    if exclude_texts:
        candidates = [
            item
            for item in candidates
            if not any(
                text_similarity(str(item["text"]), known) >= 0.90
                for known in exclude_texts
                if normalize_text(known)
            )
        ]

    if not candidates:
        return [], {
            "reason": "source_ocr_only_matched_context_or_known_dialogue",
            "frame_count": len(frame_records),
            "observation_count": len(observations),
            "dialogue_evidence": False,
        }

    candidates.sort(key=lambda item: (item["first_seen"], item["last_seen"]))
    # Partition the confirmed acoustic parent interval at subtitle-change
    # midpoints.  This avoids arbitrary fixed-width recovery windows while
    # ensuring that the complete confirmed utterance remains covered.
    boundaries = [float(region_start)]
    for left, right in zip(candidates, candidates[1:]):
        midpoint = (float(left["last_seen"]) + float(right["first_seen"])) / 2.0
        boundaries.append(max(boundaries[-1], min(float(region_end), midpoint)))
    boundaries.append(float(region_end))
    events: List[Dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        start = boundaries[index]
        end = boundaries[index + 1]
        if end - start < 0.08:
            continue
        events.append(
            dict(
                candidate,
                start=round(start, 3),
                end=round(end, 3),
            )
        )
    return events, {
        "reason": "stable_source_language_subtitle_events",
        "frame_count": len(frame_records),
        "observation_count": len(observations),
        "event_count": len(events),
        "dialogue_evidence": True,
    }
