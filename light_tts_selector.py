#!/usr/bin/env python3
"""Lightweight candidate TTS duration selector used between Step 3 and routing."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Tuple


VOICE_BY_LANGUAGE = {
    "zh": "zh-CN-XiaoxiaoNeural",
    "en": "en-US-JennyNeural",
    "ja": "ja-JP-NanamiNeural",
    "ko": "ko-KR-SunHiNeural",
    "fr": "fr-FR-DeniseNeural",
    "de": "de-DE-KatjaNeural",
    "es": "es-ES-ElviraNeural",
    "ru": "ru-RU-SvetlanaNeural",
}


class LightTTSBackend(Protocol):
    name: str
    extension: str
    signature: str

    def synthesize(self, text: str, language: str, output_path: Path) -> None:
        ...


def normalize_translation_candidates(
    raw_candidates: Any,
    fallback_text: Any,
    *,
    maximum: int = 5,
) -> List[Dict[str, Any]]:
    """Normalize candidate objects without inventing translations."""
    normalized: List[Dict[str, Any]] = []
    seen = set()
    if isinstance(raw_candidates, list):
        for value in raw_candidates:
            item = dict(value) if isinstance(value, Mapping) else {"text": value}
            text = str(item.get("text", "")).strip()
            if not text or text in seen:
                continue
            item["text"] = text
            normalized.append(item)
            seen.add(text)
            if len(normalized) >= maximum:
                break
    fallback = str(fallback_text or "").strip()
    if not normalized and fallback:
        normalized.append({"text": fallback})
    return normalized


def _language_code(value: Any) -> str:
    language = str(value or "").strip().lower().replace("_", "-")
    aliases = {"chinese": "zh", "mandarin": "zh", "english": "en"}
    language = aliases.get(language, language)
    return language.split("-", 1)[0] or "en"


def _safe_id(value: Any) -> str:
    return "".join(char if char.isalnum() or char in "_.-" else "_" for char in str(value))


def _atomic_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def audio_duration(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as handle:
            return handle.getnframes() / float(handle.getframerate())
    except Exception:
        from pydub import AudioSegment

        return len(AudioSegment.from_file(path)) / 1000.0


class EdgeTTSBackend:
    name = "edge-tts"
    extension = ".mp3"

    def __init__(self) -> None:
        try:
            import edge_tts  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("edge-tts 未安装；请安装 edge-tts 或使用 espeak 后端。") from exc
        self.rate = os.getenv("AUTODUB_LIGHT_TTS_RATE", "+0%")
        self.voices = {
            language: os.getenv(
                f"AUTODUB_LIGHT_TTS_VOICE_{language.upper()}", default_voice
            )
            for language, default_voice in VOICE_BY_LANGUAGE.items()
        }
        self.signature = f"{self.name}:{self.rate}:{json.dumps(self.voices, sort_keys=True)}"

    def synthesize(self, text: str, language: str, output_path: Path) -> None:
        import edge_tts

        lang = _language_code(language)
        voice = self.voices.get(lang, self.voices["en"])

        async def save() -> None:
            await edge_tts.Communicate(text=text, voice=voice, rate=self.rate).save(str(output_path))

        asyncio.run(save())


class EspeakBackend:
    name = "espeak"
    extension = ".wav"

    def __init__(self, executable: Optional[str] = None) -> None:
        self.executable = executable or shutil.which("espeak-ng") or shutil.which("espeak") or ""
        if not self.executable:
            raise RuntimeError("未找到 espeak-ng/espeak。")
        self.speed = os.getenv("AUTODUB_LIGHT_TTS_SPEED", "175")
        listing = subprocess.run([self.executable, "--voices"], check=True,
                                 capture_output=True, text=True, timeout=15).stdout
        self.available = set()
        for line in listing.splitlines()[1:]:
            parts = line.split()
            if len(parts) >= 5:
                self.available.update((parts[1], parts[3], parts[4].split("/")[-1]))
        self.checked = set()
        self.signature = f"{self.name}:{self.executable}:{self.speed}:{sorted(self.available)}"

    def voice_for(self, language: str) -> str:
        lang = _language_code(language)
        choices = ("cmn", "zh", "zh-cmn") if lang == "zh" else (lang,)
        for voice in choices:
            if voice in self.available:
                return voice
        raise RuntimeError(f"eSpeak 无 {lang} 音色；可用音色: {sorted(self.available)}")

    def _generate(self, text: str, voice: str, output_path: Path) -> None:
        subprocess.run(
            [self.executable, "-v", voice, "-s", self.speed, "-w", str(output_path), "--stdin"],
            input=text, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            text=True, timeout=60,
        )

    def preflight(self, language: str) -> None:
        voice = self.voice_for(language)
        if voice not in self.checked:
            with tempfile.TemporaryDirectory(prefix="light_tts_check_") as directory:
                path = Path(directory) / "check.wav"
                self._generate("你好" if _language_code(language) == "zh" else "test", voice, path)
                if audio_duration(path) <= 0:
                    raise RuntimeError(f"eSpeak {voice} 最小合成自检失败")
            self.checked.add(voice)
            print(f"LightTTS preflight: voice={voice} PASS")

    def synthesize(self, text: str, language: str, output_path: Path) -> None:
        self.preflight(language)
        self._generate(text, self.voice_for(language), output_path)


def create_light_tts_backend(name: str = "auto") -> LightTTSBackend:
    selected = str(name or "auto").strip().lower()
    if selected == "edge-tts":
        return EdgeTTSBackend()
    if selected in {"espeak", "espeak-ng"}:
        return EspeakBackend()
    if selected != "auto":
        raise ValueError(f"不支持的轻量 TTS 后端: {name}")
    try:
        return EdgeTTSBackend()
    except RuntimeError:
        try:
            return EspeakBackend()
        except RuntimeError as exc:
            raise RuntimeError(
                "没有可用的轻量 TTS：请安装 edge-tts 或 espeak-ng，"
                "也可设置 AUTODUB_LIGHT_TTS_BACKEND。"
            ) from exc


class LightTTSSelector:
    """Generate candidate previews and choose the closest duration."""

    def __init__(
        self,
        work_dir: Any,
        *,
        backend: Optional[LightTTSBackend] = None,
        backend_name: str = "auto",
    ) -> None:
        self.root = Path(work_dir) / "light_tts_selector"
        self.audio_dir = self.root / "audio"
        self.results_path = self.root / "selection_results.json"
        self.backend = backend or create_light_tts_backend(backend_name)
        self.audio_dir.mkdir(parents=True, exist_ok=True)

    def _load(self) -> Dict[str, Any]:
        if not self.results_path.exists():
            return {"version": 1, "candidates": {}, "selections": {}}
        try:
            payload = json.loads(self.results_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"version": 1, "candidates": {}, "selections": {}}
        payload.setdefault("version", 1)
        payload.setdefault("candidates", {})
        payload.setdefault("selections", {})
        return payload

    def _fingerprint(
        self,
        segment_id: str,
        index: int,
        text: str,
        language: str,
        target_duration: float,
        available_duration: float,
    ) -> str:
        data = {
            "segment_id": segment_id,
            "candidate_index": index,
            "text": text,
            "language": language,
            "target_duration": round(target_duration, 6),
            "available_duration": round(available_duration, 6),
            "backend": self.backend.signature,
        }
        encoded = json.dumps(data, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _duration_fit(
        actual: float,
        target: float,
        available: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Judge speech duration against an ideal span and a hard time slot.

        Short speech can retain a natural tail pause.  Long speech is accepted
        only while it remains inside the segment's available mixing window.
        """
        actual, target = float(actual), max(0.4, float(target))
        available = max(target, float(available if available is not None else target))
        delta = actual - target
        absolute = abs(delta)
        relative = absolute / max(0.4, target)
        if delta <= 0:
            ratio = actual / target
            if ratio >= 0.72 or absolute <= 0.70:
                status = "suitable"
            elif ratio >= 0.50 or absolute <= 1.10:
                status = "borderline"
            else:
                status = "rejected"
        else:
            slot_overrun = actual - available
            if slot_overrun > max(0.12, 0.05 * available):
                status = "rejected"
            elif absolute <= 0.25 or relative <= 0.10:
                status = "suitable"
            elif actual <= available and relative <= 0.30:
                status = "borderline"
            else:
                status = "rejected"
        return {
            "duration_fit": status,
            "duration_error": absolute,
            "duration_relative_error": relative,
        }

    @staticmethod
    def _speech_window(segment: Mapping[str, Any]) -> Tuple[float, float]:
        """Return (ideal speech duration, maximum available segment duration)."""
        start = float(segment["start"])
        end = float(segment["end"])
        available = max(0.4, end - start)
        starts: List[float] = []
        ends: List[float] = []
        for item in segment.get("qwen3_time_stamps", []) or []:
            if not isinstance(item, Mapping):
                continue
            try:
                word_start = float(item.get("start"))
                word_end = float(item.get("end", word_start))
            except (TypeError, ValueError):
                continue
            if word_end < start - 0.25 or word_start > end + 0.25:
                continue
            starts.append(max(start, word_start))
            ends.append(min(end, max(word_start, word_end)))
        if not starts or not ends:
            return available, available
        # Small edge allowance preserves natural consonant release without
        # treating all diarization padding as speech that TTS must fill.
        ideal = min(available, max(0.4, max(ends) - min(starts) + 0.12))
        return ideal, available

    def _choose_candidate(
        self,
        successful: List[Dict[str, Any]],
        candidates: List[Dict[str, Any]],
        target_duration: float,
        *,
        measured: bool,
        available_duration: Optional[float] = None,
    ) -> Tuple[Dict[str, Any], str]:
        """Prefer trusted OCR only when measured speech duration remains usable."""
        for row in successful:
            row.update(self._duration_fit(
                row["actual_duration"], target_duration, available_duration
            ))
        closest = min(
            successful,
            key=lambda row: (float(row["duration_error"]), int(row["candidate_index"])),
        )
        if not measured:
            return closest, "closest_estimated_duration"
        direct = [
            row for row in successful
            if candidates[int(row["candidate_index"])].get("_selection_priority")
            in {"ocr_direct", "ocr_high"}
        ]
        repaired = [
            row for row in successful
            if candidates[int(row["candidate_index"])].get("_selection_priority")
            in {"ocr_repaired", "ocr_hybrid"}
        ]
        standard = [
            row for row in successful
            if candidates[int(row["candidate_index"])].get("_selection_priority") == "standard"
        ]
        excluded_indices = set()
        for trusted, label in ((direct, "ocr_direct"), (repaired, "ocr_assisted")):
            if not trusted:
                continue
            candidate = min(trusted, key=lambda row: float(row["duration_error"]))
            if candidate["duration_fit"] == "suitable":
                return candidate, f"{label}_duration_suitable"
            if candidate["duration_fit"] == "rejected":
                excluded_indices.update(int(row["candidate_index"]) for row in trusted)

        # Borderline means safely adjustable, not unusable.  Once OCR is
        # source-aligned, semantically validated, and inside the hard slot,
        # preserve the human subtitle instead of preferring a marginally
        # closer machine translation.
        for trusted, label in ((direct, "ocr_direct"), (repaired, "ocr_assisted")):
            borderline = [row for row in trusted if row["duration_fit"] == "borderline"]
            if borderline:
                return min(
                    borderline, key=lambda row: float(row["duration_error"])
                ), f"{label}_duration_adjustable"

        # Once timing is usable, preserve the standard natural translation
        # instead of selecting a more verbose variant merely because it fills
        # a little more of the source window.
        standard_suitable = [row for row in standard if row["duration_fit"] == "suitable"]
        if standard_suitable:
            return min(
                standard_suitable, key=lambda row: float(row["duration_error"])
            ), "standard_duration_suitable"

        standard_borderline = [row for row in standard if row["duration_fit"] == "borderline"]
        if standard_borderline:
            return min(
                standard_borderline, key=lambda row: float(row["duration_error"])
            ), "standard_duration_borderline"

        available = [
            row for row in successful
            if int(row["candidate_index"]) not in excluded_indices
        ] or successful
        closest_available = min(
            available,
            key=lambda row: (float(row["duration_error"]), int(row["candidate_index"])),
        )
        if (direct or repaired) and excluded_indices:
            return closest_available, "trusted_ocr_rejected_by_duration"
        if direct or repaired:
            return closest_available, "trusted_ocr_duration_outperformed"
        return closest_available, "closest_measured_duration"

    def select(
        self,
        segments: Iterable[Mapping[str, Any]],
        *,
        target_language: Any,
        force: bool = False,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        from translation_quality import require_translated
        segments = list(segments)
        require_translated(segments, str(target_language))
        if hasattr(self.backend, "preflight"):
            self.backend.preflight(str(target_language))
        state = self._load()
        output: List[Dict[str, Any]] = []
        selection_rows: List[Dict[str, Any]] = []
        for sequence, source in enumerate(segments):
            segment = dict(source)
            segment_id = str(segment.get("id", sequence))
            duration, available_duration = self._speech_window(segment)
            language = str(segment.get("target_lang_raw", target_language) or target_language)
            candidates = normalize_translation_candidates(
                segment.get("translations_candidates"), segment.get("text", "")
            )
            if not candidates:
                raise ValueError(f"segment {segment_id!r} 没有可用于轻量 TTS 的候选文本")
            successful: List[Dict[str, Any]] = []
            enriched: List[Dict[str, Any]] = []
            for index, candidate in enumerate(candidates):
                text = candidate["text"]
                cache_key = f"{segment_id}:{index}"
                fingerprint = self._fingerprint(
                    segment_id, index, text, language, duration, available_duration
                )
                output_path = self.audio_dir / (
                    f"segment_{_safe_id(segment_id)}_candidate_{index}{self.backend.extension}"
                )
                known = state["candidates"].get(cache_key, {})
                reusable = (
                    not force
                    and known.get("status") == "success"
                    and known.get("request_fingerprint") == fingerprint
                    and Path(str(known.get("audio_file", ""))).is_file()
                )
                if reusable:
                    try:
                        reusable = audio_duration(Path(known["audio_file"])) > 0
                    except Exception:
                        reusable = False
                if reusable:
                    record = dict(known)
                else:
                    try:
                        self.backend.synthesize(text, language, output_path)
                        actual = audio_duration(output_path)
                        if actual <= 0:
                            raise ValueError("empty TTS audio")
                        record = {
                            "segment_id": segment_id,
                            "candidate_index": index,
                            "text": text,
                            "target_duration": duration,
                            "available_duration": available_duration,
                            "actual_duration": actual,
                            "duration_error": abs(actual - duration),
                            "audio_file": str(output_path.resolve()),
                            "backend": self.backend.name,
                            "request_fingerprint": fingerprint,
                            "status": "success",
                            "error": "",
                        }
                    except Exception as exc:
                        record = {
                            "segment_id": segment_id,
                            "candidate_index": index,
                            "text": text,
                            "target_duration": duration,
                            "audio_file": str(output_path.resolve()),
                            "backend": self.backend.name,
                            "request_fingerprint": fingerprint,
                            "status": "failed",
                            "error": str(exc),
                        }
                    state["candidates"][cache_key] = record
                    _atomic_json(self.results_path, state)
                candidate_row = dict(candidate)
                candidate_row.update(
                    {
                        "light_tts_status": record.get("status"),
                        "light_tts_duration": record.get("actual_duration"),
                        "duration_error": record.get("duration_error"),
                        "preview_audio": record.get("audio_file"),
                        "light_tts_error": record.get("error", ""),
                    }
                )
                if record.get("status") == "success":
                    candidate_row.update(self._duration_fit(
                        record["actual_duration"], duration, available_duration
                    ))
                enriched.append(candidate_row)
                if record.get("status") == "success":
                    successful.append(record)
            if not successful:
                errors = [row.get("light_tts_error", "unknown error") for row in enriched]
                # Text estimate is explicit and never labelled as measured duration.
                for index, candidate in enumerate(candidates):
                    text = candidate["text"]
                    estimate = max(0.15, len(re.findall(r"[\u3400-\u9fff]", text)) / 4.0 +
                                   len(re.findall(r"[A-Za-z]+", text)) / 2.5)
                    successful.append({"candidate_index": index, "text": text,
                                       "actual_duration": estimate, "audio_file": None,
                                       "duration_error": abs(estimate-duration), "estimated": True})
                print(f"⚠️ segment {segment_id} LightTTS fallback: {errors}")
            selected, selection_reason = self._choose_candidate(
                successful,
                candidates,
                duration,
                measured=not any(row.get("estimated") for row in successful),
                available_duration=available_duration,
            )
            selected_index = int(selected["candidate_index"])
            for index, candidate in enumerate(enriched):
                candidate["selected"] = index == selected_index
            selection = {
                "segment_id": segment_id,
                "selected_index": selected_index,
                "selected_text": selected["text"],
                "target_duration": duration,
                "available_duration": available_duration,
                "actual_duration": selected["actual_duration"],
                "duration_error": selected["duration_error"],
                "duration_relative_error": selected.get("duration_relative_error"),
                "duration_fit": selected.get("duration_fit"),
                "selection_reason": selection_reason,
                "preview_audio": selected["audio_file"],
                "backend": self.backend.name,
                "status": "fallback" if selected.get("estimated") else "success",
                "duration_source": "text_estimate" if selected.get("estimated") else "measured_audio",
                "fallback_reason": "; ".join(r.get("light_tts_error", "") for r in enriched) if selected.get("estimated") else "",
            }
            state["selections"][segment_id] = selection
            _atomic_json(self.results_path, state)
            segment["text"] = selected["text"]
            # Candidate provenance is an internal decision aid and must not leak
            # into the frontend-facing segment schema.
            for candidate in enriched:
                candidate.pop("_selection_priority", None)
            segment["translations_candidates"] = enriched
            segment["light_tts_selection"] = selection
            segment["light_tts_status"] = selection["status"]
            segment["light_tts_fallback_reason"] = selection["fallback_reason"]
            output.append(segment)
            selection_rows.append(selection)
            print(
                f"[Translation {segment_id}] "
                f"source={candidates[selected_index].get('_selection_priority', 'standard')} "
                f"duration={float(selected['actual_duration']):.3f}/{duration:.3f}s "
                f"slot={available_duration:.3f}s "
                f"fit={selected.get('duration_fit')} selected={selection_reason}"
            )
        failures = sum(r["status"] == "fallback" for r in selection_rows)
        fraction = float(os.getenv("AUTODUB_LIGHT_TTS_MAX_FALLBACK_FRACTION", "0.5"))
        if failures >= 2 and failures / max(1, len(selection_rows)) > fraction:
            raise RuntimeError(f"LightTTS widespread failure: {failures}/{len(selection_rows)}; see {self.results_path}")
        print(f"LightTTS: measured={len(selection_rows)-failures} fallback={failures}")
        return output, selection_rows
