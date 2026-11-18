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

# Kyutai Pocket TTS 内置语言（无中文），其余语言走 edge-tts/espeak。
# 路由判定用短码；pocket-tts 的 load_model 需要下列全名。
POCKET_LANGUAGES = {"en", "fr", "de", "pt", "it", "es", "nl"}
POCKET_LANGUAGE_NAMES = {
    "en": "english",
    "fr": "french",
    "de": "german",
    "pt": "portuguese",
    "it": "italian",
    "es": "spanish",
    "nl": "dutch",
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


def classify_coarse_recovery_duration(
    *,
    actual: float,
    target: float,
    available: float,
    measured: bool,
    minimum_slot_ratio: float = 0.65,
    maximum_slot_ratio: float = 1.50,
) -> Dict[str, Any]:
    """Classify recovered TTS against its physical slot, not word timing.

    ``target`` remains diagnostic because it is useful for natural pacing, but
    it must not decide whether audio physically fits a coarse recovery window.
    """
    actual = max(0.0, float(actual))
    target = max(0.0, float(target))
    available = max(0.0, float(available))
    target_ratio = actual / target if actual > 0 and target > 0 else 0.0
    slot_ratio = actual / available if actual > 0 and available > 0 else 0.0
    if not measured:
        status = "suitable"
    elif slot_ratio < float(minimum_slot_ratio):
        status = "shorter_than_window"
    elif slot_ratio > float(maximum_slot_ratio):
        status = "too_long_blocking"
    else:
        status = "suitable"
    return {
        "status": status,
        "target_ratio": round(target_ratio, 4),
        "slot_ratio": round(slot_ratio, 4),
    }


def _language_code(value: Any) -> str:
    language = str(value or "").strip().lower().replace("_", "-")
    aliases = {"chinese": "zh", "mandarin": "zh", "english": "en"}
    language = aliases.get(language, language)
    return language.split("-", 1)[0] or "en"


def _routes_to_pocket(language: Any) -> bool:
    return _language_code(language) in POCKET_LANGUAGES


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


class PocketTTSBackend:
    """Kyutai Pocket TTS (CPU, ~100M) as a light-TTS duration preview backend.

    按语言惰性加载模型并缓存，``generate_audio`` 返回 24kHz 的 PCM 张量。
    只对 ``POCKET_LANGUAGES`` 里的内置语言路由到这里；中文等其余语言仍走
    edge-tts/espeak（见 ``LightTTSSelector._backend_for``）。
    """

    name = "pocket-tts"
    extension = ".wav"

    def __init__(self, voice: Optional[str] = None) -> None:
        self.voice = voice or os.getenv("AUTODUB_POCKET_TTS_VOICE", "alba")
        self.signature = f"{self.name}:{self.voice}"
        self._models: Dict[str, Any] = {}
        self._voices: Dict[str, Any] = {}
        self._checked: set = set()

    @staticmethod
    def _pocket_language(language: str) -> str:
        lang = _language_code(language)
        if lang not in POCKET_LANGUAGES:
            raise RuntimeError(
                f"Pocket TTS 不支持语言 {lang!r}（可用: {sorted(POCKET_LANGUAGES)}）"
            )
        return lang

    def _model(self, lang: str) -> Any:
        if lang not in self._models:
            # huggingface_hub 在 import 时读取 HF_ENDPOINT；内网机器可设置
            # AUTODUB_POCKET_HF_ENDPOINT（例如 https://hf-mirror.com）走镜像。
            mirror = os.getenv("AUTODUB_POCKET_HF_ENDPOINT", "").strip()
            if mirror and not os.getenv("HF_ENDPOINT"):
                os.environ["HF_ENDPOINT"] = mirror
            try:
                from pocket_tts import TTSModel
            except ImportError as exc:
                raise RuntimeError(
                    "pocket-tts 未安装；请 pip install pocket-tts（需 torch），"
                    "或设置 AUTODUB_LIGHT_TTS_BACKEND=edge-tts/espeak 跳过。"
                ) from exc
            self._models[lang] = TTSModel.load_model(
                language=POCKET_LANGUAGE_NAMES[lang]
            )
        return self._models[lang]

    def _voice(self, lang: str) -> Any:
        if lang not in self._voices:
            self._voices[lang] = self._model(lang).get_state_for_audio_prompt(self.voice)
        return self._voices[lang]

    def preflight(self, language: str) -> None:
        lang = self._pocket_language(language)
        if lang in self._checked:
            return
        model = self._model(lang)
        voice = self._voice(lang)
        audio = model.generate_audio(voice, "test")
        if audio is None or (hasattr(audio, "numel") and audio.numel() == 0):
            raise RuntimeError(f"Pocket TTS {lang} 最小合成自检失败")
        self._checked.add(lang)
        print(f"Pocket TTS preflight: lang={lang} voice={self.voice} PASS")

    def synthesize(self, text: str, language: str, output_path: Path) -> None:
        lang = self._pocket_language(language)
        model = self._model(lang)
        voice = self._voice(lang)
        audio = model.generate_audio(voice, text)
        self._write_wav(audio, int(getattr(model, "sample_rate", 24000)), output_path)

    @staticmethod
    def _write_wav(audio: Any, sample_rate: int, output_path: Path) -> None:
        import numpy as np

        if hasattr(audio, "detach"):
            samples = np.asarray(audio.detach().cpu().numpy())
        else:
            samples = np.asarray(audio)
        pcm = np.clip(samples.reshape(-1).astype(np.float32), -1.0, 1.0)
        pcm = (pcm * 32767.0).astype(np.int16)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(output_path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            handle.writeframes(pcm.tobytes())


def create_light_tts_backend(name: str = "auto") -> LightTTSBackend:
    selected = str(name or "auto").strip().lower()
    if selected in {"pocket-tts", "pocket"}:
        return PocketTTSBackend()
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
                "没有可用的轻量 TTS：请安装 edge-tts、espeak-ng 或 pocket-tts，"
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
        self.backend = backend  # 显式传入时绕过按语言路由（测试/回退兼容）
        self.backend_name = str(backend_name or "auto").strip().lower()
        self._routed_pocket: Optional[LightTTSBackend] = None
        self._routed_fallback: Optional[LightTTSBackend] = None
        self._pocket_disabled = False
        self.audio_dir.mkdir(parents=True, exist_ok=True)

    def _single_backend(self) -> LightTTSBackend:
        if self._routed_fallback is None:
            self._routed_fallback = create_light_tts_backend(self.backend_name)
        return self._routed_fallback

    def _pocket_backend(self) -> LightTTSBackend:
        if self._routed_pocket is None:
            self._routed_pocket = PocketTTSBackend()
        return self._routed_pocket

    def _backend_for(self, language: str) -> LightTTSBackend:
        """按语言选择轻量后端：Pocket TTS 覆盖内置语言，其余走 edge/espeak。"""
        if self.backend is not None:
            backend = self.backend
        elif self.backend_name not in {"", "auto"}:
            backend = self._single_backend()
        elif _routes_to_pocket(language) and not self._pocket_disabled:
            try:
                backend = self._pocket_backend()
                if hasattr(backend, "preflight"):
                    backend.preflight(str(language))
                return backend
            except Exception as exc:
                print(f"⚠️ Pocket TTS 不可用，回退 edge/espeak: {exc}")
                self._pocket_disabled = True
                backend = self._single_backend()
        else:
            backend = self._single_backend()
        if hasattr(backend, "preflight"):
            backend.preflight(str(language))
        return backend

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
        backend: LightTTSBackend,
    ) -> str:
        data = {
            "segment_id": segment_id,
            "candidate_index": index,
            "text": text,
            "language": language,
            "target_duration": round(target_duration, 6),
            "available_duration": round(available_duration, 6),
            "backend": backend.signature,
        }
        encoded = json.dumps(data, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _duration_fit(
        actual: float,
        target: float,
        available: Optional[float] = None,
        provenance: str = "",
    ) -> Dict[str, Any]:
        """Judge speech duration against an ideal span and a hard time slot.

        Short speech can retain a natural tail pause.  Long speech is accepted
        only while it remains inside the segment's available mixing window.
        """
        actual, target = float(actual), max(0.4, float(target))
        available = max(target, float(available if available is not None else target))
        delta = actual - target
        absolute = abs(delta)
        ratio = actual / target if target > 0 else 0.0
        relative = abs(ratio - 1.0)
        required_speed = actual / available if available > 0 else None
        trusted_ocr = str(provenance or "").lower() in {
            "ocr_direct", "ocr_repaired", "ocr_hybrid", "ocr_assisted",
        }
        if trusted_ocr:
            lower = float(os.getenv("AUTODUB_LIGHT_TTS_OCR_MIN_RATIO", "0.68"))
            upper = float(os.getenv("AUTODUB_LIGHT_TTS_OCR_MAX_RATIO", "1.35"))
        elif target < 1.0:
            lower = float(os.getenv("AUTODUB_LIGHT_TTS_TINY_MIN_RATIO", "0.60"))
            upper = float(os.getenv("AUTODUB_LIGHT_TTS_TINY_MAX_RATIO", "1.40"))
        else:
            lower = float(os.getenv("AUTODUB_LIGHT_TTS_MIN_RATIO", "0.75"))
            upper = float(os.getenv("AUTODUB_LIGHT_TTS_MAX_RATIO", "1.25"))
        borderline_lower = max(0.05, lower - 0.15)
        borderline_upper = upper + 0.15
        if actual > available:
            # A wider semantic/naturalness band never expands the physical
            # slot.  Only a modest compression remains eligible here.
            maximum_speed = float(os.getenv(
                "AUTODUB_LIGHT_TTS_OCR_MAX_SPEED"
                if trusted_ocr else "AUTODUB_LIGHT_TTS_MAX_SPEED",
                "1.15" if trusted_ocr else "1.10",
            ))
            status = (
                "speed_adjustable"
                if required_speed is not None and required_speed <= maximum_speed
                else "rejected"
            )
        elif lower <= ratio <= upper:
            status = "suitable"
        elif borderline_lower <= ratio <= borderline_upper:
            status = "borderline"
        else:
            status = "rejected"
        return {
            "duration_fit": status,
            "duration_error": absolute,
            "duration_relative_error": relative,
            "duration_ratio": ratio,
            "duration_lower_ratio": lower,
            "duration_upper_ratio": upper,
            "duration_provenance": str(provenance or "standard"),
            "required_speed_factor": required_speed,
        }

    @staticmethod
    def _speech_window(segment: Mapping[str, Any]) -> Tuple[float, float]:
        """Return (ideal speech duration, maximum available segment duration)."""
        start = float(segment["start"])
        end = float(segment["end"])
        available = max(0.4, end - start)
        try:
            explicit_start = float(segment.get("tts_placement_start"))
            explicit_target = float(segment.get("tts_target_duration"))
            explicit_hard_end = float(
                segment.get(
                    "tts_hard_end",
                    segment.get("tts_placement_end"),
                )
            )
        except (TypeError, ValueError):
            explicit_start = explicit_target = explicit_hard_end = 0.0
        if (
            start <= explicit_start < explicit_hard_end <= end + 0.001
            and explicit_target > 0.0
        ):
            hard_duration = max(0.4, explicit_hard_end - explicit_start)
            return min(max(0.4, explicit_target), hard_duration), hard_duration
        # Coarse recovery explicitly means that only the outer acoustic bounds
        # are trusted.  Speaker alignment can carry stale/collapsed Qwen word
        # timestamps into a copied segment; using those values would turn a
        # valid five-second slot into a fictitious 0.44-second target.
        if segment.get("timestamp_precision") == "coarse":
            try:
                placement_start = float(segment.get("tts_placement_start"))
                placement_end = float(segment.get("tts_placement_end"))
                available_end = float(segment.get("tts_available_end", end))
            except (TypeError, ValueError):
                placement_start = placement_end = available_end = 0.0
            if (
                start <= placement_start < placement_end <= end + 0.001
                and placement_start < available_end <= end + 0.001
            ):
                ideal = max(0.4, placement_end - placement_start)
                # Step 5 enforces placement_end as a hard collision boundary;
                # a translation must therefore fit this acoustic island, not
                # consume later silence or a later island in the outer span.
                return ideal, ideal
            return available, available
        try:
            mix_start = float(segment.get("mix_start", segment.get("speech_start")))
            mix_end = float(segment.get("mix_end", segment.get("speech_end")))
        except (TypeError, ValueError):
            mix_start = mix_end = 0.0
        if start <= mix_start < mix_end <= end + 0.001:
            available = max(0.4, mix_end - mix_start)
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
            priority = candidates[int(row["candidate_index"])].get(
                "_selection_priority", "standard"
            )
            row.update(self._duration_fit(
                row["actual_duration"], target_duration, available_duration,
                priority,
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
        native_standard = [
            row for row in successful
            if (
                candidates[int(row["candidate_index"])].get(
                    "_selection_priority"
                ) == "native_standard"
                or candidates[int(row["candidate_index"])].get(
                    "_candidate_role"
                ) == "native_standard"
            )
        ]
        compact = [
            row for row in successful
            if candidates[int(row["candidate_index"])].get(
                "_candidate_role"
            ) == "faithful_compact"
        ]
        standard = [
            row for row in successful
            if candidates[int(row["candidate_index"])].get("_selection_priority")
            in {"native_standard", "faithful_variant", "standard"}
        ]
        # Audit-failed/unavailable OCR remains in diagnostics but is never a
        # selectable dubbing candidate.
        reference_only = [
            row for row in successful
            if candidates[int(row["candidate_index"])].get("_selection_priority")
            == "ocr_reference"
        ]
        excluded_indices = {
            int(row["candidate_index"]) for row in reference_only
        }
        excluded_indices.update(
            int(row["candidate_index"])
            for row in standard
            if row["duration_fit"] == "rejected"
        )
        # Deterministic provenance order: exact OCR remains ahead of repaired
        # OCR, including when it needs no more than 10% downstream atempo.
        trusted_tiers = (
            (direct, "ocr_direct", "suitable"),
            (direct, "ocr_direct", "speed_adjustable"),
            (repaired, "ocr_assisted", "suitable"),
            (repaired, "ocr_assisted", "speed_adjustable"),
            (direct, "ocr_direct", "borderline"),
            (repaired, "ocr_assisted", "borderline"),
        )
        for trusted, label, fit in trusted_tiers:
            usable = [row for row in trusted if row["duration_fit"] == fit]
            if usable:
                return min(
                    usable, key=lambda row: float(row["duration_error"])
                ), f"{label}_duration_{fit}"
        for row in direct + repaired:
            if row["duration_fit"] == "rejected":
                excluded_indices.add(int(row["candidate_index"]))

        # Once timing is usable, preserve the standard natural translation
        # instead of selecting a more verbose variant merely because it fills
        # a little more of the source window.
        native_suitable = [
            row for row in native_standard if row["duration_fit"] == "suitable"
        ]
        if native_suitable:
            return min(
                native_suitable, key=lambda row: float(row["duration_error"])
            ), "native_standard_duration_suitable"

        # A natural standard line remains preferable inside the explicit
        # ratio tolerance. Duration is a soft tie-breaker here; it must not
        # silently replace good dialogue with a flatter compact paraphrase.
        for fit, reason in (
            ("borderline", "native_standard_duration_borderline"),
            ("speed_adjustable", "native_standard_bounded_speed"),
        ):
            usable_native = [
                row for row in native_standard if row["duration_fit"] == fit
            ]
            if usable_native:
                return min(
                    usable_native, key=lambda row: float(row["duration_error"])
                ), reason

        compact_suitable = [
            row for row in compact if row["duration_fit"] == "suitable"
        ]
        if compact_suitable:
            return min(
                compact_suitable, key=lambda row: float(row["duration_error"])
            ), "faithful_compact_after_native_rejected"

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
        from translation_quality import is_skipped, require_translated
        segments = list(segments)
        require_translated(segments, str(target_language))
        segments = [segment for segment in segments if not is_skipped(segment)]
        if not segments:
            return [], []
        state = self._load()
        output: List[Dict[str, Any]] = []
        selection_rows: List[Dict[str, Any]] = []
        for sequence, source in enumerate(segments):
            segment = dict(source)
            segment_id = str(segment.get("id", sequence))
            duration, available_duration = self._speech_window(segment)
            language = str(segment.get("target_lang_raw", target_language) or target_language)
            backend = self._backend_for(language)
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
                    segment_id, index, text, language, duration, available_duration, backend
                )
                output_path = self.audio_dir / (
                    f"segment_{_safe_id(segment_id)}_candidate_{index}{backend.extension}"
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
                        backend.synthesize(text, language, output_path)
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
                            "backend": backend.name,
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
                            "backend": backend.name,
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
                        record["actual_duration"], duration, available_duration,
                        candidate.get("_selection_priority", "standard"),
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
                "required_speed_factor": selected.get("required_speed_factor"),
                "duration_fit": selected.get("duration_fit"),
                "selection_reason": selection_reason,
                "selected_candidate_role": candidates[selected_index].get(
                    "_candidate_role", "unknown"
                ),
                "preview_audio": selected["audio_file"],
                "backend": backend.name,
                "status": "fallback" if selected.get("estimated") else "success",
                "duration_source": "text_estimate" if selected.get("estimated") else "measured_audio",
                "fallback_reason": "; ".join(r.get("light_tts_error", "") for r in enriched) if selected.get("estimated") else "",
            }
            state["selections"][segment_id] = selection
            _atomic_json(self.results_path, state)
            segment["text"] = selected["text"]
            segment["translation_selection_source"] = str(
                candidates[selected_index].get("_selection_priority", "standard")
            )
            segment["translation_candidate_role"] = str(
                candidates[selected_index].get("_candidate_role", "unknown")
            )
            segment["translation_duration_ratio"] = selected.get("duration_ratio")
            segment["translation_duration_ratio_bounds"] = [
                selected.get("duration_lower_ratio"),
                selected.get("duration_upper_ratio"),
            ]
            # Candidate provenance is an internal decision aid and must not leak
            # into the frontend-facing segment schema.
            for candidate in enriched:
                candidate.pop("_selection_priority", None)
                candidate.pop("_candidate_role", None)
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
            print(
                f"⚠️ LightTTS 大面积测量失败: {failures}/{len(selection_rows)}；"
                f"已使用文本时长估算继续，详见 {self.results_path}"
            )
        print(f"LightTTS: measured={len(selection_rows)-failures} fallback={failures}")
        return output, selection_rows
