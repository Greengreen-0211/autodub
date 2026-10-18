#!/usr/bin/env python3
"""Multi-model TTS router and AutoDub adapter.

The router groups work by model and starts one isolated Conda worker per model.
Each worker loads its model once and checkpoints every segment immediately.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import wave
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from scheme3_policy import (
    SCHEME3_ENGINES,
    normalize_emotion,
    normalize_language,
    select_scheme3_tts_engine,
)
from tts_preflight import cache_environment, offline_mode, preflight_models
from tts_quality import duration_quality, electricity_quality


MODEL_DISPLAY = {
    "f5tts": "F5-TTS",
    "indextts2": "IndexTTS2",
    "cosyvoice3": "CosyVoice3",
    "confucius4": "Confucius4-TTS",
    "omnivoice": "OmniVoice",
}
MODEL_ENVS = {
    "f5tts": "f5-tts",
    "indextts2": "indextts",
    "cosyvoice3": "cosyvoice3",
    "confucius4": "confucius4_tts",
    "omnivoice": "omnivoice",
}
MODEL_ORDER = ("f5tts", "indextts2", "cosyvoice3", "confucius4", "omnivoice")


def _is_precise_reference_segment(segment: Dict[str, Any]) -> bool:
    """Whether a segment has audio/text alignment safe for voice reference."""
    return not (
        str(segment.get("timestamp_precision", "")).lower() == "coarse"
        or str(segment.get("asr_recovery_status", "")).lower()
        == "accepted_coarse"
    )


def _language_code(language: Any) -> str:
    value = str(language or "").strip().lower().replace("_", "-")
    aliases = {"chinese": "zh", "mandarin": "zh", "english": "en"}
    value = aliases.get(value, value)
    return value.split("-", 1)[0]


def select_tts_model(language: Any, emotion: Any) -> str:
    """The fixed, user-approved scheme-two routing rule."""
    lang = _language_code(language)
    normalized_emotion = str(emotion or "neutral").strip().lower()
    if lang == "zh":
        return "cosyvoice3"
    if lang == "en":
        if normalized_emotion == "neutral":
            return "confucius4"
        return "cosyvoice3"
    return "omnivoice"


def resolve_scheme3_engine(
    segment: Dict[str, Any],
    target_language: str,
    emotion: str,
    text: str,
) -> tuple[str, str, bool]:
    """Resolve Step 4 engine while preserving the Step 3 routing decision."""
    expected = select_scheme3_tts_engine(target_language, emotion, text)
    if "tts_engine" not in segment:
        return expected, expected, True
    saved = segment["tts_engine"]
    engine = saved if isinstance(saved, str) else ""
    if engine not in SCHEME3_ENGINES:
        raise ValueError(
            f"segment {segment.get('id', '<unknown>')!r} has invalid tts_engine: {saved!r}"
        )
    return engine, expected, False


def _atomic_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _load_results(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return {str(item["segment_id"]): item for item in payload}
    return {str(key): value for key, value in payload.items()}


def _wav_duration(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as handle:
            return handle.getnframes() / float(handle.getframerate())
    except Exception:
        from pydub import AudioSegment
        return len(AudioSegment.from_file(path)) / 1000.0


def _safe_id(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _request_fingerprint(item: Dict[str, Any]) -> str:
    payload = {
        key: item.get(key)
        for key in (
            "text", "language", "target_language", "emotion", "speaker",
            "ref_audio", "ref_text", "target_duration", "model",
        )
    }
    payload["version"] = "tts-request-v2.2.1"
    payload["worker_sha256"] = hashlib.sha256(Path(__file__).with_name("tts_worker.py").read_bytes()).hexdigest()
    payload["policy_sha256"] = hashlib.sha256(Path(__file__).with_name("scheme3_policy.py").read_bytes()).hexdigest()
    configuration_keys = {
        "f5tts":("F5TTS_REPO","F5TTS_MODEL_PATH","F5TTS_VOCODER_LOCAL_PATH"),
        "indextts2":("INDEX_TTS_ROOT","INDEX_TTS_MODEL_DIR"),
        "cosyvoice3":("COSYVOICE_REPO","COSYVOICE_MODEL_DIR","AUTODUB_COSY_ONNX_PROVIDER"),
        "confucius4":("CONFUCIUS4_REPO",), "omnivoice":("OMNIVOICE_MODEL_DIR",),
    }
    payload["model_configuration"] = {key:os.getenv(key, "") for key in configuration_keys.get(item.get("model"),())}
    payload["cache"] = cache_environment(str(item.get("model") or "f5tts"))
    payload["duration_limits"] = [os.getenv("AUTODUB_TTS_MAX_DURATION_ERROR", "0.6"),
                                  os.getenv("AUTODUB_TTS_MAX_DURATION_RATIO_ERROR", "0.25")]
    ref_audio = Path(str(item.get("ref_audio") or ""))
    if ref_audio.is_file():
        digest = hashlib.sha256()
        with ref_audio.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        payload["ref_audio_sha256"] = digest.hexdigest()
    else:
        payload["ref_audio_sha256"] = ""
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tts_timing(segment: Dict[str, Any]) -> tuple[float, float]:
    """Return TTS placement and duration without mutating public boundaries."""
    start = float(segment["start"])
    end = float(segment["end"])
    legacy = (start, max(0.0, end - start))
    if not (
        str(segment.get("asr_recovery_status") or "") == "accepted_coarse"
        and str(segment.get("timestamp_precision") or "") == "coarse"
    ):
        return legacy
    try:
        placement_start = float(segment.get("tts_placement_start"))
        placement_end = float(segment.get("tts_placement_end"))
    except (TypeError, ValueError):
        return legacy
    if not (start <= placement_start < placement_end <= end + 0.001):
        return legacy
    return placement_start, placement_end - placement_start


def _file_sha256(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atempo_factor(current: float, target: float) -> tuple[float, bool, float]:
    """Time-scale factor with a configurable safety range (变速限制).

    Returns ``(factor, capped, raw)``. ``capped`` is True only when a
    speed-up was limited (``raw > maximum``): that direction produces
    unnaturally fast speech, so the resulting over-length clip is accepted
    rather than re-stretched. A slow-down clamp (``raw < minimum``) is not
    flagged as ``capped`` because an under-length clip is a genuine synthesis
    failure that should still trigger the duration retry.
    """
    if target <= 0.1 or abs(current - target) < 0.2:
        return 1.0, False, 1.0
    minimum = float(os.getenv("AUTODUB_TTS_ATEMPO_MIN", "0.8"))
    maximum = float(os.getenv("AUTODUB_TTS_ATEMPO_MAX", "1.25"))
    # ffmpeg's single atempo filter only accepts [0.5, 2.0]; clamp defensively.
    minimum = max(0.5, min(minimum, 2.0))
    maximum = max(0.5, min(maximum, 2.0))
    if minimum > maximum:
        minimum, maximum = maximum, minimum
    raw = current / target
    capped = raw > maximum
    return max(minimum, min(raw, maximum)), capped, raw


def _apply_atempo(source: Path, destination: Path, target_duration: float) -> Path:
    current = _wav_duration(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    factor, _, _ = _atempo_factor(current, target_duration)
    if factor == 1.0:
        shutil.copy2(source, destination)
        return destination
    temporary = destination.with_name(destination.stem + ".tmp.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(source), "-filter:a", f"atempo={factor:.6f}", "-vn", str(temporary)],
        check=True,
    )
    os.replace(temporary, destination)
    return destination


def _trim_silence(source: Path, destination: Path) -> Path:
    """Trim leading/trailing silence (TTS首尾静音).

    Leading silence delays the dub relative to the on-screen mouth movement;
    trailing silence needlessly extends the clip. A fully-silent clip is left
    untouched rather than dropped.
    """
    if not _env_bool("AUTODUB_TTS_TRIM_SILENCE", True):
        shutil.copy2(source, destination)
        return destination
    from pydub import AudioSegment
    from pydub import silence as pydub_silence
    clip = AudioSegment.from_wav(str(source))
    if len(clip) <= 300:
        shutil.copy2(source, destination)
        return destination
    threshold = float(os.getenv("AUTODUB_TTS_SILENCE_THRESH_DBFS", "-45"))
    min_len = int(os.getenv("AUTODUB_TTS_SILENCE_MIN_LEN_MS", "100"))
    pad = int(os.getenv("AUTODUB_TTS_SILENCE_PAD_MS", "30"))
    ranges = pydub_silence.detect_nonsilent(clip, min_silence_len=min_len, silence_thresh=threshold)
    if not ranges:
        shutil.copy2(source, destination)
        return destination
    start = max(0, ranges[0][0] - pad)
    end = min(len(clip), ranges[-1][1] + pad)
    if end <= start:
        shutil.copy2(source, destination)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    clip[start:end].export(str(destination), format="wav")
    return destination


def _postprocess(source: Path, destination: Path, target_duration: float) -> Dict[str, Any]:
    """Trim silence then time-scale to target; return post-processing metrics."""
    trimmed = destination.with_name(destination.stem + ".trim.wav")
    try:
        pre_trim = _wav_duration(source)
        _trim_silence(source, trimmed)
        post_trim = _wav_duration(trimmed)
        factor, capped, raw = _atempo_factor(post_trim, target_duration)
        _apply_atempo(trimmed, destination, target_duration)
    finally:
        if trimmed.exists():
            trimmed.unlink(missing_ok=True)
    return {
        "pre_trim_duration": pre_trim,
        "post_trim_duration": post_trim,
        "atempo_factor": factor,
        "atempo_raw_factor": round(raw, 6),
        "atempo_capped": bool(capped),
        "actual_duration": _wav_duration(destination),
    }


def _worker_environment(model: str) -> Dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    environment["CUDA_VISIBLE_DEVICES"] = os.getenv("AUTODUB_TTS_GPU", "0")
    environment.pop("PYTHONPATH", None)
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
        environment.pop(key, None)
    environment.update(cache_environment(model))
    environment["HF_HUB_OFFLINE"] = "1" if offline_mode()=="strict_offline" else "0"
    environment["TRANSFORMERS_OFFLINE"] = environment["HF_HUB_OFFLINE"]
    environment["MODELSCOPE_OFFLINE"] = environment["HF_HUB_OFFLINE"]
    if model == "f5tts":
        repo = os.getenv("F5TTS_REPO", "/data0/goldenseed/cjx/F5-TTS")
        environment["PYTHONPATH"] = str(Path(repo) / "src")
    elif model == "indextts2":
        environment["INDEX_TTS_ROOT"] = os.getenv(
            "INDEX_TTS_ROOT", "/data0/goldenseed/models/index-tts"
        )
        environment["INDEX_TTS_MODEL_DIR"] = os.getenv(
            "INDEX_TTS_MODEL_DIR", "/data0/goldenseed/models/index-tts/checkpoints"
        )
    elif model == "confucius4":
        environment.pop("HF_ENDPOINT", None)
    return environment


def _run_worker(model: str, plan: Path, result: Path, force: bool = False) -> int:
    worker = Path(__file__).with_name("tts_worker.py")
    env_var = f"AUTODUB_{model.upper()}_ENV"
    env_name = os.getenv(env_var, MODEL_ENVS[model])
    conda = shutil.which("conda") or "/opt/anaconda3/bin/conda"
    command = [
        conda, "run", "--no-capture-output", "-n", env_name,
        "python", str(worker), "--model", model, "--plan", str(plan),
        "--result", str(result),
    ]
    if force:
        command.append("--force")
    print(f"启动 {MODEL_DISPLAY[model]} worker（模型仅加载一次）")
    return subprocess.run(command, env=_worker_environment(model), check=False).returncode


def synthesize(
    text: str,
    language: str,
    emotion: str,
    ref_audio: str,
    output_path: str,
    target_duration: Optional[float] = None,
    *,
    ref_text: str = "",
    force: bool = False,
    scheme: str = "scheme2",
) -> Dict[str, Any]:
    """Unified one-item API; the pipeline uses :class:`TTSRouter` for batching."""
    destination = Path(output_path).resolve()
    work = destination.parent / (destination.stem + "_router_work")
    raw = work / "raw.wav"
    plan = work / "plan.json"
    result = work / "results.json"
    if scheme not in {"scheme2", "scheme3"}:
        raise ValueError(f"unsupported TTS scheme: {scheme}")
    model = (
        select_scheme3_tts_engine(language, emotion, text)
        if scheme == "scheme3"
        else select_tts_model(language, emotion)
    )
    canonical_language = normalize_language(language)
    synthesis_language = _language_code(language) if model == "omnivoice" else canonical_language
    item = {
        "segment_id": "0", "text": text, "language": synthesis_language,
        "target_language": canonical_language,
        "emotion": normalize_emotion(emotion), "speaker": "SPEAKER_00",
        "ref_audio": str(Path(ref_audio).resolve()), "ref_text": ref_text,
        "raw_path": str(raw), "target_duration": float(target_duration or 0.0),
        "model": model, "tts_engine": model,
    }
    item["request_fingerprint"] = _request_fingerprint(item)
    preflight_models([model])
    _atomic_json(plan, {"items": [item]})
    code = _run_worker(model, plan, result, force=force)
    results = _load_results(result)
    metadata = results.get("0", {})
    if code != 0 or metadata.get("status") != "success":
        raise RuntimeError(metadata.get("error", f"{model} worker exited with {code}"))
    metadata.update(_postprocess(raw, destination, float(target_duration or 0.0)))
    metadata["file"] = str(destination)
    metadata["target_duration"] = float(target_duration or 0.0)
    metadata["duration_error"] = metadata["actual_duration"] - metadata["target_duration"] if target_duration else None
    metadata.update(electricity_quality(destination))
    _atomic_json(result, results)
    return metadata


class TTSRouter:
    def __init__(self, config: Any, *, scheme: str = "scheme2"):
        if scheme not in {"baseline", "scheme2", "scheme3"}:
            raise ValueError(f"unsupported TTS scheme: {scheme}")
        self.config = config
        self.scheme = scheme
        # Keep the already-deployed scheme-two checkpoint path unchanged.
        directory = "tts_router" if scheme == "scheme2" else ("tts_router_baseline" if scheme=="baseline" else "tts_router_scheme3")
        self.root = Path(config.temp_dir) / directory
        self.raw_dir = self.root / "raw"
        self.final_dir = self.root / "final"
        self.plan_path = self.root / "plan.json"
        self.result_path = self.root / "results.json"

    @staticmethod
    def _reference_score(segment: Dict[str, Any], clip: Any) -> float:
        from pydub import silence
        duration = max(0.0, float(segment["end"]) - float(segment["start"]))
        nonsilent = silence.detect_nonsilent(
            clip,
            min_silence_len=180,
            silence_thresh=(clip.dBFS - 18.0 if clip.dBFS != float("-inf") else -50.0),
        )
        speech_ratio = sum(end - start for start, end in nonsilent) / max(1, len(clip))
        duration_score = max(0.0, 1.0 - abs(duration - 5.0) / 8.0)
        loudness_score = max(0.0, min(1.0, (clip.dBFS + 50.0) / 35.0)) if clip.dBFS != float("-inf") else 0.0
        penalty = 0.5 if segment.get("speaker_boundary_ambiguous") else 0.0
        return 2.0 * speech_ratio + duration_score + loudness_score - penalty

    def _speaker_references(self, segments: List[Dict[str, Any]], vocal_path: str) -> Dict[str, Dict[str, str]]:
        from pydub import AudioSegment
        full_audio = AudioSegment.from_wav(vocal_path)
        references: Dict[str, Dict[str, str]] = {}
        speakers = sorted({str(segment.get("speaker", "SPEAKER_00")) for segment in segments})
        for speaker in speakers:
            candidates = []
            for index, segment in enumerate(segments):
                if str(segment.get("speaker", "SPEAKER_00")) != speaker:
                    continue
                # Coarse ASR recovery provides only an outer dialogue window.
                # Its audio may contain additional words, pauses, or another
                # turn that are absent from the recovered transcript.  Such a
                # clip is valid for dubbing, but unsafe as paired reference
                # audio/text for a voice-cloning model.
                if not _is_precise_reference_segment(segment):
                    continue
                start = max(0, int(round(float(segment["start"]) * 1000)))
                end = min(len(full_audio), int(round(float(segment["end"]) * 1000)))
                clip = full_audio[start:end]
                if len(clip) < 500:
                    continue
                candidates.append((self._reference_score(segment, clip), index, segment, clip))
            if not candidates:
                print(
                    f"⚠️ {speaker} 没有可用的精确边界参考音频；"
                    "粗边界恢复句仍不会用于声纹参考，该说话人将单独降级"
                )
                continue
            candidates.sort(key=lambda item: (item[0], len(item[3])), reverse=True)
            chosen = []
            total_ms = 0
            # Prefer complete short ASR spans: F5 clips references above 12s,
            # which otherwise leaves the full transcript paired with half audio.
            bounded = [candidate for candidate in candidates if len(candidate[3]) <= 10000]
            for candidate in bounded or candidates:
                if chosen and total_ms + len(candidate[3]) > 10000:
                    continue
                chosen.append(candidate)
                total_ms += len(candidate[3])
                if total_ms >= 10000:
                    break
            chosen.sort(key=lambda item: float(item[2]["start"]))
            audio = AudioSegment.empty()
            texts = []
            for _, _, segment, clip in chosen:
                audio += clip
                source_text = str(segment.get("original_text") or segment.get("qwen3_raw_text") or segment.get("text") or "").strip()
                if source_text:
                    texts.append(source_text)
            safe_speaker = _safe_id(speaker)
            ref_path = self.root / "references" / f"ref_{safe_speaker}.wav"
            txt_path = ref_path.with_suffix(".txt")
            ref_path.parent.mkdir(parents=True, exist_ok=True)
            audio.export(ref_path, format="wav")
            txt_path.write_text(" ".join(texts), encoding="utf-8")
            references[speaker] = {"audio": str(ref_path), "text": " ".join(texts)}
            print(f"参考音频 {speaker}: 候选 {len(candidates)}，选中 {len(chosen)}，{len(audio)/1000.0:.2f}s")
        return references

    def run(self, segments: Iterable[Dict[str, Any]], vocal_path: str, *, force: bool = False) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        from translation_quality import is_skipped, require_translated
        # Segments without any safe target-language text never reach a TTS
        # engine. Step 5 leaves their position silent instead of restoring the
        # source vocal, so the final mix cannot leak the source language.
        segment_list = [dict(segment) for segment in segments if not is_skipped(segment)]
        require_translated(segment_list, self.config.target_lang, getattr(self.config, "source_lang", ""))
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.final_dir.mkdir(parents=True, exist_ok=True)
        references = self._speaker_references(segment_list, vocal_path)
        previous = _load_results(self.result_path)
        items = []
        unreferenced_results = []
        seen = set()
        for index, segment in enumerate(segment_list):
            text = str(segment.get("text", "")).strip()
            segment_id = str(segment.get("id", index))
            if not text or segment_id in seen:
                raise ValueError(f"invalid/duplicate TTS segment: {segment_id}")
            seen.add(segment_id)
            speaker = str(segment.get("speaker", "SPEAKER_00"))
            if speaker not in references:
                failure = {
                    "segment_id": segment_id,
                    "status": "failed",
                    "error": "no_safe_precise_reference_audio",
                    "planned_engine": segment.get("tts_engine"),
                    "actual_engine": None,
                    "fallback_count": 0,
                }
                previous[segment_id] = failure
                unreferenced_results.append(failure)
                continue
            placement_start, duration = _tts_timing(segment)
            raw_emotion = segment.get("raw_emotion", segment.get("emotion_raw", segment.get("emotion")))
            if "emotion_reliable" in segment:
                emotion = normalize_emotion(segment.get("emotion")) if segment["emotion_reliable"] else "neutral"
            else:
                emotion = normalize_emotion(raw_emotion, score=segment.get("emotion_score"), duration=duration)
            target = normalize_language(segment.get("target_lang", self.config.target_lang))
            target_raw = str(segment.get("target_lang_raw", self.config.target_lang)).lower()
            if self.scheme == "scheme3":
                model, expected, missing = resolve_scheme3_engine(segment, target, emotion, text)
                if missing or model != expected:
                    print(f"⚠️ segment={segment_id} saved_engine={model} expected={expected} legacy_fallback={missing}")
            else:
                model = "indextts2" if self.scheme=="baseline" else select_tts_model(self.config.target_lang, emotion)
            job = {
                "segment_id": segment_id, "text": text, "planned_text": text,
                "language": _language_code(target_raw) if model=="omnivoice" else target,
                "target_language": target, "target_language_raw": target_raw,
                "source_language": normalize_language(segment.get("source_lang", getattr(self.config, "source_lang", "unknown"))),
                "emotion": emotion, "emotion_score": segment.get("emotion_score"),
                "speaker": speaker, "ref_audio": references[speaker]["audio"],
                "ref_text": references[speaker]["text"], "target_duration": duration,
                "start": placement_start, "model": model, "tts_engine": model,
                "placement_source": (
                    segment.get("recovery_timing_method")
                    if placement_start != float(segment["start"])
                    else "segment_start"
                ),
                "planned_engine": model, "fallback_count": 0, "fallback_reason": "",
                "translations_candidates": segment.get("translations_candidates", []),
                "attempted_engines": [model], "attempted_texts": [text],
            }
            self._job_paths(job)
            job["request_fingerprint"] = _request_fingerprint(job)
            job["planned_request_fingerprint"] = job["request_fingerprint"]
            items.append(job)

        active = []
        for item in items:
            known = previous.get(item["segment_id"], {})
            reused = False
            if (not force and known.get("status")=="success" and
                    not (getattr(self.config,"disable_tts_fallback",False) and known.get("actual_engine")!=item["planned_engine"]) and
                    known.get("planned_request_fingerprint")==item["planned_request_fingerprint"]):
                resumed = dict(item, model=known.get("actual_engine", item["model"]),
                               text=known.get("tts_text", item["text"]))
                resumed["language"] = _language_code(resumed["target_language_raw"]) if resumed["model"]=="omnivoice" else resumed["target_language"]
                self._job_paths(resumed)
                resumed["request_fingerprint"] = _request_fingerprint(resumed)
                try:
                    valid = known.get("request_fingerprint")==resumed["request_fingerprint"]
                    actual = _wav_duration(Path(known["file"])) if valid else 0
                    measured_quality = duration_quality(actual, item["target_duration"])["duration_quality"]
                    saved_quality = str(known.get("duration_quality", ""))
                    if (valid and known.get("audio_sha256")==_file_sha256(Path(known["file"])) and
                            (
                                measured_quality == "passed"
                                or saved_quality == "degraded"
                                or bool(known.get("atempo_capped"))
                            )):
                        item.update(resumed, fallback_count=known.get("fallback_count",0),
                                    fallback_reason=known.get("fallback_reason",""))
                        previous[item["segment_id"]] = dict(known)
                        reused = True
                except Exception:
                    pass
            if not reused:
                item["force_regenerate"] = True
                active.append(item)
        _atomic_json(self.result_path, previous)
        if active:
            self._execute(active, force=force)
            previous = _load_results(self.result_path)

        if not getattr(self.config, "disable_tts_fallback", False):
            # Planned engine plus at most two fallback engines. Duration
            # mismatch is not an engine failure and never enters this loop.
            for _ in range(2):
                retry = []
                for item in items:
                    result = previous.get(item["segment_id"], {})
                    if result.get("status")=="success":
                        continue
                    chain = self._fallback_chain(item["planned_engine"], item["target_language"])
                    model = next((m for m in chain if m not in item["attempted_engines"]), None)
                    if model is None:
                        continue
                    # No partial work in a fallback round until its model preflight passes.
                    item["fallback_reason"] += ("; " if item["fallback_reason"] else "") + str(result.get("error","worker_failed"))
                    item["fallback_count"] += 1
                    item["attempted_engines"].append(model)
                    item["model"] = model
                    item["tts_engine"] = model
                    item["language"] = _language_code(item["target_language_raw"]) if model=="omnivoice" else item["target_language"]
                    self._job_paths(item)
                    item["request_fingerprint"] = _request_fingerprint(item)
                    item["force_regenerate"] = True
                    retry.append(item)
                if not retry:
                    if any(previous.get(i["segment_id"],{}).get("status")!="success" and
                           any(m not in i["attempted_engines"] for m in self._fallback_chain(i["planned_engine"],i["target_language"])) for i in items):
                        continue
                    break
                self._execute(retry, force=True)
                previous = _load_results(self.result_path)

        clips, output_rows = [], []
        for item in items:
            sid = item["segment_id"]
            metadata = dict(previous.get(sid, {}))
            metadata.update(segment_id=sid, planned_engine=item["planned_engine"],
                            actual_engine=item["model"], tts_engine=item["model"],
                            model_key=item["model"], model=MODEL_DISPLAY[item["model"]],
                            selected_model=MODEL_DISPLAY[item["model"]],
                            fallback_count=item["fallback_count"], fallback_reason=item["fallback_reason"],
                            tts_text=item["text"], planned_text=item["planned_text"],
                            planned_request_fingerprint=item["planned_request_fingerprint"],
                            request_fingerprint=item["request_fingerprint"], target_duration=item["target_duration"],
                            generation_request={k:item[k] for k in ("text","language","target_language","emotion",
                                               "speaker","ref_audio","ref_text","target_duration","start","model")},
                            placement_start=item["start"],
                            placement_source=item.get("placement_source", "segment_start"),
                            speaker=item["speaker"], emotion=item["emotion"],
                            source_language=item["source_language"], target_language=item["target_language"],
                            target_language_raw=item["target_language_raw"])
            if metadata.get("status")=="success":
                clips.append({"segment_id":sid, "start":item["start"], "file":metadata["file"],
                              "tts_engine":item["model"], "request_fingerprint":item["request_fingerprint"]})
            output_rows.append(metadata)
            previous[sid] = metadata
            print(f"[TTS] segment={sid} planned={item['planned_engine']} actual={item['model']} "
                  f"target={item['target_duration']:.2f}s actual={float(metadata.get('actual_duration') or 0):.2f}s "
                  f"status={metadata.get('status','failed')} fallback={item['fallback_count']}")
        _atomic_json(self.result_path, previous)
        _atomic_json(self.plan_path, {"items":items})
        output_rows.extend(unreferenced_results)
        print("TTS quality failures:", [{"id":r["segment_id"], "error":r.get("error")} for r in output_rows if r.get("status")!="success"])
        return clips, output_rows

    def _job_paths(self, item):
        suffix = f"_{item['model']}"
        safe = _safe_id(item["segment_id"])
        item["raw_path"] = str(self.raw_dir/f"segment_{safe}{suffix}.wav")
        item["final_path"] = str(self.final_dir/f"segment_{safe}{suffix}.wav")

    @staticmethod
    def _fallback_chain(model, language):
        defaults = {"confucius4":["f5tts","indextts2"], "cosyvoice3":["f5tts"],
                    "indextts2":["f5tts"], "f5tts":["cosyvoice3"], "omnivoice":[]}
        raw = os.getenv("AUTODUB_TTS_FALLBACK_CHAINS", "")
        chains = json.loads(raw) if raw else defaults
        chain = chains.get(model, [])
        if not isinstance(chain, list) or any(m not in MODEL_ORDER for m in chain):
            raise ValueError(f"invalid fallback chain for {model}: {chain}")
        if language not in {"zh","en"} and any(m!="omnivoice" for m in chain):
            raise ValueError("non-Chinese/English fallback must stay on OmniVoice")
        # Planned engine plus at most two distinct fallbacks: three engines in
        # total, with no repeated execution of a deterministically failed one.
        return chain[:2]

    def _execute(self, items, *, force):
        # Each worker receives ONLY its pending jobs; raw reuse must match fingerprints.
        results = _load_results(self.result_path)
        for model in MODEL_ORDER:
            pending = [item for item in items if item["model"]==model]
            if not pending:
                continue
            try:
                preflight_models([model])
            except RuntimeError as exc:
                for item in pending:
                    results[item["segment_id"]] = {
                        "segment_id": item["segment_id"],
                        "status": "failed",
                        "error": str(exc),
                        "request_fingerprint": item["request_fingerprint"],
                    }
                _atomic_json(self.result_path, results)
                continue
            _atomic_json(self.plan_path, {"items":pending})
            for item in pending:
                results[item["segment_id"]] = {
                    "segment_id":item["segment_id"],"status":"pending","error":"",
                    "request_fingerprint":item["request_fingerprint"],
                }
            _atomic_json(self.result_path, results)
            code = _run_worker(model, self.plan_path, self.result_path, force=force)
            results = _load_results(self.result_path)
            for item in pending:
                sid = item["segment_id"]
                metadata = dict(results.get(sid, {}))
                if metadata.get("status")=="pending":
                    metadata.update(status="failed",error=f"{model} worker exited with {code}")
                if metadata.get("status")=="success":
                    try:
                        post = _postprocess(Path(item["raw_path"]), Path(item["final_path"]), item["target_duration"])
                        actual = post["actual_duration"]
                        metadata.update(post)
                        metadata.update(file=item["final_path"], actual_duration=actual,
                                        audio_sha256=_file_sha256(Path(item["final_path"])),
                                        **duration_quality(actual, item["target_duration"]))
                        # When atempo was deliberately capped to keep the voice
                        # natural, keep that fact in the audit metadata.
                        if metadata["atempo_capped"] and metadata["duration_quality"]=="failed":
                            metadata["duration_quality_note"] = "atempo capped: duration mismatch accepted to avoid unnatural speed"
                        metadata.update(electricity_quality(Path(item["final_path"])))
                        # 电音检测：block policy turns a detected artifact into a
                        # failure so the fallback chain can try another engine.
                        if (metadata.get("electricity_quality")=="failed" and
                                os.getenv("AUTODUB_TTS_ELECTRICITY_POLICY","warn").strip().lower()=="block"):
                            metadata.update(status="failed", error=metadata.get("electricity_reason","electric artifact detected"))
                        elif metadata["duration_quality"]=="failed":
                            # Keep the best generated clip instead of dropping
                            # the sentence.  The mismatch remains auditable and
                            # Step 5 can still place the usable audio.
                            metadata.update(
                                status="success",
                                duration_quality="degraded",
                                duration_degraded_reason=(
                                    f"best_available_duration: {actual:.3f}/{item['target_duration']:.3f}"
                                ),
                            )
                    except Exception as exc:
                        metadata.update(status="failed",error=f"duration adaptation failed: {exc}")
                results[sid] = metadata
            _atomic_json(self.result_path, results)
