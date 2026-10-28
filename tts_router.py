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
import sys
import time
import wave
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from scheme3_policy import (
    SCHEME3_ENGINES,
    normalize_emotion,
    normalize_language,
    select_scheme3_tts_engine,
)
from tts_preflight import cache_environment, offline_mode, preflight_models
from tts_quality import (
    audible_speech_quality,
    duration_adaptation_safety,
    duration_quality,
    electricity_quality,
    natural_duration_fit,
    voice_consistency_quality,
    voice_reference_similarity,
)


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
GENERATION_REQUEST_KEYS = (
    "text", "language", "target_language", "emotion", "speaker",
    "ref_audio", "ref_text", "target_duration", "start", "model",
    "reference_segment_id", "reference_purity_score",
    "measure_natural_duration",
    "translation_selection_source",
)


def _generation_attempt_limit() -> int:
    """Hard per-sentence synthesis budget for interactive Agent latency."""
    try:
        configured = int(os.getenv("AUTODUB_TTS_MAX_GENERATION_ATTEMPTS", "5"))
    except ValueError:
        configured = 5
    return max(1, min(5, configured))


def _trial_attempt_limit(requested: Optional[int] = None) -> int:
    """Bound a user-requested one-line preview to three generations."""
    if requested is None:
        try:
            requested = int(os.getenv("AUTODUB_TTS_TRIAL_MAX_ATTEMPTS", "3"))
        except ValueError:
            requested = 3
    return max(1, min(3, int(requested)))


def _trial_content_asr_enabled() -> bool:
    configured = os.getenv("AUTODUB_TTS_CONTENT_ASR_ENABLED")
    if configured is not None:
        return configured.strip().lower() in {"1", "true", "yes", "on"}
    model_dir = os.getenv("QWEN3_ASR_MODEL_DIR", "").strip()
    return bool(model_dir and Path(model_dir).is_dir())


def _trial_candidate_rank(candidate: Dict[str, Any]) -> tuple:
    """Rank only non-leaking preview candidates; lower is better."""
    target = float(candidate.get("target_duration") or 0.0)
    actual = float(candidate.get("actual_duration") or 0.0)
    if target > 0.0 and actual > 0.0:
        import math
        duration_distance = abs(math.log(actual / target))
    else:
        duration_distance = 0.0
    duration_rank = {
        "passed": 0,
        "degraded": 1,
        "failed": 2,
    }.get(str(candidate.get("duration_quality") or "unmeasured"), 1)
    return (
        int(candidate.get("content_completeness_quality") == "failed"),
        int(candidate.get("electricity_quality") == "failed"),
        duration_rank,
        duration_distance,
        int(candidate.get("trial_attempt") or 0),
    )


def _is_precise_reference_segment(segment: Dict[str, Any]) -> bool:
    """Whether a segment has audio/text alignment safe for voice reference."""
    if (
        str(segment.get("timestamp_precision", "")).lower() == "coarse"
        or str(segment.get("asr_recovery_status", "")).lower()
        == "accepted_coarse"
        or bool(segment.get("speaker_boundary_ambiguous"))
        or bool(segment.get("speaker_reference_mixed_raw"))
    ):
        return False
    return True


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


def _minimum_pronounceable_duration(text: Any, language: Any = "") -> float:
    """Return a conservative lower speech budget for complete target text.

    This is deliberately a floor rather than a duration predictor.  It keeps
    a collapsed ASR/VAD span from asking a TTS model to pronounce a complete
    line in an impossible fraction of a second, while the hard collision
    window remains the final upper bound.
    """
    value = str(text or "").strip()
    cjk = len(re.findall(r"[\u3400-\u9fff]", value))
    words = len(re.findall(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*", value))
    clauses = len([
        part for part in re.split(r"[，,。.!！？?；;：:]+", value)
        if any(char.isalnum() or "\u3400" <= char <= "\u9fff" for char in part)
    ])
    if cjk:
        seconds = cjk * float(
            os.getenv("AUTODUB_TTS_MIN_TARGET_SECONDS_PER_CJK", "0.18")
        )
    elif words:
        seconds = words * float(
            os.getenv("AUTODUB_TTS_MIN_TARGET_SECONDS_PER_WORD", "0.22")
        )
    else:
        seconds = len(value) * 0.08
    if clauses > 1:
        seconds += (clauses - 1) * float(
            os.getenv("AUTODUB_TTS_MIN_TARGET_SECONDS_PER_EXTRA_CLAUSE", "0.10")
        )
    return max(
        float(os.getenv("AUTODUB_TTS_MIN_TARGET_UTTERANCE_SECONDS", "0.35")),
        seconds,
    )


def _request_fingerprint(item: Dict[str, Any]) -> str:
    payload = {
        key: item.get(key)
        for key in (
            "text", "language", "target_language", "emotion", "speaker",
            "ref_audio", "ref_text", "target_duration", "model",
            "measure_natural_duration", "translation_selection_source",
        )
    }
    payload["version"] = "tts-request-v2.9.0-generation-only"
    payload["worker_sha256"] = hashlib.sha256(Path(__file__).with_name("tts_worker.py").read_bytes()).hexdigest()
    payload["policy_sha256"] = hashlib.sha256(Path(__file__).with_name("scheme3_policy.py").read_bytes()).hexdigest()
    payload["quality_sha256"] = hashlib.sha256(Path(__file__).with_name("tts_quality.py").read_bytes()).hexdigest()
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
    payload["duration_fit_policy"] = {
        "force_fit": os.getenv("AUTODUB_TTS_FORCE_FIT", "1"),
        "force_fit_tolerance": os.getenv(
            "AUTODUB_TTS_FORCE_FIT_TOLERANCE", "0.03"
        ),
        "maximum_safe_compression_ratio": os.getenv(
            "AUTODUB_TTS_MAX_SAFE_COMPRESSION_RATIO", "1.50"
        ),
        "maximum_safe_slowdown_ratio": os.getenv(
            "AUTODUB_TTS_MAX_SAFE_SLOWDOWN_RATIO", "1.25"
        ),
    }
    payload["quality_policy"] = {
        "electricity": os.getenv("AUTODUB_TTS_ELECTRICITY_POLICY", "warn"),
        "voice_consistency": os.getenv(
            "AUTODUB_TTS_VOICE_CONSISTENCY_POLICY", "block"
        ),
        "minimum_voice_consistency": os.getenv(
            "AUTODUB_TTS_MIN_VOICE_CONSISTENCY", "0.45"
        ),
    }
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


def _content_audit_fingerprint(item: Dict[str, Any], metadata: Dict[str, Any]) -> str:
    """Fingerprint post-generation content checks without invalidating audio.

    Changing ASR verification policy must re-audit an existing WAV, not cause
    every stochastic TTS backend to synthesize the whole project again.
    """
    path = Path(str(metadata.get("file") or item.get("final_path") or ""))
    audio_sha = str(metadata.get("audio_sha256") or "")
    if not audio_sha and path.is_file():
        audio_sha = _file_sha256(path)
    payload = {
        "version": "tts-content-audit-v1",
        "audio_sha256": audio_sha,
        "expected_text": str(item.get("text") or ""),
        "reference_text": str(item.get("ref_text") or ""),
        "language": str(item.get("target_language") or ""),
        "verifier_sha256": hashlib.sha256(
            Path(__file__).with_name("tts_content_verifier.py").read_bytes()
        ).hexdigest(),
        "enabled": os.getenv("AUTODUB_TTS_CONTENT_ASR_ENABLED", "auto"),
        "minimum_similarity": os.getenv(
            "AUTODUB_TTS_CONTENT_ASR_MIN_SIMILARITY", "language_default"
        ),
        "short_minimum_similarity": os.getenv(
            "AUTODUB_TTS_CONTENT_ASR_SHORT_MIN_SIMILARITY", "0.90"
        ),
        "safe_extra_ratio": os.getenv(
            "AUTODUB_TTS_CONTENT_ASR_MAX_SAFE_EXTRA_RATIO", "0.25"
        ),
        "reference_leak_min_coverage": os.getenv(
            "AUTODUB_TTS_REFERENCE_LEAK_MIN_COVERAGE", "0.50"
        ),
        "reference_leak_same_language_similarity": os.getenv(
            "AUTODUB_TTS_REFERENCE_LEAK_SAME_LANG_SIMILARITY", "0.82"
        ),
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _legacy_generation_request_matches(
    item: Dict[str, Any], metadata: Dict[str, Any]
) -> bool:
    """Safely migrate audio made before audit/generation fingerprints split."""
    saved = metadata.get("generation_request")
    if not isinstance(saved, dict):
        return False
    for key in GENERATION_REQUEST_KEYS:
        current = item.get(key)
        previous = saved.get(key)
        if key in {"target_duration", "start", "reference_purity_score"}:
            try:
                if current is None and previous is None:
                    continue
                if abs(float(current) - float(previous)) > 1e-6:
                    return False
                continue
            except (TypeError, ValueError):
                return False
        if current != previous:
            return False
    return True


def _tts_timing(segment: Dict[str, Any]) -> tuple[float, float]:
    """Return TTS placement and duration without mutating public boundaries."""
    start = float(segment["start"])
    end = float(segment["end"])
    legacy = (start, max(0.0, end - start))
    if not (
        str(segment.get("asr_recovery_status") or "") == "accepted_coarse"
        and str(segment.get("timestamp_precision") or "") == "coarse"
    ):
        try:
            mix_start = float(segment.get("mix_start", segment.get("speech_start")))
            mix_end = float(segment.get("mix_end", segment.get("speech_end")))
        except (TypeError, ValueError):
            return legacy
        if start <= mix_start < mix_end <= end + 0.001:
            # A sentence reconstructed from several timestamp atoms must use
            # the union of all of its words.  Never inherit a stale inner
            # fragment window merely because mix_start/mix_end survived a
            # previous split or recovery pass.
            word_starts: List[float] = []
            word_ends: List[float] = []
            timestamp_sets = (
                segment.get("qwen3_time_stamps"),
                segment.get("speaker_timing_time_stamps"),
            )
            for timestamp_items in timestamp_sets:
                for item in timestamp_items or []:
                    if not isinstance(item, dict):
                        continue
                    try:
                        word_start = float(item.get("start"))
                        word_end = float(item.get("end", word_start))
                    except (TypeError, ValueError):
                        continue
                    if word_end < start - 0.001 or word_start > end + 0.001:
                        continue
                    word_starts.append(max(start, word_start))
                    word_ends.append(min(end, max(word_start, word_end)))
            if word_starts and word_ends:
                mix_start = min(mix_start, min(word_starts))
                mix_end = max(mix_end, max(word_ends))
            if start <= mix_start < mix_end <= end + 0.001:
                return mix_start, mix_end - mix_start
        return legacy
    try:
        placement_start = float(segment.get("tts_placement_start"))
        placement_end = float(segment.get("tts_placement_end"))
    except (TypeError, ValueError):
        return legacy
    if not (start <= placement_start < placement_end <= end + 0.001):
        return legacy
    return placement_start, placement_end - placement_start


def _tts_window(segment: Dict[str, Any]) -> tuple[float, float, float]:
    """Return placement, ideal spoken duration and exclusive hard end."""
    placement, legacy_duration = _tts_timing(segment)
    try:
        explicit_start = float(segment.get("tts_placement_start", placement))
        target = float(segment.get("tts_target_duration", legacy_duration))
        hard_end = float(
            segment.get(
                "tts_hard_end",
                segment.get("tts_placement_end", placement + legacy_duration),
            )
        )
    except (TypeError, ValueError):
        return placement, legacy_duration, placement + legacy_duration
    outer_start = float(segment.get("start", placement))
    outer_end = float(segment.get("end", hard_end))
    if not (outer_start <= explicit_start < hard_end <= outer_end + 0.001):
        return placement, legacy_duration, placement + legacy_duration
    hard_duration = hard_end - explicit_start
    target = max(0.1, min(target, hard_duration))
    return explicit_start, target, hard_end


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
    minimum = float(os.getenv("AUTODUB_TTS_ATEMPO_MIN", "0.9"))
    maximum = float(os.getenv("AUTODUB_TTS_ATEMPO_MAX", "1.10"))
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


def _atempo_filter_chain(factor: float) -> str:
    """Build an ffmpeg atempo chain for any positive speed factor.

    One atempo filter only accepts values in [0.5, 2.0].  Chaining filters
    lets the final safety pass fit a usable utterance into its collision-free
    window instead of cutting the end of the sentence off.
    """
    remaining = max(0.01, float(factor))
    factors: List[float] = []
    while remaining > 2.0 + 1e-9:
        factors.append(2.0)
        remaining /= 2.0
    while remaining < 0.5 - 1e-9:
        factors.append(0.5)
        remaining /= 0.5
    factors.append(max(0.5, min(2.0, remaining)))
    return ",".join(f"atempo={value:.8f}" for value in factors)


def _apply_atempo_exact(
    source: Path, destination: Path, target_duration: float
) -> Dict[str, Any]:
    """Force an over-length clip into the requested window without truncation."""
    current = _wav_duration(source)
    if current <= 0 or target_duration <= 0:
        raise ValueError("invalid duration for exact atempo fit")
    factor = current / target_duration
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.stem + ".force-fit.tmp.wav")
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error", "-i", str(source),
            "-filter:a", _atempo_filter_chain(factor), "-vn", str(temporary),
        ],
        check=True,
    )
    os.replace(temporary, destination)
    return {
        "forced_duration_fit": True,
        "forced_atempo_factor": round(factor, 6),
        "forced_atempo_chain": _atempo_filter_chain(factor),
    }


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


def _postprocess(
    source: Path,
    destination: Path,
    target_duration: float,
    *,
    allow_force_fit: bool = True,
) -> Dict[str, Any]:
    """Trim silence then time-scale to target; return post-processing metrics."""
    trimmed = destination.with_name(destination.stem + ".trim.wav")
    try:
        pre_trim = _wav_duration(source)
        _trim_silence(source, trimmed)
        post_trim = _wav_duration(trimmed)
        factor, capped, raw = _atempo_factor(post_trim, target_duration)
        _apply_atempo(trimmed, destination, target_duration)
        forced = {}
        adaptation = duration_adaptation_safety(post_trim, target_duration)
        natural_duration = _wav_duration(destination)
        force_fit = _env_bool("AUTODUB_TTS_FORCE_FIT", True)
        tolerance = float(os.getenv("AUTODUB_TTS_FORCE_FIT_TOLERANCE", "0.03"))
        # Short speech may naturally leave a pause.  Only over-length speech
        # risks losing words, so the default final pass compresses rather than
        # stretching concise translations to fill silence.
        if (
            allow_force_fit
            and force_fit
            and target_duration > 0.04
            and natural_duration > target_duration + max(0.005, tolerance)
        ):
            if adaptation["unsafe_duration_compression"]:
                forced = {
                    "forced_duration_fit": False,
                    "force_fit_skipped": True,
                    "force_fit_skip_reason": "unsafe_compression_ratio",
                }
            else:
                forced = _apply_atempo_exact(trimmed, destination, target_duration)
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
        **adaptation,
        **forced,
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
    engine: Optional[str] = None,
    max_attempts: Optional[int] = None,
) -> Dict[str, Any]:
    """Generate up to three previews with one user-selected TTS engine."""
    destination = Path(output_path).resolve()
    work = destination.parent / (destination.stem + "_router_work")
    if scheme not in {"scheme2", "scheme3"}:
        raise ValueError(f"unsupported TTS scheme: {scheme}")
    model = str(engine or "").strip().lower() or (
        select_scheme3_tts_engine(language, emotion, text)
        if scheme == "scheme3"
        else select_tts_model(language, emotion)
    )
    if model not in MODEL_ORDER:
        raise ValueError(f"unsupported TTS engine: {model}")
    attempt_limit = _trial_attempt_limit(max_attempts)
    canonical_language = normalize_language(language)
    synthesis_language = _language_code(language) if model == "omnivoice" else canonical_language
    preflight_models([model])
    work.mkdir(parents=True, exist_ok=True)
    candidates: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    target = float(target_duration or 0.0)
    trial_items = []
    for attempt in range(1, attempt_limit + 1):
        segment_id = f"trial_{attempt}"
        raw = work / f"raw_attempt_{attempt}.wav"
        preview = work / f"preview_attempt_{attempt}.wav"
        item = {
            "segment_id": segment_id, "text": text,
            "language": synthesis_language,
            "target_language": canonical_language,
            "emotion": normalize_emotion(emotion), "speaker": "SPEAKER_00",
            "ref_audio": str(Path(ref_audio).resolve()), "ref_text": ref_text,
            "raw_path": str(raw), "target_duration": target,
            "model": model, "tts_engine": model,
            "trial_attempt": attempt, "trial_preview_path": str(preview),
        }
        item["request_fingerprint"] = _request_fingerprint(item)
        trial_items.append(item)

    # Generate the three stochastic candidates in one worker process so the
    # selected model is loaded only once. Force generation prevents an older
    # preview cache from silently counting as a new attempt.
    plan = work / "plan.json"
    result = work / "worker_results.json"
    _atomic_json(plan, {"items": trial_items})
    code = _run_worker(model, plan, result, force=True)
    worker_results = _load_results(result)
    for item in trial_items:
        attempt = int(item["trial_attempt"])
        raw = Path(item["raw_path"])
        preview = Path(item["trial_preview_path"])
        metadata = dict(worker_results.get(item["segment_id"], {}))
        if metadata.get("status") != "success":
            failures.append({
                "attempt": attempt,
                "error": metadata.get(
                    "error", f"{model} worker exited with {code}"
                ),
            })
            continue
        try:
            metadata.update(_postprocess(raw, preview, target))
        except Exception as exc:
            failures.append({"attempt": attempt, "error": str(exc)})
            continue
        metadata.update({
            "file": str(preview),
            "target_duration": target,
            "duration_error": (
                metadata["actual_duration"] - target if target else None
            ),
            "trial_attempt": attempt,
            "engine": model,
            "actual_engine": model,
        })
        if target:
            metadata.update(duration_quality(metadata["actual_duration"], target))
            if metadata["duration_quality"] == "failed":
                metadata["duration_quality"] = "degraded"
        metadata.update(electricity_quality(preview))
        metadata.update(audible_speech_quality(preview, expected_text=text))
        candidates.append(metadata)

    if not candidates:
        raise RuntimeError(
            f"{model} trial synthesis failed in all {attempt_limit} attempts: "
            + "; ".join(str(row.get("error") or "unknown") for row in failures)
        )

    if _trial_content_asr_enabled():
        audit_plan = work / "content_asr_plan.json"
        audit_output = work / "content_asr_results.json"
        _atomic_json(audit_plan, {"items": [
            {
                "segment_id": str(candidate["trial_attempt"]),
                "text": text,
                "reference_text": ref_text,
                "language": canonical_language,
                "file": candidate["file"],
            }
            for candidate in candidates
        ]})
        completed = subprocess.run([
            sys.executable,
            str(Path(__file__).with_name("tts_content_verifier.py")),
            "--plan", str(audit_plan), "--output", str(audit_output),
        ], check=False, env=os.environ.copy())
        if completed.returncode != 0 or not audit_output.is_file():
            raise RuntimeError(
                "TTS trial content verification failed; refusing unchecked previews"
            )
        audits = json.loads(audit_output.read_text(encoding="utf-8"))
        for candidate in candidates:
            candidate.update(audits.get(str(candidate["trial_attempt"])) or {})

    safe_candidates = [
        candidate for candidate in candidates
        if not candidate.get("reference_leakage_detected")
        and candidate.get("reference_leakage_quality") != "failed"
        and candidate.get("audible_speech_quality") != "failed"
    ]
    if not safe_candidates:
        raise RuntimeError(
            "all generated TTS trial candidates were rejected as reference "
            "leakage or unreliable audio"
        )
    chosen = min(safe_candidates, key=_trial_candidate_rank)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(chosen["file"], destination)
    metadata = dict(chosen)
    metadata.update(
        file=str(destination),
        status="success",
        trial_attempt_count=len(candidates) + len(failures),
        trial_attempt_limit=attempt_limit,
        trial_selected_attempt=int(chosen["trial_attempt"]),
        trial_engine=model,
        attempt_history=[
            {
                "attempt": int(candidate["trial_attempt"]),
                "actual_duration": candidate.get("actual_duration"),
                "duration_quality": candidate.get("duration_quality"),
                "audible_speech_quality": candidate.get("audible_speech_quality"),
                "content_completeness_quality": candidate.get(
                    "content_completeness_quality"
                ),
                "reference_leakage_detected": bool(
                    candidate.get("reference_leakage_detected")
                ),
                "selected": candidate is chosen,
            }
            for candidate in candidates
        ] + failures,
    )
    strict_pass = (
        metadata.get("duration_quality") in {"passed", "unmeasured", None}
        and metadata.get("audible_speech_quality") != "failed"
        and metadata.get("content_completeness_quality") != "failed"
    )
    metadata["adoption_recommended"] = bool(strict_pass)
    metadata["requires_duration_resolution"] = not strict_pass
    if not strict_pass:
        metadata["warning"] = {
            "code": "TTS_TRIAL_BEST_AVAILABLE",
            "message": (
                "No strict-pass result was produced in three or fewer attempts; "
                "returning the best safe generated preview."
            ),
        }
    _atomic_json(work / "results.json", {"0": metadata})
    return metadata


class TTSRouter:
    def __init__(
        self,
        config: Any,
        *,
        scheme: str = "scheme2",
        duration_rewriter: Optional[Callable[..., Optional[Dict[str, Any]]]] = None,
        duration_candidate_selector: Optional[
            Callable[..., Optional[Dict[str, Any]]]
        ] = None,
    ):
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
        self.candidate_dir = self.root / "candidate_vault"
        self.candidate_path = self.root / "candidate_vault.json"
        self.duration_rewriter = duration_rewriter
        self.duration_candidate_selector = duration_candidate_selector

    def _remember_candidate(
        self, item: Dict[str, Any], metadata: Dict[str, Any], stage: str
    ) -> None:
        """Persist every audible synthesis so later retries cannot erase it."""
        source = Path(str(metadata.get("file") or item.get("final_path") or ""))
        if not source.is_file() or not source.stat().st_size:
            return
        measured = dict(metadata)
        if not measured.get("audible_speech_quality"):
            measured.update(
                audible_speech_quality(source, expected_text=item.get("text", ""))
            )
        if measured.get("audible_speech_quality") != "passed":
            return
        audio_sha = str(measured.get("audio_sha256") or _file_sha256(source))
        reference = Path(str(item.get("ref_audio") or ""))
        reference_sha = _file_sha256(reference) if reference.is_file() else ""
        sid = str(item["segment_id"])
        self.candidate_dir.mkdir(parents=True, exist_ok=True)
        archived = self.candidate_dir / f"segment_{_safe_id(sid)}_{audio_sha[:16]}.wav"
        if not archived.is_file():
            shutil.copy2(source, archived)
        try:
            registry = json.loads(self.candidate_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            registry = {}
        rows = list(registry.get(sid) or [])
        record = {
            **measured,
            "segment_id": sid,
            "file": str(archived),
            "audio_sha256": audio_sha,
            "candidate_stage": stage,
            "candidate_text": str(item.get("text") or ""),
            "planned_text": str(item.get("planned_text") or ""),
            "reference_audio_sha256": reference_sha,
            "generation_request": {
                key: item.get(key) for key in GENERATION_REQUEST_KEYS
            },
            "request_fingerprint": item.get("request_fingerprint"),
        }
        replaced = False
        for index, existing in enumerate(rows):
            if str(existing.get("audio_sha256")) == audio_sha:
                rows[index] = {**existing, **record}
                replaced = True
                break
        if not replaced:
            rows.append(record)
        registry[sid] = rows
        _atomic_json(self.candidate_path, registry)

    @staticmethod
    def _candidate_score(candidate: Dict[str, Any], target: float) -> tuple:
        """Rank semantics and audibility before duration convenience."""
        content_rank = {
            "passed": 0,
            "degraded": 1,
            "": 2,
            "failed": 3,
        }.get(str(candidate.get("content_completeness_quality") or ""), 2)
        semantic_selection_rank = int(
            candidate.get("duration_selected_attempt") is None
        )
        voice_rank = {
            "passed": 0,
            "degraded": 1,
            "": 1,
            "failed": 2,
        }.get(str(candidate.get("voice_consistency_quality") or ""), 1)
        electricity_rank = int(candidate.get("electricity_quality") == "failed")
        duration = float(
            candidate.get("post_trim_duration")
            or candidate.get("actual_duration")
            or 0.0
        )
        if duration <= 0.0 or target <= 0.0:
            distance = float("inf")
        else:
            import math
            distance = abs(math.log(duration / target))
        return (
            content_rank,
            semantic_selection_rank,
            voice_rank,
            electricity_rank,
            distance,
        )

    def _adopt_best_generated_candidate(
        self, item: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Promote the best audible generated clip as the final Agent fallback."""
        try:
            registry = json.loads(self.candidate_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None
        current_text = str(item.get("planned_text") or item.get("text") or "")
        current_reference = Path(str(item.get("ref_audio") or ""))
        current_reference_sha = (
            _file_sha256(current_reference) if current_reference.is_file() else ""
        )
        compatible = []
        for candidate in registry.get(str(item["segment_id"]), []) or []:
            path = Path(str(candidate.get("file") or ""))
            if not path.is_file() or not path.stat().st_size:
                continue
            # Reference-speech leakage is a hard safety failure, never a
            # scoring penalty. It cannot enter best-candidate fallback even if
            # its voice, duration and target-text metrics look excellent.
            if (
                bool(candidate.get("reference_leakage_detected"))
                or candidate.get("reference_leakage_quality") == "failed"
            ):
                continue
            if (
                self._content_asr_enabled()
                and candidate.get("reference_leakage_quality") != "passed"
            ):
                continue
            if current_text not in {
                str(candidate.get("planned_text") or ""),
                str(candidate.get("candidate_text") or ""),
            }:
                continue
            generation = dict(candidate.get("generation_request") or {})
            if str(generation.get("target_language") or "") != str(
                item.get("target_language") or ""
            ):
                continue
            if str(generation.get("speaker") or "") != str(
                item.get("speaker") or ""
            ):
                continue
            if str(generation.get("emotion") or "") != str(
                item.get("emotion") or ""
            ):
                continue
            if str(candidate.get("reference_audio_sha256") or "") != (
                current_reference_sha
            ):
                continue
            compatible.append(candidate)
        if not compatible:
            return None
        chosen = min(
            compatible,
            key=lambda row: self._candidate_score(
                row, float(item.get("target_duration") or 0.0)
            ),
        )
        generation = dict(chosen.get("generation_request") or {})
        # Adopt the generated voice/text, but keep the current timeline and
        # target window.  A stale candidate must never move a sentence back to
        # an older timestamp merely because its waveform was better.
        for key in ("text", "model", "language"):
            if key in generation:
                item[key] = generation[key]
        item["tts_engine"] = item["model"]
        destination = self.final_dir / (
            f"segment_{_safe_id(item['segment_id'])}_{item['model']}_best_available.wav"
        )
        source = Path(str(chosen["file"]))
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        fit = {}
        actual = _wav_duration(destination)
        target = float(item.get("target_duration") or 0.0)
        if target > 0.04 and actual > target + 0.01:
            fit = _apply_atempo_exact(destination, destination, target)
            actual = _wav_duration(destination)
        item["final_path"] = str(destination)
        item["request_fingerprint"] = _request_fingerprint(item)
        result = dict(chosen)
        original_error = str(result.get("error") or "strict quality checks failed")
        result.update(
            status="success",
            file=str(destination),
            preview_file=str(destination),
            preview_available=True,
            adoption_recommended=True,
            agent_degraded_candidate=True,
            agent_degraded_reason=original_error,
            error="",
            duration_quality="degraded",
            duration_degraded_reason="agent_best_generated_candidate",
            actual_duration=actual,
            audio_sha256=_file_sha256(destination),
            request_fingerprint=item["request_fingerprint"],
            generation_request={
                key: item.get(key) for key in GENERATION_REQUEST_KEYS
            },
            generation_attempt_count=int(
                item.get("generation_attempt_count") or 0
            ),
            generation_attempt_limit=_generation_attempt_limit(),
            **fit,
        )
        print(
            f"[TTS degraded] segment={item['segment_id']} 严格验收全部失败；"
            f"采用最佳已生成候选继续成片: {item['model']} "
            f"{actual:.2f}/{target:.2f}s"
        )
        return result

    def _content_asr_enabled(self) -> bool:
        configured = os.getenv("AUTODUB_TTS_CONTENT_ASR_ENABLED")
        if configured is not None:
            return configured.strip().lower() in {"1", "true", "yes", "on"}
        model_dir = os.getenv("QWEN3_ASR_MODEL_DIR", "").strip()
        return bool(model_dir and Path(model_dir).is_dir())

    def _run_content_asr_audit(self, items, results):
        """Batch-transcribe successful TTS and reject swallowed content."""
        if not self._content_asr_enabled():
            return results
        updated = dict(results)
        pending = []
        for item in items:
            metadata = updated.get(item["segment_id"], {})
            if (
                metadata.get("status") != "success"
                and metadata.get("audible_speech_quality") != "passed"
            ):
                continue
            audit_fingerprint = _content_audit_fingerprint(item, metadata)
            if (
                metadata.get("content_completeness_quality") == "passed"
                and metadata.get("reference_leakage_quality") == "passed"
                and metadata.get("content_asr_request_fingerprint")
                == audit_fingerprint
            ):
                continue
            path = Path(str(metadata.get("file") or item.get("final_path") or ""))
            if path.is_file():
                pending.append({
                    "segment_id": item["segment_id"],
                    "text": item["text"],
                    "reference_text": item.get("ref_text", ""),
                    "language": item["target_language"],
                    "file": str(path),
                    "request_fingerprint": item["request_fingerprint"],
                    "content_audit_fingerprint": audit_fingerprint,
                })
        if not pending:
            return updated
        plan = self.root / "content_asr_plan.json"
        output = self.root / "content_asr_results.json"
        _atomic_json(plan, {"items": pending})
        completed = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("tts_content_verifier.py")),
                "--plan", str(plan), "--output", str(output),
            ],
            check=False,
            env=os.environ.copy(),
        )
        if completed.returncode != 0 or not output.is_file():
            raise RuntimeError(
                "TTS content ASR verification failed; refusing an unchecked final video"
            )
        audits = json.loads(output.read_text(encoding="utf-8"))
        for row in pending:
            sid = str(row["segment_id"])
            metadata = dict(updated.get(sid, {}))
            audit = dict(audits.get(sid) or {})
            if not audit:
                raise RuntimeError(f"TTS content ASR omitted segment {sid}")
            metadata.update(audit)
            metadata["content_asr_request_fingerprint"] = row[
                "content_audit_fingerprint"
            ]
            if (
                audit.get("content_completeness_quality") != "passed"
                or audit.get("reference_leakage_detected")
            ):
                reason = (
                    "reference_audio_text_leakage:"
                    + str(audit.get("reference_leakage_reason") or "detected")
                    if audit.get("reference_leakage_detected")
                    else str(audit.get("content_completeness_reason") or "mismatch")
                )
                metadata.update(
                    status="failed",
                    error=(
                        "TTS content verification failed: " + reason
                    ),
                    preview_file=metadata.get("file"),
                    preview_available=True,
                    adoption_recommended=False,
                )
            updated[sid] = metadata
            matching = next(
                (item for item in items if item["segment_id"] == sid), None
            )
            if matching is not None:
                self._remember_candidate(matching, metadata, "content_audit")
        _atomic_json(self.result_path, updated)
        return updated

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
        confidence = float(segment.get("speaker_assignment_confidence", 1.0) or 0.0)
        boundary_penalty = 0.35 if "speaker_boundary" in (
            segment.get("segment_split_reasons") or []
        ) else 0.0
        return (
            2.0 * speech_ratio + duration_score + loudness_score
            + 0.8 * confidence - boundary_penalty
        )

    def _speaker_references(self, segments: List[Dict[str, Any]], vocal_path: str) -> Dict[str, Dict[str, str]]:
        from pydub import AudioSegment
        full_audio = AudioSegment.from_wav(vocal_path)
        references: Dict[str, Dict[str, str]] = {}
        manifest = {"version": 2, "policy": "single_contiguous_pure_reference", "speakers": {}}
        speakers = sorted({str(segment.get("speaker", "SPEAKER_00")) for segment in segments})
        for speaker in speakers:
            candidates = []
            for index, segment in enumerate(segments):
                if str(segment.get("speaker", "SPEAKER_00")) != speaker:
                    continue
                source_segments = [segment]
                if segment.get("speaker_reference_mixed_raw"):
                    source_segments = list(
                        segment.get("speaker_reference_components") or []
                    )
                # Coarse ASR recovery provides only an outer dialogue window.
                # Its audio may contain additional words, pauses, or another
                # turn that are absent from the recovered transcript.  Such a
                # clip is valid for dubbing, but unsafe as paired reference
                # audio/text for a voice-cloning model.
                for component_index, source_segment in enumerate(source_segments):
                    source_segment = dict(source_segment)
                    source_segment["speaker"] = speaker
                    source_segment.setdefault(
                        "id", f"{segment.get('id', index)}:{component_index}"
                    )
                    if not _is_precise_reference_segment(source_segment):
                        continue
                    start = max(0, int(round(float(source_segment["start"]) * 1000)))
                    end = min(len(full_audio), int(round(float(source_segment["end"]) * 1000)))
                    clip = full_audio[start:end]
                    if len(clip) < 500:
                        continue
                    candidates.append((
                        self._reference_score(source_segment, clip),
                        index,
                        source_segment,
                        clip,
                    ))
            if not candidates:
                print(
                    f"⚠️ {speaker} 没有可用的精确边界参考音频；"
                    "粗边界恢复句仍不会用于声纹参考，该说话人将单独降级"
                )
                continue
            candidates.sort(key=lambda item: (item[0], len(item[3])), reverse=True)
            confident = [
                item for item in candidates
                if float(item[2].get("speaker_assignment_confidence", 1.0) or 0.0)
                >= 0.72
            ]
            selection_pool = confident or candidates
            # Never concatenate separate utterances: a mistaken global cluster
            # could otherwise mix two real people into one cloning reference.
            preferred = [item for item in selection_pool if 2500 <= len(item[3]) <= 8000]
            bounded = [item for item in selection_pool if len(item[3]) <= 10000]
            chosen = (preferred or bounded or selection_pool)[0]
            score, source_index, source_segment, audio = chosen
            source_text = str(
                source_segment.get("original_text")
                or source_segment.get("qwen3_raw_text")
                or source_segment.get("text")
                or ""
            ).strip()
            safe_speaker = _safe_id(speaker)
            ref_path = self.root / "references" / f"ref_{safe_speaker}.wav"
            txt_path = ref_path.with_suffix(".txt")
            ref_path.parent.mkdir(parents=True, exist_ok=True)
            audio.export(ref_path, format="wav")
            txt_path.write_text(source_text, encoding="utf-8")
            references[speaker] = {
                "audio": str(ref_path),
                "text": source_text,
                "speaker": speaker,
                "start": float(source_segment["start"]),
                "end": float(source_segment["end"]),
                "source_segment_id": str(source_segment.get("id", source_index)),
                "purity_score": round(float(score), 4),
                "duration": round(len(audio) / 1000.0, 3),
                "assignment_confidence": float(
                    source_segment.get("speaker_assignment_confidence", 1.0)
                    or 0.0
                ),
            }
            manifest["speakers"][speaker] = {
                "candidate_count": len(candidates),
                "confident_candidate_count": len(confident),
                "selected_segment_id": source_segment.get("id", source_index),
                "start": float(source_segment["start"]),
                "end": float(source_segment["end"]),
                "duration": round(len(audio) / 1000.0, 3),
                "score": round(float(score), 4),
                "assignment_confidence": source_segment.get("speaker_assignment_confidence"),
                "confidence_degraded": not bool(confident),
                "boundary_ambiguous": bool(source_segment.get("speaker_boundary_ambiguous")),
                "concatenated": False,
                "candidates": [
                    {
                        "segment_id": item[2].get("id", item[1]),
                        "start": float(item[2]["start"]),
                        "end": float(item[2]["end"]),
                        "score": round(float(item[0]), 4),
                        "assignment_confidence": item[2].get(
                            "speaker_assignment_confidence"
                        ),
                    }
                    for item in candidates
                ],
            }
            print(f"参考音频 {speaker}: 候选 {len(candidates)}，选中单段 ID={source_segment.get('id', source_index)}，{len(audio)/1000.0:.2f}s")
        # Step 2 speaker labels remain untouched: they define dialogue turns and
        # timestamps.  For voice cloning only, conservatively reuse a stronger
        # reference when two labels have near-identical acoustic proxies.  This
        # reduces voice drift caused by an over-split diarization without ever
        # joining sentences or changing their boundaries.
        similarity_threshold = float(os.getenv(
            "AUTODUB_TTS_REFERENCE_IDENTITY_SIMILARITY", "0.90"
        ))
        minimum_reliable = float(os.getenv(
            "AUTODUB_TTS_MIN_RELIABLE_REFERENCE_SECONDS", "1.50"
        ))
        minimum_reference_confidence = float(os.getenv(
            "AUTODUB_TTS_MIN_RELIABLE_REFERENCE_CONFIDENCE", "0.72"
        ))
        similarities = []
        similarity_by_pair = {}
        reference_names = sorted(references)
        for left_index, left in enumerate(reference_names):
            for right in reference_names[left_index + 1:]:
                similarity = voice_reference_similarity(
                    references[left]["audio"], references[right]["audio"]
                )
                if similarity is None:
                    continue
                similarity_by_pair[frozenset((left, right))] = similarity
                similarities.append({
                    "left": left,
                    "right": right,
                    "similarity": round(similarity, 6),
                    "merged": False,
                })

        # Reliable references are immutable anchors. A short label may attach
        # directly to its single closest anchor, but anchors are never merged
        # with each other and short-to-short transitive chains are forbidden.
        # This avoids collapsing two same-gender actors through an accidental
        # A~B~C similarity bridge.
        anchors = [
            name for name in reference_names
            if float(references[name].get("duration", 0.0)) >= minimum_reliable
            and float(references[name].get("assignment_confidence", 1.0))
            >= minimum_reference_confidence
        ]
        voice_identity_map = {name: name for name in reference_names}
        for speaker in reference_names:
            if speaker in anchors:
                continue
            matches = [
                (similarity_by_pair.get(frozenset((speaker, anchor))), anchor)
                for anchor in anchors
            ]
            matches = [
                (similarity, anchor) for similarity, anchor in matches
                if similarity is not None and similarity >= similarity_threshold
            ]
            if matches:
                _, best_anchor = max(
                    matches,
                    key=lambda item: (
                        item[0],
                        float(references[item[1]].get("purity_score", 0.0)),
                    ),
                )
                voice_identity_map[speaker] = best_anchor
                for row in similarities:
                    if {row["left"], row["right"]} == {speaker, best_anchor}:
                        row["merged"] = True
                        break

        for speaker in reference_names:
            canonical = voice_identity_map[speaker]
            canonical_reference = dict(references[canonical])
            original = references[speaker]
            reused = speaker != canonical
            references[speaker] = {
                **canonical_reference,
                "requested_speaker": speaker,
                "voice_identity": canonical,
                "voice_identity_reused": reused,
                "original_reference_segment_id": original.get(
                    "source_segment_id"
                ),
                "original_reference_duration": original.get("duration"),
                "reference_reliability": (
                    "reliable"
                    if (
                        float(canonical_reference.get("duration", 0.0))
                        >= minimum_reliable
                        and float(canonical_reference.get(
                            "assignment_confidence", 1.0
                        )) >= minimum_reference_confidence
                    )
                    else "short_or_low_confidence_degraded"
                ),
            }
            manifest["speakers"][speaker].update({
                "voice_identity": canonical,
                "voice_identity_reused": reused,
                "selected_voice_reference_segment_id": canonical_reference.get(
                    "source_segment_id"
                ),
                "selected_voice_reference_duration": canonical_reference.get(
                    "duration"
                ),
                "reference_reliability": references[speaker][
                    "reference_reliability"
                ],
            })
        manifest.update({
            "version": 3,
            "policy": "sentence_boundaries_preserved_voice_identity_reuse_v1",
            "minimum_reliable_reference_seconds": minimum_reliable,
            "minimum_reliable_reference_confidence": minimum_reference_confidence,
            "voice_identity_similarity_threshold": similarity_threshold,
            "voice_identity_map": voice_identity_map,
            "voice_identity_similarities": similarities,
        })
        _atomic_json(self.root / "reference_manifest.json", manifest)
        return references

    @staticmethod
    def _nearest_safe_reference(
        segment: Dict[str, Any], references: Dict[str, Dict[str, Any]]
    ) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Return the temporally nearest precise reference as a no-drop fallback."""
        eligible = {
            speaker: reference
            for speaker, reference in references.items()
            if str(reference.get("reference_reliability", "reliable"))
            == "reliable"
        }
        if not eligible:
            return None, None
        center = (
            float(segment.get("start", 0.0))
            + float(segment.get("end", segment.get("start", 0.0)))
        ) / 2.0
        speaker, reference = min(
            eligible.items(),
            key=lambda item: (
                abs(
                    center
                    - (
                        float(item[1].get("start", center))
                        + float(item[1].get("end", center))
                    )
                    / 2.0
                ),
                -float(item[1].get("purity_score", 0.0) or 0.0),
                item[0],
            ),
        )
        return speaker, reference

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
            reference = references.get(speaker)
            reference_fallback_speaker = None
            reference_fallback_reason = ""
            if reference is None and _env_bool(
                "AUTODUB_TTS_ALLOW_CROSS_SPEAKER_REFERENCE", True
            ):
                reference_fallback_speaker, reference = self._nearest_safe_reference(
                    segment, references
                )
            if reference is None:
                failure = {
                    "segment_id": segment_id,
                    "status": "failed",
                    "error": "no_safe_reference_audio_for_speaker",
                    "planned_engine": segment.get("tts_engine"),
                    "actual_engine": None,
                    "fallback_count": 0,
                }
                previous[segment_id] = failure
                unreferenced_results.append(failure)
                continue
            if reference_fallback_speaker is not None:
                reference_fallback_reason = "nearest_safe_precise_reference"
                print(
                    f"⚠️ segment={segment_id} speaker={speaker} 没有安全参考音频；"
                    f"使用最近可靠声纹 {reference_fallback_speaker}，保留整句并标记降级"
                )
            placement_start, source_duration, hard_end = _tts_window(segment)
            raw_emotion = segment.get("raw_emotion", segment.get("emotion_raw", segment.get("emotion")))
            if "emotion_reliable" in segment:
                emotion = normalize_emotion(segment.get("emotion")) if segment["emotion_reliable"] else "neutral"
            else:
                emotion = normalize_emotion(raw_emotion, score=segment.get("emotion_score"), duration=source_duration)
            target = normalize_language(segment.get("target_lang", self.config.target_lang))
            target_raw = str(segment.get("target_lang_raw", self.config.target_lang)).lower()
            hard_duration = max(0.0, hard_end - placement_start)
            minimum_pronounceable = _minimum_pronounceable_duration(
                text, target
            )
            duration = min(
                hard_duration,
                max(source_duration, minimum_pronounceable),
            )
            if self.scheme == "scheme3":
                model, expected, missing = resolve_scheme3_engine(segment, target, emotion, text)
                if missing or model != expected:
                    print(f"⚠️ segment={segment_id} saved_engine={model} expected={expected} legacy_fallback={missing}")
            else:
                model = "indextts2" if self.scheme=="baseline" else select_tts_model(self.config.target_lang, emotion)
            placement_source = "segment_start"
            if placement_start != float(segment["start"]):
                placement_source = (
                    segment.get("recovery_timing_method")
                    or "word_aligned_speech_window"
                )
            job = {
                "segment_id": segment_id, "text": text, "planned_text": text,
                "language": _language_code(target_raw) if model=="omnivoice" else target,
                "target_language": target, "target_language_raw": target_raw,
                "source_language": normalize_language(segment.get("source_lang", getattr(self.config, "source_lang", "unknown"))),
                "emotion": emotion, "emotion_score": segment.get("emotion_score"),
                "speaker": speaker, "ref_audio": reference["audio"],
                "ref_text": reference["text"], "target_duration": duration,
                "source_target_duration": source_duration,
                "minimum_pronounceable_duration": minimum_pronounceable,
                "target_duration_floor_applied": (
                    duration > source_duration + 0.001
                ),
                "reference_segment_id": reference.get("source_segment_id"),
                "reference_purity_score": reference.get("purity_score"),
                "reference_duration": reference.get("duration"),
                "reference_reliability": reference.get("reference_reliability", "reliable"),
                "voice_identity": reference.get("voice_identity", speaker),
                "voice_identity_reused": bool(reference.get("voice_identity_reused", False)),
                "reference_fallback": reference_fallback_speaker is not None,
                "reference_fallback_speaker": reference_fallback_speaker,
                "reference_fallback_reason": reference_fallback_reason,
                "start": placement_start, "model": model, "tts_engine": model,
                "end": hard_end,
                "hard_duration": hard_duration,
                "allow_overlap": False,
                "allowed_overlap_seconds": 0.0,
                "source_overlap_evidence": "none",
                "placement_source": placement_source,
                "planned_engine": model, "fallback_count": 0, "fallback_reason": "",
                "translations_candidates": segment.get("translations_candidates", []),
                "attempted_engines": [model], "attempted_texts": [text],
                "source_segment": segment,
                "translation_selection_source": str(
                    segment.get("translation_selection_source", "standard")
                ),
                "duration_attempt": 0,
                "measure_natural_duration": self.duration_rewriter is not None,
                "generation_attempt_count": 0,
            }
            self._job_paths(job)
            job["request_fingerprint"] = _request_fingerprint(job)
            job["planned_request_fingerprint"] = job["request_fingerprint"]
            items.append(job)

        # A per-sentence emotion route can make one character change timbre
        # every few seconds even when every individual generation is valid.
        # Keep the rule decision for audit, but use one dominant engine for all
        # sentences sharing the same acoustic voice identity. Engine fallback
        # is still available when that dominant engine cannot generate a
        # complete utterance.
        if _env_bool("AUTODUB_TTS_SPEAKER_ENGINE_LOCK", True):
            by_identity: Dict[str, List[Dict[str, Any]]] = {}
            for item in items:
                by_identity.setdefault(
                    str(item.get("voice_identity") or item["speaker"]), []
                ).append(item)
            for identity, identity_items in by_identity.items():
                if len(identity_items) < 2:
                    continue
                engine_stats: Dict[str, tuple[int, float]] = {}
                for item in identity_items:
                    engine = str(item["planned_engine"])
                    count, seconds = engine_stats.get(engine, (0, 0.0))
                    engine_stats[engine] = (
                        count + 1,
                        seconds + float(item.get("target_duration") or 0.0),
                    )
                locked_engine = max(
                    engine_stats,
                    key=lambda engine: (
                        engine_stats[engine][0],
                        engine_stats[engine][1],
                        -MODEL_ORDER.index(engine),
                    ),
                )
                if len(engine_stats) > 1:
                    print(
                        f"🔒 voice_identity={identity} 固定 TTS 引擎为 "
                        f"{locked_engine}；原路由={engine_stats}"
                    )
                for item in identity_items:
                    rule_engine = str(item["planned_engine"])
                    item["rule_planned_engine"] = rule_engine
                    item["speaker_engine_locked"] = rule_engine != locked_engine
                    item["model"] = locked_engine
                    item["tts_engine"] = locked_engine
                    item["planned_engine"] = locked_engine
                    item["attempted_engines"] = [locked_engine]
                    item["language"] = (
                        _language_code(item["target_language_raw"])
                        if locked_engine == "omnivoice"
                        else item["target_language"]
                    )
                    self._job_paths(item)
                    item["request_fingerprint"] = _request_fingerprint(item)
                    item["planned_request_fingerprint"] = item[
                        "request_fingerprint"
                    ]

        # The next dub onset is the final collision boundary. Apply it before
        # synthesis so duration feedback targets a physically usable slot,
        # rather than letting Step 5 discover and truncate an overlap later.
        chronological = sorted(items, key=lambda row: float(row["start"]))
        for position, item in enumerate(chronological):
            hard_end = float(item["end"])
            if position + 1 < len(chronological):
                following = chronological[position + 1]
                current_source = item.get("source_segment") or {}
                following_source = following.get("source_segment") or {}
                try:
                    source_overlap = max(
                        0.0,
                        float(current_source.get("end"))
                        - float(following_source.get("start")),
                    )
                except (TypeError, ValueError):
                    source_overlap = 0.0
                distinct_speakers = str(item.get("speaker")) != str(
                    following.get("speaker")
                )
                reliable_overlap = (
                    source_overlap
                    >= float(os.getenv("AUTODUB_TTS_MIN_SOURCE_OVERLAP", "0.12"))
                    and distinct_speakers
                    and not bool(current_source.get("speaker_boundary_ambiguous"))
                    and not bool(following_source.get("speaker_boundary_ambiguous"))
                )
                allowed_overlap = (
                    min(
                        source_overlap,
                        float(os.getenv("AUTODUB_TTS_MAX_SOURCE_OVERLAP", "1.50")),
                    )
                    if reliable_overlap else 0.0
                )
                item["allowed_overlap_seconds"] = round(allowed_overlap, 6)
                item["allow_overlap"] = allowed_overlap > 0.0
                item["source_overlap_evidence"] = (
                    "distinct_speaker_source_intersection"
                    if allowed_overlap > 0.0 else "none"
                )
                hard_end = min(
                    hard_end,
                    float(following["start"]) + allowed_overlap,
                )
            item["end"] = max(float(item["start"]), hard_end)
            item["hard_duration"] = max(0.0, item["end"] - float(item["start"]))
            if item["hard_duration"] > 0.04:
                item["target_duration"] = min(
                    float(item["target_duration"]), item["hard_duration"]
                )
            item["request_fingerprint"] = _request_fingerprint(item)
            item["planned_request_fingerprint"] = item["request_fingerprint"]

        active = []
        for item in items:
            known = previous.get(item["segment_id"], {})
            item["generation_attempt_count"] = max(
                int(item.get("generation_attempt_count") or 0),
                int(known.get("generation_attempt_count") or 0),
            )
            reused = False
            saved_generation = known.get("generation_request")
            if not isinstance(saved_generation, dict):
                saved_generation = {}
            # A fallback engine or a duration-directed rewrite may differ from
            # the original Step 3 plan.  It is safe to migrate that successful
            # waveform only while the Step 3 source text itself is unchanged.
            # Otherwise an old tts_text must not hide a newly edited sentence.
            current_text = str(item.get("text") or "")
            planned_text_unchanged = (
                str(known.get("planned_text") or "") == current_text
                or (
                    int(known.get("duration_translation_attempts") or 0) > 0
                    and str(known.get("tts_text") or "") == current_text
                )
            )
            migration_probe = dict(item)
            if planned_text_unchanged:
                migration_probe.update(
                    model=known.get("actual_engine", item["model"]),
                    text=known.get("tts_text", item["text"]),
                )
                if "measure_natural_duration" in saved_generation:
                    migration_probe["measure_natural_duration"] = (
                        saved_generation["measure_natural_duration"]
                    )
            migration_probe["language"] = (
                _language_code(migration_probe["target_language_raw"])
                if migration_probe["model"] == "omnivoice"
                else migration_probe["target_language"]
            )
            legacy_generation_match = (
                planned_text_unchanged
                and _legacy_generation_request_matches(migration_probe, known)
            )
            planned_compatible = (
                known.get("planned_request_fingerprint")
                == item["planned_request_fingerprint"]
                or legacy_generation_match
                or (
                    int(known.get("duration_translation_attempts") or 0) > 0
                    and str(known.get("tts_text") or "") == str(item["text"])
                )
            )
            if (not force and known.get("status")=="success" and
                    known.get("audible_speech_quality")=="passed" and
                    not (getattr(self.config,"disable_tts_fallback",False) and known.get("actual_engine")!=item["planned_engine"]) and
                    planned_compatible):
                resumed = dict(item, model=known.get("actual_engine", item["model"]),
                               text=known.get("tts_text", item["text"]))
                if legacy_generation_match and "measure_natural_duration" in saved_generation:
                    # Preserve the exact historical generation request while
                    # migrating its metadata.  No worker runs on this path.
                    resumed["measure_natural_duration"] = saved_generation[
                        "measure_natural_duration"
                    ]
                elif known.get("native_duration_control"):
                    resumed["measure_natural_duration"] = False
                resumed["language"] = _language_code(resumed["target_language_raw"]) if resumed["model"]=="omnivoice" else resumed["target_language"]
                self._job_paths(resumed)
                resumed["request_fingerprint"] = _request_fingerprint(resumed)
                try:
                    valid = (
                        known.get("request_fingerprint")
                        == resumed["request_fingerprint"]
                        or legacy_generation_match
                    )
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
                        if (
                            known.get("request_fingerprint")
                            != resumed["request_fingerprint"]
                        ):
                            # The waveform request is identical; only the old
                            # fingerprint incorrectly included audit policy.
                            # Migrate metadata in place and let content ASR run
                            # again under its independent fingerprint.
                            known = dict(known)
                            known["request_fingerprint"] = resumed[
                                "request_fingerprint"
                            ]
                            known["planned_request_fingerprint"] = item[
                                "planned_request_fingerprint"
                            ]
                            known["generation_request"] = {
                                key: resumed.get(key)
                                for key in GENERATION_REQUEST_KEYS
                            }
                            known.pop("content_asr_request_fingerprint", None)
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

        # Decide whether the planned engine can produce a naturally usable
        # utterance before treating it as a successful routing result.  This
        # lets an unsafe duration outcome enter the same fallback chain as a
        # worker/quality failure instead of being hidden by extreme atempo.
        if self.duration_rewriter is not None:
            previous = self._run_duration_retranslation(items, previous)
        previous = self._run_content_asr_audit(items, previous)

        if not getattr(self.config, "disable_tts_fallback", False):
            # Planned engine plus at most two fallback engines. A moderate
            # mismatch remains a degraded success, while a result requiring
            # unsafe compression is an engine-quality failure and retries.
            for _ in range(2):
                retry = []
                for item in items:
                    result = previous.get(item["segment_id"], {})
                    if result.get("status")=="success":
                        continue
                    if int(item.get("generation_attempt_count") or 0) >= (
                        _generation_attempt_limit()
                    ):
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
                    item["duration_attempt"] = 0
                    item["language"] = _language_code(item["target_language_raw"]) if model=="omnivoice" else item["target_language"]
                    self._job_paths(item)
                    item["request_fingerprint"] = _request_fingerprint(item)
                    item["force_regenerate"] = True
                    retry.append(item)
                if not retry:
                    if any(previous.get(i["segment_id"],{}).get("status")!="success" and
                           int(i.get("generation_attempt_count") or 0) < _generation_attempt_limit() and
                           any(m not in i["attempted_engines"] for m in self._fallback_chain(i["planned_engine"],i["target_language"])) for i in items):
                        continue
                    break
                self._execute(retry, force=True)
                previous = _load_results(self.result_path)
                if self.duration_rewriter is not None:
                    previous = self._run_duration_retranslation(retry, previous)
                previous = self._run_content_asr_audit(retry, previous)

        # Strict checks guide retries, but they must not make an autonomous
        # Agent abandon the whole deliverable.  Promote the best audible TTS
        # candidate only after every configured engine/rewrite has failed.
        for item in items:
            sid = item["segment_id"]
            if previous.get(sid, {}).get("status") == "success":
                continue
            adopted = self._adopt_best_generated_candidate(item)
            if adopted is not None:
                previous[sid] = adopted
        _atomic_json(self.result_path, previous)

        clips, output_rows = [], []
        for item in items:
            sid = item["segment_id"]
            metadata = dict(previous.get(sid, {}))
            metadata.update(segment_id=sid, planned_engine=item["planned_engine"],
                            rule_planned_engine=item.get("rule_planned_engine", item["planned_engine"]),
                            speaker_engine_locked=bool(item.get("speaker_engine_locked", False)),
                            actual_engine=item["model"], tts_engine=item["model"],
                            model_key=item["model"], model=MODEL_DISPLAY[item["model"]],
                            selected_model=MODEL_DISPLAY[item["model"]],
                            fallback_count=item["fallback_count"], fallback_reason=item["fallback_reason"],
                            tts_text=item["text"], planned_text=item["planned_text"],
                            planned_request_fingerprint=item["planned_request_fingerprint"],
                            request_fingerprint=item["request_fingerprint"], target_duration=item["target_duration"],
                            source_target_duration=item.get("source_target_duration"),
                            minimum_pronounceable_duration=item.get(
                                "minimum_pronounceable_duration"
                            ),
                            target_duration_floor_applied=bool(
                                item.get("target_duration_floor_applied", False)
                            ),
                            generation_attempt_count=int(
                                item.get("generation_attempt_count") or 0
                            ),
                            generation_attempt_limit=_generation_attempt_limit(),
                            generation_request={
                                key: item.get(key)
                                for key in GENERATION_REQUEST_KEYS
                            },
                            placement_start=item["start"],
                            placement_end=item["end"],
                            placement_source=item.get("placement_source", "segment_start"),
                            speaker=item["speaker"], emotion=item["emotion"],
                            reference_manifest_path=str(
                                self.root / "reference_manifest.json"
                            ),
                             reference_segment_id=item.get("reference_segment_id"),
                             reference_purity_score=item.get("reference_purity_score"),
                             reference_duration=item.get("reference_duration"),
                             reference_reliability=item.get("reference_reliability"),
                             voice_identity=item.get("voice_identity", item["speaker"]),
                             voice_identity_reused=item.get("voice_identity_reused", False),
                            reference_fallback=item.get("reference_fallback", False),
                            reference_fallback_speaker=item.get("reference_fallback_speaker"),
                            reference_fallback_reason=item.get("reference_fallback_reason", ""),
                            source_language=item["source_language"], target_language=item["target_language"],
                            target_language_raw=item["target_language_raw"])
            if metadata.get("status")=="success":
                clips.append({"segment_id":sid, "start":item["start"], "end":item["end"], "file":metadata["file"],
                              "tts_engine":item["model"], "request_fingerprint":item["request_fingerprint"],
                              "allow_overlap":bool(item.get("allow_overlap", False)),
                              "allowed_overlap_seconds":float(item.get("allowed_overlap_seconds", 0.0)),
                              "source_overlap_evidence":item.get("source_overlap_evidence", "none")})
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

    def _run_duration_retranslation(self, items, results):
        """Measure formal TTS, request at most two rewrites, keep the closest."""
        maximum_rewrites = max(
            0, min(2, int(os.getenv("AUTODUB_TTS_DURATION_REWRITES", "2")))
        )
        updated_results = dict(results)
        for item in items:
            sid = item["segment_id"]
            metadata = dict(updated_results.get(sid, {}))
            if metadata.get("status") != "success":
                continue
            if (
                int(metadata.get("duration_translation_attempts") or 0) > 0
                and Path(str(metadata.get("file") or "")).is_file()
            ):
                # A completed three-version decision is immutable on resume.
                # Its raw attempt files are not part of the public checkpoint.
                continue
            candidates = []

            def remember(attempt_index, rewrite_info=None):
                current = dict(_load_results(self.result_path).get(sid, {}))
                if current.get("status") != "success":
                    return None
                natural = float(
                    current.get("post_trim_duration")
                    or current.get("pre_trim_duration")
                    or current.get("actual_duration")
                    or 0.0
                )
                fit = natural_duration_fit(
                    natural,
                    item["target_duration"],
                    str(
                        (item.get("source_segment") or {}).get(
                            "translation_selection_source", "standard"
                        )
                    ),
                )
                candidate = {
                    "attempt": attempt_index,
                    "text": item["text"],
                    "model": item["model"],
                    "raw_path": item["raw_path"],
                    "final_path": item["final_path"],
                    "request_fingerprint": item["request_fingerprint"],
                    "metadata": current,
                    "fit": fit,
                    "rewrite": rewrite_info or {},
                }
                candidates.append(candidate)
                return candidate

            current_candidate = remember(0)
            if current_candidate is None:
                continue
            for rewrite_index in range(1, maximum_rewrites + 1):
                if int(item.get("generation_attempt_count") or 0) >= (
                    _generation_attempt_limit()
                ):
                    break
                if current_candidate["fit"]["natural_duration_fit"] == "passed":
                    break
                try:
                    rewrite = self.duration_rewriter(
                        dict(item.get("source_segment") or {}),
                        item["text"],
                        current_candidate["fit"],
                        [
                            {
                                "text": candidate["text"],
                                **candidate["fit"],
                            }
                            for candidate in candidates
                        ],
                    )
                except Exception as exc:
                    print(
                        f"⚠️ segment={sid} 时长定向重译失败，保留已有候选: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    continue
                rewritten_text = str((rewrite or {}).get("text") or "").strip()
                if not rewritten_text or rewritten_text in item["attempted_texts"]:
                    continue
                item["text"] = rewritten_text
                item["attempted_texts"].append(rewritten_text)
                item["duration_attempt"] = rewrite_index
                self._job_paths(item)
                item["request_fingerprint"] = _request_fingerprint(item)
                item["force_regenerate"] = True
                self._execute([item], force=True)
                current_candidate = remember(rewrite_index, rewrite)
                if current_candidate is None:
                    break

            valid = [
                candidate for candidate in candidates
                if candidate["metadata"].get("status") == "success"
                and candidate["fit"]["natural_duration"] > 0.0
            ]
            if not valid:
                continue
            passing = [
                candidate for candidate in valid
                if candidate["fit"]["natural_duration_fit"] == "passed"
            ]
            selection_pool = passing or valid
            chosen = None
            selector_result = None
            if self.duration_candidate_selector is not None and len(valid) > 1:
                try:
                    selector_result = self.duration_candidate_selector(
                        dict(item.get("source_segment") or {}),
                        [
                            {
                                "attempt": candidate["attempt"],
                                "text": candidate["text"],
                                **candidate["fit"],
                                "rewrite_audit": candidate.get("rewrite") or {},
                            }
                            for candidate in valid
                        ],
                    )
                    selected_attempt = int(
                        (selector_result or {}).get("selected_attempt")
                    )
                    chosen = next(
                        (
                            candidate for candidate in selection_pool
                            if int(candidate["attempt"]) == selected_attempt
                        ),
                        None,
                    )
                except Exception as exc:
                    print(
                        f"⚠️ segment={sid} 译文上下文终选失败，使用本地排序: "
                        f"{type(exc).__name__}: {exc}"
                    )
            if chosen is None:
                if selector_result is not None:
                    selector_result = None
                # Rewrites are not intrinsically better than the original.
                # Prefer an explicitly context-approved rewrite; otherwise
                # preserve the earlier/native wording before using duration as
                # the final tie-breaker.
                chosen = min(
                    selection_pool,
                    key=lambda candidate: (
                        -int(bool((candidate.get("rewrite") or {}).get(
                            "preferred_for_context", False
                        ))),
                        0 if int(candidate["attempt"]) == 0 else 1,
                        float(candidate["fit"]["duration_log_distance"]),
                    ),
                )
            item.update({
                "text": chosen["text"],
                "model": chosen["model"],
                "tts_engine": chosen["model"],
                "raw_path": chosen["raw_path"],
                "final_path": chosen["final_path"],
                "request_fingerprint": chosen["request_fingerprint"],
                "duration_attempt": chosen["attempt"],
            })
            final_metadata = dict(chosen["metadata"])
            native_duration_control = False
            chosen_safety = duration_adaptation_safety(
                chosen["fit"]["natural_duration"], item["target_duration"]
            )
            # F5 can control output duration inside the acoustic model.  Use
            # that as the preferred final pass after selecting the best text,
            # but only when the unconstrained result is already close enough
            # to be credible.  A huge duration anomaly may contain babble or
            # reference leakage and must fall back to another engine instead
            # of being hidden inside native duration control.
            if (
                item["model"] == "f5tts"
                and chosen["fit"]["natural_duration_fit"] != "passed"
                and chosen_safety["safe_to_force_fit"]
                and not chosen_safety["duration_anomaly"]
                and int(item.get("generation_attempt_count") or 0)
                < _generation_attempt_limit()
            ):
                native_item = dict(item)
                native_item["measure_natural_duration"] = False
                safe = _safe_id(native_item["segment_id"])
                attempt_suffix = (
                    f"_duration_{native_item['duration_attempt']}"
                    if int(native_item.get("duration_attempt", 0) or 0) else ""
                )
                native_item["raw_path"] = str(
                    self.raw_dir
                    / f"segment_{safe}_f5tts{attempt_suffix}_native.wav"
                )
                native_item["final_path"] = str(
                    self.final_dir
                    / f"segment_{safe}_f5tts{attempt_suffix}_native.wav"
                )
                native_item["request_fingerprint"] = _request_fingerprint(native_item)
                native_item["force_regenerate"] = True
                self._execute([native_item], force=True)
                native_metadata = dict(
                    _load_results(self.result_path).get(sid, {})
                )
                if native_metadata.get("status") == "success":
                    native_duration = float(
                        native_metadata.get("post_trim_duration")
                        or native_metadata.get("actual_duration")
                        or 0.0
                    )
                    native_safety = duration_adaptation_safety(
                        native_duration, item["target_duration"]
                    )
                    if not native_safety["unsafe_duration_compression"]:
                        item.update(native_item)
                        final_metadata = native_metadata
                        native_duration_control = True
            try:
                post = _postprocess(
                    Path(item["raw_path"]), Path(item["final_path"]),
                    item["target_duration"], allow_force_fit=False,
                )
                final_metadata.update(post)
                hard_duration = float(
                    item.get("hard_duration") or item["target_duration"]
                )
                if (
                    hard_duration > 0.04
                    and float(post["actual_duration"]) > hard_duration + 0.01
                ):
                    collision_safety = duration_adaptation_safety(
                        float(post.get("post_trim_duration") or post["actual_duration"]),
                        hard_duration,
                    )
                    final_metadata.update({
                        f"collision_{key}": value
                        for key, value in collision_safety.items()
                    })
                    if collision_safety["unsafe_duration_compression"]:
                        final_metadata.update(
                            status="failed",
                            duration_quality="failed",
                            duration_degraded_reason=(
                                "unsafe_compression_after_duration_retries"
                            ),
                            error=(
                                "generated audio requires unsafe compression: "
                                f"{collision_safety['duration_ratio']:.3f}x > "
                                f"{collision_safety['maximum_safe_compression_ratio']:.3f}x"
                            ),
                            file=item["final_path"],
                            preview_file=item["final_path"],
                            preview_available=True,
                            adoption_recommended=False,
                        )
                    else:
                        forced = _apply_atempo_exact(
                            Path(item["final_path"]), Path(item["final_path"]),
                            hard_duration,
                        )
                        final_metadata.update(forced)
                        final_metadata["forced_duration_reason"] = (
                            "exceeds_collision_window_after_three_translations"
                        )
                        final_metadata["actual_duration"] = _wav_duration(
                            Path(item["final_path"])
                        )
                actual = float(final_metadata["actual_duration"])
                final_metadata.update(duration_quality(actual, item["target_duration"]))
                final_metadata.update(audible_speech_quality(
                    Path(item["final_path"]), expected_text=item["text"]
                ))
                if final_metadata.get("audible_speech_quality") == "failed":
                    final_metadata.update(
                        status="failed",
                        error=(
                            "generated WAV has no reliable audible speech: "
                            + str(final_metadata.get("audible_speech_reason") or "unknown")
                        ),
                        preview_file=item["final_path"],
                        preview_available=True,
                        adoption_recommended=False,
                    )
                if (
                    final_metadata.get("status") != "failed"
                    and final_metadata.get("duration_quality") == "failed"
                ):
                    final_metadata.update(
                        duration_quality="degraded",
                        duration_degraded_reason=(
                            "closest_valid_translation_after_duration_retries"
                        ),
                    )
                final_metadata.update(chosen["fit"])
                if (
                    final_metadata.get("status") != "failed"
                    and chosen["fit"]["natural_duration_fit"] != "passed"
                    and chosen_safety.get("unsafe_duration_slowdown")
                ):
                    # A severely under-length natural generation often means
                    # swallowed syllables or an incomplete utterance.  Small
                    # atempo stretching can make its file length look valid but
                    # cannot restore missing words, so let the engine fallback
                    # chain regenerate the sentence instead.
                    final_metadata.update(
                        status="failed",
                        duration_quality="failed",
                        duration_degraded_reason=(
                            "implausibly_short_natural_generation"
                        ),
                        error=(
                            "natural TTS output is too short to trust; possible "
                            "swallowed or missing speech: "
                            f"ratio={chosen['fit']['natural_duration_ratio']:.3f}"
                        ),
                        preview_file=item["final_path"],
                        preview_available=True,
                        adoption_recommended=False,
                    )
                final_metadata.update(
                    file=item["final_path"],
                    audio_sha256=_file_sha256(Path(item["final_path"])),
                    request_fingerprint=item["request_fingerprint"],
                    duration_translation_attempts=len(valid),
                    duration_rewrite_count=max(0, len(valid) - 1),
                    duration_selected_attempt=chosen["attempt"],
                    duration_selection_reason=(
                        "f5_native_duration_final_pass"
                        if native_duration_control else (
                            "context_native_candidate_selection"
                            if selector_result else (
                                "semantic_first_natural_fit" if passing
                                else "semantic_first_best_available_translation"
                            )
                        )
                    ),
                    duration_candidate_selector=selector_result or {},
                    native_duration_control=bool(native_duration_control),
                    duration_attempt_history=[
                        {
                            "attempt": candidate["attempt"],
                            "text": candidate["text"],
                            "natural_duration": round(
                                candidate["fit"]["natural_duration"], 4
                            ),
                            "ratio": round(
                                candidate["fit"]["natural_duration_ratio"], 4
                            ),
                            "fit": candidate["fit"]["natural_duration_fit"],
                            "direction": candidate["fit"][
                                "duration_rewrite_direction"
                            ],
                            "rewrite_audit": candidate.get("rewrite") or {},
                        }
                        for candidate in candidates
                    ],
                )
                if final_metadata.get("status") != "failed":
                    final_metadata["status"] = "success"
            except Exception as exc:
                final_metadata.update(
                    status="failed",
                    error=f"final duration selection failed: {exc}",
                )
            updated_results[sid] = final_metadata
            self._remember_candidate(
                item, final_metadata, "duration_selection"
            )
            print(
                f"[TTS duration] segment={sid} attempts={len(valid)} "
                f"selected={chosen['attempt']} natural="
                f"{chosen['fit']['natural_duration']:.2f}/"
                f"{item['target_duration']:.2f}s ratio="
                f"{chosen['fit']['natural_duration_ratio']:.3f}"
            )
            _atomic_json(self.result_path, updated_results)
        return updated_results

    def _job_paths(self, item):
        suffix = f"_{item['model']}"
        duration_attempt = int(item.get("duration_attempt", 0) or 0)
        if duration_attempt:
            suffix += f"_duration_{duration_attempt}"
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
            model_items = [item for item in items if item["model"]==model]
            exhausted = [
                item for item in model_items
                if int(item.get("generation_attempt_count") or 0)
                >= _generation_attempt_limit()
            ]
            for item in exhausted:
                results[item["segment_id"]] = {
                    **dict(results.get(item["segment_id"], {})),
                    "segment_id": item["segment_id"],
                    "status": "failed",
                    "error": "generation_attempt_budget_exhausted",
                    "request_fingerprint": item["request_fingerprint"],
                    "generation_attempt_count": int(
                        item.get("generation_attempt_count") or 0
                    ),
                    "generation_attempt_limit": _generation_attempt_limit(),
                }
            pending = [item for item in model_items if item not in exhausted]
            if not pending:
                if exhausted:
                    _atomic_json(self.result_path, results)
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
            for item in pending:
                item["generation_attempt_count"] = (
                    int(item.get("generation_attempt_count") or 0) + 1
                )
            _atomic_json(self.plan_path, {"items":pending})
            for item in pending:
                results[item["segment_id"]] = {
                    "segment_id":item["segment_id"],"status":"pending","error":"",
                    "request_fingerprint":item["request_fingerprint"],
                    "generation_attempt_count":item["generation_attempt_count"],
                    "generation_attempt_limit":_generation_attempt_limit(),
                }
            _atomic_json(self.result_path, results)
            code = _run_worker(model, self.plan_path, self.result_path, force=force)
            results = _load_results(self.result_path)
            for item in pending:
                sid = item["segment_id"]
                metadata = dict(results.get(sid, {}))
                metadata.update(
                    generation_attempt_count=int(
                        item.get("generation_attempt_count") or 0
                    ),
                    generation_attempt_limit=_generation_attempt_limit(),
                )
                if metadata.get("status")=="pending":
                    metadata.update(status="failed",error=f"{model} worker exited with {code}")
                if metadata.get("status")=="success":
                    try:
                        if self.duration_rewriter is None:
                            post = _postprocess(
                                Path(item["raw_path"]), Path(item["final_path"]),
                                item["target_duration"],
                            )
                        else:
                            post = _postprocess(
                                Path(item["raw_path"]), Path(item["final_path"]),
                                item["target_duration"], allow_force_fit=False,
                            )
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
                        metadata.update(audible_speech_quality(
                            Path(item["final_path"]), expected_text=item["text"]
                        ))
                        metadata.update(voice_consistency_quality(
                            Path(item["ref_audio"]), Path(item["final_path"])
                        ))
                        # 电音检测：block policy turns a detected artifact into a
                        # failure so the fallback chain can try another engine.
                        if metadata.get("audible_speech_quality") == "failed":
                            metadata.update(
                                status="failed",
                                error=(
                                    "generated WAV has no reliable audible speech: "
                                    + str(metadata.get("audible_speech_reason") or "unknown")
                                ),
                                preview_file=item["final_path"],
                                preview_available=True,
                                adoption_recommended=False,
                            )
                        elif (metadata.get("electricity_quality")=="failed" and
                                os.getenv("AUTODUB_TTS_ELECTRICITY_POLICY","warn").strip().lower()=="block"):
                            metadata.update(status="failed", error=metadata.get("electricity_reason","electric artifact detected"))
                        elif (metadata.get("voice_consistency_quality")=="failed" and
                                os.getenv("AUTODUB_TTS_VOICE_CONSISTENCY_POLICY","block").strip().lower()=="block" and
                                item.get("reference_reliability", "reliable") == "reliable"):
                            metadata.update(status="failed", error=metadata.get("voice_consistency_reason","voice consistency check failed"))
                        elif (
                            metadata.get("unsafe_duration_compression")
                            and self.duration_rewriter is None
                        ):
                            metadata.update(
                                status="failed",
                                error=(
                                    "generated audio requires unsafe compression: "
                                    f"{float(metadata.get('duration_ratio') or 0):.3f}x > "
                                    f"{float(metadata.get('maximum_safe_compression_ratio') or 0):.3f}x"
                                ),
                                preview_file=item["final_path"],
                                preview_available=True,
                                adoption_recommended=False,
                            )
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
                self._remember_candidate(item, metadata, "engine_generation")
            _atomic_json(self.result_path, results)
