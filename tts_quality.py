"""Duration quality and complete-dialogue gate before Step 5."""
from __future__ import annotations
import hashlib
import json
import os
import math
from pathlib import Path


def duration_quality(actual: float, target: float) -> dict:
    absolute=abs(actual-target)
    ratio=actual/target if target>0 else 0.0
    limit=max(float(os.getenv("AUTODUB_TTS_MAX_DURATION_ERROR", "0.6")),
              target*float(os.getenv("AUTODUB_TTS_MAX_DURATION_RATIO_ERROR", "0.25")))
    return {"duration_error":actual-target,"duration_ratio":ratio,
            "duration_quality":"passed" if actual>0 and target>0 and absolute<=limit else "failed"}


def duration_adaptation_safety(actual: float, target: float) -> dict:
    """Describe whether a generated clip can be time-scaled without abuse.

    ``actual / target`` is deliberately the single source of truth.  A very
    long utterance is often more than a timing problem: it can contain model
    babble, copied reference speech, or added words.  Hiding that evidence
    behind a 5-10x ``atempo`` chain produces an apparently valid WAV whose
    content is no longer intelligible.  Such a result must be retried or
    exposed as a degraded preview, never silently force-fitted.
    """
    actual = float(actual or 0.0)
    target = float(target or 0.0)
    ratio = actual / target if actual > 0.0 and target > 0.0 else 0.0
    maximum_compression = max(
        1.0,
        float(os.getenv("AUTODUB_TTS_MAX_SAFE_COMPRESSION_RATIO", "1.50")),
    )
    maximum_slowdown = max(
        1.0,
        float(os.getenv("AUTODUB_TTS_MAX_SAFE_SLOWDOWN_RATIO", "1.25")),
    )
    required_speed = ratio
    required_slowdown = (1.0 / ratio) if ratio > 0.0 else 0.0
    compression_safe = 0.0 < ratio <= maximum_compression
    slowdown_safe = ratio > 0.0 and required_slowdown <= maximum_slowdown
    return {
        "duration_ratio": ratio,
        "required_atempo_factor": required_speed,
        "required_slowdown_factor": required_slowdown,
        "maximum_safe_compression_ratio": maximum_compression,
        "maximum_safe_slowdown_ratio": maximum_slowdown,
        "safe_to_force_fit": bool(compression_safe and slowdown_safe),
        "unsafe_duration_compression": bool(ratio > maximum_compression),
        "unsafe_duration_slowdown": bool(
            ratio > 0.0 and required_slowdown > maximum_slowdown
        ),
        "duration_anomaly": bool(
            ratio
            > float(os.getenv("AUTODUB_TTS_GENERATION_ANOMALY_RATIO", "2.00"))
        ),
    }


def natural_duration_fit(actual: float, target: float, provenance: str = "") -> dict:
    """Judge unwarped, silence-trimmed TTS against the source speech window.

    Every decision is expressed as ``synthesized / source-slot``.  Trusted OCR
    wording receives a wider *naturalness* band, but never a wider physical
    collision window.  This decision must be made before atempo;
    postprocessed audio is not evidence that the translation itself fits
    naturally.
    """
    actual = float(actual or 0.0)
    target = float(target or 0.0)
    ratio = actual / target if actual > 0.0 and target > 0.0 else 0.0
    absolute = abs(actual - target)
    trusted_ocr = str(provenance or "").lower() in {
        "ocr_direct", "ocr_repaired", "ocr_hybrid", "ocr_assisted",
    }
    if trusted_ocr:
        lower = float(os.getenv("AUTODUB_TTS_OCR_NATURAL_MIN_RATIO", "0.68"))
        upper = float(os.getenv("AUTODUB_TTS_OCR_NATURAL_MAX_RATIO", "1.35"))
        rule = "trusted_ocr_ratio"
    elif target < 1.0:
        lower = float(os.getenv("AUTODUB_TTS_NATURAL_MIN_RATIO_TINY", "0.60"))
        upper = float(os.getenv("AUTODUB_TTS_NATURAL_MAX_RATIO_TINY", "1.40"))
        rule = "tiny_ratio"
    elif target < 2.0:
        lower = float(os.getenv("AUTODUB_TTS_NATURAL_MIN_RATIO_SHORT", "0.75"))
        upper = float(os.getenv("AUTODUB_TTS_NATURAL_MAX_RATIO_SHORT", "1.25"))
        rule = "short_ratio"
    else:
        lower = float(os.getenv("AUTODUB_TTS_NATURAL_MIN_RATIO", "0.80"))
        upper = float(os.getenv("AUTODUB_TTS_NATURAL_MAX_RATIO", "1.20"))
        passed = lower <= ratio <= upper
        rule = "standard_ratio"
    passed = actual > 0.0 and lower <= ratio <= upper
    direction = "suitable"
    if not passed:
        direction = "expand" if actual < target else "compress"
    distance = abs(math.log(max(ratio, 1e-6))) if ratio > 0.0 else float("inf")
    return {
        "natural_duration": actual,
        "target_duration": target,
        "natural_duration_ratio": ratio,
        "natural_duration_error": actual - target,
        "natural_duration_absolute_error": absolute,
        "natural_duration_lower_ratio": lower,
        "natural_duration_upper_ratio": upper,
        "natural_duration_rule": rule,
        "natural_duration_provenance": str(provenance or "standard"),
        "natural_duration_fit": "passed" if passed else "failed",
        "duration_rewrite_direction": direction,
        "duration_log_distance": distance,
    }


def electricity_metrics(path):
    """Spectral metrics that reveal metallic / robotic ("电音") artifacts.

    Vocoder artifacts make the spectrum flatter (noise-like) and add high
    frequency energy; natural speech is strongly harmonic (very low spectral
    flatness).  Metrics are computed at the native sample rate so that 8-24 kHz
    artifacts are visible.  Returns None when the clip cannot be analysed.
    """
    try:
        import numpy as np
        import librosa
    except ImportError:
        return None
    try:
        y, sr = librosa.load(str(path), sr=None, mono=True)
    except Exception:
        return None
    if y is None or len(y) < int(sr * 0.05):
        return None
    rms = float(np.sqrt(np.mean(np.asarray(y, dtype=np.float64) ** 2)))
    if rms <= 1e-4:
        return {"spectral_flatness": None, "hf_energy_ratio": None,
                "rms": 0.0, "duration": round(len(y) / sr, 4), "sample_rate": sr,
                "skipped": "silent"}
    n_fft = 2048
    power = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=512)) ** 2
    eps = 1e-12
    flatness = np.exp(np.mean(np.log(power + eps), axis=0)) / (np.mean(power, axis=0) + eps)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    hf_mask = freqs >= 8000
    hf_ratio = float(power[hf_mask].sum() / (power.sum() + eps)) if hf_mask.any() else 0.0
    return {
        "spectral_flatness": round(float(np.median(flatness)), 6),
        "hf_energy_ratio": round(hf_ratio, 6),
        "rms": round(rms, 6),
        "duration": round(len(y) / sr, 4),
        "sample_rate": sr,
    }


def electricity_quality(path):
    """Classify a synthesized clip for 电音 (metallic/robotic) artifacts.

    Policy (env AUTODUB_TTS_ELECTRICITY_POLICY):
      "off"           -> skip the analysis entirely (fast path)
      "warn" (default)-> compute and record metrics, never fail the segment
      "block"         -> compute and fail the segment when an artifact is found,
                         which triggers the engine fallback chain in the router

    Thresholds are deliberately conservative because a false positive under
    "block" would waste GPU re-synthesis; calibrate them against real 电音
    samples before enabling "block".
    """
    policy = os.getenv("AUTODUB_TTS_ELECTRICITY_POLICY", "warn").strip().lower()
    if policy == "off":
        return {"electricity_quality": "skipped", "electricity_policy": policy}
    metrics = electricity_metrics(path)
    if metrics is None:
        return {"electricity_quality": "skipped", "electricity_policy": policy,
                "electricity_reason": "analysis failed"}
    if metrics.get("skipped"):
        return {"electricity_quality": "skipped", "electricity_policy": policy,
                **metrics}
    max_sf = float(os.getenv("AUTODUB_TTS_MAX_SPECTRAL_FLATNESS", "0.05"))
    max_hf = float(os.getenv("AUTODUB_TTS_MAX_HF_ENERGY_RATIO", "0.08"))
    sf = metrics["spectral_flatness"] or 0.0
    hf = metrics["hf_energy_ratio"] or 0.0
    failed = sf > max_sf and hf > max_hf
    reason = ""
    if failed:
        reason = (f"spectral_flatness={sf:.4f}>{max_sf} and "
                  f"hf_energy_ratio={hf:.4f}>{max_hf}")
    return {"electricity_quality": "failed" if failed else "passed",
            "electricity_policy": policy,
            "electricity_reason": reason,
            **metrics}


def _voice_proxy_embedding(path):
    """Return a lightweight timbre embedding for post-TTS consistency audit.

    This is intentionally a proxy built from normalized MFCC statistics, not
    a replacement for WeSpeaker clustering. It is language-independent enough
    to catch gross reference/voice mismatches and remains advisory by default.
    """
    try:
        import numpy as np
        import librosa
        y, sr = librosa.load(str(path), sr=16000, mono=True)
    except Exception:
        return None
    if y is None or len(y) < 1600:
        return None
    try:
        y, _ = librosa.effects.trim(y, top_db=35)
        if len(y) < 1600:
            return None
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        delta = librosa.feature.delta(mfcc)
        vector = np.concatenate([
            np.mean(mfcc, axis=1),
            np.std(mfcc, axis=1),
            np.mean(delta, axis=1),
            np.std(delta, axis=1),
        ]).astype(float)
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 1e-9 else None
    except Exception:
        return None


def voice_consistency_quality(reference_path, synthesized_path):
    """Audit whether synthesized timbre remains close to its chosen reference.

    Policy mirrors the electric-artifact gate. ``warn`` records the result;
    ``block`` lets the router try its existing fallback chain. The conservative
    default threshold should be calibrated with production samples before
    enabling blocking.
    """
    policy = os.getenv(
        "AUTODUB_TTS_VOICE_CONSISTENCY_POLICY", "warn"
    ).strip().lower()
    if policy == "off":
        return {
            "voice_consistency_quality": "skipped",
            "voice_consistency_policy": policy,
        }
    reference = _voice_proxy_embedding(reference_path)
    synthesized = _voice_proxy_embedding(synthesized_path)
    if reference is None or synthesized is None:
        return {
            "voice_consistency_quality": "skipped",
            "voice_consistency_policy": policy,
            "voice_consistency_reason": "feature extraction unavailable",
            "voice_consistency_backend": "mfcc_proxy",
        }
    similarity = float(reference @ synthesized)
    minimum = float(os.getenv(
        "AUTODUB_TTS_MIN_VOICE_CONSISTENCY", "0.45"
    ))
    failed = similarity < minimum
    return {
        "voice_consistency_quality": "failed" if failed else "passed",
        "voice_consistency_policy": policy,
        "voice_consistency_similarity": round(similarity, 6),
        "voice_consistency_threshold": minimum,
        "voice_consistency_backend": "mfcc_proxy",
        "voice_consistency_reason": (
            f"proxy_similarity={similarity:.4f}<{minimum:.4f}"
            if failed else ""
        ),
    }


def stage_fingerprint(state, config, backend):
    segments=[{k:s.get(k) for k in ("id","text","start","end","speaker","emotion","tts_engine",
                                   "translation_request_fingerprint", "tts_placement_start",
                                   "tts_hard_end", "tts_target_duration",
                                   "tts_timing_precision", "tts_timing_reason",
                                   "speaker_boundary_ambiguous",
                                   "translation_selection_source")} for s in state.get("segments_step3",[])]
    media={}
    for key in ("vocals_path","accomp_path"):
        path=Path(str(state.get(key) or ""))
        if path.is_file():
            info=path.stat()
            media[key]={"path":str(path.resolve()),"size":info.st_size,"mtime_ns":info.st_mtime_ns}
    payload={"version":"tts-stage-v2.5.0-context-ratio-overlap","segments":segments,"media":media,"backend":backend,
             "input_file":state.get("input_file"),
             "target_lang":str(config.target_lang).lower(),
             "separator":state.get("separator",state.get("separator_backend",os.getenv("AUTODUB_SEPARATOR","htdemucs")))}
    return hashlib.sha256(json.dumps(payload,ensure_ascii=False,sort_keys=True).encode()).hexdigest()


def validate_mix_state(state, config, *, allow_missing=False, filter_missing=True):
    from tts_router import _wav_duration
    rows=state.get("segments_step3",[])
    manifest=state.get("tts_stage_manifest",{})
    if not rows or manifest.get("fingerprint") != stage_fingerprint(state,config,state.get("tts_backend")):
        raise RuntimeError("Step5 blocked: target language/separator/TTS fingerprint missing or changed; rerun Step4")
    clips=state.get("audio_clips",[])
    by_id={str(c.get("segment_id")):c for c in clips}
    ids=[str(s.get("id",i)) for i,s in enumerate(rows)]
    results={str(r.get("segment_id")):r for r in state.get("tts_router_results",[])}
    # Segments degraded by the translation policy are expected to have no dub;
    # they are excluded from the completeness arithmetic but must not gain one.
    skipped={str(s.get("id",i)) for i,s in enumerate(rows)
             if s.get("translation_status")=="skipped"}
    dubbing=[sid for sid in ids if sid not in skipped]
    quality_issues=[]
    integrity_issues=[]
    if len(by_id)!=len(clips) or len(set(ids))!=len(ids) or set(by_id)-set(ids):
        raise RuntimeError("Step5 blocked: duplicate/unknown segment IDs in TTS clips")
    for sid in ids:
        if sid in skipped:
            if sid in by_id or sid in results:
                integrity_issues.append(f"{sid}:skipped segment unexpectedly produced TTS audio")
            continue
        result=results.get(sid,{})
        clip=by_id.get(sid,{})
        path=Path(str(clip.get("file") or ""))
        if result.get("status")!="success" or not path.is_file() or not path.stat().st_size:
            quality_issues.append(f"{sid}:missing or unsuccessful WAV")
            continue
        try:
            if _wav_duration(path)<=0:
                quality_issues.append(f"{sid}:empty WAV")
                continue
            if clip.get("request_fingerprint") != result.get("request_fingerprint"):
                raise ValueError("clip/result generation fingerprint mismatch")
            from tts_router import _request_fingerprint, _file_sha256
            if not result.get("generation_request") or _request_fingerprint(result["generation_request"]) != result.get("request_fingerprint"):
                raise ValueError("full generation fingerprint missing or changed; rerun Step4")
            if result.get("audio_sha256") != _file_sha256(path):
                raise ValueError("WAV content checksum mismatch")
        except Exception as exc:
            integrity_issues.append(f"{sid}:{exc}")
    if manifest.get("success")!=len(dubbing) or len(clips)!=len(dubbing):
        quality_issues.append(f"incomplete Step4: {manifest.get('success')}/{len(dubbing)} dubbing segments"
                              f" (of {len(rows)} total, {len(skipped)} skipped)")
    if integrity_issues:
        raise RuntimeError("Step5 blocked by integrity error: " + "; ".join(integrity_issues))
    if quality_issues:
        print("⚠️ Step5 发现可降级的 TTS 缺失:",quality_issues)
        # Only verified readable clips may reach the mixer.
        if filter_missing:
            valid=[]
            for c in clips:
                try:
                    if _wav_duration(Path(c["file"]))>0:
                        valid.append(c)
                except Exception:
                    pass
            state["audio_clips"]=valid
    return quality_issues
