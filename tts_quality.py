"""Duration quality and complete-dialogue gate before Step 5."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path


def duration_quality(actual: float, target: float) -> dict:
    absolute=abs(actual-target)
    ratio=actual/target if target>0 else 0.0
    limit=max(float(os.getenv("AUTODUB_TTS_MAX_DURATION_ERROR", "0.6")),
              target*float(os.getenv("AUTODUB_TTS_MAX_DURATION_RATIO_ERROR", "0.25")))
    return {"duration_error":actual-target,"duration_ratio":ratio,
            "duration_quality":"passed" if actual>0 and target>0 and absolute<=limit else "failed"}


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


def stage_fingerprint(state, config, backend):
    segments=[{k:s.get(k) for k in ("id","text","start","end","speaker","emotion","tts_engine",
                                   "translation_request_fingerprint")} for s in state.get("segments_step3",[])]
    media={}
    for key in ("vocals_path","accomp_path"):
        path=Path(str(state.get(key) or ""))
        if path.is_file():
            info=path.stat()
            media[key]={"path":str(path.resolve()),"size":info.st_size,"mtime_ns":info.st_mtime_ns}
    payload={"version":"tts-stage-v2.2.1","segments":segments,"media":media,"backend":backend,
             "input_file":state.get("input_file"),
             "target_lang":str(config.target_lang).lower(),
             "separator":state.get("separator",state.get("separator_backend",os.getenv("AUTODUB_SEPARATOR","htdemucs")))}
    return hashlib.sha256(json.dumps(payload,ensure_ascii=False,sort_keys=True).encode()).hexdigest()


def validate_mix_state(state, config, *, allow_missing=False):
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
    issues=[]
    if len(by_id)!=len(clips) or len(set(ids))!=len(ids) or set(by_id)-set(ids):
        raise RuntimeError("Step5 blocked: duplicate/unknown segment IDs in TTS clips")
    for sid in ids:
        if sid in skipped:
            if sid in by_id or sid in results:
                issues.append(f"{sid}:skipped segment unexpectedly produced TTS audio")
            continue
        result=results.get(sid,{})
        clip=by_id.get(sid,{})
        try:
            path=Path(str(clip.get("file") or ""))
            if result.get("status")!="success" or not path.is_file() or not path.stat().st_size:
                raise ValueError("missing or unsuccessful WAV")
            if _wav_duration(path)<=0:
                raise ValueError("empty WAV")
            if clip.get("request_fingerprint") != result.get("request_fingerprint"):
                raise ValueError("clip/result generation fingerprint mismatch")
            from tts_router import _request_fingerprint, _file_sha256
            if not result.get("generation_request") or _request_fingerprint(result["generation_request"]) != result.get("request_fingerprint"):
                raise ValueError("full generation fingerprint missing or changed; rerun Step4")
            if result.get("audio_sha256") != _file_sha256(path):
                raise ValueError("WAV content checksum mismatch")
        except Exception as exc:
            issues.append(f"{sid}:{exc}")
    if manifest.get("success")!=len(dubbing) or len(clips)!=len(dubbing):
        issues.append(f"incomplete Step4: {manifest.get('success')}/{len(dubbing)} dubbing segments"
                      f" (of {len(rows)} total, {len(skipped)} skipped)")
    if issues and not allow_missing:
        raise RuntimeError("Step5 blocked: " + "; ".join(issues))
    if issues:
        print("⚠️ --allow-missing-tts explicit override:",issues)
        # Only verified readable clips may reach the mixer.
        valid=[]
        for c in clips:
            try:
                if _wav_duration(Path(c["file"]))>0:
                    valid.append(c)
            except Exception:
                pass
        state["audio_clips"]=valid
