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
    issues=[]
    if len(by_id)!=len(clips) or len(set(ids))!=len(ids) or set(by_id)-set(ids):
        raise RuntimeError("Step5 blocked: duplicate/unknown segment IDs in TTS clips")
    for sid in ids:
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
    if manifest.get("success")!=len(rows) or len(clips)!=len(rows):
        issues.append(f"incomplete Step4: {manifest.get('success')}/{len(rows)}")
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
