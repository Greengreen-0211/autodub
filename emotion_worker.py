#!/usr/bin/env python3
"""Isolated, single-load emotion2vec worker."""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
import wave
from pathlib import Path
from typing import Any, Sequence

from scheme3_policy import EMOTION_LABEL_MAP, normalize_emotion


RAW_LABEL_ALIASES = {
    "anger": "angry", "angry": "angry",
    "disgust": "disgusted", "disgusted": "disgusted",
    "fear": "fearful", "fearful": "fearful",
    "happiness": "happy", "happy": "happy",
    "neutral": "neutral", "other": "other",
    "sadness": "sad", "sad": "sad",
    "surprise": "surprised", "surprised": "surprised",
    "unknown": "unknown",
}
def atomic_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def flatten_once(value: Any) -> list[Any]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        value = list(value) if isinstance(value, Sequence) and not isinstance(value, str) else [value]
    if len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    return value


def raw_label(value: Any) -> str:
    raw = str(value or "unknown").strip().lower()
    for candidate in reversed([part.strip() for part in raw.replace("｜", "/").split("/")]):
        if candidate in RAW_LABEL_ALIASES:
            return RAW_LABEL_ALIASES[candidate]
    return raw or "unknown"


def extract_prediction(payload: Any) -> tuple[str, float]:
    item = payload[0] if isinstance(payload, list) and payload else payload
    if not isinstance(item, dict):
        raise TypeError(f"unexpected FunASR result: {type(item)!r}")
    labels = flatten_once(item.get("labels", []))
    scores = [float(score) for score in flatten_once(item.get("scores", []))]
    if not labels or len(labels) != len(scores):
        raise ValueError(f"invalid labels/scores: labels={labels!r}, scores={scores!r}")
    best = max(range(len(scores)), key=scores.__getitem__)
    return raw_label(labels[best]), scores[best]


def load_results(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return {str(k): v for k, v in value.items()}


def audio_duration(path: str) -> float:
    try:
        with wave.open(path, "rb") as handle:
            return handle.getnframes() / float(handle.getframerate())
    except Exception:
        import soundfile as sf
        info = sf.info(path)
        return info.frames / float(info.samplerate)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    os.environ["PYTHONNOUSERSITE"] = "1"
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    result_path = Path(args.result)
    results = load_results(result_path)
    model_dir = Path(args.model_dir).expanduser().resolve()
    if not model_dir.is_dir() or not any(model_dir.iterdir()):
        raise FileNotFoundError(f"emotion2vec local model is missing: {model_dir}")

    from funasr import AutoModel

    print(f"emotion2vec model={model_dir} device={args.device}")
    model = AutoModel(model=str(model_dir), hub="ms", device=args.device, disable_update=True)
    items = list(plan.get("items", []))
    for index, item in enumerate(items, start=1):
        segment_id = str(item["segment_id"])
        started = time.perf_counter()
        print(f"[Emotion {index}/{len(items)}] segment_id={segment_id}")
        try:
            prediction = model.generate(
                str(item["audio_path"]),
                output_dir=str(result_path.parent / "raw_outputs"),
                granularity="utterance",
                extract_embedding=False,
            )
            raw, score = extract_prediction(prediction)
            duration = float(item.get("duration_seconds") or audio_duration(str(item["audio_path"])))
            normalized = normalize_emotion(raw, score=score, duration=duration)
            fallback_reason = ""
            if duration < float(os.getenv("AUTODUB_EMOTION_MIN_DURATION", "0.8")):
                fallback_reason = "shorter_than_min_duration"
            elif score < float(os.getenv("AUTODUB_EMOTION_MIN_SCORE", "0.5")):
                fallback_reason = "below_min_confidence"
            elif raw not in EMOTION_LABEL_MAP:
                fallback_reason = "unknown_label"
            results[segment_id] = {
                **results.get(segment_id, {}),
                "segment_id": segment_id,
                "audio_path": str(item["audio_path"]),
                "raw_emotion": raw,
                "normalized_emotion": normalized,
                "emotion_score": float(score),
                "duration_seconds": duration,
                "fallback_reason": fallback_reason,
                "inference_seconds": time.perf_counter() - started,
                "status": "success",
                "error": "",
            }
            print(f"  -> raw={raw} normalized={results[segment_id]['normalized_emotion']} score={score:.4f}")
        except Exception as exc:
            traceback.print_exc()
            results[segment_id] = {
                **results.get(segment_id, {}),
                "segment_id": segment_id,
                "audio_path": str(item["audio_path"]),
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
        atomic_json(result_path, results)


if __name__ == "__main__":
    main()
