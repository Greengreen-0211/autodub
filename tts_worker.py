#!/usr/bin/env python3
"""Isolated TTS worker. One invocation loads one model and handles its batch."""

from __future__ import annotations

import argparse
from functools import wraps
import inspect
import json
import os
import sys
import time
import traceback
import wave
from pathlib import Path
from typing import Any


COSY_REPO = "/data0/goldenseed/qgd/CosyVoice"
COSY_MODEL = "/data0/goldenseed/qgd/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B"
CONFUCIUS_REPO = "/data0/goldenseed/qgd/Confucius4-TTS"
OMNI_MODEL = "/data0/goldenseed/qgd/models/OmniVoice"
F5_REPO = "/data0/goldenseed/cjx/F5-TTS"
F5_MODEL = "/data0/goldenseed/cjx/F5-TTS/ckpts/F5TTS_v1_Base/model_1250000.safetensors"
INDEX_TTS_ROOT = "/data0/goldenseed/models/index-tts"
INDEX_TTS_MODEL_DIR = "/data0/goldenseed/models/index-tts/checkpoints"

EMOTION_ZH = {
    "happy": "开心", "surprised": "惊讶", "sad": "悲伤", "angry": "愤怒",
    "fearful": "害怕", "disgusted": "厌恶", "neutral": "自然",
}
EMOTION_EN = {
    "happy": "happy and cheerful", "surprised": "surprised", "sad": "sad",
    "angry": "angry", "fearful": "fearful", "disgusted": "disgusted",
    "neutral": "natural",
}


def atomic_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def load_results(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return {str(key): item for key, item in value.items()}


def wav_duration(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as handle:
            return handle.getnframes() / float(handle.getframerate())
    except Exception:
        import soundfile as sf
        info = sf.info(str(path))
        return info.frames / float(info.samplerate)


def configure_strict_offline() -> None:
    """Resolve HF resources locally before the worker's network guard runs."""
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", MODELSCOPE_OFFLINE="1")
    import huggingface_hub as hub
    from huggingface_hub import file_download, _snapshot_download

    for name, module in (("hf_hub_download", file_download),
                         ("snapshot_download", _snapshot_download)):
        original = getattr(hub, name)
        if getattr(original, "_autodub_local_only", False):
            continue
        @wraps(original)
        def local_download(*args, _download=original, **kwargs):
            kwargs["local_files_only"] = True
            return _download(*args, **kwargs)
        local_download._autodub_local_only = True
        setattr(hub, name, local_download)
        setattr(module, name, local_download)

    import requests
    def offline_request(*args, **kwargs):
        raise RuntimeError("strict_offline: network request disabled; pre-download runtime resources")
    requests.sessions.Session.request = offline_request


def cosy_instruction(language: str, emotion: str) -> str:
    if language == "en":
        feeling = EMOTION_EN.get(emotion, emotion)
        return (
            "You are a helpful assistant. "
            f"Please speak in a {feeling} tone while preserving the reference speaker's voice."
            "<|endofprompt|>"
        )
    feeling = EMOTION_ZH.get(emotion, emotion)
    return f"You are a helpful assistant. 请用{feeling}语气表达，同时保持参考说话人的音色。<|endofprompt|>"


class CosyVoiceBackend:
    def __init__(self) -> None:
        repo = Path(os.getenv("COSYVOICE_REPO", COSY_REPO)).resolve()
        model_dir = os.getenv("COSYVOICE_MODEL_DIR", COSY_MODEL)
        os.chdir(repo)
        sys.path.insert(0, str(repo))
        sys.path.insert(0, str(repo / "third_party" / "Matcha-TTS"))
        import torch
        import torchaudio
        import onnxruntime as ort
        # Scope the provider override to this isolated worker, not the repository.
        provider = os.getenv("AUTODUB_COSY_ONNX_PROVIDER", "cpu").lower()
        if provider not in {"cpu", "cuda"}:
            raise ValueError("AUTODUB_COSY_ONNX_PROVIDER must be cpu or cuda")
        original_session = ort.InferenceSession
        def make_session(*args, **kwargs):
            if provider == "cpu":
                kwargs["providers"] = ["CPUExecutionProvider"]
            session = original_session(*args, **kwargs)
            print("CosyVoice3 ONNX providers:", session.get_providers())
            return session
        ort.InferenceSession = make_session
        from cosyvoice.cli.cosyvoice import AutoModel
        self.torch = torch
        self.torchaudio = torchaudio
        print(f"Loading CosyVoice3 from {model_dir}")
        self.model = AutoModel(model_dir=model_dir)
        frontend = getattr(self.model, "frontend", None)
        print(f"CosyVoice3 torch_device={'cuda' if torch.cuda.is_available() else 'cpu'} "
              f"text_frontend={getattr(frontend, 'text_frontend', 'unknown')} onnx={provider}")

    def generate(self, item: dict[str, Any], destination: Path) -> float:
        emotion = str(item.get("emotion") or "neutral").lower()
        if emotion == "neutral":
            ref_text = str(item.get("ref_text") or "").strip()
            if not ref_text:
                raise RuntimeError("CosyVoice3 zero-shot requires reference text")
            generator = self.model.inference_zero_shot(
                item["text"], "You are a helpful assistant.<|endofprompt|>" + ref_text,
                item["ref_audio"], stream=False,
            )
        else:
            generator = self.model.inference_instruct2(
                item["text"], cosy_instruction(item["language"], emotion),
                item["ref_audio"], stream=False,
            )
        chunks = []
        for value in generator:
            speech = value["tts_speech"]
            if speech.dim() == 1:
                speech = speech.unsqueeze(0)
            chunks.append(speech.detach().cpu())
        if not chunks:
            raise RuntimeError("CosyVoice3 returned no audio")
        audio = self.torch.cat(chunks, dim=-1)
        self.torchaudio.save(str(destination), audio, self.model.sample_rate)
        return audio.shape[-1] / float(self.model.sample_rate)


class F5TTSBackend:
    """Load the verified F5-TTS model once and synthesize the worker batch."""

    def __init__(self) -> None:
        repo = Path(os.getenv("F5TTS_REPO", F5_REPO)).resolve()
        checkpoint = Path(os.getenv("F5TTS_MODEL_PATH", F5_MODEL)).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"F5-TTS checkpoint is missing: {checkpoint}")
        os.chdir(repo)
        sys.path.insert(0, str(repo / "src"))
        from f5_tts.api import F5TTS

        device = "cuda"
        kwargs: dict[str, Any] = {
            "model": "F5TTS_v1_Base",
            "ckpt_file": str(checkpoint),
            "device": device,
            "hf_cache_dir": os.getenv("HF_HUB_CACHE"),
        }
        parameters = inspect.signature(F5TTS).parameters
        local_vocoder = os.getenv("F5TTS_VOCODER_LOCAL_PATH", "").strip()
        if not local_vocoder:
            for candidate in (
                repo / "ckpts" / "vocos-mel-24khz",
                repo / "checkpoints" / "vocos-mel-24khz",
            ):
                if candidate.exists():
                    local_vocoder = str(candidate)
                    break
        if "vocoder_local_path" in parameters and local_vocoder:
            kwargs["vocoder_local_path"] = local_vocoder
        if "load_vocoder_from_local" in parameters:
            kwargs["load_vocoder_from_local"] = bool(local_vocoder)
        print(f"Loading F5-TTS from {checkpoint}")
        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        constructor_args = kwargs if accepts_kwargs else {
            key: value for key, value in kwargs.items() if key in parameters
        }
        self.model = F5TTS(**constructor_args)

    def generate(self, item: dict[str, Any], destination: Path) -> float:
        ref_text = str(item.get("ref_text") or "").strip()
        if not ref_text:
            raise RuntimeError("F5-TTS requires reference text for offline inference")
        kwargs: dict[str, Any] = {
            "ref_file": item["ref_audio"],
            "ref_text": ref_text,
            "gen_text": item["text"],
            "file_wave": str(destination),
            "show_info": print,
        }
        parameters = inspect.signature(self.model.infer).parameters
        target_duration = float(item.get("target_duration") or 0.0)
        if "fix_duration" in parameters and target_duration > 0:
            from f5_tts.infer.utils_infer import preprocess_ref_audio_text
            # F5's duration includes the *processed* reference, not just output.
            # Keep ref_file unchanged so infer() hits its preprocessing cache.
            processed_ref, _ = preprocess_ref_audio_text(item["ref_audio"], ref_text, show_info=print)
            kwargs["fix_duration"] = wav_duration(Path(processed_ref)) + target_duration
            print(f"F5-TTS native target duration: {target_duration:.3f}s")
        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        infer_args = kwargs if accepts_kwargs else {
            key: value for key, value in kwargs.items() if key in parameters
        }
        result = self.model.infer(**infer_args)
        if not destination.is_file():
            if not isinstance(result, tuple) or len(result) < 2:
                raise RuntimeError("F5-TTS returned no WAV and no waveform tuple")
            import numpy as np
            import soundfile as sf
            waveform, sample_rate = result[0], int(result[1])
            sf.write(str(destination), np.asarray(waveform), sample_rate)
        return wav_duration(destination)


class IndexTTSBackend:
    """The existing baseline IndexTTS2 API, hosted in an isolated worker."""

    def __init__(self) -> None:
        root = Path(os.getenv("INDEX_TTS_ROOT", INDEX_TTS_ROOT)).resolve()
        model_dir = Path(os.getenv("INDEX_TTS_MODEL_DIR", INDEX_TTS_MODEL_DIR)).resolve()
        config = model_dir / "config.yaml"
        if not (root / "indextts" / "infer_v2.py").is_file():
            raise FileNotFoundError(f"IndexTTS2 code is missing: {root}")
        if not config.is_file():
            raise FileNotFoundError(f"IndexTTS2 config is missing: {config}")
        os.chdir(root)
        sys.path.insert(0, str(root))
        from indextts.infer_v2 import IndexTTS2
        print(f"Loading IndexTTS2 from {model_dir}")
        kwargs = dict(model_dir=str(model_dir), cfg_path=str(config), use_fp16=True)
        if "use_cuda_kernel" in inspect.signature(IndexTTS2).parameters:
            # The supported torch implementation avoids a Ninja/CUDA build.
            kwargs["use_cuda_kernel"] = False
        self.model = IndexTTS2(**kwargs)

    def generate(self, item: dict[str, Any], destination: Path) -> float:
        self.model.infer(
            spk_audio_prompt=item["ref_audio"],
            text=item["text"],
            output_path=str(destination),
            verbose=False,
        )
        if not destination.is_file():
            raise RuntimeError("IndexTTS2 returned without creating a WAV")
        return wav_duration(destination)


class ConfuciusBackend:
    def __init__(self) -> None:
        repo = Path(os.getenv("CONFUCIUS4_REPO", CONFUCIUS_REPO)).resolve()
        os.chdir(repo)
        sys.path.insert(0, str(repo))
        import torch
        import torchaudio
        from confuciustts.cli.inference import ConfuciusTTS
        self.torch = torch
        self.torchaudio = torchaudio
        config = repo / "config" / "inference_config.yaml"
        print(f"Loading Confucius4-TTS from {config}")
        self.model = ConfuciusTTS(config_path=str(config), device="cuda" if torch.cuda.is_available() else "cpu")

    def generate(self, item: dict[str, Any], destination: Path) -> float:
        language = str(item.get("language") or "en").lower()
        audio = self.model.generate(
            text=item["text"],
            lang=language if language in {"zh", "en"} else "en",
            prompt_wav=item["ref_audio"],
            verbose=False,
        )
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        audio = audio.detach().cpu()
        self.torchaudio.save(str(destination), audio, self.model.sample_rate)
        return audio.shape[-1] / float(self.model.sample_rate)


class OmniVoiceBackend:
    def __init__(self) -> None:
        import numpy as np
        import soundfile as sf
        import torch
        from omnivoice import OmniVoice
        self.np = np
        self.sf = sf
        model_dir = os.getenv("OMNIVOICE_MODEL_DIR", OMNI_MODEL)
        print(f"Loading OmniVoice offline from {model_dir}")
        self.model = OmniVoice.from_pretrained(
            model_dir,
            device_map="cuda:0" if torch.cuda.is_available() else "cpu",
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )

    def generate(self, item: dict[str, Any], destination: Path) -> float:
        # OmniVoice does not support happy/sad/angry instructions. Keep emotion
        # only in metadata and deliberately omit the instruct argument.
        audios = self.model.generate(
            text=item["text"], language=item["language"], ref_audio=item["ref_audio"],
            ref_text=str(item.get("ref_text") or "").strip() or None,
        )
        if not audios:
            raise RuntimeError("OmniVoice returned no audio")
        audio = self.np.asarray(audios[0], dtype=self.np.float32)
        self.sf.write(str(destination), audio, self.model.sampling_rate)
        return audio.shape[-1] / float(self.model.sampling_rate)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["f5tts", "indextts2", "cosyvoice3", "confucius4", "omnivoice"],
        required=True,
    )
    parser.add_argument("--plan", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    os.environ["PYTHONNOUSERSITE"] = "1"
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
        os.environ.pop(key, None)

    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    items = [item for item in plan.get("items", []) if item.get("model") == args.model]
    result_path = Path(args.result)
    results = load_results(result_path)
    pending = []
    for item in items:
        raw = Path(item["raw_path"])
        known = results.get(str(item["segment_id"]), {})
        if (
            raw.exists()
            and raw.stat().st_size > 0
            and not args.force
            and not item.get("force_regenerate", False)
            and (not item.get("request_fingerprint") or
                 known.get("request_fingerprint") == item.get("request_fingerprint"))
        ):
            if known.get("status") != "success":
                results[str(item["segment_id"])] = {
                    **known, "segment_id": str(item["segment_id"]), "status": "success",
                    "generated_duration": wav_duration(raw), "inference_seconds": known.get("inference_seconds"),
                    "tts_engine": item.get("tts_engine", args.model),
                    "model_key": args.model,
                    "request_fingerprint": item.get("request_fingerprint"),
                    "reused_existing": True, "error": "",
                }
            continue
        pending.append(item)
        results[str(item["segment_id"])] = {
            **known, "segment_id": str(item["segment_id"]), "status": "pending",
            "tts_engine": item.get("tts_engine", args.model), "model_key": args.model,
            "request_fingerprint": item.get("request_fingerprint"),
            "error": "",
        }
    atomic_json(result_path, results)
    if not pending:
        print(f"{args.model}: no pending segments")
        return

    if os.getenv("AUTODUB_TTS_OFFLINE_MODE", "strict_offline") == "strict_offline":
        configure_strict_offline()

    backend = {
        "f5tts": F5TTSBackend,
        "indextts2": IndexTTSBackend,
        "cosyvoice3": CosyVoiceBackend,
        "confucius4": ConfuciusBackend,
        "omnivoice": OmniVoiceBackend,
    }[args.model]()
    for index, item in enumerate(pending, start=1):
        segment_id = str(item["segment_id"])
        raw = Path(item["raw_path"])
        raw.parent.mkdir(parents=True, exist_ok=True)
        temporary = raw.with_name(raw.stem + ".tmp.wav")
        print(f"[{args.model} {index}/{len(pending)}] segment_id={segment_id}")
        started = time.perf_counter()
        try:
            generated_duration = backend.generate(item, temporary)
            os.replace(temporary, raw)
            elapsed = time.perf_counter() - started
            results[segment_id] = {
                **results.get(segment_id, {}),
                "segment_id": segment_id,
                "status": "success",
                "tts_engine": item.get("tts_engine", args.model),
                "model_key": args.model,
                "request_fingerprint": item.get("request_fingerprint"),
                "generated_duration": float(generated_duration),
                "inference_seconds": elapsed,
                "rtf": elapsed / generated_duration if generated_duration > 0 else None,
                "error": "",
            }
        except Exception as exc:
            traceback.print_exc()
            if temporary.exists():
                temporary.unlink()
            results[segment_id] = {
                **results.get(segment_id, {}), "segment_id": segment_id,
                "status": "failed", "error": f"{type(exc).__name__}: {exc}",
            }
        atomic_json(result_path, results)


if __name__ == "__main__":
    main()
