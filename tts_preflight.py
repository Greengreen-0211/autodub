"""Read-only model/cache preflight; downloads only with explicit --download."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

MODEL_NAMES = ("f5tts", "indextts2", "cosyvoice3", "confucius4", "omnivoice")
CACHE_KEYS = {"f5tts":"F5", "indextts2":"INDEXTTS2", "cosyvoice3":"COSYVOICE3",
              "confucius4":"CONFUCIUS", "omnivoice":"OMNIVOICE"}


def cache_environment(model: str) -> dict[str, str]:
    default = "/data0/goldenseed/qgd/hf_cache" if model=="confucius4" else str(Path.home()/".cache/huggingface")
    root = os.getenv(f"AUTODUB_{CACHE_KEYS[model]}_HF_HOME", os.getenv("HF_HOME", default))
    hub = os.getenv(f"AUTODUB_{CACHE_KEYS[model]}_HF_HUB_CACHE", str(Path(root)/"hub"))
    return {"HF_HOME":root, "HF_HUB_CACHE":hub, "HUGGINGFACE_HUB_CACHE":hub,
            "TRANSFORMERS_CACHE":hub}


def offline_mode() -> str:
    mode = os.getenv("AUTODUB_TTS_OFFLINE_MODE", "strict_offline")
    if mode not in {"strict_offline", "allow_download"}:
        raise ValueError("AUTODUB_TTS_OFFLINE_MODE must be strict_offline or allow_download")
    return mode


def cached_file(cache: Path, repo: str, filename: str):
    base = cache / ("models--"+repo.replace("/", "--"))
    ref = base/"refs/main"
    snapshots = [base/"snapshots"/ref.read_text().strip()] if ref.is_file() else []
    # Without a main ref, HF may accept a commit explicitly only; do not pretend
    # that an arbitrary snapshot is usable by an unpinned from_pretrained call.
    for snapshot in snapshots:
        path = snapshot/filename
        if path.is_file() and path.stat().st_size:
            return path
    return None


def requirements(model: str):
    files, hubs = [], []
    if model == "f5tts":
        repo=Path(os.getenv("F5TTS_REPO", "/data0/goldenseed/cjx/F5-TTS"))
        files += [repo/"src/f5_tts/api.py", Path(os.getenv("F5TTS_MODEL_PATH", str(repo/"ckpts/F5TTS_v1_Base/model_1250000.safetensors")))]
        local=os.getenv("F5TTS_VOCODER_LOCAL_PATH", "")
        if not local:
            local=next((str(p) for p in (repo/"ckpts/vocos-mel-24khz",repo/"checkpoints/vocos-mel-24khz") if p.is_dir()), "")
        if local:
            files += [Path(local)/"config.yaml",Path(local)/"pytorch_model.bin"]
        else:
            hubs += [("charactr/vocos-mel-24khz", "config.yaml"), ("charactr/vocos-mel-24khz", "pytorch_model.bin")]
    elif model == "indextts2":
        repo=Path(os.getenv("INDEX_TTS_ROOT", "/data0/goldenseed/models/index-tts"))
        ckpt=Path(os.getenv("INDEX_TTS_MODEL_DIR", str(repo/"checkpoints")))
        files += [repo/"indextts/infer_v2.py",ckpt/"config.yaml",ckpt/"gpt.pth",ckpt/"s2mel.pth",ckpt/"bpe.model",
                  ckpt/"wav2vec2bert_stats.pt", ckpt/"qwen0.6bemo4-merge/config.json", ckpt/"qwen0.6bemo4-merge/model.safetensors"]
        hubs += [("facebook/w2v-bert-2.0", "preprocessor_config.json"),
                 ("facebook/w2v-bert-2.0", "config.json"),
                 ("facebook/w2v-bert-2.0", "__weights__"),
                 ("amphion/MaskGCT", "semantic_codec/model.safetensors"),
                 ("funasr/campplus", "campplus_cn_common.bin")]
        config = ckpt/"config.yaml"
        if config.is_file():
            import yaml
            settings = yaml.safe_load(config.read_text())
            files += [ckpt/settings[key] for key in ("emo_matrix", "spk_matrix")]
            vocoder = settings["vocoder"]["name"]
            local_vocoder = Path(vocoder) if Path(vocoder).is_absolute() else repo/vocoder
            if local_vocoder.is_dir():
                files += [local_vocoder/"config.json", local_vocoder/"bigvgan_generator.pt"]
            else:
                hubs += [(vocoder,"config.json"), (vocoder,"bigvgan_generator.pt")]
    elif model == "cosyvoice3":
        repo=Path(os.getenv("COSYVOICE_REPO", "/data0/goldenseed/qgd/CosyVoice"))
        ckpt=Path(os.getenv("COSYVOICE_MODEL_DIR", str(repo/"pretrained_models/Fun-CosyVoice3-0.5B")))
        files += [repo/"cosyvoice/cli/cosyvoice.py"] + [ckpt/name for name in
                  ("cosyvoice3.yaml","llm.pt","flow.pt","hift.pt","campplus.onnx","speech_tokenizer_v3.onnx",
                   "CosyVoice-BlankEN/config.json")]
    elif model == "confucius4":
        repo=Path(os.getenv("CONFUCIUS4_REPO", "/data0/goldenseed/qgd/Confucius4-TTS"))
        config=repo/"config/inference_config.yaml"
        files += [config,repo/"confuciustts/cli/inference.py"]
        if config.is_file():
            import yaml
            paths=yaml.safe_load(config.read_text())["paths"]
            files += [repo/paths["w2v_stat"],repo/paths["tokenizer_path"]/"tokenizer_config.json"]
            hubs += [("netease-youdao/Confucius4-TTS", paths["t2s_checkpoint"]),
                     ("netease-youdao/Confucius4-TTS", paths["s2a_checkpoint"]),
                     ("funasr/campplus", paths["style_encoder"]["checkpoint"])]
            for name in ("w2v_bert_path", "vocoder_path"):
                location=paths[name]
                if (repo/location).is_dir():
                    files += [repo/location/"config.json"]
                    if name=="vocoder_path":
                        files.append(repo/location/"bigvgan_generator.pt")
                    elif not any((repo/location).glob("*.safetensors")) and not (repo/location/"pytorch_model.bin").is_file():
                        files.append(repo/location/"model.safetensors")
                else:
                    hubs += [(location,"config.json"),
                             (location,"bigvgan_generator.pt" if name=="vocoder_path" else "__weights__")]
                    if name=="w2v_bert_path":
                        hubs.append((location,"preprocessor_config.json"))
    elif model == "omnivoice":
        ckpt=Path(os.getenv("OMNIVOICE_MODEL_DIR", "/data0/goldenseed/qgd/models/OmniVoice"))
        files += [ckpt/"config.json"]
        if not any(ckpt.glob("*.safetensors")) and not (ckpt/"pytorch_model.bin").is_file():
            files.append(ckpt/"model.safetensors")
        for index in ckpt.glob("*.index.json"):
            files += [ckpt/name for name in set(json.loads(index.read_text())["weight_map"].values())]
    else:
        raise ValueError(model)
    return files,hubs


def check_model(model: str, *, download=False) -> dict:
    cache=Path(cache_environment(model)["HF_HUB_CACHE"])
    missing=[]
    try:
        files,hubs=requirements(model)
        missing += [str(p) for p in files if not p.is_file() or not p.stat().st_size]
        for repo,name in hubs:
            if download and (name=="__weights__" or not cached_file(cache,repo,name)):
                from huggingface_hub import hf_hub_download, snapshot_download
                if name=="__weights__":
                    snapshot_download(repo,cache_dir=str(cache))
                else:
                    hf_hub_download(repo,filename=name,cache_dir=str(cache))
            if name=="__weights__":
                valid=cached_file(cache,repo,"model.safetensors") or cached_file(cache,repo,"pytorch_model.bin")
                index=cached_file(cache,repo,"model.safetensors.index.json") or cached_file(cache,repo,"pytorch_model.bin.index.json")
                if not valid and index:
                    shards=set(json.loads(index.read_text())["weight_map"].values())
                    valid=bool(shards) and all(cached_file(cache,repo,s) for s in shards)
                if not valid:
                    missing.append(f"{repo}: model weights/shards (cache={cache})")
            elif not cached_file(cache,repo,name):
                missing.append(f"{repo}/{name} (cache={cache})")
    except Exception as exc:
        missing.append(f"preflight error: {type(exc).__name__}: {exc}")
    return {"model":model,"cache":str(cache),"status":"failed" if missing else "passed", "missing":missing,
            "repair":f"python tts_preflight.py --models {model} --download (local checkpoints must be supplied separately)"}


def preflight_models(models) -> list[dict]:
    report=[check_model(model, download=offline_mode()=="allow_download") for model in sorted(set(models))]
    failures=[row for row in report if row["status"]=="failed"]
    for row in report:
        print("TTS preflight:",json.dumps(row,ensure_ascii=False))
    if failures:
        raise RuntimeError("TTS model preflight failed BEFORE generation: " +
                           "; ".join(f"{r['model']}: {r['missing']} ; {r['repair']}" for r in failures))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+",choices=MODEL_NAMES,default=list(MODEL_NAMES))
    parser.add_argument("--download",action="store_true",help="Explicitly download missing HF dependencies")
    args=parser.parse_args()
    report=[check_model(m,download=args.download) for m in args.models]
    print(json.dumps(report,ensure_ascii=False,indent=2))
    raise SystemExit(1 if any(r["status"]=="failed" for r in report) else 0)

if __name__=="__main__":
    main()
