# -*- coding: utf-8 -*-
"""Single-GPU HunyuanOCR HTTP service for AutoDub v2.

The model loading and inference path intentionally mirrors the already verified
``test_hunyuan_ocr.py`` implementation. Run exactly one Uvicorn worker: multiple
workers would load multiple model copies onto the same GPU.
"""

import os
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

# The verified HunyuanOCR environment uses an isolated transformers build. It
# must win sys.path resolution before transformers/PIL are imported.
PRIVATE_LIBS_DIR = os.getenv("HUNYUAN_PRIVATE_LIBS_DIR", "/data0/goldenseed/lqz77/my_libs")
if os.path.isdir(PRIVATE_LIBS_DIR) and PRIVATE_LIBS_DIR not in sys.path:
    sys.path.insert(0, PRIVATE_LIBS_DIR)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import cv2
import torch
import transformers
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field
from transformers import AutoProcessor, HunYuanVLForConditionalGeneration


MODEL_DIR = os.getenv("HUNYUAN_OCR_MODEL_DIR", "/data0/goldenseed/models/HunyuanOCR")
MAX_NEW_TOKENS = int(os.getenv("HUNYUAN_OCR_MAX_NEW_TOKENS", "1024"))
HEARTBEAT_SECONDS = float(os.getenv("HUNYUAN_OCR_HEARTBEAT_SECONDS", "15"))
MAX_IMAGE_EDGE = int(os.getenv("HUNYUAN_OCR_MAX_IMAGE_EDGE", "0"))
INFER_DTYPE = torch.bfloat16
OCR_PROMPT_VERSION = "official-spotting-v1"
# Use the exact Spotting instruction recommended by the HunyuanOCR authors.
# This task emits line-level <ref>text</ref><quad>coordinates</quad> records.
OCR_INSTRUCTION = "检测并识别图片中的文字，将文本坐标格式化输出。"

processor = None
model = None
INFERENCE_LOCK = threading.Lock()
CURRENT_REQUEST: Dict[str, Any] = {}


class OCRRequest(BaseModel):
    video_path: str
    frame_index: int = 5
    timestamp_seconds: Optional[float] = None
    request_id: Optional[str] = None
    max_new_tokens: Optional[int] = Field(default=None, ge=1)


def extract_frame(video_path: str, frame_index: int, timestamp_seconds: Optional[float] = None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")

    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            raise ValueError(f"无法读取视频总帧数: {video_path}")

        if timestamp_seconds is not None:
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            if fps <= 0:
                raise ValueError(f"无法读取视频帧率: {video_path}")
            frame_index = int(max(0.0, float(timestamp_seconds)) * fps)

        frame_index = max(0, min(total - 1, int(frame_index)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            raise ValueError(f"无法读取视频帧: {video_path}")
    finally:
        cap.release()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame)
    if MAX_IMAGE_EDGE > 0 and max(image.size) > MAX_IMAGE_EDGE:
        scale = MAX_IMAGE_EDGE / max(image.size)
        resized = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        image = image.resize(resized, Image.Resampling.LANCZOS)
    return image, frame_index


def load_model() -> None:
    global processor, model
    print(f"正在加载 HunyuanOCR: {MODEL_DIR}", flush=True)
    print(f"transformers 来源: {transformers.__file__}", flush=True)
    print(f"transformers 版本: {transformers.__version__}", flush=True)

    if not os.path.isfile(os.path.join(MODEL_DIR, "config.json")):
        raise FileNotFoundError(f"HunyuanOCR 模型目录不完整: {MODEL_DIR}")
    if not torch.cuda.is_available():
        raise RuntimeError("当前 hunyuan_ocr 环境不可见 CUDA，拒绝回退到 CPU 超慢推理。")

    # Keep this path identical to the successful Stage 5 script.
    processor = AutoProcessor.from_pretrained(MODEL_DIR, use_fast=False)
    try:
        model = HunYuanVLForConditionalGeneration.from_pretrained(
            MODEL_DIR,
            torch_dtype=INFER_DTYPE,
            device_map={"": "cuda:0"},
            attn_implementation="sdpa",
        )
    except TypeError:
        model = HunYuanVLForConditionalGeneration.from_pretrained(
            MODEL_DIR,
            torch_dtype=INFER_DTYPE,
            attn_implementation="sdpa",
        ).to("cuda:0")

    model.eval()
    device = next(model.parameters()).device
    if device.type != "cuda":
        raise RuntimeError(f"HunyuanOCR 实际加载到了 {device}，请检查 CUDA 环境。")
    print(f"HunyuanOCR 加载完成: device={device}, dtype={model.dtype}", flush=True)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    load_model()
    yield


app = FastAPI(title="HunyuanOCR Service v2", lifespan=lifespan)


def _gpu_snapshot() -> Dict[str, Any]:
    if not torch.cuda.is_available():
        return {"available": False}
    return {
        "available": True,
        "name": torch.cuda.get_device_name(0),
        "allocated_mib": round(torch.cuda.memory_allocated(0) / 1024**2, 1),
        "reserved_mib": round(torch.cuda.memory_reserved(0) / 1024**2, 1),
    }


def _generation_heartbeat(stop_event: threading.Event, request_id: str, started_at: float) -> None:
    if HEARTBEAT_SECONDS <= 0:
        return
    while not stop_event.wait(HEARTBEAT_SECONDS):
        elapsed = time.monotonic() - started_at
        print(f"[OCR {request_id}] generation still running: {elapsed:.1f}s", flush=True)


@app.get("/health")
def health():
    device = str(next(model.parameters()).device) if model is not None else ""
    return {
        "status": "ok" if model is not None else "loading",
        "model_loaded": model is not None,
        "busy": INFERENCE_LOCK.locked(),
        "current_request": dict(CURRENT_REQUEST),
        "device": device,
        "gpu": _gpu_snapshot(),
        "model_dir": MODEL_DIR,
        "transformers": transformers.__version__,
        "private_transformers": os.path.abspath(transformers.__file__).startswith(
            os.path.abspath(PRIVATE_LIBS_DIR)
        ),
        "ocr_prompt_version": OCR_PROMPT_VERSION,
    }


@app.post("/extract_text")
def extract_text(req: OCRRequest):
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="HunyuanOCR 尚未加载完成")
    if not INFERENCE_LOCK.acquire(blocking=False):
        raise HTTPException(
            status_code=409,
            detail={"message": "HunyuanOCR 正在处理其他请求", "current_request": CURRENT_REQUEST},
        )

    request_id = req.request_id or uuid.uuid4().hex[:8]
    started_at = time.monotonic()
    timings: Dict[str, float] = {}
    CURRENT_REQUEST.clear()
    CURRENT_REQUEST.update(
        {
            "request_id": request_id,
            "video_path": req.video_path,
            "timestamp_seconds": req.timestamp_seconds,
            "ocr_prompt_version": OCR_PROMPT_VERSION,
            "started_unix": round(time.time(), 3),
            "stage": "extract_frame",
        }
    )

    try:
        print(
            f"[OCR {request_id}] request video={req.video_path}, frame_index={req.frame_index}, "
            f"timestamp={req.timestamp_seconds}",
            flush=True,
        )

        phase_start = time.monotonic()
        image, actual_frame_index = extract_frame(
            req.video_path,
            req.frame_index,
            req.timestamp_seconds,
        )
        timings["extract_frame"] = round(time.monotonic() - phase_start, 3)
        print(
            f"[OCR {request_id}] frame extracted: index={actual_frame_index}, size={image.size}",
            flush=True,
        )

        messages = [
            {"role": "system", "content": ""},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "placeholder"},
                    {"type": "text", "text": OCR_INSTRUCTION},
                ],
            },
        ]

        CURRENT_REQUEST["stage"] = "preprocess"
        phase_start = time.monotonic()
        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = processor(
            text=[prompt],
            images=[image],
            padding=True,
            return_tensors="pt",
        ).to(next(model.parameters()).device)
        timings["preprocess"] = round(time.monotonic() - phase_start, 3)
        input_tokens = int(inputs["input_ids"].shape[-1])
        print(f"[OCR {request_id}] inputs prepared: input_tokens={input_tokens}", flush=True)

        CURRENT_REQUEST["stage"] = "generate"
        generation_started = time.monotonic()
        heartbeat_stop = threading.Event()
        heartbeat = threading.Thread(
            target=_generation_heartbeat,
            args=(heartbeat_stop, request_id, generation_started),
            daemon=True,
        )
        heartbeat.start()
        token_limit = min(req.max_new_tokens or MAX_NEW_TOKENS, MAX_NEW_TOKENS)
        try:
            with torch.inference_mode():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=token_limit,
                    do_sample=False,
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        finally:
            heartbeat_stop.set()
            heartbeat.join(timeout=1.0)
        timings["generate"] = round(time.monotonic() - generation_started, 3)
        print(f"[OCR {request_id}] generation finished: {timings['generate']}s", flush=True)

        CURRENT_REQUEST["stage"] = "decode"
        phase_start = time.monotonic()
        input_ids = inputs["input_ids"]
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(input_ids, generated_ids)
        ]
        text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        timings["decode"] = round(time.monotonic() - phase_start, 3)
        timings["total"] = round(time.monotonic() - started_at, 3)
        print(
            f"[OCR {request_id}] decode finished: text_len={len(text)}, total={timings['total']}s",
            flush=True,
        )

        return {
            "status": "success",
            "request_id": request_id,
            "text": text,
            "frame_index": actual_frame_index,
            "timestamp_seconds": req.timestamp_seconds,
            "input_tokens": input_tokens,
            "max_new_tokens": token_limit,
            "ocr_prompt_version": OCR_PROMPT_VERSION,
            "timings": timings,
        }
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        print(f"[OCR {request_id}] CUDA OOM: {exc}", flush=True)
        raise HTTPException(status_code=507, detail=f"HunyuanOCR CUDA OOM: {exc}") from exc
    except HTTPException:
        raise
    except Exception as exc:
        print(f"[OCR {request_id}] error: {type(exc).__name__}: {exc}", flush=True)
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
    finally:
        CURRENT_REQUEST.clear()
        INFERENCE_LOCK.release()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, workers=1)
