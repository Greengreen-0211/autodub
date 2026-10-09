# -*- coding: utf-8 -*-
"""Preflight check for the local HunyuanOCR service used by AutoDub v2."""

import argparse
import json
import sys
import time
from urllib.parse import urlsplit, urlunsplit

import requests


def health_url(endpoint: str) -> str:
    parsed = urlsplit(endpoint)
    return urlunsplit((parsed.scheme, parsed.netloc, "/health", "", ""))


def main() -> int:
    parser = argparse.ArgumentParser(description="检查 HunyuanOCR 本地服务并可选执行单帧 OCR")
    parser.add_argument("--url", default="http://127.0.0.1:8000/extract_text")
    parser.add_argument("--video", default="", help="可选：用于真实 OCR 测试的视频路径")
    parser.add_argument("--timestamp", type=float, default=2.0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()

    session = requests.Session()
    session.trust_env = False

    print(f"[1/2] 健康检查: {health_url(args.url)}")
    response = session.get(health_url(args.url), timeout=5)
    response.raise_for_status()
    health = response.json()
    print(json.dumps(health, ensure_ascii=False, indent=2))

    if not health.get("model_loaded"):
        print("❌ OCR 模型尚未加载完成。")
        return 2
    if health.get("busy"):
        print("❌ OCR 服务正忙。不要重复提交请求，请先观察 current_request。")
        return 3
    if not health.get("private_transformers", True):
        print("❌ 服务没有使用已验证的 my_libs/transformers。")
        return 4
    if not args.video:
        print("✅ 健康检查通过。传入 --video 可继续执行真实 OCR。")
        return 0

    payload = {
        "video_path": args.video,
        "timestamp_seconds": args.timestamp,
        "request_id": "manual-preflight",
        "max_new_tokens": args.max_new_tokens,
    }
    print(f"[2/2] OCR 请求: {json.dumps(payload, ensure_ascii=False)}")
    started = time.monotonic()
    try:
        response = session.post(
            args.url,
            json=payload,
            timeout=(3, args.timeout),
        )
    except requests.exceptions.ReadTimeout:
        print(
            f"❌ 客户端等待超过 {args.timeout:.0f}s。不要立即重试；先访问 /health "
            "查看服务是否仍处于 generate 阶段。"
        )
        return 5

    elapsed = time.monotonic() - started
    print(f"HTTP {response.status_code}, client_elapsed={elapsed:.2f}s")
    try:
        print(json.dumps(response.json(), ensure_ascii=False, indent=2))
    except ValueError:
        print(response.text)
    response.raise_for_status()
    print("✅ HunyuanOCR 单帧服务测试通过。")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except requests.RequestException as exc:
        print(f"❌ OCR 服务检查失败: {exc}")
        raise SystemExit(1) from exc
