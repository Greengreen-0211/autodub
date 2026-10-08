#!/usr/bin/env python3
"""Verify that a skipped existing WAV preserves scheme-three timing metadata."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import wave
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        raw = work / "segment_0_f5tts.wav"
        with wave.open(str(raw), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(24000)
            handle.writeframes(b"\x00\x00" * 2400)
        plan = work / "plan.json"
        result = work / "results.json"
        plan.write_text(
            json.dumps(
                {
                    "items": [
                        {
                            "segment_id": "0",
                            "model": "f5tts",
                            "tts_engine": "f5tts",
                            "raw_path": str(raw),
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        expected = {
            "segment_id": "0",
            "status": "success",
            "tts_engine": "f5tts",
            "model_key": "f5tts",
            "selected_model": "F5-TTS",
            "inference_seconds": 12.34,
            "rtf": 0.56,
            "actual_duration": 1.23,
        }
        result.write_text(json.dumps({"0": expected}), encoding="utf-8")
        subprocess.run(
            [
                sys.executable,
                str(root / "tts_worker.py"),
                "--model",
                "f5tts",
                "--plan",
                str(plan),
                "--result",
                str(result),
            ],
            check=True,
        )
        actual = json.loads(result.read_text(encoding="utf-8"))["0"]
        for key in (
            "inference_seconds",
            "rtf",
            "actual_duration",
            "selected_model",
            "tts_engine",
        ):
            assert actual[key] == expected[key], (key, actual[key], expected[key])
    print("scheme3 resume metadata test: PASS")


if __name__ == "__main__":
    main()
