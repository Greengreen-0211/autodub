#!/usr/bin/env python3
"""Local tests for candidate normalization, duration selection, and resume."""

from __future__ import annotations

import tempfile
import wave
from pathlib import Path

from light_tts_selector import LightTTSSelector, normalize_translation_candidates
from scheme3_policy import build_scheme3_segments


class FakeLightTTS:
    name = "fake-light-tts"
    extension = ".wav"
    signature = "fake-light-tts:v1"

    def __init__(self, durations):
        self.durations = durations
        self.calls = []

    def synthesize(self, text: str, language: str, output_path: Path) -> None:
        self.calls.append((text, language, str(output_path)))
        duration = float(self.durations.get(text, 3.0))
        sample_rate = 16000
        frame_count = int(round(duration * sample_rate))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(output_path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            handle.writeframes(b"\x00\x00" * frame_count)


def check_candidate_normalization() -> None:
    candidates = normalize_translation_candidates(
        [
            {"text": "short"},
            {"text": "short"},
            {"text": "medium wording"},
            {"text": "a somewhat longer wording"},
            {"text": "four"},
            {"text": "five"},
            {"text": "ignored sixth"},
        ],
        "fallback",
    )
    assert 3 <= len(candidates) <= 5
    assert [item["text"] for item in candidates[:3]] == [
        "short",
        "medium wording",
        "a somewhat longer wording",
    ]


def check_selection_and_resume() -> None:
    short = "A brief line."
    medium = "This is a medium length translation for speech."
    long = (
        "This candidate deliberately contains more than twenty English words so the "
        "saved selection reaches the long sentence branch of the routing policy."
    )
    backend = FakeLightTTS({short: 0.8, medium: 1.5, long: 2.05})
    segment = {
        "id": 7,
        "text": medium,
        "start": 1.0,
        "end": 3.0,
        "emotion": "neutral",
        "speaker": "SPEAKER_00",
        "translations_candidates": [
            {"text": short},
            {"text": medium},
            {"text": long},
        ],
    }
    with tempfile.TemporaryDirectory() as directory:
        selector = LightTTSSelector(directory, backend=backend)
        selected, rows = selector.select([segment], target_language="en")
        assert len(backend.calls) == 3
        assert selected[0]["text"] == long
        assert rows[0]["selected_index"] == 2
        assert abs(rows[0]["duration_error"] - 0.05) < 1e-6
        assert sum(item["selected"] for item in selected[0]["translations_candidates"]) == 1
        assert selector.results_path.is_file()

        routed = build_scheme3_segments(selected, source_lang="en", target_lang="en")
        assert routed[0]["text"] == long
        assert routed[0]["tts_engine"] == "confucius4"

        resumed, resumed_rows = selector.select([segment], target_language="en")
        assert len(backend.calls) == 3
        assert resumed[0]["text"] == long
        assert resumed_rows == rows

        changed = dict(segment)
        changed["translations_candidates"] = [
            {"text": short},
            {"text": "Changed medium candidate."},
            {"text": long},
        ]
        selector.select([changed], target_language="en")
        assert len(backend.calls) == 4


if __name__ == "__main__":
    check_candidate_normalization()
    check_selection_and_resume()
    print("light TTS selector tests: PASS")

