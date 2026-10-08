#!/usr/bin/env python3
"""Pure local tests for scheme-three routing, emotion policy, and JSON."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from scheme3_policy import (
    CANONICAL_EMOTIONS,
    SCHEME3_ENGINES,
    analyze_text_complexity,
    build_scheme3_segments,
    normalize_emotion,
    select_scheme3_tts_engine,
    validate_scheme3_segments,
    write_segments_json,
)
from tts_router import select_scheme3_tts_engine as router_select_scheme3
from tts_router import select_tts_model as select_scheme2
from tts_router import _request_fingerprint
from tts_router import _worker_environment, resolve_scheme3_engine
from emotion_recognizer import normalize_emotion as recognizer_normalize_emotion


SHORT_ZH = "今天天气不错。"
SHORT_EN = "This is a short sentence."
LONG_EN = "This sentence contains enough carefully selected words to explain a complicated engineering idea while preserving context for every listener in the audience."
LONG_ZH = "这是一个需要保留完整上下文和所有重要细节的中文长句子，用于检查长句优先路由规则。"
COMPLEX_ZH = "现在开始，保持安静，仔细听。"


def check_routes() -> None:
    cases = [
        ("zh", "neutral", SHORT_ZH, "f5tts", "zh neutral short"),
        ("en", "neutral", SHORT_EN, "f5tts", "en neutral short"),
        ("zh", "angry", SHORT_ZH, "indextts2", "zh angry short"),
        ("en", "sad", SHORT_EN, "indextts2", "en sad short"),
        ("zh", "happy", SHORT_ZH, "cosyvoice3", "zh happy short"),
        ("en", "surprised", SHORT_EN, "cosyvoice3", "en surprised short"),
        ("en", "neutral", LONG_EN, "confucius4", "en neutral long"),
        ("zh", "happy", LONG_ZH, "confucius4", "zh happy long"),
        ("en", "sad", LONG_EN, "confucius4", "en sad long"),
        ("zh", "angry", COMPLEX_ZH, "confucius4", "zh angry complex"),
        ("ja", "neutral", "こんにちは", "omnivoice", "ja neutral"),
        ("ja", "happy", "うれしい", "omnivoice", "ja happy"),
        ("fr", "sad", "Bonjour", "omnivoice", "fr sad"),
    ]
    for language, emotion, text, expected, label in cases:
        actual = select_scheme3_tts_engine(language, emotion, text)
        assert actual == expected, (label, actual, expected)
        assert router_select_scheme3(language, emotion, text) == expected
        print(f"{label}: {actual}")
    # Scheme-two remains available and unchanged.
    assert select_scheme2("zh", "neutral") == "cosyvoice3"
    assert select_scheme2("en", "neutral") == "confucius4"
    assert select_scheme2("en", "sad") == "cosyvoice3"
    assert select_scheme2("ja", "neutral") == "omnivoice"


def check_emotions() -> None:
    assert normalize_emotion("surprise") == "surprised"
    assert normalize_emotion("surprised") == "surprised"
    assert recognizer_normalize_emotion("surprised") == "surprised"
    assert normalize_emotion("fear") == "fearful"
    assert normalize_emotion("disgust") == "disgusted"
    assert normalize_emotion("other") == "neutral"
    assert normalize_emotion("unknown-label") == "neutral"
    assert normalize_emotion("happy", score=0.49, duration=2.0) == "neutral"
    assert normalize_emotion("happy", score=0.99, duration=0.79) == "neutral"
    assert normalize_emotion("happy", score=0.99, duration=0.8) == "happy"


def check_complexity() -> None:
    assert not analyze_text_complexity(SHORT_ZH, "zh")["is_long_or_complex"]
    assert not analyze_text_complexity(SHORT_EN, "en")["is_long_or_complex"]
    assert analyze_text_complexity(LONG_EN, "en")["is_long"]
    assert analyze_text_complexity(LONG_ZH, "zh")["is_long"]
    assert analyze_text_complexity(COMPLEX_ZH, "zh")["is_complex"]


def check_json() -> None:
    rows = build_scheme3_segments(
        [
            {
                "id": 7,
                "text": "Great!",
                "start": 1.2,
                "end": 3.8,
                "raw_emotion": "surprise",
                "emotion_score": 0.91,
                "speaker": "SPEAKER_00",
            }
        ],
        source_lang="English",
        target_lang="zh",
    )
    required = {"text", "start", "end", "source_lang", "target_lang", "emotion", "speaker", "tts_engine"}
    assert required <= rows[0].keys()
    assert rows[0]["emotion"] == "surprised"
    assert rows[0]["tts_engine"] == "cosyvoice3"
    assert rows[0]["source_lang"] == "en"
    assert rows[0]["target_lang"] == "zh"
    assert rows[0]["emotion"] in CANONICAL_EMOTIONS
    assert rows[0]["tts_engine"] in SCHEME3_ENGINES
    validate_scheme3_segments(rows)
    other = build_scheme3_segments(
        [{"text": "こんにちは", "start": 0.0, "end": 1.2, "emotion": "happy"}],
        source_lang="Japanese",
        target_lang="ja",
    )[0]
    assert other["source_lang"] == "other"
    assert other["target_lang"] == "other"
    assert other["target_lang_raw"] == "ja"
    assert other["tts_engine"] == "omnivoice"
    with tempfile.TemporaryDirectory() as directory:
        path = write_segments_json(Path(directory) / "input_segments.json", rows)
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded == rows


def check_fingerprint() -> None:
    job = {
        "text": "hello",
        "language": "en",
        "target_language": "en",
        "emotion": "neutral",
        "speaker": "SPEAKER_00",
        "ref_audio": "/tmp/ref.wav",
        "ref_text": "reference",
        "target_duration": 1.0,
        "model": "f5tts",
    }
    original = _request_fingerprint(job)
    assert _request_fingerprint(dict(job)) == original
    changed = dict(job)
    changed["text"] = "changed text"
    assert _request_fingerprint(changed) != original


def check_step3_engine_boundary() -> None:
    step3 = build_scheme3_segments(
        [{"id": 1, "text": SHORT_EN, "start": 0.0, "end": 1.2, "emotion": "neutral"}],
        source_lang="en",
        target_lang="en",
    )[0]
    model, expected, fallback = resolve_scheme3_engine(
        step3, step3["target_lang"], step3["emotion"], step3["text"]
    )
    assert (model, expected, fallback) == ("f5tts", "f5tts", False)

    # Step 4 must execute the value persisted by Step 3 even if the current
    # policy would now choose a different model.
    persisted = dict(step3, text=LONG_EN, tts_engine="f5tts")
    model, expected, fallback = resolve_scheme3_engine(
        persisted, persisted["target_lang"], persisted["emotion"], persisted["text"]
    )
    assert model == "f5tts" and expected == "confucius4" and not fallback

    missing = dict(persisted)
    missing.pop("tts_engine")
    assert resolve_scheme3_engine(
        missing, missing["target_lang"], missing["emotion"], missing["text"]
    ) == ("confucius4", "confucius4", True)

    invalid = dict(step3, tts_engine="not-a-model")
    try:
        resolve_scheme3_engine(
            invalid, invalid["target_lang"], invalid["emotion"], invalid["text"]
        )
    except ValueError as exc:
        assert "invalid tts_engine" in str(exc)
    else:
        raise AssertionError("invalid persisted tts_engine was not rejected")

    for invalid_value in (None, "", "F5TTS"):
        invalid = dict(step3, tts_engine=invalid_value)
        try:
            resolve_scheme3_engine(
                invalid, invalid["target_lang"], invalid["emotion"], invalid["text"]
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid persisted tts_engine was accepted: {invalid_value!r}")


def check_worker_pythonpath_isolation() -> None:
    import os

    previous = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = "/polluted/f5/src"
    try:
        assert _worker_environment("cosyvoice3").get("PYTHONPATH") is None
        assert _worker_environment("confucius4").get("PYTHONPATH") is None
        assert _worker_environment("omnivoice").get("PYTHONPATH") is None
        assert _worker_environment("indextts2").get("PYTHONPATH") is None
        assert _worker_environment("f5tts")["PYTHONPATH"].replace("\\", "/").endswith(
            "/data0/goldenseed/cjx/F5-TTS/src"
        )
    finally:
        if previous is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = previous


if __name__ == "__main__":
    check_routes()
    check_emotions()
    check_complexity()
    check_json()
    check_fingerprint()
    check_step3_engine_boundary()
    check_worker_pythonpath_isolation()
    print("scheme3 policy tests: PASS")
