"""Regression tests for the TTS voice-quality fixes.

Covers the three TTS sub-tasks from the leader's board:
  * TTS 首尾静音 -> tts_router._trim_silence
  * 变速限制     -> tts_router._atempo_factor / _apply_atempo
  * 电音检测     -> tts_quality.electricity_quality

Runs offline (no GPU / no model downloads). Place in the repo root and run:
    python test_tts_voice_quality.py
"""
import importlib.util
import math
import os
import shutil
import struct
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import tts_router
from tts_quality import (
    duration_adaptation_safety,
    electricity_quality,
    voice_consistency_quality,
)
from tts_content_verifier import content_completeness_quality

HAS_LIBROSA = importlib.util.find_spec("librosa") is not None


class VoiceConsistencyTests(unittest.TestCase):
    def test_off_policy_skips_without_loading_audio_features(self):
        with patch.dict(os.environ, {
            "AUTODUB_TTS_VOICE_CONSISTENCY_POLICY": "off"
        }):
            result = voice_consistency_quality("missing.wav", "missing.wav")
        self.assertEqual(result["voice_consistency_quality"], "skipped")
        self.assertEqual(result["voice_consistency_policy"], "off")


class ContentCompletenessTests(unittest.TestCase):
    def test_explicit_content_asr_switch_overrides_server_model_export(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {
            "QWEN3_ASR_MODEL_DIR": tmp,
            "AUTODUB_TTS_CONTENT_ASR_ENABLED": "0",
        }):
            router = tts_router.TTSRouter(
                SimpleNamespace(temp_dir=tmp), scheme="scheme3"
            )
            self.assertFalse(router._content_asr_enabled())
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {
            "QWEN3_ASR_MODEL_DIR": tmp,
            "AUTODUB_TTS_CONTENT_ASR_ENABLED": "1",
        }):
            router = tts_router.TTSRouter(
                SimpleNamespace(temp_dir=tmp), scheme="scheme3"
            )
            self.assertTrue(router._content_asr_enabled())

    def test_short_chinese_line_rejects_one_swallowed_character(self):
        result = content_completeness_quality("杰克你好", "杰你好", "zh")
        self.assertEqual(result["content_completeness_quality"], "failed")

    def test_punctuation_difference_does_not_fail_complete_speech(self):
        result = content_completeness_quality(
            "现在闭上眼睛，走吧。", "现在闭上眼睛走吧", "zh"
        )
        self.assertEqual(result["content_completeness_quality"], "passed")

    def test_missing_english_word_is_rejected_for_short_line(self):
        result = content_completeness_quality(
            "Give me your hand", "Give me hand", "en"
        )
        self.assertEqual(result["content_completeness_quality"], "failed")

    def test_safe_chinese_particle_is_not_mistaken_for_missing_content(self):
        result = content_completeness_quality("手给我", "把手给我", "zh")
        self.assertEqual(result["content_completeness_quality"], "passed")
        self.assertEqual(
            result["content_completeness_match_rule"],
            "expected_contained_with_safe_particles",
        )
        self.assertEqual(result["content_completeness_warnings"], [
            "tts_asr_safe_extra=把"
        ])

    def test_added_negation_is_not_treated_as_a_safe_particle(self):
        result = content_completeness_quality("给我", "别给我", "zh")
        self.assertEqual(result["content_completeness_quality"], "failed")

    def test_missing_expected_particle_still_fails(self):
        result = content_completeness_quality("把手给我", "手给我", "zh")
        self.assertEqual(result["content_completeness_quality"], "failed")

    def test_content_policy_has_separate_fingerprint_from_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "candidate.wav"
            write_wav(audio, 1.0, sample=tone())
            item = {
                "text": "你好，杰克。",
                "language": "zh",
                "target_language": "zh",
                "emotion": "neutral",
                "speaker": "A",
                "target_duration": 1.0,
                "model": "f5tts",
                "final_path": str(audio),
            }
            metadata = {
                "file": str(audio),
                "audio_sha256": tts_router._file_sha256(audio),
            }
            with patch.dict(os.environ, {
                "AUTODUB_TTS_CONTENT_ASR_MIN_SIMILARITY": "0.72"
            }):
                generation_a = tts_router._request_fingerprint(item)
                audit_a = tts_router._content_audit_fingerprint(item, metadata)
            with patch.dict(os.environ, {
                "AUTODUB_TTS_CONTENT_ASR_MIN_SIMILARITY": "0.80"
            }):
                generation_b = tts_router._request_fingerprint(item)
                audit_b = tts_router._content_audit_fingerprint(item, metadata)
            self.assertEqual(generation_a, generation_b)
            self.assertNotEqual(audit_a, audit_b)

    def test_legacy_audio_migrates_only_when_generation_request_matches(self):
        item = {
            "text": "你好，杰克。",
            "language": "zh",
            "target_language": "zh",
            "emotion": "neutral",
            "speaker": "A",
            "ref_audio": "speaker.wav",
            "ref_text": "reference",
            "target_duration": 0.8,
            "start": 7.71,
            "model": "cosyvoice3",
            "reference_segment_id": 4,
            "reference_purity_score": 0.9,
            "translation_selection_source": "standard",
        }
        metadata = {
            "generation_request": {
                key: item.get(key)
                for key in tts_router.GENERATION_REQUEST_KEYS
            }
        }
        self.assertTrue(
            tts_router._legacy_generation_request_matches(item, metadata)
        )
        changed = dict(item, text="完全不同的文本")
        self.assertFalse(
            tts_router._legacy_generation_request_matches(changed, metadata)
        )

    def test_agent_fallback_uses_best_generated_candidate_not_source_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            router = tts_router.TTSRouter(
                SimpleNamespace(temp_dir=tmp), scheme="scheme3"
            )
            weak = Path(tmp) / "weak.wav"
            complete = Path(tmp) / "complete.wav"
            write_wav(weak, 1.0, sample=tone(freq=330.0))
            write_wav(complete, 1.0, sample=tone(freq=550.0))
            item = {
                "segment_id": "0",
                "text": "你好，杰克。",
                "planned_text": "你好，杰克。",
                "language": "zh",
                "target_language": "zh",
                "emotion": "neutral",
                "speaker": "A",
                "ref_audio": "speaker.wav",
                "ref_text": "reference",
                "target_duration": 1.0,
                "start": 7.71,
                "model": "f5tts",
                "measure_natural_duration": True,
                "translation_selection_source": "standard",
            }
            item["request_fingerprint"] = tts_router._request_fingerprint(item)
            router._remember_candidate(item, {
                "file": str(weak),
                "actual_duration": 1.0,
                "audible_speech_quality": "passed",
                "content_completeness_quality": "failed",
                "voice_consistency_quality": "passed",
                "error": "content mismatch",
            }, "content_audit")
            router._remember_candidate(item, {
                "file": str(complete),
                "actual_duration": 1.0,
                "audible_speech_quality": "passed",
                "content_completeness_quality": "passed",
                "voice_consistency_quality": "passed",
                "error": "strict duration policy",
            }, "content_audit")
            item["start"] = 9.25
            adopted = router._adopt_best_generated_candidate(item)
            self.assertIsNotNone(adopted)
            self.assertTrue(adopted["agent_degraded_candidate"])
            self.assertEqual(adopted["status"], "success")
            self.assertEqual(item["start"], 9.25)
            self.assertEqual(
                tts_router._file_sha256(Path(adopted["file"])),
                tts_router._file_sha256(complete),
            )


def write_wav(path, duration_s, rate=16000, sample=None):
    """Write a mono 16-bit PCM WAV. `sample(i)` returns float in [-1, 1]."""
    if sample is None:
        sample = lambda i: 0.0
    frames = int(duration_s * rate)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"".join(
            struct.pack("<h", int(max(-1.0, min(1.0, sample(i))) * 32767))
            for i in range(frames)
        ))


def tone(rate=16000, freq=440.0):
    return lambda i: 0.5 * math.sin(2 * math.pi * freq * i / rate)


def noise(seed=12345):
    state = seed

    def f(i):
        nonlocal state
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        return (state % 2000) / 1000.0 - 1.0
    return f


def copy_adapt(source, destination, target_duration):
    shutil.copy2(source, destination)
    return destination


class AtempoFactorTests(unittest.TestCase):
    def test_within_tolerance_is_unchanged(self):
        factor, capped, raw = tts_router._atempo_factor(1.00, 1.05)
        self.assertEqual((factor, capped, raw), (1.0, False, 1.0))

    def test_speedup_is_capped(self):
        with patch.dict(os.environ, {"AUTODUB_TTS_ATEMPO_MIN": "0.8",
                                     "AUTODUB_TTS_ATEMPO_MAX": "1.25"}):
            factor, capped, raw = tts_router._atempo_factor(3.0, 1.0)
        self.assertEqual(factor, 1.25)
        self.assertTrue(capped)
        self.assertEqual(raw, 3.0)

    def test_slowdown_is_clamped_but_not_capped(self):
        # An under-length clip is a real failure, not an accepted state.
        factor, capped, raw = tts_router._atempo_factor(0.1, 1.0)
        self.assertEqual(factor, 0.9)
        self.assertFalse(capped)
        self.assertEqual(raw, 0.1)

    def test_exact_filter_chain_supports_large_speedup(self):
        chain = tts_router._atempo_filter_chain(5.0)
        factors = [float(part.split("=")[1]) for part in chain.split(",")]
        self.assertTrue(all(0.5 <= factor <= 2.0 for factor in factors))
        self.assertAlmostEqual(math.prod(factors), 5.0, places=6)


class TrimSilenceTests(unittest.TestCase):
    def test_leading_and_trailing_silence_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src.wav"
            dst = Path(tmp) / "dst.wav"
            # 0.5s silence + 1.0s tone + 0.5s silence = 2.0s at 16 kHz.
            write_wav(src, 2.0, sample=lambda i: tone()(i) if 8000 <= i < 24000 else 0.0)
            tts_router._trim_silence(src, dst)
            trimmed = tts_router._wav_duration(dst)
            self.assertLess(trimmed, 1.5)
            self.assertGreater(trimmed, 0.8)

    def test_fully_silent_clip_is_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src.wav"
            dst = Path(tmp) / "dst.wav"
            write_wav(src, 1.0)
            tts_router._trim_silence(src, dst)
            self.assertAlmostEqual(tts_router._wav_duration(dst), 1.0, delta=0.05)


class PostprocessTests(unittest.TestCase):
    def test_postprocess_records_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "raw.wav"
            dst = Path(tmp) / "final.wav"
            write_wav(src, 2.0, sample=lambda i: tone()(i) if 8000 <= i < 24000 else 0.0)
            with patch("tts_router._apply_atempo", side_effect=copy_adapt):
                post = tts_router._postprocess(src, dst, 1.5)
            self.assertAlmostEqual(post["pre_trim_duration"], 2.0, delta=0.05)
            self.assertLess(post["post_trim_duration"], 1.5)
            self.assertGreater(post["post_trim_duration"], 0.8)
            self.assertAlmostEqual(post["actual_duration"], post["post_trim_duration"], delta=0.05)
            self.assertIn("atempo_factor", post)
            self.assertIn("atempo_capped", post)
            self.assertTrue(dst.is_file())

    def test_postprocess_force_fits_only_moderate_overlength_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "raw.wav"
            dst = Path(tmp) / "final.wav"
            write_wav(src, 1.4, sample=tone())

            def exact(source, destination, target):
                write_wav(destination, target, sample=tone())
                return {
                    "forced_duration_fit": True,
                    "forced_atempo_factor": 1.4,
                    "forced_atempo_chain": "atempo=1.4",
                }

            with patch(
                "tts_router._trim_silence",
                side_effect=lambda source, destination: shutil.copy2(
                    source, destination
                ),
            ), patch("tts_router._apply_atempo", side_effect=copy_adapt), patch(
                "tts_router._apply_atempo_exact", side_effect=exact
            ):
                post = tts_router._postprocess(src, dst, 1.0)
            self.assertTrue(post["forced_duration_fit"])
            self.assertAlmostEqual(post["actual_duration"], 1.0, delta=0.05)

    def test_postprocess_keeps_preview_and_skips_unsafe_force_fit(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "raw.wav"
            dst = Path(tmp) / "final.wav"
            write_wav(src, 3.0, sample=tone())
            with patch(
                "tts_router._trim_silence",
                side_effect=lambda source, destination: shutil.copy2(
                    source, destination
                ),
            ), patch("tts_router._apply_atempo", side_effect=copy_adapt), patch(
                "tts_router._apply_atempo_exact"
            ) as exact:
                post = tts_router._postprocess(src, dst, 1.0)
            exact.assert_not_called()
            self.assertTrue(post["force_fit_skipped"])
            self.assertTrue(post["unsafe_duration_compression"])
            self.assertAlmostEqual(post["actual_duration"], 3.0, delta=0.05)

    def test_confucius_example_exceeds_safe_compression_limit(self):
        result = duration_adaptation_safety(5.35, 3.44)
        self.assertAlmostEqual(result["duration_ratio"], 5.35 / 3.44)
        self.assertTrue(result["unsafe_duration_compression"])
        self.assertFalse(result["safe_to_force_fit"])


@unittest.skipUnless(HAS_LIBROSA, "librosa not installed")
class ElectricityTests(unittest.TestCase):
    def test_clean_tone_is_not_flagged(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tone.wav"
            write_wav(path, 1.0, rate=16000, sample=tone())
            result = electricity_quality(path)
            self.assertIn(result["electricity_quality"], ("passed", "skipped"))

    def test_broadband_noise_is_flagged(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "noise.wav"
            # 48 kHz white noise has energy well above 8 kHz and a flat spectrum.
            write_wav(path, 1.0, rate=48000, sample=noise())
            result = electricity_quality(path)
            self.assertEqual(result["electricity_quality"], "failed")

    def test_policy_off_skips_analysis(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tone.wav"
            write_wav(path, 1.0, rate=16000, sample=tone())
            with patch.dict(os.environ, {"AUTODUB_TTS_ELECTRICITY_POLICY": "off"}):
                result = electricity_quality(path)
            self.assertEqual(result["electricity_quality"], "skipped")
            self.assertEqual(result["electricity_policy"], "off")


if __name__ == "__main__":
    unittest.main(verbosity=2)
