"""Tests for the Step 4 elastic window and its handoff to the mixer."""

import ast
from contextlib import ExitStack
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
from types import SimpleNamespace
from typing import Any, Dict, List
import unittest
from unittest.mock import Mock, patch
import wave

from elastic_timing import (
    ElasticTimingSettings, conflicting_elastic_clips, decide_elastic_timing,
    plan_elastic_windows,
)
from mix_timing import plan_complete_mix_schedule, plan_mix_windows
from scheme3_policy import build_scheme3_segments
from tts_quality import validate_mix_state
import tts_router


def source_state(next_start=4.0):
    return {
        "duration": 6.0,
        "src_lang": "en",
        "segments_step3": [
            {"id": 1, "start": 1.0, "end": 3.0, "speaker": "A", "text": "今天见到你真高兴。"},
            {"id": 2, "start": next_start, "end": 5.0, "speaker": "B", "text": "再见。"},
        ],
        "tts_timing_report": {
            "status": "ok",
            "primary_diagnostics": {"duration": 6.0, "speech_duration": 3.0},
            "fused_speech_regions": [[1.0, 3.0], [next_start, 5.0]],
        },
    }


def first_window(state):
    return {
        **plan_elastic_windows(state, ElasticTimingSettings(enabled=True))["1"],
        "placement_start": 1.0,
        "legacy_end": 3.0,
    }


class ElasticWindowTests(unittest.TestCase):
    def test_a_natural_audio_already_fits(self):
        timing = decide_elastic_timing(first_window(source_state()), 1.8)
        self.assertEqual(timing["reason"], "fits_original_window")
        self.assertEqual(timing["allowed_end"], 3.0)
        self.assertFalse(timing["applied"])
        self.assertFalse(timing["needs_speed_adjustment"])

    def test_b_uses_only_the_required_part_of_confirmed_silence(self):
        timing = decide_elastic_timing(first_window(source_state()), 2.3)
        self.assertTrue(timing["applied"])
        self.assertAlmostEqual(timing["available_extension_seconds"], 0.4)
        self.assertAlmostEqual(timing["extension_seconds"], 0.3)
        self.assertAlmostEqual(timing["allowed_end"], 3.3)
        self.assertFalse(timing["needs_speed_adjustment"])

    def test_c_insufficient_extension_returns_original_window(self):
        timing = decide_elastic_timing(first_window(source_state()), 3.0)
        self.assertEqual(timing["reason"], "insufficient_elastic_window")
        self.assertAlmostEqual(timing["available_extension_seconds"], 0.4)
        self.assertEqual(timing["allowed_end"], 3.0)
        self.assertTrue(timing["needs_speed_adjustment"])

    def test_d_next_start_limits_the_window_even_if_translation_is_skipped(self):
        state = source_state(next_start=3.35)
        state["segments_step3"][1]["translation_status"] = "skipped"
        window = first_window(state)
        self.assertAlmostEqual(window["maximum_end"], 3.2)
        self.assertFalse(decide_elastic_timing(window, 2.3)["applied"])
        self.assertTrue(decide_elastic_timing(window, 2.2)["applied"])
        close = decide_elastic_timing(first_window(source_state(next_start=3.1)), 2.1)
        self.assertEqual(close["reason"], "next_sentence_safety_gap")

    def test_e_existing_overlap_blocks_both_source_segments(self):
        state = source_state(next_start=2.75)
        windows = plan_elastic_windows(state, ElasticTimingSettings(enabled=True))
        self.assertEqual([row["reason"] for row in windows.values()], ["source_overlap"] * 2)
        self.assertFalse(decide_elastic_timing(first_window(state), 2.3)["applied"])

    def test_disabled_keeps_the_existing_route(self):
        self.assertEqual(plan_elastic_windows(source_state(), ElasticTimingSettings()), {})

    def test_gap_without_vad_is_not_evidence(self):
        state = source_state()
        state["tts_timing_report"]["fused_speech_regions"] = []
        timing = decide_elastic_timing(first_window(state), 2.3)
        self.assertEqual(timing["reason"], "no_confirmed_trailing_silence")
        self.assertFalse(timing["applied"])

    def test_silence_must_be_contiguous_and_clear_of_unresolved_dialogue(self):
        state = source_state()
        state["tts_timing_report"]["fused_speech_regions"][0][1] = 3.05
        self.assertFalse(decide_elastic_timing(first_window(state), 2.3)["applied"])
        state = source_state()
        state["asr_unresolved_dialogue_regions"] = [{"start": 3.18, "end": 3.35}]
        window = first_window(state)
        self.assertEqual(window["maximum_end"], 3.18)
        self.assertFalse(decide_elastic_timing(window, 2.3)["applied"])

    def test_last_line_cannot_borrow_beyond_video_or_vad_coverage(self):
        state = source_state()
        state["segments_step3"].pop()
        state["tts_timing_report"]["fused_speech_regions"].pop()
        state["duration"] = 3.25
        self.assertFalse(decide_elastic_timing(first_window(state), 2.3)["applied"])
        state["duration"] = 6.0
        state["tts_timing_report"]["primary_diagnostics"]["duration"] = 3.2
        self.assertFalse(decide_elastic_timing(first_window(state), 2.3)["applied"])

    def test_refined_onset_does_not_move_to_make_the_wav_fit(self):
        window = dict(first_window(source_state()), placement_start=1.1)
        self.assertTrue(decide_elastic_timing(window, 2.3)["applied"])
        self.assertFalse(decide_elastic_timing(window, 2.31)["applied"])
        window["legacy_end"] = 2.95
        timing = decide_elastic_timing(window, 2.3)
        self.assertEqual(timing["reason"], "narrower_legacy_window")

    def test_actual_final_clip_can_invalidate_a_previously_safe_extension(self):
        timing = decide_elastic_timing(first_window(source_state()), 2.3)
        clips = [
            {"segment_id": "1", "start": 1.0, "end": 3.3, "source_start": 1.0},
            {"segment_id": "2", "start": 4.0, "end": 5.0, "source_start": 4.0},
        ]
        results = [
            {"segment_id": "1", "actual_duration": 2.3, "elastic_timing": timing},
            {"segment_id": "2", "actual_duration": 1.9},
        ]
        self.assertEqual(conflicting_elastic_clips(
            clips, results, 5.2, maximum_early_shift_seconds=0.35,
        ), {"1"})
        self.assertFalse(conflicting_elastic_clips(
            clips, results, 6.0, maximum_early_shift_seconds=0.35,
        ))


def write_wav(path, duration):
    rate = 16000
    cycle = struct.pack("<80h", *(round(6000 * math.sin(2 * math.pi * i / 80)) for i in range(80)))
    frames = round(duration * rate)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        output.writeframes((cycle * (frames // 80 + 1))[:frames * 2])


def load_pipeline(state, translator):
    # The production entry point imports model runtimes at module load.
    # Execute the actual orchestration with lightweight state/model fixtures.
    class StateManager:
        def __init__(self, _config):
            pass

        def load(self):
            return deepcopy(state)

        def save(self, changed):
            state.update(changed)

    script = Path(__file__).with_name("auto_dubbing_ver_2.0.py")
    tree = ast.parse(script.read_text(encoding="utf-8"))
    names = {"_env_bool", "Mixer", "run_step_4", "run_step_5", "restore_skipped_vocals"}
    definitions = [node for node in tree.body if getattr(node, "name", None) in names]
    from pydub import AudioSegment, silence

    namespace = {
        "DubbingConfig": object, "StateManager": StateManager,
        "AdvancedTranslator": lambda _config: translator, "os": os,
        "Any": Any, "Dict": Dict, "List": List, "Path": Path,
        "AudioSegment": AudioSegment, "silence": silence,
        "json": json, "subprocess": subprocess,
    }
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(script), "exec"), namespace)
    return namespace["run_step_4"], namespace["run_step_5"]


@unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is required for waveform comparisons")
class ElasticTimingIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.stack.enter_context(patch.dict(os.environ, {
            "AUTODUB_TTS_CONTENT_ASR_ENABLED": "0",
            "AUTODUB_TTS_DURATION_RETRANSLATION": "1",
            "AUTODUB_TTS_DURATION_REWRITES": "1",
            "AUTODUB_TTS_TRIM_SILENCE": "1",
        }))

    def test_a_and_c_match_legacy_audio_when_the_window_is_unused(self):
        for duration in (1.8, 3.0):
            with self.subTest(duration=duration):
                raw, old, new = [self.root / name for name in ("raw.wav", "old.wav", "new.wav")]
                write_wav(raw, duration)
                baseline = tts_router._postprocess(raw, old, 2.0)
                elastic = tts_router._postprocess(raw, new, 2.0, elastic_window=first_window(source_state()))
                self.assertFalse(elastic.pop("elastic_timing")["applied"])
                self.assertEqual(baseline, elastic)
                self.assertEqual(old.read_bytes(), new.read_bytes())

    def prepare_pipeline(self, *, video_duration=6.0, final_wav_duration=1.0):
        state = source_state()
        state["duration"] = video_duration
        state["segments_step3"] = build_scheme3_segments(
            state["segments_step3"], source_lang="en", target_lang="zh",
        )
        state["vocals_path"] = str(self.root / "vocals.wav")
        state["accomp_path"] = str(self.root / "accomp.wav")
        config = SimpleNamespace(
            temp_dir=str(self.root), output_dir=str(self.root), project_name="elastic_test",
            target_lang="zh", enable_elastic_timing=True, openai_api_key="test-fixture",
            disable_tts_fallback=True, tts_backend="scheme3", subtitle_mode="off",
        )
        translator = Mock()
        translator.rewrite_for_duration.return_value = None
        reference = self.root / "reference.wav"
        write_wav(reference, 1.0)

        def worker(_model, plan_path, result_path, **_kwargs):
            items = json.loads(plan_path.read_text(encoding="utf-8"))["items"]
            results = json.loads(result_path.read_text(encoding="utf-8"))
            for item in items:
                sid = item["segment_id"]
                write_wav(Path(item["raw_path"]), 2.3 if sid == "1" else final_wav_duration)
                results[sid].update(status="success")
            result_path.write_text(json.dumps(results), encoding="utf-8")
            return 0

        self.stack.enter_context(patch("tts_router.preflight_models"))
        synthesis = self.stack.enter_context(patch("tts_router._run_worker", side_effect=worker))
        self.stack.enter_context(patch.object(tts_router.TTSRouter, "_speaker_references", return_value={
            speaker: {"audio": str(reference), "text": "source"} for speaker in ("A", "B")
        }))
        self.stack.enter_context(patch("tts_router.audible_speech_quality", return_value={"audible_speech_quality": "passed"}))
        self.stack.enter_context(patch("tts_router.electricity_quality", return_value={"electricity_quality": "passed"}))
        self.stack.enter_context(patch("tts_router.voice_consistency_quality", return_value={}))
        atempo = self.stack.enter_context(patch("tts_router._apply_atempo", wraps=tts_router._apply_atempo))
        return state, config, translator, synthesis, atempo

    def test_step4_preserves_natural_wave_and_rejects_stale_mode_on_resume(self):
        state, config, translator, synthesis, atempo = self.prepare_pipeline()
        step4, step5 = load_pipeline(state, translator)
        step4(config)

        clip, next_clip = state["audio_clips"]
        result = state["tts_router_results"][0]
        self.assertEqual(state["segments_step3"][0]["end"], 3.0)
        self.assertEqual(clip["start"], 1.0)
        self.assertAlmostEqual(clip["end"], 3.3)
        self.assertAlmostEqual(tts_router._wav_duration(Path(clip["file"])), 2.3)
        self.assertEqual(result["atempo_factor"], 1.0)
        self.assertEqual(result["natural_duration_fit"], "passed")
        self.assertTrue(result["elastic_timing"]["applied"])
        self.assertTrue(all("segment_1_" not in str(call.args[0]) for call in atempo.call_args_list))
        translator.rewrite_for_duration.assert_not_called()
        self.assertEqual(validate_mix_state(state, config), [])
        timeline = plan_complete_mix_schedule(plan_mix_windows([
            {**item, "audio_duration_seconds": tts_router._wav_duration(Path(item["file"]))}
            for item in (clip, next_clip)
        ], 6.0), 6.0)
        self.assertAlmostEqual(timeline[0]["complete_end"], 3.3)
        self.assertGreaterEqual(timeline[1]["mix_start"] - timeline[0]["complete_end"], 0.15)
        self.assertFalse(timeline[0]["truncated_to_window"])

        video = self.root / "fixture.mp4"
        subprocess.run([
            "ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
            "color=c=black:s=96x64:r=10:d=6", "-an", "-c:v", "mpeg4", str(video),
        ], check=True, capture_output=True)
        step5(config, str(video))
        mix_audit = json.loads((self.root / "elastic_test_mix_timing_audit.json").read_text(encoding="utf-8"))
        self.assertTrue((self.root / "final_video.mp4").is_file())
        self.assertAlmostEqual(mix_audit[0]["audible_duration"], 2.3)
        self.assertEqual(mix_audit[0]["mixed_overlap_seconds"], 0.0)
        self.assertFalse(mix_audit[0]["truncated_to_window"])

        synthesis.reset_mock()
        step4(config)
        synthesis.assert_not_called()

        config.enable_elastic_timing = False
        with self.assertRaisesRegex(RuntimeError, "rerun Step4"):
            step5(config, str(video))
        step4(config)
        synthesis.assert_called()
        self.assertEqual(state["audio_clips"][0]["end"], 3.0)
        self.assertNotIn("elastic_timing", state["tts_router_results"][0])
        self.assertNotIn("tts_elastic_timing", state["segments_step3"][0])
        self.assertEqual(validate_mix_state(state, config), [])

    def test_step4_revokes_a_reservation_if_final_wav_must_start_early(self):
        state, config, translator, _, _ = self.prepare_pipeline(
            video_duration=5.2, final_wav_duration=2.7,
        )
        step4, _ = load_pipeline(state, translator)
        step4(config)
        first_timing = state["tts_router_results"][0]["elastic_timing"]
        self.assertEqual(first_timing["reason"], "rejected_by_mix_schedule")
        self.assertFalse(first_timing["applied"])
        self.assertEqual(state["audio_clips"][0]["end"], 3.0)


if __name__ == "__main__":
    unittest.main()
