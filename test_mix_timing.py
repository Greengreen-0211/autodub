import unittest

from mix_timing import (
    plan_complete_mix_schedule,
    plan_mix_windows,
    plan_speech_gain,
    validate_mix_audit,
)


class MixTimingTests(unittest.TestCase):
    def test_quiet_dialogue_receives_bounded_headroom_safe_gain(self):
        plan = plan_speech_gain(
            -24.0, -25.0, -8.0,
            target_margin_db=6.0,
            minimum_margin_db=3.0,
            maximum_gain_db=6.0,
            peak_ceiling_dbfs=-1.0,
        )
        self.assertEqual(plan["speech_gain_db"], 5.0)
        self.assertEqual(plan["speech_margin_db"], 6.0)
        self.assertEqual(plan["mix_audibility_quality"], "passed")

    def test_gain_never_crosses_peak_ceiling_and_reports_degradation(self):
        plan = plan_speech_gain(
            -24.0, -22.0, -2.0,
            target_margin_db=6.0,
            minimum_margin_db=3.0,
            maximum_gain_db=6.0,
            peak_ceiling_dbfs=-1.0,
        )
        self.assertEqual(plan["speech_gain_db"], 1.0)
        self.assertEqual(plan["mix_audibility_quality"], "degraded")

    def test_non_overlapping_clips_are_bounded_by_next_onset(self):
        planned = plan_mix_windows([
            {"segment_id": 1, "start": 1.0, "end": 4.0},
            {"segment_id": 2, "start": 3.2, "end": 5.0},
        ], 10.0)
        self.assertEqual(planned[0]["hard_end"], 3.2)
        self.assertEqual(planned[0]["available_ms"], 2200)
        self.assertEqual(planned[1]["hard_end"], 5.0)

    def test_explicit_source_overlap_is_preserved(self):
        planned = plan_mix_windows([
            {"start": 1.0, "end": 4.0, "allowed_overlap_seconds": 0.35},
            {"start": 3.2, "end": 5.0},
        ], 10.0)
        self.assertAlmostEqual(planned[0]["hard_end"], 3.55)
        self.assertTrue(planned[0]["allow_overlap"])

    def test_boolean_overlap_without_measured_budget_is_not_unbounded(self):
        planned = plan_mix_windows([
            {"start": 1.0, "end": 4.0, "allow_overlap": True},
            {"start": 3.2, "end": 5.0},
        ], 10.0)
        self.assertEqual(planned[0]["hard_end"], 3.2)

    def test_video_end_is_always_a_hard_limit(self):
        planned = plan_mix_windows([
            {"start": 8.5, "end": 12.0},
        ], 10.0)
        self.assertEqual(planned[0]["hard_end"], 10.0)
        self.assertEqual(planned[0]["available_ms"], 1500)

    def test_complete_clip_uses_silent_gap_instead_of_cutting_tail(self):
        planned = plan_mix_windows([
            {
                "segment_id": 1,
                "start": 1.0,
                "end": 2.0,
                "audio_duration_seconds": 2.5,
            },
            {
                "segment_id": 2,
                "start": 4.0,
                "end": 5.0,
                "audio_duration_seconds": 1.0,
            },
        ], 10.0)
        scheduled = plan_complete_mix_schedule(planned, 10.0)
        self.assertEqual(scheduled[0]["mix_start"], 1.0)
        self.assertEqual(scheduled[0]["complete_end"], 3.5)
        self.assertEqual(scheduled[0]["unavoidable_overlap_seconds"], 0.0)
        self.assertFalse(scheduled[0]["truncated_to_window"])

    def test_complete_clip_reports_collision_instead_of_cutting_tail(self):
        planned = plan_mix_windows([
            {
                "segment_id": 1,
                "start": 1.0,
                "source_start": 0.5,
                "end": 2.0,
                "audio_duration_seconds": 3.0,
            },
            {
                "segment_id": 2,
                "start": 3.0,
                "end": 4.0,
                "audio_duration_seconds": 1.0,
            },
        ], 10.0)
        scheduled = plan_complete_mix_schedule(
            planned, 10.0, maximum_early_shift_seconds=0.3
        )
        self.assertAlmostEqual(scheduled[0]["mix_start"], 0.7)
        self.assertAlmostEqual(
            scheduled[0]["unavoidable_overlap_seconds"], 0.7
        )
        self.assertFalse(scheduled[0]["truncated_to_window"])

    def test_complete_clip_never_moves_before_source_sentence(self):
        planned = plan_mix_windows([
            {
                "segment_id": 1,
                "start": 1.2,
                "source_start": 1.0,
                "end": 2.0,
                "audio_duration_seconds": 3.0,
            },
            {
                "segment_id": 2,
                "start": 3.0,
                "end": 4.0,
                "audio_duration_seconds": 1.0,
            },
        ], 10.0)
        scheduled = plan_complete_mix_schedule(
            planned, 10.0, maximum_early_shift_seconds=0.35
        )
        self.assertAlmostEqual(scheduled[0]["mix_start"], 1.0)

    def test_final_complete_clip_moves_earlier_instead_of_being_cut(self):
        planned = plan_mix_windows([{
            "segment_id": 1,
            "start": 8.5,
            "end": 9.0,
            "audio_duration_seconds": 2.0,
        }], 10.0)
        scheduled = plan_complete_mix_schedule(planned, 10.0)
        self.assertEqual(scheduled[0]["mix_start"], 8.0)
        self.assertEqual(scheduled[0]["complete_end"], 10.0)
        self.assertFalse(scheduled[0]["truncated_to_window"])

    def test_mix_audit_requires_every_verified_clip(self):
        with self.assertRaisesRegex(RuntimeError, "missing=.*2"):
            validate_mix_audit(["1", "2"], [{
                "segment_id": "1", "audible_duration": 1.0,
                "original_wav_duration": 1.0, "clip_rms": 100,
                "mix_verified": True,
            }])

    def test_mix_audit_blocks_tail_truncation_that_can_swallow_words(self):
        with self.assertRaisesRegex(RuntimeError, "would trim"):
            validate_mix_audit(["1"], [{
                "segment_id": "1", "audible_duration": 0.7,
                "original_wav_duration": 1.0, "clip_rms": 100,
                "mix_verified": True,
            }])

    def test_mix_audit_accepts_complete_non_silent_clip(self):
        validate_mix_audit(["1"], [{
            "segment_id": "1", "audible_duration": 1.0,
            "original_wav_duration": 1.0, "clip_rms": 100,
            "mix_verified": True,
        }])

    def test_agent_mode_records_complete_clip_collision(self):
        issues = validate_mix_audit(["1"], [{
            "segment_id": "1", "audible_duration": 1.0,
            "original_wav_duration": 1.0, "clip_rms": 100,
            "mix_verified": True,
            "unavoidable_overlap_seconds": 0.25,
        }], allow_degraded=True)
        self.assertTrue(any("collision boundary" in issue for issue in issues))

    def test_agent_mode_records_tail_trim_without_blocking_delivery(self):
        issues = validate_mix_audit(["1"], [{
            "segment_id": "1", "audible_duration": 0.7,
            "original_wav_duration": 1.0, "clip_rms": 100,
            "mix_verified": True,
        }], allow_degraded=True)
        self.assertTrue(any("would trim" in issue for issue in issues))


if __name__ == "__main__":
    unittest.main()
