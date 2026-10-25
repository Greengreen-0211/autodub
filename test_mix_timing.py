import unittest

from mix_timing import plan_mix_windows, validate_mix_audit


class MixTimingTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
