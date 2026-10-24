import unittest

from mix_timing import plan_mix_windows


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
            {"start": 1.0, "end": 4.0, "allow_overlap": True},
            {"start": 3.2, "end": 5.0},
        ], 10.0)
        self.assertEqual(planned[0]["hard_end"], 4.0)

    def test_video_end_is_always_a_hard_limit(self):
        planned = plan_mix_windows([
            {"start": 8.5, "end": 12.0},
        ], 10.0)
        self.assertEqual(planned[0]["hard_end"], 10.0)
        self.assertEqual(planned[0]["available_ms"], 1500)


if __name__ == "__main__":
    unittest.main()
