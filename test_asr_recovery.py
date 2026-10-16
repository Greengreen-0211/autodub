"""Regression tests for evidence-driven dialogue recovery."""

import unittest

from asr_recovery import choose_consensus_candidate, ocr_source_events
from ocr_v211_integration import _recover_deferred_from_ocr


class OCRTimelineTests(unittest.TestCase):
    def test_dense_ocr_recovers_multiple_ordered_events(self):
        records = [
            {"timestamp": 40.10, "text": "<ref>Come on!</ref><quad>x</quad>"},
            {"timestamp": 40.45, "text": "<ref>Come on</ref><quad>x</quad>"},
            {"timestamp": 42.00, "text": "<ref>Do the magic!</ref><quad>x</quad>"},
            {"timestamp": 42.35, "text": "<ref>Do the magic.</ref><quad>x</quad>"},
        ]
        events, decision = ocr_source_events(
            records, "en", region_start=40.0, region_end=43.0
        )
        self.assertEqual(
            [event["text"].rstrip(".!") for event in events],
            ["Come on", "Do the magic"],
        )
        self.assertEqual(events[0]["start"], 40.0)
        self.assertEqual(events[-1]["end"], 43.0)
        self.assertLessEqual(events[0]["end"], events[1]["start"])
        self.assertTrue(decision["dialogue_evidence"])

    def test_one_ocr_observation_blocks_instead_of_becoming_truth(self):
        events, decision = ocr_source_events(
            [{"timestamp": 1.0, "text": "Come on!"}],
            "en",
            region_start=0.5,
            region_end=1.5,
        )
        self.assertEqual(events, [])
        self.assertTrue(decision["dialogue_evidence"])
        self.assertEqual(decision["reason"], "source_language_ocr_seen_but_not_stable")

    def test_short_content_word_difference_is_not_merged(self):
        records = [
            {"timestamp": 1.0, "text": "Come on!"},
            {"timestamp": 1.3, "text": "Come in!"},
        ]
        events, decision = ocr_source_events(
            records, "en", region_start=0.8, region_end=1.6
        )
        self.assertEqual(events, [])
        self.assertTrue(decision["dialogue_evidence"])

    def test_deferred_parent_can_become_multiple_coarse_segments(self):
        regions = [
            {
                "id": "deferred-0001",
                "start": 40.0,
                "end": 43.0,
                "candidate_kind": "confirmed_empty_primary_chunk",
                "dialogue_evidence": True,
            }
        ]
        records = {
            "deferred-0001": [
                {"timestamp": 40.1, "text": "Come on!"},
                {"timestamp": 40.5, "text": "Come on!"},
                {"timestamp": 42.0, "text": "Do the magic!"},
                {"timestamp": 42.4, "text": "Do the magic!"},
            ]
        }
        recovered, remaining, audit = _recover_deferred_from_ocr(
            regions, records, "en"
        )
        self.assertEqual([item["text"] for item in recovered], ["Come on!", "Do the magic!"])
        self.assertEqual(remaining, [])
        self.assertEqual(recovered[0]["asr_recovered_parent_start"], 40.0)
        self.assertEqual(recovered[-1]["asr_recovered_parent_end"], 43.0)
        self.assertEqual(audit[0]["status"], "accepted_coarse")

    def test_confirmed_dialogue_with_unstable_ocr_remains_blocking(self):
        regions = [
            {
                "id": "deferred-0002",
                "start": 1.0,
                "end": 2.0,
                "candidate_kind": "confirmed_empty_primary_chunk",
                "dialogue_evidence": True,
            }
        ]
        recovered, remaining, _ = _recover_deferred_from_ocr(
            regions,
            {"deferred-0002": [{"timestamp": 1.5, "text": "Come on!"}]},
            "en",
        )
        self.assertEqual(recovered, [])
        self.assertEqual(remaining[0]["asr_recovery_status"], "unresolved_blocking")
        self.assertTrue(remaining[0]["dialogue_evidence"])


class AcousticConsensusTests(unittest.TestCase):
    def test_short_asr_conflict_requires_near_exact_consensus(self):
        candidates = [
            {"text": "Do the magic", "asr_repair_source": "vocals"},
            {"text": "To the magic", "asr_repair_source": "source"},
        ]
        selected, decision = choose_consensus_candidate(
            candidates,
            threshold=0.78,
            start=1.0,
            end=2.0,
            precise_origin="precise",
            coarse_origin="coarse",
        )
        self.assertIsNone(selected)
        self.assertGreaterEqual(decision["effective_threshold"], 0.94)


if __name__ == "__main__":
    unittest.main()
