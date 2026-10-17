"""Regression tests for evidence-driven dialogue recovery."""

import unittest

from asr_recovery import (
    acoustic_dialogue_score,
    build_recovery_groups,
    classify_recovery_evidence,
    choose_consensus_candidate,
    ocr_source_events,
)
from ocr_v211_integration import (
    _load_initial_recovery_state,
    _partition_deferred_regions,
    _recover_deferred_from_ocr,
)


class OCRTimelineTests(unittest.TestCase):
    def test_step1b_rerun_uses_immutable_initial_inventory(self):
        state = {
            "asr_deferred_regions_initial": [
                {"id": "original", "start": 40.0, "end": 43.0}
            ],
            "asr_deferred_regions": [],
            "asr_deferred_regions_remaining": [],
            "asr_uncovered_regions_initial": [
                {"start": 40.0, "end": 43.0}
            ],
            "asr_uncovered_regions": [],
            "asr_uncovered_regions_remaining": [],
            "asr_recovery_report_initial": [{"status": "step1a"}],
            "asr_recovery_report": [{"status": "mutated"}],
            "ocr_optimized_manifest": {"version": "already-ran"},
        }
        deferred, uncovered, report = _load_initial_recovery_state(state)
        self.assertEqual([item["id"] for item in deferred], ["original"])
        self.assertEqual(uncovered, [{"start": 40.0, "end": 43.0}])
        self.assertEqual(report, [{"status": "step1a"}])

        # Callers may mutate a run-local copy without changing the checkpoint.
        deferred.clear()
        self.assertEqual(state["asr_deferred_regions_initial"][0]["id"], "original")

    def test_unprocessed_legacy_checkpoint_can_seed_initial_inventory(self):
        deferred, uncovered, report = _load_initial_recovery_state(
            {
                "asr_deferred_regions": [{"id": "legacy"}],
                "asr_uncovered_regions": [{"start": 1.0, "end": 2.0}],
                "asr_recovery_report": [{"status": "legacy-step1a"}],
            }
        )
        self.assertEqual(deferred, [{"id": "legacy"}])
        self.assertEqual(uncovered, [{"start": 1.0, "end": 2.0}])
        self.assertEqual(report, [{"status": "legacy-step1a"}])

    def test_processed_legacy_checkpoint_fails_instead_of_losing_dialogue(self):
        with self.assertRaisesRegex(RuntimeError, "重新运行 --step 1a"):
            _load_initial_recovery_state(
                {
                    "asr_deferred_regions": [],
                    "asr_uncovered_regions": [],
                    "asr_recovery_report": [],
                    "ocr_optimized_manifest": {"version": "legacy-step1b"},
                    "segments_ocr_corrected": [{"id": 0, "text": "hello"}],
                }
            )

    def test_adjacent_regions_group_for_inference_but_keep_members(self):
        groups = build_recovery_groups(
            [
                {"id": "a", "start": 0.05, "end": 4.25},
                {"id": "b", "start": 4.25, "end": 5.16},
                {"id": "c", "start": 10.3, "end": 11.6},
            ],
            context=0.8,
            media_duration=12.0,
        )
        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0]["member_ids"], ["a", "b"])
        self.assertEqual((groups[0]["start"], groups[0]["end"]), (0.05, 5.16))
        self.assertEqual(groups[0]["probe_start"], 0.0)
        self.assertEqual(groups[1]["probe_end"], 12.0)

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

    def test_context_only_subtitle_is_not_assigned_to_core(self):
        events, decision = ocr_source_events(
            [
                {"timestamp": 8.8, "text": "Previous sentence."},
                {"timestamp": 9.1, "text": "Previous sentence."},
            ],
            "en",
            region_start=10.0,
            region_end=11.0,
        )
        self.assertEqual(events, [])
        self.assertFalse(decision["dialogue_evidence"])

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
    def test_acoustic_score_separates_weak_short_noise_from_auditable_activity(self):
        weak, _ = acoustic_dialogue_score(
            duration=0.4,
            vocal_dbfs=-48.0,
            source_dbfs=-20.0,
            acoustic_text_observed=False,
            primary_speech_confirmed=True,
        )
        probable, evidence = acoustic_dialogue_score(
            duration=1.2,
            vocal_dbfs=-24.0,
            source_dbfs=-20.0,
            acoustic_text_observed=False,
            primary_speech_confirmed=True,
        )
        self.assertLess(weak, 0.55)
        self.assertGreaterEqual(probable, 0.55)
        self.assertIn("vocal_not_buried_in_mix", evidence["reasons"])

    def test_energy_only_never_confirms_dialogue(self):
        classification = classify_recovery_evidence(
            acoustic_text_observed=False,
            acoustic_score=0.60,
            acoustic_candidate_threshold=0.40,
        )
        self.assertFalse(classification["dialogue_evidence"])
        self.assertTrue(classification["acoustic_candidate"])
        self.assertEqual(
            classification["evidence_status"],
            "acoustic_candidate_without_text_evidence",
        )

    def test_observed_asr_text_confirms_dialogue(self):
        classification = classify_recovery_evidence(
            acoustic_text_observed=True,
            acoustic_score=0.10,
            acoustic_candidate_threshold=0.40,
        )
        self.assertTrue(classification["dialogue_evidence"])
        self.assertFalse(classification["acoustic_candidate"])

    def test_energy_only_region_is_audited_without_blocking(self):
        regions = [
            {
                "id": "recovery-group-0000",
                "start": 0.05,
                "end": 5.22,
                "dialogue_evidence": False,
                "acoustic_candidate": True,
                "acoustic_dialogue_score": 0.60,
                "asr_recovery_status": "unresolved_uncertain",
            }
        ]
        blocking, acoustic, non_dialogue = _partition_deferred_regions(regions)
        self.assertEqual(blocking, [])
        self.assertEqual(non_dialogue, [])
        self.assertEqual(len(acoustic), 1)
        self.assertEqual(
            acoustic[0]["asr_recovery_status"],
            "acoustic_candidate_unconfirmed",
        )

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
