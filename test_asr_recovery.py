"""Regression tests for evidence-driven dialogue recovery."""

import unittest
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from asr_recovery import (
    acoustic_dialogue_score,
    build_recovery_groups,
    classify_recovery_evidence,
    choose_consensus_candidate,
    ocr_source_events,
    reconcile_coarse_recovery_boundaries,
)
from light_tts_selector import (
    LightTTSSelector,
    classify_coarse_recovery_duration,
)
from ocr_translation_reference import prepare_ocr_translation_references
from run_step2_wespeaker import invalidate_downstream_state
from speaker_alignment import merge_adjacent_speaker_turns
from speech_activity import (
    attach_coarse_tts_placement,
    attach_precise_tts_placement,
    fuse_speech_regions,
)
from tts_router import _tts_timing
from ocr_v211_integration import (
    _load_initial_recovery_state,
    _partition_deferred_regions,
    _recover_deferred_from_ocr,
    apply_stable_bilingual_entity_corrections,
    filter_unsupported_recovered_micro_segments,
    ocr_recovery_timing_plausibility,
    persistent_ocr_overlay_texts,
    preserve_speaker_timing_skeleton,
)


class OCRTimelineTests(unittest.TestCase):
    def test_persistent_text_across_distant_regions_is_overlay(self):
        records = [
            {"segment_id": "a", "timestamp": 1.0, "text": "没有毛绒绒"},
            {"segment_id": "b", "timestamp": 8.0, "text": "没有毛绒绒"},
            {"segment_id": "c", "timestamp": 16.0, "text": "没有毛绒绒"},
        ]
        self.assertEqual(
            persistent_ocr_overlay_texts(records), ["没有毛绒绒"]
        )

    def test_repeated_local_subtitle_is_not_overlay(self):
        records = [
            {"segment_id": "a", "timestamp": 1.0, "text": "来吧"},
            {"segment_id": "a", "timestamp": 1.2, "text": "来吧"},
            {"segment_id": "a", "timestamp": 1.4, "text": "来吧"},
        ]
        self.assertEqual(persistent_ocr_overlay_texts(records), [])

    def test_ocr_text_rewrite_keeps_separate_speaker_timing_skeleton(self):
        source = {
            "text": "wrong word. Next line.",
            "qwen3_time_stamps": [
                {"text": "wrong", "start": 0.1, "end": 0.5},
                {"text": "word", "start": 0.6, "end": 1.0},
                {"text": "Next", "start": 2.0, "end": 2.3},
                {"text": "line", "start": 2.4, "end": 2.8},
            ],
        }
        result = {"text": "right word. Next line.", "qwen3_time_stamps": []}
        preserve_speaker_timing_skeleton(source, result)
        self.assertTrue(result["speaker_timing_skeleton_valid"])
        self.assertEqual(result["speaker_timing_text"], source["text"])
        self.assertEqual(
            result["speaker_timing_time_stamps"], source["qwen3_time_stamps"]
        )
        self.assertEqual(result["qwen3_time_stamps"], [])
        self.assertFalse(result["corrected_token_alignment_valid"])

    def test_impossible_dense_ocr_recovery_timing_is_rejected(self):
        audit = ocr_recovery_timing_plausibility(
            "没有毛绒绒", 20.41, 20.70
        )
        self.assertFalse(audit["plausible"])
        self.assertGreater(audit["lexical_rate"], audit["max_lexical_rate"])

    def test_short_ocr_interjection_remains_possible(self):
        audit = ocr_recovery_timing_plausibility("来吧", 1.0, 1.30)
        self.assertTrue(audit["plausible"])

    def test_single_unpunctuated_ocr_line_cannot_create_dialogue_from_energy(self):
        regions = [{
            "id": "acoustic-only",
            "start": 20.27,
            "end": 21.08,
            "acoustic_candidate": True,
            "dialogue_evidence": False,
        }]
        records = {"acoustic-only": [
            {"timestamp": 20.30, "text": "没有毛绒绒"},
            {"timestamp": 20.60, "text": "没有毛绒绒"},
        ]}
        recovered, remaining, audit = _recover_deferred_from_ocr(
            regions, records, "zh"
        )
        self.assertEqual(recovered, [])
        self.assertEqual(remaining[0]["asr_recovery_status"], "unresolved_uncertain")
        self.assertEqual(audit[0]["status"], "rejected_unverified_ocr_text")

    def test_unverified_single_han_retry_is_quarantined_not_spoken(self):
        segments = [{
            "id": 6,
            "start": 20.27,
            "end": 20.97,
            "text": "是。",
            "asr_origin": "coverage_retry_cross_source_consensus",
            "asr_recovery_status": "accepted_precise",
        }]
        kept, uncertain, audit = filter_unsupported_recovered_micro_segments(
            segments, []
        )
        self.assertEqual(kept, [])
        self.assertEqual(len(uncertain), 1)
        self.assertEqual(uncertain[0]["asr_recovery_status"], "unresolved_uncertain")
        self.assertFalse(audit[0]["accepted"])

    def test_primary_single_han_reply_is_never_filtered(self):
        segments = [{
            "id": 1,
            "start": 1.0,
            "end": 1.5,
            "text": "好。",
            "asr_origin": "primary",
            "asr_recovery_status": "primary",
        }]
        kept, uncertain, audit = filter_unsupported_recovered_micro_segments(
            segments, []
        )
        self.assertEqual(kept, segments)
        self.assertEqual(uncertain, [])
        self.assertEqual(audit, [])

    def test_stable_matching_ocr_can_confirm_single_han_retry(self):
        segments = [{
            "id": 1,
            "start": 1.0,
            "end": 1.8,
            "text": "好。",
            "asr_origin": "coverage_retry_cross_source_consensus",
        }]
        records = [
            {"timestamp": 1.2, "text": "好。"},
            {"timestamp": 1.5, "text": "好。"},
        ]
        kept, uncertain, audit = filter_unsupported_recovered_micro_segments(
            segments, records
        )
        self.assertEqual(len(kept), 1)
        self.assertEqual(uncertain, [])
        self.assertTrue(audit[0]["accepted"])

    def test_ocr_generated_punctuation_does_not_admit_single_character_text(self):
        regions = [{
            "id": "punctuated-artifact",
            "start": 20.27,
            "end": 20.97,
            "acoustic_candidate": True,
            "acoustic_dialogue_score": 0.70,
            "dialogue_evidence": False,
        }]
        records = {"punctuated-artifact": [
            {"timestamp": 20.30, "text": "是。"},
            {"timestamp": 20.60, "text": "是。"},
        ]}
        recovered, remaining, audit = _recover_deferred_from_ocr(
            regions, records, "zh"
        )
        self.assertEqual(recovered, [])
        self.assertEqual(remaining[0]["asr_recovery_status"], "unresolved_uncertain")
        # Single-character OCR is discarded before event admission.
        self.assertEqual(audit[0]["status"], "deferred")

    def test_coarse_tts_placement_uses_acoustic_onset_without_moving_bounds(self):
        segment = {
            "id": 7,
            "start": 40.0,
            "end": 50.0,
            "text": "Do the magic!",
            "asr_recovery_status": "accepted_coarse",
            "timestamp_precision": "coarse",
            "ocr_subtitle_first_seen": 43.0,
        }
        refined, audit = attach_coarse_tts_placement(
            [segment], [(40.1, 40.8), (43.4, 46.2)]
        )
        self.assertEqual(refined[0]["start"], 40.0)
        self.assertEqual(refined[0]["end"], 50.0)
        self.assertEqual(refined[0]["tts_placement_start"], 43.34)
        self.assertEqual(refined[0]["tts_placement_end"], 46.28)
        self.assertTrue(audit[0]["accepted"])
        placement, duration = _tts_timing(refined[0])
        self.assertAlmostEqual(placement, 43.34)
        self.assertAlmostEqual(duration, 2.94)
        ideal, available = LightTTSSelector._speech_window(refined[0])
        self.assertAlmostEqual(ideal, 2.94)
        self.assertAlmostEqual(available, 2.94)

    def test_coarse_window_does_not_span_a_long_internal_silence(self):
        segment = {
            "id": 10,
            "start": 10.0,
            "end": 20.0,
            "text": "Recovered dialogue",
            "asr_recovery_status": "accepted_coarse",
            "timestamp_precision": "coarse",
        }
        refined, audit = attach_coarse_tts_placement(
            [segment], [(10.5, 12.0), (15.0, 17.0)]
        )
        self.assertEqual(refined[0]["tts_placement_start"], 10.44)
        self.assertEqual(refined[0]["tts_placement_end"], 12.08)
        self.assertEqual(
            refined[0]["coarse_timing_excluded_islands"], [[15.0, 17.0]]
        )
        self.assertTrue(refined[0]["coarse_timing_requires_local_recovery"])
        self.assertEqual(audit[0]["excluded_islands"], [[15.0, 17.0]])

    def test_primary_coarse_timing_keeps_all_owned_islands_but_not_silence(self):
        segment = {
            "id": 11, "start": 10.0, "end": 20.0,
            "text": "One complete sentence.",
            "asr_recovery_status": "primary", "timestamp_precision": "coarse",
        }
        refined, _audit = attach_coarse_tts_placement(
            [segment], [(10.5, 12.0), (15.0, 17.0)]
        )
        self.assertEqual(refined[0]["tts_placement_start"], 10.44)
        self.assertEqual(refined[0]["tts_hard_end"], 17.08)
        self.assertLess(refined[0]["tts_target_duration"], 4.0)
        self.assertEqual(len(refined[0]["tts_speech_islands"]), 2)

    def test_precise_tts_uses_mix_window_not_transport_bounds(self):
        placement, duration = _tts_timing({
                "start": 10.0,
                "end": 13.0,
                "speech_start": 10.25,
                "speech_end": 12.4,
                "mix_start": 10.25,
                "mix_end": 12.4,
            })
        self.assertEqual(placement, 10.25)
        self.assertAlmostEqual(duration, 2.15)

    def test_precise_tts_expands_stale_mix_window_to_complete_sentence_words(self):
        placement, duration = _tts_timing({
            "start": 10.0,
            "end": 15.0,
            "mix_start": 13.0,
            "mix_end": 14.0,
            "qwen3_time_stamps": [
                {"text": "Complete", "start": 10.4, "end": 11.2},
                {"text": "sentence", "start": 13.0, "end": 14.0},
            ],
        })
        self.assertEqual(placement, 10.4)
        self.assertAlmostEqual(duration, 3.6)

    def test_precise_segment_is_unchanged_by_coarse_timing_refiner(self):
        segment = {
            "id": 1,
            "start": 13.09,
            "end": 17.29,
            "text": "Elsa, wake up!",
            "asr_recovery_status": "primary",
            "timestamp_precision": "word",
            "qwen3_time_stamps": [
                {"text": "Elsa", "start": 13.09, "end": 13.40}
            ],
        }
        refined, audit = attach_coarse_tts_placement([segment], [(13.1, 17.2)])
        self.assertEqual(refined[0], segment)
        self.assertEqual(audit, [])

    def test_precise_timing_counts_speech_islands_not_long_internal_pause(self):
        segment = {
            "id": 2, "start": 10.0, "end": 15.0, "timestamp_precision": "word",
            "qwen3_time_stamps": [
                {"text": "Hello", "start": 10.2, "end": 10.8},
                {"text": "again", "start": 13.0, "end": 13.7},
            ],
        }
        refined, audit = attach_precise_tts_placement(
            [segment], [(10.18, 10.82), (12.98, 13.72)]
        )
        self.assertLess(refined[0]["tts_target_duration"], 2.0)
        self.assertGreater(refined[0]["tts_hard_end"] - refined[0]["tts_placement_start"], 3.0)
        self.assertEqual(len(refined[0]["tts_internal_pauses"]), 1)
        self.assertEqual(audit[0]["precision"], "word_vad_consensus")

    def test_precise_vad_cannot_delay_past_first_word_onset(self):
        segment = {
            "id": 3, "start": 7.71, "end": 8.53,
            "text": "Hello, Jack.", "timestamp_precision": "word",
            "qwen3_time_stamps": [
                {"text": "Hello", "start": 7.76, "end": 8.05},
                {"text": "Jack", "start": 8.08, "end": 8.48},
            ],
        }
        refined, _ = attach_precise_tts_placement(
            [segment], [(7.84, 8.50)]
        )
        self.assertLessEqual(refined[0]["tts_placement_start"], 7.76)
        self.assertGreaterEqual(refined[0]["tts_target_duration"], 0.70)

    def test_precise_short_word_keeps_trailing_silence_as_physical_window(self):
        segment = {
            "id": 4, "start": 20.0, "end": 21.2,
            "text": "Go.", "timestamp_precision": "word",
            "qwen3_time_stamps": [
                {"text": "Go", "start": 20.10, "end": 20.25},
            ],
        }
        refined, audit = attach_precise_tts_placement(
            [segment], [(20.08, 20.28)]
        )
        row = refined[0]
        self.assertEqual(row["tts_aligned_speech_end"], 20.33)
        self.assertEqual(row["tts_hard_end"], 21.2)
        self.assertGreaterEqual(row["tts_target_duration"], 0.35)
        self.assertEqual(audit[0]["hard_end"], 21.2)

    def test_short_sentence_borrows_bounded_leading_slack_before_next_turn(self):
        segment = {
            "id": 6, "start": 41.97, "end": 43.49,
            "text": "Step up.", "timestamp_precision": "word",
            "qwen3_time_stamps": [
                {"text": "Step", "start": 42.93, "end": 43.33},
                {"text": "up", "start": 43.33, "end": 43.49},
            ],
        }
        refined, audit = attach_precise_tts_placement(
            [segment], [(42.83, 43.49)]
        )
        row = refined[0]
        self.assertAlmostEqual(row["tts_original_placement_start"], 42.87)
        # The pronunciation slack remains diagnostic, while the public
        # sentence boundary is now the authoritative playback anchor.
        self.assertAlmostEqual(row["tts_placement_start"], 41.97)
        self.assertTrue(row["tts_sentence_start_anchored"])
        self.assertAlmostEqual(row["tts_leading_slack_used"], 0.18)
        self.assertAlmostEqual(row["tts_target_duration"], 0.80)
        self.assertEqual(row["tts_hard_end"], 43.49)
        self.assertAlmostEqual(audit[0]["leading_slack_used"], 0.18)

    def test_leading_slack_is_capped_and_never_crosses_sentence_start(self):
        segment = {
            "id": 9, "start": 10.0, "end": 10.5,
            "text": "A complete longer sentence.",
            "timestamp_precision": "word",
            "qwen3_time_stamps": [
                {"text": "sentence", "start": 10.42, "end": 10.5},
            ],
        }
        refined, _ = attach_precise_tts_placement(
            [segment], [(10.40, 10.5)]
        )
        row = refined[0]
        self.assertGreaterEqual(row["tts_placement_start"], row["start"])
        self.assertLessEqual(row["tts_leading_slack_used"], 0.35)

    def test_ambiguous_boundary_does_not_borrow_leading_slack(self):
        segment = {
            "id": 10, "start": 30.0, "end": 31.0,
            "text": "Step up.", "timestamp_precision": "word",
            "speaker_boundary_ambiguous": True,
            "qwen3_time_stamps": [
                {"text": "Step", "start": 30.70, "end": 30.85},
                {"text": "up", "start": 30.85, "end": 31.0},
            ],
        }
        refined, _ = attach_precise_tts_placement(
            [segment], [(30.68, 31.0)]
        )
        self.assertEqual(refined[0]["tts_leading_slack_used"], 0.0)

    def test_long_pause_places_continuous_translation_at_dominant_island(self):
        segment = {
            "id": 7, "start": 43.49, "end": 49.61,
            "text": "Now hold on to the railing.",
            "timestamp_precision": "word",
            "speaker_boundary_ambiguous": True,
            "qwen3_time_stamps": [
                {"text": "Now", "start": 43.49, "end": 43.73},
                {"text": "hold", "start": 48.13, "end": 48.45},
                {"text": "on", "start": 48.45, "end": 48.69},
                {"text": "the", "start": 48.69, "end": 48.77},
                {"text": "railing", "start": 48.77, "end": 49.17},
            ],
        }
        refined, audit = attach_precise_tts_placement(
            [segment], [(43.47, 43.75), (48.11, 49.19)]
        )
        row = refined[0]
        self.assertEqual(
            row["tts_timing_reason"],
            "dominant_island_after_long_pause",
        )
        self.assertAlmostEqual(row["tts_original_placement_start"], 43.49)
        self.assertAlmostEqual(row["tts_placement_start"], 48.07)
        self.assertAlmostEqual(row["tts_aligned_speech_end"], 49.25)
        self.assertEqual(row["tts_hard_end"], 49.61)
        self.assertLess(row["tts_target_duration"], 2.0)
        self.assertEqual(row["tts_internal_pauses"], [[43.73, 48.13]])
        self.assertEqual(
            row["tts_dominant_word_island"],
            {
                "index": 1,
                "start": 48.13,
                "end": 49.17,
                "duration": 1.04,
                "activity_share": 0.8125,
            },
        )
        self.assertEqual(
            audit[0]["dominant_word_island"],
            row["tts_dominant_word_island"],
        )
        self.assertFalse(row["tts_onset_shift_limited"])
        self.assertFalse(row["tts_sentence_start_anchored"])

    def test_long_interval_with_words_at_tail_uses_audible_onset(self):
        segment = {
            "id": 12, "start": 57.74, "end": 60.75,
            "text": "Hold on.", "timestamp_precision": "word",
            "qwen3_time_stamps": [
                {"text": "Hold", "start": 59.95, "end": 60.30},
                {"text": "on", "start": 60.30, "end": 60.69},
            ],
        }
        refined, audit = attach_precise_tts_placement(
            [segment], [(59.93, 60.71)]
        )
        row = refined[0]
        self.assertAlmostEqual(row["tts_unbounded_placement_start"], 59.90)
        self.assertAlmostEqual(row["tts_placement_start"], 59.90)
        self.assertEqual(
            row["tts_timing_reason"], "trailing_word_island_consensus"
        )
        self.assertFalse(row["tts_sentence_start_anchored"])
        self.assertFalse(row["tts_onset_shift_limited"])
        self.assertTrue(row["tts_trailing_word_island_mode"])
        self.assertFalse(audit[0]["onset_shift_limited"])

    def test_long_interval_with_trailing_room_keeps_sentence_anchor(self):
        segment = {
            "id": 5, "start": 37.32, "end": 41.20,
            "text": "Go on.", "timestamp_precision": "word",
            "qwen3_time_stamps": [
                {"text": "Go", "start": 38.22, "end": 38.42},
                {"text": "on", "start": 38.42, "end": 38.68},
            ],
        }
        refined, _ = attach_precise_tts_placement(
            [segment], [(38.20, 38.70)]
        )
        row = refined[0]
        self.assertAlmostEqual(row["tts_placement_start"], 37.32)
        self.assertTrue(row["tts_sentence_start_anchored"])
        self.assertFalse(row["tts_trailing_word_island_mode"])

    def test_legacy_bounded_shift_requires_explicit_anchor_opt_out(self):
        segment = {
            "id": 12, "start": 57.74, "end": 60.75,
            "text": "Hold on.", "timestamp_precision": "word",
            "qwen3_time_stamps": [
                {"text": "Hold", "start": 59.95, "end": 60.30},
                {"text": "on", "start": 60.30, "end": 60.69},
            ],
        }
        with patch.dict(
            os.environ,
            {"AUTODUB_TTS_PRESERVE_SENTENCE_START": "0"},
        ):
            refined, _ = attach_precise_tts_placement(
                [segment], [(59.93, 60.71)]
            )
        self.assertAlmostEqual(refined[0]["tts_placement_start"], 58.04)
        self.assertFalse(refined[0]["tts_sentence_start_anchored"])

    def test_ambiguous_implausible_word_span_uses_local_ratio_guard(self):
        segment = {
            "id": 7, "start": 43.49, "end": 49.61,
            "text": "Hold on to the railing.",
            "timestamp_precision": "word",
            "speaker_boundary_ambiguous": True,
            "qwen3_time_stamps": [
                {"text": "Hold", "start": 43.51, "end": 44.2},
                {"text": "on", "start": 44.3, "end": 45.0},
                {"text": "to", "start": 45.1, "end": 45.8},
                {"text": "the", "start": 46.0, "end": 47.0},
                {"text": "railing", "start": 47.1, "end": 49.25},
            ],
        }
        refined, audit = attach_precise_tts_placement(
            [segment], [(43.5, 49.3)]
        )
        row = refined[0]
        self.assertEqual(
            row["tts_timing_reason"], "ambiguous_word_span_ratio_guard"
        )
        # The ambiguous word span still controls the natural speech target,
        # while the sentence's trailing silence remains a separate physical
        # fitting window.
        self.assertLess(
            row["tts_aligned_speech_end"] - row["tts_placement_start"],
            3.6,
        )
        self.assertEqual(row["tts_hard_end"], row["end"])
        self.assertLess(row["tts_target_duration"], 3.6)
        self.assertLess(row["tts_timing_confidence"], 0.7)
        self.assertIsNotNone(audit[0]["ratio_guard"])

    def test_source_activity_can_only_extend_existing_vocal_edges(self):
        fused = fuse_speech_regions(
            [(5.0, 6.0)], [(1.0, 2.0), (4.9, 6.15)]
        )
        self.assertEqual(len(fused), 1)
        self.assertAlmostEqual(fused[0][0], 4.9)
        self.assertAlmostEqual(fused[0][1], 6.15)

    def test_coarse_tts_placement_is_idempotent(self):
        segment = {
            "id": 8,
            "start": 50.0,
            "end": 60.0,
            "text": "This is amazing!",
            "asr_recovery_status": "accepted_coarse",
            "timestamp_precision": "coarse",
        }
        once, _ = attach_coarse_tts_placement([segment], [(53.0, 55.0)])
        twice, _ = attach_coarse_tts_placement(once, [(53.0, 55.0)])
        self.assertEqual(once, twice)

    def test_coarse_without_activity_keeps_legacy_timing(self):
        segment = {
            "id": 9,
            "start": 60.0,
            "end": 64.0,
            "text": "Hello!",
            "asr_recovery_status": "accepted_coarse",
            "timestamp_precision": "coarse",
        }
        refined, audit = attach_coarse_tts_placement([segment], [])
        self.assertEqual(refined[0], segment)
        self.assertFalse(audit[0]["accepted"])
        self.assertEqual(_tts_timing(refined[0]), (60.0, 4.0))

    def test_step2_does_not_merge_away_a_coarse_tts_window(self):
        segments = [
            {
                "id": 1,
                "start": 40.0,
                "end": 42.0,
                "text": "Come on!",
                "speaker": "SPEAKER_00",
                "timestamp_precision": "word",
                "asr_recovery_status": "primary",
            },
            {
                "id": 2,
                "start": 42.1,
                "end": 48.0,
                "text": "Do the magic!",
                "speaker": "SPEAKER_00",
                "timestamp_precision": "coarse",
                "asr_recovery_status": "accepted_coarse",
                "tts_placement_start": 43.2,
                "tts_placement_end": 45.4,
            },
        ]
        merged = merge_adjacent_speaker_turns(segments)
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[1]["tts_placement_start"], 43.2)

    @staticmethod
    def _entity_segment(text="Aurora, wake up!"):
        stamps = [
            {"text": "Aurora", "start": 13.09, "end": 13.40},
            {"text": "wake", "start": 13.50, "end": 13.82},
            {"text": "up", "start": 13.84, "end": 14.05},
        ]
        return {
            "id": 1,
            "start": 13.09,
            "end": 17.29,
            "text": text,
            "qwen3_time_stamps": [dict(item) for item in stamps],
            "qwen3_time_stamps_raw": [dict(item) for item in stamps],
            "timestamp_precision": "word",
        }

    def test_stable_bilingual_entity_repairs_one_timed_name(self):
        segment = self._entity_segment()
        frames = [
            {"timestamp": 13.40, "text": "Elsa!\n艾莎！"},
            {"timestamp": 13.90, "text": "Elsa!\n艾莎！"},
        ]
        corrected, audit = apply_stable_bilingual_entity_corrections(
            [segment], frames, "en", "zh"
        )
        self.assertEqual(corrected[0]["text"], "Elsa, wake up!")
        self.assertEqual(corrected[0]["start"], segment["start"])
        self.assertEqual(corrected[0]["end"], segment["end"])
        self.assertEqual(corrected[0]["qwen3_time_stamps"][0]["text"], "Elsa")
        self.assertEqual(corrected[0]["qwen3_time_stamps_raw"][0]["text"], "Elsa")
        self.assertEqual(
            corrected[0]["qwen3_timestamp_status"],
            "ocr_entity_token_substitution_preserved",
        )
        self.assertTrue(audit[0]["accepted"])
        self.assertEqual(audit[0]["original_name"], "Aurora")
        self.assertEqual(audit[0]["corrected_name"], "Elsa")
        self.assertEqual(audit[0]["target_evidence"], "艾莎！")

    def test_raw_timestamp_entity_is_matched_by_word_and_time_not_list_index(self):
        segment = self._entity_segment()
        segment["qwen3_time_stamps_raw"] = [
            {"text": "noise", "start": 9.0, "end": 9.1},
            *segment["qwen3_time_stamps_raw"],
        ]
        frames = [
            {"timestamp": 13.40, "text": "Elsa!\n艾莎！"},
            {"timestamp": 13.90, "text": "Elsa!\n艾莎！"},
        ]

        corrected, audit = apply_stable_bilingual_entity_corrections(
            [segment], frames, "en", "zh"
        )

        self.assertEqual(corrected[0]["qwen3_time_stamps_raw"][0]["text"], "noise")
        self.assertEqual(corrected[0]["qwen3_time_stamps_raw"][1]["text"], "Elsa")
        self.assertTrue(audit[0]["accepted"])

    def test_true_name_supported_by_ocr_is_never_replaced(self):
        segment = self._entity_segment()
        frames = [
            {"timestamp": 13.40, "text": "Aurora!\n奥罗拉！"},
            {"timestamp": 13.90, "text": "Aurora!\n奥罗拉！"},
            {"timestamp": 13.50, "text": "Elsa!\n艾莎！"},
            {"timestamp": 13.95, "text": "Elsa!\n艾莎！"},
        ]
        corrected, audit = apply_stable_bilingual_entity_corrections(
            [segment], frames, "en", "zh"
        )
        self.assertEqual(corrected[0]["text"], segment["text"])
        self.assertEqual(audit, [])

    def test_single_frame_entity_is_not_authoritative(self):
        segment = self._entity_segment()
        corrected, audit = apply_stable_bilingual_entity_corrections(
            [segment],
            [{"timestamp": 13.40, "text": "Elsa!\n艾莎！"}],
            "en",
            "zh",
        )
        self.assertEqual(corrected[0]["text"], segment["text"])
        self.assertEqual(audit, [])

    def test_conflicting_bilingual_targets_do_not_repair(self):
        segment = self._entity_segment()
        frames = [
            {"timestamp": 13.30, "text": "Elsa!\n艾莎！"},
            {"timestamp": 13.60, "text": "Elsa!\n艾莎！"},
            {"timestamp": 13.90, "text": "Elsa!\n埃尔莎！"},
            {"timestamp": 14.10, "text": "Elsa!\n埃尔莎！"},
        ]
        corrected, audit = apply_stable_bilingual_entity_corrections(
            [segment], frames, "en", "zh"
        )
        self.assertEqual(corrected[0]["text"], segment["text"])
        self.assertEqual(audit, [])

    def test_distant_entity_subtitle_does_not_repair(self):
        segment = self._entity_segment()
        frames = [
            {"timestamp": 16.00, "text": "Elsa!\n艾莎！"},
            {"timestamp": 16.40, "text": "Elsa!\n艾莎！"},
        ]
        corrected, audit = apply_stable_bilingual_entity_corrections(
            [segment], frames, "en", "zh"
        )
        self.assertEqual(corrected[0]["text"], segment["text"])
        self.assertEqual(audit, [])

    def test_repeated_clause_boundary_moves_only_missing_suffix(self):
        segments = [
            {
                "id": 7,
                "start": 43.03,
                "end": 48.03,
                "text": "Do the magic! Do the magic! Do the",
                "timestamp_precision": "coarse",
                "asr_recovery_status": "accepted_coarse",
                "qwen3_time_stamps": [{"text": "stale", "start": 47.5, "end": 48.0}],
            },
            {
                "id": 8,
                "start": 48.03,
                "end": 59.52,
                "text": "magic Ready? Uhhm... This is amazing!",
                "timestamp_precision": "coarse",
                "asr_recovery_status": "accepted_coarse",
                "qwen3_time_stamps": [{"text": "stale", "start": 48.0, "end": 48.5}],
            },
        ]
        repaired, audit = reconcile_coarse_recovery_boundaries(segments)
        self.assertEqual(
            repaired[0]["text"],
            "Do the magic! Do the magic! Do the magic!",
        )
        self.assertEqual(repaired[1]["text"], "Ready? Uhhm... This is amazing!")
        self.assertEqual(repaired[0]["qwen3_time_stamps"], [])
        self.assertEqual(repaired[1]["qwen3_time_stamps"], [])
        self.assertEqual(audit[0]["moved_tokens"], ["magic"])
        self.assertEqual(segments[0]["text"], "Do the magic! Do the magic! Do the")
        rerun, rerun_audit = reconcile_coarse_recovery_boundaries(repaired)
        self.assertEqual(rerun, repaired)
        self.assertEqual(rerun_audit, [])

    def test_non_repeated_fragment_is_not_guessed(self):
        segments = [
            {
                "start": 1.0,
                "end": 2.0,
                "text": "We should go to",
                "timestamp_precision": "coarse",
                "asr_recovery_status": "accepted_coarse",
                "ocr_recovered": True,
            },
            {
                "start": 2.0,
                "end": 3.0,
                "text": "school now.",
                "timestamp_precision": "coarse",
                "asr_recovery_status": "accepted_coarse",
                "ocr_recovered": True,
            },
        ]
        repaired, audit = reconcile_coarse_recovery_boundaries(segments)
        self.assertEqual([item["text"] for item in repaired], [
            "We should go to", "school now.",
        ])
        self.assertEqual(audit, [])

    def test_repaired_complete_source_can_select_exact_bilingual_ocr(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory) / "ocr_optimized"
            cache_dir.mkdir(parents=True)
            frame_text = (
                "<ref>Do the magic! Do the magic!</ref>\n"
                "<ref>来点魔法！来点魔法！</ref>"
            )
            (cache_dir / "ocr_frames.json").write_text(
                json.dumps({
                    "a": {"status": "ok", "timestamp": 43.5, "segment_id": "7", "text": frame_text},
                    "b": {"status": "ok", "timestamp": 44.0, "segment_id": "7", "text": frame_text},
                }, ensure_ascii=False),
                encoding="utf-8",
            )
            result = prepare_ocr_translation_references(
                directory,
                [{"id": 7, "start": 43.0, "end": 48.0,
                  "text": "Do the magic! Do the magic!"}],
                "en",
                "zh",
            )
            reference = result["references_by_segment"]["7"][0]
            self.assertEqual(reference["target_text"], "来点魔法！来点魔法！")
            self.assertEqual(reference["reference_tier"], "exact")

    def test_repeated_ocr_subtitle_competes_when_only_count_differs(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory) / "ocr_optimized"
            cache_dir.mkdir(parents=True)
            frame_text = (
                "<ref>Do the magic! Do the magic!</ref>\n"
                "<ref>来点魔法！来点魔法！</ref>"
            )
            (cache_dir / "ocr_frames.json").write_text(
                json.dumps({
                    "a": {"status": "ok", "timestamp": 43.5, "segment_id": "7", "text": frame_text},
                    "b": {"status": "ok", "timestamp": 44.0, "segment_id": "7", "text": frame_text},
                }, ensure_ascii=False),
                encoding="utf-8",
            )
            result = prepare_ocr_translation_references(
                directory,
                [{"id": 7, "start": 43.0, "end": 48.0,
                  "text": "Do the magic! Do the magic! Do the magic!"}],
                "en",
                "zh",
            )
            reference = result["references_by_segment"]["7"][0]
            self.assertEqual(reference["target_text"], "来点魔法！来点魔法！")
            self.assertTrue(reference["repetition_count_mismatch"])
            self.assertEqual(reference["reference_policy"], "repair_or_compress")

    def test_exact_ocr_can_use_adaptive_temporal_margin(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory) / "ocr_optimized"
            cache_dir.mkdir(parents=True)
            frame_text = (
                "<ref>Hold on to the railing.</ref>\n"
                "<ref>请抓紧栏杆。</ref>"
            )
            (cache_dir / "ocr_frames.json").write_text(
                json.dumps({
                    "a": {"status": "ok", "timestamp": 11.1, "segment_id": "7", "text": frame_text},
                    "b": {"status": "ok", "timestamp": 11.3, "segment_id": "7", "text": frame_text},
                }, ensure_ascii=False),
                encoding="utf-8",
            )
            result = prepare_ocr_translation_references(
                directory,
                [{"id": 7, "start": 8.0, "end": 10.0,
                  "text": "Hold on to the railing."}],
                "en", "zh",
            )
            reference = result["references_by_segment"]["7"][0]
            self.assertTrue(reference["adaptive_time_match"])
            self.assertGreater(reference["temporal_outside_seconds"], 0.8)

    def test_coarse_duration_uses_full_slot_for_hard_gate(self):
        ideal, available = LightTTSSelector._speech_window({
            "start": 43.03,
            "end": 48.03,
            "timestamp_precision": "coarse",
            "qwen3_time_stamps": [
                {"start": 47.47, "end": 47.79, "text": "stale"}
            ],
        })
        self.assertEqual((ideal, available), (5.0, 5.0))
        decision = classify_coarse_recovery_duration(
            actual=2.75,
            target=0.44,
            available=5.0,
            measured=True,
        )
        self.assertEqual(decision["status"], "shorter_than_window")
        self.assertEqual(decision["slot_ratio"], 0.55)
        self.assertGreater(decision["target_ratio"], 6.0)

    def test_coarse_duration_allows_moderate_step4_compression(self):
        decision = classify_coarse_recovery_duration(
            actual=3.501,
            target=2.48,
            available=2.48,
            measured=True,
        )
        self.assertEqual(decision["status"], "suitable")
        self.assertEqual(decision["slot_ratio"], 1.4117)

    def test_coarse_duration_still_blocks_excessive_compression(self):
        decision = classify_coarse_recovery_duration(
            actual=3.8,
            target=2.48,
            available=2.48,
            measured=True,
        )
        self.assertEqual(decision["status"], "too_long_blocking")
        self.assertGreater(decision["slot_ratio"], 1.5)

    def test_wespeaker_step2_invalidates_dependent_results(self):
        state = {
            "segments_step1": [{"id": 0, "text": "hello"}],
            "segments_emotion": [{"id": 0, "emotion": "happy"}],
            "segments_translation_candidates": [{"id": 0, "text": "你好"}],
            "light_tts_selector_results": [{"id": 0}],
            "unrelated_diagnostic": {"keep": True},
        }
        removed = invalidate_downstream_state(state)
        self.assertIn("segments_emotion", removed)
        self.assertIn("segments_translation_candidates", removed)
        self.assertIn("light_tts_selector_results", removed)
        self.assertEqual(state["unrelated_diagnostic"], {"keep": True})

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

    def test_dense_but_impossible_ocr_is_not_promoted_to_dialogue(self):
        regions = [{
            "id": "deferred-dense",
            "start": 20.41,
            "end": 20.70,
            "candidate_kind": "confirmed_empty_primary_chunk",
            "dialogue_evidence": True,
        }]
        records = {"deferred-dense": [
            {"timestamp": 20.42, "text": "没有毛绒绒"},
            {"timestamp": 20.62, "text": "没有毛绒绒"},
        ]}
        recovered, remaining, audit = _recover_deferred_from_ocr(
            regions, records, "zh"
        )
        self.assertEqual(recovered, [])
        self.assertEqual(remaining[0]["asr_recovery_status"], "unresolved_blocking")
        self.assertTrue(remaining[0]["requires_contextual_asr_retry"])
        self.assertEqual(audit[0]["status"], "rejected_implausible_timing")

    def test_partial_ocr_acceptance_keeps_rejected_interval_unresolved(self):
        regions = [{
            "id": "deferred-partial",
            "start": 0.0,
            "end": 2.0,
            "candidate_kind": "confirmed_empty_primary_chunk",
            "dialogue_evidence": True,
        }]
        records = {"deferred-partial": [
            {"timestamp": 0.02, "text": "没有毛绒绒"},
            {"timestamp": 0.08, "text": "没有毛绒绒"},
            {"timestamp": 0.32, "text": "来吧"},
            {"timestamp": 0.50, "text": "来吧"},
        ]}
        recovered, remaining, audit = _recover_deferred_from_ocr(
            regions, records, "zh"
        )
        self.assertEqual([item["text"] for item in recovered], ["来吧"])
        self.assertEqual(len(remaining), 1)
        self.assertTrue(remaining[0]["requires_contextual_asr_retry"])
        self.assertEqual(
            recovered[0]["asr_recovered_parent_start"], recovered[0]["start"]
        )
        self.assertEqual(audit[0]["status"], "accepted_coarse")
        self.assertEqual(len(audit[0]["rejected_events"]), 1)


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
