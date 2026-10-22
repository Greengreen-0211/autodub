import unittest

try:
    import numpy as np
except ImportError:  # pragma: no cover - production WeSpeaker env has numpy
    np = None

from run_step2_wespeaker import (
    _centroid_for_centers,
    _cluster_centroid,
    _cluster_pair_evidence,
    _robust_identity_match,
    _select_change_boundaries,
    _soft_continuity_threshold,
    _temporal_assignment_context,
    attach_speaker_confidence,
    reassemble_final_speaker_segments,
)
from speaker_alignment import AlignmentConfig, align_segments_with_rttm, smooth_rttm_spans


@unittest.skipIf(np is None, "numpy is required")
class SpeakerClusteringEvidenceTests(unittest.TestCase):
    def test_boundary_centroid_excludes_crossing_windows(self):
        windows = [
            {
                "start": 0.0,
                "end": 1.5,
                "center": 0.75,
                "embedding": np.asarray([1.0, 0.0]),
            },
            {
                "start": 1.0,
                "end": 2.5,
                "center": 1.75,
                "embedding": np.asarray([0.0, 1.0]),
            },
        ]
        centroid, count, _ = _centroid_for_centers(
            windows, 0.0, 2.0, boundary=2.0, side="left"
        )
        self.assertEqual(count, 1)
        np.testing.assert_allclose(centroid, [1.0, 0.0])

    def test_punctuation_without_reliable_support_stays_soft(self):
        region = ("input", 0.0, 8.0)
        windows = [
            {"region": region, "center": value}
            for value in (1.0, 2.0, 3.0, 5.0, 6.0, 7.0)
        ]
        candidate = {
            "time": 4.0,
            "score": 0.70,
            "similarity": 0.30,
            "gap": 0.30,
            "left_count": 3,
            "right_count": 3,
            "left_cohesion": 0.90,
            "right_cohesion": 0.90,
            "left_text": "hello",
            "right_text": "there",
            "punctuation": "strong",
            "same_asr_segment": False,
            "timestamp_reliable": False,
        }
        selected, _ = _select_change_boundaries([candidate], windows)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["constraint"], "soft")

    def test_reliable_multi_source_boundary_can_be_hard(self):
        region = ("input", 0.0, 8.0)
        windows = [
            {"region": region, "center": value}
            for value in (1.0, 2.0, 3.0, 5.0, 6.0, 7.0)
        ]
        candidate = {
            "time": 4.0,
            "score": 0.70,
            "similarity": 0.30,
            "gap": 0.30,
            "left_count": 3,
            "right_count": 3,
            "left_cohesion": 0.90,
            "right_cohesion": 0.90,
            "left_text": "hello",
            "right_text": "there",
            "punctuation": "strong",
            "same_asr_segment": False,
            "timestamp_reliable": True,
        }
        selected, _ = _select_change_boundaries([candidate], windows)
        self.assertEqual(selected[0]["constraint"], "hard")

    def test_strong_punctuation_soft_boundary_remains_reconcilable(self):
        threshold = _soft_continuity_threshold([{
            "same_asr_segment": True,
            "constraint": "soft",
            "punctuation": "strong",
        }])
        self.assertIsNotNone(threshold)
        self.assertGreaterEqual(threshold, 0.48)

    def test_centroid_similarity_cannot_hide_cross_turn_disagreement(self):
        turns = [
            {"embedding": np.asarray([1.0, 0.0]), "duration": 3.0, "embedding_reliable": True},
            {"embedding": np.asarray([0.8, 0.6]), "duration": 3.0, "embedding_reliable": True},
            {"embedding": np.asarray([1.0, 0.0]), "duration": 3.0, "embedding_reliable": True},
            {"embedding": np.asarray([-0.2, 0.9797959]), "duration": 3.0, "embedding_reliable": True},
        ]
        evidence = _cluster_pair_evidence({0, 1}, {2, 3}, turns)
        self.assertGreater(evidence["centroid"], 0.48)
        self.assertFalse(_robust_identity_match(evidence, 0.48, anchor_pair=True))

    def test_weak_short_turn_does_not_contaminate_reliable_prototype(self):
        turns = [
            {
                "embedding": np.asarray([1.0, 0.0]),
                "duration": 4.0,
                "embedding_reliable": True,
            },
            {
                "embedding": np.asarray([0.0, 1.0]),
                "duration": 0.4,
                "embedding_reliable": False,
            },
        ]
        centroid = _cluster_centroid({0, 1}, turns)
        np.testing.assert_allclose(centroid, [1.0, 0.0])

    def test_temporal_context_detects_non_neighbor_assignment(self):
        turns = [
            {"start": 0.0, "end": 1.0, "embedding_reliable": True},
            {"start": 1.0, "end": 1.4, "embedding_reliable": False},
            {"start": 1.4, "end": 2.4, "embedding_reliable": True},
            {"start": 3.0, "end": 4.0, "embedding_reliable": True},
        ]
        context = _temporal_assignment_context({1}, {3}, turns)
        self.assertEqual(context["target_neighbor_support"], 0)
        self.assertEqual(context["competing_reliable_neighbors"], 2)


class SpeakerAlignmentFallbackTests(unittest.TestCase):
    def test_short_real_island_is_not_erased_at_point_four_seconds(self):
        spans = [
            {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"},
            {"start": 2.0, "end": 2.4, "speaker": "SPEAKER_01"},
            {"start": 2.4, "end": 4.0, "speaker": "SPEAKER_00"},
        ]
        result = smooth_rttm_spans(spans, config=AlignmentConfig())
        self.assertEqual(len(result), 3)

    def test_even_very_short_distinct_label_is_not_erased_without_acoustics(self):
        spans = [
            {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"},
            {"start": 2.0, "end": 2.2, "speaker": "SPEAKER_01"},
            {"start": 2.2, "end": 4.0, "speaker": "SPEAKER_00"},
        ]
        self.assertEqual(len(smooth_rttm_spans(spans)), 3)

    def test_coarse_multi_speaker_overlap_is_marked_ambiguous(self):
        segments = [{
            "id": 0,
            "start": 0.0,
            "end": 4.0,
            "text": "coarse recovered dialogue",
            "speaker_boundary_eligible": False,
        }]
        spans = [
            {"start": 0.0, "end": 2.2, "speaker": "SPEAKER_00"},
            {"start": 2.2, "end": 4.0, "speaker": "SPEAKER_01"},
        ]
        result = align_segments_with_rttm(segments, spans)
        self.assertTrue(result[0]["speaker_boundary_ambiguous"])
        self.assertEqual(
            result[0]["speaker_assignment_status"],
            "ambiguous_coarse_overlap",
        )
        self.assertEqual(len(result[0]["speaker_overlap_candidates"]), 2)

    def test_global_identity_mapping_reassembles_raw_label_fragments(self):
        segments = [
            {"id": 0, "start": 0.0, "end": 1.0, "text": "one", "speaker": "SPEAKER_00"},
            {"id": 1, "start": 1.0, "end": 2.0, "text": "two", "speaker": "SPEAKER_01"},
        ]
        diagnostics = {
            "speaker_identity_map": {
                "SPEAKER_00": "SPEAKER_00",
                "SPEAKER_01": "SPEAKER_00",
            },
            "speaker_profiles": {},
        }
        mapped = attach_speaker_confidence(segments, diagnostics)
        merged = reassemble_final_speaker_segments(mapped)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["text"], "one two")


if __name__ == "__main__":
    unittest.main()
