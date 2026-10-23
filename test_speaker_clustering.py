import unittest

try:
    import numpy as np
except ImportError:  # pragma: no cover - production WeSpeaker env has numpy
    np = None

from run_step2_wespeaker import (
    _add_structural_sentence_boundaries,
    _acoustic_boundary_scores,
    _centroid_for_centers,
    _cluster_centroid,
    _cluster_pair_evidence,
    _constrained_turn_clusters,
    _robust_identity_match,
    _reconcile_turn_clusters,
    _select_change_boundaries,
    _soft_continuity_threshold,
    _temporal_assignment_context,
    attach_speaker_confidence,
    reassemble_final_speaker_segments,
    _timestamp_items,
)
from speaker_alignment import (
    AlignmentConfig,
    align_segments_with_rttm,
    merge_adjacent_speaker_turns,
    smooth_rttm_spans,
)


@unittest.skipIf(np is None, "numpy is required")
class SpeakerClusteringEvidenceTests(unittest.TestCase):
    def test_pre_ocr_timing_skeleton_remains_boundary_evidence(self):
        segment = {
            "id": 1,
            "start": 0.0,
            "end": 3.0,
            "text": "right word. Next line.",
            "qwen3_time_stamps": [],
            "speaker_timing_skeleton_valid": True,
            "speaker_timing_text": "wrong word. Next line.",
            "speaker_timing_time_stamps": [
                {"text": "wrong", "start": 0.1, "end": 0.5},
                {"text": "word", "start": 0.6, "end": 1.0},
                {"text": "Next", "start": 2.0, "end": 2.3},
                {"text": "line", "start": 2.4, "end": 2.8},
            ],
        }
        items = _timestamp_items([segment])
        self.assertEqual([item["text"] for item in items], [
            "wrong", "word", "Next", "line"
        ])
        self.assertTrue(all(
            item["timing_source"] == "pre_ocr_timing_skeleton"
            for item in items
        ))

    def test_acoustic_scan_finds_change_without_word_timestamps(self):
        region = ("input", 0.0, 6.0)
        windows = []
        for index, center in enumerate((0.75, 1.5, 2.25, 3.75, 4.5, 5.25)):
            vector = np.asarray([1.0, 0.0]) if index < 3 else np.asarray([0.0, 1.0])
            windows.append({
                "region": region,
                "start": center - 0.70,
                "end": center + 0.70,
                "center": center,
                "embedding": vector,
            })
        candidates = _acoustic_boundary_scores(
            [{"start": 0.0, "end": 6.0, "text": "rewritten"}], windows
        )
        self.assertTrue(candidates)
        self.assertEqual(
            max(candidates, key=lambda item: item["score"])["candidate_source"],
            "acoustic_scan",
        )

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
        self.assertGreaterEqual(threshold, 0.76)

    def test_reliable_sentence_end_becomes_partition_without_claiming_hard_change(self):
        candidate = {
            "time": 4.0,
            "punctuation": "strong",
            "timestamp_reliable": True,
            "score": 0.20,
        }
        windows = [{
            "region": ("input", 0.0, 8.0),
            "center": center,
        } for center in (1.0, 3.0, 5.0, 7.0)]
        boundaries = _add_structural_sentence_boundaries(
            [], [candidate], windows
        )
        self.assertEqual(len(boundaries), 1)
        self.assertEqual(boundaries[0]["constraint"], "partition")
        self.assertEqual(boundaries[0]["candidate_source"], "sentence_partition")
        self.assertEqual(boundaries[0]["region"], ("input", 0.0, 8.0))
        self.assertIsNone(boundaries[0]["between_regions"])

    def test_structural_boundary_with_clear_change_remains_soft_constraint(self):
        candidate = {
            "time": 4.0,
            "punctuation": "strong",
            "timestamp_reliable": True,
            "score": 0.48,
        }
        windows = [{
            "region": ("input", 0.0, 8.0),
            "center": center,
        } for center in (1.0, 3.0, 5.0, 7.0)]
        boundaries = _add_structural_sentence_boundaries(
            [], [candidate], windows
        )
        self.assertEqual(boundaries[0]["constraint"], "soft")

    def test_raw_core_clustering_does_not_merge_moderately_similar_voices(self):
        other = np.asarray([0.645, np.sqrt(1.0 - 0.645 ** 2)])
        turns = [
            {"embedding": np.asarray([1.0, 0.0]), "duration": 3.0,
             "embedding_reliable": True, "embedding_quality": 0.9},
            {"embedding": other, "duration": 3.0,
             "embedding_reliable": True, "embedding_quality": 0.9},
        ]
        clusters = _constrained_turn_clusters(turns, set())
        self.assertEqual(len(clusters), 2)

    def test_raw_core_clustering_merges_strong_mutual_voice_match(self):
        other = np.asarray([0.95, np.sqrt(1.0 - 0.95 ** 2)])
        turns = [
            {"embedding": np.asarray([1.0, 0.0]), "duration": 3.0,
             "embedding_reliable": True, "embedding_quality": 0.9},
            {"embedding": other, "duration": 3.0,
             "embedding_reliable": True, "embedding_quality": 0.9},
        ]
        clusters = _constrained_turn_clusters(turns, set())
        self.assertEqual(len(clusters), 1)

    def test_three_voice_sequence_is_not_forced_into_two_labels(self):
        voice_a = np.asarray([1.0, 0.0, 0.0])
        voice_b = np.asarray([0.58, np.sqrt(1.0 - 0.58 ** 2), 0.0])
        voice_c = np.asarray([0.62, 0.0, np.sqrt(1.0 - 0.62 ** 2)])
        embeddings = [
            voice_a, voice_b, voice_a,
            voice_c, voice_c, voice_c,
            voice_a, voice_b, voice_a,
        ]
        turns = [
            {
                "start": float(index * 3),
                "end": float(index * 3 + 2.5),
                "duration": 2.5,
                "embedding": embedding,
                "embedding_reliable": True,
                "embedding_quality": 0.9,
            }
            for index, embedding in enumerate(embeddings)
        ]
        raw = _constrained_turn_clusters(turns, {(2, 3)})
        raw.sort(key=lambda cluster: min(cluster))
        identities, _, diagnostics = _reconcile_turn_clusters(
            raw,
            turns,
            set(),
            {(2, 3)},
            {(2, 3): {
                "same_asr_segment": True,
                "constraint": "soft",
                "punctuation": "strong",
            }},
        )
        self.assertEqual(len(raw), 3)
        self.assertEqual(len(identities), 3)
        self.assertEqual(diagnostics["identity_merges"], [])

    def test_moderate_similarity_cannot_erase_soft_acoustic_boundary(self):
        other = np.asarray([0.645, np.sqrt(1.0 - 0.645 ** 2)])
        turns = [
            {"start": 0.0, "end": 2.0, "duration": 2.0,
             "embedding": np.asarray([1.0, 0.0]), "embedding_reliable": True,
             "embedding_quality": 0.9},
            {"start": 2.0, "end": 4.0, "duration": 2.0,
             "embedding": other, "embedding_reliable": True,
             "embedding_quality": 0.9},
            {"start": 6.0, "end": 8.0, "duration": 2.0,
             "embedding": np.asarray([1.0, 0.0]), "embedding_reliable": True,
             "embedding_quality": 0.9},
            {"start": 8.0, "end": 10.0, "duration": 2.0,
             "embedding": other, "embedding_reliable": True,
             "embedding_quality": 0.9},
        ]
        boundary = {
            "same_asr_segment": True,
            "constraint": "soft",
            "punctuation": "strong",
        }
        identities, mapping, diagnostics = _reconcile_turn_clusters(
            [{0, 2}, {1, 3}],
            turns,
            set(),
            {(0, 1)},
            {(0, 1): boundary},
        )
        self.assertEqual(len(identities), 2)
        self.assertNotEqual(mapping[0], mapping[1])
        self.assertEqual(diagnostics["identity_merges"], [])

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

    def test_single_long_turn_does_not_merge_at_ordinary_prototype_threshold(self):
        other = np.asarray([0.74, np.sqrt(1.0 - 0.74 ** 2)])
        turns = [
            {"start": 0.0, "end": 3.0, "duration": 3.0,
             "embedding": np.asarray([1.0, 0.0]), "embedding_reliable": True,
             "embedding_quality": 0.9},
            {"start": 6.0, "end": 9.0, "duration": 3.0,
             "embedding": np.asarray([1.0, 0.0]), "embedding_reliable": True,
             "embedding_quality": 0.9},
            {"start": 3.0, "end": 6.0, "duration": 3.0,
             "embedding": other, "embedding_reliable": True,
             "embedding_quality": 0.9},
        ]
        identities, mapping, diagnostics = _reconcile_turn_clusters(
            [{0, 1}, {2}], turns, set(), set(), {}
        )
        self.assertEqual(len(identities), 2)
        self.assertNotEqual(mapping[0], mapping[1])
        self.assertEqual(diagnostics["identity_merges"], [])

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
    def test_corrected_text_splits_using_original_timing_skeleton(self):
        segments = [{
            "id": 0,
            "start": 0.0,
            "end": 3.0,
            "text": "right word. Next line.",
            "qwen3_time_stamps": [],
            "speaker_timing_skeleton_valid": True,
            "speaker_timing_text": "wrong word. Next line.",
            "speaker_timing_time_stamps": [
                {"text": "wrong", "start": 0.1, "end": 0.5},
                {"text": "word", "start": 0.6, "end": 1.0},
                {"text": "Next", "start": 2.0, "end": 2.3},
                {"text": "line", "start": 2.4, "end": 2.8},
            ],
        }]
        spans = [
            {"start": 0.0, "end": 1.5, "speaker": "SPEAKER_00"},
            {"start": 1.5, "end": 3.0, "speaker": "SPEAKER_01"},
        ]
        result = align_segments_with_rttm(segments, spans)
        self.assertEqual([item["text"] for item in result], [
            "right word.", "Next line."
        ])
        self.assertEqual([item["speaker"] for item in result], [
            "SPEAKER_00", "SPEAKER_01"
        ])
        self.assertTrue(all(not item["qwen3_time_stamps"] for item in result))

    def test_reassembly_preserves_strong_boundary_between_parent_sentences(self):
        segments = [
            {"id": 0, "parent_segment_id": 3, "start": 0.0, "end": 1.0,
             "text": "First sentence.", "speaker": "SPEAKER_00"},
            {"id": 1, "parent_segment_id": 4, "start": 1.2, "end": 2.0,
             "text": "Second sentence.", "speaker": "SPEAKER_00"},
        ]
        self.assertEqual(len(merge_adjacent_speaker_turns(segments)), 2)

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
