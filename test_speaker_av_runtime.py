"""Boundary contracts for the isolated 3D-Speaker runtime adapter."""
from types import SimpleNamespace
import unittest

import numpy as np

from speaker_av_runtime import inclusive_visual_anchor_embeddings


class _Joint:
    @staticmethod
    def arrange_labels(labels, a_st=0):
        mapping = {}
        output = []
        for label in labels:
            if label not in mapping:
                mapping[label] = a_st + len(mapping)
            output.append(mapping[label])
        return np.asarray(output)


class VisualAudioAnchorBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.joint = _Joint()
        self.audio_embeddings = np.asarray([[1.0, 0.0]])
        self.audio_labels = np.asarray([0])
        self.audio_times = np.asarray([[18.33, 19.83]])
        self.conf = SimpleNamespace(face_det_stride=5)

    def anchors(self, end):
        visual_times = np.asarray([18.72, 18.92, 19.12, 19.32, 19.52, end])
        return inclusive_visual_anchor_embeddings(
            self.joint,
            self.audio_embeddings,
            self.audio_labels,
            np.zeros(len(visual_times), dtype=int),
            self.audio_times,
            visual_times,
            self.conf,
        )

    def test_exactly_one_second_overlap_is_accepted(self):
        intervals, embeddings, durations = self.anchors(19.72)
        self.assertEqual(len(intervals), 1)
        self.assertEqual(set(embeddings), {1})
        self.assertAlmostEqual(durations[1], 1.0)

    def test_overlap_below_one_second_is_rejected(self):
        intervals, embeddings, durations = self.anchors(19.71)
        self.assertEqual(intervals, [])
        self.assertEqual(embeddings, {})
        self.assertEqual(durations, {})


if __name__ == "__main__":
    unittest.main()
