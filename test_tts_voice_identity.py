import os
import unittest
from unittest.mock import patch

from tts_quality import voice_consistency_quality
from tts_voice_identity import (
    build_voice_profiles,
    conservative_identity_map,
)


class DotVector(list):
    def __matmul__(self, other):
        return sum(float(left) * float(right) for left, right in zip(self, other))


class VoiceIdentityTests(unittest.TestCase):
    def _references(self):
        return {
            "A": {
                "reference_reliability": "reliable",
                "purity_score": 3.0,
                "duration": 3.0,
            },
            "B": {
                "reference_reliability": "reliable",
                "purity_score": 4.0,
                "duration": 5.0,
            },
            "C": {
                "reference_reliability": "reliable",
                "purity_score": 2.0,
                "duration": 4.0,
            },
        }

    def test_mutual_nearest_fragments_reuse_stronger_reference(self):
        profiles = {
            "A": {"embedding": [1.0, 0.0], "cohesion": 0.95, "support_duration": 3.0},
            "B": {"embedding": [0.995, 0.10], "cohesion": 0.96, "support_duration": 5.0},
            "C": {"embedding": [0.0, 1.0], "cohesion": 0.90, "support_duration": 4.0},
        }
        intervals = {
            "A": [(0.0, 1.0)],
            "B": [(2.0, 3.0)],
            "C": [(4.0, 5.0)],
        }
        with patch.dict(os.environ, {
            "AUTODUB_TTS_WESPEAKER_IDENTITY_SIMILARITY": "0.90",
            "AUTODUB_TTS_WESPEAKER_IDENTITY_MARGIN": "0.05",
        }):
            identity_map, audit = conservative_identity_map(
                profiles, intervals, self._references()
            )
        self.assertEqual(identity_map["A"], "B")
        self.assertEqual(identity_map["B"], "B")
        self.assertEqual(identity_map["C"], "C")
        self.assertEqual(len(audit["merge_decisions"]), 1)

    def test_simultaneous_labels_can_never_share_tts_identity(self):
        profiles = {
            "A": {"embedding": [1.0, 0.0], "cohesion": 0.95, "support_duration": 3.0},
            "B": {"embedding": [0.999, 0.02], "cohesion": 0.96, "support_duration": 5.0},
        }
        intervals = {"A": [(0.0, 2.0)], "B": [(1.0, 3.0)]}
        references = {key: self._references()[key] for key in profiles}
        with patch.dict(os.environ, {
            "AUTODUB_TTS_WESPEAKER_IDENTITY_SIMILARITY": "0.90",
        }):
            identity_map, audit = conservative_identity_map(
                profiles, intervals, references
            )
        self.assertEqual(identity_map, {"A": "A", "B": "B"})
        self.assertTrue(audit["similarities"][0]["temporal_overlap"])

    def test_profile_builder_downweights_ambiguous_sentence(self):
        segments = [
            {
                "id": 0, "speaker": "A", "start": 0.0, "end": 2.0,
                "speaker_assignment_confidence": 0.95,
                "wespeaker_tts_embedding": [1.0, 0.0],
                "wespeaker_tts_embedding_windows": 3,
                "wespeaker_tts_embedding_cohesion": 0.95,
            },
            {
                "id": 1, "speaker": "A", "start": 3.0, "end": 4.0,
                "speaker_assignment_confidence": 0.50,
                "speaker_boundary_ambiguous": True,
                "wespeaker_tts_embedding": [0.0, 1.0],
                "wespeaker_tts_embedding_windows": 1,
                "wespeaker_tts_embedding_cohesion": 0.40,
            },
        ]
        profiles, _ = build_voice_profiles(segments)
        self.assertGreater(profiles["A"]["embedding"][0], 0.99)
        self.assertLess(profiles["A"]["embedding"][1], 0.10)

    def test_wespeaker_low_similarity_warns_without_blocking(self):
        with patch.dict(os.environ, {
            "AUTODUB_TTS_WESPEAKER_VOICE_POLICY": "warn",
            "AUTODUB_TTS_MIN_WESPEAKER_VOICE_SIMILARITY": "0.80",
        }), patch(
            "tts_quality._preferred_voice_embedding",
            side_effect=[
                (DotVector([1.0, 0.0]), "wespeaker_simamresnet100"),
                (DotVector([0.5, 0.5]), "wespeaker_simamresnet100"),
            ],
        ):
            result = voice_consistency_quality("reference.wav", "tts.wav")
        self.assertEqual(result["voice_consistency_quality"], "degraded")
        self.assertEqual(
            result["voice_consistency_backend"], "wespeaker_simamresnet100"
        )
        self.assertEqual(result["voice_consistency_policy"], "warn")


if __name__ == "__main__":
    unittest.main()

