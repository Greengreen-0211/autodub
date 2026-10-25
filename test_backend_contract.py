import unittest

from backend_contract import serialize_trial


class TrialContractTests(unittest.TestCase):
    def test_generated_overlength_audio_is_a_playable_degraded_result(self):
        result = serialize_trial({
            "trial_id": "trial-1",
            "segment_id": "4",
            "engine": "confucius4",
            "status": "failed",
            "file": "/private/trial.wav",
            "target_duration": 3.44,
            "actual_duration": 5.35,
            "duration_error": 1.91,
            "duration_quality": "failed",
            "error": "duration exceeded sentence window",
        }, artifact_resolver=lambda _value: "artifact-trial-1")

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["duration_quality"], "degraded")
        self.assertEqual(result["audio_artifact_id"], "artifact-trial-1")
        self.assertTrue(result["preview_available"])
        self.assertFalse(result["adoption_recommended"])
        self.assertTrue(result["requires_duration_resolution"])
        self.assertIsNone(result["error"])
        self.assertEqual(result["warning"]["code"], "TTS_DURATION_OUT_OF_RANGE")
        self.assertAlmostEqual(result["duration_ratio"], 5.35 / 3.44)

    def test_real_generation_failure_remains_failed(self):
        result = serialize_trial({
            "trial_id": "trial-2",
            "segment_id": "4",
            "engine": "confucius4",
            "status": "failed",
            "error": "model process crashed",
        })
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["preview_available"])
        self.assertEqual(result["error"]["code"], "TTS_TRIAL_FAILED")

    def test_measured_overrun_is_recovered_even_with_generic_old_error(self):
        result = serialize_trial({
            "trial_id": "trial-legacy",
            "segment_id": "4",
            "engine": "confucius4",
            "status": "failed",
            "audio_artifact_id": "artifact-existing",
            "target_duration": 3.44,
            "actual_duration": 5.35,
            "error": "trial synthesis failed",
        })
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["duration_quality"], "degraded")
        self.assertTrue(result["preview_available"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
