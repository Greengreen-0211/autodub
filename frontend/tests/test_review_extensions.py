"""Emotion overrides, observable routing and independent TTS comparisons."""
import copy
from uuid import uuid4

from test_workflow import WorkflowHarness
from workbench import Workbench, activity_html, available_tts_engines, trial_view, review_permissions, stage_html


class ReviewExtensionTest(WorkflowHarness):
    def test_completed_workbench_is_readonly_even_from_stale_tab(self):
        self.checkpoint("tts")
        bench = Workbench(self.client)
        stale, editor, segment = bench.select(bench.restore(self.job_id), "seg_0001")
        self.action("resume")
        self.drain()
        before = copy.deepcopy(self.job())
        permissions = review_permissions(before, self.segments()[0])
        self.assertFalse(any(permissions[k] for k in ["source", "translation", "emotion", "tts"]))
        self.assertTrue(permissions["tts_visible"])
        self.assertNotIn(" current", stage_html(before))
        self.assert_code("JOB_COMPLETED", lambda: bench.action(stale, "rework", "tts"))
        self.assert_code("JOB_COMPLETED", lambda: bench.action(stale, "rework", "translation"))
        self.assert_code("JOB_COMPLETED", lambda: bench.tts_trial_action(stale, editor, "trial_tts", "cosyvoice3"))
        self.assert_code("JOB_COMPLETED", lambda: bench.tts_trial_action(stale, editor, "adopt_tts_trial", "trial-old"))
        self.assert_code("JOB_COMPLETED", lambda: bench.edit(stale, editor, "emotion_label", "happy"))
        self.assertEqual(self.job(), before)

    def test_route_reasons_match_policy_priority(self):
        self.checkpoint("routing")
        first, second, long = self.segments()
        self.assertIn("angry", first["routing"]["reason"])
        self.assertIn("普通短句优先速度", second["routing"]["reason"])
        self.assertIn("长句/复杂句规则优先于情绪", long["routing"]["reason"])
        self.assertIn("阈值 28", long["routing"]["reason"])
        self.assertIn(first["routing"]["reason"], activity_html(self.job(), self.client.list_events(self.job_id)["items"]))
        other = self.client.create_job(None, "ja", "review", uuid4().hex)["job_id"]
        self.job_id = other
        self.checkpoint("routing")
        self.assertTrue(all(s["routing"]["planned_engine"] == "omnivoice" for s in self.segments()))
        self.assertIn("不属于中文/英文", self.segments()[2]["routing"]["reason"])
        self.assertEqual([e[1] for e in available_tts_engines(self.job())], ["omnivoice"])

    def test_emotion_override_preserves_raw_text_and_recomputes_only_dependents(self):
        self.checkpoint("tts")
        self.action("rework", from_stage="translation", scope="segments", segment_ids=["seg_0001"], force=True)
        self.drain()
        original = copy.deepcopy(self.segments()[0])
        result = self.edit("emotion_label", "happy")
        after = self.segments()[0]
        self.assertEqual(after["emotion"]["raw_label"], "angry")
        self.assertEqual(after["emotion"]["origin"], "human")
        self.assertEqual(after["emotion"]["label"], "happy")
        self.assertEqual(after["source"], original["source"])
        self.assertEqual(after["translation"], original["translation"])
        self.assertEqual(after["tts"]["status"], "stale")
        self.assertEqual(result["resume_from"], "routing")
        self.assertEqual(self.job()["current_stage"], "light_tts")
        self.assertEqual(self.segments()[1]["tts"]["status"], "success")
        self.action("resume")
        self.drain()
        self.assertEqual(self.segments()[0]["routing"]["planned_engine"], "cosyvoice3")
        self.assertIn("人工确认情绪为 happy", self.segments()[0]["routing"]["reason"])
        revision = self.job()["revision"]
        self.assertTrue(self.edit("emotion_label", "happy")["unchanged"])
        self.assertEqual(revision, self.job()["revision"])
        self.assert_code("INVALID_EMOTION", lambda: self.edit("emotion_label", "other"))
        self.continue_until("tts")
        self.action("rework", from_stage="emotion", scope="segments", segment_ids=["seg_0001"], force=True)
        self.drain()
        bench = Workbench(self.client)
        state, editor, segment = bench.select(bench.restore(self.job_id), "seg_0001")
        self.assert_code("UNSAVED_CHANGES", lambda: bench.confirm_review(state, editor, segment["source"]["effective_text"], segment["translation"]["text"], "sad"))

    def test_trial_preserves_formal_and_final_until_explicit_adoption(self):
        self.checkpoint("tts")
        self.action("resume")
        self.drain()
        original = copy.deepcopy(self.segments()[0])
        final_id = next(a["artifact_id"] for a in self.job()["artifacts"] if a["kind"] == "final_video" and not a["stale"])
        request = {"request_id": uuid4().hex, "expected_revision": self.job()["revision"], "action": "trial_tts",
                   "scope": "segments", "segment_ids": ["seg_0001"], "engine": "cosyvoice3"}
        operation = self.client.perform_action(self.job_id, request)
        self.assertEqual(operation, self.client.perform_action(self.job_id, request))
        self.assert_code("JOB_BUSY", lambda: self.action("trial_tts", scope="segments", segment_ids=["seg_0001"], engine="f5tts"))
        self.drain()
        self.assertEqual(self.job()["status"], "completed")
        self.assertEqual(self.segments()[0]["tts"], original["tts"])
        self.assertFalse(next(a for a in self.job()["artifacts"] if a["artifact_id"] == final_id)["stale"])
        trial = self.segments()[0]["tts_trials"][-1]
        self.assertEqual(trial["status"], "success")
        completed = next(e for e in self.client.list_events(self.job_id)["items"] if e["type"] == "tts.trial_completed")
        self.assertEqual(completed["operation_id"], operation["operation_id"])
        self.assertIsNone(trial["actual_duration"])
        self.assert_code("NOT_FOUND", lambda: self.action("adopt_tts_trial", scope="segments", segment_ids=["seg_0002"], trial_id=trial["trial_id"]))
        self.action("adopt_tts_trial", scope="segments", segment_ids=["seg_0001"], trial_id=trial["trial_id"])
        adopted = self.segments()[0]
        self.assertEqual(adopted["tts"]["actual_engine"], "cosyvoice3")
        self.assertEqual(adopted["tts"]["model_origin"], "human_trial")
        self.assertEqual(adopted["tts"]["audio_artifact_id"], trial["audio_artifact_id"])
        self.assertEqual(adopted["routing"], original["routing"])
        self.assertEqual(adopted["translation"], original["translation"])
        self.assertTrue(next(a for a in self.job()["artifacts"] if a["artifact_id"] == final_id)["stale"])
        self.assertEqual(self.job()["resume_from"], "mixing")
        self.assertEqual(self.job()["current_stage"], "tts")
        # Replacing a formal selection does not invalidate an unchanged comparison
        # recording: users can compare several engines and return to an earlier one.
        self.action("trial_tts", scope="segments", segment_ids=["seg_0001"], engine="confucius4")
        self.drain()
        other_trial = self.segments()[0]["tts_trials"][-1]
        self.action("adopt_tts_trial", scope="segments", segment_ids=["seg_0001"], trial_id=other_trial["trial_id"])
        self.action("adopt_tts_trial", scope="segments", segment_ids=["seg_0001"], trial_id=trial["trial_id"])
        self.assertEqual(self.segments()[0]["tts"]["actual_engine"], "cosyvoice3")
        self.action("resume")
        self.drain()
        self.assertEqual(self.job()["status"], "completed")

    def test_failed_and_obsolete_trials_cannot_replace_formal(self):
        self.checkpoint("tts")
        original = copy.deepcopy(self.segments()[0]["tts"])
        self.client.inject_tts_failure(self.job_id, "seg_0001")
        self.action("trial_tts", scope="segments", segment_ids=["seg_0001"], engine="cosyvoice3")
        self.drain()
        failed = self.segments()[0]["tts_trials"][-1]
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(self.segments()[0]["tts"], original)
        self.assert_code("MISSING_PREREQUISITE", lambda: self.action("adopt_tts_trial", scope="segments", segment_ids=["seg_0001"], trial_id=failed["trial_id"]))
        self.action("trial_tts", scope="segments", segment_ids=["seg_0001"], engine="confucius4")
        self.drain()
        successful = self.segments()[0]["tts_trials"][-1]
        html, audio, adoptable = trial_view(self.job(), self.segments()[0], successful["trial_id"], self.client.get_artifact_url)
        self.assertTrue(adoptable)
        self.assertIsNone(audio)  # Mock media are never passed to playback.
        self.assertIn("无真实音频", html)
        self.edit("emotion_label", "sad")
        self.action("resume")
        self.drain()
        self.action("resume")
        self.drain()
        self.assert_code("STALE_TRIAL", lambda: self.action("adopt_tts_trial", scope="segments", segment_ids=["seg_0001"], trial_id=successful["trial_id"]))
        self.assert_code("UNSUPPORTED_MODEL", lambda: self.action("trial_tts", scope="segments", segment_ids=["seg_0001"], engine="imaginary-model"))
