"""Stage gates, locked old steps and rework through the same gates."""
from test_workflow import WorkflowHarness
from workbench import Workbench, review_permissions


class SequentialReviewTest(WorkflowHarness):
    def test_each_gate_stops_before_future_work_and_opens_one_section(self):
        self.action("run")
        for stage, permission in [("ocr", "source"), ("emotion", "emotion"),
                                  ("light_tts", "translation"), ("routing", "routing"), ("tts", "tts")]:
            self.drain()
            job = self.job()
            self.assertEqual(job["current_stage"], stage)
            self.assertEqual(job["status"], "waiting_review")
            p = review_permissions(job, self.segments()[0])
            self.assertEqual([k for k in ["source", "emotion", "translation", "routing", "tts"] if p[k]], [permission])
            index = next(i for i, s in enumerate(job["stages"]) if s["id"] == stage)
            self.assertTrue(all(s["attempt"] == 0 for s in job["stages"][index + 1:]))
            for _ in range(60):
                self.client.tick(self.job_id)
            self.assertEqual(self.job(), job)
            self.action("resume")
        self.drain()
        self.assertEqual(self.job()["status"], "completed")

    def test_emotion_save_does_not_skip_other_pending_sentences(self):
        self.checkpoint("emotion")
        bench = Workbench(self.client)
        state, editor, s = bench.select(bench.restore(self.job_id), "seg_0001")
        self.assert_code("UNSAVED_CHANGES", lambda: bench.confirm_review(state, editor, "", "", "happy"))
        state, _ = bench.edit(state, editor, "emotion_label", "happy")
        self.assertEqual(self.job()["current_stage"], "emotion")
        self.assertEqual(self.job()["resume_from"], "translation")
        self.assert_code("REVIEW_LOCKED", lambda: bench.edit(*bench.select(state, "seg_0001")[:2], "translation_text", "不能提前编辑。"))
        self.continue_until("light_tts")
        self.assertTrue(all(s["translation"]["status"] == "success" for s in self.segments()))
        self.continue_until("routing")
        self.assertTrue(all(s["routing"]["status"] == "success" for s in self.segments()))
        self.assertEqual(self.segments()[0]["routing"]["planned_engine"], "cosyvoice3")

    def test_prior_steps_lock_and_rework_reopens_only_target_gate(self):
        self.checkpoint("routing")
        bench = Workbench(self.client)
        state, editor, s = bench.select(bench.restore(self.job_id), "seg_0001")
        for field, value in [("emotion_label", "sad"), ("translation_text", "禁止跨阶段保存。")]:
            self.assert_code("REVIEW_LOCKED", lambda: bench.edit(state, editor, field, value))
        self.assert_code("REVIEW_LOCKED", lambda: bench.action(state, "rework", "translation"))
        self.continue_until("tts")
        state = bench.action(bench.restore(self.job_id), "rework", "translation")
        self.drain()
        self.assertEqual(self.job()["current_stage"], "light_tts")
        p = review_permissions(self.job(), self.segments()[0])
        self.assertTrue(p["translation"])
        self.assertFalse(p["emotion"] or p["routing"] or p["tts_visible"])
        self.continue_until("tts")
        bench.action(bench.restore(self.job_id), "rework", "emotion")
        self.drain()
        p = review_permissions(self.job(), self.segments()[0])
        self.assertTrue(p["emotion"])
        self.assertFalse(p["translation_visible"] or p["routing_visible"] or p["tts_visible"])

    def test_auto_mode_finishes_without_review_gates(self):
        from uuid import uuid4
        self.job_id = self.client.create_job(None, "zh", "auto", uuid4().hex)["job_id"]
        self.action("run")
        self.drain()
        self.assertEqual(self.job()["status"], "completed")
        self.assertFalse(any(e["type"] == "stage.review_required" for e in self.client.list_events(self.job_id)["items"]))
