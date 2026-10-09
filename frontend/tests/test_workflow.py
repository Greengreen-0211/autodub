"""Behavioral contract checks; no Gradio or model packages needed."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend_client import BackendError
from mock_backend import MockBackendClient
from workbench import (Workbench, activity_html, changed_outputs, details_html, edit_notice, event_html,
                       media_html, playback_values, review_permissions, segment_rows, stage_html, summary_html)


class WorkflowHarness(unittest.TestCase):
    def setUp(self):
        (ROOT / ".tmp").mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT / ".tmp")
        self.db = Path(self.tmp.name) / "mock.sqlite3"
        self.client = MockBackendClient(self.db, start_worker=False)
        self.job_id = self.client.create_job("same.mp4", "zh", "review", uuid4().hex)["job_id"]

    def tearDown(self):
        self.client.close()
        self.tmp.cleanup()

    def job(self):
        return self.client.get_job(self.job_id)

    def segments(self):
        return self.client.list_segments(self.job_id)["items"]

    def action(self, action, **extra):
        return self.client.perform_action(self.job_id, {
            "request_id": uuid4().hex, "expected_revision": self.job()["revision"],
            "action": action, "instruction": None, **extra})

    def drain(self):
        for _ in range(40):
            if not self.job()["active_operation"]:
                return
            self.client.tick(self.job_id)
        self.fail("worker failed to reach a boundary")

    def checkpoint(self, target):
        self.action("run")
        self.drain()
        self.continue_until(target)

    def continue_until(self, target):
        for _ in range(12):
            if self.job()["current_stage"] == target and not self.job()["active_operation"]:
                return
            self.assertNotEqual(self.job()["status"], "completed", "passed the requested checkpoint")
            self.action("resume")
            self.drain()
        self.fail("did not reach requested checkpoint")

    def edit(self, field, value, sid="seg_0001", **extra):
        return self.client.edit_segment(self.job_id, sid, {
            "request_id": uuid4().hex, "expected_revision": self.job()["revision"],
            "field": field, "value": value, **extra})

    def assert_code(self, code, callback):
        with self.assertRaises(BackendError) as caught:
            callback()
        self.assertEqual(caught.exception.code, code)

class WorkflowTest(WorkflowHarness):
    def test_tts_return_translation_pin_and_new_result(self):
        self.checkpoint("tts")
        self.assertEqual(self.job()["status"], "waiting_review")
        first = self.segments()[0]
        self.assertEqual(first["routing"]["planned_engine"], "indextts2")
        self.assertEqual(first["tts"]["actual_engine"], "f5tts")
        self.action("resume")
        self.drain()
        self.assertEqual(self.job()["status"], "completed")
        old_artifact = self.segments()[0]["tts"]["audio_artifact_id"]
        self.action("rework", from_stage="translation", scope="segments", segment_ids=["seg_0001"], force=True)
        self.assertEqual(self.job()["current_stage"], "translation")
        self.assertTrue(all(a["stale"] for a in self.job()["artifacts"] if a["kind"] == "final_video"))
        self.assertTrue(next(a for a in self.job()["artifacts"] if a["artifact_id"] == old_artifact)["stale"])
        self.drain()
        self.assertEqual(self.job()["current_stage"], "light_tts")
        text = "这是我们最后的机会。"
        result = self.edit("translation_text", text)
        self.assertEqual(result["resume_from"], "light_tts")
        self.action("resume")
        self.drain()
        self.action("resume")
        self.drain()
        self.continue_until("tts")
        s = self.segments()[0]
        self.assertTrue(s["translation"]["human_pinned"])
        self.assertEqual(s["tts"]["synthesized_text"], text)
        self.assertNotEqual(s["tts"]["audio_artifact_id"], old_artifact)
        self.assertEqual(self.segments()[1]["tts"]["status"], "success")
        # force retranslation must preserve the manually pinned text.
        self.action("rework", from_stage="translation", scope="segments", segment_ids=["seg_0001"], force=True)
        self.drain()
        self.assertEqual(self.segments()[0]["translation"]["text"], text)
        self.action("resume")
        self.drain()
        self.action("resume")
        self.drain()
        self.continue_until("mixing")
        self.assertEqual(self.job()["status"], "completed")
        self.assertEqual(len([a for a in self.job()["artifacts"] if a["kind"] == "final_video" and not a["stale"]]), 1)

    def test_source_overlay_context_dependencies_and_raw_asr(self):
        self.checkpoint("ocr")
        original = self.segments()[1]["source"]["raw_asr_text"]
        self.assertEqual(original, "I can here you.")
        result = self.edit("source_text", "I cannot hear you.", sid="seg_0002")
        self.assertEqual(result["resume_from"], "diarization")
        translation_deps = next(i for i in result["affected"] if i["stage"] == "translation")
        self.assertEqual(result["invalidated"], [])
        self.assertEqual(len(translation_deps["segment_ids"]), 3)
        self.assertEqual(self.segments()[1]["source"]["word_alignment_status"], "stale")
        self.action("resume")
        self.drain()
        self.continue_until("light_tts")
        s = self.segments()[1]
        self.assertEqual(s["source"]["raw_asr_text"], original)
        self.assertEqual(s["source"]["ocr_corrected_text"], "I can hear you.")
        self.assertIn("I cannot hear you.", s["translation"]["text"])
        self.assertTrue(s["history"])

    def test_revision_busy_idempotency_and_empty_validation(self):
        self.checkpoint("ocr")
        payload = {"request_id": "repeat", "expected_revision": 1, "field": "source_text", "value": "Corrected."}
        first = self.client.edit_segment(self.job_id, "seg_0001", payload)
        self.assertEqual(first, self.client.edit_segment(self.job_id, "seg_0001", payload))
        self.assertEqual(self.job()["revision"], 2)
        self.assert_code("IDEMPOTENCY_CONFLICT", lambda: self.client.edit_segment(self.job_id, "seg_0001", {**payload, "value": "Different."}))
        self.assert_code("REVISION_CONFLICT", lambda: self.client.edit_segment(self.job_id, "seg_0001", {**payload, "request_id": "old"}))
        self.assert_code("EMPTY_TEXT", lambda: self.edit("source_text", "  "))
        self.action("resume")
        self.assert_code("JOB_BUSY", lambda: self.edit("source_text", "Busy."))
        self.assert_code("UNSUPPORTED_SCOPE", lambda: self.client.perform_action(self.job_id, {"request_id": "unsupported", "expected_revision": 2, "action": "pause_after_stage", "instruction": "change"}))

    def test_failed_sentence_gate_and_local_retry(self):
        self.checkpoint("routing")
        self.client.inject_tts_failure(self.job_id, "seg_0002")
        op = self.action("resume")
        self.drain()
        self.assertEqual(self.job()["status"], "partial_failed")
        self.assertEqual(self.client.get_operation(self.job_id, op["operation_id"])["status"], "partial_failed")
        self.assertEqual(self.segments()[1]["tts"]["status"], "failed")
        retained = self.segments()[0]["tts"]["audio_artifact_id"]
        self.assertFalse(any(a["kind"] == "final_video" for a in self.job()["artifacts"]))
        self.action("rework", from_stage="tts", scope="segments", segment_ids=["seg_0002"], force=True)
        self.drain()
        self.assertEqual(self.segments()[0]["tts"]["audio_artifact_id"], retained)
        self.assertEqual(self.segments()[1]["tts"]["status"], "success")
        self.action("resume")
        self.drain()
        self.assertEqual(self.job()["status"], "completed")

    def test_pause_at_boundary_progress_and_snapshot_restore(self):
        op = self.action("run")
        self.client.tick(self.job_id)
        running = self.job()
        self.assertEqual(running["current_stage"], "separation")
        self.assertIsNone(running["stages"][0]["progress"])
        pause = self.action("pause_after_stage")
        self.assertEqual(pause["operation_id"], op["operation_id"])
        self.assertEqual(pause["pause_boundary"], "separation")
        self.client.tick(self.job_id)
        self.assertEqual(self.job()["status"], "paused")
        self.assertEqual(self.job()["resume_from"], "asr")
        snapshot = self.job()
        restarted = MockBackendClient(self.db, start_worker=False)
        self.assertEqual(restarted.get_job(self.job_id), snapshot)
        state = Workbench(restarted).restore(self.job_id)
        state = Workbench(restarted).refresh(state)
        self.assertEqual(len(state["events"]), len({e["sequence"] for e in state["events"]}))
        self.action("resume")
        restarted.tick(self.job_id)
        self.drain()
        self.assertEqual(self.job()["current_stage"], "ocr")
        restarted.close()

    def test_two_jobs_stable_ids_and_event_pagination(self):
        second = self.client.create_job("same.mp4", "ja", "auto", "second")
        self.assertNotEqual(second["job_id"], self.job_id)
        self.assertEqual(second, self.client.create_job("same.mp4", "ja", "auto", "second"))
        self.checkpoint("tts")
        cursor, seen = 0, []
        while True:
            page = self.client.list_events(self.job_id, cursor, limit=3)
            seen.extend(e["sequence"] for e in page["items"])
            cursor = page["next_sequence"]
            if not page["has_more"]:
                break
        self.assertEqual(seen, list(range(1, self.job()["snapshot_sequence"] + 1)))
        self.assertEqual(self.client.get_job(second["job_id"])["revision"], 1)
        self.assertEqual(self.client.list_segments(second["job_id"])["total"], 0)
        third = self.segments()[2]
        self.assertEqual(third["emotion"]["label"], "happy")
        self.assertEqual(third["routing"]["planned_engine"], "confucius4")
        self.assertEqual(third["routing"]["inputs"]["thresholds"]["zh_long_chars"], 28)

    def test_stale_candidate_and_missing_route_cannot_tts(self):
        self.checkpoint("routing")
        cid = self.segments()[0]["translation"]["candidates"][0]["candidate_id"]
        self.action("rework", from_stage="translation", scope="segments", segment_ids=["seg_0001"], force=True)
        self.drain()
        self.assert_code("STALE_CANDIDATE", lambda: self.edit("candidate_id", cid))
        current = self.segments()[0]["translation"]["candidates"][0]["candidate_id"]
        self.edit("candidate_id", current)
        self.assertTrue(self.segments()[0]["translation"]["human_pinned"])
        self.assert_code("MISSING_PREREQUISITE", lambda: self.action("rework", from_stage="tts", scope="segments", segment_ids=["seg_0001"]))
        self.assert_code("UNSUPPORTED_SCOPE", lambda: self.action("rework", from_stage="asr", scope="segments", segment_ids=["seg_0001"]))

    def test_multiple_edits_accumulate_dependencies_and_clear_review(self):
        self.checkpoint("tts")
        self.edit("translation_text", "第一句已修改。", sid="seg_0001")
        self.edit("translation_text", "第二句也已修改。", sid="seg_0002")
        op = self.action("resume")
        self.assertEqual(set(op["affected_segment_ids"]), {"seg_0001", "seg_0002"})
        self.drain()
        self.action("resume")
        self.drain()
        self.continue_until("tts")
        self.assertEqual(self.segments()[0]["tts"]["synthesized_text"], "第一句已修改。")
        self.assertEqual(self.segments()[1]["tts"]["synthesized_text"], "第二句也已修改。")
        self.assertEqual([s["id"] for s in self.job()["stages"] if s["status"] == "waiting_review"], ["tts"])

    def test_controller_keeps_editor_version_and_escapes_untrusted_html(self):
        self.checkpoint("routing")
        bench = Workbench(self.client)
        state, editor, s = bench.select(bench.restore(self.job_id), "seg_0001")
        self.edit("source_text", '<script>alert("x")</script>')
        polled = bench.refresh(state)
        self.assertEqual(editor["expected_revision"], 1)
        self.assertEqual(polled["job"]["revision"], 2)
        self.assert_code("REVISION_CONFLICT", lambda: bench.edit(polled, editor, "translation_text", "unsaved draft"))
        render = details_html(self.segments()[0]) + event_html(polled["events"]) + summary_html(polled["job"]) + stage_html(polled["job"])
        self.assertNotIn("<script>", render)
        self.assertIn("&lt;script&gt;", render)
        self.assertTrue(all(a["url"] is None for a in self.job()["artifacts"]))

    def test_idle_poll_emits_no_unchanged_view_fields(self):
        self.checkpoint("routing")
        bench = Workbench(self.client)
        state = bench.restore(self.job_id)

        def displayed(current):
            return [summary_html(current["job"]), stage_html(current["job"]),
                    activity_html(current["job"], current["events"]), segment_rows(current["segments"]),
                    details_html(current["segments"][0])]

        self.assertTrue(all(changed_outputs(state, displayed(state))))
        state = bench.refresh(state)
        self.assertFalse(any(changed_outputs(state, displayed(state))))
        # Selecting a different sentence should update details, not rewrite the phase strip/list.
        values = displayed(state)
        values[-1] = details_html(state["segments"][1])
        self.assertEqual(changed_outputs(state, values), [False, False, False, False, True])

    def test_review_wait_and_identical_save_never_expire_results(self):
        self.checkpoint("ocr")
        snapshot, segments = self.job(), self.segments()
        for _ in range(120):
            self.client.tick(self.job_id)
        self.assertEqual(self.job(), snapshot)
        result = self.edit("source_text", segments[0]["source"]["effective_text"])
        self.assertTrue(result["unchanged"])
        self.assertEqual(self.job(), snapshot)
        self.assertEqual(self.segments(), segments)
        self.assertIn("内容未变化", edit_notice(result))

    def test_edit_keeps_checkpoint_and_unstarted_dependencies_pending(self):
        self.checkpoint("ocr")
        self.edit("source_text", "A different source.")
        job = self.job()
        self.assertEqual(job["current_stage"], "ocr")
        self.assertEqual(job["resume_from"], "diarization")
        self.assertIsNone(job["active_operation"])
        self.assertTrue(all(s["status"] == "pending" for s in job["stages"][3:]))
        self.assertTrue(all(s["tts"]["status"] == "pending" and s["translation"]["status"] == "pending"
                            for s in self.segments()))
        self.assertIn("确认后从说话人", activity_html(job, []))
        self.action("resume")
        self.assertEqual(self.job()["current_stage"], "diarization")
        self.assertIsNone(self.job()["pending_edit"])

    def test_real_old_results_need_update_but_unchanged_source_preserves_them(self):
        self.checkpoint("tts")
        snapshot = self.job()
        self.edit("source_text", self.segments()[0]["source"]["effective_text"])
        self.assertEqual(self.job(), snapshot)
        self.edit("source_text", "Changed after TTS.")
        self.assertEqual(self.job()["current_stage"], "tts")
        self.assertEqual(self.job()["resume_from"], "diarization")
        self.assertEqual(self.segments()[0]["tts"]["status"], "stale")
        self.assertTrue(all(a["stale"] for a in self.job()["artifacts"]))
        self.assertIn("上次合成文案（待更新）", media_html(self.job(), self.segments()[0]))

    def test_legacy_mock_never_generated_results_are_normalized(self):
        self.checkpoint("ocr")
        with self.client.lock, self.client._connect() as db:
            job = self.client._read(db, self.job_id)
            for record in job["stages"][3:]:
                record["status"] = "stale"
            for segment in job["_segments"]:
                segment["translation"]["status"] = "stale"
                segment["tts"]["status"] = "stale"
            job["current_stage"] = "diarization"
            self.client._event(job, "edit.applied", "legacy source edit", "diarization")
            self.client._write(db, job)
        restored = self.job()
        self.assertEqual(restored["current_stage"], "ocr")
        self.assertTrue(all(s["status"] == "pending" for s in restored["stages"][3:]))
        self.assertEqual(self.segments()[0]["tts"]["status"], "pending")

    def test_media_slots_resolve_candidates_and_clear_obsolete_audio(self):
        self.checkpoint("tts")
        job, segment = self.job(), self.segments()[0]
        kinds = ["final_video", "dialogue_audio", "background_audio", "source_audio", "light_audio", "tts_audio"]
        job["artifacts"] = [{"artifact_id": kind, "kind": kind, "stale": False, "mock": False, "url": None}
                            for kind in kinds]
        segment["source"]["audio_artifact_id"] = "source_audio"
        segment["translation"]["candidates"] = [{"candidate_id": "c1", "stale": False, "audio_artifact_id": "light_audio"},
                                                  {"candidate_id": "old", "stale": True, "audio_artifact_id": "light_audio"}]
        segment["tts"]["audio_artifact_id"] = "tts_audio"
        urls = lambda jid, aid: f"https://media.example/{aid}"
        expected = [urls(self.job_id, kind) for kind in kinds]
        self.assertEqual(playback_values(job, segment, urls), expected)
        self.assertIsNone(playback_values(job, segment, urls, "old")[4])
        job["artifacts"][-1]["stale"] = True
        self.assertIsNone(playback_values(job, segment, urls)[5])
        job["artifacts"][-1]["stale"] = False
        segment["tts"]["status"] = "failed"
        self.assertIsNone(playback_values(job, segment, urls)[5])
        job["artifacts"][0]["mock"] = True
        self.assertIsNone(playback_values(job, segment, urls)[0])

    def test_review_sections_lock_and_unsaved_drafts_block_continue(self):
        self.checkpoint("ocr")
        permissions = review_permissions(self.job(), self.segments()[0])
        self.assertTrue(permissions["source"])
        self.assertFalse(permissions["translation_visible"])
        self.assertFalse(permissions["tts_visible"])
        bench = Workbench(self.client)
        state, editor, segment = bench.select(bench.restore(self.job_id), "seg_0001")
        self.assert_code("UNSAVED_CHANGES", lambda: bench.confirm_review(state, editor, "unsaved", ""))
        self.assertIsNone(self.job()["active_operation"])
        bench.confirm_review(state, editor, segment["source"]["effective_text"], "")
        self.assertFalse(review_permissions(self.job(), segment)["source"])
        self.drain()
        permissions = review_permissions(self.job(), self.segments()[0])
        self.assertFalse(permissions["source"])
        self.assertTrue(permissions["emotion"])
        self.assertFalse(permissions["translation"])
        self.continue_until("light_tts")
        self.assertTrue(review_permissions(self.job(), self.segments()[0])["translation"])
        state, editor, segment = bench.select(bench.restore(self.job_id), "seg_0001")
        self.assert_code("UNSAVED_CHANGES", lambda: bench.confirm_review(state, editor, segment["source"]["effective_text"], "draft"))
        bench.confirm_review(state, editor, segment["source"]["effective_text"], segment["translation"]["text"])
        self.drain()
        self.assertTrue(review_permissions(self.job(), self.segments()[0])["routing"])
        self.assertFalse(review_permissions(self.job(), self.segments()[0])["tts_visible"])
        self.continue_until("tts")
        permissions = review_permissions(self.job(), self.segments()[0])
        self.assertFalse(permissions["source"])
        self.assertFalse(permissions["translation"])
        self.assertTrue(permissions["tts"])
        self.action("rework", from_stage="translation", scope="segments", segment_ids=["seg_0001"], force=True)
        self.drain()
        self.assertTrue(review_permissions(self.job(), self.segments()[0])["translation"])

    def test_review_confirmation_rejects_an_external_revision_change(self):
        self.checkpoint("ocr")
        bench = Workbench(self.client)
        state, editor, segment = bench.select(bench.restore(self.job_id), "seg_0001")
        self.edit("source_text", "Changed in another browser.")
        self.assert_code("REVISION_CONFLICT", lambda: bench.confirm_review(state, editor, segment["source"]["effective_text"], ""))
        self.assertIsNone(self.job()["active_operation"])


if __name__ == "__main__":
    (ROOT / ".tmp").mkdir(exist_ok=True)
    unittest.main()
