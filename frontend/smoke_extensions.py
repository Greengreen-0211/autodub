"""Deterministic v0.2 mock interaction smoke; no Gradio, models or media."""
import json
import tempfile
from pathlib import Path
from uuid import uuid4

from mock_backend import MockBackendClient


def main():
    root = Path(__file__).resolve().parent
    (root / ".tmp").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root / ".tmp") as temp:
        client = MockBackendClient(Path(temp) / "mock.sqlite3", start_worker=False)
        try:
            jid = client.create_job(None, "zh", "review", uuid4().hex)["job_id"]

            def action(name, **extra):
                return client.perform_action(jid, {"request_id": uuid4().hex,
                    "expected_revision": client.get_job(jid)["revision"], "action": name, **extra})

            def drain():
                for _ in range(40):
                    if not client.get_job(jid)["active_operation"]:
                        return
                    client.tick(jid)
                raise AssertionError("mock worker did not stop")

            def advance(target):
                for _ in range(12):
                    if client.get_job(jid)["current_stage"] == target:
                        return
                    action("resume")
                    drain()
                raise AssertionError("review target not reached")

            action("run")
            drain()
            action("resume")
            drain()
            edit = client.edit_segment(jid, "seg_0001", {"request_id": uuid4().hex,
                "expected_revision": client.get_job(jid)["revision"], "field": "emotion_label", "value": "happy"})
            action("resume")
            drain()
            advance("routing")
            first = client.list_segments(jid)["items"][0]
            assert first["emotion"]["raw_label"] == "angry" and first["routing"]["planned_engine"] == "cosyvoice3"
            action("resume")
            drain()
            original = client.list_segments(jid)["items"][0]["tts"]
            trial_operation = action("trial_tts", scope="segments", segment_ids=["seg_0001"], engine="f5tts")
            drain()
            assert client.list_segments(jid)["items"][0]["tts"] == original
            adoption = action("adopt_tts_trial", scope="segments", segment_ids=["seg_0001"], trial_id=trial_operation["trial_id"])
            first = client.list_segments(jid)["items"][0]
            assert first["tts"]["actual_engine"] == "f5tts" and first["routing"]["planned_engine"] == "cosyvoice3"
            action("resume")
            drain()
            payload = {"mock": True, "contract": "v0.2 proposed supplement", "emotion_edit": edit,
                       "trial_operation": trial_operation, "adoption_operation": adoption,
                       "job": client.get_job(jid), "segments": client.list_segments(jid)["items"],
                       "events": client.list_events(jid)["items"]}
            assert payload["job"]["status"] == "completed"
            destination = root / "examples/mock_review_extensions.json"
            destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps({"status": "passed", "original_emotion": "angry", "human_emotion": "happy",
                              "routed_engine": "cosyvoice3", "adopted_engine": "f5tts", "fixture": str(destination)}, ensure_ascii=False))
        finally:
            client.close()


if __name__ == "__main__":
    main()
