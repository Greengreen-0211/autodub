"""Drive the mock contract without UI/ML; export compact backend review fixtures."""
import json
import tempfile
from pathlib import Path
from uuid import uuid4

from mock_backend import MockBackendClient
from workbench import Workbench

ROOT = Path(__file__).resolve().parent


def main():
    (ROOT / ".tmp").mkdir(exist_ok=True)
    (ROOT / "examples").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT / ".tmp") as temp:
        client = MockBackendClient(Path(temp) / "mock.sqlite3", start_worker=False)
        ui = Workbench(client)
        state = ui.create(None, "zh", "review")

        def drain(current):
            for _ in range(40):
                if not client.get_job(current["job_id"])["active_operation"]:
                    return ui.refresh(current)
                client.tick(current["job_id"])
            raise RuntimeError("mock worker did not pause")

        def advance(current, target):
            for _ in range(12):
                if current["job"]["current_stage"] == target and not current["job"]["active_operation"]:
                    return current
                current = drain(ui.action(current, "resume"))
            raise AssertionError("review target not reached")

        state = drain(ui.action(state, "run"))
        state, editor, segment = ui.select(state, "seg_0002")
        raw = segment["source"]["raw_asr_text"]
        state, source_edit = ui.edit(state, editor, "source_text", "I cannot hear you.")
        state = advance(state, "tts")
        tts_snapshot = state["job"]
        state = drain(ui.action(state, "rework", "translation"))
        state, editor, segment = ui.select(state, "seg_0002")
        state, translation_edit = ui.edit(state, editor, "translation_text", "我听不见你。")
        state = advance(state, "mixing")
        segment = next(s for s in state["segments"] if s["segment_id"] == "seg_0002")
        assert state["job"]["status"] == "completed"
        assert segment["source"]["raw_asr_text"] == raw
        assert segment["tts"]["synthesized_text"] == "我听不见你。"
        assert segment["translation"]["human_pinned"]
        fixture = {"notice": "模拟合同样例；无 HTTP 服务、真实模型或媒体", "contract_status": "PROPOSED_v0.1",
                   "tts_review_job": tts_snapshot, "source_edit_result": source_edit,
                   "translation_edit_result": translation_edit, "completed_job": state["job"],
                   "segments": state["segments"], "event_page": client.list_events(state["job_id"], 0),
                   "last_operation": client.get_operation(state["job_id"], state["events"][-1]["operation_id"])}
        path = ROOT / "examples/mock_contract.json"
        path.write_text(json.dumps(fixture, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"status": "passed", "job_revision": state["job"]["revision"],
                          "raw_asr_preserved": True, "human_pinned": True,
                          "tts_text": segment["tts"]["synthesized_text"], "fixture": str(path)}, ensure_ascii=False))
        client.close()


if __name__ == "__main__":
    main()
