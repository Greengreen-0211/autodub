"""Zero-network checks for the targeted OCR host entry and state contract."""
import ast
import copy
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
TREE = ast.parse((ROOT / "auto_dubbing_ver_2.0.py").read_text(encoding="utf-8-sig"))


class MemoryState:
    current = None

    def __init__(self, config):
        pass

    def load(self):
        return copy.deepcopy(self.current)

    def delete_keys(self, keys):
        for key in keys:
            self.current.pop(key, None)

    def save(self, values):
        self.current.update(copy.deepcopy(values))


def host_entry():
    node = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == "run_step_1b")
    namespace = {"DubbingConfig": object, "StateManager": MemoryState,
                 "ASROCRCorrector": object, "Path": Path,
                 "__file__": str(ROOT / "auto_dubbing_ver_2.0.py")}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "host_entry", "exec"), namespace)
    return namespace["run_step_1b"]


class OCRHostTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.video = Path(self.temp.name) / "sample.mp4"
        self.video.write_bytes(b"test input stat only; OCR disabled")
        self.config = SimpleNamespace(temp_dir=self.temp.name, ocr_enabled=False)
        self.raw = [{"id": 0, "start": 0.0, "end": 2.0, "text": "Original speech.",
                     "qwen3_time_stamps": [], "custom_upstream": "preserved"}]
        MemoryState.current = {"input_file": str(self.video), "duration": 2.0,
                               "src_lang": "en", "segments_qwen3_raw": self.raw,
                               "segments_step1_asr": self.raw, "segments_step3": ["stale"],
                               "translation_candidates_fingerprint": "stale",
                               "custom_state": "preserved"}

    def tearDown(self):
        self.temp.cleanup()

    def test_ocr_off_preserves_text_boundaries_and_existing_fields(self):
        with patch.dict(os.environ, {"AUTODUB_OCR_MODE": "frozen_b"}):
            host_entry()(self.config, str(self.video), disable_ocr=True)
        state = MemoryState.current
        self.assertEqual(state["segments_qwen3_raw"], self.raw)
        self.assertEqual(state["segments_step1_asr"], self.raw)
        self.assertEqual(state["custom_state"], "preserved")
        for key, value in self.raw[0].items():
            self.assertEqual(state["segments_step1"][0][key], value)
        self.assertEqual(state["segments_step1"], state["segments_ocr_corrected"])
        self.assertEqual(state["ocr_optimized_manifest"]["mode"], "off")

    def test_rerun_invalidates_downstream_checkpoints(self):
        host_entry()(self.config, str(self.video), disable_ocr=True)
        self.assertNotIn("segments_step3", MemoryState.current)
        self.assertNotIn("translation_candidates_fingerprint", MemoryState.current)

    def test_mismatched_video_is_rejected(self):
        other = Path(self.temp.name) / "other.mp4"
        other.write_bytes(b"other")
        with self.assertRaises(RuntimeError):
            host_entry()(self.config, str(other), disable_ocr=True)

    def test_step_all_uses_shared_step1b_and_keeps_later_steps(self):
        step1 = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == "run_step_1")
        calls = {n.func.id for n in ast.walk(step1) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        self.assertIn("run_step_1b", calls)
        main = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        calls = {n.func.id for n in ast.walk(main) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        for name in ("run_step_1", "run_step_1b", "run_step_2", "run_step_emotion", "run_step_3", "run_step_4", "run_step_5"):
            self.assertIn(name, calls)


if __name__ == "__main__":
    unittest.main(verbosity=2)
