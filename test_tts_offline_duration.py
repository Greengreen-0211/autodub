"""Regression checks for cached HF loading, F5 duration and complete dialogue."""
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
import wave

import tts_worker
import tts_router
from tts_preflight import check_model, requirements


def wav(path, seconds):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(1000)
        stream.writeframes(b"\x01\x00" * int(seconds * 1000))


class OfflineTests(unittest.TestCase):
    def modules(self, download):
        hub = ModuleType("huggingface_hub")
        hub.hf_hub_download = download
        hub.snapshot_download = download
        hub.file_download = SimpleNamespace(hf_hub_download=download)
        hub._snapshot_download = SimpleNamespace(snapshot_download=download)
        requests = SimpleNamespace(sessions=SimpleNamespace(Session=SimpleNamespace(request=lambda: None)))
        return hub, requests

    def test_all_hf_entrypoints_force_local_even_when_false_is_passed(self):
        calls = []
        def download(*args, **kwargs):
            calls.append((args, kwargs))
            return "/cached/file"
        hub, requests = self.modules(download)
        with patch.dict(sys.modules, {"huggingface_hub": hub, "requests": requests}), patch.dict(os.environ):
            tts_worker.configure_strict_offline()
            for function in (hub.hf_hub_download, hub.file_download.hf_hub_download,
                             hub.snapshot_download, hub._snapshot_download.snapshot_download):
                self.assertEqual(function("repo", local_files_only=False, cache_dir="custom", revision="pinned"), "/cached/file")
            self.assertTrue(all(kwargs["local_files_only"] for _, kwargs in calls))
            self.assertTrue(all(kwargs["cache_dir"] == "custom" and kwargs["revision"] == "pinned" for _, kwargs in calls))
            with self.assertRaisesRegex(RuntimeError, "network request disabled"):
                requests.sessions.Session.request("https://example.invalid")

    def test_offline_setup_is_idempotent(self):
        hub, requests = self.modules(lambda *a, **k: "cached")
        with patch.dict(sys.modules, {"huggingface_hub": hub, "requests": requests}), patch.dict(os.environ):
            tts_worker.configure_strict_offline()
            original = hub.hf_hub_download
            tts_worker.configure_strict_offline()
            self.assertIs(hub.hf_hub_download, original)

    def test_missing_cache_is_not_replaced_by_network_or_success(self):
        def missing(*args, **kwargs):
            self.assertTrue(kwargs["local_files_only"])
            raise FileNotFoundError("cache missing")
        hub, requests = self.modules(missing)
        with patch.dict(sys.modules, {"huggingface_hub": hub, "requests": requests}), patch.dict(os.environ):
            tts_worker.configure_strict_offline()
            with self.assertRaisesRegex(FileNotFoundError, "cache missing"):
                hub.hf_hub_download("missing")

    def test_index_preflight_includes_runtime_hub_dependencies(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"INDEX_TTS_ROOT": tmp, "INDEX_TTS_MODEL_DIR": tmp}):
            _, hubs = requirements("indextts2")
            self.assertIn(("amphion/MaskGCT", "semantic_codec/model.safetensors"), hubs)
            self.assertIn(("funasr/campplus", "campplus_cn_common.bin"), hubs)
            self.assertIn(("facebook/w2v-bert-2.0", "__weights__"), hubs)

    def test_missing_index_codec_is_reported_before_workers(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {
            "INDEX_TTS_ROOT": tmp, "INDEX_TTS_MODEL_DIR": tmp,
            "AUTODUB_INDEXTTS2_HF_HOME": tmp, "AUTODUB_INDEXTTS2_HF_HUB_CACHE": tmp,
        }):
            report = check_model("indextts2")
            self.assertEqual(report["status"], "failed")
            self.assertIn("amphion/MaskGCT/semantic_codec/model.safetensors", " ".join(report["missing"]))

    def test_index_config_vocoder_and_matrices_are_checked(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"INDEX_TTS_ROOT": tmp, "INDEX_TTS_MODEL_DIR": tmp}):
            Path(tmp, "config.yaml").write_text("configuration fixture")
            config = {"emo_matrix": "feat2.pt", "spk_matrix": "feat1.pt",
                      "vocoder": {"name": "nvidia/bigvgan_v2_22khz_80band_256x"}}
            with patch.dict(sys.modules, {"yaml": SimpleNamespace(safe_load=lambda text: config)}):
                files, hubs = requirements("indextts2")
            self.assertIn(Path(tmp, "feat1.pt"), files)
            self.assertIn(Path(tmp, "feat2.pt"), files)
            self.assertIn((config["vocoder"]["name"], "bigvgan_generator.pt"), hubs)


class DurationTests(unittest.TestCase):
    def test_reference_uses_pure_components_from_reassembled_identity(self):
        class FakeAudio:
            dBFS = -20.0

            def __init__(self, length=30000):
                self.length = int(length)

            def __len__(self):
                return self.length

            def __getitem__(self, item):
                start = int(item.start or 0)
                stop = int(item.stop if item.stop is not None else self.length)
                return FakeAudio(max(0, stop - start))

            def export(self, path, format="wav"):
                Path(path).write_bytes(b"RIFF")

        fake_pydub = ModuleType("pydub")
        fake_pydub.AudioSegment = SimpleNamespace(
            from_wav=lambda path: FakeAudio()
        )
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            sys.modules, {"pydub": fake_pydub}
        ):
            router = tts_router.TTSRouter(
                SimpleNamespace(temp_dir=tmp), scheme="scheme3"
            )
            router._reference_score = lambda segment, clip: float(segment["end"])
            segments = [{
                "id": 9, "start": 0.0, "end": 8.0, "speaker": "A",
                "text": "joined", "speaker_reference_mixed_raw": True,
                "speaker_reference_components": [
                    {"id": "9:0", "start": 0.0, "end": 3.0,
                     "text": "first", "speaker_assignment_confidence": 0.9},
                    {"id": "9:1", "start": 3.2, "end": 8.0,
                     "text": "second", "speaker_assignment_confidence": 0.9},
                ],
            }]
            reference = router._speaker_references(segments, "unused.wav")["A"]
            self.assertEqual(reference["source_segment_id"], "9:1")
            self.assertEqual(reference["text"], "second")
            manifest = json.loads(
                (router.root / "reference_manifest.json").read_text(encoding="utf-8")
            )
            self.assertFalse(manifest["speakers"]["A"]["concatenated"])

    def test_reference_precision_gate_only_rejects_coarse_recovery(self):
        self.assertTrue(tts_router._is_precise_reference_segment({}))
        self.assertTrue(tts_router._is_precise_reference_segment({
            "timestamp_precision": "word", "asr_source": "primary"
        }))
        self.assertFalse(tts_router._is_precise_reference_segment({
            "timestamp_precision": "coarse"
        }))
        self.assertFalse(tts_router._is_precise_reference_segment({
            "asr_recovery_status": "accepted_coarse"
        }))
        self.assertFalse(tts_router._is_precise_reference_segment({
            "speaker_boundary_ambiguous": True
        }))
        self.assertTrue(tts_router._is_precise_reference_segment({
            "speaker_assignment_confidence": 0.60
        }))
        self.assertTrue(tts_router._is_precise_reference_segment({
            "speaker_assignment_confidence": 0.90
        }))

    def test_f5_native_duration_includes_processed_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp, "reference.wav")
            processed = Path(tmp, "processed.wav")
            wav(raw, 18)
            wav(processed, 6.2)
            calls = []
            class Model:
                def infer(self, ref_file, ref_text, gen_text, file_wave, show_info, fix_duration=None):
                    calls.append((ref_file, gen_text, fix_duration))
                    wav(file_wave, fix_duration - 6.2)
            backend = tts_worker.F5TTSBackend.__new__(tts_worker.F5TTSBackend)
            backend.model = Model()
            utils = SimpleNamespace(preprocess_ref_audio_text=lambda *a, **k: (str(processed), "reference"))
            with patch.dict(sys.modules, {"f5_tts.infer.utils_infer": utils}):
                for target in (1.543, 7.09, 21.17):
                    item = {"ref_audio": str(raw), "ref_text": "Reference transcript.",
                            "text": "目标文本", "target_duration": target}
                    duration = backend.generate(item, Path(tmp, "output.wav"))
                    self.assertAlmostEqual(duration, target, places=2)
                    self.assertAlmostEqual(calls[-1][2], 6.2 + target)
                    self.assertEqual(calls[-1][0], str(raw))

    def test_f5_legacy_api_without_duration_keeps_existing_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            class Model:
                def infer(self, ref_file, ref_text, gen_text, file_wave, show_info):
                    wav(file_wave, 1)
            backend = tts_worker.F5TTSBackend.__new__(tts_worker.F5TTSBackend)
            backend.model = Model()
            self.assertEqual(backend.generate({"ref_audio": "ref.wav", "ref_text": "Reference.",
                                              "text": "文本", "target_duration": 7}, Path(tmp, "out.wav")), 1)

    def test_reference_keeps_complete_short_audio_and_matching_text(self):
        from pydub import AudioSegment
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp, "vocals.wav")
            AudioSegment.silent(duration=30000).export(source, format="wav")
            router = tts_router.TTSRouter(SimpleNamespace(temp_dir=tmp), scheme="scheme3")
            router._reference_score = lambda segment, clip: float(segment["id"])
            segments = [{"id": 0, "start": 0, "end": 7, "speaker": "A", "original_text": "Complete short sentence."},
                        {"id": 1, "start": 8, "end": 29, "speaker": "A", "original_text": "Long transcript."}]
            ref = router._speaker_references(segments, str(source))["A"]
            self.assertAlmostEqual(tts_worker.wav_duration(Path(ref["audio"])), 7)
            self.assertEqual(ref["text"], "Complete short sentence.")

    def test_reference_excludes_coarse_recovery_segments(self):
        from pydub import AudioSegment
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp, "vocals.wav")
            AudioSegment.silent(duration=20000).export(source, format="wav")
            router = tts_router.TTSRouter(SimpleNamespace(temp_dir=tmp), scheme="scheme3")
            router._reference_score = lambda segment, clip: float(segment["id"])
            segments = [
                {"id": 1, "start": 0, "end": 4, "speaker": "A",
                 "original_text": "Clean precise reference.",
                 "timestamp_precision": "word", "asr_source": "primary"},
                {"id": 99, "start": 5, "end": 12, "speaker": "A",
                 "original_text": "Recovered words.",
                 "timestamp_precision": "coarse",
                 "asr_recovery_status": "accepted_coarse"},
            ]
            ref = router._speaker_references(segments, str(source))["A"]
            self.assertEqual(ref["text"], "Clean precise reference.")
            self.assertAlmostEqual(tts_worker.wav_duration(Path(ref["audio"])), 4)

    def test_reference_skips_speaker_with_only_coarse_recovery(self):
        from pydub import AudioSegment
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp, "vocals.wav")
            AudioSegment.silent(duration=5000).export(source, format="wav")
            router = tts_router.TTSRouter(SimpleNamespace(temp_dir=tmp), scheme="scheme3")
            segments = [{
                "id": 1, "start": 0, "end": 4, "speaker": "A",
                "original_text": "Recovered only.",
                "timestamp_precision": "coarse",
                "asr_recovery_status": "accepted_coarse",
            }]
            self.assertEqual(router._speaker_references(segments, str(source)), {})

    def test_worker_still_loads_once_for_two_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = root/"plan.json"
            result = root/"result.json"
            plan.write_text(json.dumps({"items": [
                {"segment_id": str(i), "model": "f5tts", "raw_path": str(root/f"raw{i}.wav")}
                for i in range(2)]}))
            loads, generated = [], []
            class Backend:
                def __init__(self): loads.append(1)
                def generate(self, item, path):
                    generated.append(item["segment_id"])
                    wav(path, 1)
                    return 1
            with patch.object(tts_worker, "F5TTSBackend", Backend), patch.object(tts_worker, "configure_strict_offline"), patch.object(sys, "argv", [
                "worker", "--model", "f5tts", "--plan", str(plan), "--result", str(result)
            ]):
                tts_worker.main()
            self.assertEqual(loads, [1])
            self.assertEqual(generated, ["0", "1"])
            self.assertTrue(all(row["status"] == "success" for row in json.loads(result.read_text()).values()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
