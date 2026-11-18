"""Offline checks for the optional current native HTTP adapter."""
from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch
from speaker_av_support import write_json, read_json, sha

ROOT = Path(__file__).resolve().parent
ADAPTER = ROOT / 'integration/native_http_adapter'
spec = importlib.util.spec_from_file_location('av_binding_tested', ADAPTER / 'av_routing_binding.py')
binding = importlib.util.module_from_spec(spec); spec.loader.exec_module(binding)


class BindingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        rows = []
        for name in sorted(binding.REQUIRED) + ['fixture_%d.py' % i for i in range(41 - len(binding.REQUIRED))]:
            p = self.root / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(b'# fixture\n'); rows.append({'path': name, 'sha256': sha(p)})
        self.marker = self.root / 'task_service/BACKEND_AV_ROUTING_MANIFEST.json'
        self.record = {'schema': 'backend-av-routing-binding-v1', 'base_commit': 'f0577d9c7252bb22ae24494d5c9791d517aaba4e',
            'backend_commit': '1' * 40, 'files': rows}
        write_json(self.marker, self.record)

    def tearDown(self): self.temp.cleanup()

    def test_whole_reviewed_snapshot_required(self):
        result = binding.verified_baseline(self.root, sha(self.root / 'auto_dubbing_ver_2.0.py'))
        self.assertTrue(result['routing_branch_binding'])

    def test_modified_source_rejected(self):
        (self.root / 'fixture_3.py').write_text('# changed')
        with self.assertRaisesRegex(RuntimeError, 'modified'):
            binding.verified_baseline(self.root)

    def test_old_asr_not_silently_bound_to_the_new_runtime(self):
        with self.assertRaisesRegex(RuntimeError, 'new task'):
            binding.verified_baseline(self.root, 'unknown-old-entry')

    def test_unreviewed_install_keeps_original_verifier(self):
        self.marker.unlink()
        self.assertIsNone(binding.verified_baseline(self.root))

    def test_duplicate_manifest_entries_rejected(self):
        self.record['files'][-1] = self.record['files'][0]; write_json(self.marker, self.record)
        with self.assertRaisesRegex(RuntimeError, 'duplicate'):
            binding.verified_baseline(self.root)

    def test_missing_router_is_not_a_complete_binding(self):
        self.record['files'] = [x for x in self.record['files'] if x['path'] != 'diarization_router.py']
        extra = self.root / 'extra.py'; extra.write_bytes(b'# extra\n')
        self.record['files'].append({'path': 'extra.py', 'sha256': sha(extra)}); write_json(self.marker, self.record)
        with self.assertRaisesRegex(RuntimeError, 'modules are missing'):
            binding.verified_baseline(self.root)


class NativeContextTests(unittest.TestCase):
    def exercise(self, with_video):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); video = root / 'input.mp4'; video.write_bytes(b'fixture-video')
            server = types.ModuleType('task_service.server'); server.digest, server.read_json = sha, read_json
            package = types.ModuleType('task_service'); package.__path__ = []
            with patch.dict(sys.modules, {'task_service': package, 'task_service.server': server}):
                spec = importlib.util.spec_from_file_location('native_audio_under_test', ADAPTER / 'native_audio.py')
                module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
            runner = module.NativeRunner(root / 'work')
            job = {'job_id': 'job1', 'active_operation': 'op1', '_audio': '/vocals.wav', '_audio_sha256': 'fixed',
                '_backend_entry_sha256': 'fixed-entry', '_native': [], '_raw': [], 'revision': 0, 'duration': 45.93}
            if with_video: job['_state'] = {'input_file': str(video)}
            work = runner.root / 'job1/op1/diarization'
            captured = {}
            def popen(*args, **kwargs):
                captured.update(kwargs)
                write_json(work / 'result.json', {'segments': [], 'metadata': {}})
                return types.SimpleNamespace(wait=lambda: 0)
            with patch.object(module.subprocess, 'Popen', side_effect=popen), patch.dict(os.environ, {'DEEPSEEK_API_KEY': 'test-key'}):
                runner('diarization', job)
            plan = read_json(work / 'plan.json')
            self.assertNotIn('DEEPSEEK_API_KEY', captured['env'])
            if with_video:
                self.assertEqual(plan['_diarization_video'], str(video.resolve()))
                self.assertEqual(plan['_diarization_video_sha256'], sha(video))
                self.assertEqual(plan['_diarization_duration'], 45.93)
            else:
                self.assertNotIn('_diarization_video', plan)

    def test_video_context_reaches_native_plan(self): self.exercise(True)
    def test_audio_only_tasks_do_not_invent_a_video(self): self.exercise(False)


class InstallerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        spec = importlib.util.spec_from_file_location('native_adapter_installer_tested', ADAPTER / 'install.py')
        self.installer = importlib.util.module_from_spec(spec); spec.loader.exec_module(self.installer)
        base = read_json(ADAPTER / 'BASE.json')
        paths = [row['path'] for row in base['files']] + ['diarization_router.py', 'speaker_av_support.py',
            'speaker_av_runtime.py', 'speaker_av_worker.py', 'speaker_av_resources.py', 'speaker_av_gates.py']
        rows = [{'path': p, 'source': p, 'sha256': sha(ROOT / p)} for p in paths]
        for target, name in [('task_service/native_audio.py', 'native_audio.py'),
            ('task_service/native_trials.py', 'native_trials.py'), ('task_service/av_routing_binding.py', 'av_routing_binding.py'),
            ('tools/start_video_workflow.py', 'start_video_workflow.py')]:
            rows.append({'path': target, 'source': 'integration/native_http_adapter/' + name, 'sha256': sha(ADAPTER / name)})
        self.record = {'schema': 'backend-av-routing-binding-v1', 'base_commit': 'f0577d9c7252bb22ae24494d5c9791d517aaba4e',
            'backend_commit': '1' * 40, 'files': rows}
        self.manifest = self.root.parent / (self.root.name + '_manifest.json'); write_json(self.manifest, self.record)
        self.native = self.root / 'task_service/native_audio.py'
        self.native.parent.mkdir(); self.native.write_bytes((ADAPTER / 'native_audio.py').read_bytes())

    def tearDown(self):
        self.manifest.unlink(missing_ok=True); self.temp.cleanup()

    def test_explicit_apply_creates_backup_and_verifiable_complete_binding(self):
        root, record, changes = self.installer.prepare(self.root, self.manifest, ROOT)
        with patch.object(self.installer.socket, 'socket') as sock:
            sock.return_value.__enter__.return_value.connect_ex.return_value = 1
            backup = self.installer.apply(root, record, changes)
        self.assertTrue(backup.is_file())
        self.assertTrue(binding.verified_baseline(self.root)['routing_branch_binding'])

    def test_unknown_user_edit_is_rejected_before_write(self):
        changed = self.root / 'auto_dubbing_ver_2.0.py'; changed.write_bytes(b'# user-edited')
        with self.assertRaisesRegex(RuntimeError, 'unreviewed edits'):
            self.installer.prepare(self.root, self.manifest, ROOT)
        self.assertEqual(changed.read_bytes(), b'# user-edited')

    def test_live_HTTP_port_blocks_source_replacement(self):
        root, record, changes = self.installer.prepare(self.root, self.manifest, ROOT)
        before = self.native.read_bytes()
        with patch.object(self.installer.socket, 'socket') as sock:
            sock.return_value.__enter__.return_value.connect_ex.return_value = 0
            with self.assertRaisesRegex(RuntimeError, 'active'):
                self.installer.apply(root, record, changes)
        self.assertEqual(self.native.read_bytes(), before)
        self.assertFalse((self.root / 'logs').exists())


class EntryPointTests(unittest.TestCase):
    def function(self, file, name, namespace):
        tree = ast.parse((ROOT / file).read_text(encoding='utf-8'))
        fn = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == name)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), file, 'exec'), namespace)
        return namespace[name]

    def test_main_pipeline_dispatches_through_router(self):
        store, config, old = object(), types.SimpleNamespace(), Mock()
        fn = self.function('auto_dubbing_ver_2.0.py', 'run_step_2', {'DubbingConfig': object,
            'StateManager': lambda _: store, '_run_step_2_legacy': old})
        with patch('diarization_router.run_routed') as routed:
            fn(config)
        self.assertIs(routed.call_args.args[0], config)
        self.assertIs(routed.call_args.args[1], store)
        old.assert_not_called()

    def test_standalone_wespeaker_dispatches_through_same_router(self):
        old = Mock()
        fn = self.function('run_step2_wespeaker.py', 'main', {'TEMP_DIR': '/owned/temp', '_run_wespeaker_legacy': old})
        with patch('diarization_router.run_bridge') as routed:
            fn()
        routed.assert_called_once_with('/owned/temp', old)
        old.assert_not_called()


class DeliveryBindingTests(unittest.TestCase):
    """The delivery must name the commit actually containing its runtime."""
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.repo = Path(self.temp.name)
        module_spec = importlib.util.spec_from_file_location('prepare_manifest_tested', ADAPTER / 'prepare_manifest.py')
        self.module = importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(self.module)
        self.base_manifest = read_json(ADAPTER / 'BASE.json')
        names = [x['path'] for x in self.base_manifest['files']] + list(self.module.MODULES)
        names += ['integration/native_http_adapter/' + x for x in self.module.HTTP.values()]
        for name in names:
            path = self.repo / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b'# fixture\n')
        write_json(self.repo / 'integration/native_http_adapter/BASE.json', self.base_manifest)
        self.git('init', '-q')
        self.git('add', '.')
        self.git('commit', '-qm', 'fixture base')
        self.module.BASE = self.git('rev-parse', 'HEAD').strip()
        self.changed = self.repo / 'diarization_router.py'

    def tearDown(self): self.temp.cleanup()

    def git(self, *args):
        return subprocess.run(['git', '-C', str(self.repo), '-c', 'user.name=AutoDub test',
            '-c', 'user.email=autodub-test@example.invalid', '-c', 'core.autocrlf=false', *args],
            check=True, capture_output=True, text=True).stdout

    def commit_feature(self):
        self.changed.write_bytes(b'# feature\n'); self.git('add', 'diarization_router.py')
        self.git('commit', '-qm', 'fixture feature')
        return self.git('rev-parse', 'HEAD').strip()

    def test_requires_a_real_feature_commit(self):
        with self.assertRaisesRegex(RuntimeError, 'Commit the AV feature'):
            self.module.prepare(self.repo)

    def test_binds_all_51_installed_sources_to_actual_head(self):
        head = self.commit_feature()
        record = self.module.prepare(self.repo)
        self.assertEqual(record['backend_commit'], head)
        self.assertEqual(len(record['files']), 51)
        self.assertEqual(len({x['path'] for x in record['files']}), 51)
        self.assertIn('task_service/av_routing_binding.py', {x['path'] for x in record['files']})

    def test_uncommitted_runtime_is_rejected(self):
        self.commit_feature(); self.changed.write_bytes(b'# uncommitted\n')
        with self.assertRaisesRegex(RuntimeError, 'differs from the actual commit'):
            self.module.prepare(self.repo)

    def test_uncommitted_missing_source_is_rejected(self):
        self.commit_feature(); self.changed.unlink()
        with self.assertRaisesRegex(RuntimeError, 'Missing runtime source'):
            self.module.prepare(self.repo)

    def test_checkout_line_endings_do_not_forge_new_identity(self):
        self.commit_feature(); self.changed.write_bytes(b'# feature\r\n')
        record = self.module.prepare(self.repo)
        row = next(x for x in record['files'] if x['path'] == 'diarization_router.py')
        self.assertEqual(row['sha256'], hashlib.sha256(b'# feature\n').hexdigest())


if __name__ == '__main__': unittest.main()
