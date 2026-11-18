"""Routing integration contracts, with no GPU, model, or provider execution."""
from pathlib import Path
from types import SimpleNamespace
import copy
import os
import io
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

import diarization_router as router
from speaker_av_support import assess_evidence, read_json, write_json, sha, fingerprint


GOOD = {'status': 'completed', 'visual_features': 77, 'voice_prototypes_from_visual': 1,
        'audio_clusters_with_visual_match': 2, 'visual_fusion_applied': True, 'audio_branch_matches_B': True}
EMPTY = {'status': 'unavailable_no_visual_features', 'visual_features': 0, 'visual_fusion_applied': False,
         'gates': {'gate_histogram': {'fqa_rejected': 45}, 'fqa_distribution': {'max': .12673316895961761}}}


class EvidenceTests(unittest.TestCase):
    def test_real_empty_Titanic_evidence_is_not_B_success(self):
        self.assertEqual(assess_evidence(EMPTY)[:2], (False, 'no_visual_features'))

    def test_features_without_usable_anchor_are_insufficient(self):
        p = dict(GOOD, status='no_usable_visual_anchors', visual_features=1, visual_fusion_applied=False)
        self.assertEqual(assess_evidence(p)[:2], (False, 'no_usable_visual_anchors'))

    def test_valid_partial_anchors_do_not_require_an_invented_coverage_ratio(self):
        self.assertTrue(assess_evidence(GOOD)[0])

    def test_no_partition_change_is_not_a_fallback_trigger(self):
        self.assertTrue(assess_evidence(dict(GOOD, partition_changed=False))[0])

    def test_inconsistent_audio_branch_is_not_accepted(self):
        self.assertEqual(assess_evidence(dict(GOOD, audio_branch_matches_B=False))[:2], (False, 'audio_branch_mismatch'))

    def test_malformed_counts_are_not_evidence(self):
        self.assertFalse(assess_evidence(dict(GOOD, voice_prototypes_from_visual='bad'))[0])


class RouterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = SimpleNamespace(temp_dir=str(self.root))
        self.store = router.BridgeStore(self.root / 'project_state.json')
        self.video, self.vocals = self.root / 'video.mp4', self.root / 'vocals.wav'
        self.video.write_bytes(b'fixture-only-video'); self.vocals.write_bytes(b'fixture-only-vocals')
        self.raw = [{'id': 's1', 'start': 0, 'end': 2, 'text': 'fixed source', 'human_source_pinned': True}]
        self.store.save({'segments_step1': self.raw, 'segments_qwen3_raw': copy.deepcopy(self.raw),
            'vocals_path': str(self.vocals), 'input_file': str(self.video), 'audio_clips': [{'file': 'old'}],
            'segments_step2': [{'id': 'old', 'speaker': 'STALE'}]})
        self.request = {'fingerprint': 'fixed-av-request', 'video': str(self.video), 'video_sha256': sha(self.video),
            'vocals': str(self.vocals), 'vocals_sha256': sha(self.vocals), 'duration': 2, 'settings': {}}
        self.environ = patch.dict(os.environ, {'AUTODUB_DIAR_ROUTING_MODE': 'auto', 'AUTODUB_DIAR_BACKEND': 'wespeaker'})
        self.environ.start()
        self.make = patch.object(router, 'make_request', return_value=self.request); self.make.start()
        self.legacy_calls = []
        def legacy():
            self.legacy_calls.append(dict(mode=os.environ['AUTODUB_DIAR_ROUTING_MODE'],
                backend=os.environ['AUTODUB_DIAR_BACKEND'], rebuild=os.environ.get('AUTODUB_DIAR_FORCE_REBUILD')))
            self.store.save({'segments_step2': [{'id': 's1', 'start': 0, 'end': 2, 'text': 'fixed source', 'speaker': 'OLD_NATIVE'}]})
        self.legacy = legacy

    def tearDown(self):
        self.make.stop(); self.environ.stop(); self.temp.cleanup()

    def invoke(self, proof=GOOD, runner=None, aligner=None, legacy=None):
        if runner is None: runner = Mock(return_value=({'evidence': copy.deepcopy(proof)}, False))
        if aligner is None:
            aligner = Mock(return_value=([{'id': 's1', 'start': 0, 'end': 2, 'text': 'fixed source', 'speaker': 'SPEAKER_01'}],
                {'C_identities': 1, 'unresolved_segment_ids': []}))
        router.run_routed(self.config, self.store, legacy or self.legacy, worker_runner=runner, aligner=aligner)
        return self.store.load(), runner, aligner

    def test_usable_C_is_adopted_without_legacy_or_B(self):
        state, _, _ = self.invoke()
        self.assertFalse(self.legacy_calls)
        self.assertEqual(state['diarization_routing']['selected_backend'], '3dspeaker_av')
        self.assertFalse(state['diarization_routing']['audio_B_adopted'])
        self.assertNotIn('audio_clips', state)
        self.assertEqual(state['segments_qwen3_raw'], self.raw)

    def test_no_visual_features_uses_original_native_method(self):
        state, _, alignment = self.invoke(EMPTY)
        self.assertEqual(len(self.legacy_calls), 1)
        self.assertEqual(state['segments_step2'][0]['speaker'], 'OLD_NATIVE')
        self.assertEqual(state['diarization_routing']['reason_code'], 'no_visual_features')
        self.assertEqual(state['diarization_routing']['selected_backend'], 'wespeaker')
        self.assertTrue(state['diarization_routing']['fallback_used'])
        self.assertTrue(state['diarization_routing']['fallback_attempted'])
        self.assertFalse(state['diarization_routing']['audio_B_adopted'])
        alignment.assert_not_called()
        self.assertEqual(state['diarization_routing']['evidence']['gates']['gate_histogram']['fqa_rejected'], 45)

    def test_no_anchor_also_uses_legacy(self):
        state, _, _ = self.invoke(dict(GOOD, status='no_usable_visual_anchors', visual_fusion_applied=False))
        self.assertEqual(state['diarization_routing']['reason_code'], 'no_usable_visual_anchors')
        self.assertEqual(len(self.legacy_calls), 1)

    def test_timeout_reason_is_distinct_from_weak_evidence(self):
        state, _, _ = self.invoke(runner=Mock(side_effect=router.AVUnavailable('av_timeout', 'timed out')))
        self.assertEqual(state['diarization_routing']['reason_code'], 'av_timeout')
        self.assertEqual(state['diarization_routing']['selected_backend'], 'wespeaker')

    def test_bad_C_alignment_falls_back(self):
        state, _, _ = self.invoke(aligner=Mock(side_effect=router.AVUnavailable('invalid_av_rttm', 'checksum mismatch')))
        self.assertEqual(state['diarization_routing']['reason_code'], 'invalid_av_rttm')

    def test_both_failures_are_not_reported_as_success(self):
        with self.assertRaisesRegex(RuntimeError, 'native also failed'):
            self.invoke(EMPTY, legacy=Mock(side_effect=RuntimeError('native also failed')))
        state = self.store.load()
        self.assertEqual(state['diarization_routing']['status'], 'failed')
        self.assertIsNone(state['diarization_routing']['selected_backend'])
        self.assertFalse(state['diarization_routing']['fallback_used'])
        self.assertTrue(state['diarization_routing']['fallback_attempted'])
        self.assertNotIn('segments_step2', state)

    def test_input_mutation_stops_both_methods_and_invalidates_old_tts(self):
        def change(*_):
            self.vocals.write_bytes(b'changed'); return {'evidence': EMPTY}, False
        with self.assertRaisesRegex(router.AVUnavailable, '输入文件发生变化'):
            self.invoke(runner=Mock(side_effect=change))
        self.assertFalse(self.legacy_calls)
        self.assertNotIn('audio_clips', self.store.load())
        self.assertEqual(self.store.load()['diarization_routing']['status'], 'failed')
        self.assertFalse(self.store.load()['diarization_routing']['fallback_attempted'])

    def test_bridge_then_cli_is_idempotent_and_preserves_later_results(self):
        self.invoke()
        self.store.save({'audio_clips': [{'file': 'new-verified-output'}]})
        state, runner, _ = self.invoke()
        runner.assert_not_called()
        self.assertEqual(state['audio_clips'], [{'file': 'new-verified-output'}])

    def test_edit_text_reuses_audio_request_but_realigns_and_invalidates_tts(self):
        self.invoke()
        self.store.save({'segments_step1': [dict(self.raw[0], text='human edited')], 'audio_clips': [{'file': 'stale'}]})
        state, runner, alignment = self.invoke()
        runner.assert_called_once(); alignment.assert_called_once()
        self.assertNotIn('audio_clips', state)

    def test_missing_video_fallback_is_idempotent_too(self):
        with patch.object(router, 'make_request', side_effect=router.AVUnavailable('video_context_missing', 'no video')):
            self.invoke()
            self.invoke()
        self.assertEqual(len(self.legacy_calls), 1)

    def test_legacy_bypass_keeps_original_behavior(self):
        with patch.dict(os.environ, {'AUTODUB_DIAR_ROUTING_MODE': 'legacy'}):
            router.run_routed(self.config, self.store, self.legacy)
        self.assertEqual(len(self.legacy_calls), 1)
        self.assertEqual(self.store.load()['diarization_routing']['reason_code'], 'explicit_legacy_mode')

    def test_explicit_rollback_does_not_leave_a_stale_AV_label(self):
        self.invoke()
        with patch.dict(os.environ, {'AUTODUB_DIAR_ROUTING_MODE': 'legacy'}):
            router.run_routed(self.config, self.store, self.legacy)
        self.assertEqual(self.store.load()['diarization_routing']['selected_backend'], 'wespeaker')

    def test_fallback_environment_does_not_leak_into_next_stage(self):
        self.invoke(EMPTY)
        self.assertEqual(self.legacy_calls[0], {'mode': 'legacy', 'backend': 'wespeaker', 'rebuild': '1'})
        self.assertEqual(os.environ['AUTODUB_DIAR_ROUTING_MODE'], 'auto')


class SecurityTests(unittest.TestCase):
    def test_model_child_does_not_receive_provider_credentials(self):
        with patch.dict(os.environ, {'DEEPSEEK_API_KEY': 'fixture', 'OPENAI_API_KEY': 'fixture', 'GITHUB_TOKEN': 'fixture', 'HTTPS_PROXY': 'fixture'}):
            env = router.worker_environment()
        for key in ('DEEPSEEK_API_KEY', 'OPENAI_API_KEY', 'GITHUB_TOKEN', 'HTTPS_PROXY'):
            self.assertNotIn(key, env)
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '0')

    def test_actual_C_alignment_preserves_two_distinct_source_speakers(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); rttm = root / 'C.rttm'
            rttm.write_text('SPEAKER input 1 0.0 1.0 <NA> <NA> 1 <NA> <NA>\nSPEAKER input 1 1.0 1.0 <NA> <NA> 2 <NA> <NA>\n')
            state = {'segments_step1': [{'id': 0, 'start': 0, 'end': 1, 'text': 'Hello.'},
                {'id': 1, 'start': 1, 'end': 2, 'text': 'Answer.'}]}
            original = copy.deepcopy(state)
            rows, audit = router.align_C(state, {'rttm': str(rttm), 'rttm_sha256': sha(rttm), 'duration': 2}, root)
            self.assertEqual({x['speaker'] for x in rows}, {'SPEAKER_00', 'SPEAKER_01'})
            self.assertEqual(audit['unresolved_segment_ids'], [])
            self.assertEqual(state, original)

    def test_C_alignment_rejects_another_tasks_RTTM(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp); root = base / 'owned'; root.mkdir(); p = base / 'outside.rttm'; p.write_text('outside')
            with self.assertRaisesRegex(router.AVUnavailable, 'RTTM'):
                router.align_C({'segments_step1': []}, {'rttm': str(p), 'rttm_sha256': sha(p)}, root)

    def test_invalid_RTTM_clock_is_rejected(self):
        from speaker_av_support import read_rttm
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / 'bad.rttm'; p.write_text('SPEAKER input 1 0.0 4.0 <NA> <NA> 1 <NA> <NA>\n')
            with self.assertRaisesRegex(ValueError, 'clock'):
                read_rttm(p, 2)

    def test_timeout_terminates_and_reaps_worker_before_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            child = Mock(stdout=io.StringIO('fixture progress\n'))
            child.wait.side_effect = [subprocess.TimeoutExpired('fixture', 1), 0]
            request = {'fingerprint': 'timeout-test', 'settings': {'python': '/unused', 'timeout_seconds': 1}}
            with patch.object(router.subprocess, 'Popen', return_value=child):
                with self.assertRaisesRegex(router.AVUnavailable, '超时'):
                    router.run_worker(request, root)
            child.terminate.assert_called_once(); self.assertEqual(child.wait.call_count, 2)


if __name__ == '__main__':
    unittest.main()
