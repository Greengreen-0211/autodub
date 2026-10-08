"""Synthetic zero-network preflight, runnable without torch/GPU/API credentials."""
import ast
import copy
import json
from pathlib import Path
import tempfile
import unittest
import contextlib
import io
from unittest.mock import patch

from .engine import Engine, FrozenPipeline, keep, split_on_pauses, update_segment


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / 'ocr_v211_integration.py'
MAIN_PIPELINE = ROOT / 'auto_dubbing_ver_2.0.py'


def embedded_prompt():
    tree = ast.parse(PIPELINE.read_text(encoding='utf-8-sig'))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'FROZEN_B_PROMPT' for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError('embedded FROZEN_B_PROMPT missing')


FROZEN_B_PROMPT = embedded_prompt()


def request(sid='arbitrary-id'):
    return {'id': sid, 'source_language': 'zh', 'qwen_asr': '今天我们学习机气学习的基础知识。',
            'ocr_candidates': [{'line_id': 'L1', 'text': '今天我们学习机器学习的基础知识。(10,20),(600,80)'}]}


def assessment(item, **overrides):
    rows = []
    for candidate in item['contrast_candidates']:
        row = {'candidate_id': candidate['candidate_id'], 'difference_types': ['lexical_repair'],
               'correspondence': 'local', 'error_evidence': {'basis': 'context_disambiguated',
               'asr_alternative': 'ruled_out', 'observation': 'Local lexical error supported by matching context.'},
               'independence': 'independent', 'confidence': 0.99}
        row.update(overrides)
        rows.append(row)
    return {'id': item['id'], 'judgments': rows}


def response(payload):
    items = json.loads(payload['messages'][1]['content'])['items']
    return {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(
        {'assessments': [assessment(item) for item in items]}, ensure_ascii=False)}}]}


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipe = FrozenPipeline(FROZEN_B_PROMPT)

    def test_bbox_not_replacement(self):
        p = self.pipe.prepare(request())
        self.assertEqual(p['request']['ocr_candidates'][0]['text'], '今天我们学习机器学习的基础知识。')
        self.assertEqual(len(p['item']['contrast_candidates']), 1)
        self.assertNotIn('(10,20)', p['item']['contrast_candidates'][0]['replacement'])

    def test_real_numbers_preserved(self):
        r = request()
        r['ocr_candidates'][0]['text'] = '今天学习2026年的知识(10,20),(600,80)'
        self.assertIn('2026', self.pipe.prepare(r)['request']['ocr_candidates'][0]['text'])

    def test_exact_unicode_and_repeated_span(self):
        f = self.pipe.contract.exact_application
        self.assertEqual(f('cat one cat two', 8, 11, 'cat', 'dog'), 'cat one dog two')
        self.assertEqual(f('😀猫跑', 1, 2, '猫', '狗'), '😀狗跑')
        with self.assertRaises(ValueError): f('abc', True, 2, 'b', 'x')
        with self.assertRaises(ValueError): f('abc', 0, 1, 'b', 'x')

    def test_complete_chain_accept(self):
        p = self.pipe.prepare(request())
        d = self.pipe.decide(p, assessment(p['item']))
        self.assertTrue(d['accepted'])
        self.assertEqual(d['corrected_text'], '今天我们学习机器学习的基础知识。')
        self.assertIsNotNone(d['c3_decision']['downstream_decision']['final_base_reason'])

    def test_dependency_and_surface_veto(self):
        p = self.pipe.prepare(request())
        for changes in ({'independence': 'requires_other_change'}, {'difference_types': ['surface_form']},
                        {'confidence': 0.84}, {'correspondence': 'conflicting'}):
            self.assertFalse(self.pipe.decide(p, assessment(p['item'], **changes))['accepted'])

    def test_partial_independent_accept(self):
        r = request()
        r['qwen_asr'] += '然后了解神经网落的基本原理。'
        r['ocr_candidates'][0]['text'] = '今天我们学习机器学习的基础知识。然后了解神经网络的基本原理。'
        p = self.pipe.prepare(r)
        a = assessment(p['item'])
        self.assertEqual(len(a['judgments']), 2)
        a['judgments'][1]['independence'] = 'requires_other_change'
        d = self.pipe.decide(p, a)
        self.assertTrue(d['accepted'])
        self.assertEqual(sum(bool(x['downstream'] and x['downstream'].get('downstream', {}).get('applied'))
                             for x in d['candidate_audit']), 1)

    def test_tampered_inventory_keep(self):
        p = self.pipe.prepare(request())
        a = assessment(p['item'])
        p['inventory']['candidates'][0]['replacement'] = 'totally unrelated'
        self.assertFalse(self.pipe.decide(p, a)['accepted'])

    def test_final_combination_must_pass_base_guard(self):
        text = 'abcdefghijklmnopqrst'
        proposal = {'action': 'CORRECT', 'reason': 'local replacements', 'edits': [
            {'original_span': 'abcdef', 'replacement': 'uvwxyz', 'evidence_line_ids': ['L1'], 'confidence': .99, 'reason': 'local'},
            {'original_span': 'opqrst', 'replacement': 'UVWXYZ', 'evidence_line_ids': ['L1'], 'confidence': .99, 'reason': 'local'}]}
        # Isolate combination validation: two individually accepted edits still
        # must not skip the real frozen core's total-edit-ratio check.
        with patch.object(self.pipe.a1, 'decide', return_value={
                'final_accepted': True, 'p3_reason': 'accepted', 'p4_reason': 'accepted', 'alignment': None}):
            result = self.pipe.c1r.decide(text, [{'line_id': 'L1', 'text': 'uvwxyzghijklmnUVWXYZ'}], proposal,
                                         core=self.pipe.core, a1=self.pipe.a1, p3=self.pipe.p3, p4=self.pipe.p4)
        self.assertFalse(result['accepted'])
        self.assertEqual(result['reason'], 'combined_base_rejected')
        self.assertTrue(all(not e['applied'] for e in result['edits']))

    def test_missing_key_no_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            e = Engine(tmp, prompt=FROZEN_B_PROMPT)
            self.assertEqual(e.run([request()])['arbitrary-id']['reason'], 'missing_api_key')
            self.assertFalse((Path(tmp) / 'api_audit.jsonl').exists())

    def test_no_candidates_no_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = request(); r['ocr_candidates'] = []
            e = Engine(tmp, prompt=FROZEN_B_PROMPT, transport=lambda _: self.fail('must not call'))
            self.assertFalse(e.run([r])[r['id']]['accepted'])

    def test_http_contract_cache_and_budget(self):
        calls = []
        def transport(payload):
            calls.append(payload)
            return response(payload)
        with tempfile.TemporaryDirectory() as tmp:
            requests = [request(f'variable-{i}') for i in range(9)]
            e = Engine(tmp, prompt=FROZEN_B_PROMPT, transport=transport, max_calls=2)
            self.assertEqual(sum(d['accepted'] for d in e.run(requests).values()), 9)
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[0]['model'], 'deepseek-v4-pro')
            self.assertNotIn('temperature', calls[0])
            self.assertEqual(calls[0]['thinking'], {'type': 'disabled'})
            again = Engine(tmp, prompt=FROZEN_B_PROMPT, transport=lambda _: self.fail('cache missed'), max_calls=2).run(requests)
            self.assertTrue(all(d['cache_hit'] for d in again.values()))
            self.assertFalse(e.run([request('new-id')])['new-id']['accepted'])

    def test_unknown_enum_counted_retry_bounded(self):
        calls = []
        def bad(payload):
            calls.append(1)
            value = response(payload)
            data = json.loads(value['choices'][0]['message']['content'])
            data['assessments'][0]['judgments'][0]['independence'] = 'independence_unknown'
            value['choices'][0]['message']['content'] = json.dumps(data)
            return value
        with tempfile.TemporaryDirectory() as tmp:
            e = Engine(tmp, prompt=FROZEN_B_PROMPT, transport=bad)
            self.assertFalse(e.run([request()])['arbitrary-id']['accepted'])
            self.assertEqual(len(calls), 2)
            e.run([request()]); self.assertEqual(len(calls), 2)

    def test_truncated_response_and_cache_corruption_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            def incomplete(payload):
                value = response(payload); value['choices'][0]['finish_reason'] = 'length'; return value
            e = Engine(tmp, prompt=FROZEN_B_PROMPT, transport=incomplete, max_calls=1)
            self.assertFalse(e.run([request()])['arbitrary-id']['accepted'])
            (Path(tmp) / 'cache.jsonl').write_text('{broken', encoding='utf-8')
            with self.assertRaises(ValueError): e.run([request()])

    def test_timestamps_invalidated_and_original_preserved(self):
        s = {'id': 0, 'start': 1., 'end': 3., 'text': 'a cat',
             'qwen3_time_stamps': [{'text': 'cat', 'start': 1.1, 'end': 2.} ]}
        d = update_segment(s, {'accepted': True, 'corrected_text': 'a dog', 'reason': 'accepted'})
        self.assertNotIn('qwen3_time_stamps', d)
        self.assertEqual(d['qwen3_time_stamps_original'], s['qwen3_time_stamps'])
        self.assertEqual((d['start'], d['end']), (1., 3.))
        self.assertIn('qwen3_time_stamps', update_segment(s, keep(s['text'], 'keep')))
        s['qwen3_time_stamps'][0]['end'] = 9.
        self.assertNotIn('qwen3_time_stamps', update_segment(s, keep(s['text'], 'keep')))

    def test_rttm_cannot_receive_stale_words(self):
        source = {'id': 0, 'text': 'a cat', 'start': 0., 'end': 2.,
                  'qwen3_time_stamps': [{'text': 'a', 'start': 0., 'end': .9}, {'text': 'cat', 'start': 1., 'end': 2.}]}
        updated = update_segment(source, {'accepted': True, 'corrected_text': 'a dog', 'reason': 'accepted'})
        self.assertNotIn('qwen3_time_stamps', updated)
        self.assertEqual(updated['text'], 'a dog')
        self.assertEqual(updated['qwen3_time_stamps_original'], source['qwen3_time_stamps'])

    def test_natural_boundaries_and_no_silent_hard_cut(self):
        ranges = split_on_pauses(0, 62000, [(24100, 24600), (48900, 49300)])
        self.assertEqual(ranges[0][1], 24350)
        self.assertEqual(ranges[-1][1], 62000)
        self.assertTrue(all(a[1] == b[0] for a, b in zip(ranges, ranges[1:])))
        self.assertEqual(split_on_pauses(0, 62000, [])[0][2], 'unsafe_continuous_speech')
        self.assertEqual(split_on_pauses(0, 30000, []), [(0, 30000, 'range_end')])

    def test_integrated_pipeline_keeps_main_stages_and_uses_optimized_step1b(self):
        optimized = PIPELINE.read_text(encoding='utf-8-sig')
        main_source = MAIN_PIPELINE.read_text(encoding='utf-8-sig')
        self.assertIn('run_step_1b_optimized', main_source)
        self.assertIn('--ocr-preflight', main_source)
        tree = ast.parse(main_source)
        names = {getattr(node, 'name', '') for node in tree.body}
        for required in ('DubbingConfig', 'Qwen3ASRProcessor', 'HunyuanOCRClient', 'ASROCRCorrector',
                         'SpeakerDiarization', 'AdvancedTranslator', 'IndexTTSWrapper', 'Mixer',
                         'run_step_1a', 'run_step_1b', 'run_step_2', 'run_step_3', 'run_step_4', 'run_step_5', 'main'):
            self.assertIn(required, names)
        integration_names = {getattr(node, 'name', '') for node in ast.parse(optimized).body}
        self.assertIn('OCRV211Client', integration_names)
        self.assertIn('OCRV211Corrector', integration_names)
        self.assertIn('run_step_1b_optimized', integration_names)

    def test_prompt_is_embedded_and_frozen(self):
        import hashlib
        self.assertEqual(hashlib.sha256(FROZEN_B_PROMPT.encode()).hexdigest().upper(),
                         '2F6A3832F14AFC269B71D9B820DEBF10EBA4732CA4C4A40F9C571DC19B9A0471')
        self.assertNotIn('system_prompt_B.txt', PIPELINE.read_text(encoding='utf-8-sig'))


def run():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
    captured = io.StringIO()
    with patch('socket.socket.connect', side_effect=AssertionError('network forbidden in preflight')), \
            patch('socket.create_connection', side_effect=AssertionError('network forbidden in preflight')), \
            contextlib.redirect_stdout(captured):
        result = unittest.TextTestRunner(verbosity=1).run(suite)
    if not result.wasSuccessful():
        print(captured.getvalue())
        raise SystemExit(1)
    print(f'OCR_OPTIMSED_PREFLIGHT=PASSED tests={result.testsRun} API_CALLS=0 GPU_REQUIRED=0')


if __name__ == '__main__':
    run()
