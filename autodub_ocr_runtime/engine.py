"""Variable-size, budgeted C8 B classification with the complete frozen guard chain."""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
from collections import Counter

from . import VERSION

HERE = Path(__file__).resolve().parent
MODEL = 'deepseek-v4-pro'
PROMPT_SHA256 = '2F6A3832F14AFC269B71D9B820DEBF10EBA4732CA4C4A40F9C571DC19B9A0471'


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def append(path, value):
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
        handle.flush()
        os.fsync(handle.fileno())


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_bytes(canonical(value))
    os.replace(temp, path)


class FrozenPipeline:
    def __init__(self, prompt):
        manifest = json.loads((HERE / 'FROZEN_SOURCES.json').read_text(encoding='utf-8'))
        for relative, entry in manifest.items():
            path = HERE.parent / relative
            if hashlib.sha256(path.read_bytes()).hexdigest().upper() != entry['sha256']:
                raise RuntimeError(f'frozen_dependency_hash_mismatch:{path.name}')
        if not isinstance(prompt, str):
            raise TypeError('frozen_B_prompt_must_be_embedded_string')
        self.prompt = prompt
        if hashlib.sha256(prompt.encode('utf-8')).hexdigest().upper() != PROMPT_SHA256:
            raise RuntimeError('frozen_B_prompt_hash_mismatch')
        frozen_path = str(HERE / 'frozen')
        if frozen_path not in sys.path:
            sys.path.insert(0, frozen_path)
        names = {
            'inventory': 'v12_c4_candidate_inventory_r4', 'contract': 'v12_c8_exact_context',
            'sanitizer': 'v12_c7_ocr_structure_sanitizer_r1', 'adapter': 'v12_c4_candidate_adapter_r4',
            'c1r': 'v12_c1r_checked_adapter', 'a1': 'v12_a1_local_deletion',
            'p3': 'replay_p3_guard_candidates_v1', 'p4': 'replay_v11_p4_task_boundary_v1',
            'core': 'run_ocr_asr_correction_ablation',
        }
        for key, name in names.items():
            module = importlib.import_module(name)
            if Path(module.__file__).resolve().parent != HERE / 'frozen':
                raise RuntimeError(f'foreign_module_collision:{name}')
            setattr(self, key, module)
        self.binding = digest({'version': VERSION, 'sources': manifest,
                               'engine': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                               'model': MODEL, 'thinking': 'disabled', 'max_tokens': 8192})

    def prepare(self, raw_request):
        clean, audit = self.sanitizer.sanitize_request(raw_request)
        inventory = self.inventory.build_inventory(clean)
        item = self.contract.build_model_item(clean, inventory, inventory_impl=self.inventory)
        return {'request': clean, 'sanitizer': audit, 'inventory': inventory, 'item': item}

    def decide(self, prepared, assessment):
        return self.adapter.decide(prepared['request'], prepared['inventory'], assessment,
                                   c1r=self.c1r, core=self.core, a1=self.a1, p3=self.p3, p4=self.p4)


def keep(text, reason):
    return {'accepted': False, 'corrected_text': text, 'reason': reason, 'candidate_audit': []}


class Engine:
    def __init__(self, workdir, *, prompt, api_key='', base_url='https://api.deepseek.com', proxy_url='',
                 max_calls=6, max_attempts=2, timeout=180, transport=None, scope='default'):
        self.pipeline = FrozenPipeline(prompt)
        self.directory = Path(workdir)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.api_key, self.base_url, self.proxy_url = api_key, base_url, proxy_url
        self.max_calls, self.max_attempts = int(max_calls), int(max_attempts)
        if self.max_calls < 0 or not 1 <= self.max_attempts <= 2:
            raise ValueError('invalid_budget; calls>=0 and attempts in [1,2] required')
        self.timeout, self.transport = timeout, transport
        self.scope = scope

    def _post(self, items):
        payload = {'model': MODEL, 'messages': [
            {'role': 'system', 'content': self.pipeline.prompt},
            {'role': 'user', 'content': json.dumps({'items': items}, ensure_ascii=False, separators=(',', ':'))}],
            'thinking': {'type': 'disabled'}, 'response_format': {'type': 'json_object'},
            'max_tokens': 8192, 'stream': False}
        if self.transport is not None:
            return self.transport(payload)
        import requests
        with requests.Session() as session:
            session.trust_env = False
            if self.proxy_url:
                session.proxies.update({'http': self.proxy_url, 'https': self.proxy_url})
            response = session.post(self.base_url.rstrip('/') + '/chat/completions',
                                    headers={'Authorization': 'Bearer ' + self.api_key},
                                    json=payload, timeout=self.timeout)
            response.raise_for_status()
            return response.json()

    def run(self, raw_requests):
        ids = [item['id'] for item in raw_requests]
        if len(set(ids)) != len(ids):
            raise ValueError('duplicate_segment_ids')
        # Exclusive writer also prevents two simultaneous jobs consuming one budget.
        lock = self.directory / 'writer.lock'
        with lock.open('x', encoding='utf-8') as handle:
            handle.write(str(os.getpid()))
        try:
            return self._run(raw_requests)
        finally:
            lock.unlink()

    def _run(self, raw_requests):
        prepared, results, keys = {}, {}, {}
        for request in raw_requests:
            sid = request['id']
            try:
                prepared[sid] = self.pipeline.prepare(request)
                keys[sid] = digest({'binding': self.pipeline.binding, 'scope': self.scope,
                                    'item': prepared[sid]['item'], 'endpoint': self.base_url})
                if not prepared[sid]['item']['contrast_candidates']:
                    results[sid] = keep(request['qwen_asr'], 'no_classifiable_candidates')
            except Exception as exc:
                results[sid] = keep(request['qwen_asr'], 'prepare_error:' + str(exc))
        audit_path, cache_path = self.directory / 'api_audit.jsonl', self.directory / 'cache.jsonl'
        # A truncated/corrupt ledger stops safely instead of silently resetting budget.
        logs, cached = read_jsonl(audit_path), read_jsonl(cache_path)
        starts = [row for row in logs if row.get('event') == 'start']
        calls = len(starts)
        attempts = Counter(key for row in starts for key in row['keys'])
        cache = {row['key']: row for row in cached}
        for sid in prepared:
            if sid in results:
                continue
            if keys[sid] in cache:
                row = cache[keys[sid]]
                # Always rerun the complete guard on cached model assessments.
                results[sid] = self.pipeline.decide(prepared[sid], row['assessment'])
                results[sid]['cache_hit'] = True
        pending = [sid for sid in prepared if sid not in results]
        while pending:
            eligible = [sid for sid in pending if attempts[keys[sid]] < self.max_attempts]
            if not eligible or calls >= self.max_calls or (not self.api_key and self.transport is None):
                break
            batch = eligible[:8]
            items = [prepared[sid]['item'] for sid in batch]
            submitted = {sid: [c['candidate_id'] for c in prepared[sid]['item']['contrast_candidates']] for sid in batch}
            calls += 1
            batch_keys = [keys[sid] for sid in batch]
            # Reserve BEFORE the network: interrupted requests consume the budget too.
            append(audit_path, {'event': 'start', 'call': calls, 'keys': batch_keys,
                                'binding': self.pipeline.binding, 'request_items': items})
            attempts.update(batch_keys)
            try:
                raw = self._post(items)
                append(audit_path, {'event': 'response', 'call': calls, 'response': raw})
                choice = raw['choices'][0]
                if choice.get('finish_reason') != 'stop':
                    raise ValueError('incomplete_response')
                assessments = self.pipeline.inventory.parse_classification_response(choice['message']['content'], submitted)
                decisions = {sid: self.pipeline.decide(prepared[sid], assessments[sid]) for sid in batch}
                for sid in batch:
                    append(cache_path, {'key': keys[sid], 'id': sid, 'assessment': assessments[sid], 'call': calls})
                    results[sid] = decisions[sid]
                print(f'[OCR B] cached={len(results)}/{len(raw_requests)} calls={calls}/{self.max_calls}')
            except Exception as exc:
                # Do not persist request headers or credentials in errors.
                append(audit_path, {'event': 'error', 'call': calls, 'error_type': type(exc).__name__,
                                   'reason': str(exc).replace(self.api_key, '[REDACTED]') if self.api_key else str(exc)})
            pending = [sid for sid in pending if sid not in results]
        for sid in pending:
            reason = 'missing_api_key' if not self.api_key and self.transport is None else 'budget_or_attempts_exhausted'
            results[sid] = keep(prepared[sid]['request']['qwen_asr'], reason)
        report = {'version': VERSION, 'binding': self.pipeline.binding, 'api_calls_total': calls,
                  'max_calls': self.max_calls, 'raw_requests': raw_requests,
                  'prepared': prepared, 'decisions': results}
        atomic_json(self.directory / 'last_run.json', report)
        return results


def update_segment(segment, decision):
    result = copy.deepcopy(segment)
    original = segment['text']
    corrected = decision.get('corrected_text', original) if decision.get('accepted') else original
    result['text'] = corrected
    result['ocr_original_asr'] = original
    result['ocr_optimized'] = {'version': VERSION, 'accepted': corrected != original, 'reason': decision['reason']}
    stamps = result.get('qwen3_time_stamps') or []
    invalid = corrected != original
    previous = float(segment['start'])
    for item in stamps:
        try:
            start, end = float(item['start']), float(item['end'])
            invalid |= not (math.isfinite(start) and math.isfinite(end) and previous <= start <= end <= float(segment['end']))
            previous = end
        except (ValueError, TypeError, KeyError):
            invalid = True
    if invalid and stamps:
        result['qwen3_time_stamps_original'] = result.pop('qwen3_time_stamps')
        result['qwen3_timestamp_status'] = 'invalidated_text_changed' if corrected != original else 'invalidated_invalid_bounds'
    elif stamps:
        result['qwen3_timestamp_status'] = 'unchanged_asr_valid_bounds'
    return result


def split_on_pauses(start, end, pauses, max_ms=25000, search_ms=5000, hard_limit_ms=45000):
    """Non-overlapping ownership ranges; natural pause nearest the soft target.

    Hard-limit cuts are flagged. The ASR integration refuses them instead of
    passing truncated words into automatic OCR repair.
    """
    if not 0 <= start < end or not 0 < max_ms <= hard_limit_ms:
        raise ValueError('invalid_chunk_parameters')
    output, cursor = [], start
    while end - cursor > max_ms:
        target = cursor + max_ms
        choices = [(a + b) // 2 for a, b in pauses
                   if a >= cursor + max(350, max_ms - search_ms)
                   and b <= min(end, cursor + hard_limit_ms, target + search_ms)]
        if choices:
            cut = min(choices, key=lambda x: abs(x - target))
            reason = 'natural_pause'
        elif end - cursor <= hard_limit_ms:
            break
        else:
            cut, reason = cursor + hard_limit_ms, 'unsafe_continuous_speech'
        output.append((cursor, cut, reason))
        cursor = cut
    output.append((cursor, end, 'range_end'))
    return output
