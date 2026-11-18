"""AV-first speaker routing with the original backend as the only fallback."""
from pathlib import Path
import contextlib
import copy
import os
import subprocess
import sys
import threading
import wave
from speaker_av_support import AV_COMMIT, assess_evidence, fingerprint, read_json, read_rttm, sha, write_json

VERSION = 'av-first-native-fallback-v1'
DOWNSTREAM = ('segments_step2', 'segments_step2_timing_refined', 'tts_timing_report', 'tts_voice_embedding_audit',
    'segments_emotion', 'segments_audio_analyzed', 'segments_step3', 'segments_scheme3', 'segments_json_path',
    'audio_clips', 'segments_translation_candidates', 'translation_candidates_fingerprint',
    'light_tts_selector_results', 'light_tts_selector_results_path', 'emotion_results', 'tts_router_results',
    'tts_backend', 'tts_stage_manifest', 'tts_routing_summary', 'tts_routing_report_path',
    'speaker_clustering', 'estimated_speaker_count', 'estimated_speaker_range', 'speaker_count_confidence',
    'dialogue_assigned_speaker_count')


class AVUnavailable(RuntimeError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


@contextlib.contextmanager
def temporary_environment(values):
    previous = {k: os.environ.get(k) for k in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for k, v in previous.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v


def settings():
    home = Path.home()
    return {'python': os.getenv('AUTODUB_3DSPEAKER_PYTHON', str(home / '.conda/envs/3dspeaker_av_lqz77_065629c3/bin/python')),
        'source': os.getenv('AUTODUB_3DSPEAKER_SOURCE', '/data0/goldenseed/lqz77/third_party/3D-Speaker_065629c313ea'),
        'models': os.getenv('AUTODUB_3DSPEAKER_MODELS', '/data0/goldenseed/models/3D-Speaker_AV_065629c313ea'),
        'commit': AV_COMMIT, 'gpu': '0', 'threads': 2, 'seed': 0,
        'timeout_seconds': max(1, int(os.getenv('AUTODUB_3DSPEAKER_TIMEOUT', '1800')))}


def make_request(state, config):
    video = state.get('input_file') or state.get('video_path') or os.getenv('AUTODUB_DIAR_VIDEO', '')
    if not video or not Path(video).is_file():
        raise AVUnavailable('video_context_missing', '没有可用的原视频路径；音频任务沿用原说话人方法。')
    vocals = Path(state['vocals_path'])
    if not vocals.is_file():
        raise AVUnavailable('audio_missing', '分离人声音轨不存在。')
    opts = settings()
    for path in (Path(opts['python']), Path(opts['source']) / 'AUTODUB_SOURCE_MANIFEST.json',
                 Path(opts['models']) / 'VERIFIED_MANIFEST.json'):
        if not path.is_file():
            raise AVUnavailable('av_resources_unavailable', '3D-Speaker现有环境或校验清单缺失：' + str(path))
    from speaker_av_resources import verify_resources
    try:
        verify_resources(opts)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        raise AVUnavailable('av_resource_integrity_failed', '3D-Speaker源码或权重校验失败：' + str(exc)) from exc
    with wave.open(str(vocals)) as f:
        duration = f.getnframes() / f.getframerate()
    request = {'version': VERSION, 'video': str(Path(video).resolve()), 'video_sha256': sha(video),
        'vocals': str(vocals.resolve()), 'vocals_sha256': sha(vocals), 'duration': duration,
        'settings': opts, 'source_manifest_sha256': sha(Path(opts['source']) / 'AUTODUB_SOURCE_MANIFEST.json'),
        'model_manifest_sha256': sha(Path(opts['models']) / 'VERIFIED_MANIFEST.json')}
    request['adapter_sha256'] = {name: sha(Path(__file__).with_name(name)) for name in
        ('speaker_av_support.py', 'speaker_av_runtime.py', 'speaker_av_gates.py', 'speaker_av_resources.py', 'speaker_av_worker.py')}
    request['fingerprint'] = fingerprint(request)
    return request


def worker_environment():
    env = os.environ.copy()
    for key in list(env):
        if any(x in key.upper() for x in ('API_KEY', 'PASSWORD', 'SECRET', 'TOKEN')) or key in {
                'PYTHONPATH', 'PYTHONHOME', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY',
                'http_proxy', 'https_proxy', 'all_proxy'}:
            env.pop(key, None)
    env.update(CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
        NUMBA_NUM_THREADS='2', PYTHONNOUSERSITE='1', PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', MODELSCOPE_OFFLINE='1')
    return env


def run_worker(request, root):
    root.mkdir(parents=True, exist_ok=True)
    result_path = root / 'result.json'
    if result_path.is_file():
        result = read_json(result_path)
        if result.get('fingerprint') == request['fingerprint']:
            if not assess_evidence(result.get('evidence'))[0]:
                return result, True
            path = Path(result.get('rttm', ''))
            if path.resolve().is_relative_to(root.resolve()) and path.is_file() and sha(path) == result.get('rttm_sha256'):
                return result, True
    plan = root / 'plan.json'
    write_json(plan, request)
    worker = Path(__file__).with_name('speaker_av_worker.py')
    with (root / 'run.log').open('a', encoding='utf-8') as log:
        # Inherit the caller's process group so HTTP cancellation also reaches
        # this model process. No GPU-bearing grandchildren are launched.
        child = subprocess.Popen([request['settings']['python'], '-I', '-u', '-B', str(worker), '--plan', str(plan)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', env=worker_environment())
        def stream():
            for line in child.stdout:
                log.write(line); log.flush(); print(line, end='', flush=True)
        reader = threading.Thread(target=stream, daemon=True)
        reader.start()
        try:
            code = child.wait(timeout=request['settings']['timeout_seconds'])
        except subprocess.TimeoutExpired:
            child.terminate()
            try: child.wait(timeout=15)
            except subprocess.TimeoutExpired: child.kill(); child.wait()
            raise AVUnavailable('av_timeout', '视听处理超时，已停止其模型进程。')
        finally:
            reader.join(timeout=10)
        if code != 0 or not result_path.is_file():
            raise AVUnavailable('av_worker_failed', '视听子进程失败；详情：' + str(root / 'run.log'))
    result = read_json(result_path)
    if result.get('fingerprint') != request['fingerprint']:
        raise AVUnavailable('av_result_mismatch', '视听结果指纹与本次输入不一致。')
    return result, False


def align_C(state, result, root):
    from run_step2_wespeaker import assign_speakers, reassemble_final_speaker_segments
    path = Path(result['rttm']).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file() or sha(path) != result.get('rttm_sha256'):
        raise AVUnavailable('invalid_av_rttm', '视听RTTM路径或校验不符合本任务。')
    spans = read_rttm(path, result.get('duration'))
    labels = sorted({x['speaker'] for x in spans})
    names = {x: 'SPEAKER_%02d' % i for i, x in enumerate(labels)}
    for x in spans: x['speaker'] = names[x['speaker']]
    rows = reassemble_final_speaker_segments(assign_speakers(copy.deepcopy(state['segments_step1']), spans))
    if not rows or len({str(x['id']) for x in rows}) != len(rows):
        raise AVUnavailable('invalid_av_alignment', '视听分句结果为空或句ID重复。')
    unresolved = []
    for x in rows:
        overlap = sum(max(0, min(float(x['end']), s['end']) - max(float(x['start']), s['start'])) for s in spans)
        if float(x['end']) <= float(x['start']) or not str(x.get('text', '')).strip():
            raise AVUnavailable('invalid_av_alignment', '视听分句包含无效文本或时间窗。')
        if overlap <= 1e-6 or x.get('speaker') not in set(names.values()):
            x['speaker'] = 'UNRESOLVED_' + str(x['id']); unresolved.append(x['id'])
        x.update(speaker_assignment='3dspeaker_av', speaker_cluster_source='3D-Speaker multimodal')
    return rows, {'label_map': names, 'unresolved_segment_ids': unresolved, 'C_identities': len(names)}


def run_routed(config, store, legacy, *, legacy_backend=None, worker_runner=None, aligner=None):
    """Both CLI and the standalone bridge call this same idempotent decision."""
    mode = os.getenv('AUTODUB_DIAR_ROUTING_MODE', 'auto').strip().lower()
    if mode not in {'auto', 'legacy'}:
        raise ValueError('AUTODUB_DIAR_ROUTING_MODE must be auto or legacy')
    if mode == 'legacy':
        value = legacy()
        completed = store.load() or {}
        if completed.get('segments_step2'):
            actual = legacy_backend or os.getenv('AUTODUB_DIAR_BACKEND', 'speechbrain')
            path = Path(config.temp_dir) / 'diarization_routing.json'
            audit = {'version': VERSION, 'mode': 'legacy', 'status': 'completed',
                'planned_backend': actual, 'selected_backend': actual, 'fallback_used': False, 'fallback_attempted': False,
                'audio_B_adopted': False, 'reason_code': 'explicit_legacy_mode',
                'reason': '按配置直接执行原说话人方法。', 'segments_fingerprint': fingerprint(completed['segments_step2'])}
            write_json(path, audit)
            store.save({'diarization_routing': audit, 'diarization_routing_path': str(path)})
        return value
    state = store.load() or {}
    if not state.get('segments_step1'):
        return legacy()
    fallback = legacy_backend or os.getenv('AUTODUB_DIAR_FALLBACK_BACKEND') or os.getenv('AUTODUB_DIAR_BACKEND', 'wespeaker')
    if fallback not in {'wespeaker', 'speechbrain'}:
        raise ValueError('Fallback must be the original wespeaker or speechbrain backend')
    audit_path = Path(config.temp_dir) / 'diarization_routing.json'
    audit = {'version': VERSION, 'mode': mode, 'planned_backend': '3dspeaker_av', 'fallback_backend': fallback,
             'status': 'running', 'selected_backend': None, 'fallback_used': False, 'fallback_attempted': False, 'audio_B_adopted': False}
    def publish(value):
        write_json(audit_path, value)
        store.save({'diarization_routing': value, 'diarization_routing_path': str(audit_path)})
    def reusable(input_key):
        prior = state.get('diarization_routing') or {}
        return prior.get('status') == 'completed' and prior.get('input_fingerprint') == input_key and bool(state.get('segments_step2')) and prior.get('segments_fingerprint') == fingerprint(state['segments_step2'])
    try:
        request = make_request(state, config)
        input_key = fingerprint({'request': request['fingerprint'], 'source': state['segments_step1'], 'fallback': fallback})
        prior = state.get('diarization_routing') or {}
        if reusable(input_key):
            print('[说话人路由] 复用本输入已完成的 %s 归属。' % prior['selected_backend'], flush=True)
            return
        audit['input_fingerprint'] = input_key
        root = Path(config.temp_dir) / 'diarization_av' / request['fingerprint'][:20]
        print('[说话人路由] 优先尝试3D-Speaker视听；不足时回退 %s。' % fallback, flush=True)
        result, reused = (worker_runner or run_worker)(request, root)
        audit.update(evidence=result.get('evidence'), av_cache_reused=reused, av_log=str(root / 'run.log'))
        if sha(request['video']) != request['video_sha256'] or sha(request['vocals']) != request['vocals_sha256']:
            raise AVUnavailable('input_changed', '视听处理期间输入文件发生变化。')
        usable, reason, message = assess_evidence(result.get('evidence'))
        if not usable:
            raise AVUnavailable(reason, message)
        rows, alignment = (aligner or align_C)(state, result, root)
        store.delete_keys(DOWNSTREAM)
        store.save({'segments_step2': rows, 'speaker_clustering': {'algorithm': '3dspeaker_av',
            'speaker_count': alignment['C_identities'], 'routing_reason': reason},
            'estimated_speaker_count': alignment['C_identities'], 'dialogue_assigned_speaker_count': len({x['speaker'] for x in rows})})
        audit.update(status='completed', selected_backend='3dspeaker_av', reason_code=reason, reason=message,
                     alignment=alignment, segments_fingerprint=fingerprint(rows))
        publish(audit)
        print('[说话人路由] 采用3D-Speaker视听结果；%s' % message, flush=True)
        return
    except (AVUnavailable, OSError, ValueError, KeyError, TypeError, RuntimeError, wave.Error, subprocess.SubprocessError) as exc:
        audit.update(reason_code=getattr(exc, 'code', 'av_processing_error'), reason=str(exc))
        # Changes during inference are integrity errors, not weak evidence: do
        # not let either backend publish results bound to an altered waveform.
        if audit['reason_code'] == 'input_changed':
            store.delete_keys(DOWNSTREAM)
            audit['status'] = 'failed'; publish(audit); raise
        if 'input_fingerprint' not in audit:
            # Missing video/resources can be a stable, legitimate audio-only
            # request. Cache the successful original method too, so a bridge
            # followed by the main CLI does not run WeSpeaker twice.
            def file_identity(value):
                path = Path(value) if value else None
                return {'path': str(path), 'sha256': sha(path)} if path and path.is_file() else {'path': str(value or ''), 'sha256': None}
            opts = settings()
            audit['input_fingerprint'] = fingerprint({'version': VERSION, 'fallback': fallback,
                'source': state['segments_step1'], 'vocals': file_identity(state.get('vocals_path')),
                'video': file_identity(state.get('input_file') or state.get('video_path') or os.getenv('AUTODUB_DIAR_VIDEO')),
                'settings': opts, 'source_manifest': file_identity(Path(opts['source']) / 'AUTODUB_SOURCE_MANIFEST.json'),
                'model_manifest': file_identity(Path(opts['models']) / 'VERIFIED_MANIFEST.json')})
            if reusable(audit['input_fingerprint']):
                print('[说话人路由] 复用本输入已完成的原方法回退结果。', flush=True)
                return
        print('[说话人路由] %s 回退原 %s；不采用3D-Speaker纯音频B。' % (str(exc), fallback), flush=True)
    store.delete_keys(DOWNSTREAM)
    audit['fallback_attempted'] = True
    try:
        with temporary_environment({'AUTODUB_DIAR_ROUTING_MODE': 'legacy', 'AUTODUB_DIAR_BACKEND': fallback,
                                    'AUTODUB_DIAR_FORCE_REBUILD': '1'}):
            legacy()
        completed = store.load() or {}
        if not completed.get('segments_step2'):
            raise RuntimeError('Original fallback returned no segments_step2')
        audit.update(status='completed', selected_backend=fallback, fallback_used=True, segments_fingerprint=fingerprint(completed['segments_step2']))
        publish(audit)
        print('[说话人路由] 已采用原 %s，回退原因：%s。' % (fallback, audit['reason_code']), flush=True)
    except Exception as exc:
        audit.update(status='failed', selected_backend=None, fallback_error=str(exc))
        publish(audit)
        raise


class BridgeStore:
    def __init__(self, path): self.path = Path(path)
    def load(self): return read_json(self.path) if self.path.is_file() else None
    def save(self, value):
        state = self.load() or {}; state.update(value); write_json(self.path, state)
    def delete_keys(self, keys):
        state = self.load() or {}
        for key in keys: state.pop(key, None)
        write_json(self.path, state)


def run_bridge(temp_dir, legacy):
    from types import SimpleNamespace
    return run_routed(SimpleNamespace(temp_dir=str(temp_dir)), BridgeStore(Path(temp_dir) / 'project_state.json'),
                      legacy, legacy_backend='wespeaker')
