"""Explicit one-model worker, original reference and duration policy, server only."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import threading
import time

from task_service.server import digest, read_json
from tools.probe_formal_tts import HASHES
from tools.try_formal_tts import hash_resources

ROOT = Path(__file__).resolve().parents[1]


def verify_sources():
    from task_service.native_audio import verify_backend_baseline
    reviewed = verify_backend_baseline()
    if reviewed.get('routing_branch_binding') is True:
        return  # The reviewed branch manifest already checks every runtime file.

    baseline = read_json(ROOT / 'task_service/BACKEND_BASELINE_20261011.json')
    hashes = {entry['path']: entry['sha256'] for entry in baseline['runtime_files']}
    for name, sha in hashes.items():
        if hashlib.sha256((ROOT / name).read_bytes().replace(b'\r\n', b'\n')).hexdigest() != sha:
            raise RuntimeError('冻结后端文件已变化: ' + name)


def expected_environment(model, report, baseline=None):
    """Legacy trial reports list only the three newly probed models."""
    matches = [e for e in report.get('models', []) if e.get('model') == model]
    if not matches and model in {'f5tts', 'confucius4'}:
        baseline = baseline if baseline is not None else read_json(ROOT / 'logs/formal_tts_preflight_20261012_215421_8528ac/report.json')
        if baseline.get('status') != 'passed':
            raise RuntimeError('原正式模型的环境预检未通过。')
        matches = [e for e in baseline.get('models', []) if e.get('model') == model and e.get('environment_ok') is True]
    if len(matches) != 1 or not isinstance(matches[0].get('environment'), str) or not matches[0]['environment'].strip():
        raise RuntimeError(f'{model} 缺少唯一的已核验环境记录，请重新预检。')
    return matches[0]['environment']


def resource_manifest(model, destination):
    records = hash_resources([model], destination)
    # These local shards are consumed by the worker but not enumerated in the
    # frozen preflight's minimum existence list. Record them without changing it.
    extra = []
    if model == 'omnivoice':
        directory = Path(os.getenv('OMNIVOICE_MODEL_DIR', '/data0/goldenseed/qgd/models/OmniVoice'))
        for pattern in ('*.safetensors', '*.bin', '*.index.json'):
            extra.extend(directory.glob(pattern))
        if not any(p.suffix in {'.safetensors', '.bin'} for p in extra):
            raise RuntimeError('OmniVoice 本地权重缺失。')
    elif model == 'cosyvoice3':
        repo = Path(os.getenv('COSYVOICE_REPO', '/data0/goldenseed/qgd/CosyVoice'))
        directory = Path(os.getenv('COSYVOICE_MODEL_DIR', str(repo / 'pretrained_models/Fun-CosyVoice3-0.5B'))) / 'CosyVoice-BlankEN'
        for pattern in ('*.safetensors', '*.bin', '*.index.json', '*token*.json'):
            extra.extend(directory.glob(pattern))
    known = {r['resolved_path'] for r in records}
    for path in sorted(extra):
        resolved = path.resolve(strict=True)
        if str(resolved) in known:
            continue
        info = resolved.stat()
        sha = digest(resolved)
        if (info.st_size, info.st_mtime_ns) != (resolved.stat().st_size, resolved.stat().st_mtime_ns):
            raise RuntimeError('校验期间模型文件发生变化。')
        records.append({'model': model, 'source': str(path), 'path': str(path), 'resolved_path': str(resolved),
            'bytes': info.st_size, 'mtime_ns': info.st_mtime_ns, 'sha256': sha})
        known.add(str(resolved))
    destination.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    return records


def gpu_preflight(directory):
    if shutil.disk_usage(directory).free < 1024 ** 3:
        raise RuntimeError('结果盘剩余空间不足 1 GiB，未加载模型。')
    probe = subprocess.run(['nvidia-smi', '-i', '0', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
        capture_output=True, text=True, timeout=15)
    if probe.returncode or int(probe.stdout.strip()) < 12288:
        raise RuntimeError('GPU0 可用显存不足 12 GiB，未加载模型。')


class NativeTrialRunner:
    def __init__(self, report):
        self.report = report
        self.closed = threading.Event()
        self.lock = threading.Lock()
        self.child = None

    def close(self):
        self.closed.set()
        self.stop_child()

    def stop_child(self):
        with self.lock:
            child = self.child
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                return
            try:
                child.wait(15)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait(15)

    def __call__(self, plan, directory):
        if self.closed.is_set():
            raise RuntimeError("服务正在关闭。")
        import fcntl
        from tts_router import MODEL_ENVS, _worker_environment, _request_fingerprint, _apply_atempo, _wav_duration
        from tts_quality import duration_quality
        if self.closed.is_set():
            raise RuntimeError('服务正在关闭。')
        verify_sources()
        model = plan['model']
        entry = next((e for e in self.report['registry'] if e['engine'] == model and e['available']), None)
        if entry is None or plan['target_language'] != 'en' or _request_fingerprint(plan) != plan['request_fingerprint']:
            raise RuntimeError('模型能力或输入指纹已变化。')
        expected = expected_environment(model, self.report)
        environment = os.getenv(f'AUTODUB_{model.upper()}_ENV', MODEL_ENVS[model])
        if environment != expected:
            raise RuntimeError('独立环境名称已变化，请重新预检。')
        directory = directory.resolve()
        directory.mkdir(parents=True, exist_ok=False)
        with (ROOT / 'temp/audio_review_gpu.lock').open('a') as gpu_lock:
            try:
                fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError('已有本项目模型操作占用 GPU 锁，请稍后单句重试。')
            gpu_preflight(directory)
            manifest = resource_manifest(model, directory / 'model_files.json')
            gpu_preflight(directory)
            if self.closed.is_set():
                raise RuntimeError('服务正在关闭。')
            plan_path, result_path = directory / 'plan.json', directory / 'worker_result.json'
            plan_path.write_text(json.dumps({'items': [plan]}, ensure_ascii=False, indent=2), encoding='utf-8')
            command = [shutil.which('conda') or '/opt/anaconda3/bin/conda', 'run', '--no-capture-output',
                '-n', environment, 'python', '-u', '-B', str(ROOT / 'tts_worker.py'), '--model', model,
                '--plan', str(plan_path), '--result', str(result_path), '--force']
            env = _worker_environment(model)
            for key in ('DEEPSEEK_API_KEY', 'OPENAI_API_KEY'):
                env.pop(key, None)
            env.update(CUDA_VISIBLE_DEVICES='0', AUTODUB_TTS_GPU='0', AUTODUB_TTS_OFFLINE_MODE='strict_offline',
                OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONUNBUFFERED='1', PYTHONNOUSERSITE='1')
            log_path = directory / 'worker.log'
            with log_path.open('xb') as log, self.lock:
                if self.closed.is_set():
                    raise RuntimeError('服务正在关闭。')
                self.child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(gpu_lock.fileno(),))
                child = self.child
            (directory / 'worker.json').write_text(json.dumps({'pid': child.pid, 'command': command}), encoding='utf-8')
            print(f'试合成 {model} 单句；日志: {log_path}', flush=True)
            deadline = time.monotonic() + 600
            try:
                while child.poll() is None:
                    if self.closed.is_set() or time.monotonic() > deadline:
                        self.stop_child()
                        raise RuntimeError('试合成中断或超过 600 秒；原配音保留。')
                    time.sleep(.25)
                if child.returncode != 0:
                    raise RuntimeError(f'{model} worker 退出码 {child.returncode}，日志: {log_path}')
            finally:
                with self.lock:
                    self.child = None
            data = read_json(result_path)
            result = data.get(plan['segment_id'], {}) if isinstance(data, dict) else next((r for r in data if str(r['segment_id']) == plan['segment_id']), {})
            if result.get('status') != 'success' or result.get('request_fingerprint') != plan['request_fingerprint']:
                raise RuntimeError(f'{model} 未生成有效结果: {result.get("error", "缺少结果")}')
            raw, final = Path(plan['raw_path']), Path(plan['final_path'])
            for path in (raw, final):
                if not path.resolve().is_relative_to(directory) or path.is_symlink():
                    raise RuntimeError('输出路径越界。')
            _apply_atempo(raw, final, plan['target_duration'])
            duration = _wav_duration(final)
            quality = duration_quality(duration, plan['target_duration'])
            if quality['duration_quality'] != 'passed':
                raise RuntimeError('实际合成时长未通过原门禁；原配音保留。')
            if _request_fingerprint(plan) != plan['request_fingerprint']:
                raise RuntimeError('合成期间参考音频或模型配置已变化。')
            for record in manifest:
                stat = Path(record['path']).stat()
                if Path(record['path']).resolve() != Path(record['resolved_path']) or (stat.st_size, stat.st_mtime_ns) != (record['bytes'], record['mtime_ns']):
                    raise RuntimeError('合成期间模型文件发生变化。')
            result.update(segment_id=plan['segment_id'], status='success', file=str(final), audio_sha256=digest(final),
                target_duration=plan['target_duration'], actual_duration=duration, **quality,
                actual_engine=model, planned_engine=model, tts_text=plan['text'], planned_text=plan['text'],
                fallback_count=0, fallback_reason='', model_files=manifest,
                generation_request={k: plan[k] for k in ('text', 'language', 'target_language', 'emotion', 'speaker', 'ref_audio', 'ref_text', 'target_duration', 'model')})
            (directory / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
            return result
