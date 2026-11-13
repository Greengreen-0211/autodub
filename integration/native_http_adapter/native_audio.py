"""Server-only adapter to the pinned upstream WeSpeaker/emotion pipeline."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading

from task_service.server import digest, read_json

ROOT = Path(__file__).resolve().parents[1]
SB_FILES = ['hyperparams.yaml', 'embedding_model.ckpt', 'mean_var_norm_emb.ckpt', 'classifier.ckpt', 'label_encoder.txt']


def wespeaker_probe_source():
    """Check the bridge environment separately from the main pipeline environment."""
    return '''import importlib.util,json,sys
names=["numpy","kaldiio","onnxruntime","torch","torchaudio","soundfile","sklearn","umap","hdbscan"]
missing=[m for m in names if not importlib.util.find_spec(m)]
providers=[]
provider_error=None
if "onnxruntime" not in missing:
 try:
  import onnxruntime
  providers=onnxruntime.get_available_providers()
 except Exception as exc:
  provider_error=str(exc)
print(json.dumps({"python":sys.executable,"missing":missing,"onnx_providers":providers,"onnx_error":provider_error}))
sys.exit(1 if missing or provider_error or "CUDAExecutionProvider" not in providers else 0)
'''


def verify_backend_baseline(asr_hash=None, *, root=ROOT):
    """An explicit reviewed source upgrade, rather than requiring old ASR code forever."""
    from task_service.av_routing_binding import verified_baseline
    reviewed = verified_baseline(root, asr_hash)
    if reviewed is not None:
        return reviewed
    baseline = read_json(Path(root) / 'task_service/BACKEND_BASELINE_20261011.json')
    if baseline.get('backend_commit') != '0b94dcd5059bf96afed3da0b2beea809234458b9':
        raise RuntimeError('后端基线不是本次核对的 ASR 恢复/TTS 音质版本。')
    import hashlib
    for item in baseline['files']:
        path = Path(root) / item['path']
        if hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest() != item['sha256']:
            raise RuntimeError('后端文件与已核对版本不同: ' + item['path'])
    entry = next(item for item in baseline['files'] if item['path'] == 'auto_dubbing_ver_2.0.py')
    accepted = {entry['previous_sha256'], entry['previous_sha256_lf'], entry['sha256']}
    accepted.update(entry.get('accepted_previous_sha256', []))
    accepted.add(digest(Path(root) / entry['path']))
    if asr_hash is not None and asr_hash not in accepted:
        raise RuntimeError('ASR 结果来自未经核对的后端版本，停止模型处理。')
    return baseline


def preflight(*, emotion=True):
    if sys.platform != 'linux':
        raise RuntimeError('真实模型只在 Linux 服务器运行。')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise RuntimeError('本次仅使用物理 GPU 0，请设置 CUDA_VISIBLE_DEVICES=0。')
    backend = os.environ.get('AUTODUB_DIAR_BACKEND', 'wespeaker').lower()
    if backend not in {'wespeaker', 'speechbrain'}:
        raise RuntimeError('未知说话人后端；不会自动更换模型。')
    baseline = verify_backend_baseline()
    modules = ('torch', 'pydub') if backend == 'wespeaker' else ('torch', 'torchaudio', 'speechbrain', 'sklearn', 'soundfile', 'pydub')
    missing = [m for m in modules if not importlib.util.find_spec(m)]
    if missing:
        raise RuntimeError('当前环境缺少模块: ' + ', '.join(missing))
    sb = Path(os.environ.get('SB_ECAPA_MODEL_DIR', '/data0/goldenseed/models/sb_ecapa')).resolve()
    if backend == 'speechbrain' and any(not (sb / name).is_file() for name in SB_FILES):
        raise RuntimeError('SpeechBrain 模型不完整: ' + str(sb))
    gpu = subprocess.run(['nvidia-smi', '--id=0', '--query-gpu=index,name,memory.free', '--format=csv,noheader,nounits'],
                         check=True, capture_output=True, text=True, timeout=15).stdout.strip()
    if int(gpu.rsplit(',', 1)[1].strip()) < 4096:
        raise RuntimeError('GPU 0 空闲显存不足 4 GiB，请稍后再试。')
    if shutil.disk_usage(ROOT).free < 512 * 1024 * 1024:
        raise RuntimeError('剩余磁盘不足 512 MiB。')
    report = {'gpu': gpu, 'backend_commit': baseline['backend_commit'], 'speaker_backend': backend}
    if backend == 'speechbrain':
        report.update(speaker_model_dir=str(sb), speaker_sha256={name: digest(sb / name) for name in SB_FILES},
                      speaker_device='cpu (unchanged backend default)')
    else:
        code = Path(os.environ.get('WESPEAKER_ROOT', '/data0/goldenseed/chy/wespeaker_voxconverse_v2')).resolve()
        checkpoint = Path(os.environ.get('SIMAM_MODEL_PATH', '/data0/goldenseed/models/wespeaker_simamresnet100/speaker-embedding.onnx')).resolve()
        example = code / 'wespeaker/examples/voxconverse/v2'
        sources = [example / 'wespeaker/diar/extract_emb.py', example / 'wespeaker/diar/make_rttm.py', ROOT / 'make_fbank_direct.py']
        if not checkpoint.is_file() or checkpoint.stat().st_size == 0 or any(not p.is_file() for p in sources):
            raise RuntimeError(f'WeSpeaker 代码或权重未就绪: code={code}, checkpoint={checkpoint}。请返回资源检查结果，不需要他人密码。')
        environment = os.environ.get('WESPEAKER_ENV_NAME', 'wespeaker_diar')
        command = wespeaker_command('-c', wespeaker_probe_source())
        probe = subprocess.run(command, capture_output=True, text=True, timeout=45)
        if probe.returncode:
            raise RuntimeError(f'WeSpeaker 环境 {environment} 未就绪: {probe.stdout[-500:]} {probe.stderr[-500:]}。不会创建或修改环境。')
        report.update(wespeaker_root=str(code), simam_checkpoint=str(checkpoint),
                      simam_sha256=digest(checkpoint), wespeaker_environment=environment, wespeaker_probe=probe.stdout.strip(),
                      wespeaker_code_sha256={str(p.relative_to(code)) if p.is_relative_to(code) else p.name: digest(p) for p in sources},
                      speaker_device='cuda:0', upstream_batch_size=96)
    if emotion:
        candidates = [os.environ.get('EMOTION2VEC_MODEL_DIR', ''), '/data0/goldenseed/qgd/models/emotion2vec_plus_base',
                      '/data0/goldenseed/models/emotion2vec_plus_base']
        configured = os.environ.get('EMOTION2VEC_MODEL_DIR')
        model = next((Path(p).resolve() for p in (candidates[:1] if configured else candidates) if p and Path(p).is_dir()
                      and any(Path(p).glob('*.pt'))), None)
        if model is None:
            raise RuntimeError('未找到 emotion2vec_plus_base 本地权重。请提供 EMOTION2VEC_MODEL_DIR；不会下载模型。')
        conda = shutil.which('conda') or '/opt/anaconda3/bin/conda'
        env_name = os.environ.get('AUTODUB_EMOTION_ENV', 'emotion2vec')
        command = [conda, 'run', '--no-capture-output', '-n', env_name, 'python', '-c',
                   'import importlib.util,json,sys; found=bool(importlib.util.find_spec("funasr")); print(json.dumps({"funasr":found,"python":sys.executable})); sys.exit(0 if found else 1)']
        probe = subprocess.run(command, capture_output=True, text=True, timeout=45)
        if probe.returncode:
            raise RuntimeError('独立情绪环境未就绪: ' + env_name + '; ' + probe.stderr[-500:])
        os.environ['EMOTION2VEC_MODEL_DIR'] = str(model)
        report.update(emotion_model_dir=str(model), emotion_environment=env_name, emotion_probe=probe.stdout.strip(),
                      emotion_sha256={p.name: digest(p) for p in model.iterdir() if p.is_file() and p.suffix in {'.pt', '.json', '.yaml'}})
    return report


def wespeaker_command(*arguments):
    conda = shutil.which('conda') or '/opt/anaconda3/bin/conda'
    return [conda, 'run', '--no-capture-output', '-n', os.environ.get('WESPEAKER_ENV_NAME', 'wespeaker_diar'), 'python', *arguments]


def prepare_source_rows(rows):
    """Do not apply old word/text alignment to manually changed transcripts."""
    import copy
    prepared = copy.deepcopy(rows)
    for row in prepared:
        if row.get('word_alignment_status') == 'stale':
            row['qwen3_time_stamps'] = []
            row['qwen3_sentence_candidates'] = []
    return prepared


class NativeRunner:
    def __init__(self, root):
        self.root = Path(root)
        self.lock = threading.Lock()
        self.process = None
        self.closed = False
        self.speaker_backend = os.environ.get('AUTODUB_DIAR_BACKEND', 'wespeaker').lower()

    def __call__(self, stage, job):
        work = self.root / job['job_id'] / job['active_operation'] / stage
        work.mkdir(parents=True, exist_ok=False)
        plan = {key: job[key] for key in ('_audio', '_audio_sha256', '_backend_entry_sha256', '_native', '_raw', 'revision')}
        context = job.get('_state') or {}
        if stage == 'diarization' and context.get('input_file'):
            video = Path(context['input_file']).resolve(strict=True)
            plan.update(_diarization_video=str(video), _diarization_video_sha256=digest(video),
                        _diarization_duration=job.get('duration'))
        (work / 'plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding='utf-8')
        environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': '0', 'PYTHONNOUSERSITE': '1',
            'PYTHONIOENCODING': 'utf-8', 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
            'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'MODELSCOPE_OFFLINE': '1',
            'AUTODUB_DIAR_BACKEND': self.speaker_backend, 'AUTODUB_EMOTION_DEVICE': 'cuda:0',
            'SB_ECAPA_MODEL_DIR': os.environ.get('SB_ECAPA_MODEL_DIR', '/data0/goldenseed/models/sb_ecapa')}
        for key in ('DEEPSEEK_API_KEY', 'OPENAI_API_KEY'):
            environment.pop(key, None)
        with (work / 'run.log').open('xb') as log:
            with self.lock:
                if self.closed:
                    raise RuntimeError('模型执行器已关闭。')
                self.process = subprocess.Popen([sys.executable, '-u', '-B', '-m', 'task_service.native_audio',
                    'phase', '--stage', stage, '--work', str(work)], cwd=ROOT, env=environment,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                process = self.process
            code = process.wait()
            with self.lock:
                self.process = None
        if code:
            raise RuntimeError(f'{stage} 真实模型执行失败（退出码 {code}）；日志: {work / "run.log"}')
        return read_json(work / 'result.json')

    def close(self):
        with self.lock:
            self.closed = True
            process = self.process
            if process and process.poll() is None:
                # Kill only this adapter's own session (includes the isolated emotion child).
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


def phase(stage, work):
    import fcntl
    lock_path = ROOT / 'temp/audio_review_gpu.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('另一个音频审核服务正在运行模型，请稍后重试。')
        run_phase(stage, work)


def run_phase(stage, work):
    work = work.resolve(strict=True)
    plan = read_json(work / 'plan.json')
    audio = Path(plan['_audio']).resolve(strict=True)
    if digest(audio) != plan['_audio_sha256']:
        raise RuntimeError('ASR 输入 WAV 已变化，拒绝模型处理。')
    report = preflight(emotion=stage == 'emotion')
    entry = ROOT / 'auto_dubbing_ver_2.0.py'
    report['backend_sha256'] = digest(entry)
    if not plan.get('_backend_entry_sha256'):
        raise RuntimeError('ASR 运行缺少后端源码身份记录。')
    verify_backend_baseline(plan['_backend_entry_sha256'])
    report['asr_backend_sha256'] = plan['_backend_entry_sha256']
    (work / 'preflight.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    spec = importlib.util.spec_from_file_location('autodub_native_audio', entry)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    config = module.DubbingConfig()
    config.temp_dir = str(work / 'native')
    config.output_dir = str(work / 'output')
    if report['speaker_backend'] == 'speechbrain':
        config.speaker_model_dir = report['speaker_model_dir']
    if stage == 'emotion':
        config.emotion_model_dir = report['emotion_model_dir']
    if stage == 'diarization' and report['speaker_backend'] == 'wespeaker':
        config.temp_dir = str(work / 'bridge/temp/input')
        os.environ['AUTODUB_WORKDIR'] = str(work / 'bridge')
        os.environ['AUTODUB_VIDEO'] = 'input.wav'
    Path(config.temp_dir).mkdir(parents=True)
    state = {'vocals_path': str(audio), 'source_language': 'Chinese',
             'segments_qwen3_raw': plan['_raw'], 'segments_step1': prepare_source_rows(plan['_native'])}
    if plan.get('_diarization_video'):
        video = Path(plan['_diarization_video']).resolve(strict=True)
        if digest(video) != plan['_diarization_video_sha256']:
            raise RuntimeError('Original video changed before diarization.')
        state.update(input_file=str(video), duration=plan.get('_diarization_duration'))
    if stage == 'emotion':
        state['segments_step2'] = plan['_native']
    module.StateManager(config).save(state)
    if stage == 'diarization':
        if report['speaker_backend'] == 'wespeaker':
            # Only the isolated environment loads WeSpeaker/ONNX. Existing source
            # generates the RTTM; native step 2 attaches PR7 global identities.
            subprocess.run(wespeaker_command(str(ROOT / 'run_step2_wespeaker.py')),
                           cwd=ROOT, env=os.environ.copy(), check=True)
        module.run_step_2(config)
    else:
        module.run_step_emotion(config, force=True)
    result = module.StateManager(config).load()
    key = 'segments_step2' if stage == 'diarization' else 'segments_emotion'
    if not result.get(key):
        raise RuntimeError('后端未产生当前阶段结果。')
    if digest(audio) != plan['_audio_sha256']:
        raise RuntimeError('模型执行期间输入文件变化，拒绝发布。')
    metadata = {name: result[name] for name in ('speaker_clustering', 'estimated_speaker_count',
                'estimated_speaker_range', 'speaker_count_confidence', 'diarization_routing',
                'diarization_routing_path', 'dialogue_assigned_speaker_count') if name in result}
    metadata['speaker_backend'] = (result.get('diarization_routing') or {}).get('selected_backend') or report['speaker_backend']
    (work / 'result.json').write_text(json.dumps({'segments': result[key], 'revision': plan['revision'], 'metadata': metadata,
        'backend_sha256': report['backend_sha256'], 'input_unchanged': True,
        'audio_input': 'separated_vocals' if plan.get('_diarization_video') else 'unseparated', 'separation_executed': False}, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['preflight', 'phase'])
    parser.add_argument('--stage', choices=['diarization', 'emotion'])
    parser.add_argument('--work', type=Path)
    args = parser.parse_args()
    if args.command == 'preflight':
        print(json.dumps(preflight(), ensure_ascii=False, indent=2), flush=True)
    else:
        if not args.stage or not args.work:
            parser.error('phase 需要 --stage 和 --work')
        phase(args.stage, args.work)


if __name__ == '__main__':
    main()
