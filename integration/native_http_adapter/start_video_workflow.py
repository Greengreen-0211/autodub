"""Launch one unified workbench. The user starts inference in the browser."""
import argparse
from datetime import datetime
import getpass
import importlib.metadata
import json
import os
from pathlib import Path
import secrets
import signal
import socket
import subprocess
import sys
import time
from urllib.request import build_opener, ProxyHandler, Request
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from task_service.native_trials import verify_sources
from task_service.server import read_json, digest
from task_service.native_video import inspect_video
from tools.start_asr_workbench import loopback_environment

VIDEO = Path('/data0/goldenseed/lqz77/integration/test/test.mp4')
VIDEO_SHA = 'e45106c9f5fa1678a75298932e6ab5ac1db48ef66a91025b037efbe18f3ad477'


def stop_record(path):
    record = read_json(path)
    if record.get('profile') != 'real_video_workflow': raise RuntimeError('只能停止本统一工作台。')
    expected = [
        [sys.executable, '-u', '-B', '-m', 'task_service.video_server', '--store', str(ROOT / 'temp/video_workflow'),
         '--registry', record['registry'], '--port', '8772'],
        [sys.executable, '-u', '-B', str(ROOT / 'frontend/app.py'), '--host', '127.0.0.1', '--port', '7868',
         '--api-url', 'http://127.0.0.1:8772/api/v1', '--video-workflow', '--initial-video', str(VIDEO)]]
    processes = record.get('processes')
    if not isinstance(processes, list) or len(processes) != 2: raise RuntimeError('停止记录无效。')
    for process, command in zip(processes, expected):
        if process['command'] != command: raise RuntimeError('记录的命令不属于此入口。')
        proc = Path('/proc') / str(process['pid'])
        if not proc.exists(): continue
        actual = proc.joinpath('cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
        if actual != command: raise RuntimeError('PID对应的命令已变化，不发送停止信号。')
        os.kill(process['pid'], signal.SIGTERM)
        print('已停止本入口进程：' + str(process['pid']), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stop-record', type=Path)
    args = parser.parse_args()
    if sys.platform != 'linux': raise SystemExit('只在服务器启动，不在Windows加载模型。')
    if args.stop_record:
        stop_record(args.stop_record)
        return
    verify_sources()
    if not VIDEO.is_file() or digest(VIDEO) != VIDEO_SHA: raise RuntimeError('预填视频与刚才核验的文件不同。')
    metadata = inspect_video(VIDEO)
    registry = ROOT / 'logs/tts_trials_preflight_20261012_231657_d1b3e0/report.json'
    report = read_json(registry)
    if report.get('status') != 'passed': raise RuntimeError('已有模型能力登记未通过。')
    log_root = ROOT / 'logs' / ('video_workflow_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid4().hex[:6])
    log_root.mkdir(parents=True)
    print('日志目录：' + str(log_root), flush=True)
    for port in (8772, 7868):
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try: probe.bind(('127.0.0.1', port))
            except OSError: raise RuntimeError(f'端口{port}已有监听者；不停止其他入口。') from None
    key = os.environ.get('DEEPSEEK_API_KEY') or getpass.getpass('请输入本次DeepSeek API Key（不回显，不写文件）：').strip()
    if not key: raise RuntimeError('未输入翻译凭证；没有启动服务。')
    env = loopback_environment({**os.environ, 'DEEPSEEK_API_KEY': key, 'AUTODUB_API_TOKEN': secrets.token_urlsafe(32),
        'PYTHONNOUSERSITE': '1', 'PYTHONIOENCODING': 'utf-8', 'PYTHONUNBUFFERED': '1', 'CUDA_VISIBLE_DEVICES': '0',
        'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2',
        'GRADIO_ANALYTICS_ENABLED': 'False', 'GRADIO_TEMP_DIR': str(ROOT / 'temp/gradio_video_workflow'),
        'AUTODUB_DIAR_BACKEND': 'wespeaker', 'AUTODUB_DIAR_ROUTING_MODE': os.environ.get('AUTODUB_DIAR_ROUTING_MODE', 'auto'), 'WESPEAKER_ROOT': os.environ.get('WESPEAKER_ROOT') or '/data0/goldenseed/chy/wespeaker_voxconverse_v2',
        'SIMAM_MODEL_PATH': os.environ.get('SIMAM_MODEL_PATH') or '/data0/goldenseed/models/wespeaker_simamresnet100/speaker-embedding.onnx',
        'EMOTION2VEC_MODEL_DIR': os.environ.get('EMOTION2VEC_MODEL_DIR') or '/data0/goldenseed/qgd/models/emotion2vec_plus_base',
        'AUTODUB_EMOTION_ENV': 'emotion2vec', 'AUTODUB_LIGHT_TTS_BACKEND': 'espeak',
        'AUTODUB_TTS_OFFLINE_MODE': 'strict_offline', 'AUTODUB_TTS_GPU': '0',
        'QWEN3_ASR_MODEL_DIR': '/data0/goldenseed/models/Qwen3-ASR-1.7B',
        'QWEN3_MAX_INFERENCE_BATCH_SIZE': '1', 'AUTODUB_TTS_CONTENT_ASR_ENABLED': '1',
        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'MODELSCOPE_OFFLINE': '1',
        'AUTODUB_TRANSLATION_RETRIES': '2', 'AUTODUB_TRANSLATION_BATCH_SIZE': '10',
        'AUTODUB_TRANSLATION_FAILURE_POLICY': 'best_available',
        'AUTODUB_OCR_REQUIRED': '0', 'DEEPSEEK_BASE_URL': 'https://api.deepseek.com', 'DEEPSEEK_MODEL': 'deepseek-chat'})
    env.pop('OPENAI_API_KEY', None)
    commands = [
        [sys.executable, '-u', '-B', '-m', 'task_service.video_server', '--store', str(ROOT / 'temp/video_workflow'), '--registry', str(registry), '--port', '8772'],
        [sys.executable, '-u', '-B', str(ROOT / 'frontend/app.py'), '--host', '127.0.0.1', '--port', '7868',
         '--api-url', 'http://127.0.0.1:8772/api/v1', '--video-workflow', '--initial-video', str(VIDEO)]]
    children = []
    opener = build_opener(ProxyHandler({})).open
    try:
        for command, name, url in zip(commands, ('api', 'ui'), ('http://127.0.0.1:8772/api/v1/health', 'http://127.0.0.1:7868/')):
            child_env = env.copy()
            if name == 'ui':
                child_env.pop('DEEPSEEK_API_KEY', None); child_env['CUDA_VISIBLE_DEVICES'] = ''
            with (log_root / (name + '.log')).open('xb') as log:
                child = subprocess.Popen(command, cwd=ROOT, env=child_env, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            children.append(child)
            headers = {'Authorization': 'Bearer ' + env['AUTODUB_API_TOKEN']} if name == 'api' else {}
            deadline = time.monotonic() + 60
            while True:
                if child.poll() is not None: raise RuntimeError(name + '启动失败，查看 ' + str(log_root / (name + '.log')))
                try:
                    with opener(Request(url, headers=headers), timeout=1) as response:
                        if response.status == 200: break
                except OSError: pass
                if time.monotonic() > deadline: raise RuntimeError(name + '启动超时。')
                time.sleep(.5)
        record = {'profile': 'real_video_workflow', 'ui_url': 'http://127.0.0.1:7868/',
            'backend_commit': json.loads((ROOT / ('task_service/BACKEND_AV_ROUTING_MANIFEST.json' if (ROOT / 'task_service/BACKEND_AV_ROUTING_MANIFEST.json').is_file() else 'task_service/BACKEND_BASELINE_20261011.json')).read_bytes())['backend_commit'],
            'api_url': 'http://127.0.0.1:8772/api/v1', 'initial_video': str(VIDEO), 'input_sha256': VIDEO_SHA,
            'duration': metadata['duration'], 'registry': str(registry), 'logs': str(log_root),
            'gradio_version': importlib.metadata.version('gradio'), 'models_loaded': False, 'task_created': False,
            'api_calls': False, 'verification': 'ready for user browser end-to-end run; not yet passed',
            'processes': [{'pid': c.pid, 'command': command} for c, command in zip(children, commands)]}
        (log_root / 'startup.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)
        print('启动成功。页面已预填test.mp4：创建新任务 → 开始执行 → 逐阶段确认 → 成品。', flush=True)
    except BaseException:
        for child in children:
            if child.poll() is None:
                child.terminate()
                try: child.wait(20)
                except subprocess.TimeoutExpired: print('本入口子进程尚未退出，请检查日志。', flush=True)
        raise
    finally:
        key = ''
        env.pop('DEEPSEEK_API_KEY', None)


if __name__ == '__main__': main()
