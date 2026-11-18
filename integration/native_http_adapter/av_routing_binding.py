"""Strict identity bridge for an explicitly installed AV feature-branch snapshot."""
from pathlib import Path
import hashlib
import json
import re

REQUIRED = {'auto_dubbing_ver_2.0.py', 'run_step2_wespeaker.py', 'speaker_alignment.py', 'diarization_router.py',
    'speaker_av_worker.py', 'speaker_av_runtime.py', 'speaker_av_support.py', 'speaker_av_gates.py',
    'speaker_av_resources.py', 'tts_router.py', 'mix_timing.py', 'autodub_ocr_runtime/FROZEN_SOURCES.json',
    'task_service/native_audio.py', 'task_service/native_trials.py', 'task_service/av_routing_binding.py',
    'tools/start_video_workflow.py'}


def verified_baseline(root, asr_hash=None):
    root = Path(root).resolve()
    marker = root / 'task_service/BACKEND_AV_ROUTING_MANIFEST.json'
    if not marker.is_file():
        return None
    record = json.loads(marker.read_text(encoding='utf-8'))
    if record.get('schema') != 'backend-av-routing-binding-v1' or record.get('base_commit') != 'f0577d9c7252bb22ae24494d5c9791d517aaba4e':
        raise RuntimeError('Unrecognized AV branch source binding.')
    if not re.fullmatch(r'[0-9a-f]{40}', str(record.get('backend_commit', ''))):
        raise RuntimeError('Missing reviewed Git commit for AV branch.')
    files = record.get('files') or []
    if len(files) < 41 or len({x['path'] for x in files}) != len(files):
        raise RuntimeError('Incomplete or duplicate AV branch file manifest.')
    if not REQUIRED <= {x['path'] for x in files}:
        raise RuntimeError('Required AV runtime/HTTP modules are missing from the binding.')
    entries = {}
    for item in files:
        path = (root / item['path']).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise RuntimeError('Invalid branch source path: ' + item['path'])
        actual = hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        if actual != item['sha256']:
            raise RuntimeError('Reviewed AV source was modified: ' + item['path'])
        entries[item['path']] = actual
    entry = root / 'auto_dubbing_ver_2.0.py'
    accepted = {entries.get('auto_dubbing_ver_2.0.py'), hashlib.sha256(entry.read_bytes()).hexdigest()}
    if asr_hash is not None and asr_hash not in accepted:
        raise RuntimeError('Create a new task with this reviewed backend; old ASR snapshots were not rebound silently.')
    return dict(record, routing_branch_binding=True)
