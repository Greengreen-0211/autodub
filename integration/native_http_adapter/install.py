"""Apply the reviewed delivery to the existing HTTP service, with a source backup.

Default is read-only preflight. The source manifest is emitted with the delivery
after the feature commit is published; it binds every installed runtime file.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import socket
import time
import uuid
import zipfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def digest(path): return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def prepare(root, manifest_path, source_root=REPO):
    root, source_root = Path(root).resolve(), Path(source_root).resolve()
    manifest = read(manifest_path)
    if manifest.get('schema') != 'backend-av-routing-binding-v1' or manifest.get('base_commit') != 'f0577d9c7252bb22ae24494d5c9791d517aaba4e':
        raise RuntimeError('Unexpected delivery manifest.')
    if not re.fullmatch(r'[0-9a-f]{40}', str(manifest.get('backend_commit', ''))):
        raise RuntimeError('Delivery is not bound to an actual Git commit.')
    if not (root / 'task_service/native_audio.py').is_file():
        raise RuntimeError('Target is not the existing HTTP runtime.')
    original = read(HERE / 'BASE.json')
    allowed = {row['path']: {row['sha256']} for row in original['files']}
    for record in (HERE / 'ADAPTER_BASE.json', root / 'task_service/BACKEND_BASELINE_20261011.json', root / 'task_service/BACKEND_AV_ROUTING_MANIFEST.json'):
        if not record.is_file(): continue
        data = read(record)
        for item in data.get('files', []) + data.get('runtime_files', []):
            values = {v for k, v in item.items() if 'sha256' in k and isinstance(v, str)}
            values.update(item.get('accepted_previous_sha256', []))
            allowed.setdefault(item['path'], set()).update(values)
    changes = []
    mandatory = set(allowed) & {row['path'] for row in original['files']}
    mandatory.update({'diarization_router.py', 'speaker_av_support.py', 'speaker_av_runtime.py', 'speaker_av_worker.py',
        'speaker_av_resources.py', 'speaker_av_gates.py', 'task_service/native_audio.py', 'task_service/native_trials.py',
        'task_service/av_routing_binding.py', 'tools/start_video_workflow.py'})
    if not mandatory <= {row['path'] for row in manifest['files']}:
        raise RuntimeError('Required runtime/HTTP source is missing from the delivery.')
    for row in manifest['files']:
        source, target = (source_root / row.get('source', row['path'])).resolve(), (root / row['path']).resolve()
        if not source.is_relative_to(source_root) or not target.is_relative_to(root) or digest(source) != row['sha256']:
            raise RuntimeError('Source checksum/path differs: ' + row['path'])
        if target.is_file() and digest(target) not in allowed.get(row['path'], set()) | {row['sha256']}:
            raise RuntimeError('Existing file has unreviewed edits; untouched: ' + row['path'])
        changes.append((row['path'], source.read_bytes()))
    if len(changes) < 41 or len({x[0] for x in changes}) != len(changes):
        raise RuntimeError('Incomplete or duplicate delivery manifest.')
    return root, manifest, changes


def apply(root, manifest, changes):
    for port in (7868, 8772):
        with socket.socket() as sock:
            sock.settimeout(.5)
            if sock.connect_ex(('127.0.0.1', port)) == 0:
                raise RuntimeError('Stop the recorded workbench/API processes before applying; port %d is active.' % port)
    marker = 'task_service/BACKEND_AV_ROUTING_MANIFEST.json'
    payload = dict(manifest)
    payload['files'] = [{k: x[k] for k in ('path', 'sha256')} for x in manifest['files']]
    changes = changes + [(marker, (json.dumps(payload, ensure_ascii=False, indent=2) + '\n').encode())]
    backup = root / 'logs/source_update_backups' / ('before_av_routing_' + time.strftime('%Y%m%d_%H%M%S_') + uuid.uuid4().hex[:8] + '.zip')
    backup.parent.mkdir(parents=True, exist_ok=True)
    existed = []
    with zipfile.ZipFile(backup, 'x', zipfile.ZIP_DEFLATED) as z:
        for name, _ in changes:
            target = (root / name).resolve()
            if not target.is_relative_to(root): raise RuntimeError('Target escapes runtime.')
            if target.is_file(): z.write(target, name); existed.append(name)
        z.writestr('BACKUP_INDEX.json', json.dumps({'targets': [x[0] for x in changes], 'existed': existed}))
    try:
        for name, content in changes:
            target = root / name; target.parent.mkdir(parents=True, exist_ok=True)
            temp = target.with_name(target.name + '.av-update.tmp'); temp.write_bytes(content); os.replace(temp, target)
    except BaseException:
        with zipfile.ZipFile(backup) as z:
            for name, _ in changes:
                target = (root / name).resolve()
                if not target.is_relative_to(root): continue
                if name in existed: target.write_bytes(z.read(name))
                elif target.is_file(): target.unlink()
        raise
    print('Updated source only. Backup: ' + str(backup), flush=True)
    return backup


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--manifest', type=Path, default=REPO / 'AV_BRANCH_SOURCE_MANIFEST.json')
    p.add_argument('--apply', action='store_true')
    args = p.parse_args()
    root, manifest, changes = prepare(args.root, args.manifest)
    print('Reviewed branch commit: ' + manifest['backend_commit'])
    print('Preflight passed: %d source files; models/environments/task database are not targets.' % len(changes))
    if args.apply: apply(root, manifest, changes)


if __name__ == '__main__': main()
