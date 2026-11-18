"""Bind the HTTP delivery to a real, locally committed feature snapshot.

Run in a Git checkout after committing the feature. This does not install it.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BASE = 'f0577d9c7252bb22ae24494d5c9791d517aaba4e'
MODULES = ('diarization_router.py', 'speaker_av_support.py', 'speaker_av_runtime.py',
           'speaker_av_resources.py', 'speaker_av_gates.py', 'speaker_av_worker.py')
HTTP = {'task_service/native_audio.py': 'native_audio.py',
        'task_service/native_trials.py': 'native_trials.py',
        'task_service/av_routing_binding.py': 'av_routing_binding.py',
        'tools/start_video_workflow.py': 'start_video_workflow.py'}


def git(repo, *args):
    result = subprocess.run(['git', '-C', str(repo), *args], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, check=True)
    return result.stdout


def prepare(repo):
    repo = Path(repo).resolve()
    if Path(git(repo, 'rev-parse', '--show-toplevel').decode().strip()).resolve() != repo:
        raise RuntimeError('Use the backend Git repository root.')
    head = git(repo, 'rev-parse', 'HEAD').decode().strip()
    git(repo, 'merge-base', '--is-ancestor', BASE, head)
    if head == BASE:
        raise RuntimeError('Commit the AV feature before binding its source identity.')
    previous = json.loads((repo / 'integration/native_http_adapter/BASE.json').read_text(encoding='utf-8'))
    names = [row['path'] for row in previous['files']] + list(MODULES)
    pairs = [(name, name) for name in names]
    pairs.extend((target, 'integration/native_http_adapter/' + source) for target, source in HTTP.items())
    if len({name for name, _ in pairs}) != len(pairs):
        raise RuntimeError('Duplicate runtime path.')
    rows = []
    for target, source in pairs:
        path = (repo / source).resolve()
        if not path.is_relative_to(repo) or not path.is_file():
            raise RuntimeError('Missing runtime source: ' + source)
        content = path.read_bytes().replace(b'\r\n', b'\n')
        committed = git(repo, 'show', head + ':' + source).replace(b'\r\n', b'\n')
        if content != committed:
            raise RuntimeError('Runtime differs from the actual commit: ' + source)
        rows.append({'path': target, 'source': source, 'sha256': hashlib.sha256(content).hexdigest()})
    return {'schema': 'backend-av-routing-binding-v1', 'base_commit': BASE,
            'backend_commit': head, 'files': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=REPO)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    record = prepare(args.repo)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(record, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
    print('Bound actual Git commit: ' + record['backend_commit'])
    print('Runtime/HTTP files: %d; manifest: %s' % (len(record['files']), args.output))


if __name__ == '__main__': main()
