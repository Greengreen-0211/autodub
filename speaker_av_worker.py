"""Isolated 3D-Speaker audio/visual worker. This entry never runs legacy A."""
from pathlib import Path
import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import traceback

# -I deliberately excludes user site and script-directory imports.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from speaker_av_support import assess_evidence, fingerprint, read_json, write_json, sha, phase_valid, mark_phase, read_rttm, say
from speaker_av_resources import verify_resources


def prepare(run, request):
    for name in ('common', 'B', 'C', 'models', 'checkpoints', 'cache'):
        (run / name).mkdir(parents=True, exist_ok=True)
    if sha(request['video']) != request['video_sha256'] or sha(request['vocals']) != request['vocals_sha256']:
        raise RuntimeError('Input changed before model loading')
    options = request['settings']
    assets = verify_resources(options)
    for row in assets:
        source = Path(row['path'])
        target = run / 'models' / source.relative_to(Path(options['models']))
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file():
            if sha(target) != row['sha256']: raise RuntimeError('Owned model link/config changed')
        elif source.suffix in {'.json', '.yaml', '.md', '.txt'}:
            shutil.copyfile(source, target)
        else:
            target.symlink_to(source.resolve())
    gpu = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=name,memory.free',
                                  '--format=csv,noheader,nounits'], text=True).strip()
    if int(gpu.rsplit(',', 1)[1]) < 5000 or shutil.disk_usage(run).free < 512 * 1024**2:
        raise RuntimeError('Not enough free GPU0 memory (5GiB) or disk (512MiB) for AV; no model loaded')
    ffmpeg, ffprobe = shutil.which('ffmpeg'), shutil.which('ffprobe')
    preferred = Path('/home/goldenseed/bin/ffmpeg')
    if (preferred / 'ffmpeg').is_file(): ffmpeg, ffprobe = str(preferred / 'ffmpeg'), str(preferred / 'ffprobe')
    if not ffmpeg or not ffprobe: raise RuntimeError('ffmpeg/ffprobe missing')
    media = json.loads(subprocess.check_output([ffprobe, '-v', 'error', '-show_streams', '-show_format', '-of', 'json', request['video']], text=True))
    video = next((x for x in media['streams'] if x['codec_type'] == 'video'), None)
    if video is None: raise RuntimeError('Input has no video stream')
    if abs(float(video.get('start_time', 0))) > .02 or abs(float(media['format']['duration']) - request['duration']) > .2:
        raise RuntimeError('Audio/video origins or durations do not match the tested synchronization policy')
    key = request['fingerprint']
    if not phase_valid(run, 'media', key):
        subprocess.run([ffmpeg, '-nostdin', '-y', '-v', 'warning', '-threads', '2', '-i', request['vocals'],
            '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', str(run / 'common/input.wav')], check=True, timeout=180)
        subprocess.run([ffmpeg, '-nostdin', '-y', '-v', 'warning', '-threads', '2', '-i', request['video'],
            '-map', '0:v:0', '-an', '-vf', 'fps=25', '-filter_threads', '2', '-c:v', 'libx264',
            '-crf', '18', '-preset', 'veryfast', '-threads', '2', str(run / 'common/input.mp4')], check=True, timeout=300)
        mark_phase(run, 'media', key, [run / 'common/input.wav', run / 'common/input.mp4'])
    write_json(run / 'protocol.json', {'fingerprint': key, 'duration': request['duration'], 'upstream': options['commit']})
    write_json(run / 'preflight.json', {'gpu': gpu, 'source': options['source'], 'model_files_checked': len(assets),
        'gpu_index': 0, 'cpu_threads': 2, 'pretrained_models_loaded': False})


def execute(plan):
    request = read_json(plan)
    run = plan.parent.resolve()
    prepare(run, request)
    import torch
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable in configured AV environment')
    torch.set_num_threads(2)
    test = torch.ones((4, 4), device='cuda:0'); assert float((test @ test)[0, 0]) == 4
    del test
    import speaker_av_runtime as runtime
    import speaker_av_gates as gates
    runtime.SOURCE = gates.SOURCE = Path(request['settings']['source'])
    stages = [('audio', runtime.audio_b, ['B/labels.json', 'B/audio_branch.rttm']),
        ('vision', lambda r: gates.run_vision(r, runtime.vision), ['C/visual_embeddings.pkl', 'C/face_evidence.json', 'C/local_tracks.json', 'C/gates/summary.json']),
        ('joint', runtime.audiovisual_c, ['C/fusion_audit.json'])]
    for stage, call, outputs in stages:
        if not phase_valid(run, stage, request['fingerprint']):
            say('3D-Speaker开始：' + stage)
            call(run)
            files = [run / x for x in outputs]
            if stage == 'vision':
                files += [run / x['image'] for x in read_json(run / 'C/face_evidence.json')]
                files += [run / 'C/gates' / x['path'] for x in read_json(run / 'C/gates/summary.json')['images']]
                files += [run / 'C/gates/frames.csv', run / 'C/gates/frames.jsonl']
            if stage == 'joint' and (run / 'C/audiovisual.rttm').is_file():
                files += [run / 'C/audiovisual.rttm', run / 'C/labels.json']
            mark_phase(run, stage, request['fingerprint'], files)
        gc.collect(); torch.cuda.empty_cache()
    proof = read_json(run / 'C/fusion_audit.json')
    proof['gates'] = read_json(run / 'C/gates/summary.json')
    usable, reason, message = assess_evidence(proof)
    result = {'fingerprint': request['fingerprint'], 'duration': request['duration'], 'evidence': proof,
        'status': 'usable' if usable else 'insufficient_visual_evidence', 'reason_code': reason, 'reason': message,
        'upstream_commit': request['settings']['commit'], 'audio_B_adopted': False}
    if usable:
        output = run / 'C/audiovisual.rttm'
        read_rttm(output, request['duration'])
        result.update(rttm=str(output), rttm_sha256=sha(output))
    if sha(request['video']) != request['video_sha256'] or sha(request['vocals']) != request['vocals_sha256']:
        raise RuntimeError('Input changed during AV execution')
    write_json(run / 'result.json', result)
    say('视听证据：' + message)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', required=True, type=Path)
    args = p.parse_args()
    try: execute(args.plan.resolve(strict=True))
    except Exception:
        traceback.print_exc(); return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
