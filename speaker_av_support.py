"""Small stdlib helpers shared by the isolated visual speaker worker."""
from pathlib import Path
import hashlib
import json
import math
import os
import time
import uuid

AV_COMMIT = '065629c313eaf1a01c65c640c46d77e61e9607b4'


def say(message):
    print(time.strftime('[%H:%M:%S] ') + str(message), flush=True)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex[:8] + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def read_rttm(path, duration=None):
    rows = []
    for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
        p = line.split()
        if not p or p[0].startswith('#'):
            continue
        if len(p) < 8 or p[0] != 'SPEAKER':
            raise ValueError('Malformed RTTM')
        start, length = float(p[3]), float(p[4])
        if not math.isfinite(start) or not math.isfinite(length) or start < 0 or length <= 0:
            raise ValueError('Invalid RTTM time')
        if duration is not None and start + length > duration + .1:
            raise ValueError('RTTM exceeds the input clock')
        rows.append({'start': start, 'end': start + length, 'speaker': p[7]})
    if not rows:
        raise ValueError('Empty RTTM')
    return sorted(rows, key=lambda x: (x['start'], x['end']))


def phase_valid(run, phase, key):
    p = Path(run) / 'checkpoints' / (phase + '.json')
    if not p.is_file():
        return False
    row = read_json(p)
    outputs = row.get('outputs') or []
    return row.get('key') == key and bool(outputs) and all(
        Path(x['path']).resolve().is_relative_to(Path(run).resolve()) and Path(x['path']).is_file()
        and sha(x['path']) == x['sha256'] for x in outputs)


def mark_phase(run, phase, key, files):
    write_json(Path(run) / 'checkpoints' / (phase + '.json'), {'key': key,
        'outputs': [{'path': str(p), 'sha256': sha(p)} for p in files]})


def assess_evidence(proof):
    """Gate actual official anchor use, not cluster count or visual appearance."""
    if not isinstance(proof, dict):
        return False, 'invalid_evidence', '视听结果没有有效的证据记录。'
    if proof.get('status') == 'unavailable_no_visual_features' or proof.get('visual_features') == 0:
        return False, 'no_visual_features', '没有人脸通过口型活跃与人脸质量筛选。'
    if proof.get('status') == 'no_usable_visual_anchors':
        return False, 'no_usable_visual_anchors', '存在人脸特征，但未形成可用的视听声音锚点。'
    if proof.get('status') != 'completed':
        return False, 'av_not_completed', '视听处理没有正常完成。'
    if proof.get('audio_branch_matches_B') is not True:
        return False, 'audio_branch_mismatch', '联合处理的声学分区与本次独立音频分区不一致或没有核对记录。'
    try:
        features = int(proof.get('visual_features', 0))
        prototypes = int(proof.get('voice_prototypes_from_visual', 0))
        matches = int(proof.get('audio_clusters_with_visual_match', 0))
    except (ValueError, TypeError, OverflowError):
        return False, 'invalid_evidence', '视听证据计数格式无效。'
    if features <= 0 or prototypes <= 0 or matches <= 0 or proof.get('visual_fusion_applied') is not True:
        return False, 'no_usable_visual_anchors', '没有实际参与匹配的视听声音锚点。'
    return True, 'usable_visual_anchors', '已有有效视听锚点参与说话人匹配。'
