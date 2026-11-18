"""Adapter of the verified 3D-Speaker audio/vision/joint path, without A or providers.

Upstream models and thresholds remain in the pinned installed repository. Only
resource limits, offline paths, progress and evidence recording are adapted.
"""
from pathlib import Path
import gc
import importlib
import json
import os
import pickle
import random
import runpy
import sys
from speaker_av_support import (say, read_json, write_json, sha, fingerprint, phase_valid, mark_phase, read_rttm)

# Filled from the validated request in the isolated worker before any model load.
SOURCE = None

def pickle_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('wb') as stream:
        pickle.dump(value, stream)
    os.replace(temporary, path)

def pickle_read(path):
    with Path(path).open('rb') as stream:
        return pickle.load(stream)

def setup_source():
    sys.path.insert(0, str(SOURCE))
    sys.path.insert(0, str(SOURCE / 'egs/3dspeaker/speaker-diarization/local'))

def upstream_module(name, path):
    # Upstream CLI modules parse argv at import time; provide only their script name.
    previous = sys.argv
    sys.argv = [str(path)]
    try:
        return importlib.import_module(name)
    finally:
        sys.argv = previous

def ort_resource_adapter(audit):
    import torch
    import onnxruntime as ort
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable in the trial environment.')
    if hasattr(ort, 'preload_dlls'):
        ort.preload_dlls()
    original = ort.InferenceSession

    def session(model, *args, **kwargs):
        options = kwargs.get('sess_options')
        if options is None and args and hasattr(args[0], 'intra_op_num_threads'):
            options = args[0]
        if options is None:
            options = ort.SessionOptions()
            kwargs['sess_options'] = options
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        options.log_severity_level = 2
        providers = kwargs.get('providers') or ['CUDAExecutionProvider', 'CPUExecutionProvider']
        updated = []
        wanted_cuda = False
        for item in providers:
            name, settings = item if isinstance(item, tuple) else (item, {})
            if name == 'CUDAExecutionProvider':
                wanted_cuda = True
                settings = dict(settings, device_id=0, gpu_mem_limit=2 * 1024**3)
                updated.append((name, settings))
            else:
                updated.append(item)
        kwargs['providers'] = updated
        instance = original(model, *args, **kwargs)
        actual = instance.get_providers()
        if wanted_cuda and (not actual or actual[0] != 'CUDAExecutionProvider'):
            raise RuntimeError('A requested CUDA ONNX session fell back to CPU: ' + str(model))
        audit.append({'model': str(model), 'providers': actual, 'threads': 2,
                      'gpu_mem_limit_per_session': 2 * 1024**3})
        return instance
    ort.InferenceSession = session
    return ort, original

def audio_b(run):
    setup_source()
    import numpy as np
    import torch
    torch.set_num_threads(2)
    config = read_json(run / 'protocol.json')
    common = run / 'common'
    key = fingerprint({'protocol': config['fingerprint'], 'actual_wave_sha256': sha(common / 'input.wav')})
    vad_file, emb_file = common / 'vad.json', common / 'audio_embeddings.pkl'
    if not phase_valid(run, 'vad', key):
        say('VAD：加载已校验的本地FSMN模型。')
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks
        vad = pipeline(task=Tasks.voice_activity_detection, model=str(run / 'models/vad'),
                       device='cuda', disable_update=True)
        result = vad(str(common / 'input.wav'))
        item = result[0] if isinstance(result, list) else result
        rows = {}
        for start_ms, end_ms in item['value']:
            start, end = round(float(start_ms) / 1000, 3), round(float(end_ms) / 1000, 3)
            if not 0 <= start < end <= config['duration'] + 0.1:
                raise RuntimeError('Invalid VAD source times.')
            rows['input_' + str(start) + '_' + str(end)] = {
                'file': str(common / 'input.wav'), 'start': start, 'stop': end}
        if not rows:
            raise RuntimeError('FSMN found no speech; no synthetic fallback is used.')
        write_json(vad_file, rows)
        mark_phase(run, 'vad', key, [vad_file])
        say('VAD完成：' + str(len(rows)) + '段。')
        del vad
        gc.collect()
        torch.cuda.empty_cache()
    if not phase_valid(run, 'audio_embeddings', key):
        subseg = common / 'subsegments.json'
        official_subsegments(run)
        items = read_json(subseg)
        from speakerlab.models.campplus.DTDNN import CAMPPlus
        from speakerlab.process.processor import FBank
        from speakerlab.utils.fileio import load_audio
        from speakerlab.utils.utils import circle_pad
        model = CAMPPlus(feat_dim=80, embedding_size=192)
        state = torch.load(str(run / 'models/campplus/campplus_cn_en_common.pt'), map_location='cpu')
        model.load_state_dict(state, strict=True)
        model.eval().to('cuda:0')
        extractor = FBank(80, 16000, mean_nor=True)
        wav = load_audio(str(common / 'input.wav'), obj_fs=16000)
        selected = list(items.values())
        clips = [wav[0, int(x['start'] * 16000):int(x['stop'] * 16000)] for x in selected]
        if not clips or any(x.numel() == 0 for x in clips):
            raise RuntimeError('Empty real-audio embedding window.')
        length = max(x.shape[0] for x in clips)
        clips = torch.stack([circle_pad(x, length) for x in clips]).unsqueeze(1)
        features = []
        with torch.no_grad():
            for start in range(0, len(clips), 16):
                batch = clips[start:start + 16].to('cuda:0')
                feats = torch.vmap(extractor)(batch)
                features.append(model(feats).cpu())
                say('CAM++声纹：' + str(min(start + 16, len(clips))) + '/' + str(len(clips)))
        values = torch.cat(features).numpy()
        if values.shape != (len(selected), 192) or not np.isfinite(values).all():
            raise RuntimeError('Invalid CAM++ output.')
        pickle_write(emb_file, {'embeddings': values, 'times': [[x['start'], x['stop']] for x in selected]})
        mark_phase(run, 'audio_embeddings', key, [emb_file, subseg])
    from speakerlab.utils.config import Config, yaml_config_loader
    from speakerlab.utils.builder import build
    conf = Config(yaml_config_loader(str(SOURCE / 'egs/3dspeaker/speaker-diarization/conf/diar_video.yaml')))
    cluster = build('audio_cluster', conf)
    stat = pickle_read(emb_file)
    np.random.seed(0)
    random.seed(0)
    labels = cluster(stat['embeddings'])
    write_json(run / 'B/labels.json', [int(x) for x in labels])
    output_rttm(run / 'B/audio_branch.rttm', stat['times'], labels)
    read_rttm(run / 'B/audio_branch.rttm', config['duration'])

def official_subsegments(run):
    setup_source()
    script = SOURCE / 'egs/3dspeaker/speaker-diarization/local/prepare_subseg_json.py'
    sys.argv = [str(script), '--vad', str(run / 'common/vad.json'),
                '--out_file', str(run / 'common/subsegments.json')]
    runpy.run_path(str(script), run_name='__main__')

def output_rttm(path, times, labels):
    setup_source()
    import numpy as np
    helper = upstream_module('cluster_and_postprocess',
        SOURCE / 'egs/3dspeaker/speaker-diarization/local/cluster_and_postprocess.py')
    labels = np.asarray(labels, dtype=int)
    mapped = np.zeros(len(labels), dtype=int)
    for index, label in enumerate(np.unique(labels)):
        mapped[labels == label] = index
    path.parent.mkdir(parents=True, exist_ok=True)
    helper.make_rttms(list(zip(times, mapped)), str(path), 'input')

def vision(run):
    setup_source()
    import cv2
    cv2.setNumThreads(2)
    import numpy as np
    from vision_processer import VisionProcesser
    from speakerlab.utils.config import yaml_config_loader
    conf = yaml_config_loader(str(SOURCE / 'egs/3dspeaker/speaker-diarization/conf/diar_video.yaml'))
    vad = read_json(run / 'common/vad.json')
    intervals = sorted([[x['start'], x['stop']] for x in vad.values()])
    merged = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    audit, tracks = [], []
    sessions = []
    ort, original_session = ort_resource_adapter(sessions)
    images = run / 'C/faces'
    images.mkdir(exist_ok=True)

    class AuditedVision(VisionProcesser):
        def process_one_shot(self, frames, face_det_frames, audio, frame_st=None):
            self.audit_start = int(frame_st)
            self.audit_block = len(self.elapsed_time['faceTime'])
            super().process_one_shot(frames, face_det_frames, audio, frame_st)
            say('视觉块 ' + str(self.audit_block + 1) + '：原视频 ' +
                format(frame_st / 25, '.2f') + '–' + format((frame_st + len(frames)) / 25, '.2f') +
                's；累计有效脸特征 ' + str(len(audit)))

        def evaluate_fr(self, frames, local_tracks, scores):
            accepted = super().evaluate_fr(frames, local_tracks, scores)
            for tid, track in enumerate(local_tracks):
                score = np.asarray(scores[tid], dtype=float)
                finite = score[np.isfinite(score)]
                tracks.append({'block': self.audit_block, 'track': tid,
                    'start': (self.audit_start + int(track['track']['frame'][0])) / 25,
                    'end': (self.audit_start + int(track['track']['frame'][-1]) + 1) / 25,
                    'positive_score_fraction': float(np.mean(finite > 0)) if len(finite) else None})
            for frame_index in accepted['frameI']:
                frame_index = int(frame_index)
                active = []
                for tid, track in enumerate(local_tracks):
                    matches = np.flatnonzero(track['track']['frame'] == frame_index)
                    if not len(matches):
                        continue
                    position = int(matches[0])
                    local_score = scores[tid]
                    values = local_score[max(position - 2, 0):min(position + 3, len(local_score) - 1)]
                    score = float(np.mean(values)) if len(values) else float('nan')
                    if score > 0:
                        active.append((tid, track['track']['bbox'][position], score))
                if len(active) != 1:
                    raise RuntimeError('Audit active-face mapping differs from upstream acceptance.')
                tid, bbox, score = active[0]
                x1, y1, x2, y2 = [int(x) for x in bbox]
                frame = frames[frame_index]
                crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
                file = images / ('feature_%04d.jpg' % len(audit))
                if not crop.size or not cv2.imwrite(str(file), crop, [cv2.IMWRITE_JPEG_QUALITY, 80]):
                    raise RuntimeError('Unable to save accepted face evidence.')
                audit.append({'feature_index': len(audit), 'time': (self.audit_start + frame_index) / 25,
                    'block': self.audit_block, 'track': tid, 'asd_score': score,
                    'image': str(file.relative_to(run))})
            return accepted

    temporary = run / 'C/visual_embeddings.pkl.partial'
    try:
        model = AuditedVision(str(run / 'common/input.mp4'), str(run / 'common/input.wav'),
            merged, str(temporary), str(run / 'models/visual'), conf, device='cuda', device_id=0,
            out_video_path=None)
        model.run()
        stat = pickle_read(temporary)
        if stat['embeddings'].shape != (len(audit), 512) or not np.isfinite(stat['embeddings']).all():
            raise RuntimeError('Invalid visual embeddings/audit correspondence.')
        if len(stat['times']) != len(audit) or any(abs(float(t) - a['time']) > 1e-5 for t, a in zip(stat['times'], audit)):
            raise RuntimeError('Visual evidence timestamps differ from actual model feature times.')
        os.replace(temporary, run / 'C/visual_embeddings.pkl')
        write_json(run / 'C/face_evidence.json', audit)
        write_json(run / 'C/local_tracks.json', tracks)
        write_json(run / 'C/onnx_sessions.json', sessions)
    finally:
        ort.InferenceSession = original_session

def audiovisual_c(run):
    setup_source()
    import numpy as np
    from speakerlab.utils.config import Config, yaml_config_loader
    from speakerlab.utils.builder import build
    stat = pickle_read(run / 'common/audio_embeddings.pkl')
    visual = pickle_read(run / 'C/visual_embeddings.pkl')
    proof = {'visual_features': len(visual['times']), 'visual_fusion_applied': False,
             'note': 'Visual cluster IDs are not ground-truth character names.'}
    if not len(visual['times']):
        (run / 'C/audiovisual.rttm').unlink(missing_ok=True)
        (run / 'C/labels.json').unlink(missing_ok=True)
        proof.update(status='unavailable_no_visual_features', reason='No face passed upstream ASD/quality gates.')
        write_json(run / 'C/fusion_audit.json', proof)
        return
    conf = Config(yaml_config_loader(str(SOURCE / 'egs/3dspeaker/speaker-diarization/conf/diar_video.yaml')))
    joint = build('cluster', conf)
    original_audio, original_vision = joint.audio_cluster, joint.vision_cluster
    original_get, original_overlap = joint.get_vlist_embs, joint.overlap_spks
    saved = {}

    def audio(*args, **kwargs):
        labels = original_audio(*args, **kwargs)
        saved['audio'] = labels.copy()
        return labels

    def faces(*args, **kwargs):
        labels = original_vision(*args, **kwargs)
        saved['visual'] = labels.copy()
        return labels

    def anchors(*args, **kwargs):
        value = original_get(*args, **kwargs)
        proof['visual_anchor_intervals'] = [[float(a), float(b), int(c)] for a, b, c in value[0]]
        proof['voice_prototypes_from_visual'] = len(value[1])
        return value

    matches = []

    def overlap(*args, **kwargs):
        value = original_overlap(*args, **kwargs)
        matches.append([int(x) for x in value])
        return value

    joint.audio_cluster, joint.vision_cluster = audio, faces
    joint.get_vlist_embs, joint.overlap_spks = anchors, overlap
    np.random.seed(0)
    random.seed(0)
    labels = joint(stat['embeddings'], visual['embeddings'], stat['times'], visual['times'], conf)
    baseline = np.asarray(read_json(run / 'B/labels.json'))
    internal = saved['audio']
    if not np.array_equal(baseline[:, None] == baseline[None, :], internal[:, None] == internal[None, :]):
        raise RuntimeError('C internal audio partition differs from B; visual-only comparison is invalid.')
    applied = any(matches)
    proof.update(status='completed' if applied else 'no_usable_visual_anchors',
        visual_fusion_applied=applied, audio_clusters_with_visual_match=sum(bool(x) for x in matches),
        audio_branch_matches_B=True, visual_cluster_labels=[int(x) for x in saved['visual']])
    write_json(run / 'C/fusion_audit.json', proof)
    write_json(run / 'C/labels.json', [int(x) for x in labels])
    output_rttm(run / 'C/audiovisual.rttm', stat['times'], labels)
    read_rttm(run / 'C/audiovisual.rttm', read_json(run / 'protocol.json')['duration'])
