"""Instrument the existing visual worker without changing its acceptance decisions."""
from pathlib import Path
import csv
import json
import math
import sys
from speaker_av_support import read_json, write_json, say

SOURCE = None


def classify(scores):
    if not scores:
        return 'no_kept_track'
    positives = sum(x > 0 for x in scores)
    return 'no_active_face' if positives == 0 else ('multiple_active_faces' if positives > 1 else 'pending_fqa')


def quality_gate(value):
    if not math.isfinite(value):
        return 'nonfinite_fqa'
    return 'fqa_rejected' if value < .7 else 'accepted'


def verify_acceptance(candidates, frames):
    predicted = {int(x['local_frame']) for x in candidates if x['original_would_accept']}
    actual = {int(x) for x in frames}
    if predicted != actual:
        raise RuntimeError('Gate trace differs from the unchanged publisher acceptance set.')


class GateTrace:
    def __init__(self, base, out):
        self.base, self.out = base, Path(out)
        self.rows, self.images, self.image_counts = [], [], {}
        self.detected_boxes, self.kept_tracks = 0, 0

    def save_image(self, image, row):
        import cv2
        kind = row['gate']
        count = self.image_counts.get(kind, 0)
        if count >= 3 or not image.size:
            return
        filename = 'images/%s_%02d_t%.3f.jpg' % (kind, count, row['time'])
        if cv2.imwrite(str(self.out / filename), image, [cv2.IMWRITE_JPEG_QUALITY, 80]):
            self.image_counts[kind] = count + 1
            row['image'] = filename
            self.images.append({'path': filename, 'gate': kind, 'time': row['time'], 'fqa': row.get('fqa')})

    def __enter__(self):
        import numpy as np
        import cv2
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / 'images').mkdir(exist_ok=True)
        (self.out / 'frames.jsonl').write_text('', encoding='utf-8')
        self.original_fr = self.base.evaluate_fr
        self.original_detect = self.base.face_detection
        tracer = self

        def detection(processor, frames):
            result = tracer.original_detect(processor, frames)
            tracer.detected_boxes += sum(len(x) for x in result)
            return result

        def evaluate(processor, frames, tracks, scores):
            tracer.kept_tracks += len(tracks)
            local, pending = [], []
            for frame in range(0, len(frames), processor.face_det_stride):
                values = []
                for tid, track in enumerate(tracks):
                    matches = np.flatnonzero(track['track']['frame'] == frame)
                    if not len(matches):
                        continue
                    position = int(matches[0])
                    score = scores[tid]
                    selected = score[max(position - 2, 0):min(position + 3, len(score) - 1)]
                    value = float(np.mean(selected)) if len(selected) else float('nan')
                    values.append((tid, value))
                row = {'block': processor.audit_block, 'local_frame': frame,
                    'time': (processor.audit_start + frame) / 25,
                    'tracks': [{'id': tid, 'smoothed_asd': v if math.isfinite(v) else None} for tid, v in values],
                    'active_count': sum(v > 0 for _, v in values),
                    'asd_nonfinite': sum(not math.isfinite(v) for _, v in values),
                    'gate': classify([v for _, v in values]), 'fqa': None}
                local.append(row)
                if row['gate'] == 'pending_fqa':
                    pending.append(row)
                elif row['gate'] in {'no_active_face', 'multiple_active_faces'}:
                    factor = min(1, 960 / frames[frame].shape[1])
                    image = cv2.resize(frames[frame], (int(frames[frame].shape[1] * factor), int(frames[frame].shape[0] * factor)))
                    tracer.save_image(image, row)
            original_quality = processor.face_quality_evaluator
            cursor = 0

            def measured_quality(image):
                nonlocal cursor
                if cursor >= len(pending):
                    raise RuntimeError('FQA calls differ from the original unique-active-face rule.')
                row = pending[cursor]
                cursor += 1
                raw = original_quality(image)  # Exactly one original model call.
                value = float(raw)
                row.update(fqa=value if math.isfinite(value) else None, gate=quality_gate(value),
                    original_would_accept=not (value < .7))
                tracer.save_image(image, row)
                return raw

            processor.face_quality_evaluator = measured_quality
            try:
                accepted = tracer.original_fr(processor, frames, tracks, scores)
            finally:
                processor.face_quality_evaluator = original_quality
            if cursor != len(pending):
                raise RuntimeError('An FQA candidate was not matched to an actual original call.')
            verify_acceptance(pending, accepted['frameI'])
            tracer.rows.extend(local)
            with (tracer.out / 'frames.jsonl').open('a', encoding='utf-8') as stream:
                for row in local:
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            return accepted  # Same feature object; no replacement labels or embeddings.

        self.base.evaluate_fr, self.base.face_detection = evaluate, detection
        return self

    def __exit__(self, exc_type, exc, tb):
        import numpy as np
        self.base.evaluate_fr, self.base.face_detection = self.original_fr, self.original_detect
        counts = {}
        for row in self.rows:
            counts[row['gate']] = counts.get(row['gate'], 0) + 1
        qualities = [x['fqa'] for x in self.rows if x['fqa'] is not None]
        summary = {'status': 'complete' if exc_type is None else 'partial', 'error': str(exc) if exc else None,
            'sampled_frames': len(self.rows), 'detected_boxes': self.detected_boxes, 'kept_tracks': self.kept_tracks,
            'gate_histogram': counts, 'fqa_calls': sum('original_would_accept' in x for x in self.rows),
            'nonfinite_asd_values': sum(x['asd_nonfinite'] for x in self.rows),
            'fqa_threshold': .7, 'asd_threshold': 0, 'policy_changed': False,
            'original_acceptance_set_checked': exc_type is None,
            'fqa_distribution': {key: float(np.quantile(qualities, q)) for key, q in
                [('min', 0), ('q25', .25), ('median', .5), ('q75', .75), ('max', 1)]} if qualities else None,
            'images': self.images}
        write_json(self.out / 'summary.json', summary)
        with (self.out / 'frames.csv').open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['time', 'block', 'local_frame', 'active_count', 'asd_nonfinite', 'gate', 'fqa', 'image'], extrasaction='ignore')
            writer.writeheader()
            writer.writerows(self.rows)
        say('视觉门控计数：' + json.dumps(counts, ensure_ascii=False))


def run_vision(run, vision_worker):
    sys.path.insert(0, str(SOURCE))
    sys.path.insert(0, str(SOURCE / 'egs/3dspeaker/speaker-diarization/local'))
    from vision_processer import VisionProcesser
    with GateTrace(VisionProcesser, run / 'C/gates'):
        vision_worker(run)
