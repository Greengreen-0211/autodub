#!/usr/bin/env python3
"""
Step 2 WeSpeaker 桥接脚本 v2.2
内嵌自适应聚类 + Qwen3 时间戳拆段 + 碎片合并 + 相邻合并
"""

import json
import math
import os
import re
import shutil
import subprocess
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None
try:
    import kaldiio
except ImportError:
    kaldiio = None

AUTODUB_WORKDIR = os.getenv("AUTODUB_WORKDIR", os.path.dirname(os.path.abspath(__file__)))
AUTODUB_VIDEO = os.getenv("AUTODUB_VIDEO", "")
WESPEAKER_ROOT = os.getenv("WESPEAKER_ROOT", "/data0/goldenseed/chy/wespeaker_voxconverse_v2")
SIMAM_MODEL_PATH = os.getenv("SIMAM_MODEL_PATH", "/data0/goldenseed/models/wespeaker_simamresnet100/speaker-embedding.onnx")

TEMP_DIR = os.path.join(AUTODUB_WORKDIR, "temp", Path(AUTODUB_VIDEO).stem if AUTODUB_VIDEO else "input")
WORK_DIR = os.path.join(TEMP_DIR, "wespeaker_diar")
WESPEAKER_EXAMPLE_DIR = os.path.join(WESPEAKER_ROOT, "wespeaker", "examples", "voxconverse", "v2")


def load_state():
    with open(os.path.join(TEMP_DIR, "project_state.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def save_state(state):
    path = os.path.join(TEMP_DIR, "project_state.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    print(f"✅ 状态已保存: {path}")


def run_cmd(cmd, env=None, cwd=None, stdout_file=None):
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    print(f"▶ {' '.join(cmd[:6])}...")
    if stdout_file:
        with open(stdout_file, "w") as f:
            subprocess.run(cmd, env=full_env, cwd=cwd, check=True, stdout=f)
    else:
        subprocess.run(cmd, env=full_env, cwd=cwd, check=True)


def prepare_inputs(state):
    os.makedirs(WORK_DIR, exist_ok=True)
    for sub in ["fbank", "embedding", "labels"]:
        p = os.path.join(WORK_DIR, sub)
        if os.path.exists(p):
            if os.path.islink(p) or os.path.isfile(p):
                os.unlink(p)
            elif os.path.isdir(p):
                shutil.rmtree(p)

    utt_id = "input"
    vocal_path = state["vocals_path"]

    with open(os.path.join(WORK_DIR, "wav.scp"), "w", encoding="utf-8") as f:
        f.write(f"{utt_id} {os.path.abspath(vocal_path)}\n")

    with open(os.path.join(WORK_DIR, "oracle_sad"), "w", encoding="utf-8") as f:
        for idx, seg in enumerate(state["segments_step1"]):
            s, e = float(seg["start"]), float(seg["end"])
            if e > s:
                seg_id = f"{utt_id}-{int(s*1000):08d}-{int(e*1000):08d}"
                f.write(f"{seg_id} {utt_id} {s:.3f} {e:.3f}\n")

    print(f"✅ WeSpeaker 输入已生成: {WORK_DIR}")
    return utt_id


def run_wespeaker_pipeline():
    env = {}
    pp = os.environ.get("PYTHONPATH", "")
    wespeaker_pkg = os.path.join(WESPEAKER_ROOT, "wespeaker")
    if wespeaker_pkg not in pp:
        env["PYTHONPATH"] = f"{wespeaker_pkg}:{pp}" if pp else wespeaker_pkg
    env["LD_LIBRARY_PATH"] = os.path.join(sys.prefix, "lib")

    print("\n=== [Step 2-W] WeSpeaker: Fbank 提取 ===")
    run_cmd([
        sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "make_fbank_direct.py"),
        "--scp", os.path.join(WORK_DIR, "wav.scp"),
        "--segments", os.path.join(WORK_DIR, "oracle_sad"),
        "--ark-path", os.path.join(WORK_DIR, "fbank", "fbank.ark"),
        "--subseg-cmn", "true",
    ], env=env)

    print("\n=== [Step 2-W] WeSpeaker: SimAMResNet100 提取 Embedding ===")
    run_cmd([
        "python3", "wespeaker/diar/extract_emb.py",
        "--scp", os.path.join(WORK_DIR, "fbank", "fbank.scp"),
        "--ark-path", os.path.join(WORK_DIR, "embedding", "emb.ark"),
        "--source", SIMAM_MODEL_PATH,
        "--device", "cuda",
        "--batch-size", "96",
        "--frame-shift", "10",
        "--window-secs", "1.5",
        "--period-secs", "0.75",
        "--subseg-cmn", "true",
    ], env=env, cwd=WESPEAKER_EXAMPLE_DIR)

    print("\n=== [Step 2-W] 自适应 UMAP + HDBSCAN 聚类 ===")
    run_adaptive_clustering()

    print("\n=== [Step 2-W] WeSpeaker: 生成 RTTM ===")
    rttm_path = os.path.join(WORK_DIR, "result.rttm")
    run_cmd([
        "python3", "wespeaker/diar/make_rttm.py",
        "--labels", os.path.join(WORK_DIR, "labels"),
        "--channel", "1",
    ], env=env, cwd=WESPEAKER_EXAMPLE_DIR, stdout_file=rttm_path)

    return rttm_path


def run_adaptive_clustering():
    if not kaldiio or not np:
        raise RuntimeError("自适应聚类需要 kaldiio 和 numpy")

    import umap
    import hdbscan

    emb_scp = os.path.join(WORK_DIR, "embedding", "emb.scp")
    labels_out = os.path.join(WORK_DIR, "labels")
    os.makedirs(os.path.dirname(labels_out), exist_ok=True)

    utt_embs = defaultdict(list)
    utt_keys = defaultdict(list)
    with kaldiio.ReadHelper(f"scp:{emb_scp}") as reader:
        for key, mat in reader:
            utt = key.split('-')[0]
            emb = mat.mean(axis=0) if mat.ndim == 2 else mat
            utt_embs[utt].append(emb)
            utt_keys[utt].append(key)

    with open(labels_out, "w", encoding="utf-8") as f:
        for utt, embs in utt_embs.items():
            X = np.stack(embs)
            n = len(X)

            if n < 20:
                strategies = [
                    {"mcs": 3, "ms": 1, "method": "leaf"},
                    {"mcs": 5, "ms": 1, "method": "eom"},
                    {"mcs": 5, "ms": 1, "method": "leaf"},
                ]
            elif n < 50:
                strategies = [
                    {"mcs": 5, "ms": 1, "method": "eom"},
                    {"mcs": 5, "ms": 1, "method": "leaf"},
                    {"mcs": 3, "ms": 1, "method": "leaf"},
                ]
            elif n < 100:
                strategies = [
                    {"mcs": 8, "ms": 2, "method": "eom"},
                    {"mcs": 6, "ms": 1, "method": "eom"},
                    {"mcs": 5, "ms": 1, "method": "leaf"},
                    {"mcs": 4, "ms": 1, "method": "leaf"},
                ]
            else:
                strategies = [
                    {"mcs": 12, "ms": 2, "method": "eom"},
                    {"mcs": 8, "ms": 1, "method": "eom"},
                    {"mcs": 6, "ms": 1, "method": "leaf"},
                    {"mcs": 5, "ms": 1, "method": "leaf"},
                ]

            results = []
            for strategy in strategies:
                try:
                    reducer = umap.UMAP(
                        n_components=min(32, max(2, n - 2)),
                        metric="cosine",
                        n_neighbors=min(15, max(5, n // 5)),
                        min_dist=0.0,
                        random_state=42,
                        n_jobs=1,
                    )
                    emb_2d = reducer.fit_transform(X)

                    clusterer = hdbscan.HDBSCAN(
                        min_cluster_size=strategy["mcs"],
                        min_samples=strategy["ms"],
                        cluster_selection_method=strategy["method"],
                        allow_single_cluster=True,
                        core_dist_n_jobs=1,
                    )
                    labels = clusterer.fit_predict(emb_2d)
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    print(f"  尝试 {strategy}: {n_clusters} 个有效簇")
                    results.append((n_clusters, strategy, labels))
                except Exception as e:
                    print(f"  ⚠️ 策略 {strategy} 失败: {e}")
                    continue

            valid = [(n, s, l) for n, s, l in results if n >= 2]
            if not valid:
                valid = results

            eom_valid = [(n, s, l) for n, s, l in valid if s["method"] == "eom" and n >= 2]
            if eom_valid:
                best_n_clusters, best_strategy, best_labels = max(eom_valid, key=lambda x: x[0])
            else:
                leaf_valid = [(n, s, l) for n, s, l in valid if s["method"] == "leaf"]
                if leaf_valid:
                    mid_leaf = [(n, s, l) for n, s, l in leaf_valid if 3 <= n <= 5]
                    if mid_leaf:
                        best_n_clusters, best_strategy, best_labels = max(mid_leaf, key=lambda x: x[0])
                    else:
                        best_n_clusters, best_strategy, best_labels = max(leaf_valid, key=lambda x: x[0])
                else:
                    best_n_clusters, best_strategy, best_labels = max(valid, key=lambda x: x[0])

            print(f"✅ 选中策略: {best_strategy}, 最终 {best_n_clusters} 个簇")

            for key, label in zip(utt_keys[utt], best_labels):
                f.write(f"{key} {label}\n")


def parse_rttm(rttm_path):
    spans = []
    with open(rttm_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8 or parts[0] != "SPEAKER":
                continue
            start = float(parts[3])
            duration = float(parts[4])
            speaker = parts[7]
            spans.append({"start": start, "end": start + duration, "speaker": speaker})
    spans = sorted(spans, key=lambda x: x["start"])
    return smooth_short_speaker_islands(spans)


def smooth_short_speaker_islands(spans):
    if len(spans) < 3:
        return spans
    smoothed = [span.copy() for span in spans]
    corrected = 0
    index = 1
    while index < len(smoothed) - 1:
        left = smoothed[index - 1]
        middle = smoothed[index]
        right = smoothed[index + 1]
        duration = float(middle["end"]) - float(middle["start"])
        left_gap = float(middle["start"]) - float(left["end"])
        right_gap = float(right["start"]) - float(middle["end"])
        is_short_island = (
            str(left["speaker"]) == str(right["speaker"])
            and str(middle["speaker"]) != str(left["speaker"])
            and duration <= 0.5
            and left_gap <= 0.15
            and right_gap <= 0.15
        )
        if not is_short_island:
            index += 1
            continue
        smoothed[index - 1] = {
            "start": float(left["start"]),
            "end": float(right["end"]),
            "speaker": str(left["speaker"]),
        }
        del smoothed[index:index + 2]
        corrected += 1
        index = max(1, index - 1)
    if corrected:
        print(f"✅ RTTM 单窗抖动平滑: 修正 {corrected} 个 A-B-A 短说话人孤岛。")
    return smoothed


def normalize_speaker(speaker):
    value = str(speaker or "SPEAKER_00")
    match = re.fullmatch(r"(?:spk|speaker)[_-]?(\d+)", value, flags=re.IGNORECASE)
    if match:
        return f"SPEAKER_{int(match.group(1)):02d}"
    if value.isdigit():
        return f"SPEAKER_{int(value):02d}"
    return value


def speaker_overlaps(start, end, rttm_spans):
    overlaps = {}
    for span in rttm_spans:
        overlap = max(0.0, min(end, float(span["end"])) - max(start, float(span["start"])))
        if overlap > 0:
            speaker = str(span["speaker"])
            overlaps[speaker] = overlaps.get(speaker, 0.0) + overlap
    return overlaps


def speaker_for_timestamp(item, rttm_spans, fallback):
    start = float(item.get("start", 0.0))
    end = float(item.get("end", start))
    overlaps = speaker_overlaps(start, end, rttm_spans)
    if end > start:
        probe = min(end, start + 0.001)
        starting_spans = [span for span in rttm_spans if float(span["start"]) <= probe < float(span["end"])]
        if starting_spans:
            starting_speaker = str(starting_spans[0]["speaker"])
            if any(speaker != starting_speaker for speaker in overlaps):
                return normalize_speaker(starting_speaker)
    if overlaps:
        return normalize_speaker(max(overlaps, key=overlaps.get))
    midpoint = (start + end) / 2.0
    nearest = [span for span in rttm_spans if float(span["start"]) <= midpoint <= float(span["end"])]
    if nearest:
        return normalize_speaker(nearest[0]["speaker"])
    return fallback


def ends_sentence(item):
    text = str(item.get("text", "")).strip()
    return bool(re.search(r'[.!?。！？]["\'\u2019\u201d)\]]*$', text))


def snap_speaker_turns_to_punctuation(timestamp_items, speakers):
    if len(timestamp_items) < 3:
        return speakers
    snapped = list(speakers)
    corrected = 0
    index = 1
    while index < len(snapped):
        previous_speaker = snapped[index - 1]
        next_speaker = snapped[index]
        if previous_speaker == next_speaker or ends_sentence(timestamp_items[index - 1]):
            index += 1
            continue
        run_end = index + 1
        while run_end < len(snapped) and snapped[run_end] == next_speaker:
            run_end += 1
        search_end = min(run_end, index + 3)
        punctuation_index = next(
            (candidate for candidate in range(index, search_end) if ends_sentence(timestamp_items[candidate])),
            None,
        )
        if punctuation_index is None:
            index = run_end
            continue
        following_tokens = run_end - punctuation_index - 1
        boundary_start = float(timestamp_items[index].get("start", 0.0))
        punctuation_end = float(timestamp_items[punctuation_index].get("end", boundary_start))
        snap_duration = max(0.0, punctuation_end - boundary_start)
        if following_tokens < 2 or snap_duration > 1.20:
            index = run_end
            continue
        for candidate in range(index, punctuation_index + 1):
            snapped[candidate] = previous_speaker
        corrected += 1
        index = punctuation_index + 1
    if corrected:
        print(f"✅ RTTM 边界标点吸附: 修正 {corrected} 个句中提前换人边界。")
    return snapped


def split_text_by_weights(text, weights):
    if len(weights) <= 1:
        return [text.strip()]
    char_positions = [
        index for index, char in enumerate(text)
        if char == "'" or unicodedata.category(char).startswith(("L", "N"))
    ]
    total_weight = sum(weights)
    if not char_positions or total_weight <= 0:
        return [text.strip()] + [""] * (len(weights) - 1)
    boundaries = [0]
    consumed = 0
    for weight in weights[:-1]:
        consumed += weight
        normalized_index = min(
            len(char_positions) - 1,
            max(1, round(consumed / total_weight * len(char_positions))),
        )
        boundaries.append(char_positions[normalized_index])
    boundaries.append(len(text))
    return [text[boundaries[index]:boundaries[index + 1]].strip() for index in range(len(weights))]


def split_segment_on_speaker_turns(segment, timestamp_items, rttm_spans, fallback_speaker):
    if not timestamp_items:
        return [segment]
    item_speakers = [speaker_for_timestamp(item, rttm_spans, fallback_speaker) for item in timestamp_items]
    item_speakers = snap_speaker_turns_to_punctuation(timestamp_items, item_speakers)

    groups = []
    for item, speaker in zip(timestamp_items, item_speakers):
        if not groups or groups[-1]["speaker"] != speaker:
            groups.append({"speaker": speaker, "items": []})
        groups[-1]["items"].append(item)
    if len(groups) <= 1:
        return [segment]
    weights = []
    for group in groups:
        weight = sum(
            max(
                1,
                sum(
                    char == "'" or unicodedata.category(char).startswith(("L", "N"))
                    for char in str(item.get("text", ""))
                ),
            )
            for item in group["items"]
        )
        weights.append(weight)
    text_parts = split_text_by_weights(str(segment.get("text", "")), weights)
    raw_ranges = [
        (
            min(float(item.get("start", segment["start"])) for item in group["items"]),
            max(float(item.get("end", segment["end"])) for item in group["items"]),
        )
        for group in groups
    ]
    boundaries = [float(segment["start"])]
    for left, right in zip(raw_ranges, raw_ranges[1:]):
        boundaries.append((left[1] + right[0]) / 2.0)
    boundaries.append(float(segment["end"]))
    split_segments = []
    for index, (group, text_part) in enumerate(zip(groups, text_parts)):
        if not text_part:
            continue
        new_segment = segment.copy()
        new_segment["parent_segment_id"] = segment.get("id")
        new_segment["start"] = round(boundaries[index], 3)
        new_segment["end"] = round(boundaries[index + 1], 3)
        new_segment["text"] = text_part
        new_segment["speaker"] = group["speaker"]
        new_segment["speaker_split_by_rttm"] = True
        new_segment["qwen3_time_stamps"] = group["items"]
        split_segments.append(new_segment)
    return split_segments or [segment]


def fix_interjection_speakers(segments, rttm_spans):
    INTERJECTION_RE = re.compile(
        r"^(?:uh+h*|huh+|yeah+|yes+|ok+a*y*|what\?*|oh+|ah+|um+m*|hmm+|wow+|hey+|no+|right[.,]?|嗯|啊|哦|哎|呀|呢|吧|吗|哼|哈|呃|欸|啥|咋|喔|喏|呗|嘛)$",
        flags=re.IGNORECASE,
    )
    fixed = 0
    for seg in segments:
        text = str(seg.get("text", "")).strip()
        duration = float(seg["end"]) - float(seg["start"])
        if duration >= 1.2 or not INTERJECTION_RE.match(text):
            continue
        midpoint = (float(seg["start"]) + float(seg["end"])) / 2.0
        best_spk = None
        best_dist = float("inf")
        for span in rttm_spans:
            if float(span["start"]) <= midpoint <= float(span["end"]):
                best_spk = span["speaker"]
                break
            dist = min(abs(midpoint - float(span["start"])), abs(midpoint - float(span["end"])))
            if dist < best_dist:
                best_dist = dist
                best_spk = span["speaker"]
        if best_spk and normalize_speaker(best_spk) != seg["speaker"]:
            seg["speaker"] = normalize_speaker(best_spk)
            seg["speaker_fixed_by_midpoint"] = True
            fixed += 1
    if fixed:
        print(f"✅ 短语气词精确落点: 修正 {fixed} 个片段的 speaker 归属。")
    return segments


def merge_fragment_speakers(segments):
    total = sum(float(s["end"]) - float(s["start"]) for s in segments)
    threshold = max(1.5, total * 0.03)
    stats = defaultdict(lambda: {"duration": 0.0, "count": 0, "mid_time": 0.0})
    for seg in segments:
        spk = seg["speaker"]
        dur = float(seg["end"]) - float(seg["start"])
        stats[spk]["duration"] += dur
        stats[spk]["count"] += 1
        stats[spk]["mid_time"] += (float(seg["start"]) + float(seg["end"])) / 2 * dur
    for spk in stats:
        if stats[spk]["duration"] > 0:
            stats[spk]["mid_time"] /= stats[spk]["duration"]

    fragments = [
        spk for spk, info in stats.items()
        if info["duration"] < threshold or info["count"] < 2
    ]
    majors = [spk for spk in stats if spk not in fragments]

    if not fragments or not majors:
        return segments

    print(f"🧹 自适应碎片合并: 阈值 {threshold:.1f}s, {len(fragments)} 个碎片簇 -> 合并")
    for spk in fragments:
        print(f"   碎片 {spk}: {stats[spk]['count']} 段, {stats[spk]['duration']:.1f}s")

    for seg in segments:
        if seg["speaker"] in fragments:
            t = (float(seg["start"]) + float(seg["end"])) / 2
            nearest = min(majors, key=lambda spk: abs(stats[spk]["mid_time"] - t))
            seg["speaker"] = nearest
            seg["speaker_merged_from"] = seg.get("speaker", "")

    for new_id, segment in enumerate(segments):
        segment["id"] = new_id
    return segments


def merge_adjacent_same_speaker(segments, max_gap=0.5, min_merge_duration=1.5):
    """合并同一说话人、间隔极短的相邻片段，避免 TTS 碎裂"""
    if len(segments) < 2:
        return segments
    merged = [segments[0].copy()]
    for seg in segments[1:]:
        last = merged[-1]
        gap = float(seg["start"]) - float(last["end"])
        same_speaker = last["speaker"] == seg["speaker"]
        last_dur = float(last["end"]) - float(last["start"])
        seg_dur = float(seg["end"]) - float(seg["start"])
        should_merge = (
            same_speaker
            and gap <= max_gap
            and (last_dur < min_merge_duration or seg_dur < min_merge_duration)
        )
        if should_merge:
            last["end"] = seg["end"]
            last["text"] = (last.get("text", "") + " " + seg.get("text", "")).strip()
            if "qwen3_time_stamps" in last and "qwen3_time_stamps" in seg:
                last["qwen3_time_stamps"].extend(seg["qwen3_time_stamps"])
            last["speaker_merged_adjacent"] = True
        else:
            merged.append(seg.copy())
    for i, seg in enumerate(merged):
        seg["id"] = i
    if len(merged) < len(segments):
        print(f"✅ 同 speaker 相邻合并: {len(segments)} -> {len(merged)} 段")
    return merged


def assign_speakers(segments, rttm_spans):
    result = []
    split_count = 0
    for source_segment in segments:
        seg = source_segment.copy()
        start = float(seg["start"])
        end = float(seg["end"])
        overlaps = speaker_overlaps(start, end, rttm_spans)
        dominant = max(overlaps, key=overlaps.get) if overlaps else "SPEAKER_00"
        dominant = normalize_speaker(dominant)
        timestamp_items = list(seg.get("qwen3_time_stamps") or [])
        split_segments = split_segment_on_speaker_turns(seg, timestamp_items, rttm_spans, dominant)
        if len(split_segments) > 1:
            split_count += 1
            result.extend(split_segments)
            continue
        seg["speaker"] = dominant
        result.append(seg)

    for new_id, segment in enumerate(result):
        segment["id"] = new_id
    if split_count:
        print(f"✅ 根据 Qwen3 时间戳和 RTTM 换人边界拆分了 {split_count} 个 ASR 片段。")

    result = fix_interjection_speakers(result, rttm_spans)
    result = merge_fragment_speakers(result)
    result = merge_adjacent_same_speaker(result, max_gap=0.5, min_merge_duration=1.5)
    return result


def main():
    print("=== [Step 2] WeSpeaker + 自适应聚类 说话人分离 ===")
    print(f"当前 Python: {sys.executable}")
    state = load_state()
    if "segments_step1" not in state:
        print("❌ state 中缺少 segments_step1，请先运行 Step 1a/1b")
        sys.exit(1)

    prepare_inputs(state)
    rttm_path = run_wespeaker_pipeline()
    print("\n=== [Step 2-W] 解析 RTTM 回填 AutoDub Segments ===")
    rttm_spans = parse_rttm(rttm_path)
    aligned = assign_speakers(state["segments_step1"], rttm_spans)

    unique = sorted(set(s["speaker"] for s in aligned))
    print(f"🔍 最终识别出 {len(unique)} 个说话人: {unique}")

    state["segments_step2"] = aligned
    save_state(state)
    print("\n✅ Step 2 完成。请切回 autodub_server 环境继续运行 Step 3。")


if __name__ == "__main__":
    main()
