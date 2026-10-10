#!/usr/bin/env python3
"""
Step 2 WeSpeaker bridge.

Speech activity is detected from the complete vocal stem, independently of
ASR coverage. Qwen3 word gaps provide candidate boundaries, WeSpeaker verifies
local acoustic changes, and constrained turn clustering protects rare voices
without requiring a user-supplied speaker count.
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


def _env_float(name, default):
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return float(default)


def _env_int(name, default):
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return int(default)


SAD_MIN_SILENCE_MS = _env_int("AUTODUB_DIAR_SAD_MIN_SILENCE_MS", 350)
SAD_KEEP_MS = _env_int("AUTODUB_DIAR_SAD_KEEP_MS", 180)
SAD_MERGE_GAP_MS = _env_int("AUTODUB_DIAR_SAD_MERGE_GAP_MS", 450)
SAD_MIN_REGION_MS = _env_int("AUTODUB_DIAR_SAD_MIN_REGION_MS", 300)
SAD_DB_OFFSET = _env_float("AUTODUB_DIAR_SAD_DB_OFFSET", 18.0)
SAD_FLOOR_DBFS = _env_float("AUTODUB_DIAR_SAD_FLOOR_DBFS", -50.0)
BOUNDARY_CONTEXT_SECONDS = _env_float("AUTODUB_DIAR_BOUNDARY_CONTEXT_SECONDS", 2.3)
BOUNDARY_GUARD_SECONDS = _env_float("AUTODUB_DIAR_BOUNDARY_GUARD_SECONDS", 0.3)
BOUNDARY_MIN_TURN_SECONDS = _env_float("AUTODUB_DIAR_MIN_TURN_SECONDS", 0.8)
BOUNDARY_SCORE_OVERRIDE = os.getenv("AUTODUB_DIAR_BOUNDARY_SCORE")
STRONG_SPEAKER_SIMILARITY = _env_float("AUTODUB_DIAR_STRONG_SIMILARITY", 0.52)
WEAK_SPEAKER_SIMILARITY = _env_float("AUTODUB_DIAR_WEAK_SIMILARITY", 0.36)


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


def _merge_time_ranges(ranges, *, max_gap_ms):
    merged = []
    for start, end in sorted(ranges):
        if end <= start:
            continue
        if merged and start - merged[-1][1] <= max_gap_ms:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def detect_speech_regions(vocal_path):
    """Detect speech-bearing regions independently from the ASR transcript.

    These ranges are only an SAD mask for speaker embeddings.  They are not
    sentence boundaries, so merging a short pause cannot merge speaker labels.
    """
    try:
        import soundfile as sf

        if np is None:
            raise RuntimeError("numpy 不可用")
        samples, sample_rate = sf.read(vocal_path, dtype="float32")
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        if len(samples) <= 0 or sample_rate <= 0:
            return []

        frame_samples = max(1, round(sample_rate * 0.030))
        hop_samples = max(1, round(sample_rate * 0.010))
        starts = np.arange(0, len(samples), hop_samples, dtype=int)
        rms = np.asarray([
            math.sqrt(float(np.mean(np.square(samples[start:start + frame_samples]))) + 1e-12)
            for start in starts
        ])
        frame_dbfs = 20.0 * np.log10(np.maximum(rms, 1e-12))
        reference_rms = math.sqrt(float(np.mean(np.square(samples))) + 1e-12)
        reference_dbfs = 20.0 * math.log10(max(reference_rms, 1e-12))
        silence_thresh = max(SAD_FLOOR_DBFS, reference_dbfs - SAD_DB_OFFSET)
        active = np.flatnonzero(frame_dbfs >= silence_thresh)
        duration_ms = len(samples) / sample_rate * 1000.0
        if len(active):
            raw = []
            run_start = int(active[0])
            previous = run_start
            max_inactive_frames = max(1, round(SAD_MIN_SILENCE_MS / 10.0))
            for frame_index in map(int, active[1:]):
                if frame_index - previous > max_inactive_frames:
                    raw.append([run_start * 10, previous * 10 + 30])
                    run_start = frame_index
                previous = frame_index
            raw.append([run_start * 10, previous * 10 + 30])
        else:
            raw = []
        padded = [
            [max(0, start - SAD_KEEP_MS), min(duration_ms, end + SAD_KEEP_MS)]
            for start, end in raw
        ]
        merged = _merge_time_ranges(padded, max_gap_ms=SAD_MERGE_GAP_MS)
        regions = [
            (start / 1000.0, end / 1000.0)
            for start, end in merged
            if end - start >= SAD_MIN_REGION_MS
        ]
        if regions:
            covered = sum(end - start for start, end in regions)
            print(
                f"✅ 独立语音活动检测: {len(regions)} 段, "
                f"覆盖 {covered:.1f}s/{duration_ms / 1000.0:.1f}s, "
                f"阈值 {silence_thresh:.1f} dBFS"
            )
            return regions
        print("⚠️ 独立语音活动检测未找到有效区域，将使用完整人声音轨。")
        return [(0.0, duration_ms / 1000.0)]
    except Exception as exc:
        raise RuntimeError(f"无法对人声音轨执行独立语音活动检测: {exc}") from exc


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

    speech_regions = detect_speech_regions(vocal_path)
    with open(os.path.join(WORK_DIR, "oracle_sad"), "w", encoding="utf-8") as f:
        for s, e in speech_regions:
            seg_id = f"{utt_id}-{int(s*1000):08d}-{int(e*1000):08d}"
            f.write(f"{seg_id} {utt_id} {s:.3f} {e:.3f}\n")

    print(f"✅ WeSpeaker 输入已生成: {WORK_DIR}")
    return utt_id


def run_wespeaker_pipeline(segments_step1):
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

    print("\n=== [Step 2-W] 时序边界 + 稀有说话人保护聚类 ===")
    if not run_temporal_constrained_clustering(segments_step1):
        print("⚠️ 词级时间戳或边界证据不足，回退到 UMAP + HDBSCAN 聚类。")
        run_adaptive_clustering()

    print("\n=== [Step 2-W] WeSpeaker: 生成 RTTM ===")
    rttm_path = os.path.join(WORK_DIR, "result.rttm")
    run_cmd([
        "python3", "wespeaker/diar/make_rttm.py",
        "--labels", os.path.join(WORK_DIR, "labels"),
        "--channel", "1",
    ], env=env, cwd=WESPEAKER_EXAMPLE_DIR, stdout_file=rttm_path)

    return rttm_path


def _normalized_rows(values):
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def _unit_vector(value):
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def _embedding_window(key, value):
    """Decode WeSpeaker's SAD/subsegment key into an absolute time window."""
    match = re.match(r"^(.*)-(\d{8})-(\d{8})-(\d{8})-(\d{8})$", key)
    if not match:
        raise ValueError(f"无法解析 WeSpeaker embedding key: {key}")
    region_start = int(match.group(2)) / 1000.0
    region_end = int(match.group(3)) / 1000.0
    start = region_start + int(match.group(4)) / 100.0
    end = region_start + int(match.group(5)) / 100.0
    return {
        "key": key,
        "region": (match.group(1), region_start, region_end),
        "start": start,
        "end": end,
        "center": (start + end) / 2.0,
        "embedding": _unit_vector(value),
    }


def _read_embedding_windows(emb_scp):
    windows = []
    with kaldiio.ReadHelper(f"scp:{emb_scp}") as reader:
        for key, value in reader:
            windows.append(_embedding_window(key, value))
    windows.sort(key=lambda item: (item["center"], item["start"]))
    return windows


def _timestamp_items(segments):
    """Collect valid absolute word timestamps without inventing missing words."""
    items = []
    for segment in segments:
        for raw in segment.get("qwen3_time_stamps") or []:
            try:
                start = float(raw.get("start"))
                end = float(raw.get("end", start))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(start) or not math.isfinite(end):
                continue
            items.append({
                "start": start,
                "end": max(start, end),
                "text": str(raw.get("text", "")),
            })
    items.sort(key=lambda item: (item["start"], item["end"]))
    deduplicated = []
    for item in items:
        signature = (round(item["start"], 3), round(item["end"], 3), item["text"])
        if deduplicated and signature == deduplicated[-1][0]:
            continue
        deduplicated.append((signature, item))
    return [item for _, item in deduplicated]


def _centroid_for_centers(windows, start, end):
    values = [
        item["embedding"]
        for item in windows
        if start <= item["center"] <= end
    ]
    if not values:
        return None, 0
    return _unit_vector(np.mean(np.stack(values), axis=0)), len(values)


def _candidate_boundary_scores(segments, windows):
    items = _timestamp_items(segments)
    if len(items) < 2:
        return []
    candidates = []
    for left, right in zip(items, items[1:]):
        if right["start"] < left["start"]:
            continue
        boundary = (left["end"] + right["start"]) / 2.0
        left_center, left_count = _centroid_for_centers(
            windows,
            boundary - BOUNDARY_CONTEXT_SECONDS,
            boundary - BOUNDARY_GUARD_SECONDS,
        )
        right_center, right_count = _centroid_for_centers(
            windows,
            boundary + BOUNDARY_GUARD_SECONDS,
            boundary + BOUNDARY_CONTEXT_SECONDS,
        )
        if left_center is None or right_center is None:
            continue
        similarity = float(left_center @ right_center)
        candidates.append({
            "time": boundary,
            "score": 1.0 - similarity,
            "similarity": similarity,
            "gap": max(0.0, right["start"] - left["end"]),
            "left_count": left_count,
            "right_count": right_count,
            "left_text": left["text"],
            "right_text": right["text"],
        })
    return candidates


def _adaptive_boundary_threshold(candidates):
    if BOUNDARY_SCORE_OVERRIDE not in (None, ""):
        try:
            return float(BOUNDARY_SCORE_OVERRIDE)
        except ValueError:
            print(
                f"⚠️ 忽略无效 AUTODUB_DIAR_BOUNDARY_SCORE="
                f"{BOUNDARY_SCORE_OVERRIDE!r}"
            )
    values = np.asarray([item["score"] for item in candidates], dtype=float)
    ordered = np.sort(values)
    minimum_lower = max(2, int(math.ceil(len(ordered) * 0.20)))
    minimum_upper = max(2, int(math.ceil(len(ordered) * 0.10)))
    gaps = [
        (float(ordered[index + 1] - ordered[index]), index)
        for index in range(minimum_lower - 1, len(ordered) - minimum_upper)
    ]
    if gaps:
        natural_gap, gap_index = max(gaps)
        if natural_gap >= 0.06:
            midpoint = float((ordered[gap_index] + ordered[gap_index + 1]) / 2.0)
            return min(0.60, max(0.42, midpoint))
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    # The threshold follows each video's own score distribution.  The bounds
    # are model-level safety rails, not an assumed number of speakers.
    return min(0.62, max(0.44, median + max(0.06, 1.2 * mad)))


def _boundary_location(regions, timestamp):
    matches = [
        region for region in regions
        if region[1] + BOUNDARY_MIN_TURN_SECONDS
        <= timestamp
        <= region[2] - BOUNDARY_MIN_TURN_SECONDS
    ]
    if matches:
        return {
            "region": min(matches, key=lambda item: item[2] - item[1]),
            "between_regions": None,
        }
    for left, right in zip(regions, regions[1:]):
        if left[2] - 0.20 <= timestamp <= right[1] + 0.20:
            return {"region": None, "between_regions": (left, right)}
    return None


def _select_change_boundaries(candidates, windows):
    if not candidates:
        return [], None
    threshold = _adaptive_boundary_threshold(candidates)
    regions = sorted(set(item["region"] for item in windows), key=lambda item: item[1])
    eligible = []
    for candidate in candidates:
        location = _boundary_location(regions, candidate["time"])
        if location is None:
            continue
        required = threshold
        if candidate["gap"] >= 0.60:
            required -= 0.025
        if candidate["score"] >= required:
            enriched = dict(candidate)
            enriched.update(location)
            eligible.append(enriched)

    # Non-maximum suppression keeps one acoustic peak for one real change.
    selected = []
    for candidate in sorted(eligible, key=lambda item: item["score"], reverse=True):
        location_key = candidate["region"] or candidate["between_regions"]
        if any(
            (item["region"] or item["between_regions"]) == location_key
            and abs(item["time"] - candidate["time"])
            < BOUNDARY_MIN_TURN_SECONDS
            for item in selected
        ):
            continue
        selected.append(candidate)
    selected.sort(key=lambda item: item["time"])
    return selected, threshold


def _build_acoustic_turns(windows, boundaries):
    by_region = defaultdict(list)
    for index, window in enumerate(windows):
        by_region[window["region"]].append(index)
    boundary_times = defaultdict(list)
    for item in boundaries:
        if item["region"] is not None:
            boundary_times[item["region"]].append(float(item["time"]))

    turns = []
    cannot_links = set()
    turns_by_region = {}
    for region in sorted(by_region, key=lambda item: item[1]):
        cuts = [region[1], *sorted(boundary_times.get(region, [])), region[2]]
        region_turns = []
        for interval_index, (start, end) in enumerate(zip(cuts, cuts[1:])):
            is_last_interval = interval_index == len(cuts) - 2
            indices = [
                index for index in by_region[region]
                if start <= windows[index]["center"]
                and (
                    windows[index]["center"] < end
                    or (is_last_interval and windows[index]["center"] <= end)
                )
            ]
            if not indices:
                continue
            duration = end - start
            guard = min(BOUNDARY_GUARD_SECONDS, max(0.0, duration * 0.12))
            interior = [
                index for index in indices
                if start + guard <= windows[index]["center"] <= end - guard
            ]
            source = interior or indices
            embedding = _unit_vector(np.mean(np.stack([
                windows[index]["embedding"] for index in source
            ]), axis=0))
            turn_index = len(turns)
            turns.append({
                "start": start,
                "end": end,
                "duration": duration,
                "region": region,
                "window_indices": indices,
                "embedding": embedding,
            })
            region_turns.append(turn_index)
        for left, right in zip(region_turns, region_turns[1:]):
            cannot_links.add((min(left, right), max(left, right)))
        turns_by_region[region] = region_turns
    for item in boundaries:
        if item["between_regions"] is None:
            continue
        left_region, right_region = item["between_regions"]
        left_turns = turns_by_region.get(left_region) or []
        right_turns = turns_by_region.get(right_region) or []
        if left_turns and right_turns:
            left = left_turns[-1]
            right = right_turns[0]
            cannot_links.add((min(left, right), max(left, right)))
    return turns, cannot_links


def _cluster_centroid(cluster, turns):
    values = []
    weights = []
    for index in cluster:
        values.append(turns[index]["embedding"])
        weights.append(max(0.75, math.sqrt(max(turns[index]["duration"], 0.0))))
    return _unit_vector(np.average(np.stack(values), axis=0, weights=weights))


def _clusters_conflict(left, right, cannot_links):
    return any(
        (min(a, b), max(a, b)) in cannot_links
        for a in left
        for b in right
    )


def _merge_clusters(clusters, left_index, right_index):
    merged = set(clusters[left_index]) | set(clusters[right_index])
    return [
        cluster for index, cluster in enumerate(clusters)
        if index not in (left_index, right_index)
    ] + [merged]


def _constrained_turn_clusters(turns, cannot_links):
    clusters = [{index} for index in range(len(turns))]

    # First join only high-confidence identities in the original 256-D space.
    while True:
        centroids = [_cluster_centroid(cluster, turns) for cluster in clusters]
        best = None
        for left in range(len(clusters)):
            for right in range(left + 1, len(clusters)):
                if _clusters_conflict(clusters[left], clusters[right], cannot_links):
                    continue
                similarity = float(centroids[left] @ centroids[right])
                if similarity < STRONG_SPEAKER_SIMILARITY:
                    continue
                if best is None or similarity > best[0]:
                    best = (similarity, left, right)
        if best is None:
            break
        clusters = _merge_clusters(clusters, best[1], best[2])

    # Complete the partition as a constrained graph-colouring problem.  A
    # strong acoustic change creates a hard edge.  Very dissimilar residual
    # identities also create an edge.  DSATUR then uses the fewest identities
    # it can without breaking those edges, so a speaker with only one short
    # line is not discarded merely because it cannot form a density cluster.
    centroids = [_cluster_centroid(cluster, turns) for cluster in clusters]
    conflicts = [set() for _ in clusters]
    for left in range(len(clusters)):
        for right in range(left + 1, len(clusters)):
            similarity = float(centroids[left] @ centroids[right])
            if (
                _clusters_conflict(clusters[left], clusters[right], cannot_links)
                or similarity < WEAK_SPEAKER_SIMILARITY
            ):
                conflicts[left].add(right)
                conflicts[right].add(left)

    colors = {}
    while len(colors) < len(clusters):
        uncolored = [index for index in range(len(clusters)) if index not in colors]

        def priority(index):
            neighbor_colors = {colors[item] for item in conflicts[index] if item in colors}
            duration = sum(turns[item]["duration"] for item in clusters[index])
            return (len(neighbor_colors), len(conflicts[index]), duration)

        node = max(uncolored, key=priority)
        used = sorted(set(colors.values()))
        available = [
            color for color in used
            if all(
                other not in conflicts[node]
                for other, other_color in colors.items()
                if other_color == color
            )
        ]
        if available:
            def color_similarity(color):
                members = [index for index, value in colors.items() if value == color]
                return float(np.mean([
                    centroids[node] @ centroids[index] for index in members
                ]))

            colors[node] = max(available, key=color_similarity)
        else:
            colors[node] = max(used, default=-1) + 1

    grouped = defaultdict(set)
    for node, color in colors.items():
        grouped[color].update(clusters[node])
    return list(grouped.values())


def run_temporal_constrained_clustering(segments):
    """Cluster speaker turns while preserving local acoustic change evidence."""
    if not kaldiio or np is None:
        return False
    emb_scp = os.path.join(WORK_DIR, "embedding", "emb.scp")
    labels_out = os.path.join(WORK_DIR, "labels")
    windows = _read_embedding_windows(emb_scp)
    if len(windows) < 4:
        return False
    candidates = _candidate_boundary_scores(segments, windows)
    boundaries, threshold = _select_change_boundaries(candidates, windows)
    if not boundaries:
        return False

    print(
        f"✅ 自适应换人阈值: {threshold:.3f}; "
        f"{len(candidates)} 个词间候选中保留 {len(boundaries)} 个。"
    )
    for item in boundaries:
        print(
            f"   换人候选 {item['time']:.3f}s: "
            f"change={item['score']:.3f}, gap={item['gap']:.3f}s, "
            f"{item['left_text']!r} -> {item['right_text']!r}"
        )

    turns, cannot_links = _build_acoustic_turns(windows, boundaries)
    if len(turns) < 2:
        return False
    clusters = _constrained_turn_clusters(turns, cannot_links)
    clusters.sort(key=lambda cluster: min(turns[index]["start"] for index in cluster))
    turn_labels = {}
    for label, cluster in enumerate(clusters):
        for turn_index in cluster:
            turn_labels[turn_index] = label

    window_labels = {}
    for turn_index, turn in enumerate(turns):
        for window_index in turn["window_indices"]:
            window_labels[window_index] = turn_labels[turn_index]
    if len(window_labels) != len(windows):
        return False

    os.makedirs(os.path.dirname(labels_out), exist_ok=True)
    with open(labels_out, "w", encoding="utf-8") as stream:
        for index, window in enumerate(windows):
            stream.write(f"{window['key']} {window_labels[index]}\n")

    print(
        f"✅ 时序约束聚类: {len(turns)} 个声学话轮 -> "
        f"{len(clusters)} 位说话人（允许稀有说话人保留）。"
    )
    return True


def _fill_noise_labels(values, labels):
    """Attach HDBSCAN noise windows to their nearest stable speaker."""
    labels = np.asarray(labels, dtype=int).copy()
    stable = sorted(set(labels) - {-1})
    if not stable:
        return np.zeros(len(labels), dtype=int)
    normalized = _normalized_rows(values)
    centroids = []
    for label in stable:
        centroid = normalized[labels == label].mean(axis=0)
        centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
        centroids.append(centroid)
    for index in np.flatnonzero(labels == -1):
        labels[index] = stable[int(np.argmax(np.asarray(centroids) @ normalized[index]))]
    return labels


def score_clustering(values, raw_labels):
    """Score a candidate without knowing the number of speakers in advance."""
    raw_labels = np.asarray(raw_labels, dtype=int)
    noise_ratio = float(np.mean(raw_labels == -1)) if len(raw_labels) else 1.0
    labels = _fill_noise_labels(values, raw_labels)
    unique = sorted(set(labels.tolist()))
    normalized = _normalized_rows(values)
    centroids = {}
    within_scores = []
    small_windows = 0
    for label in unique:
        mask = labels == label
        count = int(mask.sum())
        centroid = normalized[mask].mean(axis=0)
        centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
        centroids[label] = centroid
        within_scores.extend((normalized[mask] @ centroid).tolist())
        if count < max(2, math.ceil(len(labels) * 0.04)):
            small_windows += count

    cohesion = float(np.mean(within_scores)) if within_scores else 0.0
    if len(unique) > 1:
        similarities = [
            float(centroids[left] @ centroids[right])
            for pos, left in enumerate(unique)
            for right in unique[pos + 1:]
        ]
        separation = 1.0 - max(similarities)
    else:
        separation = 0.0

    switches = int(np.sum(labels[1:] != labels[:-1])) if len(labels) > 1 else 0
    switch_ratio = switches / max(1, len(labels) - 1)
    short_islands = 0
    run_start = 0
    for index in range(1, len(labels) + 1):
        if index == len(labels) or labels[index] != labels[run_start]:
            if index - run_start <= 1:
                short_islands += 1
            run_start = index
    island_ratio = short_islands / max(1, len(labels))
    small_ratio = small_windows / max(1, len(labels))

    # Reward compact and separated voices, but penalize unstable timelines,
    # noise, tiny clusters, and unnecessary model complexity.
    score = (
        cohesion
        + 0.70 * separation
        - 0.45 * switch_ratio
        - 0.60 * island_ratio
        - 0.50 * noise_ratio
        - 0.45 * small_ratio
        - 0.025 * max(0, len(unique) - 1)
    )
    metrics = {
        "score": score,
        "speakers": len(unique),
        "cohesion": cohesion,
        "separation": separation,
        "switch_ratio": switch_ratio,
        "noise_ratio": noise_ratio,
        "small_ratio": small_ratio,
    }
    return score, labels, metrics


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

            if n < 4:
                print(f"⚠️ 仅有 {n} 个声纹窗口，按单说话人保守处理。")
                for key in utt_keys[utt]:
                    f.write(f"{key} 0\n")
                continue

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
            reduced_cache = {}
            for strategy in strategies:
                try:
                    reducer_key = (min(15, max(2, n // 5)), min(32, max(2, n - 2)))
                    if reducer_key not in reduced_cache:
                        reducer = umap.UMAP(
                            n_components=reducer_key[1],
                            metric="cosine",
                            n_neighbors=min(n - 1, reducer_key[0]),
                            min_dist=0.0,
                            random_state=42,
                            n_jobs=1,
                        )
                        reduced_cache[reducer_key] = reducer.fit_transform(X)
                    emb_2d = reduced_cache[reducer_key]

                    clusterer = hdbscan.HDBSCAN(
                        min_cluster_size=strategy["mcs"],
                        min_samples=strategy["ms"],
                        cluster_selection_method=strategy["method"],
                        allow_single_cluster=True,
                        core_dist_n_jobs=1,
                    )
                    raw_labels = clusterer.fit_predict(emb_2d)
                    score, labels, metrics = score_clustering(X, raw_labels)
                    print(
                        f"  尝试 {strategy}: {metrics['speakers']} 人, "
                        f"评分={score:.3f}, 类内={metrics['cohesion']:.3f}, "
                        f"类间={metrics['separation']:.3f}, "
                        f"切换率={metrics['switch_ratio']:.3f}, "
                        f"噪声率={metrics['noise_ratio']:.3f}"
                    )
                    results.append((score, strategy, labels, metrics))
                except Exception as e:
                    print(f"  ⚠️ 策略 {strategy} 失败: {e}")
                    continue

            # Always compare against the conservative one-speaker explanation.
            single_score, single_labels, single_metrics = score_clustering(
                X, np.zeros(n, dtype=int)
            )
            results.append((single_score, {"method": "single"}, single_labels, single_metrics))
            if not results:
                raise RuntimeError("没有可用的声纹聚类结果")
            best_score, best_strategy, best_labels, best_metrics = max(
                results, key=lambda item: item[0]
            )

            print(
                f"✅ 自动选择: {best_metrics['speakers']} 位说话人, "
                f"策略={best_strategy}, 评分={best_score:.3f}"
            )

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
    from speaker_alignment import smooth_rttm_spans

    return smooth_rttm_spans(spans)


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
    from speaker_alignment import align_segments_with_rttm, merge_adjacent_speaker_turns

    result = align_segments_with_rttm(segments, rttm_spans)
    split_count = max(0, len(result) - len(segments))
    ambiguous_count = sum(
        bool(item.get("speaker_boundary_ambiguous")) for item in result
    )
    if split_count:
        print(f"✅ 句子边界与 RTTM 联合分段: 新增 {split_count} 个片段。")
    if ambiguous_count:
        print(f"⚠️ {ambiguous_count} 个片段的说话人边界证据不足，已保守保留整段。")
    merged = merge_adjacent_speaker_turns(result)
    if len(merged) < len(result):
        print(f"✅ 同说话人片段重组: {len(result)} -> {len(merged)} 段。")
    return merged


def main():
    print("=== [Step 2] WeSpeaker + 时序约束聚类 说话人分离 ===")
    print(f"当前 Python: {sys.executable}")
    state = load_state()
    if "segments_step1" not in state:
        print("❌ state 中缺少 segments_step1，请先运行 Step 1a/1b")
        sys.exit(1)

    prepare_inputs(state)
    rttm_path = run_wespeaker_pipeline(state["segments_step1"])
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
