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


def _env_bool(name, default):
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() not in {"0", "false", "no", "off"}


SAD_MIN_SILENCE_MS = _env_int("AUTODUB_DIAR_SAD_MIN_SILENCE_MS", 350)
SAD_KEEP_MS = _env_int("AUTODUB_DIAR_SAD_KEEP_MS", 180)
SAD_MERGE_GAP_MS = _env_int("AUTODUB_DIAR_SAD_MERGE_GAP_MS", 450)
SAD_MIN_REGION_MS = _env_int("AUTODUB_DIAR_SAD_MIN_REGION_MS", 300)
SAD_DB_OFFSET = _env_float("AUTODUB_DIAR_SAD_DB_OFFSET", 18.0)
SAD_FLOOR_DBFS = _env_float("AUTODUB_DIAR_SAD_FLOOR_DBFS", -50.0)
BOUNDARY_CONTEXT_SECONDS = _env_float("AUTODUB_DIAR_BOUNDARY_CONTEXT_SECONDS", 2.3)
BOUNDARY_MIN_CONTEXT_SECONDS = _env_float(
    "AUTODUB_DIAR_BOUNDARY_MIN_CONTEXT_SECONDS", 1.10
)
BOUNDARY_GUARD_SECONDS = _env_float("AUTODUB_DIAR_BOUNDARY_GUARD_SECONDS", 0.3)
BOUNDARY_MIN_TURN_SECONDS = _env_float("AUTODUB_DIAR_MIN_TURN_SECONDS", 0.8)
BOUNDARY_NMS_SECONDS = _env_float("AUTODUB_DIAR_BOUNDARY_NMS_SECONDS", 1.15)
ACOUSTIC_PERSISTENCE_MIN_POINTS = _env_int(
    "AUTODUB_DIAR_ACOUSTIC_PERSISTENCE_MIN_POINTS", 2
)
ACOUSTIC_TEXT_SPLIT_MIN_SCORE = _env_float(
    "AUTODUB_DIAR_ACOUSTIC_TEXT_SPLIT_MIN_SCORE", 0.72
)
ACOUSTIC_IDENTITY_SEPARATION_MIN_SCORE = _env_float(
    "AUTODUB_DIAR_ACOUSTIC_IDENTITY_SEPARATION_MIN_SCORE", 0.62
)
BOUNDARY_SCORE_OVERRIDE = os.getenv("AUTODUB_DIAR_BOUNDARY_SCORE")
STRONG_SPEAKER_SIMILARITY = _env_float("AUTODUB_DIAR_STRONG_SIMILARITY", 0.68)
WEAK_SPEAKER_SIMILARITY = _env_float("AUTODUB_DIAR_WEAK_SIMILARITY", 0.36)
PROTOTYPE_MERGE_SIMILARITY = _env_float(
    "AUTODUB_DIAR_PROTOTYPE_MERGE_SIMILARITY", 0.60
)
LOW_QUALITY_ASSIGN_SIMILARITY = _env_float(
    "AUTODUB_DIAR_LOW_QUALITY_ASSIGN_SIMILARITY", 0.36
)
LOW_QUALITY_ASSIGN_MARGIN = _env_float(
    "AUTODUB_DIAR_LOW_QUALITY_ASSIGN_MARGIN", 0.05
)
SENTENCE_CONTINUITY_SIMILARITY = _env_float(
    "AUTODUB_DIAR_SENTENCE_CONTINUITY_SIMILARITY", 0.42
)
HARD_BOUNDARY_SCORE_MARGIN = _env_float(
    "AUTODUB_DIAR_HARD_BOUNDARY_SCORE_MARGIN", 0.08
)
HARD_BOUNDARY_MIN_WINDOWS = _env_int(
    "AUTODUB_DIAR_HARD_BOUNDARY_MIN_WINDOWS", 3
)
HARD_BOUNDARY_MIN_COHESION = _env_float(
    "AUTODUB_DIAR_HARD_BOUNDARY_MIN_COHESION", 0.58
)
IDENTITY_CROSS_MIN_QUANTILE = _env_float(
    "AUTODUB_DIAR_IDENTITY_CROSS_MIN_QUANTILE", 0.40
)
PROTOTYPE_ANCHOR_MIN_SECONDS = _env_float(
    "AUTODUB_DIAR_PROTOTYPE_ANCHOR_MIN_SECONDS", 4.0
)
PROTOTYPE_ANCHOR_MIN_TURNS = _env_int(
    "AUTODUB_DIAR_PROTOTYPE_ANCHOR_MIN_TURNS", 2
)
SOFT_BOUNDARY_MIN_PENALTY = _env_float(
    "AUTODUB_DIAR_SOFT_BOUNDARY_MIN_PENALTY", 0.025
)
SOFT_BOUNDARY_MAX_PENALTY = _env_float(
    "AUTODUB_DIAR_SOFT_BOUNDARY_MAX_PENALTY", 0.10
)
SINGLE_TURN_PROTOTYPE_PENALTY = _env_float(
    "AUTODUB_DIAR_SINGLE_TURN_PROTOTYPE_PENALTY", 0.035
)
NON_MUTUAL_PROTOTYPE_PENALTY = _env_float(
    "AUTODUB_DIAR_NON_MUTUAL_PROTOTYPE_PENALTY", 0.035
)
SANDWICH_SINGLETON_PENALTY = _env_float(
    "AUTODUB_DIAR_SANDWICH_SINGLETON_PENALTY", 0.12
)
IDENTITY_CONSISTENCY_MARGIN = _env_float(
    "AUTODUB_DIAR_IDENTITY_CONSISTENCY_MARGIN", 0.13
)
ADAPTIVE_SEED_MIN = _env_float(
    "AUTODUB_DIAR_ADAPTIVE_SEED_MIN", 0.66
)
ADAPTIVE_SEED_MAX = _env_float(
    "AUTODUB_DIAR_ADAPTIVE_SEED_MAX", 0.76
)
ADAPTIVE_RECONNECT_MIN = _env_float(
    "AUTODUB_DIAR_ADAPTIVE_RECONNECT_MIN", 0.48
)
ADAPTIVE_RECONNECT_MAX = _env_float(
    "AUTODUB_DIAR_ADAPTIVE_RECONNECT_MAX", 0.68
)
ADAPTIVE_ASSIGNMENT_MIN = _env_float(
    "AUTODUB_DIAR_ADAPTIVE_ASSIGNMENT_MIN", 0.40
)
ADAPTIVE_ASSIGNMENT_MAX = _env_float(
    "AUTODUB_DIAR_ADAPTIVE_ASSIGNMENT_MAX", 0.56
)
MAX_COMBINED_MERGE_PENALTY = _env_float(
    "AUTODUB_DIAR_MAX_COMBINED_MERGE_PENALTY", 0.10
)
MULTI_PROTOTYPE_MAX_COUNT = _env_int(
    "AUTODUB_DIAR_MULTI_PROTOTYPE_MAX_COUNT", 3
)
MULTI_PROTOTYPE_COVERAGE = _env_float(
    "AUTODUB_DIAR_MULTI_PROTOTYPE_COVERAGE", 0.67
)
FRAGMENTATION_SINGLETON_RATIO = _env_float(
    "AUTODUB_DIAR_FRAGMENTATION_SINGLETON_RATIO", 0.55
)
FRAGMENTATION_RELAXATION = _env_float(
    "AUTODUB_DIAR_FRAGMENTATION_RELAXATION", 0.04
)
STRUCTURAL_SOFT_CHANGE_MIN = _env_float(
    "AUTODUB_DIAR_STRUCTURAL_SOFT_CHANGE_MIN", 0.44
)
RELIABLE_TURN_MIN_SECONDS = _env_float(
    "AUTODUB_DIAR_RELIABLE_TURN_MIN_SECONDS", 1.8
)
RELIABLE_TURN_MIN_WINDOWS = _env_int(
    "AUTODUB_DIAR_RELIABLE_TURN_MIN_WINDOWS", 2
)
RELIABLE_TURN_MIN_COHESION = _env_float(
    "AUTODUB_DIAR_RELIABLE_TURN_MIN_COHESION", 0.55
)
HARD_SILENCE_SECONDS = _env_float(
    "AUTODUB_HARD_SILENCE_SECONDS", 0.75
)
CLUSTERING_DIAGNOSTICS_PATH = os.path.join(
    WORK_DIR, "speaker_clustering_diagnostics.json"
)
PAIRWISE_LABELS_PATH = os.path.join(WORK_DIR, "labels_wespeaker_pairwise")
PAIRWISE_RTTM_PATH = os.path.join(WORK_DIR, "result_wespeaker_pairwise.rttm")
CLUSTER_COMPARISON_PATH = os.path.join(
    WORK_DIR, "speaker_clustering_comparison.json"
)

DOWNSTREAM_STATE_KEYS = (
    "speaker_clustering_comparison",
    "segments_step2_wespeaker_pairwise",
    "segments_emotion",
    "segments_audio_analyzed",
    "segments_step3",
    "segments_scheme3",
    "segments_json_path",
    "audio_clips",
    "segments_translation_candidates",
    "translation_candidates_fingerprint",
    "light_tts_selector_results",
    "light_tts_selector_results_path",
    "emotion_results",
    "tts_router_results",
    "tts_backend",
    "tts_stage_manifest",
    "tts_routing_summary",
    "tts_routing_report_path",
    "asr_tts_duration_blocked_ids",
)


def invalidate_downstream_state(state):
    """Remove products whose segment IDs/timing depend on Step 2 output."""
    removed = []
    for key in DOWNSTREAM_STATE_KEYS:
        if key in state:
            state.pop(key, None)
            removed.append(key)
    return removed


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


def detect_speech_regions(vocal_path):
    """Detect speech-bearing regions independently from the ASR transcript.

    These ranges are only an SAD mask for speaker embeddings.  They are not
    sentence boundaries, so merging a short pause cannot merge speaker labels.
    """
    try:
        from speech_activity import detect_speech_regions as shared_detect

        regions, diagnostics = shared_detect(
            vocal_path,
            min_silence_ms=SAD_MIN_SILENCE_MS,
            keep_ms=SAD_KEEP_MS,
            merge_gap_ms=SAD_MERGE_GAP_MS,
            min_region_ms=SAD_MIN_REGION_MS,
            db_offset=SAD_DB_OFFSET,
            floor_dbfs=SAD_FLOOR_DBFS,
        )
        if regions:
            covered = sum(end - start for start, end in regions)
            print(
                f"✅ 独立语音活动检测: {len(regions)} 段, "
                f"覆盖 {covered:.1f}s/{diagnostics['duration']:.1f}s, "
                f"阈值 {diagnostics['threshold_dbfs']:.1f} dBFS"
            )
            return regions
        print("⚠️ 独立语音活动检测未找到有效区域。")
        # Preserve the historical WeSpeaker behaviour for silent or unusual
        # files.  Step 1 coverage auditing intentionally does not use this
        # fallback because a full-duration false positive would restore noise.
        duration = float(diagnostics.get("duration", 0.0))
        return [(0.0, duration)] if duration > 0.0 else []
    except Exception as exc:
        raise RuntimeError(f"无法对人声音轨执行独立语音活动检测: {exc}") from exc


def prepare_inputs(state):
    os.makedirs(WORK_DIR, exist_ok=True)
    if os.path.exists(CLUSTERING_DIAGNOSTICS_PATH):
        os.unlink(CLUSTERING_DIAGNOSTICS_PATH)
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

    cluster_mode = os.getenv(
        "AUTODUB_DIAR_CLUSTER_MODE", "compare"
    ).strip().lower()
    if cluster_mode not in {"repaired", "pairwise", "compare"}:
        raise ValueError(
            "AUTODUB_DIAR_CLUSTER_MODE must be repaired, pairwise, or compare"
        )
    # Diagnostics describe one concrete label run.  Never let a previous
    # repaired run remap a fresh pairwise/adaptive baseline.
    if os.path.isfile(CLUSTERING_DIAGNOSTICS_PATH):
        os.remove(CLUSTERING_DIAGNOSTICS_PATH)

    if cluster_mode == "pairwise":
        print("\n=== [Step 2-W] WeSpeaker 两两余弦平均链接聚类 ===")
        run_wespeaker_pairwise_clustering(
            os.path.join(WORK_DIR, "labels")
        )
    else:
        print("\n=== [Step 2-W] 时序边界 + 稀有说话人保护聚类 ===")
        if not run_temporal_constrained_clustering(segments_step1):
            print("⚠️ 词级时间戳或边界证据不足，回退到 UMAP + HDBSCAN 聚类。")
            run_adaptive_clustering()
        if cluster_mode == "compare":
            shutil.copy2(
                os.path.join(WORK_DIR, "labels"),
                os.path.join(WORK_DIR, "labels_repaired"),
            )
            print("\n=== [Step 2-W] 旁路生成 WeSpeaker 两两聚类基线 ===")
            run_wespeaker_pairwise_clustering(PAIRWISE_LABELS_PATH)
            run_cmd([
                "python3", "wespeaker/diar/make_rttm.py",
                "--labels", PAIRWISE_LABELS_PATH,
                "--channel", "1",
            ], env=env, cwd=WESPEAKER_EXAMPLE_DIR, stdout_file=PAIRWISE_RTTM_PATH)

    print("\n=== [Step 2-W] WeSpeaker: 生成 RTTM ===")
    rttm_path = os.path.join(WORK_DIR, "result.rttm")
    run_cmd([
        "python3", "wespeaker/diar/make_rttm.py",
        "--labels", os.path.join(WORK_DIR, "labels"),
        "--channel", "1",
    ], env=env, cwd=WESPEAKER_EXAMPLE_DIR, stdout_file=rttm_path)

    return rttm_path


def run_wespeaker_pairwise_clustering(labels_out):
    """Legacy pairwise baseline on the exact same WeSpeaker embeddings.

    This mirrors the repository's former cosine/average-linkage path.  It is
    deliberately free of word-boundary and temporal cannot-link constraints,
    making it a clean A/B baseline for the repaired clustering path.
    """
    if kaldiio is None or np is None:
        raise RuntimeError("WeSpeaker 两两聚类需要 kaldiio 和 numpy")
    from sklearn.cluster import AgglomerativeClustering

    distance_threshold = _env_float("SPEAKER_CLUSTER_THRESHOLD", 0.5)
    emb_scp = os.path.join(WORK_DIR, "embedding", "emb.scp")
    utt_embs = defaultdict(list)
    utt_keys = defaultdict(list)
    with kaldiio.ReadHelper(f"scp:{emb_scp}") as reader:
        for key, matrix in reader:
            utterance = key.split("-")[0]
            embedding = matrix.mean(axis=0) if matrix.ndim == 2 else matrix
            utt_embs[utterance].append(embedding)
            utt_keys[utterance].append(key)

    os.makedirs(os.path.dirname(labels_out), exist_ok=True)
    with open(labels_out, "w", encoding="utf-8") as stream:
        for utterance, embeddings in utt_embs.items():
            values = _normalized_rows(np.stack(embeddings))
            if len(values) == 1:
                labels = np.zeros(1, dtype=int)
            else:
                kwargs = {
                    "n_clusters": None,
                    "linkage": "average",
                    "distance_threshold": distance_threshold,
                }
                try:
                    labels = AgglomerativeClustering(
                        metric="cosine", **kwargs
                    ).fit_predict(values)
                except TypeError:
                    labels = AgglomerativeClustering(
                        affinity="cosine", **kwargs
                    ).fit_predict(values)
            for key, label in zip(utt_keys[utterance], labels):
                stream.write(f"{key} {int(label)}\n")
            print(
                f"  {utterance}: {len(values)} 个窗口，"
                f"两两聚类得到 {len(set(int(value) for value in labels))} 人，"
                f"distance_threshold={distance_threshold:.3f}"
            )


def _normalized_rows(values):
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def _unit_vector(value):
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def _robust_embedding_pool(values):
    """Aggregate windows without letting one noisy window move the identity."""
    values = _normalized_rows(np.asarray(values, dtype=np.float32))
    if len(values) <= 2:
        return _unit_vector(np.mean(values, axis=0)), len(values)
    similarities = values @ values.T
    medoid_index = int(np.argmax(np.median(similarities, axis=1)))
    support = similarities[medoid_index]
    floor = max(0.20, float(np.quantile(support, 0.25)))
    retained = values[support >= floor]
    if len(retained) < 2:
        retained = values[[medoid_index]]
    weights = np.clip(retained @ values[medoid_index], 0.05, 1.0)
    pooled = np.average(retained, axis=0, weights=weights)
    return _unit_vector(pooled), int(len(retained))


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
    """Collect valid timestamps and recover punctuation from the transcript."""
    from speaker_alignment import normalize_timestamp_items

    items = []
    for segment_index, segment in enumerate(segments):
        if segment.get("speaker_boundary_eligible") is False:
            # Coarse ASR recovery keeps safe outer speech bounds only.  It may
            # receive a speaker by RTTM overlap later, but must never create a
            # word-level speaker-change constraint.
            continue
        use_timing_skeleton = bool(
            segment.get("speaker_timing_skeleton_valid")
            and segment.get("speaker_timing_time_stamps")
        )
        transcript = str(
            segment.get("speaker_timing_text")
            if use_timing_skeleton else segment.get("text", "")
        )
        raw_items = list(
            segment.get("speaker_timing_time_stamps")
            if use_timing_skeleton else segment.get("qwen3_time_stamps") or []
        )
        try:
            chunk_start = float(segment.get("start", 0.0))
            chunk_end = float(segment.get("end", chunk_start))
        except (TypeError, ValueError):
            continue
        normalized, alignment = normalize_timestamp_items(
            raw_items,
            transcript=transcript,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
        )
        # Broken forced-alignment output (for example, most words collapsed
        # onto the chunk endpoint) is audit evidence only.  It must never
        # create temporal speaker-change constraints.
        if alignment.get("status") == "unusable":
            continue
        for item in normalized:
            items.append({
                "start": float(item["start"]),
                "end": float(item["end"]),
                "text": str(item.get("text", "")),
                "punctuation_after": str(item.get("punctuation_after", "none")),
                "segment_index": segment_index,
                "chunk_start": chunk_start,
                "chunk_end": chunk_end,
                "timestamp_status": alignment.get("status", "unusable"),
                "timing_source": (
                    "pre_ocr_timing_skeleton"
                    if use_timing_skeleton else "current_segment"
                ),
            })
    items.sort(key=lambda item: (item["start"], item["end"]))
    deduplicated = []
    for item in items:
        signature = (round(item["start"], 3), round(item["end"], 3), item["text"])
        if deduplicated and signature == deduplicated[-1][0]:
            continue
        deduplicated.append((signature, item))
    return [item for _, item in deduplicated]


def _centroid_for_centers(windows, start, end, *, boundary=None, side=None):
    values = [
        item["embedding"]
        for item in windows
        if start <= item["center"] <= end
        # A WeSpeaker window may have its centre on the requested side while
        # still containing speech from the other side.  Do not let such a
        # cross-boundary window manufacture or hide a speaker change.
        and (
            boundary is None
            or side == "left" and float(item["end"]) <= boundary + 0.05
            or side == "right" and float(item["start"]) >= boundary - 0.05
        )
    ]
    if not values:
        return None, 0, 0.0
    stacked = np.stack(values)
    centroid = _unit_vector(np.mean(stacked, axis=0))
    cohesion = float(np.mean(stacked @ centroid))
    return centroid, len(values), cohesion


def _adaptive_context_seconds(boundary, left, right):
    """Limit acoustic context to the locally supported transcript range."""
    available_left = max(0.0, boundary - float(left.get("chunk_start", boundary)))
    available_right = max(0.0, float(right.get("chunk_end", boundary)) - boundary)
    available = min(available_left, available_right)
    if left.get("segment_index") != right.get("segment_index"):
        # Transport-chunk boundaries are especially vulnerable to pulling in
        # a neighbouring voice, so keep their context deliberately local.
        available = min(available, 1.60)
    return min(
        BOUNDARY_CONTEXT_SECONDS,
        max(BOUNDARY_MIN_CONTEXT_SECONDS, available),
    )


def _candidate_boundary_scores(segments, windows):
    items = _timestamp_items(segments)
    if len(items) < 2:
        return []
    candidates = []
    for left, right in zip(items, items[1:]):
        if right["start"] < left["start"]:
            continue
        boundary = (left["end"] + right["start"]) / 2.0
        context_seconds = _adaptive_context_seconds(boundary, left, right)
        left_center, left_count, left_cohesion = _centroid_for_centers(
            windows,
            boundary - context_seconds,
            boundary - BOUNDARY_GUARD_SECONDS,
            boundary=boundary,
            side="left",
        )
        right_center, right_count, right_cohesion = _centroid_for_centers(
            windows,
            boundary + BOUNDARY_GUARD_SECONDS,
            boundary + context_seconds,
            boundary=boundary,
            side="right",
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
            "left_cohesion": left_cohesion,
            "right_cohesion": right_cohesion,
            "context_seconds": context_seconds,
            "left_text": left["text"],
            "right_text": right["text"],
            "punctuation": left.get("punctuation_after", "none"),
            "left_segment_index": left.get("segment_index"),
            "right_segment_index": right.get("segment_index"),
            "same_asr_segment": (
                left.get("segment_index") == right.get("segment_index")
            ),
            "timestamp_reliable": (
                left.get("timestamp_status") == "valid"
                and right.get("timestamp_status") == "valid"
            ),
        })
    return candidates


def _acoustic_boundary_scores(segments, windows):
    """Propose high-contrast boundaries without requiring word timestamps.

    OCR rewrites and failed alignments can remove textual timing even though
    speaker embeddings still contain a clear local change.  These candidates
    remain soft evidence and must pass a stricter acoustic gate.
    """
    candidates = []
    by_region = defaultdict(list)
    for window in windows:
        by_region[window["region"]].append(window)
    for region_windows in by_region.values():
        ordered = sorted(region_windows, key=lambda item: item["center"])
        region_candidates = []
        for left_window, right_window in zip(ordered, ordered[1:]):
            boundary = (
                float(left_window["center"]) + float(right_window["center"])
            ) / 2.0
            left_center, left_count, left_cohesion = _centroid_for_centers(
                ordered,
                boundary - BOUNDARY_CONTEXT_SECONDS,
                boundary - BOUNDARY_GUARD_SECONDS,
                boundary=boundary,
                side="left",
            )
            right_center, right_count, right_cohesion = _centroid_for_centers(
                ordered,
                boundary + BOUNDARY_GUARD_SECONDS,
                boundary + BOUNDARY_CONTEXT_SECONDS,
                boundary=boundary,
                side="right",
            )
            if left_center is None or right_center is None:
                continue
            containing = [
                (index, segment)
                for index, segment in enumerate(segments)
                if float(segment.get("start", 0.0)) <= boundary
                <= float(segment.get("end", 0.0))
            ]
            segment_index = containing[0][0] if containing else None
            similarity = float(left_center @ right_center)
            region_candidates.append({
                "time": boundary,
                "score": 1.0 - similarity,
                "similarity": similarity,
                "gap": 0.0,
                "left_count": left_count,
                "right_count": right_count,
                "left_cohesion": left_cohesion,
                "right_cohesion": right_cohesion,
                "context_seconds": BOUNDARY_CONTEXT_SECONDS,
                "left_text": "<acoustic>",
                "right_text": "<acoustic>",
                "punctuation": "none",
                "left_segment_index": segment_index,
                "right_segment_index": segment_index,
                "same_asr_segment": True,
                "timestamp_reliable": False,
                "candidate_source": "acoustic_scan",
            })
        # Retain local maxima only. A single sharp window is useful for
        # identity diagnostics, but it is not enough to split transcript text.
        for index, candidate in enumerate(region_candidates):
            neighbourhood = region_candidates[
                max(0, index - 1):min(len(region_candidates), index + 2)
            ]
            neighbours = [item for item in neighbourhood if item is not candidate]
            support_floor = max(0.50, float(candidate["score"]) - 0.12)
            persistence_count = sum(
                float(item["score"]) >= support_floor for item in neighbourhood
            )
            candidate["persistence_count"] = int(persistence_count)
            candidate["persistence_ratio"] = round(
                persistence_count / max(1, len(neighbourhood)), 4
            )
            candidate["persistent_acoustic_change"] = bool(
                persistence_count >= ACOUSTIC_PERSISTENCE_MIN_POINTS
            )
            if all(candidate["score"] >= item["score"] for item in neighbours):
                candidates.append(candidate)
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
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    # Estimate ordinary within-speaker variation for one continuous SAD
    # region.  Punctuation provides the structural evidence; this threshold
    # only decides whether that textual boundary also has an acoustic change.
    return min(0.52, max(0.42, median + max(0.04, 0.75 * mad)))


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
        return [], {}
    regions = sorted(set(item["region"] for item in windows), key=lambda item: item[1])
    located = []
    for candidate in candidates:
        location = _boundary_location(regions, candidate["time"])
        if location is None:
            continue
        enriched = dict(candidate)
        enriched.update(location)
        located.append(enriched)

    def location_key(item):
        return item["region"] or ("between", *item["between_regions"])

    grouped = defaultdict(list)
    for candidate in located:
        grouped[location_key(candidate)].append(candidate)
    thresholds = {
        key: _adaptive_boundary_threshold(group)
        for key, group in grouped.items()
    }

    eligible = []
    for candidate in located:
        key = location_key(candidate)
        threshold = thresholds[key]
        candidate["threshold"] = threshold
        punctuation = candidate.get("punctuation", "none")
        if punctuation == "strong" and candidate["score"] >= threshold:
            multi_source_support = bool(
                candidate.get("timestamp_reliable")
                and min(candidate["left_count"], candidate["right_count"])
                >= HARD_BOUNDARY_MIN_WINDOWS
                and min(
                    candidate.get("left_cohesion", 0.0),
                    candidate.get("right_cohesion", 0.0),
                ) >= HARD_BOUNDARY_MIN_COHESION
                and (
                    candidate["gap"] >= 0.12
                    or not candidate.get("same_asr_segment", True)
                )
            )
            candidate["constraint"] = "hard" if (
                multi_source_support
                and candidate["score"]
                >= max(0.55, threshold + HARD_BOUNDARY_SCORE_MARGIN)
            ) else "soft"
            candidate["hard_evidence"] = multi_source_support
            candidate["text_split_allowed"] = True
            candidate["boundary_role"] = "speaker_and_text"
            eligible.append(candidate)
            continue

        # Speaker changes without punctuation remain possible, but need a
        # sustained pause, a much stronger acoustic peak, and no nearby
        # punctuated explanation for the same peak.  This prevents cutting
        # ordinary phrases such as "Wake up" or "by yourself".
        acoustic_scan = candidate.get("candidate_source") == "acoustic_scan"
        strict_threshold = max(
            0.62 if acoustic_scan else 0.58,
            threshold + 0.12,
        )
        if candidate["score"] < strict_threshold:
            continue
        if acoustic_scan:
            if (
                min(candidate["left_count"], candidate["right_count"]) < 2
                or min(
                    candidate["left_cohesion"], candidate["right_cohesion"]
                ) < 0.60
            ):
                continue
        elif candidate["gap"] < 0.25:
            continue
        punctuated_neighbor = any(
            other.get("punctuation") == "strong"
            and location_key(other) == key
            and abs(other["time"] - candidate["time"]) <= 1.25
            and other["score"] >= thresholds[key]
            and other["score"] >= candidate["score"] - 0.10
            for other in located
        )
        if not punctuated_neighbor:
            # A non-punctuated acoustic peak can locate a useful split, but is
            # not strong enough to permanently assert two different identities.
            persistent = bool(candidate.get("persistent_acoustic_change"))
            candidate["text_split_allowed"] = bool(
                not acoustic_scan
                or (
                    persistent
                    and candidate["score"] >= ACOUSTIC_TEXT_SPLIT_MIN_SCORE
                )
            )
            # A sustained acoustic change can be strong enough to keep two
            # identities separate even when it is not safe to cut transcript
            # text. Keeping these decisions independent prevents the stricter
            # anti-over-splitting policy from silently merging real speakers.
            # Persistence is useful evidence for preserving a local turn and
            # auditing an unsplittable transcript, but it is not proof that
            # the voices can never belong to the same global identity.
            candidate["constraint"] = "soft"
            candidate["boundary_role"] = (
                "speaker_and_text"
                if candidate["text_split_allowed"]
                else (
                    "identity_separation_only"
                    if acoustic_scan
                    and persistent
                    and candidate["score"]
                    >= ACOUSTIC_IDENTITY_SEPARATION_MIN_SCORE
                    else "identity_diagnostic_only"
                )
            )
            eligible.append(candidate)

    # Non-maximum suppression keeps one acoustic peak for one real change.
    selected = []
    for candidate in sorted(
        eligible,
        key=lambda item: (item.get("punctuation") == "strong", item["score"]),
        reverse=True,
    ):
        key = location_key(candidate)
        if any(
            location_key(item) == key
            and abs(item["time"] - candidate["time"])
            < BOUNDARY_NMS_SECONDS
            for item in selected
        ):
            continue
        selected.append(candidate)
    selected.sort(key=lambda item: item["time"])
    return selected, thresholds


def _add_structural_sentence_boundaries(
    selected, timestamp_candidates, windows
):
    """Partition long ASR chunks at reliable sentence ends.

    A structural partition always creates a cleaner acoustic turn.  It becomes
    a soft identity constraint only when the local embedding change is also
    meaningful; otherwise it is partition-only and makes no speaker claim.
    """
    combined = list(selected)
    regions = sorted(
        set(item["region"] for item in windows), key=lambda item: item[1]
    )
    for candidate in timestamp_candidates:
        if (
            candidate.get("punctuation") != "strong"
            or not candidate.get("timestamp_reliable")
        ):
            continue
        if any(
            abs(float(item["time"]) - float(candidate["time"]))
            < BOUNDARY_NMS_SECONDS
            for item in combined
        ):
            continue
        location = _boundary_location(regions, float(candidate["time"]))
        if location is None:
            continue
        structural = dict(candidate)
        structural.update(location)
        structural.update({
            "constraint": (
                "soft"
                if float(candidate.get("score", 0.0))
                >= STRUCTURAL_SOFT_CHANGE_MIN
                else "partition"
            ),
            "candidate_source": "sentence_partition",
            "hard_evidence": False,
            "structural_partition": True,
            "text_split_allowed": True,
            "boundary_role": "sentence_partition",
        })
        combined.append(structural)
    combined.sort(key=lambda item: item["time"])
    return combined


def _is_high_confidence_identity_change(item):
    """Require independent evidence before making a local change a cannot-link."""
    if not item.get("persistent_acoustic_change"):
        return False
    timestamp_support = bool(item.get("timestamp_reliable"))
    acoustic_support = bool(
        int(item.get("persistence_count") or 0) >= 2
        and int(item.get("left_count") or 0) >= 2
        and int(item.get("right_count") or 0) >= 2
        and float(item.get("left_cohesion") or 0.0)
        >= HARD_BOUNDARY_MIN_COHESION
        and float(item.get("right_cohesion") or 0.0)
        >= HARD_BOUNDARY_MIN_COHESION
    )
    return bool(timestamp_support or acoustic_support)


def _build_acoustic_turns(windows, boundaries):
    by_region = defaultdict(list)
    for index, window in enumerate(windows):
        by_region[window["region"]].append(index)
    boundaries_by_region = defaultdict(list)
    for item in boundaries:
        if item["region"] is not None:
            boundaries_by_region[item["region"]].append(item)

    turns = []
    hard_cannot_links = set()
    soft_cannot_links = set()
    boundary_evidence = {}
    turns_by_region = {}
    for region in sorted(by_region, key=lambda item: item[1]):
        region_boundaries = sorted(
            boundaries_by_region.get(region, []), key=lambda item: item["time"]
        )
        cuts = [
            region[1],
            *[float(item["time"]) for item in region_boundaries],
            region[2],
        ]
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
            source_embeddings = np.stack([
                windows[index]["embedding"] for index in source
            ])
            embedding, retained_window_count = _robust_embedding_pool(
                source_embeddings
            )
            cohesion = float(np.mean(source_embeddings @ embedding))
            duration_score = min(1.0, max(0.0, duration) / 3.0)
            window_score = min(1.0, len(source) / 3.0)
            cohesion_score = min(1.0, max(0.0, (cohesion - 0.25) / 0.55))
            quality = 0.40 * duration_score + 0.25 * window_score + 0.35 * cohesion_score
            reliable = (
                duration >= RELIABLE_TURN_MIN_SECONDS
                and len(source) >= RELIABLE_TURN_MIN_WINDOWS
                and cohesion >= RELIABLE_TURN_MIN_COHESION
            )
            turn_index = len(turns)
            turns.append({
                "start": start,
                "end": end,
                "duration": duration,
                "region": region,
                "window_indices": indices,
                "embedding": embedding,
                "embedding_window_count": len(source),
                "embedding_retained_window_count": retained_window_count,
                "embedding_cohesion": cohesion,
                "embedding_quality": quality,
                "embedding_reliable": reliable,
            })
            region_turns.append(turn_index)
        for boundary in region_boundaries:
            boundary_time = float(boundary["time"])
            left_candidates = [
                index
                for index in region_turns
                if turns[index]["end"] <= boundary_time + 1e-6
            ]
            right_candidates = [
                index
                for index in region_turns
                if turns[index]["start"] >= boundary_time - 1e-6
            ]
            if not left_candidates or not right_candidates:
                continue
            left = max(left_candidates, key=lambda index: turns[index]["end"])
            right = min(right_candidates, key=lambda index: turns[index]["start"])
            edge = (min(left, right), max(left, right))
            if boundary.get("constraint") == "hard":
                hard_cannot_links.add(edge)
            elif boundary.get("constraint") == "soft":
                if _is_high_confidence_identity_change(boundary):
                    hard_cannot_links.add(edge)
                    boundary["identity_constraint"] = (
                        "hard_persistent_acoustic_change"
                    )
                else:
                    soft_cannot_links.add(edge)
            boundary_evidence[edge] = boundary
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
            edge = (min(left, right), max(left, right))
            if item.get("constraint") == "hard":
                hard_cannot_links.add(edge)
            elif item.get("constraint") == "soft":
                if _is_high_confidence_identity_change(item):
                    hard_cannot_links.add(edge)
                    item["identity_constraint"] = (
                        "hard_persistent_acoustic_change"
                    )
                else:
                    soft_cannot_links.add(edge)
            boundary_evidence[edge] = item
    return turns, hard_cannot_links, soft_cannot_links, boundary_evidence


def _cluster_centroid(cluster, turns):
    members = list(cluster)
    reliable_members = [
        index for index in members if turns[index].get("embedding_reliable")
    ]
    # Once an identity has reliable speech, weak/very short turns may be
    # assigned to it but must not drag its prototype toward another voice.
    prototype_members = reliable_members or members
    values = []
    weights = []
    for index in prototype_members:
        values.append(turns[index]["embedding"])
        weights.append(max(0.75, math.sqrt(max(turns[index]["duration"], 0.0))))
    return _unit_vector(np.average(np.stack(values), axis=0, weights=weights))


def _cluster_pair_evidence(left, right, turns):
    """Describe cross-cluster agreement instead of trusting centroids alone."""
    left_members = [
        index for index in left if turns[index].get("embedding_reliable")
    ] or list(left)
    right_members = [
        index for index in right if turns[index].get("embedding_reliable")
    ] or list(right)
    values = np.asarray([
        float(turns[a]["embedding"] @ turns[b]["embedding"])
        for a in left_members
        for b in right_members
    ], dtype=float)
    matrix = np.asarray([
        [
            float(turns[a]["embedding"] @ turns[b]["embedding"])
            for b in right_members
        ]
        for a in left_members
    ], dtype=float)
    centroid = float(
        _cluster_centroid(left, turns) @ _cluster_centroid(right, turns)
    )
    if not len(values):
        return {
            "centroid": centroid,
            "median": centroid,
            "q25": centroid,
            "minimum": centroid,
            "support_median": centroid,
            "support_q25": centroid,
            "_directional_support": np.asarray([centroid], dtype=float),
            "pair_count": 0,
        }
    left_support = np.max(matrix, axis=1)
    right_support = np.max(matrix, axis=0)
    directional_support = np.concatenate([left_support, right_support])
    return {
        "centroid": centroid,
        "median": float(np.median(values)),
        "q25": float(np.quantile(values, 0.25)),
        "minimum": float(np.min(values)),
        # Each turn only needs to match one reliable acoustic state of the
        # opposite identity.  This keeps shouting/whispering variants from
        # being rejected merely because every cross-pair is not similar.
        "support_median": float(np.median(directional_support)),
        "support_q25": float(np.quantile(directional_support, 0.25)),
        "_directional_support": directional_support,
        "pair_count": int(len(values)),
    }


def _robust_identity_match(evidence, threshold, *, anchor_pair):
    """Require broad turn-level support before merging established voices."""
    centroid_ok = evidence["centroid"] >= threshold
    directional_support = np.asarray(
        evidence.get("_directional_support", []), dtype=float
    )
    support_floor = threshold - 0.03
    support_coverage = (
        float(np.mean(directional_support >= support_floor))
        if len(directional_support) else 0.0
    )
    multi_prototype_ok = bool(
        evidence.get("pair_count", 0) >= 4
        and evidence.get("support_median", evidence["median"])
        >= threshold - 0.01
        and evidence.get("support_q25", evidence["q25"]) >= support_floor
        and support_coverage >= 0.80
        and evidence["centroid"] >= threshold - 0.06
    )
    if not (centroid_ok or multi_prototype_ok):
        return False
    if not anchor_pair:
        return bool(
            evidence["median"] >= threshold - 0.04
            or multi_prototype_ok
        )
    quantile_floor = max(IDENTITY_CROSS_MIN_QUANTILE, threshold - 0.06)
    return bool(
        (
            centroid_ok
            and evidence["median"] >= threshold - 0.02
            and evidence["q25"] >= quantile_floor
        )
        or multi_prototype_ok
    )


def _soft_continuity_threshold(boundary_items):
    """Return the evidence threshold for undoing a non-final local split."""
    eligible = [
        item for item in boundary_items
        if bool(item.get("same_asr_segment"))
        and item.get("constraint", "soft") == "soft"
    ]
    if not eligible:
        return None
    # A soft boundary is evidence, not a cannot-link.  Convert its acoustic
    # strength into a continuous penalty so weak sentence cuts are easy to
    # reconnect while a clear local voice change still needs stronger proof.
    threshold = max(
        PROTOTYPE_MERGE_SIMILARITY,
        SENTENCE_CONTINUITY_SIMILARITY + 0.06,
    ) + _soft_boundary_penalty(eligible)
    # A persistent local voice change is negative identity evidence.  It may
    # still be a false split, but reconnecting across it must require near-
    # duplicate acoustic evidence instead of ordinary sentence continuity.
    persistent_change = any(
        _is_high_confidence_identity_change(item)
        for item in eligible
    )
    return max(threshold, 0.76) if persistent_change else threshold


def _soft_boundary_penalty(boundary_items):
    """Return a bounded identity-merge penalty for soft boundary evidence."""
    soft_items = [
        item for item in boundary_items
        if item.get("constraint", "soft") == "soft"
    ]
    if not soft_items:
        return 0.0
    penalties = []
    for item in soft_items:
        score = item.get("score")
        try:
            score = float(score)
        except (TypeError, ValueError):
            # Missing diagnostics must stay conservative, but no longer turn
            # a soft split into the old near-hard 0.76 gate.
            penalties.append(
                (SOFT_BOUNDARY_MIN_PENALTY + SOFT_BOUNDARY_MAX_PENALTY) / 2
            )
            continue
        strength = min(1.0, max(0.0, (score - 0.40) / 0.30))
        penalties.append(
            SOFT_BOUNDARY_MIN_PENALTY
            + strength * (SOFT_BOUNDARY_MAX_PENALTY - SOFT_BOUNDARY_MIN_PENALTY)
        )
    return min(SOFT_BOUNDARY_MAX_PENALTY, max(penalties))


def _temporal_assignment_context(source_group, target_group, turns):
    """Use neighbours only to reject implausible short-turn assignments."""
    source_group = set(source_group)
    target_group = set(target_group)
    ordered = sorted(
        range(len(turns)), key=lambda index: (turns[index]["start"], turns[index]["end"])
    )
    positions = {turn_index: position for position, turn_index in enumerate(ordered)}
    neighbours = set()
    for turn_index in source_group:
        position = positions[turn_index]
        for step in (-1, 1):
            cursor = position + step
            while 0 <= cursor < len(ordered):
                candidate = ordered[cursor]
                if candidate not in source_group:
                    neighbours.add(candidate)
                    break
                cursor += step
    support = sum(index in target_group for index in neighbours)
    competing_reliable = sum(
        index not in target_group and turns[index].get("embedding_reliable")
        for index in neighbours
    )
    return {
        "neighbor_count": len(neighbours),
        "target_neighbor_support": support,
        "competing_reliable_neighbors": competing_reliable,
    }


def _sandwich_protection_penalty(
    left_group,
    right_group,
    left_profile,
    right_profile,
    turns,
    boundary_evidence=None,
):
    """Protect a real A-B-A island only when both local boundaries support it."""
    if left_profile["turn_count"] == 1:
        singleton_group, target_group = left_group, right_group
    elif right_profile["turn_count"] == 1:
        singleton_group, target_group = right_group, left_group
    else:
        return 0.0
    context = _temporal_assignment_context(singleton_group, target_group, turns)
    if context["target_neighbor_support"] < 2:
        return 0.0
    boundary_evidence = boundary_evidence or {}
    singleton = next(iter(singleton_group))
    target_neighbours = sorted(
        index for index in target_group
        if (min(index, singleton), max(index, singleton)) in boundary_evidence
    )
    supported_edges = 0
    for neighbour in target_neighbours:
        item = boundary_evidence[
            (min(neighbour, singleton), max(neighbour, singleton))
        ]
        if item.get("constraint") in {"soft", "hard"}:
            supported_edges += 1
    return (
        SANDWICH_SINGLETON_PENALTY
        if supported_edges >= 2
        else 0.0
    )


def _clusters_conflict(left, right, cannot_links):
    return any(
        (min(a, b), max(a, b)) in cannot_links
        for a in left
        for b in right
    )


def _cluster_reliability(cluster, turns):
    members = list(cluster)
    durations = [max(0.0, float(turns[index]["duration"])) for index in members]
    weights = [max(0.75, math.sqrt(duration)) for duration in durations]
    qualities = [
        float(turns[index].get("embedding_quality", 0.0)) for index in members
    ]
    quality = float(np.average(qualities, weights=weights)) if qualities else 0.0
    reliable_duration = sum(
        duration
        for index, duration in zip(members, durations)
        if turns[index].get("embedding_reliable")
    )
    reliable = reliable_duration >= RELIABLE_TURN_MIN_SECONDS
    return quality, reliable


def _identity_profile(cluster, turns):
    members = sorted(cluster)
    duration = sum(max(0.0, float(turns[index]["duration"])) for index in members)
    reliable_members = [
        index for index in members if turns[index].get("embedding_reliable")
    ]
    reliable_duration = sum(
        max(0.0, float(turns[index]["duration"]))
        for index in reliable_members
    )
    quality, _ = _cluster_reliability(cluster, turns)
    anchor = (
        reliable_duration >= PROTOTYPE_ANCHOR_MIN_SECONDS
        or (
            len(members) >= PROTOTYPE_ANCHOR_MIN_TURNS
            and reliable_duration >= RELIABLE_TURN_MIN_SECONDS
        )
    )
    # A clean single turn is already useful acoustic evidence even when it is
    # not long/repeated enough to be a high-confidence anchor.  Treat it as a
    # prototype with a small merge penalty instead of leaving it permanently
    # isolated behind the former 0.82 singleton threshold.
    prototype = bool(
        anchor
        or (
            reliable_members
            and reliable_duration >= RELIABLE_TURN_MIN_SECONDS
            and quality >= max(0.45, RELIABLE_TURN_MIN_COHESION - 0.08)
        )
    )
    return {
        "turn_count": len(members),
        "duration": duration,
        "reliable_turn_count": len(reliable_members),
        "reliable_duration": reliable_duration,
        "quality": quality,
        "anchor": anchor,
        "prototype": prototype,
    }


def _cluster_prototypes(cluster, turns, maximum=None):
    """Choose a few real turns that cover distinct acoustic states."""
    maximum = max(1, int(maximum or MULTI_PROTOTYPE_MAX_COUNT))
    members = [
        index for index in sorted(cluster)
        if turns[index].get("embedding_reliable")
    ] or sorted(cluster)
    if len(members) <= maximum:
        return members

    def turn_weight(index):
        duration = max(0.0, float(turns[index].get("duration", 0.0)))
        quality = max(0.0, float(turns[index].get("embedding_quality", 0.0)))
        return quality * math.sqrt(max(0.25, duration))

    # Start from the cleanest useful turn, then use farthest-first selection.
    # The selected items remain actual embeddings rather than an average that
    # can fall between neutral, shouted and whispered states.
    chosen = [max(members, key=turn_weight)]
    while len(chosen) < maximum:
        candidates = [index for index in members if index not in chosen]
        if not candidates:
            break
        candidate = min(
            candidates,
            key=lambda index: max(
                float(turns[index]["embedding"] @ turns[other]["embedding"])
                for other in chosen
            ),
        )
        nearest = max(
            float(turns[candidate]["embedding"] @ turns[other]["embedding"])
            for other in chosen
        )
        # A nearly identical extra prototype adds no coverage information.
        if nearest >= 0.88:
            break
        chosen.append(candidate)
    return chosen


def _prototype_coverage(source, target, turns, threshold):
    """Measure how much of source is explained by target's acoustic states."""
    members = [
        index for index in source if turns[index].get("embedding_reliable")
    ] or list(source)
    prototypes = _cluster_prototypes(target, turns)
    if not members or not prototypes:
        return {"coverage": 0.0, "median": -1.0, "minimum": -1.0}
    best = np.asarray([
        max(
            float(turns[index]["embedding"] @ turns[prototype]["embedding"])
            for prototype in prototypes
        )
        for index in members
    ], dtype=float)
    floor = max(IDENTITY_CROSS_MIN_QUANTILE, threshold - 0.08)
    return {
        "coverage": float(np.mean(best >= floor)),
        "median": float(np.median(best)),
        "minimum": float(np.min(best)),
        "prototype_count": len(prototypes),
    }


def _cluster_consistency(cluster, turns):
    """Measure whether one proposed identity remains acoustically coherent."""
    members = [
        index for index in cluster if turns[index].get("embedding_reliable")
    ] or list(cluster)
    if len(members) < 2:
        return {"median": 1.0, "q25": 1.0, "minimum": 1.0, "pair_count": 0}
    values = np.asarray([
        float(turns[left]["embedding"] @ turns[right]["embedding"])
        for position, left in enumerate(members)
        for right in members[position + 1:]
    ], dtype=float)
    return {
        "median": float(np.median(values)),
        "q25": float(np.quantile(values, 0.25)),
        "minimum": float(np.min(values)),
        "pair_count": int(len(values)),
    }


def _merge_preserves_identity(left, right, turns, threshold):
    """Accept multi-state voices while rejecting unsupported cluster chaining."""
    combined = set(left) | set(right)
    consistency = _cluster_consistency(combined, turns)
    left_coverage = _prototype_coverage(left, right, turns, threshold)
    right_coverage = _prototype_coverage(right, left, turns, threshold)
    floor = max(IDENTITY_CROSS_MIN_QUANTILE, threshold - IDENTITY_CONSISTENCY_MARGIN)
    coverage_required = MULTI_PROTOTYPE_COVERAGE
    # A singleton has only one observation, so its direct match must clear the
    # threshold rather than obtaining a free pass from a one-element ratio.
    if min(len(left), len(right)) == 1:
        coverage_required = 1.0
    accepted = bool(
        left_coverage["coverage"] >= coverage_required
        and right_coverage["coverage"] >= coverage_required
        and min(left_coverage["median"], right_coverage["median"])
        >= threshold - 0.06
        and (
            consistency["q25"] >= floor
            or min(left_coverage["coverage"], right_coverage["coverage"])
            >= 0.80
        )
    )
    consistency.update({
        "left_prototype_coverage": left_coverage["coverage"],
        "right_prototype_coverage": right_coverage["coverage"],
        "left_prototype_count": left_coverage.get("prototype_count", 0),
        "right_prototype_count": right_coverage.get("prototype_count", 0),
    })
    return accepted, consistency


def _adaptive_identity_thresholds(similarities, group_count):
    """Calibrate seed/reconnect/assignment gates from legal nearest matches."""
    nearest = defaultdict(list)
    for (left, right), similarity in similarities.items():
        nearest[left].append(float(similarity))
        nearest[right].append(float(similarity))
    nearest_values = np.asarray(
        [max(values) for values in nearest.values() if values], dtype=float
    )
    all_values = np.asarray(list(similarities.values()), dtype=float)
    if not len(nearest_values):
        reconnect = min(
            ADAPTIVE_RECONNECT_MAX,
            max(ADAPTIVE_RECONNECT_MIN, PROTOTYPE_MERGE_SIMILARITY),
        )
        confidence = 0.0
        center = None
        spread = None
    else:
        center = float(np.median(nearest_values))
        spread = float(np.median(np.abs(nearest_values - center)))
        if len(nearest_values) >= 4:
            estimate = center - max(0.05, 0.75 * spread)
            confidence = min(1.0, len(nearest_values) / max(4.0, group_count))
        else:
            # Too few identities cannot reveal two distribution modes. Keep a
            # conservative prior. A two/three-identity clip does not provide
            # enough evidence to lower the global gate from one close pair.
            estimate = PROTOTYPE_MERGE_SIMILARITY
            confidence = 0.35
        reconnect = min(
            ADAPTIVE_RECONNECT_MAX,
            max(ADAPTIVE_RECONNECT_MIN, estimate),
        )
    if len(nearest_values):
        upper = float(np.quantile(nearest_values, 0.75))
    else:
        upper = STRONG_SPEAKER_SIMILARITY
    seed = min(
        ADAPTIVE_SEED_MAX,
        max(ADAPTIVE_SEED_MIN, reconnect + 0.10, upper),
    )
    assignment = min(
        ADAPTIVE_ASSIGNMENT_MAX,
        max(ADAPTIVE_ASSIGNMENT_MIN, reconnect - 0.10),
    )
    return {
        "seed": float(seed),
        "reconnect": float(reconnect),
        "assignment": float(assignment),
        "confidence": float(confidence),
        "nearest_values": nearest_values,
        "all_values": all_values,
        "nearest_median": center,
        "nearest_mad": spread,
    }


def _adaptive_identity_threshold(similarities):
    """Compatibility wrapper for diagnostics/tests using the former helper."""
    mapping = {
        (index, index + 1): float(value)
        for index, value in enumerate(similarities)
    }
    return _adaptive_identity_thresholds(
        mapping, len(similarities) + 1
    )["reconnect"]


def _combined_merge_penalty(*values):
    """Combine uncertainty without stacking several weak signals to 0.78."""
    usable = [max(0.0, float(value)) for value in values if value is not None]
    return min(MAX_COMBINED_MERGE_PENALTY, max(usable, default=0.0))


def _pair_evidence_diagnostics(evidence):
    """Keep report fields JSON-safe while hiding internal vector evidence."""
    result = {}
    for key, value in evidence.items():
        if key.startswith("_"):
            continue
        if key == "pair_count":
            result[key] = int(value)
        elif np.isscalar(value):
            result[key] = round(float(value), 4)
    return result


def _merge_clusters(clusters, left_index, right_index):
    merged = set(clusters[left_index]) | set(clusters[right_index])
    return [
        cluster for index, cluster in enumerate(clusters)
        if index not in (left_index, right_index)
    ] + [merged]


def _constrained_turn_clusters(
    turns,
    cannot_links,
    soft_links=None,
    seed_threshold=None,
):
    """Build conservative speaker cores from pure acoustic turns.

    Earlier code greedily merged at a low centroid threshold and then colored
    the remaining conflict graph.  Graph coloring minimizes label count, so
    acoustically similar voices could collapse to two alternating identities.
    Here a merge must be mutual-nearest and supported by the distribution of
    all cross-turn similarities.  Uncertain turns remain separate for the
    later audited assignment phase instead of being forced into a color.
    """
    clusters = [{index} for index in range(len(turns))]
    soft_links = soft_links or set()
    if seed_threshold is None:
        legal_similarities = {
            (left, right): float(
                turns[left]["embedding"] @ turns[right]["embedding"]
            )
            for left in range(len(turns))
            for right in range(left + 1, len(turns))
            if (left, right) not in cannot_links
        }
        seed_threshold = _adaptive_identity_thresholds(
            legal_similarities, len(turns)
        )["seed"]
    while True:
        eligible = {}
        rankings = defaultdict(list)
        for left in range(len(clusters)):
            for right in range(left + 1, len(clusters)):
                if _clusters_conflict(clusters[left], clusters[right], cannot_links):
                    continue
                evidence = _cluster_pair_evidence(
                    clusters[left], clusters[right], turns
                )
                reliable_pair = bool(
                    any(turns[index].get("embedding_reliable") for index in clusters[left])
                    and any(turns[index].get("embedding_reliable") for index in clusters[right])
                )
                threshold = (
                    seed_threshold
                    if reliable_pair
                    else min(ADAPTIVE_SEED_MAX, seed_threshold + 0.04)
                )
                # A soft boundary must survive long enough to be audited by
                # reconciliation, but it is never an absolute cannot-link.
                if _clusters_conflict(
                    clusters[left], clusters[right], soft_links
                ):
                    threshold = min(0.84, threshold + 0.06)
                if not _robust_identity_match(
                    evidence, threshold, anchor_pair=True
                ):
                    continue
                robust_score = min(
                    evidence["centroid"],
                    evidence["median"],
                    evidence["q25"],
                )
                eligible[(left, right)] = (robust_score, evidence, threshold)
                rankings[left].append((robust_score, right))
                rankings[right].append((robust_score, left))
        if not eligible:
            break
        for values in rankings.values():
            values.sort(reverse=True)
        mutual = []
        for (left, right), (score, evidence, threshold) in eligible.items():
            if (
                rankings[left][0][1] == right
                and rankings[right][0][1] == left
            ):
                mutual.append((score, left, right, evidence, threshold))
        if not mutual:
            break
        _, left, right, _, _ = max(mutual, key=lambda item: item[0])
        clusters = _merge_clusters(clusters, left, right)
    return clusters


def _partition_signature(partition):
    return tuple(sorted(tuple(sorted(node)) for node in partition))


def _partition_turn_groups(partition, raw_clusters):
    return [
        {
            turn_index
            for raw_index in node
            for turn_index in raw_clusters[raw_index]
        }
        for node in partition
    ]


def _partition_soft_boundary_ratio(turn_groups, soft_cannot_links):
    if not soft_cannot_links:
        return 0.0
    owner = {}
    for label, group in enumerate(turn_groups):
        for turn_index in group:
            owner[turn_index] = label
    violations = sum(
        owner.get(left) == owner.get(right)
        for left, right in soft_cannot_links
    )
    return float(violations) / len(soft_cannot_links)


def _has_two_core_single_bridge(node, adjacency, raw_clusters, turns):
    """Detect a single-link bridge between two independently supported cores.

    The v7 search intentionally allows several acoustic states of one person
    to form a connected graph.  The unsafe case is narrower: two multi-atom,
    sufficiently long cores connected by only one pairwise edge.  Removing
    that bridge must not collapse both communities into one identity merely
    because the overall graph is technically connected.
    """
    atoms = set(node)
    if len(atoms) < 4:
        return False

    edges = {
        (min(left, right), max(left, right))
        for left in atoms
        for right in adjacency.get(left, set())
        if left != right
    }

    def reliable_duration(component):
        return sum(
            max(0.0, float(turns[turn_index].get("duration", 0.0)))
            for raw_index in component
            for turn_index in raw_clusters[raw_index]
            if turns[turn_index].get("embedding_reliable")
        )

    for blocked_left, blocked_right in edges:
        start = min(atoms)
        visited = set()
        stack = [start]
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            for neighbour in adjacency.get(current, set()):
                if {
                    current, neighbour
                } == {blocked_left, blocked_right}:
                    continue
                if neighbour in atoms and neighbour not in visited:
                    stack.append(neighbour)
        if visited == atoms:
            continue
        other = atoms - visited
        if (
            len(visited) >= 2
            and len(other) >= 2
            and reliable_duration(visited) >= 2.40
            and reliable_duration(other) >= 2.40
        ):
            return True
    return False


def _score_identity_partition(
    partition,
    raw_clusters,
    turns,
    hard_cannot_links,
    soft_cannot_links,
    turn_similarity=None,
    raw_pair_support=None,
    merge_support_floor=0.48,
    established_support_floor=0.72,
):
    """Score a complete identity hypothesis without assuming speaker count.

    A good partition must explain repeated acoustic identities, keep those
    identities separated, and avoid manufacturing weak one-turn speakers.
    Hard boundaries remain absolute. Soft boundaries are deliberately only a
    small cost because a real speaker may occur on both sides of a turn cut.
    """
    turn_groups = _partition_turn_groups(partition, raw_clusters)
    if any(
        _clusters_conflict(group, group, hard_cannot_links)
        for group in turn_groups
    ):
        return None

    turn_count = max(1, len(turns))
    if turn_similarity is None:
        embeddings = np.stack([turn["embedding"] for turn in turns])
        turn_similarity = embeddings @ embeddings.T
    owner = {}
    for label, group in enumerate(turn_groups):
        for turn_index in group:
            owner[turn_index] = label

    centroids = [_cluster_centroid(group, turns) for group in turn_groups]
    repeated_support = []
    assignment_margins = []
    repeated_turn_count = 0
    singleton_turns = []
    for turn_index, turn in enumerate(turns):
        label = owner[turn_index]
        peers = turn_groups[label] - {turn_index}
        outside = [
            other
            for other in range(len(turns))
            if owner.get(other) != label
        ]
        outside_score = (
            max(
                float(turn_similarity[turn_index, other])
                for other in outside
            )
            if outside else None
        )
        if peers:
            own_score = max(
                float(turn_similarity[turn_index, peer])
                for peer in peers
            )
            repeated_support.append(own_score)
            # A one-cluster solution has no competing identity and therefore
            # no evidence of assignment stability. Treating the missing rival
            # as zero would give it an artificial +1 margin and collapse every
            # recording into one speaker.
            if outside_score is not None:
                assignment_margins.append(own_score - outside_score)
            repeated_turn_count += 1
        else:
            singleton_turns.append(turn_index)

    cohesion = (
        float(np.median(repeated_support)) if repeated_support else 0.0
    )
    assignment_margin = (
        float(np.median(assignment_margins)) if assignment_margins else 0.0
    )
    if len(centroids) > 1:
        nearest_other = [
            max(
                float(centroid @ other)
                for other_index, other in enumerate(centroids)
                if other_index != index
            )
            for index, centroid in enumerate(centroids)
        ]
        separation = 1.0 - float(np.median(nearest_other))
    else:
        separation = 0.0

    singleton_ratio = len(singleton_turns) / turn_count
    low_quality_singleton_ratio = sum(
        not turns[index].get("embedding_reliable")
        or float(turns[index].get("duration", 0.0))
        < 1.20
        for index in singleton_turns
    ) / turn_count
    reuse_ratio = repeated_turn_count / turn_count
    complexity = (len(partition) - 1) / max(1, len(turns) - 1)
    soft_violation_ratio = _partition_soft_boundary_ratio(
        turn_groups, soft_cannot_links
    )
    # Judge a merged identity by directional prototype coverage: every raw
    # atom needs one convincing peer. Penalising every cross-pair fragmented
    # legitimate voices that have several supported acoustic states.
    directional_support = []
    established_merge_penalties = []
    identity_component_counts = []
    for node in partition:
        atoms = sorted(node)
        if len(atoms) < 2:
            identity_component_counts.append(1)
            continue
        adjacency = {atom: set() for atom in atoms}
        for left in atoms:
            peer_support = []
            for right in atoms:
                if right == left:
                    continue
                pair = (min(left, right), max(left, right))
                if raw_pair_support is not None:
                    support = raw_pair_support[pair]
                else:
                    evidence = _cluster_pair_evidence(
                        raw_clusters[left], raw_clusters[right], turns
                    )
                    support = max(
                        evidence["centroid"],
                        evidence.get("support_median", evidence["median"]),
                    )
                peer_support.append(support)
                if support >= merge_support_floor:
                    adjacency[left].add(right)
            best_support = max(peer_support) if peer_support else -1.0
            directional_support.append(best_support)
            if len(raw_clusters[left]) >= 2:
                established_merge_penalties.append(
                    max(0.0, established_support_floor - best_support)
                )
        visited = set()
        component_count = 0
        for atom in atoms:
            if atom in visited:
                continue
            component_count += 1
            stack = [atom]
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                stack.extend(adjacency[current] - visited)
        identity_component_counts.append(component_count)
        # Multi-state identities may form A1-A2-A3 chains, but two unrelated
        # internally coherent subgroups must never be silently unioned.
        if component_count > 1:
            return None
        if _has_two_core_single_bridge(
            node, adjacency, raw_clusters, turns
        ):
            return None
    unsupported_merge_penalty = (
        float(np.mean([
            max(0.0, merge_support_floor - support)
            for support in directional_support
        ]))
        if directional_support else 0.0
    )
    established_merge_penalty = (
        float(np.mean(established_merge_penalties))
        if established_merge_penalties else 0.0
    )
    # Two already-established cores are not short-turn assignment material.
    # Reject a hypothesis when one of those cores has no sufficiently close
    # peer in the proposed identity instead of compensating with speaker-count
    # savings elsewhere in the recording.
    if established_merge_penalties and max(established_merge_penalties) > 0.04:
        return None

    prototype_coverage = (
        float(np.mean([
            support >= merge_support_floor for support in directional_support
        ])) if directional_support else 0.0
    )
    # There is deliberately no ordinary singleton or speaker-count penalty.
    # Those terms caused opposite failures on different videos. Supported
    # reuse earns a reward; weak islands and unsupported merges remain costly.
    score = (
        0.72 * assignment_margin
        + 0.34 * cohesion
        + 0.22 * separation
        + 0.18 * reuse_ratio
        + 0.12 * prototype_coverage
        - 0.38 * low_quality_singleton_ratio
        - 0.06 * soft_violation_ratio
        - 0.90 * unsupported_merge_penalty
        - 0.78 * established_merge_penalty
    )
    return {
        "score": float(score),
        "speaker_count": len(partition),
        "cohesion": cohesion,
        "assignment_margin": assignment_margin,
        "separation": separation,
        "reuse_ratio": reuse_ratio,
        "singleton_ratio": singleton_ratio,
        "low_quality_singleton_ratio": low_quality_singleton_ratio,
        "complexity": complexity,
        "soft_boundary_violation_ratio": soft_violation_ratio,
        "unsupported_merge_penalty": unsupported_merge_penalty,
        "established_merge_penalty": established_merge_penalty,
        "prototype_coverage": prototype_coverage,
        "identity_component_count_max": max(identity_component_counts or [1]),
    }


def _search_global_identity_partition(
    raw_clusters,
    turns,
    hard_cannot_links,
    soft_cannot_links,
    seed_partitions=(),
    beam_width=6,
):
    """Compare complete speaker partitions instead of stopping at local pairs."""
    initial = [{index} for index in range(len(raw_clusters))]
    seen = {}
    embeddings = np.stack([turn["embedding"] for turn in turns])
    turn_similarity = embeddings @ embeddings.T
    raw_pair_support = {}
    for left in range(len(raw_clusters)):
        for right in range(left + 1, len(raw_clusters)):
            evidence = _cluster_pair_evidence(
                raw_clusters[left], raw_clusters[right], turns
            )
            raw_pair_support[(left, right)] = max(
                evidence["centroid"],
                evidence.get("support_median", evidence["median"]),
            )
    support_values = np.asarray(list(raw_pair_support.values()), dtype=float)
    if len(support_values):
        merge_support_floor = float(np.clip(
            np.quantile(support_values, 0.25), 0.38, 0.56
        ))
        established_support_floor = float(np.clip(
            np.median(support_values) + 0.12, 0.62, 0.78
        ))
    else:
        merge_support_floor = 0.48
        established_support_floor = 0.72

    def consider(partition):
        signature = _partition_signature(partition)
        if signature in seen:
            return None
        metrics = _score_identity_partition(
            partition,
            raw_clusters,
            turns,
            hard_cannot_links,
            soft_cannot_links,
            turn_similarity=turn_similarity,
            raw_pair_support=raw_pair_support,
            merge_support_floor=merge_support_floor,
            established_support_floor=established_support_floor,
        )
        if metrics is None:
            return None
        normalized = [set(node) for node in signature]
        seen[signature] = (normalized, metrics)
        return normalized, metrics

    consider(initial)
    for partition in seed_partitions:
        consider(partition)

    frontier = [initial]
    while frontier:
        next_candidates = []
        for partition in frontier:
            turn_groups = _partition_turn_groups(partition, raw_clusters)
            for left in range(len(partition)):
                for right in range(left + 1, len(partition)):
                    if _clusters_conflict(
                        turn_groups[left],
                        turn_groups[right],
                        hard_cannot_links,
                    ):
                        continue
                    merged = partition[left] | partition[right]
                    candidate = [
                        set(node)
                        for index, node in enumerate(partition)
                        if index not in (left, right)
                    ] + [merged]
                    result = consider(candidate)
                    if result is not None:
                        next_candidates.append(result)
        if not next_candidates:
            break
        next_candidates.sort(
            key=lambda item: item[1]["score"], reverse=True
        )
        frontier = [
            partition for partition, _ in next_candidates[:beam_width]
        ]

    candidates = sorted(
        seen.values(), key=lambda item: item[1]["score"], reverse=True
    )
    # Select a hypothesis that is also stable across near-optimal partitions.
    # This is deterministic consensus and never assumes a requested K.
    best_raw_score = candidates[0][1]["score"]
    near = [item for item in candidates if item[1]["score"] >= best_raw_score - 0.08]

    def coassignment(partition):
        owner = {}
        for label, node in enumerate(partition):
            for raw_index in node:
                owner[raw_index] = label
        return {
            (left, right): owner[left] == owner[right]
            for left in range(len(raw_clusters))
            for right in range(left + 1, len(raw_clusters))
        }

    near_maps = [coassignment(partition) for partition, _ in near]
    rescored = []
    for partition, metrics in candidates:
        mapping = coassignment(partition)
        if mapping and near_maps:
            stability = float(np.mean([
                np.mean([mapping[pair] == other[pair] for pair in mapping])
                for other in near_maps
            ]))
        else:
            stability = 1.0
        updated = dict(metrics)
        updated["partition_stability"] = stability
        updated["selection_score"] = metrics["score"] + 0.08 * stability
        rescored.append((partition, updated))
    rescored.sort(key=lambda item: item[1]["selection_score"], reverse=True)
    selected_partition, selected_metrics = rescored[0]
    selected_metrics = dict(selected_metrics)
    selected_metrics.update({
        "merge_support_floor": merge_support_floor,
        "established_support_floor": established_support_floor,
    })
    best_by_count = {}
    for _, metrics in rescored:
        count = metrics["speaker_count"]
        if count not in best_by_count:
            best_by_count[count] = metrics
    audit = [
        {
            key: (round(value, 4) if isinstance(value, float) else value)
            for key, value in metrics.items()
        }
        for _, metrics in sorted(
            best_by_count.items(), key=lambda item: item[0]
        )
    ]
    return selected_partition, selected_metrics, audit, len(candidates)


def _assign_weak_identity_nodes(
    partition, raw_clusters, turns, hard_cannot_links, assignment_threshold
):
    """Attach short/weak nodes only after stable identities are established."""
    nodes = [set(node) for node in partition]
    history = []

    def members(node):
        return {
            turn_index
            for raw_index in node
            for turn_index in raw_clusters[raw_index]
        }

    while True:
        groups = [members(node) for node in nodes]
        profiles = [_identity_profile(group, turns) for group in groups]
        weak_indices = [
            index for index, profile in enumerate(profiles)
            if not profile["prototype"] or profile["reliable_duration"] < 1.20
        ]
        stable_indices = [
            index for index, profile in enumerate(profiles)
            if profile["prototype"] and profile["reliable_duration"] >= 1.20
        ]
        chosen = None
        for weak_index in weak_indices:
            rankings = []
            for stable_index in stable_indices:
                if weak_index == stable_index or _clusters_conflict(
                    groups[weak_index], groups[stable_index], hard_cannot_links
                ):
                    continue
                evidence = _cluster_pair_evidence(
                    groups[weak_index], groups[stable_index], turns
                )
                support = max(
                    evidence["centroid"],
                    evidence.get("support_median", evidence["median"]),
                )
                context = _temporal_assignment_context(
                    groups[weak_index], groups[stable_index], turns
                )
                rankings.append((support, stable_index, evidence, context))
            rankings.sort(reverse=True, key=lambda item: item[0])
            if not rankings:
                continue
            best = rankings[0]
            runner_up = rankings[1][0] if len(rankings) > 1 else -1.0
            margin = best[0] - runner_up
            context = best[3]
            if best[0] < assignment_threshold or margin < LOW_QUALITY_ASSIGN_MARGIN:
                continue
            if (
                context["competing_reliable_neighbors"] > 0
                and context["target_neighbor_support"] == 0
                and best[0] < assignment_threshold + 0.08
            ):
                continue
            candidate = (best[0], margin, weak_index, best[1], best[2], context)
            if chosen is None or (candidate[0], candidate[1]) > (chosen[0], chosen[1]):
                chosen = candidate
        if chosen is None:
            break
        support, margin, weak_index, stable_index, evidence, context = chosen
        weak_node = set(nodes[weak_index])
        stable_node = set(nodes[stable_index])
        merged = weak_node | stable_node
        nodes = [
            node for index, node in enumerate(nodes)
            if index not in (weak_index, stable_index)
        ] + [merged]
        history.append({
            "raw_speakers": [f"SPEAKER_{index:02d}" for index in sorted(merged)],
            "similarity": round(support, 4),
            "margin": round(margin, 4),
            "reason": "delayed_unique_weak_turn_assignment",
            "target_neighbor_support": context["target_neighbor_support"],
            "competing_reliable_neighbors": context["competing_reliable_neighbors"],
            "prototype_support_median": round(
                evidence.get("support_median", evidence["median"]), 4
            ),
        })
    return nodes, history


def _reconcile_turn_clusters(
    raw_clusters,
    turns,
    hard_cannot_links,
    soft_cannot_links,
    boundary_evidence,
):
    """Resolve global identities while leaving local RTTM boundaries intact.

    Raw identities are deliberately kept for sentence alignment.  Global
    identity resolution uses three different kinds of evidence in order:
    sentence continuity across a soft split, reliable prototype reconnection,
    and finally unique assignment of a short/weak identity to a prototype.
    Hard boundaries are the only absolute cannot-links; soft boundaries add a
    continuous evidence penalty and partitions are identity-neutral.
    """
    identity_nodes = [{index} for index in range(len(raw_clusters))]
    merge_history = []

    def turn_members(nodes):
        return {
            turn_index
            for raw_index in nodes
            for turn_index in raw_clusters[raw_index]
        }

    def edge_evidence(left_turns, right_turns):
        matches = []
        for left_turn in left_turns:
            for right_turn in right_turns:
                edge = (
                    min(left_turn, right_turn), max(left_turn, right_turn)
                )
                if edge in boundary_evidence:
                    matches.append(boundary_evidence[edge])
        return matches

    initial_turn_groups = [turn_members(nodes) for nodes in identity_nodes]
    initial_centroids = [
        _cluster_centroid(group, turns) for group in initial_turn_groups
    ]
    unconstrained_similarities = {}
    for left in range(len(initial_turn_groups)):
        for right in range(left + 1, len(initial_turn_groups)):
            if _clusters_conflict(
                initial_turn_groups[left],
                initial_turn_groups[right],
                hard_cannot_links,
            ):
                continue
            unconstrained_similarities[(left, right)] = float(
                initial_centroids[left] @ initial_centroids[right]
            )
    calibration = _adaptive_identity_thresholds(
        unconstrained_similarities, len(initial_turn_groups)
    )
    seed_threshold = calibration["seed"]
    prototype_threshold = calibration["reconnect"]
    assignment_threshold = calibration["assignment"]

    identity_candidates = []
    initial_profiles = [
        _identity_profile(group, turns) for group in initial_turn_groups
    ]
    for left in range(len(initial_turn_groups)):
        for right in range(left + 1, len(initial_turn_groups)):
            left_group = initial_turn_groups[left]
            right_group = initial_turn_groups[right]
            evidence = edge_evidence(left_group, right_group)
            pair_evidence = _cluster_pair_evidence(
                left_group, right_group, turns
            )
            hard_conflict = _clusters_conflict(
                left_group, right_group, hard_cannot_links
            )
            soft_conflict = _clusters_conflict(
                left_group, right_group, soft_cannot_links
            )
            continuity_threshold = _soft_continuity_threshold(evidence)
            continuity = continuity_threshold is not None
            similarity = pair_evidence["centroid"]
            if hard_conflict:
                initial_decision = "blocked_by_hard_boundary"
            elif continuity:
                initial_decision = (
                    "sentence_continuity_candidate"
                    if similarity >= continuity_threshold
                    else "below_sentence_continuity_threshold"
                )
            elif initial_profiles[left]["prototype"] and initial_profiles[right]["prototype"]:
                initial_decision = (
                    "prototype_candidate"
                    if similarity >= (
                        prototype_threshold
                        + _soft_boundary_penalty(evidence)
                    )
                    else "below_prototype_threshold"
                )
            elif initial_profiles[left]["prototype"] or initial_profiles[right]["prototype"]:
                initial_decision = (
                    "short_turn_assignment_candidate"
                    if similarity >= assignment_threshold
                    else "below_short_turn_threshold"
                )
            else:
                initial_decision = "no_reliable_prototype"
            identity_candidates.append({
                "left_raw_speaker": f"SPEAKER_{left:02d}",
                "right_raw_speaker": f"SPEAKER_{right:02d}",
                "similarity": round(similarity, 4),
                "cross_turn_median": round(pair_evidence["median"], 4),
                "cross_turn_q25": round(pair_evidence["q25"], 4),
                "cross_turn_minimum": round(pair_evidence["minimum"], 4),
                "prototype_support_median": round(
                    pair_evidence.get("support_median", pair_evidence["median"]),
                    4,
                ),
                "prototype_support_q25": round(
                    pair_evidence.get("support_q25", pair_evidence["q25"]),
                    4,
                ),
                "hard_conflict": hard_conflict,
                "soft_conflict": soft_conflict,
                "sentence_continuity": continuity,
                "left_anchor": initial_profiles[left]["anchor"],
                "right_anchor": initial_profiles[right]["anchor"],
                "left_prototype": initial_profiles[left]["prototype"],
                "right_prototype": initial_profiles[right]["prototype"],
                "soft_boundary_penalty": round(
                    _soft_boundary_penalty(evidence), 4
                ),
                "initial_decision": initial_decision,
            })

    while len(identity_nodes) > 1:
        turn_groups = [turn_members(nodes) for nodes in identity_nodes]
        profiles = [_identity_profile(group, turns) for group in turn_groups]
        similarities = {}
        pair_evidence_by_pair = {}
        evidence_by_pair = {}
        for left in range(len(identity_nodes)):
            for right in range(left + 1, len(identity_nodes)):
                if _clusters_conflict(
                    turn_groups[left], turn_groups[right], hard_cannot_links
                ):
                    continue
                pair_evidence = _cluster_pair_evidence(
                    turn_groups[left], turn_groups[right], turns
                )
                similarities[(left, right)] = pair_evidence["centroid"]
                pair_evidence_by_pair[(left, right)] = pair_evidence
                evidence_by_pair[(left, right)] = edge_evidence(
                    turn_groups[left], turn_groups[right]
                )

        if not similarities:
            break

        # Phase 1: undo a likely false local split inside one ASR sentence.
        # This targets cases such as "Yeah, yeah, I did / leave my wallet..."
        # without removing the sentence/turn boundary itself.
        best = None
        for (left, right), similarity in similarities.items():
            continuity_threshold = _soft_continuity_threshold(
                evidence_by_pair[(left, right)]
            )
            continuity = continuity_threshold is not None
            sandwich_penalty = 0.0
            if continuity:
                sandwich_penalty = _sandwich_protection_penalty(
                    turn_groups[left],
                    turn_groups[right],
                    profiles[left],
                    profiles[right],
                    turns,
                    boundary_evidence,
                )
                if sandwich_penalty:
                    continuity_threshold = max(
                        continuity_threshold, 0.76
                    )
            robust = pair_evidence_by_pair[(left, right)]
            if (
                not continuity
                or not _robust_identity_match(
                    robust,
                    continuity_threshold,
                    anchor_pair=False,
                )
            ):
                continue
            consistent, consistency = _merge_preserves_identity(
                turn_groups[left],
                turn_groups[right],
                turns,
                continuity_threshold,
            )
            if not consistent:
                continue
            if best is None or similarity > best[0]:
                best = (
                    similarity,
                    left,
                    right,
                    continuity_threshold,
                    "same_sentence_soft_boundary",
                    similarity,
                    consistency,
                    {
                        "soft_boundary": round(
                            _soft_boundary_penalty(
                                evidence_by_pair[(left, right)]
                            ), 4
                        ),
                        "sandwich_protection": round(sandwich_penalty, 4),
                    },
                )

        # Phase 2: reconnect reliable prototypes. Mutual-nearest is strong
        # evidence, but no longer an absolute requirement: a non-mutual pair
        # may merge when it clears an additional margin and the combined
        # identity remains coherent. This recovers A1/A2/A3 chains without
        # forcing the overall speaker count downward.
        if best is None:
            anchor_rankings = defaultdict(list)
            for (left, right), similarity in similarities.items():
                if profiles[left]["prototype"] and profiles[right]["prototype"]:
                    soft_penalty = _soft_boundary_penalty(
                        evidence_by_pair[(left, right)]
                    )
                    robust_score = min(
                        similarity,
                        pair_evidence_by_pair[(left, right)]["median"],
                    ) - soft_penalty
                    anchor_rankings[left].append((robust_score, right))
                    anchor_rankings[right].append((robust_score, left))
            for values in anchor_rankings.values():
                values.sort(reverse=True)
            for (left, right), similarity in similarities.items():
                if not (
                    profiles[left]["prototype"]
                    and profiles[right]["prototype"]
                ):
                    continue
                boundary_penalty = _soft_boundary_penalty(
                    evidence_by_pair[(left, right)]
                )
                single_turn_penalty = (
                    SINGLE_TURN_PROTOTYPE_PENALTY
                    if min(
                        profiles[left]["turn_count"],
                        profiles[right]["turn_count"],
                    ) < 2
                    else 0.0
                )
                structural_partition = any(
                    item.get("constraint") == "partition"
                    for item in evidence_by_pair[(left, right)]
                )
                mutual = bool(
                    anchor_rankings[left]
                    and anchor_rankings[left][0][1] == right
                    and anchor_rankings[right]
                    and anchor_rankings[right][0][1] == left
                )
                non_mutual_penalty = (
                    0.0 if mutual else NON_MUTUAL_PROTOTYPE_PENALTY
                )
                # A clean A-B-A interjection is exactly where temporal
                # proximity is most misleading. Preserve that potential rare
                # speaker unless it is exceptionally close acoustically to A.
                sandwich_penalty = _sandwich_protection_penalty(
                    turn_groups[left],
                    turn_groups[right],
                    profiles[left],
                    profiles[right],
                    turns,
                    boundary_evidence,
                )
                combined_penalty = _combined_merge_penalty(
                    boundary_penalty,
                    single_turn_penalty,
                    non_mutual_penalty,
                )
                required_threshold = min(
                    ADAPTIVE_SEED_MAX,
                    prototype_threshold + combined_penalty,
                )
                if sandwich_penalty:
                    required_threshold = max(required_threshold, 0.76)
                elif single_turn_penalty and not structural_partition:
                    # One observation is too little to establish a global
                    # identity. Only an explicit sentence partition may use
                    # the ordinary reconnect gate; other singleton prototypes
                    # need exceptional acoustic agreement.
                    required_threshold = max(required_threshold, 0.76)
                if not _robust_identity_match(
                    pair_evidence_by_pair[(left, right)],
                    required_threshold,
                    anchor_pair=True,
                ):
                    continue
                consistent, consistency = _merge_preserves_identity(
                    turn_groups[left],
                    turn_groups[right],
                    turns,
                    required_threshold,
                )
                if not consistent:
                    continue
                adjusted_score = similarity - combined_penalty - sandwich_penalty
                if best is None or adjusted_score > best[0]:
                    best = (
                        adjusted_score,
                        left,
                        right,
                        required_threshold,
                        (
                            "mutual_reliable_prototype_match"
                            if mutual
                            else "strong_non_mutual_prototype_match"
                        ),
                        similarity,
                        consistency,
                        {
                            "soft_boundary": round(boundary_penalty, 4),
                            "single_turn": round(single_turn_penalty, 4),
                            "non_mutual": round(non_mutual_penalty, 4),
                            "sandwich_protection": round(
                                sandwich_penalty, 4
                            ),
                            "combined": round(combined_penalty, 4),
                            "structural_partition": structural_partition,
                        },
                    )

        # Phase 3: a short or weak identity may join an established prototype
        # only when its best legal match is unique.  No target speaker count is
        # supplied or forced.
        if best is None:
            assignments = []
            for source in range(len(identity_nodes)):
                if profiles[source]["prototype"]:
                    continue
                ranked = []
                for target in range(len(identity_nodes)):
                    if not profiles[target]["prototype"] or source == target:
                        continue
                    pair = (min(source, target), max(source, target))
                    if pair not in similarities:
                        continue
                    ranked.append((similarities[pair], target))
                ranked.sort(reverse=True)
                if not ranked:
                    continue
                similarity, target = ranked[0]
                source_profile = profiles[source]
                quality_penalty = max(
                    0.0, 0.60 - float(source_profile.get("quality", 0.0))
                ) * 0.15
                duration_penalty = (
                    0.04 if float(source_profile.get("duration", 0.0)) < 1.20
                    else 0.0
                )
                reliability_penalty = (
                    0.03
                    if int(source_profile.get("reliable_turn_count", 0)) == 0
                    else 0.0
                )
                source_threshold = min(
                    prototype_threshold,
                    assignment_threshold
                    + quality_penalty
                    + duration_penalty
                    + reliability_penalty,
                )
                pair = (min(source, target), max(source, target))
                boundary_penalty = _soft_boundary_penalty(
                    evidence_by_pair[pair]
                )
                source_threshold = min(
                    ADAPTIVE_RECONNECT_MAX,
                    source_threshold
                    + min(boundary_penalty, MAX_COMBINED_MERGE_PENALTY),
                )
                margin = (
                    similarity - ranked[1][0]
                    if len(ranked) > 1 else 1.0
                )
                required_margin = (
                    LOW_QUALITY_ASSIGN_MARGIN
                    + (0.03 if source_profile["duration"] < 1.20 else 0.0)
                    + (0.02 if source_profile["quality"] < 0.50 else 0.0)
                )
                temporal_context = _temporal_assignment_context(
                    turn_groups[source], turn_groups[target], turns
                )
                # Temporal context is deliberately veto-only.  An A-B-A
                # interjection must not be swallowed merely because both
                # neighbours are A, but a very short turn should need stronger
                # acoustic evidence before jumping to a non-neighbour identity.
                if (
                    source_profile["duration"] < 1.20
                    and temporal_context["neighbor_count"]
                    and not temporal_context["target_neighbor_support"]
                ):
                    required_margin += 0.03
                if temporal_context["competing_reliable_neighbors"] >= 2:
                    required_margin += 0.02
                if temporal_context["target_neighbor_support"]:
                    required_margin = max(0.02, required_margin - 0.02)
                if not _robust_identity_match(
                    pair_evidence_by_pair[pair],
                    source_threshold,
                    anchor_pair=False,
                ):
                    continue
                if margin < required_margin:
                    continue
                consistent, consistency = _merge_preserves_identity(
                    turn_groups[source],
                    turn_groups[target],
                    turns,
                    source_threshold,
                )
                if not consistent:
                    continue
                assignments.append(
                    (
                        similarity - boundary_penalty,
                        source,
                        target,
                        margin,
                        source_threshold,
                        similarity,
                        consistency,
                        {"soft_boundary": round(boundary_penalty, 4)},
                    )
                )
            if assignments:
                (
                    adjusted_score,
                    source,
                    target,
                    margin,
                    source_threshold,
                    similarity,
                    consistency,
                    penalty_components,
                ) = max(assignments)
                best = (
                    adjusted_score,
                    min(source, target),
                    max(source, target),
                    source_threshold,
                    "unique_short_turn_to_prototype",
                    similarity,
                    consistency,
                    penalty_components,
                )

        if best is None:
            break
        adjusted_score, left, right, threshold, reason, *audit = best
        similarity = audit[0] if audit else adjusted_score
        consistency = audit[1] if len(audit) > 1 else _cluster_consistency(
            turn_groups[left] | turn_groups[right], turns
        )
        penalty_components = audit[2] if len(audit) > 2 else {}
        if left == right:
            # Defensive guard for future changes to assignment selection.
            break
        merged = identity_nodes[left] | identity_nodes[right]
        merge_history.append({
            "raw_speakers": [
                f"SPEAKER_{index:02d}" for index in sorted(merged)
            ],
            "similarity": round(similarity, 4),
            "adjusted_score": round(adjusted_score, 4),
            "threshold": round(threshold, 4),
            "reason": reason,
            "combined_consistency_median": round(
                consistency["median"], 4
            ),
            "combined_consistency_q25": round(consistency["q25"], 4),
            "prototype_coverage": round(min(
                float(consistency.get("left_prototype_coverage", 1.0)),
                float(consistency.get("right_prototype_coverage", 1.0)),
            ), 4),
            "penalty_components": penalty_components,
        })
        identity_nodes = [
            nodes
            for index, nodes in enumerate(identity_nodes)
            if index not in (left, right)
        ] + [merged]

    # The conservative stages above provide a useful candidate, but local
    # reciprocal merges can become trapped in a fragmented partition. Compare
    # complete hypotheses over the same raw acoustic atoms. This does not alter
    # ASR text, time boundaries, embeddings, or impose a requested speaker
    # count; it only selects the partition that best explains all turns.
    first_pass_speaker_count = len(identity_nodes)
    first_pass_groups = [turn_members(nodes) for nodes in identity_nodes]
    first_pass_profiles = [
        _identity_profile(group, turns) for group in first_pass_groups
    ]
    singleton_ratio = (
        sum(profile["turn_count"] == 1 for profile in first_pass_profiles)
        / max(1, len(first_pass_profiles))
    )
    retention_ratio = len(identity_nodes) / max(1, len(raw_clusters))
    fragmentation_score = 0.65 * singleton_ratio + 0.35 * retention_ratio
    selected_nodes, selected_metrics, partition_audit, candidate_count = (
        _search_global_identity_partition(
            raw_clusters,
            turns,
            hard_cannot_links,
            soft_cannot_links,
            seed_partitions=(identity_nodes,),
        )
    )
    selected_nodes, delayed_assignments = _assign_weak_identity_nodes(
        selected_nodes,
        raw_clusters,
        turns,
        hard_cannot_links,
        assignment_threshold,
    )
    first_signature = _partition_signature(identity_nodes)
    selected_signature = _partition_signature(selected_nodes)
    global_selection_changed = selected_signature != first_signature
    fragmentation_retry_triggered = global_selection_changed
    fragmentation_retry_merges = max(
        0, first_pass_speaker_count - len(selected_nodes)
    )
    if global_selection_changed:
        # Local merge history is only an audit trail for the final partition.
        # If global selection deliberately undoes an over-merge, do not report
        # that rejected local merge as though it were still active.
        retained_history = []
        for item in merge_history:
            raw_indices = {
                int(label.rsplit("_", 1)[-1])
                for label in item.get("raw_speakers", [])
            }
            if any(raw_indices <= nodes for nodes in selected_nodes):
                retained_history.append(item)
        merge_history = retained_history
        previous_groups = [turn_members(nodes) for nodes in identity_nodes]
        for nodes in selected_nodes:
            if any(nodes <= previous for previous in identity_nodes):
                continue
            combined_turns = turn_members(nodes)
            evidence_values = []
            for left in range(len(previous_groups)):
                if not (identity_nodes[left] & nodes):
                    continue
                for right in range(left + 1, len(previous_groups)):
                    if not (identity_nodes[right] & nodes):
                        continue
                    evidence_values.append(
                        _cluster_pair_evidence(
                            previous_groups[left], previous_groups[right], turns
                        )["centroid"]
                    )
            consistency = _cluster_consistency(combined_turns, turns)
            similarity = (
                float(np.median(evidence_values))
                if evidence_values else consistency["median"]
            )
            merge_history.append({
                "raw_speakers": [
                    f"SPEAKER_{index:02d}" for index in sorted(nodes)
                ],
                "similarity": round(similarity, 4),
                "adjusted_score": round(selected_metrics["score"], 4),
                "threshold": 0.0,
                "reason": "global_partition_model_selection",
                "combined_consistency_median": round(
                    consistency["median"], 4
                ),
                "combined_consistency_q25": round(
                    consistency["q25"], 4
                ),
                "prototype_coverage": round(min(
                    float(consistency.get("left_prototype_coverage", 1.0)),
                    float(consistency.get("right_prototype_coverage", 1.0)),
                ), 4),
                "penalty_components": {
                    "global_objective": round(selected_metrics["score"], 4),
                },
            })
    identity_nodes = selected_nodes

    ordered_identity_nodes = sorted(
        identity_nodes,
        key=lambda nodes: min(
            turns[turn_index]["start"]
            for raw_index in nodes
            for turn_index in raw_clusters[raw_index]
        ),
    )
    identity_clusters = []
    raw_to_final = {}
    for final_label, nodes in enumerate(ordered_identity_nodes):
        combined = set()
        for raw_index in nodes:
            raw_to_final[raw_index] = final_label
            combined.update(raw_clusters[raw_index])
        identity_clusters.append(combined)

    reliable_turns = sum(bool(turn.get("embedding_reliable")) for turn in turns)
    similarity_values = calibration["all_values"]
    nearest_similarity_values = calibration["nearest_values"]
    final_identity_profiles = []
    for final_label, cluster in enumerate(identity_clusters):
        profile = _identity_profile(cluster, turns)
        consistency = _cluster_consistency(cluster, turns)
        final_identity_profiles.append({
            "speaker": f"SPEAKER_{final_label:02d}",
            "turn_count": profile["turn_count"],
            "duration": round(profile["duration"], 3),
            "reliable_turn_count": profile["reliable_turn_count"],
            "embedding_quality": round(profile["quality"], 4),
            "prototype_anchor": bool(profile["anchor"]),
            "prototype_eligible": bool(profile["prototype"]),
            "prototype_count": len(_cluster_prototypes(cluster, turns)),
            "consistency_median": round(consistency["median"], 4),
            "consistency_q25": round(consistency["q25"], 4),
        })
    diagnostics = {
        "algorithm": "stable_core_identity_consensus_v11_sentence_first",
        "turn_count": len(turns),
        "raw_speaker_count": len(raw_clusters),
        "speaker_count": len(identity_clusters),
        "hard_constraint_count": len(hard_cannot_links),
        "soft_constraint_count": len(soft_cannot_links),
        "reliable_turn_count": reliable_turns,
        "low_quality_turn_count": len(turns) - reliable_turns,
        "speaker_count_lower_bound": 2 if hard_cannot_links else 1,
        "speaker_count_confidence": round(
            sum(float(turn.get("embedding_quality", 0.0)) for turn in turns)
            / max(1, len(turns)),
            4,
        ),
        "identity_merges": merge_history,
        "delayed_weak_assignments": delayed_assignments,
        "seed_similarity_threshold": round(seed_threshold, 4),
        "prototype_similarity_threshold": round(prototype_threshold, 4),
        "short_turn_assignment_threshold": round(assignment_threshold, 4),
        "sentence_continuity_threshold": round(
            SENTENCE_CONTINUITY_SIMILARITY, 4
        ),
        "raw_core_similarity_threshold": round(
            seed_threshold, 4
        ),
        "fragmentation_score": round(fragmentation_score, 4),
        "fragmentation_singleton_ratio": round(singleton_ratio, 4),
        "fragmentation_retry_triggered": fragmentation_retry_triggered,
        "fragmentation_retry_merges": fragmentation_retry_merges,
        "first_pass_speaker_count": first_pass_speaker_count,
        "second_pass_speaker_count": len(identity_clusters),
        "global_partition_changed": global_selection_changed,
        "global_partition_split_count": max(
            0, len(identity_clusters) - first_pass_speaker_count
        ),
        "global_partition_merge_count": max(
            0, first_pass_speaker_count - len(identity_clusters)
        ),
        "global_partition_candidate_count": candidate_count,
        "global_partition_selected_score": round(
            selected_metrics["score"], 4
        ),
        "global_partition_selected_metrics": {
            key: (round(value, 4) if isinstance(value, float) else value)
            for key, value in selected_metrics.items()
        },
        "global_partition_candidates": partition_audit,
        "identity_candidates": identity_candidates,
        "identity_threshold_calibration": {
            "pair_count": int(len(similarity_values)),
            "median": round(float(np.median(similarity_values)), 4)
            if len(similarity_values) else None,
            "q25": round(float(np.quantile(similarity_values, 0.25)), 4)
            if len(similarity_values) else None,
            "q75": round(float(np.quantile(similarity_values, 0.75)), 4)
            if len(similarity_values) else None,
            "adaptive_threshold": round(prototype_threshold, 4),
            "seed_threshold": round(seed_threshold, 4),
            "assignment_floor": round(assignment_threshold, 4),
            "confidence": round(calibration["confidence"], 4),
            "nearest_pair_count": int(len(nearest_similarity_values)),
            "nearest_median": (
                round(float(np.median(nearest_similarity_values)), 4)
                if len(nearest_similarity_values) else None
            ),
            "nearest_q25": (
                round(float(np.quantile(nearest_similarity_values, 0.25)), 4)
                if len(nearest_similarity_values) else None
            ),
            "nearest_q75": (
                round(float(np.quantile(nearest_similarity_values, 0.75)), 4)
                if len(nearest_similarity_values) else None
            ),
        },
        "final_identity_profiles": final_identity_profiles,
        "raw_speaker_profiles": {
            f"SPEAKER_{index:02d}": {
                "turn_count": profile["turn_count"],
                "duration": round(profile["duration"], 3),
                "reliable_turn_count": profile["reliable_turn_count"],
                "reliable_duration": round(
                    profile["reliable_duration"], 3
                ),
                "embedding_quality": round(profile["quality"], 4),
                "prototype_anchor": bool(profile["anchor"]),
                "prototype_eligible": bool(profile["prototype"]),
            }
            for index, profile in enumerate(initial_profiles)
        },
    }
    return identity_clusters, raw_to_final, diagnostics


def run_temporal_constrained_clustering(segments):
    """Cluster speaker turns while preserving local acoustic change evidence."""
    if not kaldiio or np is None:
        return False
    emb_scp = os.path.join(WORK_DIR, "embedding", "emb.scp")
    labels_out = os.path.join(WORK_DIR, "labels")
    windows = _read_embedding_windows(emb_scp)
    if len(windows) < 4:
        return False
    timestamp_candidates = _candidate_boundary_scores(segments, windows)
    acoustic_candidates = _acoustic_boundary_scores(segments, windows)
    candidates = timestamp_candidates + acoustic_candidates
    boundaries, thresholds = _select_change_boundaries(candidates, windows)
    boundaries = _add_structural_sentence_boundaries(
        boundaries, timestamp_candidates, windows
    )
    if not boundaries:
        return False
    structural_count = sum(
        bool(item.get("structural_partition")) for item in boundaries
    )

    threshold_values = list(thresholds.values())
    if threshold_values:
        threshold_text = (
            f"{threshold_values[0]:.3f}"
            if max(threshold_values) - min(threshold_values) < 0.001
            else f"{min(threshold_values):.3f}~{max(threshold_values):.3f}"
        )
    else:
        threshold_text = "n/a"
    print(
        f"✅ 分区自适应换人阈值: {threshold_text}; "
        f"{len(timestamp_candidates)} 个词间候选 + "
        f"{len(acoustic_candidates)} 个纯声学候选中保留 "
        f"{len(boundaries) - structural_count} 个声学换人边界 + "
        f"{structural_count} 个句末结构分段。"
    )
    for item in boundaries:
        boundary_label = (
            "结构分段" if item.get("structural_partition") else "换人候选"
        )
        print(
            f"   {boundary_label} {item['time']:.3f}s: "
            f"change={item['score']:.3f}, gap={item['gap']:.3f}s, "
            f"punctuation={item.get('punctuation', 'none')}, "
            f"constraint={item.get('constraint', 'soft')}, "
            f"source={item.get('candidate_source', 'word_timestamp')}, "
            f"{item['left_text']!r} -> {item['right_text']!r}"
        )

    (
        turns,
        hard_cannot_links,
        soft_cannot_links,
        boundary_evidence,
    ) = _build_acoustic_turns(windows, boundaries)
    if len(turns) < 2:
        return False
    # Only a hard acoustic boundary is a true cannot-link. A soft boundary is
    # retained as graded evidence for reconciliation; treating it as absolute
    # here was the main source of one-speaker-per-sentence fragmentation.
    legal_seed_similarities = {
        (left, right): float(
            turns[left]["embedding"] @ turns[right]["embedding"]
        )
        for left in range(len(turns))
        for right in range(left + 1, len(turns))
        if (left, right) not in hard_cannot_links
    }
    raw_seed_threshold = _adaptive_identity_thresholds(
        legal_seed_similarities, len(turns)
    )["seed"]
    raw_clusters = _constrained_turn_clusters(
        turns,
        hard_cannot_links,
        soft_links=soft_cannot_links,
        seed_threshold=raw_seed_threshold,
    )
    raw_clusters.sort(
        key=lambda cluster: min(turns[index]["start"] for index in cluster)
    )
    identity_clusters, raw_to_final, diagnostics = _reconcile_turn_clusters(
        raw_clusters,
        turns,
        hard_cannot_links,
        soft_cannot_links,
        boundary_evidence,
    )
    raw_turn_labels = {}
    final_turn_labels = {}
    for raw_label, cluster in enumerate(raw_clusters):
        for turn_index in cluster:
            raw_turn_labels[turn_index] = raw_label
            final_turn_labels[turn_index] = raw_to_final[raw_label]

    window_labels = {}
    for turn_index, turn in enumerate(turns):
        for window_index in turn["window_indices"]:
            # RTTM intentionally keeps raw labels so identity reconciliation
            # cannot remove a sentence/speaker boundary.
            window_labels[window_index] = raw_turn_labels[turn_index]
    if len(window_labels) != len(windows):
        return False

    speaker_profiles = {}
    audited_profiles = {
        item["speaker"]: item
        for item in diagnostics.get("final_identity_profiles", [])
    }
    for label, cluster in enumerate(identity_clusters):
        quality, reliable = _cluster_reliability(cluster, turns)
        speaker_name = f"SPEAKER_{label:02d}"
        audit = audited_profiles.get(speaker_name, {})
        speaker_profiles[speaker_name] = {
            "turn_count": len(cluster),
            "duration": round(sum(turns[index]["duration"] for index in cluster), 3),
            "embedding_quality": round(quality, 4),
            "reliable": reliable,
            "consistency_median": audit.get("consistency_median"),
            "consistency_q25": audit.get("consistency_q25"),
            "prototype_anchor": bool(audit.get("prototype_anchor")),
        }
    diagnostics["speaker_count"] = len(identity_clusters)
    diagnostics["speaker_count_preferred"] = len(identity_clusters)
    diagnostics["speaker_count_range"] = [
        diagnostics["speaker_count_lower_bound"],
        len(identity_clusters),
    ]
    diagnostics["speaker_profiles"] = speaker_profiles
    diagnostics["speaker_identity_map"] = {
        f"SPEAKER_{raw_label:02d}": f"SPEAKER_{final_label:02d}"
        for raw_label, final_label in sorted(raw_to_final.items())
    }
    diagnostics["turns"] = [
        {
            "start": round(turn["start"], 3),
            "end": round(turn["end"], 3),
            "duration": round(turn["duration"], 3),
            "embedding_window_count": turn["embedding_window_count"],
            "embedding_cohesion": round(turn["embedding_cohesion"], 4),
            "embedding_quality": round(turn["embedding_quality"], 4),
            "embedding_reliable": bool(turn["embedding_reliable"]),
            "speaker_raw": f"SPEAKER_{raw_turn_labels[index]:02d}",
            "speaker": f"SPEAKER_{final_turn_labels[index]:02d}",
        }
        for index, turn in enumerate(turns)
    ]
    diagnostics["boundary_counts"] = {
        "partition": sum(
            item.get("constraint") == "partition" for item in boundaries
        ),
        "soft": sum(item.get("constraint") == "soft" for item in boundaries),
        "hard": sum(item.get("constraint") == "hard" for item in boundaries),
        "identity": sum(
            item.get("constraint") == "identity" for item in boundaries
        ),
        "text_split_allowed": sum(
            bool(item.get("text_split_allowed", True)) for item in boundaries
        ),
        "identity_diagnostic_only": sum(
            item.get("boundary_role") == "identity_diagnostic_only"
            for item in boundaries
        ),
        "identity_separation_only": sum(
            item.get("boundary_role") == "identity_separation_only"
            for item in boundaries
        ),
    }
    diagnostics["turn_pair_similarity"] = [
        {
            "left_turn": left,
            "right_turn": right,
            **_pair_evidence_diagnostics(_cluster_pair_evidence(
                {left}, {right}, turns
            )),
            "hard_conflict": (left, right) in hard_cannot_links,
            "soft_conflict": (left, right) in soft_cannot_links,
        }
        for left in range(len(turns))
        for right in range(left + 1, len(turns))
    ]
    diagnostics["boundaries"] = [
        {
            "time": round(float(item["time"]), 3),
            "score": round(float(item["score"]), 4),
            "threshold": round(float(item.get("threshold", 0.0)), 4),
            "punctuation": item.get("punctuation", "none"),
            "constraint": item.get("constraint", "soft"),
            "identity_constraint": item.get(
                "identity_constraint", item.get("constraint", "soft")
            ),
            "same_asr_segment": bool(item.get("same_asr_segment")),
            "timestamp_reliable": bool(item.get("timestamp_reliable")),
            "left_window_count": int(item.get("left_count", 0)),
            "right_window_count": int(item.get("right_count", 0)),
            "left_cohesion": round(float(item.get("left_cohesion", 0.0)), 4),
            "right_cohesion": round(float(item.get("right_cohesion", 0.0)), 4),
            "context_seconds": round(float(item.get("context_seconds", 0.0)), 3),
            "hard_evidence": bool(item.get("hard_evidence", False)),
            "candidate_source": item.get("candidate_source", "word_timestamp"),
            "structural_partition": bool(item.get("structural_partition")),
            "text_split_allowed": bool(item.get("text_split_allowed", True)),
            "boundary_role": item.get("boundary_role", "speaker_and_text"),
            "persistent_acoustic_change": bool(
                item.get("persistent_acoustic_change", False)
            ),
            "persistence_count": int(item.get("persistence_count", 0)),
            "persistence_ratio": round(
                float(item.get("persistence_ratio", 0.0)), 4
            ),
        }
        for item in boundaries
    ]
    with open(CLUSTERING_DIAGNOSTICS_PATH, "w", encoding="utf-8") as stream:
        json.dump(diagnostics, stream, ensure_ascii=False, indent=2)

    os.makedirs(os.path.dirname(labels_out), exist_ok=True)
    with open(labels_out, "w", encoding="utf-8") as stream:
        for index, window in enumerate(windows):
            stream.write(f"{window['key']} {window_labels[index]}\n")

    print(
        f"✅ 时序约束聚类: {len(turns)} 个声学话轮 -> "
        f"{len(raw_clusters)} 个原始身份 -> {len(identity_clusters)} 位全局说话人"
        f"（硬约束 {len(hard_cannot_links)}, "
        f"软约束 {len(soft_cannot_links)}, 可靠话轮 {diagnostics['reliable_turn_count']}）。"
    )
    print(
        f"   自适应身份阈值: seed={diagnostics['seed_similarity_threshold']:.3f}, "
        f"reconnect={diagnostics['prototype_similarity_threshold']:.3f}, "
        f"assignment={diagnostics['short_turn_assignment_threshold']:.3f}, "
        f"sentence={diagnostics['sentence_continuity_threshold']:.3f}"
    )
    diagnostics["raw_core_similarity_threshold"] = round(
        raw_seed_threshold, 4
    )
    diagnostics["raw_core_soft_boundary_penalty"] = 0.06
    if diagnostics["fragmentation_retry_triggered"]:
        print(
            "   🔎 全局身份模型已替换局部结果: "
            f"speaker {diagnostics['first_pass_speaker_count']} -> "
            f"{diagnostics['second_pass_speaker_count']}, "
            f"objective={diagnostics['global_partition_selected_score']:.3f}, "
            f"candidates={diagnostics['global_partition_candidate_count']}"
        )
    else:
        print(
            "   🔎 全局身份模型保留局部结果: "
            f"speaker={diagnostics['second_pass_speaker_count']}, "
            f"objective={diagnostics['global_partition_selected_score']:.3f}, "
            f"candidates={diagnostics['global_partition_candidate_count']}"
        )
    if diagnostics["identity_merges"]:
        for item in diagnostics["identity_merges"]:
            print(
                "   身份合并 "
                f"{'+'.join(item['raw_speakers'])}: "
                f"similarity={item['similarity']:.3f}, "
                f"threshold={item['threshold']:.3f}, "
                f"reason={item['reason']}"
            )
    else:
        print("   身份合并: 无候选满足全局身份规则。")
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


def assign_speakers(segments, rttm_spans, diagnostics=None):
    from speaker_alignment import (
        AlignmentConfig,
        align_segments_with_rttm,
    )

    result = align_segments_with_rttm(
        segments,
        rttm_spans,
        config=AlignmentConfig(hard_silence_seconds=HARD_SILENCE_SECONDS),
        speaker_boundary_evidence=(diagnostics or {}).get("boundaries"),
    )
    split_count = max(0, len(result) - len(segments))
    ambiguous_count = sum(
        bool(item.get("speaker_boundary_ambiguous")) for item in result
    )
    if split_count:
        print(f"✅ 句子边界与 RTTM 联合分段: 新增 {split_count} 个片段。")
    if ambiguous_count:
        print(f"⚠️ {ambiguous_count} 个片段的说话人边界证据不足，已保守保留整段。")
    # Speaker labels are metadata.  Never concatenate text or extend timing
    # merely because two adjacent atoms currently share a label: a mistaken
    # label would otherwise destroy the evidence needed to correct it later.
    return result


def load_clustering_diagnostics():
    if not os.path.isfile(CLUSTERING_DIAGNOSTICS_PATH):
        return None
    try:
        with open(CLUSTERING_DIAGNOSTICS_PATH, "r", encoding="utf-8") as stream:
            payload = json.load(stream)
        return payload if isinstance(payload, dict) else None
    except (OSError, ValueError) as exc:
        print(f"⚠️ 无法读取说话人聚类诊断: {exc}")
        return None


def save_clustering_diagnostics(diagnostics):
    """Persist post-alignment counts as part of the clustering audit."""
    if not diagnostics:
        return
    temporary_path = CLUSTERING_DIAGNOSTICS_PATH + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as stream:
        json.dump(diagnostics, stream, ensure_ascii=False, indent=2)
    os.replace(temporary_path, CLUSTERING_DIAGNOSTICS_PATH)


def attach_speaker_confidence(segments, diagnostics):
    """Map global identities without changing text, timing, or segment count."""
    profiles = diagnostics.get("speaker_profiles") or {}
    raw_profiles = diagnostics.get("raw_speaker_profiles") or {}
    identity_map = diagnostics.get("speaker_identity_map") or {}
    inverse_identity_map = defaultdict(list)
    for raw_name, final_name in identity_map.items():
        inverse_identity_map[normalize_speaker(final_name)].append(
            normalize_speaker(raw_name)
        )
    enriched = []
    for source in segments:
        segment = source.copy()
        raw_speaker = normalize_speaker(segment.get("speaker"))
        final_speaker = normalize_speaker(
            identity_map.get(raw_speaker, raw_speaker)
        )
        segment["speaker_raw"] = raw_speaker
        segment["speaker"] = final_speaker
        segment["boundary_confidence"] = float(
            segment.get("speaker_assignment_confidence", 0.0)
        )
        profile = profiles.get(final_speaker) or {}
        raw_profile = raw_profiles.get(raw_speaker) or {}
        if profile or raw_profile:
            profile_quality = float(profile.get("embedding_quality", 0.0))
            raw_quality = float(
                raw_profile.get("embedding_quality", profile_quality)
            )
            consistency = profile.get("consistency_q25")
            if consistency is None:
                consistency = profile_quality
            identity_confidence = min(
                profile_quality,
                raw_quality,
                max(0.0, float(consistency)),
            )
            raw_member_count = max(
                1, len(inverse_identity_map.get(final_speaker) or [raw_speaker])
            )
            if raw_member_count > 1:
                identity_confidence *= max(
                    0.55, 1.0 - 0.08 * (raw_member_count - 1)
                )
            segment["identity_confidence"] = round(
                max(0.0, min(1.0, identity_confidence)), 4
            )
            # Compatibility field now reflects identity certainty rather than
            # local RTTM overlap or profile quality alone.
            segment["speaker_confidence"] = segment["identity_confidence"]
            segment["identity_raw_member_count"] = raw_member_count
            segment["identity_mapping_status"] = (
                "global_identity_merge"
                if raw_member_count > 1
                else "single_raw_identity"
            )
            segment["speaker_assignment"] = diagnostics.get(
                "algorithm", "prototype_identity_clustering_v2"
            )
        else:
            segment["identity_confidence"] = 0.0
            segment["speaker_confidence"] = 0.0
            segment["identity_mapping_status"] = "missing_identity_profile"
        segment["merge_confidence"] = None
        enriched.append(segment)
    return enriched


def update_dialogue_identity_diagnostics(segments, diagnostics):
    """Separate acoustic identities from identities visible in final text."""
    if not diagnostics:
        return diagnostics
    acoustic_speakers = sorted(
        (diagnostics.get("speaker_profiles") or {}).keys()
        or {
            normalize_speaker(value)
            for value in (diagnostics.get("speaker_identity_map") or {}).values()
        }
    )
    dialogue_speakers = sorted({
        normalize_speaker(segment.get("speaker"))
        for segment in segments
        if segment.get("speaker") is not None
    })
    diagnostics["acoustic_identity_count"] = len(acoustic_speakers)
    diagnostics["acoustic_identities"] = acoustic_speakers
    diagnostics["dialogue_assigned_identity_count"] = len(dialogue_speakers)
    diagnostics["dialogue_assigned_identities"] = dialogue_speakers
    diagnostics["unassigned_acoustic_identities"] = sorted(
        set(acoustic_speakers) - set(dialogue_speakers)
    )
    diagnostics["ambiguous_dialogue_segment_count"] = sum(
        bool(segment.get("speaker_boundary_ambiguous")) for segment in segments
    )
    diagnostics["unsplittable_multi_speaker_segment_count"] = sum(
        segment.get("speaker_assignment_status")
        == "multi_speaker_text_unsplittable"
        for segment in segments
    )
    diagnostics["sentence_segmentation_contract"] = "sentence_first_v1"
    diagnostics["long_internal_pause_segment_count"] = sum(
        bool(segment.get("contains_long_internal_pause"))
        for segment in segments
    )
    diagnostics["protected_internal_speaker_change_count"] = sum(
        len(segment.get("unsplit_identity_boundaries") or [])
        for segment in segments
    )
    return diagnostics


def reassemble_final_speaker_segments(segments):
    """Annotate logical continuity without modifying atomic text or timing."""
    annotated = []
    group_index = -1
    previous = None
    for source in segments:
        segment = source.copy()
        same_raw_identity = bool(
            previous
            and normalize_speaker(previous.get("speaker"))
            == normalize_speaker(segment.get("speaker"))
            and normalize_speaker(previous.get("speaker_raw"))
            == normalize_speaker(segment.get("speaker_raw"))
        )
        gap = (
            float(segment["start"]) - float(previous["end"])
            if previous else float("inf")
        )
        protected_boundary = bool(
            not previous
            or previous.get("hard_silence_after")
            or segment.get("hard_silence_before")
            or previous.get("speaker_boundary_ambiguous")
            or segment.get("speaker_boundary_ambiguous")
            or "speaker_boundary"
            in (previous.get("segment_split_reasons") or [])
            or "hard_silence_boundary"
            in (previous.get("segment_split_reasons") or [])
        )
        continuation = bool(
            same_raw_identity
            and 0.0 <= gap <= 0.80
            and not protected_boundary
        )
        if not continuation:
            group_index += 1
            member_index = 0
        else:
            member_index = int(previous.get("speaker_group_member_index", 0)) + 1
        segment["speaker_group_id"] = f"speaker_group_{group_index:04d}"
        segment["speaker_group_member_index"] = member_index
        segment["speaker_group_logical_only"] = True
        segment["speaker_continuation_candidate"] = continuation
        annotated.append(segment)
        previous = segment
    return annotated


def assign_speakers_to_fixed_atoms(atoms, rttm_spans):
    """Project an RTTM hypothesis onto existing atoms without splitting them."""
    projected = []
    for source in atoms:
        segment = source.copy()
        start = float(segment["start"])
        end = float(segment["end"])
        overlaps = speaker_overlaps(start, end, rttm_spans)
        overlap_total = sum(overlaps.values())
        if overlaps:
            raw_speaker = normalize_speaker(max(overlaps, key=overlaps.get))
            dominant = max(overlaps.values())
            dominance = dominant / overlap_total if overlap_total > 0.0 else 0.0
        else:
            raw_speaker = "SPEAKER_00"
            dominance = 0.0
        segment["speaker"] = raw_speaker
        segment["speaker_raw"] = raw_speaker
        segment["speaker_overlap_dominance"] = round(dominance, 4)
        segment["speaker_assignment_confidence"] = round(dominance, 4)
        segment["boundary_confidence"] = round(dominance, 4)
        segment["identity_confidence"] = None
        segment["speaker_confidence"] = None
        segment["merge_confidence"] = None
        segment["speaker_assignment_status"] = (
            "fixed_atom_pairwise_dominant_overlap"
            if dominance >= 0.80
            else "fixed_atom_pairwise_ambiguous_overlap"
        )
        segment["speaker_boundary_ambiguous"] = bool(
            len(overlaps) > 1 and dominance < 0.80
        )
        segment["speaker_overlap_candidates"] = [
            {
                "speaker": normalize_speaker(speaker),
                "seconds": round(seconds, 3),
                "ratio": round(
                    seconds / overlap_total if overlap_total > 0.0 else 0.0,
                    4,
                ),
            }
            for speaker, seconds in sorted(
                overlaps.items(), key=lambda item: item[1], reverse=True
            )
        ]
        segment["comparison_segmentation"] = "fixed_repaired_atoms_v1"
        projected.append(segment)
    return projected


def annotate_pairwise_boundary_disagreements(repaired, pairwise):
    """Flag possible over-merges without letting the baseline rewrite labels."""
    if len(repaired) != len(pairwise):
        return repaired
    annotated = [item.copy() for item in repaired]
    for index in range(1, len(annotated)):
        repaired_change = normalize_speaker(annotated[index - 1].get("speaker")) != (
            normalize_speaker(annotated[index].get("speaker"))
        )
        pairwise_change = normalize_speaker(pairwise[index - 1].get("speaker")) != (
            normalize_speaker(pairwise[index].get("speaker"))
        )
        if pairwise_change and not repaired_change:
            for target in (annotated[index - 1], annotated[index]):
                target["identity_merge_disputed_by_pairwise"] = True
                confidence = target.get("identity_confidence")
                if confidence is not None:
                    target["identity_confidence"] = round(
                        min(float(confidence), 0.65), 4
                    )
                    target["speaker_confidence"] = target["identity_confidence"]
        elif repaired_change and not pairwise_change:
            annotated[index]["identity_split_disputed_by_pairwise"] = True
    return annotated


def build_clustering_comparison(repaired, pairwise):
    """Compare labels on one immutable set of sentence atoms."""
    overlap_by_pair = defaultdict(float)
    pairwise_duration = defaultdict(float)
    for baseline in pairwise:
        baseline_start = float(baseline["start"])
        baseline_end = float(baseline["end"])
        baseline_speaker = str(baseline.get("speaker", "SPEAKER_00"))
        pairwise_duration[baseline_speaker] += max(
            0.0, baseline_end - baseline_start
        )
        for candidate in repaired:
            overlap = max(
                0.0,
                min(baseline_end, float(candidate["end"]))
                - max(baseline_start, float(candidate["start"])),
            )
            if overlap:
                overlap_by_pair[
                    (baseline_speaker, str(candidate.get("speaker", "SPEAKER_00")))
                ] += overlap

    mapping = {}
    for baseline_speaker in pairwise_duration:
        candidates = [
            (seconds, repaired_speaker)
            for (source, repaired_speaker), seconds in overlap_by_pair.items()
            if source == baseline_speaker
        ]
        if candidates:
            mapping[baseline_speaker] = max(candidates)[1]

    matched = sum(
        seconds
        for (baseline_speaker, repaired_speaker), seconds in overlap_by_pair.items()
        if mapping.get(baseline_speaker) == repaired_speaker
    )
    total = sum(pairwise_duration.values())
    repaired_labels = [
        normalize_speaker(item.get("speaker")) for item in repaired
    ]
    pairwise_labels = [
        normalize_speaker(item.get("speaker")) for item in pairwise
    ]
    same_atomic_units = bool(
        len(repaired) == len(pairwise)
        and all(
            left.get("atomic_segment_id") == right.get("atomic_segment_id")
            and abs(float(left["start"]) - float(right["start"])) < 1e-6
            and abs(float(left["end"]) - float(right["end"])) < 1e-6
            and str(left.get("text", "")) == str(right.get("text", ""))
            for left, right in zip(repaired, pairwise)
        )
    )
    coassignment = {
        "both_same": 0,
        "repaired_same_pairwise_different": 0,
        "repaired_different_pairwise_same": 0,
        "both_different": 0,
    }
    if same_atomic_units:
        for left in range(len(repaired_labels)):
            for right in range(left + 1, len(repaired_labels)):
                repaired_same = repaired_labels[left] == repaired_labels[right]
                pairwise_same = pairwise_labels[left] == pairwise_labels[right]
                if repaired_same and pairwise_same:
                    coassignment["both_same"] += 1
                elif repaired_same:
                    coassignment["repaired_same_pairwise_different"] += 1
                elif pairwise_same:
                    coassignment["repaired_different_pairwise_same"] += 1
                else:
                    coassignment["both_different"] += 1
    repaired_boundaries = []
    pairwise_boundaries = []
    if same_atomic_units:
        repaired_boundaries = [
            repaired_labels[index - 1] != repaired_labels[index]
            for index in range(1, len(repaired_labels))
        ]
        pairwise_boundaries = [
            pairwise_labels[index - 1] != pairwise_labels[index]
            for index in range(1, len(pairwise_labels))
        ]
    mapped_sources = defaultdict(set)
    for baseline_speaker, repaired_speaker in mapping.items():
        mapped_sources[repaired_speaker].add(baseline_speaker)
    mapping_collisions = {
        repaired_speaker: sorted(sources)
        for repaired_speaker, sources in mapped_sources.items()
        if len(sources) > 1
    }
    return {
        "version": 2,
        "comparison_unit": "immutable_sentence_atom",
        "same_atomic_units": same_atomic_units,
        "atomic_segment_count": len(repaired) if same_atomic_units else None,
        "production_algorithm": "stable_core_identity_consensus_v11_sentence_first",
        "baseline_algorithm": "wespeaker_pairwise_cosine_average_linkage",
        "pairwise_distance_threshold": _env_float(
            "SPEAKER_CLUSTER_THRESHOLD", 0.5
        ),
        "repaired": {
            "speaker_count": len({str(item.get("speaker")) for item in repaired}),
            "segment_count": len(repaired),
            "ambiguous_segment_count": sum(
                bool(item.get("speaker_boundary_ambiguous")) for item in repaired
            ),
        },
        "wespeaker_pairwise": {
            "speaker_count": len({str(item.get("speaker")) for item in pairwise}),
            "segment_count": len(pairwise),
            "ambiguous_segment_count": sum(
                bool(item.get("speaker_boundary_ambiguous")) for item in pairwise
            ),
        },
        "baseline_to_repaired_label_map": mapping,
        "many_to_one_mapping_collisions": mapping_collisions,
        "many_to_one_mapping_collision_count": len(mapping_collisions),
        "coassignment": coassignment,
        "adjacent_boundary_disagreement": {
            "repaired_missed_pairwise_change": sum(
                baseline and not candidate
                for candidate, baseline in zip(
                    repaired_boundaries, pairwise_boundaries
                )
            ),
            "repaired_extra_change": sum(
                candidate and not baseline
                for candidate, baseline in zip(
                    repaired_boundaries, pairwise_boundaries
                )
            ),
            "both_detect_change": sum(
                candidate and baseline
                for candidate, baseline in zip(
                    repaired_boundaries, pairwise_boundaries
                )
            ),
        },
        "duration_weighted_label_agreement": round(
            matched / total if total > 0.0 else 0.0, 4
        ),
        "note": (
            "Agreement is label-invariant after maximum-overlap mapping; "
            "it is not a ground-truth accuracy score. Coassignment and "
            "boundary disagreements separate merge risk from split risk."
        ),
    }


def main():
    print("=== [Step 2] WeSpeaker + 时序约束聚类 说话人分离 ===")
    print(f"当前 Python: {sys.executable}")
    state = load_state()
    if "segments_step1" not in state:
        print("❌ state 中缺少 segments_step1，请先运行 Step 1a/1b")
        sys.exit(1)
    unresolved = state.get("asr_unresolved_dialogue_regions") or []
    if unresolved and _env_bool("ASR_REQUIRE_COMPLETE_DIALOGUE", False):
        spans = ", ".join(
            f"{float(item['start']):.2f}-{float(item['end']):.2f}s"
            for item in unresolved[:8]
        )
        print(f"❌ 仍有确认对白未恢复，拒绝执行说话人聚类: {spans}")
        sys.exit(2)
    if unresolved:
        spans = ", ".join(
            f"{float(item['start']):.2f}-{float(item['end']):.2f}s"
            for item in unresolved[:8]
        )
        print(f"⚠️ 未恢复对白已记入降级报告，继续说话人聚类: {spans}")

    prepare_inputs(state)
    rttm_path = run_wespeaker_pipeline(state["segments_step1"])
    print("\n=== [Step 2-W] 解析 RTTM 回填 AutoDub Segments ===")
    rttm_spans = parse_rttm(rttm_path)
    diagnostics = load_clustering_diagnostics()
    aligned = assign_speakers(
        state["segments_step1"], rttm_spans, diagnostics=diagnostics
    )
    if diagnostics:
        aligned = attach_speaker_confidence(aligned, diagnostics)
        aligned = reassemble_final_speaker_segments(aligned)
        update_dialogue_identity_diagnostics(aligned, diagnostics)
        save_clustering_diagnostics(diagnostics)

    clustering_comparison = None
    pairwise_aligned = None
    if (
        os.getenv("AUTODUB_DIAR_CLUSTER_MODE", "compare").strip().lower()
        == "compare"
        and os.path.isfile(PAIRWISE_RTTM_PATH)
    ):
        pairwise_spans = parse_rttm(PAIRWISE_RTTM_PATH)
        # A/B must use exactly the production atoms.  Letting each hypothesis
        # split and merge its own text made speaker counts incomparable and
        # allowed an over-merged result to score deceptively well.
        pairwise_aligned = assign_speakers_to_fixed_atoms(
            aligned, pairwise_spans
        )
        clustering_comparison = build_clustering_comparison(
            aligned, pairwise_aligned
        )
        aligned = annotate_pairwise_boundary_disagreements(
            aligned, pairwise_aligned
        )
        clustering_comparison.update({
            "repaired_labels_path": os.path.join(WORK_DIR, "labels_repaired"),
            "repaired_rttm_path": rttm_path,
            "pairwise_labels_path": PAIRWISE_LABELS_PATH,
            "pairwise_rttm_path": PAIRWISE_RTTM_PATH,
        })
        pairwise_segments_path = os.path.join(
            WORK_DIR, "segments_step2_wespeaker_pairwise.json"
        )
        with open(pairwise_segments_path, "w", encoding="utf-8") as stream:
            json.dump(pairwise_aligned, stream, ensure_ascii=False, indent=2)
        clustering_comparison["pairwise_segments_path"] = pairwise_segments_path
        with open(CLUSTER_COMPARISON_PATH, "w", encoding="utf-8") as stream:
            json.dump(clustering_comparison, stream, ensure_ascii=False, indent=2)
        print(
            "🔬 聚类 A/B: "
            f"修复版={clustering_comparison['repaired']['speaker_count']} 人，"
            f"WeSpeaker两两={clustering_comparison['wespeaker_pairwise']['speaker_count']} 人，"
            "映射后时长一致率="
            f"{clustering_comparison['duration_weighted_label_agreement']:.1%}"
        )

    unique = sorted(set(s["speaker"] for s in aligned))
    print(f"🔍 最终识别出 {len(unique)} 个说话人: {unique}")
    if diagnostics and diagnostics.get("unassigned_acoustic_identities"):
        print(
            "⚠️ 声学身份数与文本可分配身份数不同: "
            f"acoustic={diagnostics['acoustic_identity_count']}, "
            f"dialogue={diagnostics['dialogue_assigned_identity_count']}, "
            "未安全落到独立文本段的身份="
            f"{diagnostics['unassigned_acoustic_identities']}"
        )

    removed = invalidate_downstream_state(state)
    if removed:
        print(
            "🧹 Step 2 结果已变化，清除下游旧缓存: "
            + ", ".join(removed)
        )
    state["segments_step2"] = aligned
    if clustering_comparison is not None:
        state["speaker_clustering_comparison"] = clustering_comparison
        state["segments_step2_wespeaker_pairwise"] = pairwise_aligned
    if diagnostics:
        state["speaker_clustering"] = diagnostics
        state["estimated_speaker_count"] = diagnostics.get(
            "speaker_count_preferred", len(unique)
        )
        state["estimated_speaker_range"] = diagnostics.get(
            "speaker_count_range", [len(unique), len(unique)]
        )
        state["speaker_count_confidence"] = diagnostics.get(
            "speaker_count_confidence", 0.0
        )
        state["dialogue_assigned_speaker_count"] = diagnostics.get(
            "dialogue_assigned_identity_count", len(unique)
        )
    else:
        for key in (
            "speaker_clustering",
            "estimated_speaker_count",
            "estimated_speaker_range",
            "speaker_count_confidence",
            "dialogue_assigned_speaker_count",
        ):
            state.pop(key, None)
    save_state(state)
    print("\n✅ Step 2 完成。请切回 autodub_server 环境继续运行 Step 3。")


if __name__ == "__main__":
    main()
