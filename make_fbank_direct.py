#!/usr/bin/env python3
"""直接提取 fbank，绕过 make_fbank.sh 的 shell 黑盒"""

import argparse
import os
import numpy as np
import soundfile as sf
import torch
import kaldiio
import torchaudio.compliance.kaldi as kaldi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scp", required=True)
    parser.add_argument("--segments", required=True)
    parser.add_argument("--ark-path", required=True)
    parser.add_argument("--subseg-cmn", default="true")
    args = parser.parse_args()

    # 1. 读音频（自动处理立体声转单声道）
    with open(args.scp) as f:
        utt_id, wav_path = f.readline().strip().split(None, 1)

    data, sr = sf.read(wav_path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)  # 立体声/多声道 → 单声道
    waveform = torch.from_numpy(data).unsqueeze(0)  # (1, samples)

    # 2. 重采样到 16kHz（WeSpeaker 默认）
    if sr != 16000:
        import torchaudio
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
        sr = 16000

    # 3. 读 segments（4列: seg_id utt_id begin end）
    with open(args.segments) as f:
        segs = [line.strip().split() for line in f if line.strip()]

    # 4. 提取 fbank 并写入 Kaldi ark/scp
    os.makedirs(os.path.dirname(args.ark_path), exist_ok=True)
    scp_path = args.ark_path.replace(".ark", ".scp")

    with kaldiio.WriteHelper(f"ark,scp:{args.ark_path},{scp_path}") as writer:
        for seg_id, utt, b, e in segs:
            s = int(float(b) * sr)
            t = int(float(e) * sr)
            seg_wav = waveform[:, s:t]

            # 跳过极短片段（不足 400 采样点）
            if seg_wav.shape[-1] < 400:
                print(f"⚠️ 跳过过短片段 {seg_id}: {seg_wav.shape[-1]} samples")
                continue

            fbank = kaldi.fbank(
                seg_wav,  # 确保一维 (samples,)
                htk_compat=True,
                sample_frequency=sr,
                use_energy=False,
                window_type="hanning",
                num_mel_bins=80,
                dither=0.0,
                frame_shift=10,
            )

            if args.subseg_cmn.lower() == "true":
                fbank = fbank - fbank.mean(dim=0, keepdim=True)

            writer(seg_id, fbank.numpy())

    print(f"✅ 成功写入 fbank 到 {args.ark_path}")


if __name__ == "__main__":
    main()
