# AutoDub v2.0 Scheme 3 五模型 TTS Router

本包已按 `AutoDub_v2.2.1_translation_emotion_tts_issues.md` 修补，见
`CHANGELOG_v2.2.1.md`。默认 TTS 后端为 scheme3，配置优先级为命令行 > 环境变量 > 默认值。

2026-10-08 定向加入已在 `ocr_test` 验证的 OCR 2.1.1 Step 1b，不覆盖主入口，
不改翻译、情绪、LightTTS、Scheme3 policy 或五模型 worker。新增
`ocr_v211_integration.py`、`autodub_ocr_runtime/` 和已验证的 `ocr_server_v2.py`。

Step 1b 默认为 `frozen_b`：按片段 20%/50%/80% 三帧抽取 HunyuanOCR，
进行结构清洗、局部候选构造、Frozen B classifier 和完整保守 gate。
无合格证据时保留原 ASR；结果写入原有 `segments_step1`、
`segments_ocr_corrected`，新增 `ocr_optimized_manifest` 和段级审计元数据。
`--step 1b` 与 `--step all` 调用同一新入口，重跑会使下游旧结果失效。
帧缓存及审计位于 `temp/<视频名>/ocr_optimized/`。

```bash
python auto_dubbing_ver_2.0.py --ocr-preflight
export AUTODUB_OCR_MODE=frozen_b
export AUTODUB_OCR_MAX_CALLS=6
export AUTODUB_OCR_MAX_ATTEMPTS=2
# DEEPSEEK_API_KEY 只从环境或交互输入取得，不写入包或日志。
python -u auto_dubbing_ver_2.0.py "$AUTODUB_VIDEO" --lang zh --step 1b --require-ocr
```

`AUTODUB_OCR_MODE=off` 或 `--disable-ocr` 用于不做 OCR 的对照；
`AUTODUB_OCR_MODE=legacy` 显式保留旧 classifier 兼容模式。
`--ocr-preflight` 是零网络自检，不代表真实视频准确率提升。

## 1. 新流程

```text
Step 1a  Demucs + Qwen3-ASR
Step 1b  HunyuanOCR + DeepSeek 保守纠错
Step 2   说话人识别
Step 2.5 emotion2vec_plus_base（逐 segment）
Step 3   逐句校验/补译 -> 1-5 个候选（极短句1-2个） -> 轻量 TTS 时长筛选 -> Scheme 3 路由
         + output/<视频名>_segments.json
Step 4   按 tts_engine 分组，每个模型启动一个隔离 worker
Step 5   完整性/语言/分离器/生成指纹/WAV校验 -> 混音并生成 final_video.mp4
```

Scheme 3 的优先级固定为：

```text
非 zh/en                         -> omnivoice
zh/en 长句或复杂句               -> confucius4
zh/en angry/sad/fearful/disgusted -> indextts2
zh/en happy/surprised             -> cosyvoice3
zh/en neutral 普通短句             -> f5tts
```

长句/复杂句规则优先于情绪。Step3 使用 `target_lang + emotion + text` 保存
`tts_engine`；Step4 默认执行这个字段，fallback 单独记录 actual_engine，不改写计划路由。

## 2. 文件

- `auto_dubbing_ver_2.0.py`：主流程，保留 baseline/scheme2，增加 scheme3。
- `scheme3_policy.py`：语言、七类情绪、文本复杂度、路由和 JSON 的单一规则源。
- `emotion_recognizer.py`：主进程切片、断点合并和旧 checkpoint 迁移。
- `emotion_worker.py`：独立环境一次加载 emotion2vec，批量推理。
- `light_tts_selector.py`：生成候选试听、测量实际时长、选择最接近原片段的译文并断点保存。
- `tts_router.py`：参考音频选择、分组调度、时长适配和断点保存。
- `tts_worker.py`：F5-TTS、IndexTTS2、CosyVoice3、Confucius4、OmniVoice 批量 worker。
- `test_scheme3_policy.py`：纯逻辑路由、情绪、复杂度和 JSON 测试。
- `test_scheme3_resume.py`：断点跳过时的计时和引擎元数据保留测试。
- `test_light_tts_selector.py`：候选格式、时长最优选择、路由顺序和 selector resume 测试。

## 3. 语言与情绪

语言对外统一为 `zh / en / other`。对于日语等语言，JSON 中
`target_lang=other`，另保留 `target_lang_raw=ja` 供 OmniVoice 合成时使用。

情绪只允许：

```text
neutral angry sad fearful disgusted happy surprised
```

`fear -> fearful`、`disgust -> disgusted`、`surprise -> surprised`。未知标签、
低置信度或过短片段标记原因，可参考同一说话人的相邻可靠结果，否则回退为 neutral。
原始预测保留在 `raw_emotion`、`emotion_raw_label`、`emotion_raw_score`，详见变更说明。

可调参数：

```bash
export AUTODUB_EMOTION_MIN_SCORE=0.5
export AUTODUB_EMOTION_MIN_DURATION=0.8
```

## 4. 长句/复杂句

MD 只规定“长句/复杂句优先”，没有指定数值。本项目默认为：

```text
中文：不少于 28 个 CJK 字符
英文：不少于 20 个单词
复杂句：至少 3 个逗号/分号/冒号分隔的分句
```

可集中调整：

```bash
export AUTODUB_SCHEME3_ZH_LONG_CHARS=28
export AUTODUB_SCHEME3_EN_LONG_WORDS=20
export AUTODUB_SCHEME3_COMPLEX_CLAUSES=3
```

## 5. 五个 TTS 环境

```text
f5tts       env=f5-tts
            repo=/data0/goldenseed/cjx/F5-TTS
            model=/data0/goldenseed/cjx/F5-TTS/ckpts/F5TTS_v1_Base/model_1250000.safetensors

indextts2   env=indextts
            root=/data0/goldenseed/models/index-tts

cosyvoice3  env=cosyvoice3
            repo=/data0/goldenseed/qgd/CosyVoice

confucius4  env=confucius4_tts
            repo=/data0/goldenseed/qgd/Confucius4-TTS

omnivoice   env=omnivoice
            model=/data0/goldenseed/qgd/models/OmniVoice
```

worker 全部设置 `PYTHONNOUSERSITE=1`。F5-TTS 使用本地 checkpoint/vocoder；
IndexTTS2 复用已验证 baseline API；OmniVoice 不传情绪 instruct。

## 6. 运行

不要使用 `--step all`。Step 1a—3 按原 README 逐步运行，包括：

```bash
conda activate autodub_server
python -u auto_dubbing_ver_2.0.py "$AUTODUB_VIDEO" --lang zh --step emotion
python -u auto_dubbing_ver_2.0.py "$AUTODUB_VIDEO" --lang zh --step 3
```

Step 3 会生成 `output/<视频名>_segments.json`。

Step 3 默认优先使用 `edge-tts` 作为轻量试听后端，未安装时自动尝试
`espeak-ng/espeak`。它只负责候选时长筛选，不替代或修改五个正式 TTS worker。

```bash
pip install -r light_tts_requirements.txt
# 可选：固定离线后端
export AUTODUB_LIGHT_TTS_BACKEND=espeak
```

如需兼容旧流程临时关闭筛选，可设置 `AUTODUB_LIGHT_TTS_ENABLED=0`。默认开启。

翻译候选以源 segment、源/目标语言和翻译模型生成请求指纹，并保存到 project
state。指纹一致时直接复用候选，避免 Step 3 重启后候选变化导致试听缓存失效。

运行 Scheme 3：

```bash
conda activate autodub_server
python -u auto_dubbing_ver_2.0.py "$AUTODUB_VIDEO" \
  --lang zh --step 4 --tts-backend scheme3
python -u auto_dubbing_ver_2.0.py "$AUTODUB_VIDEO" --lang zh --step 5
```

原有方案仍保留：

```bash
# IndexTTS2 baseline（从主环境启动隔离 indextts worker）
python -u auto_dubbing_ver_2.0.py "$AUTODUB_VIDEO" \
  --lang zh --step 4 --tts-backend indextts

# Scheme 2（router 是兼容别名）
python -u auto_dubbing_ver_2.0.py "$AUTODUB_VIDEO" \
  --lang zh --step 4 --tts-backend scheme2
```

## 7. JSON 结构

每个 segment 至少包含：

```json
{
  "text": "最终用于TTS的文本",
  "translations_candidates": [
    {"text": "候选1", "light_tts_duration": 1.8, "duration_error": 0.2},
    {"text": "候选2", "light_tts_duration": 2.0, "duration_error": 0.0},
    {"text": "候选3", "light_tts_duration": 2.4, "duration_error": 0.4}
  ],
  "start": 1.2,
  "end": 3.8,
  "source_lang": "en",
  "target_lang": "zh",
  "emotion": "angry",
  "speaker": "SPEAKER_00",
  "tts_engine": "indextts2"
}
```

额外保留原始情绪、置信度、复杂度、生成时长和断点信息。
`light_tts_selection` 记录被选候选、试听文件、目标时长、实际时长、误差和后端。
轻量 TTS 选择完成后才调用 `scheme3_policy`，因此正式路由使用最终选中的 `text`。

## 8. 断点续跑

- 情绪 worker 每段原子保存；旧 `other` checkpoint 会按七类规则重新归一化。
- 轻量 TTS 每个候选完成后原子写入
  `temp/<视频>/light_tts_selector/selection_results.json`；候选文本、目标时长、
  语言和后端配置指纹一致且试听文件存在时直接复用。
- Scheme 2 仍使用 `temp/<视频>/tts_router`。
- Scheme 3 独立使用 `temp/<视频>/tts_router_scheme3`，不会误复用 Scheme 2 WAV。
- 已成功且引擎匹配的 WAV 直接 skip，原有 `inference_seconds`、`rtf`、
  `actual_duration`、`selected_model`、`tts_engine` 不被清空。
- Scheme 3 额外校验文本、语言、情绪、参考音频和时长的请求指纹；指纹改变时
  只重生受影响的 segment，不误用旧 WAV。
- 强制重算使用 `--force-emotion` 或 `--force-tts`。

## 9. 本地检查与服务器验证

本地可执行：

```bash
python -m py_compile auto_dubbing_ver_2.0.py scheme3_policy.py \
  emotion_recognizer.py emotion_worker.py light_tts_selector.py \
  tts_router.py tts_worker.py
python test_scheme3_policy.py
python test_scheme3_resume.py
python test_light_tts_selector.py
python test_translation_emotion_tts_issues.py
python tts_preflight.py --models f5tts indextts2 cosyvoice3 confucius4 omnivoice
```

本地测试不等于 GPU 模型验收。上传服务器后必须分别执行五模型单条、
批量分组、resume 和短视频 Step 3—5，再能声称方案三端到端成功。

## 2026-10-08：Step4 / Step5 实测问题修复

- 保持 `strict_offline` 禁网；仅在隔离 worker 中将 HuggingFace 的文件和
  snapshot 加载显式设为 `local_files_only=True`，不再先发起元数据网络请求。
- IndexTTS2 preflight 增加实际运行需要的 MaskGCT、w2v-bert、CAMPPlus、
  BigVGAN 与情绪/说话人矩阵检查；使用已支持的 torch CUDA 实现，
  `use_cuda_kernel=False`，不安装 Ninja、不编译扩展、不修改模型仓库。
- 优先使用总长不超过 10 秒的完整 ASR 参考片段，保持音频与原文对应。
  F5 如支持原生 `fix_duration`，传入“预处理后参考时长 + 目标输出时长”，
  避免跨语言字节数估算造成配音过短；旧 API 仍兼容。
- Scheme3 policy、Step3 保存的路由、OCR frozen_b、LightTTS、Step5 完整性
  gate 与时长质量阈值均不变。worker 文件指纹变化自动使旧 TTS 缓存失效。
- 新回归测试：`python test_tts_offline_duration.py`；真实验收需重新运行
  `--step 4`、`--step 5`，不能用 `--allow-missing-tts` 掩盖失败。
