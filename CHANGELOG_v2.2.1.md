# 翻译、情绪与 TTS 问题修补（2026-10-07）

此包基于 `autodub_v2_scheme3_lighttts_final.zip`。OCR、人声分离、ASR 和说话人分离算法未修改。
下述命令用于解压后的独立工程。本次交付仅更新 ZIP，没有部署到现有流水线。

## 翻译

- 按 segment 独立校验 ID、主译文、全部候选、目标文字脚本与异常原文复用。
- 中文目标必须含中文且不能以英文为主体；英文目标拒绝中文残留；其他支持语种检查
  对应脚本。对法德西等共用拉丁字母语言，脚本检查不能代替语言识别，仍依赖模型与人工抽验。
- 批量合格句保留；漏 ID、缺字段、空值、错误目标语言的句子单独补译。
  默认额外重试2次，`AUTODUB_TRANSLATION_RETRIES` 可设0-3。SDK 自动重试关闭，避免次数叠加。
- 补译失败保存 `translation_status=failed`、ID、错误及尝试次数，`text` 置空；Step3停止，
  不把原文静默送入 TTS。成功句立即保存 checkpoint，恢复时只请求失败或指纹变化的句子。
- 极短句允许1-2个自然候选；其他句3-5个。长度接近仅警告，不判失败。
- 每句包含 previous_text/next_text（含跨批次上下文）；只翻译当前片段。
  相邻同 speaker 的极短残片保存 `translation_merge_suggestion`，不自动更改分段或时间戳。
- 指纹包含规则、语言校验、提示词、模型、源/目标语言、文本、时间范围、说话人与上下文。
  同语言输入显式 passthrough。专有名词确需原样保留时，可在对应源 segment 显式设置
  `translation_allow_identity=true`；此标志不绕过目标脚本要求。
- 局部重算：

```bash
python auto_dubbing_ver_2.0.py "$AUTODUB_VIDEO" --lang zh --step 3 --retranslate-ids 0,7
```

也可在 `segments_translation_candidates` 中修正已成功句子的 text/candidates；保留其请求指纹，
重跑 Step3 时会重新校验这些候选，再按文本指纹刷新对应试听与正式 WAV。
不要直接更改 segments_step3 绕过 Step3 校验。

## 轻量 TTS

- eSpeak启动读取 --voices，在实际支持的 cmn/zh/zh-cmn 中选择普通话音色。
- 第一轮筛选前执行该音色最小合成检查；不可用后端快速停止。
- 单个候选失败继续其他候选，记录错误；单个 segment 全失败可用文本长度估算，明确标记
  light_tts_status=fallback、duration_source=text_estimate，估算值不冒充测量值。
- 至少2个 segment 全失败且失败比例高于阈值时停止；
  `AUTODUB_LIGHT_TTS_MAX_FALLBACK_FRACTION=0.5`。
- 使用现有缓存结构；候选文本、时间、语言或后端变化使对应缓存失效。

## 情绪

- 保存原预测与最终路由情绪、emotion_status（success/too_short/low_confidence/failed）、
  emotion_reliable、emotion_fallback_reason、emotion_model。
- 仅为短/低可靠片段参考同 speaker、1.5秒以内的相邻可靠结果；邻居情绪冲突时保留 neutral。
  可靠的真实情绪变化不平滑覆盖。继承结果标注 same_speaker_neighbor 和邻句 IDs。
- 默认时长阈值0.8秒、分数阈值0.5；可按类设置，例如 `AUTODUB_EMOTION_MIN_SCORE_ANGRY=0.7`。
  `AUTODUB_EMOTION_CONTEXT_GAP` 配置邻接最大间隔。
- 输出类别数量与分数范围/均值，检测分数饱和。模型原分数不宣称为校准后的概率；
  真正的阈值校准仍需带标签的验证集。
- Scheme3 的五模型优先级保持原规则，改为使用上游已经确认的可靠/平滑后情绪。

## TTS 模型、缓存与离线预检

- CLI --tts-backend > AUTODUB_TTS_BACKEND > scheme3。显式 indextts baseline 也通过隔离
  worker运行，主环境不直接导入模型。
- 所有 worker 清理代理/PYTHONPATH污染并设置 PYTHONNOUSERSITE=1，F5单独补充其src路径。
- 每次运行在首个 worker前检查所有计划模型的本地文件及HF依赖；缺失时报告具体文件、
  当前缓存与修复命令，避免部分生成后才暴露缺失。
- HF_HOME、HF_HUB_CACHE、HUGGINGFACE_HUB_CACHE、TRANSFORMERS_CACHE 按模型统一设置。
  独立变量（也支持对应 *_HF_HUB_CACHE）：

```bash
export AUTODUB_F5_HF_HOME=/home/goldenseed/.cache/huggingface
export AUTODUB_CONFUCIUS_HF_HOME=/data0/goldenseed/qgd/hf_cache
# 另有 AUTODUB_INDEXTTS2_HF_HOME / AUTODUB_COSYVOICE3_HF_HOME / AUTODUB_OMNIVOICE_HF_HOME
export AUTODUB_TTS_OFFLINE_MODE=strict_offline
python tts_preflight.py --models f5tts confucius4
# 有意补全缓存时独立运行；本次交付未执行下载：
python tts_preflight.py --models f5tts confucius4 --download
```

strict_offline为默认；`allow_download`显式允许预检补全HF资源。本地checkpoint需用户预先提供，
预检不会安装环境或模型代码。预检验证文件存在、非空、HF main引用及分片完整，不承诺二进制/GPU兼容。
F5检查 Vocos；Confucius读取其真实 YAML paths，检查t2s/s2a、CAMPPlus、w2v-bert、BigVGAN等依赖。
CosyVoice ONNX默认CPU（避免无效CUDA provider）；可设 `AUTODUB_COSY_ONNX_PROVIDER=cuda`。
模型打印实际Torch/ONNX设备和文字前端。严格离线时未缓存的ModelScope资源请求立即失败，
不会挂到失效代理重试；wetext/ttsfrd缓存和GPU二进制仍需在已有模型环境单独验收。

## Fallback、时长门禁与恢复

- 默认链 Confucius4 -> F5 -> IndexTTS2；CosyVoice3 -> F5；IndexTTS2 -> F5；F5 -> CosyVoice3。
  非中英默认不跨到不支持语言的模型。每句无循环，fallback模型先预检。
- 自定义 `AUTODUB_TTS_FALLBACK_CHAINS` 为JSON字典；严格实验使用 `--disable-tts-fallback`。
  禁用时不复用以前由fallback模型产生的WAV。
- 记录 planned_engine、actual_engine、fallback_reason、fallback_count；Step3 tts_engine 保持计划值。
- WAV生成后先做现有atempo适配，再检查误差；超限时先重选候选重试，再尝试模型fallback。
  记录 tts_text（实际合成文本）、planned_text、duration_error、duration_ratio、duration_quality。
- 默认容忍误差为 max(0.6秒, 目标时长*25%)；超出则判失败，避免WAV存在即算成功。
  可设 AUTODUB_TTS_MAX_DURATION_ERROR、AUTODUB_TTS_MAX_DURATION_RATIO_ERROR。
  AUTODUB_TTS_DURATION_RETRIES默认1，可设0-3。
- 每个worker只收到待处理列表；每轮同一模型加载一次。候选重试或fallback若重新进入同一模型，
  需另开一次worker。无待处理模型不加载。
- 全生成指纹包含文本、语言、情绪、参考WAV内容、参考文字、目标时长、模型、模型路径/缓存、
  路由规则版本、worker版本；保存WAV SHA256。文本或模型配置变化只重算受影响句子。
- Step5检查全部segment/WAV数量与IDs、状态、可读性、正时长、内容SHA256、生成指纹、
  Step4成功数以及语言/分离器/媒体状态；旧状态缺少新manifest须先重跑Step4。
  明确接受缺句才使用 `--allow-missing-tts`，输入视频或总fingerprint不一致仍不允许。

## 验证

```bash
python -m py_compile auto_dubbing_ver_2.0.py translation_quality.py emotion_recognizer.py \
  emotion_worker.py light_tts_selector.py scheme3_policy.py tts_router.py tts_worker.py \
  tts_preflight.py tts_quality.py
python test_scheme3_policy.py
python test_scheme3_resume.py
python test_light_tts_selector.py
python test_translation_emotion_tts_issues.py
```

新增回归使用模拟API/worker，覆盖逐句补译、错误语言拦截、短候选、上下文、局部缓存、
eSpeak音色、LightTTS降级、情绪继承、模型预检、fallback、时长重选、resume与Step5门禁。
测试不调用真实API或五模型推理；上线前需独立工程中的真实短视频验收。
