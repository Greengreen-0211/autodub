# 改动说明 —— 轻量 TTS 时长预览后端替换为 Pocket TTS（按语言路由）

分支：`fix/backend-light-tts-pocket`（基于 `origin/backend` @ `f6072fa`）

## 背景

轻量 TTS 在 Step 3 里用来给每个翻译候选合成一段**预览**、量时长、选「时长最贴合时间窗」的那句，不是最终配音。原来只有 edge-tts / espeak 两个后端。

组长建议把 [Pocket TTS](https://github.com/kyutai-labs/pocket-tts)（Kyutai Labs 的 CPU 轻量神经 TTS，~100M）替换进来看效果。Pocket TTS 内置语言为英/法/德/葡/意/西/荷，**不支持中文**，所以需要按语言路由：内置语言走 Pocket TTS，中文等其余语言仍走 edge-tts/espeak。

## 涉及文件

| 文件 | 改动 |
|---|---|
| `light_tts_selector.py` | 新增 `PocketTTSBackend` + `POCKET_LANGUAGES` + `_routes_to_pocket`；`LightTTSSelector` 增加按语言路由 `_backend_for`；`select()` 逐句按语言选后端 |
| `light_tts_requirements.txt` | 加入 `pocket-tts` |

## 改动详情

### 1. `PocketTTSBackend`

- 实现 `LightTTSBackend` 协议：`name="pocket-tts"`、`extension=".wav"`、`synthesize(text, language, output_path)`。
- 按语言惰性加载 `TTSModel.load_model(language=...)` 并缓存（`_models`/`_voices`），默认音色 `alba`（可用 `AUTODUB_POCKET_TTS_VOICE` 覆盖）。
- **语言名映射**：路由用短码（`en/fr/de/...`），但 pocket-tts 的 `load_model` 需要全名（`english/french/german/portuguese/italian/spanish/dutch`），由 `POCKET_LANGUAGE_NAMES` 转换。实测传 `"en"` 会报 `Config file not found`，必须用全名。
- `generate_audio(voice, text)` 返回 24kHz PCM 张量，转成 16-bit WAV 落盘。
- `preflight(language)` 做一次最小合成自检；`pocket-tts` 未安装或自检失败时抛 `RuntimeError`。
- **内网/镜像**：模型权重默认从 huggingface.co 下载。本机 HF 直连不通、`hf-mirror.com` 可达，可用 `AUTODUB_POCKET_HF_ENDPOINT=https://hf-mirror.com` 走镜像（代码在 import `pocket_tts` 前设置 `HF_ENDPOINT`；已显式设过 `HF_ENDPOINT` 则不覆盖）。

### 2. 按语言路由（`LightTTSSelector._backend_for`）

- 显式传入 `backend=`（测试/回退）→ 直接用，不路由。
- `AUTODUB_LIGHT_TTS_BACKEND` 非 `auto` → 强制该后端，不路由。
- `auto`（默认）→ 语言在 `POCKET_LANGUAGES`（en/fr/de/pt/it/es/nl）走 Pocket TTS，其余（zh/ja/ko/ru…）走 edge-tts/espeak。
- Pocket TTS 加载/自检失败 → 打印警告、本次运行内禁用 Pocket，回退 edge-tts/espeak（不影响中文片段）。
- `select()` 里逐句用 `_backend_for(language)` 取后端，`extension`/`signature`/`name`/`synthesize` 全部随之后端走；`light_tts_selection.backend` 现在如实记录每句实际用的后端。

### 3. `create_light_tts_backend`

- 新增 `pocket-tts`/`pocket` 分支（供 `AUTODUB_LIGHT_TTS_BACKEND=pocket-tts` 强制使用）。

## 关于「前端是否隐藏轻量 TTS 合成效果」

后端现状：预览音频经 `backend_contract.py::serialize_candidate` 把 `preview_audio` 映射成 `audio_artifact_id` 传给前端，同时 `light_tts_selection.backend` 记录每句实际后端（pocket-tts/edge-tts/espeak）。

**结论：是否隐藏是前端侧的渲染决定，后端不需要额外改动**——前端只要不渲染 `audio_artifact_id`，或对 `light_tts_selection.backend != pocket-tts` 的片段打标/隐藏即可。本次未改 `backend_contract.py`，避免动前端契约。

## 新增/相关环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `AUTODUB_LIGHT_TTS_BACKEND` | `auto` | 强制指定轻量后端（pocket-tts/edge-tts/espeak/auto） |
| `AUTODUB_POCKET_TTS_VOICE` | `alba` | Pocket TTS 默认音色名 |
| `AUTODUB_POCKET_HF_ENDPOINT` | 空 | 可选 HF 镜像，如 `https://hf-mirror.com`（仅在未设 `HF_ENDPOINT` 时生效） |

## 安装 Pocket TTS

```bash
pip install pocket-tts   # Linux CPU 可加 --extra-index-url https://download.pytorch.org/whl/cpu
```

> 未安装时不影响主流程：英文等片段会自动回退到 edge-tts/espeak（带警告日志）。

### 本机实测安装记录（2026-11-18）

- 环境：`autodub_server`（`/home/goldenseed/.conda/envs/autodub_server`，Python 3.14.4 / torch 2.9.1+cu128 / numpy 2.4.2）
- `pip install --dry-run` 确认**只新增 pocket-tts-4.0.1，不升级/降级任何现有包**；随后实际安装成功。
- 安装前 pip 快照：`/data0/goldenseed/cjx/backups/pip_freeze_autodub_server_20261118_164739.txt`
- 首次加载模型需联网下载权重；本机直连 huggingface.co 不通，用 `AUTODUB_POCKET_HF_ENDPOINT=https://hf-mirror.com` 成功下载并缓存（英文模型）。

## 测试

- 离线路由自测：`_routes_to_pocket`、pocket 语言门、`create_light_tts_backend("pocket-tts")`、显式后端绕过、auto 路由、强制后端名——**全过**。
- `python3 test_light_tts_selector.py`：**PASS**（显式后端路径不回归）。
- `python3 test_asr_recovery.py`：**Ran 66 tests — OK**。
- **真实合成验证（装包后）**：
  - `PocketTTSBackend.preflight("en")` → `Pocket TTS preflight: lang=en voice=alba PASS`
  - `synthesize(...)` 英文 → 2.48s WAV，24kHz，正常
  - 中文 → 被语言门正确拒绝（`Pocket TTS 不支持语言 'zh'`）
  - `LightTTSSelector(auto).select()` 真实英文片段 → `backend=pocket-tts`、2.96s/3.0s、`fit=suitable`，二次运行命中缓存
  - 中文片段 → 走 espeak（本机 `edge-tts` 未安装，属既有状态），合成 2.72s 正常

> 注：本机 `autodub_server` 环境里 `edge-tts` 未安装，中文实际由 espeak 承担（`create_light_tts_backend("auto")` 自动降级）。这是改动前就存在的状态，本次未改动。

## 备份

改前原文件备份于：

```
/data0/goldenseed/cjx/backups/light_tts_selector.py.20261116_184944.bak
/data0/goldenseed/cjx/backups/light_tts_requirements.txt.20261116_184944.bak
```

## 提交

`light_tts_selector.py`、`light_tts_requirements.txt`、`改动说明.md` 提交并推送至 `fix/backend-light-tts-pocket`。

> 仓库根目录下另有未跟踪文件 `run_phase_a.sh`（非本次改动、非我创建），**未纳入提交**。
