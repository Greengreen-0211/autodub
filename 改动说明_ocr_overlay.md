# 改动说明 —— OCR 画面叠加文字混入配音的修复

分支：`fix/backend-ocr-overlay-filter`（基于 `origin/backend` @ `78ccc7e`）

## 背景

用样例视频（110s，买瓜片段）测试时发现 OCR 会把**画面上的非对白文字**当成字幕混进流程：

- OCR 读到了水果摊招牌 `草莓`、`特甜`、`香草莓`、`5元2斤`、`5元/斤`；
- 这些文字被当成"稳定出现两次以上的源语言字幕"，提升成 ASR 片段
  （`origin='ocr_deferred_source_subtitle'`），随后会被翻译并配音。

另外画面里还有直播观看人数（`6人正在看`、`9人正在看`）、路牌（`上安路干洗`、
`批发站`、`门诊部`）、车牌（`冀A·60`）等叠加文字，同样有被误采纳的风险。

## 根因

`asr_recovery._ocr_lines()` 解析 OCR 输出时**丢弃了包围盒坐标**，只返回文字。
于是下游所有判断只能基于文本内容：

- 文本层面这几类文字完全合法（是真汉字、长度合适、语言匹配）；
- 已有的 `persistent_ocr_overlay_texts` 只能识别跨多个片段长期存在的叠加文字，
  对「只在某个恢复区间出现几次」的招牌无能为力。

而样例视频里，真正的字幕**位置高度固定**：底部中间
（x≈272–727，y≈924–991），招牌在画面中部，观看人数虽在同一水平带但贴左边缘
（x≈112–158）。**位置是这里唯一干净可用的判据。**

## 修复思路

用「能与真实 ASR 文本对上的 OCR 行」反推该视频的字幕版面区域，再让恢复流程
只在该区域内采纳文字。

1. `asr_recovery._ocr_lines_with_bbox()`：保留每行文字及其包围盒，
   `_ocr_lines()` 退化为只取文本，保持既有调用不变；
2. `ocr_v211_integration.estimate_subtitle_region()`：遍历 OCR 帧，
   把与所在时间段 ASR 文本匹配（归一化后是子串，或相似度 ≥ 0.55）的行作为
   字幕样本，取左右/上下边界的 3%/97% 分位数，再加边距得到字幕带；
   样本少于 4 个时返回 `None`，调用方退回原行为（宁可不过滤也不误删正文）；
3. `ocr_source_events(..., subtitle_region=...)`：框中心落在字幕带外的
   OCR 行不参与事件聚类，并在决策里记录 `outside_subtitle_region` 计数；
4. `_recover_deferred_from_ocr()` 透传该区域，`run_step_1b_optimized()`
   计算并打印估计出的字幕区域。

## 涉及文件

| 文件 | 改动 |
|---|---|
| `asr_recovery.py` | 新增 `_bbox_from_coordinates` / `_ocr_lines_with_bbox` / `_inside_subtitle_region`；`_ocr_lines` 改为复用；`ocr_source_events` 新增 `subtitle_region` 参数与过滤；`_SPOTTING_RECORD` 捕获 `<quad>` 坐标 |
| `ocr_v211_integration.py` | 新增 `estimate_subtitle_region()`；`_recover_deferred_from_ocr()` 新增 `subtitle_region` 参数；`run_step_1b_optimized()` 计算并透传 |
| `test_asr_recovery.py` | 新增 `SubtitleRegionTests`（7 个用例） |

## 样例视频实测结果

同一份 Step 1a 检查点、同一份 OCR 帧缓存，只改代码：

| | 修复前 | 修复后 |
|---|---|---|
| Step 1 片段数 | 23 | **18** |
| OCR 恢复采纳 | 1/5 组（5 个招牌事件） | **0/5 组** |
| 假字幕 | `草莓`、`特甜`、`香草莓`、`5元2斤`、`5元/斤` | 无 |
| 3.8–9.48s 区域 | 生成 5 段假对白 | 记为 `unresolved_blocking`，**不生成伪文本** |

估计出的字幕区域：`x=[226,772] y=[904,1011]`（50 个样本）。

修复后 Step 1 片段全部为 `primary`（真实 ASR 对白）：

```
0   0.00-  1.73 有一个人前来买瓜。
1  11.90- 20.68 生意行吗？……哥们儿，这瓜多少钱一斤呢？两块钱一斤。
...
17 107.08-110.46 喂，华哥，杀人了！杀人了！
```

> 说明：3.8–9.48s 实际是摊位空镜，但声学检测到有人声活动，因此仍被记为
> 「未恢复的确认对白」并降级提示。这是保守行为——**不生成伪文本**，
> 也不会阻断流程（除非显式开启 `asr_require_complete_dialogue`）。

## 兼容性

- `_ocr_lines()` 行为不变（仍返回纯文本列表），既有调用方无需改动；
- `ocr_source_events()` 的 `subtitle_region` 是可选参数，默认 `None` 时行为与原来完全一致；
- 无法估计字幕区域（样本不足）时自动退回原逻辑；
- 不带坐标的 OCR 输出（纯文本）一律放行，不做位置过滤。

## 测试

- `python3 test_asr_recovery.py`：**Ran 73 tests — OK**（原 66 + 新增 7）
- `python3 -m autodub_ocr_runtime.selftest`：**Ran 18 tests — OK**
- `python3 test_ocr_host_integration.py`：**Ran 4 tests — OK**
- 样例视频端到端重跑 Step 1b：见上表

新增用例覆盖：坐标解析、逐行包围盒、区域判定（字幕保留 / 招牌剔除 / 观看人数剔除 /
无坐标放行）、恢复流程拒绝招牌、恢复流程保留真字幕、区域估计、样本不足返回 `None`。

## 备份

改前原文件备份于：

```
/data0/goldenseed/cjx/backups/asr_recovery.py.20261119_194351.bak
/data0/goldenseed/cjx/backups/ocr_v211_integration.py.20261119_194351.bak
```

## 提交

`asr_recovery.py`、`ocr_v211_integration.py`、`test_asr_recovery.py`、
`改动说明_ocr_overlay.md` 提交并推送至 `fix/backend-ocr-overlay-filter`。

> 仓库根目录下另有未跟踪文件 `run_phase_a.sh`（非本次改动、非我创建），**未纳入提交**。
