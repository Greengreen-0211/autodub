# 改动说明 —— 短句用屏幕字幕纠正 ASR 识别错误

分支：`fix/backend-ocr-short-subtitle`
**基于 `fix/backend-ocr-overlay-filter`**（依赖其字幕区域估计，需先合并该分支）

## 问题

样例视频《人民的名义》片段中，原话「带他过来」被 ASR 识别成「丫头，过来。」，
译文随之变成 "Girl, come here"。屏幕字幕 OCR 明明读到了正确的「带他过来」，
但 OCR 纠错链路没有改动这句话。

## 根因（两层，都已用真实数据验证）

**第一层：候选根本没送给模型。**

`autodub_ocr_runtime/frozen/v12_c4_candidate_inventory_r4.py` 会先生成候选
`丫头，→ 带他`，但因为这是「单向锚点」且锚点只有「过来」2 个内容词，
低于 `MIN_ONE_SIDED_CONTENT_ANCHOR = 3`，被判为 `weak_one_sided_anchor`，
状态降级为 `diagnostic_only`。于是 `classifiable_candidate_count = 0`，
冻结 B 收到的结果是 `no_classifiable_candidates`，模型压根没被问到。

（对比同视频 id=10 的 `你这→你说`，锚点「房子不是你的」6 个字，就能通过。）

**第二层：即使送进去，模型也会拒绝。**

模拟把阈值临时改成 2 后真实调用了一次 API，模型返回：

```json
"difference_types": ["unresolved"],
"error_evidence": {
  "basis": "none",
  "asr_alternative": "unknown",
  "observation": "No unchanged context distinguishes 丫头 from 带他; OCR line is unrelated and alignment is one-sided."
},
"confidence": 0.0
```

原因很清楚：整句只有 4 个字，**不存在任何「未变化的上下文」**。而冻结 B 的
prompt（哈希冻结）明确规定「仅字幕不同不足以判定识别错误」。所以这是
**设计上的能力边界**，不是参数调整能解决的。

## 修复方案

在集成层（`ocr_v211_integration.py`，非冻结）增加一条**范围很窄、可审计**的规则
`apply_short_utterance_subtitle_overrides()`，用屏幕上唯一的整句字幕替换短句 ASR。
只有在**同时满足全部条件**时才生效：

1. 冻结 B 没有改动该句（绝不叠加第二次改写）；
2. 该句很短：内容词 ≤ 12；
3. 时间窗内**只有一条**源语言字幕（多条不同字幕 = 有歧义，放弃）；
4. 该字幕至少出现在 **2 帧**（排除闪过的瞬时文字）；
5. 字幕覆盖该片段时长的 **≥ 50%**；
6. 字幕文本落在**字幕版面区域**内（复用 overlay-filter 的
   `estimate_subtitle_region`，招牌/路牌/观看人数天然被排除）；
7. 字幕与 ASR 的内容长度比在 **0.5x ~ 2x**（避免用长句替换短句）。

命中后：
- 用 `update_segment()` 写入新文本（自动作废词级时间戳）；
- 调 `preserve_speaker_timing_skeleton()` 保留原始时序骨架给说话人边界检测；
- `ocr_optimized.reason = "short_utterance_subtitle_override"`；
- 写入 state 的 `asr_ocr_subtitle_override_report` 审计列表。

## 涉及文件

| 文件 | 改动 |
|---|---|
| `ocr_v211_integration.py` | 新增 `_content_token_count` / `_window_subtitle_lines` / `apply_short_utterance_subtitle_overrides`；`run_step_1b_optimized()` 在冻结 B 之后调用，并把字幕区域计算提前复用；state 新增 `asr_ocr_subtitle_override_report` |
| `test_asr_recovery.py` | 新增 `ShortSubtitleOverrideTests`（7 个用例） |

## 实测结果（样例视频）

Step 1b 输出：

```
ℹ️ 字幕区域估计: x=[291,708] y=[810,933] (样本 37)
✅ OCR 短句字幕覆盖: 3 句改用屏幕字幕文本。
```

| 片段 | 时间 | ASR | 覆盖后 | 帧数 | 覆盖率 |
|---|---|---|---|---|---|
| 8 | 31.47–32.12 | 丫头，过来。 | **带他过来** | 3 | 0.668 |
| 10 | 55.81–56.88 | 你这房子不是你的。 | 你说房子不是你的 | 3 | 0.667 |
| 14 | 62.69–63.45 | 这谁呀？ | 这是谁呀 | 3 | 0.666 |

目标句「带他过来」已修正，其余两句也都是字幕更准确的同义修正。

## 风险与边界

- **会信任字幕**：这条规则绕过 LLM 判定，直接采用屏幕字幕。中文剧集字幕通常是
  逐字听写稿，准确性高；但若某视频字幕是意译/改写，则可能覆盖掉本来正确的 ASR。
  为此加了「短句 + 唯一字幕 + 多帧稳定 + 覆盖过半 + 长度相当 + 位置在字幕带内」
  六重约束，且全部改动都有审计记录。
- **不处理长句**：长句仍有未变化上下文，冻结 B 能正常判断，不需要这条规则。
- **依赖 overlay-filter 分支**：没有字幕区域估计时本规则完全不生效（保守）。

## 测试

- `python3 test_asr_recovery.py`：**Ran 80 tests — OK**（原 73 + 新增 7）
- `python3 -m autodub_ocr_runtime.selftest`：**Ran 18 tests — OK**
- `python3 test_ocr_host_integration.py`：**Ran 4 tests — OK**

新增用例覆盖：正常覆盖、无区域不生效、冻结 B 已改则不叠加、多字幕歧义放弃、
单帧不稳定放弃、长句不动、字幕在区域外忽略。

## 备份

改前原文件备份于：

```
/data0/goldenseed/cjx/backups/ocr_v211_integration.py.20261119_203535.bak
```

## 提交

`ocr_v211_integration.py`、`test_asr_recovery.py`、`改动说明_ocr_short_subtitle.md`
提交并推送至 `fix/backend-ocr-short-subtitle`。

> 仓库根目录下另有未跟踪文件 `run_phase_a.sh`（非本次改动、非我创建），**未纳入提交**。
