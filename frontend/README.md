# AutoDub 第10阶段模拟工作台

当前交付：M1/M2 的 Gradio 模拟前端与可执行 mock 任务层。接口 v0.1 仍为团队评审草案，未实现真实 HTTP 服务。页面所有执行、路由、失败和产物均有模拟标记；真实视频、GPU 模型和生成媒体在服务器运行。

## GitHub 分支运行

在 `frontend` 分支根目录新增 `frontend/`，不移动原有后端文件。mock 优先读取本阶段冻结 source 中的 `scheme3_policy.py`；仓库布局则读取上一级已有的同名文件，并检查其 SHA-256，版本不匹配时停止。2026-10-09 核对分支提交 `635d01b77f1d3f8cfdca520a3a4e3625206274b9` 的规则文件与冻结快照完全一致，无需再上传一份。

下载/克隆仓库后，在仓库根目录执行（Python 3.12）：

规则版本校验允许 Git 在 Windows 上进行 LF/CRLF 换行转换；不允许内容或路由规则静默改变。

```powershell
python -m venv frontend/.venv
& 'frontend/.venv/Scripts/python.exe' -m pip install -r frontend/requirements.txt
& 'frontend/.venv/Scripts/python.exe' frontend/app.py --host 127.0.0.1 --port 7860
```

Linux：

```bash
python3.12 -m venv frontend/.venv
frontend/.venv/bin/python -m pip install -r frontend/requirements.txt
frontend/.venv/bin/python frontend/app.py --host 127.0.0.1 --port 7860
```

mock UI 仅依赖 Gradio，勿使用后端模型环境的 requirements 替代。接口和联调说明放在 `docs/stage10/`。本机数据库、环境、缓存及 ZIP 备份均不提交。

展示调整：阶段方块前提供 agent 式执行动态，显示模块调用、路由选择和 fallback。方块只显示编号和阶段名称；未开始灰色、完成绿色、运行蓝色、错误红色、待确认黄色、旧结果待更新橙色。方块详情仅悬停显示，移开收起，点击不会固定展开；键盘聚焦也可阅读。执行记录仍可点击查看事件依据，查看不会改变执行阶段。

最新体验调整：保留原 Soft/indigo 视觉与已确认布局，移除 Gradio 轮询 pending 的变淡效果，所有视图字段按变化更新。当前执行行有局部旋转/扫光，阶段块不闪烁。宽屏居中，左侧视频与四列句子列表，右侧源文/译文与本句配音返工；设置、日志和模型详情按需展开。执行动态显示最近六条记录。窄屏上下排列，旧页面需刷新后加载新版。

## 本机启动

本次已在独立 `.venv` 中安装并验证 Gradio 6.1.0。Windows PowerShell：

```powershell
Set-Location -LiteralPath 'D:\Documents\PyCharmProjects\autodub\project_log\第10阶段\可视化交互与前后端接口\frontend'
& '.venv\Scripts\python.exe' app.py --host 127.0.0.1 --port 7860
```

打开 <http://127.0.0.1:7860>。`--port` 可更换端口；`--db` 可指定独立模拟数据库。默认只监听本机，`share=False`。

若重新创建前端环境，使用 Python 3.12，仅安装此目录的 UI 依赖：

```powershell
Set-Location -LiteralPath 'D:\Documents\PyCharmProjects\autodub\project_log\第10阶段\可视化交互与前后端接口\frontend'
New-Item -ItemType Directory -Force -Path '.tmp' | Out-Null
$env:TEMP = Join-Path (Get-Location) '.tmp'
$env:TMP = $env:TEMP
python -m venv .venv
& '.venv\Scripts\python.exe' -m pip install -r requirements.txt
& '.venv\Scripts\python.exe' app.py --host 127.0.0.1 --port 7860
```

`requirements.lock.txt` 记录本次 Windows / Python 3.12 的实际安装版本；不是服务器环境验收结果。不要安装项目根的模型环境清单，也不要将 Windows `.venv` 上传到 Linux。

## 演示闭环

1. 展开“新建 / 恢复任务”，创建新模拟任务，选择中文与人工确认，点击上方开始。不上传视频也可演示；上传视频只作源视频预览，模拟执行始终使用内置三句样例。
2. OCR 检查点，点选 `seg_0002`。原始 ASR 为 `I can here you.`，OCR 为 `I can hear you.`。输入有效源文 `I cannot hear you.`，保存。原始字段保留，逐词对齐变为 stale；仍停在 OCR，后续未生成的步骤保持灰色。模拟后端返回三句上下文/参考依赖处理计划；仅已有旧结果置 stale。相同源文不更新版本、不使结果失效。
3. 确认后先停在情绪审阅，核对或修正后再确认，才执行翻译和轻量试听。此时只开放译文与候选审阅。手动源文的模拟翻译显示 `【模拟译文 · zh】I cannot hear you.`，不宣称真实翻译。
4. 确认译文后停在独立路由审阅，核对计划模型和依据，再确认才执行正式 TTS。TTS 完成后单独配音审阅：第一句计划 IndexTTS2、模拟 fallback 实际 F5-TTS；第三句长/复杂句优先 Confucius4。无真实音频，时长/RTF 不填假数。
5. 选中第二句，点击“从 TTS 回到翻译”。旧 TTS 标为待更新；等待译文/轻量试听检查点，输入 `我听不见你。`，保存并锁定。若要修正情绪，使用独立的“从 TTS 回到情绪”。
6. 保存译文后确认先重新轻量检查，再确认进入路由审阅，再确认进入正式 TTS。`TTS 读取文案` 必须为手动译文。配音审核确认后混音，成品仍是无媒体占位。
7. 在成品合成前的 TTS 审阅步骤可以返回翻译；成品流程完成后工作台锁定，配音结果只读，保留查看/试听，不再提供编辑、返工、试合成或采用入口。

“保存”只改变有效文本，不自动执行。编辑后“确认并继续”消费已保存文案；“返回翻译”发起重新计算，保留人工译文锁定。多句连续保存会累计过期依赖，后端决定实际重算范围。

原声/分离/轻量 TTS/正式 TTS/成品播放器已有 artifact ID/URL 绑定。候选选择可切换轻量试听；stale、mock、失败音频不会送到播放器。OCR 审计链接和候选时长/误差也有对应位置。mock 无真实媒体，不验证服务器播放。逐项核对和接口评审补充见上一级 `FRONTEND_AUDIT_20261009.md`。

## 情绪修正与其他模型试合成（v0.2 模拟补充）

1. 路由检查点的执行动态直接显示每句模型及实际命中理由；长句/复杂句列出译文字符/词/分句数量与阈值，优先级沿用 Scheme3。原声时间段不冒充路由长度阈值，方案图中的 RTF 不作为本任务测量。
2. “情绪审阅”显示原识别、当前采用和来源。在独立 emotion 检查点可改七种规范情绪；原标签、置信度与可靠性保留。保存仍停原检查点，确认才重算。其他步骤情绪只读，TTS 时通过“从 TTS 回到情绪”明确返回。
3. 配音审阅的“其他模型试合成与比较”：选择支持目标语言的模型，点“用所选模型试合成”。原正式配音和既有成品保留，试合成独立排队/完成/失败，记录可以切换；播放器并排/上下按响应式布局显示原配音与选中试合成。
4. 比较后点“采用所选试合成，替换本句配音”，才替换正式结果并标记成品待更新；自动路由理由仍保留，实际模型标注人工采用。确认后重新混音。失败、未完成、过期、跨句或依赖改变的试合成不能采用。真实服务质量门禁待联调，mock 无音频与实测质量。
5. “仅重做当前句 TTS”仍按自动路由执行；采用试合成只选择本次结果，不永久改路由规则。运行中控件锁定，试合成期间阶段暂停禁用。情绪/译文不同字段保存不会覆盖另一字段的未保存草稿，确认会拦截未保存内容。

正式接口补充见上一级 `接口补充_v0.2_情绪与TTS试合成.md`（待后端评审）；不改 v0.1 原草案。可执行样例 `smoke_extensions.py` 导出 `examples/mock_review_extensions.json`。

完成态不再标为“当前步骤”：配音区默认收起并改名“本句配音结果（只读）”。展开仅用于看结果、试听和历史比较。工作台回调读取最新任务状态，旧标签页也不能绕过完成锁定。部分失败和 TTS 待确认仍保留对应修复操作。此为当前前端交互策略；通用后端 rework 合同能力保留，正式服务终态/重开任务语义待团队冻结。

## 其他验证入口

审阅顺序：OCR → 情绪 → 译文与轻量试听（light_tts）→ 路由 → 正式配音（tts）。每次只有一个当前板块；未来隐藏，过去折叠只读。返回翻译或情绪后再开放对应步骤。保存保留当前位置，确认才推进；未保存草稿或版本冲突禁止确认。播放器仍为模拟媒体槽位。完整策略见上一级 `接口补充_v0.3_逐阶段审核.md`；旧任务不会被静默回退，验证完整新顺序请新建人工确认任务。

- 下一次当前句 TTS 模拟失败：在检查点展开异常演示并点击，再进入 TTS。失败会显示原因并禁止混音；仅重做失败句，成功句保留。
- 当前阶段结束后暂停：登记请求，当前模拟阶段安全结束后停下，明确显示实际边界。
- 浏览器刷新：BrowserState 保存最近 job_id，SQLite 恢复任务、日志、版本和历史。未保存草稿不跨刷新保留。已有操作由独立 mock worker 推进，不依赖轮询；同一数据库重新启动单个进程可继续运行操作。
- 多标签版本冲突：编辑器保存使用载入句子时的任务 revision；409 时草稿不变，显示需要重新载入。进入新审阅检查点自动载入当前句，同一检查点轮询只更新可编辑状态，不覆盖输入值。载入/切换句子会替换编辑器内容。
- 能力边界：按钮读取 capabilities。逐句 ASR、时间修改、自然语言定向指令暂不提供。

## 代码边界

| 文件 | 用途 |
|---|---|
| `app.py` / `styles.css` | Gradio 布局、十阶段高亮、按钮、单组试听位置、响应式样式 |
| `backend_client.py` | v0.1 的八个 BackendClient 方法与公开错误 |
| `workbench.py` | UI 控制器、增量事件、稳定 ID、编辑版本和安全呈现 |
| `mock_backend.py` / `fixtures.py` | 独立模拟任务服务、SQLite 事务/幂等、检查点、依赖过期和 mock worker |
| `smoke_demo.py` | 无 Gradio / 无模型的完整闭环及合同样例导出 |
| `tests/test_workflow.py` | 闭环、冲突、暂停、恢复、依赖累计和失败门禁检查 |
| `examples/mock_contract.json` | 实际运行 smoke 导出的 Job/Segment/Event/Operation/EditResult 样例 |

模拟后端只读取最新快照中的无 ML 依赖 `scheme3_policy.py`，明确使用默认阈值；路由判断不在 UI 内重复实现。没有导入流水线、执行 shell/GPU 命令或写入 `project_state.json`。SQLite 数据仅位于此目录 `.data/`，原 ZIP 快照保留。

mock 目前只有 zh/en/ja 三种固定样例。模拟“翻译”和“音频结果”不是质量测试。模型依赖、媒体 URL/Range、真实校验、服务地址和认证需后端提供。后续实现 HTTP BackendClient 时还需接入实际媒体、事件过期 410 的恢复逻辑，并移除模拟专属控件；本次不宣称已完成真实对接。

## 检查

轻量闭环检查仅依赖 Python 标准库（`app.py` 的 UI 需 Gradio）：

```powershell
Set-Location -LiteralPath 'D:\Documents\PyCharmProjects\autodub\project_log\第10阶段\可视化交互与前后端接口\frontend'
& '.venv\Scripts\python.exe' -m unittest discover -s tests -v
& '.venv\Scripts\python.exe' smoke_demo.py
& '.venv\Scripts\python.exe' smoke_extensions.py
```

本次验证和后端评审问题见上一级 `VALIDATION_20261009.md`、`HANDOFF_20261009.md`。

## 修改前备份

每次 UI 修改前运行 `python backup_frontend.py --label before-layout-change`，保存不可覆盖的代码 ZIP 与现有截图；索引和恢复说明见 `backups/README.md`。用户仅要求布局调整时，保留现有主题、配色、字体和控件设计。此次恢复沿用原 Soft/indigo 样式，保留防闪烁与字段变化后才更新的修复。

深色模式自定义区域已有对应配色，沿用相同布局。深色预览见 `examples/preview_dark_colors.png`，浅色预览仍为 `examples/preview_restored_style.png`。

执行动态当前行带局部状态动画：排队/处理为旋转圆圈和柔和扫光，等待人工确认为较慢扫光；完成/失败/暂停/未启动静止。系统选择减少动态效果时关闭动画。动画不改变模型状态、进度或模拟标记，整块视图不会因为轮询反复变淡。
