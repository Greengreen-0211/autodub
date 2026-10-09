# 第10阶段模拟前端验收记录

日期：2026-10-09  
范围：Windows 本机 mock UI / 合同行为检查；不包括真实模型、真实媒体、HTTP 服务或服务器部署。

## 已验证

### 轻量行为检查

`frontend/tests/test_workflow.py`：9 项全部通过，最后一次输出 `Ran 9 tests ... OK`。

| 场景 | 验证要点 |
|---|---|
| TTS 返回翻译闭环 | 旧音频/成品占位 stale；编辑后新 revision/attempt；TTS 文案等于人工锁定译文 |
| 源文覆盖 | raw ASR/OCR 不变；逐词对齐 stale；模拟后端扩展三句上下文依赖 |
| 并发与幂等 | 旧 revision 409、运行中 JOB_BUSY、相同 request_id 重复复用、不同正文冲突、空文本拒绝 |
| 部分失败 | 失败句有原因；阻止混音；局部 TTS 重做保留其他成功句 |
| 多句连续编辑 | 累计依赖，不丢失第一句修改；确认后旧检查点不继续显示待确认 |
| 阶段暂停与恢复 | 实际阶段完成后暂停；无总数 progress=null；同 DB 重建 client 恢复状态 |
| 任务隔离与事件分页 | 同名视频不同 job_id；事件顺序/分页完整；第三句复杂句优先路由 |
| 候选和前置条件 | 过期候选拒绝；人工选择锁定；上游过期时禁止仅重做 TTS；逐句 ASR 不支持 |
| UI 控制器和安全呈现 | 轮询不升级编辑器基准版本；旧版本保存冲突；HTML 按数据转义；mock 媒体 URL=null |

`frontend/smoke_demo.py`：完整 OCR 源文修改 → 路由 → TTS → 返回翻译 → 锁定译文 → 重做 TTS → 混音模拟通过，最终 revision=4、TTS 文案 `我听不见你。`，原始 ASR 保留。

生成合同实例：`frontend/examples/mock_contract.json`，含检查点 Job、编辑响应、成品 Job、Segment、EventPage、Operation。只作为 v0.1 评审数据。

### Gradio 与浏览器

- 独立 `.venv` 的 Gradio 6.1.0 启动成功；本机地址 `http://127.0.0.1:7860`；`share=False`。`pip check` 输出 `No broken requirements found.`。
- 浏览器验收任务 `job_2c5938db9af5`：r1 创建；OCR 选表格第二句、修正源文后 r2；返回翻译后 r3；保存锁定译文后 r4；正式 TTS 和最终模拟混音完成。
- 浏览器直接观察到源文 `I cannot hear you.` 被后续模拟翻译使用；原始 ASR `I can here you.` / OCR `I can hear you.` 保留。
- 输入译文 `我听不见你。` 后点击刷新状态，输入内容保持。保存后 human_pinned=True；正式 TTS 详情的读取文案与输入完全相同。
- 返回翻译时当前高亮为 translation；旧 light_tts/tts 占位明确显示过期；完成后新占位 r4，与旧占位区分。
- 服务重新启动、浏览器重新载入后，自动恢复 r4 完成状态；重新选第二句可看到人工源文、锁定译文和 3 条历史。
- 在浏览器可观测的 1046 px / 486 px 视口检查了 5 列 / 2 列阶段布局，文档宽度分别 1034 / 474 px，未发生文档横向溢出。10 列宽屏分支由 CSS 定义；当前侧栏预览以窄屏布局为主。
- 截图：`frontend/examples/preview_workbench.png`；服务本机日志：`frontend/.data/local-ui.log`。当前预览保留在 Codex 浏览器标签页。

### 快照与环境

- `SOURCE_MANIFEST_20261008.json` 的 40 项源码文件大小和 SHA-256 全部匹配。
- 最新快照没有代码编辑；模拟路由编译读取纯策略文件，避免在快照目录产生 `__pycache__`。
- 没有加载模型、处理视频、安装 GPU 环境、下载权重、运行服务器命令或修改 Stage8/9 冻结项。
- 前端实际 Python/依赖版本记录于 `frontend/requirements.lock.txt`。这是本机验收环境；服务器环境尚未核实。

## 待验证：与后端评审

1. 确认 v0.1 和 mock JSON 的字段，提供真实 capabilities、Job/Segment/Event/Operation/错误响应。
2. 确认业务 revision 与事件 sequence 的区别、幂等期限、409 和事件历史 410 的快照恢复。
3. 人工源文/译文覆盖、human_pinned、实际依赖图和参考音频依赖由后端落地；mock 的三句扩展依赖不是对真实服务的断言。
4. 定义真实子阶段完成事件、TTS 计划与实际引擎、失败/fallback 原因、时长来源、实际门禁结果。
5. 任务隔离、逐句重做、部分成功门禁、原子状态写入、安全暂停和持久化 worker。
6. 真实媒体的 URL、MIME/Range、版本和 stale 规则，接口地址/认证/文件限制；届时补 HTTP 客户端与实际播放器绑定。

本次没有把 Gradio 自动组件端点升级为业务服务，没有宣称完成真实流水线端到端运行。拟议服务器阶段目录仍未验证部署。

## 同日展示调整：agent 式执行动态

- 按用户截图反馈，将原大阶段卡片改为只显示编号/名称的小方块；阶段状态用灰/绿/蓝/橙/红区分。悬停或点击可查看状态、模块/模型、尝试次数、开始/结束时间；普通无变化轮询保留展开状态。
- 方块前增加模块执行动态：当前模块/任务、已使用模块、计划模型选择、fallback、人工保存和返工记录。点击记录显示原始事件及结构化依据；模拟标记保留。
- `stage.started/completed/failed` mock 事件补充 model/attempt/progress，不新增或假装真实服务能力。阶段完成和下一阶段启动之间显示“正在准备下一阶段”，避免重复宣称已完成模块仍在运行。
- 修改后原 9 项闭环检查全部通过，smoke 演示通过并刷新合同样例。浏览器实测小方块点击详情、刷新不收起、原任务恢复、已使用 Demucs/Qwen3-ASR/HunyuanOCR 模块的动态记录。
- 新 UI 示例任务：`job_bffd88b4e649`，保留原先 r4 人工编辑验收任务，未覆盖已有任务。

## 同日实测反馈修复：轮询闪烁与宽屏布局

- 在已安装 Gradio 6.1.0 前端资源中确认 HTML 的 `.pending` 样式为 `opacity:.2`，其他 pending 样式还包含 flash 动画；即使回调 show_progress=hidden，轮询期间 HTML 容器仍绑定 pending 类。此前自定义 running 呼吸/旋转也是视觉变化来源。
- 删除自定义动画，作用域 CSS 取消 pending 变淡/flash 与 generating 动画。所有 18 个视图输出用值比较过滤；无变化轮询只更新内部 State，不重复提交表格、下拉框、详情、日志、按钮等。
- 新增空闲轮询回归：第二次相同快照不输出任何重复视图字段；更换当前句只更新相应详情。共 10 项行为检查全部通过。
- 浏览器手动刷新和模拟执行时检查到所有已显示 HTML 容器 opacity=1，当前阶段及执行图标 animationName=none；未保存的测试译文在刷新后保留，随后显式载入清除测试草稿，未保存到任务。
- 宽屏重构为居中、最大 CSS 宽度 1240px 的审阅工作区。左侧原视频/成品与四列句子导航，右侧有效源文、译文、保存和试听返工；新建/恢复任务、候选、模型依据按需展开，日志进入“过程记录”页签。动态只显示当前状态和两条最近记录。
- 可观测 1920px 视口文档宽度 1905px；恢复默认 640px 视口文档宽度 625px，左右工作区上下回流，无文档横向溢出。
- 宽屏预览：`frontend/examples/preview_workspace_desktop.png`。旧浏览器需刷新以加载新版脚本与样式。

这些是本机 Gradio 界面的修复，不依赖模拟任务速度。真实后端接入仍需服务器联调，沿用相同的不闪烁呈现规则。

## 原视觉恢复与版本备份

- 用户明确要求只改布局，恢复 Soft/indigo 主题、白色阴影按钮、紫色标签、浅蓝任务卡与原字号留白；顶部执行动态重新展示六条近期记录。
- 本机 UI 重启成功；1920 宽屏 DOM document.scrollWidth=1905，无文档横向溢出。执行动态/状态 HTML opacity 均为 1，工作流容器透明，避免 Group 灰底改变原视觉。显式载入句子后显示原文和译文。
- 10 项行为检查通过。业务代码与字段差异更新不变；原快照/服务器模型不受影响。
- 恢复版截图：frontend/examples/preview_restored_style.png。旧截图继续保留，不将恢复版称为旧代码精确回滚。
- 每次修改前备份要求写入 frontend/AGENTS.md；backup_frontend.py 生成不可覆盖的时间戳 ZIP，附 SHA-256 并检查 ZIP/文件内容，当前基线已备份。以前没有完整代码备份的事实已记录。

## 深色模式颜色验证

- 修改前备份 before-dark-colors；仅追加 dark CSS，原浅色规则逐行一致，布局/业务 Python 文件字节一致。
- 浏览器实际切换 Dark/Light：dark 主标题 rgb(229,231,235)，卡片 rgb(23,28,39)，阶段完成文字 rgb(184,223,198)/底 rgb(29,48,40)，待确认文字 rgb(238,215,153)/底 rgb(48,43,30)。模型详情和事件日志背景为深色，正文为浅色。
- 切回 Light：卡片 rgb(255,255,255)、任务卡 rgb(241,245,252)、完成状态底 rgb(240,248,243)、主标题 rgb(38,49,64)，与确认版配色一致。未改变布局尺寸或防闪烁规则。
- 修复版截图 frontend/examples/preview_dark_colors.png；服务正常重启。本次为颜色修复，验证使用浏览器检查与基线对照，未重复运行未改变的后端检查。

## 当前执行行局部动画

- 用户明确授权旋转圆圈/扫光，与此前禁止整块内容反复变淡区分。修改前已有 before-activity-motion 备份。
- 浏览器排队/执行 spinner animationName=ad-activity-spin、duration=0.9s，transform 两次采样不同；扫光 animationName=ad-activity-sweep、duration=3.2s，位移两次不同。待人工确认扫光为 6s，连续两次位移改变。
- 活动 HTML opacity=1，阶段方块 animationName=none；已创建、完成、报错、暂停无 motion 标记；系统减少动态效果偏好下关闭动画。
- 10 项行为检查通过；本机服务重启正常，无真实模型或服务器操作。预览 frontend/examples/preview_activity_motion.png。

## 审阅状态 / TTS / 详情交互

- 15 项行为检查通过，新增等待不失效、相同保存不失效、保存保留检查点/未运行依赖、旧 mock 兼容及媒体 ID/候选/过期清除；smoke_demo 完整闭环通过，重新导出 mock_contract.json。
- 浏览器：r1 OCR 检查点保存相同原文，r1 不变、下游 pending；实际修改后 r2 仍停 OCR，resume_from=diarization，无 active operation；点击确认才 queued 说话人。
- 阶段块不再使用 details/open；鼠标指向块 tooltip display=block，移开 display=none，点击未锁定。执行记录仍可展开。
- TTS native Soft 组件与六个媒体输出接线，OCR 审计链接输出接线；实际媒体播放仍待真实后端。当前 mock 的所有媒体值为 None，不生成假音频。
- 新版核对清单 FRONTEND_AUDIT_20261009.md。截图 frontend/examples/preview_review_pending.png 与 frontend/examples/preview_tts_review.png。

## 按阶段开放审阅

- 17 项标准库行为检查通过，新增阶段权限、未保存草稿拦截和外部版本冲突校验。
- 浏览器实际验证：TTS 检查点源文/译文折叠只读；展开源文的输入框为 disabled。返回翻译后路由检查点自动载入并开放译文。草稿点击确认返回 UNSAVED_CHANGES 且保留草稿。
- 保存译文后 r4 仍停路由审阅，确认才处理轻量 TTS/路由；再次确认进入 TTS。实际合成文案显示已保存的“最后还有一次机会。”，源文/译文再次折叠只读。
- 固定子组件与 CSS 显隐避免 Gradio 布局更新丢失内容；没有更换已确认布局和浅色/dark 配色。深色 TTS 预览已实际保存为 frontend/examples/preview_tts_review.png。
- 本轮使用独立演示任务 job_1879db997750；仅运行 mock，没有真实媒体或模型处理。

## 路由、情绪和 TTS 模型比较扩展

- 修改前备份 before-routing-emotion-tts-options，历史 progressive-review 备份保留。
- 21 项标准库检查通过：新增规则优先级与理由、人工情绪保留原数据并传播、试合成/采用分离、失败/旧依赖/跨句/未知模型/忙碌/幂等门禁。smoke_demo 与 smoke_extensions 均通过；v0.2 固定样例已实际导出。
- 原 source 清单 40 项 SHA-256 全一致。没有改 Stage8/9 或原 ZIP 快照，没有连接 GPU/服务器或调用真实 TTS。
- 浏览器独立新任务 job_6118422a75ef：路由实际显示 33 个汉字、阈值 28、3 个分句/阈值 3；人工 happy 后显示 CosyVoice3 与人工情绪理由。情绪保存没有覆盖已输入译文，确认仍拦截草稿。
- 同任务正式 TTS CosyVoice3 与试合成 F5-TTS 分开。试合成完成前后原结果保留，明确采用后才变为 F5-TTS、来源为人工采用，自动 CosyVoice3 依据保留；完成后确认重新混音入口保留。
- 下拉值保持修正与显式选择门禁：当前句 ID 持续显示，选择模型后试合成按钮开放，选择完成记录后采用开放；已采用的同一结果禁止重复点击。浅色/dark/宽屏/窄屏样式沿用。
- 真实音频试听与质量门禁仍待后端；mock 时长、RTF 等均 null，播放器无假音频。接口补充单独立 v0.2 文件，没有覆盖 v0.1 草案。
- 多个试合成之间可反复采用：正式引用切换不使未改变依赖的比较录音失效，检查覆盖 CosyVoice3→Confucius4→CosyVoice3。新增试合成完成/采用事件关联对应 operation_id。
- 实际预览：frontend/examples/preview_routing_reasons.png、preview_tts_models.png、preview_tts_models_narrow.png。最终浏览器停在 r6 TTS 审阅，正式 F5-TTS 保留，新 Confucius4 试合成可供选择比较；没有自动采用。

## 完成态只读（后续用户修正）

- 22 项检查通过。新增完成后 source/translation/emotion/tts 权限全 false，无当前阶段高亮；旧页面调用编辑/重做/返回翻译/试合成/采用均在工作台返回 JOB_COMPLETED，Job 快照无变化。
- 浏览器独立样例 job_6118422a75ef r6 完成混音。配音区自动收起、标题变“本句配音结果（只读）”；展开后 redo/back/try/adopt 四按钮均 visible=false。顶部只剩刷新，阶段全完成、无当前标记。
- 保留结果、试听、原声和历史比较，不开放修改。完成时反馈和说明改为只读，隐藏保存/继续提示。部分失败/TTS 待确认权限保留。
- 本次是工作台交互策略修正，不改原 v0.1 通用服务合同。前后均保存代码备份；最新预览 frontend/examples/preview_completed_readonly.png。


## 逐阶段审核（取代早期合并审核）

- 26 项行为检查通过，五个安全暂停点为 OCR / emotion / light_tts / routing / tts。每次仅一个活动权限；60 次等待 tick 状态不变；未来阶段 attempt 为 0。自动模式直接完成。
- 两个模拟 smoke 完整通过，人工文本、情绪与采用模型仍传到最终结果。单句情绪修改不会漏掉其他尚未翻译句子。跨阶段保存与返工由 Workbench 新鲜快照拦截。
- 浏览器样例 job_9d2b283f1558 r2：OCR、情绪、译文按序单独开放；保存译文后仍停译文审阅，重做轻量结果后再次确认才路由。路由时旧情绪/译文只读，TTS 隐藏。预览 frontend/examples/preview_sequential_routing.png。
- 冻结 source 40 项哈希无变化；原 v0.1 保留，新合同补充单独为 v0.3。真实暂停、媒体与模型仍待服务器后端联调。

- 浏览器继续验证 TTS 单独开放及返回情绪：r3 只开放 emotion，translation/routing/tts 均不可见。实际预览 frontend/examples/preview_sequential_emotion.png。

