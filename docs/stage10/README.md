# Stage 10 前端交付与联调

当前为明确标记的模拟工作台：逐阶段人工审核、文本/情绪修改、TTS 独立试合成与明确采用、返工及完成态只读。真实 HTTP BackendClient、音频/视频和服务器模型尚未接入。

仓库根目录保留原有后端文件，新增 `frontend/` 与 `docs/stage10/`。运行方式见 `frontend/README.md`，Python 3.12，仅安装 `frontend/requirements.txt`。不要求本机安装模型环境。

人工模式顺序：OCR → 情绪 → 译文/轻量试听 → 路由 → 正式配音 → 混音。v0.3 替代 v0.1/v0.2 的早期合并审核安排；源文与人工译文锁定、稳定句子 ID、revision 和依赖失效规则仍有效。

后端评审顺序：v0.1 基础合同 → v0.2 情绪/试合成/采用 → v0.3 独立审核和情绪返工 → mock JSON 样例。先确认实际能力、暂停点、版本校验、依赖失效与媒体 URL，再实现 HTTP 任务服务及流水线适配。UI 不直接修改 project_state.json。

验证命令（在仓库根目录）：

```text
python -m unittest discover -s frontend/tests -v
python frontend/smoke_demo.py
python frontend/smoke_extensions.py
```

模拟服务读取现有根目录 scheme3_policy.py，SHA-256 必须与交付清单一致；不复制或替换仓库规则。Mock 通过不代表真实模型/音频验收。
