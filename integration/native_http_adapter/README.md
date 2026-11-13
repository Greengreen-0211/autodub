# 现有Native HTTP服务的适配文件

这些文件是当前前端工程服务的配套适配，不是另一套模型。backend分支没有整个前端工程，不会通过拉取本分支自动启动网页服务。

1. native_audio.py把原视频及SHA/时长加入内部plan，校验后写入说话人状态；AV/legacy桥接结果均经同一主Step2幂等检查；实际路由和原因随metadata返回。
2. native_trials.py认可严格审阅清单已验证的分支源码，保留原检查作为未安装分支时的默认。
3. av_routing_binding.py检查完整运行源码、路径、SHA及feature实际commit；旧版本ASR不静默重新绑定，部署后创建新任务。
4. start_video_workflow.py保留现有页面配置，启动记录读取已安装feature commit并显式传递auto/legacy模式。

部署清单必须绑定实际 feature 提交，不能用原 backend 提交冒充。先将本次代码提交到指定分支，再在该 Git 克隆中执行下列命令；脚本逐文件核对当前源码与实际 HEAD。提交可以先留在本地，但交付/运行记录必须明确是否已经推送。BASE.json/ADAPTER_BASE.json仅供原文件识别，不是最终部署清单。

```bash
cd /data0/goldenseed/lqz77/integration/autodub_3dspeaker_feature || exit 1
/home/goldenseed/.conda/envs/autodub_server/bin/python -B \
integration/native_http_adapter/prepare_manifest.py \
--output /data0/goldenseed/lqz77/integration/autodub_3dspeaker_routing_20261113/backend/AV_BRANCH_SOURCE_MANIFEST.json
```

在已停下工作台/API后，对当前工程目录先预检，再应用：

```bash
cd /data0/goldenseed/lqz77/integration/autodub_3dspeaker_routing_20261113/backend || exit 1
/home/goldenseed/.conda/envs/autodub_server/bin/python -B \
integration/native_http_adapter/install.py --root /data0/goldenseed/lqz77/integration/autodub

# 预检通过后应用。自动备份受影响的源码；模型、环境及任务库不在更新范围。
/home/goldenseed/.conda/envs/autodub_server/bin/python -B \
integration/native_http_adapter/install.py --root /data0/goldenseed/lqz77/integration/autodub --apply
```

安装器遇到未审阅的已有源码修改会停止并指出路径；不会强行覆盖。工作台或API端口仍在监听时不应用。完成后按原完整启动方式重启，创建新视频任务；OCR服务沿用原管理方式。

这一适配针对现有canonical运行目录；若自行换目录，还须同步检查旧工具中写死的ROOT，不要把另一目录的runtime误当成本分支。安装/回退应使用打印的源码备份和已有精确进程记录，禁止按python进程名批量杀进程。当前尚无新分支的服务器部署回执。
