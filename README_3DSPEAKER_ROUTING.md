# 3D-Speaker视听优先，原说话人方法回退

本分支替换说话人阶段的选择方式，其余OCR、情绪、翻译、TTS与混音实现保持backend原有规则。基于backend `f0577d9c7252bb22ae24494d5c9791d517aaba4e`，复用已在服务器试过的3D-Speaker `065629c313eaf1a01c65c640c46d77e61e9607b4`。三个完整配音案例已获用户主观反馈“人物归类清楚很多、整体效果不错”；这不是独立逐帧标注或DER结果，新路由分支本身仍需服务器联调。

## 采用和回退规则

| 实际结果 | 本流水线采用 |
|---|---|
| 有人脸特征、有可用视听声音原型、确有声学簇参与匹配，RTTM及输入校验通过 | 3D-Speaker视听C |
| 无人脸通过ASD/FQA筛选，如已试过的泰坦尼克号片段 | 原backend WeSpeaker |
| 有人脸特征但无可用锚点/匹配 | 原backend WeSpeaker |
| 视听环境缺失、资源校验失败、处理超时/失败、结果格式错误 | 记录不同原因，回退原方法 |
| 视频/人声音轨在执行期间被改动 | 停止发布；不把完整性错误伪装成弱证据 |
| C和B分组恰好一致 | 仍可采用C；没有标签变化不等于没有证据 |

**不自动采用3D-Speaker纯音频B作为降级结果。** B是联合模型的必要声学分支及诊断对照；最终路由只选择C或原backend方法。默认原方法为WeSpeaker；若已有配置明确指定SpeechBrain，可继续采用该原后端。视听锚点有效仅说明可用，不保证角色归属一定正确。

`auto_dubbing_ver_2.0.py`的Step2与独立`run_step2_wespeaker.py`都接入同一个路由器。原实现分别保留为legacy函数。相同输入的桥接调用和主入口不会重复运行模型；改文本会重新对齐并使下游过期，视频/音轨/模型清单/适配代码变化会改变视听指纹。不会把C写进WeSpeaker目录或把CAM++特征冒充SimAM证据。

## 配置

```bash
export AUTODUB_DIAR_ROUTING_MODE=auto
export AUTODUB_DIAR_BACKEND=wespeaker
export AUTODUB_3DSPEAKER_PYTHON=/home/goldenseed/.conda/envs/3dspeaker_av_lqz77_065629c3/bin/python
export AUTODUB_3DSPEAKER_SOURCE=/data0/goldenseed/lqz77/third_party/3D-Speaker_065629c313ea
export AUTODUB_3DSPEAKER_MODELS=/data0/goldenseed/models/3D-Speaker_AV_065629c313ea
```

默认即auto；显式设置有助于部署记录。`AUTODUB_DIAR_ROUTING_MODE=legacy`直接执行原方法，可用于回退验证；会记录明确的legacy选择，避免留下上次C的显示。`AUTODUB_DIAR_FALLBACK_BACKEND`可显式指定wespeaker/speechbrain。`AUTODUB_3DSPEAKER_TIMEOUT`默认1800秒。

只使用GPU0、CPU线程2，现有独立3D环境，模型离线。不安装环境、不下载权重、不改已部署第三方源码或ASD/FQA/联合聚类阈值。启动前校验已有84份源码、14个资源及GPU/磁盘条件；只有单个视听worker持有模型，退出后再执行原方法。主HTTP进程组取消也会结束该worker。音频任务或没有原视频上下文的调用明确回到原方法。

## 可查的实际决策

任务temp内`diarization_routing.json`及状态字段`diarization_routing`保留：

- `planned_backend`、`selected_backend`、`fallback_backend`、`fallback_attempted`、`fallback_used`、`audio_B_adopted=false`；
- `reason_code`及说明，例如`no_visual_features`、`no_usable_visual_anchors`、`av_timeout`；
- 实际特征/原型/匹配数、逐采样门控计数及FQA分布、worker日志路径；
- 输入指纹、缓存使用、C标签映射、无C覆盖片段ID、结果指纹。

源文、原ASR和已有人工锁保留；使用backend现有RTTM/字词对齐和分句。C未覆盖的对白显式使用`UNRESOLVED_句ID`的隔离参考，不伪造新增视觉人物。两条方法都失败时记录失败，不把“开始回退”当成“回退成功”。

## 现有网页服务

现有NativeRunner仅转发音频，且旧服务固定校验0b94dcd源码；只替换backend文件不足以在网页上启用C。对应兼容层位于`integration/native_http_adapter/`：转发并校验原视频、返回实际采用方法/原因、用已审阅分支的完整文件清单替代旧版本身份校验、启动记录显示实际feature提交。没有部署清单时旧验证仍然有效，不提供“跳过校验”开关。具体见该目录README。

## 验证边界

本地检查仅模型无关合同、入口/HTTP上下文、原聚类回归、语法及包完整性；真实模型和完整视频只在服务器执行。此前三段成片验证了C接入下游的可行性，没有实测本分支自动回退链，不能据单测宣称已服务器验收。模型/阈值/预训练版本更新时应升路由/协议版本，不能复用旧证据冒充新模型结果。
