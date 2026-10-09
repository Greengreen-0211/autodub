"""AutoDub Stage 10 mock Gradio workbench. Run: python app.py"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("GRADIO_TEMP_DIR", str(ROOT / ".data/gradio"))

import gradio as gr

from backend_client import BackendError, EMOTION_CHOICES
from mock_backend import MockBackendClient
from workbench import (Workbench, activity_html, candidate_label, changed_outputs, details_html, edit_notice, empty_session, event_html, log_text,
                       media_html, ocr_evidence_html, playback_values, review_key, review_permissions, review_policy_html, review_section_header, segment_rows, stage_html, summary_html,
                       ENGINE_NAMES, available_tts_engines, emotion_html, trial_view, routing_html)


def build_app(client=None):
    client = client or MockBackendClient(ROOT / ".data/mock.sqlite3")
    bench = Workbench(client)
    with gr.Blocks(title="AutoDub · 模拟工作台") as demo:
        session = gr.State(empty_session())
        editor = gr.State(None)
        stored_job = gr.BrowserState("", storage_key="autodub_stage10_mock_job", secret="autodub-stage10-mock-v1")
        gr.HTML('<div class="brand"><div><h1>AutoDub 工作台</h1><p>逐句审阅 · 人工修正 · 返回翻译</p></div>'
                '<span class="mock-badge">模拟数据 · 无真实模型调用</span></div>')
        header = gr.HTML(summary_html(None), elem_id="job-header")
        with gr.Column(elem_id="workflow-panel"):
            activity = gr.HTML(activity_html(None, []), elem_id="activity-panel")
            stages = gr.HTML(stage_html(None), elem_id="stage-panel")
        with gr.Row(elem_id="job-actions"):
            run = gr.Button("开始模拟执行", variant="primary", interactive=False)
            resume = gr.Button("确认并继续", interactive=False)
            pause = gr.Button("当前阶段结束后暂停", interactive=False)
            refresh = gr.Button("刷新状态")
        with gr.Tabs(elem_id="workspace-tabs"):
            with gr.Tab("审阅与配音"):
                with gr.Row(equal_height=False, elem_id="review-layout"):
                    with gr.Column(scale=4, min_width=360, elem_id="review-sidebar"):
                        with gr.Accordion("新建 / 恢复任务", open=False, elem_id="task-settings"):
                            language = gr.Dropdown([("中文", "zh"), ("英语", "en"), ("日语", "ja")], value="zh", label="目标语言")
                            mode = gr.Radio([("人工确认", "review"), ("自动模拟", "auto")], value="review", label="运行模式")
                            create = gr.Button("创建新模拟任务", size="sm")
                            with gr.Accordion("恢复已有任务", open=False):
                                restore_id = gr.Textbox(label="任务 ID", placeholder="job_…")
                                restore = gr.Button("恢复此任务", size="sm")
                        with gr.Tabs(elem_id="video-tabs"):
                            with gr.Tab("原视频"):
                                video = gr.Video(label="源视频（可选）", sources=["upload"], autoplay=False, height=210)
                            with gr.Tab("成品"):
                                final_video = gr.Video(label="成品视频", interactive=False, autoplay=False, height=210)
                        gr.Markdown("### 句子列表")
                        table = gr.Dataframe(headers=["句子 ID", "时间(s)", "原文摘要", "状态"],
                                             datatype=["str"] * 4, value=[], interactive=False, type="array", wrap=True,
                                             column_widths=["78px", "52px", "146px", "54px"],
                                             max_height=300, label="点击一句开始审阅", elem_id="sentence-list")
                        with gr.Accordion("分离音频", open=False):
                            dialogue_audio = gr.Audio(label="分离人声", interactive=False, autoplay=False)
                            background_audio = gr.Audio(label="背景音", interactive=False, autoplay=False)
                    with gr.Column(scale=7, min_width=370, elem_id="sentence-editor"):
                        gr.Markdown("### 当前句编辑")
                        review_context = gr.HTML("运行至检查点后，对应审阅板块会自动展开。")
                        with gr.Row(elem_id="sentence-selection"):
                            picker = gr.Dropdown([], label="当前句 · 稳定 ID", interactive=True, scale=3)
                            load_sentence = gr.Button("载入 / 更新版本", scale=0, min_width=150)
                        editor_version = gr.Textbox(label="编辑器版本", value="尚未载入句子", interactive=False, show_label=False, elem_id="editor-version")
                        with gr.Column(elem_id="review-source"):
                            source_section = gr.HTML(review_section_header("source", "源文 / OCR 审阅", False))
                            with gr.Column(elem_classes="review-content"):
                                source = gr.Textbox(label="有效源文", lines=2, placeholder="输入确定的正确原文", interactive=False)
                                save_source = gr.Button("保存源文", interactive=False)
                                gr.Markdown("保存后确认继续；离开 OCR 检查点后，此板块只读。")
                                with gr.Accordion("原始识别与 OCR 依据", open=False):
                                    raw = gr.Textbox(label="原始 ASR（保留）", interactive=False, lines=2)
                                    ocr = gr.Textbox(label="OCR 修正文（保留）", interactive=False, lines=2)
                                    ocr_evidence = gr.HTML(ocr_evidence_html(None, None, client.get_artifact_url))
                        with gr.Column(elem_id="review-emotion"):
                            emotion_section = gr.HTML(review_section_header("emotion", "情绪审阅", False))
                            with gr.Column(elem_classes="review-content"):
                                emotion_evidence = gr.HTML(emotion_html(None))
                                with gr.Row():
                                    emotion = gr.Dropdown(EMOTION_CHOICES, value=None, label="当前采用情绪", interactive=False)
                                    save_emotion = gr.Button("保存情绪修正", interactive=False)
                                gr.Markdown("保留原识别。确认继续后情绪只读；需要再改时从 TTS 返回情绪审阅。")
                        with gr.Column(elem_id="review-translation"):
                            translation_section = gr.HTML(review_section_header("translation", "译文与候选审阅", False))
                            with gr.Column(elem_classes="review-content"):
                                translation = gr.Textbox(label="人工译文（保存后锁定）", lines=3, placeholder="确定的目标文案；不会被返工静默覆盖", interactive=False)
                                save_translation = gr.Button("保存译文（锁定并更新版本）", interactive=False)
                                with gr.Row():
                                    candidates = gr.Dropdown([], label="译文候选 · 试听时长 / 误差", interactive=False)
                                    pin_candidate = gr.Button("锁定当前候选", interactive=False)
                                gr.Markdown("确认继续后译文只读；需要修改时，从 TTS 返回翻译审阅。")
                                light_audio = gr.Audio(label="轻量 TTS 试听", interactive=False, autoplay=False)
                        with gr.Column(elem_id="review-routing"):
                            routing_section = gr.HTML(review_section_header("routing", "模型路由审阅", False))
                            with gr.Column(elem_classes="review-content"):
                                routing_detail = gr.HTML()
                                gr.Markdown("核对计划模型及路由依据。确认后才开始正式配音。")
                        with gr.Column(elem_id="audio-review"):
                            tts_section = gr.HTML(review_section_header("tts", "本句配音与返工", False))
                            with gr.Column(elem_classes="review-content"):
                                media = gr.HTML(media_html(None, None))
                                with gr.Row():
                                    tts_audio = gr.Audio(label="正式配音试听", interactive=False, autoplay=False)
                                with gr.Accordion("本句原声对照", open=False):
                                    source_audio = gr.Audio(label="本句原声", interactive=False, autoplay=False)
                                with gr.Row(elem_classes="editor-actions"):
                                    redo = gr.Button("仅重做当前句 TTS", interactive=False)
                                    back = gr.Button("从 TTS 回到翻译", interactive=False)
                                    back_emotion = gr.Button("从 TTS 回到情绪", interactive=False)
                                with gr.Column(elem_id="model-trials"):
                                    gr.Markdown("#### 其他模型试合成与比较")
                                    with gr.Row():
                                        trial_model = gr.Dropdown([], value=None, label="试合成模型", interactive=False, filterable=False)
                                        try_model = gr.Button("用所选模型试合成", interactive=False)
                                    trial_picker = gr.Dropdown([], value=None, label="本句试合成记录", interactive=False, filterable=False)
                                    trial_detail = gr.HTML(trial_view(None, None, None, client.get_artifact_url)[0])
                                    trial_audio = gr.Audio(label="所选试合成试听", interactive=False, autoplay=False)
                                    adopt_trial = gr.Button("采用所选试合成，替换本句配音", interactive=False)
                                    gr.Markdown("试合成保留原配音。采用后成品需重新合成；上方“仅重做当前句 TTS”仍按自动路由执行。模拟模式无真实音频。", elem_id="trial-help")
                        gr.Markdown("保存后点击上方 **确认并继续**，让后续步骤使用修改后的文案。", elem_id="editing-hint")
                        notice = gr.Textbox(label="操作反馈", value="新建任务，或从左侧选择一句进行审阅。", interactive=False, lines=2, elem_id="operation-notice")
                        with gr.Accordion("模型与版本详情", open=False):
                            detail = gr.HTML(details_html(None))
            with gr.Tab("过程记录"):
                events = gr.HTML(event_html([]))
                with gr.Accordion("详细事件日志", open=False):
                    logs = gr.Textbox(label="日志（模拟）", lines=14, interactive=False, elem_id="runtime-log")
                with gr.Accordion("模拟异常演示", open=False):
                    fail = gr.Button("下一次当前句 TTS 模拟失败", interactive=False, size="sm", scale=0)
                    gr.Markdown("用于检查失败提示与本句重做。")
        timer = gr.Timer(1.0)

        view_outputs = [session, header, stages, table, picker, events, logs, detail, media,
                        run, resume, pause, redo, back, save_source, save_translation, pin_candidate, fail, activity,
                        final_video, dialogue_audio, background_audio, source_audio, light_audio, tts_audio, ocr_evidence,
                        source_section, translation_section, tts_section, review_context, source, translation, candidates,
                        emotion_section, emotion_evidence, emotion, save_emotion,
                        trial_model, try_model, trial_picker, trial_detail, trial_audio, adopt_trial,
                        routing_section, routing_detail, back_emotion]
        editor_outputs = [editor, editor_version, raw, ocr]

        def view(state, loaded_segment=None, clear_editor=False, load_fields=None):
            job = state["job"]
            selected = next((s for s in state["segments"] if s["segment_id"] == state["selected_id"]), None)
            idle = bool(job and not job["active_operation"] and not job.get("mock_worker_error"))
            loaded = bool(selected and state["editor_loaded_id"] == selected["segment_id"])
            caps = job["capabilities"] if job else {}
            completed = bool(job and job["status"] == "completed")
            translated = bool(selected and selected["translation"]["status"] == "success")
            routed = bool(selected and selected["routing"].get("status") == "success")
            rework_stages = caps.get("rework_stages", [])
            permissions = review_permissions(job, selected)
            source_update = gr.update(interactive=bool(permissions["source"] and loaded))
            translation_update = gr.update(interactive=bool(permissions["translation"] and loaded))
            candidate_choices = [(candidate_label(c), c["candidate_id"]) for c in (selected or {}).get("translation", {}).get("candidates", []) if not c.get("stale")]
            current_candidate = state.get("candidate_preview_id") or (selected or {}).get("translation", {}).get("selected_candidate_id")
            if current_candidate not in [c[1] for c in candidate_choices]:
                current_candidate = None
            candidate_update = gr.update(choices=candidate_choices, value=current_candidate, interactive=bool(permissions["translation"] and loaded))
            emotion_update = gr.update(value=state.get("emotion_draft", (selected or {}).get("emotion", {}).get("label")), interactive=bool(permissions["emotion"] and loaded))
            engines = available_tts_engines(job)
            model_choice = state.get("trial_model_choice")
            if model_choice not in [e[1] for e in engines]:
                model_choice = None
                state["trial_model_choice"] = model_choice
            trial_model_update = gr.update(choices=engines, value=model_choice, interactive=bool(permissions["tts"] and loaded and caps.get("trial_tts_models") and routed and translated))
            if clear_editor:
                source_update.update(value="")
                translation_update.update(value="")
                candidate_update.update(choices=[], value=None)
                emotion_update.update(value=None)
                trial_model_update.update(value=None)
            if loaded_segment is not None:
                fields = load_fields if load_fields is not None else {"source_text", "translation_text", "candidate_id", "emotion_label", "trial_model"}
                if "source_text" in fields:
                    source_update.update(value=loaded_segment["source"]["effective_text"])
                if "translation_text" in fields or "candidate_id" in fields:
                    translation_update.update(value=loaded_segment["translation"]["text"])
                    choices = [(candidate_label(c), c["candidate_id"]) for c in loaded_segment["translation"]["candidates"] if not c.get("stale")]
                    candidate_update.update(choices=choices, value=current_candidate)
                if "emotion_label" in fields:
                    emotion_update.update(value=loaded_segment["emotion"]["label"])
                if "trial_model" in fields:
                    trial_model_update.update(value=model_choice)
            context = "运行至检查点后，对应审阅板块会自动展开。"
            if permissions["source"]:
                context = "**OCR 审阅** · 核对当前源文。确认继续后源文锁定，进入后续处理。"
            elif permissions["emotion"]:
                context = "**情绪审阅** · 核对并修正识别结果。确认后才进入翻译。"
            elif permissions["translation"]:
                context = "**译文与轻量试听审阅** · 核对译文和候选。确认后才计算模型路由。"
            elif permissions["routing"]:
                context = "**模型路由审阅** · 核对计划模型及依据。情绪和译文已只读；确认后才开始正式配音。"
            elif permissions["tts"]:
                context = "**配音审阅** · 试听并确认，或重做当前句 / 返回翻译。已确认的文本保持只读。"
            elif completed:
                context = "**任务已完成** · 结果仅供查看和试听，编辑、返工和模型替换已锁定。"
            elif job and job["active_operation"]:
                context = "**执行中** · 已有文本只读，当前步骤完成后自动开放对应审阅。"
            tts_open = permissions["tts"]
            media_values = playback_values(job, selected, client.get_artifact_url, state.get("candidate_preview_id"))
            formal_started = bool(job and any(s["id"] == "tts" and s["attempt"] > 0 for s in job["stages"]))
            media_values[-1] = gr.update(value=media_values[-1], visible=formal_started)
            trials = (selected or {}).get("tts_trials", [])
            trial_choices = [(f"{ENGINE_NAMES.get(t['engine'], t['engine'])} · {'待更新' if t.get('stale') else {'success':'已完成','queued':'排队中','running':'运行中','failed':'失败'}.get(t['status'],t['status'])} · r{t['revision']} · {t['trial_id'][-4:]}", t["trial_id"]) for t in trials]
            selected_trial = state.get("trial_preview_id")
            trial_html, trial_url, adoptable = trial_view(job, selected, selected_trial or "__unselected__", client.get_artifact_url)
            values = (summary_html(job), stage_html(job), segment_rows(state["segments"]),
                    gr.update(choices=[s["segment_id"] for s in state["segments"]], value=state["selected_id"]),
                    event_html(state["events"]), log_text(state["events"]), details_html(selected), media_html(job, selected),
                    gr.update(interactive=bool(idle and job["status"] == "created"), visible=bool(not job or job["status"] == "created")),
                    gr.update(interactive=bool(idle and job["status"] != "created" and job["resume_from"]), visible=bool(job and job["status"] not in {"created", "completed"})),
                    gr.update(interactive=bool(job and job["active_operation"] and job.get("active_operation_kind") != "tts_trial"), visible=not completed),
                    gr.update(interactive=bool(permissions["tts"] and routed and translated and caps.get("rerun_tts_segments")), visible=permissions["tts"]),
                    gr.update(interactive=bool(permissions["tts"] and "translation" in rework_stages), visible=permissions["tts"]),
                    gr.update(interactive=bool(permissions["source"] and loaded), visible=permissions["source"]),
                    gr.update(interactive=bool(permissions["translation"] and loaded), visible=permissions["translation"]),
                    gr.update(interactive=bool(permissions["translation"] and loaded and current_candidate), visible=permissions["translation"]),
                    gr.update(interactive=bool(idle and selected and not completed)), activity_html(job, state["events"]),
                    *media_values,
                    ocr_evidence_html(job, selected, client.get_artifact_url),
                    review_section_header("source", "源文 / OCR 审阅", permissions["source"]),
                    review_section_header("translation", "译文与候选审阅", permissions["translation"]),
                    review_section_header("tts", "本句配音结果" if completed else "本句配音与返工", tts_open),
                    review_policy_html(permissions, context, tts_open), source_update, translation_update, candidate_update,
                    review_section_header("emotion", "情绪审阅", permissions["emotion"]), emotion_html(selected), emotion_update,
                    gr.update(interactive=bool(permissions["emotion"] and loaded), visible=permissions["emotion"]),
                    {**trial_model_update, "visible": not completed}, gr.update(interactive=bool(permissions["tts"] and loaded and caps.get("trial_tts_models") and model_choice and routed and translated), visible=not completed),
                    gr.update(choices=trial_choices, value=selected_trial, interactive=bool(idle and trials)), trial_html, trial_url,
                    gr.update(interactive=bool(adoptable and loaded), visible=permissions["tts"]),
                    review_section_header("routing", "模型路由审阅", permissions["routing"]),
                    routing_html(selected),
                    gr.update(interactive=bool(permissions["tts"] and "emotion" in rework_stages), visible=permissions["tts"]))
            changes = changed_outputs(state, values)
            # Gradio 6.1 reconstructs components even for skip update dictionaries.
            # For unchanged dropdowns, send the tracked scalar value rather than an
            # empty config update; this preserves choices and the visible selection.
            dropdowns = {picker, candidates, emotion, trial_model, trial_picker}
            return (state, *(value if changed else value.get("value") if output in dropdowns else gr.skip()
                             for value, changed, output in zip(values, changes, view_outputs[1:])))

        def editor_view(edit_state, s):
            return (edit_state, f"{edit_state['segment_id']} · 基于任务 r{edit_state['expected_revision']} 保存",
                    s["source"]["raw_asr_text"], s["source"]["ocr_corrected_text"])

        def error_text(error):
            return f"{error.code}：{error}"

        def refresh_view(state):
            try:
                state = bench.refresh(state)
                key = review_key(state)
                if key and key != state.get("auto_review_key") and state["selected_id"]:
                    state, edit_state, s = bench.select(state, state["selected_id"])
                    feedback = "任务已完成，当前句结果已载入，只读查看。" if state["job"]["status"] == "completed" else "已进入新的审阅步骤并载入当前句。"
                    return (*view(state, s), *editor_view(edit_state, s), feedback)
                return (*view(state), *[gr.skip() for _ in editor_outputs], gr.skip())
            except BackendError as error:
                gr.Warning(error_text(error))
                return tuple(gr.skip() for _ in view_outputs + editor_outputs + [notice])

        def create_job(v, lang, run_mode):
            state = bench.create(v, lang, run_mode)
            return (*view(state, clear_editor=True), state["job_id"], state["job_id"], None,
                    "尚未载入句子", "", "",
                    "新模拟任务已创建；点击开始。视频内容不参与模拟执行。")

        def restore_job(job_id):
            try:
                state = bench.restore(job_id)
                return (*view(state, clear_editor=True), job_id or "", None, "恢复后请载入句子", "", "",
                        "已恢复任务和事件历史。" if job_id else "等待创建模拟任务。")
            except BackendError as error:
                return (*[gr.skip() for _ in view_outputs], gr.skip(),
                        *[gr.skip() for _ in editor_outputs], error_text(error))

        def select_sentence(sid, state):
            try:
                state, edit_state, s = bench.select(state, sid)
                feedback = "当前句结果已载入，只读查看。" if state["job"]["status"] == "completed" else "句子已载入；保存使用此编辑版本。"
                return (*view(state, s), *editor_view(edit_state, s), feedback)
            except BackendError as error:
                return (*[gr.skip() for _ in view_outputs + editor_outputs], error_text(error))

        def select_row(state, evt: gr.SelectData):
            sid = evt.row_value[0] if evt.row_value else None
            return select_sentence(sid, state)

        def edit_sentence(state, edit_state, value, field):
            try:
                current = bench.refresh(state)
                selected = next((s for s in current["segments"] if s["segment_id"] == current["selected_id"]), None)
                permission = {"source_text": "source", "emotion_label": "emotion"}.get(field, "translation")
                if not review_permissions(current["job"], selected)[permission]:
                    raise BackendError("REVIEW_LOCKED", "该文本当前只读，请在对应审阅步骤修改。", 422)
                state, result = bench.edit(state, edit_state, field, value)
                emotion_draft, model_choice = state.get("emotion_draft"), state.get("trial_model_choice")
                candidate_preview = state.get("candidate_preview_id")
                state, edit_state, s = bench.select(state, state["selected_id"])
                if field != "emotion_label":
                    state["emotion_draft"] = emotion_draft
                state["trial_model_choice"] = model_choice
                if field == "emotion_label":
                    state["candidate_preview_id"] = candidate_preview
                return (*view(state, s, load_fields={field}), *editor_view(edit_state, s), edit_notice(result))
            except BackendError as error:
                # No editor outputs on conflicts: preserve unsaved user input.
                return (*view(bench.refresh(state)), *[gr.skip() for _ in editor_outputs], error_text(error))

        def action(state, action_name, from_stage=None):
            try:
                return (*view(bench.action(state, action_name, from_stage)), "请求已接受；等待模拟阶段事件。未保存输入不会用于运行。")
            except BackendError as error:
                return (*view(bench.refresh(state)), error_text(error))

        def confirm_review(state, edit_state, source_text, translation_text, emotion_label):
            try:
                state = bench.confirm_review(state, edit_state, source_text, translation_text, emotion_label)
                return (*view(state), "已确认；当前文本锁定，开始后续处理。")
            except BackendError as error:
                return (*view(bench.refresh(state)), error_text(error))

        def simulate_failure(state):
            try:
                bench.ensure_open(state)
                client.inject_tts_failure(state["job_id"], state["selected_id"])
                return (*view(bench.refresh(state)), "下一次当前句 TTS 将模拟失败；失败后可仅重做此句。")
            except BackendError as error:
                return (*view(bench.refresh(state)), error_text(error))

        def preview_candidate(candidate_id, state):
            state = bench.refresh(state)
            state["candidate_preview_id"] = candidate_id
            return view(state)

        def preview_trial(trial_id, state):
            state = bench.refresh(state)
            state["trial_preview_id"] = trial_id
            return view(state)

        def choose_emotion(value, state):
            state = bench.refresh(state)
            state["emotion_draft"] = value
            return view(state)

        def choose_trial_model(value, state):
            state = bench.refresh(state)
            state["trial_model_choice"] = value
            return view(state)

        def trial_action(state, edit_state, value, action_name):
            try:
                state = bench.tts_trial_action(state, edit_state, action_name, value)
                model_choice = state.get("trial_model_choice")
                state, edit_state, s = bench.select(state, state["selected_id"])
                state["trial_model_choice"] = model_choice
                if action_name == "adopt_tts_trial":
                    state["trial_preview_id"] = value
                return (*view(state, s, load_fields=set()), *editor_view(edit_state, s),
                        "试合成请求已接受；原正式配音保留。" if action_name == "trial_tts" else "已替换本句正式配音；确认后重新合成成品。")
            except BackendError as error:
                return (*view(bench.refresh(state)), *[gr.skip() for _ in editor_outputs], error_text(error))

        serial = {"concurrency_id": "mock-ui", "concurrency_limit": 1, "show_progress": "hidden"}
        create.click(create_job, [video, language, mode], view_outputs + [stored_job, restore_id] + editor_outputs + [notice], **serial)
        demo.load(restore_job, [stored_job], view_outputs + [stored_job] + editor_outputs + [notice], **serial)
        restore.click(restore_job, [restore_id], view_outputs + [stored_job] + editor_outputs + [notice], **serial)
        timer.tick(refresh_view, [session], view_outputs + editor_outputs + [notice], **serial)
        refresh.click(refresh_view, [session], view_outputs + editor_outputs + [notice], **serial)
        picker.input(select_sentence, [picker, session], view_outputs + editor_outputs + [notice], **serial)
        load_sentence.click(select_sentence, [picker, session], view_outputs + editor_outputs + [notice], **serial)
        table.select(select_row, [session], view_outputs + editor_outputs + [notice], **serial)
        save_source.click(lambda st, ed, text: edit_sentence(st, ed, text, "source_text"), [session, editor, source], view_outputs + editor_outputs + [notice], **serial)
        save_translation.click(lambda st, ed, text: edit_sentence(st, ed, text, "translation_text"), [session, editor, translation], view_outputs + editor_outputs + [notice], **serial)
        save_emotion.click(lambda st, ed, value: edit_sentence(st, ed, value, "emotion_label"), [session, editor, emotion], view_outputs + editor_outputs + [notice], **serial)
        emotion.input(choose_emotion, [emotion, session], view_outputs, **serial)
        pin_candidate.click(lambda st, ed, cid: edit_sentence(st, ed, cid, "candidate_id"), [session, editor, candidates], view_outputs + editor_outputs + [notice], **serial)
        candidates.input(preview_candidate, [candidates, session], view_outputs, **serial)
        run.click(lambda st: action(st, "run"), [session], view_outputs + [notice], **serial)
        resume.click(confirm_review, [session, editor, source, translation, emotion], view_outputs + [notice], **serial)
        pause.click(lambda st: action(st, "pause_after_stage"), [session], view_outputs + [notice], **serial)
        redo.click(lambda st: action(st, "rework", "tts"), [session], view_outputs + [notice], **serial)
        back.click(lambda st: action(st, "rework", "translation"), [session], view_outputs + [notice], **serial)
        back_emotion.click(lambda st: action(st, "rework", "emotion"), [session], view_outputs + [notice], **serial)
        fail.click(simulate_failure, [session], view_outputs + [notice], **serial)
        trial_picker.input(preview_trial, [trial_picker, session], view_outputs, **serial)
        trial_model.input(choose_trial_model, [trial_model, session], view_outputs, **serial)
        try_model.click(lambda st, ed, value: trial_action(st, ed, value, "trial_tts"), [session, editor, trial_model], view_outputs + editor_outputs + [notice], **serial)
        adopt_trial.click(lambda st, ed, value: trial_action(st, ed, value, "adopt_tts_trial"), [session, editor, trial_picker], view_outputs + editor_outputs + [notice], **serial)
    return demo, client


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--db", type=Path, default=ROOT / ".data/mock.sqlite3")
    args = parser.parse_args()
    backend = MockBackendClient(args.db)
    app, _ = build_app(backend)
    try:
        app.queue().launch(server_name=args.host, server_port=args.port, share=False,
                           css=(ROOT / "styles.css").read_text(encoding="utf-8"),
                           theme=gr.themes.Soft(font=["system-ui", "sans-serif"]))
    finally:
        backend.close()
