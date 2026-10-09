"""Frontend boundary for the PROPOSED v0.1 contract; no HTTP service assumed."""
from __future__ import annotations

from typing import Any, Protocol

STAGES = (
    ("separation", "音频分离"), ("asr", "语音识别"), ("ocr", "OCR 核对"),
    ("diarization", "说话人"), ("emotion", "情绪"), ("translation", "翻译"),
    ("light_tts", "轻量 TTS"), ("routing", "模型路由"),
    ("tts", "正式 TTS"), ("mixing", "混音合成"),
)
STAGE_IDS = tuple(s[0] for s in STAGES)
REVIEW_STAGES = {"ocr", "emotion", "light_tts", "routing", "tts"}
LABELS = dict(STAGES)
EMOTION_CHOICES = [("中性 · neutral", "neutral"), ("开心 · happy", "happy"), ("惊讶 · surprised", "surprised"),
                   ("愤怒 · angry", "angry"), ("悲伤 · sad", "sad"), ("恐惧 · fearful", "fearful"), ("厌恶 · disgusted", "disgusted")]


class BackendError(Exception):
    def __init__(self, code: str, message: str, status: int = 409, **details: Any):
        super().__init__(message)
        self.code, self.status, self.details = code, status, details


class BackendClient(Protocol):
    def create_job(self, video: str | None, target_language: str, mode: str,
                   request_id: str) -> dict: ...
    def get_job(self, job_id: str) -> dict: ...
    def list_segments(self, job_id: str, offset: int = 0, limit: int = 50) -> dict: ...
    def list_events(self, job_id: str, after_sequence: int = 0, limit: int = 200) -> dict: ...
    def edit_segment(self, job_id: str, segment_id: str, payload: dict) -> dict: ...
    def perform_action(self, job_id: str, payload: dict) -> dict: ...
    def get_operation(self, job_id: str, operation_id: str) -> dict: ...
    def get_artifact_url(self, job_id: str, artifact_id: str) -> str | None: ...
