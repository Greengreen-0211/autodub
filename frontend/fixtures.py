"""Small, explicit demo corpus. Durations, emotions and failures are simulations."""
CORPUS = [
    ("We have one last chance.", "We have one last chance.",
     "我们还有最后一次机会。", "We have one last chance.", "最後のチャンスがある。", "angry"),
    ("I can here you.", "I can hear you.",
     "我能听见你。", "I can hear you.", "声が聞こえる。", "neutral"),
    ("Before the sun rises, we must cross the river, and bring everyone home safely.",
     "Before the sun rises, we must cross the river, and bring everyone home safely.",
     "在太阳升起之前，我们必须穿过这条危险的河流，并把每一个人都安全地带回家。",
     "Before the sun rises, we must cross the river, and bring everyone home safely.",
     "日が昇る前に川を渡り、全員を無事に家へ連れ帰ろう。", "happy"),
]


def new_segments() -> list[dict]:
    return [
        {"segment_id": f"seg_{i + 1:04d}", "parent_segment_id": None,
         "revision": 1, "start": i * 5.0, "end": i * 5.0 + 4.0,
         "speaker": "SPEAKER_00" if i != 1 else "SPEAKER_01",
         "source": {"raw_asr_text": raw, "ocr_corrected_text": None,
                    "effective_text": raw, "origin": "asr",
                    "word_alignment_status": "valid", "audio_artifact_id": None},
         "ocr": {"available": False, "evidence_artifact_id": None, "guard_reason": "pending"},
         "translation": {"text": "", "status": "pending", "origin": "model",
                         "human_pinned": False, "selected_candidate_id": None,
                         "candidates": [], "error": None},
         "emotion": {"label": emotion, "raw_label": emotion, "recognized_label": emotion, "score": None,
                     "reliable": True, "fallback_reason": None, "origin": "model", "status": "pending"},
         "routing": {}, "tts": {"status": "pending", "audio_artifact_id": None},
         "tts_trials": [],
         "history": []}
        for i, (raw, corrected, zh, en, ja, emotion) in enumerate(CORPUS)
    ]


def demo_translation(segment: dict, language: str) -> str:
    index = int(segment["segment_id"].split("_")[1]) - 1
    row = CORPUS[index]
    if segment["source"]["effective_text"] == row[1]:
        return row[{"zh": 2, "en": 3, "ja": 4}[language]]
    # This visibly shows propagation of a source override without pretending to translate it.
    return f"【模拟译文 · {language}】{segment['source']['effective_text']}"
