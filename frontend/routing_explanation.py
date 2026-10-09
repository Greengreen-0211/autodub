"""Explain a backend policy decision using its recorded inputs; never choose engines."""


def explain_route(code, inputs, emotion_origin="model"):
    language = inputs.get("target_language", "未知")
    language_name = {"zh": "中文", "en": "英文", "ja": "日文"}.get(language, language)
    emotion = inputs.get("emotion", "neutral")
    complexity = inputs.get("tts_complexity", {})
    thresholds = inputs.get("thresholds", {})
    prefix = f"目标语为{language_name}；"
    source = "人工确认情绪" if emotion_origin == "human" else "路由采用情绪"
    if code == "other_language":
        return prefix + "不属于中文/英文，优先使用 OmniVoice 的多语言覆盖。"
    if code == "long_or_complex":
        conditions = []
        if complexity.get("is_long"):
            if complexity.get("language") == "zh":
                conditions.append(f"译文含 {complexity.get('cjk_chars')} 个汉字，达到长句阈值 {thresholds.get('zh_long_chars')}")
            else:
                conditions.append(f"译文含 {complexity.get('words')} 个词，达到长句阈值 {thresholds.get('en_long_words')}")
        if complexity.get("is_complex"):
            conditions.append(f"分句数 {complexity.get('clause_count')}，达到复杂句阈值 {thresholds.get('complex_clause_count')}")
        condition = "；".join(conditions) or "后端判定为长句/复杂句"
        return prefix + condition + f"；长句/复杂句规则优先于情绪（{emotion}），选择 Confucius4。"
    if code == "zh_en_strong_emotion":
        return prefix + f"未命中长句/复杂句，{source}为 {emotion}，选择情绪表达优先的 IndexTTS2。"
    if code == "positive_emotion":
        return prefix + f"未命中长句/复杂句，{source}为 {emotion}，选择兼顾自然度与情绪的 CosyVoice3。"
    if code == "neutral_short":
        return prefix + f"未命中长句/复杂句，{source}为 {emotion}，普通短句优先速度，选择 F5-TTS。"
    return "后端未提供可解释的规则记录。"
