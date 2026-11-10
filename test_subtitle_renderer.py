import json
from pathlib import Path
import tempfile
import unittest

from subtitle_renderer import (
    build_subtitle_events,
    plan_subtitle_layout,
    read_ocr_subtitle_observations,
    split_long_subtitle_events,
    write_ass,
)


class SubtitleRendererTests(unittest.TestCase):
    def test_ocr_quad_is_normalized_and_bilingual_card_is_unioned(self):
        payload = {
            "a": {
                "status": "ok",
                "timestamp": 2.0,
                "text": (
                    "<ref>Source line</ref><quad>[(250,760),(750,760),(750,810),(250,810)]</quad>"
                    "<ref>原字幕</ref><quad>[(300,825),(700,825),(700,875),(300,875)]</quad>"
                ),
            }
        }
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "ocr_frames.json"
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            rows = read_ocr_subtitle_observations(path, 1920, 1080)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["bbox"], [0.25, 0.76, 0.75, 0.875])

    def test_plain_compact_ocr_keeps_bottom_dialogue_and_filters_watermarks(self):
        payload = {
            "a": {
                "status": "ok",
                "timestamp": 30.897,
                "text": (
                    "没有毛绒绒(18,47),(118,86)"
                    "bilibili(124,31),(200,92)"
                    "臣妾以瓜尔佳氏一族起誓(312,888),(685,947)"
                ),
            },
            "b": {
                "status": "ok",
                "timestamp": 31.815,
                "text": "没有毛绒绒 bilibili(17,30),(200,93)",
            },
        }
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "ocr_frames.json"
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            rows = read_ocr_subtitle_observations(path, 1920, 1080)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["timestamp"], 30.897)
        self.assertEqual(rows[0]["texts"], ["臣妾以瓜尔佳氏一族起誓"])
        self.assertEqual(rows[0]["bbox"], [0.312, 0.888, 0.685, 0.947])

    def test_events_use_final_tts_text_and_actual_mix_timing(self):
        rows = build_subtitle_events(
            [{"id": 3, "start": 1.0, "end": 2.0, "text": "旧译文"}],
            [{"segment_id": 3, "status": "success", "tts_text": "最终配音文本"}],
            [{"segment_id": 3, "start": 1.2, "audible_end": 2.3}],
            10.0,
        )
        self.assertEqual(rows[0]["text"], "最终配音文本")
        self.assertEqual((rows[0]["start"], rows[0]["end"]), (1.2, 2.3))

    def test_unmixed_or_failed_lines_do_not_get_subtitles(self):
        rows = build_subtitle_events(
            [
                {"id": 1, "text": "未混音"},
                {"id": 2, "text": "失败"},
            ],
            [{"segment_id": 2, "status": "failed", "tts_text": "失败"}],
            [{"segment_id": 2, "start": 1, "audible_end": 2}],
            3,
        )
        self.assertEqual(rows, [])

    def test_target_is_always_above_existing_subtitle(self):
        events = [{"segment_id": "0", "start": 1, "end": 2, "text": "新的字幕"}]
        observations = [{"timestamp": 1.5, "bbox": [0.2, 0.78, 0.8, 0.86]}]
        layout = plan_subtitle_layout(events, observations, 1920, 1080, target_language="zh")
        event = layout["events"][0]
        self.assertEqual(layout["mode"], "above_existing_subtitle")
        self.assertLessEqual(event["y"], 0.78 * 1080 - event["gap_pixels"])
        self.assertFalse(event["overlaps_existing_subtitle"])

    def test_1080p_default_font_is_slightly_larger(self):
        events = [{"segment_id": "0", "start": 1, "end": 2, "text": "Readable text"}]
        observations = [{"timestamp": 1.5, "bbox": [0.2, 0.82, 0.8, 0.9]}]
        layout = plan_subtitle_layout(events, observations, 1920, 1080, target_language="en")
        self.assertGreaterEqual(layout["font_size"], 56)

    def test_long_english_line_becomes_timed_consecutive_cards(self):
        events = [{
            "segment_id": "7",
            "start": 10.0,
            "end": 15.0,
            "text": (
                "I stayed awake because the sky was still bright, "
                "and I wanted to wait until you came home safely."
            ),
        }]
        chunks = split_long_subtitle_events(events, "en")
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["start"], 10.0)
        self.assertEqual(chunks[-1]["end"], 15.0)
        self.assertTrue(chunks[0]["text"].endswith(","))
        self.assertTrue(all(
            chunks[index]["end"] == chunks[index + 1]["start"]
            for index in range(len(chunks) - 1)
        ))
        self.assertTrue(all(chunk["end"] - chunk["start"] >= 0.84 for chunk in chunks))
        self.assertEqual(
            " ".join(chunk["text"] for chunk in chunks).replace("  ", " "),
            events[0]["text"],
        )

    def test_short_window_does_not_flash_multiple_cards(self):
        events = [{
            "segment_id": "8", "start": 1.0, "end": 1.7,
            "text": "This text may be long but its spoken window is short.",
        }]
        chunks = split_long_subtitle_events(events, "en")
        self.assertEqual(len(chunks), 1)

    def test_long_chinese_line_prefers_punctuation_boundary(self):
        events = [{
            "segment_id": "9", "start": 3.0, "end": 7.0,
            "text": "天还醒着，所以我也醒着，我想等你平安回来。",
        }]
        chunks = split_long_subtitle_events(events, "zh")
        self.assertGreater(len(chunks), 1)
        self.assertTrue(chunks[0]["text"].endswith("，"))
        self.assertEqual("".join(chunk["text"] for chunk in chunks), events[0]["text"])

    def test_missing_ocr_uses_safe_expanded_canvas_not_top(self):
        events = [{"segment_id": "0", "start": 1, "end": 2, "text": "新的字幕"}]
        layout = plan_subtitle_layout(events, [], 1280, 720, target_language="zh")
        event = layout["events"][0]
        self.assertEqual(layout["mode"], "expanded_bottom_canvas")
        self.assertGreater(event["y"], 720)
        self.assertEqual(event["placement_source"], "expanded_bottom_canvas")
        self.assertTrue(layout["no_overlap_guaranteed"])

    def test_insufficient_space_uses_expanded_canvas(self):
        events = [{
            "segment_id": "0",
            "start": 1,
            "end": 2,
            "text": "这是一条足够长并且需要显示为两行的目标语言字幕文本",
        }]
        observations = [{"timestamp": 1.5, "bbox": [0.2, 0.08, 0.8, 0.14]}]
        layout = plan_subtitle_layout(events, observations, 1920, 1080, target_language="zh")
        self.assertEqual(layout["mode"], "expanded_bottom_canvas")
        self.assertGreater(layout["output_height"], 1080)
        self.assertGreater(layout["events"][0]["y"], 1080)

    def test_ass_has_two_line_wrap_and_position_override(self):
        events = [{
            "segment_id": "0", "start": 1, "end": 2,
            "text": "这是一条足够长并且应当自动换成两行显示的目标语言字幕",
        }]
        layout = plan_subtitle_layout(events, [], 1920, 1080, target_language="zh")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "target.ass"
            write_ass(path, layout)
            content = path.read_text(encoding="utf-8-sig")
        self.assertIn("\\pos(", content)
        self.assertIn("\\N", content)


if __name__ == "__main__":
    unittest.main()
