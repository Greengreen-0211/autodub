"""Regression tests for v2.2.1 issues, with no API/GPU/model downloads."""
import argparse
import ast
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import wave

from translation_quality import (
    applicable_glossary_entries,
    prepare_tts_punctuation,
    translate_segments,
    require_translated,
    validate_target_text,
)
from light_tts_selector import LightTTSSelector, EspeakBackend
from emotion_recognizer import apply_emotion_reliability
from scheme3_policy import build_scheme3_segments, describe_scheme3_decision
import tts_router
from tts_quality import (
    audible_speech_quality,
    duration_quality,
    natural_duration_fit,
    stage_fingerprint,
    validate_mix_state,
)
from tts_preflight import cache_environment, check_model, cached_file, requirements

ROOT=Path(__file__).resolve().parent


def wav(path, duration=1.0):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with wave.open(str(path),"wb") as h:
        h.setnchannels(1); h.setsampwidth(2); h.setframerate(1000)
        # Audible deterministic fixture. Tests that need silence create it
        # explicitly so the formal audible-speech gate remains meaningful.
        h.writeframes(b"\x10\x27"*int(duration*1000))


def source(sid=0,text="This is a full sentence."):
    return {"id":sid,"text":text,"start":float(sid),"end":float(sid)+1,"speaker":"A"}


def translation(sid, tiny=False):
    texts=["等等"] if tiny else ["这是完整的一句话。","这是一句完整的话。","这是个完整句子。"]
    return {"id":str(sid),"text":texts[0],"translations_candidates":[{"text":t} for t in texts]}


class TranslationTests(unittest.TestCase):
    def test_common_words_cannot_become_entity_aliases(self):
        glossary = [{
            "source_forms": ["Olaf", "Where", "you"],
            "target_form": "雪宝",
        }]
        self.assertEqual(
            applicable_glossary_entries({
                "text": "Where are you?",
                "_ocr_translation_references": [{
                    "source_text": "Olaf. Where are you?",
                }],
            }, glossary),
            [],
        )
        self.assertEqual(
            applicable_glossary_entries({"text": "Olaf, where are you?"}, glossary),
            [{"source_forms": ["Olaf"], "target_form": "雪宝"}],
        )

    def test_multi_sentence_comma_chain_restores_strong_boundaries(self):
        source_text = (
            "I just can't. The sky's awake, so I'm awake. "
            "So we have to play."
        )
        target_text = "我睡不着，天还没睡我也睡不着，我们来玩游戏吧。"
        restored, ready, reason = prepare_tts_punctuation(
            target_text, source_text, "zh"
        )
        self.assertTrue(ready)
        self.assertEqual(
            restored,
            "我睡不着。天还没睡我也睡不着。我们来玩游戏吧。",
        )
        self.assertEqual(reason, "deterministic_source_boundaries")

    def test_punctuation_restoration_never_changes_words(self):
        text = "来点魔法，来点魔法"
        restored, ready, _ = prepare_tts_punctuation(
            text, "Do the magic! Do the magic!", "zh"
        )
        self.assertTrue(ready)
        self.assertEqual(restored, "来点魔法！来点魔法！")

    def test_truncated_source_keeps_all_known_sentence_boundaries(self):
        restored, ready, _ = prepare_tts_punctuation(
            "来点魔法，来点魔法，来点",
            "Do the magic! Do the magic! Do the",
            "zh",
        )
        self.assertTrue(ready)
        self.assertEqual(restored, "来点魔法！来点魔法！来点。")

    def test_source_abbreviation_is_not_treated_as_sentence_boundary(self):
        restored, ready, reason = prepare_tts_punctuation(
            "史密斯博士来了", "Dr. Smith is here.", "zh"
        )
        self.assertTrue(ready)
        self.assertEqual(restored, "史密斯博士来了。")
        self.assertEqual(reason, "deterministic_terminal")

    def test_missing_boundary_evidence_uses_deterministic_terminal(self):
        restored, ready, reason = prepare_tts_punctuation(
            "我睡不着天还没睡我们来玩", "I can't. The sky is awake. We play.", "zh"
        )
        self.assertTrue(ready)
        self.assertEqual(restored, "我睡不着天还没睡我们来玩。")
        self.assertEqual(reason, "deterministic_preserve_words:missing_strong_boundary_evidence")

    def test_chinese_ellipsis_preserves_source_pause_boundary(self):
        text = "准备好了吗？嗯……太神奇了！"
        restored, ready, reason = prepare_tts_punctuation(
            text, "Ready? Uhhm... This is amazing!", "zh"
        )
        self.assertTrue(ready)
        self.assertEqual(restored, text)
        self.assertEqual(reason, "source_boundaries_preserved")

    def test_chinese_ellipsis_is_a_valid_terminal(self):
        text = "我想想……"
        restored, ready, _ = prepare_tts_punctuation(
            text, "Let me think...", "zh"
        )
        self.assertTrue(ready)
        self.assertEqual(restored, text)

    def test_variable_length_ellipsis_forms_are_boundaries(self):
        source_text = "Wait...What now."
        for punctuation in ("..", "...", ".....", "…", "……", "………", "．．", "．．．．"):
            with self.subTest(punctuation=punctuation):
                text = f"等等{punctuation}现在怎么办。"
                restored, ready, reason = prepare_tts_punctuation(
                    text, source_text, "zh"
                )
                self.assertTrue(ready)
                self.assertEqual(restored, text)
                self.assertEqual(reason, "source_boundaries_preserved")

    def test_ellipsis_without_surrounding_spaces_is_not_a_decimal(self):
        text = "等等……现在怎么办。"
        restored, ready, reason = prepare_tts_punctuation(
            text, "Wait...What now.", "zh"
        )
        self.assertTrue(ready)
        self.assertEqual(restored, text)
        self.assertEqual(reason, "source_boundaries_preserved")

    def test_missing_terminal_inherits_source_ellipsis(self):
        restored, ready, reason = prepare_tts_punctuation(
            "我想想", "Let me think...", "zh"
        )
        self.assertTrue(ready)
        self.assertEqual(restored, "我想想……")
        self.assertEqual(reason, "deterministic_terminal")

    def test_translation_candidates_are_punctuated_before_light_tts(self):
        source_segment = source(
            text=(
                "I just can't. The sky's awake, so I'm awake. "
                "So we have to play."
            )
        )

        def request(_items):
            return {"translations": [{
                "id": "0",
                "text": "我睡不着，天还没睡我也睡不着，我们来玩游戏吧。",
                "translations_candidates": [
                    {"text": "我睡不着，天还没睡我也睡不着，我们来玩游戏吧。"},
                    {"text": "我实在睡不着，天空还醒着我也醒着，我们必须玩。"},
                ],
            }]}

        rows = translate_segments(
            [source_segment], "en", "zh", "test", request, retries=0
        )
        self.assertEqual(
            rows[0]["text"],
            "我睡不着。天还没睡我也睡不着。我们来玩游戏吧。",
        )
        self.assertTrue(all(
            candidate["text"].count("。") >= 3
            for candidate in rows[0]["translations_candidates"]
        ))

    def test_partial_batch_kept_only_failed_retried(self):
        calls=[]; checkpoints=[]
        def request(items):
            calls.append([r["id"] for r in items])
            return {"translations":[translation(0)] if len(items)>1 else [translation(1)]}
        rows=translate_segments([source(0),source(1)],"en","zh","test",request,
                                checkpoint=lambda r: checkpoints.append(json.loads(json.dumps(r))))
        self.assertEqual(calls,[["0","1"],["1"]])
        self.assertTrue(all(r["translation_status"]=="success" for r in rows))
        self.assertEqual(checkpoints[0][0]["translation_status"],"success")

    def test_content_validation_failure_is_retried_in_isolation(self):
        calls = []
        def request(items):
            calls.append([row["id"] for row in items])
            if len(calls) == 1:
                return {"translations": [{
                    "id": "0", "text": "Any",
                    "translations_candidates": [{"text": "Any"}],
                }]}
            return {"translations": [translation(0, tiny=True)]}
        rows = translate_segments(
            [source(text="Wait.")], "en", "zh", "test", request, retries=2
        )
        self.assertEqual(calls, [["0"], ["0"]])
        self.assertEqual(rows[0]["translation_status"], "success")
        self.assertEqual(rows[0]["text"], "等等。")

    def test_invalid_sibling_candidate_uses_best_valid_without_retry(self):
        def request(items):
            t=translation(0); t["translations_candidates"][1]["text"]="Any"
            return {"translations":[t]}
        rows=translate_segments([source()],"en","zh","test",request,retries=2)
        self.assertEqual(rows[0]["translation_attempts"],1)
        self.assertEqual(rows[0]["translation_status"],"degraded_best_available")
        self.assertTrue(rows[0]["text"])
        self.assertEqual(require_translated(rows,"zh","en"), [])

    def test_no_key_no_silent_source_fallback(self):
        rows=translate_segments([source()],"en","zh","test",None)
        self.assertEqual(rows[0]["translation_status"],"failed")
        with self.assertRaises(RuntimeError):
            require_translated(rows,"zh","en",policy="fail_fast")

    def test_primary_wrong_language_even_good_candidates(self):
        def request(items):
            t=translation(0); t["text"]="Any"; return {"translations":[t]}
        rows=translate_segments([source()],"en","zh","test",request,retries=0)
        self.assertEqual(rows[0]["translation_status"],"degraded_best_available")
        self.assertEqual(rows[0]["text"],"这是完整的一句话。")

    def test_tiny_and_equal_length_candidates(self):
        rows=translate_segments([source(text="Wait.")],"en","zh","test",
                                lambda items:{"translations":[translation(0,tiny=True)]})
        self.assertEqual(rows[0]["translation_status"],"success")
        t=translation(0);t["translations_candidates"]=[{"text":s} for s in ("这是句子","这句话语","一句话语")]
        rows=translate_segments([source()],"en","zh","test",lambda items:{"translations":[t]})
        self.assertEqual(rows[0]["translation_status"],"success")
        self.assertIn("candidate_lengths_similar",rows[0]["translation_warnings"])

    def test_context_and_force_one_cached_id(self):
        calls=[]
        def request(items):
            calls.extend(items); return {"translations":[translation(i["id"]) for i in items]}
        segments=[source(0),source(1)]
        cached=translate_segments(segments,"en","zh","test",request,batch_size=1)
        self.assertEqual(calls[0]["next_text"],segments[1]["text"])
        self.assertEqual(calls[1]["previous_text"],segments[0]["text"])
        self.assertEqual(calls[0]["dialogue_context"][0]["relative_index"], 1)
        self.assertTrue(calls[0]["dialogue_context"][0]["same_speaker"])
        self.assertEqual(
            cached[0]["translation_dialogue_context"],
            calls[0]["dialogue_context"],
        )
        calls.clear()
        translate_segments(segments,"en","zh","test",request,cached=cached,force_ids={"1"})
        self.assertEqual([c["id"] for c in calls],["1"])
        calls.clear()
        translate_segments(segments,"en","zh","new-model",request,cached=cached)
        self.assertEqual(len(calls),2)

    def test_repeated_high_coverage_medium_ocr_reaches_semantic_audit(self):
        captured = []
        segment = source()
        segment["_ocr_translation_references"] = [{
            "source_text": segment["text"],
            "target_text": "这是完整的一句话。",
            "confidence": "medium",
            "reference_policy": "repair_or_compress",
            "source_coverage": 0.95,
            "missing_rate": 0.04,
            "extra_rate": 0.02,
            "frame_support": 3,
        }]

        def request(items):
            captured.extend(items)
            return {"translations": [translation(0)]}

        translate_segments([segment], "en", "zh", "test", request, retries=0)
        self.assertEqual(
            captured[0]["ocr_translation_references"][0]["usage_policy"],
            "preferred_if_complete",
        )

    def test_language_and_identity_rules(self):
        for text,target in (("Any","zh"),("中文 sentence","en"),("","en"),("Hello","ja")):
            with self.assertRaises(ValueError): validate_target_text(text,target)
        with self.assertRaises(ValueError):validate_target_text("Any","en","Any","fr")
        self.assertEqual(validate_target_text("这是NASA的计划。","zh"),"这是NASA的计划。")

    def test_same_language_passthrough_resumes(self):
        rows=translate_segments([source()],"en","en","test",None)
        require_translated(rows,"en","en")
        again=translate_segments([source()],"en","en","test",None,cached=rows)
        self.assertTrue(again[0]["translation_cache_hit"])

    def test_malformed_row_does_not_discard_valid_sibling(self):
        def request(items):
            return {"translations":[translation(0),{"id":"1","text":"缺失候选"}]}
        rows=translate_segments([source(0),source(1)],"en","zh","test",request,retries=0)
        self.assertEqual(
            [r["translation_status"] for r in rows],
            ["success", "degraded_best_available"],
        )

    def test_skip_failed_policy_marks_and_unblocks_later_stages(self):
        rows=translate_segments([source(0),source(1)],"en","zh","test",None)
        with patch.dict(os.environ,{"AUTODUB_TRANSLATION_FAILURE_POLICY":"skip_failed"}):
            skipped=require_translated(rows,"zh","en")
            self.assertEqual([str(s["id"]) for s in skipped],["0","1"])
            self.assertEqual([r["translation_status"] for r in rows],["skipped","skipped"])
            self.assertTrue(all(r["dubbing_enabled"] is False for r in rows))
            # Step 4 re-runs the same gate; degraded segments are not re-reported.
            self.assertEqual(require_translated(rows,"zh","en"),[])

    def test_skip_policy_leaves_valid_segments_alone(self):
        rows=translate_segments([source(0)],"en","zh","test",
                                lambda items:{"translations":[translation(0)]})
        with patch.dict(os.environ,{"AUTODUB_TRANSLATION_FAILURE_POLICY":"skip_failed"}):
            self.assertEqual(require_translated(rows,"zh","en"),[])
        self.assertEqual(rows[0]["translation_status"],"success")
        self.assertNotIn("dubbing_enabled",rows[0])

    def test_best_available_is_default_and_fail_fast_is_explicit(self):
        rows=translate_segments([source()],"en","zh","test",None)
        with patch.dict(os.environ,{},clear=False):
            os.environ.pop("AUTODUB_TRANSLATION_FAILURE_POLICY",None)
            skipped=require_translated(rows,"zh","en")
            self.assertEqual([str(row["id"]) for row in skipped],["0"])
        rows=translate_segments([source()],"en","zh","test",None)
        with patch.dict(os.environ,{"AUTODUB_TRANSLATION_FAILURE_POLICY":"fail_fast"}):
            with self.assertRaises(RuntimeError):require_translated(rows,"zh","en")
        rows=translate_segments([source()],"en","zh","test",None)
        with patch.dict(os.environ,{"AUTODUB_TRANSLATION_FAILURE_POLICY":"nonsense"}):
            skipped=require_translated(rows,"zh","en")
            self.assertEqual([str(row["id"]) for row in skipped],["0"])


class LightTests(unittest.TestCase):
    def test_zh_voice_detection_and_selftest(self):
        def run(command,**kwargs):
            if command[-1]=="--voices":
                return SimpleNamespace(stdout="Pty Language Age/Gender VoiceName File\n 5 zh M Mandarin asia/zh\n 5 en M English en\n")
            wav(command[command.index("-w")+1],.5)
            return SimpleNamespace(stdout="")
        with patch("light_tts_selector.subprocess.run",side_effect=run):
            backend=EspeakBackend("fake-espeak")
            self.assertEqual(backend.voice_for("zh"),"zh")
            backend.preflight("zh")

    def test_candidate_failure_continues_and_all_failed_falls_back(self):
        class Backend:
            name="fake";extension=".wav";signature="test"
            def synthesize(self,text,language,path):
                if text!="good": raise RuntimeError("candidate error")
                wav(path)
        with tempfile.TemporaryDirectory() as tmp:
            selector=LightTTSSelector(tmp,backend=Backend())
            rows,_=selector.select([dict(source(),text="bad",translations_candidates=[{"text":"bad"},{"text":"good"}])],target_language="en")
            self.assertEqual(rows[0]["text"],"good")
            self.assertEqual(rows[0]["translations_candidates"][0]["light_tts_status"],"failed")
            rows,results=selector.select([dict(source(2),text="bad",translations_candidates=[{"text":"bad"}])],target_language="en")
            self.assertEqual(rows[0]["light_tts_status"],"fallback")
            self.assertEqual(results[0]["duration_source"],"text_estimate")

    def test_wrong_language_never_reaches_backend(self):
        class Backend:
            name="fake";extension=".wav";signature="test"
            def synthesize(self,*args): raise AssertionError("should not synthesize")
        with tempfile.TemporaryDirectory() as tmp:
            rows, results = LightTTSSelector(tmp,backend=Backend()).select(
                [source(text="Any")],target_language="zh"
            )
            self.assertEqual(rows, [])
            self.assertEqual(results, [])


class EmotionTests(unittest.TestCase):
    def test_short_neighbor_inheritance_preserves_raw(self):
        rows=[dict(source(0),raw_emotion="happy",emotion_score=.95,emotion_status="success"),
              dict(source(1),end=1.3,raw_emotion="angry",emotion_score=1,emotion_status="success")]
        with patch.dict(os.environ,{"AUTODUB_EMOTION_MIN_DURATION":"0.8"}):
            out=apply_emotion_reliability(rows)
        self.assertEqual(out[1]["emotion_status"],"too_short")
        self.assertEqual(out[1]["emotion"],"happy")
        self.assertEqual(out[1]["emotion_raw_label"],"angry")
        routed=build_scheme3_segments(out,source_lang="en",target_lang="en")
        self.assertEqual(routed[1]["tts_engine"],"cosyvoice3")

    def test_no_cross_speaker_or_high_confidence_override(self):
        rows=[dict(source(0),raw_emotion="happy",emotion_score=.95,emotion_status="success"),
              dict(source(1),end=1.3,speaker="B",raw_emotion="angry",emotion_score=1,emotion_status="success"),
              dict(source(2),raw_emotion="sad",emotion_score=.99,emotion_status="success")]
        out=apply_emotion_reliability(rows)
        self.assertEqual(out[1]["emotion"],"neutral")
        self.assertFalse(out[1]["emotion_reliable"])
        self.assertEqual(out[2]["emotion"],"sad")

    def test_invalid_scores_fail_closed(self):
        out=apply_emotion_reliability([dict(source(),raw_emotion="angry",emotion_score=float("nan"),emotion_status="success")])
        self.assertFalse(out[0]["emotion_reliable"])
        self.assertEqual(out[0]["emotion"],"neutral")


LONG_EN="alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon"


class RoutingPolicyTests(unittest.TestCase):
    def decision(self,text,emotion):
        return describe_scheme3_decision("en",emotion,text)

    def test_default_precedence_keeps_complexity_first_and_flags_the_override(self):
        d=self.decision(LONG_EN,"angry")
        self.assertEqual(d["planned_engine"],"confucius4")
        self.assertEqual(d["rule_code"],"long_or_complex_text")
        self.assertEqual(d["inputs"]["emotion_precedence"],"complexity_first")
        self.assertTrue(d["inputs"]["emotion_overridden_by_complexity"])

    def test_strong_emotion_first_is_opt_in_and_strong_only(self):
        with patch.dict(os.environ,{"AUTODUB_SCHEME3_EMOTION_PRECEDENCE":"strong_emotion_first"}):
            strong=self.decision(LONG_EN,"angry")
            positive=self.decision(LONG_EN,"happy")
        self.assertEqual(strong["planned_engine"],"indextts2")
        self.assertEqual(strong["rule_code"],"strong_emotion")
        self.assertFalse(strong["inputs"]["emotion_overridden_by_complexity"])
        # A positive emotion is not a strong emotion; complexity still wins.
        self.assertEqual(positive["planned_engine"],"confucius4")

    def test_unknown_precedence_falls_back_to_the_shipped_rule(self):
        with patch.dict(os.environ,{"AUTODUB_SCHEME3_EMOTION_PRECEDENCE":"nonsense"}):
            d=self.decision("short line","angry")
        self.assertEqual(d["inputs"]["emotion_precedence"],"complexity_first")
        self.assertEqual(d["planned_engine"],"indextts2")

    def test_neutral_long_text_is_not_an_emotion_override(self):
        d=self.decision(LONG_EN,"neutral")
        self.assertEqual(d["planned_engine"],"confucius4")
        self.assertFalse(d["inputs"]["emotion_overridden_by_complexity"])

    def test_build_scheme3_segments_accepts_an_explicit_precedence(self):
        routed=build_scheme3_segments([dict(source(),text=LONG_EN,emotion="angry")],
                                      source_lang="en",target_lang="en",
                                      emotion_precedence="strong_emotion_first")
        self.assertEqual(routed[0]["tts_engine"],"indextts2")
        self.assertEqual(routed[0]["routing_inputs"]["emotion_precedence"],"strong_emotion_first")
        self.assertFalse(routed[0]["emotion_overridden_by_complexity"])


class RouterTests(unittest.TestCase):
    def setup_router(self,tmp,disabled=False):
        ref=Path(tmp)/"ref.wav";wav(ref)
        router=tts_router.TTSRouter(SimpleNamespace(temp_dir=tmp,target_lang="en",source_lang="en",disable_tts_fallback=disabled),scheme="scheme3")
        router._speaker_references=lambda *args:{"A":{"audio":str(ref),"text":"source text"}}
        return router

    def worker(self,calls,failed):
        def run(model,plan,result,force=False):
            calls.append(model)
            results=json.loads(Path(result).read_text())
            for item in json.loads(Path(plan).read_text())["items"]:
                sid=item["segment_id"]
                if model in failed:
                    results[sid].update(status="failed",error="model failed")
                else:
                    wav(item["raw_path"],item["target_duration"])
                    results[sid].update(status="success",generated_duration=item["target_duration"])
            Path(result).write_text(json.dumps(results))
            return 0
        return run

    def test_fallback_preserves_planned_and_resume_loads_none(self):
        calls=[]
        with tempfile.TemporaryDirectory() as tmp, patch("tts_router.preflight_models"):
            router=self.setup_router(tmp)
            with patch("tts_router._run_worker",side_effect=self.worker(calls,{"confucius4"})):
                clips,results=router.run([dict(source(),tts_engine="confucius4")],"unused")
                self.assertEqual(calls,["confucius4","f5tts"])
                self.assertEqual(results[0]["planned_engine"],"confucius4")
                self.assertEqual(results[0]["actual_engine"],"f5tts")
                self.assertEqual(results[0]["fallback_count"],1)
                calls.clear()
                router.run([dict(source(),tts_engine="confucius4")],"unused")
                self.assertEqual(calls,[])
                router.run([dict(source(text="changed words"),tts_engine="confucius4")],"unused")
                self.assertTrue(calls)

    def test_disabled_fallback(self):
        calls=[]
        with tempfile.TemporaryDirectory() as tmp, patch("tts_router.preflight_models"):
            router=self.setup_router(tmp,disabled=True)
            with patch("tts_router._run_worker",side_effect=self.worker(calls,{"confucius4"})):
                clips,results=router.run([dict(source(),tts_engine="confucius4")],"unused")
            self.assertEqual(clips,[]);self.assertEqual(calls,["confucius4"])
            self.assertEqual(results[0]["status"],"failed")

    def test_only_measured_reliable_source_overlap_is_preserved(self):
        calls=[]
        with tempfile.TemporaryDirectory() as tmp, patch("tts_router.preflight_models"):
            router=self.setup_router(tmp)
            rows=[
                dict(source(0), start=0.0, end=2.0, speaker="A", tts_engine="f5tts"),
                dict(source(1), start=1.6, end=2.6, speaker="B", tts_engine="f5tts"),
            ]
            with patch("tts_router._run_worker",side_effect=self.worker(calls,set())):
                clips,_=router.run(rows,"unused")
            self.assertAlmostEqual(clips[0]["allowed_overlap_seconds"],0.4)
            self.assertTrue(clips[0]["allow_overlap"])

            rows[1]["speaker_boundary_ambiguous"]=True
            with patch("tts_router._run_worker",side_effect=self.worker([],set())):
                clips,_=router.run(rows,"unused",force=True)
            self.assertEqual(clips[0]["allowed_overlap_seconds"],0.0)

    def test_missing_speaker_reference_uses_nearest_safe_reference(self):
        calls = []
        def postprocess(source_path, destination, target_duration):
            shutil.copy2(source_path, destination)
            return {
                "actual_duration": target_duration,
                "atempo_capped": False,
            }
        with tempfile.TemporaryDirectory() as tmp, patch(
            "tts_router.preflight_models"
        ), patch("tts_router._postprocess", side_effect=postprocess), patch(
            "tts_router.electricity_quality", return_value={}
        ), patch("tts_router.voice_consistency_quality", return_value={}):
            router = self.setup_router(tmp)
            segment = dict(source(), speaker="B", tts_engine="f5tts")
            with patch(
                "tts_router._run_worker",
                side_effect=self.worker(calls, set()),
            ):
                clips, results = router.run([segment], "unused")
        self.assertEqual(len(clips), 1)
        self.assertEqual(results[0]["status"], "success")
        self.assertTrue(results[0]["reference_fallback"])
        self.assertEqual(results[0]["reference_fallback_speaker"], "A")

    def test_preflight_precedes_any_worker(self):
        with tempfile.TemporaryDirectory() as tmp, patch("tts_router.preflight_models",side_effect=RuntimeError("missing Vocos")),patch("tts_router._run_worker") as worker:
            clips, rows = self.setup_router(tmp).run([dict(source(),tts_engine="f5tts")],"unused")
            self.assertEqual(clips, [])
            self.assertEqual(rows[0]["status"], "failed")
            worker.assert_not_called()

    def test_missing_first_fallback_can_try_next(self):
        calls=[]
        def preflight(models):
            if "f5tts" in models: raise RuntimeError("missing Vocos")
        with tempfile.TemporaryDirectory() as tmp,patch("tts_router.preflight_models",side_effect=preflight):
            router=self.setup_router(tmp)
            with patch("tts_router._run_worker",side_effect=self.worker(calls,{"confucius4"})):
                clips,rows=router.run([dict(source(),tts_engine="confucius4")],"unused")
            self.assertEqual(calls,["confucius4","indextts2"])
            self.assertEqual(rows[0]["actual_engine"],"indextts2")
            self.assertIn("missing Vocos",rows[0]["fallback_reason"])

    def test_implausibly_short_audio_is_not_kept_as_success(self):
        calls=[]
        def worker(model,plan,result,force=False):
            calls.append(model)
            data=json.loads(Path(result).read_text())
            for item in json.loads(Path(plan).read_text())["items"]:
                duration=.1 if item["text"]=="short" else 1
                wav(item["raw_path"],duration)
                data[item["segment_id"]].update(status="success",generated_duration=duration)
            Path(result).write_text(json.dumps(data)); return 0
        def adapt(src,dst,duration):shutil.copy2(src,dst);return dst
        with tempfile.TemporaryDirectory() as tmp,patch("tts_router.preflight_models"),patch("tts_router._run_worker",side_effect=worker),patch("tts_router._apply_atempo",side_effect=adapt):
            router=self.setup_router(tmp)
            clips,results=router.run([dict(source(text="short"),tts_engine="f5tts",translations_candidates=[{"text":"short"},{"text":"a longer candidate"}])],"unused")
            self.assertEqual(results[0]["status"],"failed")
            self.assertEqual(clips, [])
            self.assertEqual(results[0]["audible_speech_quality"],"failed")
            self.assertEqual(results[0]["tts_text"],"short")
            self.assertGreaterEqual(len(calls),1)

    def test_unsafe_overlength_generation_uses_engine_fallback(self):
        calls=[]
        def worker(model,plan,result,force=False):
            calls.append(model)
            data=json.loads(Path(result).read_text())
            for item in json.loads(Path(plan).read_text())["items"]:
                duration=2.0 if model=="confucius4" else 1.0
                wav(item["raw_path"],duration)
                data[item["segment_id"]].update(status="success",generated_duration=duration)
            Path(result).write_text(json.dumps(data)); return 0
        def postprocess(src,dst,target,**_kwargs):
            shutil.copy2(src,dst)
            actual=tts_router._wav_duration(Path(src))
            return {
                "pre_trim_duration":actual,
                "post_trim_duration":actual,
                "actual_duration":actual,
                "atempo_factor":1.0,
                "atempo_raw_factor":actual/target,
                "atempo_capped":actual>target,
                **tts_router.duration_adaptation_safety(actual,target),
            }
        with tempfile.TemporaryDirectory() as tmp,patch(
            "tts_router.preflight_models"
        ),patch("tts_router._run_worker",side_effect=worker),patch(
            "tts_router._postprocess",side_effect=postprocess
        ),patch("tts_router.electricity_quality",return_value={}),patch(
            "tts_router.voice_consistency_quality",return_value={}
        ):
            router=self.setup_router(tmp)
            _,results=router.run([
                dict(source(),tts_engine="confucius4")
            ],"unused")
        self.assertEqual(calls,["confucius4","f5tts"])
        self.assertEqual(results[0]["status"],"success")
        self.assertEqual(results[0]["actual_engine"],"f5tts")
        self.assertIn("unsafe compression",results[0]["fallback_reason"])

    def test_formal_tts_can_request_two_targeted_retranslations(self):
        calls = []
        rewrites = iter(["rewrite one", "rewrite two"])

        def worker(model, plan, result, force=False):
            calls.append(model)
            data = json.loads(Path(result).read_text())
            durations = {"initial words": 2.0, "rewrite one": 1.4, "rewrite two": 1.1}
            for item in json.loads(Path(plan).read_text())["items"]:
                duration = durations[item["text"]]
                wav(item["raw_path"], duration)
                data[item["segment_id"]].update(
                    status="success", generated_duration=duration
                )
            Path(result).write_text(json.dumps(data))
            return 0

        def postprocess(src, dst, target, **_kwargs):
            shutil.copy2(src, dst)
            natural = tts_router._wav_duration(Path(src))
            return {
                "pre_trim_duration": natural,
                "post_trim_duration": natural,
                "actual_duration": natural,
                "atempo_factor": 1.0,
                "atempo_raw_factor": natural / target,
                "atempo_capped": False,
            }

        def rewrite(_segment, _text, _feedback, _history):
            return {"text": next(rewrites), "native_fluency": "pass"}

        with tempfile.TemporaryDirectory() as tmp, patch(
            "tts_router.preflight_models"
        ), patch("tts_router._run_worker", side_effect=worker), patch(
            "tts_router._postprocess", side_effect=postprocess
        ), patch("tts_router.electricity_quality", return_value={}), patch(
            "tts_router.voice_consistency_quality", return_value={}
        ):
            ref = Path(tmp) / "ref.wav"
            wav(ref)
            router = tts_router.TTSRouter(
                SimpleNamespace(
                    temp_dir=tmp, target_lang="en", source_lang="en",
                    disable_tts_fallback=False,
                ),
                scheme="scheme3",
                duration_rewriter=rewrite,
            )
            router._speaker_references = lambda *_args: {
                "A": {"audio": str(ref), "text": "source text"}
            }
            segment = dict(
                source(text="initial words"), end=2.0, tts_engine="f5tts",
                tts_placement_start=0.0, tts_target_duration=1.0,
                tts_hard_end=2.0,
            )
            clips, results = router.run([segment], "unused")
            first_calls = list(calls)
            quality_state = {
                "segments_step3": [dict(segment, text="rewrite two")],
                "audio_clips": clips,
                "tts_router_results": results,
                "tts_backend": "scheme3",
            }
            quality_config = SimpleNamespace(target_lang="en")
            quality_state["tts_stage_manifest"] = {
                "fingerprint": stage_fingerprint(
                    quality_state, quality_config, "scheme3"
                ),
                "success": 1,
            }
            self.assertEqual(
                validate_mix_state(quality_state, quality_config), []
            )
            calls.clear()
            resumed_clips, resumed_results = router.run([segment], "unused")
            self.assertEqual(calls, [])
            self.assertEqual(resumed_results[0]["tts_text"], "rewrite two")
            self.assertEqual(len(resumed_clips), 1)
            rewritten_segment = dict(segment, text="rewrite two")
            router.run([rewritten_segment], "unused")
            self.assertEqual(calls, [])
        self.assertEqual(first_calls, ["f5tts", "f5tts", "f5tts"])
        self.assertEqual(results[0]["tts_text"], "rewrite two")
        self.assertEqual(results[0]["duration_rewrite_count"], 2)
        self.assertEqual(results[0]["duration_selected_attempt"], 2)
        self.assertEqual(len(results[0]["duration_attempt_history"]), 3)
        self.assertEqual(len(clips), 1)

    def test_f5_uses_native_duration_for_credible_near_miss(self):
        calls=[]
        def worker(model,plan,result,force=False):
            data=json.loads(Path(result).read_text())
            for item in json.loads(Path(plan).read_text())["items"]:
                calls.append(bool(item.get("measure_natural_duration")))
                duration=1.3 if item.get("measure_natural_duration") else 1.0
                wav(item["raw_path"],duration)
                data[item["segment_id"]].update(status="success",generated_duration=duration)
            Path(result).write_text(json.dumps(data)); return 0
        def postprocess(src,dst,target,**_kwargs):
            shutil.copy2(src,dst)
            actual=tts_router._wav_duration(Path(src))
            return {
                "pre_trim_duration":actual,"post_trim_duration":actual,
                "actual_duration":actual,"atempo_factor":1.0,
                "atempo_raw_factor":actual/target,"atempo_capped":False,
                **tts_router.duration_adaptation_safety(actual,target),
            }
        with tempfile.TemporaryDirectory() as tmp,patch(
            "tts_router.preflight_models"
        ),patch("tts_router._run_worker",side_effect=worker),patch(
            "tts_router._postprocess",side_effect=postprocess
        ),patch("tts_router.electricity_quality",return_value={}),patch(
            "tts_router.voice_consistency_quality",return_value={}
        ):
            router=self.setup_router(tmp)
            router.duration_rewriter=lambda *_args,**_kwargs:None
            segment=dict(source(),tts_engine="f5tts")
            _,results=router.run([segment],"unused")
            first_calls=list(calls)
            calls.clear()
            router.run([segment],"unused")
        self.assertEqual(first_calls,[True,False])
        self.assertEqual(calls,[])
        self.assertTrue(results[0]["native_duration_control"])
        self.assertEqual(
            results[0]["duration_selection_reason"],
            "f5_native_duration_final_pass",
        )

    def test_duration_candidate_selector_can_preserve_contextual_wording(self):
        rewrites = iter(["awkward longer one", "awkward longer two"])
        durations = {
            "natural answer": 0.65,
            "awkward longer one": 0.68,
            "awkward longer two": 0.70,
        }
        def worker(_model, plan, result, force=False):
            data = json.loads(Path(result).read_text())
            for item in json.loads(Path(plan).read_text())["items"]:
                wav(item["raw_path"], durations[item["text"]])
                data[item["segment_id"]].update(status="success")
            Path(result).write_text(json.dumps(data))
            return 0
        def postprocess(src, dst, target, **_kwargs):
            shutil.copy2(src, dst)
            actual = tts_router._wav_duration(Path(src))
            return {
                "pre_trim_duration": actual, "post_trim_duration": actual,
                "actual_duration": actual, "atempo_factor": 1.0,
                "atempo_raw_factor": actual / target, "atempo_capped": False,
                **tts_router.duration_adaptation_safety(actual, target),
            }
        selected_payloads = []
        with tempfile.TemporaryDirectory() as tmp, patch(
            "tts_router.preflight_models"
        ), patch("tts_router._run_worker", side_effect=worker), patch(
            "tts_router._postprocess", side_effect=postprocess
        ), patch("tts_router.electricity_quality", return_value={}), patch(
            "tts_router.voice_consistency_quality", return_value={}
        ):
            router = self.setup_router(tmp, disabled=True)
            router.duration_rewriter = lambda *_a, **_k: {"text": next(rewrites)}
            router.duration_candidate_selector = lambda _segment, candidates: (
                selected_payloads.append(candidates)
                or {"selected_attempt": 0, "reason": "context is more natural"}
            )
            _, results = router.run([
                dict(source(text="natural answer"), tts_engine="f5tts")
            ], "unused")
        self.assertEqual(results[0]["duration_selected_attempt"], 0)
        self.assertEqual(results[0]["tts_text"], "natural answer")
        self.assertEqual(len(selected_payloads[0]), 3)


class QualityTests(unittest.TestCase):
    def mix(self,tmp):
        p=Path(tmp)/"a.wav";wav(p)
        state={"segments_step3":[source()],"audio_clips":[{"segment_id":"0","file":str(p),"request_fingerprint":"fp"}],
               "tts_router_results":[{"segment_id":"0","status":"success","request_fingerprint":"fp",
                                      "audible_speech_quality":"passed"}],"tts_backend":"scheme3"}
        request={"text":source()["text"],"language":"en","target_language":"en","emotion":"neutral",
                 "speaker":"A","ref_audio":str(p),"ref_text":"source","target_duration":1,"model":"f5tts"}
        fp=tts_router._request_fingerprint(request)
        state["audio_clips"][0]["request_fingerprint"]=fp
        state["tts_router_results"][0].update(request_fingerprint=fp,generation_request=request,
                                              audio_sha256=tts_router._file_sha256(p))
        config=SimpleNamespace(target_lang="en")
        state["tts_stage_manifest"]={"fingerprint":stage_fingerprint(state,config,"scheme3"),"success":1}
        return state,config

    def test_complete_audio_gate_and_language_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            state,config=self.mix(tmp);validate_mix_state(state,config)
            with self.assertRaises(RuntimeError):validate_mix_state(state,SimpleNamespace(target_lang="zh"))
            Path(state["audio_clips"][0]["file"]).unlink()
            issues=validate_mix_state(state,config)
            self.assertTrue(issues)
            self.assertEqual(state["audio_clips"],[])

    def test_duration_gate(self):
        self.assertEqual(duration_quality(2.49,5.68)["duration_quality"],"failed")
        self.assertEqual(duration_quality(5.6,5.68)["duration_quality"],"passed")
        self.assertEqual(duration_quality(0.45,0.70)["duration_quality"],"failed")
        self.assertEqual(duration_quality(0.60,0.70)["duration_quality"],"passed")

    def test_natural_duration_fit_uses_percentage_rules(self):
        self.assertEqual(natural_duration_fit(0.7, 0.5)["natural_duration_fit"], "failed")
        self.assertEqual(natural_duration_fit(0.4, 0.5)["natural_duration_fit"], "passed")
        self.assertEqual(natural_duration_fit(2.5, 2.0)["duration_rewrite_direction"], "compress")
        self.assertEqual(natural_duration_fit(1.5, 2.0)["duration_rewrite_direction"], "expand")

    def test_trusted_ocr_uses_wider_percentage_band(self):
        self.assertEqual(
            natural_duration_fit(2.6, 2.0)["natural_duration_fit"],
            "failed",
        )
        trusted = natural_duration_fit(2.6, 2.0, "ocr_direct")
        self.assertEqual(trusted["natural_duration_fit"], "passed")
        self.assertEqual(trusted["natural_duration_rule"], "trusted_ocr_ratio")

    def test_silent_wav_fails_audible_speech_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "silent.wav"
            with wave.open(str(path), "wb") as stream:
                stream.setnchannels(1)
                stream.setsampwidth(2)
                stream.setframerate(1000)
                stream.writeframes(b"\x00\x00" * 800)
            result = audible_speech_quality(path, expected_text="把手给我")
            self.assertEqual(result["audible_speech_quality"], "failed")
            self.assertIn("near_silent", result["audible_speech_reason"])

    def test_light_duration_fit_is_ratio_based_and_ocr_aware(self):
        standard = LightTTSSelector._duration_fit(2.6, 2.0, 3.0, "standard")
        trusted = LightTTSSelector._duration_fit(2.6, 2.0, 3.0, "ocr_direct")
        self.assertEqual(standard["duration_fit"], "borderline")
        self.assertEqual(trusted["duration_fit"], "suitable")
        self.assertAlmostEqual(trusted["duration_ratio"], 1.3)

    def test_tampered_wav_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            state,config=self.mix(tmp)
            path=Path(state["audio_clips"][0]["file"])
            data=bytearray(path.read_bytes()); data[-1]=1; path.write_bytes(data)
            with self.assertRaises(RuntimeError):validate_mix_state(state,config)

    def test_skipped_segment_is_allowed_to_have_no_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            state,config=self.mix(tmp)
            state["segments_step3"].append(dict(source(1),translation_status="skipped",dubbing_enabled=False))
            state["tts_stage_manifest"]={"fingerprint":stage_fingerprint(state,config,"scheme3"),"success":1}
            validate_mix_state(state,config)
            # A skipped segment that somehow produced audio is still an error.
            state["audio_clips"].append(dict(state["audio_clips"][0],segment_id="1"))
            with self.assertRaises(RuntimeError):validate_mix_state(state,config)

    def test_cache_environment_and_proxy_isolation(self):
        with patch.dict(os.environ,{"AUTODUB_F5_HF_HOME":"/custom/f5","PYTHONPATH":"/bad","HTTP_PROXY":"http://bad","AUTODUB_TTS_OFFLINE_MODE":"strict_offline"}):
            env=tts_router._worker_environment("f5tts")
            self.assertEqual(env["HF_HOME"],"/custom/f5")
            self.assertNotIn("HTTP_PROXY",env)
            self.assertEqual(env["PYTHONNOUSERSITE"],"1")
            self.assertEqual(env["HF_HUB_OFFLINE"],"1")
        with patch.dict(os.environ,{"AUTODUB_TTS_OFFLINE_MODE":"allow_download"}):
            self.assertEqual(tts_router._worker_environment("confucius4")["HF_HUB_OFFLINE"],"0")

    def test_missing_models_report_file_and_cache(self):
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{"F5TTS_REPO":tmp,"F5TTS_MODEL_PATH":tmp+"/missing.safetensors","AUTODUB_F5_HF_HOME":tmp}):
            report=check_model("f5tts")
            self.assertEqual(report["status"],"failed")
            self.assertIn("missing.safetensors"," ".join(report["missing"]))
            self.assertIn("vocos", " ".join(report["missing"]))

    def test_confucius_bigvgan_uses_real_generator_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            config=Path(tmp)/"config/inference_config.yaml"
            config.parent.mkdir();config.write_text("paths: test")
            parsed={"paths":{"w2v_stat":"checkpoints/stats.pt","tokenizer_path":"checkpoints",
                             "t2s_checkpoint":"t2s_model.safetensors","s2a_checkpoint":"s2a_model.pt",
                             "style_encoder":{"checkpoint":"campplus_cn_common.bin"},
                             "w2v_bert_path":"facebook/w2v-bert-2.0",
                             "vocoder_path":"nvidia/bigvgan_v2_22khz_80band_256x"}}
            with patch.dict(os.environ,{"CONFUCIUS4_REPO":tmp}),patch.dict(sys.modules,{"yaml":SimpleNamespace(safe_load=lambda text:parsed)}):
                files,hubs=requirements("confucius4")
            self.assertIn(("nvidia/bigvgan_v2_22khz_80band_256x","bigvgan_generator.pt"),hubs)
            self.assertNotIn(("nvidia/bigvgan_v2_22khz_80band_256x","__weights__"),hubs)

    def test_cli_environment_default_and_explicit_override(self):
        tree=ast.parse((ROOT/"auto_dubbing_ver_2.0.py").read_text(encoding="utf-8-sig"))
        main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="main")
        configs=[]
        def config(path):
            obj=SimpleNamespace();configs.append(obj);return obj
        globals={"argparse":argparse,"os":os,"sys":sys,"DubbingConfig":config,
                 "run_step_1b":lambda *a,**k:None}
        exec(compile(ast.Module(body=[main],type_ignores=[]),"main","exec"),globals)
        with patch.dict(os.environ,{"AUTODUB_TTS_BACKEND":"scheme3"}),patch.object(sys,"argv",["main","input.mp4","--step","1b"]):
            globals["main"]()
        self.assertEqual(configs[-1].tts_backend,"scheme3")
        with patch.dict(os.environ,{"AUTODUB_TTS_BACKEND":"scheme3"}),patch.object(sys,"argv",["main","input.mp4","--step","1b","--tts-backend","scheme2"]):
            globals["main"]()
        self.assertEqual(configs[-1].tts_backend,"scheme2")


class PipelineTests(unittest.TestCase):
    def functions(self,names,globals):
        tree=ast.parse((ROOT/"auto_dubbing_ver_2.0.py").read_text(encoding="utf-8-sig"))
        nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
        globals.update(DubbingConfig=object,Path=Path,os=os,hashlib=__import__("hashlib"),
                       json=json,List=list,Dict=dict,Any=object)
        exec(compile(ast.Module(body=nodes,type_ignores=[]),"pipeline","exec"),globals)
        return globals

    def test_step3_failure_checkpoint_degrades_without_stale_tts(self):
        state={"segments_step2":[source()],"src_lang":"en", "segments_step3":[source()],"audio_clips":[{}]}
        class Manager:
            def __init__(self,c):pass
            def load(self):return json.loads(json.dumps(state))
            def save(self,data):state.update(json.loads(json.dumps(data)))
            def delete_keys(self,keys):
                for k in keys:state.pop(k,None)
        class Translator:
            def __init__(self,c):pass
            def translate(self,segments,src,target,**kwargs):
                return translate_segments(segments,src,target,"test",None,checkpoint=kwargs["checkpoint"])
        with tempfile.TemporaryDirectory() as tmp:
            config=SimpleNamespace(
                target_lang="zh",model_name="test",openai_api_key="",
                retranslate_ids=set(),temp_dir=tmp,output_dir=tmp,
                project_name="test",light_tts_enabled=False,
            )
            globals=self.functions({
                "run_step_3", "_refine_step3_timing",
                "_translation_candidates_fingerprint",
                "_routing_report_rows", "summarize_routing", "print_routing_report",
            },{"StateManager":Manager,"AdvancedTranslator":Translator})
            globals["run_step_3"](config)
            self.assertNotIn("audio_clips",state)
            self.assertEqual(state["segments_step3"][0]["translation_status"],"skipped")
            self.assertFalse(state["segments_step3"][0]["dubbing_enabled"])

    def test_step4_manifest_and_step5_work_together(self):
        with tempfile.TemporaryDirectory() as tmp:
            ref=Path(tmp)/"reference.wav";wav(ref)
            state={"segments_step3":[dict(source(),tts_engine="confucius4")],"src_lang":"en",
                   "vocals_path":str(ref),"accomp_path":str(ref),"duration":1.0}
            class Manager:
                def __init__(self,c):pass
                def load(self):return json.loads(json.dumps(state))
                def save(self,data):state.update(json.loads(json.dumps(data)))
            mixed=[]
            class Mixer:
                def __init__(self,c):pass
                def run(self,*args):mixed.append(args)
            config=SimpleNamespace(temp_dir=tmp,target_lang="en",disable_tts_fallback=False)
            globals=self.functions({"run_step_4","run_step_5","restore_skipped_vocals"},
                                   {"StateManager":Manager,"Mixer":Mixer})
            worker=RouterTests().worker([],{"confucius4"})
            with patch("tts_router.preflight_models"),patch("tts_router._run_worker",side_effect=worker),patch.object(tts_router.TTSRouter,"_speaker_references",return_value={"A":{"audio":str(ref),"text":"source"}}),patch("scheme3_policy.write_segments_json"):
                config.output_dir=tmp;config.project_name="test"
                globals["run_step_4"](config,backend="scheme3")
                self.assertEqual(state["segments_step3"][0]["tts_engine"],"confucius4")
                self.assertEqual(state["segments_step3"][0]["actual_engine"],"f5tts")
                globals["run_step_5"](config,"unused.mp4")
            self.assertEqual(len(mixed),1)

    def test_skipped_segment_gets_no_tts_and_does_not_restore_original_vocal(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Long enough that the skipped span (1.0s-2.0s) is actually overlaid.
            ref=Path(tmp)/"reference.wav";wav(ref,3.0)
            dub=dict(source(0),tts_engine="confucius4",translation_status="success")
            skipped=dict(source(1),tts_engine="f5tts",translation_status="skipped",dubbing_enabled=False)
            state={"segments_step3":[dub,skipped],"src_lang":"en","vocals_path":str(ref),
                   "accomp_path":str(ref),"duration":1.0}
            class Manager:
                def __init__(self,c):pass
                def load(self):return json.loads(json.dumps(state))
                def save(self,data):state.update(json.loads(json.dumps(data)))
            mixed=[]
            class Mixer:
                def __init__(self,c):pass
                def run(self,*args):mixed.append(args)
            config=SimpleNamespace(
                temp_dir=tmp,target_lang="en",disable_tts_fallback=False,
                tts_backend="scheme3",
            )
            globals=self.functions({"run_step_4","run_step_5","restore_skipped_vocals"},
                                   {"StateManager":Manager,"Mixer":Mixer})
            with patch("tts_router.preflight_models"),patch("tts_router._run_worker",side_effect=RouterTests().worker([],{})),patch.object(tts_router.TTSRouter,"_speaker_references",return_value={"A":{"audio":str(ref),"text":"source"}}),patch("scheme3_policy.write_segments_json"),patch.object(sys.modules["tts_quality"],"validate_mix_state",return_value=[]):
                config.output_dir=tmp;config.project_name="test"
                globals["run_step_4"](config,backend="scheme3")
                self.assertEqual(state["segments_step3"][1]["tts_status"],"skipped")
                self.assertEqual([c["segment_id"] for c in state["audio_clips"]],["0"])
                self.assertEqual(state["tts_stage_manifest"]["total"],2)
                captured=io.StringIO()
                with redirect_stdout(captured):
                    globals["run_step_5"](config,"unused.mp4")
            self.assertEqual(len(mixed),1)
            self.assertEqual(Path(mixed[0][1]), ref)
            self.assertIn("只保留伴奏/静音",captured.getvalue())

    def test_routing_report_summarizes_emotion_coverage(self):
        neutral=dict(source(0),text="hello",emotion="neutral",emotion_reliable=False,
                     tts_engine="f5tts",routing_rule_code="neutral_short_text",
                     routing_inputs={"target_language":"en","emotion":"neutral",
                                     "tts_complexity":{"is_long":False,"is_complex":False,
                                                       "cjk_chars":0,"words":1,"clause_count":1}})
        overridden=dict(source(1),text=LONG_EN,emotion="angry",emotion_raw_label="angry",
                        emotion_reliable=True,tts_engine="confucius4",
                        routing_rule_code="long_or_complex_text",emotion_overridden_by_complexity=True,
                        routing_inputs={"target_language":"en","emotion":"angry",
                                        "tts_complexity":{"is_long":True,"is_complex":False,
                                                          "cjk_chars":0,"words":21,"clause_count":1}})
        globals=self.functions({"_routing_report_rows","summarize_routing","print_routing_report"},{})
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"routing.json"
            summary=globals["print_routing_report"]([neutral,overridden],str(path))
            self.assertEqual(summary["by_engine"],{"confucius4":1,"f5tts":1})
            self.assertEqual(summary["by_rule_code"],{"long_or_complex_text":1,"neutral_short_text":1})
            self.assertEqual(summary["emotion_routed_segments"],0)
            self.assertEqual(summary["emotion_overridden_by_complexity"],1)
            self.assertEqual(summary["emotion_reliability"],{"reliable_segments":1,"unreliable_segments":1})
            written=json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(written["rows"][1]["rule_code"],"long_or_complex_text")
            self.assertEqual(written["rows"][1]["words"],21)

    def test_worker_does_not_reuse_mismatched_request(self):
        import tts_worker
        generated=[]
        class Backend:
            def generate(self,item,path):generated.append(item);wav(path);return 1.0
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);raw=root/"raw.wav";wav(raw)
            plan=root/"plan.json";result=root/"result.json"
            plan.write_text(json.dumps({"items":[{"segment_id":"0","model":"f5tts","raw_path":str(raw),"request_fingerprint":"new"}]}))
            result.write_text(json.dumps({"0":{"status":"success","request_fingerprint":"old"}}))
            with patch.object(tts_worker,"F5TTSBackend",Backend),patch.object(sys,"argv",["worker","--model","f5tts","--plan",str(plan),"--result",str(result)]),patch.dict(os.environ,{"AUTODUB_TTS_OFFLINE_MODE":"allow_download"}):
                tts_worker.main()
            self.assertEqual(len(generated),1)
            self.assertEqual(json.loads(result.read_text())["0"]["request_fingerprint"],"new")


if __name__=="__main__":
    unittest.main(verbosity=2)
