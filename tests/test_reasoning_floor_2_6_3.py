"""
Lowest reasoning setting on every request path (cat-stack 2.6.3).

Three defects, each found by a live preflight on 2026-10-14:
  1. The batch request builder mapped thinking_budget=0 to None, so OpenAI
     and Gemini batch jobs carried no reasoning setting and ran at the
     provider default. The synchronous path already passed 0 through.
  2. GPT-6 was not treated as a reasoning model: it got temperature=0, which
     it rejects (fatal in a batch job, which has no retry), and no
     reasoning_effort. Lowest accepted: Luna "none", Astra and 6.1 Sol "low".
  3. gemini-3.5-flash-lite rejects thinkingBudget: 0 with a bare "invalid
     argument" that the 400 fallback cannot recognise; its lowest is
     thinkingLevel "minimal". gemini-3.8-flash cannot stop reasoning and its
     documented lowest is thinkingLevel "low".
"""

import pytest

from catstack._providers import UnifiedLLMClient, _google_thinking_level_floor

SCHEMA = {"type": "object", "properties": {"1": {"type": "string", "enum": ["0", "1"]}},
          "required": ["1"]}
MSGS = [{"role": "user", "content": "Categorize this text response."}]


def payload(provider, model, thinking_budget=0):
    c = UnifiedLLMClient(provider=provider, api_key="k", model=model)
    return c._build_payload(messages=MSGS, json_schema=SCHEMA, creativity=0,
                            thinking_budget=thinking_budget)


@pytest.mark.parametrize("model,effort", [
    ("gpt-6-luna", "none"), ("gpt-6-astra", "low"), ("gpt-6.1-sol", "low")])
def test_gpt6_lowest_effort_and_no_temperature(model, effort):
    p = payload("openai", model)
    assert p["reasoning_effort"] == effort
    assert "temperature" not in p


@pytest.mark.parametrize("model,cfg", [
    ("gemini-3.5-flash-lite", {"thinkingLevel": "minimal"}),
    ("gemini-3.8-flash", {"thinkingLevel": "low"}),
    ("gemini-4-argon", {"thinkingBudget": 0}),
])
def test_gemini_lowest_thinking(model, cfg):
    assert payload("google", model)["generationConfig"]["thinkingConfig"] == cfg


def test_gemini_floor_matching():
    assert _google_thinking_level_floor("models/gemini-3.8-flash") == "low"
    assert _google_thinking_level_floor("gemini-3.8-flash-lite-tts") is None
    assert _google_thinking_level_floor("gemini-3.5-flash") is None


def test_gemini_positive_budget_unchanged():
    assert payload("google", "gemini-3.8-flash", 2048)["generationConfig"]["thinkingConfig"] == \
        {"thinkingBudget": 2048}


@pytest.mark.parametrize("provider,model,check", [
    ("openai", "gpt-6-luna", lambda p: p["body"]["reasoning_effort"] == "none"),
    ("google", "gemini-3.5-flash-lite",
     lambda p: p["request"]["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "minimal"}),
    ("anthropic", "claude-sonnet-5-5", lambda p: p["params"]["thinking"] == {"type": "between_tools"}),
])
def test_batch_lines_carry_lowest_setting(provider, model, check):
    import catstack._batch as B
    lines = []
    real = B._build_jsonl_line

    def rec(*a, **k):
        line = real(*a, **k); lines.append(line); return line

    class Stop(Exception):
        pass

    def stop(*a, **k):
        raise Stop()

    from unittest.mock import patch
    with patch.object(B, "_build_jsonl_line", side_effect=rec), \
         patch.object(B, "_upload_jsonl", side_effect=stop), \
         patch.object(B, "_create_batch_job", side_effect=stop):
        with pytest.raises(Stop):
            B._run_one_batch_job(
                cfg={"provider": provider, "api_key": "k", "model": model},
                items=["a response"],
                prompt_params={"categories_str": "1. a", "json_schema": SCHEMA,
                               "creativity": 0, "thinking_budget": 0})
    assert lines and check(lines[0])


@pytest.mark.parametrize("key,num", [
    ("1", "1"), ("1. Work", "1"), (".1", "1"), ("#2", "2"), ("(3)", "3"), ("[12]", "12"),
    ("Category 1", ""), ("", "")])
def test_leading_category_number(key, num):
    from catstack._utils import _leading_category_number
    assert _leading_category_number(key) == num


def test_dot_prefixed_keys_validate():
    # DeepSeek V4 Pro on Hugging Face, observed 2026-10-14
    import json
    from catstack._utils import validate_classification_json
    from catstack.text_functions_ensemble import _normalize_json_keys
    reply = json.dumps({".1": "0", ".2": "1", ".3": "0"})
    ok, parsed = validate_classification_json(reply, 3)
    assert ok and parsed == {"1": "0", "2": "1", "3": "0"}
    assert _normalize_json_keys(json.loads(reply), {"1", "2", "3"}) == {"1": "0", "2": "1", "3": "0"}


@pytest.mark.parametrize("model,off", [
    ("moonshotai/Kimi-K2.6:deepinfra", True), ("moonshotai/Kimi-K2.6:novita", True),
    ("Qwen/Qwen3.8-27B:deepinfra", True), ("moonshotai/Kimi-K3:deepinfra", False),
    ("deepseek-ai/DeepSeek-V4.1-Flash:deepinfra", False)])
def test_hf_reasoning_off_switch(model, off):
    p = payload("huggingface", model)
    assert (p.get("chat_template_kwargs") == {"enable_thinking": False}) == off


def test_boolean_values_normalize():
    # Kimi Linear on Hugging Face, observed 2026-10-14
    import json
    from catstack._utils import validate_classification_json, _normalize_category_value
    from catstack.text_functions_ensemble import _normalize_json_keys
    assert [_normalize_category_value(v) for v in (True, False, "true", " FALSE ", 1, "1", "yes")] == \
        ["1", "0", "1", "0", 1, "1", "yes"]
    reply = json.dumps({"1": True, "2": False, "3": "true"})
    ok, parsed = validate_classification_json(reply, 3)
    assert ok and parsed == {"1": "1", "2": "0", "3": "1"}
    assert _normalize_json_keys(json.loads(reply), {"1", "2", "3"}) == {"1": "1", "2": "0", "3": "1"}
    # anything else still fails rather than being guessed at
    assert validate_classification_json(json.dumps({"1": "yes", "2": "0", "3": "0"}), 3)[0] is False


def test_single_validator_and_usable_rule():
    # the ensemble's validator is now the _utils one (".1" keys, true/false values)
    import json
    from catstack.text_functions import validate_classification_json as tf_validate
    from catstack._utils import classification_reply_is_usable as usable
    assert tf_validate(json.dumps({".1": "1", "2": False, "3": "0"}), 3)[0] is True
    assert usable(json.dumps({"3": "1"}), 13)            # sparse: usable
    assert usable(json.dumps({"1": True}), 13)
    assert not usable(json.dumps({"1": "Family"}), 13)   # category names: not usable
    assert not usable(json.dumps({"14": "1"}), 13)       # out of range
    assert not usable('{"1":"e"}', 13)                    # error sentinel
    assert not usable("not json", 13)
