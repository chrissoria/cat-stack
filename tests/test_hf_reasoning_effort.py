"""HF Qwen3-family reasoning off: reasoning_effort "none" rides with enable_thinking.

The Hugging Face router serves some models through OpenRouter, which ignores
chat_template_kwargs; Qwen3.8-27B on :deepinfra and :novita reasoned on every
call with enable_thinking=False alone (verified 2026-11-16). OpenRouter honors
reasoning_effort "none". A router that rejects the field must have it dropped,
never lowered to "low" (which would turn reasoning on).
"""

from unittest.mock import MagicMock, patch

import pytest

from cat_stack._providers import UnifiedLLMClient, apply_model_params


@pytest.mark.parametrize("model", [
    "Qwen/Qwen3.8-27B:deepinfra", "Qwen/Qwen3.8-27B:novita", "Qwen/Qwen3-8B",
    "moonshotai/Kimi-K2.6:deepinfra",
])
def test_zero_budget_sends_both_switches(model):
    p = apply_model_params({}, "huggingface", model, creativity=0, thinking_budget=0)
    assert p["chat_template_kwargs"] == {"enable_thinking": False}
    assert p["reasoning_effort"] == "none"


@pytest.mark.parametrize("model,budget", [
    ("Qwen/Qwen3.8-27B:deepinfra", 4000),          # reasoning requested
    ("Qwen/Qwen3.8-27B:deepinfra", None),          # reasoning left alone
    ("meta-llama/Llama-3.3-70B-Instruct", 0),      # family without the switch
    ("Qwen/Qwen3.8-2.4T-A95B:novita", 0),          # host rejects enable_thinking
])
def test_no_reasoning_effort_outside_the_gate(model, budget):
    p = apply_model_params({}, "huggingface", model, thinking_budget=budget)
    assert "reasoning_effort" not in p


def test_cached_rejection_leaves_it_off():
    p = apply_model_params({}, "huggingface", "Qwen/Qwen3.8-27B:deepinfra", thinking_budget=0,
                           overrides={"hf_no_reasoning_effort": True})
    assert "reasoning_effort" not in p
    assert p["chat_template_kwargs"] == {"enable_thinking": False}


def _resp(status_code=200, body="", json_data=None):
    import requests
    r = MagicMock()
    r.status_code = status_code
    r.headers = {}
    r.text = body
    r.json.return_value = json_data or {}
    if status_code >= 400:
        r.raise_for_status.side_effect = requests.exceptions.HTTPError(
            f"{status_code} Error", response=r)
    else:
        r.raise_for_status = MagicMock()
    return r


OK = {"choices": [{"message": {"content": '{"1": "1"}'}}]}


@patch("cat_stack._providers.time.sleep")
@patch("cat_stack._providers.requests.post")
def test_rejected_reasoning_effort_is_dropped_not_lowered(mock_post, mock_sleep, capsys):
    sent = []

    def post(url, headers=None, json=None, timeout=None):
        sent.append(dict(json))
        if "reasoning_effort" in json:
            return _resp(400, body="Extra inputs are not permitted, field: 'reasoning_effort'")
        return _resp(200, json_data=OK)

    mock_post.side_effect = post
    client = UnifiedLLMClient(provider="huggingface", api_key="k", model="Qwen/Qwen3.8-27B:deepinfra")
    msgs = [{"role": "user", "content": "x"}]
    reply, err = client.complete(messages=msgs, creativity=0, thinking_budget=0)
    assert err is None and reply == '{"1": "1"}'
    assert [("reasoning_effort" in s) for s in sent] == [True, False]
    assert all(s.get("reasoning_effort") != "low" for s in sent)
    assert sent[-1]["chat_template_kwargs"] == {"enable_thinking": False}
    assert "does not accept reasoning_effort" in capsys.readouterr().out

    # later calls on the same client skip the field up front
    sent.clear()
    client.complete(messages=msgs, creativity=0, thinking_budget=0)
    assert len(sent) == 1 and "reasoning_effort" not in sent[0]
