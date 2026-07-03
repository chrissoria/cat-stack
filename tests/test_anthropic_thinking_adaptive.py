"""
Tests for Anthropic extended-thinking payload shaping.

Background: newer Anthropic generations (Opus 4.7 / 4.8, Sonnet 5, Fable 5)
removed the legacy fixed-budget thinking API. Sending
`thinking: {"type": "enabled", "budget_tokens": N}` to them returns HTTP 400 —
they require adaptive thinking (`thinking: {"type": "adaptive"}`). Older models
(Opus 4.6, Sonnet 4.6, and earlier) still accept the explicit budget.

Pre-fix, _build_anthropic_payload always emitted the fixed-budget form, so
`classify(..., thinking_budget=N)` on a current model hard-400'd with no
fallback (unlike the temperature case, which had one).

The fix mirrors the temperature handling:
  1. Proactive: a prefix table (_ANTHROPIC_ADAPTIVE_THINKING) consulted by
     _anthropic_uses_adaptive_thinking() makes _build_anthropic_payload emit the
     adaptive form for the known-affected models (and skip temperature, which
     those models also reject).
  2. Safety net: complete() detects a thinking/budget_tokens rejection 400,
     rewrites the payload to adaptive, caches the decision, and retries —
     covering future model families not yet in the table.

Also covers _parse_anthropic_response preferring a tool_use block over a text
preamble, since the thinking path uses tool_choice="auto" (a preamble can
precede the structured tool call).
"""

from unittest.mock import patch, MagicMock

import pytest

from cat_stack._providers import (
    UnifiedLLMClient,
    _anthropic_uses_adaptive_thinking,
)


def _anthropic_client(model):
    return UnifiedLLMClient(provider="anthropic", api_key="fake", model=model)


def _resp(status_code=200, body="", json_data=None):
    import requests
    r = MagicMock()
    r.status_code = status_code
    r.headers = {}
    r.text = body
    r.json.return_value = json_data or {}
    if status_code >= 400:
        r.raise_for_status.side_effect = requests.exceptions.HTTPError(
            f"{status_code} Error", response=r,
        )
    else:
        r.raise_for_status = MagicMock()
    return r


# A model that rejects fixed-budget thinking and asks for adaptive.
THINKING_REJECTED_BODY = (
    '{"type":"error","error":{"type":"invalid_request_error",'
    '"message":"`thinking.budget_tokens` is not supported on this model; '
    'use adaptive thinking."}}'
)


class TestCapabilityHelper:
    @pytest.mark.parametrize(
        "model",
        ["claude-opus-4-7", "claude-opus-4-8", "claude-sonnet-5", "claude-fable-5"],
    )
    def test_new_models_use_adaptive(self, model):
        assert _anthropic_uses_adaptive_thinking(model) is True

    @pytest.mark.parametrize(
        "model",
        ["claude-opus-4-6", "claude-sonnet-4-6", "claude-sonnet-4-5", "claude-3-opus"],
    )
    def test_older_models_use_fixed_budget(self, model):
        assert _anthropic_uses_adaptive_thinking(model) is False


class TestProactiveShape:
    @pytest.mark.parametrize(
        "model", ["claude-opus-4-8", "claude-sonnet-5", "claude-fable-5"]
    )
    def test_new_model_emits_adaptive_and_no_temperature(self, model):
        client = _anthropic_client(model)
        payload = client._build_anthropic_payload(
            [{"role": "user", "content": "hi"}],
            json_schema={"type": "object"},
            creativity=0.3,
            thinking_budget=2000,
        )
        assert payload["thinking"] == {"type": "adaptive"}
        # Depth is carried via output_config.effort, graded from the budget
        # (2000 <= 2048 -> "low").
        assert payload["output_config"] == {"effort": "low"}
        assert "temperature" not in payload
        # Forced tool_choice is not allowed with thinking on.
        assert payload["tool_choice"] == {"type": "auto"}

    @pytest.mark.parametrize("model", ["claude-opus-4-6", "claude-sonnet-4-6"])
    def test_older_model_keeps_fixed_budget(self, model):
        client = _anthropic_client(model)
        payload = client._build_anthropic_payload(
            [{"role": "user", "content": "hi"}],
            json_schema={"type": "object"},
            creativity=0.3,
            thinking_budget=2000,
        )
        assert payload["thinking"] == {"type": "enabled", "budget_tokens": 2000}
        # Thinking requires temperature == 1 on the legacy path.
        assert payload["temperature"] == 1
        assert payload["tool_choice"] == {"type": "auto"}

    def test_max_tokens_gets_headroom_over_budget(self):
        client = _anthropic_client("claude-opus-4-6")
        payload = client._build_anthropic_payload(
            [{"role": "user", "content": "hi"}],
            thinking_budget=8000,
            max_tokens=4096,
        )
        # budget (8000) >= max_tokens (4096), so max_tokens is bumped above it.
        assert payload["max_tokens"] == 8000 + 4096

    def test_thinking_off_unchanged(self):
        client = _anthropic_client("claude-opus-4-8")
        payload = client._build_anthropic_payload(
            [{"role": "user", "content": "hi"}],
            json_schema={"type": "object"},
            creativity=None,
            thinking_budget=0,
        )
        assert "thinking" not in payload
        assert payload["tool_choice"] == {"type": "tool", "name": "return_categories"}


class TestRuntimeFallback:
    @patch("cat_stack._providers.time.sleep")
    @patch("cat_stack._providers.requests.post")
    def test_400_rewrites_to_adaptive_and_retries(self, mock_post, mock_sleep):
        """A future adaptive model not in the table gets the fixed-budget payload,
        the API 400s, and complete() rewrites to adaptive + retries."""
        first = _resp(status_code=400, body=THINKING_REJECTED_BODY)
        second = _resp(
            status_code=200,
            json_data={"content": [{"type": "tool_use", "input": {"1": "1"}}]},
        )
        mock_post.side_effect = [first, second]

        # Not in _ANTHROPIC_ADAPTIVE_THINKING → proactive path builds fixed-budget.
        client = _anthropic_client("claude-opus-9-future")
        result, err = client.complete(
            messages=[{"role": "user", "content": "classify this"}],
            json_schema={"type": "object"},
            thinking_budget=2000,
        )
        assert err is None, f"unexpected error: {err}"

        second_payload = mock_post.call_args_list[1].kwargs["json"]
        assert second_payload["thinking"] == {"type": "adaptive"}
        # Effort carried over from the rejected budget_tokens (2000 -> "low").
        assert second_payload["output_config"] == {"effort": "low"}
        assert "temperature" not in second_payload
        assert getattr(client, "_anthropic_thinking_adaptive", False) is True

    @patch("cat_stack._providers.time.sleep")
    @patch("cat_stack._providers.requests.post")
    def test_unrelated_400_not_caught(self, mock_post, mock_sleep):
        """A 400 unrelated to thinking must not rewrite the thinking payload."""
        mock_post.return_value = _resp(400, '{"error":"messages must be a list"}')
        client = _anthropic_client("claude-opus-9-future")
        result, err = client.complete(
            messages=[{"role": "user", "content": "x"}],
            json_schema={"type": "object"},
            thinking_budget=2000,
            max_retries=1,
        )
        assert result is None
        assert err is not None
        assert getattr(client, "_anthropic_thinking_adaptive", False) is False


class TestParsePrefersToolUse:
    def test_tool_use_preferred_over_text_preamble(self):
        client = _anthropic_client("claude-opus-4-8")
        response_json = {
            "content": [
                {"type": "thinking", "thinking": ""},
                {"type": "text", "text": "Let me classify this."},
                {"type": "tool_use", "input": {"category": "A"}},
            ]
        }
        assert client._parse_anthropic_response(response_json) == '{"category": "A"}'

    def test_text_returned_when_no_tool_use(self):
        client = _anthropic_client("claude-opus-4-8")
        response_json = {"content": [{"type": "text", "text": "plain answer"}]}
        assert client._parse_anthropic_response(response_json) == "plain answer"

    def test_empty_content_returns_empty_string(self):
        client = _anthropic_client("claude-opus-4-8")
        assert client._parse_anthropic_response({"content": []}) == ""
