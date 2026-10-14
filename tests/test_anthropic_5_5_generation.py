"""
Tests for the Anthropic 5.5-generation request shape (cat-stack 2.6.1).

Background: Claude Sonnet 5.5, Opus 5.5 and Fable 5.1 reject forced tool use
(`tool_choice: {"type": "tool"}` returns 400 "tool_choice: type "tool" and
"any" are not supported for this model.") on the Messages API and the Message
Batches API. cat-stack forced its `return_categories` tool whenever thinking
was off, so every classification request to these models failed. Observed
live 2026-10-14: a 5-row batch classify on claude-sonnet-5-5 ended with 0/5
requests succeeded.

The fix:
  1. These models get the JSON through structured outputs
     (`output_config.format`, same schema) instead of a forced tool.
  2. Reasoning off (thinking_budget 0 / None) sends the lowest reasoning
     setting each model accepts, because these models think by default:
     Sonnet 5.5 `between_tools`, Haiku 5.5 `disabled`, Opus 5.5 / Fable 5.1
     effort "low" (thinking cannot be turned off).
  3. Opus 5 / 5.5, Haiku 5.5 and Mythos join the temperature and
     adaptive-thinking tables.
  4. complete() switches a client to structured outputs on a runtime 400 that
     names tool_choice, for families not yet listed.
"""

import json
from unittest.mock import patch, MagicMock

import pytest

from cat_stack._providers import (
    UnifiedLLMClient,
    _anthropic_rejects_forced_tool,
    _anthropic_reasoning_floor,
    _anthropic_supports_temperature,
    _anthropic_uses_adaptive_thinking,
)


SCHEMA = {
    "type": "object",
    "properties": {"1": {"type": "string", "enum": ["0", "1"]},
                   "2": {"type": "string", "enum": ["0", "1"]}},
    "required": ["1", "2"],
}
MESSAGES = [{"role": "user", "content": "Categorize this text response."}]


def _client(model):
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


FORCED_TOOL_REJECTED_BODY = (
    '{"type":"error","error":{"type":"invalid_request_error",'
    '"message":"tool_choice: type \\"tool\\" and \\"any\\" are not supported '
    'for this model."}}'
)


class TestCapabilityTables:
    @pytest.mark.parametrize("model", [
        "claude-sonnet-5-5", "claude-opus-5-5", "claude-haiku-5-5",
        "claude-fable-5-1", "claude-mythos-5-1",
    ])
    def test_new_generation_rejects_forced_tool(self, model):
        assert _anthropic_rejects_forced_tool(model) is True

    @pytest.mark.parametrize("model", [
        "claude-sonnet-5", "claude-opus-5", "claude-fable-5",
        "claude-opus-4-8", "claude-sonnet-4-6", "claude-haiku-4-5",
    ])
    def test_earlier_models_keep_forced_tool(self, model):
        assert _anthropic_rejects_forced_tool(model) is False

    @pytest.mark.parametrize("model", [
        "claude-opus-5", "claude-opus-5-5", "claude-haiku-5-5",
        "claude-sonnet-5-5", "claude-fable-5-1", "claude-mythos-5-1",
    ])
    def test_new_generation_is_adaptive_and_temperature_free(self, model):
        assert _anthropic_uses_adaptive_thinking(model) is True
        assert _anthropic_supports_temperature(model) is False

    def test_reasoning_floor(self):
        assert _anthropic_reasoning_floor("claude-sonnet-5-5") == {"thinking": {"type": "between_tools"}}
        assert _anthropic_reasoning_floor("claude-haiku-5-5") == {"thinking": {"type": "disabled"}}
        assert _anthropic_reasoning_floor("claude-opus-5-5") == {"effort": "low"}
        assert _anthropic_reasoning_floor("claude-fable-5-1") == {"effort": "low"}
        for m in ("claude-sonnet-5", "claude-opus-4-8", "claude-sonnet-4-6"):
            assert _anthropic_reasoning_floor(m) is None


class TestPayloadShape:
    @pytest.mark.parametrize("model", [
        "claude-sonnet-5-5", "claude-opus-5-5", "claude-haiku-5-5", "claude-fable-5-1",
    ])
    def test_structured_outputs_replace_forced_tool(self, model):
        payload = _client(model)._build_anthropic_payload(
            MESSAGES, json_schema=SCHEMA, creativity=0.0, thinking_budget=0,
        )
        assert "tools" not in payload and "tool_choice" not in payload
        fmt = payload["output_config"]["format"]
        assert fmt["type"] == "json_schema"
        assert fmt["schema"]["properties"] == SCHEMA["properties"]
        assert fmt["schema"]["additionalProperties"] is False
        # the caller's schema is not mutated
        assert "additionalProperties" not in SCHEMA
        # temperature 0 is not sent to models that reject sampling params
        assert "temperature" not in payload

    def test_sonnet_5_5_reasoning_off_is_between_tools(self):
        payload = _client("claude-sonnet-5-5")._build_anthropic_payload(
            MESSAGES, json_schema=SCHEMA, creativity=0.0, thinking_budget=0,
        )
        assert payload["thinking"] == {"type": "between_tools"}
        # between_tools 400s at effort above high; no effort is sent
        assert "effort" not in payload["output_config"]

    def test_haiku_5_5_reasoning_off_is_disabled(self):
        payload = _client("claude-haiku-5-5")._build_anthropic_payload(
            MESSAGES, json_schema=SCHEMA, thinking_budget=None,
        )
        assert payload["thinking"] == {"type": "disabled"}

    @pytest.mark.parametrize("model", ["claude-opus-5-5", "claude-fable-5-1"])
    def test_always_thinking_models_get_low_effort(self, model):
        payload = _client(model)._build_anthropic_payload(
            MESSAGES, json_schema=SCHEMA, thinking_budget=0,
        )
        assert "thinking" not in payload
        assert payload["output_config"]["effort"] == "low"
        assert payload["output_config"]["format"]["type"] == "json_schema"

    def test_thinking_budget_still_maps_to_adaptive_effort(self):
        payload = _client("claude-sonnet-5-5")._build_anthropic_payload(
            MESSAGES, json_schema=SCHEMA, thinking_budget=10000,
        )
        assert payload["thinking"] == {"type": "adaptive"}
        assert payload["output_config"]["effort"] == "high"
        assert "format" in payload["output_config"]

    @pytest.mark.parametrize("model", ["claude-sonnet-5", "claude-opus-4-8"])
    def test_earlier_models_unchanged(self, model):
        payload = _client(model)._build_anthropic_payload(
            MESSAGES, json_schema=SCHEMA, thinking_budget=0,
        )
        assert payload["tool_choice"] == {"type": "tool", "name": "return_categories"}
        assert "thinking" not in payload
        assert "output_config" not in payload

    def test_older_model_keeps_temperature(self):
        payload = _client("claude-sonnet-4-6")._build_anthropic_payload(
            MESSAGES, json_schema=SCHEMA, creativity=0.0, thinking_budget=0,
        )
        assert payload["temperature"] == 0.0


class TestResponseParsing:
    def test_structured_output_after_thinking_block(self):
        body = {"content": [
            {"type": "thinking", "thinking": "", "signature": "x"},
            {"type": "text", "text": '{"1": "1", "2": "0"}'},
        ]}
        out = _client("claude-sonnet-5-5")._parse_anthropic_response(body)
        assert json.loads(out) == {"1": "1", "2": "0"}


class TestBatchLine:
    def test_batch_request_uses_structured_outputs(self):
        from cat_stack._batch import _build_jsonl_line
        client = _client("claude-sonnet-5-5")
        payload = client._build_payload(
            messages=MESSAGES, json_schema=SCHEMA, creativity=0.0, thinking_budget=None,
        )
        line = _build_jsonl_line("anthropic", "item-0", payload, "claude-sonnet-5-5")
        params = line["params"]
        assert "tool_choice" not in params
        assert params["output_config"]["format"]["type"] == "json_schema"
        assert params["thinking"] == {"type": "between_tools"}
        assert "temperature" not in params


class TestRuntimeFallback:
    @patch("cat_stack._providers.time.sleep")
    @patch("cat_stack._providers.requests.post")
    def test_forced_tool_400_switches_to_structured_outputs(self, mock_post, _sleep):
        # an unlisted future model that rejects forced tool use
        ok = {"content": [{"type": "text", "text": '{"1": "0", "2": "1"}'}],
              "stop_reason": "end_turn"}
        mock_post.side_effect = [
            _resp(400, FORCED_TOOL_REJECTED_BODY),
            _resp(200, json.dumps(ok), ok),
        ]
        client = _client("claude-sonnet-9")
        reply, err = client.complete(MESSAGES, json_schema=SCHEMA)
        assert err is None
        assert json.loads(reply) == {"1": "0", "2": "1"}
        retried = mock_post.call_args_list[1].kwargs["json"]
        assert "tool_choice" not in retried and "tools" not in retried
        assert retried["output_config"]["format"]["type"] == "json_schema"
        # cached: the next payload from this client goes straight to structured outputs
        nxt = client._build_anthropic_payload(MESSAGES, json_schema=SCHEMA)
        assert "tool_choice" not in nxt
