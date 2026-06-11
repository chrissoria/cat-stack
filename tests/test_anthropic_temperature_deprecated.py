"""
Tests for the Anthropic `temperature`-deprecation handling.

Background: newer Anthropic models (claude-opus-4-7, claude-opus-4-8)
deprecated the `temperature` parameter and reject any request that sends it
with HTTP 400:

  {"type":"error","error":{"type":"invalid_request_error",
   "message":"`temperature` is deprecated for this model."}}

Pre-fix, _build_anthropic_payload unconditionally set `temperature`, so every
request to those models failed and classify() produced all-NA columns.

The fix mirrors the OpenAI reasoning-model handling:
  1. Proactive: a prefix table (_ANTHROPIC_TEMPERATURE_DEPRECATED) consulted
     by _anthropic_supports_temperature() makes _build_anthropic_payload skip
     `temperature` up-front for the known-deprecated models.
  2. Safety net: complete() detects "temperature" + "deprecated" in a 400
     body, pops the param, caches the decision on the client, and retries —
     covering future model families not yet in the table.

Models that still accept `temperature` (sonnet-4-6, opus-4-6, …) are
unaffected.
"""

from unittest.mock import patch, MagicMock

import pytest

from cat_stack._providers import (
    UnifiedLLMClient,
    _anthropic_supports_temperature,
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


TEMP_DEPRECATED_BODY = (
    '{"type":"error","error":{"type":"invalid_request_error",'
    '"message":"`temperature` is deprecated for this model."}}'
)


class TestCapabilityHelper:
    @pytest.mark.parametrize("model", ["claude-opus-4-7", "claude-opus-4-8"])
    def test_known_deprecated_models_report_unsupported(self, model):
        assert _anthropic_supports_temperature(model) is False

    @pytest.mark.parametrize(
        "model",
        ["claude-sonnet-4-6", "claude-opus-4-6", "claude-sonnet-4-5-20250929"],
    )
    def test_supported_models_report_supported(self, model):
        assert _anthropic_supports_temperature(model) is True


class TestProactiveSkip:
    def test_opus_47_payload_omits_temperature(self):
        client = _anthropic_client("claude-opus-4-7")
        payload = client._build_anthropic_payload(
            [{"role": "user", "content": "hi"}],
            json_schema={"type": "object"},
            creativity=0,
            thinking_budget=0,
        )
        assert "temperature" not in payload

    def test_sonnet_payload_keeps_temperature(self):
        """Regression guard: sonnet must still receive temperature so the
        deterministic-output behavior the project relies on is unchanged."""
        client = _anthropic_client("claude-sonnet-4-6")
        payload = client._build_anthropic_payload(
            [{"role": "user", "content": "hi"}],
            json_schema={"type": "object"},
            creativity=0,
            thinking_budget=0,
        )
        assert payload.get("temperature") == 0


class TestRuntimeFallback:
    @patch("cat_stack._providers.time.sleep")
    @patch("cat_stack._providers.requests.post")
    def test_400_deprecated_strips_and_retries(self, mock_post, mock_sleep):
        """Safety net for a model NOT in the prefix table: temperature is sent,
        the API 400s with 'deprecated', and complete() strips + retries."""
        first = _resp(status_code=400, body=TEMP_DEPRECATED_BODY)
        second = _resp(
            status_code=200,
            json_data={"content": [{"type": "text", "text": '{"1":"1"}'}]},
        )
        mock_post.side_effect = [first, second]

        # A future model not in _ANTHROPIC_TEMPERATURE_DEPRECATED, so the
        # proactive skip does NOT fire and temperature enters the payload.
        client = _anthropic_client("claude-opus-4-99-future")
        result, err = client.complete(
            messages=[{"role": "user", "content": "classify this"}],
            creativity=0,
        )
        assert err is None, f"unexpected error: {err}"

        second_payload = mock_post.call_args_list[1].kwargs["json"]
        assert "temperature" not in second_payload, (
            f"temperature not stripped on retry: {second_payload}"
        )
        # Decision cached on the client for subsequent rows.
        assert getattr(client, "_anthropic_temperature_unsupported", False) is True

    @patch("cat_stack._providers.time.sleep")
    @patch("cat_stack._providers.requests.post")
    def test_unrelated_400_not_caught(self, mock_post, mock_sleep):
        """A 400 about something else must not trigger temperature stripping."""
        mock_post.return_value = _resp(400, '{"error":"messages must be a list"}')
        client = _anthropic_client("claude-opus-4-99-future")
        result, err = client.complete(
            messages=[{"role": "user", "content": "x"}],
            creativity=0,
            max_retries=1,
        )
        assert result is None
        assert err is not None
        assert getattr(client, "_anthropic_temperature_unsupported", False) is False
