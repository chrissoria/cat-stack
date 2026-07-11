"""
Tests for the chat_template_kwargs strip-and-retry handler.

Background: classify()'s default `thinking_budget=0` causes
_build_openai_payload to inject `chat_template_kwargs={"enable_thinking":False}`
into HuggingFace payloads. The Groq router (which sits behind HF Inference
Providers for Llama-3.x and openai/gpt-oss families) rejects unknown
properties with HTTP 400:

  {"message":"chat_template_kwargs: property 'chat_template_kwargs' is
   unsupported","type":"invalid_request_error","param":"validation_error",
   "code":"wrong_api_format"}

Pre-fix the call retried until exhausted, every row failed.

Fix mirrors the existing `response_format` strip-and-retry pattern: detect
"chat_template_kwargs" + "unsupported" in the 400 body, pop the param,
retry immediately, and warn only once per client.

NOTE (2026-06-12 reasoning audit): payload injection is now gated to model
families whose chat template honors `enable_thinking` (Qwen3-family only —
see `_hf_model_needs_enable_thinking_off`). Llama/gpt-oss never receive the
kwarg anymore, so these tests use a Qwen3 model to exercise the strip path,
which remains as the safety net for routers that reject the kwarg even for
Qwen models.
"""

from unittest.mock import patch, MagicMock

import pytest

from cat_stack._providers import UnifiedLLMClient


def _make_client():
    # huggingface provider + a Qwen3-family model: injection of
    # chat_template_kwargs is gated to families that honor enable_thinking
    # (see _hf_model_needs_enable_thinking_off), so only a Qwen3 model
    # exercises the strip-and-retry path these tests cover.
    return UnifiedLLMClient(
        provider="huggingface",
        api_key="fake",
        model="Qwen/Qwen3-235B-A22B-Instruct",
    )


def _resp(status_code=200, body="", json_data=None):
    import requests
    r = MagicMock()
    r.status_code = status_code
    r.headers = {}
    r.text = body
    r.json.return_value = json_data or {}
    if status_code >= 400:
        # Match real requests: raise_for_status raises for 4xx/5xx
        r.raise_for_status.side_effect = requests.exceptions.HTTPError(
            f"{status_code} Error", response=r,
        )
    else:
        r.raise_for_status = MagicMock()
    return r


GROQ_REJECTION_BODY = (
    '{"message":"chat_template_kwargs: property '
    "'chat_template_kwargs' is unsupported"
    '","type":"invalid_request_error","param":"validation_error",'
    '"code":"wrong_api_format"}'
)


class TestStripAndRetry:
    @patch("cat_stack._providers.time.sleep")
    @patch("cat_stack._providers.requests.post")
    def test_400_unsupported_chat_template_kwargs_strips_and_retries(
        self, mock_post, mock_sleep
    ):
        """First call: 400 rejecting chat_template_kwargs.
        Second call: payload no longer has the kwarg → 200 success."""
        first = _resp(status_code=400, body=GROQ_REJECTION_BODY)
        second = _resp(
            status_code=200,
            json_data={"choices": [{"message": {"content": '{"1":"1"}'}}]},
        )
        mock_post.side_effect = [first, second]

        client = _make_client()
        # Force thinking_budget=0 path so the kwarg is in the payload.
        result, err = client.complete(
            messages=[{"role": "user", "content": "classify this"}],
            thinking_budget=0,
        )
        assert err is None, f"unexpected error: {err}"
        assert result == '{"1":"1"}'

        # The second call should NOT have chat_template_kwargs in its payload
        second_call_payload = mock_post.call_args_list[1].kwargs["json"]
        assert "chat_template_kwargs" not in second_call_payload, (
            f"chat_template_kwargs not stripped on retry: {second_call_payload}"
        )

    @patch("cat_stack._providers.time.sleep")
    @patch("cat_stack._providers.requests.post")
    def test_warning_printed_only_once(self, mock_post, mock_sleep, capsys):
        """The strip warning should print on first failure, not on every
        subsequent call to complete() that re-triggers the strip."""
        # Two complete() calls; each: 400 → 200
        sequence = [
            _resp(400, GROQ_REJECTION_BODY),
            _resp(200, json_data={"choices": [{"message": {"content": "a"}}]}),
            _resp(400, GROQ_REJECTION_BODY),
            _resp(200, json_data={"choices": [{"message": {"content": "b"}}]}),
        ]
        mock_post.side_effect = sequence

        client = _make_client()
        client.complete(messages=[{"role": "user", "content": "x"}], thinking_budget=0)
        client.complete(messages=[{"role": "user", "content": "y"}], thinking_budget=0)

        out = capsys.readouterr().out
        # The warning marker appears at most once
        assert out.count("does not accept chat_template_kwargs") == 1, out

    @patch("cat_stack._providers.time.sleep")
    @patch("cat_stack._providers.requests.post")
    def test_unrelated_400_is_not_caught_by_this_handler(
        self, mock_post, mock_sleep
    ):
        """A 400 about something else (e.g. malformed message) should not
        accidentally trigger chat_template_kwargs stripping."""
        other_body = '{"error":"messages must be a list"}'
        # max_retries=1 so this hits the failure path quickly
        mock_post.return_value = _resp(400, other_body)

        client = _make_client()
        result, err = client.complete(
            messages=[{"role": "user", "content": "x"}],
            thinking_budget=0,
            max_retries=1,
        )
        # Should not strip and retry — should surface the error.
        assert result is None
        assert err is not None

    @patch("cat_stack._providers.time.sleep")
    @patch("cat_stack._providers.requests.post")
    def test_no_strip_when_kwarg_not_in_payload(self, mock_post, mock_sleep):
        """If somehow the 400 fires but the payload doesn't contain the
        kwarg (defensive), the handler should fall through, not loop."""
        # thinking_budget=None → no chat_template_kwargs added
        # Even if the (unexpected) body mentions chat_template_kwargs, the
        # handler should not infinite-loop trying to strip a missing key.
        mock_post.return_value = _resp(400, GROQ_REJECTION_BODY)

        client = _make_client()
        result, err = client.complete(
            messages=[{"role": "user", "content": "x"}],
            thinking_budget=None,  # kwarg NOT added
            max_retries=1,
        )
        # Without the kwarg in the payload, the handler condition
        # `if "chat_template_kwargs" in payload` is False — we should
        # surface the original error rather than spinning.
        assert mock_post.call_count == 1
        assert err is not None
