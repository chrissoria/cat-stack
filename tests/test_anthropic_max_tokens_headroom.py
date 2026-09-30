"""
Tests for Anthropic output room on the adaptive-thinking models, and for
replies cut off at max_tokens.

Background: the adaptive-thinking models (Opus 4.7 / 4.8, Sonnet 5, Fable 5)
think by default, with or without a `thinking` field in the request, and the
thinking tokens count against `max_tokens`. cat-stack sent the 4096 default
whenever no thinking budget was set, so a long deliberation could use all of
it: the reply held only a thinking block (stop_reason "max_tokens"), which
parsed to "" and `complete()` returned ("", None), an empty success. Observed
live on claude-sonnet-5 in collapse_themes()'s extract-unique batches
(2026-09-29): 4096 of 4096 output tokens were thinking, and the package kept
those batches unchanged without saying why.

The fix:
  1. `apply_model_params` raises max_tokens to at least
     _ANTHROPIC_ADAPTIVE_MAX_TOKENS_FLOOR for adaptive-thinking models, whether
     or not a thinking budget was set (every Anthropic call site goes through it).
  2. `complete()` treats stop_reason "max_tokens" as incomplete: it retries
     once with double the room (capped), then returns an error instead of a
     truncated reply.
"""

import copy
from unittest.mock import patch, MagicMock

import pytest

from cat_stack._providers import (
    UnifiedLLMClient,
    apply_model_params,
    _ANTHROPIC_ADAPTIVE_MAX_TOKENS_FLOOR as FLOOR,
    _ANTHROPIC_MAX_TOKENS_RETRY_CAP as CAP,
)

ADAPTIVE = ["claude-sonnet-5", "claude-opus-4-8", "claude-fable-5"]
OLDER = ["claude-opus-4-6", "claude-sonnet-4-6", "claude-3-5-haiku-20241022"]


def _resp(json_data):
    r = MagicMock()
    r.status_code = 200
    r.headers = {}
    r.text = ""
    r.json.return_value = json_data
    r.raise_for_status = MagicMock()
    return r


THINKING_ONLY = {
    "stop_reason": "max_tokens",
    "content": [{"type": "thinking", "thinking": "..."}],
}
CUT_MID_LIST = {
    "stop_reason": "max_tokens",
    "content": [{"type": "thinking", "thinking": "..."},
                {"type": "text", "text": "1. Employment\n2. Educa"}],
}
COMPLETE = {
    "stop_reason": "end_turn",
    "content": [{"type": "thinking", "thinking": "..."},
                {"type": "text", "text": "1. Employment\n2. Education"}],
}


class TestHeadroomFloor:
    @pytest.mark.parametrize("model", ADAPTIVE)
    def test_adaptive_model_gets_floor_without_thinking_budget(self, model):
        payload = apply_model_params({"max_tokens": 4096}, "anthropic", model, creativity=0)
        assert payload["max_tokens"] == FLOOR

    @pytest.mark.parametrize("model", ADAPTIVE)
    def test_adaptive_model_small_leaf_limit_raised(self, model):
        # the image/pdf leaves send 1024-2048 for short answers
        payload = apply_model_params({"max_tokens": 1024}, "anthropic", model)
        assert payload["max_tokens"] == FLOOR

    def test_adaptive_model_with_budget_still_at_least_floor(self):
        payload = apply_model_params({"max_tokens": 4096}, "anthropic", "claude-sonnet-5",
                                     thinking_budget=8000)
        assert payload["thinking"] == {"type": "adaptive"}
        assert payload["max_tokens"] == max(8000 + 4096, FLOOR)

    def test_adaptive_model_larger_limit_kept(self):
        payload = apply_model_params({"max_tokens": FLOOR + 5000}, "anthropic", "claude-sonnet-5")
        assert payload["max_tokens"] == FLOOR + 5000

    @pytest.mark.parametrize("model", OLDER)
    def test_older_model_unchanged(self, model):
        payload = apply_model_params({"max_tokens": 4096}, "anthropic", model, creativity=0)
        assert payload["max_tokens"] == 4096

    def test_no_max_tokens_key_not_added(self):
        payload = apply_model_params({}, "anthropic", "claude-sonnet-5")
        assert "max_tokens" not in payload

    def test_runtime_adaptive_flag_gets_floor(self):
        # a future family discovered adaptive by complete()'s 400 fallback
        payload = apply_model_params({"max_tokens": 4096}, "anthropic", "claude-opus-9-future",
                                     overrides={"anthropic_thinking_adaptive": True})
        assert payload["max_tokens"] == FLOOR

    def test_client_payload_gets_floor(self):
        client = UnifiedLLMClient(provider="anthropic", api_key="fake", model="claude-sonnet-5")
        payload = client._build_anthropic_payload([{"role": "user", "content": "hi"}], creativity=0)
        assert payload["max_tokens"] == FLOOR
        assert "temperature" not in payload


class TestTruncatedReply:
    def _client(self, model="claude-sonnet-5"):
        return UnifiedLLMClient(provider="anthropic", api_key="fake", model=model)

    def _recording(self, mock_post, bodies):
        sent = []

        def fake_post(*args, **kwargs):
            sent.append(copy.deepcopy(kwargs["json"]))
            return _resp(bodies[len(sent) - 1])

        mock_post.side_effect = fake_post
        return sent

    @patch("cat_stack._providers.time.sleep")
    @patch("cat_stack._providers.requests.post")
    def test_thinking_only_reply_retried_with_more_room(self, mock_post, mock_sleep):
        sent = self._recording(mock_post, [THINKING_ONLY, COMPLETE])
        result, err = self._client().complete(
            messages=[{"role": "user", "content": "list"}], force_json=False)
        assert err is None
        assert result == "1. Employment\n2. Education"
        assert [p["max_tokens"] for p in sent] == [FLOOR, min(FLOOR * 2, CAP)]

    @patch("cat_stack._providers.time.sleep")
    @patch("cat_stack._providers.requests.post")
    def test_cut_off_twice_is_an_error_not_an_empty_success(self, mock_post, mock_sleep):
        self._recording(mock_post, [THINKING_ONLY, THINKING_ONLY])
        result, err = self._client().complete(
            messages=[{"role": "user", "content": "list"}], force_json=False)
        assert result is None
        assert "max_tokens" in err and "only thinking" in err
        assert mock_post.call_count == 2

    @patch("cat_stack._providers.time.sleep")
    @patch("cat_stack._providers.requests.post")
    def test_cut_mid_answer_is_retried_then_reported(self, mock_post, mock_sleep):
        self._recording(mock_post, [CUT_MID_LIST, CUT_MID_LIST])
        result, err = self._client().complete(
            messages=[{"role": "user", "content": "list"}], force_json=False)
        assert result is None
        assert "cut off" in err and "only thinking" not in err

    @patch("cat_stack._providers.time.sleep")
    @patch("cat_stack._providers.requests.post")
    def test_no_retry_when_no_attempt_left(self, mock_post, mock_sleep):
        self._recording(mock_post, [THINKING_ONLY])
        result, err = self._client().complete(
            messages=[{"role": "user", "content": "list"}], force_json=False, max_retries=1)
        assert result is None and "max_tokens" in err
        assert mock_post.call_count == 1

    @patch("cat_stack._providers.time.sleep")
    @patch("cat_stack._providers.requests.post")
    def test_complete_reply_untouched(self, mock_post, mock_sleep):
        sent = self._recording(mock_post, [COMPLETE])
        result, err = self._client().complete(
            messages=[{"role": "user", "content": "list"}], force_json=False)
        assert err is None and result.startswith("1. Employment")
        assert len(sent) == 1

    @patch("cat_stack._providers.time.sleep")
    @patch("cat_stack._providers.requests.post")
    def test_older_model_cut_off_also_reported(self, mock_post, mock_sleep):
        # the truncation check is not limited to thinking models
        sent = self._recording(mock_post, [CUT_MID_LIST, COMPLETE])
        result, err = self._client("claude-sonnet-4-6").complete(
            messages=[{"role": "user", "content": "list"}], force_json=False)
        assert err is None
        assert [p["max_tokens"] for p in sent] == [4096, FLOOR]
