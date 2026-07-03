"""
Tests for temperature gating in the calls/ strategy leaves (stepback, CoVe,
top_n).

These per-strategy modules build the Anthropic payload directly (raw requests /
the Anthropic SDK), bypassing UnifiedLLMClient._build_anthropic_payload — so the
temperature-deprecation fix there did NOT cover them. On the newest Anthropic
generation (Opus 4.7+, Sonnet 5, Fable 5) `temperature` returns 400; these
leaves swallow exceptions, so the failure was silent (lost stepback insight /
CoVe verification / top_n categories). Each leaf now routes sampling params
through the shared apply_model_params() shaper, same as the central builder.
"""

from unittest.mock import patch, MagicMock

import pytest

from cat_stack.calls.stepback import get_stepback_insight_anthropic
from cat_stack.calls.top_n import get_anthropic_top_n
from cat_stack.calls.CoVe import chain_of_verification_anthropic


def _text_resp(text="ok"):
    r = MagicMock()
    r.status_code = 200
    r.raise_for_status = MagicMock()
    r.json.return_value = {"content": [{"type": "text", "text": text}]}
    return r


NEW = ["claude-opus-4-8", "claude-sonnet-5", "claude-fable-5"]
OLD = ["claude-sonnet-4-6", "claude-opus-4-6"]


class TestStepbackLeaf:
    @pytest.mark.parametrize("model", NEW)
    def test_new_model_omits_temperature(self, model):
        with patch("requests.post", return_value=_text_resp("insight")) as mp:
            get_stepback_insight_anthropic("q", "key", model, creativity=0.3)
        assert "temperature" not in mp.call_args.kwargs["json"]

    @pytest.mark.parametrize("model", OLD)
    def test_old_model_keeps_temperature(self, model):
        with patch("requests.post", return_value=_text_resp("insight")) as mp:
            get_stepback_insight_anthropic("q", "key", model, creativity=0.3)
        assert mp.call_args.kwargs["json"]["temperature"] == 0.3


class TestTopNLeaf:
    @pytest.mark.parametrize("model", NEW)
    def test_new_model_omits_temperature(self, model):
        with patch("requests.post", return_value=_text_resp("cats")) as mp:
            get_anthropic_top_n("prompt", model, "anthropic", "broad", "key", "rq", 0.3)
        assert "temperature" not in mp.call_args.kwargs["json"]

    @pytest.mark.parametrize("model", OLD)
    def test_old_model_keeps_temperature(self, model):
        with patch("requests.post", return_value=_text_resp("cats")) as mp:
            get_anthropic_top_n("prompt", model, "anthropic", "broad", "key", "rq", 0.3)
        assert mp.call_args.kwargs["json"]["temperature"] == 0.3


class TestCoVeLeaf:
    def _mock_client(self):
        client = MagicMock()
        msg = MagicMock()
        msg.content = [MagicMock(text="a")]
        client.messages.create.return_value = msg
        return client

    @pytest.mark.parametrize("model", NEW)
    def test_new_model_omits_temperature(self, model):
        client = self._mock_client()
        chain_of_verification_anthropic(
            "init", "<<INITIAL_REPLY>>", "<<QUESTION>>", "<<INITIAL_REPLY>><<VERIFICATION_QA>>",
            client, model, 0.3, remove_numbering=lambda q: q,
        )
        # No create() call should carry temperature on the new models.
        for call in client.messages.create.call_args_list:
            assert "temperature" not in call.kwargs

    def test_old_model_keeps_temperature(self):
        client = self._mock_client()
        chain_of_verification_anthropic(
            "init", "<<INITIAL_REPLY>>", "<<QUESTION>>", "<<INITIAL_REPLY>><<VERIFICATION_QA>>",
            client, "claude-sonnet-4-6", 0.3, remove_numbering=lambda q: q,
        )
        assert client.messages.create.call_args_list, "no create() calls made"
        assert all(
            call.kwargs.get("temperature") == 0.3
            for call in client.messages.create.call_args_list
        )
