"""
Tests for shared param shaping in the image/pdf strategy leaves
(image_stepback, pdf_stepback, image_CoVe, pdf_CoVe) and the Google
generationConfig placement fix in the text stepback leaf.

These leaves build provider payloads directly; they now route sampling
params through apply_model_params(), which (a) fixes silent temperature
400s on the newest Anthropic generation and (b) places Google params in a
top-level generationConfig.
"""

from unittest.mock import patch, MagicMock

import pytest

from cat_stack.calls.stepback import get_stepback_insight_google
from cat_stack.calls.image_stepback import get_image_stepback_insight_anthropic
from cat_stack.calls.pdf_stepback import get_pdf_stepback_insight_anthropic
from cat_stack.calls.image_CoVe import image_chain_of_verification_anthropic
from cat_stack.calls.pdf_CoVe import pdf_chain_of_verification_anthropic


def _anthropic_text_resp(text="ok"):
    r = MagicMock()
    r.status_code = 200
    r.raise_for_status = MagicMock()
    r.json.return_value = {"content": [{"type": "text", "text": text}]}
    return r


def _google_text_resp(text="ok"):
    r = MagicMock()
    r.status_code = 200
    r.raise_for_status = MagicMock()
    r.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": text}]}}]
    }
    return r


NEW = ["claude-opus-4-8", "claude-sonnet-5", "claude-fable-5"]
OLD_MODEL = "claude-sonnet-4-6"


class TestImagePdfStepbackAnthropic:
    @pytest.mark.parametrize("fn", [
        get_image_stepback_insight_anthropic,
        get_pdf_stepback_insight_anthropic,
    ])
    @pytest.mark.parametrize("model", NEW)
    def test_new_model_omits_temperature(self, fn, model):
        with patch("requests.post", return_value=_anthropic_text_resp()) as mp:
            fn("q", "key", model, creativity=0.3)
        assert "temperature" not in mp.call_args.kwargs["json"]

    @pytest.mark.parametrize("fn", [
        get_image_stepback_insight_anthropic,
        get_pdf_stepback_insight_anthropic,
    ])
    def test_old_model_keeps_temperature(self, fn):
        with patch("requests.post", return_value=_anthropic_text_resp()) as mp:
            fn("q", "key", OLD_MODEL, creativity=0.3)
        assert mp.call_args.kwargs["json"]["temperature"] == 0.3


class TestImagePdfCoVeAnthropic:
    """The multimodal CoVe leaves loop over requests.post via an inner
    make_anthropic_request helper — no create() call may carry temperature
    on the new models, and all must on the old ones."""

    def _run(self, fn, model, content_key):
        with patch("requests.post", return_value=_anthropic_text_resp("a")) as mp:
            fn(
                "init", "<<INITIAL_REPLY>>", "<<QUESTION>>",
                "<<INITIAL_REPLY>><<VERIFICATION_QA>>",
                None,  # deprecated client arg
                model, 0.3,
                remove_numbering=lambda q: q,
                **{content_key: {"type": "image"}},
                api_key="key",
            )
        assert mp.call_args_list, "no requests made"
        return [c.kwargs["json"] for c in mp.call_args_list]

    @pytest.mark.parametrize("model", NEW)
    def test_image_cove_new_model_omits_temperature(self, model):
        for payload in self._run(image_chain_of_verification_anthropic, model, "image_content"):
            assert "temperature" not in payload

    def test_image_cove_old_model_keeps_temperature(self):
        for payload in self._run(image_chain_of_verification_anthropic, OLD_MODEL, "image_content"):
            assert payload["temperature"] == 0.3

    @pytest.mark.parametrize("model", NEW)
    def test_pdf_cove_new_model_omits_temperature(self, model):
        for payload in self._run(pdf_chain_of_verification_anthropic, model, "pdf_content"):
            assert "temperature" not in payload

    def test_pdf_cove_old_model_keeps_temperature(self):
        for payload in self._run(pdf_chain_of_verification_anthropic, OLD_MODEL, "pdf_content"):
            assert payload["temperature"] == 0.3


class TestStepbackGooglePlacement:
    def test_generation_config_at_top_level(self):
        """Regression: generationConfig was spread inside contents[0], where
        Gemini does not honor it — creativity was silently dropped."""
        with patch("requests.post", return_value=_google_text_resp()) as mp:
            get_stepback_insight_google("q", "key", "gemini-2.5-flash", creativity=0.3)
        payload = mp.call_args.kwargs["json"]
        assert payload["generationConfig"] == {"temperature": 0.3}
        assert "generationConfig" not in payload["contents"][0]

    def test_no_creativity_no_generation_config(self):
        with patch("requests.post", return_value=_google_text_resp()) as mp:
            get_stepback_insight_google("q", "key", "gemini-2.5-flash", creativity=None)
        assert "generationConfig" not in mp.call_args.kwargs["json"]
