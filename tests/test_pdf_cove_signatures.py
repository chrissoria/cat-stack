"""
Tests for C11: pdf_chain_of_verification_openai and _mistral now accept
the api_key (+ base_url for openai) kwargs that pdf_functions.py passes,
and execute via direct HTTP requests instead of an SDK client.

Bug shape (verified empirically): pre-fix signatures ended at pdf_content,
so calling either function with `api_key=...` raised
  TypeError: got an unexpected keyword argument 'api_key'.

Every chain_of_verification=True call to pdf_multi_class on OpenAI,
Mistral, Perplexity, HuggingFace, or xAI crashed on the CoVe step.

After the fix, both functions follow the same pattern as
pdf_chain_of_verification_anthropic / _google: accept api_key (+ base_url
for openai), use requests.post directly, return initial_reply on any
error or when api_key is None.
"""

import inspect
from unittest.mock import patch, MagicMock

from cat_stack.calls.pdf_CoVe import (
    pdf_chain_of_verification_openai,
    pdf_chain_of_verification_mistral,
)


def _pdf_content():
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64,FAKE"}}


def _common_kwargs():
    return dict(
        initial_reply="initial categorization",
        step2_prompt="step2 <<INITIAL_REPLY>>",
        step3_prompt="step3 <<QUESTION>>",
        step4_prompt="step4 <<INITIAL_REPLY>> <<VERIFICATION_QA>>",
        client=None,
        user_model="gpt-4o-mini",
        creativity=0,
        remove_numbering=lambda x: x,
        pdf_content=_pdf_content(),
    )


class TestSignatures:
    def test_openai_signature_accepts_api_key_and_base_url(self):
        sig = inspect.signature(pdf_chain_of_verification_openai)
        assert "api_key" in sig.parameters
        assert "base_url" in sig.parameters

    def test_mistral_signature_accepts_api_key(self):
        sig = inspect.signature(pdf_chain_of_verification_mistral)
        assert "api_key" in sig.parameters

    def test_both_keep_client_for_back_compat(self):
        """The deprecated `client` param stays so existing callers that
        pass `client=None` still work."""
        for fn in (pdf_chain_of_verification_openai, pdf_chain_of_verification_mistral):
            sig = inspect.signature(fn)
            assert "client" in sig.parameters

    def test_both_no_longer_use_sdk_client_in_body(self):
        """Migration to direct HTTP — no `client.chat.completions.create`
        / `client.chat.complete` SDK calls left in either function."""
        for fn in (pdf_chain_of_verification_openai, pdf_chain_of_verification_mistral):
            src = inspect.getsource(fn)
            assert "client.chat.completions.create" not in src
            assert "client.chat.complete" not in src
            assert "requests.post" in src


class TestApiKeyGuard:
    def test_openai_no_api_key_returns_initial_reply(self):
        """When api_key is None, the function must short-circuit with
        the initial reply instead of attempting the request."""
        kwargs = _common_kwargs()
        # api_key omitted → defaults to None
        result = pdf_chain_of_verification_openai(**kwargs)
        assert result == "initial categorization"

    def test_mistral_no_api_key_returns_initial_reply(self):
        kwargs = _common_kwargs()
        result = pdf_chain_of_verification_mistral(**kwargs)
        assert result == "initial categorization"


class TestErrorFallback:
    @patch("requests.post")
    def test_openai_returns_initial_reply_on_http_error(self, mock_post):
        """Any non-success response falls back to initial_reply (matches
        the existing anthropic/google CoVe variants' behavior)."""
        import requests as r
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = r.exceptions.HTTPError(
            "500", response=mock_response
        )
        mock_post.return_value = mock_response

        kwargs = _common_kwargs()
        kwargs["api_key"] = "sk-fake"
        result = pdf_chain_of_verification_openai(**kwargs)
        assert result == "initial categorization"

    @patch("requests.post")
    def test_mistral_returns_initial_reply_on_http_error(self, mock_post):
        import requests as r
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = r.exceptions.HTTPError(
            "500", response=mock_response
        )
        mock_post.return_value = mock_response

        kwargs = _common_kwargs()
        kwargs["api_key"] = "fake"
        result = pdf_chain_of_verification_mistral(**kwargs)
        assert result == "initial categorization"


class TestEndpointHonorsBaseUrl:
    @patch("requests.post")
    def test_openai_default_endpoint_is_openai(self, mock_post):
        """When base_url is not provided, hit the OpenAI endpoint."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "ok"}}]
        }
        mock_post.return_value = mock_response

        kwargs = _common_kwargs()
        kwargs["api_key"] = "sk-fake"
        pdf_chain_of_verification_openai(**kwargs)

        # The endpoint URL passed to requests.post must be OpenAI's
        first_call_url = mock_post.call_args_list[0].args[0]
        assert first_call_url == "https://api.openai.com/v1/chat/completions"

    @patch("requests.post")
    def test_openai_honors_base_url_for_compatible_providers(self, mock_post):
        """Custom base_url (Perplexity, HF, xAI, …) is honored."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "ok"}}]
        }
        mock_post.return_value = mock_response

        kwargs = _common_kwargs()
        kwargs["api_key"] = "sk-fake"
        kwargs["base_url"] = "https://api.perplexity.ai"
        pdf_chain_of_verification_openai(**kwargs)

        first_call_url = mock_post.call_args_list[0].args[0]
        assert first_call_url == "https://api.perplexity.ai/chat/completions"
