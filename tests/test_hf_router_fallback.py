"""
Tests for C10: lazy HuggingFace router fallback.

Before the fix, UnifiedLLMClient.__init__ called _detect_huggingface_endpoint
and discarded the return — burning two probe POSTs (with the API key) on
every client construction, while never populating self._custom_endpoint.

After the fix:
  - __init__ does no probing. Explicit `:router` suffix is honoured
    directly (no probe needed); otherwise self._custom_endpoint stays None.
  - First `complete()` call uses the resolved endpoint.
  - On HTTP 400 with a "wrong router" body, _try_hf_router_fallback probes
    all five known specific routers (Novita, Together, SambaNova, Cerebras,
    Fireworks) plus the generic router, caches the first working one, and
    retries.
  - Subsequent calls use the cached endpoint — no more probes.
"""

from unittest.mock import patch, MagicMock

import pytest

from cat_stack._providers import (
    UnifiedLLMClient,
    _detect_huggingface_endpoint,
    _HF_ROUTER_ENDPOINTS,
    PROVIDER_CONFIG,
)


class TestInitNoEagerProbe:
    def test_init_without_suffix_does_no_probing(self):
        """__init__ must not call requests.post — the eager probe is gone."""
        with patch("cat_stack._providers.requests.post") as mock_post:
            UnifiedLLMClient(
                provider="huggingface",
                api_key="fake-key",
                model="meta-llama/Llama-3.3-70B-Instruct",
            )
        assert not mock_post.called, (
            "Constructing a HuggingFace client must not send probe POSTs"
        )

    def test_init_without_suffix_leaves_custom_endpoint_none(self):
        client = UnifiedLLMClient(
            provider="huggingface",
            api_key="fake-key",
            model="meta-llama/Llama-3.3-70B-Instruct",
        )
        assert client._custom_endpoint is None
        assert client.model == "meta-llama/Llama-3.3-70B-Instruct"

    def test_init_with_explicit_suffix_sets_endpoint_and_strips_model(self):
        client = UnifiedLLMClient(
            provider="huggingface",
            api_key="fake-key",
            model="Qwen/Qwen3-VL-235B:novita",
        )
        assert client._custom_endpoint == (
            f"{_HF_ROUTER_ENDPOINTS['novita']}/chat/completions"
        )
        assert client.model == "Qwen/Qwen3-VL-235B"

    def test_init_explicit_suffix_does_no_probing(self):
        with patch("cat_stack._providers.requests.post") as mock_post:
            UnifiedLLMClient(
                provider="huggingface",
                api_key="fake-key",
                model="Qwen/Qwen3-VL-235B:novita",
            )
        assert not mock_post.called

    def test_non_huggingface_provider_no_endpoint_no_lock_issues(self):
        client = UnifiedLLMClient(
            provider="openai", api_key="fake", model="gpt-4o"
        )
        assert client._custom_endpoint is None
        # Lock is per-instance regardless of provider — should exist for all
        assert hasattr(client, "_endpoint_lock")


class TestIsHfWrongRouter400:
    def _client(self, provider="huggingface"):
        return UnifiedLLMClient(provider=provider, api_key="fake", model="dummy/model")

    def test_recognizes_generic_router_model_not_supported(self):
        c = self._client()
        body = (
            '{"error":{"message":"The requested model X is not supported by '
            'any provider you have enabled.","type":"invalid_request_error",'
            '"param":"model","code":"model_not_supported"}}'
        )
        assert c._is_hf_wrong_router_400(body) is True

    def test_recognizes_specific_router_error(self):
        c = self._client()
        body = '{"error":"Model not supported by provider together"}'
        assert c._is_hf_wrong_router_400(body) is True

    def test_does_not_trigger_on_model_not_found(self):
        """A truly nonexistent model shouldn't trigger probing — no router will help."""
        c = self._client()
        body = '{"error":{"code":"model_not_found","message":"does not exist"}}'
        assert c._is_hf_wrong_router_400(body) is False

    def test_does_not_trigger_for_non_hf_provider(self):
        c = self._client(provider="openai")
        # Even with a matching body, non-HF providers must not trigger HF fallback
        body = '"Model not supported by provider together"'
        assert c._is_hf_wrong_router_400(body) is False

    def test_does_not_trigger_on_unrelated_400(self):
        c = self._client()
        body = '{"error":{"code":"invalid_request","message":"bad max_tokens"}}'
        assert c._is_hf_wrong_router_400(body) is False


class TestTryHfRouterFallback:
    @patch("cat_stack._providers._detect_huggingface_endpoint")
    def test_caches_returned_endpoint_on_success(self, mock_detect):
        mock_detect.return_value = _HF_ROUTER_ENDPOINTS["together"]
        client = UnifiedLLMClient(
            provider="huggingface",
            api_key="fake",
            model="meta-llama/Llama-3.3-70B-Instruct",
        )

        ok = client._try_hf_router_fallback(
            failed_endpoint=PROVIDER_CONFIG["huggingface"]["endpoint"]
        )

        assert ok is True
        assert client._custom_endpoint == (
            f"{_HF_ROUTER_ENDPOINTS['together']}/chat/completions"
        )
        # _detect_huggingface_endpoint should have been called with the
        # failed endpoint in skip (as a base URL)
        args, kwargs = mock_detect.call_args
        assert "skip" in kwargs
        assert PROVIDER_CONFIG["huggingface"]["endpoint"].replace(
            "/chat/completions", ""
        ) in kwargs["skip"]

    @patch("cat_stack._providers._detect_huggingface_endpoint")
    def test_returns_false_when_no_alternative_works(self, mock_detect):
        mock_detect.return_value = None
        client = UnifiedLLMClient(
            provider="huggingface",
            api_key="fake",
            model="meta-llama/Llama-3.3-70B-Instruct",
        )

        ok = client._try_hf_router_fallback(
            failed_endpoint=PROVIDER_CONFIG["huggingface"]["endpoint"]
        )

        assert ok is False
        assert client._custom_endpoint is None

    @patch("cat_stack._providers._detect_huggingface_endpoint")
    def test_uses_cached_endpoint_if_different_from_failed(self, mock_detect):
        """If a previous fallback already cached an endpoint that's not the
        one that just failed, return True without re-probing."""
        client = UnifiedLLMClient(
            provider="huggingface",
            api_key="fake",
            model="meta-llama/Llama-3.3-70B-Instruct",
        )
        client._custom_endpoint = f"{_HF_ROUTER_ENDPOINTS['together']}/chat/completions"

        ok = client._try_hf_router_fallback(
            failed_endpoint=PROVIDER_CONFIG["huggingface"]["endpoint"]
        )

        assert ok is True
        assert not mock_detect.called, "should reuse cached endpoint, not re-probe"


class TestDetectHuggingfaceEndpointSkip:
    @patch("cat_stack._providers.requests.post")
    def test_legacy_mode_probes_only_two_endpoints(self, mock_post):
        """Without skip, preserves the pre-C10 probe count for existing
        image_functions / pdf_functions callers."""
        mock_post.return_value = MagicMock(status_code=400)
        _detect_huggingface_endpoint(api_key="fake", model="some-model")
        assert mock_post.call_count == 2

    @patch("cat_stack._providers.requests.post")
    def test_skip_mode_probes_all_routers(self, mock_post):
        """With skip set, probes generic + all five specific routers
        (minus skipped ones) — six candidates total, five after skipping one."""
        mock_post.return_value = MagicMock(status_code=400)
        _detect_huggingface_endpoint(
            api_key="fake",
            model="some-model",
            skip={PROVIDER_CONFIG["huggingface"]["endpoint"].replace("/chat/completions", "")},
        )
        # 6 candidates total - 1 skipped = 5 probes
        assert mock_post.call_count == 5

    @patch("cat_stack._providers.requests.post")
    def test_skip_mode_returns_none_when_all_fail(self, mock_post):
        mock_post.return_value = MagicMock(status_code=400)
        result = _detect_huggingface_endpoint(
            api_key="fake",
            model="some-model",
            skip={PROVIDER_CONFIG["huggingface"]["endpoint"].replace("/chat/completions", "")},
        )
        assert result is None

    @patch("cat_stack._providers.requests.post")
    def test_legacy_mode_returns_generic_when_all_fail(self, mock_post):
        """Backward-compat: image_functions / pdf_functions callers expect
        a base URL even on failure."""
        mock_post.return_value = MagicMock(status_code=400)
        result = _detect_huggingface_endpoint(api_key="fake", model="some-model")
        assert result is not None
        assert "router.huggingface.co" in result

    @patch("cat_stack._providers.requests.post")
    def test_explicit_suffix_path_returns_immediately_without_probing(self, mock_post):
        """When the model has an explicit known router suffix, return that
        endpoint directly — no probe POSTs needed."""
        result = _detect_huggingface_endpoint(
            api_key="fake", model="Qwen/Qwen3-VL-235B:novita"
        )
        assert result == _HF_ROUTER_ENDPOINTS["novita"]
        assert not mock_post.called
