"""
Tests for H-PROV: consolidate + tighten provider detection.

Pre-fix:
  - Two near-duplicate functions (detect_provider, _detect_model_source)
    with divergent substring rules (5/20 disagreements on the smoke probe).
  - Bare-substring matching produced false-positive routings:
    `qwen-o3-coder` → openai (because "o3" matched before "qwen"),
    `llama3.1:8b` → huggingface (because "llama" matched before any
    Ollama-shape check).
  - Ollama-style `name:tag` syntax was auto-routed to whichever family
    pattern matched the prefix.

Post-fix:
  - Anchored regex patterns with word boundaries replace bare substrings.
  - `org/model` format → huggingface explicitly (catches HF-hosted
    Mistral, Qwen, Llama with org prefix).
  - `name:tag` (no slash) raises ValueError suggesting
    provider='local' or provider='ollama' — auto-detection is
    intentionally disabled for Ollama because the failure mode (connection
    refused on port 11434) is confusing for users who meant a hosted model.
  - `_detect_model_source` is a thin shim over detect_provider so both
    paths route identically.
  - `provider="local"` is a friendlier alias for `provider="ollama"`.
"""

import pytest

from cat_stack._providers import (
    detect_provider,
    _detect_model_source,
    _normalize_provider,
)


class TestExplicitProviderPassThrough:
    def test_explicit_provider_returned_lowercased(self):
        assert detect_provider("gpt-4o", provider="OpenAI") == "openai"
        assert detect_provider("anything", provider="ANTHROPIC") == "anthropic"


class TestLocalAlias:
    def test_local_normalizes_to_ollama(self):
        assert _normalize_provider("local") == "ollama"
        assert _normalize_provider("LOCAL") == "ollama"
        assert _normalize_provider("Local") == "ollama"

    def test_local_works_through_detect_provider(self):
        assert detect_provider("llama3.1:8b", provider="local") == "ollama"
        assert detect_provider("any-name", provider="LOCAL") == "ollama"

    def test_ollama_still_works(self):
        assert detect_provider("llama3.1:8b", provider="ollama") == "ollama"
        assert _normalize_provider("ollama") == "ollama"

    def test_other_providers_pass_through_unchanged(self):
        for name in ("openai", "anthropic", "google", "mistral", "perplexity",
                     "xai", "huggingface", "huggingface-together"):
            assert _normalize_provider(name) == name

    def test_normalize_preserves_falsy(self):
        assert _normalize_provider(None) is None
        assert _normalize_provider("") == ""


class TestPatternMatching:
    """Word-boundary anchored matching — first match wins."""

    def test_openai_o_series_models(self):
        assert detect_provider("o1-preview") == "openai"
        assert detect_provider("o3-mini") == "openai"
        assert detect_provider("o5-pro") == "openai"

    def test_openai_gpt_models(self):
        assert detect_provider("gpt-4o") == "openai"
        assert detect_provider("gpt-4o-mini") == "openai"
        assert detect_provider("gpt-5") == "openai"
        assert detect_provider("gpt-3.5-turbo") == "openai"

    def test_anthropic_claude(self):
        assert detect_provider("claude-haiku-4-5-20251001") == "anthropic"
        assert detect_provider("claude-sonnet-4-5-20250929") == "anthropic"
        assert detect_provider("claude") == "anthropic"

    def test_google_gemini_and_gemma(self):
        assert detect_provider("gemini-2.5-flash") == "google"
        assert detect_provider("gemini-2.0-pro") == "google"
        assert detect_provider("gemma-3") == "google"

    def test_mistral(self):
        assert detect_provider("mistral-large-latest") == "mistral"
        assert detect_provider("mistral-7b") == "mistral"
        assert detect_provider("open-mistral-nemo") == "mistral"
        assert detect_provider("mixtral-8x7b") == "mistral"

    def test_xai_grok(self):
        assert detect_provider("grok-2") == "xai"
        assert detect_provider("grok-4") == "xai"

    def test_perplexity(self):
        assert detect_provider("sonar") == "perplexity"
        assert detect_provider("sonar-pro") == "perplexity"
        assert detect_provider("pplx-7b") == "perplexity"

    def test_huggingface_family_names_without_org(self):
        assert detect_provider("llama-3.3-70b") == "huggingface"
        assert detect_provider("deepseek-r1") == "huggingface"
        assert detect_provider("qwen2.5-coder") == "huggingface"


class TestHuggingFaceOrgFormat:
    """org/model format → huggingface, unconditionally."""

    def test_org_slash_model(self):
        assert detect_provider("meta-llama/Llama-3.3-70B-Instruct") == "huggingface"

    def test_org_slash_model_with_router_suffix(self):
        assert detect_provider("Qwen/Qwen3-VL-235B:novita") == "huggingface"

    def test_hf_hosted_mistral_routes_to_hf_not_mistral_ai(self):
        """Regression: pre-fix substring matching caught 'mistral' and
        routed mistralai/Mistral-7B-v0.1 to Mistral.ai instead of HF."""
        assert detect_provider("mistralai/Mistral-7B-v0.1") == "huggingface"


class TestOllamaShapeRequiresExplicit:
    def test_name_colon_tag_raises_value_error(self):
        with pytest.raises(ValueError, match="Ollama"):
            detect_provider("llama3.1:8b")

    def test_error_mentions_both_local_and_ollama(self):
        with pytest.raises(ValueError) as exc:
            detect_provider("qwen2.5-coder:7b")
        msg = str(exc.value)
        assert "local" in msg
        assert "ollama" in msg

    def test_explicit_local_works(self):
        assert detect_provider("llama3.1:8b", provider="local") == "ollama"

    def test_explicit_ollama_works(self):
        assert detect_provider("llama3.1:8b", provider="ollama") == "ollama"


class TestSubstringCollisionsFixed:
    """Cases the bare-substring matcher got wrong."""

    def test_qwen_o3_coder_routes_to_huggingface_not_openai(self):
        """Pre-fix: 'o3' substring matched before 'qwen'."""
        assert detect_provider("qwen-o3-coder") == "huggingface"

    def test_deepseek_grok_merge_routes_to_huggingface(self):
        """Pre-fix: detect_provider matched 'grok' first, _detect_model_source
        matched 'deepseek' first — divergent."""
        assert detect_provider("deepseek-grok-merge") == "huggingface"

    def test_perplexity_in_model_name_routes_to_strongest_family_token(self):
        """`perplexity-llama-finetune` tokenizes to ['perplexity', 'llama',
        'finetune']. 'perplexity' isn't a family prefix; 'llama' is, so the
        model routes to HuggingFace deliberately — model-family signals
        beat container-name signals."""
        assert detect_provider("perplexity-llama-finetune") == "huggingface"

    def test_meta_keyword_no_longer_false_positives(self):
        """Pre-fix: 'meta' substring matched any model with letters m-e-t-a,
        could route 'meta-claude-experiment' to anthropic (claude wins) but
        'meta-anything-else' to huggingface. New rules: no bare 'meta'
        substring."""
        # Even meta-claude routes by the strongest signal (claude pattern)
        assert detect_provider("meta-claude-experiment") == "anthropic"


class TestDetectModelSourceDelegation:
    """Back-compat shim. Same input → same output as detect_provider."""

    def test_delegates_to_detect_provider(self):
        for name in ("gpt-4o", "claude-haiku-4-5-20251001", "gemini-2.5-flash",
                     "mistral-large-latest", "grok-2", "sonar-pro"):
            assert _detect_model_source(name, "auto") == detect_provider(name)

    def test_o_series_no_longer_diverges(self):
        """Pre-fix: detect_provider returned 'openai' but _detect_model_source
        raised ValueError. Now both return 'openai'."""
        assert _detect_model_source("o1-preview", "auto") == "openai"
        assert _detect_model_source("o3-mini", "auto") == "openai"

    def test_claude_code_passthrough_preserved(self):
        """The claude-code special-case (CLI subprocess) must still pass
        through _detect_model_source even though detect_provider would
        route it via the claude regex."""
        assert _detect_model_source("anything", "claude-code") == "claude-code"

    def test_explicit_provider_passes_through(self):
        assert _detect_model_source("any-model", "openai") == "openai"
        assert _detect_model_source("any-model", "local") == "ollama"


class TestUnclassifiableModels:
    def test_unknown_model_raises_value_error(self):
        with pytest.raises(ValueError, match="auto-detect"):
            detect_provider("totally-unknown-vendor-x")

    def test_error_message_lists_supported_providers(self):
        with pytest.raises(ValueError) as exc:
            detect_provider("mystery-model")
        msg = str(exc.value).lower()
        # Should hint at every legitimate provider
        for prov in ("openai", "anthropic", "google", "mistral", "perplexity",
                     "xai", "huggingface", "ollama"):
            assert prov in msg
