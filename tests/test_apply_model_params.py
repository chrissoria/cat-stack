"""
Unit tests for apply_model_params() — the shared sampling/reasoning param
shaper in _providers.py.

Every payload builder (the central _build_*_payload methods AND the direct
builders in calls/, image_functions.py, pdf_functions.py) routes through this
one function, so its behavior is the contract for how creativity /
thinking_budget translate to provider wire params.
"""

import pytest

from cat_stack._providers import (
    apply_model_params,
    _openai_reasoning_effort_floor,
)


class TestAnthropic:
    def test_old_model_gets_temperature(self):
        p = apply_model_params({}, "anthropic", "claude-sonnet-4-6", creativity=0.3)
        assert p["temperature"] == 0.3

    @pytest.mark.parametrize(
        "model", ["claude-opus-4-7", "claude-opus-4-8", "claude-sonnet-5", "claude-fable-5"]
    )
    def test_new_model_skips_temperature(self, model):
        p = apply_model_params({}, "anthropic", model, creativity=0.3)
        assert "temperature" not in p

    def test_override_blocks_temperature_on_old_model(self):
        p = apply_model_params(
            {}, "anthropic", "claude-sonnet-4-6", creativity=0.3,
            overrides={"anthropic_temperature_unsupported": True},
        )
        assert "temperature" not in p

    def test_legacy_thinking_budget(self):
        p = apply_model_params(
            {"max_tokens": 4096}, "anthropic", "claude-sonnet-4-6",
            creativity=0.3, thinking_budget=2000,
        )
        assert p["thinking"] == {"type": "enabled", "budget_tokens": 2000}
        # Thinking forces temperature=1 on the legacy path; creativity ignored.
        assert p["temperature"] == 1
        assert "output_config" not in p

    def test_legacy_thinking_budget_floored_at_1024(self):
        p = apply_model_params(
            {"max_tokens": 4096}, "anthropic", "claude-sonnet-4-6", thinking_budget=100,
        )
        assert p["thinking"]["budget_tokens"] == 1024

    @pytest.mark.parametrize(
        "budget,effort", [(500, "low"), (2048, "low"), (5000, "medium"), (20000, "high")]
    )
    def test_adaptive_thinking_effort_tiers(self, budget, effort):
        p = apply_model_params(
            {"max_tokens": 4096}, "anthropic", "claude-sonnet-5", thinking_budget=budget,
        )
        assert p["thinking"] == {"type": "adaptive"}
        assert p["output_config"] == {"effort": effort}
        assert "temperature" not in p

    def test_adaptive_override_forces_adaptive_on_old_model(self):
        p = apply_model_params(
            {"max_tokens": 4096}, "anthropic", "claude-sonnet-4-6",
            thinking_budget=2000, overrides={"anthropic_thinking_adaptive": True},
        )
        assert p["thinking"] == {"type": "adaptive"}
        assert p["output_config"] == {"effort": "low"}

    def test_max_tokens_headroom_bump(self):
        p = apply_model_params(
            {"max_tokens": 4096}, "anthropic", "claude-sonnet-4-6", thinking_budget=8000,
        )
        assert p["max_tokens"] == 8000 + 4096

    def test_no_max_tokens_key_no_bump(self):
        # SDK-kwargs callers may not carry max_tokens in the shaped dict;
        # the shaper must not invent one.
        p = apply_model_params({}, "anthropic", "claude-sonnet-4-6", thinking_budget=8000)
        assert "max_tokens" not in p

    def test_no_params_no_keys(self):
        p = apply_model_params({}, "anthropic", "claude-sonnet-4-6")
        assert p == {}


class TestGoogle:
    def test_temperature_inside_generation_config(self):
        p = apply_model_params({}, "google", "gemini-2.5-flash", creativity=0.4)
        assert p["generationConfig"]["temperature"] == 0.4

    def test_existing_generation_config_preserved(self):
        p = apply_model_params(
            {"generationConfig": {"responseMimeType": "application/json"}},
            "google", "gemini-2.5-flash", creativity=0.4,
        )
        assert p["generationConfig"]["responseMimeType"] == "application/json"
        assert p["generationConfig"]["temperature"] == 0.4

    def test_thinking_budget_floored_at_128(self):
        p = apply_model_params({}, "google", "gemini-2.5-flash", thinking_budget=64)
        assert p["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 128}

    def test_zero_budget_sent_explicitly(self):
        p = apply_model_params({}, "google", "gemini-2.5-flash", thinking_budget=0)
        assert p["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 0}

    def test_zero_budget_uses_cached_floor(self):
        p = apply_model_params(
            {}, "google", "gemini-2.5-flash", thinking_budget=0,
            overrides={"google_thinking_floor": 128},
        )
        assert p["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 128}

    def test_no_params_no_generation_config(self):
        p = apply_model_params({}, "google", "gemini-2.5-flash")
        assert "generationConfig" not in p


class TestOpenAI:
    def test_normal_model_gets_temperature(self):
        p = apply_model_params({}, "openai", "gpt-4o", creativity=0.7)
        assert p["temperature"] == 0.7

    @pytest.mark.parametrize("model", ["o1", "o3-mini", "o4-mini", "gpt-5", "gpt-5.4"])
    def test_reasoning_model_skips_temperature(self, model):
        p = apply_model_params({}, "openai", model, creativity=0.7)
        assert "temperature" not in p

    @pytest.mark.parametrize(
        "budget,effort", [(1000, "low"), (8192, "medium"), (9000, "high")]
    )
    def test_reasoning_model_effort_tiers(self, budget, effort):
        p = apply_model_params({}, "openai", "gpt-5", thinking_budget=budget)
        assert p["reasoning_effort"] == effort

    def test_zero_budget_uses_family_floor(self):
        p = apply_model_params({}, "openai", "o3-mini", thinking_budget=0)
        assert p["reasoning_effort"] == _openai_reasoning_effort_floor("o3-mini")

    def test_zero_budget_override_wins(self):
        p = apply_model_params(
            {}, "openai", "o3-mini", thinking_budget=0,
            overrides={"reasoning_effort_override": "low"},
        )
        assert p["reasoning_effort"] == "low"

    def test_reasoning_model_no_budget_no_effort(self):
        p = apply_model_params({}, "openai", "gpt-5")
        assert "reasoning_effort" not in p


class TestXAI:
    def test_temperature_and_graded_effort(self):
        p = apply_model_params({}, "xai", "grok-4-1", creativity=0.5, thinking_budget=5000)
        assert p["temperature"] == 0.5
        assert p["reasoning_effort"] == "medium"

    def test_zero_budget_floors_at_low(self):
        p = apply_model_params({}, "xai", "grok-4-1", thinking_budget=0)
        assert p["reasoning_effort"] == "low"

    def test_non_reasoning_variant_skips_effort(self):
        p = apply_model_params(
            {}, "xai", "grok-4-1-fast-non-reasoning", creativity=0.5, thinking_budget=5000,
        )
        assert "reasoning_effort" not in p
        assert p["temperature"] == 0.5

    def test_cached_rejection_skips_effort(self):
        p = apply_model_params(
            {}, "xai", "grok-4-1", thinking_budget=5000,
            overrides={"xai_no_reasoning_effort": True},
        )
        assert "reasoning_effort" not in p

    def test_no_budget_no_effort(self):
        p = apply_model_params({}, "xai", "grok-4-1", creativity=0.5)
        assert "reasoning_effort" not in p


class TestOllama:
    @pytest.mark.parametrize(
        "budget,think", [(0, "low"), (1000, "low"), (5000, "medium"), (20000, "high")]
    )
    def test_gpt_oss_enum_grading(self, budget, think):
        p = apply_model_params({}, "ollama", "gpt-oss:20b", thinking_budget=budget)
        assert p["think"] == think

    def test_bool_family(self):
        p = apply_model_params({}, "ollama", "qwen3:8b", thinking_budget=5000)
        assert p["think"] is True
        p = apply_model_params({}, "ollama", "qwen3:8b", thinking_budget=0)
        assert p["think"] is False

    def test_non_reasoning_model_no_think_field(self):
        p = apply_model_params({}, "ollama", "llama3.3", thinking_budget=5000)
        assert "think" not in p

    def test_temperature_set(self):
        p = apply_model_params({}, "ollama", "llama3.3", creativity=0.2)
        assert p["temperature"] == 0.2


class TestHuggingFace:
    def test_temperature_set(self):
        p = apply_model_params({}, "huggingface", "meta-llama/Llama-3.3-70B", creativity=0.2)
        assert p["temperature"] == 0.2

    def test_qwen3_zero_budget_disables_thinking(self):
        p = apply_model_params({}, "huggingface", "Qwen/Qwen3-8B", thinking_budget=0)
        assert p["chat_template_kwargs"] == {"enable_thinking": False}

    def test_qwen3_positive_budget_no_kwarg(self):
        p = apply_model_params({}, "huggingface", "Qwen/Qwen3-8B", thinking_budget=4000)
        assert "chat_template_kwargs" not in p

    def test_non_qwen_no_kwarg(self):
        p = apply_model_params(
            {}, "huggingface", "meta-llama/Llama-3.3-70B", thinking_budget=0,
        )
        assert "chat_template_kwargs" not in p


class TestDefaultProviders:
    @pytest.mark.parametrize("provider", ["mistral", "perplexity", "something-new"])
    def test_plain_temperature(self, provider):
        p = apply_model_params({}, provider, "some-model", creativity=0.6)
        assert p == {"temperature": 0.6}

    def test_none_creativity_untouched(self):
        p = apply_model_params({}, "mistral", "mistral-small")
        assert p == {}


class TestInPlaceContract:
    def test_mutates_and_returns_same_dict(self):
        payload = {"model": "claude-sonnet-4-6", "max_tokens": 4096}
        out = apply_model_params(payload, "anthropic", "claude-sonnet-4-6", creativity=0.3)
        assert out is payload
        assert payload["temperature"] == 0.3
