"""
Tests for the cross-provider thinking_budget -> effort mapping.

`thinking_budget` is a single user-facing token-count knob. Providers whose API
takes a literal token budget (Google, older Anthropic) get it verbatim; the
providers whose API takes an effort ENUM used to collapse every positive budget
to "high", so the same budget behaved very differently depending on provider.

The shared `_thinking_budget_to_effort` table now grades a positive budget into
low / medium / high (<=2048 low, <=8192 medium, else high), and every
effort-enum provider consults it, so the same thinking_budget produces
comparable reasoning intensity everywhere:

  * Anthropic adaptive (Opus 4.7+, Sonnet 5, Fable 5) -> output_config.effort
  * OpenAI reasoning models                           -> reasoning_effort
  * xAI hybrid grok                                    -> reasoning_effort
  * Ollama gpt-oss (enum family)                       -> think

Bool-only families (Ollama qwen3/deepseek, HF Qwen3) can only toggle on/off and
keep doing so. `thinking_budget=0` is "off" and represented per-provider.
"""

import pytest

from cat_stack._providers import (
    UnifiedLLMClient,
    _thinking_budget_to_effort,
    _ollama_think_value,
)


def _client(model, provider):
    return UnifiedLLMClient(model=model, provider=provider, api_key="fake")


class TestHelperThresholds:
    @pytest.mark.parametrize(
        "budget,expected",
        [
            (1, "low"),
            (2048, "low"),
            (2049, "medium"),
            (8192, "medium"),
            (8193, "high"),
            (100000, "high"),
        ],
    )
    def test_tiers(self, budget, expected):
        assert _thinking_budget_to_effort(budget) == expected

    def test_never_emits_xhigh_or_max(self):
        # Capped at "high" so every effort-enum provider accepts the value.
        assert _thinking_budget_to_effort(10**9) == "high"


class TestAnthropicAdaptiveEffort:
    @pytest.mark.parametrize(
        "budget,tier", [(1000, "low"), (5000, "medium"), (12000, "high")]
    )
    def test_effort_graded_from_budget(self, budget, tier):
        c = _client("claude-opus-4-8", "anthropic")
        p = c._build_anthropic_payload(
            [{"role": "user", "content": "hi"}], thinking_budget=budget
        )
        assert p["thinking"] == {"type": "adaptive"}
        assert p["output_config"] == {"effort": tier}

    def test_off_sends_no_output_config(self):
        c = _client("claude-opus-4-8", "anthropic")
        p = c._build_anthropic_payload(
            [{"role": "user", "content": "hi"}], thinking_budget=0
        )
        assert "thinking" not in p
        assert "output_config" not in p


class TestOpenAIReasoningEffort:
    @pytest.mark.parametrize(
        "budget,tier", [(1000, "low"), (5000, "medium"), (12000, "high")]
    )
    def test_graded(self, budget, tier):
        c = _client("o3", "openai")
        p = c._build_openai_payload(
            [{"role": "user", "content": "hi"}], thinking_budget=budget
        )
        assert p["reasoning_effort"] == tier

    def test_zero_uses_floor_not_a_tier(self):
        # budget=0 keeps the per-model off-floor ("none"/"minimal"), not "low".
        c = _client("gpt-5", "openai")
        p = c._build_openai_payload(
            [{"role": "user", "content": "hi"}], thinking_budget=0
        )
        assert p["reasoning_effort"] in ("none", "minimal")


class TestXaiReasoningEffort:
    @pytest.mark.parametrize(
        "budget,tier", [(1000, "low"), (5000, "medium"), (12000, "high")]
    )
    def test_graded(self, budget, tier):
        c = _client("grok-4.3", "xai")
        p = c._build_openai_payload(
            [{"role": "user", "content": "hi"}], thinking_budget=budget
        )
        assert p["reasoning_effort"] == tier

    def test_zero_is_low_floor(self):
        c = _client("grok-4.3", "xai")
        p = c._build_openai_payload(
            [{"role": "user", "content": "hi"}], thinking_budget=0
        )
        assert p["reasoning_effort"] == "low"


class TestOllamaThinkValue:
    @pytest.mark.parametrize(
        "budget,tier", [(1000, "low"), (5000, "medium"), (12000, "high")]
    )
    def test_gpt_oss_enum_graded(self, budget, tier):
        assert _ollama_think_value("gpt-oss:20b", budget) == tier

    def test_gpt_oss_off(self):
        assert _ollama_think_value("gpt-oss:20b", 0) == "low"

    @pytest.mark.parametrize("budget", [1000, 5000, 12000])
    def test_bool_family_on_off_only(self, budget):
        # qwen3 can't grade — any positive budget is just "on".
        assert _ollama_think_value("qwen3:8b", budget) is True
        assert _ollama_think_value("qwen3:8b", 0) is False


class TestCrossProviderParity:
    """The same thinking_budget yields the same tier on every enum provider."""

    @pytest.mark.parametrize(
        "budget,tier", [(1500, "low"), (4096, "medium"), (20000, "high")]
    )
    def test_same_tier_everywhere(self, budget, tier):
        anth = _client("claude-sonnet-5", "anthropic")._build_anthropic_payload(
            [{"role": "user", "content": "x"}], thinking_budget=budget
        )["output_config"]["effort"]
        openai = _client("o4-mini", "openai")._build_openai_payload(
            [{"role": "user", "content": "x"}], thinking_budget=budget
        )["reasoning_effort"]
        xai = _client("grok-4.3", "xai")._build_openai_payload(
            [{"role": "user", "content": "x"}], thinking_budget=budget
        )["reasoning_effort"]
        ollama = _ollama_think_value("gpt-oss:120b", budget)
        assert anth == openai == xai == ollama == tier
