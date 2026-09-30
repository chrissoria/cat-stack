"""
collapse_themes(thinking_budget=...): the reasoning depth passed to every model call a collapse makes.

Default "auto": low effort for the models that think whether asked or not (the adaptive-thinking Anthropic models,
which otherwise run at high effort and made a long collapse slow and costly), and every other model left at its own
default, so reasoning is never switched ON for a model where it is optional. None restores each model's default for
every model; an int is sent as given.
"""

from unittest.mock import MagicMock, patch

import pytest

from catstack import collapse_themes
from catstack.collapse_themes import _AUTO_LOW_BUDGET, _resolve_thinking


def _numbered(messages, **kwargs):
    """Echo back the batch's labels as a numbered list (a no-op merge)."""
    content = messages[0]["content"]
    blob = content.split("```")[1]
    labels = [x.strip() for x in blob.split(";") if x.strip()]
    return "\n".join(f"{i + 1}. {lab.split(' (')[0]}" for i, lab in enumerate(labels)), None


def _run(provider, model, **kw):
    inst = MagicMock()
    inst.complete.side_effect = _numbered
    with patch("catstack.collapse_themes.detect_provider", return_value=provider), \
         patch("catstack.collapse_themes.UnifiedLLMClient", return_value=inst):
        collapse_themes([f"theme {i}" for i in range(12)] * 2, api_key="k", user_model=model, passes=1,
                        batch_size=40, embedding_merge_threshold=None, dedupe_threshold=1.0,
                        final_consolidation=False, shuffle=False, **kw)
    return [c.kwargs for c in inst.complete.call_args_list]


class TestResolve:
    @pytest.mark.parametrize("model", ["claude-sonnet-5", "claude-opus-4-8", "claude-fable-5"])
    def test_auto_is_low_for_always_thinking_models(self, model):
        assert _resolve_thinking("auto", "anthropic", model) == _AUTO_LOW_BUDGET

    @pytest.mark.parametrize("provider,model", [("anthropic", "claude-sonnet-4-6"), ("openai", "gpt-4o"),
                                                ("huggingface", "Qwen/Qwen3-32B"), ("google", "gemini-2.5-flash"),
                                                ("ollama", "qwen3:8b")])
    def test_auto_leaves_other_models_at_their_default(self, provider, model):
        assert _resolve_thinking("auto", provider, model) is None

    def test_explicit_values_pass_through(self):
        assert _resolve_thinking(None, "anthropic", "claude-sonnet-5") is None
        assert _resolve_thinking(8000, "openai", "gpt-4o") == 8000
        assert _resolve_thinking(0, "anthropic", "claude-sonnet-5") == 0


class TestEveryCallGetsIt:
    def test_default_sonnet_calls_run_at_low_effort(self):
        calls = _run("anthropic", "claude-sonnet-5")
        assert calls and all(c.get("thinking_budget") == _AUTO_LOW_BUDGET for c in calls)

    def test_default_openai_calls_untouched(self):
        calls = _run("openai", "gpt-4o")
        assert calls and all("thinking_budget" not in c for c in calls)

    def test_none_restores_model_default(self):
        calls = _run("anthropic", "claude-sonnet-5", thinking_budget=None)
        assert calls and all("thinking_budget" not in c for c in calls)

    def test_explicit_budget_on_every_call_including_top_n(self):
        calls = _run("openai", "gpt-4o", thinking_budget=4000, top_n=3)
        assert len(calls) >= 2                      # the pass and the top_n step
        assert all(c.get("thinking_budget") == 4000 for c in calls)
        assert any("EXACTLY 3" in c["messages"][0]["content"] for c in calls)
