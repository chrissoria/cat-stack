"""Dispatch tests for model_source='claude-agent' (the cat-claws backend).

Mocked — cat-claws is patched so no live agent CLI is needed. Verifies cat-stack
routes the new provider to the adapter, flattens messages the same way the
claude-code CLI path does, surfaces adapter errors, and degrades politely when
cat-claws is not installed. All behavior is gated on the new provider value, so
these tests touch no existing provider path.
"""
import sys
from unittest.mock import patch

from catstack._providers import (
    PROVIDER_CONFIG,
    UnifiedLLMClient,
    _detect_model_source,
    detect_provider,
)


def _client():
    return UnifiedLLMClient(provider="claude-agent", api_key="", model="claude-sonnet-5")


def test_provider_config_has_claude_agent():
    assert "claude-agent" in PROVIDER_CONFIG
    assert PROVIDER_CONFIG["claude-agent"]["endpoint"] is None


def test_detection_recognizes_claude_agent():
    assert detect_provider("claude-sonnet-5", provider="claude-agent") == "claude-agent"
    assert _detect_model_source("claude-sonnet-5", "claude-agent") == "claude-agent"


def test_dispatch_routes_to_adapter_and_flattens_messages():
    captured = {}

    class FakeAdapter:
        async def one_shot(self, prompt, system_prompt, model, thinking_budget=0):
            captured.update(prompt=prompt, system_prompt=system_prompt,
                            model=model, thinking_budget=thinking_budget)
            return '{"1": "1"}', None

    with patch("catclaws._adapters.get_adapter", return_value=FakeAdapter()):
        text, err = _client().complete(
            messages=[
                {"role": "system", "content": "sys A"},
                {"role": "user", "content": "user B"},
            ],
            thinking_budget=0,
        )
    assert (text, err) == ('{"1": "1"}', None)
    # Message flattening mirrors _call_claude_cli (system vs user split).
    assert captured["system_prompt"] == "sys A"
    assert captured["prompt"] == "user B"
    assert captured["model"] == "claude-sonnet-5"


def test_dispatch_surfaces_adapter_error():
    class FakeAdapter:
        async def one_shot(self, prompt, system_prompt, model, thinking_budget=0):
            return None, "rate-limited: five_hour limit reached"

    with patch("catclaws._adapters.get_adapter", return_value=FakeAdapter()):
        text, err = _client().complete(messages=[{"role": "user", "content": "x"}])
    assert text is None and "rate-limited" in err


def test_missing_cat_agent_degrades_politely():
    # Simulate cat-claws not installed: importing catclaws._adapters raises.
    with patch.dict(sys.modules, {"catclaws._adapters": None}):
        text, err = _client().complete(messages=[{"role": "user", "content": "x"}])
    assert text is None
    assert "pip install cat-stack[agent]" in err  # hint, not a raw traceback
