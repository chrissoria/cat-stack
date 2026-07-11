"""Dispatch tests for the cat-claws agent backends (claude-agent + codex-agent).

Mocked — cat-claws is patched so no live agent is needed. One parameterized
suite over both providers (cases moved from test_claude_agent_dispatch.py,
not copy-pasted): PROVIDER_CONFIG presence, detection, dispatch + message
flattening + adapter-name routing, adapter-error surfacing, and polite
degradation when cat-claws is not installed. Plus the codex-agent image/PDF
guards. All behavior is gated on the provider value, so these tests touch no
existing provider path.
"""
import sys
from unittest.mock import patch

import pytest

from catstack._providers import (
    PROVIDER_CONFIG,
    _AGENT_BACKENDS,
    UnifiedLLMClient,
    _detect_model_source,
    detect_provider,
)

# provider -> (expected adapter name, example model, expected install hint)
SPECS = {
    "claude-agent": ("claude", "claude-sonnet-5", "pip install cat-stack[agent]"),
    "codex-agent": ("codex", "gpt-5.5", 'pip install "cat-stack[codex-agent]"'),
}


def _client(provider, model):
    return UnifiedLLMClient(provider=provider, api_key="", model=model)


def test_spec_table_covers_backend_table():
    assert set(SPECS) == set(_AGENT_BACKENDS)


@pytest.mark.parametrize("provider", sorted(SPECS))
class TestAgentBackendDispatch:
    def test_provider_config_entry(self, provider):
        assert provider in PROVIDER_CONFIG
        assert PROVIDER_CONFIG[provider]["endpoint"] is None

    def test_detection_recognizes_provider(self, provider):
        _, model, _ = SPECS[provider]
        assert detect_provider(model, provider=provider) == provider
        assert _detect_model_source(model, provider) == provider

    def test_dispatch_routes_to_adapter_and_flattens_messages(self, provider):
        adapter_name, model, _ = SPECS[provider]
        captured = {}

        class FakeAdapter:
            async def one_shot(self, prompt, system_prompt, model, thinking_budget=0):
                captured.update(prompt=prompt, system_prompt=system_prompt,
                                model=model, thinking_budget=thinking_budget)
                return '{"1": "1"}', None

        def fake_get_adapter(name):
            captured["adapter_name"] = name
            return FakeAdapter()

        with patch("catclaws._adapters.get_adapter", side_effect=fake_get_adapter):
            text, err = _client(provider, model).complete(
                messages=[
                    {"role": "system", "content": "sys A"},
                    {"role": "user", "content": "user B"},
                ],
                thinking_budget=0,
            )
        assert (text, err) == ('{"1": "1"}', None)
        # The provider must reach ITS adapter, not the other one.
        assert captured["adapter_name"] == adapter_name
        # Message flattening mirrors _call_claude_cli (system vs user split).
        assert captured["system_prompt"] == "sys A"
        assert captured["prompt"] == "user B"
        assert captured["model"] == model

    def test_dispatch_surfaces_adapter_error(self, provider):
        _, model, _ = SPECS[provider]

        class FakeAdapter:
            async def one_shot(self, prompt, system_prompt, model, thinking_budget=0):
                return None, "rate-limited: five_hour limit reached"

        with patch("catclaws._adapters.get_adapter", return_value=FakeAdapter()):
            text, err = _client(provider, model).complete(
                messages=[{"role": "user", "content": "x"}]
            )
        assert text is None and "rate-limited" in err

    def test_missing_cat_claws_degrades_politely(self, provider):
        _, model, hint = SPECS[provider]
        # Simulate cat-claws not installed: importing catclaws._adapters raises.
        with patch.dict(sys.modules, {"catclaws._adapters": None}):
            text, err = _client(provider, model).complete(
                messages=[{"role": "user", "content": "x"}]
            )
        assert text is None
        assert hint in err  # hint, not a raw traceback
        assert f"model_source='{provider}'" in err


class TestCodexAgentMultimodalGuards:
    """codex-agent is text-only this release: image/PDF raise a clear error
    BEFORE any file is touched (dummy paths never hit the filesystem)."""

    def test_image_guard(self):
        from catstack.image_functions import image_multi_class

        with pytest.raises(ValueError, match="codex-agent"):
            image_multi_class(
                "a drawing", ["/nonexistent/img.png"], ["A"], api_key="",
                user_model="gpt-5.5", model_source="codex-agent",
            )

    def test_pdf_guard(self):
        from catstack.pdf_functions import pdf_multi_class

        with pytest.raises(ValueError, match="codex-agent"):
            pdf_multi_class(
                "a form", ["/nonexistent/doc.pdf"], ["A"], api_key="",
                user_model="gpt-5.5", model_source="codex-agent",
            )

    def test_guard_message_points_at_claude_agent(self):
        from catstack.image_functions import image_multi_class

        with pytest.raises(ValueError, match="claude-agent"):
            image_multi_class(
                "a drawing", ["/nonexistent/img.png"], ["A"], api_key="",
                user_model="gpt-5.5", model_source="codex-agent",
            )
