"""Dispatch tests for the cat-claws agent backends (claude-agent + codex-agent).

Mocked — cat-claws is patched so no live agent is needed. One parameterized
suite over both providers (cases moved from test_claude_agent_dispatch.py,
not copy-pasted): PROVIDER_CONFIG presence, detection, dispatch + message
flattening + adapter-name routing, adapter-error surfacing, and polite
degradation when cat-claws is not installed. Plus image and PDF-page
classification routing for both backends. All behavior is gated on the provider value, so these tests touch no
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


class _CapturingAdapter:
    def __init__(self):
        self.calls = []

    async def one_shot(self, prompt, system_prompt, model, thinking_budget=0, **kw):
        self.calls.append(dict(prompt=prompt, model=model, **kw))
        return '{"1": "1"}', None


@pytest.fixture
def png_file(tmp_path):
    from PIL import Image
    path = tmp_path / "img.png"
    Image.new("RGB", (8, 8), "red").save(path)
    return str(path)


@pytest.fixture
def pdf_file(tmp_path):
    import fitz  # PyMuPDF
    path = tmp_path / "doc.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "hello")
    doc.save(path)
    return str(path)


@pytest.mark.parametrize("provider", sorted(SPECS))
class TestAgentMultimodalRouting:
    """Image and PDF-page classification reach the provider's OWN cat-claws
    adapter with the image attached (codex-agent used to be refused with a
    "not yet supported" guard; cat-claws 0.3.2 takes images on both)."""

    def _patched(self, provider):
        adapter_name, _, _ = SPECS[provider]
        adapter, names = _CapturingAdapter(), []

        def fake_get_adapter(name):
            names.append(name)
            return adapter

        return adapter, names, adapter_name, patch(
            "catclaws._adapters.get_adapter", side_effect=fake_get_adapter)

    def test_image_classification(self, provider, png_file):
        from catstack.image_functions import image_multi_class

        _, model, _ = SPECS[provider]
        adapter, names, adapter_name, p = self._patched(provider)
        with p:
            image_multi_class("a drawing", [png_file], ["Red", "Blue"], api_key="",
                              user_model=model, model_source=provider)
        assert names and set(names) == {adapter_name}
        images = adapter.calls[0]["images"]
        assert len(images) == 1 and images[0]["media_type"] == "image/png"

    def test_pdf_page_classification(self, provider, pdf_file):
        from catstack.pdf_functions import pdf_multi_class

        _, model, _ = SPECS[provider]
        adapter, names, adapter_name, p = self._patched(provider)
        with p:
            pdf_multi_class("a form", [pdf_file], ["A", "B"], api_key="",
                            user_model=model, model_source=provider)
        assert set(names) == {adapter_name}
        assert adapter.calls[0]["images"][0]["media_type"] == "image/png"

    def test_pdf_text_mode_sends_no_image(self, provider, pdf_file):
        from catstack.pdf_functions import pdf_multi_class

        _, model, _ = SPECS[provider]
        adapter, _, _, p = self._patched(provider)
        with p:
            pdf_multi_class("a form", [pdf_file], ["A", "B"], api_key="",
                            user_model=model, model_source=provider, mode="text")
        assert "images" not in adapter.calls[0]
