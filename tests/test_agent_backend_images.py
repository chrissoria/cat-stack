"""Multimodal content through the cat-claws agent backends, and surfaced
summarize() errors.

Two regressions, both hit by summarize(input_type="image",
model_source="claude-agent"):

1. `_call_agent_backend` joined message contents as strings, so an image
   message (a list of content blocks) raised TypeError and the image never
   reached the adapter, even though the cat-claws adapters accept
   `images=[{"media_type", "data"}]`.
2. summarize() collected each row's error but never wrote it out, so the
   failure above showed only processing_status="error" with no reason.

Mocked: cat-claws is patched, no live agent is needed.
"""
import base64
import io
from unittest.mock import patch

import pytest

from catstack._providers import UnifiedLLMClient, _split_agent_content

B64 = base64.b64encode(b"fake-image-bytes").decode()


# --- _split_agent_content ----------------------------------------------------

def test_plain_string_is_text_only():
    assert _split_agent_content("hello") == ("hello", [])


@pytest.mark.parametrize("block, media_type", [
    ({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": B64}},
     "image/png"),
    ({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{B64}", "detail": "high"}},
     "image/png"),
    ({"type": "inline_data", "mime_type": "image/png", "data": B64}, "image/png"),
    # image/jpg is normalized to the MIME type the API accepts
    ({"type": "image_url", "image_url": {"url": f"data:image/jpg;base64,{B64}"}}, "image/jpeg"),
])
def test_each_provider_image_shape_is_extracted(block, media_type):
    text, images = _split_agent_content([{"type": "text", "text": "describe"}, block])
    assert text == "describe"
    assert images == [{"media_type": media_type, "data": B64}]


def test_unknown_and_remote_blocks_are_dropped_not_stringified():
    text, images = _split_agent_content([
        {"type": "text", "text": "a"},
        {"type": "image_url", "image_url": {"url": "https://example.com/x.png"}},
        {"type": "mystery", "payload": 1},
        "not-a-dict",
    ])
    assert text == "a"
    assert images == []


# --- dispatch -----------------------------------------------------------------

class CapturingAdapter:
    def __init__(self, reply='{"summary": "ok"}', error=None):
        self.calls, self.reply, self.error = [], reply, error

    async def one_shot(self, prompt, system_prompt, model, thinking_budget=0, **kw):
        self.calls.append(dict(prompt=prompt, system_prompt=system_prompt,
                               model=model, **kw))
        return (None, self.error) if self.error else (self.reply, None)


def _complete(messages, adapter):
    client = UnifiedLLMClient(provider="claude-agent", api_key="", model="claude-sonnet-5")
    with patch("catclaws._adapters.get_adapter", return_value=adapter):
        return client.complete(messages=messages)


def test_image_message_reaches_adapter_as_images():
    adapter = CapturingAdapter()
    text, err = _complete([
        {"role": "system", "content": "sys"},
        {"role": "user", "content": [
            {"type": "text", "text": "summarize this figure"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{B64}"}},
        ]},
    ], adapter)
    assert err is None
    call = adapter.calls[0]
    assert call["prompt"] == "summarize this figure"
    assert call["system_prompt"] == "sys"
    assert call["images"] == [{"media_type": "image/png", "data": B64}]


def test_text_only_call_passes_no_images_kwarg():
    # Text-only calls must be unchanged: no `images` argument at all.
    adapter = CapturingAdapter()
    _complete([{"role": "user", "content": "plain"}], adapter)
    assert "images" not in adapter.calls[0]
    assert adapter.calls[0]["prompt"] == "plain"


# --- summarize() end to end (mocked adapter) -----------------------------------

@pytest.fixture
def png_path(tmp_path):
    from PIL import Image
    path = tmp_path / "fig.png"
    Image.new("RGB", (8, 8), "white").save(path)
    return str(path)


def _summarize(png_path, adapter):
    from catstack import summarize
    with patch("catclaws._adapters.get_adapter", return_value=adapter):
        return summarize(
            input_data=[png_path], input_type="image", input_mode="visual",
            description="a test figure", user_model="claude-sonnet-5",
            model_source="claude-agent", batch_retries=0,
        )


def test_summarize_image_via_claude_agent(png_path):
    adapter = CapturingAdapter(reply='{"summary": "Key finding."}')
    df = _summarize(png_path, adapter)
    assert df.loc[0, "processing_status"] == "success"
    assert df.loc[0, "summary"] == "Key finding."
    assert df.loc[0, "error_message"] == ""
    sent = adapter.calls[0]["images"]
    assert len(sent) == 1 and sent[0]["media_type"] == "image/png"


def test_summarize_failure_reports_its_reason(png_path):
    adapter = CapturingAdapter(error="rate-limited: five_hour limit reached")
    df = _summarize(png_path, adapter)
    assert df.loc[0, "processing_status"] == "error"
    assert "rate-limited" in df.loc[0, "error_message"]


# --- sign-in preflight ------------------------------------------------------------

class _NotSignedIn(ConnectionError):
    pass


@pytest.mark.real_auth
def test_signed_out_summarize_stops_before_any_row(png_path):
    """A signed-out agent stops the run up front with cat-claws' sign-in
    instructions, instead of every row failing on "not logged in"."""
    import catclaws
    adapter = CapturingAdapter()

    def refuse(agent):
        raise _NotSignedIn(f"{agent}: not signed in -- run catclaws.login()")

    with patch.object(catclaws, "ensure_signed_in", side_effect=refuse):
        with pytest.raises(ConnectionError, match="catclaws.login"):
            _summarize(png_path, adapter)
    assert adapter.calls == []


@pytest.mark.real_auth
def test_preflight_checks_the_right_agent():
    import catclaws
    from catstack._providers import _require_agent_sign_in
    seen = []
    with patch.object(catclaws, "ensure_signed_in", side_effect=seen.append):
        _require_agent_sign_in("claude-agent")
        _require_agent_sign_in("codex-agent")
        _require_agent_sign_in("anthropic")  # not an agent backend: no check
    assert seen == ["claude", "codex"]
