"""Qwen3.8 Max (2.4T-A95B) rejects enable_thinking on its HF host: send no
switch, and record it as reasoning by default. Other Qwen3 models keep the
off switch."""

from cat_stack._providers import (
    _hf_model_needs_enable_thinking_off,
    _hf_model_reasons_by_default,
)


def test_qwen38_max_gets_no_switch():
    for m in ("Qwen/Qwen3.8-2.4T-A95B", "Qwen/Qwen3.8-2.4T-A95B:novita"):
        assert not _hf_model_needs_enable_thinking_off(m)
        assert _hf_model_reasons_by_default(m)


def test_other_qwen3_keep_the_off_switch():
    for m in ("Qwen/Qwen3.8-27B:deepinfra", "Qwen/Qwen3.8-Flash-Next:featherless-ai",
              "Qwen/Qwen3.5-397B-A17B"):
        assert _hf_model_needs_enable_thinking_off(m)
        assert not _hf_model_reasons_by_default(m)
