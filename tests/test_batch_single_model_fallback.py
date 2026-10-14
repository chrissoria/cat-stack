"""
batch_mode=True with a single model on a provider that has no batch API
(cat-stack 2.6.2).

Before 2.6.2 this raised "batch_mode=True is not supported for provider
'huggingface'", while the same provider inside a multi-model batch run fell
back to synchronous calls. A caller that keeps one call site and only swaps
the model therefore could not leave batch_mode on. Now the single-model path
falls back the same way, with a printed notice, and batch-capable providers
are unaffected.
"""

from unittest.mock import patch

import pandas as pd
import pytest

import importlib
C = importlib.import_module("catstack.classify")  # the module, not the re-exported function
from catstack._batch import UNSUPPORTED_BATCH_PROVIDERS


CATS = ["Work", "Family"]


def _fake_frame(*a, **k):
    return pd.DataFrame({"input_data": ["x"], "processing_status": ["success"],
                         "category_1": [0], "category_2": [1]})


@pytest.mark.parametrize("provider,model", [
    ("huggingface", "Qwen/Qwen3.8-27B:deepinfra"),
    ("huggingface", "moonshotai/Kimi-Linear-48B-A3B-Instruct:featherless-ai"),
    ("ollama", "llama3.2"),
    ("perplexity", "sonar"),
])
def test_unsupported_provider_falls_back_to_sync(provider, model, capsys):
    assert provider in UNSUPPORTED_BATCH_PROVIDERS
    with patch.object(C, "classify_ensemble", side_effect=_fake_frame) as sync, \
         patch("catstack._batch.run_batch_classify") as batch:
        out = C.classify(["a response"], CATS, models=[(model, provider, "key")],
                         batch_mode=True, add_other=False, check_verbosity=False)
    assert sync.called and not batch.called
    assert list(out["processing_status"]) == ["success"]
    assert "has no batch API" in capsys.readouterr().out


@pytest.mark.parametrize("provider,model", [
    ("anthropic", "claude-haiku-5-5"),
    ("openai", "gpt-6-luna"),
    ("google", "gemini-3.5-flash-lite"),
])
def test_batch_provider_still_uses_batch(provider, model, capsys):
    with patch.object(C, "classify_ensemble", side_effect=_fake_frame) as sync, \
         patch("catstack._batch.run_batch_classify", side_effect=_fake_frame) as batch:
        C.classify(["a response"], CATS, models=[(model, provider, "key")],
                   batch_mode=True, add_other=False, check_verbosity=False)
    assert batch.called and not sync.called
    assert "has no batch API" not in capsys.readouterr().out


def test_provider_resolution():
    r = C._resolve_model_provider
    assert r(("Qwen/Qwen3.8-27B:deepinfra", "huggingface", "k")) == "huggingface"
    assert r({"model": "claude-haiku-5-5", "provider": "anthropic"}) == "anthropic"
    assert r(("", None)) is None
