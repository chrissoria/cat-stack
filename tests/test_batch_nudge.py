"""
Tests for the batch-mode cost nudge in classify().

Large synchronous text runs that batch_mode=True would accept get a one-line
tip (~50% cheaper via the async batch API). The tip must only fire when
opting in would actually work — never for small runs, non-text input,
batch-incompatible options, or providers without a batch API — and must
never affect the run itself.
"""

from unittest.mock import patch

import pandas as pd
import pytest

import cat_stack


NUDGE_MARKER = "qualifies for batch_mode=True"

CATEGORIES = ["Employment", "Family", "Other"]  # includes catch-all -> no add_other prompt


def _run_classify(capsys, n_rows=600, **kwargs):
    """Run classify() with the ensemble mocked out; return captured stdout."""
    rows = [f"response {i}" for i in range(n_rows)]
    dummy = pd.DataFrame({"input_data": rows})
    defaults = dict(
        input_data=rows,
        categories=CATEGORIES,
        api_key="k",
        user_model="gpt-4o",
        model_source="openai",
        add_other=False,
        check_verbosity=False,
        json_formatter=False,
    )
    defaults.update(kwargs)
    with patch("cat_stack.classify.classify_ensemble", return_value=dummy):
        result = cat_stack.classify(**defaults)
    assert result is dummy
    return capsys.readouterr().out


class TestNudgeFires:
    def test_large_single_model_run(self, capsys):
        out = _run_classify(capsys, n_rows=600)
        assert NUDGE_MARKER in out
        assert "~600 API calls" in out

    def test_ensemble_counts_capable_models_only(self, capsys):
        out = _run_classify(
            capsys,
            n_rows=300,
            models=[
                ("gpt-4o", "openai", "k1"),
                ("claude-sonnet-5", "anthropic", "k2"),
                ("llama3.3", "ollama", None),  # no batch API — excluded
            ],
        )
        assert NUDGE_MARKER in out
        assert "~600 API calls" in out
        assert "2 batch-capable model(s)" in out

    def test_auto_provider_resolved(self, capsys):
        out = _run_classify(capsys, n_rows=600, model_source="auto")
        assert NUDGE_MARKER in out


class TestNudgeSilent:
    def test_small_run(self, capsys):
        out = _run_classify(capsys, n_rows=100)
        assert NUDGE_MARKER not in out

    def test_batchless_provider(self, capsys):
        out = _run_classify(
            capsys, n_rows=600, user_model="llama3.3", model_source="ollama",
        )
        assert NUDGE_MARKER not in out

    def test_categories_per_call_incompatible(self, capsys):
        out = _run_classify(capsys, n_rows=600, categories_per_call=2)
        assert NUDGE_MARKER not in out

    def test_cove_incompatible(self, capsys):
        out = _run_classify(capsys, n_rows=600, chain_of_verification=True)
        assert NUDGE_MARKER not in out

    def test_progress_callback_incompatible(self, capsys):
        out = _run_classify(capsys, n_rows=600, progress_callback=lambda *a: None)
        assert NUDGE_MARKER not in out

    def test_batch_mode_already_on(self, capsys):
        # No tip when the user already opted in; batch path is mocked.
        rows = [f"response {i}" for i in range(600)]
        dummy = pd.DataFrame({"input_data": rows})
        with patch("cat_stack.classify.classify_ensemble", return_value=dummy), \
             patch("cat_stack._batch.run_batch_classify", return_value=dummy):
            cat_stack.classify(
                input_data=rows,
                categories=CATEGORIES,
                api_key="k",
                user_model="gpt-4o",
                model_source="openai",
                add_other=False,
                check_verbosity=False,
                json_formatter=False,
                batch_mode=True,
            )
        assert NUDGE_MARKER not in capsys.readouterr().out


class TestNudgeNeverBreaksTheRun:
    def test_weird_input_data_is_safe(self, capsys):
        """The nudge swallows its own errors; classification proceeds."""

        class NoLen:
            def __iter__(self):
                return iter(["a", "b"])

        dummy = pd.DataFrame({"input_data": ["a", "b"]})
        with patch("cat_stack.classify.classify_ensemble", return_value=dummy):
            result = cat_stack.classify(
                input_data=NoLen(),
                categories=CATEGORIES,
                api_key="k",
                user_model="gpt-4o",
                model_source="openai",
                add_other=False,
                check_verbosity=False,
                json_formatter=False,
            )
        assert result is dummy
