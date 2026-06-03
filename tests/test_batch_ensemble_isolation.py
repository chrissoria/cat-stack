"""
Tests for task #28 (H-BATCH): per-model failure isolation in
run_batch_ensemble_classify and run_batch_ensemble_summarize.

Pre-fix behavior: the ThreadPoolExecutor loop did `future.result()` with
no try/except. Any exception from ONE model's batch job (BatchJobFailedError,
BatchJobExpiredError, TimeoutError, RuntimeError, RequestException, etc.)
propagated out of the loop, killing the entire ensemble run — even when
other models had already completed successfully.

Post-fix: each future is wrapped; failures are logged, the model's result
is recorded as an empty dict ({}), and the loop continues. The downstream
code's `.get(idx, (None, "Missing from batch results"))` pattern handles
the empty dict cleanly.
"""

from unittest.mock import patch, MagicMock

import pytest

from cat_stack._batch import (
    BatchJobFailedError,
    run_batch_ensemble_classify,
    run_batch_ensemble_summarize,
)


def _make_cfg(model: str, provider: str = "anthropic"):
    """Minimal model_config dict matching what prepare_model_configs() returns."""
    return {
        "model": model,
        "provider": provider,
        "api_key": "fake-key",
        "sanitized_name": model.replace("-", "_").replace(".", "_"),
    }


def _prompt_params(categories):
    """Minimal prompt_params dict — used by _run_one_batch_job but we'll be
    mocking that function out, so values don't matter except to satisfy
    the dispatch path."""
    return {
        "categories_str": "\n".join(f"{i+1}. {c}" for i, c in enumerate(categories)),
        "json_schema": {"type": "object"},
        "creativity": 0,
        "multi_label": True,
    }


class TestEnsembleClassifyIsolation:
    """run_batch_ensemble_classify should not abort when one model fails."""

    def test_one_model_failure_does_not_kill_ensemble(self, capsys):
        """Model A's batch raises BatchJobFailedError; model B succeeds.
        The ensemble should return a DataFrame containing both columns
        — model A's empty, model B's filled."""
        categories = ["yes", "no"]
        items = ["item-1", "item-2"]
        cfg_a = _make_cfg("claude-bad", "anthropic")
        cfg_b = _make_cfg("claude-good", "anthropic")
        configs = [cfg_a, cfg_b]
        pp = {"claude-bad": _prompt_params(categories), "claude-good": _prompt_params(categories)}

        # Model B returns realistic per-item results
        good_results = {0: ('{"1":"1","2":"0"}', None), 1: ('{"1":"0","2":"1"}', None)}

        def fake_run_one(cfg, items, pp_dict, *args, **kwargs):
            if cfg["model"] == "claude-bad":
                raise BatchJobFailedError("simulated batch failure")
            return good_results

        with patch("cat_stack._batch._run_one_batch_job", side_effect=fake_run_one):
            df = run_batch_ensemble_classify(
                items=items,
                model_configs=configs,
                categories=categories,
                prompt_params_per_model=pp,
                consensus_threshold="majority",
                fail_strategy="partial",
            )

        # Ensemble returned a DataFrame (didn't raise) — that's the
        # primary regression check.
        assert df is not None
        assert len(df) == len(items)

        # Both model columns are present
        cols = list(df.columns)
        assert any("claude_bad" in c for c in cols), f"missing claude-bad column in {cols}"
        assert any("claude_good" in c for c in cols), f"missing claude-good column in {cols}"

        # Failure was logged
        captured = capsys.readouterr().out
        assert "claude-bad" in captured
        assert "failed" in captured.lower()
        assert "BatchJobFailedError" in captured

    def test_generic_exception_also_isolated(self):
        """Not just batch-specific exceptions — ANY exception should be
        isolated so a programming error in one model's pipeline doesn't
        kill the rest of the run."""
        categories = ["yes", "no"]
        items = ["item-1"]
        cfg_a = _make_cfg("claude-buggy", "anthropic")
        cfg_b = _make_cfg("claude-ok", "anthropic")
        configs = [cfg_a, cfg_b]
        pp = {c["model"]: _prompt_params(categories) for c in configs}

        def fake_run_one(cfg, items, pp_dict, *args, **kwargs):
            if cfg["model"] == "claude-buggy":
                raise RuntimeError("unexpected internal error")
            return {0: ('{"1":"1","2":"0"}', None)}

        with patch("cat_stack._batch._run_one_batch_job", side_effect=fake_run_one):
            df = run_batch_ensemble_classify(
                items=items,
                model_configs=configs,
                categories=categories,
                prompt_params_per_model=pp,
                consensus_threshold="majority",
                fail_strategy="partial",
            )

        assert df is not None
        assert len(df) == 1

    def test_all_models_fail_returns_empty_dataframe_not_exception(self):
        """Edge case: every model fails. We should still return a
        DataFrame (with all-empty columns), not raise."""
        categories = ["yes", "no"]
        items = ["only-item"]
        cfg_a = _make_cfg("claude-x", "anthropic")
        cfg_b = _make_cfg("claude-y", "anthropic")
        configs = [cfg_a, cfg_b]
        pp = {c["model"]: _prompt_params(categories) for c in configs}

        def fake_run_one(cfg, items, pp_dict, *args, **kwargs):
            raise BatchJobFailedError(f"{cfg['model']} died")

        with patch("cat_stack._batch._run_one_batch_job", side_effect=fake_run_one):
            df = run_batch_ensemble_classify(
                items=items,
                model_configs=configs,
                categories=categories,
                prompt_params_per_model=pp,
                consensus_threshold="majority",
                fail_strategy="partial",
            )

        assert df is not None
        assert len(df) == 1


class TestEnsembleSummarizeIsolation:
    """run_batch_ensemble_summarize should not abort when one model fails."""

    def test_one_model_failure_does_not_kill_ensemble(self, capsys):
        items = ["text-1", "text-2"]
        cfg_a = _make_cfg("claude-bad", "anthropic")
        cfg_b = _make_cfg("claude-good", "anthropic")
        configs = [cfg_a, cfg_b]
        pp = {"claude-bad": {}, "claude-good": {}}

        good_results = {
            0: ('{"summary":"This is the first summary"}', None),
            1: ('{"summary":"This is the second summary"}', None),
        }

        def fake_run_one(cfg, items, pp_dict, *args, **kwargs):
            if cfg["model"] == "claude-bad":
                raise BatchJobFailedError("simulated summarize failure")
            return good_results

        # The summarize path calls _synthesize_summaries which is heavy —
        # patch it to just return the first valid summary verbatim.
        with patch("cat_stack._batch._run_one_batch_summarize_job", side_effect=fake_run_one), \
             patch("cat_stack.text_functions_ensemble._synthesize_summaries", side_effect=lambda summaries, original_text, **kw: list(summaries.values())[0]):
            df = run_batch_ensemble_summarize(
                items=items,
                model_configs=configs,
                prompt_params_per_model=pp,
            )

        assert df is not None
        assert len(df) == 2

        # claude-bad column is empty, claude-good column has content
        assert df["summary_claude_bad"].iloc[0] == ""
        assert df["summary_claude_good"].iloc[0] != ""

        # claude-bad appears in failed_models
        assert "claude_bad" in df["failed_models"].iloc[0]

        # Failure was logged
        captured = capsys.readouterr().out
        assert "claude-bad" in captured
        assert "failed" in captured.lower()
