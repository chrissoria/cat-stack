"""
Test for the parallel chunked-classification isolation extension to
task #28 (H-BATCH).

Same fault shape as the batch ensemble: `text_functions.py:786` had
`future.result()` with no try/except, and the worker function
(`_call_chunk`) didn't catch its own exceptions. So a single chunk
hitting a transient network glitch (or anything else outside the
`client.complete()` retry budget) aborted the entire parallel chunk
loop, losing every other chunk's work.

Post-fix: chunk exceptions are caught at the `future.result()` boundary,
a warning is logged, and the loop continues. The chunks that already
completed are preserved.
"""

import sys

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock


def _input_series(n=6):
    """Minimal pandas Series for explore_common_categories chunked path."""
    return pd.Series([f"text-{i}" for i in range(n)])


class TestChunkedParallelIsolation:
    def test_one_chunk_exception_does_not_kill_parallel_loop(self, capsys):
        """If _call_chunk raises (e.g., network glitch), other chunks
        still finish and the function returns successfully.

        We patch `client.complete` so the second job raises but the
        first and third return real-looking replies.
        """
        from cat_stack.text_functions import explore_common_categories

        # 3 chunks total (divisions=3, iterations=1).
        # Use a Lock + dedicated "raised" flag so exactly one thread raises;
        # bare integer increments aren't thread-safe in Python.
        import threading
        lock = threading.Lock()
        state = {"calls": 0, "raised_once": False}

        def fake_complete(messages, **kwargs):
            with lock:
                state["calls"] += 1
                if not state["raised_once"]:
                    state["raised_once"] = True
                    raise ConnectionError("simulated network glitch on chunk 1")
            return ("category A\ncategory B\n", None)

        with patch("cat_stack.text_functions.UnifiedLLMClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.complete.side_effect = fake_complete
            mock_client_cls.return_value = mock_client

            result = explore_common_categories(
                input_data=_input_series(n=6),
                api_key="fake-key",
                model="gpt-4o",
                provider="openai",
                divisions=3,
                iterations=1,
                max_workers=3,  # parallel mode
                return_raw=True,  # skip merge step (only test parallel loop)
            )

        # The function returned (didn't propagate the ConnectionError)
        assert result is not None

        # The warning was printed to stderr — confirms the catch path fired
        captured = capsys.readouterr()
        # explore_common_categories writes warnings via sys.stderr.write
        assert "ConnectionError" in captured.err or "Skipping" in captured.err, (
            f"expected catch-path warning in stderr; got: {captured.err}"
        )

        # At least 2 chunks reached fake_complete — proves the loop
        # didn't abort on the first exception. (One thread raised;
        # at least one OTHER thread successfully ran. Without the fix,
        # the second + third would have been killed by future.result()
        # re-raising on the failing one.)
        assert state["calls"] >= 2, f"loop aborted early; only {state['calls']} chunks ran"
        assert state["raised_once"], "expected the simulated exception to fire"

    def test_pre_fix_regression(self):
        """Sanity check: with all chunks succeeding, parallel mode
        produces a non-empty result (proves we didn't break the
        happy path)."""
        from cat_stack.text_functions import explore_common_categories

        with patch("cat_stack.text_functions.UnifiedLLMClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.complete.return_value = ("good category\nanother category\n", None)
            mock_client_cls.return_value = mock_client

            result = explore_common_categories(
                input_data=_input_series(n=4),
                api_key="fake-key",
                model="gpt-4o",
                provider="openai",
                divisions=2,
                iterations=1,
                max_workers=2,
                return_raw=True,
            )

        assert result is not None
