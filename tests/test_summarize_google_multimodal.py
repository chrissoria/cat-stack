"""
Tests for the _call_google_multimodal tuple unpacking in summarize_ensemble.

Regression: both Google branches in summarize_ensemble's per-item closure
did `response = _call_google_multimodal(...)` and then `if error:` — but
_call_google_multimodal returns (reply, error), so `error` was undefined
and `summarize_single_item` raised UnboundLocalError, caught by the
surrounding `except Exception:` and turned into a silent failure for
every Google PDF/image summary.

After the fix, both sites destructure `response, error = ...`.
"""

import inspect

from cat_stack.text_functions_ensemble import summarize_ensemble


def test_no_remaining_bug_pattern_in_summarize_ensemble():
    """No call site in summarize_ensemble should drop the error half of the
    _call_google_multimodal tuple."""
    src = inspect.getsource(summarize_ensemble)

    bug_pattern = "response = _call_google_multimodal("
    fixed_pattern = "response, error = _call_google_multimodal("

    assert bug_pattern not in src, (
        f"Found broken call site that drops the error half of the tuple. "
        f"_call_google_multimodal returns (reply, error); the caller must "
        f"destructure both."
    )
    assert src.count(fixed_pattern) == 2, (
        f"Expected exactly 2 fixed call sites (one PDF, one image), "
        f"found {src.count(fixed_pattern)}."
    )


def test_call_google_multimodal_returns_two_tuple():
    """Sanity check on the helper's return contract — if this changes,
    callers across the codebase need updating."""
    from cat_stack.text_functions_ensemble import _call_google_multimodal

    src = inspect.getsource(_call_google_multimodal)
    # Every return statement should produce a 2-tuple (reply, error).
    # The function uses tuple returns like `return reply, None` /
    # `return None, "..."` — quick grep on `return ` lines.
    return_lines = [
        line.strip()
        for line in src.splitlines()
        if line.strip().startswith("return ") and "None" in line or line.strip().startswith("return reply")
    ]
    assert return_lines, "Expected at least one return statement"
    for line in return_lines:
        # Each return is either `return X, Y` (2-tuple) or `return reply, None`.
        # Reject any single-value returns.
        assert "," in line, (
            f"_call_google_multimodal has a single-value return: {line!r}. "
            f"All returns must be (reply, error) 2-tuples."
        )
