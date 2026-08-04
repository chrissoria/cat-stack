"""api_key must be optional so subscription/CLI backends work without one.

Regression test for the live failure on 2026-08-04:
    explore(..., model_source="claude-agent")
    -> TypeError: explore() missing 1 required positional argument: 'api_key'
The provider layer already validates that HTTP providers get a key
(text_functions raises when api_key is falsy for non-subscription sources);
the wrappers just must not require it positionally.
"""
import inspect
from unittest.mock import patch

import catstack


def test_signatures_default_api_key_to_none():
    for fn in (catstack.explore, catstack.extract):
        assert inspect.signature(fn).parameters["api_key"].default is None


@patch("catstack.explore.explore_common_categories", return_value=["a", "b"])
def test_explore_subscription_backend_needs_no_key(mock_explore):
    out = catstack.explore(["text one", "text two"], description="q",
                           model_source="claude-agent")
    assert out == ["a", "b"]
    assert mock_explore.call_args.kwargs["api_key"] is None
