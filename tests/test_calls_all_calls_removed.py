"""
Tests for C12: the dead, partially-broken `calls/all_calls.py` is removed.

Before: 621 lines of code that duplicated every function in stepback.py +
CoVe.py + the top_n helpers. Two of the CoVe variants there
(`chain_of_verification_anthropic`, `chain_of_verification_google`) raised
NameError when called (undefined `properties`, missing `import json`,
undefined `thinking_budget`). Nothing in src/ called them; nothing in the
5 sibling packages imported them. The 8 names that `calls/__init__.py`
re-exported from there now come from the working leaf modules.
"""

import importlib
import os


def test_all_calls_module_file_removed():
    """The 621-line dead module should no longer exist on disk."""
    import catstack.calls as calls_pkg
    pkg_dir = os.path.dirname(calls_pkg.__file__)
    all_calls_path = os.path.join(pkg_dir, "all_calls.py")
    assert not os.path.exists(all_calls_path), (
        f"all_calls.py still exists at {all_calls_path}; should be deleted "
        f"and its re-exports replaced with imports from .stepback / .CoVe."
    )


def test_all_calls_module_not_importable():
    """`from catstack.calls import all_calls` and `import catstack.calls.all_calls`
    should both fail."""
    import importlib
    try:
        importlib.import_module("catstack.calls.all_calls")
    except (ImportError, ModuleNotFoundError):
        return  # expected
    raise AssertionError("catstack.calls.all_calls is still importable")


def test_public_re_exports_still_work():
    """The 8 names previously exported via all_calls remain importable
    from `catstack.calls` (now sourced from the leaf modules)."""
    from catstack.calls import (
        get_stepback_insight_openai,
        get_stepback_insight_anthropic,
        get_stepback_insight_google,
        get_stepback_insight_mistral,
        chain_of_verification_openai,
        chain_of_verification_anthropic,
        chain_of_verification_google,
        chain_of_verification_mistral,
    )
    for fn in (
        get_stepback_insight_openai, get_stepback_insight_anthropic,
        get_stepback_insight_google, get_stepback_insight_mistral,
        chain_of_verification_openai, chain_of_verification_anthropic,
        chain_of_verification_google, chain_of_verification_mistral,
    ):
        assert callable(fn)


def test_re_exports_resolve_to_leaf_modules_not_all_calls():
    """The two formerly-broken CoVe variants must resolve to the working
    leaf-module versions, not the dead duplicates."""
    from catstack.calls import (
        chain_of_verification_anthropic,
        chain_of_verification_google,
    )
    assert chain_of_verification_anthropic.__module__ == "catstack.calls.CoVe"
    assert chain_of_verification_google.__module__ == "catstack.calls.CoVe"


def test_stepback_re_exports_resolve_to_stepback_module():
    from catstack.calls import (
        get_stepback_insight_openai,
        get_stepback_insight_anthropic,
        get_stepback_insight_google,
        get_stepback_insight_mistral,
    )
    for fn in (
        get_stepback_insight_openai, get_stepback_insight_anthropic,
        get_stepback_insight_google, get_stepback_insight_mistral,
    ):
        assert fn.__module__ == "catstack.calls.stepback", (
            f"{fn.__name__} resolves to {fn.__module__}; expected catstack.calls.stepback"
        )
