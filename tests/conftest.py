"""Shared fixtures.

The cat-claws agent backends run a sign-in preflight
(_providers._require_agent_sign_in -> catclaws.ensure_signed_in) that shells
out to the real agent CLI. Unit tests must not depend on this machine's login
state, so it is a no-op by default; tests marked `real_auth` exercise it with
cat-claws mocked.
"""
import pytest


@pytest.fixture(autouse=True)
def _no_agent_sign_in_preflight(request, monkeypatch):
    if request.node.get_closest_marker("real_auth"):
        return
    try:
        import catclaws
    except ImportError:
        return
    if hasattr(catclaws, "ensure_signed_in"):
        monkeypatch.setattr(catclaws, "ensure_signed_in", lambda *a, **k: None)


def pytest_configure(config):
    config.addinivalue_line("markers", "real_auth: run the real agent sign-in preflight")
