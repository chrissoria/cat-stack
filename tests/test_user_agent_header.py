"""2.0.0b5: both HTTP request paths must send a browser User-Agent.

Some providers fronted by a WAF (the HF router's featherless-ai backend)
403 the default `python-requests` agent, surfacing as spurious classification
failures. `_get_headers` (main call) and `_detect_huggingface_endpoint`
(routing probe) must both set a browser-like User-Agent.
"""
import catstack._providers as pv


def test_get_headers_sets_browser_user_agent():
    client = pv.UnifiedLLMClient(provider="huggingface", api_key="sk-test", model="some/model")
    headers = client._get_headers()
    assert "User-Agent" in headers
    assert "Mozilla" in headers["User-Agent"]
    # auth + content-type still present
    assert headers["Content-Type"] == "application/json"
    assert any("sk-test" in str(v) for v in headers.values())


def test_detect_endpoint_probe_sends_user_agent(monkeypatch):
    captured = {}

    class _Resp:
        status_code = 200

    def _fake_post(url, headers=None, json=None, timeout=None):
        captured["headers"] = headers or {}
        return _Resp()

    monkeypatch.setattr(pv.requests, "post", _fake_post)
    pv._detect_huggingface_endpoint("sk-test", "some/model")
    assert "User-Agent" in captured["headers"]
    assert "Mozilla" in captured["headers"]["User-Agent"]
