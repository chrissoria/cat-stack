"""
Tests for task #44 (HF-SMALL-MODEL): strip response_format on persistent
5xx errors from gateways that reliably reject json_object with non-JSON
error bodies (e.g., HF's router for Llama-3.2-1B).

Key behaviors:
  - On a 5xx without Retry-After AND `response_format` in payload AND we
    haven't tried stripping yet: strip + immediate retry.
  - On a 5xx WITH Retry-After: trust the hint, don't strip — server is
    explicitly saying "transient overload" not "your payload is broken."
  - Once stripped + succeeded, the decision is cached on the client
    (`self._skip_response_format = True`) so subsequent rows skip
    `response_format` from the start.
"""

from unittest.mock import patch, MagicMock

from cat_stack._providers import UnifiedLLMClient


def _client():
    return UnifiedLLMClient(provider="openai", api_key="fake", model="gpt-4o")


# Pass a real-looking schema to force `response_format` into the payload.
# Without it, `_build_payload` doesn't include the key and the strip path
# is a no-op (which is the intended behavior — see TestStripWorksForVariousStatusCodes
# tests that exercise the path with a schema).
_FAKE_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
}


def _complete(client, messages=None):
    return client.complete(
        messages=messages or [{"role": "user", "content": "hi"}],
        json_schema=_FAKE_SCHEMA,
    )


def _response(status_code=200, headers=None, text="", json_data=None):
    r = MagicMock()
    r.status_code = status_code
    r.headers = headers or {}
    r.text = text
    r.json.return_value = json_data or {}
    r.raise_for_status = MagicMock()
    return r


def _ok(content="ok"):
    return _response(
        status_code=200,
        json_data={"choices": [{"message": {"content": content}}]},
    )


class TestStripOnPersistent5xx:
    @patch("cat_stack._providers.time.sleep")
    @patch("cat_stack._providers.requests.post")
    def test_strip_on_502_without_retry_after(self, mock_post, mock_sleep, capsys):
        """502 with HTML body (no Retry-After) → strip response_format,
        immediate retry, success.

        We can't compare `call_args_list[0]` vs. `call_args_list[1]` payloads
        directly because the mock stores by reference and the payload dict
        is mutated (`pop`) between calls. We verify the strip happened by:
          - The warning is printed.
          - The final (post-strip) call payload has no response_format.
          - There were exactly 2 POSTs.
          - No sleep — strip is immediate retry, not a backoff.
        """
        first = _response(
            status_code=502,
            headers={},
            text="<!DOCTYPE html><html>...gateway error...</html>",
        )
        mock_post.side_effect = [first, _ok()]

        c = _client()
        result, err = _complete(c)
        captured = capsys.readouterr()

        assert err is None
        assert result == "ok"
        assert mock_post.call_count == 2
        # The current payload (which is what the second call sent, after the
        # pop) must not have response_format.
        assert "response_format" not in mock_post.call_args.kwargs["json"]
        # The warning printed → confirms the strip code path fired.
        assert "Retrying without response_format" in captured.out
        # No sleep — strip is immediate retry.
        assert mock_sleep.call_count == 0

    @patch("cat_stack._providers.time.sleep")
    @patch("cat_stack._providers.requests.post")
    def test_503_with_retry_after_does_not_strip(self, mock_post, mock_sleep):
        """503 + Retry-After means "transient overload, come back in N
        seconds" — that's NOT a payload-format complaint. Trust the hint."""
        first = _response(status_code=503, headers={"Retry-After": "3"})
        mock_post.side_effect = [first, _ok()]

        c = _client()
        _complete(c)

        # The second call should still have response_format
        second_payload = mock_post.call_args_list[1].kwargs["json"]
        assert "response_format" in second_payload, (
            "Retry-After hint should not trigger the response_format strip"
        )
        # And we should have slept the 3s the server told us to
        assert mock_sleep.call_args.args[0] == 3.0

    @patch("cat_stack._providers.time.sleep")
    @patch("cat_stack._providers.requests.post")
    def test_strip_caches_decision_for_future_calls(self, mock_post, mock_sleep):
        """Once we strip + succeed, subsequent complete() calls on the same
        client should drop response_format from the start — no wasted
        first-call-5xx-then-retry cycle for every row."""
        first = _response(status_code=502, headers={}, text="<html>err</html>")
        mock_post.side_effect = [first, _ok(), _ok()]

        c = _client()
        _complete(c, [{"role": "user", "content": "row 1"}])
        _complete(c, [{"role": "user", "content": "row 2"}])

        # Call 3 (the second row's first attempt) must NOT have response_format
        assert mock_post.call_count == 3
        third_payload = mock_post.call_args_list[2].kwargs["json"]
        assert "response_format" not in third_payload, (
            "client should remember response_format is unsupported on this endpoint"
        )

    @patch("cat_stack._providers.time.sleep")
    @patch("cat_stack._providers.requests.post")
    def test_strip_only_once_per_call(self, mock_post, mock_sleep):
        """If after stripping we STILL get 5xx, fall through to normal
        backoff — don't try to strip a second time on the same call."""
        responses = [
            _response(status_code=502, headers={}, text="<html>1</html>"),
            _response(status_code=502, headers={}, text="<html>2</html>"),
            _response(status_code=502, headers={}, text="<html>3</html>"),
            _ok(),
        ]
        mock_post.side_effect = responses

        c = _client()
        result, _ = _complete(c)

        # Strip happened on call 1; calls 2 and 3 went through normal backoff
        assert result == "ok"
        # Sleep should have fired on calls 2 and 3 (post-strip backoff)
        assert mock_sleep.call_count >= 1

    @patch("cat_stack._providers.requests.post")
    def test_no_strip_when_payload_has_no_response_format(self, mock_post):
        """If response_format isn't in payload (force_json=False), the
        strip path is a no-op."""
        mock_post.side_effect = [
            _response(status_code=502, headers={}, text="<html>err</html>"),
            _response(status_code=502, headers={}, text="<html>err</html>"),
            _response(status_code=502, headers={}, text="<html>err</html>"),
            _response(status_code=502, headers={}, text="<html>err</html>"),
            _response(status_code=502, headers={}, text="<html>err</html>"),
        ]

        c = _client()
        result, err = c.complete(
            messages=[{"role": "user", "content": "hi"}],
            force_json=False,  # → no response_format in payload
        )

        assert result is None
        assert err is not None
        # All 5 calls went out (no strip-skipped retry)
        assert mock_post.call_count == 5
        for call in mock_post.call_args_list:
            assert "response_format" not in call.kwargs["json"]


class TestStripWorksForVariousStatusCodes:
    @patch("cat_stack._providers.time.sleep")
    @patch("cat_stack._providers.requests.post")
    def test_500(self, mock_post, _sleep):
        mock_post.side_effect = [_response(500, text="err"), _ok()]
        result, _ = _complete(_client())
        assert result == "ok"
        assert "response_format" not in mock_post.call_args_list[1].kwargs["json"]

    @patch("cat_stack._providers.time.sleep")
    @patch("cat_stack._providers.requests.post")
    def test_502(self, mock_post, _sleep):
        mock_post.side_effect = [_response(502, text="err"), _ok()]
        result, _ = _complete(_client())
        assert result == "ok"
        assert "response_format" not in mock_post.call_args_list[1].kwargs["json"]

    @patch("cat_stack._providers.time.sleep")
    @patch("cat_stack._providers.requests.post")
    def test_504(self, mock_post, _sleep):
        mock_post.side_effect = [_response(504, text="err"), _ok()]
        result, _ = _complete(_client())
        assert result == "ok"
        assert "response_format" not in mock_post.call_args_list[1].kwargs["json"]
