"""
Tests for H-RETRY: jitter + Retry-After + total-time cap in
UnifiedLLMClient.complete() and OSError handling in _call_claude_cli.

Pre-fix issues:
  - No jitter on backoff → all concurrent rate-limited workers wake at
    the same instant (thundering herd).
  - Retry-After header ignored → provider's explicit retry hint
    overridden by our exponential schedule.
  - No total-time cap → 5 retries × 5× multiplier on 429s could block
    a single call for 310s+.
  - _call_claude_cli didn't catch OSError → a multi-MB prompt that
    overflowed argv (E2BIG on macOS/Linux) crashed the caller instead
    of returning (None, error).
"""

from unittest.mock import patch, MagicMock

import pytest

from cat_stack._providers import (
    UnifiedLLMClient,
    _parse_retry_after,
    _backoff_with_jitter,
    _MAX_TOTAL_WAIT_SECONDS,
)


# ── _parse_retry_after ──────────────────────────────────────────────────

class TestParseRetryAfter:
    def test_integer_seconds(self):
        assert _parse_retry_after("30") == 30.0
        assert _parse_retry_after("0") == 0.0

    def test_float_seconds(self):
        assert _parse_retry_after("60.5") == 60.5

    def test_negative_seconds_clamped_to_zero(self):
        assert _parse_retry_after("-10") == 0.0

    def test_none_input(self):
        assert _parse_retry_after(None) is None

    def test_empty_input(self):
        assert _parse_retry_after("") is None

    def test_garbage_input(self):
        assert _parse_retry_after("not-a-number-or-date") is None

    def test_http_date_in_future(self):
        from email.utils import format_datetime
        from datetime import datetime, timedelta, timezone
        future = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=45))
        parsed = _parse_retry_after(future)
        assert parsed is not None
        assert 40 <= parsed <= 50

    def test_http_date_in_past_clamps_to_zero(self):
        from email.utils import format_datetime
        from datetime import datetime, timedelta, timezone
        past = format_datetime(datetime.now(timezone.utc) - timedelta(seconds=60))
        assert _parse_retry_after(past) == 0.0


# ── _backoff_with_jitter ───────────────────────────────────────────────

class TestBackoffWithJitter:
    def test_within_jitter_bounds(self):
        """Result is always in [0.5 * base, 1.5 * base]."""
        for attempt in range(6):
            base = 2.0 * (2 ** attempt) * 1.0
            samples = [_backoff_with_jitter(2.0, attempt) for _ in range(100)]
            for s in samples:
                assert 0.5 * base <= s <= 1.5 * base, (
                    f"attempt={attempt}, base={base}, sample={s}"
                )

    def test_multiplier_applied(self):
        """multiplier=5 (used for 429) shifts the band 5x."""
        for _ in range(50):
            normal = _backoff_with_jitter(2.0, 0, multiplier=1.0)
            rate_limit = _backoff_with_jitter(2.0, 0, multiplier=5.0)
            # rate_limit band [5, 15], normal band [1, 3] — disjoint
            assert rate_limit >= 5.0
            assert normal <= 3.0

    def test_randomness_produces_different_samples(self):
        """Two consecutive calls should differ — verifies jitter is real."""
        samples = {_backoff_with_jitter(2.0, 3) for _ in range(20)}
        assert len(samples) > 1, "jitter should produce varied samples"


# ── complete() honors Retry-After ───────────────────────────────────────

def _client():
    return UnifiedLLMClient(provider="openai", api_key="fake", model="gpt-4o")


def _response(status_code=200, headers=None, text="", json_data=None):
    r = MagicMock()
    r.status_code = status_code
    r.headers = headers or {}
    r.text = text
    r.json.return_value = json_data or {}
    r.raise_for_status = MagicMock()
    return r


class TestRetryAfterHonored:
    @patch("cat_stack._providers.time.sleep")
    @patch("cat_stack._providers.requests.post")
    def test_429_with_retry_after_uses_that_value(self, mock_post, mock_sleep):
        """When provider sends Retry-After: 7, we sleep ~7s, not the
        exponential default."""
        first = _response(status_code=429, headers={"Retry-After": "7"})
        second = _response(
            status_code=200,
            headers={},
            json_data={"choices": [{"message": {"content": "ok"}}]},
        )
        mock_post.side_effect = [first, second]

        result, err = _client().complete(messages=[{"role": "user", "content": "hi"}])
        assert err is None
        assert result == "ok"

        # Exactly one sleep, with the Retry-After value (no jitter overlay
        # because the header is authoritative).
        assert mock_sleep.call_count == 1
        slept = mock_sleep.call_args.args[0]
        assert slept == 7.0, f"expected 7s sleep from Retry-After, got {slept}"

    @patch("cat_stack._providers.time.sleep")
    @patch("cat_stack._providers.requests.post")
    def test_429_without_retry_after_uses_jittered_exponential(
        self, mock_post, mock_sleep
    ):
        """No Retry-After header → fall back to jittered exponential with
        the 5x multiplier."""
        first = _response(status_code=429, headers={})
        second = _response(
            status_code=200,
            json_data={"choices": [{"message": {"content": "ok"}}]},
        )
        mock_post.side_effect = [first, second]

        _client().complete(messages=[{"role": "user", "content": "hi"}])

        assert mock_sleep.call_count == 1
        slept = mock_sleep.call_args.args[0]
        # attempt=0, initial_delay=2, multiplier=5 → base=10, jitter [5,15]
        assert 5.0 <= slept <= 15.0, f"jittered exponential out of range: {slept}"

    @patch("cat_stack._providers.time.sleep")
    @patch("cat_stack._providers.requests.post")
    def test_503_with_retry_after_uses_that_value(self, mock_post, mock_sleep):
        first = _response(status_code=503, headers={"Retry-After": "3"})
        second = _response(
            status_code=200,
            json_data={"choices": [{"message": {"content": "ok"}}]},
        )
        mock_post.side_effect = [first, second]

        _client().complete(messages=[{"role": "user", "content": "hi"}])

        assert mock_sleep.call_count == 1
        assert mock_sleep.call_args.args[0] == 3.0


# ── Total-time cap ──────────────────────────────────────────────────────

class TestTotalTimeCap:
    @patch("cat_stack._providers.time.sleep")
    @patch("cat_stack._providers.time.monotonic")
    @patch("cat_stack._providers.requests.post")
    def test_cap_aborts_retry_when_elapsed_plus_wait_exceeds_max(
        self, mock_post, mock_monotonic, mock_sleep
    ):
        """If we've already slept ~290s and the next wait would push past
        300s, return the error instead of sleeping again."""
        # monotonic: start=0, after-first-failure=290 (close to cap)
        mock_monotonic.side_effect = [0.0, 290.0, 290.0]
        # 429 with Retry-After: 60 (60 + 290 = 350 > 300 cap)
        mock_post.return_value = _response(
            status_code=429, headers={"Retry-After": "60"}
        )

        result, err = _client().complete(messages=[{"role": "user", "content": "hi"}])
        assert err is not None
        assert "Rate limit" in err
        # No sleep should have happened — cap aborted before sleeping
        assert mock_sleep.call_count == 0


# ── _call_claude_cli OSError ────────────────────────────────────────────

class TestClaudeCliOSError:
    @patch("subprocess.run")
    def test_os_error_returns_helpful_message_not_crash(self, mock_run):
        """E2BIG on macOS/Linux when argv is too large. Should surface as
        an error tuple, not bubble up and crash the caller."""
        mock_run.side_effect = OSError(7, "Argument list too long")

        client = UnifiedLLMClient(provider="claude-code", api_key="fake", model="sonnet")
        result, err = client.complete(messages=[{"role": "user", "content": "huge"}])

        assert result is None
        assert err is not None
        assert "subprocess failed" in err
        assert "Argument list too long" in err

    @patch("subprocess.run")
    def test_os_error_does_not_retry_max_retries_times(self, mock_run):
        """E2BIG is deterministic for this prompt size — retrying would
        just produce the same failure. The OSError catch is OUTSIDE the
        retry loop so we abort after one attempt."""
        mock_run.side_effect = OSError(7, "Argument list too long")

        client = UnifiedLLMClient(provider="claude-code", api_key="fake", model="sonnet")
        client.complete(messages=[{"role": "user", "content": "huge"}])

        assert mock_run.call_count == 1, (
            f"expected exactly 1 call (OSError caught outside loop), got {mock_run.call_count}"
        )
