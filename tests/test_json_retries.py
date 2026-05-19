"""
Tests for the json_retries parameter on classify().

json_retries controls how many times the classifier re-asks the LLM when the
response fails JSON validation (separate from max_retries, which handles
transport/API failures). The retry path also appends a "Respond with ONLY
valid JSON" nudge to the prompt.

We mock UnifiedLLMClient.complete so no network calls happen.
"""

import pandas as pd
import pytest

from cat_stack.classify import classify


VALID_REPLY = '{"1":1,"2":0,"3":0}'      # passes validate_classification_json
INVALID_REPLY = '{"foo":"bar"}'           # parses as JSON but fails category-shape check


@pytest.fixture
def df():
    return pd.DataFrame({"text": ["hello world"]})


def _make_complete_mock(replies):
    """Return (mock_fn, call_log) where mock_fn yields replies[i] on call i+1."""
    call_log = []

    def fake_complete(self, messages, json_schema=None, creativity=None,
                     thinking_budget=None, max_retries=5):
        idx = len(call_log)
        call_log.append({"messages": messages})
        reply = replies[min(idx, len(replies) - 1)]
        return reply, None  # (reply, error)

    return fake_complete, call_log


def test_json_retries_zero_makes_one_call(df, monkeypatch):
    """json_retries=0 → exactly 1 LLM call even if JSON is invalid."""
    fake, log = _make_complete_mock([INVALID_REPLY])
    monkeypatch.setattr(
        "cat_stack._providers.UnifiedLLMClient.complete", fake
    )

    classify(
        input_data=df,
        categories=["A", "B", "C"],
        user_model="gpt-4o",
        api_key="sk-fake",
        json_retries=0,
        batch_retries=0,
        check_verbosity=False,
        add_other=False,
        parallel=False,
    )

    assert len(log) == 1, f"expected 1 call, got {len(log)}"


def test_json_retries_two_makes_three_calls_on_persistent_invalid(df, monkeypatch):
    """json_retries=2 → up to 3 calls (initial + 2 retries) on persistent invalid JSON."""
    fake, log = _make_complete_mock([INVALID_REPLY, INVALID_REPLY, INVALID_REPLY])
    monkeypatch.setattr(
        "cat_stack._providers.UnifiedLLMClient.complete", fake
    )

    classify(
        input_data=df,
        categories=["A", "B", "C"],
        user_model="gpt-4o",
        api_key="sk-fake",
        json_retries=2,
        batch_retries=0,
        check_verbosity=False,
        add_other=False,
        parallel=False,
    )

    assert len(log) == 3, f"expected 3 calls, got {len(log)}"


def test_json_retries_stops_early_on_valid(df, monkeypatch):
    """Retry loop must exit as soon as valid JSON arrives."""
    # invalid, then valid — should be exactly 2 calls (not 3)
    fake, log = _make_complete_mock([INVALID_REPLY, VALID_REPLY, VALID_REPLY])
    monkeypatch.setattr(
        "cat_stack._providers.UnifiedLLMClient.complete", fake
    )

    classify(
        input_data=df,
        categories=["A", "B", "C"],
        user_model="gpt-4o",
        api_key="sk-fake",
        json_retries=5,
        batch_retries=0,
        check_verbosity=False,
        add_other=False,
        parallel=False,
    )

    assert len(log) == 2, f"expected 2 calls (invalid then valid), got {len(log)}"


def test_json_retries_appends_nudge_on_retry(df, monkeypatch):
    """Retry attempts must append the 'Respond with ONLY valid JSON' nudge."""
    fake, log = _make_complete_mock([INVALID_REPLY, VALID_REPLY])
    monkeypatch.setattr(
        "cat_stack._providers.UnifiedLLMClient.complete", fake
    )

    classify(
        input_data=df,
        categories=["A", "B", "C"],
        user_model="gpt-4o",
        api_key="sk-fake",
        json_retries=2,
        batch_retries=0,
        check_verbosity=False,
        add_other=False,
        parallel=False,
    )

    first_last_msg = log[0]["messages"][-1]["content"]
    retry_last_msg = log[1]["messages"][-1]["content"]

    assert "Respond with ONLY valid JSON" not in first_last_msg
    assert "Respond with ONLY valid JSON" in retry_last_msg
