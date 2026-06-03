"""
Tests for task #28 (H-BATCH): Anthropic terminal-state inspection +
per-model failure isolation.

Anthropic uses `"ended"` as the only terminal `processing_status`, with
detailed outcome in `request_counts`. The polling code previously treated
"ended" as full success without inspecting counts. Result: a batch where
every request errored silently looked like success to the caller; the
DataFrame came back with all-None values for that model and no clear
log signal that the whole batch was dead.

The fix raises BatchJobFailedError / BatchJobExpiredError for uniformly
failed/canceled/expired batches, and prints a warning for partial cases.
"""

from unittest.mock import patch, MagicMock

import pytest

from cat_stack._batch import (
    _inspect_anthropic_terminal_state,
    BatchJobFailedError,
    BatchJobExpiredError,
)


class TestInspectAnthropicTerminalState:
    def test_full_success_returns_silently(self, capsys):
        status = {"request_counts": {"succeeded": 10, "errored": 0, "canceled": 0, "expired": 0}}
        assert _inspect_anthropic_terminal_state(status, "job-1") is None
        assert capsys.readouterr().out == ""

    def test_all_errored_raises_failed(self):
        status = {"request_counts": {"succeeded": 0, "errored": 5, "canceled": 0, "expired": 0}}
        with pytest.raises(BatchJobFailedError) as exc_info:
            _inspect_anthropic_terminal_state(status, "job-X")
        msg = str(exc_info.value)
        assert "0/5" in msg
        assert "errored=5" in msg
        assert "job-X" in msg

    def test_all_canceled_raises_expired(self):
        status = {"request_counts": {"succeeded": 0, "errored": 0, "canceled": 5, "expired": 0}}
        with pytest.raises(BatchJobExpiredError) as exc_info:
            _inspect_anthropic_terminal_state(status, "job-Y")
        assert "canceled" in str(exc_info.value)
        assert "job-Y" in str(exc_info.value)

    def test_all_expired_raises_expired(self):
        status = {"request_counts": {"succeeded": 0, "errored": 0, "canceled": 0, "expired": 5}}
        with pytest.raises(BatchJobExpiredError) as exc_info:
            _inspect_anthropic_terminal_state(status, "job-Z")
        assert "expired" in str(exc_info.value)

    def test_mixed_failure_modes_no_success_raises_failed(self):
        """When 0 succeeded but failures are mixed (errored+canceled+expired),
        report as BatchJobFailedError with the full breakdown."""
        status = {"request_counts": {"succeeded": 0, "errored": 2, "canceled": 2, "expired": 1}}
        with pytest.raises(BatchJobFailedError) as exc_info:
            _inspect_anthropic_terminal_state(status, "job-mix")
        msg = str(exc_info.value)
        assert "0/5" in msg
        assert "errored=2" in msg
        assert "canceled=2" in msg
        assert "expired=1" in msg

    def test_partial_success_with_errors_prints_warning(self, capsys):
        """Partial: some succeeded, some failed → return None + warning."""
        status = {"request_counts": {"succeeded": 3, "errored": 2, "canceled": 0, "expired": 0}}
        assert _inspect_anthropic_terminal_state(status, "job-partial") is None
        captured = capsys.readouterr().out
        assert "partial" in captured
        assert "succeeded=3" in captured
        assert "errored=2" in captured

    def test_empty_counts_returns_silently(self, capsys):
        """Edge case: empty request_counts (e.g., batch with no requests) →
        return None, no warning, no exception."""
        status = {"request_counts": {}}
        assert _inspect_anthropic_terminal_state(status, "job-empty") is None
        assert capsys.readouterr().out == ""

    def test_missing_request_counts_key_returns_silently(self, capsys):
        """Defensive: status_data with no request_counts key at all → return
        silently rather than crashing."""
        status = {}
        assert _inspect_anthropic_terminal_state(status, "job-missing") is None
        assert capsys.readouterr().out == ""


class TestPollBatchJobAnthropicInspection:
    """Verify _poll_batch_job invokes the inspection helper for Anthropic
    when state reaches "ended"."""

    @patch("cat_stack._batch.requests.get")
    def test_ended_with_all_errored_raises_failed(self, mock_get):
        """Polling an Anthropic batch that ends with 0 successes should
        raise BatchJobFailedError from the inspection helper."""
        from cat_stack._batch import _poll_batch_job

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "processing_status": "ended",
            "request_counts": {"succeeded": 0, "errored": 3, "canceled": 0, "expired": 0},
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        with pytest.raises(BatchJobFailedError) as exc_info:
            _poll_batch_job(
                provider="anthropic",
                api_key="fake-key",
                job_id="msgbatch_test123",
                interval=0.01,
                timeout=10.0,
            )
        assert "0/3" in str(exc_info.value)

    @patch("cat_stack._batch.requests.get")
    def test_ended_with_full_success_returns(self, mock_get):
        """Polling an Anthropic batch that ends with all successes should
        return the status data without raising."""
        from cat_stack._batch import _poll_batch_job

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "processing_status": "ended",
            "request_counts": {"succeeded": 5, "errored": 0, "canceled": 0, "expired": 0},
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = _poll_batch_job(
            provider="anthropic",
            api_key="fake-key",
            job_id="msgbatch_ok",
            interval=0.01,
            timeout=10.0,
        )
        assert result["processing_status"] == "ended"
        assert result["request_counts"]["succeeded"] == 5

    @patch("cat_stack._batch.requests.get")
    def test_ended_with_partial_returns_with_warning(self, mock_get, capsys):
        """Partial success returns status_data + prints a warning."""
        from cat_stack._batch import _poll_batch_job

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "processing_status": "ended",
            "request_counts": {"succeeded": 4, "errored": 1, "canceled": 0, "expired": 0},
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = _poll_batch_job(
            provider="anthropic",
            api_key="fake-key",
            job_id="msgbatch_partial",
            interval=0.01,
            timeout=10.0,
        )
        assert result["processing_status"] == "ended"
        captured = capsys.readouterr().out
        assert "partial" in captured
