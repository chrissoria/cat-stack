"""
Tests for _run_one_sync_model — the sync-fallback used by
run_batch_ensemble_classify when an ensemble includes unsupported-batch
providers (huggingface, huggingface-together, perplexity, ollama).

Regression: the function previously did `raw = client.complete(...)` —
but `complete()` returns a `(text, error)` tuple. The result:
  - Success path: raw = (text_str, None); extract_json(tuple) returned
    the schema-error sentinel '{"1":"e"}' for every row.
  - Error path: raw = (None, "error msg"); the error message was lost in
    the same way, hidden behind a generic schema failure.

So every ensemble batch run that included an unsupported-batch provider
produced silent garbage from that model's column.

Fix: `raw, err = client.complete(...)`; if err, record (None, err); else
record (extract_json(raw), None) — matching the pattern at
_run_one_sync_summarize_model:1187.
"""

from unittest.mock import patch, MagicMock

from cat_stack._batch import _run_one_sync_model


def _patched_prompt(messages_text="test"):
    """Returns a minimal valid messages list."""
    return [{"role": "user", "content": messages_text}]


class TestRunOneSyncModelTupleUnpacking:
    @patch("cat_stack._batch.UnifiedLLMClient")
    def test_successful_response_returns_parsed_json_not_error_sentinel(
        self, mock_client_cls
    ):
        """Regression: success used to return '{"1":"e"}' because
        extract_json was called on the tuple instead of the string."""
        mock_client = MagicMock()
        mock_client.complete.return_value = ('{"1":"1","2":"0"}', None)
        mock_client_cls.return_value = mock_client

        with patch(
            "cat_stack.text_functions_ensemble.build_text_classification_prompt",
            return_value=_patched_prompt(),
        ):
            cfg = {
                "model": "test-model",
                "provider": "huggingface",
                "api_key": "fake-key",
            }
            items = ["resp 1", "resp 2"]
            prompt_params = {"categories_str": "1. A\n2. B", "json_schema": None}

            result = _run_one_sync_model(cfg, items, prompt_params)

        assert set(result.keys()) == {0, 1}
        for idx in (0, 1):
            json_str, err = result[idx]
            assert err is None, f"item {idx} unexpected error: {err}"
            assert json_str is not None
            assert json_str != '{"1":"e"}', (
                f"item {idx} regressed to error sentinel — tuple unpacking broken again"
            )
            assert '"1"' in json_str and '"2"' in json_str

    @patch("cat_stack._batch.UnifiedLLMClient")
    def test_provider_error_propagates_in_err_field_not_swallowed(
        self, mock_client_cls
    ):
        """Regression: an API error used to be hidden behind the same
        generic '{"1":"e"}' sentinel because we never read the err half
        of the tuple."""
        mock_client = MagicMock()
        mock_client.complete.return_value = (None, "API rate limit exceeded")
        mock_client_cls.return_value = mock_client

        with patch(
            "cat_stack.text_functions_ensemble.build_text_classification_prompt",
            return_value=_patched_prompt(),
        ):
            cfg = {
                "model": "test-model",
                "provider": "perplexity",
                "api_key": "fake-key",
            }
            items = ["resp"]
            prompt_params = {"categories_str": "1. A", "json_schema": None}

            result = _run_one_sync_model(cfg, items, prompt_params)

        json_str, err = result[0]
        assert json_str is None
        assert err == "API rate limit exceeded", (
            "real provider error should surface in err, not be swallowed"
        )

    @patch("cat_stack._batch.UnifiedLLMClient")
    def test_exception_caught_and_stored_as_error(self, mock_client_cls):
        """Exceptions raised by client.complete (e.g., transport errors)
        are caught and stored — same behavior as before the fix."""
        mock_client = MagicMock()
        mock_client.complete.side_effect = ConnectionError("network unreachable")
        mock_client_cls.return_value = mock_client

        with patch(
            "cat_stack.text_functions_ensemble.build_text_classification_prompt",
            return_value=_patched_prompt(),
        ):
            cfg = {
                "model": "test-model",
                "provider": "ollama",
                "api_key": None,
            }
            items = ["resp"]
            prompt_params = {"categories_str": "1. A", "json_schema": None}

            result = _run_one_sync_model(cfg, items, prompt_params)

        json_str, err = result[0]
        assert json_str is None
        assert "network unreachable" in err

    @patch("cat_stack._batch.UnifiedLLMClient")
    def test_mixed_success_and_error_items_independently_tracked(
        self, mock_client_cls
    ):
        """Per-item independence: a failing call on item 1 shouldn't
        affect item 0 or item 2."""
        mock_client = MagicMock()
        mock_client.complete.side_effect = [
            ('{"1":"1"}', None),
            (None, "transient 503"),
            ('{"1":"0"}', None),
        ]
        mock_client_cls.return_value = mock_client

        with patch(
            "cat_stack.text_functions_ensemble.build_text_classification_prompt",
            return_value=_patched_prompt(),
        ):
            cfg = {
                "model": "test-model",
                "provider": "huggingface",
                "api_key": "fake-key",
            }
            items = ["a", "b", "c"]
            prompt_params = {"categories_str": "1. A", "json_schema": None}

            result = _run_one_sync_model(cfg, items, prompt_params)

        assert result[0] == ('{"1":"1"}', None)
        assert result[1] == (None, "transient 503")
        assert result[2] == ('{"1":"0"}', None)
