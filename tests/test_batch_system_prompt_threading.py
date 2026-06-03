"""
Tests for task #31 (H-BATCH-PROMPT): system_prompt threading through the
batch path.

Pre-fix: `classify(system_prompt="...", batch_mode=True)` silently
dropped the system_prompt. The producers in classify.py (both the
single-model `prompt_params` dict at L805-817 and the ensemble
`prompt_params_per_model` dict at L838-852) didn't include
"system_prompt" as a key, and the consumers in _batch.py
(`_run_one_batch_job` at L764 and `_run_one_sync_model` at L862)
didn't pass `system_prompt=` to `build_text_classification_prompt`.

The sync path at text_functions_ensemble.py:3132-3144 has always
passed it through. So `classify(system_prompt="X", batch_mode=False)`
and `classify(system_prompt="X", batch_mode=True)` produced different
prompts for the same input — silently dropping a feature the user
explicitly invoked (especially load-bearing for users running
`prompt_tune()` then immediately submitting a batch).

Post-fix: both producers add "system_prompt" to their params dict;
both consumers read it back out and forward it to the prompt builder.
The system_prompt now appears in the JSONL payload that batch APIs
receive, matching the sync-mode output.
"""

from unittest.mock import patch, MagicMock

import pytest

from cat_stack._batch import _run_one_batch_job, _run_one_sync_model


def _cfg(model="gpt-4o", provider="openai"):
    return {
        "model": model,
        "provider": provider,
        "api_key": "fake-key",
        "sanitized_name": model.replace("-", "_").replace(".", "_"),
    }


def _params(system_prompt=""):
    return {
        "categories_str": "1. yes\n2. no",
        "survey_question_context": "",
        "examples_text": "",
        "chain_of_thought": False,
        "context_prompt": False,
        "step_back_prompt": False,
        "stepback_insights": {},
        "system_prompt": system_prompt,
        "json_schema": {"type": "object"},
        "creativity": 0,
        "thinking_budget": 0,
        "multi_label": True,
    }


class TestBatchJobForwardsSystemPrompt:
    """_run_one_batch_job must pass prompt_params['system_prompt'] to
    build_text_classification_prompt — otherwise the JSONL payload
    submitted to OpenAI/Anthropic/etc. is missing the user's custom
    system instruction."""

    @patch("cat_stack._batch._poll_batch_job")
    @patch("cat_stack._batch._download_batch_results", return_value="")
    @patch("cat_stack._batch._parse_batch_results", return_value={})
    @patch("cat_stack._batch._create_batch_job", return_value="job-xyz")
    @patch("cat_stack._batch._upload_jsonl", return_value="file-xyz")
    def test_system_prompt_reaches_prompt_builder(
        self, _upload, _create, _parse, _download, _poll
    ):
        custom_instr = "You are a benevolent panda classifier. Answer in JSON only."

        with patch(
            "cat_stack.text_functions_ensemble.build_text_classification_prompt"
        ) as mock_build:
            mock_build.return_value = [{"role": "user", "content": "hi"}]

            _run_one_batch_job(
                cfg=_cfg(),
                items=["text-1"],
                prompt_params=_params(system_prompt=custom_instr),
            )

            # build_text_classification_prompt got called with our system_prompt
            assert mock_build.called
            kwargs = mock_build.call_args.kwargs
            assert kwargs.get("system_prompt") == custom_instr, (
                f"expected system_prompt={custom_instr!r}, got {kwargs.get('system_prompt')!r}"
            )

    @patch("cat_stack._batch._poll_batch_job")
    @patch("cat_stack._batch._download_batch_results", return_value="")
    @patch("cat_stack._batch._parse_batch_results", return_value={})
    @patch("cat_stack._batch._create_batch_job", return_value="job-xyz")
    @patch("cat_stack._batch._upload_jsonl", return_value="file-xyz")
    def test_empty_system_prompt_when_not_set(
        self, _upload, _create, _parse, _download, _poll
    ):
        """Missing key in prompt_params → empty string default (matches
        build_text_classification_prompt's own default)."""
        params = _params()
        del params["system_prompt"]

        with patch(
            "cat_stack.text_functions_ensemble.build_text_classification_prompt"
        ) as mock_build:
            mock_build.return_value = [{"role": "user", "content": "hi"}]

            _run_one_batch_job(
                cfg=_cfg(),
                items=["text-1"],
                prompt_params=params,
            )

            kwargs = mock_build.call_args.kwargs
            assert kwargs.get("system_prompt") == ""


class TestSyncFallbackForwardsSystemPrompt:
    """_run_one_sync_model is invoked for batch-unsupported providers
    (huggingface, perplexity, ollama) when the user calls
    classify(batch_mode=True). It must thread system_prompt too —
    otherwise mixed-provider ensembles get inconsistent prompts."""

    def test_system_prompt_reaches_prompt_builder(self):
        custom_instr = "Be precise. Output only JSON."

        with patch(
            "cat_stack.text_functions_ensemble.build_text_classification_prompt"
        ) as mock_build, patch(
            "cat_stack._batch.UnifiedLLMClient"
        ) as mock_client_cls:
            mock_build.return_value = [{"role": "user", "content": "hi"}]
            mock_client = MagicMock()
            mock_client.complete.return_value = ('{"1":"1","2":"0"}', None)
            mock_client_cls.return_value = mock_client

            _run_one_sync_model(
                cfg=_cfg(model="meta-llama/Llama-3.1-8B-Instruct", provider="huggingface"),
                items=["text-1"],
                prompt_params=_params(system_prompt=custom_instr),
            )

            assert mock_build.called
            kwargs = mock_build.call_args.kwargs
            assert kwargs.get("system_prompt") == custom_instr


class TestProducerSideIncludesSystemPrompt:
    """The producers in classify.py must add system_prompt to the
    prompt_params dict so the consumer sees it. Audit by checking the
    source contains the key — direct unit-testing classify() would
    require mocking out the entire API stack."""

    def test_single_model_batch_path_includes_system_prompt(self):
        from pathlib import Path
        src = Path("src/catstack/classify.py").read_text()
        # The single-model batch dict is the one that calls run_batch_classify.
        # Find it by anchoring on the call.
        single_model_block_start = src.index("prompt_params = {")
        single_model_block_end = src.index("result = run_batch_classify(", single_model_block_start)
        block = src[single_model_block_start:single_model_block_end]
        assert '"system_prompt": system_prompt' in block, (
            "single-model batch path is missing system_prompt in its prompt_params dict"
        )

    def test_ensemble_batch_path_includes_system_prompt(self):
        from pathlib import Path
        src = Path("src/catstack/classify.py").read_text()
        # The ensemble batch dict is the one that calls run_batch_ensemble_classify.
        ensemble_block_start = src.index("prompt_params_per_model = {")
        ensemble_block_end = src.index("result = run_batch_ensemble_classify(", ensemble_block_start)
        block = src[ensemble_block_start:ensemble_block_end]
        assert '"system_prompt": system_prompt' in block, (
            "ensemble batch path is missing system_prompt in its prompt_params_per_model dict"
        )
