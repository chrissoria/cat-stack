"""
Tests for gather_stepback_insights and its callers.

Regression test for a bug where summarize_ensemble called
gather_stepback_insights with kwargs (context=, question=) that the function
didn't accept, causing TypeError on every summarize(step_back_prompt=True).

After the fix:
  - gather_stepback_insights takes a pre-built stepback_prompt
  - classify_ensemble templates the prompt around survey_question
  - summarize_ensemble templates the prompt around its summarization goal
"""

from unittest.mock import patch

import pytest

from cat_stack.text_functions_ensemble import (
    gather_stepback_insights,
    summarize_ensemble,
)


class TestGatherStepbackInsights:
    @patch("cat_stack.text_functions_ensemble._get_stepback_insight")
    def test_accepts_stepback_prompt(self, mock_get_insight):
        """The new signature takes stepback_prompt as the prepared question."""
        mock_get_insight.return_value = ("Some insight text", True)

        configs = [
            {
                "model": "gpt-4o",
                "provider": "openai",
                "api_key": "fake",
                "creativity": None,
            }
        ]
        prompt = "What are the underlying factors when classifying customer feedback?"
        result = gather_stepback_insights(configs, stepback_prompt=prompt, creativity=0.5)

        assert "gpt-4o" in result
        prompt_used, insight = result["gpt-4o"]
        assert prompt_used == prompt
        assert insight == "Some insight text"

    def test_rejects_empty_prompt(self):
        """Empty stepback_prompt must raise TypeError with a clear message."""
        with pytest.raises(TypeError, match="stepback_prompt is required"):
            gather_stepback_insights(
                [{"model": "x", "provider": "openai", "api_key": "y"}],
                stepback_prompt="",
            )

    def test_rejects_old_survey_question_kwarg(self):
        """survey_question is no longer accepted as a kwarg (breaking change)."""
        with pytest.raises(TypeError):
            gather_stepback_insights(
                [{"model": "x", "provider": "openai", "api_key": "y"}],
                survey_question="Why did you move?",
            )

    @patch("cat_stack.text_functions_ensemble._get_stepback_insight")
    def test_skips_ollama_models(self, mock_get_insight):
        """Ollama models are skipped — their stepback path is different."""
        mock_get_insight.return_value = ("insight", True)

        configs = [
            {"model": "gpt-4o", "provider": "openai", "api_key": "k1", "creativity": None},
            {"model": "llama3.1", "provider": "ollama", "api_key": None, "creativity": None},
        ]
        result = gather_stepback_insights(configs, "test prompt")

        assert "gpt-4o" in result
        assert "llama3.1" not in result
        assert mock_get_insight.call_count == 1


class TestSummarizeStepbackRegression:
    """Regression test: summarize(step_back_prompt=True) used to TypeError
    on the gather_stepback_insights call with unsupported context=/question= kwargs."""

    @patch("cat_stack.text_functions_ensemble._get_stepback_insight")
    @patch("cat_stack.text_functions_ensemble.prepare_model_configs")
    def test_summarize_step_back_prompt_does_not_typeerror(
        self, mock_prepare, mock_get_insight
    ):
        mock_get_insight.return_value = ("insight", True)
        mock_prepare.return_value = [
            {
                "model": "gpt-4o",
                "provider": "openai",
                "api_key": "fake",
                "sanitized_name": "gpt_4o",
                "creativity": None,
            }
        ]

        try:
            summarize_ensemble(
                input_data=["Some text to summarize"],
                models=[("gpt-4o", "openai", "fake-key")],
                step_back_prompt=True,
                input_description="customer feedback",
            )
        except TypeError as e:
            msg = str(e)
            if "gather_stepback_insights" in msg or "unexpected keyword" in msg:
                raise AssertionError(
                    f"summarize_ensemble(step_back_prompt=True) regressed: {e}"
                )
        except Exception:
            # Downstream classification failures aren't what we're testing
            pass

        assert mock_get_insight.called, "stepback path not reached"

    @patch("cat_stack.text_functions_ensemble._get_stepback_insight")
    @patch("cat_stack.text_functions_ensemble.prepare_model_configs")
    def test_summarize_stepback_prompt_includes_focus(
        self, mock_prepare, mock_get_insight
    ):
        """The summarize caller should incorporate the `focus` kwarg into the prompt."""
        mock_get_insight.return_value = ("insight", True)
        mock_prepare.return_value = [
            {
                "model": "gpt-4o",
                "provider": "openai",
                "api_key": "fake",
                "sanitized_name": "gpt_4o",
                "creativity": None,
            }
        ]

        try:
            summarize_ensemble(
                input_data=["Some text"],
                models=[("gpt-4o", "openai", "fake-key")],
                step_back_prompt=True,
                focus="emotional content",
            )
        except Exception:
            pass

        # Inspect the prompt that was actually sent
        assert mock_get_insight.called
        # 2nd positional arg of _get_stepback_insight is the stepback prompt
        prompt_sent = mock_get_insight.call_args.args[1]
        assert "focus on emotional content" in prompt_sent
