"""
Tests for the categories="auto" path in classify_ensemble.

Regression test for a bug where the lazy import at the start of the
auto-categories branch did `from .main import extract` — but main.py
doesn't exist; the function lives in extract.py. Every call to
classify(categories="auto", ...) raised ModuleNotFoundError.
"""

from unittest.mock import patch, MagicMock

import pytest

from cat_stack.classify import classify_ensemble


def _fake_extract_result():
    return {
        "top_categories": ["Positive", "Negative"],
        "counts_df": MagicMock(),
        "raw_top_text": "",
    }


def _fake_model_configs():
    return [
        {
            "model": "gpt-4o",
            "provider": "openai",
            "api_key": "fake",
            "sanitized_name": "gpt_4o",
        }
    ]


class TestCategoriesAuto:
    @patch("cat_stack.extract.extract")
    @patch("cat_stack.text_functions_ensemble.prepare_model_configs")
    def test_auto_categories_does_not_raise_module_not_found(
        self, mock_prepare, mock_extract
    ):
        """classify_ensemble(categories='auto') must clear the lazy import."""
        mock_extract.return_value = _fake_extract_result()
        mock_prepare.return_value = _fake_model_configs()

        try:
            classify_ensemble(
                input_data=["text1", "text2"],
                categories="auto",
                survey_question="What is your favorite thing?",
                models=[("gpt-4o", "openai", "fake-key")],
            )
        except ModuleNotFoundError as e:
            raise AssertionError(
                f"classify_ensemble(categories='auto') regressed: {e}"
            )
        except Exception:
            # Downstream classification failures aren't what we're testing
            pass

        assert mock_extract.called, "extract() not invoked — auto branch skipped"
        kwargs = mock_extract.call_args.kwargs
        assert kwargs["input_data"] == ["text1", "text2"]
        assert kwargs["description"] == "What is your favorite thing?"
        assert kwargs["input_type"] == "text"
        assert kwargs["user_model"] == "gpt-4o"

    @patch("cat_stack.extract.extract")
    @patch("cat_stack.text_functions_ensemble.prepare_model_configs")
    def test_auto_categories_requires_survey_question_for_text(
        self, mock_prepare, mock_extract
    ):
        """Text input with empty survey_question must still raise TypeError."""
        mock_extract.return_value = _fake_extract_result()
        mock_prepare.return_value = _fake_model_configs()

        with pytest.raises(TypeError, match="survey_question is required"):
            classify_ensemble(
                input_data=["text1", "text2"],
                categories="auto",
                survey_question="",
                models=[("gpt-4o", "openai", "fake-key")],
            )
