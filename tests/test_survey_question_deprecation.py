"""
Tests for the soft deprecation of `survey_question=` across the cat-stack
public entry points. `description=` is now the canonical content-neutral
parameter; `survey_question=` keeps working but emits a DeprecationWarning
and is mirrored into `description` when `description` is empty.

Entry points covered:
  - cat_stack.classify
  - cat_stack.extract
  - cat_stack.prompt_tune

(explore() doesn't accept survey_question; summarize() never did.)
"""

import warnings
from unittest.mock import patch, MagicMock

import pytest

import cat_stack


# ── extract ──────────────────────────────────────────────────────────────

class TestExtractDeprecation:
    @patch("cat_stack.extract.collapse_themes", return_value=["A"])
    @patch("cat_stack.extract.explore_common_categories")
    def test_survey_question_emits_deprecation_warning(self, mock_explore, mock_collapse):
        mock_explore.return_value = ["A", "B"]  # raw labels (return_raw=True contract)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cat_stack.extract(
                input_data=["x"],
                api_key="fake",
                survey_question="Why did you move?",
                input_type="text",
                user_model="gpt-4o-mini",
                model_source="openai",
                iterations=1,
            )
        dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert dep, f"expected DeprecationWarning, got: {[str(w.message) for w in caught]}"
        assert "survey_question" in str(dep[0].message)

    @patch("cat_stack.extract.collapse_themes", return_value=["A"])
    @patch("cat_stack.extract.explore_common_categories")
    def test_survey_question_value_mirrored_to_inner_survey_question_kw(self, mock_explore, mock_collapse):
        """The legacy value still has to flow into the downstream call —
        explore_common_categories still accepts survey_question= as its kwarg."""
        mock_explore.return_value = ["A", "B"]  # raw labels (return_raw=True contract)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cat_stack.extract(
                input_data=["x"],
                api_key="fake",
                survey_question="Why did you move?",
                input_type="text",
                user_model="gpt-4o-mini",
                model_source="openai",
                iterations=1,
            )
        passed = mock_explore.call_args.kwargs
        assert passed["survey_question"] == "Why did you move?", (
            f"survey_question value lost on the way down: {passed!r}"
        )

    @patch("cat_stack.extract.collapse_themes", return_value=["A"])
    @patch("cat_stack.extract.explore_common_categories")
    def test_description_only_does_not_warn(self, mock_explore, mock_collapse):
        mock_explore.return_value = ["A", "B"]  # raw labels (return_raw=True contract)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cat_stack.extract(
                input_data=["x"],
                api_key="fake",
                description="Why did you move?",
                input_type="text",
                user_model="gpt-4o-mini",
                model_source="openai",
                iterations=1,
            )
        dep = [w for w in caught if issubclass(w.category, DeprecationWarning)
               and "survey_question" in str(w.message)]
        assert not dep, "no warning expected when only description= is used"

    @patch("cat_stack.extract.collapse_themes", return_value=["A"])
    @patch("cat_stack.extract.explore_common_categories")
    def test_description_wins_when_both_set(self, mock_explore, mock_collapse):
        mock_explore.return_value = ["A", "B"]  # raw labels (return_raw=True contract)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cat_stack.extract(
                input_data=["x"],
                api_key="fake",
                description="canonical",
                survey_question="legacy",
                input_type="text",
                user_model="gpt-4o-mini",
                model_source="openai",
                iterations=1,
            )
        passed = mock_explore.call_args.kwargs
        # The resolved value passed downstream comes from description.
        assert passed["survey_question"] == "canonical", (
            f"description should win when both are set: {passed!r}"
        )


# ── classify ─────────────────────────────────────────────────────────────

class TestClassifyDeprecation:
    @patch("cat_stack.classify.classify_ensemble")
    def test_survey_question_emits_deprecation_warning(self, mock_ensemble):
        mock_ensemble.return_value = MagicMock()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cat_stack.classify(
                input_data=["x"],
                categories=["a", "b"],
                api_key="fake",
                user_model="gpt-4o-mini",
                model_source="openai",
                survey_question="Why did you move?",
                add_other=False,
                check_verbosity=False,
            )
        dep = [w for w in caught if issubclass(w.category, DeprecationWarning)
               and "survey_question" in str(w.message)]
        assert dep, f"expected DeprecationWarning for survey_question, got: {[str(w.message) for w in caught]}"

    @patch("cat_stack.classify.classify_ensemble")
    def test_classify_passes_canonical_value_downstream(self, mock_ensemble):
        mock_ensemble.return_value = MagicMock()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cat_stack.classify(
                input_data=["x"],
                categories=["a", "b"],
                api_key="fake",
                user_model="gpt-4o-mini",
                model_source="openai",
                survey_question="legacy value",
                add_other=False,
                check_verbosity=False,
            )
        kwargs = mock_ensemble.call_args.kwargs
        # The internal call must still receive survey_question (the
        # text_functions_ensemble code path uses it) — value preserved.
        assert kwargs.get("survey_question") == "legacy value", (
            f"survey_question not forwarded: {kwargs!r}"
        )
        # And input_description should also be set (mirrored from the
        # deprecated value when description was empty).
        assert kwargs.get("input_description") == "legacy value", (
            f"description not mirrored from survey_question: {kwargs!r}"
        )

    @patch("cat_stack.classify.classify_ensemble")
    def test_description_only_no_warning(self, mock_ensemble):
        mock_ensemble.return_value = MagicMock()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cat_stack.classify(
                input_data=["x"],
                categories=["a", "b"],
                api_key="fake",
                user_model="gpt-4o-mini",
                model_source="openai",
                description="canonical description",
                add_other=False,
                check_verbosity=False,
            )
        dep = [w for w in caught if issubclass(w.category, DeprecationWarning)
               and "survey_question" in str(w.message)]
        assert not dep, "no warning when only description= is used"


# ── prompt_tune ──────────────────────────────────────────────────────────

class TestPromptTuneDeprecation:
    def test_survey_question_emits_deprecation_warning(self):
        """We can't easily mock the whole prompt_tune internals, so we just
        check that the warning fires before the function progresses far
        enough to need API access. The warning is the very first thing the
        function does after the docstring."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with pytest.raises(Exception):
                # Will fail somewhere downstream once it tries to do real
                # work, but the warning fires first.
                cat_stack.prompt_tune(
                    input_data=["x"],
                    categories=["a", "b"],
                    api_key="fake",
                    user_model="gpt-4o-mini",
                    model_source="openai",
                    survey_question="legacy",
                    sample_size=1,
                    max_iterations=1,
                    ui="terminal",
                )
        dep = [w for w in caught if issubclass(w.category, DeprecationWarning)
               and "survey_question" in str(w.message)]
        assert dep, f"expected DeprecationWarning, got: {[str(w.message) for w in caught]}"
