"""
Tests for the description -> survey_question mirror in classify().

`description=` is the canonical, documented way to give data context, but
downstream prompt assembly (the "Context:" line, step-back, and
categories="auto") keys off `survey_question`. Before the mirror, callers
passing only description= — which includes the cat-survey/cat-pol/cat-web/
cat-ademic classify wrappers — silently lost their context framing in text
prompts.
"""

import warnings
from unittest.mock import patch

import pandas as pd

import cat_stack


CATEGORIES = ["Employment", "Family", "Other"]


def _classify(**kwargs):
    dummy = pd.DataFrame({"input_data": ["a"]})
    defaults = dict(
        input_data=["a"],
        categories=CATEGORIES,
        api_key="k",
        user_model="gpt-4o",
        model_source="openai",
        add_other=False,
        check_verbosity=False,
        json_formatter=False,
    )
    defaults.update(kwargs)
    with patch("cat_stack.classify.classify_ensemble", return_value=dummy) as m:
        cat_stack.classify(**defaults)
    return m.call_args.kwargs


class TestDescriptionMirror:
    def test_description_reaches_survey_question_channel(self):
        kw = _classify(description="Why did you move?")
        assert kw["survey_question"] == "Why did you move?"
        assert kw["input_description"] == "Why did you move?"

    def test_description_alone_emits_no_deprecation_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            _classify(description="Why did you move?")

    def test_survey_question_still_mirrors_to_description(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            kw = _classify(survey_question="Why did you move?")
        assert kw["survey_question"] == "Why did you move?"
        assert kw["input_description"] == "Why did you move?"

    def test_both_passed_keeps_channels_distinct(self):
        # e.g. cat-vader: description=social context, survey_question=feed question
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            kw = _classify(description="Bluesky posts", survey_question="What topics?")
        assert kw["survey_question"] == "What topics?"
        assert kw["input_description"] == "Bluesky posts"

    def test_neither_passed_stays_empty(self):
        kw = _classify()
        assert kw["survey_question"] == ""
        assert kw["input_description"] == ""
