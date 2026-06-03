r"""
Tests for _extract_balanced_json — the stdlib helper that replaced
regex.findall(r'\{(?:[^{}]|(?R))*\}', ...).

The helper backs extract_json() in text_functions.py plus the JSON
extraction in image_functions / pdf_functions / text_functions_ensemble.
Previously these used the `regex` module's recursive subpattern, which had
two issues: it didn't know about JSON string semantics (would truncate at
the first `}` inside a string), and one call site forgot to import it
(text_functions_ensemble.py:1412, the C4 bug).
"""

import pytest

from cat_stack._utils import _extract_balanced_json
from cat_stack.text_functions import extract_json


class TestExtractBalancedJson:
    def test_simple_object(self):
        assert _extract_balanced_json('{"a": 1}') == '{"a": 1}'

    def test_with_surrounding_text(self):
        text = 'Here is the result: {"summary": "hello"} and then trailing prose.'
        assert _extract_balanced_json(text) == '{"summary": "hello"}'

    def test_nested_object(self):
        text = '{"outer": {"inner": "value"}}'
        assert _extract_balanced_json(text) == text

    def test_deeply_nested(self):
        text = '{"a": {"b": {"c": {"d": 1}}}}'
        assert _extract_balanced_json(text) == text

    def test_returns_first_top_level_object(self):
        text = '{"first": 1} extra {"second": 2}'
        assert _extract_balanced_json(text) == '{"first": 1}'

    def test_empty_object(self):
        assert _extract_balanced_json('{}') == '{}'

    def test_no_braces(self):
        assert _extract_balanced_json('just prose with no JSON') is None

    def test_only_open_brace(self):
        assert _extract_balanced_json('text { unmatched') is None

    def test_only_close_brace(self):
        assert _extract_balanced_json('text } unmatched') is None

    def test_string_with_brace_inside_not_truncated(self):
        """Regression: regex.findall(r'\\{(?:[^{}]|(?R))*\\}', ...) would
        truncate this at the first } inside the string value. The stdlib
        helper is string-aware and returns the full object."""
        text = '{"summary": "see Fig {3} for details"}'
        assert _extract_balanced_json(text) == text

    def test_string_with_open_brace_inside(self):
        text = '{"caption": "expression {y} = f(x)"}'
        assert _extract_balanced_json(text) == text

    def test_escaped_quote_in_string(self):
        text = r'{"key": "he said \"hi\" and left"}'
        assert _extract_balanced_json(text) == text

    def test_escaped_backslash_in_string(self):
        text = r'{"path": "C:\\Users\\foo"}'
        assert _extract_balanced_json(text) == text

    def test_none_input(self):
        assert _extract_balanced_json(None) is None

    def test_empty_string(self):
        assert _extract_balanced_json('') is None

    def test_unbalanced_extra_close(self):
        """Extra closing braces at depth 0 are ignored; the first balanced
        object still returns."""
        text = '}}{"key": "value"}}'
        assert _extract_balanced_json(text) == '{"key": "value"}'

    def test_multiline_object(self):
        text = '''Here:
        {
            "a": 1,
            "b": 2
        }
        end'''
        result = _extract_balanced_json(text)
        assert result is not None
        assert '"a": 1' in result
        assert '"b": 2' in result


class TestExtractJsonBackwardCompat:
    """extract_json() is in the public API. Its observable behavior should
    match the old regex-backed implementation for typical inputs."""

    def test_classification_json_passthrough(self):
        reply = 'The answer is {"1":"1","2":"0","3":"1"}'
        # Whitespace stripping is part of extract_json's contract
        assert extract_json(reply) == '{"1":"1","2":"0","3":"1"}'

    def test_returns_error_sentinel_when_no_json(self):
        assert extract_json("no json here") == '{"1":"e"}'

    def test_none_input(self):
        assert extract_json(None) == '{"1":"e"}'

    def test_strips_brackets_inside_match(self):
        # extract_json strips [ and ] from the match
        reply = '{"1":"[1]","2":"0"}'
        result = extract_json(reply)
        assert '[' not in result
        assert ']' not in result
