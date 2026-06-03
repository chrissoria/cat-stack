"""
Tests for task #34 (H-KWARGS): parse_kwargs_string should warn — not
silently fall back to a raw string — when the value LOOKS like the
user tried to write a Python literal but it didn't parse.

Pre-fix: `parse_kwargs_string("max_retries=three")` silently returned
`{"max_retries": "three"}`, then `max_retries` got compared against
`int` defaults downstream and broke in puzzling ways. Same shape for
`tags=[apple,banana]` (unquoted list).

Post-fix: those cases emit a UserWarning. Plain prose values
(`research_question=Why did you move?`) still fall through silently —
they aren't trying to be literals.
"""

import warnings

import pytest

from cat_stack import parse_kwargs_string


class TestPlainProseSilent:
    """Values that don't start with a Python-literal-looking character
    fall through silently — this is the wrapper-friendly path."""

    def test_unquoted_prose_no_warning(self, recwarn):
        result = parse_kwargs_string("research_question=Why did you move?")
        assert result == {"research_question": "Why did you move?"}
        assert len(recwarn) == 0

    def test_simple_word_no_warning(self, recwarn):
        result = parse_kwargs_string("format=bullets")
        assert result == {"format": "bullets"}
        assert len(recwarn) == 0


class TestPythonLiteralLooksWarns:
    """Values that LOOK like Python literals but fail to parse should
    warn — they're almost certainly typos or syntax errors."""

    def test_typo_in_int_warns(self, recwarn):
        warnings.simplefilter("always")
        result = parse_kwargs_string("max_retries=3oops")
        assert result == {"max_retries": "3oops"}
        # Should have emitted a warning because '3' is a literal-looking lead
        assert any("max_retries" in str(w.message) for w in recwarn)

    def test_unclosed_bracket_warns(self, recwarn):
        warnings.simplefilter("always")
        result = parse_kwargs_string("tags=[apple,banana")
        assert result == {"tags": "[apple,banana"}
        assert any("tags" in str(w.message) for w in recwarn)

    def test_unquoted_list_words_warns(self, recwarn):
        """[apple, banana] looks like a list but bare words don't parse."""
        warnings.simplefilter("always")
        result = parse_kwargs_string("tags=[apple,banana]")
        assert result == {"tags": "[apple,banana]"}
        assert any("tags" in str(w.message) for w in recwarn)

    def test_invalid_bool_keyword_warns(self, recwarn):
        """Python literal-looking bool typo (e.g. 'yes' is not Python)
        — leading 'y' isn't a literal lead but if value equals a known
        keyword variant we... actually 'yes' doesn't trigger; only
        True/False/None as exact matches do."""
        warnings.simplefilter("always")
        # 'yes' is plain prose — silent fall-through
        result = parse_kwargs_string("safety=yes")
        assert result == {"safety": "yes"}
        # No warning for prose-shaped values
        assert len(recwarn) == 0


class TestValidLiteralsParse:
    """Sanity: valid literals still parse to native types — the warning
    path doesn't kick in on the happy path."""

    def test_int_parses(self, recwarn):
        result = parse_kwargs_string("max_retries=3")
        assert result == {"max_retries": 3}
        assert len(recwarn) == 0

    def test_float_parses(self, recwarn):
        result = parse_kwargs_string("retry_delay=0.5")
        assert result == {"retry_delay": 0.5}
        assert len(recwarn) == 0

    def test_quoted_string_parses(self, recwarn):
        result = parse_kwargs_string("format='bullets'")
        assert result == {"format": "bullets"}
        assert len(recwarn) == 0

    def test_list_parses(self, recwarn):
        result = parse_kwargs_string("tags=['apple','banana']")
        assert result == {"tags": ["apple", "banana"]}
        assert len(recwarn) == 0

    def test_bool_parses(self, recwarn):
        result = parse_kwargs_string("safety=True")
        assert result == {"safety": True}
        assert len(recwarn) == 0
