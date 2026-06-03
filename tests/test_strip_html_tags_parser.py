"""
Tests for H-HTML (#36): strip_html_tags now uses html.parser.HTMLParser
instead of regex, fixing three concrete failure modes:

  1. Attribute values containing `>` — the old regex `[^>]*` terminated
     at the `>` inside a quoted attribute value, leaving the rest of the
     tag as visible text.
  2. Missing void elements (br, hr, area, etc.) — the old hardcoded
     list only handled input/meta/link/img.
  3. Nested same-tag content — the non-greedy regex matched the wrong
     close tag, leaving the outer close as literal text.

Happy-path behavior (junk-tag stripping, entity decoding, whitespace
collapse) is preserved.
"""

import pytest

from cat_stack import strip_html_tags


class TestBugsTheNewParserFixes:
    """Direct empirical verification of the three failure modes."""

    def test_attribute_with_gt_does_not_leak_into_text(self):
        """`<a href="https://x.y/?q=>foo">label</a>` previously left
        `foo">label` visible because the regex `[^>]*` terminated at
        the `>` inside the quoted href value. Now the parser tokenizes
        the attribute correctly."""
        html = '<p>before <a href="https://x.y/?q=>foo">label</a> after</p>'
        result = strip_html_tags(html)
        assert "foo" not in result, (
            f"href attribute value should be stripped; got: {result!r}"
        )
        assert "label" in result
        assert "before" in result and "after" in result

    def test_void_br_separates_adjacent_text(self):
        """The old hardcoded void list missed `br`. A real parser knows
        it's a void element, so `foo<br>bar` becomes `foo bar`."""
        html = "<p>foo<br>bar</p>"
        result = strip_html_tags(html)
        assert "foo" in result
        assert "bar" in result
        assert "foobar" not in result, (
            f"<br> should separate adjacent text; got: {result!r}"
        )

    def test_void_hr_does_not_leak_as_text(self):
        """Same issue for hr/area/base/col/embed/source/track/wbr — all
        missing from the old hardcoded list."""
        for void in ("hr", "area", "base", "col", "embed", "source", "track", "wbr"):
            html = f"<p>before<{void}>after</p>"
            result = strip_html_tags(html)
            assert void not in result.lower(), (
                f"void element <{void}> leaked into output: {result!r}"
            )

    def test_html_comments_are_stripped(self):
        """HTML comments with `>` inside (e.g. `<!-- > -->`) confused the
        old `[^>]*` regex into partial matches. Parser dispatches them
        cleanly to handle_comment (default: do nothing → stripped)."""
        html = "<p>before</p><!-- inline > comment with > inside --><p>after</p>"
        result = strip_html_tags(html)
        assert "before" in result
        assert "after" in result
        assert "comment" not in result, (
            f"comment text should be stripped; got: {result!r}"
        )


class TestHappyPathPreserved:
    def test_strips_simple_tags(self):
        assert strip_html_tags("<p>Hello, <b>world</b>!</p>") == "Hello, world !"

    def test_decodes_entities(self):
        assert strip_html_tags("<p>Tom &amp; Jerry</p>") == "Tom & Jerry"
        assert strip_html_tags("<p>&lt;not a tag&gt;</p>") == "<not a tag>"
        assert strip_html_tags("<p>caf&eacute;</p>") == "café"

    def test_collapses_whitespace(self):
        assert strip_html_tags("<p>foo\n\n\t bar  </p>").strip() == "foo bar"

    def test_strips_junk_tag_contents(self):
        html = (
            "<html><head><style>body { color: red; }</style></head>"
            "<body><nav>menu</nav><main>real content</main>"
            "<footer>copyright</footer></body></html>"
        )
        result = strip_html_tags(html)
        assert "real content" in result
        assert "color" not in result      # style content stripped
        assert "menu" not in result        # nav content stripped
        assert "copyright" not in result   # footer content stripped

    def test_strips_script_with_complex_content(self):
        """Common real-world script body shouldn't leak into output."""
        html = (
            "<p>real text</p>"
            "<script>var x = {a: 1, b: '</p>'}; window.x = x;</script>"
            "<p>more real text</p>"
        )
        result = strip_html_tags(html)
        assert "real text" in result
        assert "more real text" in result
        assert "window.x" not in result
        assert "var x" not in result

    def test_empty_input_returns_empty(self):
        assert strip_html_tags("") == ""

    def test_plain_text_passthrough(self):
        assert strip_html_tags("just plain text") == "just plain text"

    def test_malformed_html_does_not_raise(self):
        """html.parser is permissive; some malformed shapes still
        shouldn't raise. The fallback re.sub strip exists for paranoia."""
        for bad in (
            "<p>unclosed paragraph",
            "<>>>>",
            "<p><b>nested unclosed",
            "x < y && y > z",  # ambiguous: < and > as math
        ):
            # Should not raise
            result = strip_html_tags(bad)
            assert isinstance(result, str)


class TestPublicAPISignatureUnchanged:
    """cat-web sibling re-exports strip_html_tags. The signature must
    stay (str) -> str so the import surface doesn't break."""

    def test_signature_is_str_to_str(self):
        result = strip_html_tags("<p>x</p>")
        assert isinstance(result, str)

    def test_is_importable_from_cat_stack_root(self):
        from cat_stack import strip_html_tags as imported  # noqa: F401
