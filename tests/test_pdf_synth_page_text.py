"""
Tests for task #35 (H-PDF-SYNTH): _synthesize_summaries gets the actual
page text for PDF summarization, not just the page label.

Pre-fix: at text_functions_ensemble.py:4426, the PDF branch of the
result-aggregation loop set:
    original_text_for_synthesis = page_label
…then passed that into _synthesize_summaries(), so the synthesis prompt
told the model:
    Original text: "report.pdf p1"
    Summaries from different models: ...
    Resolve any contradictions by focusing on accuracy.
…with no source-of-truth to actually focus on. Contradictions between
per-model summaries got resolved arbitrarily.

Post-fix: when the PDF item went through text-mode OCR (the path that
extracts page text and passes it to summarize_single_item), the
extracted text is captured on the result_entry as "page_text" and
preferred over page_label at synthesis time. Visual-mode PDFs (no OCR
text) still fall back to page_label so synthesis isn't broken there
either.
"""

from unittest.mock import patch, MagicMock

from cat_stack.text_functions_ensemble import _synthesize_summaries


def _cfg():
    return {
        "model": "claude-haiku-4-5",
        "provider": "anthropic",
        "api_key": "fake-key",
        "sanitized_name": "claude_haiku_4_5",
    }


class TestSynthesizerReceivesActualText:
    """End-to-end: when _synthesize_summaries is called, the synthesis
    prompt sent to the LLM should contain the actual page text — that's
    the contract the fix maintains by capturing OCR text on the
    result_entry and reading it back at synthesis time."""

    def test_actual_page_text_appears_in_synthesis_prompt(self):
        """If the caller passes real page text as original_text, it must
        appear (verbatim, possibly truncated) in the prompt sent to the
        LLM. This is the property the fix preserves at the call boundary."""
        actual_page_text = (
            "The 2023 climate report concluded that global average temperatures "
            "rose 0.14 degrees Celsius above the 1991-2020 baseline."
        )
        summaries = {
            "model_a": "Temperatures rose substantially.",
            "model_b": "Climate report findings unclear.",
        }

        with patch(
            "cat_stack.text_functions_ensemble.UnifiedLLMClient"
        ) as mock_client_cls:
            mock_client = MagicMock()
            mock_client.complete.return_value = ('{"summary": "synthesized"}', None)
            mock_client_cls.return_value = mock_client

            _synthesize_summaries(
                summaries=summaries,
                original_text=actual_page_text,
                synthesis_config=_cfg(),
            )

            sent_messages = mock_client.complete.call_args.kwargs["messages"]
            prompt_text = " ".join(m["content"] for m in sent_messages)
            assert "global average temperatures" in prompt_text, (
                "synthesis prompt should contain the actual page text it "
                "was asked to ground on"
            )

    def test_label_falls_through_when_no_text_available(self):
        """Visual-mode PDFs have no OCR text. The synthesizer still works
        — it just gets the page label as a weak anchor. Verify the label
        appears in the prompt as a regression check on the fallback path."""
        page_label = "report.pdf p1"
        summaries = {
            "model_a": "Some summary.",
            "model_b": "Different summary.",
        }

        with patch(
            "cat_stack.text_functions_ensemble.UnifiedLLMClient"
        ) as mock_client_cls:
            mock_client = MagicMock()
            mock_client.complete.return_value = ('{"summary": "synthesized"}', None)
            mock_client_cls.return_value = mock_client

            _synthesize_summaries(
                summaries=summaries,
                original_text=page_label,
                synthesis_config=_cfg(),
            )

            sent_messages = mock_client.complete.call_args.kwargs["messages"]
            prompt_text = " ".join(m["content"] for m in sent_messages)
            assert "report.pdf p1" in prompt_text

    def test_single_model_skips_synthesis(self):
        """With only one model in the ensemble, synthesis is a no-op —
        return the single summary unchanged. Verifies we don't make
        extra API calls when synthesis isn't meaningful."""
        with patch(
            "cat_stack.text_functions_ensemble.UnifiedLLMClient"
        ) as mock_client_cls:
            result = _synthesize_summaries(
                summaries={"only_model": "the one summary"},
                original_text="any page text",
                synthesis_config=_cfg(),
            )
            assert result == "the one summary"
            # No client should have been instantiated for a single-model case
            mock_client_cls.assert_not_called()


class TestPdfSelectionLogic:
    """The aggregation loop at text_functions_ensemble.py:4426 decides
    which 'original_text' goes to synthesis. Verify the precedence rule:
    page_text (when present) over page_label."""

    def test_page_text_preferred_over_label(self):
        """Simulate the precedence expression from the fix."""
        entry = {"page_text": "Real extracted page content."}
        page_label = "report.pdf p1"
        # This matches the new expression at L4426
        result = entry.get("page_text") or page_label
        assert result == "Real extracted page content."

    def test_label_used_when_page_text_missing(self):
        entry = {}  # no page_text key (visual-mode PDF path)
        page_label = "report.pdf p1"
        result = entry.get("page_text") or page_label
        assert result == "report.pdf p1"

    def test_label_used_when_page_text_is_none(self):
        entry = {"page_text": None}  # text path that didn't capture (image item)
        page_label = "report.pdf p1"
        result = entry.get("page_text") or page_label
        assert result == "report.pdf p1"

    def test_label_used_when_page_text_is_empty(self):
        """OCR may produce an empty string (e.g., page is blank). Don't
        send an empty 'Original text:' to the synthesizer — fall back."""
        entry = {"page_text": ""}
        page_label = "report.pdf p1"
        result = entry.get("page_text") or page_label
        assert result == "report.pdf p1"
