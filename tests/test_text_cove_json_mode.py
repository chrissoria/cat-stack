"""
Tests for task #32 (H-COVE): text-mode CoVe Step 4 now requests JSON
output where the provider supports it.

`calls/CoVe.py` is part of the public API surface (re-exported via
`calls/__init__.py.__all__`). The four `chain_of_verification_*`
functions were missing per-provider JSON-mode hints on Step 4, which
meant any caller invoking them directly (it's a public option) got
free-form text where downstream `extract_json()` expected JSON-shaped
output.

Per-provider mechanism (matching calls/pdf_CoVe.py and
calls/image_CoVe.py):
  - OpenAI:    response_format={"type": "json_object"}
  - Mistral:   response_format={"type": "json_object"}
  - Google:    generationConfig.responseMimeType = "application/json"
  - Anthropic: no JSON-mode kwarg available (Anthropic's messages API
               has no response_format; the prompt itself instructs JSON
               output, consistent with image_CoVe.py / pdf_CoVe.py
               anthropic variants).
"""

from unittest.mock import MagicMock

from cat_stack.calls.CoVe import (
    chain_of_verification_openai,
    chain_of_verification_google,
    chain_of_verification_mistral,
)


def _no_op_remove_numbering(s):
    return s.strip()


class TestOpenAIStep4JsonMode:
    def test_step4_uses_response_format_json_object(self):
        """The third call to chat.completions.create() is Step 4; it must
        include response_format={"type": "json_object"}."""
        client = MagicMock()
        # Mock chat.completions.create to return objects with the SDK shape
        client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="ok"))]
        )

        chain_of_verification_openai(
            initial_reply='{"1":"1"}',
            step2_prompt="Q gen: <<INITIAL_REPLY>>",
            step3_prompt="Answer: <<QUESTION>>",
            step4_prompt="Final: <<INITIAL_REPLY>> <<VERIFICATION_QA>>",
            client=client,
            user_model="gpt-4o",
            creativity=0,
            remove_numbering=_no_op_remove_numbering,
        )

        # Step 4 is the last call. Verify response_format is set.
        last_call = client.chat.completions.create.call_args_list[-1]
        assert last_call.kwargs.get("response_format") == {"type": "json_object"}, (
            f"Step 4 should request JSON mode; got kwargs: {last_call.kwargs}"
        )

    def test_step2_and_step3_do_NOT_use_json_mode(self):
        """Steps 2 (question generation) and 3 (per-question answers)
        are free-text; only Step 4 should be JSON-mode."""
        client = MagicMock()
        client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Question 1?"))]
        )

        chain_of_verification_openai(
            initial_reply='{"1":"1"}',
            step2_prompt="Q gen: <<INITIAL_REPLY>>",
            step3_prompt="Answer: <<QUESTION>>",
            step4_prompt="Final: <<INITIAL_REPLY>> <<VERIFICATION_QA>>",
            client=client,
            user_model="gpt-4o",
            creativity=0,
            remove_numbering=_no_op_remove_numbering,
        )

        # First call (Step 2) → no response_format
        assert "response_format" not in client.chat.completions.create.call_args_list[0].kwargs


class TestMistralStep4JsonMode:
    def test_step4_uses_response_format_json_object(self):
        client = MagicMock()
        client.chat.complete.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="ok"))]
        )

        chain_of_verification_mistral(
            initial_reply='{"1":"1"}',
            step2_prompt="Q gen: <<INITIAL_REPLY>>",
            step3_prompt="Answer: <<QUESTION>>",
            step4_prompt="Final: <<INITIAL_REPLY>> <<VERIFICATION_QA>>",
            client=client,
            user_model="mistral-large-latest",
            creativity=0,
            remove_numbering=_no_op_remove_numbering,
        )

        last_call = client.chat.complete.call_args_list[-1]
        assert last_call.kwargs.get("response_format") == {"type": "json_object"}


class TestGoogleStep4JsonMode:
    def test_step4_uses_response_mime_type_application_json(self):
        """Google uses responseMimeType inside generationConfig, NOT
        response_format. Verify Step 4's payload sets it."""
        calls = []

        def fake_make_google_request(url, headers, payload):
            calls.append(payload)
            return {
                "candidates": [
                    {"content": {"parts": [{"text": "Q1?\nQ2?"}]}}
                ]
            }

        chain_of_verification_google(
            initial_reply='{"1":"1"}',
            prompt="What is X?",
            step2_prompt="Q gen: <<INITIAL_REPLY>>",
            step3_prompt="Answer: <<QUESTION>>",
            step4_prompt="Final: <<PROMPT>> <<INITIAL_REPLY>> <<VERIFICATION_QA>>",
            url="https://example.test",
            headers={},
            creativity=0,
            remove_numbering=_no_op_remove_numbering,
            make_google_request=fake_make_google_request,
        )

        # Step 4 is the LAST call. Verify generationConfig.responseMimeType.
        step4_payload = calls[-1]
        gen_cfg = step4_payload.get("generationConfig", {})
        assert gen_cfg.get("responseMimeType") == "application/json", (
            f"Step 4 should request JSON via responseMimeType; got: {gen_cfg}"
        )
