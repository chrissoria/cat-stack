# PDF-aware Chain of Verification (CoVe) functions for various LLM providers
# These functions include the PDF page in verification steps for accurate document-based categorization


def pdf_chain_of_verification_openai(
    initial_reply,
    step2_prompt,
    step3_prompt,
    step4_prompt,
    client,  # Deprecated, kept for backward compatibility
    user_model,
    creativity,
    remove_numbering,
    pdf_content,
    api_key=None,
    base_url=None,
):
    """
    Execute Chain of Verification (CoVe) process for PDF pages with OpenAI-compatible
    providers. The PDF page (as image) is included in verification steps for accurate
    assessment. Returns the verified reply or the initial reply if any error occurs.

    Uses direct HTTP requests instead of the OpenAI SDK so the function works for
    any OpenAI-compatible provider (OpenAI, Perplexity, HuggingFace, xAI, ...) given
    the right `base_url`.

    Args:
        pdf_content: The PDF page content in OpenAI format (image_url dict).
        api_key: Bearer token for the provider.
        base_url: Provider root URL (e.g. https://api.openai.com/v1). Defaults to OpenAI.
    """
    import requests

    if api_key is None:
        return initial_reply

    endpoint = (base_url or "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    def make_openai_request(messages, json_mode=False):
        payload = {"model": user_model, "messages": messages}
        if creativity is not None:
            payload["temperature"] = creativity
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        response = requests.post(endpoint, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]

    try:
        # STEP 2: Generate verification questions (text only)
        step2_filled = step2_prompt.replace('<<INITIAL_REPLY>>', initial_reply)
        verification_questions = make_openai_request(
            [{'role': 'user', 'content': step2_filled}]
        )

        # STEP 3: Answer verification questions WITH the PDF page
        questions_list = [
            remove_numbering(q)
            for q in verification_questions.split('\n')
            if q.strip()
        ]
        verification_qa = []
        for question in questions_list:
            step3_filled = step3_prompt.replace('<<QUESTION>>', question)
            message_content = [
                {"type": "text", "text": step3_filled},
                pdf_content,
            ]
            answer = make_openai_request(
                [{'role': 'user', 'content': message_content}]
            )
            verification_qa.append(f"Q: {question}\nA: {answer}")

        # STEP 4: Final corrected categorization WITH the PDF page
        verification_qa_text = "\n\n".join(verification_qa)
        step4_filled = (step4_prompt
            .replace('<<INITIAL_REPLY>>', initial_reply)
            .replace('<<VERIFICATION_QA>>', verification_qa_text))
        final_message_content = [
            {"type": "text", "text": step4_filled},
            pdf_content,
        ]
        verified_reply = make_openai_request(
            [{'role': 'user', 'content': final_message_content}],
            json_mode=True,
        )
        return verified_reply

    except Exception:
        return initial_reply


def pdf_chain_of_verification_anthropic(
    initial_reply,
    step2_prompt,
    step3_prompt,
    step4_prompt,
    client,  # Deprecated, kept for backward compatibility
    user_model,
    creativity,
    remove_numbering,
    pdf_content,
    api_key=None
):
    """
    Execute Chain of Verification (CoVe) process for PDF pages with Anthropic Claude.
    The PDF page is included in verification steps for accurate assessment.
    Returns the verified reply or initial reply if error occurs.

    Uses direct HTTP requests instead of Anthropic SDK.

    Args:
        pdf_content: The PDF page content in Anthropic format (dict with type: "document")
        api_key: Anthropic API key for authentication
    """
    import requests

    if api_key is None:
        return initial_reply

    endpoint = "https://api.anthropic.com/v1/messages"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01"
    }

    def make_anthropic_request(messages, max_tokens=4096):
        """Helper to make Anthropic API requests."""
        payload = {
            "model": user_model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if creativity is not None:
            payload["temperature"] = creativity

        response = requests.post(endpoint, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()

        content = result.get("content", [])
        if content and content[0].get("type") == "text":
            return content[0].get("text", "")
        return ""

    try:
        # STEP 2: Generate verification questions (text only)
        step2_filled = step2_prompt.replace('<<INITIAL_REPLY>>', initial_reply)

        verification_questions = make_anthropic_request(
            [{'role': 'user', 'content': step2_filled}]
        )

        # STEP 3: Answer verification questions WITH the PDF page
        questions_list = [
            remove_numbering(q)
            for q in verification_questions.split('\n')
            if q.strip()
        ]
        verification_qa = []

        for question in questions_list:
            step3_filled = step3_prompt.replace('<<QUESTION>>', question)

            # Include PDF page in the verification question
            message_content = [
                {"type": "text", "text": step3_filled},
                pdf_content
            ]

            answer = make_anthropic_request(
                [{'role': 'user', 'content': message_content}]
            )
            verification_qa.append(f"Q: {question}\nA: {answer}")

        # STEP 4: Final corrected categorization WITH the PDF page
        verification_qa_text = "\n\n".join(verification_qa)

        step4_filled = (step4_prompt
            .replace('<<INITIAL_REPLY>>', initial_reply)
            .replace('<<VERIFICATION_QA>>', verification_qa_text))

        # Include PDF page in final categorization
        final_message_content = [
            {"type": "text", "text": step4_filled},
            pdf_content
        ]

        verified_reply = make_anthropic_request(
            [{'role': 'user', 'content': final_message_content}]
        )

        return verified_reply

    except Exception as e:
        return initial_reply


def pdf_chain_of_verification_google(
    initial_reply,
    prompt,
    step2_prompt,
    step3_prompt,
    step4_prompt,
    url,
    headers,
    creativity,
    remove_numbering,
    make_google_request,
    pdf_data,
    mime_type
):
    """
    Execute Chain of Verification (CoVe) process for PDF pages with Google Gemini.
    The PDF page is included in verification steps for accurate assessment.
    Returns the verified reply or initial reply if error occurs.

    Args:
        pdf_data: Base64 encoded PDF page data
        mime_type: MIME type of the content (e.g., "application/pdf")
    """
    import time

    try:
        # STEP 2: Generate verification questions (text only)
        step2_filled = step2_prompt.replace('<<INITIAL_REPLY>>', initial_reply)

        payload_step2 = {
            "contents": [{
                "parts": [{"text": step2_filled}]
            }],
            **({"generationConfig": {"temperature": creativity}} if creativity is not None else {})
        }

        result_step2 = make_google_request(url, headers, payload_step2)
        verification_questions = result_step2["candidates"][0]["content"]["parts"][0]["text"]

        # STEP 3: Answer verification questions WITH the PDF page
        questions_list = [
            remove_numbering(q)
            for q in verification_questions.split('\n')
            if q.strip()
        ]
        verification_qa = []

        for question in questions_list:
            time.sleep(2)  # Rate limit handling
            step3_filled = step3_prompt.replace('<<QUESTION>>', question)

            # Include PDF page in the verification question
            payload_step3 = {
                "contents": [{
                    "parts": [
                        {"text": step3_filled},
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": pdf_data
                            }
                        }
                    ]
                }],
                **({"generationConfig": {"temperature": creativity}} if creativity is not None else {})
            }

            result_step3 = make_google_request(url, headers, payload_step3)
            answer = result_step3["candidates"][0]["content"]["parts"][0]["text"]
            verification_qa.append(f"Q: {question}\nA: {answer}")

        # STEP 4: Final corrected categorization WITH the PDF page
        verification_qa_text = "\n\n".join(verification_qa)

        step4_filled = (step4_prompt
            .replace('<<PROMPT>>', prompt)
            .replace('<<INITIAL_REPLY>>', initial_reply)
            .replace('<<VERIFICATION_QA>>', verification_qa_text))

        # Include PDF page in final categorization
        payload_step4 = {
            "contents": [{
                "parts": [
                    {"text": step4_filled},
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": pdf_data
                        }
                    }
                ]
            }],
            "generationConfig": {
                "responseMimeType": "application/json",
                **({"temperature": creativity} if creativity is not None else {})
            }
        }

        result_step4 = make_google_request(url, headers, payload_step4)
        verified_reply = result_step4["candidates"][0]["content"]["parts"][0]["text"]

        return verified_reply

    except Exception as e:
        return initial_reply


def pdf_chain_of_verification_mistral(
    initial_reply,
    step2_prompt,
    step3_prompt,
    step4_prompt,
    client,  # Deprecated, kept for backward compatibility
    user_model,
    creativity,
    remove_numbering,
    pdf_content,
    api_key=None,
):
    """
    Execute Chain of Verification (CoVe) process for PDF pages with Mistral AI.
    The PDF page (as image) is included in verification steps for accurate
    assessment. Returns the verified reply or the initial reply if any error occurs.

    Uses direct HTTP requests instead of the mistralai SDK.

    Args:
        pdf_content: The PDF page content in Mistral format (image_url dict).
        api_key: Mistral API key.
    """
    import requests

    if api_key is None:
        return initial_reply

    endpoint = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    def make_mistral_request(messages, json_mode=False):
        payload = {"model": user_model, "messages": messages}
        if creativity is not None:
            payload["temperature"] = creativity
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        response = requests.post(endpoint, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]

    try:
        # STEP 2: Generate verification questions (text only)
        step2_filled = step2_prompt.replace('<<INITIAL_REPLY>>', initial_reply)
        verification_questions = make_mistral_request(
            [{'role': 'user', 'content': step2_filled}]
        )

        # STEP 3: Answer verification questions WITH the PDF page
        questions_list = [
            remove_numbering(q)
            for q in verification_questions.split('\n')
            if q.strip()
        ]
        verification_qa = []
        for question in questions_list:
            step3_filled = step3_prompt.replace('<<QUESTION>>', question)
            message_content = [
                {"type": "text", "text": step3_filled},
                pdf_content,
            ]
            answer = make_mistral_request(
                [{'role': 'user', 'content': message_content}]
            )
            verification_qa.append(f"Q: {question}\nA: {answer}")

        # STEP 4: Final corrected categorization WITH the PDF page
        verification_qa_text = "\n\n".join(verification_qa)
        step4_filled = (step4_prompt
            .replace('<<INITIAL_REPLY>>', initial_reply)
            .replace('<<VERIFICATION_QA>>', verification_qa_text))
        final_message_content = [
            {"type": "text", "text": step4_filled},
            pdf_content,
        ]
        verified_reply = make_mistral_request(
            [{'role': 'user', 'content': final_message_content}],
            json_mode=True,
        )
        return verified_reply

    except Exception:
        return initial_reply
