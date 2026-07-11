# Image-aware Chain of Verification (CoVe) functions for various LLM providers
# These functions include the image in verification steps for accurate image-based categorization


def image_chain_of_verification_openai(
    initial_reply,
    step2_prompt,
    step3_prompt,
    step4_prompt,
    client,  # Deprecated, kept for backward compatibility
    user_model,
    creativity,
    remove_numbering,
    image_content,
    api_key=None,
    base_url=None,
):
    """
    Execute Chain of Verification (CoVe) process for images with OpenAI-compatible
    providers. The image is included in verification steps for accurate assessment.
    Returns the verified reply or initial reply if error occurs.

    Uses direct HTTP requests instead of the OpenAI SDK so the function works for
    any OpenAI-compatible provider (OpenAI, Perplexity, HuggingFace, xAI, ...) given
    the right `base_url`.

    Args:
        image_content: The image content in OpenAI format (image_url dict)
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
        # Sampling params via the shared shaper (skips temperature for OpenAI
        # reasoning models, which reject non-default values).
        from cat_stack._providers import apply_model_params
        apply_model_params(payload, "openai", user_model, creativity=creativity)
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        response = requests.post(endpoint, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]

    try:
        # STEP 2: Generate verification questions (text only - questions about the categorization)
        step2_filled = step2_prompt.replace('<<INITIAL_REPLY>>', initial_reply)
        verification_questions = make_openai_request(
            [{'role': 'user', 'content': step2_filled}]
        )

        # STEP 3: Answer verification questions WITH the image
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
                image_content
            ]
            answer = make_openai_request(
                [{'role': 'user', 'content': message_content}]
            )
            verification_qa.append(f"Q: {question}\nA: {answer}")

        # STEP 4: Final corrected categorization WITH the image
        verification_qa_text = "\n\n".join(verification_qa)
        step4_filled = (step4_prompt
            .replace('<<INITIAL_REPLY>>', initial_reply)
            .replace('<<VERIFICATION_QA>>', verification_qa_text))
        final_message_content = [
            {"type": "text", "text": step4_filled},
            image_content
        ]
        verified_reply = make_openai_request(
            [{'role': 'user', 'content': final_message_content}],
            json_mode=True,
        )
        return verified_reply

    except Exception:
        return initial_reply


def image_chain_of_verification_anthropic(
    initial_reply,
    step2_prompt,
    step3_prompt,
    step4_prompt,
    client,  # Deprecated, kept for backward compatibility
    user_model,
    creativity,
    remove_numbering,
    image_content,
    api_key=None
):
    """
    Execute Chain of Verification (CoVe) process for images with Anthropic Claude.
    The image is included in verification steps for accurate assessment.
    Returns the verified reply or initial reply if error occurs.

    Uses direct HTTP requests instead of Anthropic SDK.

    Args:
        image_content: The image content in Anthropic format (dict with type: "image")
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
        # Sampling params via the shared shaper (skips temperature on
        # Anthropic models that 400 on it: Opus 4.7+, Sonnet 5, Fable 5).
        from cat_stack._providers import apply_model_params
        apply_model_params(payload, "anthropic", user_model, creativity=creativity)

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

        # STEP 3: Answer verification questions WITH the image
        questions_list = [
            remove_numbering(q)
            for q in verification_questions.split('\n')
            if q.strip()
        ]
        verification_qa = []

        for question in questions_list:
            step3_filled = step3_prompt.replace('<<QUESTION>>', question)

            # Include image in the verification question
            message_content = [
                {"type": "text", "text": step3_filled},
                image_content
            ]

            answer = make_anthropic_request(
                [{'role': 'user', 'content': message_content}]
            )
            verification_qa.append(f"Q: {question}\nA: {answer}")

        # STEP 4: Final corrected categorization WITH the image
        verification_qa_text = "\n\n".join(verification_qa)

        step4_filled = (step4_prompt
            .replace('<<INITIAL_REPLY>>', initial_reply)
            .replace('<<VERIFICATION_QA>>', verification_qa_text))

        # Include image in final categorization
        final_message_content = [
            {"type": "text", "text": step4_filled},
            image_content
        ]

        verified_reply = make_anthropic_request(
            [{'role': 'user', 'content': final_message_content}]
        )

        return verified_reply

    except Exception as e:
        return initial_reply


def image_chain_of_verification_google(
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
    image_data,
    mime_type
):
    """
    Execute Chain of Verification (CoVe) process for images with Google Gemini.
    The image is included in verification steps for accurate assessment.
    Returns the verified reply or initial reply if error occurs.

    Args:
        image_data: Base64 encoded image data
        mime_type: MIME type of the image (e.g., "image/jpeg")
    """
    import time

    try:
        # Sampling params via the shared shaper (this variant doesn't receive
        # the model name; the Google branch keys off the provider only).
        from cat_stack._providers import apply_model_params

        # STEP 2: Generate verification questions (text only)
        step2_filled = step2_prompt.replace('<<INITIAL_REPLY>>', initial_reply)

        payload_step2 = apply_model_params(
            {"contents": [{"parts": [{"text": step2_filled}]}]},
            "google", "", creativity=creativity,
        )

        result_step2 = make_google_request(url, headers, payload_step2)
        verification_questions = result_step2["candidates"][0]["content"]["parts"][0]["text"]

        # STEP 3: Answer verification questions WITH the image
        questions_list = [
            remove_numbering(q)
            for q in verification_questions.split('\n')
            if q.strip()
        ]
        verification_qa = []

        for question in questions_list:
            time.sleep(2)  # Rate limit handling
            step3_filled = step3_prompt.replace('<<QUESTION>>', question)

            # Include image in the verification question
            payload_step3 = apply_model_params(
                {
                    "contents": [{
                        "parts": [
                            {"text": step3_filled},
                            {
                                "inline_data": {
                                    "mime_type": mime_type,
                                    "data": image_data
                                }
                            }
                        ]
                    }],
                },
                "google", "", creativity=creativity,
            )

            result_step3 = make_google_request(url, headers, payload_step3)
            answer = result_step3["candidates"][0]["content"]["parts"][0]["text"]
            verification_qa.append(f"Q: {question}\nA: {answer}")

        # STEP 4: Final corrected categorization WITH the image
        verification_qa_text = "\n\n".join(verification_qa)

        step4_filled = (step4_prompt
            .replace('<<PROMPT>>', prompt)
            .replace('<<INITIAL_REPLY>>', initial_reply)
            .replace('<<VERIFICATION_QA>>', verification_qa_text))

        # Include image in final categorization
        payload_step4 = apply_model_params(
            {
                "contents": [{
                    "parts": [
                        {"text": step4_filled},
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": image_data
                            }
                        }
                    ]
                }],
                "generationConfig": {"responseMimeType": "application/json"},
            },
            "google", "", creativity=creativity,
        )

        result_step4 = make_google_request(url, headers, payload_step4)
        verified_reply = result_step4["candidates"][0]["content"]["parts"][0]["text"]

        return verified_reply

    except Exception as e:
        return initial_reply


def image_chain_of_verification_mistral(
    initial_reply,
    step2_prompt,
    step3_prompt,
    step4_prompt,
    client,  # Deprecated, kept for backward compatibility
    user_model,
    creativity,
    remove_numbering,
    image_content,
    api_key=None,
):
    """
    Execute Chain of Verification (CoVe) process for images with Mistral AI.
    The image is included in verification steps for accurate assessment.
    Returns the verified reply or initial reply if error occurs.

    Uses direct HTTP requests instead of the mistralai SDK.

    Args:
        image_content: The image content in Mistral format (dict with image_url)
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
        # Sampling params via the shared shaper.
        from cat_stack._providers import apply_model_params
        apply_model_params(payload, "mistral", user_model, creativity=creativity)
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

        # STEP 3: Answer verification questions WITH the image
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
                image_content
            ]
            answer = make_mistral_request(
                [{'role': 'user', 'content': message_content}]
            )
            verification_qa.append(f"Q: {question}\nA: {answer}")

        # STEP 4: Final corrected categorization WITH the image
        verification_qa_text = "\n\n".join(verification_qa)
        step4_filled = (step4_prompt
            .replace('<<INITIAL_REPLY>>', initial_reply)
            .replace('<<VERIFICATION_QA>>', verification_qa_text))
        final_message_content = [
            {"type": "text", "text": step4_filled},
            image_content
        ]
        verified_reply = make_mistral_request(
            [{'role': 'user', 'content': final_message_content}],
            json_mode=True,
        )
        return verified_reply

    except Exception:
        return initial_reply
