# openai chain of verification calls

def chain_of_verification_openai(
    initial_reply,
    step2_prompt,
    step3_prompt,
    step4_prompt,
    client,
    user_model,
    creativity,
    remove_numbering
):
    """
    Execute Chain of Verification (CoVe) process.
    Returns the verified reply or initial reply if error occurs.
    """
    try:
        # Sampling params via the shared shaper (skips temperature for OpenAI
        # reasoning models, which reject non-default values). Computed once,
        # reused across the 3 steps.
        from cat_stack._providers import apply_model_params
        _params = apply_model_params({}, "openai", user_model, creativity=creativity)

        # STEP 2: Generate verification questions
        step2_filled = step2_prompt.replace('<<INITIAL_REPLY>>', initial_reply)

        verification_response = client.chat.completions.create(
            model=user_model,
            messages=[{'role': 'user', 'content': step2_filled}],
            **_params
        )

        verification_questions = verification_response.choices[0].message.content

        # STEP 3: Answer verification questions
        questions_list = [
            remove_numbering(q)
            for q in verification_questions.split('\n')
            if q.strip()
        ]
        verification_qa = []

        # Prompting each question individually
        for question in questions_list:
            step3_filled = step3_prompt.replace('<<QUESTION>>', question)

            answer_response = client.chat.completions.create(
                model=user_model,
                messages=[{'role': 'user', 'content': step3_filled}],
                **_params
            )

            answer = answer_response.choices[0].message.content
            verification_qa.append(f"Q: {question}\nA: {answer}")

        # STEP 4: Final corrected categorization
        # Request strict JSON via response_format — Step 4 produces the
        # final per-category labeling, which downstream extract_json()
        # expects to be JSON-shaped. Matches the pattern in
        # calls/image_CoVe.py and calls/pdf_CoVe.py for the OpenAI variant.
        verification_qa_text = "\n\n".join(verification_qa)

        step4_filled = (step4_prompt
            .replace('<<INITIAL_REPLY>>', initial_reply)
            .replace('<<VERIFICATION_QA>>', verification_qa_text))

        final_response = client.chat.completions.create(
            model=user_model,
            messages=[{'role': 'user', 'content': step4_filled}],
            response_format={"type": "json_object"},
            **_params
        )

        verified_reply = final_response.choices[0].message.content

        return verified_reply

    except Exception as e:
        return initial_reply

# anthropic chain of verification calls

def chain_of_verification_anthropic(
    initial_reply,
    step2_prompt,
    step3_prompt,
    step4_prompt,
    client,
    user_model,
    creativity,
    remove_numbering
):
    """
    Execute Chain of Verification (CoVe) process for Anthropic Claude.
    Returns the verified reply or initial reply if error occurs.
    """
    try:
        # Sampling params via the shared shaper (skips temperature on
        # Anthropic models that 400 on it: Opus 4.7+, Sonnet 5, Fable 5).
        # Computed once, reused across the 3 steps.
        from cat_stack._providers import apply_model_params
        _params = apply_model_params({}, "anthropic", user_model, creativity=creativity)

        # STEP 2: Generate verification questions
        step2_filled = step2_prompt.replace('<<INITIAL_REPLY>>', initial_reply)

        verification_response = client.messages.create(
            model=user_model,
            messages=[{'role': 'user', 'content': step2_filled}],
            max_tokens=4096,
            **_params
        )

        verification_questions = verification_response.content[0].text

        # STEP 3: Answer verification questions
        questions_list = [
            remove_numbering(q)
            for q in verification_questions.split('\n')
            if q.strip()
        ]
        verification_qa = []

        # Prompting each question individually
        for question in questions_list:
            step3_filled = step3_prompt.replace('<<QUESTION>>', question)

            answer_response = client.messages.create(
                model=user_model,
                messages=[{'role': 'user', 'content': step3_filled}],
                max_tokens=4096,
                **_params
            )

            answer = answer_response.content[0].text
            verification_qa.append(f"Q: {question}\nA: {answer}")

        # STEP 4: Final corrected categorization
        verification_qa_text = "\n\n".join(verification_qa)

        step4_filled = (step4_prompt
            .replace('<<INITIAL_REPLY>>', initial_reply)
            .replace('<<VERIFICATION_QA>>', verification_qa_text))

        final_response = client.messages.create(
            model=user_model,
            messages=[{'role': 'user', 'content': step4_filled}],
            max_tokens=4096,
            **_params
        )

        verified_reply = final_response.content[0].text

        return verified_reply

    except Exception as e:
        return initial_reply

# google chain of verification calls
def chain_of_verification_google(
    initial_reply,
    prompt,
    step2_prompt,
    step3_prompt,
    step4_prompt,
    url,
    headers,
    creativity,
    remove_numbering,
    make_google_request
):
    import time
    """
    Execute Chain of Verification (CoVe) process for Google Gemini.
    Returns the verified reply or initial reply if error occurs.
    """
    try:
        # Sampling params via the shared shaper (this variant doesn't receive
        # the model name; the Google branch keys off the provider only).
        from cat_stack._providers import apply_model_params

        # STEP 2: Generate verification questions
        step2_filled = step2_prompt.replace('<<INITIAL_REPLY>>', initial_reply)

        payload_step2 = apply_model_params(
            {"contents": [{"parts": [{"text": step2_filled}]}]},
            "google", "", creativity=creativity,
        )

        result_step2 = make_google_request(url, headers, payload_step2)
        verification_questions = result_step2["candidates"][0]["content"]["parts"][0]["text"]
        
        # STEP 3: Answer verification questions
        questions_list = [
            remove_numbering(q) 
            for q in verification_questions.split('\n') 
            if q.strip()
        ]
        verification_qa = []
        
        for question in questions_list:
            time.sleep(2)  # temporary rate limit handling
            step3_filled = step3_prompt.replace('<<QUESTION>>', question)
            
            payload_step3 = apply_model_params(
                {"contents": [{"parts": [{"text": step3_filled}]}]},
                "google", "", creativity=creativity,
            )

            result_step3 = make_google_request(url, headers, payload_step3)
            answer = result_step3["candidates"][0]["content"]["parts"][0]["text"]
            verification_qa.append(f"Q: {question}\nA: {answer}")
        
        # STEP 4: Final corrected categorization
        # Request strict JSON via responseMimeType — Step 4 produces the
        # final per-category labeling, which downstream extract_json()
        # expects to be JSON-shaped. Matches the pattern in
        # calls/pdf_CoVe.py for the Google variant. responseMimeType
        # belongs inside generationConfig (top-level is rejected by Gemini).
        verification_qa_text = "\n\n".join(verification_qa)

        step4_filled = (step4_prompt
            .replace('<<PROMPT>>', prompt)
            .replace('<<INITIAL_REPLY>>', initial_reply)
            .replace('<<VERIFICATION_QA>>', verification_qa_text))

        payload_step4 = apply_model_params(
            {
                "contents": [{"parts": [{"text": step4_filled}]}],
                "generationConfig": {"responseMimeType": "application/json"},
            },
            "google", "", creativity=creativity,
        )

        result_step4 = make_google_request(url, headers, payload_step4)
        verified_reply = result_step4["candidates"][0]["content"]["parts"][0]["text"]

        return verified_reply

    except Exception as e:
        return initial_reply

# mistral chain of verification calls

def chain_of_verification_mistral(
    initial_reply,
    step2_prompt,
    step3_prompt,
    step4_prompt,
    client,
    user_model,
    creativity,
    remove_numbering
):
    """
    Execute Chain of Verification (CoVe) process for Mistral AI.
    Returns the verified reply or initial reply if error occurs.
    """
    try:
        # Sampling params via the shared shaper. Computed once, reused
        # across the 3 steps.
        from cat_stack._providers import apply_model_params
        _params = apply_model_params({}, "mistral", user_model, creativity=creativity)

        # STEP 2: Generate verification questions
        step2_filled = step2_prompt.replace('<<INITIAL_REPLY>>', initial_reply)

        verification_response = client.chat.complete(
            model=user_model,
            messages=[{'role': 'user', 'content': step2_filled}],
            **_params
        )
        
        verification_questions = verification_response.choices[0].message.content
        
        # STEP 3: Answer verification questions
        questions_list = [
            remove_numbering(q) 
            for q in verification_questions.split('\n') 
            if q.strip()
        ]
        verification_qa = []
        
        # Prompting each question individually
        for question in questions_list:
            step3_filled = step3_prompt.replace('<<QUESTION>>', question)
            
            answer_response = client.chat.complete(
                model=user_model,
                messages=[{'role': 'user', 'content': step3_filled}],
                **_params
            )
            
            answer = answer_response.choices[0].message.content
            verification_qa.append(f"Q: {question}\nA: {answer}")
        
        # STEP 4: Final corrected categorization
        # Request strict JSON via response_format — Step 4 produces the
        # final per-category labeling, which downstream extract_json()
        # expects to be JSON-shaped. Matches the pattern in
        # calls/pdf_CoVe.py for the Mistral variant.
        verification_qa_text = "\n\n".join(verification_qa)

        step4_filled = (step4_prompt
            .replace('<<INITIAL_REPLY>>', initial_reply)
            .replace('<<VERIFICATION_QA>>', verification_qa_text))

        final_response = client.chat.complete(
            model=user_model,
            messages=[{'role': 'user', 'content': step4_filled}],
            response_format={"type": "json_object"},
            **_params
        )
        
        verified_reply = final_response.choices[0].message.content

        return verified_reply

    except Exception as e:
        return initial_reply
