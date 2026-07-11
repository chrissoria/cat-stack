# Stepback prompting functions for various LLM providers

import requests


def get_stepback_insight_openai(
    stepback,
    api_key,
    user_model,
    model_source="openai",
    creativity=None
):
    """
    Get stepback insight from OpenAI-compatible APIs.
    Supports OpenAI, Perplexity, Huggingface, and xAI.

    Uses direct HTTP requests instead of OpenAI SDK for lighter dependencies.
    """
    # Determine the base URL based on model source
    if model_source == "huggingface":
        from cat_stack._providers import _detect_huggingface_endpoint
        base_url = _detect_huggingface_endpoint(api_key, user_model)
    elif model_source == "huggingface-together":
        base_url = "https://router.huggingface.co/together/v1"
    elif model_source == "perplexity":
        base_url = "https://api.perplexity.ai"
    elif model_source == "xai":
        base_url = "https://api.x.ai/v1"
    else:
        base_url = "https://api.openai.com/v1"

    endpoint = f"{base_url}/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": user_model,
        "messages": [{"role": "user", "content": stepback}],
    }

    # Sampling params via the shared shaper (skips temperature for OpenAI
    # reasoning models, which reject non-default values).
    from cat_stack._providers import apply_model_params
    apply_model_params(payload, model_source or "openai", user_model, creativity=creativity)

    try:
        response = requests.post(endpoint, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        stepback_insight = result["choices"][0]["message"]["content"]

        return stepback_insight, True

    except Exception as e:
        return None, False


def get_stepback_insight_anthropic(
    stepback,
    api_key,
    user_model,
    model_source="anthropic",
    creativity=None
):
    """
    Get stepback insight from Anthropic Claude.

    Uses direct HTTP requests instead of Anthropic SDK for lighter dependencies.
    """
    import requests

    endpoint = "https://api.anthropic.com/v1/messages"

    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01"
    }

    payload = {
        "model": user_model,
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": stepback}],
    }

    # Sampling params via the shared shaper (skips temperature on Anthropic
    # models that 400 on it: Opus 4.7+, Sonnet 5, Fable 5).
    from cat_stack._providers import apply_model_params
    apply_model_params(payload, "anthropic", user_model, creativity=creativity)

    try:
        response = requests.post(endpoint, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()

        # Parse response - Anthropic returns content as a list
        content = result.get("content", [])
        if content and content[0].get("type") == "text":
            stepback_insight = content[0].get("text", "")
            return stepback_insight, True

        return None, False

    except Exception as e:
        return None, False


def get_stepback_insight_google(
    stepback,
    api_key,
    user_model,
    model_source="google",
    creativity=None
):
    """
    Get stepback insight from Google Gemini.
    """
    import requests
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{user_model}:generateContent?key={api_key}"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    payload = {
        "contents": [{"parts": [{"text": stepback}]}],
    }

    # Sampling params via the shared shaper. Also fixes placement: Gemini
    # takes generationConfig at the top level of the request body (it was
    # previously spread inside contents[0], where it is not honored).
    from cat_stack._providers import apply_model_params
    apply_model_params(payload, "google", user_model, creativity=creativity)


    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # Raise error for bad status codes
        
        result = response.json()
        stepback_insight = result['candidates'][0]['content']['parts'][0]['text']
        
        return stepback_insight, True
        
    except Exception as e:
        return None, False


def get_stepback_insight_mistral(
    stepback,
    api_key,
    user_model,
    model_source="mistral",
    creativity=None
):
    """
    Get stepback insight from Mistral AI.
    """
    import requests

    endpoint = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": user_model,
        "messages": [{'role': 'user', 'content': stepback}],
    }

    # Sampling params via the shared shaper.
    from cat_stack._providers import apply_model_params
    apply_model_params(payload, "mistral", user_model, creativity=creativity)

    try:
        response = requests.post(endpoint, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        stepback_insight = result["choices"][0]["message"]["content"]

        return stepback_insight, True

    except Exception as e:
        return None, False



def get_stepback_insight_via_complete(
    stepback,
    api_key,
    user_model,
    model_source,
    creativity=None,
):
    """Stepback insight via the central UnifiedLLMClient.complete().

    For providers with no direct HTTP endpoint (claude-agent / codex-agent /
    claude-code), which route through complete() rather than a
    provider-specific requests.post.
    Returns (insight_text, True) on success, (None, False) otherwise.
    """
    from cat_stack._providers import UnifiedLLMClient
    try:
        client = UnifiedLLMClient(
            provider=model_source, api_key=api_key or "", model=user_model
        )
        insight, error = client.complete(
            messages=[{"role": "user", "content": stepback}],
            creativity=creativity,
            force_json=False,  # stepback insight is free text, not JSON
        )
        if error or not insight:
            return None, False
        return insight, True
    except Exception:
        return None, False
