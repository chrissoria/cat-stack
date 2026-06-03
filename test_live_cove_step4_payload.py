"""
Live empirical validation of H-COVE (#32): each provider's gateway
accepts the Step 4 JSON-mode payload shape that calls/CoVe.py now sends.

Uses raw requests.post (no SDK installs) — consistent with the rest of
the cat-stack live tests and matching how UnifiedLLMClient itself talks
to provider APIs.

For each provider, send the exact Step 4 payload shape that the
post-fix CoVe function sends, then verify the response is 200 and the
returned text is parseable JSON.

  OpenAI:      response_format={"type": "json_object"}        in POST body
  Mistral:     response_format={"type": "json_object"}        in POST body
  Google:      generationConfig.responseMimeType = "application/json"
  HF (router): response_format={"type": "json_object"}        in POST body
               (HF's OpenAI-compatible router at router.huggingface.co/v1)

  Anthropic:   SKIPPED — no native JSON-mode kwarg in Anthropic's
               messages API; H-COVE intentionally didn't change the
               anthropic Step 4 (relies on prompt instruction, matching
               image_CoVe.py / pdf_CoVe.py anthropic variants).

Cost: ~50 input + ~30 output tokens per provider. <$0.001 total.
"""

import json
import os
import sys
import time

import requests
from dotenv import load_dotenv

load_dotenv("/Users/chrissoria/Documents/Research/Categorization_AI_experiments/.env", override=True)
openai_key = os.getenv("OPENAI_API_KEY")
google_key = os.getenv("GOOGLE_API_KEY")
mistral_key = os.getenv("MISTRAL_API_KEY")
hf_key = os.getenv("HUGGINGFACE_API_KEY")
assert openai_key and google_key and mistral_key and hf_key, "missing key(s) in .env"

# The Step 4 prompt shape that the post-fix CoVe builds (compressed for the smoke probe)
STEP4_USER_CONTENT = (
    "Initial classification: {\"1\":\"1\",\"2\":\"0\",\"3\":\"0\"}\n"
    "Verification Q&A:\n"
    "Q: Does 'I moved for a new job' relate to career?\nA: Yes.\n"
    "Q: Does it relate to family?\nA: No.\n\n"
    "Categories: 1=career, 2=family, 3=housing.\n"
    "Return ONLY the final corrected JSON object. Keys: \"1\", \"2\", \"3\". "
    "Each value: \"0\" or \"1\". No prose, no markdown."
)


def _check_json(provider, status, body_text):
    if status != 200:
        print(f"\n→ FAIL — {provider} returned {status}")
        print(f"  Body: {body_text[:300]}")
        return False
    try:
        parsed = json.loads(body_text)
        print(f"→ PASS — {provider} Step 4 returned valid JSON: {parsed}")
        return True
    except json.JSONDecodeError as e:
        print(f"\n→ FAIL — {provider} returned non-JSON: {e}")
        print(f"  Body: {body_text[:300]}")
        return False


def probe_openai_compatible(label, endpoint, model, api_key):
    """Single OpenAI-compatible chat.completions call carrying the
    Step 4 payload shape (with response_format)."""
    print(f"\n{'='*72}\nPROVIDER: {label}\n{'='*72}")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": STEP4_USER_CONTENT}],
        "response_format": {"type": "json_object"},
        "temperature": 0,
    }
    t0 = time.time()
    resp = requests.post(endpoint, headers=headers, json=payload, timeout=60)
    dt = time.time() - t0
    print(f"Elapsed: {dt:.1f}s")
    print(f"Status: {resp.status_code}")
    if resp.status_code != 200:
        return _check_json(label, resp.status_code, resp.text)
    try:
        content = resp.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        print(f"\n→ FAIL — {label} 200 but missing choices/content: {e}")
        print(f"  Body: {resp.text[:300]}")
        return False
    print(f"Content: {content!r}")
    return _check_json(label, 200, content)


def probe_google(label, model, api_key):
    """Google uses generationConfig.responseMimeType — different payload shape."""
    print(f"\n{'='*72}\nPROVIDER: {label}\n{'='*72}")
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": STEP4_USER_CONTENT}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0,
        },
    }
    t0 = time.time()
    resp = requests.post(endpoint, headers=headers, json=payload, timeout=60)
    dt = time.time() - t0
    print(f"Elapsed: {dt:.1f}s")
    print(f"Status: {resp.status_code}")
    if resp.status_code != 200:
        return _check_json(label, resp.status_code, resp.text)
    try:
        content = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as e:
        print(f"\n→ FAIL — {label} 200 but missing candidates: {e}")
        return False
    print(f"Content: {content!r}")
    return _check_json(label, 200, content)


results = {}

results["openai (gpt-4o-mini)"] = probe_openai_compatible(
    "OpenAI gpt-4o-mini",
    endpoint="https://api.openai.com/v1/chat/completions",
    model="gpt-4o-mini",
    api_key=openai_key,
)

results["mistral (small)"] = probe_openai_compatible(
    "Mistral mistral-small-latest",
    endpoint="https://api.mistral.ai/v1/chat/completions",
    model="mistral-small-latest",
    api_key=mistral_key,
)

results["hf router (Llama-3.1-8B)"] = probe_openai_compatible(
    "HuggingFace Llama-3.1-8B-Instruct (OpenAI-compat router)",
    endpoint="https://router.huggingface.co/v1/chat/completions",
    model="meta-llama/Llama-3.1-8B-Instruct",
    api_key=hf_key,
)

results["google (gemini-2.5-flash)"] = probe_google(
    "Google gemini-2.5-flash",
    model="gemini-2.5-flash",
    api_key=google_key,
)

print(f"\n{'='*72}\nSUMMARY\n{'='*72}")
for label, ok in results.items():
    print(f"  {label:36s}  {'PASS' if ok else 'FAIL'}")

sys.exit(0 if all(results.values()) else 1)
