"""
Live empirical validation of PREFLIGHT (#40): _sanitize_google_schema
strips `additionalProperties` (and friends) before the payload reaches
Google's responseSchema.

Two probes against Gemini Flash:

  A. Pre-fix shape: send the classify_ensemble preflight schema verbatim
     (with `additionalProperties: false`). Expected: 400 Bad Request
     with a body that mentions the rejected key.

  B. Post-fix shape: send the same schema after _sanitize_google_schema()
     strips the bad key. Expected: 200 OK, model returns the requested
     JSON object.

Cost: ~$0 — two small calls to gemini-2.5-flash, ~50 tokens each.
"""

import os
import json

import requests
from dotenv import load_dotenv

load_dotenv("/Users/chrissoria/Documents/Research/Categorization_AI_experiments/.env", override=True)
api_key = os.getenv("GOOGLE_API_KEY")
assert api_key, "GOOGLE_API_KEY missing from .env"

from cat_stack._providers import _sanitize_google_schema

MODEL = "gemini-2.5-flash"
ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"

# The exact preflight schema sent by classify_ensemble at L680
PREFLIGHT_SCHEMA = {
    "type": "object",
    "properties": {"1": {"type": "string"}},
    "required": ["1"],
    "additionalProperties": False,
}

PROMPT = 'Reply with exactly: {"1":"0"}'


def make_payload(schema):
    return {
        "contents": [{"parts": [{"text": PROMPT}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": schema,
            "temperature": 0,
        },
    }


def probe(label, schema):
    print(f"\n{'='*72}")
    print(f"PROBE: {label}")
    print('='*72)
    print(f"Schema (responseSchema field):")
    print(f"  {json.dumps(schema)}")

    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
    resp = requests.post(ENDPOINT, headers=headers, json=make_payload(schema), timeout=30)
    print(f"\nStatus: {resp.status_code}")
    if resp.status_code == 200:
        body = resp.json()
        try:
            text = body["candidates"][0]["content"]["parts"][0]["text"]
            print(f"Returned: {text!r}")
        except Exception as e:
            print(f"Could not parse candidates: {e}")
            print(f"Body keys: {list(body.keys())}")
    else:
        snippet = resp.text[:400]
        print(f"Body: {snippet}")
    return resp.status_code


print(f"Model: {MODEL}")
print(f"Endpoint: {ENDPOINT.split('?')[0]}")

# Probe A: raw schema (pre-fix shape)
status_pre = probe("A. raw schema with additionalProperties: false (pre-fix shape)",
                   PREFLIGHT_SCHEMA)

# Probe B: sanitized schema (post-fix shape)
sanitized = _sanitize_google_schema(PREFLIGHT_SCHEMA)
status_post = probe("B. _sanitize_google_schema() applied (post-fix shape)",
                    sanitized)

print(f"\n{'='*72}")
print("SUMMARY")
print('='*72)
print(f"  pre-fix  (raw schema):       {status_pre}  {'← bug confirmed' if status_pre == 400 else '← bug not reproduced'}")
print(f"  post-fix (sanitized schema): {status_post}  {'← fix works' if status_post == 200 else '← fix did not resolve'}")

if status_pre == 400 and status_post == 200:
    print("\n→ PASS — bug claim verified empirically; fix verified empirically.")
elif status_pre != 400:
    print("\n→ INFO — Google didn't 400 on the raw schema. The bug may have been "
          "resolved server-side, or the model accepts it now. Fix is harmless either way.")
elif status_post != 200:
    print("\n→ FAIL — sanitizer-stripped schema still doesn't work.")
