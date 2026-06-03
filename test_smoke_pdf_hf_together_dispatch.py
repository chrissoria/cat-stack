"""
Smoke test for the H-PDF dispatch bug.

Before the fix, `pdf_multi_class(model_source="huggingface-together", ...)`
was accepted upstream (validated in `explore_pdf_categories` and the
`_call_openai_compatible` / `_call_openai_text_only` helpers all branched
on it), but `_process_single_page`'s `mode == "text"` and `mode == "image"`
dispatches hard-coded `["openai", "perplexity", "huggingface", "xai"]` —
so HF-together fell through to `raise ValueError("Unknown source!")`.

This test confirms the dispatch now reaches `_call_openai_text_only` /
`_call_openai_compatible` for HF-together. We don't need a successful
classification — any outcome that ISN'T "Unknown source" proves the
dispatch hole is closed (auth errors, transient 5xx, etc. all count as
"got past dispatch").
"""

import os
import sys
import traceback

from dotenv import load_dotenv

load_dotenv("/Users/chrissoria/Documents/Research/Categorization_AI_experiments/.env", override=True)

hf_key = os.getenv("HUGGINGFACE_API_KEY")
assert hf_key, "HUGGINGFACE_API_KEY missing from .env"

PDF_PATH = "/Users/chrissoria/Documents/Research/socialcapital/pop_network_mortality_slides.pdf"
assert os.path.exists(PDF_PATH), f"test PDF not found: {PDF_PATH}"

# A model Together actually serves.  If Together drops/renames it, the call
# will surface as 404 / "model not found" — that still proves dispatch
# worked (the failure happens INSIDE the call, not at the ValueError).
MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"

from cat_stack.pdf_functions import pdf_multi_class


def run_one(mode_name):
    print(f"\n{'='*70}")
    print(f"MODE: {mode_name}")
    print('='*70)
    try:
        result = pdf_multi_class(
            pdf_description="research presentation slides",
            pdf_input=PDF_PATH,
            categories=["network analysis", "mortality study"],
            api_key=hf_key,
            user_model=MODEL,
            mode=mode_name,
            model_source="huggingface-together",
            chain_of_thought=False,
            chain_of_verification=False,
        )
        print(f"OK — returned: {type(result).__name__}")
        if hasattr(result, 'shape'):
            print(f"  shape: {result.shape}")
        return "ok"
    except ValueError as e:
        msg = str(e)
        if "Unknown source" in msg:
            print(f"FAIL — dispatch hole still open: {msg}")
            return "dispatch_fail"
        print(f"non-dispatch ValueError (acceptable): {msg}")
        return "ok"
    except Exception as e:
        # Auth errors, 5xx, model-not-found, etc. all prove dispatch worked.
        print(f"non-dispatch error (acceptable, dispatch passed): {type(e).__name__}: {str(e)[:200]}")
        return "ok"


results = {
    "text": run_one("text"),
    "image": run_one("image"),
}

print(f"\n{'='*70}")
print("SUMMARY")
print('='*70)
for mode, outcome in results.items():
    marker = "PASS" if outcome == "ok" else "FAIL"
    print(f"  {mode:8s} {marker}")

if any(v == "dispatch_fail" for v in results.values()):
    sys.exit(1)
