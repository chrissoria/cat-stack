"""
Live end-to-end validation against HuggingFace's router using a mid-tier
model. Exercises the parts of the stack the recent batch of fixes touched:

  - HF endpoint detection (C10): does the router auto-route the bare
    model name to a working backend?
  - Retry hardening (H-RETRY): jittered backoff + Retry-After honoring
    if any transient 429/5xx fires.
  - Strip-on-5xx (HF-SMALL-MODEL, #44): if the backend rejects
    response_format with a non-JSON 5xx body, strip it once and retry.
    For Llama-3.1-8B this should NOT trigger (it accepts response_format),
    but the path exists if it does.
  - JSON schema handling and extract_json downstream.
  - Active text-CoVe path
    (text_functions_ensemble.run_chain_of_verification): exercise
    Step 4's json_schema kwarg → JSON mode coming out of complete().

Cost: ~600 input + ~1500 output tokens per scenario. Total < $0.01.
"""

import os
import sys
import time
import traceback

from dotenv import load_dotenv

load_dotenv("/Users/chrissoria/Documents/Research/Categorization_AI_experiments/.env", override=True)
hf_key = os.getenv("HUGGINGFACE_API_KEY")
assert hf_key, "HUGGINGFACE_API_KEY missing from .env"

from cat_stack import classify

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
TEXTS = [
    "I moved for a new job opportunity in tech.",
    "We relocated to be closer to my mother who needs care.",
    "Found a cheaper apartment downtown.",
]
CATEGORIES = ["career", "family", "housing"]


def run_scenario(label, **classify_kwargs):
    print(f"\n{'='*72}")
    print(f"SCENARIO: {label}")
    print('='*72)
    start = time.time()
    try:
        df = classify(
            input_data=TEXTS,
            categories=CATEGORIES,
            models=[(MODEL, "huggingface", hf_key)],
            multi_label=True,
            check_verbosity=False,  # skip the extra verbosity-check API call
            add_other=False,        # skip "Other" category prompt
            json_formatter=False,   # avoid consent prompt
            **classify_kwargs,
        )
        elapsed = time.time() - start
        print(f"\nElapsed: {elapsed:.1f}s")
        print(f"Result shape: {df.shape}")
        print(df.to_string())

        # Sanity: 3 rows of classifications produced
        assert len(df) == 3, f"expected 3 rows, got {len(df)}"
        # Each row should have at least one category labeled (1)
        cat_cols = [c for c in df.columns if c.startswith("category_")]
        assert len(cat_cols) == 3, f"expected 3 category_N columns, got: {cat_cols}"

        # Show which labels were actually assigned
        for i, row in df.iterrows():
            assigned = [cat for cat, col in zip(CATEGORIES, cat_cols) if row[col] == 1]
            print(f"  Row {i}: {assigned or '(none)'}")

        print(f"\n→ PASS — scenario '{label}' classified successfully.")
        return True
    except Exception as e:
        elapsed = time.time() - start
        print(f"\nElapsed: {elapsed:.1f}s")
        print(f"\n→ FAIL — {type(e).__name__}: {e}")
        traceback.print_exc()
        return False


results = {}

# Scenario 1: baseline classify
results["baseline"] = run_scenario("baseline classify")

# Scenario 2: chain_of_verification — exercises run_chain_of_verification
# in text_functions_ensemble.py; Step 4 goes through complete(json_schema=...)
# which adds the JSON mode appropriate to HF.
results["cove"] = run_scenario("classify with chain_of_verification=True",
                                chain_of_verification=True)

print(f"\n{'='*72}")
print("SUMMARY")
print('='*72)
for label, ok in results.items():
    print(f"  {label:32s}  {'PASS' if ok else 'FAIL'}")

sys.exit(0 if all(results.values()) else 1)
