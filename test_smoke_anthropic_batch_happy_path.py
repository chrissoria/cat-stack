"""
Live smoke test for H-BATCH (task #28).

Two fixes landed:
  1. `_inspect_anthropic_terminal_state` is invoked when an Anthropic
     batch reaches `processing_status == "ended"`. For full-success
     batches it returns silently (no behavior change). For all-errored
     batches it raises BatchJobFailedError (new). For partials it prints
     a warning and returns normally.
  2. Per-model failure isolation in `run_batch_ensemble_classify` and
     `run_batch_ensemble_summarize` — if one model raises, the others
     still complete and the DataFrame returns with that model's column
     empty.

This live test exercises (1) on the happy path — submit a tiny Anthropic
batch with valid payload, confirm the inspection helper doesn't disrupt
normal completion. Skips the isolation test because that requires
waiting for a real Anthropic batch to complete (5-30 min) and the unit
tests in tests/test_batch_ensemble_isolation.py cover the behavior with
mocked failures.

Expect: 1-10 minutes wall time for the Anthropic batch to complete.
"""

import os
import sys
import time

from dotenv import load_dotenv

load_dotenv("/Users/chrissoria/Documents/Research/Categorization_AI_experiments/.env", override=True)

anthropic_key = os.getenv("ANTHROPIC_API_KEY")
assert anthropic_key, "ANTHROPIC_API_KEY missing from .env"

from cat_stack import classify

print("Submitting Anthropic batch (claude-haiku, 2 items, 2 categories)...")
print("This will take 1-10 minutes wall time — batch APIs have no SLA on speed.")
start = time.time()

df = classify(
    input_data=[
        "I moved for a new job opportunity in tech.",
        "We relocated to be closer to my mother who needs care.",
    ],
    categories=["career", "family"],
    models=[("claude-haiku-4-5-20251001", "anthropic", anthropic_key)],
    batch_mode=True,
    batch_poll_interval=15.0,
    batch_timeout=900.0,  # 15 min cap
    multi_label=True,
)

elapsed = time.time() - start
print(f"\nBatch completed in {elapsed:.0f}s")
print(f"\nDataFrame shape: {df.shape}")
print(df.to_string())

# Smoke check: 2 rows, no exception, expected categories present
assert len(df) == 2, f"expected 2 rows, got {len(df)}"

# The classify output has columns like: response, career, family,
# claude_haiku_4_5_20251001_career, claude_haiku_4_5_20251001_family, etc.
# classify() returns columns "category_1", "category_2", ... + per-model columns.
# Successful classification gives 1/0 per category.
cols = list(df.columns)
has_cat_cols = any(c.startswith("category_") for c in cols)
assert has_cat_cols, f"expected category_N columns in {cols}"

# Verify both rows succeeded — the inspection helper would have raised
# if state=ended + errored>0, so reaching this assert proves the helper
# returned silently on the all-success path.
assert (df["processing_status"] == "success").all(), \
    f"expected all rows to succeed, got: {df['processing_status'].tolist()}"

# Row 0 ("moved for job") → career=1, family=0
# Row 1 ("closer to mother") → career=0, family=1
assert df.loc[0, "category_1"] == 1, "row 0 should be classified as career"
assert df.loc[1, "category_2"] == 1, "row 1 should be classified as family"

print("\nPASS — Anthropic batch happy path works after H-BATCH fix.")
