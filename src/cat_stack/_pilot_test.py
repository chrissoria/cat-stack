"""
Pilot test module for CatLLM.

Runs classification on a small random sample and asks the user to validate
results before committing to the full (potentially expensive) classification run.
"""

import random


def run_pilot_test(
    input_data,
    categories,
    models,
    classify_ensemble_fn,
    ensemble_kwargs,
    sample_size=10,
):
    """
    Run a pilot classification on a random sample and collect user feedback.

    Classifies a small random subset, displays the results, and asks the user
    to mark each as correct or incorrect. Returns a summary so the caller can
    decide whether to proceed.

    Args:
        input_data: The full input data (list or Series).
        categories: List of category names.
        models: Models list (same format as classify()).
        classify_ensemble_fn: The classify_ensemble callable.
        ensemble_kwargs: Dict of keyword arguments to forward to classify_ensemble.
        sample_size: Number of random items to test. Default 10.

    Returns:
        dict with keys:
            - "proceed": bool — True if user chose to continue
            - "accuracy": float — fraction marked correct (0-1)
            - "feedback": list of dicts with "input", "classifications", "correct"
            - "sample_indices": list of int indices that were sampled
        Returns None if user cancels before completing feedback.
    """
    import pandas as pd

    # Convert to list for indexing
    if isinstance(input_data, pd.Series):
        items_list = input_data.tolist()
    else:
        items_list = list(input_data)

    n_total = len(items_list)
    if n_total == 0:
        print("[CatLLM] No items to pilot test.")
        return {"proceed": True, "accuracy": 1.0, "feedback": [], "sample_indices": []}

    # Sample
    actual_sample_size = min(sample_size, n_total)
    sample_indices = sorted(random.sample(range(n_total), actual_sample_size))
    sample_items = [items_list[i] for i in sample_indices]

    print(f"\n[CatLLM] Running pilot test on {actual_sample_size} random item(s)...")
    print("=" * 60)

    # Run classification on the sample
    pilot_kwargs = dict(ensemble_kwargs)
    # Override settings for pilot: no file saving, no progress callback
    pilot_kwargs["filename"] = None
    pilot_kwargs["save_directory"] = None
    pilot_kwargs["progress_callback"] = None
    pilot_kwargs["input_data"] = sample_items
    pilot_kwargs["categories"] = categories
    pilot_kwargs["models"] = models

    try:
        pilot_result = classify_ensemble_fn(**pilot_kwargs)
    except Exception as e:
        print(f"\n[CatLLM] Pilot test classification failed: {e}")
        print("  Skipping pilot test and proceeding with full classification.\n")
        return {"proceed": True, "accuracy": 0.0, "feedback": [], "sample_indices": sample_indices}

    # Extract classifications from the result DataFrame
    # Category columns are named like "category_name" with values 0/1,
    # or for ensemble they may be "category_name_consensus"
    feedback = []
    is_multi_model = len(models) > 1

    print(f"\n{'=' * 60}")
    print("PILOT TEST RESULTS — Please review each classification")
    print(f"{'=' * 60}\n")

    for row_idx in range(len(pilot_result)):
        row = pilot_result.iloc[row_idx]
        input_text = sample_items[row_idx]

        # Truncate long inputs for display
        display_text = str(input_text)
        if len(display_text) > 200:
            display_text = display_text[:200] + "..."

        print(f"--- Item {row_idx + 1}/{actual_sample_size} ---")
        print(f"  Input: {display_text}\n")

        # Find which categories were assigned (value = 1)
        # Columns are numbered: category_1, category_2, ... (single model)
        # or category_1_consensus, category_2_consensus, ... (multi-model)
        assigned = []
        for cat_idx, cat in enumerate(categories, 1):
            if is_multi_model:
                col = f"category_{cat_idx}_consensus"
            else:
                col = f"category_{cat_idx}"

            if col in pilot_result.columns:
                val = row[col]
                if val is not None and str(val) == "1":
                    assigned.append(cat)

        if assigned:
            print(f"  Classified as: {', '.join(assigned)}")
        else:
            print("  Classified as: (none)")

        print()

        # Ask for feedback
        try:
            answer = input("  Is this correct? (Y/n/q to quit): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n\n[CatLLM] Pilot test cancelled.")
            return None

        if answer in ("q", "quit", "exit"):
            print("\n[CatLLM] Pilot test cancelled by user.")
            return None

        is_correct = answer in ("", "y", "yes")
        feedback.append({
            "input": input_text,
            "classifications": assigned,
            "correct": is_correct,
        })
        print()

    # Summarize
    n_correct = sum(1 for f in feedback if f["correct"])
    n_total_fb = len(feedback)
    accuracy = n_correct / n_total_fb if n_total_fb > 0 else 0.0
    pct = accuracy * 100

    print(f"{'=' * 60}")
    print(f"PILOT TEST SUMMARY: {n_correct}/{n_total_fb} correct ({pct:.0f}%)")
    print(f"{'=' * 60}\n")

    if accuracy < 0.5:
        print(
            "  WARNING: Less than half of the pilot classifications were marked correct.\n"
            "  Consider revising your categories — adding descriptions and examples\n"
            "  significantly improves accuracy. For example:\n"
            "\n"
            '    Instead of:  "Positive"\n'
            '    Consider:    "Positive: The response expresses satisfaction, approval,\n'
            "     or happiness (e.g., 'I love this product', 'Great experience')\"\n"
        )
    elif accuracy < 0.8:
        print(
            "  Some classifications were incorrect. You may want to refine your\n"
            "  category definitions before running the full classification.\n"
        )
    else:
        print("  Classifications look good!\n")

    # Ask whether to proceed
    try:
        proceed_answer = input(
            "  Proceed with full classification? (Y/n): "
        ).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\n[CatLLM] Classification cancelled.")
        return {"proceed": False, "accuracy": accuracy, "feedback": feedback, "sample_indices": sample_indices}

    proceed = proceed_answer in ("", "y", "yes")

    if not proceed:
        print("\n[CatLLM] Classification cancelled. Adjust your categories and try again.\n")
    else:
        print("\n[CatLLM] Proceeding with full classification...\n")

    return {
        "proceed": proceed,
        "accuracy": accuracy,
        "feedback": feedback,
        "sample_indices": sample_indices,
    }
