"""
Pilot test module for CatLLM.

Provides two capabilities:
1. collect_corrections() — classify a small sample and collect category-level
   user corrections. Used by prompt_tune() for APO and by classify(pilot_test=True).
2. run_pilot_test() — wrapper that collects corrections and asks whether to proceed.
"""

import random


def collect_corrections(
    input_data,
    categories,
    models,
    classify_ensemble_fn,
    ensemble_kwargs,
    sample_size=10,
    system_prompt="",
):
    """
    Classify a random sample and collect per-category user corrections.

    Args:
        input_data: The full input data (list or Series).
        categories: List of category names.
        models: Models list (same format as classify()).
        classify_ensemble_fn: The classify_ensemble callable.
        ensemble_kwargs: Dict of keyword arguments to forward to classify_ensemble.
        sample_size: Number of random items to test. Default 10.
        system_prompt: Optional system prompt to use for this classification run.

    Returns:
        dict with keys:
            - "corrections": list of dicts, each with:
                - "input": str — the input text
                - "original": dict — {category_name: 0/1} as model classified
                - "corrected": dict — {category_name: 0/1} after user corrections
                - "changed": list of str — category names that were flipped
            - "accuracy": float — fraction of items with zero corrections (0-1)
            - "category_accuracy": float — fraction of individual category
              decisions that were correct (0-1)
            - "total_flips": int — total number of category-level corrections
            - "sample_indices": list of int indices that were sampled
        Returns None if user cancels (q/quit/Ctrl-C).
    """
    import pandas as pd

    # Convert to list for indexing
    if isinstance(input_data, pd.Series):
        items_list = input_data.tolist()
    else:
        items_list = list(input_data)

    n_total = len(items_list)
    if n_total == 0:
        print("[CatLLM] No items to test.")
        return {
            "corrections": [], "accuracy": 1.0, "category_accuracy": 1.0,
            "total_flips": 0, "sample_indices": [],
        }

    # Sample
    actual_sample_size = min(sample_size, n_total)
    sample_indices = sorted(random.sample(range(n_total), actual_sample_size))
    sample_items = [items_list[i] for i in sample_indices]

    print(f"\n[CatLLM] Classifying {actual_sample_size} random item(s)...")
    print("=" * 60)

    # Run classification on the sample
    pilot_kwargs = dict(ensemble_kwargs)
    pilot_kwargs["filename"] = None
    pilot_kwargs["save_directory"] = None
    pilot_kwargs["progress_callback"] = None
    pilot_kwargs["input_data"] = sample_items
    pilot_kwargs["categories"] = categories
    pilot_kwargs["models"] = models
    if system_prompt:
        pilot_kwargs["system_prompt"] = system_prompt

    try:
        pilot_result = classify_ensemble_fn(**pilot_kwargs)
    except Exception as e:
        print(f"\n[CatLLM] Classification failed: {e}")
        return None

    is_multi_model = len(models) > 1
    corrections = []

    print(f"\n{'=' * 60}")
    print("RESULTS — Review each classification")
    print("Enter category numbers to flip (e.g. '1,3'), or press Enter if correct.")
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

        # Read per-category values
        cat_values = {}
        for cat_idx, cat in enumerate(categories, 1):
            if is_multi_model:
                col = f"category_{cat_idx}_consensus"
            else:
                col = f"category_{cat_idx}"

            val = 0
            if col in pilot_result.columns:
                raw = row[col]
                if raw is not None and str(raw) == "1":
                    val = 1
            cat_values[cat] = val

        # Display each category with its assignment
        print("  Categories:")
        for cat_idx, cat in enumerate(categories, 1):
            val = cat_values[cat]
            marker = "1" if val else "0"
            cat_display = cat if len(cat) <= 60 else cat[:57] + "..."
            print(f"    {cat_idx}. {cat_display:<60s} = {marker}")
        print()

        # Ask for corrections
        try:
            answer = input(
                "  Numbers to flip (e.g. '1,3'), Enter if correct, 'q' to quit: "
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n\n[CatLLM] Cancelled.")
            return None

        if answer in ("q", "quit", "exit"):
            print("\n[CatLLM] Cancelled by user.")
            return None

        # Parse which categories to flip
        original = dict(cat_values)
        corrected = dict(cat_values)
        changed = []

        if answer:
            try:
                nums = [int(x.strip()) for x in answer.split(",") if x.strip()]
            except ValueError:
                print("  (Could not parse input — treating as no corrections)")
                nums = []

            for num in nums:
                if 1 <= num <= len(categories):
                    cat_name = categories[num - 1]
                    corrected[cat_name] = 1 - corrected[cat_name]
                    changed.append(cat_name)
                else:
                    print(f"  (Ignoring invalid number: {num})")

            if changed:
                print(f"  Flipped: {', '.join(changed)}")

        corrections.append({
            "input": input_text,
            "original": original,
            "corrected": corrected,
            "changed": changed,
        })
        print()

    # Compute stats
    n_total_fb = len(corrections)
    n_perfect = sum(1 for c in corrections if not c["changed"])
    total_flips = sum(len(c["changed"]) for c in corrections)
    total_decisions = n_total_fb * len(categories)
    accuracy = n_perfect / n_total_fb if n_total_fb > 0 else 0.0
    cat_accuracy = (total_decisions - total_flips) / total_decisions if total_decisions > 0 else 1.0

    return {
        "corrections": corrections,
        "accuracy": accuracy,
        "category_accuracy": cat_accuracy,
        "total_flips": total_flips,
        "sample_indices": sample_indices,
    }


def run_pilot_test(
    input_data,
    categories,
    models,
    classify_ensemble_fn,
    ensemble_kwargs,
    sample_size=10,
):
    """
    Run a pilot classification, collect corrections, and ask whether to proceed.

    Thin wrapper around collect_corrections() that prints a summary and asks
    the user to confirm before the full classification run.

    Returns:
        dict with "proceed" key (bool) plus all keys from collect_corrections(),
        or None if cancelled.
    """
    result = collect_corrections(
        input_data=input_data,
        categories=categories,
        models=models,
        classify_ensemble_fn=classify_ensemble_fn,
        ensemble_kwargs=ensemble_kwargs,
        sample_size=sample_size,
    )

    if result is None:
        return None

    # Print summary
    n_total = len(result["corrections"])
    n_perfect = sum(1 for c in result["corrections"] if not c["changed"])
    n_corrected = n_total - n_perfect
    pct = result["accuracy"] * 100
    cat_pct = result["category_accuracy"] * 100

    print(f"{'=' * 60}")
    print(f"PILOT TEST SUMMARY")
    print(f"  Items fully correct:     {n_perfect}/{n_total} ({pct:.0f}%)")
    print(f"  Items with corrections:  {n_corrected}/{n_total}")
    print(f"  Category-level accuracy: {cat_pct:.1f}%")
    print(f"{'=' * 60}\n")

    if result["accuracy"] < 0.5:
        print(
            "  WARNING: Less than half of the pilot classifications were fully correct.\n"
            "  Consider revising your categories — adding descriptions and examples\n"
            "  significantly improves accuracy. You can also use prompt_tune() to\n"
            "  automatically optimize the classification prompt.\n"
        )
    elif result["accuracy"] < 0.8:
        print(
            "  Some classifications needed corrections. Consider using prompt_tune()\n"
            "  to optimize the prompt before running the full classification.\n"
        )
    else:
        print("  Classifications look good!\n")

    # Ask whether to proceed
    try:
        answer = input("  Proceed with full classification? (Y/n): ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\n[CatLLM] Classification cancelled.")
        result["proceed"] = False
        return result

    result["proceed"] = answer in ("", "y", "yes")

    if not result["proceed"]:
        print("\n[CatLLM] Classification cancelled. Adjust your categories and try again.\n")
    else:
        print("\n[CatLLM] Proceeding with full classification...\n")

    return result
