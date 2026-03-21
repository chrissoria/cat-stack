"""
Pilot test module for CatLLM.

Provides two capabilities:
1. collect_corrections() — classify a small sample and collect category-level
   user corrections via a browser UI. Used by prompt_tune() and classify(pilot_test=True).
2. run_pilot_test() — wrapper that collects corrections and asks whether to proceed.
"""

import random


def compute_metrics(corrections):
    """
    Compute cell-level accuracy, sensitivity, and precision from corrections.

    Each (item, category) pair is a cell. The model's original output is the
    prediction; the user's corrected output is ground truth.

    - TP: model=1, truth=1  (correctly identified)
    - FP: model=1, truth=0  (false alarm — user flipped 1→0)
    - FN: model=0, truth=1  (missed — user flipped 0→1)
    - TN: model=0, truth=0  (correctly excluded)

    Returns:
        dict with "accuracy", "sensitivity", "precision" (each 0-1 float).
        When a denominator is zero (e.g. no positives), that metric is 1.0.
    """
    tp = fp = fn = tn = 0
    for c in corrections:
        for cat, orig_val in c["original"].items():
            truth_val = c["corrected"][cat]
            if orig_val == 1 and truth_val == 1:
                tp += 1
            elif orig_val == 1 and truth_val == 0:
                fp += 1
            elif orig_val == 0 and truth_val == 1:
                fn += 1
            else:
                tn += 1

    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total > 0 else 1.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0

    return {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "precision": precision,
    }


def collect_corrections(
    input_data,
    categories,
    models,
    classify_ensemble_fn,
    ensemble_kwargs,
    sample_size=10,
    system_prompt="",
    ui="browser",
):
    """
    Classify a random sample and collect per-category user corrections.

    Opens a browser-based review UI where the user can toggle category
    checkboxes for each item, then submit all corrections at once.

    Args:
        input_data: The full input data (list or Series).
        categories: List of category names.
        models: Models list (same format as classify()).
        classify_ensemble_fn: The classify_ensemble callable.
        ensemble_kwargs: Dict of keyword arguments to forward to classify_ensemble.
        sample_size: Number of random items to test. Default 10.
        system_prompt: Optional system prompt to use for this classification run.
        ui: Review interface to use. "browser" (default) opens a local web page
            with checkboxes. "terminal" uses text-based input.

    Returns:
        dict with keys:
            - "corrections": list of dicts, each with:
                - "input": str — the input text
                - "original": dict — {category_name: 0/1} as model classified
                - "corrected": dict — {category_name: 0/1} after user corrections
                - "changed": list of str — category names that were flipped
            - "metrics": dict with "accuracy", "sensitivity", "precision"
              (each 0-1 float), computed cell-wise across all (item, category)
              pairs.
            - "total_flips": int — total number of category-level corrections
            - "sample_indices": list of int indices that were sampled
        Returns None if user cancels.
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
            "corrections": [],
            "metrics": {"accuracy": 1.0, "sensitivity": 1.0, "precision": 1.0},
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

    # Extract per-item category values from the result DataFrame
    review_items = []
    for row_idx in range(len(pilot_result)):
        row = pilot_result.iloc[row_idx]
        input_text = sample_items[row_idx]

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

        review_items.append({
            "input": input_text,
            "values": cat_values,
        })

    # Collect corrections via the chosen UI
    if ui == "browser":
        corrections = _collect_via_browser(review_items, categories)
    else:
        corrections = _collect_via_terminal(review_items, categories)

    if corrections is None:
        return None

    total_flips = sum(len(c["changed"]) for c in corrections)
    metrics = compute_metrics(corrections)

    return {
        "corrections": corrections,
        "metrics": metrics,
        "total_flips": total_flips,
        "sample_indices": sample_indices,
    }


def _collect_via_browser(review_items, categories):
    """Open a browser-based review UI and return corrections."""
    from ._review_ui import open_review_ui
    return open_review_ui(review_items, categories)


def _collect_via_terminal(review_items, categories):
    """Collect corrections via terminal text input (fallback)."""
    corrections = []
    n = len(review_items)

    print(f"\n{'=' * 60}")
    print("RESULTS — Review each classification")
    print("Enter category numbers to flip (e.g. '1,3'), or press Enter if correct.")
    print(f"{'=' * 60}\n")

    for idx, item in enumerate(review_items):
        input_text = item["input"]
        cat_values = item["values"]

        display_text = str(input_text)
        if len(display_text) > 200:
            display_text = display_text[:200] + "..."

        print(f"--- Item {idx + 1}/{n} ---")
        print(f"  Input: {display_text}\n")

        print("  Categories:")
        for cat_idx, cat in enumerate(categories, 1):
            val = cat_values[cat]
            marker = "1" if val else "0"
            cat_display = cat if len(cat) <= 60 else cat[:57] + "..."
            print(f"    {cat_idx}. {cat_display:<60s} = {marker}")
        print()

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

    return corrections


def run_pilot_test(
    input_data,
    categories,
    models,
    classify_ensemble_fn,
    ensemble_kwargs,
    sample_size=10,
    ui="browser",
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
        ui=ui,
    )

    if result is None:
        return None

    # Print summary
    m = result["metrics"]

    print(f"{'=' * 60}")
    print(f"PILOT TEST SUMMARY")
    print(f"  Accuracy:    {m['accuracy'] * 100:.1f}%")
    print(f"  Sensitivity: {m['sensitivity'] * 100:.1f}%")
    print(f"  Precision:   {m['precision'] * 100:.1f}%")
    print(f"  Corrections: {result['total_flips']}")
    print(f"{'=' * 60}\n")

    avg = (m["accuracy"] + m["sensitivity"] + m["precision"]) / 3
    if avg < 0.7:
        print(
            "  WARNING: Average score is below 70%.\n"
            "  Consider revising your categories — adding descriptions and examples\n"
            "  significantly improves accuracy. You can also use prompt_tune() to\n"
            "  automatically optimize the classification prompt.\n"
        )
    elif avg < 0.9:
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
