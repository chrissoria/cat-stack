"""
Pilot test module for CatLLM.

Runs classification on a small random sample and asks the user to validate
results at the category level before committing to the full classification run.
User corrections are formatted as few-shot examples to fine-tune the prompt.
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
    Run a pilot classification on a random sample and collect category-level feedback.

    Classifies a small random subset, displays each item with its per-category
    assignments, and lets the user correct individual categories by number.
    Corrections are formatted as prompt examples for the full run.

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
            - "accuracy": float — fraction of items with zero corrections (0-1)
            - "corrections": list of dicts, each with:
                - "input": str — the input text
                - "original": dict — {category_name: 0/1} as model classified
                - "corrected": dict — {category_name: 0/1} after user corrections
                - "changed": list of str — category names that were flipped
            - "correction_examples": str — formatted text to inject into prompts
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
        return {
            "proceed": True, "accuracy": 1.0, "corrections": [],
            "correction_examples": "", "sample_indices": [],
        }

    # Sample
    actual_sample_size = min(sample_size, n_total)
    sample_indices = sorted(random.sample(range(n_total), actual_sample_size))
    sample_items = [items_list[i] for i in sample_indices]

    print(f"\n[CatLLM] Running pilot test on {actual_sample_size} random item(s)...")
    print("=" * 60)

    # Run classification on the sample
    pilot_kwargs = dict(ensemble_kwargs)
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
        return {
            "proceed": True, "accuracy": 0.0, "corrections": [],
            "correction_examples": "", "sample_indices": sample_indices,
        }

    is_multi_model = len(models) > 1
    corrections = []

    print(f"\n{'=' * 60}")
    print("PILOT TEST RESULTS — Review each classification")
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
        # Columns: category_1, category_2, ... (single) or category_1_consensus, ... (multi)
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
            # Truncate long category names for display
            cat_display = cat if len(cat) <= 60 else cat[:57] + "..."
            print(f"    {cat_idx}. {cat_display:<60s} = {marker}")
        print()

        # Ask for corrections
        try:
            answer = input(
                "  Numbers to flip (e.g. '1,3'), Enter if correct, 'q' to quit: "
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n\n[CatLLM] Pilot test cancelled.")
            return None

        if answer in ("q", "quit", "exit"):
            print("\n[CatLLM] Pilot test cancelled by user.")
            return None

        # Parse which categories to flip
        original = dict(cat_values)
        corrected = dict(cat_values)
        changed = []

        if answer:
            try:
                nums = [int(x.strip()) for x in answer.split(",") if x.strip()]
            except ValueError:
                print("  (Could not parse input — treating as no corrections)\n")
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

    # Build summary
    n_perfect = sum(1 for c in corrections if not c["changed"])
    n_with_corrections = sum(1 for c in corrections if c["changed"])
    n_total_fb = len(corrections)
    accuracy = n_perfect / n_total_fb if n_total_fb > 0 else 0.0
    pct = accuracy * 100

    # Count total category-level flips
    total_flips = sum(len(c["changed"]) for c in corrections)
    total_decisions = n_total_fb * len(categories)
    cat_accuracy = (total_decisions - total_flips) / total_decisions * 100 if total_decisions > 0 else 100.0

    print(f"{'=' * 60}")
    print(f"PILOT TEST SUMMARY")
    print(f"  Items fully correct:    {n_perfect}/{n_total_fb} ({pct:.0f}%)")
    print(f"  Items with corrections: {n_with_corrections}/{n_total_fb}")
    print(f"  Category-level accuracy: {cat_accuracy:.1f}% ({total_decisions - total_flips}/{total_decisions})")
    print(f"{'=' * 60}\n")

    # Build correction examples for prompt injection
    correction_examples = _build_correction_examples(corrections, categories)

    if n_with_corrections > 0:
        print(f"  {n_with_corrections} correction(s) will be used as examples to guide")
        print("  the full classification run.\n")

    if accuracy < 0.5:
        print(
            "  WARNING: Less than half of the pilot classifications were fully correct.\n"
            "  Consider revising your categories — adding descriptions and examples\n"
            "  significantly improves accuracy. For example:\n"
            "\n"
            '    Instead of:  "Positive"\n'
            '    Consider:    "Positive: The response expresses satisfaction, approval,\n'
            "     or happiness (e.g., 'I love this product', 'Great experience')\"\n"
        )
    elif accuracy < 0.8:
        print(
            "  Some classifications needed corrections. Your corrections will be\n"
            "  used to improve the full run.\n"
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
        return {
            "proceed": False, "accuracy": accuracy, "corrections": corrections,
            "correction_examples": correction_examples, "sample_indices": sample_indices,
        }

    proceed = proceed_answer in ("", "y", "yes")

    if not proceed:
        print("\n[CatLLM] Classification cancelled. Adjust your categories and try again.\n")
    else:
        print("\n[CatLLM] Proceeding with full classification...\n")

    return {
        "proceed": proceed,
        "accuracy": accuracy,
        "corrections": corrections,
        "correction_examples": correction_examples,
        "sample_indices": sample_indices,
    }


def _build_correction_examples(corrections, categories):
    """
    Format user corrections as few-shot examples for prompt injection.

    Includes both corrected items (to fix mistakes) and a sample of items
    the model got fully correct (to reinforce good behavior).

    Args:
        corrections: List of correction dicts from pilot test.
        categories: List of category names.

    Returns:
        str: Formatted correction examples text, or "" if no corrections.
    """
    # Separate corrected items from fully-correct items
    corrected_items = [c for c in corrections if c["changed"]]
    correct_items = [c for c in corrections if not c["changed"]]

    if not corrected_items:
        return ""

    lines = [
        "The following are reference examples based on prior review. "
        "Use these to calibrate your classifications:"
    ]

    # Include all corrected items as examples
    for item in corrected_items:
        input_text = str(item["input"])
        if len(input_text) > 300:
            input_text = input_text[:300] + "..."

        lines.append(f'\nText: "{input_text}"')
        lines.append("Correct classification:")
        for cat_idx, cat in enumerate(categories, 1):
            val = item["corrected"].get(cat, 0)
            lines.append(f"  {cat_idx}. {cat} = {val}")

    # Include up to 3 correct items as positive reinforcement
    if correct_items:
        sample_correct = correct_items[:3]
        for item in sample_correct:
            input_text = str(item["input"])
            if len(input_text) > 300:
                input_text = input_text[:300] + "..."

            lines.append(f'\nText: "{input_text}"')
            lines.append("Correct classification:")
            for cat_idx, cat in enumerate(categories, 1):
                val = item["corrected"].get(cat, 0)
                lines.append(f"  {cat_idx}. {cat} = {val}")

    return "\n".join(lines)
