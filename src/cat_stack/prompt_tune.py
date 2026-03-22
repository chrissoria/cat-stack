"""
Automatic Prompt Optimization (APO) for CatLLM.

Iteratively refines the classification system prompt by analyzing errors
at the per-category level:
1. Classify a random sample with the current prompt
2. Collect category-level user corrections
3. For each category with errors, ask an LLM to generate a targeted
   instruction that addresses that category's specific mistakes
4. Assemble per-category instructions into a new system prompt
5. Re-classify the same sample with the new prompt
6. Compare accuracy — keeping the best prompt
7. Repeat until convergence or max iterations

Categories are never modified — only the system prompt changes.
"""

from typing import Union

from ._category_analysis import has_other_category
from ._pilot_test import collect_corrections
from .text_functions_ensemble import classify_ensemble
from ._providers import UnifiedLLMClient, detect_provider


def _compute_per_category_metrics(corrections, categories):
    """
    Compute per-category TP/FP/FN/TN and derived metrics.

    Returns:
        dict mapping category name -> {"tp", "fp", "fn", "tn",
        "accuracy", "sensitivity", "precision"}.
    """
    per_cat = {cat: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for cat in categories}

    for c in corrections:
        for cat in categories:
            orig = c["original"][cat]
            truth = c["corrected"][cat]
            if orig == 1 and truth == 1:
                per_cat[cat]["tp"] += 1
            elif orig == 1 and truth == 0:
                per_cat[cat]["fp"] += 1
            elif orig == 0 and truth == 1:
                per_cat[cat]["fn"] += 1
            else:
                per_cat[cat]["tn"] += 1

    for cat in categories:
        d = per_cat[cat]
        total = d["tp"] + d["fp"] + d["fn"] + d["tn"]
        d["accuracy"] = (d["tp"] + d["tn"]) / total if total > 0 else 1.0
        d["sensitivity"] = d["tp"] / (d["tp"] + d["fn"]) if (d["tp"] + d["fn"]) > 0 else 1.0
        d["precision"] = d["tp"] / (d["tp"] + d["fp"]) if (d["tp"] + d["fp"]) > 0 else 1.0

    return per_cat


def prompt_tune(
    input_data,
    categories,
    api_key=None,
    user_model="gpt-4o",
    model_source="auto",
    models=None,
    description="",
    survey_question: str = "",
    sample_size: int = 10,
    max_iterations: int = 3,
    multi_label: bool = True,
    creativity: float = None,
    use_json_schema: bool = True,
    consensus_threshold: Union[str, float] = "unanimous",
    max_retries: int = 5,
    input_mode: str = None,
    ui: str = "browser",
    optimize: str = "balanced",
    add_other: Union[str, bool] = "prompt",
):
    """
    Automatically optimize the classification prompt using user feedback.

    Uses a coordinate-descent approach: one category changes at a time so we
    can isolate what's helping, while the meta-LLM always sees the full error
    context across all categories.

    Flow:
        1. BASELINE — Classify a random sample using the user's raw categories
           (no system prompt). The user reviews and corrects the model's output.
           This establishes the starting accuracy.

        2. PER-CATEGORY OPTIMIZATION — For each category that had errors
           (worst-first), the meta-LLM generates a targeted instruction.
           It receives ALL errors across ALL categories for full context
           (e.g. boundary confusion between "Positive" and "Neutral") but
           is asked to output guidance for only the target category.

           After each instruction is generated, the sample is re-classified
           to validate improvement:
             - Improved → keep the instruction, try again if errors remain
             - No change → try a different wording (up to max_iterations)
             - Regressed → revert the instruction, try different wording

           Each category gets up to max_iterations attempts before moving on
           to the next. All other category instructions are held constant.

        3. FINAL ASSESSMENT — Before/after comparison showing baseline vs
           best metrics, per-category breakdown, and the assembled prompt.

    The output is a system prompt with per-category instructions, e.g.:
        Classification guidance per category:

        - Neutral: Assign only for factual statements with no emotional valence...
        - Positive: Assign when the text expresses satisfaction or approval...

    Categories that never had errors are omitted — the model already handles
    them correctly without extra guidance. Categories themselves are never
    modified, only the classification instructions change.

    Args:
        input_data: The data to classify (list of text strings or pandas Series).
        categories (list): List of category names for classification.
        api_key (str): API key for the model provider (single-model mode).
        user_model (str): Model name to use. Default "gpt-4o".
        model_source (str): Provider. Default "auto".
        models (list): For multi-model mode, list of (model, provider, api_key)
            tuples. Overrides user_model/api_key/model_source.
        description (str): Description of the input data context.
        survey_question (str): The survey question (provides context).
        sample_size (int): Number of random items to test per iteration. Default 10.
        max_iterations (int): Maximum instruction attempts per category. Each
            error category gets up to this many tries to find an instruction
            that improves it before moving on. Default 3.
        multi_label (bool): Multi-label classification. Default True.
        creativity (float): Temperature setting. None uses model default.
        use_json_schema (bool): Use JSON schema for structured output. Default True.
        consensus_threshold: For multi-model mode. Default "unanimous".
        max_retries (int): Max retries per API call. Default 5.
        input_mode (str): Input mode override. Default None (auto-detect).
        ui (str): Review interface for corrections. "browser" (default) opens
            a local web page with checkboxes. "terminal" uses text-based input.
        optimize (str): Which metric to maximize across iterations.
            "balanced" (default) — average of accuracy, sensitivity, precision.
            "precision" — optimize for precision (minimize false positives).
            "sensitivity" — optimize for sensitivity (minimize false negatives).
        add_other (str or bool): Controls auto-addition of an "Other" catch-all
            category when none is detected.
            - "prompt" (default): Ask the user to accept or reject the suggestion.
            - True: Silently add "Other" without prompting.
            - False: Never add "Other".

    Returns:
        dict with keys:
            - "system_prompt": str — the optimized system prompt (best found)
            - "iterations": list of dicts, each with:
                - "label": str — e.g. "baseline" or "Neutral attempt 2"
                - "system_prompt": str — the prompt used for this run
                - "metrics": dict with "accuracy", "sensitivity", "precision"
                - "per_category": dict mapping category -> metrics dict
                - "total_flips": int — total corrections made
            - "per_category_summary": dict — per-category metrics from the
                best-scoring iteration

    Example:
        >>> import cat_stack as cat
        >>>
        >>> result = cat.prompt_tune(
        ...     input_data=df['responses'],
        ...     categories=["Positive", "Negative", "Neutral"],
        ...     api_key="your-api-key",
        ...     sample_size=10,
        ...     max_iterations=3,
        ... )
        >>> print(result["system_prompt"])
        >>> results = cat.classify(
        ...     input_data=df['responses'],
        ...     categories=["Positive", "Negative", "Neutral"],
        ...     api_key="your-api-key",
        ...     system_prompt=result["system_prompt"],
        ... )
    """
    # Build models list
    if models is None:
        models = [(user_model, model_source, api_key)]

    # Auto-append "Other" catch-all category if missing
    if add_other and categories and categories != "auto":
        if not has_other_category(categories):
            if add_other == "prompt":
                print(
                    "\n[CatLLM] It looks like your categories may not include a catch-all\n"
                    "  'Other' option. Adding one can improve accuracy by giving the\n"
                    "  model an outlet for ambiguous responses instead of forcing them\n"
                    "  into ill-fitting categories.\n"
                    "  (If you already have a catch-all under a different name, choose 'n'.)\n"
                )
                try:
                    answer = input("  Add 'Other' to your categories? (Y/n): ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    answer = "n"
                if answer in ("", "y", "yes"):
                    categories = list(categories) + ["Other"]
                    print(f"  -> Categories are now: {categories}\n")
                else:
                    print("  -> Keeping original categories.\n")
            else:
                # add_other=True — silently add
                categories = list(categories) + ["Other"]
                print(
                    f"[CatLLM] Auto-added 'Other' catch-all category. "
                    f"Categories are now: {categories}  "
                    f"(set add_other=False to disable)"
                )

    # Build ensemble kwargs (shared across all iterations)
    ensemble_kwargs = dict(
        input_description=description,
        survey_question=survey_question,
        creativity=creativity,
        safety=False,
        chain_of_thought=False,
        chain_of_verification=False,
        step_back_prompt=False,
        context_prompt=False,
        thinking_budget=0,
        use_json_schema=use_json_schema,
        fail_strategy="partial",
        max_retries=max_retries,
        retry_delay=1.0,
        row_delay=0.0,
        consensus_threshold=consensus_threshold,
        multi_label=multi_label,
        input_mode=input_mode,
    )

    # Pick the model/key to use for the meta-optimization LLM calls
    meta_model, meta_source, meta_key = models[0]
    if meta_source == "auto":
        meta_source = detect_provider(meta_model)

    # Resolve optimization target to a scoring function
    _optimize_fns = {
        "balanced": lambda m: (m["accuracy"] + m["sensitivity"] + m["precision"]) / 3,
        "precision": lambda m: m["precision"],
        "sensitivity": lambda m: m["sensitivity"],
    }
    if optimize not in _optimize_fns:
        raise ValueError(
            f"optimize must be 'balanced', 'precision', or 'sensitivity', got '{optimize}'"
        )
    _target_fn = _optimize_fns[optimize]

    current_prompt = ""
    # Per-category instructions that accumulate across categories.
    # Maps category name -> instruction string for that category.
    cat_instructions = {}
    best_prompt = ""
    best_target = -1.0

    print(f"\n{'=' * 60}")
    print(f"PROMPT TUNING — {sample_size} sample(s), up to {max_iterations} iteration(s) per category")
    print(f"  Optimizing for: {optimize}")
    print(f"{'=' * 60}")

    # ── Step 1: Baseline classification + user corrections ───────────
    print(f"\n--- Baseline (no prompt) ---")
    print("  Current prompt: (default — no custom instruction)\n")

    result = collect_corrections(
        input_data=input_data,
        categories=categories,
        models=models,
        classify_ensemble_fn=classify_ensemble,
        ensemble_kwargs=ensemble_kwargs,
        sample_size=sample_size,
        system_prompt=current_prompt,
        ui=ui,
    )

    if result is None:
        print("\n[CatLLM] Prompt tuning cancelled.")
        return {
            "system_prompt": "",
            "iterations": [],
            "best_iteration": 0,
            "per_category_summary": {},
        }

    corrections = result["corrections"]
    metrics = result["metrics"]
    total_flips = result["total_flips"]
    baseline_target = _target_fn(metrics)

    # Save ground truth from user corrections for auto-scoring later iterations
    sample_indices = result["sample_indices"]
    ground_truth = {
        i: c["corrected"] for i, c in zip(sample_indices, corrections)
    }

    # Per-category metrics from baseline
    per_cat = _compute_per_category_metrics(corrections, categories)

    # Print baseline summary
    _print_classification_summary("Baseline", metrics, per_cat, categories, total_flips)

    baseline_data = {
        "label": "baseline",
        "system_prompt": "",
        "metrics": metrics,
        "per_category": per_cat,
        "total_flips": total_flips,
    }
    iterations = [baseline_data]

    best_target = baseline_target
    best_prompt = ""

    if total_flips == 0:
        print("\n  All classifications correct — no tuning needed.")
    else:
        # ── Step 2: Iterate through each error category ──────────────
        cats_with_errors = sorted(
            [cat for cat in categories if per_cat[cat]["fp"] > 0 or per_cat[cat]["fn"] > 0],
            key=lambda c: per_cat[c]["fp"] + per_cat[c]["fn"],
            reverse=True,
        )

        print(f"\n  Categories with errors ({len(cats_with_errors)}): {', '.join(cats_with_errors)}")

        for cat_idx, target_cat in enumerate(cats_with_errors, 1):
            cat_errors = per_cat[target_cat]["fp"] + per_cat[target_cat]["fn"]
            print(f"\n{'─' * 60}")
            print(f"  Category {cat_idx}/{len(cats_with_errors)}: {target_cat} ({cat_errors} errors)")
            print(f"  Up to {max_iterations} iteration(s)")

            prev_instruction = cat_instructions.get(target_cat, "")

            for attempt in range(1, max_iterations + 1):
                print(f"\n    Attempt {attempt}/{max_iterations}...")

                instruction = _generate_category_instruction(
                    target_category=target_cat,
                    corrections=corrections,
                    categories=categories,
                    per_cat=per_cat,
                    current_instruction=cat_instructions.get(target_cat, ""),
                    description=description,
                    survey_question=survey_question,
                    multi_label=multi_label,
                    optimize=optimize,
                    meta_model=meta_model,
                    meta_source=meta_source,
                    meta_key=meta_key,
                    max_retries=max_retries,
                )

                if not instruction:
                    print(f"    Failed to generate instruction — skipping")
                    break

                # Apply this instruction and re-classify
                cat_instructions[target_cat] = instruction
                current_prompt = _assemble_prompt(cat_instructions, categories)

                print(f"    Instruction: {instruction}")
                print(f"    Re-classifying...")

                result = _classify_and_score(
                    input_data=input_data,
                    categories=categories,
                    models=models,
                    classify_ensemble_fn=classify_ensemble,
                    ensemble_kwargs=ensemble_kwargs,
                    sample_indices=sample_indices,
                    ground_truth=ground_truth,
                    system_prompt=current_prompt,
                )

                if result is None:
                    print("\n[CatLLM] Re-classification failed.")
                    # Revert this category
                    if prev_instruction:
                        cat_instructions[target_cat] = prev_instruction
                    else:
                        cat_instructions.pop(target_cat, None)
                    current_prompt = _assemble_prompt(cat_instructions, categories)
                    break

                corrections = result["corrections"]
                metrics = result["metrics"]
                total_flips = result["total_flips"]
                target_score = _target_fn(metrics)
                per_cat = _compute_per_category_metrics(corrections, categories)

                new_cat_errors = per_cat[target_cat]["fp"] + per_cat[target_cat]["fn"]

                _print_classification_summary(
                    f"{target_cat} attempt {attempt}", metrics, per_cat, categories, total_flips,
                )

                iterations.append({
                    "label": f"{target_cat} attempt {attempt}",
                    "system_prompt": current_prompt,
                    "metrics": metrics,
                    "per_category": per_cat,
                    "total_flips": total_flips,
                })

                # Track best overall
                if target_score > best_target:
                    best_target = target_score
                    best_prompt = current_prompt

                # Check improvement for this category
                if new_cat_errors < cat_errors:
                    print(f"    Improved: {target_cat} errors {cat_errors} -> {new_cat_errors}")
                    prev_instruction = instruction
                    cat_errors = new_cat_errors
                    if new_cat_errors == 0:
                        print(f"    {target_cat}: all errors fixed!")
                        break
                    # Continue trying if there are remaining errors and attempts left
                elif new_cat_errors == cat_errors:
                    print(f"    No change for {target_cat} ({cat_errors} errors)")
                    # Instruction didn't help — try again with a different one
                else:
                    print(f"    Regressed: {target_cat} errors {cat_errors} -> {new_cat_errors}")
                    # Revert this attempt
                    if prev_instruction:
                        cat_instructions[target_cat] = prev_instruction
                    else:
                        cat_instructions.pop(target_cat, None)
                    current_prompt = _assemble_prompt(cat_instructions, categories)

                if total_flips == 0:
                    print("\n  All classifications correct — stopping early!")
                    break

            if total_flips == 0:
                break

    # ── Step 3: Final validation ─────────────────────────────────────
    # Assemble the final prompt and record it
    current_prompt = _assemble_prompt(cat_instructions, categories)
    if not best_prompt and current_prompt:
        best_prompt = current_prompt

    # Find the best iteration by target score
    best_iter_data = max(iterations, key=lambda it: _target_fn(it["metrics"]))

    # Final summary with before/after comparison
    print(f"\n{'=' * 60}")
    print(f"PROMPT TUNING COMPLETE")
    print(f"  Classification runs:  {len(iterations)}")
    print(f"  Optimized for:        {optimize}")

    if len(iterations) >= 2:
        baseline = iterations[0]

        b = baseline["metrics"]
        f_ = best_iter_data["metrics"]

        def _delta(after, before):
            diff = after - before
            return f"+{diff:.0%}" if diff >= 0 else f"{diff:.0%}"

        print(f"\n  Before vs After (baseline vs best):")
        print(f"    {'Metric':<12s}  {'Baseline':>8s}  {'Best':>8s}  {'Change':>8s}")
        print(f"    {chr(9472) * 12}  {chr(9472) * 8}  {chr(9472) * 8}  {chr(9472) * 8}")
        for metric_name in ("accuracy", "sensitivity", "precision"):
            print(
                f"    {metric_name:<12s}  {b[metric_name]:>7.0%}  {f_[metric_name]:>7.0%}"
                f"  {_delta(f_[metric_name], b[metric_name]):>8s}"
            )
        print(
            f"    {'errors':<12s}  {baseline['total_flips']:>8d}  {best_iter_data['total_flips']:>8d}"
            f"  {best_iter_data['total_flips'] - baseline['total_flips']:>+8d}"
        )

        # Per-category before/after
        print(f"\n  Per-category change (baseline -> best):")
        print(f"    {'Category':<30s}  {'Acc':>11s}  {'Sens':>11s}  {'Prec':>11s}  {'Err':>8s}")
        print(f"    {chr(9472) * 30}  {chr(9472) * 11}  {chr(9472) * 11}  {chr(9472) * 11}  {chr(9472) * 8}")
        for cat in categories:
            b_cat = baseline["per_category"][cat]
            f_cat = best_iter_data["per_category"][cat]
            b_err = b_cat["fp"] + b_cat["fn"]
            f_err = f_cat["fp"] + f_cat["fn"]
            cat_display = cat if len(cat) <= 30 else cat[:27] + "..."
            print(
                f"    {cat_display:<30s}"
                f"  {b_cat['accuracy']:.0%}->{f_cat['accuracy']:.0%}"
                f"  {b_cat['sensitivity']:.0%}->{f_cat['sensitivity']:.0%}"
                f"  {b_cat['precision']:.0%}->{f_cat['precision']:.0%}"
                f"  {b_err}->{f_err}"
            )
    else:
        b = iterations[0]["metrics"] if iterations else {}
        if b:
            print(f"\n  Results:  acc={b['accuracy']:.0%}  sens={b['sensitivity']:.0%}  prec={b['precision']:.0%}")

    if best_prompt:
        print(f"\n  Optimized prompt:")
        for line in best_prompt.split("\n"):
            print(f"    {line}")
    else:
        print(f"\n  Best prompt: (default — no custom instruction needed)")
    print(f"{'=' * 60}\n")

    return {
        "system_prompt": best_prompt,
        "iterations": iterations,
        "per_category_summary": best_iter_data["per_category"],
    }


def _classify_and_score(
    input_data,
    categories,
    models,
    classify_ensemble_fn,
    ensemble_kwargs,
    sample_indices,
    ground_truth,
    system_prompt="",
):
    """
    Re-classify the same sample items and auto-score against saved ground truth.

    No browser UI — the user's corrections from the baseline are reused as the
    answer key.

    Returns:
        Same format as collect_corrections() return value, or None on failure.
    """
    import pandas as pd
    from ._pilot_test import compute_metrics

    # Get the same items that were used in the baseline
    if isinstance(input_data, pd.Series):
        items_list = input_data.tolist()
    else:
        items_list = list(input_data)

    sample_items = [items_list[i] for i in sample_indices]

    # Run classification
    kwargs = dict(ensemble_kwargs)
    kwargs["filename"] = None
    kwargs["save_directory"] = None
    kwargs["progress_callback"] = None
    kwargs["input_data"] = sample_items
    kwargs["categories"] = categories
    kwargs["models"] = models
    if system_prompt:
        kwargs["system_prompt"] = system_prompt

    try:
        pilot_result = classify_ensemble_fn(**kwargs)
    except Exception as e:
        print(f"    [CatLLM] Classification failed: {e}")
        return None

    is_multi_model = len(models) > 1

    # Extract model predictions and score against ground truth
    corrections = []
    for row_idx, sample_idx in enumerate(sample_indices):
        row = pilot_result.iloc[row_idx]
        input_text = sample_items[row_idx]
        truth = ground_truth[sample_idx]

        original = {}
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
            original[cat] = val

        changed = [cat for cat in categories if original[cat] != truth[cat]]

        corrections.append({
            "input": input_text,
            "original": original,
            "corrected": truth,
            "changed": changed,
        })

    total_flips = sum(len(c["changed"]) for c in corrections)
    metrics = compute_metrics(corrections)

    return {
        "corrections": corrections,
        "metrics": metrics,
        "total_flips": total_flips,
        "sample_indices": sample_indices,
    }


def _print_classification_summary(label, metrics, per_cat, categories, total_flips):
    """Print a compact classification summary table."""
    print(f"\n  {label} results:")
    print(f"    Overall:  acc={metrics['accuracy']:.0%}  sens={metrics['sensitivity']:.0%}  prec={metrics['precision']:.0%}  flips={total_flips}")
    print()
    print(f"    {'Category':<40s}  {'Acc':>5s}  {'Sens':>5s}  {'Prec':>5s}  {'FP':>3s}  {'FN':>3s}")
    print(f"    {chr(9472) * 40}  {chr(9472) * 5}  {chr(9472) * 5}  {chr(9472) * 5}  {chr(9472) * 3}  {chr(9472) * 3}")
    for cat in categories:
        d = per_cat[cat]
        cat_display = cat if len(cat) <= 40 else cat[:37] + "..."
        print(
            f"    {cat_display:<40s}  {d['accuracy']:>4.0%}  {d['sensitivity']:>4.0%}"
            f"  {d['precision']:>4.0%}  {d['fp']:>3d}  {d['fn']:>3d}"
        )


def _assemble_prompt(cat_instructions, categories):
    """
    Combine per-category instructions into a single system prompt.

    Only includes categories that have instructions (i.e. ones that had
    errors and got a targeted fix). Categories with no errors are omitted
    — the model already handles them correctly with default behavior.
    """
    if not cat_instructions:
        return ""

    lines = ["Classification guidance per category:", ""]
    for cat in categories:
        if cat in cat_instructions:
            lines.append(f"- {cat}: {cat_instructions[cat]}")

    return "\n".join(lines)


def _generate_category_instruction(
    target_category,
    corrections,
    categories,
    per_cat,
    current_instruction,
    description,
    survey_question,
    multi_label,
    optimize,
    meta_model,
    meta_source,
    meta_key,
    max_retries,
):
    """
    Generate a targeted instruction for one category, given full error context.

    The meta-LLM sees ALL errors across ALL categories so it understands the
    full picture (e.g. boundary confusion between categories), but is asked
    to produce an instruction only for the target category.

    Returns:
        str: A one-or-two sentence instruction for the target category, or None.
    """
    # Build full error context across all categories
    all_error_lines = []
    target_fp_examples = []
    target_fn_examples = []
    target_correct_examples = []

    for item in corrections:
        input_text = str(item["input"])
        if len(input_text) > 200:
            input_text = input_text[:200] + "..."

        # Full context: all errors for this item
        if item["changed"]:
            entry = f'Text: "{input_text}"\n'
            for cat in item["changed"]:
                orig = item["original"][cat]
                corr = item["corrected"][cat]
                marker = " <<<" if cat == target_category else ""
                entry += f"    - {cat}: model={orig}, correct={corr}{marker}\n"
            all_error_lines.append(entry)

        # Target category specifics
        orig = item["original"][target_category]
        truth = item["corrected"][target_category]
        if orig == 1 and truth == 0:
            target_fp_examples.append(f'  - "{input_text}"')
        elif orig == 0 and truth == 1:
            target_fn_examples.append(f'  - "{input_text}"')
        elif orig == truth:
            target_correct_examples.append(f'  - "{input_text}" = {orig}')

    all_errors_text = "\n".join(all_error_lines) if all_error_lines else "(no errors)"

    # Target category error summary
    target_section = f'ERRORS FOR "{target_category}" SPECIFICALLY:\n'
    fp = per_cat[target_category]["fp"]
    fn = per_cat[target_category]["fn"]
    if target_fp_examples:
        target_section += f"  False positives ({fp} — model wrongly assigned this category):\n"
        target_section += "\n".join(target_fp_examples) + "\n"
    if target_fn_examples:
        target_section += f"  False negatives ({fn} — model missed this category):\n"
        target_section += "\n".join(target_fn_examples) + "\n"
    if target_correct_examples:
        target_section += f"  Correct ({len(target_correct_examples)} — preserve these):\n"
        target_section += "\n".join(target_correct_examples[:5]) + "\n"

    # Per-category metrics summary
    metrics_lines = []
    for cat in categories:
        d = per_cat[cat]
        errors = d["fp"] + d["fn"]
        marker = " <<<" if cat == target_category else ""
        metrics_lines.append(
            f"    {cat}: acc={d['accuracy']:.0%} sens={d['sensitivity']:.0%} "
            f"prec={d['precision']:.0%} (FP={d['fp']}, FN={d['fn']}){marker}"
        )
    metrics_text = "\n".join(metrics_lines)

    # Context
    context_parts = []
    if description:
        context_parts.append(f"Data: {description}")
    if survey_question:
        context_parts.append(f"Question: {survey_question}")
    context_text = "; ".join(context_parts) if context_parts else ""

    # Current instruction
    current_text = f'\nCURRENT INSTRUCTION FOR THIS CATEGORY:\n"{current_instruction}"\n' if current_instruction else ""

    optimize_guidance = {
        "balanced": "",
        "precision": " Focus especially on reducing false positives.",
        "sensitivity": " Focus especially on reducing false negatives.",
    }

    meta_prompt = f"""You are improving a text classification system. Your job is to write an
instruction for the category "{target_category}" that fixes its errors.
{f"Context: {context_text}" if context_text else ""}
{"Multi-label: a text can belong to multiple categories." if multi_label else "Single-label: each text belongs to exactly one category."}

PER-CATEGORY PERFORMANCE:
{metrics_text}

ALL ERRORS ACROSS ALL CATEGORIES (<<< marks errors involving your target):
{all_error_lines and chr(10).join(all_error_lines) or "(no errors)"}

{target_section}
{current_text}
Write a 1-2 sentence instruction for the category "{target_category}" that tells
a classifier when to assign and when NOT to assign it. Use the full error context
above to understand how this category relates to others, but only output guidance
for "{target_category}".{optimize_guidance[optimize]}

Return ONLY the instruction. No preamble, no quotes, no formatting."""

    try:
        client = UnifiedLLMClient(
            provider=meta_source,
            api_key=meta_key,
            model=meta_model,
        )

        reply, error = client.complete(
            messages=[{"role": "user", "content": meta_prompt}],
            force_json=False,
            max_retries=max_retries,
        )

        if error:
            print(f"    [CatLLM] Error for {target_category}: {error}")
            return None

        instruction = reply.strip().strip('"').strip("'")
        return instruction

    except Exception as e:
        print(f"    [CatLLM] Failed for {target_category}: {e}")
        return None
