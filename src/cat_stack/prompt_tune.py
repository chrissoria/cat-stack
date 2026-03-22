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
):
    """
    Automatically optimize the classification prompt using user feedback.

    Runs an iterative loop: classify a sample, collect user corrections,
    then for each category that had errors, ask an LLM to generate a
    targeted instruction for that category. Per-category instructions are
    assembled into the system prompt. Categories themselves are never
    modified.

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
        max_iterations (int): Maximum optimization iterations. Default 3.
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

    Returns:
        dict with keys:
            - "system_prompt": str — the optimized system prompt (best found)
            - "iterations": list of dicts, each with:
                - "iteration": int
                - "system_prompt": str — the prompt used
                - "metrics": dict with "accuracy", "sensitivity", "precision"
                - "per_category": dict mapping category -> metrics dict
                - "total_flips": int — total corrections made
            - "best_iteration": int — which iteration produced the best prompt
            - "per_category_summary": dict mapping category -> aggregated metrics

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

    iterations = []
    current_prompt = ""
    # Per-category instructions that accumulate across iterations.
    # Maps category name -> instruction string for that category.
    cat_instructions = {}
    best_prompt = ""
    best_target = -1.0
    best_iteration = 0

    # Accumulators for per-category summary across all iterations
    agg_per_cat = {cat: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for cat in categories}

    print(f"\n{'=' * 60}")
    print(f"PROMPT TUNING — up to {max_iterations} iteration(s), {sample_size} sample(s) each")
    print(f"  Optimizing for: {optimize}")
    print(f"{'=' * 60}")

    for iteration in range(1, max_iterations + 1):
        print(f"\n--- Iteration {iteration}/{max_iterations} ---")

        if current_prompt:
            display_prompt = current_prompt
            if len(display_prompt) > 200:
                display_prompt = display_prompt[:200] + "..."
            print(f"  Current prompt: {display_prompt}\n")
        else:
            print("  Current prompt: (default — no custom instruction)\n")

        # Step 1: Classify sample and collect corrections
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
            break

        corrections = result["corrections"]
        metrics = result["metrics"]
        total_flips = result["total_flips"]
        target_score = _target_fn(metrics)

        # Per-category metrics for this iteration
        per_cat = _compute_per_category_metrics(corrections, categories)

        # Accumulate into summary
        for cat in categories:
            for key in ("tp", "fp", "fn", "tn"):
                agg_per_cat[cat][key] += per_cat[cat][key]

        # Print iteration summary
        print(f"\n  Iteration {iteration} results:")
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

        iterations.append({
            "iteration": iteration,
            "system_prompt": current_prompt,
            "metrics": metrics,
            "per_category": per_cat,
            "total_flips": total_flips,
        })

        # Track best
        if target_score > best_target:
            best_target = target_score
            best_prompt = current_prompt
            best_iteration = iteration

        # Perfect score — no need to continue
        if total_flips == 0:
            print("\n  All classifications correct — no further tuning needed.")
            break

        # Last iteration — don't generate a new prompt
        if iteration == max_iterations:
            print(f"\n  Reached max iterations ({max_iterations}).")
            break

        # Step 2: Pick the worst-performing category and generate an instruction
        # for it. Only one category changes per iteration so we can isolate
        # what's helping. The meta-LLM sees ALL errors for full context.
        cats_with_errors = [
            cat for cat in categories
            if per_cat[cat]["fp"] > 0 or per_cat[cat]["fn"] > 0
        ]
        if not cats_with_errors:
            break

        # Rank by total errors (FP + FN), pick the worst
        worst_cat = max(
            cats_with_errors,
            key=lambda c: per_cat[c]["fp"] + per_cat[c]["fn"],
        )
        worst_errors = per_cat[worst_cat]["fp"] + per_cat[worst_cat]["fn"]

        print(f"\n  Targeting: {worst_cat} ({worst_errors} errors)")

        instruction = _generate_category_instruction(
            target_category=worst_cat,
            corrections=corrections,
            categories=categories,
            per_cat=per_cat,
            current_instruction=cat_instructions.get(worst_cat, ""),
            description=description,
            survey_question=survey_question,
            multi_label=multi_label,
            optimize=optimize,
            meta_model=meta_model,
            meta_source=meta_source,
            meta_key=meta_key,
            max_retries=max_retries,
        )
        if instruction:
            cat_instructions[worst_cat] = instruction
            print(f"    Updated: {instruction}")
        else:
            print(f"    Failed to generate instruction (keeping previous)")

        # Assemble per-category instructions into system prompt
        current_prompt = _assemble_prompt(cat_instructions, categories)

    # Build per-category summary from accumulated counts
    per_category_summary = {}
    for cat in categories:
        d = agg_per_cat[cat]
        total = d["tp"] + d["fp"] + d["fn"] + d["tn"]
        per_category_summary[cat] = {
            "tp": d["tp"], "fp": d["fp"], "fn": d["fn"], "tn": d["tn"],
            "accuracy": (d["tp"] + d["tn"]) / total if total > 0 else 1.0,
            "sensitivity": d["tp"] / (d["tp"] + d["fn"]) if (d["tp"] + d["fn"]) > 0 else 1.0,
            "precision": d["tp"] / (d["tp"] + d["fp"]) if (d["tp"] + d["fp"]) > 0 else 1.0,
        }

    # Final summary
    print(f"\n{'=' * 60}")
    print(f"PROMPT TUNING COMPLETE")
    print(f"  Iterations run:  {len(iterations)}")
    print(f"  Best iteration:  {best_iteration}")
    print(f"  Optimized for:   {optimize}")
    print(f"  Best target:     {best_target * 100:.1f}%")
    print(f"\n  Aggregated per-category results:")
    print(f"    {'Category':<40s}  {'Acc':>5s}  {'Sens':>5s}  {'Prec':>5s}  {'FP':>3s}  {'FN':>3s}")
    print(f"    {chr(9472) * 40}  {chr(9472) * 5}  {chr(9472) * 5}  {chr(9472) * 5}  {chr(9472) * 3}  {chr(9472) * 3}")
    for cat in categories:
        d = per_category_summary[cat]
        cat_display = cat if len(cat) <= 40 else cat[:37] + "..."
        print(
            f"    {cat_display:<40s}  {d['accuracy']:>4.0%}  {d['sensitivity']:>4.0%}"
            f"  {d['precision']:>4.0%}  {d['fp']:>3d}  {d['fn']:>3d}"
        )
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
        "best_iteration": best_iteration,
        "per_category_summary": per_category_summary,
    }


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
