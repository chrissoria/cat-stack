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

        # Step 2: For each category with errors, generate a targeted instruction
        print("\n  Generating per-category instructions...")
        cats_with_errors = [
            cat for cat in categories
            if per_cat[cat]["fp"] > 0 or per_cat[cat]["fn"] > 0
        ]

        for cat in cats_with_errors:
            instruction = _generate_category_instruction(
                category=cat,
                per_cat_metrics=per_cat[cat],
                corrections=corrections,
                categories=categories,
                current_instruction=cat_instructions.get(cat, ""),
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
                cat_instructions[cat] = instruction
                print(f"    {cat}: updated")
            else:
                print(f"    {cat}: failed (keeping previous)")

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
    category,
    per_cat_metrics,
    corrections,
    categories,
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
    Generate a targeted instruction for a single category based on its errors.

    Sends the meta-LLM only the errors relevant to this category and asks
    for a concise instruction that would fix them.

    Returns:
        str: A one-or-two sentence instruction for this category, or None.
    """
    fp = per_cat_metrics["fp"]
    fn = per_cat_metrics["fn"]

    # Collect examples relevant to this category's errors
    fp_examples = []
    fn_examples = []
    correct_examples = []

    for item in corrections:
        input_text = str(item["input"])
        if len(input_text) > 200:
            input_text = input_text[:200] + "..."

        orig = item["original"][category]
        truth = item["corrected"][category]

        if orig == 1 and truth == 0:
            fp_examples.append(f'  - "{input_text}" (model said 1, should be 0)')
        elif orig == 0 and truth == 1:
            fn_examples.append(f'  - "{input_text}" (model said 0, should be 1)')
        elif orig == truth:
            correct_examples.append(f'  - "{input_text}" = {orig}')

    # Build the focused prompt
    error_section = ""
    if fp_examples:
        error_section += f"FALSE POSITIVES ({fp} — model wrongly assigned this category):\n"
        error_section += "\n".join(fp_examples) + "\n\n"
    if fn_examples:
        error_section += f"FALSE NEGATIVES ({fn} — model missed this category):\n"
        error_section += "\n".join(fn_examples) + "\n\n"

    correct_section = ""
    if correct_examples:
        correct_section = "CORRECTLY CLASSIFIED (preserve these):\n"
        correct_section += "\n".join(correct_examples[:5]) + "\n\n"

    # Context
    context_parts = []
    if description:
        context_parts.append(f"Data: {description}")
    if survey_question:
        context_parts.append(f"Question: {survey_question}")
    context_text = "; ".join(context_parts) if context_parts else ""

    # Other categories for boundary context
    other_cats = [c for c in categories if c != category]
    other_cats_str = ", ".join(other_cats) if other_cats else "(none)"

    # Current instruction for this category
    current_text = f'\nCURRENT INSTRUCTION FOR THIS CATEGORY:\n"{current_instruction}"\n' if current_instruction else ""

    optimize_guidance = {
        "balanced": "",
        "precision": " Focus especially on reducing false positives.",
        "sensitivity": " Focus especially on reducing false negatives.",
    }

    meta_prompt = f"""You are writing a classification guideline for the category "{category}".
{f"Context: {context_text}" if context_text else ""}
Other categories in the scheme: {other_cats_str}
{"Multi-label: a text can belong to multiple categories." if multi_label else "Single-label: each text belongs to exactly one category."}
{current_text}
{error_section}{correct_section}Write a 1-2 sentence instruction that tells a classifier when to assign or not assign the category "{category}". Address the specific errors above — clarify what distinguishes this category from others and what should NOT be included.{optimize_guidance[optimize]}

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
            print(f"    [CatLLM] Error for {category}: {error}")
            return None

        instruction = reply.strip().strip('"').strip("'")
        return instruction

    except Exception as e:
        print(f"    [CatLLM] Failed for {category}: {e}")
        return None
