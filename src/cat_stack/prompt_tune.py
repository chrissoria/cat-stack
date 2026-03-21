"""
Automatic Category Optimization for CatLLM.

Iteratively refines category definitions by:
1. Classifying a random sample with the current categories
2. Collecting category-level user corrections
3. Asking an LLM to analyze the errors and produce enriched category
   descriptions (keeping names fixed)
4. Re-classifying with the improved categories
5. Keeping the best-scoring category set
6. Repeating until convergence or max iterations
"""

import json
from typing import Union

from ._pilot_test import collect_corrections
from .text_functions_ensemble import classify_ensemble
from ._providers import UnifiedLLMClient, detect_provider


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
    Automatically optimize category definitions using user feedback.

    Runs an iterative loop: classify a sample, collect user corrections, ask
    an LLM to analyze the errors and enrich the category descriptions, then
    re-classify to verify improvement. Returns the best category definitions.

    Category names are kept fixed — only descriptions and examples are added
    or refined. This avoids overfitting to the small sample while improving
    the model's understanding of what each category means.

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
            - "categories": list of str — the optimized category definitions
            - "iterations": list of dicts, each with:
                - "iteration": int
                - "categories": list of str — categories used
                - "metrics": dict with "accuracy", "sensitivity", "precision"
                - "total_flips": int — total corrections made
            - "best_iteration": int — which iteration produced the best categories

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
        >>> print(result["categories"])
        >>> # Use the optimized categories for full classification
        >>> results = cat.classify(
        ...     input_data=df['responses'],
        ...     categories=result["categories"],
        ...     api_key="your-api-key",
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

    # Extract original category names (before any " — " descriptions)
    original_names = [c.split(" — ")[0].split(" - ")[0].strip() for c in categories]

    iterations = []
    current_categories = list(categories)
    best_categories = list(categories)
    best_target = -1.0
    best_iteration = 0

    print(f"\n{'=' * 60}")
    print(f"CATEGORY TUNING — up to {max_iterations} iteration(s), {sample_size} sample(s) each")
    print(f"  Optimizing for: {optimize}")
    print(f"{'=' * 60}")

    for iteration in range(1, max_iterations + 1):
        print(f"\n--- Iteration {iteration}/{max_iterations} ---")

        # Show current categories
        print("  Categories:")
        for i, cat in enumerate(current_categories, 1):
            cat_display = cat if len(cat) <= 70 else cat[:67] + "..."
            print(f"    {i}. {cat_display}")
        print()

        # Step 1: Classify sample and collect corrections
        result = collect_corrections(
            input_data=input_data,
            categories=current_categories,
            models=models,
            classify_ensemble_fn=classify_ensemble,
            ensemble_kwargs=ensemble_kwargs,
            sample_size=sample_size,
            ui=ui,
        )

        if result is None:
            print("\n[CatLLM] Category tuning cancelled.")
            break

        corrections = result["corrections"]
        metrics = result["metrics"]
        total_flips = result["total_flips"]
        target_score = _target_fn(metrics)

        # Print iteration summary
        print(f"\n  Iteration {iteration} results:")
        print(f"    Accuracy:    {metrics['accuracy'] * 100:.1f}%")
        print(f"    Sensitivity: {metrics['sensitivity'] * 100:.1f}%")
        print(f"    Precision:   {metrics['precision'] * 100:.1f}%")
        print(f"    Corrections: {total_flips}")

        iterations.append({
            "iteration": iteration,
            "categories": list(current_categories),
            "metrics": metrics,
            "total_flips": total_flips,
        })

        # Track best
        if target_score > best_target:
            best_target = target_score
            best_categories = list(current_categories)
            best_iteration = iteration

        # Perfect score — no need to continue
        if total_flips == 0:
            print("\n  All classifications correct — no further tuning needed.")
            break

        # Last iteration — don't generate new categories
        if iteration == max_iterations:
            print(f"\n  Reached max iterations ({max_iterations}).")
            break

        # Step 2: Generate improved categories via meta-LLM call
        print("\n  Generating improved category definitions...")
        new_categories = _generate_improved_categories(
            corrections=corrections,
            categories=current_categories,
            original_names=original_names,
            description=description,
            survey_question=survey_question,
            multi_label=multi_label,
            optimize=optimize,
            meta_model=meta_model,
            meta_source=meta_source,
            meta_key=meta_key,
            max_retries=max_retries,
        )

        if new_categories is None:
            print("  Failed to generate improved categories. Stopping.")
            break

        current_categories = new_categories

    # Final summary
    print(f"\n{'=' * 60}")
    print(f"CATEGORY TUNING COMPLETE")
    print(f"  Iterations run:  {len(iterations)}")
    print(f"  Best iteration:  {best_iteration}")
    print(f"  Optimized for:   {optimize}")
    print(f"  Best target:     {best_target * 100:.1f}%")
    print(f"\n  Optimized categories:")
    for i, cat in enumerate(best_categories, 1):
        print(f"    {i}. {cat}")
    print(f"{'=' * 60}\n")

    return {
        "categories": best_categories,
        "iterations": iterations,
        "best_iteration": best_iteration,
    }


def _generate_improved_categories(
    corrections,
    categories,
    original_names,
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
    Use an LLM to analyze classification errors and generate improved category
    definitions. Category names stay fixed; only descriptions and examples
    are added or refined.

    Returns:
        list of str: The improved category definitions, or None on failure.
    """
    # Format the error analysis
    error_lines = []
    correct_lines = []

    for item in corrections:
        input_text = str(item["input"])
        if len(input_text) > 300:
            input_text = input_text[:300] + "..."

        if item["changed"]:
            entry = f'Text: "{input_text}"\n'
            entry += "  Model output vs correct:\n"
            for cat in item["changed"]:
                orig = item["original"][cat]
                corr = item["corrected"][cat]
                entry += f"    - {cat}: model said {orig}, should be {corr}\n"
            error_lines.append(entry)
        else:
            entry = f'Text: "{input_text}"\n'
            entry += "  Classification: "
            assigned = [cat for cat, val in item["corrected"].items() if val == 1]
            entry += ", ".join(assigned) if assigned else "(none)"
            correct_lines.append(entry)

    errors_text = "\n".join(error_lines) if error_lines else "(no errors)"
    correct_text = "\n".join(correct_lines) if correct_lines else "(none)"

    # Build current categories display
    categories_str = "\n".join(f"  {i+1}. {cat}" for i, cat in enumerate(categories))
    original_names_str = "\n".join(f"  {i+1}. {name}" for i, name in enumerate(original_names))
    label_mode = "multi-label (multiple categories can apply)" if multi_label else "single-label (exactly one category)"

    # Optimization emphasis
    optimize_guidance = {
        "balanced": "Balance accuracy, sensitivity, and precision equally.",
        "precision": "Prioritize precision — add constraints that reduce false positives (model assigning 1 when it should be 0).",
        "sensitivity": "Prioritize sensitivity — add guidance that reduces false negatives (model assigning 0 when it should be 1).",
    }

    # Context
    context_parts = []
    if description:
        context_parts.append(f"Data description: {description}")
    if survey_question:
        context_parts.append(f"Survey question: {survey_question}")
    context_text = "\n".join(context_parts) if context_parts else "(no additional context)"

    # Build JSON schema for structured output
    schema_example = json.dumps(
        {name: f"{name} — description. Example: ..." for name in original_names[:2]},
        indent=2,
    )

    meta_prompt = f"""You are an expert at defining classification categories for text analysis.

TASK: Analyze the classification errors below and produce improved category definitions.
You must keep the exact category NAMES fixed but can add or refine descriptions,
clarifications, and examples for each category.

CLASSIFICATION SETUP:
- Mode: {label_mode}
- Context: {context_text}

ORIGINAL CATEGORY NAMES (these must stay exactly the same):
{original_names_str}

CURRENT CATEGORY DEFINITIONS:
{categories_str}

MISCLASSIFIED ITEMS (errors the definitions must fix):
{errors_text}

CORRECTLY CLASSIFIED ITEMS (definitions must preserve these):
{correct_text}

OPTIMIZATION TARGET:
{optimize_guidance[optimize]}

INSTRUCTIONS:
1. Analyze what the model is getting wrong — look for patterns in the errors.
   Are certain categories confused? Is the model over- or under-classifying?
2. For each category, write an improved definition that:
   - Starts with the exact original category name, followed by " — "
   - Adds a clear description of what belongs in this category
   - Clarifies boundary cases between confused categories
   - Includes 1-2 short examples if helpful
   - Is concise (aim for one line per category)
3. Return a JSON object mapping each original category name to its improved
   definition string.

Example format:
{schema_example}

Return ONLY the JSON object. No explanation, no preamble, no markdown."""

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
            print(f"  [CatLLM] Meta-prompt error: {error}")
            return None

        # Clean up and parse
        text = reply.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()

        mapping = json.loads(text)

        # Build new categories list preserving original order
        new_categories = []
        for name in original_names:
            if name in mapping:
                new_categories.append(mapping[name])
            else:
                # Fallback: keep existing definition
                idx = original_names.index(name)
                new_categories.append(categories[idx])

        return new_categories

    except json.JSONDecodeError as e:
        print(f"  [CatLLM] Could not parse LLM response as JSON: {e}")
        return None
    except Exception as e:
        print(f"  [CatLLM] Failed to generate improved categories: {e}")
        return None
