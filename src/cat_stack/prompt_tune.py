"""
Automatic Prompt Optimization (APO) for CatLLM.

Iteratively refines the classification system prompt by:
1. Classifying a random sample with the current prompt
2. Collecting category-level user corrections
3. Asking an LLM to analyze the gap between model output and corrections
4. Generating a revised prompt
5. Re-classifying the same sample with the new prompt
6. Comparing accuracy — keeping the best prompt
7. Repeating until convergence or max iterations
"""

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
):
    """
    Automatically optimize the classification prompt using user feedback.

    Runs an iterative loop: classify a sample, collect user corrections, ask
    an LLM to analyze the errors and generate a better prompt, then re-classify
    to verify improvement. Returns the best system prompt found.

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

    Returns:
        dict with keys:
            - "system_prompt": str — the optimized system prompt (best found)
            - "iterations": list of dicts, each with:
                - "iteration": int
                - "system_prompt": str — the prompt used
                - "accuracy": float — item-level accuracy (0-1)
                - "category_accuracy": float — category-level accuracy (0-1)
                - "total_flips": int — total corrections made
            - "best_iteration": int — which iteration produced the best prompt

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
        >>> # Use the optimized prompt for full classification
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

    iterations = []
    current_prompt = ""
    best_prompt = ""
    best_cat_accuracy = -1.0
    best_iteration = 0

    print(f"\n{'=' * 60}")
    print(f"PROMPT TUNING — up to {max_iterations} iteration(s), {sample_size} sample(s) each")
    print(f"{'=' * 60}")

    for iteration in range(1, max_iterations + 1):
        print(f"\n--- Iteration {iteration}/{max_iterations} ---")

        if current_prompt:
            # Truncate for display
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
        )

        if result is None:
            print("\n[CatLLM] Prompt tuning cancelled.")
            break

        corrections = result["corrections"]
        accuracy = result["accuracy"]
        cat_accuracy = result["category_accuracy"]
        total_flips = result["total_flips"]

        # Print iteration summary
        n_total = len(corrections)
        n_perfect = sum(1 for c in corrections if not c["changed"])
        cat_pct = cat_accuracy * 100

        print(f"\n  Iteration {iteration} results:")
        print(f"    Items fully correct:     {n_perfect}/{n_total}")
        print(f"    Category-level accuracy: {cat_pct:.1f}%")
        print(f"    Total corrections:       {total_flips}")

        iterations.append({
            "iteration": iteration,
            "system_prompt": current_prompt,
            "accuracy": accuracy,
            "category_accuracy": cat_accuracy,
            "total_flips": total_flips,
        })

        # Track best
        if cat_accuracy > best_cat_accuracy:
            best_cat_accuracy = cat_accuracy
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

        # Step 2: Generate improved prompt via meta-LLM call
        print("\n  Generating improved prompt...")
        new_prompt = _generate_improved_prompt(
            corrections=corrections,
            categories=categories,
            current_prompt=current_prompt,
            description=description,
            survey_question=survey_question,
            multi_label=multi_label,
            meta_model=meta_model,
            meta_source=meta_source,
            meta_key=meta_key,
            max_retries=max_retries,
        )

        if new_prompt is None:
            print("  Failed to generate improved prompt. Stopping.")
            break

        current_prompt = new_prompt

    # Final summary
    print(f"\n{'=' * 60}")
    print(f"PROMPT TUNING COMPLETE")
    print(f"  Iterations run:  {len(iterations)}")
    print(f"  Best iteration:  {best_iteration}")
    print(f"  Best accuracy:   {best_cat_accuracy * 100:.1f}% (category-level)")
    if best_prompt:
        print(f"  Optimized prompt:")
        for line in best_prompt.split("\n"):
            print(f"    {line}")
    else:
        print(f"  Best prompt: (default — no custom instruction needed)")
    print(f"{'=' * 60}\n")

    return {
        "system_prompt": best_prompt,
        "iterations": iterations,
        "best_iteration": best_iteration,
    }


def _generate_improved_prompt(
    corrections,
    categories,
    current_prompt,
    description,
    survey_question,
    multi_label,
    meta_model,
    meta_source,
    meta_key,
    max_retries,
):
    """
    Use an LLM to analyze classification errors and generate an improved prompt.

    Sends a meta-prompt with the current instruction, the errors observed, and
    asks the model to produce a better classification instruction.

    Returns:
        str: The improved system prompt, or None on failure.
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

    # Build categories description
    categories_str = "\n".join(f"  {i+1}. {cat}" for i, cat in enumerate(categories))
    label_mode = "multi-label (multiple categories can apply)" if multi_label else "single-label (exactly one category)"

    # Context
    context_parts = []
    if description:
        context_parts.append(f"Data description: {description}")
    if survey_question:
        context_parts.append(f"Survey question: {survey_question}")
    context_text = "\n".join(context_parts) if context_parts else "(no additional context)"

    # Current prompt
    current_prompt_text = current_prompt if current_prompt else "(no custom instruction — using default)"

    meta_prompt = f"""You are an expert prompt engineer optimizing a text classification system.

TASK: Analyze the classification errors below and generate an improved instruction that
would fix the errors while maintaining correct classifications.

CLASSIFICATION SETUP:
- Mode: {label_mode}
- Categories:
{categories_str}
- Context: {context_text}

CURRENT INSTRUCTION:
{current_prompt_text}

MISCLASSIFIED ITEMS (errors the instruction must fix):
{errors_text}

CORRECTLY CLASSIFIED ITEMS (the instruction must preserve these):
{correct_text}

INSTRUCTIONS FOR YOU:
1. Analyze what the model is getting wrong — look for patterns in the errors.
   Are certain categories confused? Is the model over- or under-classifying?
2. Generate an improved classification instruction that:
   - Addresses the specific error patterns you identified
   - Clarifies boundary cases between confused categories
   - Preserves correct behavior on the items that were already right
   - Is concise and actionable (not verbose)
3. Return ONLY the improved instruction text. No explanation, no preamble,
   no markdown formatting. Just the instruction itself."""

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

        # Clean up the response
        improved = reply.strip()
        # Remove markdown code fences if present
        if improved.startswith("```"):
            lines = improved.split("\n")
            # Remove first and last ``` lines
            lines = [l for l in lines if not l.strip().startswith("```")]
            improved = "\n".join(lines).strip()

        return improved

    except Exception as e:
        print(f"  [CatLLM] Failed to generate improved prompt: {e}")
        return None
