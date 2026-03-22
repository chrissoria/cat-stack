"""
Iterative category testing for CatLLM (prompt_tune).

Runs the user's categories through repeated classify → correct cycles
on fresh random samples each iteration. Categories are never modified —
the value is in the per-category diagnostic metrics that show which
categories the model handles well and which it struggles with.
"""

from typing import Union

from ._pilot_test import collect_corrections
from .text_functions_ensemble import classify_ensemble


def _compute_per_category_metrics(corrections, categories):
    """
    Compute per-category TP/FP/FN/TN and derived metrics.

    Returns:
        dict mapping category name → {"tp", "fp", "fn", "tn",
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
    Iteratively test categories against random samples and collect corrections.

    Each iteration draws a fresh random sample, classifies it with the user's
    exact categories (unchanged), and collects per-category user corrections.
    Per-category metrics are reported so the user can see which categories
    the model handles well and which need better definitions.

    Args:
        input_data: The data to classify (list of text strings or pandas Series).
        categories (list): List of category names/definitions for classification.
        api_key (str): API key for the model provider (single-model mode).
        user_model (str): Model name to use. Default "gpt-4o".
        model_source (str): Provider. Default "auto".
        models (list): For multi-model mode, list of (model, provider, api_key)
            tuples. Overrides user_model/api_key/model_source.
        description (str): Description of the input data context.
        survey_question (str): The survey question (provides context).
        sample_size (int): Number of random items to test per iteration. Default 10.
        max_iterations (int): Maximum iterations. Default 3.
        multi_label (bool): Multi-label classification. Default True.
        creativity (float): Temperature setting. None uses model default.
        use_json_schema (bool): Use JSON schema for structured output. Default True.
        consensus_threshold: For multi-model mode. Default "unanimous".
        max_retries (int): Max retries per API call. Default 5.
        input_mode (str): Input mode override. Default None (auto-detect).
        ui (str): Review interface for corrections. "browser" (default) opens
            a local web page with checkboxes. "terminal" uses text-based input.
        optimize (str): Which metric to highlight in the summary.
            "balanced" (default) — average of accuracy, sensitivity, precision.
            "precision" — focus on precision.
            "sensitivity" — focus on sensitivity.

    Returns:
        dict with keys:
            - "categories": list of str — the user's original categories (unchanged)
            - "iterations": list of dicts, each with:
                - "iteration": int
                - "metrics": dict with "accuracy", "sensitivity", "precision"
                - "per_category": dict mapping category → metrics dict
                - "total_flips": int — total corrections made
            - "best_iteration": int — which iteration had the best target score
            - "per_category_summary": dict mapping category → aggregated metrics
              across all iterations

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
        >>> # See which categories need work
        >>> for cat_name, m in result["per_category_summary"].items():
        ...     print(f"{cat_name}: acc={m['accuracy']:.0%} sens={m['sensitivity']:.0%} prec={m['precision']:.0%}")
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
    best_target = -1.0
    best_iteration = 0

    # Accumulators for per-category summary across all iterations
    agg_per_cat = {cat: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for cat in categories}

    print(f"\n{'=' * 60}")
    print(f"PROMPT TUNE — {max_iterations} iteration(s), {sample_size} sample(s) each")
    print(f"  Metric focus: {optimize}")
    print(f"  Categories:")
    for i, cat in enumerate(categories, 1):
        cat_display = cat if len(cat) <= 65 else cat[:62] + "..."
        print(f"    {i}. {cat_display}")
    print(f"{'=' * 60}")

    for iteration in range(1, max_iterations + 1):
        print(f"\n--- Iteration {iteration}/{max_iterations} ---")

        # Classify a fresh random sample and collect corrections
        result = collect_corrections(
            input_data=input_data,
            categories=categories,
            models=models,
            classify_ensemble_fn=classify_ensemble,
            ensemble_kwargs=ensemble_kwargs,
            sample_size=sample_size,
            ui=ui,
        )

        if result is None:
            print("\n[CatLLM] Prompt tune cancelled.")
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
        print(f"    {'─' * 40}  {'─' * 5}  {'─' * 5}  {'─' * 5}  {'─' * 3}  {'─' * 3}")
        for cat in categories:
            d = per_cat[cat]
            cat_display = cat if len(cat) <= 40 else cat[:37] + "..."
            print(
                f"    {cat_display:<40s}  {d['accuracy']:>4.0%}  {d['sensitivity']:>4.0%}"
                f"  {d['precision']:>4.0%}  {d['fp']:>3d}  {d['fn']:>3d}"
            )

        iterations.append({
            "iteration": iteration,
            "metrics": metrics,
            "per_category": per_cat,
            "total_flips": total_flips,
        })

        # Track best
        if target_score > best_target:
            best_target = target_score
            best_iteration = iteration

        # Perfect score — no need to continue
        if total_flips == 0:
            print("\n  All classifications correct — no further iterations needed.")
            break

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
    print(f"PROMPT TUNE COMPLETE — {len(iterations)} iteration(s)")
    print(f"\n  Aggregated per-category results:")
    print(f"    {'Category':<40s}  {'Acc':>5s}  {'Sens':>5s}  {'Prec':>5s}  {'FP':>3s}  {'FN':>3s}")
    print(f"    {'─' * 40}  {'─' * 5}  {'─' * 5}  {'─' * 5}  {'─' * 3}  {'─' * 3}")
    for cat in categories:
        d = per_category_summary[cat]
        cat_display = cat if len(cat) <= 40 else cat[:37] + "..."
        print(
            f"    {cat_display:<40s}  {d['accuracy']:>4.0%}  {d['sensitivity']:>4.0%}"
            f"  {d['precision']:>4.0%}  {d['fp']:>3d}  {d['fn']:>3d}"
        )
    print(f"{'=' * 60}\n")

    return {
        "categories": list(categories),
        "iterations": iterations,
        "best_iteration": best_iteration,
        "per_category_summary": per_category_summary,
    }
