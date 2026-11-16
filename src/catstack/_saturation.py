"""Saturation prompting: classify one text once per prompt-wording variant.

classify(saturation=True) sends every variant in catstack.prompts.saturation_prompts
(v01 ... v10) for each response, at temperature 0, so the draws differ only in
wording. The main output is, for every response and category, the proportion of
usable draws that assigned the category (the mean assignment rate across the
draws: 2 of 10 -> 0.2), with saturation_draws giving the number of usable draws
(the denominator).

Proportions: the share of usable draws coding the category 1 (a draw whose reply
can't be scored is left out of the denominator).

Failed draws: a draw lost to an API or connection error is sent again, in up to
DRAW_RETRIES extra passes over just the failed variants (with a pause before each),
before the draws are combined. A draw that returned a reply that can't be scored
is not re-sent here: it already had the JSON re-asks, and at temperature 0 the
same prompt gives the same reply. Rows left with fewer usable draws than variants
are reported at the end of the run.

A majority label is also formed, only so the row passes through the usual
consensus / retry machinery (it is not in the output):
  multi-label   category k is 1 when more than half of the usable draws code it 1;
                an exact half goes to v01's code (0 if v01 failed).
  single-label  the category with the most votes; a tie goes to v01's category
                if it is among the tied, else to the lowest-numbered tied category.
"""

import json
import time

from ._utils import classification_codes
from .prompts.saturation_prompts import VARIANT_IDS, render

SATURATION_CREATIVITY = 0  # draws differ in wording only
DRAW_RETRIES = 2           # extra passes over draws that failed with an API error

# Providers whose client accepts a thinking budget (mirrors classify_single).
_THINKING_PROVIDERS = ("google", "openai", "anthropic", "huggingface",
                       "huggingface-together", "ollama")


def _one_draw(client, messages, json_schema, provider, thinking_budget, max_retries,
              json_retries, n_cats, extract_json_fn, formatter_fallback_fn):
    """One variant's call, with classify_single's JSON re-ask and formatter fallback.

    Returns (codes dict "1".."n" -> 0/1 or None if unusable, error or None).
    """
    from ._utils import classification_reply_is_usable

    json_result = '{"1":"e"}'
    for attempt in range(json_retries + 1):
        msgs = messages
        if attempt > 0:
            last = messages[-1]
            msgs = messages[:-1] + [{**last, "content": last["content"]
                                     + "\n\nRespond with ONLY valid JSON, no explanation or additional text."}]
        reply, error = client.complete(
            messages=msgs,
            json_schema=json_schema,
            creativity=SATURATION_CREATIVITY,
            thinking_budget=thinking_budget if provider in _THINKING_PROVIDERS else None,
            max_retries=max_retries,
        )
        if error:
            return None, error
        json_result = extract_json_fn(reply)
        if classification_reply_is_usable(json_result, n_cats):
            break
        if attempt == json_retries and formatter_fallback_fn is not None:
            json_result = formatter_fallback_fn(json_result, reply)
    return classification_codes(json_result, n_cats), None


def combine_draws(draws, n_cats, multi_label=True):
    """Majority across usable draws.

    draws: list in variant order (v01 first); each a codes dict ("1".."n" -> 0/1) or None.
    Returns (majority codes dict, shares list of length n_cats, number of usable
    draws). shares[k] is the proportion of usable draws coding category k+1 as 1.
    The majority is None when no draw was usable.
    """
    usable = [d for d in draws if d is not None]
    n = len(usable)
    if n == 0:
        return None, [None] * n_cats, 0
    keys = [str(k) for k in range(1, n_cats + 1)]
    votes = [sum(int(d.get(k, 0)) for d in usable) for k in keys]
    shares = [v / n for v in votes]
    base = draws[0]                                  # v01, or None if it failed
    if multi_label:
        final = {}
        for k, v in zip(keys, votes):
            if 2 * v > n:
                final[k] = 1
            elif 2 * v == n:
                final[k] = int(base.get(k, 0)) if base is not None else 0
            else:
                final[k] = 0
    else:
        # each usable draw names its first coded category (single-label replies
        # carry one 1; a stray extra 1 counts toward the lowest-numbered one)
        picks = [next((k for k in keys if int(d.get(k, 0)) == 1), None) for d in usable]
        counts = {k: picks.count(k) for k in keys}
        top = max(counts.values())
        if top == 0:
            final = {k: 0 for k in keys}
        else:
            tied = [k for k in keys if counts[k] == top]
            base_pick = (next((k for k in keys if int(base.get(k, 0)) == 1), None)
                         if base is not None else None)
            winner = base_pick if base_pick in tied else tied[0]
            final = {k: int(k == winner) for k in keys}
    return final, shares, n


def run_saturation(client, response_text, categories_str, n_cats, survey_question="",
                   examples_text="", multi_label=True, system_prompt="", json_schema=None,
                   provider="", thinking_budget=0, max_retries=5, json_retries=2,
                   extract_json_fn=None, formatter_fallback_fn=None, variants=VARIANT_IDS,
                   draw_retries=DRAW_RETRIES, retry_delay=1.0):
    """Classify one response once per variant and combine.

    Returns (json_result string, error or None, stats). stats holds "draws" (the
    number of usable draws) and "shares" (per category, the proportion of usable
    draws coding it 1). json_result is the majority label, '{"1":"e"}' with an
    error when no draw is usable.
    """
    def draw(vid):
        messages = render(vid, response_text, categories_str, survey_question=survey_question,
                          examples_text=examples_text, multi_label=multi_label,
                          system_prompt=system_prompt)
        return _one_draw(client, messages, json_schema, provider, thinking_budget,
                         max_retries, json_retries, n_cats, extract_json_fn,
                         formatter_fallback_fn)

    codes, errors = {}, {}
    for vid in variants:
        codes[vid], errors[vid] = draw(vid)
    # Re-send only the draws an API / connection error lost
    for attempt in range(draw_retries):
        failed = [vid for vid in variants if errors[vid]]
        if not failed:
            break
        if retry_delay:
            time.sleep(retry_delay * (attempt + 1))
        for vid in failed:
            codes[vid], errors[vid] = draw(vid)
    draws = [codes[vid] for vid in variants]
    errors = [f"{vid}: {errors[vid]}" for vid in variants if errors[vid]]
    final, shares, n = combine_draws(draws, n_cats, multi_label)
    stats = {"draws": n, "shares": shares, "failed": len(errors)}
    if final is None:
        return '{"1":"e"}', "saturation: no usable draw (" + "; ".join(errors[:3]) + ")", stats
    return json.dumps({k: str(v) for k, v in final.items()}), None, stats


# ---------------------------------------------------------------------------
# Output helpers (used by build_output_dataframes)
# ---------------------------------------------------------------------------

def apply_saturation_proportions(df, stats, model_names, n_cats):
    """Replace each model's category columns with the proportion of usable draws that
    assigned the category, and add saturation_draws_{model} after processing_status.
    With several models, category_k_consensus becomes the mean of the models'
    proportions and the 0/1 agreement / resolved_by columns are dropped.
    stats is {(row, model): stats dict}; rows are matched on input_index."""
    import pandas as pd

    rows = df["input_index"].tolist()
    pos = list(df.columns).index("processing_status") + 1 if "processing_status" in df else len(df.columns)
    for m in model_names:
        got = [stats.get((r, m)) for r in rows]
        for k in range(n_cats):
            col = f"category_{k + 1}_{m}"
            if col in df:
                df[col] = pd.array([g["shares"][k] if g and g["shares"][k] is not None else None
                                    for g in got], dtype="Float64")
        df.insert(pos, f"saturation_draws_{m}",
                  pd.array([g["draws"] if g else None for g in got], dtype="Int64"))
        pos += 1
    if len(model_names) > 1:
        for k in range(1, n_cats + 1):
            cols = [f"category_{k}_{m}" for m in model_names if f"category_{k}_{m}" in df]
            if f"category_{k}_consensus" in df:
                df[f"category_{k}_consensus"] = df[cols].mean(axis=1, skipna=True).astype("Float64")
        df.drop(columns=[c for c in df.columns if c.endswith("_agreement") or c.endswith("_resolved_by")],
                inplace=True)
    # Report rows that ended with fewer usable draws than variants
    n_var = len(VARIANT_IDS)
    for m in model_names:
        d = df[f"saturation_draws_{m}"]
        short = int((d.fillna(0) < n_var).sum())
        if short:
            print(f"[CatLLM] saturation: {short} of {len(d)} rows have fewer than {n_var} usable "
                  f"draws for {m} (min {int(d.fillna(0).min())}); their proportions rest on fewer draws.")
    return df
