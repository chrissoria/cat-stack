"""Saturation prompting: classify one text once per prompt in a grid of equivalent prompts.

classify(saturation=True) sends every variant in catstack.prompts.saturation_prompts
(v01 ... v10) for each response, at temperature 0, so the draws differ only in
wording. saturation_orders > 1 crosses each wording with that many category
orders (the original order plus shuffled ones; see "The prompt grid" below), and
saturation_repeats sends each prompt more than once. The main output is, for
every response and category, the proportion of usable draws that assigned the
category (the mean assignment rate across the draws: 2 of 10 -> 0.2), with
saturation_draws giving the number of usable draws (the denominator).

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
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor

from ._utils import classification_codes
from .prompts.saturation_prompts import VARIANT_IDS, render

SATURATION_CREATIVITY = 0  # draws differ in wording (and category order) only
DRAW_RETRIES = 2           # extra passes over draws that failed with an API error
SATURATION_SEED = 20261116  # default seed for the shuffled category orders


# ---------------------------------------------------------------------------
# The prompt grid: wording variants x category orders x repeats
# ---------------------------------------------------------------------------
#
# Every wording variant is crossed with the same category orders, so wording
# and order effects can be told apart. Order 0 is always the original order,
# which makes (v01, order 0) the plain single-call prompt. The other orders
# are distinct random permutations from a seeded generator, the same for every
# response. Each shown order is numbered 1..K from the top; a reply is mapped
# back to the original category numbers before the draws are combined.
#
# K categories have only K! orders, so the grid holds at most
# len(variants) * K! distinct prompts: 20 for 2 categories and 60 for 3 with
# the ten wordings. An order count above K! is capped at K!, and every order is
# used. At temperature 0 a repeated prompt returns the same reply, so repeats
# are for models that can't run at temperature 0 (a sampling-noise control).

def max_prompts(n_cats, n_variants=len(VARIANT_IDS)):
    """Distinct prompts available: n_variants wordings x n_cats! category orders."""
    return n_variants * math.factorial(n_cats)


def category_orders(n_cats, n_orders=1, seed=SATURATION_SEED):
    """The category orders for a saturation grid, as tuples of original indexes
    (order[j] is the original 0-based index of the category shown at position j).

    The original order comes first. n_orders is capped at n_cats!; at or above
    the cap every order is returned (in lexicographic order, original first).
    Below it, the other orders are distinct permutations drawn with
    random.Random(seed), so the same (n_cats, n_orders, seed) always gives the
    same orders.
    """
    if not isinstance(n_orders, int) or isinstance(n_orders, bool) or n_orders < 1:
        raise ValueError(f"the number of category orders must be a positive integer, got {n_orders!r}")
    identity = tuple(range(n_cats))
    if n_orders >= math.factorial(n_cats):
        import itertools
        return list(itertools.permutations(identity))   # identity is first
    rng = random.Random(seed)
    orders, seen = [identity], {identity}
    while len(orders) < n_orders:
        perm = tuple(rng.sample(identity, n_cats))
        if perm not in seen:
            seen.add(perm)
            orders.append(perm)
    return orders


def draw_id(variant, order, repeat):
    """Label of one draw in the grid, e.g. 'v03_o07_r00'."""
    return f"{variant}_o{order:02d}_r{repeat:02d}"


def check_saturation_options(variants=None, orders=1, repeats=1, workers=1):
    """Validate classify()'s saturation_* options; returns the variant ids to use."""
    if variants is None:
        variants = VARIANT_IDS
    if isinstance(variants, str):
        variants = [variants]
    variants = tuple(variants)
    unknown = [v for v in variants if v not in VARIANT_IDS]
    if not variants or unknown:
        raise ValueError(f"saturation_variants must be a non-empty subset of {VARIANT_IDS}"
                         + (f"; unknown: {unknown}" if unknown else ""))
    if len(set(variants)) != len(variants):
        raise ValueError("saturation_variants has a variant more than once; use saturation_repeats "
                         "to send a prompt several times")
    for name, val in (("saturation_orders", orders), ("saturation_repeats", repeats),
                      ("saturation_workers", workers)):
        if not isinstance(val, int) or isinstance(val, bool) or val < 1:
            raise ValueError(f"{name} must be a positive integer, got {val!r}")
    return variants


def describe_grid(n_cats, variants, n_orders_requested, orders, repeats):
    """One-paragraph description of the grid for the run log."""
    n = len(variants) * len(orders) * repeats
    lines = [f"[CatLLM] saturation grid: {len(variants)} wording(s) x {len(orders)} category "
             f"order(s) x {repeats} repeat(s) = {n} calls per response per model."]
    if n_orders_requested > len(orders):
        lines.append(f"  {n_cats} categories have only {len(orders)} orders ({n_cats}!), so "
                     f"saturation_orders={n_orders_requested} was capped at {len(orders)}; "
                     f"every order is used. At most {max_prompts(n_cats, len(variants))} "
                     f"distinct prompts exist for {len(variants)} wording(s).")
    return "\n".join(lines)


def ordered_categories_str(categories, order):
    """The numbered category list as shown in one order: '1. ...\\n2. ...'."""
    return "\n".join(f"{j + 1}. {categories[i]}" for j, i in enumerate(order))


def reorder_schema(schema, order):
    """The JSON schema for one shown order. build_json_schema describes key "j"
    with category j's text; a shuffled list numbers the categories by shown
    position, so key "j" must carry the description of the category shown at j
    (on providers that put the schema in the prompt, e.g. Anthropic structured
    outputs, the model reads both). Returns the schema unchanged when it isn't
    keyed "1".."K"."""
    if not isinstance(schema, dict):
        return schema
    props = schema.get("properties")
    keys = [str(k) for k in range(1, len(order) + 1)]
    if not isinstance(props, dict) or sorted(props, key=str) != sorted(keys, key=str):
        return schema
    shown = {str(j + 1): dict(props[str(i + 1)]) for j, i in enumerate(order)}
    return {**schema, "properties": {k: shown[k] for k in keys}}


def unshuffle_codes(codes, order):
    """Map a reply's codes, keyed by shown position '1'..'K', back to the original
    category numbers. None (an unusable reply) stays None."""
    if codes is None:
        return None
    return {str(i + 1): codes.get(str(j + 1), 0) for j, i in enumerate(order)}

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
                   draw_retries=DRAW_RETRIES, retry_delay=1.0, categories=None, orders=None,
                   repeats=1, workers=1):
    """Classify one response once per prompt in the grid and combine.

    The grid is variants x orders x repeats (see category_orders). orders=None
    means the original order only; any other order needs `categories` (the
    category texts in original order) to build its shown list. workers > 1
    sends a response's draws concurrently.

    Returns (json_result string, error or None, stats). stats holds "draws" (the
    number of usable draws), "shares" (per category, the proportion of usable
    draws coding it 1), "failed", "planned" (the grid size) and "detail"
    ({draw id: the draw's 0/1 codes as a string in original category order, or
    None if unusable}). json_result is the majority label, '{"1":"e"}' with an
    error when no draw is usable.
    """
    identity = tuple(range(n_cats))
    orders = [identity] if orders is None else [tuple(o) for o in orders]
    if any(o != identity for o in orders) and categories is None:
        raise ValueError("run_saturation: shuffled category orders need `categories`")
    shown = [categories_str if o == identity else ordered_categories_str(categories, o)
             for o in orders]
    schemas = [json_schema if o == identity else reorder_schema(json_schema, o) for o in orders]
    grid = [(vid, oi, r) for vid in variants for oi in range(len(orders)) for r in range(repeats)]

    def draw(key):
        vid, oi, _ = key
        messages = render(vid, response_text, shown[oi], survey_question=survey_question,
                          examples_text=examples_text, multi_label=multi_label,
                          system_prompt=system_prompt)
        codes, error = _one_draw(client, messages, schemas[oi], provider, thinking_budget,
                                 max_retries, json_retries, n_cats, extract_json_fn,
                                 formatter_fallback_fn)
        return unshuffle_codes(codes, orders[oi]), error

    codes, errors = {}, {}

    def send(keys):
        if workers > 1 and len(keys) > 1:
            with ThreadPoolExecutor(max_workers=min(workers, len(keys))) as ex:
                for key, (c, e) in zip(keys, ex.map(draw, keys)):
                    codes[key], errors[key] = c, e
        else:
            for key in keys:
                codes[key], errors[key] = draw(key)

    send(grid)
    # Re-send only the draws an API / connection error lost
    for attempt in range(draw_retries):
        failed = [key for key in grid if errors[key]]
        if not failed:
            break
        if retry_delay:
            time.sleep(retry_delay * (attempt + 1))
        send(failed)
    draws = [codes[key] for key in grid]       # (v01, original order) first when present
    detail = {draw_id(*key): (None if codes[key] is None else
                              "".join(str(int(codes[key].get(str(k), 0))) for k in range(1, n_cats + 1)))
              for key in grid}
    errors = [f"{draw_id(*key)}: {errors[key]}" for key in grid if errors[key]]
    final, shares, n = combine_draws(draws, n_cats, multi_label)
    stats = {"draws": n, "shares": shares, "failed": len(errors), "planned": len(grid),
             "detail": detail}
    if final is None:
        return '{"1":"e"}', "saturation: no usable draw (" + "; ".join(errors[:3]) + ")", stats
    return json.dumps({k: str(v) for k, v in final.items()}), None, stats


# ---------------------------------------------------------------------------
# Output helpers (used by build_output_dataframes)
# ---------------------------------------------------------------------------

def apply_saturation_proportions(df, stats, model_names, n_cats, keep_draws=False):
    """Replace each model's category columns with the proportion of usable draws that
    assigned the category, and add saturation_draws_{model} after processing_status.
    keep_draws=True also adds saturation_detail_{model}: a JSON object mapping each
    draw id ('v03_o07_r00') to that draw's 0/1 codes as a string in original
    category order ('0100...'), or null for an unusable draw.
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
        if keep_draws:
            df.insert(pos, f"saturation_detail_{m}",
                      [json.dumps(g["detail"]) if g and "detail" in g else None for g in got])
            pos += 1
    if len(model_names) > 1:
        for k in range(1, n_cats + 1):
            cols = [f"category_{k}_{m}" for m in model_names if f"category_{k}_{m}" in df]
            if f"category_{k}_consensus" in df:
                df[f"category_{k}_consensus"] = df[cols].mean(axis=1, skipna=True).astype("Float64")
        df.drop(columns=[c for c in df.columns if c.endswith("_agreement") or c.endswith("_resolved_by")],
                inplace=True)
    # Report rows that ended with fewer usable draws than the grid planned
    planned = max((g.get("planned", len(VARIANT_IDS)) for g in stats.values() if g),
                  default=len(VARIANT_IDS))
    for m in model_names:
        d = df[f"saturation_draws_{m}"]
        short = int((d.fillna(0) < planned).sum())
        if short:
            print(f"[CatLLM] saturation: {short} of {len(d)} rows have fewer than {planned} usable "
                  f"draws for {m} (min {int(d.fillna(0).min())}); their proportions rest on fewer draws.")
    return df
