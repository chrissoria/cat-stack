"""
Theme collapsing for CatLLM.

collapse_themes() takes an already-extracted list of category/theme strings (for
example the output of explore()) and iteratively consolidates near-duplicate /
synonymous labels into a smaller list. Each pass:

    A. accept the list,
    B. PRE-CLEAN before the model — normalize + Jaro-Winkler dedup (surface
       variants) then embedding-merge (semantic near-duplicates),
    C. split the cleaned list into batches of `batch_size`,
    D. read every batch with one LLM call (extract-unique, or aggressive merge),
    E. concatenate and dedupe into a single, smaller list.

`passes` iterations run in one call, randomizing batch composition each pass so
labels stranded in separate batches get fresh chances to meet and merge.
Provider-agnostic via the same dispatch classify()/explore() use.
"""

import random
import re
import sys

import numpy as np
import pandas as pd
from jellyfish import jaro_winkler_similarity

from ._providers import UnifiedLLMClient, detect_provider
from ._utils import _clean_label

__all__ = [
    "collapse_themes",
]

_LINE_PAT = re.compile(r"^\s*\d+\s*[\.\)\-]\s*(.+)$")
_EMB_MODEL = None  # cached embedding model (loaded once per process)


def _strip_parens(label):
    """Drop parenthetical examples — '(...)' doesn't change the category."""
    return re.sub(r"\s*\([^)]*\)", "", label).strip()


def _norm_key(label):
    """Canonical dedup key: parens-stripped, lowercased, separators/order unified."""
    s = _strip_parens(label).lower().strip()
    s = re.sub(r"\s*&\s*|\s+and\s+|\s*/\s*", " / ", s)
    parts = sorted(p.strip() for p in s.split("/") if p.strip())
    return " / ".join(parts)


def _jw_dedupe(items, threshold):
    """Order-preserving dedup: normalize each label and collapse near-identical
    normalized labels with a Jaro-Winkler threshold. Returns readable forms."""
    kept_keys = []
    out = []
    for c in items:
        disp = _strip_parens(c).lower().strip()
        key = _norm_key(c)
        if not disp or not key:
            continue
        is_dup = any(
            k == key
            or (threshold < 1.0 and jaro_winkler_similarity(key, k) >= threshold)
            for k in kept_keys
        )
        if not is_dup:
            kept_keys.append(key)
            out.append(disp)
    return out


def _get_emb_model():
    """Load (once) and return cat-stack's canonical BAAI/bge-small embedder."""
    global _EMB_MODEL
    if _EMB_MODEL is None:
        from ._embeddings import load_embedding_model
        _EMB_MODEL = load_embedding_model()
    return _EMB_MODEL


def _embedding_merge(items, threshold):
    """Greedy embedding clustering: drop labels whose cosine similarity to an
    already-kept label is >= threshold. Keeps the first-seen representative."""
    if not threshold or threshold >= 1.0 or len(items) < 2:
        return items
    embs = _get_emb_model().encode(items, normalize_embeddings=True, show_progress_bar=False)
    reps, rep_embs = [], []
    for it, e in zip(items, embs):
        if rep_embs and float(np.max(np.asarray(rep_embs) @ e)) >= threshold:
            continue
        reps.append(it)
        rep_embs.append(e)
    return reps


def _quality(output, raw_embs, tau_cov=0.70, tau_red=0.85, beta=2.0):
    """Deterministic quality of a candidate taxonomy vs the raw input themes:
    coverage-weighted F-beta of recall=coverage_hard (share of raw within tau_cov
    of some output) and precision=(1 - redundancy_rate) (share of outputs with a
    near-twin >= tau_red). Embedding-only — the convergence signal for passes='auto'.
    """
    if not output:
        return 0.0
    O = _get_emb_model().encode(list(output), normalize_embeddings=True, show_progress_bar=False)
    coverage = float(((raw_embs @ O.T).max(axis=1) >= tau_cov).mean())
    if len(output) > 1:
        OO = O @ O.T
        np.fill_diagonal(OO, -1.0)
        redundancy = float((OO.max(axis=1) >= tau_red).mean())
    else:
        redundancy = 0.0
    precision = 1.0 - redundancy
    if coverage <= 0 or precision <= 0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * precision * coverage / (b2 * precision + coverage)


def _collapse_batch(client, batch, description, creativity, mode="unique"):
    """One LLM call on a single batch -> list[str].

    mode="unique": extract unique categories only (remove restatements, keep
    distinct ones) — gentle, near-idempotent, guaranteed to only remove.
    mode="merge": aggressively consolidate related labels into broader concepts
    while retaining meaningful distinctions — for a final compression step.

    Strict numbered-list prompt + strict parsing, so the reply is always a clean
    list and any stray prose is ignored. Guardrails: a failed call returns the
    batch unchanged (no data loss); in "unique" mode the output is forced to be a
    subset of the input (monotone, drift-free).
    """
    items_blob = "; ".join(batch)
    context = f' about: "{description}"' if description else ""
    if mode == "merge":
        prompt = (
            f"You are consolidating a list of category labels{context} into a smaller set of "
            "broader categories. Group labels that describe the same underlying concept and give "
            "each group ONE clear representative label — actively merge near-synonyms and closely "
            "related labels into broader themes. BUT retain nuance: do NOT over-merge — keep labels "
            "separate when they capture a genuinely distinct concept, even if related, rather than "
            "collapsing them into one vague catch-all. Prefer fewer, cleaner categories without "
            f"losing real distinctions. Labels are separated by semicolons within triple backticks: "
            f"```{items_blob}```.\n\n"
            "Return ONLY a numbered list of the consolidated categories. Each line must follow this "
            "exact format, with no other text before or after the list:\n"
            "N. label\n\n"
            "Example:\n"
            "1. Employment\n"
            "2. Education\n"
            "3. Religion"
        )
    else:
        prompt = (
            f"You are given a list of category labels{context}. "
            "Return the UNIQUE categories. Remove ONLY exact duplicates and labels that "
            "restate the SAME category in different words — when two labels are the same "
            "category, keep one of them exactly as written. KEEP every genuinely distinct "
            "category. Do NOT merge categories that are merely related, do NOT invent or "
            "broaden labels, and do NOT drop a category just to make the list shorter. "
            "If all the labels are already distinct categories, return ALL of them unchanged. "
            f"Labels are separated by semicolons within triple backticks: ```{items_blob}```.\n\n"
            "Return ONLY a numbered list, using the labels exactly as they appear. Each line "
            "must follow this exact format, with no other text before or after the list:\n"
            "N. label\n\n"
            "Example:\n"
            "1. Employment\n"
            "2. Education\n"
            "3. Religion"
        )
    reply, error = client.complete(
        messages=[{"role": "user", "content": prompt}],
        creativity=creativity,
        force_json=False,
    )
    if error:
        # No data loss: keep the batch unchanged so its categories aren't dropped.
        sys.stderr.write(f"[collapse_themes] batch failed: {error} — keeping batch unchanged\n")
        return [str(x).strip().lower() for x in batch]

    out = []
    for line in (reply or "").splitlines():
        m = _LINE_PAT.match(line.strip())
        if m:
            label = _clean_label(m.group(1)).strip(" ;.,")
            if label:
                out.append(label)

    if mode == "unique":
        # Contraction guarantee: extract-unique must only REMOVE, never add or
        # mutate. Keep only outputs that map back to an input label (by normalized
        # key), as the original input string. Makes every pass monotone and
        # drift-free, immune to intermittent model rephrasing/splitting.
        in_by_key = {}
        for x in batch:
            in_by_key.setdefault(_norm_key(x), str(x).strip().lower())
        seen, subset = set(), []
        for o in out:
            k = _norm_key(o)
            if k in in_by_key and k not in seen:
                seen.add(k)
                subset.append(in_by_key[k])
        # If parsing/matching failed entirely, fall back to the batch (no loss).
        out = subset if subset else [str(x).strip().lower() for x in batch]
    return out


def _to_counts(input_data):
    """Coerce the accepted input forms into a {category: count} dict."""
    if isinstance(input_data, pd.DataFrame):
        cols = {c.lower(): c for c in input_data.columns}
        cat_col = cols.get("category")
        cnt_col = cols.get("count")
        if cat_col is None:
            raise ValueError("DataFrame input must have a 'category' column.")
        if cnt_col is not None:
            return input_data.groupby(cat_col)[cnt_col].sum().astype(int).to_dict()
        return input_data[cat_col].value_counts().to_dict()
    if isinstance(input_data, dict):
        return {str(k): int(v) for k, v in input_data.items()}
    series = input_data if isinstance(input_data, pd.Series) else pd.Series(input_data)
    series = series.dropna().astype("string")
    return series.value_counts().to_dict()


def _collapse_once(
    client,
    items,
    *,
    description,
    batch_size,
    dedupe_threshold,
    embedding_merge_threshold,
    mode,
    shuffle,
    random_state,
    creativity,
    max_workers,
):
    """Run a single collapse pass over `items` and return the reduced list."""
    # A. accept -> {category: count}
    counts = _to_counts(items)

    # B. PRE-CLEAN before the model: normalize+JW dedup, then embedding-merge
    ordered = sorted(counts, key=counts.get, reverse=True)
    cleaned = _jw_dedupe(ordered, dedupe_threshold)
    cleaned = _embedding_merge(cleaned, embedding_merge_threshold)

    # Randomize order so batch composition varies across passes — gives near-
    # duplicates split across batches fresh chances to co-occur and merge.
    if shuffle:
        random.Random(random_state).shuffle(cleaned)

    # C. split into batches
    batches = [cleaned[i:i + batch_size] for i in range(0, len(cleaned), batch_size)]

    # D. one LLM call per batch (sequential or parallel)
    if max_workers and max_workers > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results = [None] * len(batches)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(_collapse_batch, client, b, description, creativity, mode): i
                for i, b in enumerate(batches)
            }
            for fut in as_completed(futures):
                results[futures[fut]] = fut.result()
        out = [label for r in results for label in (r or [])]
    else:
        out = []
        for batch in batches:
            out.extend(_collapse_batch(client, batch, description, creativity, mode))

    # E. dedupe the concatenated output (surface-level)
    return _jw_dedupe(out, dedupe_threshold)


def collapse_themes(
    input_data,
    api_key=None,
    description="",
    passes=1,
    max_passes=10,
    batch_size=40,
    aggressive=False,
    dedupe_threshold=0.95,
    embedding_merge_threshold=0.92,
    shuffle=True,
    user_model="gpt-4o",
    model_source="auto",
    creativity=0,
    max_workers=1,
    random_state=None,
    filename=None,
    progress_callback=None,
):
    """
    Collapse a list of extracted themes into a smaller, deduplicated list.

    Iteratively consolidates near-duplicate / synonymous category labels (for
    example the output of explore()). Each pass PRE-CLEANS before the model
    (normalize + Jaro-Winkler dedup, then embedding-merge), splits into batches,
    sends each batch to the model, and dedupes the concatenated result. Runs
    `passes` iterations, randomizing batch composition each pass so labels
    stranded in separate batches get fresh chances to merge.

    Two modes:
      - aggressive=False (default): extract-unique — only removes duplicates /
        restatements, never invents or broadens. Each pass is guaranteed monotone
        (output is a subset of its input). Use to thin a noisy list faithfully.
      - aggressive=True: conceptual merge — actively consolidates related labels
        into broader categories while retaining meaningful distinctions. Use as a
        final compression step.

    Provider-agnostic (model_source: "auto", "openai", "huggingface", ...), via
    the same dispatch classify()/explore() use.

    Args:
        input_data: Themes to collapse. list[str] (duplicates allowed), pandas
            Series, dict {category: count}, or DataFrame with "category"
            [and optional "count"] columns.
        api_key (str): API key for the model provider.
        description (str): Data/question context, injected into the prompt — e.g.
            the survey question the categories came from. Helps the model judge
            which distinctions matter.
        passes (int | str): Number of collapse iterations, or "auto" to iterate
            until the deterministic quality benchmark peaks (the recommended mode
            for a final taxonomy — pair with aggressive=True). Default 1.
        max_passes (int): Cap on iterations when passes="auto". Default 10.
        batch_size (int): Themes per LLM chunk (ceil(n / batch_size) calls per
            pass). Default 40.
        aggressive (bool): If True, use the conceptual-merge prompt (compress);
            if False, extract-unique (faithful thinning). Default False.
        dedupe_threshold (float): Jaro-Winkler similarity at/above which two
            normalized labels are deduped. Default 0.95; 1.0 = exact only.
        embedding_merge_threshold (float): Cosine similarity at/above which labels
            are merged in the pre-LLM embedding step (BAAI/bge-small). Default
            0.92. None or >=1.0 skips embeddings.
        shuffle (bool): Randomize order each pass so batch composition varies.
            Default True (improves convergence stability).
        user_model (str): Model name. Default "gpt-4o". Use a capable model —
            small models can degenerate into repetition.
        model_source (str): Provider — "auto", "openai", "huggingface", etc.
        creativity (float): Temperature. Default 0 (deterministic).
        max_workers (int): Batches processed concurrently per pass. Default 1.
        random_state (int): Seed for shuffling (per-pass seed = random_state + p).
            None = nondeterministic.
        filename (str): Optional CSV path to save the final list.
        progress_callback (callable): Optional callback(pass, passes, label).

    Returns:
        list[str]: The collapsed category list after `passes` iterations.

    Examples:
        >>> import cat_stack as cat
        >>> themes = cat.explore(df['responses'], description="Why did you move?",
        ...                      api_key=key)
        >>> # Recommended: aggressive merge, auto-stop at the quality peak
        >>> taxonomy = cat.collapse_themes(
        ...     themes, api_key=key, description="Why did you move?",
        ...     aggressive=True, passes="auto", max_workers=8,
        ... )
    """
    if not api_key:
        raise ValueError("collapse_themes() needs an api_key for the LLM call.")

    mode = "merge" if aggressive else "unique"
    provider = detect_provider(user_model, model_source)
    client = UnifiedLLMClient(provider=provider, api_key=api_key, model=user_model)

    def _pass(items, p):
        return _collapse_once(
            client, items,
            description=description,
            batch_size=batch_size,
            dedupe_threshold=dedupe_threshold,
            embedding_merge_threshold=embedding_merge_threshold,
            mode=mode,
            shuffle=shuffle,
            random_state=(None if random_state is None else random_state + p),
            creativity=creativity,
            max_workers=max_workers,
        )

    current = input_data
    if passes == "auto":
        # Iterate until the deterministic quality benchmark stops improving (the
        # peak), capped at max_passes. Quality is scored vs the ORIGINAL input
        # themes — embedding-only, model-independent at decision time. The peak is
        # the principled stop (validated across surveys and list sizes).
        raw_embs = _get_emb_model().encode(
            list(_to_counts(input_data).keys()), normalize_embeddings=True,
            show_progress_bar=False,
        )
        best, best_q = None, -1.0
        for p in range(max_passes):
            current = _pass(current, p)
            q = _quality(current, raw_embs)
            if progress_callback:
                progress_callback(p + 1, max_passes, "collapse_themes")
            if q < best_q:
                break  # quality dropped -> the previous pass was the peak
            best, best_q = current, q
        current = best if best is not None else current
    else:
        for p in range(int(passes)):
            current = _pass(current, p)
            if progress_callback:
                progress_callback(p + 1, int(passes), "collapse_themes")

    if filename:
        pd.DataFrame({"category": current}).to_csv(filename, index=False)
        print(f"Collapsed categories saved to {filename}")

    return current
