"""
Convenience helpers for thin language wrappers (Stata, future Julia/CLI).

These functions exist so each language wrapper does not have to re-implement
the same string-parsing and output-shaping logic.  They are thin layers
over the main `classify()` / `extract()` / `explore()` / `summarize()` API
— same kwargs, same behavior — plus a few parsers for the string formats
that wrappers tend to accept from their host languages.

R users typically pass native lists / tuples and do not need the string
parsers, but `classify_labels()` is useful for getting one label per row
without manually walking the DataFrame.

These helpers are intentionally side-effect free and import-safe: nothing
here imports a domain sub-package (cat-pol, cat-vader, etc.) until the user
calls `get_backend("pol")`, so importing `catstack` does not require any
domain package to be installed.
"""

from __future__ import annotations

import ast
import importlib
import re
from typing import Any, Dict, List, Optional, Tuple, Union


# -----------------------------------------------------------------------------
# Domain → module resolution
# -----------------------------------------------------------------------------

# Maps the user-facing short domain name to (python import name, pip package).
# Note: import names and pip names differ for the historical cat-vader,
# cat-ademic, and cat-web packages, which omit the underscore in their module
# name.  This dict is the single source of truth across the ecosystem.
_DOMAIN_PACKAGES: Dict[str, Tuple[str, str]] = {
    "pol":    ("cat_pol",   "cat-pol"),
    "vader":  ("catvader",  "cat-vader"),
    "ademic": ("catademic", "cat-ademic"),
    "survey": ("cat_survey", "cat-survey"),
    "cog":    ("cat_cog",   "cat-cog"),
    "web":    ("catweb",    "cat-web"),
}


def get_backend(domain: Optional[str] = None):
    """Return the Python module to call for a given domain shortform.

    Empty string or None returns the base `catstack` module.  Known domain
    names ("pol", "vader", "ademic", "survey", "cog", "web") return their
    respective sub-package module.

    Raises:
        ValueError: if `domain` is set but not in the known list.
        ImportError: if the domain package is not installed.  The error
            message tells the user the exact `catllm setup, domain(X)`
            command to fix it.

    Example:
        >>> get_backend("").__name__
        'catstack'
        >>> get_backend(None).__name__
        'catstack'
        >>> # get_backend("pol") returns the cat_pol module if installed
    """
    if not domain or not str(domain).strip():
        import catstack  # local import to avoid bootstrap cycles
        return catstack

    key = str(domain).strip().lower()
    if key not in _DOMAIN_PACKAGES:
        valid = ", ".join(_DOMAIN_PACKAGES.keys())
        raise ValueError(
            f"Unknown domain: {domain!r}. Valid: {valid}."
        )
    module_name, pip_name = _DOMAIN_PACKAGES[key]
    try:
        return importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(
            f"Domain package '{pip_name}' is not installed. "
            f"Run: catllm setup, domain({key})"
        ) from e


# -----------------------------------------------------------------------------
# String parsers (for wrappers whose host language passes options as strings)
# -----------------------------------------------------------------------------


def _strip_surrounding_quotes(s: str) -> str:
    """Strip one balanced pair of surrounding ' or " — Stata `string asis`
    artifact.  Leaves inner quotes untouched."""
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
        return s[1:-1]
    return s


def parse_kwargs_string(s: Optional[str]) -> Dict[str, Any]:
    """Parse a `"key=val, key=val"` string into a Python kwargs dict.

    Each value is run through `ast.literal_eval` so numbers, booleans,
    strings, and lists all work naturally.  Values that don't parse fall
    back to the raw string.

    Commas inside quotes / brackets are respected (no naive split).

    Returns an empty dict for empty / None input.

    Example:
        >>> parse_kwargs_string("max_retries=3, retry_delay=0.5")
        {'max_retries': 3, 'retry_delay': 0.5}
        >>> parse_kwargs_string("format='bullets', research_question='Why did you move?'")
        {'format': 'bullets', 'research_question': 'Why did you move?'}
    """
    if not s:
        return {}
    s = _strip_surrounding_quotes(str(s))
    if not s.strip():
        return {}

    # Walk character-by-character to split on commas at the top level only
    # (not inside quotes or brackets).
    pieces: List[str] = []
    buf: List[str] = []
    depth = 0
    quote_char: Optional[str] = None
    for ch in s:
        if quote_char:
            buf.append(ch)
            if ch == quote_char:
                quote_char = None
        elif ch in ('"', "'"):
            quote_char = ch
            buf.append(ch)
        elif ch in "([{":
            depth += 1
            buf.append(ch)
        elif ch in ")]}":
            depth -= 1
            buf.append(ch)
        elif ch == "," and depth == 0:
            pieces.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    if buf:
        pieces.append("".join(buf))

    kwargs: Dict[str, Any] = {}
    for p in pieces:
        if "=" not in p:
            continue
        k, _, v = p.partition("=")
        k = k.strip()
        v = v.strip()
        if not k:
            continue
        try:
            kwargs[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            kwargs[k] = v
    return kwargs


def parse_models_string(
    s: Optional[str],
    default_api_key: Optional[str] = None,
) -> Optional[List[Tuple[str, ...]]]:
    """Parse `"model provider key; model provider key"` into a list of tuples.

    Each entry is whitespace-split into 3 fields.  Two-field entries inherit
    `default_api_key` for the third position (useful when the same API key
    powers multiple cloud models in an ensemble).

    Returns None for empty / None input so callers can do `if models: ...`.

    Example:
        >>> parse_models_string("gpt-4o openai sk-...; claude-haiku-4-5 anthropic sk-ant-...")
        [('gpt-4o', 'openai', 'sk-...'), ('claude-haiku-4-5', 'anthropic', 'sk-ant-...')]
        >>> parse_models_string("qwen2.5:7b ollama _")
        [('qwen2.5:7b', 'ollama', '_')]
    """
    if not s or not str(s).strip():
        return None
    s = _strip_surrounding_quotes(str(s))
    if not s.strip():
        return None

    out: List[Tuple[str, ...]] = []
    for entry in s.split(";"):
        parts = entry.strip().split()
        if len(parts) >= 3:
            out.append(tuple(parts[:3]))
        elif len(parts) == 2 and default_api_key is not None:
            out.append((parts[0], parts[1], default_api_key))
        # 1-token or empty entries are silently dropped — they're malformed
    return out or None


# -----------------------------------------------------------------------------
# Output shaping
# -----------------------------------------------------------------------------


def short_label(s: Any) -> Any:
    """Return the short label from a "Label: definition..." string.

    Verbose category labels improve classification accuracy but are awkward
    to display in a single output cell.  `short_label("Positive: The
    respondent expresses approval.")` returns `"Positive"`.

    No-colon strings, empty strings, and non-string values are returned
    unchanged.
    """
    if isinstance(s, str) and ":" in s:
        head = s.split(":", 1)[0].strip()
        if head:
            return head
    return s


# Patterns used by classify_labels to find the per-category output columns.
_CONSENSUS_COL_PAT = re.compile(r"^category_(\d+)_consensus$")
_SINGLE_COL_PAT = re.compile(r"^category_(\d+)$")


def classify_labels(
    input_data,
    categories,
    *,
    short_labels: bool = True,
    multi_label_sep: str = "; ",
    return_full: bool = False,
    **kwargs,
):
    """Convenience wrapper around `classify()` returning one label per row.

    The standard `classify()` returns a wide DataFrame with `category_1`,
    `category_2`, ... (or `category_1_consensus`, ... in ensemble mode)
    indicator columns.  `classify_labels()` collapses that to a `list[str]`
    of length `len(input_data)`, where each entry is the assigned category
    name (joined by `multi_label_sep` if more than one category applies).

    This is the function thin language wrappers should call when the host
    language wants one labeled column per row (Stata, simple CLI tools).

    Args:
        input_data: List of texts, paths, or otherwise — same as `classify()`.
        categories: List of category names — same as `classify()`.
        short_labels: If True (default), apply `short_label()` to each
            assigned category — so `"Positive: definition..."` becomes
            `"Positive"` in the output.  Pass False to keep the full text.
        multi_label_sep: Separator used to join multiple matched categories
            for a row.  Default `"; "`.  Has no effect when only one
            category matches per row (the common case).
        return_full: If True, return `(labels, df)` so callers also have
            access to the underlying DataFrame.  Default False.
        **kwargs: All other kwargs are forwarded to `classify()`.

    Returns:
        list[str] of length `len(input_data)`, or `(labels, df)` tuple if
        `return_full=True`.

    Raises:
        RuntimeError: if `classify()` returns a DataFrame that contains
            neither `category_N` nor `category_N_consensus` columns —
            indicates that cat-stack's output schema has changed
            incompatibly.

    Example:
        >>> labels = classify_labels(
        ...     ["Great service", "Awful experience"],
        ...     ["Positive: approval", "Negative: criticism"],
        ...     api_key="...", user_model="gpt-4o-mini",
        ... )
        >>> labels
        ['Positive', 'Negative']
    """
    # Local import — `classify` lives in catstack.classify, but importing it
    # at module load time would create a circular import (classify.py
    # imports from this package indirectly).
    from .classify import classify

    df = classify(input_data=input_data, categories=categories, **kwargs)

    cols = list(df.columns)
    # Ensemble path first (more specific suffix)
    indexed: List[Tuple[int, str]] = []
    for c in cols:
        m = _CONSENSUS_COL_PAT.match(c)
        if m:
            indexed.append((int(m.group(1)), c))
    if not indexed:
        for c in cols:
            m = _SINGLE_COL_PAT.match(c)
            if m:
                indexed.append((int(m.group(1)), c))
    if not indexed:
        raise RuntimeError(
            "classify() returned no category_N or category_N_consensus "
            "columns. The output schema may have changed; this version of "
            "classify_labels cannot map the result back to user-provided "
            "category names. Got columns: " + ", ".join(cols)
        )
    indexed.sort(key=lambda t: t[0])

    # Pre-shorten the category list once if requested.
    if short_labels:
        display_cats = [short_label(c) for c in categories]
    else:
        display_cats = list(categories)

    labels_per_row: List[str] = []
    for _, row in df.iterrows():
        matched: List[str] = []
        for n, col in indexed:
            try:
                if int(row[col]) == 1:
                    cat_idx = n - 1
                    if 0 <= cat_idx < len(display_cats):
                        matched.append(str(display_cats[cat_idx]))
            except (ValueError, TypeError, KeyError):
                continue
        labels_per_row.append(multi_label_sep.join(matched))

    if return_full:
        return labels_per_row, df
    return labels_per_row


def classify_indicators(
    input_data,
    categories,
    *,
    short_labels: bool = True,
    return_full: bool = False,
    **kwargs,
):
    """Convenience wrapper around `classify()` returning per-category indicators.

    Like `classify_labels`, but instead of collapsing the wide DataFrame to
    one assigned label per row, it returns a dict mapping each category to
    a list of 0/1 indicators of length `len(input_data)`.

    This is the right shape for language wrappers that want one indicator
    variable per category (Stata's wide mode, future R `as_indicators=TRUE`
    mode) instead of a single label per row.

    Args:
        input_data: Same as `classify()`.
        categories: Same as `classify()` — list of category strings.
        short_labels: If True (default), use `short_label()` on each
            category to produce dict keys (`"Positive: defn"` → `"Positive"`).
            If False, the dict keys are the full category strings.
        return_full: If True, return `(indicators_dict, df)` so callers also
            have access to the underlying DataFrame.  Default False.
        **kwargs: All other kwargs are forwarded to `classify()`.

    Returns:
        dict[str, list[int]]: keys are category labels (short or full),
        values are 0/1 lists of length `len(input_data)`.  In ensemble mode
        the indicators come from the `category_N_consensus` columns; in
        single-model mode from `category_N`.
        Or `(dict, df)` tuple if `return_full=True`.

    Raises:
        RuntimeError: if `classify()` returns a DataFrame that contains
            neither `category_N` nor `category_N_consensus` columns
            (centralized schema canary, same trigger as `classify_labels`).

    Example:
        >>> indicators = classify_indicators(
        ...     ["I moved for the job and to be near family.",
        ...      "Lower cost of living was the only reason."],
        ...     ["Job: career", "Family: relationships", "Cost: affordability"],
        ...     api_key="...", user_model="gpt-4o-mini",
        ... )
        >>> indicators
        {'Job': [1, 0], 'Family': [1, 0], 'Cost': [0, 1]}
    """
    # Reuse classify_labels for the df + centralized schema canary.  We
    # pass short_labels=False because we want the raw df; we apply our own
    # short_label() to the dict keys below.
    _labels, df = classify_labels(
        input_data,
        categories,
        short_labels=False,
        return_full=True,
        **kwargs,
    )

    cols = list(df.columns)
    indexed: List[Tuple[int, str]] = []
    for c in cols:
        m = _CONSENSUS_COL_PAT.match(c)
        if m:
            indexed.append((int(m.group(1)), c))
    if not indexed:
        for c in cols:
            m = _SINGLE_COL_PAT.match(c)
            if m:
                indexed.append((int(m.group(1)), c))
    # classify_labels already raised RuntimeError if neither family is
    # present, so we know `indexed` is non-empty here.
    indexed.sort(key=lambda t: t[0])

    keys = [short_label(c) if short_labels else c for c in categories]

    out: Dict[str, List[int]] = {}
    for n, col in indexed:
        cat_idx = n - 1
        if not (0 <= cat_idx < len(keys)):
            continue
        key = str(keys[cat_idx])
        series = df[col]
        values: List[int] = []
        for v in series:
            try:
                values.append(1 if int(v) == 1 else 0)
            except (ValueError, TypeError):
                values.append(0)
        out[key] = values

    if return_full:
        return out, df
    return out
