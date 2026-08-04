# cat-stack

**Domain-agnostic text, image, and PDF classification engine powered by LLMs.**

`cat-stack` is the shared base package for the [CatLLM](https://github.com/chrissoria/cat-llm) ecosystem. It provides the core classification, extraction, exploration, and summarization engine that all domain-specific CatLLM packages build on.

## Installation

```bash
pip install cat-stack
```

Optional extras:

```bash
pip install cat-stack[pdf]         # PDF support (PyMuPDF)
pip install cat-stack[embeddings]  # Embedding similarity scoring
pip install cat-stack[formatter]   # JSON formatter fallback model
```

## Ecosystem

`cat-stack` is independently useful for classifying any text column. Domain-specific packages extend it with tuned prompts and workflows:

| Package | Domain |
|---------|--------|
| **cat-stack** | General-purpose text, image, PDF classification (this package) |
| **cat-survey** | Survey response classification |
| **cat-vader** | Social media text (Reddit, Twitter/X) |
| **cat-ademic** | Academic papers, PDFs, citations |
| **cat-cog** | Cognitive assessment & visual scoring (CERAD) |
| **cat-pol** | Political text (manifestos, speeches, legislation) |

Installing `cat-llm` pulls in all of the above.

## Quick Start

```python
import catstack as cat

# Classify text into predefined categories
result = cat.classify(
    input_data=df["text_column"],
    categories=["Positive", "Negative", "Neutral"],
    models=[("gpt-4o", "openai", OPENAI_KEY)],
    filename="classified.csv"
)
```

## Core API

### `classify()`
Assign predefined categories to text, images, or PDFs. Supports single-model and multi-model ensemble classification with consensus voting.

```python
cat.classify(
    input_data=df["text"],
    categories=["Cat A", "Cat B", "Cat C"],
    models=[("gpt-4o", "openai", key1), ("claude-sonnet-4-20250514", "anthropic", key2)],
    filename="results.csv"
)
```

#### Inline prompt tuning

Add `prompt_tune=True` to automatically optimize the classification prompt before the full run. A browser UI opens for you to correct a small sample, then the optimized prompt is used for all remaining items.

```python
cat.classify(
    input_data=df["text"],
    categories=["Cat A", "Cat B", "Cat C"],
    models=[("gpt-4o", "openai", key)],
    prompt_tune=15,       # tune on 15 random items, then classify all
    tune_iterations=3,    # max attempts per category (default 3)
)
```

### `prompt_tune()`
Standalone automatic prompt optimization. Iteratively refines classification prompts using user feedback — classify a sample, correct mistakes in the browser, and let the LLM generate targeted per-category instructions.

```python
result = cat.prompt_tune(
    input_data=df["text"],
    categories=["Cat A", "Cat B", "Cat C"],
    api_key="your-key",
    sample_size=15,
    max_iterations=3,
)

# Use the optimized prompt for classification
cat.classify(
    input_data=df["text"],
    categories=["Cat A", "Cat B", "Cat C"],
    api_key="your-key",
    system_prompt=result["system_prompt"],
)
```

### `extract()`
Discover categories from a corpus using LLM-driven exploration. Since v2.5.0, text consolidation runs the full explore → `collapse_themes()` pipeline (`engine="collapse"`, the default): the entire raw label inventory reaches the semantic merge — Jaro-Winkler dedup, embedding pre-merge, quality-controlled LLM passes, then a count-guided reduction to at most `max_categories`. Pass `engine="legacy"` to reproduce pre-2.5 runs (single merge call over a truncated inventory). `collapse_kwargs` forwards options to `collapse_themes()`; `max_workers` parallelizes both extraction and consolidation.

```python
cat.extract(
    input_data=df["text"],
    survey_question="What is this text about?",
    models=[("gpt-4o", "openai", key)],
)
```

### `explore()`
Raw category extraction for saturation analysis.

```python
cat.explore(
    input_data=df["text"],
    description="Describe the main themes",
    models=[("gpt-4o", "openai", key)],
)
```

### `collapse_themes()`
Consolidate a long, redundant list of extracted category labels (e.g. the output of `explore()`) into a smaller, deduplicated taxonomy. Runs the semantic merge iteratively, then applies a single deterministic embedding re-merge over the whole result to collapse cross-batch lexical siblings (e.g. "tension" / "estrangement") that batched passes leave separate. Tuned to err toward over-segmentation (keeping categories) rather than over-merging.

```python
# Basic: aggressive merge, auto-stop at the quality peak
cat.collapse_themes(
    input_data=raw_labels,            # list[str] or a frequency Series/dict
    api_key=key,
    description="Why did you move?",   # the survey question / context
    aggressive=True,
    passes="auto",
    user_model="gpt-4o",
)
```

```python
# Per-step model assignment: a cheap model thins restatements,
# a stronger model does the conceptual merge (providers can differ)
cat.collapse_themes(
    input_data=raw_labels,
    api_key=key,
    description="Why did you move?",
    aggressive=True,
    passes="auto",
    unique_model="Qwen/Qwen2.5-72B-Instruct:together",
    unique_model_source="huggingface",
    unique_passes=1,
    merge_model="Qwen/Qwen3.6-35B-A3B:together",
    merge_model_source="huggingface",
    max_workers=8,
)
```

**Parameters**

| Parameter | Default | Description |
| --- | --- | --- |
| `input_data` | — | List of category labels, or a frequency `Series`/`dict` (`label -> count`). |
| `api_key` | `None` | API key for the LLM provider. Not required for subscription/CLI backends (`claude-code`, `claude-agent`, `codex-agent`) or `ollama`. |
| `description` | `""` | The survey question or context, used in the merge prompt. |
| `passes` | `1` | Number of merge iterations, or `"auto"` to iterate until the embedding-quality benchmark peaks. |
| `max_passes` | `10` | Cap on iterations when `passes="auto"`. |
| `batch_size` | `40` | Labels per LLM chunk (`ceil(n / batch_size)` calls per pass). |
| `aggressive` | `False` | `True` = conceptual-merge prompt (compress related labels); `False` = extract-unique (faithful thinning, removes restatements only). |
| `dedupe_threshold` | `0.95` | Jaro-Winkler similarity at/above which normalized labels are deduped (`1.0` = exact only). |
| `embedding_merge_threshold` | `0.92` | Cosine similarity at/above which labels are merged in the pre-LLM embedding step. `None`/`>=1.0` disables it. |
| `shuffle` | `True` | Randomize order each pass so batch composition varies (improves convergence stability). |
| `final_consolidation` | `0.82` | Cosine threshold for one greedy global embedding re-merge after all passes, collapsing cross-batch duplicates. Conservative by design (errs toward keeping categories). `False`/`None` skips it. |
| `top_n` | `None` | If set, a final global LLM call consolidates the surviving list into at most N categories, guided by each label's generation count (frequent themes favored; overlapping labels merged rather than dropped). Guaranteed `<= top_n` via truncation plus a deterministic top-N-by-count fallback. |
| `prune` | `False` | `True` = drop conceptual duplicates keeping one representative verbatim; never renames or merges merely-related labels. |
| `user_model` | `"gpt-4o"` | Model for the merge phase. Use a capable model — small models can degenerate. |
| `model_source` | `"auto"` | Provider for `user_model` (`"auto"`, `"openai"`, `"huggingface"`, …). |
| `unique_model` | `None` | If set, run an initial extract-unique thinning phase on this (typically cheaper) model before the merge phase. `None` skips the phase (backward compatible). |
| `unique_model_source` | `"auto"` | Provider for `unique_model` — can differ from the merge phase. |
| `unique_passes` | `1` | Number of thinning passes when `unique_model` is set. |
| `merge_model` | `None` | Model for the merge phase; falls back to `user_model` when `None`. |
| `merge_model_source` | `"auto"` | Provider for `merge_model`. |
| `creativity` | `0` | Temperature (`0` = deterministic). |
| `max_workers` | `1` | Batches processed concurrently per pass. |
| `random_state` | `None` | Seed for shuffling (per-pass seed = `random_state + pass`). |
| `filename` | `None` | Optional CSV path to save the final list. |
| `progress_callback` | `None` | Optional `callback(pass, passes, label)` for progress reporting. |

### `summarize()`
Summarize text or PDF documents, with optional multi-model ensemble.

```python
cat.summarize(
    input_data=df["text"],
    models=[("gpt-4o", "openai", key)],
    filename="summaries.csv"
)
```

## Supported Providers

OpenAI, Anthropic, Google (Gemini), Mistral, Perplexity, xAI (Grok), HuggingFace, Ollama (local models).

All providers use the same `(model_name, provider, api_key)` tuple format. Provider is auto-detected from model name if omitted.

## Features

- **Automatic prompt optimization** (`prompt_tune`) — correct a small sample in a browser UI, and the system generates per-category instructions that improve accuracy
- **Multi-model ensemble** with consensus voting and agreement scores
- **Batch API support** for OpenAI, Anthropic, Google, Mistral, and xAI.
  *Caveat for Google (Gemini):* as of 2026-06, Google's batch
  scheduler routinely leaves small jobs (under a few dozen rows) in
  `BATCH_STATE_PENDING` for 30+ minutes — sometimes hours — before
  it starts processing. Google's published SLA is up to 24h. If your
  job is small and you want results back quickly, use `batch_mode=False`
  for Gemini; reserve `batch_mode=True` for large jobs where the
  50% cost discount matters more than wall-clock latency. Other
  providers' batch APIs (OpenAI, Anthropic, xAI) typically complete
  small jobs in 1-3 minutes
- **Prompt strategies**: Chain-of-Thought, Chain-of-Verification, step-back prompting, few-shot examples
- **Text, image, and PDF** input auto-detection (PDF inputs are
  validated against the `%PDF-` magic-byte header before reaching
  PyMuPDF, so a webpage saved with `.pdf` extension surfaces a clear
  `ValueError` instead of silently classifying a blank rendered page
  as `success`)
- **Embedding similarity** tiebreaker for ensemble consensus ties
- **Pilot test** — validate classifications on a small sample before committing to the full run
- **Provider-conditional HTTP timeouts** — cloud providers use a tight
  120 s per-request timeout (catches genuine hangs without waiting too
  long on transient API blips), and the Ollama provider uses a wider
  600 s per-request / 1200 s cumulative budget (accommodates the long
  per-row tails that emerge when running 14B+ models on memory-
  constrained hardware like 16 GB Macs). Power users can override per
  client: `UnifiedLLMClient(provider, key, model, request_timeout=900,
  max_total_wait=1800)`, or set a process-wide override with
  `catstack._providers.set_session_timeouts(request_timeout=..., max_total_wait=...)`

## Future work / contributions welcome

The following items are tracked but not yet implemented. PRs welcome —
each entry includes the scope I'd suggest if someone wants to pick it up.

- **Standalone SambaNova provider.** Currently SambaNova-hosted models
  are reachable through the HuggingFace router suffix
  (`meta-llama/...:sambanova`), but there's no direct
  `provider="sambanova"` path that talks to SambaNova's own
  OpenAI-compatible endpoint. Wiring it up means a new
  `PROVIDER_CONFIG` entry, the right base URL
  (`https://api.sambanova.ai/v1`), token-detection rules in
  `detect_provider`, and a smoke test against one of their cheap
  models (e.g. `Meta-Llama-3.1-8B-Instruct`).

- **Consolidate HuggingFace-suffix dispatch.** The strings
  `"huggingface"` and `"huggingface-together"` are currently
  hardcoded in ~30 dispatch sites across
  `pdf_functions.py` / `image_functions.py` /
  `text_functions_ensemble.py` / `_chunked.py`. Adding a new router
  suffix (e.g. `huggingface-fireworks`) means updating every one of
  them. The cleaner refactor is a single
  `_is_openai_compatible(model_source)` helper that matches anything
  starting with `huggingface` plus the static list
  (openai/perplexity/xai). Same shape as our existing
  `_sanitize_google_schema` helper. Touches a lot of sites but each
  edit is mechanical.

- **Meta-LLM "Senate VP" tiebreaker + batch_mode support for
  `embedding_tiebreaker`.** The existing `embedding_tiebreaker=True`
  resolves true 50/50 ties via centroid similarity, but only in
  synchronous ensemble mode. Two related extensions: (a) a meta-LLM
  tie-breaker that invokes a separate model on tied rows
  (`tie_break="meta_model"` with a configurable model); (b) extend
  the existing centroid tiebreaker to work inside `batch_mode=True`
  by running it after the batch results come back, before
  `build_output_dataframes`. The infrastructure for both is already
  in `_tiebreaker.py`; the meta-LLM variant would be a new resolver
  function called from `resolve_ties_with_centroids`.

- **Schema-permafail retry short-circuit.** When a model's
  classification permanently fails schema validation across all
  available retry budgets, the framework keeps spending API calls.
  A short-circuit that detects "this model + this input is producing
  the same invalid output N times in a row" and bails out early
  would save quota. Scope was narrowed earlier (after the
  HF-SMALL-MODEL fix reduced the wasted-retries surface area), so
  there's a real risk this stays low-value; recommend writing the
  detection metric first, instrumenting an actual run, and only
  building the short-circuit if the metric says it would have helped.

## License

GPL-3.0-or-later
