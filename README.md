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
Discover categories from a corpus using LLM-driven exploration.

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
- **Batch API support** for OpenAI, Anthropic, Google, Mistral, and xAI
- **Prompt strategies**: Chain-of-Thought, Chain-of-Verification, step-back prompting, few-shot examples
- **Text, image, and PDF** input auto-detection
- **Embedding similarity** tiebreaker for ensemble consensus ties
- **Pilot test** — validate classifications on a small sample before committing to the full run

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
