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
import cat_stack as cat

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

## License

GPL-3.0-or-later
