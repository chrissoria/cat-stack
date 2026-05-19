# Changelog

All notable changes to CatLLM will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.4.1] - 2026-05-18

### Changed
- **`batch_retries` default lowered from 2 to 1** in `classify()`, `classify_ensemble()`,
  `summarize()`, and `summarize_ensemble()`. `batch_retries` and `json_retries`
  compose multiplicatively — with the previous defaults, a stubbornly-failing
  row could hit the LLM up to `(1 + 2) * (1 + 2) = 9` times. `batch_retries`
  re-sends the identical prompt (unlike `json_retries`, which adds a
  "respond with only valid JSON" nudge), so its rescue probability is low while
  its cost in the failure tail is high. New worst case is `3 * 2 = 6` calls.
  Callers depending on the old behavior can pass `batch_retries=2` explicitly.
- Docstrings for `batch_retries` now spell out the multiplicative composition
  with `json_retries`.

### Added
- `tests/test_json_retries.py` — unit tests covering `json_retries=0` (single
  call), persistent-invalid retry exhaustion, early exit on valid JSON, and
  the retry-nudge prompt injection. Uses `monkeypatch` on
  `UnifiedLLMClient.complete` — no network calls.

---

## [1.4.0] - 2026-05-18

### Added
- **`catstack.classify_indicators(input_data, categories, *, short_labels=True,
  return_full=False, **kwargs)`** — sibling to `classify_labels` that returns
  per-category 0/1 indicator lists instead of one collapsed label per row.
  Shape: `dict[short_label, list[int]]`, length-`len(input_data)` per key.
  Use this for language wrappers that want one indicator variable per
  category (Stata's wide mode, future R `as_indicators=TRUE` mode), matching
  the wide DataFrame Python users see directly from `classify()`. Same
  centralized schema canary as `classify_labels` — raises `RuntimeError`
  once if neither `category_N` nor `category_N_consensus` columns appear.

Pure-additive release; no existing signatures or behavior change.

---

## [1.3.0] - 2026-05-18

### Improved
- **`prompt_tune()` — attempt history format simplified**: dropped score numbers
  from the history section (not useful for small models), capped at last 3
  attempts to avoid prompt bloat, and reworded to a direct imperative
  ("write something different") rather than a negation list ("do not repeat
  these"), which is harder for smaller models to follow.

---

## [1.2.0] - 2026-05-17

### Added — wrapper-friendly public helpers
Five new public helpers in `catstack._wrapper_helpers` (re-exported at the
top level) so thin language wrappers — Stata today, future Julia / CLI
bindings — can stop re-implementing the same string-parsing and output-
shaping logic. R users can opt in too; the existing R wrapper continues to
work unchanged.

- **`catstack.classify_labels(input_data, categories, *, short_labels=True,
  multi_label_sep="; ", return_full=False, **kwargs)`** — load-bearing
  convenience helper. Runs `classify()` and collapses the wide DataFrame
  to one assigned label per row. Default `short_labels=True` strips
  `"Positive: definition..."` → `"Positive"`. When `multi_label=True`
  produces more than one match per row, the matches are joined with
  `multi_label_sep` instead of being silently dropped — fixes a multi-label
  data-loss bug in the previous Stata wrapper which kept only the first
  match. Raises `RuntimeError` once if neither `category_N` nor
  `category_N_consensus` columns are present (centralized schema canary).
- **`catstack.get_backend(domain)`** — resolves a domain shortform
  (`"pol"`, `"vader"`, `"ademic"`, `"survey"`, `"cog"`, `"web"`) to its
  Python module. Empty/None returns base `catstack`. Unknown domain
  raises `ValueError` listing valid values; missing domain package raises
  `ImportError` with the exact `catllm setup, domain(X)` fix.
- **`catstack.parse_kwargs_string(s)`** — quote- and bracket-aware parser
  for `"k=v, k=v"` strings, with `ast.literal_eval` value parsing and
  string fallback. Powers Stata's `pyoptions()` escape hatch.
- **`catstack.parse_models_string(s, default_api_key=None)`** — parser for
  `"model provider key; model provider key"` ensemble strings. 2-token
  entries inherit `default_api_key`. Powers Stata's `models()` option.
- **`catstack.short_label(s)`** — `"Label: definition"` → `"Label"`.
  Standalone helper for callers that want the colon-split convenience
  without going through `classify_labels`.

Pure-additive release: no existing signatures or behavior change. Python
and R users see the same API they did in 1.1.3.

---

## [1.1.3] - 2026-05-17

### Improved
- **`prompt_tune()` — attempt history passed to meta-LLM**: each call to
  `_generate_category_instruction()` now receives the full history of previous
  instructions tried for that category (instruction text, outcome, and score
  delta). This prevents the meta-LLM from regenerating identical or structurally
  similar instructions across attempts.

### Reverted
- **`prompt_tune()` — holdout split** (introduced in 1.1.2): at the typical
  sample sizes of 10–20 corrections, a ⅓ holdout leaves only 3–6 items for
  scoring — enough variance (±33 pp per flip) to make "improved/regressed"
  decisions meaningless. The holdout concept is sound at scale but
  counterproductive here. Reverted to scoring against all corrections.

---

## [1.1.2] - 2026-05-17

### Improved
- **`prompt_tune()` — attempt history passed to meta-LLM**: each call to
  `_generate_category_instruction()` now receives the full history of previous
  instructions tried for that category (instruction text, outcome, and score
  delta). This prevents the meta-LLM from regenerating identical or structurally
  similar instructions across attempts.
- **`prompt_tune()` — holdout split**: reverted in 1.1.3 (see above).

---

## [1.1.1] - 2026-05-17

### Fixed
- **`check_ollama_model()` false-positive partial match**: a request for
  `qwen2.5:14b` would return True when only `qwen2.5:7b` was installed,
  because the matcher checked whether the requested model started with the
  installed model's family. `classify()` then proceeded with an effectively
  uninstalled model and got per-row Ollama errors → silent "0 classified"
  outcome. Now: explicit tags require an exact match; only family-only
  requests (`"qwen2.5"`) are allowed to match any installed variant.

---

## [1.1.0] - 2026-05-17

### Added
- **`two_step_classify` parameter on `classify()`**: exposes the two-step
  "natural language reasoning then JSON formatting" path that was previously
  hardcoded for Ollama. Pass `True` to force it on any provider — useful for
  lower-tier API models (gpt-4o-mini, claude-haiku, gemini-flash) that
  struggle with strict per-category JSON in a single shot. Pass `False` to
  disable, even for Ollama. Default `None` preserves the auto-enable-for-Ollama
  behavior. Per-model override is also supported via the 4-tuple options dict:
  `("gpt-4o-mini", "openai", key, {"two_step_classify": True})`. Setting
  `two_step_classify=True` also auto-enables the fine-tuned JSON formatter.

### Changed
- **Step-1 prompt for two-step classify is now a simple category list**: the
  prior YES/NO-per-category format was too structured for weak models (local
  7B, lower-tier API), which regressed to partial JSON that step 2 silently
  mapped to all-zeros. The new prompt asks for ONLY the names of the
  applicable categories, one per line — the simplest possible output. Step 2
  was updated to parse the list back into the indexed JSON schema.
- **Default `creativity=0.0` for Ollama**: classification is not creative
  generation. On qwen2.5:7b the temperature-0 default delivers +7pp accuracy
  (78% → 85% on a 40-row sentiment benchmark) and produces bit-identical
  output across runs. Users can still override by passing `creativity=`
  explicitly. Cloud providers continue to use their own model defaults.

### Fixed
- **Fine-tuned JSON formatter now actually fires on lost step-1 signal**:
  `ollama_two_step_classify()` now returns `(json_str, step1_raw, error)`
  instead of `(json_str, error)`. The call site routes `step1_raw` through
  the fine-tuned formatter when step-2 returned valid-but-all-zero JSON,
  which is the original "Ollama returns nothing for clearly-classifiable
  text" bug. The formatter is no longer overridden when step-2 produced a
  confident non-zero result.

### Benchmark
- qwen2.5:7b sentiment classification, 40 verbose-label rows, 4 categories:
  - Before all changes: ~40% accuracy (most failures: empty labels)
  - After all changes: **85% accuracy**, fully deterministic

---

## [1.0.22] - 2026-05-17

### Fixed
- **`_clean_label()` no longer strips "such as" clauses**: the regex that removed
  parenthetical content was too broad, stripping descriptive phrases like
  `(such as rising rent, job loss)` alongside count-only artifacts like `(3)`.
  It now only removes parentheticals that contain a bare number, preserving
  rich "such as" labels in the final merge output.

### Changed
- **Merge prompts now include a concrete output example**: both the neutral and
  survey merge prompts show an explicit `N. Label (such as ...)` example line
  and end with `"Each line must follow this exact format"`, making small models
  (e.g. gpt-4o-mini) reliably produce verbose labels without instruction-following
  failures.

---

## [1.0.21] - 2026-05-17

### Added
- **`_ensure_dependencies()` in `_formatter.py`**: auto-installs `transformers`,
  `torch`, `accelerate`, and `sentencepiece` on first use (~1.5 GB, one-time) instead
  of raising an ImportError. Safe to call from non-TTY sessions (Rscript, CI).
- **Lazy formatter loading**: the formatter model (~1 GB RAM) is no longer loaded at
  startup. It is loaded into RAM only on the first malformed-JSON row encountered
  during a run, saving memory and startup time when all rows parse cleanly.

### Changed
- **`json_formatter` default changed from `False` to `None`**: when `None`, the
  formatter is auto-enabled for Ollama / local-model providers (which more often emit
  malformed JSON) and disabled for all other providers. Pass `json_formatter=False`
  explicitly to opt out, or `json_formatter=True` to force-enable for any provider.
- **`ensure_formatter_available()`** no longer prompts interactively (`input()`
  removed). It now auto-downloads the model and auto-installs deps, printing a clear
  console warning instead. This makes it safe to call from R via `reticulate`.
- **Merge prompt `name_instruction`**: the "such as / parenthetical examples" guidance
  is now included in the default (non-specific) branch as a SHOULD instruction.
  Previously only `specificity="specific"` got this guidance; now all callers do.
  `specificity="specific"` upgrades it to MUST.

---

## [1.0.20] - 2026-05-16

### Added
- **Domain-keyed prompt registry** (`_prompts.py`): introduces a `PROMPTS` dict keyed
  by domain (`"neutral"`, `"survey"`, `"social"`, `"academic"`, `"policy"`, `"web"`),
  each with overrideable `"first_pass"` and `"merge"` prompt slots. A `get_prompt(domain,
  slot)` helper falls back to `"neutral"` for any slot a domain does not override.

### Changed
- **`extract()` and `explore()`** accept a new `domain: str = "neutral"` parameter.
  Passing `domain=...` selects the corresponding prompt template for both the per-chunk
  first-pass extraction and the second-pass semantic merge.
- **Default prompt is now domain-neutral**: previously both functions rendered the
  survey-flavored prompt (`"respondent"`, `"reason a respondent might give"`) for all
  callers. Now callers that do not pass `domain=...` receive a truly generic template.
  Callers relying on the old survey language should pass `domain="survey"` explicitly.
- **`explore_common_categories()`** accepts `domain: str = "neutral"` and routes both
  inline prompts through `get_prompt()` instead of hardcoding them.

---

## [1.0.19] - 2026-05-11

### Changed
- **Canonical import name normalized to `catstack`** (no separator), matching
  the rest of the cat-* family. The previous name `cat_stack` continues to
  work as a backward-compatible alias — `import cat_stack` and `import catstack`
  resolve to the same module object, and `from cat_stack.text_functions import X`
  still works. Existing code does not need to change. New code is encouraged
  to use `catstack`.
- **Source directory** renamed from `src/cat_stack/` to `src/catstack/`. The
  alias is shipped as a tiny `src/cat_stack/__init__.py` that does a
  `sys.modules` swap. Wheels include both.

---

## [1.0.18] - 2026-04-26

### Added
- **`max_workers` parameter for `explore()`**: Adds parallel execution via
  `concurrent.futures.ThreadPoolExecutor`. When `max_workers > 1`, all
  `iterations × divisions` API calls are dispatched concurrently up to the
  specified worker count, reducing wallclock time by ~5x at `max_workers=8`
  with no change to output. Sequential behaviour (`max_workers=1`) is the
  default and is unchanged. The `chunk_delay` parameter is ignored in parallel
  mode. Works across all providers (OpenAI, Mistral, Anthropic, Google, etc.).

---

## [1.0.17] - 2026-04-25

### Fixed
- **`explore()` markdown in raw output**: Applied `_clean_label` to the raw
  category list returned by `explore()` (via `explore_common_categories` with
  `return_raw=True`), so bold markers, parentheticals, and frequency suffixes
  are stripped from the saturation frequency output, matching the behaviour
  already applied in the `extract()` path.

---

## [1.0.16] - 2026-04-24

### Fixed
- **Mistral API 400 error on structured output**: `_providers.py` was sending
  `response_format: {type: "json_schema", strict: true}` to the Mistral API, which
  does not support the strict json_schema mode. Added `"mistral"` to the `json_object`
  provider list so Mistral requests use `{type: "json_object"}` instead.

---

## [1.0.15] - 2026-04-21

### Added
- **`_clean_label` utility function**: New `_utils._clean_label` strips markdown formatting
  (bold markers `**...**`, parenthetical notes `(...)`, and trailing frequency counts `: N`)
  from category label strings. This normalises output from quantized local models (e.g.
  Mistral Nemo via Ollama) that inconsistently format labels with markdown decoration.
- Applied `_clean_label` post-processing to the final category list in `text_functions.py`,
  `image_functions.py`, and `pdf_functions.py` so all extraction paths produce clean labels.

---

## [1.0.14] - 2026-04-03

### Fixed
- **Batch summarization whitespace preservation**: `_batch.py` now uses the safe
  `extract_json()` implementation that preserves spaces inside JSON string values.
  Fixes batch-mode summaries that were returning words run together (for example,
  bullet-point summaries and structured summary fields).

---

## [1.0.13] - 2026-04-03

### Added
- **`format="raw"` summarize preset**: New format with no built-in instruction, allowing
  callers to supply the full instruction via `instructions=` without any preset preamble.
  Used internally by cat-pol's `format="bill_analysis"` to prevent the default paragraph
  instruction from overriding structured extraction prompts.

### Fixed
- **400 "model not found" error handling**: Extended the `_providers.py` error handler to
  treat HTTP 400 responses containing both "not found" and "model" as model-not-found errors
  (same as 404), so they surface a clear message instead of silently failing.

---

## [1.0.12] - 2026-04-02

### Added
- **Preflight model validation**: Before classification begins, each cloud model
  receives a minimal test call to catch issues early. If the model does not exist,
  classification halts immediately with a clear error message. If the model does not
  support structured JSON output, a warning is displayed and classification proceeds
  with the prompt-based fallback. Prevents long-running jobs from failing silently
  due to a single broken model.
- **Structured output fallback warning**: When the fallback is triggered, a one-time
  message is printed: `[CatLLM] Model 'X' does not support structured JSON output.
  Falling back to prompt-based JSON parsing.`

---

## [1.0.11] - 2026-04-02

### Fixed
- **Structured output fallback for HuggingFace models**: When a model returns HTTP 400
  indicating it does not support structured outputs (`response_format: json_object`),
  the request is automatically retried without `response_format`. The prompt still
  instructs the model to return JSON, and `extract_json()` parses it from the free-text
  response. Fixes Qwen3-32B-FP8 and other quantized models on Novita that dropped
  structured output support.

## [1.0.10] - 2026-04-02

### Fixed
- **HuggingFace JSON key normalization**: Models using `json_object` mode (no strict
  schema enforcement) sometimes return keys like `"1. Category name"` instead of `"1"`.
  Added `_normalize_json_keys()` helper that extracts the leading numeric prefix,
  applied in `aggregate_results()` and all retry validation paths. This fixes
  classification failures for HuggingFace-routed models (Novita, Together, SambaNova, etc.).
- **HuggingFace Together endpoint routing**: The Together-specific endpoint
  (`router.huggingface.co/together/v1`) rejects model names with `:together` suffix
  and does not support the clean model name either. Fixed by always routing through
  the generic HuggingFace router (`router.huggingface.co/v1`) which natively handles
  all router suffixes (`:together`, `:novita`, `:sambanova`, etc.).

---

## [1.0.5] - 2026-03-23

### Added
- **Image summarization**: `summarize()` now supports image file inputs (`.jpg`, `.png`, etc.)
  with visual analysis via multimodal LLMs. Previously images only worked with `classify()`.
- **HuggingFace router suffix support**: Model names can now include a router suffix
  (e.g., `qwen/qwen3-vl-235b-a22b-instruct:novita`) to route requests to a specific
  HuggingFace Inference Provider (novita, together, sambanova, cerebras, fireworks).
  The suffix is automatically stripped from the model name before API calls.
- **`explore()` specificity improvement**: When `specificity="specific"`, category names now
  include detailed descriptions with examples (e.g., parenthetical clarifications) instead
  of bare labels.

### Fixed
- **`summarize()` JSON extraction**: Replaced `extract_json()` (which strips spaces, brackets,
  and newlines — fine for classification 0/1 output but destructive for freeform summary text)
  with a new `_extract_json_for_summary()` that preserves content. Also strips `<think>` tags
  from thinking models (Qwen3, DeepSeek).
- **`summarize()` array responses**: `extract_summary_from_json()` now handles models that
  return bullet-point summaries as JSON arrays (`{"summary": ["- point 1", "- point 2"]}`)
  instead of a single string.

---

## [1.0.0] - 2026-03-22

### Notes
- First stable release. All core features — classify, extract, explore, summarize,
  prompt_tune — are tested and production-ready.

---

## [0.4.2] - 2026-03-22

### Added
- **`format` parameter on `summarize()`**: Controls the output structure of summaries. Default `"paragraph"`.
  - `"paragraph"` — flowing prose (default, existing behavior)
  - `"bullets"` — bullet-point list of key points
  - `"one-liner"` — single-sentence summary (auto-sets max_length=40)
  - `"structured"` — labeled sections: What, Who, Why, Impact
  - `"report"` — comprehensive full-page report with Overview, Background, Key Provisions, Stakeholders/Impact, and Implementation sections (auto-sets max_length=800)
  - Format instructions are prepended to any user-provided `instructions`. User `max_length` overrides the format default.

### Fixed
- **`summarize()` error handling**: Fixed a bug where `summarize_single_item()` ignored the error return from `client.complete()` in both the text and PDF code paths. The error was stored as `_err` (unused variable) instead of being checked, causing failed API calls to silently return empty summaries instead of being detected as failures. This meant the batch retry logic (2 additional passes × 5 retries each) never fired for summarization failures. Now properly checks `if error:` and returns the error, enabling the full retry pipeline (up to 15 total attempts per item).

---

## [0.4.0] - 2026-03-21

### Added
- **`prompt_tune` parameter on `classify()`**: Inline prompt optimization — runs
  `prompt_tune()` on a subsample before the full classification, then passes the
  optimized `system_prompt` automatically.
  - `prompt_tune=True`: Tune on 10 random items (default sample size).
  - `prompt_tune=N`: Tune on N random items.
  - `tune_iterations`, `tune_ui`, `tune_optimize` control the tuning behavior.
  - Tested on UCNets a19i data: **+5 pp** cell-level accuracy (89.8% → 94.8%)
    on 100 rows with Haiku 3.0, tuning on just 15 items.

---

## [0.3.0] - 2026-03-21

### Added
- **`prompt_tune()` — Automatic Prompt Optimization (APO)**: Iteratively refines classification prompts using user feedback. Classifies a random sample, opens a browser-based review UI for corrections, then generates per-category instructions to improve accuracy. Uses coordinate-descent: one category at a time, worst-first, with full error context across all categories.
  - Browser UI (`_review_ui.py`): Self-contained HTML page with checkboxes for toggling category assignments. No external dependencies — uses Python's built-in `http.server` and `webbrowser`.
  - Terminal fallback (`ui="terminal"`): Text-based correction input for headless environments.
  - `optimize` parameter: Target metric — `"balanced"` (default, average of accuracy/sensitivity/precision), `"precision"`, or `"sensitivity"`.
  - `add_other` parameter: Auto-detects missing "Other" catch-all category, matching `classify()` convention.
  - Returns optimized `system_prompt` string that can be passed directly to `classify(system_prompt=...)`.
  - Only opens browser once (baseline). Subsequent iterations auto-score against saved ground truth.
- **`pilot_test` parameter on `classify()`**: Run a pilot classification on a small random sample before the full run. User reviews results and can cancel if accuracy is too low.
  - `pilot_test=True`: Test on 10 random items. `pilot_test=N`: Test on N items.
  - Uses the same browser review UI as `prompt_tune()`.
- **`system_prompt` parameter on `classify()`**: Custom system-level instruction prepended to classification prompts. Use `prompt_tune()` to generate an optimized one.

---

## [0.2.0] - 2026-03-20

### Added
- **`input_mode` parameter** on `classify()`, `summarize()`, and `extract()`: Separates _what you want the model to do_ from _what file type to process_. Two modes:
  - `"text"` — classify/summarize text content, regardless of source format. For images and scanned PDFs, uses LLM-based OCR to extract text first.
  - `"visual"` — classify/summarize visual features of images or rendered PDF pages.
  - `None` (default) — auto-select based on file type, preserving all existing behavior.
- **`input_type` parameter** on `classify()`, `summarize()`, and `extract()`: Explicit file type filter for directories/mixed input. Options: `"auto"` (default, auto-detect from extensions), `"pdf"`, `"image"`, `"docx"`, `"text"`.
- **LLM-based OCR** (`_ocr_extract_text()` in `text_functions_ensemble.py`): When `input_mode="text"` is used with image or PDF input, a multimodal LLM extracts visible text from the document before classification/summarization. OCR is performed once per item and shared across all ensemble models.
  - For PDFs: tries PyMuPDF text extraction first; if the page has no extractable text (scanned/image PDF), falls back to rendering the page as an image and OCR-ing it via LLM. Prints `[CatStack] Page has no extractable text. Using LLM-based OCR.`
  - For images: sends the image to the LLM with an OCR prompt.
  - Uses the first multimodal-capable model in the ensemble (skips text-only providers like Ollama).
  - Not supported with `batch_mode=True` (raises `ValueError`).
- **`_resolve_input_params()` internal function**: Resolves `input_mode`, `input_type`, and the legacy `mode` parameter into a unified `(resolved_mode, file_type, warnings)` tuple. Handles all backward compatibility:
  - Old `mode="image"/"text"/"both"` still works when `input_mode` is not set.
  - Emits a deprecation warning when both `input_mode` and `mode` are explicitly set.
  - Validates incompatible combinations (e.g., `input_mode="visual"` on text/DOCX input raises `ValueError`).
- **DOCX support in `summarize()`**: `summarize_ensemble()` now handles DOCX file auto-detection and text extraction, matching the existing behavior in `classify_ensemble()`.
- **Image support in `summarize()`**: `summarize_ensemble()` now loads and processes image files when detected, enabling image summarization.

### Changed
- **`extract()` default `input_type`** changed from `"text"` to `"auto"`. When set to `"auto"`, `extract()` calls `_detect_input_type()` to auto-detect the input format. Explicit `input_type="text"` still works as before.

---

## [2.10.0] - 2026-03-15

### Added
- **Robustness and batch features for `summarize()`**: Added `safety`, `max_retries`, `batch_retries`, `retry_delay`, `row_delay`, `fail_strategy`, `batch_mode`, `batch_poll_interval`, and `batch_timeout` parameters — achieving full parity with `classify()`.
  - **Safety incremental saves**: `safety=True` saves partial results to CSV after each row. New `_save_partial_summarize_results()` helper in `text_functions_ensemble.py`.
  - **Row delay**: `row_delay` pauses between processing rows for rate limit management.
  - **Fail strategy**: `fail_strategy="strict"` blanks the entire row if any model fails; `"partial"` (default) keeps successful results.
  - **Batch mode**: `batch_mode=True` submits summarization as async batch jobs for 50% cost savings. Supports single-model (`run_batch_summarize()`) and multi-model ensemble (`run_batch_ensemble_summarize()`) modes. PDF input raises an error (batch is text-only).
- **`_parse_batch_results()` parse_mode parameter**: Added `parse_mode="json"|"text"` to `_parse_batch_results()` in `_batch.py`. When `"text"`, skips `extract_json()` and returns raw text — needed for summarization batch results.
- **New batch summarization functions** in `_batch.py`: `_run_one_batch_summarize_job()`, `_run_one_sync_summarize_model()`, `run_batch_summarize()`, `run_batch_ensemble_summarize()`.
- **Example notebooks**:
  - `Summarizing Text and PDF Data.ipynb` — text/PDF summarization with all features
  - `Classifying Text with Local Models (Ollama).ipynb` — local model classification
  - `Ensemble Classification with Cloud and Local Models.ipynb` — ensemble patterns, temperature ensembles, parallel vs sequential execution, embedding tiebreaker
  - `Exploring Categories with explore().ipynb` — raw category extraction and saturation analysis
  - `Extracting Categories with extract().ipynb` — refined category extraction and classify workflow

### Fixed
- **`extract()` image/PDF dispatch bug**: `survey_question` was silently ignored for `input_type="image"` and `input_type="pdf"` — the code passed `description or ""` instead of the resolved survey question. Now correctly uses `resolved_survey_question` for all input types.

### Removed
- **Logprobs-based confidence scores** (`_confidence.py`): Removed the experimental logprobs feature due to unreliable behavior across providers. The module was never integrated into the public API.

---

## [2.9.0] - 2026-03-12

### Added
- **Embedding centroid tiebreaker** (`embedding_tiebreaker` parameter in `classify()`): Resolves true consensus ties (equal votes for 0 and 1) using embedding centroids built from unanimously-agreed rows. Compares tied texts to positive and negative centroids via cosine similarity. Adds `category_N_resolved_by` columns to output. Requires `pip install cat-llm[embeddings]`. Text input only, multi-model ensemble only, not supported in batch mode.
  - New parameter: `min_centroid_size` (int, default 3) — minimum unanimous rows needed to build centroids.
  - New internal module: `src/cat_stack/_tiebreaker.py`.

---

## [2.8.2] - 2026-03-11

### Added
- **`claude-code` provider backend**: Added `claude-code` as a provider in `_providers.py`. Each LLM call shells out to `claude -p` (print mode), enabling the full catllm pipeline (retries, `extract_json()`, `categories_per_call`, threading) powered by the user's Claude Code token allowance with no API key. Use via `cat.classify(..., model_source="claude-code", user_model="sonnet")` from a standalone terminal or Python script. Not usable from within a Claude Code session (nested sessions blocked by CLI).
- **`check_claude_cli_available()`** utility function in `_providers.py` (re-exported via `text_functions.py`).

### Changed
- **`/catllm:classify` conversational redesign**: Replaced the rigid step-by-step questionnaire with a conversational-first flow. After finding the file, shows data preview and a single open-ended prompt ("What would you like to do with this data?"). Parses column, categories, model preference, and context from the user's free-text response. Only asks follow-ups for missing required parameters.
- **Smart API key auto-detection**: The classify skill now probes the environment for all known API keys (OpenAI, Anthropic, Google, Mistral, xAI, HuggingFace) at startup. If found, mentions them proactively and defaults to cloud. If none found and ≤200 rows, defaults to Claude Code native mode.
- **Claude Code (Path B) hard-capped at 200 rows**: Native classification mode now enforces a strict 200-row limit instead of allowing users to proceed with a warning.
- API key validation in `text_functions.py` now skips `claude-code` provider (like Ollama).

---

## [2.8.1] - 2026-03-10

### Added
- **Claude Code classification mode in `/catllm:classify`**: Added "Claude Code (no API key)" as a third model location option alongside Cloud API and Ollama. When selected, Claude Code itself acts as the classifier — no API key or external setup needed. Supports natural-language input (e.g. `/catllm:classify survey.csv on the response column for positive, negative, neutral sentiment`), auto-discovers categories by reading data samples directly, and produces the same DataFrame output format as the API-based pipeline. Best for quick/casual classification of datasets under 200 rows. Cloud and Ollama options labeled as "core cat-llm pipeline, empirically validated" to distinguish from the native Claude Code mode.

---

## [2.8.0] - 2026-03-10

### Added
- **Chunked category classification** (`categories_per_call` parameter in `classify()`): Splits large category lists into smaller chunks, runs a separate LLM call per chunk with local 1..N numbering, and merges results back into global numbering. Reduces prompt complexity per call and can improve accuracy for large category sets (20+). Each chunk automatically gets a temporary "Other" catch-all category to give the LLM an escape hatch for ambiguous responses; the "Other" is dropped before merging. A unified "Other" column is added to the output when all real categories are 0 but at least one chunk flagged "Other". Not supported with `batch_mode=True`. Works with all input types (text, PDF, image), all providers, ensemble mode, and all prompting strategies.
  - New internal module: `src/cat_stack/_chunked.py` with `run_chunked_classification()` and `_run_single_chunk_call()`.

---

## [2.7.0] - 2026-03-07

### Added
- **Sequential ensemble mode** (`parallel` parameter in `classify()`): Controls concurrent vs sequential model execution. Default `None` auto-detects: sequential for all-local models (Ollama), parallel for cloud providers. Set `parallel=True` to force concurrent execution or `parallel=False` to force sequential. Sequential mode is useful for resource-constrained environments or debugging.
- **Ollama support for `explore()` and `extract()`**: Local Ollama models now get the same pre-flight validation as `classify()` — checks that Ollama is running, verifies the model is available, offers auto-download for missing models, and warns about system resources. New `auto_download` parameter on both functions.
- **Single-label classification mode** (`multi_label=False` in `classify()`): Switches from the default multi-label mode (multiple categories can be 1) to single-label mode (exactly one best category gets 1, all others 0). Only the prompt text changes — JSON schema, parsing, validation, ensemble consensus, and DataFrame output format are all unchanged. Works with all input types (text, PDF, image), all prompting strategies (CoT, context prompt, step-back), and batch mode.
- **Ensemble batch mode (experimental)**: `batch_mode=True` now works with multi-model ensembles. Each model submits its own batch job concurrently via `ThreadPoolExecutor`; results are merged through the existing `aggregate_results` + `build_output_dataframes` pipeline and return the same DataFrame format as synchronous ensemble mode (per-model columns, `_consensus`, `_agreement`). Providers without a batch API (HuggingFace, Perplexity, Ollama) fall back to synchronous calls automatically. Prints an `[CatLLM] NOTE: experimental` warning when used.
  - New internal helpers: `_run_one_batch_job` (extracted from `run_batch_classify`), `_run_one_sync_model` (sync fallback), and `run_batch_ensemble_classify` (orchestrator) in `src/cat_stack/_batch.py`.
- **Embedding-based similarity scores** (`embeddings=True` in `classify()`): Adds `category_N_similarity` columns (0–1 float) alongside the binary 0/1 classification columns. Uses a local sentence-transformer model (`BAAI/bge-small-en-v1.5`, 33M params, ~130MB) to compute cosine similarity between each input text and each category. Requires `pip install cat-llm[embeddings]`.
  - New parameters: `embeddings` (bool, default `False`) and `category_descriptions` (dict, default `None` — optional richer text per category for improved similarity, e.g. `{"Financial reasons (...)": "The person moved because of money, high rent, ..."}`).
  - Scores are independent per (text, category) pair — no softmax across categories. Works with single-model and ensemble modes. Skipped automatically for PDF/image input. Model downloaded from HuggingFace Hub on first use.
  - New internal module: `src/cat_stack/_embeddings.py`. New optional dependency group: `[embeddings]` (installs `sentence-transformers`).
- **`json_formatter=True` in `classify()`**: Opt-in local JSON formatter fallback that uses a fine-tuned Qwen2.5-0.5B model to fix malformed classification JSON before marking responses as failed. The formatter only runs when `extract_json()` produces invalid output — zero cost on the happy path. On first use, the model (~1GB) is downloaded from HuggingFace Hub ([chrissoria/catllm-json-formatter](https://huggingface.co/chrissoria/catllm-json-formatter)). Requires `pip install cat-llm[formatter]`.
- **`src/cat_stack/_formatter.py`**: New internal module with `ensure_formatter_available()`, `load_formatter()`, and `run_formatter()` functions for the JSON formatter fallback.
- **`[formatter]` optional dependency group**: `pip install cat-llm[formatter]` installs `torch`, `transformers`, and `accelerate`.

---

## [2.6.0] - 2026-03-05

### Added
- **`batch_mode=True` in `classify()`**: New async batch inference mode that reduces API costs by 50% and bypasses standard rate limits. Supported providers: OpenAI, Anthropic, Google (Gemini), Mistral, and xAI (Grok). Not supported: HuggingFace, Perplexity, Ollama.
  - Packages all classification requests as a JSONL file, submits a single batch job, polls for completion, and returns a DataFrame identical in format to the synchronous single-model path.
  - New parameters: `batch_poll_interval` (seconds between polls, default 30) and `batch_timeout` (max wait in seconds, default 86400 = 24h).
  - Incompatible with multi-model ensemble (`models` list with >1 entry), PDF/image input, and `progress_callback`.
  - Returns the same simplified DataFrame format as synchronous single-model mode: `category_1`, `category_2`, ... columns with no model suffix, consensus, or agreement columns.
- **`BatchJobExpiredError`**: New exception raised when a batch job expires or is cancelled. Includes the job ID for provider dashboard lookup.
- **`BatchJobFailedError`**: New exception raised when a batch job terminates in a failed state.
- **`src/cat_stack/_batch.py`**: New internal module implementing all batch logic (JSONL building, file upload, job creation, polling, result download and parsing) for all five supported providers via pure HTTP — no provider SDKs required.

### Fixed
- **Google (Gemini) batch**: Switched from file-upload to inline requests format; fixed terminal state names (`BATCH_STATE_SUCCEEDED` not `JOB_STATE_SUCCEEDED`); fixed result extraction path (`response.inlinedResponses.inlinedResponses`); fixed response ordering — Google returns results out of order, so responses are now mapped via `metadata.key` rather than positional index. Verified: ≤0.3pp accuracy delta vs synchronous calls.
- **Mistral batch**: Fixed response parsing — Mistral wraps the completion inside `response.body`, mirroring the OpenAI envelope. Verified: ≤0.4pp accuracy delta vs synchronous calls.

### Changed
- `CERAD_functions.py`: Refactored `cerad_drawn_score()` to call `classify()` directly instead of the deprecated `image_multi_class()`. All scoring logic unchanged.

---

## [2.5.0] - 2026-02-26

### Added
- **`has_other_category()` utility**: New function in `cat_stack._category_analysis` that detects whether a category list contains a catch-all / "Other" category. Uses a two-tier heuristic (anchored patterns for exact matches, phrase patterns for short categories) with an optional LLM fallback for ambiguous cases.
- **`add_other` parameter in `classify()`**: Automatically detects when categories lack a catch-all "Other" option and prompts the user to add one. Supports three modes: `"prompt"` (default, interactive), `True` (silent), `False` (disabled). Including an "Other" category improves accuracy by giving models an outlet for ambiguous responses.
- **`check_category_verbosity()` utility**: New function that uses a single LLM call to assess whether each category has a description and examples. Returns per-category flags (`has_description`, `has_examples`, `is_verbose`).
- **`check_verbosity` parameter in `classify()`**: Alerts users when categories lack descriptions or examples (1 API call). Verbose categories with descriptions and examples improve accuracy by ~7 pp over bare labels. Default `True`.
- **Evidence-based prompting strategy warnings**: `classify()` now prints informational warnings when users enable strategies that empirical evidence shows are ineffective or harmful for structured classification:
  - `chain_of_verification=True`: WARNING — degrades accuracy by ~2 pp, costs 4x API calls.
  - Few-shot examples (`example1`–`example6`): NOTE — degrades accuracy by ~1 pp, amplifies over-classification.
  - `thinking_budget > 0`: NOTE — negligible gains, high failure rates, massive latency increase.
  - `chain_of_thought=True`: NOTE — no measurable effect on accuracy.
  - `step_back_prompt=True`: NOTE — small/inconsistent gains, hurts top-tier models, 2x cost.

---

## [2.4.1] - 2026-02-19

### Fixed
- **NaN row handling in classify()**: Skipped rows (NaN input) no longer falsely list all models as failed. Previously, NaN inputs generated fake error results for every model, causing `failed_models` to contain all model names. Now skipped rows correctly show empty `failed_models` and NaN category values.

---

## [2.4.0] - 2026-02-11

### Fixed
- **Schema validation in aggregate_results**: Responses with at least one valid category key (0/1 value) are accepted, but invalid keys are now stripped before storing — prevents garbage values like `"yes"` from silently becoming phantom 0 votes in consensus.
- **Failed model output**: Failed models now produce `None`/NA in output CSVs instead of silent zeros, in both `_save_partial_results()` and `build_output_dataframes()`.
- **Batch retry detection**: Schema validation applied consistently to detect failures and verify retry success.

### Added
- **Missing keys tracking**: `aggregate_results()` now returns `missing_keys` counts per model, and a classification quality summary is printed after classification completes.

---

## [2.3.4] - 2026-02-11

### Fixed
- **HuggingFace thinking support**: Models that reason by default (e.g., Qwen3) can now be controlled via `thinking_budget=0`, which sends `chat_template_kwargs: {"enable_thinking": False}` to disable thinking mode. HuggingFace providers now correctly receive `thinking_budget` through the payload pipeline.
- **OpenAI reasoning model detection**: Added `gpt-5` to reasoning model prefix list alongside o1/o3/o4. Simplified temperature handling — reasoning models never set temperature (only default=1 is valid).

### Changed
- **Consolidated duplicate `UnifiedLLMClient`**: Removed ~930 lines of duplicated provider infrastructure from `text_functions.py`. `_providers.py` is now the single source of truth; `text_functions.py` re-exports all names for backward compatibility.
- **Added `ARCHITECTURE.md`**: Module dependency map and `classify()` call chain showing where each function and prompting strategy originates.

---

## [2.3.3] - 2026-02-11

### Fixed
- **Critical: Thinking support was applied to wrong module** — v2.3.2 fixes were only applied to `_providers.py`, but the classify pipeline imports `UnifiedLLMClient` from `text_functions.py`. All three provider fixes now applied to both modules.
- **Google thinking support**: Fixed `thinkingConfig` placement in `text_functions.py` — must be inside `generationConfig`, not at the top level. Added minimum budget of 128 tokens.
- **OpenAI reasoning support**: `reasoning_effort` now only applied to reasoning models (o1, o3, o4-series). Regular models like gpt-4o skip this parameter gracefully.
- **Anthropic thinking support**: Extended thinking + forced `tool_choice` are incompatible — now uses `tool_choice: "auto"` when thinking is enabled. Also added temperature=1 requirement and minimum budget of 1024 tokens.

---

## [2.3.2] - 2026-02-10

### Fixed
- **Google thinking support**: Fixed `thinkingConfig` placement — must be inside `generationConfig`, not at the top level. Added minimum budget of 128 tokens.
- **OpenAI reasoning support**: Fixed conflict between `reasoning_effort` and `temperature` — temperature is now omitted when reasoning is enabled (`thinking_budget > 0`).
- **Anthropic thinking support**: Temperature is now set to 1 (Anthropic requirement) when extended thinking is enabled, instead of using the user-specified creativity value.

---

## [2.3.1] - 2026-02-10

### Changed
- **Extraction defaults updated**: `divisions` changed from 5 to **12** and `iterations` changed from 3 to **8** for `extract()`, `explore()`, and the `main.py` wrapper. These new defaults were determined through empirical analysis: a 6x6 grid search over both parameters (10 repeats per cell, 360 total runs) showed that extraction consistency peaks at 12 divisions and 8 iterations, with no meaningful improvement beyond this point.

---

## [2.3.0] - 2026-02-08

### Added
- **`explore()` function**: New entry point for raw category extraction — returns every category string from every chunk across every iteration, with duplicates intact. Useful for analyzing category stability and building saturation curves.
- `return_raw` parameter on `explore_common_categories()` to support raw output mode

---

## [2.2.0] - 2025-02-08

### Added
- **Unified `classify()` API**: Added 9 missing parameters (`survey_question`, `use_json_schema`, `max_workers`, `fail_strategy`, `max_retries`, `batch_retries`, `retry_delay`, `pdf_dpi`, `auto_download`) — `classify()` is now the single entry point for all classification
- **4-tuple model format**: `(model, provider, api_key, {"creativity": 0.5})` for per-model temperature control in ensembles
- **Image/PDF auto-category extraction**: `categories="auto"` now works for images and PDFs via routing through `extract()`, not just text
- **Retry logic for image extraction**: Exponential backoff (6 attempts) for `call_model_with_image()` and `describe_image_with_vision()`
- `progress_callback` support for real-time progress tracking

### Fixed
- **Agreement calculation**: Now measures fraction of models agreeing with consensus (was incorrectly measuring fraction voting 1)
- **MIME type for Anthropic**: Normalized `image/jpg` to `image/jpeg` in `_encode_image()`, fixing 400 errors on Anthropic image API calls
- Removed dead duplicate `classify()` from `main.py`

### Changed
- HuggingFace Space app now uses `classify()` instead of `classify_ensemble()` directly
- All example/test scripts updated to use `classify()` API

---

## [2.0.0] - 2025-01-17

### Major Release: Simplified API & Ensemble Methods

Version 2.0 represents a major simplification of CatLLM's architecture and API, making it easier to install, use, and extend.

### Added
- **Ensemble classification**: Run multiple models in parallel and combine predictions
  - Cross-provider ensembles (GPT-4o + Claude + Gemini)
  - Self-consistency ensembles (same model with temperature variation)
  - Model comparison mode for side-by-side evaluation
- **Consensus voting methods**:
  - `"majority"` - 50%+ agreement required
  - `"two-thirds"` - 67%+ agreement required
  - `"unanimous"` - 100% agreement required
  - Custom numeric thresholds (e.g., `0.75` for 75%)
- **Visualization tools** in web app:
  - Classification matrix heatmap
  - Category distribution charts
  - Download buttons for all visualizations
- PDF report generation with methodology documentation

### Changed
- **Simplified to 3 core functions**:
  - `extract()` - Discover categories in your data
  - `classify()` - Assign categories to your data
  - `summarize()` - Generate summaries of your data
- **Removed SDK dependencies**: All API calls now use pure `requests` library
  - No more `openai`, `anthropic`, `google-generativeai` package requirements
  - Lighter installation, fewer dependency conflicts
  - Unified HTTP interface for all providers
- **Streamlined parameters**: Consistent parameter names across all functions
- Web app UI improvements: button alignment, Garamond font, improved layout

### Removed
- Direct SDK dependencies (openai, anthropic, google-generativeai, mistralai)
- Legacy function names (old aliases still work but are deprecated)

### Migration from 1.x
Most code will work without changes. Key differences:
- SDK-specific features (like streaming) are no longer available
- All providers now use the same HTTP-based interface
- New `models` parameter enables ensemble mode

---

## [0.1.15] - 2025-01-10

### Added
- `summarize()` function for text and PDF summarization with multi-model support
- `focus` parameter for `extract()` to prioritize specific themes during category discovery
- `progress_callback` parameter for PDF page-by-page progress updates
- Multi-model support in `classify()` via `models` parameter for ensemble classification
- Documentation for `summarize()` function in README

### Changed
- Converted web app from Gradio to Streamlit for better mobile support
- Improved PDF functionality in HuggingFace app

### Fixed
- Parameter mapping in `classify()` function
- Bug in extract function for edge cases
- Extract API now uses chat.completions for OpenAI-compatible providers

---

## [0.1.14] - 2025-01-02

### Added
- **Ollama support** for local model inference (llama3, mistral, etc.)
- Auto-download of Ollama models when not installed
- System resource checks before downloading large models
- Confirmation prompts before downloading Ollama models

### Changed
- Improved error messages and download warnings for Ollama integration

---

## [0.1.13] - 2024-12-30

### Added
- Unified HTTP-based multi-class text classification
- Multiple categories per item for PDFs and images
- Extract categories functionality for PDFs and images

### Changed
- Web app made mobile-friendly
- Auto-adjust `divisions` and `categories_per_chunk` for small datasets
- Aligned PDF function output format with text classifier

### Fixed
- Image classification output alignment with other classifiers
- Glitch causing errors in app when using image classification

---

## [0.1.12] - 2024-12-15

### Added
- **PDF document classification** with multiple processing modes:
  - `image` mode: renders pages as images for visual analysis
  - `text` mode: extracts text for text-based classification
  - `both` mode: combines image and text analysis
- **HuggingFace Spaces web app** for browser-based classification

### Changed
- Moved web app to CatLLM organization on HuggingFace

---

## [0.1.11] - 2024-12-01

### Added
- **Image classification** using vision models
- Image file upload support with description context
- Support for multiple image formats (PNG, JPG, JPEG, GIF, WEBP)

---

## [0.1.10] - 2024-11-20

### Added
- **Chain of Verification (CoVe)** prompting for improved accuracy
- **Step-back prompting** option for complex classifications
- **Context prompting** to add expert domain knowledge
- Warning messages for CoVe users about processing time

### Changed
- Refactored and tested multi_class function
- Cleaned up prompt code structure

### Fixed
- CoT prompt not producing structured output in some cases
- Error handling improvements for Google, OpenAI, and Mistral providers

---

## [0.1.9] - 2024-11-15

### Added
- **HuggingFace Inference API** support as model provider
- Auto-detection of model source based on model name
- Few-shot learning with `example1` through `example6` parameters

### Changed
- Default model for text classification set to GPT-4o

---

## [0.1.8] - 2024-11-10

### Added
- **Perplexity** as web search provider
- Advanced search with dates and confidence scores
- Formal URL output in web search function

### Changed
- Web search method no longer halts on rate limit
- Removed case sensitivity for `model_source` input

---

## [0.1.7] - 2024-11-05

### Added
- **Google search** capabilities for web search function
- Web search dataset building function
- Example script for categorizing text data

### Changed
- `creativity` parameter now optional (uses model defaults)
- Improved column names for easier understanding

### Fixed
- Error message when model is not valid
- Image inputs with file paths no longer crash the function

---

## [0.1.6] - 2024-10-25

### Added
- **xAI (Grok)** support for text classification
- Auto-create categories option in multi_class function
- Rate limit handling for OpenAI and Google

### Fixed
- Issue where whole row was converted to missing if one category wasn't output
- HuggingFace retry when incorrect JSON format is returned
- Column converting to 0s for valid rows
- Explore corpus failure when non-string value in rows

---

## [0.1.5] - 2024-10-15

### Added
- **Google (Gemini)** support for multi-class text classification
- **Anthropic (Claude)** support for CERAD and image functions
- **Mistral** support for CERAD and image functions
- Reference images provided within package for CERAD scoring

### Changed
- Updated license to be JOSS-acceptable (MIT)

---

## [0.1.4] - 2024-10-01

### Added
- `explore_common_categories()` function for automatic category discovery
- Research question parameter for guided category extraction
- Specificity parameter ("broad" or "specific") for category granularity

---

## [0.1.3] - 2024-09-15

### Added
- **CERAD cognitive assessment** scoring functions
- Support for reference images in CERAD analysis
- Option to specify whether image contains a reference

### Changed
- Separated CERAD functions into dedicated module

---

## [0.1.2] - 2024-09-01

### Added
- Image classification functions with OpenAI vision models
- UCNets example usage documentation

### Changed
- Package can now be imported as `cat_stack` instead of `cat_llm`

---

## [0.1.1] - 2024-08-15

### Added
- Logo and branding
- Improved README documentation

### Fixed
- Various small fixes and improvements

---

## [0.1.0] - 2024-08-01

### Added
- **Initial release**
- `classify()` function for multi-class text classification
- Support for OpenAI models (GPT-4, GPT-4o, GPT-3.5)
- Binary classification output (0/1) for each category
- CSV export functionality
- Basic error handling and retry logic

---

## Version History Summary

| Version | Date | Highlights |
|---------|------|------------|
| **0.2.0** | **2026-03-20** | **input_mode/input_type params, LLM-based OCR for images & scanned PDFs** |
| **2.10.0** | **2026-03-15** | **Summarize robustness & batch parity, 5 example notebooks, remove logprobs** |
| **2.9.0** | **2026-03-12** | **Embedding centroid tiebreaker for ensemble consensus ties** |
| **2.8.2** | **2026-03-11** | **Claude-code provider backend, redesigned /catllm:classify flow** |
| **2.8.1** | **2026-03-10** | **Claude Code classification mode in /catllm:classify** |
| **2.8.0** | **2026-03-10** | **Chunked category classification (categories_per_call)** |
| **2.7.0** | **2026-03-07** | **Sequential ensemble, Ollama for explore/extract, single-label, batch ensemble, embeddings, json_formatter** |
| **2.6.0** | **2026-03-05** | **Batch mode for classify (50% cost savings)** |
| **2.5.0** | **2026-02-26** | **Auto-add Other category, category verbosity check, prompting warnings** |
| **2.4.1** | **2026-02-19** | **Fix NaN row handling in classify** |
| **2.4.0** | **2026-02-11** | **Schema validation fixes, failed model output as NA** |
| **2.3.4** | **2026-02-11** | **HuggingFace thinking, OpenAI reasoning model detection** |
| **2.3.3** | **2026-02-11** | **Fix thinking support in classify pipeline (was applied to wrong module)** |
| **2.3.2** | **2026-02-10** | **Thinking fixes for Google, OpenAI, Anthropic (in _providers.py only)** |
| **2.3.1** | **2026-02-10** | **Empirically optimized extraction defaults (divisions=12, iterations=8)** |
| **2.3.0** | **2026-02-08** | **`explore()` for raw category extraction and saturation analysis** |
| **2.2.0** | **2025-02-08** | **Unified classify() API, image auto-categories, ensemble fixes** |
| **2.0.0** | **2025-01-17** | **Simplified API, ensemble methods, removed SDK dependencies** |
| 0.1.15 | 2025-01-10 | Summarization, focus parameter, Streamlit web app |
| 0.1.14 | 2025-01-02 | Ollama local inference |
| 0.1.13 | 2024-12-30 | Multi-category support, mobile web app |
| 0.1.12 | 2024-12-15 | PDF classification, HuggingFace app |
| 0.1.11 | 2024-12-01 | Image classification |
| 0.1.10 | 2024-11-20 | CoVe, step-back, context prompting |
| 0.1.9 | 2024-11-15 | HuggingFace support, few-shot learning |
| 0.1.8 | 2024-11-10 | Perplexity web search |
| 0.1.7 | 2024-11-05 | Google search, web search datasets |
| 0.1.6 | 2024-10-25 | xAI/Grok support, auto-categories |
| 0.1.5 | 2024-10-15 | Google/Anthropic/Mistral providers |
| 0.1.4 | 2024-10-01 | Category discovery function |
| 0.1.3 | 2024-09-15 | CERAD cognitive scoring |
| 0.1.2 | 2024-09-01 | Image classification |
| 0.1.1 | 2024-08-15 | Branding, documentation |
| 0.1.0 | 2024-08-01 | Initial release |

---

[0.2.0]: https://github.com/chrissoria/cat-stack/compare/v0.1.0...v0.2.0
[2.10.0]: https://github.com/chrissoria/cat-llm/compare/v2.9.0...v2.10.0
[2.9.0]: https://github.com/chrissoria/cat-llm/compare/v2.8.2...v2.9.0
[2.8.2]: https://github.com/chrissoria/cat-llm/compare/v2.8.1...v2.8.2
[2.8.1]: https://github.com/chrissoria/cat-llm/compare/v2.8.0...v2.8.1
[2.8.0]: https://github.com/chrissoria/cat-llm/compare/v2.7.0...v2.8.0
[2.7.0]: https://github.com/chrissoria/cat-llm/compare/v2.6.0...v2.7.0
[2.6.0]: https://github.com/chrissoria/cat-llm/compare/v2.5.0...v2.6.0
[2.5.0]: https://github.com/chrissoria/cat-llm/compare/v2.4.1...v2.5.0
[2.4.1]: https://github.com/chrissoria/cat-llm/compare/v2.4.0...v2.4.1
[2.4.0]: https://github.com/chrissoria/cat-llm/compare/v2.3.4...v2.4.0
[2.3.4]: https://github.com/chrissoria/cat-llm/compare/v2.3.3...v2.3.4
[2.3.3]: https://github.com/chrissoria/cat-llm/compare/v2.3.2...v2.3.3
[2.3.2]: https://github.com/chrissoria/cat-llm/compare/v2.3.1...v2.3.2
[2.3.1]: https://github.com/chrissoria/cat-llm/compare/v2.3.0...v2.3.1
[2.3.0]: https://github.com/chrissoria/cat-llm/compare/v2.2.0...v2.3.0
[2.2.0]: https://github.com/chrissoria/cat-llm/compare/v2.0.0...v2.2.0
[2.0.0]: https://github.com/chrissoria/cat-llm/compare/v0.1.15...v2.0.0
[0.1.15]: https://github.com/chrissoria/cat-llm/compare/v0.1.14...v0.1.15
[0.1.14]: https://github.com/chrissoria/cat-llm/compare/v0.1.13...v0.1.14
[0.1.13]: https://github.com/chrissoria/cat-llm/compare/v0.1.12...v0.1.13
[0.1.12]: https://github.com/chrissoria/cat-llm/compare/v0.1.11...v0.1.12
[0.1.11]: https://github.com/chrissoria/cat-llm/compare/v0.1.10...v0.1.11
[0.1.10]: https://github.com/chrissoria/cat-llm/compare/v0.1.9...v0.1.10
[0.1.9]: https://github.com/chrissoria/cat-llm/compare/v0.1.8...v0.1.9
[0.1.8]: https://github.com/chrissoria/cat-llm/compare/v0.1.7...v0.1.8
[0.1.7]: https://github.com/chrissoria/cat-llm/compare/v0.1.6...v0.1.7
[0.1.6]: https://github.com/chrissoria/cat-llm/compare/v0.1.5...v0.1.6
[0.1.5]: https://github.com/chrissoria/cat-llm/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/chrissoria/cat-llm/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/chrissoria/cat-llm/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/chrissoria/cat-llm/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/chrissoria/cat-llm/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/chrissoria/cat-llm/releases/tag/v0.1.0
