# Changelog

All notable changes to CatLLM will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **`model_source="claude-agent"` provider** — classify through a Claude
  subscription via the cat-agent SDK backend (no API key), alongside the
  existing `claude-code` subprocess shim. Dispatch mirrors the `claude-code`
  branch exactly: a `PROVIDER_CONFIG` entry, `_detect_model_source`
  recognition, a `_call_claude_agent` method (async adapter driven via
  `asyncio.run` per call; message flattening identical to `_call_claude_cli`),
  an ensemble availability check + preflight-probe skip, and an `[agent]`
  extra (`pip install cat-stack[agent]`). Entirely additive and gated on the
  new provider value: the full pre-existing test suite passes unchanged
  (507→507) plus 5 new dispatch tests, and a live `complete()` classified
  correctly through the subscription. Requires the cat-agent package
  (published separately); without it the provider returns a clear
  `pip install cat-stack[agent]` hint rather than a traceback.

### Changed
- Claude Code CLI provider: install-help URL in the "CLI not found" error
  updated to the current docs home (`code.claude.com/docs`; the old
  docs.anthropic.com path still redirects). The integration itself was
  audited against Claude Code 2.1.197 — all flags (`-p`,
  `--output-format text`, `--model`, `--system-prompt`) remain valid, and a
  live classify() run through `model_source="claude-code"` passes on the
  2.0.1 engine.

## [2.0.1] - 2026-07-03

### Fixed
- **`prompt_tune(description=...)` ran the whole tuning loop with no prompt
  context.** Same gap as the `classify()` fix in 2.0.0: `description` is the
  documented canonical param, but the tuning prompts' "Context:" line keys
  off `survey_question` downstream. The reconciliation now lives in one
  shared helper (`_resolve_description_context`) used by both entry points:
  only-`survey_question` → DeprecationWarning + mirrored to `description`;
  only-`description` → mirrored to `survey_question`; both → kept distinct.
  `explore()` and `summarize()` were audited and already route `description`
  correctly.

## [2.0.0] - 2026-07-03

First stable release of the 2.0 line — promotes 2.0.0b1–2.0.0b6 (theme
collapsing, centralized provider param shaping, current-generation Anthropic
support, graded cross-provider `thinking_budget`) to stable. Domain packages
pinning `cat-stack>=1.6.3` pick this release up automatically.

### Fixed
- **`classify(description=...)` silently lost the prompt context line for
  text input.** `description` is the documented canonical replacement for the
  deprecated `survey_question=`, but downstream prompt assembly (the
  `Context: ...` line, the step-back question, and the `categories="auto"`
  requirement) still keyed off `survey_question` — so callers passing only
  `description=`, which includes the cat-survey / cat-pol / cat-web /
  cat-ademic classify wrappers, got no context framing at all (and step-back
  / auto-categories raised despite context being provided). `description` is
  now mirrored into `survey_question` when the latter isn't given, restoring
  the exact prompt those callers produced before the rename. Callers passing
  both (e.g. cat-vader: `survey_question=` feed question +
  `description=` platform context) keep the two channels distinct.

### Added
- **Batch-mode cost nudge in `classify()`.** Large synchronous text runs now
  print a one-line tip when they qualify for `batch_mode=True` (~50% cheaper
  via the async batch API, higher rate limits, identical prompts/results).
  The tip only fires when opting in would actually work — text input, ≥ ~500
  estimated API calls (rows × batch-capable models), no batch-incompatible
  options (`categories_per_call`, `chain_of_verification`,
  `embedding_tiebreaker`, `progress_callback`), and at least one model on a
  provider with a batch API (OpenAI/Anthropic/Google/Mistral/xAI). Purely
  informational: it never changes or aborts the run, and any internal error
  in the eligibility check is swallowed.

## [2.0.0b6] - 2026-07-03

### Changed
- **Sampling/reasoning param handling is centralized in
  `apply_model_params()`** (`_providers.py`, exported). One function now
  decides which params a given provider+model accepts and in what form —
  Anthropic temperature gating + adaptive-vs-budget thinking + max_tokens
  headroom, OpenAI reasoning-model temperature skip + `reasoning_effort`,
  Google `generationConfig` temperature/`thinkingConfig`, xAI
  `reasoning_effort`, Ollama `think`, HuggingFace `chat_template_kwargs`.
  Every payload builder routes through it: the central `_build_*_payload`
  methods (and therefore `classify`/`explore`/`extract`/`summarize` and the
  batch API), all `calls/` strategy leaves (stepback / CoVe / top_n and their
  image/pdf variants), the direct per-provider builders in
  `image_functions.py` / `pdf_functions.py`, and `_call_google_multimodal`.
  Future provider quirks are fixed once instead of at every call site.
  Runtime 400 fallbacks remain in `complete()` (its cached capability flags
  feed the shaper via `UnifiedLLMClient._param_overrides()`); the direct-HTTP
  leaves get correct up-front params for known model families but still have
  no runtime net.
- **`thinking_budget` now maps consistently across providers.** It stays a
  single token-count knob, but providers whose API takes an effort *enum*
  (rather than a literal token budget) previously collapsed every positive
  budget to `"high"`, so the same value behaved very differently by provider.
  A shared table (`_thinking_budget_to_effort`) now grades a positive budget
  into `low` / `medium` / `high` (`<=2048` low, `<=8192` medium, else high) and
  every effort-enum provider consults it: OpenAI/xAI `reasoning_effort`,
  Anthropic-adaptive `output_config.effort`, and Ollama gpt-oss `think`. Token
  providers (Google, legacy Anthropic) still send the literal count; bool-only
  families (Ollama qwen3/deepseek, HF Qwen3) can only toggle on/off. **Behavior
  change:** OpenAI and xAI reasoning models now send graded effort — a small
  `thinking_budget` sends `"low"`/`"medium"` where it previously sent `"high"`.
  Thresholds are tunable in one place (`_THINKING_EFFORT_*`).

### Fixed
- **Extended thinking 400s on current Anthropic models when
  `thinking_budget > 0`.** `_build_anthropic_payload` always sent the legacy
  fixed-budget form `thinking: {"type": "enabled", "budget_tokens": N}`, which
  is rejected with a hard 400 (no fallback) on `claude-opus-4-7` /
  `claude-opus-4-8` / `claude-sonnet-5` / `claude-fable-5` — those generations
  require adaptive thinking. cat-stack now emits `thinking: {"type":
  "adaptive"}` for those models (skipping `temperature`, which they also
  reject) while keeping the explicit budget for older models (Opus 4.6, Sonnet
  4.6, …), selected by a prefix table with a runtime 400 safety net for future
  families. Also hardened `_parse_anthropic_response` to prefer a `tool_use`
  block over a text preamble, since the thinking path uses
  `tool_choice="auto"` and the model may narrate before the structured tool
  call. Default `classify()` calls are unaffected (`thinking_budget=0` sends no
  `thinking`).
- **`temperature` 400s on Sonnet-5 / Fable-5 Anthropic models when
  `creativity` is set.** The up-front skip list of models that reject the
  `temperature` sampling parameter only covered `claude-opus-4-7` /
  `claude-opus-4-8`, so `classify(..., creativity=<value>)` on
  `claude-sonnet-5` or `claude-fable-5` sent `temperature` and got a hard 400.
  Added both prefixes to the skip list, and broadened the runtime 400 safety
  net in `complete()` to match any parameter-rejection wording (Sonnet 5 /
  Fable 5 reject the param rather than calling it "deprecated"), not just the
  literal "deprecated" string. Default `classify()` calls are unaffected
  (`creativity=None` sends no `temperature`).
- **`temperature` 400s in the `calls/` strategy leaves (stepback, CoVe,
  top_n).** These per-strategy modules build the Anthropic payload directly
  (bypassing `_build_anthropic_payload`), so they still sent `temperature` on
  the newest models and — because they swallow exceptions — failed *silently*
  (lost stepback insight / CoVe verification / top_n categories) on
  `claude-opus-4-7`+, `claude-sonnet-5`, `claude-fable-5`. Each leaf now gates
  `temperature` with the same `_anthropic_supports_temperature()` check. These
  paths back the classify strategy options; `explore()` routes through
  `complete()` and was already covered by the central fix. (Superseded in the
  same release by the `apply_model_params()` centralization above, which
  extends the fix to every remaining direct payload builder.)
- **`temperature` 400s in the remaining direct Anthropic payload builders.**
  The image/pdf strategy leaves (`calls/image_stepback.py`,
  `calls/pdf_stepback.py`, `calls/image_CoVe.py`, `calls/pdf_CoVe.py`) and
  the per-provider builders inside `image_functions.py` / `pdf_functions.py`
  still sent `temperature` unconditionally, so image/PDF classification with
  `creativity` set degraded silently on `claude-opus-4-7`+, `claude-sonnet-5`,
  `claude-fable-5`. All now route through `apply_model_params()`. The same
  migration also stops sending `temperature` to OpenAI reasoning models
  (o-series / GPT-5) from these leaves — they reject non-default values.
- **Misplaced `generationConfig` broke Google stepback whenever `creativity`
  was set.** `get_stepback_insight_google` (`calls/stepback.py`) spread
  `generationConfig` *inside* `contents[0]` instead of at the top level of
  the request body. Gemini hard-400s on that shape (verified live:
  `Unknown name "generationConfig" at 'contents[0]'`), and the leaf swallows
  exceptions — so any Google classify run using the stepback strategy with
  `creativity` set silently lost the stepback insight entirely. The shaper
  migration places it correctly.
- **`_call_google_multimodal` (PDF/image ensemble path) now shapes Google
  params like the text path.** Previously `thinking_budget=0` sent nothing
  (leaving Gemini's default thinking ON, unlike the text path which sends an
  explicit zero since v1.6.8) and positive budgets skipped the 128-token
  floor. It now routes through `apply_model_params()` with the client's
  cached thinking floor.

## [2.0.0b5] - 2026-06-15

### Fixed
- **Spurious classification failures on WAF-fronted providers (notably the
  HuggingFace router's `featherless-ai` backend).** Requests were sent with no
  explicit `User-Agent`, so `requests` defaulted to `python-requests/x.y`,
  which featherless's Cloudflare bot rule intermittently 403s — surfacing as
  rows with `processing_status="error"` even though the API key, endpoint, and
  model were all correct (verified: default agent 0/15 success, browser agent
  15/15 on identical rapid calls). Both request paths now send a browser
  `User-Agent`: the main classification call (`_get_headers`) and the
  provider-detection probe (`_detect_huggingface_endpoint`). Harmless on
  providers that don't inspect the agent.

## [1.6.9] - 2026-06-13

### Fixed
- **JSON formatter could not load on transformers 4.56–4.57.x.** The
  formatter model repo's `tokenizer_config.json` stored `extra_special_tokens`
  as a list, which those transformers versions reject (`'list' object has no
  attribute 'keys'`). The formatter only loads when a classification model
  emits malformed JSON, so this surfaced only for weak local models. Two-part
  fix: (a) the HF repo `chrissoria/catllm-json-formatter` config was corrected
  (`extra_special_tokens` → `{}`; the tokens remain in the tokenizer, verified
  lossless) — this helps ALL versions with no upgrade; (b)
  `_formatter.load_formatter()` now defensively snapshots the repo and
  normalizes a list-valued `extra_special_tokens` to `{}` if it ever sees the
  error again, and tries `dtype=` (≥4.56) then falls back to `torch_dtype=`
  (<4.56) on model load. Thanks to a beta user for the diagnosis.
- **Formatter output truncated for large category sets.** `run_formatter`
  generated `max_new_tokens=128`, which truncated the JSON for 28- and
  48-category tasks; raised to 512.
- **Silent degradation when auto-installing formatter deps.** When
  `json_formatter=True` and deps were missing, the in-process pip install
  could not be imported by the same running process, and classification
  silently degraded to all-error rows. `_check_dependencies_installed()` now
  calls `importlib.invalidate_caches()`, and when deps land on disk but can't
  be imported in-process the user gets a clear "re-run" message instead of
  broken output. Pre-installing `pip install 'cat-stack[formatter]'` avoids
  the path entirely (recommended pre-step).

## [1.6.8] - 2026-06-13

### Fixed
- **Google reasoning control was silently absent at `thinking_budget=0`.**
  The Gemini payload builder only attached `thinkingConfig` when the budget
  was positive, so an "off" request sent nothing and Gemini ran at its
  provider default — measured (2026-06-12 reasoning audit) as thinking ON
  (~200+ thought tokens per trivial classification call) for Gemini 3.5
  Flash and 3.1 Pro. The builder now sends an explicit
  `thinkingConfig: {thinkingBudget: 0}`; tiers that reject 0 trigger a 400
  fallback in `complete()` that retries at Google's stated minimum (128)
  and caches the floor on the client (`_google_thinking_floor`).
- **xAI received no reasoning control at all.** `_build_payload` dropped
  `thinking_budget` for the xai provider; default-reasoning hybrids
  (grok-4.3+) therefore reasoned on every call (audit: 214 reasoning
  tokens on a probe). xai now forwards the request as
  `reasoning_effort: "low"` at budget 0 and `"high"` above; variants that
  400 on the field are handled by the existing fallback, now extended to
  cache the rejection (`_xai_no_reasoning_effort`). Two verified caveats
  (2026-06-13): (a) grok-4.3 accepts `reasoning_effort: "low"` but ignores
  it (206 reasoning tokens with or without); (b) for `*-non-reasoning`
  variants the field is WITHHELD entirely — sending `low` paradoxically
  turns reasoning back ON (0 tokens without the field, 207 with), so the
  builder skips reasoning_effort whenever the model name contains
  "non-reasoning". Net: on xAI, reasoning is controlled by model-variant
  choice, not by parameter. See docs/reasoning-controls.md.
- **Gemini constrained-decoding hang on specific inputs.** A strict
  `responseSchema` can make Gemini reproducibly time out on particular
  inputs (audit: a trivial response timed out 6/6 attempts with the
  schema attached and answered instantly without it). After two
  consecutive timeouts with a schema attached, `complete()` now drops the
  schema once and re-asks; `extract_json()` parses the JSON from the
  free-text reply.

### Added
- **One-time warning for uncontrollable reasoning.** HuggingFace-routed
  models that reason by default and honor no off-switch through the
  router (`openai/gpt-oss-*`, `moonshotai/Kimi-K2*` — the router
  400-rejects `enable_thinking` for their templates) now print a one-time
  warning that the provider's default reasoning behavior applies, instead
  of silently not delivering the requested "off". See
  `docs/reasoning-controls.md`.
- `docs/reasoning-controls.md`: per-provider table of what
  `thinking_budget = 0` actually delivers, with the 2026-06-12 audit
  methodology (reasoning-token usage probes).

## [1.6.7] - 2026-06-11

### Fixed
- **Anthropic `temperature` deprecation (Opus 4.7+).** Newer Anthropic
  models (`claude-opus-4-7`, `claude-opus-4-8`) return
  `400 "temperature is deprecated for this model."` when the
  `temperature` parameter is sent. Pre-fix, every request to these models
  failed and `classify()` produced all-`NA` columns for them. The fix
  mirrors the existing OpenAI reasoning-model handling: a prefix table
  (`_ANTHROPIC_TEMPERATURE_DEPRECATED`) + helper
  (`_anthropic_supports_temperature()`) skips `temperature` up-front in
  `_build_anthropic_payload` for the known-deprecated models, and
  `complete()` strips it on a runtime `400` (caching the decision on the
  client) as a safety net for future families not yet in the table.
  Models that still accept `temperature` (`claude-sonnet-4-6`,
  `claude-opus-4-6`, and earlier) are unaffected.

### Changed
- **`classify()` no longer truncates `input_data` in its output.**
  `build_output_dataframes` previously truncated the `input_data` column
  to a 100-char preview, which silently broke downstream joins against
  gold-standard files and fed truncated text to any pipeline reusing the
  column as input. The classify writer now emits the full
  (whitespace-collapsed) input. `summarize_ensemble` keeps its preview
  truncation intentionally — its inputs can be whole documents/PDF pages.

## [1.6.6] - 2026-06-04

### Added
- **Ollama `think` field forwarding for reasoning-capable models.**
  Cat-stack now consults a per-model-family registry
  (`_OLLAMA_REASONING_MODELS` in `_providers.py`) and injects the
  top-level `think` field on Ollama chat-completion payloads for
  reasoning-capable model families. Ollama standardized the API field
  name across all reasoning models (`think`), but value types differ
  per family — the registry encodes the format and per-budget values
  side-by-side:

  ```python
  _OLLAMA_REASONING_MODELS = (
      ("gpt-oss",      "enum", "low", "high"),  # gpt-oss expects an enum
      ("qwen3",        "bool", False, True),    # most others use boolean
      ("qwq",          "bool", False, True),
      ("deepseek-r1",  "bool", False, True),    # covers -distill-* variants
  )
  ```

  When `thinking_budget=0` is set on `classify()` (the default), Ollama
  gpt-oss receives `"think": "low"` (the model's shortest reasoning
  trace — Ollama doesn't yet support fully disabling gpt-oss
  reasoning); Ollama Qwen3 / QwQ / DeepSeek-R1 receive `"think":
  false`. Models not in the registry (gemma3, llama3.x, llama4.x,
  mistral, phi3/4, etc.) get no `think` field — the safe default. The
  cat-stack patch is harmless for any unlisted model.

  Three places touched, all surgical:
    - `_build_payload` now dispatches Ollama through
      `_build_openai_payload` with `thinking_budget` forwarded (was
      previously dropped at the `else` branch).
    - `_build_openai_payload` injects `payload["think"]` when
      `_ollama_think_value(model, budget)` returns non-None.
    - `text_functions_ensemble.py` adds `"ollama"` to the
      thinking_budget-forwarding provider lists (3 sites).

  The HuggingFace path (`chat_template_kwargs={"enable_thinking":
  False}` for HF-routed Qwen3) is untouched — different mechanism,
  different code path, no conflict.

  Surfaced during the small-tier paper run: Ollama-served gpt-oss:20b
  emits long `<think>` blocks by default that bloat per-row generation
  3-5x. With `think="low"` forwarded, output is shorter, but for our
  16 GB M1 Pro this didn't make the model viable (a separate GPU-memory
  ceiling at Q4_K_M = 14 GB resident exceeds what's available after
  macOS + Python + cat-stack take their share). The patch is still a
  real improvement for users on memory-adequate hardware and for
  Ollama-served Qwen3 / DeepSeek-R1 / QwQ users.

### Notes for maintainers
- When adding a new Ollama reasoning model: append the tuple to
  `_OLLAMA_REASONING_MODELS`. Put more-specific prefixes earlier (e.g.
  `qwen3-coder` would go before `qwen3` if their reasoning toggle
  differs).
- Magistral (Mistral's reasoning model), exaone-deep, and marco-o1 are
  intentionally OMITTED from the registry — they use system-prompt
  injection or chat-template wrappers, not the `think` field. Adding
  them to this registry would silently inject a no-op `think` field.
  See the in-code comment for the canonical list.

## [1.6.5] - 2026-06-04

### Changed
- **HTTP request timeout is now provider-conditional.** Cloud providers
  (OpenAI, Anthropic, Google, Mistral, xAI, HuggingFace, Perplexity)
  continue to use the previous 120 s per-request timeout and 300 s
  cumulative-retry budget — those settings are right for cloud-API
  latencies. **The local Ollama path now gets 600 s per request and
  1200 s cumulative**, which accommodates the long-tail per-row
  latency that emerges on memory-constrained hardware (16 GB Macs
  running 14B+ models can take 2-4+ minutes per row late in a long
  session under thermal/memory pressure). Surfaced during the
  small-tier paper run on 16 GB M1 Pro where qwen3:14b had 10 rows of
  e1b and 143 rows of a19f marked as `processing_status='error'` due
  to spurious 120-second HTTP timeouts even though Ollama was on the
  verge of producing valid output. New module-level constants in
  `_providers.py`:
    - `_REQUEST_TIMEOUT = 120.0` (cloud, unchanged)
    - `_OLLAMA_REQUEST_TIMEOUT = 600.0` (new)
    - `_OLLAMA_MAX_TOTAL_WAIT_SECONDS = 1200.0` (new)

  Plus helper functions `_request_timeout_for(provider)` and
  `_max_total_wait_for(provider)` for downstream call sites.
  `UnifiedLLMClient.complete()` now consults both per call.

  Cloud providers unaffected — same timeouts as before. Pure logging
  / control-flow change, no behavior difference for non-Ollama paths.

  **Power-user override paths** (in addition to the conditional default):
    - Construct `UnifiedLLMClient` directly with explicit
      `request_timeout=<float>` and/or `max_total_wait=<float>` kwargs.
      Per-instance overrides win over the conditional default.
    - Module-level `set_session_timeouts(request_timeout, max_total_wait)`
      registers a process-wide override that all subsequently-constructed
      clients consult. Pass `None` to clear.

  Exposing these as `classify()`-level kwargs would require threading
  through every UnifiedLLMClient construction site (~8 in the ensemble
  code) and is deferred to a follow-up. Validated end-to-end on a
  16 GB M1 Pro: 10 e1b rows that all failed under 1.6.4's 120 s
  timeout (avg 132 sec/row, max 193 sec/row) all succeeded under
  1.6.5's 600 s conditional default.

## [1.6.4] - 2026-06-03

### Changed
- **Rate-limit (429) and server-error (5xx) retry messages now name the
  provider and model.** Previously cat-stack printed bare
  `Rate limited. Waiting 13.3s...` lines, which left users guessing
  which of the 8 providers in an ensemble was throttling them.
  `_providers.py` now prefixes the message with `[provider/model]` —
  e.g. `[anthropic/claude-sonnet-4-6] Rate limited. Waiting 13.3s...`
  or `[huggingface/moonshotai/Kimi-K2.6] Server error 502. Retrying in
  4.2s...`. Surfaced by Phase 1 of the paper port (8-model ensemble run
  where rate-limit pauses appeared without attribution). Pure logging
  change — no behavior difference.

## [1.6.3] - 2026-06-03

### Fixed
- **xAI batch API contract refreshed to match the current `/v1/batches`
  endpoint.** The old implementation in `_batch.py` was written against
  an earlier draft of the xAI batch API and was broken end-to-end on
  the current API: every batch submission returned `422 Unprocessable
  Entity` immediately on create. Five xAI-specific sites in
  `_batch.py` were updated; no other provider's code paths were
  touched.

  Changes (xAI only):
    - `_build_jsonl_line`: each request is now wrapped in xAI's
      tagged-union envelope — `{batch_request_id, batch_request:
      {chat_get_completion: {…payload…}}}` — instead of the
      OpenAI-compat `{custom_id, method, url, body}` shape that the
      old endpoint accepted.
    - `_submit_batch_job`: create body now sends the required `name`
      field (was missing; was sending the no-longer-accepted
      `completion_window`); the create response's ID field is read as
      `batch_id` instead of `id`; the add-requests body wraps the list
      under a `batch_requests` key.
    - `_poll_batch_job`: state is now synthesized from the `state`
      object's counters (`num_pending`, `num_success`, `num_error`,
      `num_cancelled`) rather than a top-level state string — the
      latter no longer exists in the response. Synthesized values
      ("running", "completed", "failed", "cancelled") still fit the
      existing terminal/success-set logic.
    - `_download_batch_results`: results endpoint now returns
      paginated JSON (`{results: [...], pagination_token}`) rather
      than streaming JSONL. The fetcher now walks all pages and
      re-serializes the result objects as JSONL so the existing
      line-by-line parser is unchanged.
    - `_parse_batch_results`: result-line shape changed to
      `{batch_request_id, batch_result: {response:
      {chat_get_completion: {…}}}, error_message?}` — read the new
      field names; `chat_get_completion` is the standard chat
      completion body that `client._parse_response()` already
      handles.

  Surfaced by the same eight-provider batch-mode smoke test from the
  paper port; xAI returned `422` on submission before this fix.

### Documented
- **README caveat: Google's batch scheduler is slow on small jobs.**
  The same smoke test that surfaced the xAI patch also exposed a
  not-a-bug-but-worth-flagging operational reality: Google's batch
  queue often leaves small jobs (a few rows) in `BATCH_STATE_PENDING`
  for 30+ minutes — Google's published batch SLA is up to 24h. There's
  no cat-stack fix; small Google jobs should use `batch_mode=False`
  and reserve batch mode for jobs where the 50% discount matters more
  than wall-clock latency. README's "Features" bullet updated to call
  this out explicitly.

## [1.6.2] - 2026-06-03

### Fixed
- **`chat_template_kwargs={"enable_thinking": False}` now scoped to
  Qwen3-family models only.** The kwarg exists specifically to suppress
  Qwen3's `<think>` blocks via its chat-template `enable_thinking`
  variable; other HF-routed families (Kimi, Llama, Gemma, gpt-oss,
  Mistral) don't expose that variable and never benefited from the
  injection. Strict-validator backends (Fireworks, Groq) reject the
  unknown field with 400 — sending it to a non-Qwen model just
  bought a wasted retry + a stderr warning on every first call. New
  helper `_hf_model_needs_enable_thinking_off()` in `_providers.py`
  gates the injection by model-name prefix (`Qwen/Qwen3` matches
  Qwen3, Qwen3.5, Qwen3.6, …). The runtime 400-fallback added in
  1.6.1 stays as a safety net for unexpected cases — e.g. if a Qwen
  variant lands on a router whose validator doesn't accept the field
  even though the model needs it.

## [1.6.1] - 2026-06-03

### Fixed
- **OpenAI `reasoning_effort` now handled consistently and
  conditionally across GPT-5 sub-families.** Two changes:

  *(a) Per-family off-equivalent mapping.* OpenAI's `reasoning_effort`
  enum is not stable across model generations. The older o-series and
  gpt-5.0–5.3 accept `["minimal", "low", "medium", "high"]`; gpt-5.4+
  deprecated `"minimal"` and switched to
  `["none", "low", "medium", "high", "xhigh"]`. The previous hardcoded
  `"minimal"` for `thinking_budget=0` caused every gpt-5.4* request to
  fail with a 400 `unsupported_value` error and exhaust the retry
  budget. A new module-level constant
  `_OPENAI_REASONING_EFFORT_FLOORS` in `_providers.py` holds the
  per-prefix off value (matched longest-prefix-first), consulted by a
  new helper `_openai_reasoning_effort_floor(model)`. `_build_openai_payload`
  now picks `"none"` for `gpt-5.4`/`gpt-5.5`/`gpt-5.6`, and `"minimal"`
  for `o1`/`o3`/`o4`/`gpt-5.0–5.3`.

  *(b) Runtime fallback to `"low"` on 400 unsupported_value.* For
  model families not yet in the prefix table (future generations,
  third-party hosts that rename the enum), `UnifiedLLMClient.complete()`
  now mirrors the existing `response_format` / `chat_template_kwargs`
  fallback pattern: when a 400 mentions `reasoning_effort` and
  `unsupported`/`invalid`, retry with `"low"` (universally accepted
  across all OpenAI reasoning_effort-supporting models) and cache the
  override on `self._reasoning_effort_override` so subsequent calls
  skip the doomed value. If `"low"` itself is rejected, drop
  `reasoning_effort` entirely.

  Surfaced by a paper-pipeline smoke test against eight providers;
  gpt-5.4-mini was the only closed provider that failed every retry
  before this fix.

- **HuggingFace `chat_template_kwargs` rejection detection broadened
  to cover Fireworks-style 400 wording.** Cat-stack already strips
  `chat_template_kwargs` on routers that don't accept it (originally
  to handle Groq's `"property 'chat_template_kwargs' is unsupported"`
  message). Fireworks (now serving Kimi K2.6 via the HuggingFace
  router) phrases the same rejection as
  `"Extra inputs are not permitted, field: 'chat_template_kwargs'"` —
  no occurrence of `"unsupported"`, so the old heuristic missed it
  and the call exhausted its retry budget. The detection check now
  matches any of `("unsupported", "not permitted", "not allowed",
  "extra inputs", "extra fields", "unknown field")` alongside the
  field name. Surfaced by the same eight-provider smoke test;
  Kimi K2.6 was the only HuggingFace-routed model that failed every
  retry before this fix.

- **PDF inputs are now validated against the `%PDF-` magic-byte header.**
  PyMuPDF is famously permissive: it will happily "open" an HTML file
  saved with `.pdf` extension, render the result as a near-blank page,
  and the downstream VLM will classify the blank image and return
  `processing_status: "success"` with bogus category columns. Users
  who accidentally saved a webpage with `.pdf` on the end got a clean
  DataFrame of garbage classifications with no signal that the input
  was malformed. Added `_is_likely_pdf(path)` which scans the first
  1024 bytes for `b"%PDF-"` (matches how most PDF parsers sniff —
  technically allows leading bytes before the header, e.g. MIME-wrapped
  PDFs). Wired into `_load_pdf_files`:
    - Single bogus file → raises `ValueError` naming the file and
      explaining why we're refusing rather than letting PyMuPDF stumble
      through it.
    - List with a bogus entry → same `ValueError` naming the offender.
    - Directory glob → bogus files are warned-and-skipped per-file so
      the user sees what didn't make it into the run.

## [1.6.0] - 2026-06-03

### Fixed
- **`consensus_threshold="majority"` now uses strict majority — ties
  resolve to "0", not "1".** *(BEHAVIOR CHANGE — listed in
  ecosystem-memory as release-note-worthy.)* The old `positive_rate >=
  0.5` rule meant a 50/50 tie on an even-model ensemble (2-2 of 4,
  3-3 of 6, 1-1 of 2) arbitrarily defaulted to a positive
  classification — biasing every tied row toward false-positive label
  assignment. The fix matches: (a) the linguistic meaning of
  "majority" — *more than half*; (b) sklearn's `VotingClassifier`
  default (`np.argmax` picks the first class on a tie); (c) standard
  ML ensemble-literature treatment (Kittler 1998, Kuncheva 2004,
  Polikar 2006) — ties default to the negative class; (d)
  parliamentary procedure — a motion fails on a tie.

  Numeric thresholds are unchanged: `consensus_threshold=0.5` keeps
  the `>=` semantics — when a user passes a number they get the
  literal interpretation, including the old tie-favors-positive
  behavior if they want it.

  *Caveat for 2-model ensembles:* "majority" + 2 models effectively
  requires unanimous positive agreement (a 1-1 split can never be
  "more than half"). Use 3+ models for a non-degenerate majority
  vote, or `consensus_threshold=0.5` numerically to keep the old
  behavior.

  Discoverability note: the output DataFrame already includes
  `category_N_agreement` columns (fraction of models matching the
  consensus, 0.0-1.0) for multi-model runs — use them to gate
  downstream actions on per-row confidence. Now mentioned in the
  classify() and aggregate_results() docstrings.

  Recommended companion: pair `consensus_threshold="majority"` with
  `embedding_tiebreaker=True` (existing parameter, requires
  `cat-llm[embeddings]`) for even-model ensembles. The tiebreaker
  runs *after* aggregate_results, detects true 50/50 ties exactly,
  builds per-category centroids from unanimously-agreed rows, and
  picks the closer centroid for each tied row. Adds a
  `category_N_resolved_by` audit column (values: `"vote"` or
  `"centroid"`). Not yet supported in `batch_mode`; structured
  meta-LLM "Senate" breakers tracked as task #47.
- **`strip_html_tags` no longer leaks attribute values or misses void
  elements.** The regex implementation had two concrete failure modes:
  (a) `[^>]*` terminated at the first `>` even when inside a quoted
  attribute value, so a tag like `<a href="?q=>foo">label</a>` left
  `foo">label` as visible text — common on real pages with analytics
  URLs and JS templating; (b) the hardcoded void-element list
  (`input/meta/link/img`) missed `br/hr/area/base/col/embed/source/
  track/wbr`. Replaced with a stdlib `html.parser.HTMLParser`
  subclass (`_ReadableTextExtractor`) that tokenizes attribute
  contents correctly and knows the full void-element set. Junk-tag
  content (script/style/nav/header/footer/aside/noscript/iframe/form/
  svg) is skipped by tracking depth; HTML entities are auto-decoded
  via `convert_charrefs=True`. No new dependency — stdlib only. A
  defensive regex fallback handles the unlikely case where the parser
  itself raises. `cat-web` sibling re-exports `strip_html_tags` —
  signature `(str) -> str` unchanged, no import-surface break.
  *Known limitation:* embedded literal `</script>` inside a script
  body still terminates parsing there — this is per HTML spec (even
  browsers do it) and isn't something `html.parser` (or any
  spec-compliant parser) can fix.
- **Text-mode CoVe Step 4 now requests JSON output.** `calls/CoVe.py`'s
  four `chain_of_verification_*` functions (re-exported via
  `cat_stack.calls.__all__` — public surface for anyone who wants to
  invoke text CoVe directly without going through the unified-client
  path) didn't pass any JSON-mode hint on Step 4, the final corrected
  categorization. Downstream `extract_json()` expects JSON-shaped
  output; without the hint, OpenAI / Mistral / Google would freely
  return prose around the JSON object. Added per-provider JSON mode
  to Step 4 matching the pattern already in `calls/pdf_CoVe.py` and
  `calls/image_CoVe.py`: OpenAI and Mistral get
  `response_format={"type": "json_object"}`, Google gets
  `generationConfig.responseMimeType = "application/json"`. Anthropic
  has no native JSON-mode kwarg in its messages API — its Step 4
  stays prompt-based (also matching the existing image_CoVe.py and
  pdf_CoVe.py anthropic variants). Steps 2 and 3 (question generation
  and free-text Q/A) intentionally stay text-mode.
  - *Note:* `cat-stack`'s active text-CoVe path runs through
    `text_functions_ensemble.run_chain_of_verification` (uses
    `UnifiedLLMClient.complete(json_schema=...)` which already gets
    JSON mode for free), so this fix is primarily for external callers
    who import the SDK-shaped functions directly. The kwargs were
    added rather than deleting the file because the symbols are public
    API.
- **`prompt_tune()` no longer returns prompts that didn't beat baseline,
  and the meta-LLM now sees the full attempt history.** Two related
  bugs in `src/catstack/prompt_tune.py`:
  - At L479-480 the function had a fallback `if not best_prompt and
    current_prompt: best_prompt = current_prompt` that promoted the
    latest assembled prompt to "best" whenever no iteration actually
    improved the baseline score. But `current_prompt` contained any
    "no_change" instructions that survived without improving — so the
    function returned a prompt that demonstrably didn't beat baseline,
    contradicting the docstring's "the optimized system prompt (best
    found)" promise. Removed the fallback; when no improvement is
    found, `result["system_prompt"]` is now `""` and the final-summary
    print says *"no improvement found — keep baseline; returned
    system_prompt=''"* (replacing the prior misleading *"no custom
    instruction needed"*). Callers that unconditionally piped the
    result into `classify(system_prompt=...)` are unaffected because
    `system_prompt=""` is the default.
  - At L772 the attempt-history window was capped at the last 3 entries
    to "avoid prompt bloat." With default `tune_iterations=3` the cap
    was a no-op, but users explicitly setting `tune_iterations=5+`
    lost visibility into the first attempts and the meta-LLM
    re-proposed duds it had already tried. Removed the cap — full
    history is now included. Each entry is ~100 chars
    (`"instruction" [outcome]`) so even 20-iteration runs add <2 KB to
    a meta-prompt that already carries multi-KB error-list context.
- **PDF summary synthesis now grounds on actual page text, not the page
  label.** `summarize(input_data=<pdfs>, models=[...multiple...])` runs
  per-model summarization then calls `_synthesize_summaries()` to merge
  the per-model results into a consensus summary. The synthesizer's
  prompt frames the merge as "resolve any contradictions by focusing
  on accuracy" — anchored on an "Original text:" block. For PDF rows
  that block was just the page label (e.g., `"report.pdf p1"`) instead
  of the page's actual extracted text — the synthesizer had nothing to
  check against, so contradictions between per-model summaries got
  resolved arbitrarily. Captured the OCR-extracted page text on the
  result_entry as a new `page_text` field during the summarize loop
  (only when text-mode OCR ran), then read it back at synthesis time
  with `entry.get("page_text") or page_label`. Visual-mode PDFs (no
  OCR) fall back to page_label so synthesis still works — no
  regression. `summarize(batch_mode=True)` is already documented as
  incompatible with PDF input, so the parallel batch path didn't need
  the fix.
- **Google preflight 400 no longer fires on `additionalProperties`.**
  The preflight probe in `classify_ensemble` sends a JSON schema with
  `additionalProperties: false` (valid JSON Schema, helpful for
  catching malformed responses) which Gemini models reject — Google's
  `responseSchema` only accepts a subset of OpenAPI 3.0, excluding
  `additionalProperties`, `oneOf`/`anyOf`/`allOf`/`not`, and
  `$schema`/`$ref`/`definitions`/`patternProperties`. Every Google
  preflight 400'd and the warning "preflight test returned: Server
  error 400 after retries" became a familiar startup noise — and
  meant the actual classification calls (using the same schema) also
  paid the 400-retry budget per row. Added a recursive
  `_sanitize_google_schema()` that strips unsupported keys before the
  payload reaches `responseSchema`. Google still gets the validation
  intent (shape, required fields, types) but in a form its API
  accepts.
- **`parse_kwargs_string` now warns on probable typos.** When a value
  looks like the user was trying to write a Python literal (starts
  with `[ ( { " ' -` or a digit, or equals `True`/`False`/`None`) but
  fails `ast.literal_eval`, the function now emits a UserWarning
  before falling back to the raw string. So `max_retries=three`,
  `tags=[apple,banana]` (unquoted list), or `safety=Truee` (typo) now
  surface as warnings instead of silently degrading to strings that
  break downstream when compared against `int`/`bool` defaults. Plain
  prose values like `research_question=Why did you move?` still fall
  through silently — they aren't trying to be literals. The fallback
  itself is preserved (backwards-compat for sibling wrappers); only
  the silence is fixed.
- **Removed dead `_utils.extract_json` duplicate.** Two versions of
  `extract_json` existed: `_utils.extract_json` (used the older
  `.replace(" ", "")` approach which broke spaces in summary strings)
  and `text_functions.extract_json` (parses + reserializes, preserves
  spaces inside string values). Nothing in `src/` imported the
  `_utils` version; only one test file imported it. Removed the
  duplicate and the test now imports from `text_functions`.
- **`system_prompt` is no longer silently dropped in `batch_mode=True`.**
  `classify(system_prompt="...", batch_mode=True)` and the sync version
  diverged: sync forwarded the system_prompt through to
  `build_text_classification_prompt`, but the batch path's
  `prompt_params` dict (both the single-model and ensemble variants in
  classify.py) didn't include `system_prompt` as a key, and the two
  consumers in _batch.py (`_run_one_batch_job`, `_run_one_sync_model`)
  didn't read it back out. The user's custom system instruction —
  including the output of `prompt_tune()` — silently disappeared the
  moment they switched to batch mode. Added the key in both producers
  and the kwarg in both consumers. Sync fallback path (used for HF /
  Perplexity / Ollama within `batch_mode=True` ensembles) also
  threaded so mixed-provider ensembles see consistent prompts.
  *Known remaining gap (out of scope for this fix):*
  `prompt_params["stepback_insights"]` is still hardcoded to `{}` in
  the batch producers — step-back prompting computed in sync mode
  isn't reproduced in batch mode. Will track separately if it
  matters for users.
- **Image directory loading is now case-insensitive.** `glob.glob('*.jpg')`
  is case-sensitive on every platform — directories with mixed-case
  extensions (e.g., `IMG_001.JPG` from many phone cameras, or
  `Photo.Jpeg`) silently dropped files. Replaced the three duplicate
  inline `glob.glob` loops (in `_load_image_files`, `image_features`,
  and `image_score_drawing`) with a single shared `_load_image_files`
  implementation backed by `pathlib.Path.iterdir()` + `suffix.lower()`
  matching. The consolidation also fixes a separate bug in the
  inline loops: passing a single image path (not a directory) to
  `image_features` or `image_score_drawing` silently returned an
  empty list, because the inline glob only handled the directory case.
  Both functions now correctly handle list/single-file/directory inputs.
- **Large images now warn before the b64+API round-trip.** `_encode_image`
  prints a one-time warning (per path) when a file exceeds 20 MB —
  most provider limits cluster between 5–20 MB (Anthropic 5, Google 7,
  OpenAI 20) and a 50 MB upload silently fails at API time after a
  noticeable encoding delay. Just a heads-up — encoding still proceeds.
- **A single chunk's exception no longer kills parallel category
  exploration.** `explore_common_categories(..., max_workers>1)` used
  `pass_idx, div_idx, reply, error = future.result()` with no try/except
  and a worker (`_call_chunk`) that didn't catch exceptions itself. A
  transient network glitch in one chunk (anything outside
  `client.complete()`'s retry budget — DNS failure, TLS reset, etc.)
  would re-raise from `future.result()` and abort the entire parallel
  loop, losing every other chunk's work that had already completed.
  Wrapped the boundary in try/except: chunk exceptions are logged to
  stderr with the same `[CatStack] Warning: chunk div=X pass=Y ...`
  format the existing in-band error path uses, and the loop continues.
  Audited the other five `future.result()` sites in the codebase
  (text_functions_ensemble.py x3, _batch.py x2 already fixed above);
  the three in text_functions_ensemble.py wrap `classify_single` /
  `summarize_single_item` which already catch all exceptions internally,
  so no isolation issue there.
- **One model's batch failure no longer kills the entire ensemble run.**
  `run_batch_ensemble_classify` and `run_batch_ensemble_summarize` both
  iterated `for future in as_completed(futures): _, result =
  future.result()` with no try/except. Any exception from one model's
  batch pipeline (BatchJobFailedError, BatchJobExpiredError, TimeoutError,
  RuntimeError on missing output_file_id, RequestException on the
  submission HTTP call, etc.) propagated out of the loop and aborted
  the entire ensemble — losing every other model's results, including
  ones that had already completed successfully. Wrapped each
  `future.result()` in try/except: failures are logged with the model
  name and exception type, the failed model's results dict is set to
  `{}`, and the loop continues. Downstream `.get(idx, (None, "Missing
  from batch results"))` cleanly handles the empty dict — every row
  for the failed model gets "Missing from batch results" and the
  DataFrame returns with that model's column empty rather than
  raising. The other models' columns are populated normally.
- **Anthropic batches that end with all requests errored now surface as
  failures instead of silent empty results.** Anthropic's batch API
  uses a single terminal `processing_status` ("ended") for every
  outcome — fully succeeded, fully errored, fully canceled, fully
  expired, or any mix. The polling code previously treated "ended" as
  unconditional success and returned status_data; per-request errors
  got surfaced at parse time as `(None, error_msg)` per row. That
  works for the mixed case but is misleading when 0/N requests
  succeeded: the caller saw a DataFrame of all-None values for that
  model with no clear log signal that the entire batch was dead.
  Added `_inspect_anthropic_terminal_state(status_data, job_id)` to
  inspect `request_counts.succeeded / errored / canceled / expired`
  when state reaches "ended": (a) if 0 succeeded and all canceled
  → `BatchJobExpiredError`; (b) if 0 succeeded and all expired
  → `BatchJobExpiredError`; (c) if 0 succeeded otherwise
  → `BatchJobFailedError` with the full breakdown in the message;
  (d) if partial (some succeeded + some failed) → print warning and
  continue (parse layer still surfaces per-row errors). Combined with
  the failure-isolation fix above, an all-errored Anthropic batch
  becomes a clean per-model failure in an ensemble rather than an
  ensemble-wide abort. Verified live: a 2-item Anthropic batch with
  valid payload completes normally (state=ended, succeeded=2,
  errored=0) — the inspection helper returns silently on the
  all-success path, no behavior change for healthy batches.

### Changed
- **`description=` is now the canonical, content-neutral parameter for
  data context across all public entry points.** `survey_question=` was a
  cat-survey-era artifact bleeding into a domain-agnostic API; calling
  e.g. `extract(survey_question="...")` looked wrong for non-survey
  corpora (academic papers, social posts, support tickets). `classify`,
  `extract`, and `prompt_tune` now treat `description` as the primary
  parameter and emit a `DeprecationWarning` when `survey_question=` is
  passed; the value is mirrored into `description` (and still threaded
  through to the lower-level `survey_question` kwarg internally, so the
  existing prompt assembly is unchanged). `explore` and `summarize`
  already used `description` — their docstrings just got the
  "content-neutral" framing cleanup. cat-survey and pre-rename notebooks
  keep working unchanged but will surface a one-line warning per call.
  8 unit tests cover the warning trigger, the mirror behavior, and the
  "description wins when both set" tie-break.

- **`chat_template_kwargs` no longer breaks classify() on Groq-routed HF
  models.** `_build_openai_payload` injects
  `chat_template_kwargs={"enable_thinking": False}` whenever
  `thinking_budget=0` (the classify() default) and the provider is
  `huggingface` — useful for stopping Qwen3-family models from emitting
  `<think>` tags, but the Groq router (which sits behind HF Inference
  Providers for Llama-3.x and `openai/gpt-oss-*`) rejects the property
  outright:
  `"property 'chat_template_kwargs' is unsupported"` (HTTP 400). Pre-fix,
  every row burned all retries on this deterministic 400 and the
  resulting DataFrame was full of `processing_status: 'error'`. A live
  stress sweep documented this hitting Llama-3.3-70B-Instruct and
  openai/gpt-oss-120b (and cascading into the parallel, many-categories,
  and unicode-category-name scenarios — same root cause).
  Fix mirrors the existing `response_format` strip-and-retry pattern in
  the 400 handler: detect `"chat_template_kwargs"` + `"unsupported"` in
  the body, pop the kwarg from the payload, retry immediately, and cache
  the decision on the client (`_warned_no_chat_template_kwargs`) so
  subsequent rows skip the doomed payload from the start. Models on
  routers that *do* honor the kwarg (GLM-4.5, DeepSeek-R1 via the
  generic HF Inference Providers dispatcher) are unaffected. 4 new
  unit tests + a live smoke test against the real Groq router confirm
  the fix; the live test now classifies 3 rows in 2s where pre-fix all 3
  rows failed.

- **`pdf_multi_class` / `explore_pdf_categories` now route
  `model_source="huggingface-together"` correctly.** The upstream
  validation list (`explore_pdf_categories`) and the OpenAI-compatible
  call helpers both accepted `"huggingface-together"`, but
  `_process_single_page`'s dispatch was hard-coded to
  `["openai", "perplexity", "huggingface", "xai"]` in both the text-only
  and image/PDF branches — so an HF-Together call passed every upstream
  check, then crashed inside dispatch with
  `ValueError("Unknown source! Choose from...")`. Added
  `"huggingface-together"` to both dispatch lists. Verified live: a
  text-mode call against `meta-llama/Llama-3.3-70B-Instruct-Turbo` via
  `router.huggingface.co/together/v1` now classifies successfully end-to-end.
- **Google PDF calls no longer block indefinitely on a hung gateway.**
  `_call_google` and `_call_google_text_only` each contained their own
  identical `make_google_request` closure, and both `requests.post(...)`
  calls were missing `timeout=` — the only two POSTs in `pdf_functions.py`
  without one (every other POST in the file uses `timeout=120`). A stalled
  connection would block the worker forever. Extracted the duplicate
  closure to module-level `_google_post_with_retry(...)` with
  `timeout=120` baked in; both call sites now alias it
  (`make_google_request = _google_post_with_retry`) so all existing
  callers — including `pdf_chain_of_verification_google`, which receives
  the function as a kwarg — keep working unchanged.

- **`detect_provider` and `_detect_model_source` no longer disagree on the
  same model string.** Pre-fix, the two near-duplicate auto-detection
  functions used divergent substring rules (`_detect_model_source` was
  missing the `o1`/`o3` reasoning-model patterns); empirical probe showed
  5/20 test strings disagreed. Worst case: a user calling
  `classify(model="o3-mini", provider="auto")` crashed through one entry
  point with `ValueError` and worked through another. Bare-substring
  matching also misrouted real inputs (`qwen-o3-coder` → openai because
  `"o3"` matched before `"qwen"`, routing the user's HF API key to
  OpenAI's endpoint). Consolidated to a single token-based matcher: the
  model name is split on `-` / `_` / `.`, an explicit set of OpenAI
  o-series tokens (`o1`…`o9`) is checked only as the *first* token, then
  family-prefix matches walk the tokens in order. `_detect_model_source`
  is now a thin shim over `detect_provider` so both paths route
  identically. `org/model` format routes to HuggingFace unconditionally
  (catches HF-hosted Mistral/Qwen/Llama models that previously misrouted
  to mistral.ai). `name:tag` syntax (no slash) now raises with a
  helpful error — auto-detection is intentionally disabled for Ollama
  because the failure mode (connection refused on :11434) is confusing
  for users who meant a hosted model.

- **Models that 5xx persistently on `response_format` payloads now recover
  automatically.** Some endpoints (notably HF's router for the small
  Llama-3.2-1B variant) reliably return 502 Bad Gateway with an HTML
  error body when the request body includes
  `response_format: {"type": "json_object"}`, even though the response
  body never mentions the parameter. The existing 400-handler couldn't
  trigger because the keyword check requires `"structured"`,
  `"response_format"`, or `"json_object"` in the response body — HTML
  matches none of those. `complete()` now strips `response_format` once
  per call on a 5xx that has NO `Retry-After` header (Retry-After is a
  signal for transient overload, not a payload complaint), retries
  immediately, and caches the decision on the client instance so
  subsequent rows in the same run skip `response_format` from the start.
  After the strip, prose-mode output is parsed by `extract_json`; if
  that fails, the auto-formatter consent prompt fires for cleanup.
  Verified live: Llama-3.2-1B + catstack's actual classification payload
  went from 0 % success to producing valid classifications.
- **JSON-formatter activates on demand for users who didn't opt in.** When
  `json_formatter` isn't explicitly set (the new default), the first
  malformed-JSON row of a run now triggers an interactive consent prompt
  instead of silently producing broken output. Two paths depending on
  whether the formatter dependencies are installed:
  - **Deps installed:** *"Use the formatter for this run? (Y/n)"* —
    one-time ~1 GB RAM load.
  - **Deps missing:** *"Download deps (~1.5 GB) and use the formatter?
    (Y/n)"* — install only proceeds after the explicit yes.
  Non-TTY contexts (CI, batch scripts, headless notebooks) decline
  silently and print a one-time suggestion instead. The decision is
  cached on `formatter_state` so a 50-row run with one malformed early
  row doesn't re-prompt 49 times. Backward-compat: `json_formatter=True`
  keeps its existing eager-load + auto-install behavior (the user has
  already implicitly consented by passing `True`), and
  `json_formatter=False` opts out absolutely.
- **`_ensure_dependencies` no longer silently pip-installs ~1.5 GB of
  transformer deps.** This was an original review finding (L19): the
  auto-install ran without asking, surprising Stata / Rscript users with
  a multi-minute hang. The auto-install path now requires either explicit
  `json_formatter=True` (implicit consent) or a `(Y/n)` answer through
  the new `_prompt_formatter_consent` flow.
- **JSON-formatter fallback path is now thread-safe.** When
  `classify(json_formatter=True, ...)` runs under a `ThreadPoolExecutor`,
  the per-row `_try_formatter_fallback` helper could fire two distinct
  races: (1) multiple workers all seeing `formatter_state["_loaded"] ==
  False` and each independently invoking the ~10 s, ~1 GB
  `_loader()`; (2) concurrent `model.generate()` calls against the same
  HuggingFace transformer model — which maintains internal KV-cache
  state and is not thread-safe — could silently corrupt outputs. Added a
  `threading.Lock` to `formatter_state["_lock"]` (pre-initialized in
  `classify.py` where the dict is constructed; defensively
  `setdefault`-ed inside the helper for robustness). The locked region
  wraps both the lazy-load and the `run_formatter()` call; the
  fast-path "JSON already parsed cleanly" check happens before
  acquiring the lock, so the common case has zero overhead.
- **`UnifiedLLMClient.complete()` retry logic hardened.** Four related
  problems addressed in one pass:
  - **`Retry-After` headers are now honored** on both 429 and 5xx
    responses. Accepts integer-seconds and HTTP-date forms (RFC 7231).
    Provider's explicit retry hint takes precedence over our exponential
    schedule.
  - **Backoff is jittered** (full jitter, sample uniformly from
    `[0.5 × base, 1.5 × base]`). Concurrent ensemble workers that all hit
    a 429 at the same instant no longer wake up on the same tick. Same
    treatment for 5xx, `requests.exceptions.Timeout`, and the catch-all
    `RequestException`.
  - **Hard cap on cumulative wait time per call.** If the next planned
    sleep would push total elapsed time past `_MAX_TOTAL_WAIT_SECONDS`
    (300s, hardcoded), the loop returns the error instead of sleeping.
    Pre-fix worst case: 5 retries × 5× multiplier on 429s could block a
    single call for ~310 s.
  - **`_call_claude_cli` now catches `OSError` outside the retry loop.**
    A multi-MB prompt that overflows the OS's `ARG_MAX` raises
    `OSError [Errno 7] Argument list too long` from `subprocess.run`.
    Pre-fix, that error bubbled up out of `complete()`, breaking the
    `(text, error)` contract and crashing the caller. Now returns
    `(None, "Claude CLI subprocess failed: ...")` after a single
    attempt — E2BIG is deterministic for this prompt size, retrying is
    pointless.

### Changed
- **`provider="local"` is a recognized alias for `provider="ollama"`.**
  Normalized at the `detect_provider` and `UnifiedLLMClient.__init__`
  boundaries. Friendlier wording for users running local Ollama
  inference who don't think of it as "the Ollama provider"; `"ollama"`
  still works for back-compat.

---

## [1.5.0] - 2026-06-02

### Fixed
- **`classify(categories="auto")` no longer raises `ModuleNotFoundError`.** The
  lazy import inside `classify_ensemble` referenced a non-existent `.main`
  submodule; corrected to `.extract`. Every prior call with
  `categories="auto"` failed at import time.
- **`summarize(step_back_prompt=True)` no longer raises `TypeError`.**
  `gather_stepback_insights` was being called with `context=` / `question=`
  kwargs it didn't accept. Refactored the helper to take a prepared
  `stepback_prompt` string; `classify_ensemble` and `summarize_ensemble` now
  build their own templates at the call site. Removes the survey-specific
  vocabulary from the shared helper.
- **Google PDF/image summaries no longer silently return empty strings.**
  `_call_google_multimodal` was nested inside `classify_ensemble`'s closure
  and unreachable from `summarize_ensemble` — the `NameError` was swallowed
  by a bare `except Exception`, producing `{"summary": ""}` for every Google
  multimodal summary. Promoted the helper to module level and fixed the
  tuple unpacking at both summarize call sites.
- **`_extract_json_for_summary` no longer raises `NameError` on a missing
  `regex` import.** Replaced the `regex.findall(r'\{(?:[^{}]|(?R))*\}', ...)`
  pattern across all 6 call sites with a stdlib `_extract_balanced_json`
  helper. The helper is string-aware: inputs like
  `{"summary": "see Fig {3}"}` are now preserved correctly (the `regex`
  pattern silently truncated at the first `}` inside a string value).
- **Batch ensembles with HuggingFace / Perplexity / Ollama no longer silently
  produce schema errors from those models' columns.** The sync-fallback path
  (`_run_one_sync_model` in `_batch.py`) was treating `client.complete()` as
  returning a single value, but it returns a `(text, error)` tuple. The
  result: `extract_json` was being called on the tuple and always returned
  the error sentinel `{"1":"e"}` — for both successful and failing API
  calls. Fixed the unpacking and added an explicit error branch matching
  the parallel summarize sync-fallback at line 1187.
- **`summarize_ensemble` no longer relies on Python late binding to find
  `is_image_mode`.** The flag was assigned ~300 lines after the
  `summarize_single_item` closure that uses it. Production worked only
  because the closure was invoked after the assignment, but a future
  refactor moving the closure invocation earlier would have raised
  `NameError`. Hoisted the assignment next to `is_pdf_mode` at the top of
  the function and added `is_image_mode = False` to the DOCX-to-text
  conversion branch so both flags travel together.

### Changed
- **`ARCHITECTURE.md` updated to describe the `calls/` layer accurately.**
  Previous entries claimed `calls/*` imports from "*nothing* (leaf
  modules)" and labelled the diagram as "leaf modules, no intra-pkg
  deps". The reality is that every leaf under `calls/` (stepback, CoVe,
  top_n, plus the image_* and pdf_* variants) makes `requests.post`
  calls directly rather than routing through `UnifiedLLMClient.complete()`
  — by design, because each strategy is a shape (multi-step CoVe
  pipeline with per-step JSON-mode toggles, per-question loop, per-provider
  prompt tail) that doesn't fit `complete()`'s single-shot API. Four of
  the leaves (`stepback`, `top_n`, `image_stepback`, `pdf_stepback`) also
  do a *lazy* intra-package import of `_detect_huggingface_endpoint`
  when `model_source == "huggingface"`. Documented both facts. Also
  corrected the call-chain trace where `extract()` was sourced from
  `main.py` — the function lives in `extract.py` (`main.py` has never
  existed; this was the C1 documentation twin).
- **`catstack.fetch_url_text` is now SSRF-safe and no longer papers over
  TLS errors.** Four hardening changes in `_web_fetch.py`, all
  stdlib-only:
  - **SSRF guard.** The URL's hostname is resolved via
    `socket.getaddrinfo` before any HTTP request and rejected if any
    returned address is private, loopback, link-local, reserved,
    multicast, or unspecified. Catches AWS metadata
    (`169.254.169.254`), `localhost`, RFC1918, IPv6 loopback (`::1`),
    and the GCP metadata host pre-network. Does not defend against DNS
    rebinding — out of scope for a stdlib guard.
  - **Scheme allowlist.** `urllib.parse.urlsplit`-based parsing
    replaces the unanchored `re.match` regex; only `http` and `https`
    schemes are accepted. `file://`, `data:`, `javascript:`, `ftp://`
    are all rejected.
  - **No silent TLS bypass.** The `except SSLError: retry with
    verify=False` fallback is gone. TLS errors surface to the caller
    as ordinary fetch errors so MITM-detectable conditions stay
    visible.
  - **Streaming + byte cap.** Responses are read via
    `iter_content(chunk_size=8192)` and capped at
    `5 × _MAX_CONTENT_CHARS` bytes (~250 KB). Replaces the
    full-buffer-then-truncate pattern that would OOM on a multi-GB
    URL. `is_url` itself also rejects strings with embedded
    `\\r`/`\\n`/`\\x00`, closing a CRLF-injection sliver.
- **`pdf_multi_class(chain_of_verification=True, ...)` no longer crashes
  with `TypeError` on OpenAI / Mistral / Perplexity / HuggingFace / xAI.**
  The migration from SDK clients to direct HTTP in `calls/pdf_CoVe.py`
  was completed for the `anthropic` and `google` variants but abandoned
  half-finished for `_openai` and `_mistral`: their signatures didn't
  accept the `api_key` (and `base_url` for openai) kwargs that
  `pdf_functions.py` passes, and their bodies still called
  `client.chat.completions.create()` on a `client=None`. Finished the
  migration to match the anthropic precedent — both functions now
  accept `api_key` (+ `base_url` for openai), use `requests.post(...)`
  via an internal helper, preserve the Step-4 `response_format=
  {"type": "json_object"}` for JSON-mode output, and fall back to
  returning the initial reply on any error. The deprecated `client`
  parameter is kept on both signatures for backward compatibility with
  existing callers that pass `client=None`.
- **`image_score_drawing` and `image_features` no longer crash with
  `UnboundLocalError` on the first iteration when a provider call fails
  pre-success.** Each per-image loop now initializes `reply = None` at
  the top of the iteration. The pre-fix code only assigned `reply` on
  the success path of each provider's `try` block; any non-success
  branch (401, 403, generic network error, Anthropic's empty-content
  response, …) left `reply` unbound, and the post-dispatch
  `if reply is not None:` check then raised `UnboundLocalError`,
  crashing the entire call. Worse for multi-image batches: on a
  later-iteration failure, `reply` still held the *previous* successful
  iteration's response, silently attaching the wrong JSON to the
  failing row. Verified live: a bad API key now produces a row whose
  `model_response` contains the captured 401 and `json` is the
  `{"1":"e"}` sentinel, instead of crashing the call.
- **Anthropic image dispatch now derives `media_type` from the file
  extension instead of hardcoding `image/png` / `image/jpeg`.** Anthropic
  validates the declared `media_type` against the actual image bytes
  and returns HTTP 400 `invalid_request_error` ("The image was specified
  using the image/jpeg media type, but the image appears to be a
  image/png image") on mismatch. The three affected sites
  (`image_score_drawing` reference + user image, `image_features` user
  image) now use the same `f"image/{ext}" if ext else "image/jpeg"`
  pattern as the newer `image_multi_class` paths, with `ext` coming from
  `_encode_image`'s normalized return (`jpg` → `jpeg`, lowercase).
- **`image_score_drawing` and `image_features` now route base64 encoding
  through the shared `_encode_image` helper.** Previously, both functions
  reinvented inline base64 encoding without the helper's `jpg`→`jpeg`
  normalization, lowercase-extension handling, or extensionless-path
  guard. `image_score_drawing` additionally had an unconditional
  duplicate data-URI wrap that rewrapped error strings like
  `"Error: [Errno 2] No such file or directory: 'x.png'"` into fake
  `data:image/...;base64,Error:…` URIs and shipped them to the provider.
  Both functions now use `_encode_image` (returning
  `(encoded, ext, is_valid)`), skip invalid inputs via `continue` with
  the `{"no_valid_image": 1}` sentinel, and write the data URI from the
  normalized extension exactly once. The reference image in
  `image_score_drawing` now raises `FileNotFoundError` eagerly when the
  path can't be read, instead of silently producing a broken URI.
- **`UnifiedLLMClient.__init__` no longer probes HuggingFace endpoints
  eagerly.** Previously, *every* HuggingFace client construction issued
  two probe POSTs (with the user's API key) to `router.huggingface.co/v1`
  and `…/together/v1` to "detect" the right endpoint — and then discarded
  the result, so the detection was a no-op. Replaced with lazy
  resolution: explicit `:router` suffixes (`:novita`, `:together`,
  `:sambanova`, `:cerebras`, `:fireworks`) are honoured immediately
  without probing; for un-suffixed models, the client uses the configured
  generic endpoint and only falls back to probing alternative routers on
  the actual 400 "Model not supported by provider …" response. The
  fallback probes all five known specific routers plus the generic one,
  caches the first working endpoint on the instance, and retries —
  subsequent calls reuse the cached endpoint with no further probes.
  Thread-safe via a per-instance `threading.Lock`. The probe API now
  takes an optional `skip` set so callers can exclude endpoints that
  already failed; legacy callers (in `image_functions` / `pdf_functions`)
  preserve their two-endpoint probe behavior when `skip` is omitted.

### Removed
- **`regex` runtime dependency.** No longer needed after the JSON extraction
  refactor; all six former call sites now use stdlib via
  `_extract_balanced_json`.
- **`catstack/calls/all_calls.py`** (621 lines). The module duplicated every
  function already living in the per-strategy leaves (`stepback.py`,
  `CoVe.py`, `top_n.py`) and shipped two runtime-broken CoVe variants —
  `chain_of_verification_anthropic` (undefined `properties`, missing
  `import json`) and `chain_of_verification_google` (undefined
  `thinking_budget`). Nothing in `src/` and no sibling package imported
  it. `calls/__init__.py` now re-exports the same eight public names
  (`get_stepback_insight_*`, `chain_of_verification_*`) from the working
  leaf modules — same signatures, same `__all__`, so any external caller
  doing `from catstack.calls import chain_of_verification_anthropic`
  silently gets the working version instead of the broken duplicate.

### Added
- `tests/test_classify_auto_categories.py`, `tests/test_stepback_insights.py`,
  `tests/test_summarize_google_multimodal.py`, `tests/test_extract_balanced_json.py`
  — 28 new mocked tests covering each of the four fixes above. No network.

---

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
