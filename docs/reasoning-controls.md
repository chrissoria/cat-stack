# Reasoning controls: what `thinking_budget = 0` actually delivers

`classify(thinking_budget = 0)` requests reasoning off on every call, but
what that request *delivers* depends on the provider and model. This table
reflects cat-stack **v1.6.8** behavior (older versions delivered less — see
the CHANGELOG and the 2026-06-12 audit that motivated the fixes).

| Provider / model family | Control sent at budget 0 | Effective state |
|---|---|---|
| OpenAI GPT-5.4+ | `reasoning_effort: "none"` | Off |
| OpenAI o-series, GPT-5.0–5.3 | `reasoning_effort: "minimal"` (400-fallback to `"low"`) | Minimized |
| Anthropic | thinking omitted + forced tool call (API rejects forced tools with thinking on) | Off |
| Google Gemini | `thinkingConfig: {thinkingBudget: 0}`; tiers rejecting 0 fall back to the 128 minimum (cached) | Off (verified: Flash thoughts=0), or floor-minimized on thinking-only tiers (verified: 3.1 Pro thoughts=128 vs 243 uncontrolled) |
| xAI grok hybrid (e.g. grok-4.3) | `reasoning_effort: "low"` sent; **measured ineffective** (206 reasoning tokens with `low`, 206 default) | **Provider default (ON)** — no working off-switch on hybrids |
| xAI grok `*-non-reasoning` variant | **nothing sent** — reasoning_effort is withheld because the field paradoxically turns reasoning ON for these variants (verified: 0 tokens with no field, 207 with `low`) | Off (by variant) |
| HF-routed Qwen3 family | `chat_template_kwargs: {enable_thinking: false}` | Off (verified: 0 vs ~390 reasoning tokens) |
| HF-routed gpt-oss, Kimi K2 | **none deliverable** — the router 400-rejects `enable_thinking` for these templates and exposes no effort field | **Provider default (reasoning ON)** — one-time warning printed |
| Ollama reasoning-capable models | top-level `think` field (enum for gpt-oss, boolean for qwen3/deepseek-r1) | Off/low |
| Non-reasoning models (Gemma, Mistral Large, …) | n/a | n/a |

## Why this table exists

A 2026-06-12 audit of a 18-model benchmark run found that pre-1.6.8
versions sent *no* control to Google (the off request was a silent no-op;
Gemini's default emitted ~200+ thought tokens per trivial call) and none to
xAI (default-reasoning grok-4.3 emitted 214). Benchmarks built on those
versions compared reasoning-off models against provider-default-reasoning
models without knowing it.

## Verifying for yourself

Reasoning states are *measurable* from usage metadata — don't trust the
parameter, probe it:

- OpenAI / xAI / HF router: `usage.completion_tokens_details.reasoning_tokens`
  (or a `reasoning_content` field on the message).
- Google: `usageMetadata.thoughtsTokenCount`.
- Anthropic: thinking blocks in `content`; additionally, forced tool_choice
  is rejected outright when thinking is enabled.

Send one trivial classification request per model and read the counters.
Provider defaults are serving-side and can change without notice; re-probe
when it matters.
