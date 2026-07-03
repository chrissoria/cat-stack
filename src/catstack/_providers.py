"""
Unified LLM provider infrastructure for CatLLM.

This module provides a unified HTTP-based approach for calling multiple LLM providers
(OpenAI, Anthropic, Google, Mistral, xAI, Perplexity, HuggingFace, and Ollama)
without requiring provider-specific SDKs.
"""

import json
import random
import threading
import time
import requests

# Hard cap on total accumulated retry wait per complete() call. Prevents a
# string of transient errors from blocking a single request indefinitely
# (the bare exponential schedule could otherwise sit on a request for
# 5+ minutes). Tuned to "long enough to outlast a real provider blip,
# short enough that batch ensembles don't stall for half an hour."
_MAX_TOTAL_WAIT_SECONDS = 300.0

# Per-HTTP-request timeout, in seconds. For cloud providers (OpenAI,
# Anthropic, Google, …) inference is usually 1-10 seconds, so 120 s is
# a generous ceiling that catches genuine hangs.
#
# Local Ollama is a different regime: on memory-constrained hardware
# (e.g., 16 GB M1 Pro running a 14 B-class model), individual rows can
# take 2-4+ minutes under thermal/memory pressure. cat-stack 1.6.4
# logged frequent spurious "Request timeout" failures in those
# conditions even when Ollama was about to produce valid output.
# `_OLLAMA_REQUEST_TIMEOUT` and `_OLLAMA_MAX_TOTAL_WAIT_SECONDS` give
# the Ollama path a much longer window. Surfaced during the small-tier
# paper run, 2026-06-04.
_REQUEST_TIMEOUT = 120.0          # cloud providers
_OLLAMA_REQUEST_TIMEOUT = 600.0   # local Ollama — 5x cloud, accommodates slow-row tails
_OLLAMA_MAX_TOTAL_WAIT_SECONDS = 1200.0  # 4x cloud, since per-call timeout is also 5x


# Session-level user override. Set non-None at the start of a `classify()`
# call to override the conditional defaults for ALL UnifiedLLMClient
# instances constructed during that call without per-site arg threading.
# Single-process scope; safe under cat-stack's intra-call parallelism
# (per-call sets/resets bracket all workers).
_session_request_timeout: float = None
_session_max_total_wait: float = None


def set_session_timeouts(request_timeout: float = None, max_total_wait: float = None):
    """Set the session-level HTTP-timeout overrides. Pass None to clear."""
    global _session_request_timeout, _session_max_total_wait
    _session_request_timeout = request_timeout
    _session_max_total_wait = max_total_wait


def _request_timeout_for(provider: str) -> float:
    """Per-request HTTP timeout. Session override wins over provider default."""
    if _session_request_timeout is not None:
        return _session_request_timeout
    return _OLLAMA_REQUEST_TIMEOUT if provider == "ollama" else _REQUEST_TIMEOUT


def _max_total_wait_for(provider: str) -> float:
    """Per-call cumulative-wait cap. Session override wins."""
    if _session_max_total_wait is not None:
        return _session_max_total_wait
    return _OLLAMA_MAX_TOTAL_WAIT_SECONDS if provider == "ollama" else _MAX_TOTAL_WAIT_SECONDS


# ---------------------------------------------------------------------------
# OpenAI reasoning_effort: per-model-family off-equivalent value.
# ---------------------------------------------------------------------------
#
# Different OpenAI model generations expose different `reasoning_effort`
# enum values. The "off" value (what `thinking_budget=0` maps to) is not
# stable across families:
#
#   o1 / o3 / o4, gpt-5.0..gpt-5.3   → "minimal" (older floor)
#   gpt-5.4 / gpt-5.5 / gpt-5.6      → "none"    (new strict-off; "minimal" deprecated)
#
# A model sent the wrong floor returns a 400 `unsupported_value`. The
# table below is consulted in `_openai_reasoning_effort_floor()` to pick
# the right value up-front. For unknown future families,
# `UnifiedLLMClient.complete()` catches the 400 and falls back to "low"
# (universally accepted across all reasoning_effort-supporting models).
#
# Entries are matched longest-prefix-first so "gpt-5.4" matches before
# "gpt-5" — keep that invariant when extending.
_OPENAI_REASONING_EFFORT_FLOORS = (
    ("gpt-5.4", "none"),
    ("gpt-5.5", "none"),
    ("gpt-5.6", "none"),
    ("gpt-5",   "minimal"),  # covers 5.0, 5.1, 5.2, 5.3
    ("o1",      "minimal"),
    ("o3",      "minimal"),
    ("o4",      "minimal"),
)


def _openai_reasoning_effort_floor(model: str) -> str:
    """Return the off-equivalent reasoning_effort value for a reasoning-
    capable OpenAI model, based on its name prefix. Defaults to "minimal"
    for models not covered by the table — the safest historical value."""
    for prefix, floor in _OPENAI_REASONING_EFFORT_FLOORS:
        if model.startswith(prefix):
            return floor
    return "minimal"


# ---------------------------------------------------------------------------
# HuggingFace `chat_template_kwargs={"enable_thinking": False}` is the knob
# to suppress Qwen3-family `<think>` blocks. Other model families don't
# expose an `enable_thinking` template variable, and strict HF backends
# (Fireworks, Groq) reject the unknown field with 400 — forcing a wasted
# retry. Restrict injection to families that actually honor the flag.
#
# The runtime fallback in `complete()` (strip-on-400) stays as a safety
# net for unexpected cases — e.g. if a Qwen variant lands on a router
# whose validator doesn't accept the field.
# ---------------------------------------------------------------------------
_HF_NEEDS_ENABLE_THINKING_OFF = (
    "Qwen/Qwen3",   # covers Qwen3, Qwen3.5, Qwen3.6, …
)


def _hf_model_needs_enable_thinking_off(model: str) -> bool:
    return any(model.startswith(p) for p in _HF_NEEDS_ENABLE_THINKING_OFF)


# Router-served models measured (2026-06-12 reasoning audit) to reason by
# default with NO honored off-switch through the OpenAI-compatible router:
# the router 400-rejects `chat_template_kwargs.enable_thinking` for their
# templates, and they expose no reasoning_effort. classify() warns once per
# client so users know the provider default applies.
_HF_DEFAULT_REASONING_PREFIXES = (
    "openai/gpt-oss",
    "moonshotai/kimi-k2",
)


def _hf_model_reasons_by_default(model: str) -> bool:
    m = (model or "").lower()
    return any(m.startswith(p) for p in _HF_DEFAULT_REASONING_PREFIXES)


# Module-level: models already warned about uncontrolled reasoning, so the
# warning fires once per process even though a fresh client is built per row.
_WARNED_UNCONTROLLED_REASONING: set = set()


# ---------------------------------------------------------------------------
# Anthropic rejects the sampling parameters (`temperature`, `top_p`, `top_k`)
# with a 400 on newer generations: it began with Opus 4.7 / 4.8 and now also
# covers the Sonnet-5 / Fable-5 generation. Setting any of them to a non-default
# value returns 400 ("`temperature` is deprecated for this model." on the Opus
# 4.7/4.8 line; a plain rejection on Sonnet 5 / Fable 5) — omitting them is
# accepted. Older models (opus-4-6, sonnet-4-6, sonnet-4-5, and earlier) still
# accept `temperature`. cat-stack only ever sends `temperature` (from
# `creativity`) on Anthropic, so that is the only one we skip.
#
# This mirrors the OpenAI reasoning-model handling above — we skip `temperature`
# up-front for the known prefixes in `_build_anthropic_payload`, and
# `UnifiedLLMClient.complete()` strips it on a runtime 400 as a safety net for
# future families not yet in this table.
#
# Matched by name prefix; extend the tuple when new sampling-param-free models
# ship.
# ---------------------------------------------------------------------------
_ANTHROPIC_TEMPERATURE_DEPRECATED = (
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-sonnet-5",
    "claude-fable-5",
)


def _anthropic_supports_temperature(model: str) -> bool:
    """False for Anthropic models that reject the `temperature` param."""
    m = (model or "").lower()
    return not any(m.startswith(p) for p in _ANTHROPIC_TEMPERATURE_DEPRECATED)


# ---------------------------------------------------------------------------
# Anthropic extended-thinking API generations. Newer models (Opus 4.7 / 4.8,
# Sonnet 5, Fable 5) removed the legacy fixed-budget form —
# `thinking: {"type": "enabled", "budget_tokens": N}` returns a 400 — and use
# adaptive thinking instead (`thinking: {"type": "adaptive"}`, depth tuned via
# `output_config.effort`, which we set from thinking_budget — see
# `_thinking_budget_to_effort`). Older models (Opus 4.6, Sonnet 4.6, and
# earlier) still accept `budget_tokens` (deprecated on 4.6 but functional), so
# we keep sending it there to preserve behavior.
#
# Kept as a separate table from `_ANTHROPIC_TEMPERATURE_DEPRECATED` — the two
# constraints happen to cover the same models today but are independent and may
# diverge. `complete()` strips a rejected fixed-budget payload to adaptive on a
# runtime 400 as a safety net for families not yet listed here.
# ---------------------------------------------------------------------------
_ANTHROPIC_ADAPTIVE_THINKING = (
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-sonnet-5",
    "claude-fable-5",
)


def _anthropic_uses_adaptive_thinking(model: str) -> bool:
    """True for Anthropic models that reject fixed `budget_tokens` and require
    adaptive thinking instead."""
    m = (model or "").lower()
    return any(m.startswith(p) for p in _ANTHROPIC_ADAPTIVE_THINKING)


# ---------------------------------------------------------------------------
# Canonical thinking_budget -> effort tier.
#
# `thinking_budget` is the single user-facing reasoning knob (a token count).
# Providers whose API takes a literal token budget (Google, older Anthropic)
# receive it directly; providers whose API takes an effort ENUM instead
# (OpenAI / xAI `reasoning_effort`, Anthropic `output_config.effort`, Ollama
# gpt-oss `think`) receive a coarse tier from this one table — so the same
# thinking_budget produces comparable reasoning intensity regardless of
# provider, instead of every positive budget collapsing to "high".
#
# Only meaningful for budget > 0; each provider represents "off" (budget <= 0)
# in its own way (OpenAI floors at "none"/"minimal"/"low"; Anthropic omits the
# thinking block; Ollama sends its family's low/False value). Capped at "high"
# — the tier every effort-enum provider accepts — so cross-provider parity
# holds (Anthropic's "xhigh"/"max" are deliberately not emitted here).
#
# The two thresholds are the one place to retune the token<->tier mapping.
# ---------------------------------------------------------------------------
_THINKING_EFFORT_LOW_MAX = 2048       # budget <= this -> "low"
_THINKING_EFFORT_MEDIUM_MAX = 8192    # budget <= this -> "medium"; above -> "high"


def _thinking_budget_to_effort(thinking_budget: int) -> str:
    """Map a positive `thinking_budget` (tokens) to a low/medium/high effort
    tier shared by every effort-enum provider. Callers guard budget > 0."""
    if thinking_budget <= _THINKING_EFFORT_LOW_MAX:
        return "low"
    if thinking_budget <= _THINKING_EFFORT_MEDIUM_MAX:
        return "medium"
    return "high"


# ---------------------------------------------------------------------------
# Ollama reasoning control: per-model-family parameter format for the
# top-level `think` field on chat / generate requests.
#
# Ollama standardized on a single API field name (`think`) but the value
# type differs per model family — gpt-oss takes an enum, most others take
# a boolean. See https://docs.ollama.com/capabilities/thinking.
#
# Coverage philosophy: list every Ollama reasoning model family we know of
# AND that uses the `think` field. Reasoning models that gate via other
# mechanisms (system prompts, chat-template flags) are explicitly noted in
# the "NOT in registry" comment below and handled elsewhere — adding them
# here would silently inject a no-op `think` field, which Ollama may
# accept but won't honor, leading to surprising behavior.
#
# Entries are checked longest-prefix-first by `_ollama_think_value()`, so
# put more-specific prefixes earlier when adding (e.g. `qwen3-coder` before
# `qwen3` if they differ).
#
#   Registry tuple: (model prefix, value-format, low_value, high_value)
#
# Models in registry — `think` field works:
#   gpt-oss          — enum: "low" / "medium" / "high"  (cannot fully disable)
#   qwen3 / qwen3.*  — bool: True / False               (covers -thinking variants too)
#   qwq              — bool: True / False               (Qwen QwQ — preceded Qwen3)
#   deepseek-r1      — bool: True / False               (covers -distill variants)
#
# Models NOT in registry — different mechanism, do NOT add here:
#   magistral        — controlled via system prompt (Mistral Magistral)
#   exaone-deep      — uses Modelfile-baked reasoning, no API toggle exposed
#   marco-o1         — uses chat-template wrappers, not `think` field
#
# Models with NO reasoning (so `think` should not appear at all):
#   gemma2/3, llama3.x/4.x, mistral, mistral-nemo, qwen2.5 (non-QwQ),
#   phi3/4, granite, olmo, codestral, …
# These are NOT added; the registry's None-return for unmatched prefixes
# correctly omits the `think` field for them.
# ---------------------------------------------------------------------------
_OLLAMA_REASONING_MODELS = (
    ("gpt-oss",      "enum", "low", "high"),
    ("qwen3",        "bool", False, True),  # covers qwen3.*, qwen3-*, -thinking-* variants
    ("qwq",          "bool", False, True),
    ("deepseek-r1",  "bool", False, True),  # covers -distill-qwen, -distill-llama, etc.
)


def _ollama_think_value(model: str, thinking_budget):
    """Map cat-stack's thinking_budget to the right Ollama `think` value for
    this model family. Returns None if the model isn't in the
    reasoning-capable registry (no `think` field should be set)."""
    if thinking_budget is None:
        return None
    for prefix, fmt, low_val, high_val in _OLLAMA_REASONING_MODELS:
        if model.startswith(prefix):
            if thinking_budget == 0:
                return low_val
            # Enum families (gpt-oss) accept low/medium/high — grade them from
            # the shared token->tier table for cross-provider consistency. Bool
            # families can only toggle on/off, so a positive budget -> on.
            if fmt == "enum":
                return _thinking_budget_to_effort(thinking_budget)
            return high_val
    return None


# ---------------------------------------------------------------------------
# Shared sampling/reasoning param shaping.
#
# Single source of truth for "which sampling / reasoning params does this
# provider+model accept, and in what form". Every payload builder — the
# central `_build_*_payload` methods on UnifiedLLMClient AND the per-strategy
# leaves in `calls/` / `image_functions.py` / `pdf_functions.py` that build
# payloads directly — routes through this function, so a new provider quirk
# (a model family rejecting `temperature`, a new thinking shape, …) is fixed
# here once instead of at every call site.
#
# The function only shapes params; callers keep their own structural fields
# (model / messages / system / max_tokens / response_format / tools) and
# their own HTTP or SDK transport. `complete()`'s runtime 400 fallbacks stay
# in `complete()` — stateless callers get correct up-front params for all
# known model families but no runtime safety net.
# ---------------------------------------------------------------------------
def apply_model_params(
    payload: dict,
    provider: str,
    model: str,
    creativity: float = None,
    thinking_budget: int = None,
    overrides: dict = None,
) -> dict:
    """Apply provider/model-appropriate sampling + reasoning params to
    `payload`, in place, and return it.

    Args:
        payload: The request body (or SDK kwargs dict) built so far. For
                 Anthropic it should already carry `max_tokens` if the caller
                 wants the thinking-headroom bump; for Google, params land
                 inside `generationConfig` (created if missing).
        provider: One of "openai", "anthropic", "google", "mistral",
                  "perplexity", "xai", "huggingface", "huggingface-together",
                  "ollama". Unknown providers get the plain-temperature
                  default.
        creativity: User temperature, or None to omit.
        thinking_budget: cat-stack's cross-provider reasoning knob (tokens).
                 None → don't touch reasoning params; 0 → request reasoning
                 off in the provider's own vocabulary; >0 → provider-native
                 form (token budget or graded effort tier via
                 `_thinking_budget_to_effort`).
        overrides: Runtime-discovered capability flags cached by
                 `UnifiedLLMClient.complete()`'s 400 fallbacks (see
                 `UnifiedLLMClient._param_overrides()`). Stateless callers
                 (the `calls/` leaves) omit it.
    """
    ov = overrides or {}

    if provider == "anthropic":
        # Newer Anthropic models (Opus 4.7+, Sonnet 5, Fable 5) deprecated
        # `temperature` and 400 if it is sent. Skip it for those known
        # prefixes, and also honor the flag cached by complete()'s runtime
        # 400 fallback for future families.
        temp_ok = (
            _anthropic_supports_temperature(model)
            and not ov.get("anthropic_temperature_unsupported", False)
        )
        # Extended thinking. Newer generations (Opus 4.7+, Sonnet 5, Fable 5)
        # require adaptive thinking and 400 on the legacy fixed-budget form;
        # older models still take an explicit budget. Either way the
        # reasoning tokens count against max_tokens, so give the answer
        # headroom. On the legacy path only, temperature must be 1 when
        # thinking is on (Anthropic requirement) — the adaptive-thinking
        # models reject temperature entirely, so we never set it there.
        if thinking_budget and thinking_budget > 0:
            budget = max(thinking_budget, 1024)
            adaptive = (
                _anthropic_uses_adaptive_thinking(model)
                or ov.get("anthropic_thinking_adaptive", False)
            )
            if adaptive:
                payload["thinking"] = {"type": "adaptive"}
                # Depth on the adaptive path is controlled via effort, mapped
                # from thinking_budget the same way as the other effort-enum
                # providers. Use the raw budget (not the 1024-floored
                # `budget`) so the tier matches what OpenAI / xAI derive from
                # the same value.
                payload["output_config"] = {
                    "effort": _thinking_budget_to_effort(thinking_budget)
                }
            else:
                payload["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": budget,
                }
                if temp_ok:
                    payload["temperature"] = 1
            # `budget` doubles as a headroom hint on the adaptive path (where
            # the model, not us, decides how much to think): keep max_tokens
            # above it so reasoning doesn't crowd out the answer.
            if "max_tokens" in payload and payload["max_tokens"] <= budget:
                payload["max_tokens"] = budget + 4096
        elif creativity is not None and temp_ok:
            payload["temperature"] = creativity
        return payload

    if provider == "google":
        # Both params live inside generationConfig (top-level is rejected by
        # Gemini). An explicit zero budget is SENT (since v1.6.8): Gemini's
        # provider default is thinking ON, so omitting the field would leave
        # the uniform "reasoning off" request silently unmet. Models that
        # reject 0 (minimum-budget tiers) are handled by the 400 fallback in
        # complete(), which caches the discovered floor on the client
        # ("google_thinking_floor").
        if creativity is not None:
            payload.setdefault("generationConfig", {})["temperature"] = creativity
        if thinking_budget is not None:
            if thinking_budget > 0:
                budget = max(thinking_budget, 128)
            else:
                budget = ov.get("google_thinking_floor", 0)
            payload.setdefault("generationConfig", {})["thinkingConfig"] = {
                "thinkingBudget": budget
            }
        return payload

    if provider == "openai":
        # OpenAI reasoning models (o-series, GPT-5) only accept
        # temperature=1; reasoning_effort controls depth instead.
        is_reasoning_model = any(
            (model or "").startswith(p) for p in ("o1", "o3", "o4", "gpt-5")
        )
        if is_reasoning_model:
            if thinking_budget is not None:
                if thinking_budget > 0:
                    # Graded low/medium/high from the shared token->tier table
                    # so the same thinking_budget matches the other providers.
                    payload["reasoning_effort"] = _thinking_budget_to_effort(
                        thinking_budget
                    )
                else:
                    # Off-equivalent value depends on the model family — see
                    # `_OPENAI_REASONING_EFFORT_FLOORS`. A previously-
                    # discovered fallback (from a 400 retry in complete())
                    # wins if cached on the client.
                    payload["reasoning_effort"] = (
                        ov.get("reasoning_effort_override")
                        or _openai_reasoning_effort_floor(model)
                    )
        elif creativity is not None:
            payload["temperature"] = creativity
        return payload

    if provider == "xai":
        if creativity is not None:
            payload["temperature"] = creativity
        # Hybrid grok models accept reasoning_effort alongside temperature.
        # "low" is the lowest tier xAI exposes (no "none" / "minimal");
        # explicitly non-reasoning variants 400 on the field — complete()
        # pops it and caches "xai_no_reasoning_effort" so later rows on that
        # client skip the doomed field up front. Variants whose name already
        # encodes "non-reasoning" are off by model choice; sending
        # reasoning_effort to them turns reasoning back ON (verified
        # 2026-06-13), so leave them alone.
        if (
            thinking_budget is not None
            and not ov.get("xai_no_reasoning_effort", False)
            and "non-reasoning" not in (model or "").lower()
        ):
            payload["reasoning_effort"] = (
                "low" if thinking_budget == 0
                else _thinking_budget_to_effort(thinking_budget)
            )
        return payload

    if provider == "ollama":
        if creativity is not None:
            payload["temperature"] = creativity
        # Per-model-family reasoning control via the top-level `think` field.
        # gpt-oss expects an enum ("low"/"medium"/"high"); qwen3/deepseek-r1
        # expect a boolean. Models not in the `_OLLAMA_REASONING_MODELS`
        # registry don't support reasoning and get no `think` field (would be
        # a no-op at best, validator-confusing at worst). Without this,
        # Ollama-served gpt-oss produces long `<think>` blocks by default
        # that bloat per-row generation 3-5x.
        think_value = _ollama_think_value(model, thinking_budget)
        if think_value is not None:
            payload["think"] = think_value
        return payload

    if provider in ("huggingface", "huggingface-together"):
        if creativity is not None:
            payload["temperature"] = creativity
        # Disable thinking on model families whose chat template honors
        # `enable_thinking` (Qwen3-family). Other HF-routed models don't need
        # the kwarg, and strict-validator backends (Fireworks, Groq) reject
        # the unknown field outright — sending it to a non-Qwen model just
        # buys a wasted retry. See `_hf_model_needs_enable_thinking_off()`.
        # The runtime fallback in `complete()` still strips on 400 if a
        # router rejects the kwarg even for a model we expected to support it.
        if thinking_budget == 0 and _hf_model_needs_enable_thinking_off(model):
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        elif (
            thinking_budget == 0
            and _hf_model_reasons_by_default(model)
            and model not in _WARNED_UNCONTROLLED_REASONING
        ):
            # These router-served models reason by default and honor no
            # off-switch through the router (enable_thinking is 400-rejected
            # for their templates). Warn once per process (a fresh client is
            # built per row, so a per-instance flag would warn every row) so
            # the uniform "reasoning off" request isn't silently unmet.
            print(
                f"\n[CatLLM] WARNING: no effective reasoning control delivered "
                f"for '{model}'; the provider's default reasoning "
                f"behavior applies. See docs/reasoning-controls.md.\n"
            )
            _WARNED_UNCONTROLLED_REASONING.add(model)
        return payload

    # Other OpenAI-compatible providers (mistral, perplexity, …): plain
    # temperature, no reasoning knob.
    if creativity is not None:
        payload["temperature"] = creativity
    return payload


__all__ = [
    # Main client
    "UnifiedLLMClient",
    "PROVIDER_CONFIG",
    # Shared sampling/reasoning param shaping
    "apply_model_params",
    # Provider detection
    "detect_provider",
    "_detect_model_source",
    "_detect_huggingface_endpoint",
    # Ollama utilities
    "set_ollama_endpoint",
    "check_ollama_running",
    "list_ollama_models",
    "check_ollama_model",
    "check_system_resources",
    "get_ollama_model_size_estimate",
    "pull_ollama_model",
    "OLLAMA_MODEL_SIZES",
    # Claude Code utilities
    "check_claude_cli_available",
]


# =============================================================================
# HuggingFace Endpoint Auto-Detection
# =============================================================================

def _parse_hf_model_suffix(model: str) -> tuple:
    """
    Parse a HuggingFace model name that may have a :router suffix.

    Examples:
        "Qwen/Qwen3-VL-235B:novita" -> ("Qwen/Qwen3-VL-235B", "novita")
        "meta-llama/Llama-3-8B" -> ("meta-llama/Llama-3-8B", None)

    Returns:
        (clean_model_name, router_name_or_None)
    """
    # Only treat the last segment after ':' as a router suffix if the model
    # contains a '/' (org/model format) to avoid confusing with Ollama tags
    if ":" in model and "/" in model:
        parts = model.rsplit(":", 1)
        suffix = parts[1].lower()
        # Known HuggingFace inference provider routers
        if suffix in ("novita", "together", "sambanova", "cerebras", "fireworks"):
            return parts[0], suffix
    return model, None


# Known router suffix -> endpoint mapping
_HF_ROUTER_ENDPOINTS = {
    "novita": "https://router.huggingface.co/novita/v3/openai",
    "together": "https://router.huggingface.co/together/v1",
    "sambanova": "https://router.huggingface.co/sambanova/v1",
    "cerebras": "https://router.huggingface.co/cerebras/v1",
    "fireworks": "https://router.huggingface.co/fireworks/v1",
}


def _detect_huggingface_endpoint(api_key: str, model: str, skip: set = None) -> str:
    """
    Probe HuggingFace endpoints to find one that supports this model.

    Two call modes:
      - Legacy (skip=None): probe generic + Together only. Falls back to
        returning the generic base URL when nothing responds 200 — keeps
        existing `image_functions` / `pdf_functions` callers behaving as
        before so they can surface their own error from the eventual request.
      - Lazy-fallback (skip=non-empty set): probe generic + all five known
        router endpoints, skipping any in `skip`. Returns None when no
        candidate responds 200 — caller (e.g., UnifiedLLMClient.complete)
        should then surface the original error.

    Args:
        api_key: HuggingFace API key.
        model: Model name to test (may include `:router` suffix).
        skip: optional set of base URLs to skip (typically the URL that
            just failed at the call site).

    Returns:
        Base URL (without /chat/completions) of a working endpoint, or
        None when skip is non-empty and nothing worked.
    """
    skip = skip or set()
    clean_model, router = _parse_hf_model_suffix(model)

    # If explicit router suffix and the suffix endpoint is not skipped,
    # route directly without probing.
    if router and router in _HF_ROUTER_ENDPOINTS:
        candidate = _HF_ROUTER_ENDPOINTS[router]
        if candidate not in skip:
            return candidate

    generic_base = PROVIDER_CONFIG["huggingface"]["endpoint"].replace("/chat/completions", "")

    if skip:
        # Lazy-fallback mode: probe all known routers in priority order.
        candidates_base = [generic_base] + list(_HF_ROUTER_ENDPOINTS.values())
    else:
        # Legacy mode: only generic + Together (preserves prior behavior
        # and probe count for non-UnifiedLLMClient callers).
        candidates_base = [generic_base, _HF_ROUTER_ENDPOINTS["together"]]

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        # Match the main request path: featherless's WAF 403s the default
        # python-requests agent, which would make this probe wrongly skip a
        # working endpoint.
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
    }
    payload = {
        "model": clean_model,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 5,
    }

    for base in candidates_base:
        if base in skip:
            continue
        try:
            response = requests.post(f"{base}/chat/completions", headers=headers, json=payload, timeout=30)
            if response.status_code == 200:
                return base
        except Exception:
            continue

    # Legacy callers expect a base URL even on failure (their HTTP call
    # surfaces the real error). Lazy-fallback callers prefer None so they
    # can surface the original error rather than retrying a known-bad URL.
    if skip:
        return None
    return generic_base


# =============================================================================
# Provider Configuration
# =============================================================================

PROVIDER_CONFIG = {
    "openai": {
        "endpoint": "https://api.openai.com/v1/chat/completions",
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
    },
    "anthropic": {
        "endpoint": "https://api.anthropic.com/v1/messages",
        "auth_header": "x-api-key",
        "auth_prefix": "",
    },
    "google": {
        "endpoint": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        "auth_header": "x-goog-api-key",
        "auth_prefix": "",
    },
    "mistral": {
        "endpoint": "https://api.mistral.ai/v1/chat/completions",
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
    },
    "perplexity": {
        "endpoint": "https://api.perplexity.ai/chat/completions",
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
    },
    "xai": {
        "endpoint": "https://api.x.ai/v1/chat/completions",
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
    },
    "huggingface": {
        "endpoint": "https://router.huggingface.co/v1/chat/completions",
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
    },
    "huggingface-together": {
        "endpoint": "https://router.huggingface.co/together/v1/chat/completions",
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
    },
    "ollama": {
        "endpoint": "http://localhost:11434/v1/chat/completions",
        "auth_header": None,  # No auth required for local Ollama
        "auth_prefix": "",
    },
    "claude-code": {
        "endpoint": None,  # Uses CLI subprocess, not HTTP
        "auth_header": None,
        "auth_prefix": "",
    },
}


# =============================================================================
# Unified API Client
# =============================================================================

class UnifiedLLMClient:
    """A unified client for calling various LLM providers via HTTP."""

    def __init__(self, provider: str, api_key: str, model: str,
                 request_timeout: float = None,
                 max_total_wait: float = None):
        """
        Args:
            request_timeout (float | None): Override the per-HTTP-request
                timeout (seconds). When None, uses the provider-conditional
                default: 120 s for cloud providers, 600 s for Ollama.
                Pass an explicit float to override per call site.
            max_total_wait (float | None): Override the per-call cumulative
                retry budget (seconds). When None, uses provider-conditional
                default: 300 s for cloud, 1200 s for Ollama.
        """
        self.provider = _normalize_provider(provider)
        self.api_key = api_key
        self.model = model
        # User-level overrides for HTTP timeouts. None means "use the
        # provider-conditional default" (see _request_timeout_for /
        # _max_total_wait_for at module level).
        self._request_timeout_override = request_timeout
        self._max_total_wait_override = max_total_wait

        # Lazy HuggingFace router fallback — start with None and only
        # populate when we either (a) have an explicit router suffix, or
        # (b) the default endpoint returns a "wrong router" 400 on a real
        # request. Avoids burning two probe POSTs (and leaking the API key
        # to two endpoints) on every UnifiedLLMClient construction.
        self._custom_endpoint = None
        self._endpoint_lock = threading.Lock()

        if self.provider == "huggingface":
            clean_model, router = _parse_hf_model_suffix(model)
            if router and router in _HF_ROUTER_ENDPOINTS:
                # User was explicit about the router; honour it directly and
                # strip the suffix from the model name (specific-router
                # endpoints expect the clean name, not the suffix).
                self._custom_endpoint = f"{_HF_ROUTER_ENDPOINTS[router]}/chat/completions"
                self.model = clean_model

        if self.provider not in PROVIDER_CONFIG:
            raise ValueError(f"Unsupported provider: {provider}. "
                           f"Supported: {list(PROVIDER_CONFIG.keys())}")

        self.config = PROVIDER_CONFIG[self.provider]

    def _is_hf_wrong_router_400(self, body: str) -> bool:
        """True if a 400 response body indicates the current HF router doesn't
        carry this model (vs. truly nonexistent or a non-routing problem).

        Trigger shapes (from a smoke test against the live HF API):
          - Generic router: `{"error":{"code":"model_not_supported",...}}`
          - Specific router: `{"error":"Model not supported by provider XYZ"}`

        Intentionally NOT triggered by `model_not_found` (no router will help
        a nonexistent model), 401/403 (auth), 5xx/429 (transient), or any
        other 400 unrelated to router routing.
        """
        if self.provider != "huggingface":
            return False
        return (
            '"code":"model_not_supported"' in body
            or "Model not supported by provider" in body
        )

    def _try_hf_router_fallback(self, failed_endpoint: str) -> bool:
        """Find an HF router that has this model. Cache it on self.

        Called from `complete()` when an HF request returns a "wrong router"
        400. Probes all five known specific routers plus the generic router,
        skipping the one that just failed. Idempotent and thread-safe via
        the per-instance endpoint lock — if two concurrent callers both hit
        the fallback path, only one runs the probe.

        Returns True if a working endpoint was found and cached (caller
        should refresh and retry). Returns False if every alternative also
        rejected the model (caller should surface the original error).
        """
        failed_base = failed_endpoint.replace("/chat/completions", "")
        with self._endpoint_lock:
            # Did another thread already find a different working endpoint?
            if self._custom_endpoint:
                current_base = self._custom_endpoint.replace("/chat/completions", "")
                if current_base != failed_base:
                    return True

            new_base = _detect_huggingface_endpoint(
                self.api_key, self.model, skip={failed_base}
            )
            if new_base:
                self._custom_endpoint = f"{new_base}/chat/completions"
                return True
            return False

    def _get_endpoint(self) -> str:
        """Get the API endpoint, substituting model if needed."""
        # Use custom endpoint if set (e.g., for HuggingFace router suffixes)
        endpoint = getattr(self, "_custom_endpoint", None) or self.config["endpoint"]
        if "{model}" in endpoint:
            endpoint = endpoint.format(model=self.model)
        return endpoint

    def _get_headers(self) -> dict:
        """Build request headers for the provider."""
        # Send a browser-like User-Agent. Some providers fronted by a WAF
        # (notably the HuggingFace router's featherless-ai backend) intermittently
        # 403 the default `python-requests/x.y` agent via a Cloudflare bot rule,
        # which surfaces as spurious classification failures. A standard UA is
        # accepted everywhere and costs nothing on providers that don't care.
        headers = {
            "Content-Type": "application/json",
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        }
        auth_header = self.config["auth_header"]
        auth_prefix = self.config["auth_prefix"]

        # Some providers (like Ollama) don't require auth
        if auth_header is not None:
            headers[auth_header] = f"{auth_prefix}{self.api_key}"

        # Anthropic requires additional headers
        if self.provider == "anthropic":
            headers["anthropic-version"] = "2023-06-01"

        return headers

    def _param_overrides(self) -> dict:
        """Runtime-discovered capability flags cached on this client by
        complete()'s 400 fallbacks, in the shape `apply_model_params()`
        expects. Absent flags fall back to the static capability tables."""
        return {
            "anthropic_temperature_unsupported": getattr(
                self, "_anthropic_temperature_unsupported", False
            ),
            "anthropic_thinking_adaptive": getattr(
                self, "_anthropic_thinking_adaptive", False
            ),
            "reasoning_effort_override": getattr(
                self, "_reasoning_effort_override", None
            ),
            "xai_no_reasoning_effort": getattr(
                self, "_xai_no_reasoning_effort", False
            ),
            "google_thinking_floor": getattr(self, "_google_thinking_floor", 0),
        }

    def _build_payload(
        self,
        messages: list,
        json_schema: dict = None,
        creativity: float = None,
        max_tokens: int = 4096,
        thinking_budget: int = None,
        force_json: bool = True,
    ) -> dict:
        """Build the request payload for the specific provider."""

        if self.provider == "anthropic":
            return self._build_anthropic_payload(messages, json_schema, creativity, max_tokens, thinking_budget)
        elif self.provider == "google":
            return self._build_google_payload(messages, json_schema, creativity, thinking_budget, force_json)
        elif self.provider == "openai":
            return self._build_openai_payload(messages, json_schema, creativity, force_json, thinking_budget)
        elif self.provider in ("huggingface", "huggingface-together"):
            # HuggingFace needs thinking_budget to disable thinking on models that reason by default
            return self._build_openai_payload(messages, json_schema, creativity, force_json, thinking_budget)
        elif self.provider == "ollama":
            # Ollama threads thinking_budget to its top-level `think` field for
            # reasoning-capable models (gpt-oss accepts low/medium/high; others
            # accept booleans). Without this, gpt-oss family models emit long
            # <think> blocks by default that bloat per-row generation 3-5x.
            return self._build_openai_payload(messages, json_schema, creativity, force_json, thinking_budget)
        elif self.provider == "xai":
            # v1.6.8: forward the reasoning request. grok-4.3+ hybrids reason
            # by default (2026-06-12 audit: 214 reasoning tokens on a trivial
            # probe with no control sent); non-reasoning variants reject
            # reasoning_effort and are handled by the 400 fallback in
            # complete(), which caches the rejection on the client.
            return self._build_openai_payload(messages, json_schema, creativity, force_json, thinking_budget)
        else:
            # Other OpenAI-compatible providers (mistral, etc.)
            return self._build_openai_payload(messages, json_schema, creativity, force_json)

    def _build_openai_payload(
        self,
        messages: list,
        json_schema: dict = None,
        creativity: float = None,
        force_json: bool = True,
        thinking_budget: int = None,
    ) -> dict:
        """Build payload for OpenAI-compatible APIs.

        Args:
            force_json: If False and no json_schema, don't set response_format (for text responses)
            thinking_budget: For OpenAI reasoning-capable models, maps to
                             reasoning_effort. `thinking_budget=0` picks the
                             provider's off-equivalent value from
                             `_OPENAI_REASONING_EFFORT_FLOORS`
                             ("none" for gpt-5.4+, "minimal" for o-series
                             and gpt-5.0-5.3). `thinking_budget>0` maps to a
                             graded low/medium/high tier via the shared
                             `_thinking_budget_to_effort` table (so the same
                             budget is comparable across providers). If the
                             chosen value is rejected at runtime with 400
                             `unsupported_value`, `complete()` retries with
                             "low" (universally accepted) and caches the
                             override on the client so subsequent calls skip
                             the bad value.
        """
        payload = {
            "model": self.model,
            "messages": messages,
        }

        # Structured output
        # Ollama, HuggingFace, and Mistral only support json_object mode, not strict json_schema
        if json_schema and self.provider not in ["ollama", "huggingface", "huggingface-together", "mistral"]:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "classification_result",
                    "strict": True,
                    "schema": json_schema,
                }
            }
        elif json_schema:
            # Ollama/HuggingFace - use json_object mode
            payload["response_format"] = {"type": "json_object"}
        elif force_json:
            # No schema but force JSON output
            payload["response_format"] = {"type": "json_object"}
        # else: no response_format - allow text responses

        # Sampling + reasoning params (temperature, reasoning_effort, Ollama
        # `think`, HF chat_template_kwargs) — shared shaper, one source of
        # truth across all payload builders.
        apply_model_params(
            payload,
            self.provider,
            self.model,
            creativity=creativity,
            thinking_budget=thinking_budget,
            overrides=self._param_overrides(),
        )

        return payload

    def _build_anthropic_payload(
        self,
        messages: list,
        json_schema: dict = None,
        creativity: float = None,
        max_tokens: int = 4096,
        thinking_budget: int = None,
    ) -> dict:
        """Build payload for Anthropic API.

        Args:
            thinking_budget: Controls extended thinking for Anthropic models.
                             0 or None → thinking disabled.
                             >0 → thinking enabled with budget_tokens = max(thinking_budget, 1024).
        """
        # Extract system message if present
        system_content = None
        user_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            else:
                user_messages.append(msg)

        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": user_messages,
        }

        if system_content:
            payload["system"] = system_content

        # Temperature gating + thinking shape (adaptive vs fixed-budget) +
        # max_tokens headroom — shared shaper, one source of truth across all
        # payload builders.
        apply_model_params(
            payload,
            "anthropic",
            self.model,
            creativity=creativity,
            thinking_budget=thinking_budget,
            overrides=self._param_overrides(),
        )

        # Use tool calling for structured output (most reliable for Anthropic)
        # When thinking is enabled, forced tool_choice is not allowed — use "auto"
        if json_schema:
            payload["tools"] = [{
                "name": "return_categories",
                "description": "Return categorization results",
                "input_schema": json_schema,
            }]
            if thinking_budget and thinking_budget > 0:
                payload["tool_choice"] = {"type": "auto"}
            else:
                payload["tool_choice"] = {"type": "tool", "name": "return_categories"}

        return payload

    def _build_google_payload(
        self,
        messages: list,
        json_schema: dict = None,
        creativity: float = None,
        thinking_budget: int = None,
        force_json: bool = True,
    ) -> dict:
        """Build payload for Google Gemini API."""
        # Convert messages to Google format
        # Combine system + user messages into a single prompt
        combined_text = ""
        for msg in messages:
            if msg["role"] == "system":
                combined_text += msg["content"] + "\n\n"
            elif msg["role"] == "user":
                combined_text += msg["content"]
            elif msg["role"] == "assistant":
                combined_text += "\n\nAssistant: " + msg["content"] + "\n\n"

        payload = {
            "contents": [{"parts": [{"text": combined_text}]}],
            "generationConfig": {}
        }

        if json_schema:
            payload["generationConfig"]["responseMimeType"] = "application/json"
            payload["generationConfig"]["responseSchema"] = _sanitize_google_schema(json_schema)
        elif force_json:
            payload["generationConfig"]["responseMimeType"] = "application/json"
        # else: no mime type - allow text responses

        # Temperature + thinkingConfig (both inside generationConfig) —
        # shared shaper, one source of truth across all payload builders.
        apply_model_params(
            payload,
            "google",
            self.model,
            creativity=creativity,
            thinking_budget=thinking_budget,
            overrides=self._param_overrides(),
        )

        return payload

    def _parse_response(self, response_json: dict) -> str:
        """Parse the response based on provider format."""
        if self.provider == "anthropic":
            return self._parse_anthropic_response(response_json)
        elif self.provider == "google":
            return self._parse_google_response(response_json)
        else:
            # OpenAI-compatible
            return self._parse_openai_response(response_json)

    def _parse_openai_response(self, response_json: dict) -> str:
        """Parse OpenAI-compatible response."""
        return response_json["choices"][0]["message"]["content"]

    def _parse_anthropic_response(self, response_json: dict) -> str:
        """Parse Anthropic response (handles both text and tool use).

        A tool_use block is preferred over any text, wherever it appears: with
        extended/adaptive thinking the model uses tool_choice="auto" and may emit
        a text preamble before the tool call, so returning the first text block
        would drop the structured categories. `thinking` blocks are ignored.
        """
        content = response_json.get("content", [])
        first_text = None
        for block in content:
            btype = block.get("type")
            if btype == "tool_use":
                # Return the tool input as JSON string
                return json.dumps(block.get("input", {}))
            elif btype == "text" and first_text is None:
                first_text = block.get("text", "")
        return first_text if first_text is not None else ""

    def _parse_google_response(self, response_json: dict) -> str:
        """Parse Google Gemini response."""
        candidates = response_json.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            if parts:
                return parts[0].get("text", "")
        return ""

    def _call_claude_cli(
        self,
        messages: list,
        max_retries: int = 3,
        initial_delay: float = 2.0,
    ) -> tuple[str, str | None]:
        """
        Call the Claude CLI (claude -p) as a subprocess.

        Args:
            messages: List of message dicts with 'role' and 'content'
            max_retries: Maximum retry attempts
            initial_delay: Initial delay for exponential backoff

        Returns:
            tuple: (response_text, error_message)
        """
        import subprocess

        # Extract system and user messages
        system_parts = []
        user_parts = []
        for msg in messages:
            if msg["role"] == "system":
                system_parts.append(msg["content"])
            elif msg["role"] in ("user", "assistant"):
                user_parts.append(msg["content"])

        system_prompt = "\n\n".join(system_parts) if system_parts else None
        user_prompt = "\n\n".join(user_parts)

        # Build command
        cmd = ["claude", "-p", "--output-format", "text", "--model", self.model]
        if system_prompt:
            cmd.extend(["--system-prompt", system_prompt])
        cmd.append(user_prompt)

        try:
            for attempt in range(max_retries):
                try:
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=120,
                    )
                    if result.returncode == 0:
                        return result.stdout.strip(), None
                    else:
                        error_msg = result.stderr.strip() or f"CLI exited with code {result.returncode}"
                        if attempt < max_retries - 1:
                            wait_time = initial_delay * (2 ** attempt)
                            print(f"Claude CLI error: {error_msg}. Retrying in {wait_time}s...")
                            time.sleep(wait_time)
                        else:
                            return None, f"Claude CLI failed after {max_retries} attempts: {error_msg}"
                except subprocess.TimeoutExpired:
                    if attempt < max_retries - 1:
                        wait_time = initial_delay * (2 ** attempt)
                        print(f"Claude CLI timeout. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        return None, "Claude CLI timeout after retries"
                except FileNotFoundError:
                    return None, (
                        "Claude CLI not found. Install it: "
                        "https://docs.anthropic.com/en/docs/claude-code"
                    )

            return None, "Max retries exceeded"
        except OSError as e:
            # E2BIG on macOS/Linux: argv too long. Deterministic for this
            # prompt size — no point retrying. Surface as an error rather
            # than letting OSError bubble out and break the (text, error)
            # contract that callers depend on.
            return None, f"Claude CLI subprocess failed: {e} (prompt may be too large for argv)"

    def complete(
        self,
        messages: list,
        json_schema: dict = None,
        creativity: float = None,
        thinking_budget: int = None,
        force_json: bool = True,
        max_retries: int = 5,
        initial_delay: float = 2.0,
    ) -> tuple[str, str | None]:
        """
        Make a completion request to the LLM provider.

        Args:
            messages: List of message dicts with 'role' and 'content'
            json_schema: Optional JSON schema for structured output
            creativity: Temperature setting (None for default)
            thinking_budget: A single token-count knob for reasoning depth,
                translated to each provider's native form so the same value is
                comparable across providers (see `_thinking_budget_to_effort`):
                - Google: literal token budget (0 to disable, >0 to enable).
                - Anthropic (Opus 4.6 / Sonnet 4.6 and earlier): literal
                  `budget_tokens` (0 to disable, >0 to enable with min 1024).
                - Anthropic (Opus 4.7+, Sonnet 5, Fable 5): adaptive thinking
                  with `output_config.effort` graded from the budget.
                - OpenAI / xAI: `reasoning_effort`. `thinking_budget=0` picks
                  the model's off-floor ("none"/"minimal"/"low"); >0 maps to a
                  graded low/medium/high tier. A rejected value falls back to
                  "low" at runtime and is cached.
                - Ollama gpt-oss: graded `think` enum; bool families toggle
                  on/off. HuggingFace Qwen3: on/off via `enable_thinking`.
            force_json: If True and no json_schema, still request JSON output.
                       Set to False for text-only responses (e.g., CoVe intermediate steps)
            max_retries: Maximum retry attempts
            initial_delay: Initial delay for exponential backoff

        Returns:
            tuple: (response_text, error_message)
                   error_message is None on success
        """
        if self.provider == "claude-code":
            return self._call_claude_cli(messages, max_retries=max_retries, initial_delay=initial_delay)

        headers = self._get_headers()
        payload = self._build_payload(messages, json_schema, creativity, thinking_budget=thinking_budget, force_json=force_json)

        # If a previous call on this client already discovered the endpoint
        # rejects response_format (persistent 502s, etc.), drop it before
        # even trying. Saves N-1 wasted strip-cycles in a multi-row run.
        if getattr(self, "_skip_response_format", False) and "response_format" in payload:
            payload.pop("response_format")

        # Track cumulative wait so a long string of transient errors can't
        # block the call indefinitely. Timeouts are provider-conditional by
        # default; user overrides on the client instance (set at __init__)
        # take precedence.
        start = time.monotonic()
        request_timeout = (
            self._request_timeout_override
            if self._request_timeout_override is not None
            else _request_timeout_for(self.provider)
        )
        max_total_wait = (
            self._max_total_wait_override
            if self._max_total_wait_override is not None
            else _max_total_wait_for(self.provider)
        )
        # Per-call flag: have we already tried stripping response_format on a
        # transient error this call? Only strip once per call so we don't
        # mutate payload on every retry tick.
        stripped_response_format = False
        # v1.6.8: consecutive-timeout counter + one-shot Google schema drop
        # (see the Timeout handler below).
        timeout_count = 0
        dropped_google_schema = False

        for attempt in range(max_retries):
            endpoint = self._get_endpoint()
            try:
                response = requests.post(
                    endpoint,
                    headers=headers,
                    json=payload,
                    timeout=request_timeout,
                )

                # Check for HTTP errors
                if response.status_code == 400:
                    error_text = response.text.lower()
                    # If the model doesn't support structured outputs (json_object/json_schema),
                    # retry without response_format. The prompt still asks for JSON and
                    # extract_json() will parse it from the free-text response.
                    if "structured" in error_text or "response_format" in error_text or "json_object" in error_text:
                        if "response_format" in payload:
                            if not getattr(self, '_warned_no_structured', False):
                                print(f"\n[CatLLM] Model '{self.model}' does not support structured JSON output.")
                                print(f"  Falling back to prompt-based JSON parsing.\n")
                                self._warned_no_structured = True
                            payload.pop("response_format")
                            continue  # Retry immediately without response_format

                    # HF: some routers reject `chat_template_kwargs` outright.
                    # The wording varies per router:
                    #   Groq:      "property 'chat_template_kwargs' is unsupported"
                    #   Fireworks: "Extra inputs are not permitted, field:
                    #               'chat_template_kwargs'"
                    # The kwarg is only there to disable thinking on Qwen3-
                    # family models when thinking_budget=0 — dropping it on
                    # a router that doesn't honor it is harmless. Strip and
                    # retry, mirror the response_format pattern above.
                    _ctk_rejected = (
                        "chat_template_kwargs" in error_text
                        and any(phrase in error_text for phrase in (
                            "unsupported", "not permitted", "not allowed",
                            "extra inputs", "extra fields", "unknown field",
                        ))
                    )
                    if _ctk_rejected:
                        if "chat_template_kwargs" in payload:
                            if not getattr(self, '_warned_no_chat_template_kwargs', False):
                                print(f"\n[CatLLM] Model '{self.model}' does not accept chat_template_kwargs.")
                                print(f"  Dropping the param and retrying. (thinking-mode control may be a no-op on this router.)\n")
                                self._warned_no_chat_template_kwargs = True
                            payload.pop("chat_template_kwargs")
                            continue  # Retry immediately without chat_template_kwargs

                    # OpenAI reasoning_effort enum varies across model
                    # families — gpt-5.4+ deprecated "minimal" in favor of
                    # "none"; older models reject "none". If the model
                    # rejects our chosen value with 400 unsupported_value,
                    # fall back to "low" (universally accepted across all
                    # OpenAI reasoning-effort-supporting models) and cache
                    # the override so subsequent calls skip the doomed
                    # value. If "low" itself is rejected, drop reasoning_effort
                    # entirely.
                    if "reasoning_effort" in error_text and (
                        "unsupported" in error_text or "invalid" in error_text
                    ):
                        current = payload.get("reasoning_effort")
                        if current not in (None, "low"):
                            if not getattr(self, '_warned_reasoning_effort_fallback', False):
                                print(f"\n[CatLLM] Model '{self.model}' rejected reasoning_effort='{current}'.")
                                print(f"  Falling back to 'low' and caching for subsequent calls on this client.\n")
                                self._warned_reasoning_effort_fallback = True
                            self._reasoning_effort_override = "low"
                            payload["reasoning_effort"] = "low"
                            continue
                        elif current == "low" and "reasoning_effort" in payload:
                            # Model takes no reasoning_effort at all (e.g.
                            # xAI's explicitly non-reasoning variants).
                            # Cache so later rows on this client skip the
                            # doomed field up front (v1.6.8).
                            self._xai_no_reasoning_effort = True
                            payload.pop("reasoning_effort")
                            continue

                    # Google (v1.6.8): minimum-budget thinking tiers reject
                    # thinkingBudget: 0. Fall back to 128 (Google's stated
                    # minimum) and cache on the client.
                    if (
                        self.provider == "google"
                        and "thinking" in error_text
                        and ("budget" in error_text or "invalid" in error_text
                             or "unsupported" in error_text)
                        and payload.get("generationConfig", {})
                                   .get("thinkingConfig", {})
                                   .get("thinkingBudget") == 0
                    ):
                        self._google_thinking_floor = 128
                        payload["generationConfig"]["thinkingConfig"]["thinkingBudget"] = 128
                        print(f"\n[CatLLM] Model '{self.model}' rejected thinkingBudget=0; "
                              f"falling back to the minimum (128) and caching for this client.\n")
                        continue

                    # Anthropic rejects `temperature` on newer models (Opus
                    # 4.7+, and now the Sonnet-5 / Fable-5 generation). The
                    # wording varies by family: the Opus 4.7/4.8 line 400s with
                    # "`temperature` is deprecated for this model.", while
                    # Sonnet 5 / Fable 5 reject the parameter itself. Match any
                    # rejection phrase (not just "deprecated") so the net also
                    # covers families not yet in `_ANTHROPIC_TEMPERATURE_DEPRECATED`.
                    # Strip it, cache on the client so the payload builder skips
                    # it for subsequent rows on this client, and retry.
                    _temp_rejected = (
                        "temperature" in error_text
                        and "temperature" in payload
                        and any(phrase in error_text for phrase in (
                            "deprecated", "not supported", "unsupported",
                            "not permitted", "not allowed", "not accepted",
                            "unexpected", "extra inputs", "removed",
                        ))
                    )
                    if _temp_rejected:
                        if not getattr(self, '_warned_temperature_deprecated', False):
                            print(f"\n[CatLLM] Model '{self.model}' does not accept the temperature parameter.")
                            print(f"  Dropping it and caching for subsequent calls on this client.\n")
                            self._warned_temperature_deprecated = True
                        self._anthropic_temperature_unsupported = True
                        payload.pop("temperature")
                        continue

                    # Anthropic rejects the legacy fixed-budget thinking API
                    # (`thinking: {"type":"enabled","budget_tokens":N}`) on newer
                    # models (Opus 4.7+, Sonnet 5, Fable 5), which require adaptive
                    # thinking. Rewrite the payload to adaptive, cache the decision,
                    # and retry — safety net for families not yet in
                    # `_ANTHROPIC_ADAPTIVE_THINKING`. Those same models also reject
                    # `temperature`, so drop it here too to avoid a second round-trip.
                    _thinking = payload.get("thinking")
                    _thinking_rejected = (
                        isinstance(_thinking, dict)
                        and _thinking.get("type") == "enabled"
                        and ("thinking" in error_text or "budget_tokens" in error_text)
                        and any(phrase in error_text for phrase in (
                            "adaptive", "deprecated", "not supported", "unsupported",
                            "not permitted", "not allowed", "not accepted",
                            "unexpected", "removed",
                        ))
                    )
                    if _thinking_rejected:
                        if not getattr(self, '_warned_thinking_adaptive', False):
                            print(f"\n[CatLLM] Model '{self.model}' rejected fixed-budget thinking.")
                            print(f"  Switching to adaptive thinking and caching for this client.\n")
                            self._warned_thinking_adaptive = True
                        self._anthropic_thinking_adaptive = True
                        _prior_budget = _thinking.get("budget_tokens", 0)
                        payload["thinking"] = {"type": "adaptive"}
                        # Carry depth over to effort, matching the proactive path.
                        if _prior_budget and _prior_budget > 0:
                            payload["output_config"] = {
                                "effort": _thinking_budget_to_effort(_prior_budget)
                            }
                        if "temperature" in payload:
                            self._anthropic_temperature_unsupported = True
                            payload.pop("temperature")
                        continue

                    # HuggingFace: try other routers when the current one
                    # rejects the model with a "wrong router" 400.
                    if self._is_hf_wrong_router_400(response.text):
                        if self._try_hf_router_fallback(endpoint):
                            continue  # retry with the newly-cached endpoint
                if response.status_code == 404 or (response.status_code == 400 and "not found" in response.text.lower() and "model" in response.text.lower()):
                    return None, f"Model '{self.model}' not found for {self.provider}"
                elif response.status_code in [401, 403]:
                    return None, f"Authentication failed for {self.provider}"
                elif response.status_code == 429:
                    # Rate limited. Honor Retry-After if present; otherwise
                    # fall back to jittered exponential (5x multiplier for
                    # rate limits — they need longer cool-down than 5xx).
                    wait_time = _parse_retry_after(response.headers.get("Retry-After"))
                    if wait_time is None:
                        wait_time = _backoff_with_jitter(initial_delay, attempt, multiplier=5.0)
                    elapsed = time.monotonic() - start
                    if attempt < max_retries - 1 and elapsed + wait_time <= max_total_wait:
                        # Name the throttling provider/model so multi-model
                        # ensemble runs can attribute the slowdown.
                        print(f"[{self.provider}/{self.model}] Rate limited. Waiting {wait_time:.1f}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        return None, "Rate limit exceeded (retry budget or time cap)"
                elif response.status_code >= 500:
                    # Server error. If the server included Retry-After, it's
                    # explicitly saying "I'm overloaded, come back in N
                    # seconds" — trust that, don't treat as a payload issue.
                    # If no Retry-After and we haven't yet tried, strip
                    # response_format once. HF's router for small Llama
                    # variants reliably 502s when sent json_object with an
                    # HTML error body (no Retry-After) — stripping fixes it
                    # in practice. Cache the decision so subsequent rows on
                    # the same client skip the doomed payload from the start.
                    wait_time = _parse_retry_after(response.headers.get("Retry-After"))
                    if (
                        wait_time is None
                        and not stripped_response_format
                        and "response_format" in payload
                    ):
                        stripped_response_format = True
                        self._skip_response_format = True
                        payload.pop("response_format")
                        if not getattr(self, "_warned_no_structured", False):
                            print(
                                f"\n[CatLLM] Persistent {response.status_code} from "
                                f"'{self.model}'. Retrying without response_format "
                                f"(some endpoints reject json_object with a non-JSON "
                                f"error body)."
                            )
                            self._warned_no_structured = True
                        continue  # immediate retry, no backoff sleep
                    # Honor Retry-After if present; otherwise jittered exponential.
                    if wait_time is None:
                        wait_time = _backoff_with_jitter(initial_delay, attempt)
                    elapsed = time.monotonic() - start
                    if attempt < max_retries - 1 and elapsed + wait_time <= max_total_wait:
                        # Name the failing provider/model — same rationale as
                        # the 429 handler above.
                        print(f"[{self.provider}/{self.model}] Server error {response.status_code}. Retrying in {wait_time:.1f}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        return None, f"Server error {response.status_code} after retries"

                response.raise_for_status()
                response_json = response.json()
                result = self._parse_response(response_json)
                return result, None

            except requests.exceptions.Timeout:
                timeout_count += 1
                # v1.6.8: Gemini can reproducibly hang on specific inputs
                # when a strict responseSchema is attached (constrained-
                # decoding pathology; 2026-06-12 audit — a trivial input
                # timed out 6/6 attempts WITH the schema and answered
                # instantly without it). After two consecutive timeouts with
                # a schema attached, drop the schema once and re-ask: the
                # prompt still requests JSON and extract_json() parses it
                # from the free-text response.
                if (
                    self.provider == "google"
                    and timeout_count >= 2
                    and not dropped_google_schema
                    and "responseSchema" in payload.get("generationConfig", {})
                ):
                    dropped_google_schema = True
                    payload["generationConfig"].pop("responseSchema", None)
                    print(f"[CatLLM] Repeated timeouts from '{self.model}' with "
                          f"responseSchema attached; retrying schema-less "
                          f"(prompt-based JSON parsing).")
                    continue
                wait_time = _backoff_with_jitter(initial_delay, attempt)
                elapsed = time.monotonic() - start
                if attempt < max_retries - 1 and elapsed + wait_time <= max_total_wait:
                    print(f"Request timeout. Retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                else:
                    return None, "Request timeout after retries"

            except requests.exceptions.RequestException as e:
                wait_time = _backoff_with_jitter(initial_delay, attempt)
                elapsed = time.monotonic() - start
                if attempt < max_retries - 1 and elapsed + wait_time <= max_total_wait:
                    print(f"Request error: {e}. Retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                else:
                    return None, f"Request failed: {e}"

            except json.JSONDecodeError as e:
                return None, f"Failed to parse response JSON: {e}"

        return None, "Max retries exceeded"


# =============================================================================
# Provider Detection
# =============================================================================

def _parse_retry_after(header_value):
    """Parse a Retry-After header value into seconds, or return None if
    the header is missing/unparseable/negative.

    Accepts both forms allowed by RFC 7231:
      - Integer seconds: `Retry-After: 30`
      - HTTP-date:       `Retry-After: Wed, 21 Oct 2026 07:28:00 GMT`
    """
    if not header_value:
        return None
    s = str(header_value).strip()
    try:
        return max(0.0, float(s))
    except ValueError:
        pass
    try:
        from email.utils import parsedate_to_datetime
        from datetime import datetime, timezone
        target = parsedate_to_datetime(s)
        if target is None:
            return None
        now = datetime.now(timezone.utc)
        return max(0.0, (target - now).total_seconds())
    except Exception:
        return None


def _backoff_with_jitter(initial_delay, attempt, multiplier=1.0):
    """Full-jitter exponential backoff. Returns a value in
    [0.5 * base, 1.5 * base] where base = initial_delay * 2^attempt * multiplier.

    Jitter prevents thundering-herd behavior when multiple concurrent
    callers (e.g. ensemble workers) all hit a 429 at the same instant and
    would otherwise wake at the same retry tick.
    """
    base = initial_delay * (2 ** attempt) * multiplier
    return base * (0.5 + random.random())


# Keys Google's responseSchema rejects (subset of OpenAPI 3.0 — strict
# vs. full JSON Schema). Stripped recursively before send.
# Google docs: https://ai.google.dev/api/generate-content#schema
_GOOGLE_SCHEMA_UNSUPPORTED = frozenset({
    "additionalProperties",  # rejected on most Gemini models
    "$schema",
    "$ref",
    "$defs",
    "definitions",
    "oneOf",
    "anyOf",
    "allOf",
    "not",
    "patternProperties",
    "exclusiveMinimum",
    "exclusiveMaximum",
})


def _sanitize_google_schema(schema):
    """Recursively strip keys Google's responseSchema doesn't accept.

    The preflight probe sends `additionalProperties: false` which is
    valid JSON Schema but causes a 400 on most Gemini models. Same
    issue for `oneOf` / `anyOf` etc. when callers pass richer schemas.
    Removing these keys produces a schema that Google accepts while
    preserving the validation intent (Gemini's schema validation is
    less strict by design — the model is asked to follow the shape,
    not to formally validate).
    """
    if isinstance(schema, dict):
        return {
            k: _sanitize_google_schema(v)
            for k, v in schema.items()
            if k not in _GOOGLE_SCHEMA_UNSUPPORTED
        }
    if isinstance(schema, list):
        return [_sanitize_google_schema(item) for item in schema]
    return schema


def _normalize_provider(provider) -> str:
    """Normalize a provider name. `local` is an alias for `ollama` —
    friendlier wording for users running local inference who don't think
    of it as "the Ollama provider"."""
    if not provider:
        return provider
    p = provider.lower()
    if p == "local":
        return "ollama"
    return p


# Token-based provider detection. We tokenize the model name on `-`, `_`,
# `.` so each family name lives in its own slot — that removes the bare
# substring leakage that bit pre-fix code (e.g. `qwen-o3-coder` matching
# "o3" before "qwen" and routing the user's HF API key to OpenAI's
# endpoint). First match across (token order × family-prefix order) wins.
_FAMILY_PREFIXES = (
    ("gpt", "openai"),
    ("claude", "anthropic"),
    ("gemini", "google"),
    ("gemma", "google"),
    ("mistral", "mistral"),
    ("mixtral", "mistral"),
    ("grok", "xai"),
    ("sonar", "perplexity"),
    ("pplx", "perplexity"),
    ("llama", "huggingface"),
    ("deepseek", "huggingface"),
    ("qwen", "huggingface"),
)

# OpenAI o-series models. Must be the FIRST token in the model name —
# guards against `qwen-o3-coder` style strings where `o3` appears
# downstream and isn't actually an o-series model.
_O_SERIES_TOKENS = frozenset({f"o{n}" for n in range(1, 10)})


def detect_provider(model_name: str, provider: str = "auto") -> str:
    """Auto-detect provider from model name if not explicitly provided.

    Routing rules, in order:
      1. `provider != "auto"` → use it (`local` normalised to `ollama`).
      2. Model name contains `/` → HuggingFace (`org/model[:router]` format).
      3. Model name contains `:` (no `/`) → looks like Ollama tag syntax;
         raise ValueError. Ollama is intentionally never auto-detected — too
         easy to misroute to local inference when the user meant a hosted
         provider, and the failure mode (connection refused on port 11434)
         is confusing. Set `provider='local'` (or `'ollama'`) explicitly.
      4. Tokenize on `-`, `_`, `.`. First token in `_O_SERIES_TOKENS`
         (`o1`, …, `o9`) → openai. Otherwise, the first token-prefix
         match against `_FAMILY_PREFIXES` wins.
      5. No match → ValueError asking for explicit `provider=`.
    """
    if provider and provider.lower() != "auto":
        return _normalize_provider(provider)

    name_lower = model_name.lower()

    if "/" in name_lower:
        return "huggingface"

    if ":" in name_lower:
        raise ValueError(
            f"Model '{model_name}' looks like Ollama tag syntax (`name:tag`). "
            "Auto-detection is intentionally disabled for Ollama models — "
            "set provider='local' (or provider='ollama') explicitly to use "
            "local Ollama inference."
        )

    tokens = [t for t in name_lower.replace("_", "-").replace(".", "-").split("-") if t]
    if not tokens:
        raise ValueError(
            f"Could not auto-detect provider from '{model_name}'. "
            "Please specify provider explicitly."
        )

    if tokens[0] in _O_SERIES_TOKENS:
        return "openai"

    for token in tokens:
        for prefix, prov in _FAMILY_PREFIXES:
            if token.startswith(prefix):
                return prov

    raise ValueError(
        f"Could not auto-detect provider from '{model_name}'. "
        "Please specify provider explicitly: openai, anthropic, google, mistral, "
        "perplexity, xai, huggingface, or ollama."
    )


def _detect_model_source(user_model, model_source):
    """Back-compat shim. Delegates to `detect_provider` so both paths route
    identically; kept because internal callers (text_functions.py et al.)
    still use this name. Will be inlined in a future cleanup."""
    if model_source and model_source.lower() == "claude-code":
        return "claude-code"
    return detect_provider(user_model, provider=model_source)


# =============================================================================
# Ollama Functions
# =============================================================================

def set_ollama_endpoint(host: str = "localhost", port: int = 11434):
    """
    Configure a custom Ollama endpoint.

    Useful if Ollama is running on a different host or port.

    Args:
        host: Hostname where Ollama is running (default: localhost)
        port: Port number (default: 11434)

    Example:
        set_ollama_endpoint("192.168.1.100", 11434)
    """
    PROVIDER_CONFIG["ollama"]["endpoint"] = f"http://{host}:{port}/v1/chat/completions"


def check_ollama_running(host: str = "localhost", port: int = 11434) -> bool:
    """
    Check if Ollama is running and accessible.

    Args:
        host: Hostname where Ollama should be running
        port: Port number

    Returns:
        True if Ollama is running, False otherwise
    """
    try:
        response = requests.get(f"http://{host}:{port}/api/tags", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def list_ollama_models(host: str = "localhost", port: int = 11434) -> list:
    """
    List all models available in the local Ollama installation.

    Args:
        host: Hostname where Ollama is running
        port: Port number

    Returns:
        List of model names, or empty list if Ollama is not running
    """
    try:
        response = requests.get(f"http://{host}:{port}/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return [model["name"] for model in data.get("models", [])]
        return []
    except requests.exceptions.RequestException:
        return []


def check_ollama_model(model: str, host: str = "localhost", port: int = 11434) -> bool:
    """
    Check if a specific model is available in Ollama.

    Args:
        model: Model name to check (e.g., "llama3.2", "mistral")
        host: Hostname where Ollama is running
        port: Port number

    Returns:
        True if model is available, False otherwise
    """
    available_models = list_ollama_models(host, port)
    model_lower = model.lower()
    if ":" in model_lower:
        # User specified an explicit tag (e.g. "qwen2.5:14b") — require exact
        # match.  An installed "qwen2.5:7b" must NOT satisfy a request for
        # "qwen2.5:14b"; the previous prefix-match logic let this through,
        # which caused silent classification failures downstream.
        return any(m.lower() == model_lower for m in available_models)
    # User specified just the family (e.g. "llama3.2") — any installed
    # variant of that family counts (e.g. "llama3.2:latest", "llama3.2:7b").
    return any(
        m.lower() == model_lower or m.lower().startswith(f"{model_lower}:")
        for m in available_models
    )


def _format_bytes(size_bytes: int) -> str:
    """Format bytes into human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 ** 3:
        return f"{size_bytes / (1024 ** 2):.1f} MB"
    else:
        return f"{size_bytes / (1024 ** 3):.2f} GB"


def _parse_size_string(size_str: str) -> int:
    """Parse a size string like '2.0 GB' into bytes."""
    if size_str == "unknown":
        return 0

    size_str = size_str.strip().upper()
    try:
        if "GB" in size_str:
            return int(float(size_str.replace("GB", "").strip()) * 1024 ** 3)
        elif "MB" in size_str:
            return int(float(size_str.replace("MB", "").strip()) * 1024 ** 2)
        elif "KB" in size_str:
            return int(float(size_str.replace("KB", "").strip()) * 1024)
        else:
            return int(float(size_str.replace("B", "").strip()))
    except ValueError:
        return 0


# Common model sizes (approximate) for user reference
OLLAMA_MODEL_SIZES = {
    "llama3.2": "2.0 GB",
    "llama3.2:1b": "1.3 GB",
    "llama3.2:3b": "2.0 GB",
    "llama3.1": "4.7 GB",
    "llama3.1:8b": "4.7 GB",
    "llama3.1:70b": "40 GB",
    "llama3": "4.7 GB",
    "llama2": "3.8 GB",
    "mistral": "4.1 GB",
    "mixtral": "26 GB",
    "phi3": "2.2 GB",
    "phi3:mini": "2.2 GB",
    "gemma": "5.0 GB",
    "gemma:2b": "1.7 GB",
    "gemma:7b": "5.0 GB",
    "gemma2": "5.4 GB",
    "gemma2:2b": "1.6 GB",
    "gemma2:9b": "5.4 GB",
    "gemma2:27b": "16 GB",
    "qwen2.5": "4.7 GB",
    "qwen2.5:0.5b": "397 MB",
    "qwen2.5:1.5b": "986 MB",
    "qwen2.5:3b": "1.9 GB",
    "qwen2.5:7b": "4.7 GB",
    "deepseek-r1": "4.7 GB",
    "codellama": "3.8 GB",
    "codegemma": "5.0 GB",
    "nomic-embed-text": "274 MB",
}


def get_ollama_model_size_estimate(model: str) -> str:
    """
    Get estimated download size for an Ollama model.

    Args:
        model: Model name

    Returns:
        Human-readable size estimate or "unknown"
    """
    model_lower = model.lower()

    # Check exact match first
    if model_lower in OLLAMA_MODEL_SIZES:
        return OLLAMA_MODEL_SIZES[model_lower]

    # Check base model name (without tag)
    base_model = model_lower.split(":")[0]
    if base_model in OLLAMA_MODEL_SIZES:
        return OLLAMA_MODEL_SIZES[base_model]

    return "unknown"


def check_system_resources(model: str) -> dict:
    """
    Check if system has enough resources to download and run a model.

    Args:
        model: Model name to check

    Returns:
        dict with 'can_download', 'can_run', 'warnings', and 'details'
    """
    import shutil
    import os

    result = {
        "can_download": True,
        "can_run": True,
        "warnings": [],
        "details": {}
    }

    size_estimate = get_ollama_model_size_estimate(model)
    model_size_bytes = _parse_size_string(size_estimate)

    # Check disk space (Ollama typically stores models in ~/.ollama)
    ollama_dir = os.path.expanduser("~/.ollama")
    if not os.path.exists(ollama_dir):
        ollama_dir = os.path.expanduser("~")

    try:
        disk_usage = shutil.disk_usage(ollama_dir)
        free_space = disk_usage.free
        result["details"]["free_disk_space"] = _format_bytes(free_space)
        result["details"]["model_size"] = size_estimate

        # Need at least 1.5x model size for download + extraction
        required_space = int(model_size_bytes * 1.5) if model_size_bytes > 0 else 0

        if required_space > 0 and free_space < required_space:
            result["can_download"] = False
            result["warnings"].append(
                f"Insufficient disk space. Need ~{_format_bytes(required_space)}, "
                f"but only {_format_bytes(free_space)} available."
            )
        elif required_space > 0 and free_space < required_space * 2:
            result["warnings"].append(
                f"Low disk space warning: {_format_bytes(free_space)} available."
            )
    except Exception:
        result["details"]["free_disk_space"] = "unknown"

    # Estimate RAM requirements (rough guide: model size * 1.2 for inference)
    # This is approximate - actual requirements vary by quantization
    if model_size_bytes > 0:
        estimated_ram = model_size_bytes * 1.2
        result["details"]["estimated_ram"] = _format_bytes(int(estimated_ram))

        # Try to get system RAM (works on most systems)
        try:
            import subprocess
            if os.name == 'posix':  # Linux/macOS
                if os.path.exists('/proc/meminfo'):  # Linux
                    with open('/proc/meminfo', 'r') as f:
                        for line in f:
                            if line.startswith('MemTotal:'):
                                total_ram = int(line.split()[1]) * 1024  # Convert KB to bytes
                                break
                else:  # macOS
                    output = subprocess.check_output(['sysctl', '-n', 'hw.memsize'], text=True)
                    total_ram = int(output.strip())

                result["details"]["total_ram"] = _format_bytes(total_ram)

                if estimated_ram > total_ram * 0.8:
                    result["can_run"] = False
                    result["warnings"].append(
                        f"Model may be too large for your system. "
                        f"Requires ~{_format_bytes(int(estimated_ram))} RAM, "
                        f"but system has {_format_bytes(total_ram)}."
                    )
                elif estimated_ram > total_ram * 0.5:
                    result["warnings"].append(
                        f"Model will use significant RAM (~{_format_bytes(int(estimated_ram))})."
                    )
        except Exception:
            result["details"]["total_ram"] = "unknown"
            # If we can't check RAM, warn for large models
            if model_size_bytes > 8 * 1024 ** 3:  # > 8GB models
                result["warnings"].append(
                    f"Large model (~{size_estimate}). Ensure you have sufficient RAM."
                )

    return result


def pull_ollama_model(model: str, host: str = "localhost", port: int = 11434, auto_confirm: bool = False) -> bool:
    """
    Pull/download a model in Ollama.

    Args:
        model: Model name to pull (e.g., "llama3.2", "mistral")
        host: Hostname where Ollama is running
        port: Port number
        auto_confirm: If True, skip confirmation prompt

    Returns:
        True if model was pulled successfully, False otherwise
    """
    # Get size estimate and check system resources
    size_estimate = get_ollama_model_size_estimate(model)
    resources = check_system_resources(model)

    print(f"\n{'='*60}")
    print(f"  Model '{model}' not found locally")
    print(f"{'='*60}")
    print(f"  Model size:      {size_estimate}")
    if resources["details"].get("estimated_ram"):
        print(f"  RAM required:    ~{resources['details']['estimated_ram']}")
    if resources["details"].get("free_disk_space"):
        print(f"  Free disk space: {resources['details']['free_disk_space']}")
    if resources["details"].get("total_ram"):
        print(f"  System RAM:      {resources['details']['total_ram']}")

    # Show warnings
    if resources["warnings"]:
        print(f"\n  {'!'*50}")
        for warning in resources["warnings"]:
            print(f"  Warning: {warning}")
        print(f"  {'!'*50}")

    # Block if can't download
    if not resources["can_download"]:
        print(f"\n  Cannot download: insufficient disk space.")
        print(f"  Free up disk space and try again.")
        return False

    # Warn but allow if can't run (user might want to try anyway)
    if not resources["can_run"]:
        print(f"\n  Warning: Model may not run on this system.")
        print(f"  Consider a smaller model variant (e.g., '{model}:1b' or '{model}:3b').")

    print(f"{'='*60}")

    # Ask for confirmation
    if not auto_confirm:
        try:
            if not resources["can_run"]:
                prompt = f"\n  Download anyway? [y/N]: "
            else:
                prompt = f"\n  Download '{model}'? [y/N]: "
            response = input(prompt).strip().lower()
            if response not in ['y', 'yes']:
                print("  Download cancelled.")
                return False
        except (EOFError, KeyboardInterrupt):
            print("\n  Download cancelled.")
            return False

    print(f"\n  Downloading from Ollama registry...")
    print(f"  (Press Ctrl+C to cancel)\n")

    try:
        # Ollama pull endpoint streams the response
        response = requests.post(
            f"http://{host}:{port}/api/pull",
            json={"name": model},
            stream=True,
            timeout=None  # No timeout - large models can take a while
        )

        if response.status_code != 200:
            print(f"Failed to pull model: HTTP {response.status_code}")
            return False

        # Process streaming response to show progress
        last_status = ""
        total_size_shown = False

        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line)
                    status = data.get("status", "")

                    # Show progress for downloads
                    if "completed" in data and "total" in data:
                        completed = data["completed"]
                        total = data["total"]
                        pct = (completed / total * 100) if total > 0 else 0

                        # Show actual total size on first progress update
                        if not total_size_shown and total > 0:
                            print(f"  Actual size: {_format_bytes(total)}")
                            total_size_shown = True

                        print(f"\r  {status}: {pct:.1f}% ({_format_bytes(completed)}/{_format_bytes(total)})", end="", flush=True)
                    elif status != last_status:
                        if last_status and "completed" in str(last_status):
                            print()  # newline after progress bar
                        print(f"  {status}")
                        last_status = status

                    # Check for errors
                    if "error" in data:
                        print(f"\n  Error: {data['error']}")
                        return False

                except json.JSONDecodeError:
                    continue

        print(f"\n  Model '{model}' downloaded successfully!")
        return True

    except KeyboardInterrupt:
        print(f"\n\n  Download cancelled by user.")
        return False
    except requests.exceptions.Timeout:
        print(f"\n  Timeout while downloading model '{model}'.")
        print(f"  Try again or download manually: ollama pull {model}")
        return False
    except requests.exceptions.RequestException as e:
        print(f"\n  Error pulling model: {e}")
        return False


# =============================================================================
# Claude Code CLI Functions
# =============================================================================

def check_claude_cli_available():
    """Check if the Claude CLI (claude) is installed and available on PATH."""
    import shutil
    return shutil.which("claude") is not None
