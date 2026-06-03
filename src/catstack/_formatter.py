"""
JSON formatter fallback for CatLLM.

Uses a fine-tuned Qwen2.5-0.5B model to convert messy LLM classification
output into valid cat-llm JSON format: {"1":"0","2":"1",...}

The formatter is opt-in via json_formatter=True on classify(). It only runs
when extract_json() produces invalid output — zero cost on the happy path.

Requires: pip install cat-llm[formatter]
"""

import sys

_MERGED_MODEL_REPO = "chrissoria/catllm-json-formatter"

_SYSTEM_PROMPT = (
    "You are a JSON formatter for a text classification pipeline. "
    "You will receive a list of categories (numbered 1 to N) and a raw "
    "classification output from another model. Your job is to convert that "
    'output into the exact JSON format required:\n'
    '{"1":"0","2":"1","3":"0",...}\n\n'
    "Rules:\n"
    '- Keys are 1-indexed strings: "1", "2", ..., "N"\n'
    '- Values are ONLY "0" (category absent) or "1" (category present)\n'
    "- Include ALL N categories, even if absent\n"
    "- Output ONLY the JSON object — no explanation, no markdown, no extra text\n"
    '- If a category\'s presence is ambiguous, default to "0"'
)


def _check_dependencies():
    """Check that torch and transformers are installed."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        raise ImportError(
            "The JSON formatter requires additional dependencies.\n"
            "Install them with: pip install cat-llm[formatter]\n"
            "  (requires: torch, transformers, accelerate)"
        )


def _check_dependencies_installed() -> bool:
    """Pure check — returns True if all formatter deps import successfully.
    No side effects, no install attempts."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        import accelerate  # noqa: F401
        return True
    except ImportError:
        return False


def _install_dependencies(verbose: bool = True) -> bool:
    """Run `pip install` for the formatter deps. Caller is responsible for
    obtaining user consent before calling this — it does not prompt.

    Returns True if deps are importable after install, False otherwise.
    """
    if verbose:
        print("[CatLLM] Installing formatter dependencies (~1.5 GB)…")
    import subprocess
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet",
             "transformers", "torch", "accelerate", "sentencepiece"]
        )
    except subprocess.CalledProcessError as e:
        if verbose:
            print(
                f"[CatLLM] Failed to install formatter dependencies ({e}).\n"
                "  Install manually: pip install 'cat-stack[formatter]'"
            )
        return False
    return _check_dependencies_installed()


def _prompt_formatter_consent(model_label: str = "the current model") -> str:
    """Interactive consent prompt for the JSON formatter fallback.

    Two paths depending on whether the formatter dependencies are already
    installed:
      - Deps installed: asks whether to load the ~1 GB formatter model.
      - Deps missing:   asks whether to download deps (~1.5 GB) AND load.

    Non-TTY contexts (CI, batch scripts, headless notebooks): prints a
    one-time suggestion and returns "declined" without blocking on input.

    Returns "approved" or "declined". On approval with deps missing,
    also installs the deps before returning.
    """
    deps_installed = _check_dependencies_installed()

    if not sys.stdin.isatty():
        if deps_installed:
            print(
                f"\n[CatLLM] Malformed JSON from {model_label}. The JSON "
                "formatter could recover this — pass json_formatter=True "
                "to enable, or json_formatter=False to silence this suggestion."
            )
        else:
            print(
                f"\n[CatLLM] Malformed JSON from {model_label}. The JSON "
                "formatter could recover, but its deps (~1.5 GB) aren't "
                "installed. Run `pip install cat-stack[formatter]` and pass "
                "json_formatter=True to enable, or json_formatter=False to "
                "silence this suggestion."
            )
        return "declined"

    if deps_installed:
        prompt = (
            f"\n[CatLLM] {model_label} produced malformed JSON on the first row.\n"
            "  The JSON formatter can re-format the model's prose output\n"
            "  into valid catstack JSON for this and subsequent rows.\n"
            "    Cost: ~1 GB RAM (one-time load).\n"
            "  Use the formatter for this run? (Y/n): "
        )
    else:
        prompt = (
            f"\n[CatLLM] {model_label} produced malformed JSON on the first row.\n"
            "  The JSON formatter can re-format the model's prose output\n"
            "  into valid catstack JSON for this and subsequent rows.\n"
            "    Cost: ~1.5 GB download (transformers + torch + accelerate)\n"
            "         + ~1 GB RAM (one-time load).\n"
            "  Download deps and use the formatter? (Y/n): "
        )

    try:
        answer = input(prompt).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\n[CatLLM] No input received — skipping formatter.")
        return "declined"

    if answer in ("", "y", "yes"):
        if not deps_installed:
            if not _install_dependencies(verbose=True):
                print("[CatLLM] Continuing without formatter.")
                return "declined"
        return "approved"
    print("[CatLLM] Continuing without formatter.")
    return "declined"


def _ensure_dependencies(verbose: bool = True) -> bool:
    """Back-compat: ensure deps are installed, auto-installing if missing.

    Still used by the explicit `json_formatter=True` path where the user
    has already implicitly consented by passing True. The new
    `json_formatter=None` ("auto") path uses `_prompt_formatter_consent`
    plus `_install_dependencies` directly so the install requires an
    explicit yes.
    """
    if _check_dependencies_installed():
        return True

    if verbose:
        print(
            "\n[CatLLM] JSON formatter dependencies (transformers, torch, "
            "accelerate)\n"
            "  are not installed. Installing now (~1.5 GB download; one-time).\n"
            "  To skip this and disable the formatter, pass json_formatter=False."
        )

    return _install_dependencies(verbose=verbose)


def _is_model_cached() -> bool:
    """Check if the merged model is already in the HuggingFace cache."""
    try:
        from huggingface_hub import try_to_load_from_cache
        result = try_to_load_from_cache(_MERGED_MODEL_REPO, "config.json")
        return result is not None and not isinstance(result, type(None))
    except Exception:
        return False


def ensure_formatter_available() -> bool:
    """
    Ensure the formatter model and its Python dependencies are available.

    Auto-installs deps (transformers/torch/accelerate, ~1.5 GB) on first use
    and auto-downloads the formatter model (~1 GB) from HuggingFace on first
    use. Both events print a clear warning to the console; neither prompts
    interactively, so this function is safe to call from Rscript / non-TTY
    sessions.

    Returns:
        True if the formatter is ready to use, False on install failure.
    """
    if not _ensure_dependencies():
        return False

    if _is_model_cached():
        return True

    print(
        "\n[CatLLM] Downloading JSON formatter model (~1 GB) from\n"
        f"  HuggingFace Hub ({_MERGED_MODEL_REPO}).\n"
        "  This is a one-time download — the model is cached locally after."
    )
    return True  # actual download happens in load_formatter()


def load_formatter(device=None):
    """
    Load the merged formatter model and tokenizer.

    Args:
        device: Target device. None = auto-detect (CUDA > CPU; MPS skipped).

    Returns:
        Tuple of (model, tokenizer, device_str).
    """
    _check_dependencies()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            # Skip MPS — known PEFT/generation crash issues
            device = "cpu"

    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"[CatLLM] Loading JSON formatter on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(
        _MERGED_MODEL_REPO, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        _MERGED_MODEL_REPO, dtype=dtype, trust_remote_code=True
    )
    model = model.to(device)
    model.eval()

    print("[CatLLM] JSON formatter ready.")
    return model, tokenizer, device


def run_formatter(raw_output, categories, model, tokenizer, device):
    """
    Run the formatter model to fix malformed classification JSON.

    Args:
        raw_output: The raw (messy) output from the classification LLM.
        categories: List of category names.
        model: The loaded formatter model.
        tokenizer: The loaded tokenizer.
        device: Device string ("cuda" or "cpu").

    Returns:
        The formatter's output string (caller should run extract_json on it).
    """
    import torch

    # Build category list
    cat_lines = "\n".join(
        f"{i + 1}. {cat}" for i, cat in enumerate(categories)
    )
    user_msg = f"Categories:\n{cat_lines}\n\nRaw classification output:\n{raw_output}"

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only newly generated tokens
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
