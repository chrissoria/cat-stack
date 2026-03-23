"""
Summarization functions for CatLLM.

This module provides unified summarization for text and PDF inputs,
supporting both single-model and multi-model (ensemble) summarization.
"""

import warnings

__all__ = [
    # Main entry point
    "summarize",
    # Ensemble function
    "summarize_ensemble",
]

# Import provider infrastructure
from ._providers import (
    UnifiedLLMClient,
    detect_provider,
)

# Import the implementation functions from existing modules
from .text_functions_ensemble import (
    summarize_ensemble,
)


def summarize(
    input_data,
    api_key: str = None,
    description: str = "",
    instructions: str = "",
    format: str = "paragraph",
    max_length: int = None,
    focus: str = None,
    user_model: str = "gpt-4o",
    model_source: str = "auto",
    mode: str = "image",
    input_mode: str = None,
    input_type: str = "auto",
    pdf_dpi: int = 150,
    creativity: float = None,
    thinking_budget: int = 0,
    chain_of_thought: bool = True,
    context_prompt: bool = False,
    step_back_prompt: bool = False,
    filename: str = None,
    save_directory: str = None,
    progress_callback=None,
    models: list = None,
    max_workers: int = None,
    parallel: bool = None,
    auto_download: bool = False,
    # Robustness parameters
    safety: bool = False,
    max_retries: int = 5,
    batch_retries: int = 2,
    retry_delay: float = 1.0,
    row_delay: float = 0.0,
    fail_strategy: str = "partial",
    # Batch mode parameters
    batch_mode: bool = False,
    batch_poll_interval: float = 30.0,
    batch_timeout: float = 86400.0,
):
    """
    Summarize text or PDF data using LLMs.

    Supports single-model and multi-model (ensemble) summarization. In multi-model
    mode, summaries from all models are synthesized into a consensus summary.
    Input type is auto-detected from the data (text strings or PDF paths).

    Args:
        input_data: Data to summarize. Can be:
            - Text: list of strings, pandas Series, or single string
            - PDF: directory path, single PDF path, or list of PDF paths
        api_key (str): API key for the model provider (single-model mode)
        description (str): Description of what the content contains (provides context)
        instructions (str): Specific summarization instructions. When used with
            format, these are appended as additional instructions. Default "".
        format (str): Output format for the summary. Default "paragraph".
            - "paragraph": Flowing prose summary (default)
            - "bullets": Bullet-point list of key points
            - "one-liner": Single-sentence summary
            - "structured": Labeled sections (What, Who, Why, Impact)
            - "few-paragraphs": 2-4 paragraph summary with context and details
            - "single-page": Single-page summary, thorough but concise
            - "few-pages": Thorough multi-page summary covering all significant points
            - "report": Full-page structured report with headings (Overview,
              Background, Key Provisions, Stakeholders/Impact, Implementation)
            - "detailed-report": Exhaustive report enumerating every provision,
              with an additional Details section for exceptions and cross-references
        max_length (int): Maximum summary length in words
        focus (str): What to focus on (e.g., "main arguments", "emotional content")
        user_model (str): Model to use (default "gpt-4o")
        model_source (str): Provider - "auto", "openai", "anthropic", "google", etc.
        input_mode (str): What you want the model to do with the input. Default None.
            - None: Auto-select based on file type (text→"text", image→"visual",
              pdf→uses mode param or "visual")
            - "text": Summarize text content, regardless of source format. For images
              and scanned PDFs, uses LLM-based OCR to extract text first.
            - "visual": Summarize visual features of images/rendered PDFs.
        input_type (str): File type filter. Default "auto" (auto-detect).
            Options: "auto", "pdf", "image", "text", "docx"
        mode (str): PDF processing mode (legacy, use input_mode instead):
            - "image" (default): Render pages as images
            - "text": Extract text only
            - "both": Send both image and extracted text
        pdf_dpi (int): DPI for PDF page rendering (default 150)
        creativity (float): Temperature setting (None uses provider default)
        thinking_budget (int): Token budget for extended thinking/reasoning.
            Provider-specific: Google (thinkingConfig), OpenAI (reasoning_effort),
            Anthropic (extended thinking). Default 0 (disabled).
        chain_of_thought (bool): Enable step-by-step reasoning (default True)
        context_prompt (bool): Add expert context prefix
        step_back_prompt (bool): Enable step-back prompting
        filename (str): Output CSV filename
        save_directory (str): Directory to save results
        progress_callback: Optional callback for progress updates
        models (list): For multi-model mode, list of (model, provider, api_key) tuples
        max_workers (int): Max parallel workers for API calls. None = auto.
        parallel (bool): Controls concurrent vs sequential model execution.
            - None (default): auto-detect (sequential for all-Ollama, parallel otherwise)
            - True: force parallel execution
            - False: force sequential execution
        auto_download (bool): Auto-download missing Ollama models. Default False.
        safety (bool): If True, saves progress after each item. Requires filename.
        max_retries (int): Max retries per API call. Default 5.
        batch_retries (int): Max retries for batch-level failures. Default 2.
        retry_delay (float): Delay between retries in seconds. Default 1.0.
        row_delay (float): Delay in seconds between processing each row. Default 0.0.
        fail_strategy (str): How to handle failures - "partial" (default) or "strict".
        batch_mode (bool): If True, use async batch API (50% cost savings).
            Supported providers: openai, anthropic, google, mistral, xai.
            Not compatible with PDF input.
        batch_poll_interval (float): Seconds between batch status checks. Default 30.
        batch_timeout (float): Max seconds to wait for batch completion. Default 86400 (24h).

    Returns:
        pd.DataFrame: Results with summary column(s):
            - input_data: Original text or page label (for PDFs)
            - summary: Generated summary (or consensus for multi-model)
            - summary_<model>: Per-model summaries (multi-model only)
            - processing_status: "success", "error", "skipped"
            - failed_models: Comma-separated list (multi-model only)
            - pdf_path: Path to source PDF (PDF mode only)
            - page_index: Page number, 0-indexed (PDF mode only)

    Examples:
        >>> import cat_stack as cat
        >>>
        >>> # Single model text summarization
        >>> results = cat.summarize(
        ...     input_data=df['responses'],
        ...     description="Customer feedback",
        ...     api_key="your-api-key"
        ... )
        >>>
        >>> # PDF summarization (auto-detected)
        >>> results = cat.summarize(
        ...     input_data="/path/to/pdfs/",
        ...     description="Research papers",
        ...     mode="image",
        ...     api_key="your-api-key"
        ... )
        >>>
        >>> # With safety saves and row delay
        >>> results = cat.summarize(
        ...     input_data=df['responses'],
        ...     description="Customer feedback",
        ...     api_key="your-api-key",
        ...     safety=True,
        ...     filename="results.csv",
        ...     row_delay=1.0,
        ... )
        >>>
        >>> # Batch mode (50% cost savings)
        >>> results = cat.summarize(
        ...     input_data=df['responses'],
        ...     description="Customer feedback",
        ...     api_key="your-api-key",
        ...     batch_mode=True,
        ...     filename="batch_results.csv",
        ... )
        >>>
        >>> # Multi-model with synthesis
        >>> results = cat.summarize(
        ...     input_data=df['responses'],
        ...     models=[
        ...         ("gpt-4o", "openai", "sk-..."),
        ...         ("claude-sonnet-4-5-20250929", "anthropic", "sk-ant-..."),
        ...     ],
        ... )
    """
    # =========================================================================
    # Resolve format → instructions + max_length defaults
    # =========================================================================
    _FORMAT_PRESETS = {
        "paragraph": {
            "instructions": "Write a concise summary in paragraph form.",
            "max_length": None,
        },
        "bullets": {
            "instructions": (
                "Summarize as a bullet-point list. Each bullet should capture "
                "one key point. Use '- ' prefix for each bullet."
            ),
            "max_length": None,
        },
        "one-liner": {
            "instructions": "Summarize in a single sentence.",
            "max_length": 40,
        },
        "structured": {
            "instructions": (
                "Summarize using these labeled sections:\n"
                "- What: What does this do or say?\n"
                "- Who: Who is affected or involved?\n"
                "- Why: What is the motivation or purpose?\n"
                "- Impact: What are the key consequences or effects?"
            ),
            "max_length": None,
        },
        "few-paragraphs": {
            "instructions": (
                "Write a summary of 2-4 paragraphs. The first paragraph should "
                "state the main point. Subsequent paragraphs should cover key "
                "details, context, and implications."
            ),
            "max_length": 300,
        },
        "single-page": {
            "instructions": (
                "Write a single-page summary. Cover the main points, key details, "
                "and implications in a well-organized format. Use paragraph breaks "
                "between topics. Be thorough but fit everything on one page."
            ),
            "max_length": 500,
        },
        "few-pages": {
            "instructions": (
                "Write a thorough multi-page summary. Cover all significant points "
                "in detail. Use clear paragraph breaks between topics. Include "
                "background context, specific provisions or arguments, affected "
                "parties, and implications. Be comprehensive but well-organized."
            ),
            "max_length": 1500,
        },
        "detailed-report": {
            "instructions": (
                "Write an exhaustive, detailed report covering every significant "
                "aspect of this document. Use clear headings and be as thorough "
                "as possible — do not omit details.\n\n"
                "## Overview\n"
                "An executive summary (2-3 sentences).\n\n"
                "## Background and Context\n"
                "What is the background? What problem or situation prompted this? "
                "Include relevant history and prior actions.\n\n"
                "## Key Provisions\n"
                "Detail ALL main provisions, requirements, or arguments. "
                "Be specific about numbers, dates, names, and conditions. "
                "Do not summarize — enumerate each provision.\n\n"
                "## Stakeholders and Impact\n"
                "Who is affected? What are the expected consequences? "
                "Include both intended effects and potential concerns.\n\n"
                "## Implementation\n"
                "How will this be implemented? What is the timeline? "
                "Are there enforcement mechanisms or milestones?\n\n"
                "## Additional Details\n"
                "Any other noteworthy details, exceptions, amendments, "
                "or cross-references not covered above."
            ),
            "max_length": 3000,
        },
        # Keep "report" as alias for backward compat
        "report": {
            "instructions": (
                "Write a comprehensive full-page report covering the following sections. "
                "Use clear headings and be thorough.\n\n"
                "## Overview\n"
                "A brief executive summary (2-3 sentences).\n\n"
                "## Background and Context\n"
                "What is the background? What problem or situation prompted this? "
                "Include relevant history and prior actions.\n\n"
                "## Key Provisions\n"
                "Detail the main provisions, requirements, or arguments. "
                "Be specific about numbers, dates, names, and conditions.\n\n"
                "## Stakeholders and Impact\n"
                "Who is affected? What are the expected consequences? "
                "Include both intended effects and potential concerns.\n\n"
                "## Implementation\n"
                "How will this be implemented? What is the timeline? "
                "Are there enforcement mechanisms or milestones?"
            ),
            "max_length": 800,
        },
    }

    format_lower = format.lower() if format else "paragraph"
    if format_lower not in _FORMAT_PRESETS:
        valid = ", ".join(f'"{k}"' for k in _FORMAT_PRESETS)
        raise ValueError(f"format must be one of {valid}, got '{format}'")

    preset = _FORMAT_PRESETS[format_lower]

    # Format instructions are prepended to any user-provided instructions
    if not instructions:
        instructions = preset["instructions"]
    else:
        instructions = f"{preset['instructions']}\n\nAdditional instructions: {instructions}"

    # Use format's max_length as default only if user didn't specify one
    if max_length is None and preset["max_length"] is not None:
        max_length = preset["max_length"]

    # Map mode to pdf_mode
    pdf_mode = mode if mode in ("image", "text", "both") else "image"

    # =========================================================================
    # Batch mode — bypass summarize_ensemble entirely
    # =========================================================================
    if batch_mode:
        from ._batch import UNSUPPORTED_BATCH_PROVIDERS, run_batch_summarize
        from .text_functions_ensemble import _detect_input_type, prepare_model_configs

        # Guard: text input only
        detected_type = _detect_input_type(input_data)
        if detected_type == "pdf":
            raise ValueError(
                "batch_mode=True only supports text input, but detected input type is 'pdf'. "
                "Set batch_mode=False for PDF summarization."
            )

        # Warn if progress_callback was provided (incompatible with batch)
        if progress_callback is not None:
            print(
                "[CatLLM] WARNING: progress_callback is not available in batch_mode "
                "(no per-item progress until the job completes). Ignoring callback."
            )

        # Build models list
        if models is None:
            batch_models = [(user_model, model_source, api_key)]
        else:
            batch_models = models

        model_configs = prepare_model_configs(batch_models)
        items = list(input_data) if not isinstance(input_data, list) else input_data

        prompt_params = {
            "input_description": description,
            "summary_instructions": instructions,
            "max_length": max_length,
            "focus": focus,
            "chain_of_thought": chain_of_thought,
            "context_prompt": context_prompt,
            "step_back_prompt": step_back_prompt,
            "stepback_insights": {},
            "creativity": creativity,
        }

        if len(batch_models) == 1:
            cfg = model_configs[0]
            if cfg["provider"] in UNSUPPORTED_BATCH_PROVIDERS:
                raise ValueError(
                    f"batch_mode=True is not supported for provider '{cfg['provider']}'. "
                    f"Supported providers: openai, anthropic, google, mistral, xai."
                )
            return run_batch_summarize(
                items=items,
                cfg=cfg,
                prompt_params=prompt_params,
                filename=filename,
                save_directory=save_directory,
                batch_poll_interval=batch_poll_interval,
                batch_timeout=batch_timeout,
                fail_strategy=fail_strategy,
            )

        # Ensemble batch path
        print(
            "[CatLLM] NOTE: batch_mode=True with multiple models is experimental. "
            "Each model submits a separate batch job concurrently."
        )
        from ._batch import run_batch_ensemble_summarize
        prompt_params_per_model = {
            cfg["model"]: {
                **prompt_params,
                "creativity": cfg["creativity"] if cfg["creativity"] is not None else creativity,
            }
            for cfg in model_configs
        }
        return run_batch_ensemble_summarize(
            items=items,
            model_configs=model_configs,
            prompt_params_per_model=prompt_params_per_model,
            fail_strategy=fail_strategy,
            filename=filename,
            save_directory=save_directory,
            batch_poll_interval=batch_poll_interval,
            batch_timeout=batch_timeout,
            max_retries=max_retries,
        )

    return summarize_ensemble(
        input_data=input_data,
        api_key=api_key,
        input_description=description,
        summary_instructions=instructions,
        max_length=max_length,
        focus=focus,
        user_model=user_model,
        model_source=model_source,
        pdf_mode=pdf_mode,
        pdf_dpi=pdf_dpi,
        creativity=creativity,
        thinking_budget=thinking_budget,
        chain_of_thought=chain_of_thought,
        context_prompt=context_prompt,
        step_back_prompt=step_back_prompt,
        max_retries=max_retries,
        batch_retries=batch_retries,
        retry_delay=retry_delay,
        row_delay=row_delay,
        fail_strategy=fail_strategy,
        safety=safety,
        filename=filename,
        save_directory=save_directory,
        progress_callback=progress_callback,
        models=models,
        max_workers=max_workers,
        parallel=parallel,
        auto_download=auto_download,
        input_mode=input_mode,
        input_type=input_type,
    )
