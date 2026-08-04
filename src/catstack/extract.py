"""
Category extraction functions for CatLLM.

This module provides unified category extraction from text, image, and PDF inputs.
"""

import warnings

__all__ = [
    # Main entry point
    "extract",
    # Input-specific functions (for backward compatibility)
    "explore_common_categories",
    "explore_corpus",
    "explore_image_categories",
    "explore_pdf_categories",
]

# Import provider infrastructure
from ._providers import (
    UnifiedLLMClient,
    detect_provider,
)

# Import the implementation functions from existing modules
from .text_functions import (
    explore_common_categories,
    explore_corpus,
)

from .image_functions import (
    explore_image_categories,
)

from .pdf_functions import (
    explore_pdf_categories,
)

from .collapse_themes import collapse_themes


def extract(
    input_data,
    api_key,
    input_type="auto",
    description="",
    survey_question=None,
    max_categories=12,
    categories_per_chunk=10,
    divisions=12,
    user_model="gpt-4o",
    creativity=None,
    specificity="broad",
    research_question=None,
    mode="text",
    filename=None,
    model_source="auto",
    iterations=8,
    random_state=None,
    focus=None,
    progress_callback=None,
    chunk_delay: float = 0.0,
    auto_download: bool = False,
    input_mode=None,
    domain: str = "neutral",
    engine: str = "collapse",
    max_workers: int = 1,
    collapse_kwargs: dict = None,
):
    """
    Unified category extraction function for text, image, and PDF inputs.

    This function dispatches to the appropriate specialized explore function
    based on the `input_type` parameter, providing a single entry point for
    discovering categories in your data.

    Args:
        input_data: The data to explore. Can be:
            - For text: list of text responses or pandas Series
            - For image: directory path, single file, or list of image paths
            - For pdf: directory path, single file, or list of PDF paths
        api_key (str): API key for the model provider.
        input_type (str): Type of input data. Options:
            - "auto" (default): Auto-detect from file extensions
            - "text": Text responses
            - "image": Image files
            - "pdf": PDF documents
        description (str): Description of the data context. Content-neutral —
            for survey responses this is the question that was asked; for
            documents or posts this describes what the content is about.
        survey_question (str): Deprecated alias for `description`. Pass
            `description=` instead. If provided, emits a DeprecationWarning
            and is mirrored to `description` when `description` is empty.
        max_categories (int): Maximum number of final categories to return.
        categories_per_chunk (int): Categories to extract per chunk.
        divisions (int): Number of chunks to divide data into.
        user_model (str): Model name to use. Default "gpt-4o".
        creativity (float): Temperature setting. None uses model default.
        specificity (str): "broad" or "specific" category granularity.
        research_question (str): Optional research context.
        mode (str): Processing mode:
            - For text: Not used
            - For image: "image" (default) or "both"
            - For pdf: "text" (default), "image", or "both"
        filename (str): Optional CSV filename to save results.
        model_source (str): Provider - "auto", "openai", "anthropic", "google",
            "mistral", "huggingface", "xai", "ollama".
        iterations (int): Number of passes over the data.
        random_state (int): Random seed for reproducibility.
        focus (str): Optional focus instruction for category extraction (e.g.,
            "decisions to move", "emotional responses"). When provided, the model
            will prioritize extracting categories related to this focus.
        progress_callback (callable): Optional callback function for progress updates.
            Called as progress_callback(current_step, total_steps, step_label).
        chunk_delay (float): Delay in seconds between API calls to avoid rate
            limits. Default 0.0 (no delay).
        auto_download (bool): If True, automatically download missing Ollama
            models without prompting. Default False.
        engine (str): Consolidation engine for text input. "collapse" (default):
            run raw extraction (as explore() does) and consolidate the FULL
            inventory with collapse_themes() — semantic pre-clean, quality-
            controlled passes, then a count-guided reduction to at most
            `max_categories`. "legacy": the pre-2.5 single merge call, which
            truncates the inventory to the top max_categories*3 labels by
            exact-string count before merging; kept for reproducing older runs.
        max_workers (int): Parallel API calls for both the extraction chunks and
            the consolidation batches (text engine="collapse" only; extraction
            also honors it under "legacy"). Default 1.
        collapse_kwargs (dict): Optional overrides forwarded to collapse_themes()
            when engine="collapse" — e.g. {"prune": True} or
            {"passes": 2, "aggressive": False}. Defaults applied first:
            passes="auto", aggressive=True. `top_n` cannot be overridden here;
            it is always max_categories.

    Returns:
        dict with keys:
            - counts_df: DataFrame of categories with counts
            - top_categories: List of top category names
            - raw_top_text: Raw model output from final merge step ("" when
              engine="collapse", which has no single merge reply)

    Examples:
        >>> import catstack as cat
        >>>
        >>> # Extract categories from text responses
        >>> results = cat.extract(
        ...     input_data=df['responses'],
        ...     description="Why did you move?",
        ...     api_key="your-api-key"
        ... )
        >>> print(results['top_categories'])
        >>>
        >>> # Extract categories from images
        >>> results = cat.extract(
        ...     input_data="/path/to/images/",
        ...     description="Product photos",
        ...     input_type="image",
        ...     api_key="your-api-key"
        ... )
        >>>
        >>> # Extract categories from PDFs
        >>> results = cat.extract(
        ...     input_data="/path/to/pdfs/",
        ...     description="Research papers",
        ...     input_type="pdf",
        ...     mode="text",
        ...     api_key="your-api-key"
        ... )
    """
    input_type = input_type.lower().rstrip('s')  # Normalize: "texts" -> "text", "images" -> "image", "pdfs" -> "pdf"

    # Auto-detect input type if set to "auto"
    if input_type == "auto":
        from .text_functions_ensemble import _detect_input_type
        input_type = _detect_input_type(input_data)
        # docx → text for extraction purposes
        if input_type == "docx":
            input_type = "text"

    # `description` is the canonical content-neutral name. `survey_question`
    # is a soft-deprecated alias kept working for legacy callers
    # (notably cat-survey and pre-rename notebooks).
    if survey_question:
        warnings.warn(
            "`survey_question=` is deprecated in extract(); use `description=` "
            "instead. The value will be mirrored to `description` for now.",
            DeprecationWarning,
            stacklevel=2,
        )
        if not description:
            description = survey_question
    resolved_description = description or ""

    if input_type == "text":
        if engine not in ("collapse", "legacy"):
            raise ValueError(f"engine must be 'collapse' or 'legacy', got '{engine}'")

        if engine == "legacy":
            return explore_common_categories(
                input_data=input_data,
                api_key=api_key,
                survey_question=resolved_description,
                max_categories=max_categories,
                categories_per_chunk=categories_per_chunk,
                divisions=divisions,
                user_model=user_model,
                creativity=creativity,
                specificity=specificity,
                research_question=research_question,
                filename=filename,
                model_source=model_source,
                iterations=iterations,
                random_state=random_state,
                focus=focus,
                progress_callback=progress_callback,
                chunk_delay=chunk_delay,
                auto_download=auto_download,
                max_workers=max_workers,
                domain=domain,
            )

        # engine="collapse": raw extraction (what explore() does), then consolidate
        # the FULL inventory with collapse_themes(). Unlike the legacy merge, no
        # label is truncated away before consolidation, pre-cleaning is semantic
        # (Jaro-Winkler + embeddings) rather than exact-string, and the final
        # count-guided top_n step guarantees at most max_categories categories.
        import pandas as pd

        raw_items = explore_common_categories(
            input_data=input_data,
            api_key=api_key,
            survey_question=resolved_description,
            max_categories=max_categories,
            categories_per_chunk=categories_per_chunk,
            divisions=divisions,
            user_model=user_model,
            creativity=creativity,
            specificity=specificity,
            research_question=research_question,
            filename=None,
            model_source=model_source,
            iterations=iterations,
            random_state=random_state,
            focus=focus,
            progress_callback=progress_callback,
            return_raw=True,
            chunk_delay=chunk_delay,
            auto_download=auto_download,
            max_workers=max_workers,
            domain=domain,
        )

        # Frequency inventory in the same shape the legacy engine returned.
        def _normalize(cat):
            return "/".join(sorted(t.strip().lower() for t in str(cat).split("/")))

        flat = [str(x).strip() for x in raw_items if str(x).strip()]
        if not flat:
            raise ValueError("No categories were extracted from the model responses.")
        inv = pd.DataFrame(flat, columns=["Category"])
        inv["normalized"] = inv["Category"].map(_normalize)
        counts_df = (
            inv.groupby("normalized")
               .agg(Category=("Category", lambda x: x.value_counts().index[0]),
                    counts=("Category", "size"))
               .sort_values("counts", ascending=False)
               .reset_index(drop=True)
        )

        ck = dict(passes="auto", aggressive=True)
        ck.update(collapse_kwargs or {})
        ck["top_n"] = int(max_categories)  # the required N — not overridable
        top = collapse_themes(
            raw_items,
            api_key=api_key,
            description=resolved_description,
            user_model=user_model,
            model_source=model_source,
            creativity=0 if creativity is None else creativity,
            max_workers=max_workers,
            random_state=random_state,
            progress_callback=progress_callback,
            **ck,
        )

        if filename:
            pd.DataFrame({"rank": range(1, len(top) + 1), "category": top}).to_csv(
                filename, index=False)
            print(f"Top {len(top)} categories saved to {filename}")

        return {
            "counts_df": counts_df,
            "top_categories": top,
            "raw_top_text": "",
        }

    elif input_type == "image":
        return explore_image_categories(
            image_input=input_data,
            api_key=api_key,
            image_description=resolved_description,
            max_categories=max_categories,
            categories_per_chunk=categories_per_chunk,
            divisions=divisions,
            user_model=user_model,
            creativity=creativity,
            specificity=specificity,
            research_question=research_question,
            mode=mode if mode in ["image", "both"] else "image",
            filename=filename,
            model_source=model_source,
            iterations=iterations,
            random_state=random_state,
            progress_callback=progress_callback,
        )

    elif input_type == "pdf":
        return explore_pdf_categories(
            pdf_input=input_data,
            api_key=api_key,
            pdf_description=resolved_description,
            max_categories=max_categories,
            categories_per_chunk=categories_per_chunk,
            divisions=divisions,
            user_model=user_model,
            creativity=creativity,
            specificity=specificity,
            research_question=research_question,
            mode=mode if mode in ["text", "image", "both"] else "text",
            filename=filename,
            model_source=model_source,
            iterations=iterations,
            random_state=random_state,
            progress_callback=progress_callback,
        )

    else:
        raise ValueError(
            f"input_type '{input_type}' is not supported. "
            f"Please use one of: 'text', 'image', or 'pdf'.\n\n"
            f"Examples:\n"
            f"  - For text responses or other text data: input_type='text'\n"
            f"  - For image files (.jpg, .png, etc.): input_type='image'\n"
            f"  - For PDF documents: input_type='pdf'"
        )
