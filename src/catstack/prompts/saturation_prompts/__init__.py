"""Saturation prompts: ten wording variants of the survey classification prompt.

Variant "v01" is cat-llm 0.0.97's survey prompt exactly as cat-stack renders it
with prompt_style="catllm_0.0.97" (the default in cat-survey >= 0.4.0 and cat-llm
>= 3.5.0). Variants "v02"-"v10" reword only the surface of four pieces:

    context        the opening line that states the survey question
    categorize     the "Categorize this survey response ..." line
    json           the answer-format instruction
    (single-label versions of categorize and json, for single-label tasks)

Everything else is held fixed in every variant: the order of the parts, the
numbered category list, the examples, any system prompt (prepended exactly as
cat-stack does), and the answer contract: a JSON object keyed by category
number, with 1 for a category that applies and 0 otherwise (single-label: "1"
for the one best category, "0" for the rest). The variants are meant to change
wording only, never what is asked; whether they change outcomes is the thing a
saturation run measures.

Each variant is its own file in this folder, v01.json ... v10.json, read in file
order at import; VARIANT_IDS lists them, so a caller can loop:

    for vid in VARIANT_IDS:
        messages = render(vid, ...)

Usage:
    from catstack.prompts.saturation_prompts import render, VARIANT_IDS
    messages = render("v03", response_text=..., categories_str=...,
                      survey_question=..., multi_label=True)

Chain-of-thought, step-back and context_prompt are not supported here; the
variants cover the plain classification prompt only.
"""

from .variants import FIELDS, VARIANT_IDS, VARIANTS, load_variants, render

__all__ = ["FIELDS", "VARIANTS", "VARIANT_IDS", "load_variants", "render"]
