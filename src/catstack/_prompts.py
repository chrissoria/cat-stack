"""
Domain-keyed prompt registry.

cat-stack's extract() and explore() pipelines use two LLM prompts: a
*first-pass* per-chunk extraction prompt and a *second-pass* semantic
*merge* prompt. The wording of each is domain-shaped — survey responses
read differently than social-media posts or academic papers.

This module centralises every variant in one place. Domain-specific
sub-packages (cat-survey, cat-vader, cat-ademic, cat-pol, cat-web) call
catstack.extract/explore with `domain="<key>"` to select the appropriate
variant. The default is `"neutral"`, which contains no domain-shaped
language so direct catstack callers get generic prompts.

A domain only needs to override the slots that genuinely differ from
neutral; unspecified slots fall back to neutral via `get_prompt`.

Template placeholders:

  first_pass — {categories_per_chunk} {specificity} {context}
               {focus_text} {items_blob}
  merge      — {context} {max_categories} {name_instruction}
               {seed_with_counts}
"""

# Generic, domain-neutral templates. Used directly when the caller does
# not pass a domain, and used as the fallback for any slot a domain does
# not override.
_NEUTRAL_FIRST_PASS = (
    'Identify {categories_per_chunk} {specificity} categories present in '
    'the following texts about: "{context}".{focus_text} '
    "Items are separated by semicolons. "
    "Items are within triple backticks: ```{items_blob}``` "
    "Number your categories from 1 through {categories_per_chunk} and "
    "provide concise labels only (no descriptions)."
)

_NEUTRAL_MERGE = """
You are consolidating categories extracted from a collection of texts about: "{context}"

Task: Reduce to {max_categories} categories.

Step 1 — Cluster: Group the categories below into clusters where each cluster represents ONE distinct concept or theme. Categories that describe the same concept using different words or from different angles belong in the same cluster. For example, a category about "battery life" and a category about "charge duration" likely belong together if they reflect the same underlying concept.

Step 2 — Label: For each cluster, choose the single label that best captures the shared meaning. {name_instruction}

Step 3 — Rank: Sum the frequency counts within each cluster. Output the top {max_categories} clusters by total count.

Categories (sorted by extraction frequency):
{seed_with_counts}

Return ONLY a numbered list of {max_categories} categories.
""".strip()


# Survey: the historical cat-stack prompt, verbatim. "respondent" /
# "reason" language preserved.
_SURVEY_FIRST_PASS = (
    'Identify {categories_per_chunk} {specificity} categories of responses '
    'to the question "{context}" in the following list of responses.{focus_text} '
    "Responses are separated by semicolons. "
    "Responses are within triple backticks: ```{items_blob}``` "
    "Number your categories from 1 through {categories_per_chunk} and "
    "provide concise labels only (no descriptions)."
)

_SURVEY_MERGE = """
You are consolidating categories extracted from survey responses to: "{context}"

Task: Reduce to {max_categories} categories.

Step 1 — Cluster: Group the categories below into clusters where each cluster represents ONE distinct reason a respondent might give. Categories that describe the same reason using different words or from different angles belong in the same cluster. For example, a category about relationship quality and a category about emotional closeness likely belong together if they reflect the same underlying reason.

Step 2 — Label: For each cluster, choose the single label that best captures the shared meaning. {name_instruction}

Step 3 — Rank: Sum the frequency counts within each cluster. Output the top {max_categories} clusters by total count.

Categories (sorted by extraction frequency):
{seed_with_counts}

Return ONLY a numbered list of {max_categories} categories.
""".strip()


_SOCIAL_MERGE = """
You are consolidating categories extracted from social-media posts about: "{context}"

Task: Reduce to {max_categories} categories.

Step 1 — Cluster: Group the categories below into clusters where each cluster represents ONE distinct topic, sentiment, or behaviour expressed in the posts. Categories that describe the same underlying message using different wording, slang, or hashtags belong in the same cluster. For example, a category about "product praise" and a category about "positive recommendation" likely belong together if they reflect the same underlying sentiment.

Step 2 — Label: For each cluster, choose the single label that best captures the shared meaning. {name_instruction}

Step 3 — Rank: Sum the frequency counts within each cluster. Output the top {max_categories} clusters by total count.

Categories (sorted by extraction frequency):
{seed_with_counts}

Return ONLY a numbered list of {max_categories} categories.
""".strip()


_ACADEMIC_MERGE = """
You are consolidating categories extracted from academic texts about: "{context}"

Task: Reduce to {max_categories} categories.

Step 1 — Cluster: Group the categories below into clusters where each cluster represents ONE distinct research theme, method, or finding. Categories that describe the same scholarly concept using different terminology or framings belong in the same cluster. For example, a category about "longitudinal cohort analysis" and a category about "panel data study design" likely belong together if they reflect the same underlying research approach.

Step 2 — Label: For each cluster, choose the single label that best captures the shared meaning. {name_instruction}

Step 3 — Rank: Sum the frequency counts within each cluster. Output the top {max_categories} clusters by total count.

Categories (sorted by extraction frequency):
{seed_with_counts}

Return ONLY a numbered list of {max_categories} categories.
""".strip()


_POLICY_MERGE = """
You are consolidating categories extracted from policy documents about: "{context}"

Task: Reduce to {max_categories} categories.

Step 1 — Cluster: Group the categories below into clusters where each cluster represents ONE distinct policy area, provision, or government action. Categories that describe the same provision using different statutory language or framings belong in the same cluster. For example, a category about "Medicaid eligibility expansion" and a category about "low-income healthcare coverage extension" likely belong together if they reflect the same underlying policy mechanism.

Step 2 — Label: For each cluster, choose the single label that best captures the policy area or provision. {name_instruction}

Step 3 — Rank: Sum the frequency counts within each cluster. Output the top {max_categories} clusters by total count.

Categories (sorted by extraction frequency):
{seed_with_counts}

Return ONLY a numbered list of {max_categories} categories.
""".strip()


_WEB_MERGE = """
You are consolidating categories extracted from web content about: "{context}"

Task: Reduce to {max_categories} categories.

Step 1 — Cluster: Group the categories below into clusters where each cluster represents ONE distinct topic, claim, or content type. Categories that describe the same web content using different headlines or framings belong in the same cluster. For example, a category about "product reviews" and a category about "consumer evaluations" likely belong together if they reflect the same underlying content type.

Step 2 — Label: For each cluster, choose the single label that best captures the shared meaning. {name_instruction}

Step 3 — Rank: Sum the frequency counts within each cluster. Output the top {max_categories} clusters by total count.

Categories (sorted by extraction frequency):
{seed_with_counts}

Return ONLY a numbered list of {max_categories} categories.
""".strip()


PROMPTS = {
    "neutral": {
        "first_pass": _NEUTRAL_FIRST_PASS,
        "merge":      _NEUTRAL_MERGE,
    },
    "survey": {
        "first_pass": _SURVEY_FIRST_PASS,
        "merge":      _SURVEY_MERGE,
    },
    "social": {
        # first_pass inherits from neutral
        "merge": _SOCIAL_MERGE,
    },
    "academic": {
        "merge": _ACADEMIC_MERGE,
    },
    "policy": {
        "merge": _POLICY_MERGE,
    },
    "web": {
        "merge": _WEB_MERGE,
    },
}


def get_prompt(domain: str, slot: str) -> str:
    """Look up a prompt slot for a domain, falling back to 'neutral'.

    Args:
        domain: A key in PROMPTS (e.g. "neutral", "survey", "social",
            "academic", "policy", "web"). Unknown domains fall through
            to neutral.
        slot:   "first_pass" or "merge".

    Returns:
        The template string, with f-string-style {placeholder} markers
        that the caller fills via str.format(**kwargs).
    """
    return PROMPTS.get(domain, {}).get(slot) or PROMPTS["neutral"][slot]
