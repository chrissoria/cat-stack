"""Loads the wording variants (one vNN.json file each, in this folder) and renders them.
See the package docstring.

Placeholders: {question} in "context"; {response_text} in "categorize" and
"categorize_single". The "json" lines carry no placeholders.
"""

import json
import re
from importlib import resources

FIELDS = ("context", "categorize", "categorize_single", "json", "json_single")
_PLACEHOLDERS = {"context": {"question"}, "categorize": {"response_text"},
                 "categorize_single": {"response_text"}, "json": set(), "json_single": set()}


def load_variants(folder=None):
    """Read every vNN.json in this folder, in file-name order, and check each one.

    One file per variant, so a new or edited variant is a one-file change and a
    caller can loop over them. Each file holds "id" (matching its file name), an
    optional "note", and the five FIELDS. folder defaults to this package's folder.
    """
    out = {}
    folder = resources.files(__package__) if folder is None else folder
    for f in sorted(folder.iterdir(), key=lambda p: p.name):
        if not re.fullmatch(r"v\d{2}\.json", f.name):
            continue
        doc = json.loads(f.read_text(encoding="utf-8"))
        vid = f.name[:-5]
        if doc.get("id") != vid:
            raise ValueError(f"{f.name}: id {doc.get('id')!r} does not match the file name")
        missing = [k for k in FIELDS if not isinstance(doc.get(k), str) or not doc[k]]
        if missing:
            raise ValueError(f"{f.name}: missing or empty field(s) {missing}")
        for k in FIELDS:
            found = set(re.findall(r"{(\w+)}", doc[k]))
            if found != _PLACEHOLDERS[k]:
                raise ValueError(f"{f.name}: {k} has placeholders {sorted(found)}, "
                                 f"expected {sorted(_PLACEHOLDERS[k])}")
        out[vid] = {k: doc[k] for k in FIELDS}
    if not out:
        raise FileNotFoundError(f"no vNN.json prompt files found in {folder}")
    return out


VARIANTS = load_variants()
VARIANT_IDS = tuple(VARIANTS)


def render(variant, response_text, categories_str, survey_question="", examples_text="",
           multi_label=True, system_prompt=""):
    """Messages for one classification call, in cat-stack's message format.

    Laid out exactly as catstack.text_functions_ensemble.build_text_classification_prompt
    lays out its non-chain-of-thought prompt: context line, categorize line, the
    numbered categories, the examples, the JSON instruction; a system_prompt is
    prepended to the user message followed by a blank line.

    Args:
        variant: one of VARIANT_IDS ("v01" ... "v10").
        response_text: the text to classify.
        categories_str: the numbered category list, "1. ...\\n2. ...".
        survey_question: the question; the context line is empty when blank.
        examples_text: few-shot examples text, or "".
        multi_label: False for single-label tasks.
        system_prompt: text prepended to the user message, or "".
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANT_IDS}, got {variant!r}")
    v = VARIANTS[variant]
    context = v["context"].format(question=survey_question) if survey_question else ""
    categorize = (v["categorize"] if multi_label else v["categorize_single"]).format(
        response_text=response_text)
    json_line = v["json"] if multi_label else v["json_single"]
    user_prompt = f"""{context}
{categorize}
{categories_str}
{examples_text}
{json_line}"""
    if system_prompt:
        user_prompt = system_prompt.rstrip() + "\n\n" + user_prompt
    return [{"role": "user", "content": user_prompt}]
