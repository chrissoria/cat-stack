"""saturation_prompts: v01 is cat-stack's catllm_0.0.97 prompt exactly, and the
ten variants differ in wording only, never in layout or answer contract."""

import json
import re

import pytest

from catstack.prompts.saturation_prompts import FIELDS, VARIANT_IDS, VARIANTS, load_variants, render
from catstack.text_functions_ensemble import build_context_line, build_text_classification_prompt

CATS = "1. Partner/Spouse\n2. Financial\n3. Housing Features"
TEXT = "moved in with my wife"
Q = "Why did you move?"


@pytest.mark.parametrize("multi_label", [True, False])
@pytest.mark.parametrize("system_prompt", ["", "Coder instructions.\nLine two."])
@pytest.mark.parametrize("examples_text", ["", "Example 1: x"])
@pytest.mark.parametrize("question", [Q, ""])
def test_v01_matches_catllm_0097(multi_label, system_prompt, examples_text, question):
    expected = build_text_classification_prompt(
        response_text=TEXT, categories_str=CATS,
        survey_question_context=build_context_line(question, "catllm_0.0.97"),
        examples_text=examples_text, multi_label=multi_label,
        system_prompt=system_prompt, prompt_style="catllm_0.0.97")
    got = render("v01", TEXT, CATS, survey_question=question, examples_text=examples_text,
                 multi_label=multi_label, system_prompt=system_prompt)
    assert got == expected


def test_ten_distinct_variants():
    assert VARIANT_IDS == tuple(f"v{i:02d}" for i in range(1, 11))
    for field in ("context", "categorize", "categorize_single", "json", "json_single"):
        texts = [VARIANTS[v][field] for v in VARIANT_IDS]
        assert len(set(texts)) == 10, f"{field}: variants are not all distinct"


@pytest.mark.parametrize("variant", VARIANT_IDS)
@pytest.mark.parametrize("multi_label", [True, False])
def test_same_layout_and_contract(variant, multi_label):
    content = render(variant, TEXT, CATS, survey_question=Q, examples_text="Example 1: x",
                     multi_label=multi_label, system_prompt="SYS")[0]["content"]
    lines = content.split("\n")
    # SYS, blank, context, categorize, 3 categories, examples, json
    assert lines[0] == "SYS" and lines[1] == ""
    assert Q in lines[2]
    assert f'"{TEXT}"' in lines[3] and lines[3].endswith(":")
    assert lines[4:7] == CATS.split("\n")
    assert lines[7] == "Example 1: x"
    js = lines[8]
    assert "JSON" in js and re.search(r"\bkey", js)
    if multi_label:
        assert "1" in js and "0" in js and "present" in js
        assert "single" not in lines[3] and "one most" not in lines[3]
    else:
        assert '"1"' in js and '"0"' in js
        assert re.search(r"single|one (most|best)", lines[3] + js)
    assert len(lines) == 9


def test_unknown_variant():
    with pytest.raises(ValueError):
        render("v11", TEXT, CATS)


def test_loaded_from_one_file_per_variant():
    from importlib import resources
    names = sorted(f.name for f in resources.files("catstack.prompts.saturation_prompts").iterdir()
                   if f.name.endswith(".json"))
    assert names == [f"{v}.json" for v in VARIANT_IDS]


def _write(folder, vid, **override):
    doc = {"id": vid, **VARIANTS["v01"], **override}
    (folder / f"{vid}.json").write_text(json.dumps(doc))


def test_loader_reads_folder_in_order(tmp_path):
    for vid in ("v02", "v01", "v10"):
        _write(tmp_path, vid)
    (tmp_path / "notes.json").write_text("{}")          # not vNN.json: ignored
    assert list(load_variants(tmp_path)) == ["v01", "v02", "v10"]


@pytest.mark.parametrize("override,match", [
    ({"id": "v09"}, "does not match"),
    ({"json": ""}, "missing or empty"),
    ({"context": "A respondent was asked."}, "placeholders"),
    ({"categorize": 'Categorize "{text}":'}, "placeholders"),
])
def test_loader_rejects_bad_files(tmp_path, override, match):
    _write(tmp_path, "v01", **override)
    with pytest.raises(ValueError, match=match):
        load_variants(tmp_path)


def test_loader_empty_folder(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_variants(tmp_path)
