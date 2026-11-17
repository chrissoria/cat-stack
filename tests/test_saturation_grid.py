"""Saturation grid: wording variants x category orders x repeats.

Each wording is crossed with the same category orders (original order first,
then seeded shuffles), capped at K! orders for K categories. Replies are keyed
by shown position and mapped back to the original categories.
"""

import itertools
import json
import re
from unittest.mock import patch

import pytest

import catstack
from catstack._saturation import (category_orders, draw_id, max_prompts,
                                  ordered_categories_str, unshuffle_codes)
from catstack.prompts.saturation_prompts import VARIANT_IDS, VARIANTS

CATS = ["Partner", "Money", "Housing"]
Q = "Why did you move?"
INPUTS = ("moved in with my wife", "rent went up")


def _variant_of(prompt):
    for vid in VARIANT_IDS:
        if VARIANTS[vid]["context"].format(question=Q) in prompt:
            return vid
    raise AssertionError("prompt matches no saturation variant")


def _shown(prompt, cats):
    """Category names in the order the prompt numbers them."""
    found = {}
    for line in prompt.splitlines():
        m = re.fullmatch(r"(\d+)\. (.+)", line.strip())
        if m and m.group(2) in cats:
            found[int(m.group(1))] = m.group(2)
    assert sorted(found) == list(range(1, len(cats) + 1)), "category list not numbered 1..K"
    return [found[i] for i in sorted(found)]


def _run(replies, cats=CATS, inputs=INPUTS, **kw):
    """replies(text, variant, shown names) -> reply keyed by shown position.
    Returns (result, sent calls)."""
    sent = []

    def fake_complete(self, messages, **k):
        prompt = messages[-1]["content"]
        text = next(t for t in inputs if f'"{t}"' in prompt)
        shown = _shown(prompt, cats)
        sent.append({"variant": _variant_of(prompt), "text": text, "shown": tuple(shown),
                     "prompt": prompt})
        return replies(text, _variant_of(prompt), shown), None

    with patch("cat_stack._providers.UnifiedLLMClient.complete", fake_complete):
        out = catstack.classify(input_data=list(inputs), categories=cats, api_key="k",
                                user_model="gpt-4o-mini", description=Q, check_verbosity=False,
                                add_other=False, saturation=True, json_retries=0, **kw)
    return out, sent


def _by_name(coded):
    """A model that codes the categories named in `coded`, wherever they are shown."""
    def reply(text, vid, shown):
        return json.dumps({str(j + 1): str(int(name in coded)) for j, name in enumerate(shown)})
    return reply


# --- the grid itself --------------------------------------------------------

def test_orders_are_capped_at_k_factorial():
    assert category_orders(2, 10) == [(0, 1), (1, 0)]
    three = category_orders(3, 10)
    assert three[0] == (0, 1, 2)
    assert sorted(three) == sorted(itertools.permutations(range(3)))
    assert category_orders(3, 1) == [(0, 1, 2)]


def test_large_k_orders_are_distinct_seeded_and_start_original():
    a = category_orders(13, 10)
    assert len(a) == len(set(a)) == 10
    assert a[0] == tuple(range(13))
    assert all(sorted(o) == list(range(13)) for o in a)
    assert category_orders(13, 10) == a                      # same seed, same orders
    assert category_orders(13, 10, seed=1) != a


def test_max_prompts():
    assert max_prompts(2) == 20
    assert max_prompts(3) == 60
    assert max_prompts(4) == 240
    assert max_prompts(3, n_variants=1) == 6


def test_unshuffle_maps_shown_positions_to_original_numbers():
    order = (2, 0, 1)            # shown: Housing, Partner, Money
    assert ordered_categories_str(CATS, order) == "1. Housing\n2. Partner\n3. Money"
    assert unshuffle_codes({"1": 1, "2": 0, "3": 0}, order) == {"3": 1, "1": 0, "2": 0}
    assert unshuffle_codes(None, order) is None


def test_draw_id():
    assert draw_id("v03", 7, 0) == "v03_o07_r00"


@pytest.mark.parametrize("kw", [
    {"saturation_variants": ["v99"]},
    {"saturation_variants": []},
    {"saturation_variants": ["v01", "v01"]},
    {"saturation_orders": 0},
    {"saturation_repeats": True},
    {"saturation_workers": 1.5},
])
def test_bad_options_raise_before_any_call(kw):
    with pytest.raises(ValueError):
        _run(_by_name({"Money"}), **kw)


# --- classify() with shuffled orders ----------------------------------------

def test_replies_are_mapped_back_to_the_original_categories():
    out, sent = _run(_by_name({"Money"}), saturation_orders=3)
    assert len(sent) == 2 * 10 * 3
    orders = {s["shown"] for s in sent}
    assert len(orders) == 3 and tuple(CATS) in orders
    for text in INPUTS:                                       # every wording sees every order
        for vid in VARIANT_IDS:
            assert {s["shown"] for s in sent if s["text"] == text and s["variant"] == vid} == orders
    assert list(out["category_1"]) == [0.0, 0.0]
    assert list(out["category_2"]) == [1.0, 1.0]
    assert list(out["category_3"]) == [0.0, 0.0]
    assert list(out["saturation_draws"]) == [30, 30]


def test_schema_descriptions_follow_the_shown_order():
    schemas = []

    def fake_complete(self, messages, **k):
        shown = _shown(messages[-1]["content"], CATS)
        if '"moved in with my wife"' in messages[-1]["content"]:
            schemas.append((tuple(shown), k.get("json_schema")))
        return _by_name({"Money"})("", "", shown), None

    with patch("cat_stack._providers.UnifiedLLMClient.complete", fake_complete):
        catstack.classify(input_data=["moved in with my wife"], categories=CATS, api_key="k",
                          user_model="gpt-4o-mini", description=Q, check_verbosity=False,
                          add_other=False, saturation=True, json_retries=0, saturation_orders=3)
    assert len({s for s, _ in schemas}) == 3
    for shown, schema in schemas:
        assert [schema["properties"][str(j)]["description"] for j in (1, 2, 3)] == list(shown)
        assert schema["required"] == ["1", "2", "3"]


def test_position_bias_becomes_a_split(capsys):
    def first_shown(text, vid, shown):
        return json.dumps({str(j + 1): str(int(j == 0)) for j in range(len(shown))})

    out, sent = _run(first_shown, saturation_orders=10)       # 3 categories: capped at 6
    assert "capped at 6" in capsys.readouterr().out
    assert len(sent) == 2 * 10 * 6
    for k in (1, 2, 3):                                       # each category is first in 2 of 6
        assert list(out[f"category_{k}"]) == [pytest.approx(1 / 3)] * 2


def test_original_order_prompt_is_the_plain_single_order_prompt():
    _, plain = _run(_by_name({"Money"}))
    _, grid = _run(_by_name({"Money"}), saturation_orders=3)
    p = {(s["text"], s["variant"]): s["prompt"] for s in plain}
    g = {(s["text"], s["variant"]): s["prompt"] for s in grid if s["shown"] == tuple(CATS)}
    assert p == g


def test_variant_subset_and_repeats():
    out, sent = _run(_by_name({"Partner"}), saturation_variants=["v01"], saturation_repeats=4)
    assert len(sent) == 2 * 4
    assert {s["variant"] for s in sent} == {"v01"}
    assert len({s["prompt"] for s in sent if s["text"] == INPUTS[0]}) == 1
    assert list(out["saturation_draws"]) == [4, 4]
    assert list(out["category_1"]) == [1.0, 1.0]


def test_concurrent_draws_match_sequential():
    def mixed(text, vid, shown):              # depends on wording and order
        hit = {"Money"} if int(vid[1:]) % 2 else {shown[0]}
        return _by_name(hit)(text, vid, shown)

    seq, _ = _run(mixed, saturation_orders=4, saturation_keep_draws=True)
    par, _ = _run(mixed, saturation_orders=4, saturation_keep_draws=True, saturation_workers=8)
    cols = ["category_1", "category_2", "category_3", "saturation_draws", "saturation_detail"]
    assert seq[cols].equals(par[cols])


def test_keep_draws_records_every_draw_in_original_order():
    def unusable_v10(text, vid, shown):
        return "no json here" if vid == "v10" else _by_name({"Housing"})(text, vid, shown)

    out, _ = _run(unusable_v10, saturation_orders=2, saturation_keep_draws=True)
    detail = json.loads(out["saturation_detail"].iloc[0])
    assert len(detail) == 10 * 2
    assert set(detail) == {draw_id(v, o, 0) for v in VARIANT_IDS for o in range(2)}
    assert detail["v01_o00_r00"] == "001"
    assert detail["v01_o01_r00"] == "001"                     # shuffled, mapped back
    assert detail["v10_o00_r00"] is None and detail["v10_o01_r00"] is None
    assert list(out["saturation_draws"]) == [18, 18]
    assert list(out["category_3"]) == [1.0, 1.0]


def test_no_detail_column_by_default():
    out, _ = _run(_by_name({"Money"}))
    assert "saturation_detail" not in out.columns


def test_examples_with_shuffled_orders_warn(capsys):
    _run(_by_name({"Money"}), saturation_orders=2, examples=["'cheaper rent' -> Money"])
    assert "Examples that name a category by its number" in capsys.readouterr().out
