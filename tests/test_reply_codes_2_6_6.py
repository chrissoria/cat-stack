"""
classification_codes(): the one rule for whether a reply can be scored
(cat-stack 2.6.6).

Before 2.6.6 a reply was accepted as soon as one key read as a category
coded 0/1, and every other key was dropped and read as 0. A Qwen3.8-2.4T
reply on Hugging Face, {"<garbled>": 0, "category_2": 0, ...}, passed that
way: the garbled key read as category 1 and the "category_N" keys were
dropped, so any 1 among them would have become a silent all-zero row. Now a
0/1 code under a key that is not a category number, or a category key with
no 0/1 code, makes the reply unusable (re-ask, then the formatter, else an
error row). Sparse replies stay usable.
"""

import json

import pytest

from catstack._utils import classification_codes, classification_reply_is_usable
from catstack.text_functions_ensemble import aggregate_results

N = 15


@pytest.mark.parametrize("reply,expected", [
    ({"4": 1}, {"4": "1"}),                                   # sparse, int value
    ('{"4":"1"}', {"4": "1"}),                                # sparse, as JSON text
    ({"4": "1", "15": "0"}, {"4": "1", "15": "0"}),
    ({str(i): 0 for i in range(1, N + 1)}, {str(i): "0" for i in range(1, N + 1)}),
    ({"4": True, "5": "false"}, {"4": "1", "5": "0"}),        # booleans
    ({"4": 1.0}, {"4": "1"}),                                 # float 1.0
    ({"4. Close-knit Loyalty": "1"}, {"4": "1"}),             # numbered name key
    ({".4": "1"}, {"4": "1"}),                                # leading punctuation
    ({"4": "1", "reasoning": "the response mentions ..."}, {"4": "1"}),  # text field ignored
    ({"4": "1", "notes": {"a": "b"}, "tags": ["x"]}, {"4": "1"}),        # objects/lists ignored
    ({"4": "1", "4. Close-knit Loyalty": "0"}, {"4": "1"}),   # exact key wins
])
def test_usable(reply, expected):
    assert classification_codes(reply, N) == expected
    assert classification_reply_is_usable(json.dumps(reply) if isinstance(reply, dict) else reply, N)


@pytest.mark.parametrize("reply", [
    # the observed Qwen reply shape: garbled key + category_N keys
    {": 1, ": 0, **{f"category_{i}": 0 for i in range(2, N + 1)}},
    {"category_3": 1},                                       # category_N alone
    {"1": "0", "category_7": 1},                             # a code would be dropped
    {"4": "1", "16": "1"},                                   # out-of-range code
    {"0": "1", "1": "0"},                                    # zero-based numbering
    {"1": "The response mentions family ...", "15": "0"},     # reasoning under a category key
    {"4": None},
    {"4": "maybe"},
    {},
    {"text": "....", "categories": {"4": 1}},                # nested codes
    [{"4": 1}],
])
def test_unusable(reply):
    assert classification_codes(reply, N) is None
    assert not classification_reply_is_usable(json.dumps(reply), N)


@pytest.mark.parametrize("raw", ['{"1":"e"}', "not json", "", None])
def test_unusable_text(raw):
    assert classification_codes(raw, N) is None
    assert not classification_reply_is_usable(raw, N)


def test_aggregate_marks_dropped_code_reply_failed():
    cats = [f"c{i}" for i in range(1, N + 1)]
    bad = json.dumps({": 1, ": 0, **{f"category_{i}": 0 for i in range(2, N + 1)}})
    out = aggregate_results({"m": (bad, None)}, cats, "unanimous", "partial")
    assert out["failed_models"] == ["m"]
    out = aggregate_results({"m": ('{"4": 1}', None)}, cats, "unanimous", "partial")
    assert out["failed_models"] == []
    assert out["per_model"]["m"]["4"] == "1"
