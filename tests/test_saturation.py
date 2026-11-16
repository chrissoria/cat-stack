"""classify(saturation=True): one call per prompt-wording variant per response, at
temperature 0; each category cell is the proportion of usable draws assigning it."""

from unittest.mock import patch

import pytest

import catstack
from catstack._saturation import combine_draws
from catstack.prompts.saturation_prompts import VARIANT_IDS, VARIANTS

CATS = ["Partner", "Money", "Housing"]
Q = "Why did you move?"


def _variant_of(prompt):
    """Which variant a prompt was rendered from (by its context line)."""
    for vid in VARIANT_IDS:
        if VARIANTS[vid]["context"].format(question=Q) in prompt:
            return vid
    raise AssertionError("prompt matches no saturation variant")


def _run(replies, inputs=("moved in with my wife", "rent went up"), **kw):
    """replies(text, variant) -> reply string. Returns (result, sent calls)."""
    sent = []

    def fake_complete(self, messages, **k):
        prompt = messages[-1]["content"]
        vid = _variant_of(prompt)
        text = next(t for t in inputs if f'"{t}"' in prompt)
        sent.append({"variant": vid, "text": text, "creativity": k.get("creativity")})
        return replies(text, vid), None

    with patch("cat_stack._providers.UnifiedLLMClient.complete", fake_complete):
        out = catstack.classify(input_data=list(inputs), categories=CATS, api_key="k",
                                user_model="gpt-4o-mini", description=Q, check_verbosity=False,
                                add_other=False, saturation=True, json_retries=0, **kw)
    return out, sent


def _two_of_ten(text, vid):
    if text == "moved in with my wife":
        return '{"1":"1","2":"0","3":"0"}' if vid in ("v01", "v02") else '{"1":"0","2":"0","3":"0"}'
    return '{"1":"0","2":"1","3":"0"}'


def test_cells_are_proportions_of_draws():
    out, sent = _run(_two_of_ten)
    assert len(sent) == 2 * 10
    assert sorted({s["variant"] for s in sent}) == list(VARIANT_IDS)
    assert all(s["creativity"] == 0 for s in sent)
    assert list(out["category_1"]) == [pytest.approx(0.2), pytest.approx(0.0)]
    assert list(out["category_2"]) == [pytest.approx(0.0), pytest.approx(1.0)]
    assert list(out["saturation_draws"]) == [10, 10]
    assert str(out["category_1"].dtype) == "Float64"
    assert not any(c.startswith("saturation_share") for c in out.columns)


def test_unusable_draw_leaves_the_denominator():
    def replies(text, vid):
        if vid == "v03":
            return "I cannot answer that."
        return '{"1":"1","2":"0","3":"0"}' if vid in ("v01", "v02", "v04") else '{"1":"0","2":"0","3":"0"}'
    out, _ = _run(replies, inputs=("moved in with my wife",))
    assert out["saturation_draws"].iloc[0] == 9
    assert out["category_1"].iloc[0] == pytest.approx(3 / 9)


def test_two_models_consensus_is_mean_of_proportions():
    sent = []

    def fake_complete(self, messages, **k):
        vid = _variant_of(messages[-1]["content"])
        sent.append((self.model, vid))
        if self.model == "gpt-4o-mini":
            return ('{"1":"1","2":"0","3":"0"}' if vid in ("v01", "v02") else '{"1":"0","2":"0","3":"0"}'), None
        return ('{"1":"1","2":"0","3":"0"}' if vid in ("v01", "v02", "v03", "v04") else '{"1":"0","2":"0","3":"0"}'), None

    with patch("cat_stack._providers.UnifiedLLMClient.complete", fake_complete):
        out = catstack.classify(input_data=["moved in with my wife"], categories=CATS,
                                models=[("gpt-4o-mini", "openai", "k"), ("gpt-4o", "openai", "k")],
                                description=Q, check_verbosity=False, add_other=False,
                                saturation=True, json_retries=0, parallel=False)
    assert len(sent) == 20
    assert out["category_1_gpt_4o_mini_tauto"].iloc[0] == pytest.approx(0.2)
    assert out["category_1_gpt_4o_tauto"].iloc[0] == pytest.approx(0.4)
    assert out["category_1_consensus"].iloc[0] == pytest.approx(0.3)
    assert not any(c.endswith("_agreement") for c in out.columns)
    assert out["saturation_draws_gpt_4o_tauto"].iloc[0] == 10


@pytest.mark.parametrize("bad", [
    {"batch_mode": True}, {"chain_of_thought": True}, {"chain_of_verification": True},
    {"step_back_prompt": True}, {"context_prompt": True}, {"categories_per_call": 2},
])
def test_incompatible_options_raise(bad):
    with pytest.raises(ValueError, match="saturation"):
        catstack.classify(input_data=["x"], categories=CATS, api_key="k", user_model="gpt-4o-mini",
                          check_verbosity=False, add_other=False, saturation=True, **bad)


# --- combine_draws ----------------------------------------------------------

def _d(*bits):
    return {str(i + 1): b for i, b in enumerate(bits)}


def test_combine_shares_and_majority():
    draws = [_d(1, 0), _d(1, 0), _d(0, 1), None]
    final, shares, n = combine_draws(draws, 2, multi_label=True)
    assert n == 3 and shares == pytest.approx([2 / 3, 1 / 3])
    assert final == {"1": 1, "2": 0}


def test_combine_half_goes_to_v01():
    final, shares, n = combine_draws([_d(1, 0), _d(0, 0)], 2, multi_label=True)
    assert shares == [0.5, 0.0] and final == {"1": 1, "2": 0}
    final, _, _ = combine_draws([None, _d(1, 0), _d(0, 0)], 2, multi_label=True)
    assert final == {"1": 0, "2": 0}                     # v01 failed: half -> 0


def test_combine_single_label_tie_goes_to_v01():
    final, shares, n = combine_draws([_d(0, 1, 0), _d(1, 0, 0), _d(0, 1, 0), _d(1, 0, 0)], 3,
                                     multi_label=False)
    assert final == {"1": 0, "2": 1, "3": 0} and shares == [0.5, 0.5, 0.0]


def test_combine_no_usable_draw():
    final, shares, n = combine_draws([None, None], 2)
    assert final is None and n == 0 and shares == [None, None]


# --- failed draws are re-sent ----------------------------------------------

def _run_with_errors(answer, **kw):
    """answer(variant, attempt) -> (reply, error). Returns (result, calls per variant)."""
    calls = {}

    def fake_complete(self, messages, **k):
        vid = _variant_of(messages[-1]["content"])
        calls[vid] = calls.get(vid, 0) + 1
        return answer(vid, calls[vid])

    with patch("cat_stack._providers.UnifiedLLMClient.complete", fake_complete):
        out = catstack.classify(input_data=["moved in with my wife"], categories=CATS, api_key="k",
                                user_model="gpt-4o-mini", description=Q, check_verbosity=False,
                                add_other=False, saturation=True, json_retries=0, retry_delay=0,
                                batch_retries=0, **kw)
    return out, calls


def test_draw_lost_to_an_error_is_resent():
    def answer(vid, attempt):
        if vid == "v03" and attempt == 1:
            return None, "Connection reset by peer"
        return '{"1":"1","2":"0","3":"0"}', None
    out, calls = _run_with_errors(answer)
    assert calls["v03"] == 2 and all(calls[v] == 1 for v in VARIANT_IDS if v != "v03")
    assert out["saturation_draws"].iloc[0] == 10
    assert out["category_1"].iloc[0] == pytest.approx(1.0)


def test_draw_that_keeps_failing_is_dropped_and_reported(capsys):
    from catstack._saturation import DRAW_RETRIES

    def answer(vid, attempt):
        if vid == "v04":
            return None, "Connection reset by peer"
        return ('{"1":"1","2":"0","3":"0"}' if vid in ("v01", "v02", "v03") else
                '{"1":"0","2":"0","3":"0"}'), None
    out, calls = _run_with_errors(answer)
    assert calls["v04"] == 1 + DRAW_RETRIES
    assert out["saturation_draws"].iloc[0] == 9
    assert out["category_1"].iloc[0] == pytest.approx(3 / 9)
    assert "fewer than 10 usable draws" in capsys.readouterr().out


def test_unscorable_reply_is_not_resent():
    def answer(vid, attempt):
        return ("no idea" if vid == "v05" else '{"1":"0","2":"1","3":"0"}'), None
    out, calls = _run_with_errors(answer)
    assert calls["v05"] == 1                      # json_retries=0, no draw retry
    assert out["saturation_draws"].iloc[0] == 9
