"""
Tests for M-CONSENSUS (#37): `consensus_threshold="majority"` now
resolves 50/50 ties to "0" (strict majority) instead of "1".

Pre-fix: `aggregate_results` compared `positive_rate >= threshold`
even when the threshold came from the `"majority"` string alias
(threshold = 0.5). For even-model ensembles tied 50/50 — 2-2 of 4,
3-3 of 6, 1-1 of 2 — the tie crossed `>=` and produced a positive
consensus label. That's an arbitrary default toward false-positive
assignments and most pronounced on 2-model ensembles, where a single
disagreement always produced a positive label.

Post-fix:
  - `"majority"` (string alias) uses STRICT `>`: a tie resolves to "0".
  - Numeric thresholds (e.g. user explicitly passes `0.5`) keep the
    literal `>=` semantics — the user picked a number and gets it.
  - `"two-thirds"` and `"unanimous"` are unchanged.

Behavior change is intentional and prominent in the CHANGELOG; it is
listed under the ecosystem-memory "release-note-worthy" set.
"""

import pytest

from cat_stack.text_functions_ensemble import aggregate_results


CATEGORIES = ["yes", "no"]


def _result(json_str, error=None):
    """Shape matching what aggregate_results consumes."""
    return (json_str, error)


def _model_results_2x2(votes_for_cat1):
    """Build a 4-model result with `votes_for_cat1` positive votes for
    category 1. Category 2 is always 0 for simplicity."""
    yes_models = {f"m{i+1}": _result('{"1":"1","2":"0"}') for i in range(votes_for_cat1)}
    no_models = {f"m{i+1+votes_for_cat1}": _result('{"1":"0","2":"0"}')
                 for i in range(4 - votes_for_cat1)}
    return {**yes_models, **no_models}


class TestStrictMajorityTieBreak:
    def test_4_models_tied_2_2_resolves_to_negative(self):
        """4 models tied 2-2 → consensus '0' (pre-fix: '1')."""
        results = _model_results_2x2(votes_for_cat1=2)
        agg = aggregate_results(
            model_results=results,
            categories=CATEGORIES,
            consensus_threshold="majority",
            fail_strategy="partial",
        )
        assert agg["consensus"]["1"] == "0", (
            "majority tie should resolve to negative, got "
            f"consensus={agg['consensus']}"
        )

    def test_6_models_tied_3_3_resolves_to_negative(self):
        results = {}
        for i in range(3):
            results[f"yes{i}"] = _result('{"1":"1","2":"0"}')
        for i in range(3):
            results[f"no{i}"] = _result('{"1":"0","2":"0"}')
        agg = aggregate_results(
            model_results=results,
            categories=CATEGORIES,
            consensus_threshold="majority",
            fail_strategy="partial",
        )
        assert agg["consensus"]["1"] == "0"

    def test_2_models_tied_1_1_resolves_to_negative(self):
        """The most pronounced case — 2-model ensemble with one
        disagreement now correctly produces '0' instead of always '1'."""
        results = {
            "m1": _result('{"1":"1","2":"0"}'),
            "m2": _result('{"1":"0","2":"0"}'),
        }
        agg = aggregate_results(
            model_results=results,
            categories=CATEGORIES,
            consensus_threshold="majority",
            fail_strategy="partial",
        )
        assert agg["consensus"]["1"] == "0"


class TestCleanMajorityPreserved:
    """Sanity: actual majorities still produce '1'."""

    def test_4_models_3_1_yes(self):
        results = _model_results_2x2(votes_for_cat1=3)
        agg = aggregate_results(
            model_results=results,
            categories=CATEGORIES,
            consensus_threshold="majority",
            fail_strategy="partial",
        )
        assert agg["consensus"]["1"] == "1"

    def test_3_models_2_1_yes(self):
        """Odd ensemble (no tie possible) — 2-of-3 still wins."""
        results = {
            "m1": _result('{"1":"1","2":"0"}'),
            "m2": _result('{"1":"1","2":"0"}'),
            "m3": _result('{"1":"0","2":"0"}'),
        }
        agg = aggregate_results(
            model_results=results,
            categories=CATEGORIES,
            consensus_threshold="majority",
            fail_strategy="partial",
        )
        assert agg["consensus"]["1"] == "1"

    def test_4_models_4_0_yes(self):
        results = _model_results_2x2(votes_for_cat1=4)
        agg = aggregate_results(
            model_results=results,
            categories=CATEGORIES,
            consensus_threshold="majority",
            fail_strategy="partial",
        )
        assert agg["consensus"]["1"] == "1"


class TestNumericThresholdPreservesGreaterEqual:
    """When the user explicitly passes a numeric threshold, the literal
    `>=` semantics still apply — they chose the number, they get it."""

    def test_numeric_0_5_with_4_models_tied_uses_gte(self):
        """4 models tied 2-2, threshold=0.5 numeric → still '1' (>= 0.5).
        Only the `"majority"` string alias changes."""
        results = _model_results_2x2(votes_for_cat1=2)
        agg = aggregate_results(
            model_results=results,
            categories=CATEGORIES,
            consensus_threshold=0.5,
            fail_strategy="partial",
        )
        assert agg["consensus"]["1"] == "1", (
            "numeric 0.5 should still use >= semantics; got "
            f"consensus={agg['consensus']}"
        )

    def test_numeric_0_75_with_4_models_3_yes_does_not_cross(self):
        results = _model_results_2x2(votes_for_cat1=3)
        agg = aggregate_results(
            model_results=results,
            categories=CATEGORIES,
            consensus_threshold=0.75,
            fail_strategy="partial",
        )
        # 3/4 = 0.75 >= 0.75 → '1' (>= semantics for numeric)
        assert agg["consensus"]["1"] == "1"


class TestOtherStringAliasesUnchanged:
    def test_unanimous_still_requires_full_agreement(self):
        results = _model_results_2x2(votes_for_cat1=3)
        agg = aggregate_results(
            model_results=results,
            categories=CATEGORIES,
            consensus_threshold="unanimous",
            fail_strategy="partial",
        )
        # 3 of 4 isn't unanimous
        assert agg["consensus"]["1"] == "0"

    def test_unanimous_with_all_yes(self):
        results = _model_results_2x2(votes_for_cat1=4)
        agg = aggregate_results(
            model_results=results,
            categories=CATEGORIES,
            consensus_threshold="unanimous",
            fail_strategy="partial",
        )
        assert agg["consensus"]["1"] == "1"

    def test_two_thirds_with_3_of_4(self):
        """3 of 4 = 0.75 which is >= 0.67. Should still produce '1'.
        (This test pins the current behavior; the two-thirds precision
        bug for 4/6 = 0.667 is a separate task.)"""
        results = _model_results_2x2(votes_for_cat1=3)
        agg = aggregate_results(
            model_results=results,
            categories=CATEGORIES,
            consensus_threshold="two-thirds",
            fail_strategy="partial",
        )
        assert agg["consensus"]["1"] == "1"
