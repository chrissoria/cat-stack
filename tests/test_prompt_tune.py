"""
Tests for the prompt_tune module.

Uses synthetic correction data to test:
  - _compute_per_category_metrics  (pure metrics math)
  - _assemble_prompt               (prompt assembly from instructions)
  - _print_classification_summary  (output formatting)
  - prompt_tune end-to-end         (full flow with mocked LLM + UI)
"""

import io
from contextlib import redirect_stdout
from unittest.mock import patch, MagicMock

import pytest

from cat_stack.prompt_tune import (
    _compute_per_category_metrics,
    _assemble_prompt,
    _print_classification_summary,
    prompt_tune,
)


# ── Mock data ────────────────────────────────────────────────────────────

CATEGORIES = ["Positive", "Negative", "Neutral"]

# 5-item sample where the model got everything right
CORRECTIONS_PERFECT = [
    {
        "input": "I love this product!",
        "original": {"Positive": 1, "Negative": 0, "Neutral": 0},
        "corrected": {"Positive": 1, "Negative": 0, "Neutral": 0},
        "changed": [],
    },
    {
        "input": "Terrible experience.",
        "original": {"Positive": 0, "Negative": 1, "Neutral": 0},
        "corrected": {"Positive": 0, "Negative": 1, "Neutral": 0},
        "changed": [],
    },
    {
        "input": "It was okay I guess.",
        "original": {"Positive": 0, "Negative": 0, "Neutral": 1},
        "corrected": {"Positive": 0, "Negative": 0, "Neutral": 1},
        "changed": [],
    },
    {
        "input": "Nothing special.",
        "original": {"Positive": 0, "Negative": 0, "Neutral": 1},
        "corrected": {"Positive": 0, "Negative": 0, "Neutral": 1},
        "changed": [],
    },
    {
        "input": "Great value for money.",
        "original": {"Positive": 1, "Negative": 0, "Neutral": 0},
        "corrected": {"Positive": 1, "Negative": 0, "Neutral": 0},
        "changed": [],
    },
]

# 5-item sample with mixed errors:
#  - Item 0: model said Positive, user corrected to Neutral  (Positive FP, Neutral FN)
#  - Item 1: model said Neutral, user corrected to Negative  (Neutral FP, Negative FN)
#  - Items 2-4: correct
CORRECTIONS_WITH_ERRORS = [
    {
        "input": "The weather is warm today.",
        "original": {"Positive": 1, "Negative": 0, "Neutral": 0},
        "corrected": {"Positive": 0, "Negative": 0, "Neutral": 1},
        "changed": ["Positive", "Neutral"],
    },
    {
        "input": "I can't stand waiting in line.",
        "original": {"Positive": 0, "Negative": 0, "Neutral": 1},
        "corrected": {"Positive": 0, "Negative": 1, "Neutral": 0},
        "changed": ["Negative", "Neutral"],
    },
    {
        "input": "This is amazing!",
        "original": {"Positive": 1, "Negative": 0, "Neutral": 0},
        "corrected": {"Positive": 1, "Negative": 0, "Neutral": 0},
        "changed": [],
    },
    {
        "input": "Worst meal I ever had.",
        "original": {"Positive": 0, "Negative": 1, "Neutral": 0},
        "corrected": {"Positive": 0, "Negative": 1, "Neutral": 0},
        "changed": [],
    },
    {
        "input": "It arrived on Tuesday.",
        "original": {"Positive": 0, "Negative": 0, "Neutral": 1},
        "corrected": {"Positive": 0, "Negative": 0, "Neutral": 1},
        "changed": [],
    },
]

# After improvement: all errors fixed
CORRECTIONS_FIXED = [
    {
        "input": "The weather is warm today.",
        "original": {"Positive": 0, "Negative": 0, "Neutral": 1},
        "corrected": {"Positive": 0, "Negative": 0, "Neutral": 1},
        "changed": [],
    },
    {
        "input": "I can't stand waiting in line.",
        "original": {"Positive": 0, "Negative": 1, "Neutral": 0},
        "corrected": {"Positive": 0, "Negative": 1, "Neutral": 0},
        "changed": [],
    },
    {
        "input": "This is amazing!",
        "original": {"Positive": 1, "Negative": 0, "Neutral": 0},
        "corrected": {"Positive": 1, "Negative": 0, "Neutral": 0},
        "changed": [],
    },
    {
        "input": "Worst meal I ever had.",
        "original": {"Positive": 0, "Negative": 1, "Neutral": 0},
        "corrected": {"Positive": 0, "Negative": 1, "Neutral": 0},
        "changed": [],
    },
    {
        "input": "It arrived on Tuesday.",
        "original": {"Positive": 0, "Negative": 0, "Neutral": 1},
        "corrected": {"Positive": 0, "Negative": 0, "Neutral": 1},
        "changed": [],
    },
]


# ── _compute_per_category_metrics ────────────────────────────────────────

class TestComputePerCategoryMetrics:
    def test_perfect_classifications(self):
        result = _compute_per_category_metrics(CORRECTIONS_PERFECT, CATEGORIES)

        for cat in CATEGORIES:
            assert result[cat]["accuracy"] == 1.0
            assert result[cat]["sensitivity"] == 1.0
            assert result[cat]["precision"] == 1.0
            assert result[cat]["fp"] == 0
            assert result[cat]["fn"] == 0

    def test_errors_counted_correctly(self):
        result = _compute_per_category_metrics(CORRECTIONS_WITH_ERRORS, CATEGORIES)

        # Positive: 1 FP (item 0), 0 FN; 1 TP (item 2), 3 TN
        assert result["Positive"]["fp"] == 1
        assert result["Positive"]["fn"] == 0
        assert result["Positive"]["tp"] == 1
        assert result["Positive"]["tn"] == 3

        # Negative: 0 FP, 1 FN (item 1); 1 TP (item 3), 3 TN
        assert result["Negative"]["fp"] == 0
        assert result["Negative"]["fn"] == 1
        assert result["Negative"]["tp"] == 1
        assert result["Negative"]["tn"] == 3

        # Neutral: 1 FP (item 1), 1 FN (item 0); 1 TP (item 4), 2 TN
        assert result["Neutral"]["fp"] == 1
        assert result["Neutral"]["fn"] == 1
        assert result["Neutral"]["tp"] == 1
        assert result["Neutral"]["tn"] == 2

    def test_derived_metrics(self):
        result = _compute_per_category_metrics(CORRECTIONS_WITH_ERRORS, CATEGORIES)

        # Positive: acc=(1+3)/5=0.8, sens=1/(1+0)=1.0, prec=1/(1+1)=0.5
        assert result["Positive"]["accuracy"] == pytest.approx(0.8)
        assert result["Positive"]["sensitivity"] == pytest.approx(1.0)
        assert result["Positive"]["precision"] == pytest.approx(0.5)

        # Negative: acc=(1+3)/5=0.8, sens=1/(1+1)=0.5, prec=1/(1+0)=1.0
        assert result["Negative"]["accuracy"] == pytest.approx(0.8)
        assert result["Negative"]["sensitivity"] == pytest.approx(0.5)
        assert result["Negative"]["precision"] == pytest.approx(1.0)

        # Neutral: acc=(1+2)/5=0.6, sens=1/(1+1)=0.5, prec=1/(1+1)=0.5
        assert result["Neutral"]["accuracy"] == pytest.approx(0.6)
        assert result["Neutral"]["sensitivity"] == pytest.approx(0.5)
        assert result["Neutral"]["precision"] == pytest.approx(0.5)

    def test_empty_corrections(self):
        result = _compute_per_category_metrics([], CATEGORIES)
        for cat in CATEGORIES:
            # No data → defaults to 1.0
            assert result[cat]["accuracy"] == 1.0
            assert result[cat]["sensitivity"] == 1.0
            assert result[cat]["precision"] == 1.0

    def test_single_category(self):
        corrections = [
            {
                "input": "hello",
                "original": {"Only": 1},
                "corrected": {"Only": 0},
                "changed": ["Only"],
            },
        ]
        result = _compute_per_category_metrics(corrections, ["Only"])
        assert result["Only"]["fp"] == 1
        assert result["Only"]["fn"] == 0
        assert result["Only"]["precision"] == pytest.approx(0.0)


# ── _assemble_prompt ─────────────────────────────────────────────────────

class TestAssemblePrompt:
    def test_empty_instructions(self):
        assert _assemble_prompt({}, CATEGORIES) == ""

    def test_single_instruction(self):
        instructions = {"Neutral": "Assign only for factual statements."}
        result = _assemble_prompt(instructions, CATEGORIES)
        assert result == (
            "Classification guidance per category:\n"
            "\n"
            "- Neutral: Assign only for factual statements."
        )

    def test_multiple_instructions_preserve_category_order(self):
        instructions = {
            "Neutral": "Factual only.",
            "Positive": "Requires enthusiasm.",
        }
        result = _assemble_prompt(instructions, CATEGORIES)
        lines = result.split("\n")
        # Instructions should follow category order, not insertion order
        assert "- Positive" in lines[2]
        assert "- Neutral" in lines[3]

    def test_categories_without_instructions_omitted(self):
        instructions = {"Negative": "Must express displeasure."}
        result = _assemble_prompt(instructions, CATEGORIES)
        assert "Positive" not in result
        assert "Neutral" not in result
        assert "- Negative: Must express displeasure." in result


# ── _print_classification_summary ────────────────────────────────────────

class TestPrintClassificationSummary:
    def test_output_contains_label_and_metrics(self):
        metrics = {"accuracy": 0.8, "sensitivity": 0.7, "precision": 0.9}
        per_cat = _compute_per_category_metrics(CORRECTIONS_WITH_ERRORS, CATEGORIES)

        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_classification_summary("Baseline", metrics, per_cat, CATEGORIES, 2)
        output = buf.getvalue()

        assert "Baseline results:" in output
        assert "80%" in output
        assert "flips=2" in output
        # Check per-category rows are present
        for cat in CATEGORIES:
            assert cat in output

    def test_long_category_name_truncated(self):
        long_cat = "A" * 50
        per_cat = _compute_per_category_metrics(
            [
                {
                    "input": "x",
                    "original": {long_cat: 1},
                    "corrected": {long_cat: 1},
                    "changed": [],
                }
            ],
            [long_cat],
        )
        metrics = {"accuracy": 1.0, "sensitivity": 1.0, "precision": 1.0}

        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_classification_summary("Test", metrics, per_cat, [long_cat], 0)
        output = buf.getvalue()

        # Should be truncated to 40 chars: 37 chars + "..."
        assert "..." in output
        assert long_cat not in output  # full name should NOT appear


# ── prompt_tune (integration, mocked externals) ─────────────────────────

class TestPromptTuneIntegration:
    """
    End-to-end test of prompt_tune with mocked LLM calls and user review.

    Simulates:
      1. Baseline: model makes 2 errors, user corrects them
      2. Category optimization: meta-LLM generates an instruction
      3. Re-classification: all errors fixed → early stop
    """

    @patch("cat_stack.prompt_tune._generate_category_instruction")
    @patch("cat_stack.prompt_tune._classify_and_score")
    @patch("cat_stack.prompt_tune.collect_corrections")
    def test_full_flow_improvement(self, mock_collect, mock_score, mock_gen_instr):
        # Baseline (collect_corrections): returns corrections with errors
        baseline_result = {
            "corrections": CORRECTIONS_WITH_ERRORS,
            "metrics": {"accuracy": 0.6, "sensitivity": 0.6, "precision": 0.6},
            "total_flips": 2,
            "sample_indices": [0, 1, 2, 3, 4],
        }
        mock_collect.return_value = baseline_result

        # Iteration re-classification (_classify_and_score): all fixed
        fixed_result = {
            "corrections": CORRECTIONS_FIXED,
            "metrics": {"accuracy": 1.0, "sensitivity": 1.0, "precision": 1.0},
            "total_flips": 0,
            "sample_indices": [0, 1, 2, 3, 4],
        }
        mock_score.return_value = fixed_result

        # Meta-LLM returns an instruction for the first error category
        mock_gen_instr.return_value = "Assign only for factual statements with no emotional valence."

        result = prompt_tune(
            input_data=["text1", "text2", "text3", "text4", "text5"],
            categories=CATEGORIES,
            api_key="fake-key",
            sample_size=5,
            max_iterations=3,
            ui="terminal",
        )

        # Should return the optimized prompt
        assert "system_prompt" in result
        assert result["system_prompt"] != ""
        assert "Classification guidance per category:" in result["system_prompt"]

        # Should have 2 iterations: baseline + 1 improvement
        assert len(result["iterations"]) == 2
        assert result["iterations"][0]["label"] == "baseline"

        # Best iteration should have perfect metrics
        best_per_cat = result["per_category_summary"]
        for cat in CATEGORIES:
            assert best_per_cat[cat]["fp"] == 0
            assert best_per_cat[cat]["fn"] == 0

    @patch("cat_stack.prompt_tune.collect_corrections")
    def test_perfect_baseline_skips_tuning(self, mock_collect):
        perfect_result = {
            "corrections": CORRECTIONS_PERFECT,
            "metrics": {"accuracy": 1.0, "sensitivity": 1.0, "precision": 1.0},
            "total_flips": 0,
            "sample_indices": [0, 1, 2, 3, 4],
        }
        mock_collect.return_value = perfect_result

        result = prompt_tune(
            input_data=["a", "b", "c", "d", "e"],
            categories=CATEGORIES,
            api_key="fake-key",
            sample_size=5,
            ui="terminal",
        )

        # No tuning needed → empty prompt, single iteration
        assert result["system_prompt"] == ""
        assert len(result["iterations"]) == 1
        # collect_corrections should only be called once (baseline)
        assert mock_collect.call_count == 1

    @patch("cat_stack.prompt_tune.collect_corrections")
    def test_user_cancels_baseline(self, mock_collect):
        mock_collect.return_value = None

        result = prompt_tune(
            input_data=["a"],
            categories=CATEGORIES,
            api_key="fake-key",
            ui="terminal",
        )

        assert result["system_prompt"] == ""
        assert result["iterations"] == []

    def test_invalid_optimize_raises(self):
        with pytest.raises(ValueError, match="optimize must be"):
            prompt_tune(
                input_data=["a"],
                categories=CATEGORIES,
                api_key="fake-key",
                optimize="invalid",
            )

    @patch("cat_stack.prompt_tune._generate_category_instruction")
    @patch("cat_stack.prompt_tune._classify_and_score")
    @patch("cat_stack.prompt_tune.collect_corrections")
    def test_regression_reverts_instruction(self, mock_collect, mock_score, mock_gen_instr):
        # Baseline: 2 errors
        baseline_result = {
            "corrections": CORRECTIONS_WITH_ERRORS,
            "metrics": {"accuracy": 0.6, "sensitivity": 0.6, "precision": 0.6},
            "total_flips": 2,
            "sample_indices": [0, 1, 2, 3, 4],
        }
        mock_collect.return_value = baseline_result

        # After instruction: worse (3 errors via extra FP on Positive)
        worse_corrections = []
        for c in CORRECTIONS_WITH_ERRORS:
            item = {
                "input": c["input"],
                "original": dict(c["original"]),
                "corrected": dict(c["corrected"]),
                "changed": list(c["changed"]),
            }
            worse_corrections.append(item)
        # Add another FP on Positive for item 4
        worse_corrections[4]["original"]["Positive"] = 1
        worse_corrections[4]["changed"].append("Positive")
        worse_result = {
            "corrections": worse_corrections,
            "metrics": {"accuracy": 0.5, "sensitivity": 0.5, "precision": 0.5},
            "total_flips": 3,
            "sample_indices": [0, 1, 2, 3, 4],
        }

        mock_score.return_value = worse_result
        mock_gen_instr.return_value = "Bad instruction."

        result = prompt_tune(
            input_data=["a", "b", "c", "d", "e"],
            categories=CATEGORIES,
            api_key="fake-key",
            sample_size=5,
            max_iterations=2,
            ui="terminal",
        )

        # Regression should be reverted — prompt may end up empty if all attempts regressed
        assert "iterations" in result
        # At minimum we have baseline + attempts
        assert len(result["iterations"]) >= 2
