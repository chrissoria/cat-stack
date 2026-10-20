"""classify(examples=[...]): few-shot examples with no cap, after example1-6."""

from unittest.mock import patch

from catstack.text_functions_ensemble import collect_examples, format_examples_text


def test_collect_order_and_no_cap():
    many = [f"ex{i}" for i in range(20)]
    out = collect_examples("a", None, "c", examples=many)
    assert out == ["a", "c"] + many
    assert collect_examples(examples="one") == ["one"]
    assert collect_examples() == []


def test_format_numbers_on():
    assert format_examples_text(["a", "b"]) == "Example 1: a\nExample 2: b"
    assert format_examples_text([]) == ""


def _sent_prompts(**kw):
    import catstack
    sent = []

    def fake_complete(self, messages, **k):
        sent.append(messages[-1]["content"])
        return '{"1":"1","2":"0"}', None

    with patch("cat_stack._providers.UnifiedLLMClient.complete", fake_complete):
        catstack.classify(input_data=["moved in with my wife"], categories=["Partner", "Money"],
                          api_key="k", user_model="gpt-4o-mini", description="Why did you move?",
                          check_verbosity=False, add_other=False, **kw)
    return [m for m in sent if "Categorize this" in m]


def test_sync_path_sends_all_examples():
    many = [f'Response: "r{i}" -> {{"1":"0","2":"1"}}' for i in range(20)]
    p = _sent_prompts(example1="first", examples=many)[0]
    assert "Example 1: first" in p
    assert 'Example 21: Response: "r19"' in p
    assert p.index("Example 2: ") < p.index("Example 21: ")


def test_example1_6_unchanged_without_list():
    p = _sent_prompts(example1="only one")[0]
    assert "Example 1: only one" in p and "Example 2:" not in p


def test_batch_path_carries_examples():
    import catstack
    captured = {}

    def fake_run(items, cfg, categories, prompt_params, **k):
        captured.update(prompt_params)
        import pandas as pd
        return pd.DataFrame({"survey_input": items})

    with patch("cat_stack._batch.run_batch_classify", fake_run):
        try:
            catstack.classify(input_data=["x"], categories=["A", "B"], api_key="k",
                              user_model="gpt-4o-mini", batch_mode=True,
                              check_verbosity=False, add_other=False,
                              examples=["e1", "e2", "e3", "e4", "e5", "e6", "e7"])
        except Exception:
            pass  # only the prompt params matter here
    assert captured.get("examples_text", "").splitlines()[-1] == "Example 7: e7"
