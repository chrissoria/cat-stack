"""prompt_style: the current template is unchanged, and "catllm_0.0.97"
renders cat-llm 0.0.97's survey wording (without its stray indentation)."""

import functools

import pytest

from catstack.text_functions_ensemble import (
    PROMPT_STYLES,
    build_context_line,
    build_text_classification_prompt,
    check_prompt_style,
)

CATS = "1. Partner/Spouse\n2. Financial"


def _content(**kw):
    msgs = build_text_classification_prompt(
        response_text="moved in with my wife", categories_str=CATS, **kw)
    return msgs[-1]["content"]


def test_current_template_unchanged():
    ctx = build_context_line("Why did you move?")
    assert ctx == "Context: Why did you move?."
    assert _content(survey_question_context=ctx) == (
        "Context: Why did you move?.\n"
        'Categorize this text response "moved in with my wife" into the following categories that apply:\n'
        "1. Partner/Spouse\n2. Financial\n"
        "\n"
        'Provide your answer in JSON format where the category number is the key and "1" if present, "0" if not.'
    )


def test_legacy_template_matches_0_0_97_wording():
    ctx = build_context_line("Why did you move?", "catllm_0.0.97")
    assert ctx == "A respondent was asked: Why did you move?."
    assert _content(survey_question_context=ctx, prompt_style="catllm_0.0.97") == (
        "A respondent was asked: Why did you move?.\n"
        'Categorize this survey response "moved in with my wife" into the following categories that apply:\n'
        "1. Partner/Spouse\n2. Financial\n"
        "\n"
        "Provide your work in JSON format where the number belonging to each category is the key "
        "and a 1 if the category is present and a 0 if it is not present as key values."
    )


def test_legacy_chain_of_thought_json_line():
    out = _content(survey_question_context="", prompt_style="catllm_0.0.97", chain_of_thought=True)
    assert 'Categorize this survey response "moved in with my wife"' in out
    assert out.endswith(
        "Provide your reasoning for each category, then provide your final answer in JSON format "
        "where the number belonging to each category is the key and a 1 if the category is present "
        "and a 0 if it is not present as key values.")


def test_legacy_system_prompt_still_prepended():
    out = _content(survey_question_context="", prompt_style="catllm_0.0.97",
                   system_prompt="Answer conservatively.")
    assert out.startswith("Answer conservatively.\n\n")


def test_empty_question_gives_no_context_line():
    assert build_context_line("", "catllm_0.0.97") == ""
    assert build_context_line("") == ""


def test_unknown_style_rejected():
    assert PROMPT_STYLES == ("current", "catllm_0.0.97")
    with pytest.raises(ValueError):
        check_prompt_style("legacy")
    with pytest.raises(ValueError):
        build_text_classification_prompt("x", CATS, prompt_style="legacy")


def test_partial_carries_style_for_chunked_path():
    fn = functools.partial(build_text_classification_prompt, prompt_style="catllm_0.0.97")
    assert fn.keywords["prompt_style"] == "catllm_0.0.97"
    assert "survey response" in fn("x", CATS)[-1]["content"]


def test_classify_rejects_unknown_style_before_any_call():
    import catstack
    with pytest.raises(ValueError):
        catstack.classify(input_data=["x"], categories=["a"], api_key="k",
                          user_model="gpt-4o-mini", prompt_style="legacy")


# --- threading: batch consumers, producers, and the synchronous path --------

from unittest.mock import MagicMock, patch

from cat_stack._batch import _run_one_batch_job, _run_one_sync_model


def _cfg(model="gpt-4o", provider="openai"):
    return {"model": model, "provider": provider, "api_key": "fake-key",
            "sanitized_name": model.replace("-", "_").replace(".", "_")}


def _params(**extra):
    p = {"categories_str": "1. yes\n2. no", "survey_question_context": "", "examples_text": "",
         "chain_of_thought": False, "context_prompt": False, "step_back_prompt": False,
         "stepback_insights": {}, "system_prompt": "", "json_schema": {"type": "object"},
         "creativity": 0, "thinking_budget": 0, "multi_label": True}
    p.update(extra)
    return p


@patch("cat_stack._batch._poll_batch_job")
@patch("cat_stack._batch._download_batch_results", return_value="")
@patch("cat_stack._batch._parse_batch_results", return_value={})
@patch("cat_stack._batch._create_batch_job", return_value="job-xyz")
@patch("cat_stack._batch._upload_jsonl", return_value="file-xyz")
@pytest.mark.parametrize("style", ["catllm_0.0.97", None])
def test_batch_job_forwards_prompt_style(_upload, _create, _parse, _download, _poll, style):
    params = _params(**({"prompt_style": style} if style else {}))
    with patch("cat_stack.text_functions_ensemble.build_text_classification_prompt") as mock_build:
        mock_build.return_value = [{"role": "user", "content": "hi"}]
        _run_one_batch_job(cfg=_cfg(), items=["text-1"], prompt_params=params)
    assert mock_build.call_args.kwargs.get("prompt_style") == (style or "current")


def test_sync_fallback_forwards_prompt_style():
    with patch("cat_stack.text_functions_ensemble.build_text_classification_prompt") as mock_build, \
         patch("cat_stack._batch.UnifiedLLMClient") as mock_client_cls:
        mock_build.return_value = [{"role": "user", "content": "hi"}]
        client = MagicMock(); client.complete.return_value = ('{"1":"1","2":"0"}', None)
        mock_client_cls.return_value = client
        _run_one_sync_model(cfg=_cfg(model="meta-llama/Llama-3.1-8B-Instruct", provider="huggingface"),
                            items=["text-1"], prompt_params=_params(prompt_style="catllm_0.0.97"))
    assert mock_build.call_args.kwargs.get("prompt_style") == "catllm_0.0.97"


def test_producers_include_prompt_style():
    from pathlib import Path
    src = Path(__file__).resolve().parents[1].joinpath("src/catstack/classify.py").read_text()
    a = src.index("prompt_params = {"); b = src.index("result = run_batch_classify(", a)
    assert '"prompt_style": prompt_style' in src[a:b]
    a = src.index("prompt_params_per_model = {"); b = src.index("result = run_batch_ensemble_classify(", a)
    assert '"prompt_style": prompt_style' in src[a:b]
    assert "survey_question_context = build_context_line(survey_question, prompt_style)" in src
    a = src.index("result = classify_ensemble("); b = src.index(")\n", src.index("system_prompt=system_prompt", a))
    assert "prompt_style=prompt_style" in src[a:b]


def test_sync_path_sends_legacy_prompt():
    """classify() on the synchronous path: the request the client receives
    carries the legacy context line, noun and JSON line."""
    import catstack
    sent = []

    def fake_complete(self, messages, **kw):
        sent.append(messages[-1]["content"])
        return '{"1":"1","2":"0"}', None

    with patch("cat_stack._providers.UnifiedLLMClient.complete", fake_complete):
        catstack.classify(input_data=["moved in with my wife"], categories=["Partner", "Money"],
                          api_key="k", user_model="gpt-4o-mini", description="Why did you move?",
                          prompt_style="catllm_0.0.97", check_verbosity=False, add_other=False)
    prompts = [m for m in sent if "Categorize this" in m]   # skip the connectivity check
    assert prompts, "no classification request was built"
    p = prompts[0]
    assert p.startswith("A respondent was asked: Why did you move?.\n")
    assert 'Categorize this survey response "moved in with my wife"' in p
    assert p.endswith("a 0 if it is not present as key values.")
