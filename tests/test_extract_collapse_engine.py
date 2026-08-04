"""Tests for extract(engine=...) — mocked LLM, no live API calls.

engine="collapse" must route text input through raw extraction (explore path)
and consolidate with collapse_themes(top_n=max_categories); engine="legacy"
must preserve the pre-2.5 single-merge behavior.
"""

from unittest.mock import MagicMock, patch

from catstack import extract

RESPONSES = [f"response {i}" for i in range(12)]


def _responder(messages, **kwargs):
    """Fake LLM covering all four call shapes.

    - extraction chunk (system+user, blob of responses): fixed 5 categories
    - collapse pass (1 msg, ``` blob of labels): echo the labels
    - top_n call (1 msg, EXACTLY): 3 winners
    - legacy merge (1 msg, no ``` blob): 3 merged labels
    """
    content = messages[-1]["content"]
    if len(messages) == 2:
        return "1. employment\n2. education\n3. housing\n4. family\n5. health", None
    if "EXACTLY" in content:
        return "1. employment\n2. education\n3. housing", None
    if "```" in content:
        items = [x.strip() for x in content.split("```")[1].split(";") if x.strip()]
        return "\n".join(f"{i + 1}. {it}" for i, it in enumerate(items)), None
    return "1. legacy merged\n2. two\n3. three", None


def _mock_client():
    inst = MagicMock()
    inst.complete.side_effect = _responder
    return inst


def _patches(inst):
    return [
        patch("catstack.text_functions.UnifiedLLMClient", return_value=inst),
        patch("catstack.text_functions.detect_provider", return_value="openai"),
        patch("catstack.collapse_themes.UnifiedLLMClient", return_value=inst),
        patch("catstack.collapse_themes.detect_provider", return_value="openai"),
    ]


def test_collapse_engine_is_default_and_caps_at_max_categories():
    inst = _mock_client()
    ps = _patches(inst)
    [p.start() for p in ps]
    try:
        res = extract(RESPONSES, api_key="k", input_type="text", description="Q",
                      max_categories=3, divisions=2, iterations=1,
                      categories_per_chunk=5, random_state=0,
                      collapse_kwargs={"passes": 1, "embedding_merge_threshold": None,
                                       "final_consolidation": False,
                                       "dedupe_threshold": 1.0, "shuffle": False})
    finally:
        [p.stop() for p in ps]

    assert set(res) == {"counts_df", "top_categories", "raw_top_text"}
    assert res["top_categories"] == ["employment", "education", "housing"]
    assert res["raw_top_text"] == ""
    # inventory keeps ALL extracted labels (2 chunks x 5), nothing pre-truncated
    assert int(res["counts_df"]["counts"].sum()) == 10
    prompts = [c.kwargs["messages"][-1]["content"]
               for c in inst.complete.call_args_list]
    assert any("EXACTLY 3" in p for p in prompts)


def test_legacy_engine_preserves_old_merge():
    inst = _mock_client()
    ps = _patches(inst)
    [p.start() for p in ps]
    try:
        res = extract(RESPONSES, api_key="k", input_type="text", description="Q",
                      max_categories=3, divisions=2, iterations=1,
                      categories_per_chunk=5, random_state=0, engine="legacy")
    finally:
        [p.stop() for p in ps]

    assert res["top_categories"] == ["legacy merged", "two", "three"]
    assert res["raw_top_text"] != ""
    prompts = [c.kwargs["messages"][-1]["content"]
               for c in inst.complete.call_args_list]
    assert not any("EXACTLY" in p for p in prompts)


def test_engine_validated():
    try:
        extract(RESPONSES, api_key="k", input_type="text", engine="bogus")
        assert False, "expected ValueError for unknown engine"
    except ValueError as e:
        assert "engine" in str(e)
