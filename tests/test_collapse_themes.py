"""Tests for collapse_themes() — mocked LLM, no live API calls."""

from unittest.mock import MagicMock, patch

import catstack
from catstack import collapse_themes


def _half_complete(messages, **kwargs):
    """Fake LLM: return the first half of the batch as a numbered list (simulates
    the model dropping duplicates), so each pass contracts."""
    content = messages[0]["content"]
    blob = content.split("```")[1]
    items = [x.strip() for x in blob.split(";") if x.strip()]
    keep = items[: max(1, len(items) // 2)]
    reply = "\n".join(f"{i + 1}. {it}" for i, it in enumerate(keep))
    return reply, None


def test_importable():
    assert callable(collapse_themes)
    assert "collapse_themes" in catstack.__all__


@patch("catstack.collapse_themes.detect_provider", return_value="openai")
@patch("catstack.collapse_themes.UnifiedLLMClient")
def test_passes_loop_and_monotone(mock_cls, mock_dp):
    inst = MagicMock()
    inst.complete.side_effect = _half_complete
    mock_cls.return_value = inst

    themes = [f"category {i}" for i in range(100)]
    common = dict(api_key="k", batch_size=40, embedding_merge_threshold=None,
                  dedupe_threshold=1.0, shuffle=False)

    out1 = collapse_themes(themes, passes=1, **common)
    out3 = collapse_themes(themes, passes=3, **common)

    assert isinstance(out3, list)
    # more passes never grows the result; both stay within the input
    assert len(out3) <= len(out1) <= len(themes)
    assert inst.complete.called


@patch("catstack.collapse_themes.detect_provider", return_value="openai")
@patch("catstack.collapse_themes.UnifiedLLMClient")
def test_no_data_loss_on_error(mock_cls, mock_dp):
    inst = MagicMock()
    inst.complete.return_value = ("", "simulated API error")  # every batch errors
    mock_cls.return_value = inst

    themes = [f"category {i}" for i in range(50)]
    # Disable every merge step (pre-LLM embedding, JW dedupe, and the final global
    # consolidation) so this isolates the error path: on a failed batch nothing dropped.
    out = collapse_themes(themes, api_key="k", passes=1, batch_size=40,
                          embedding_merge_threshold=None, dedupe_threshold=1.0,
                          final_consolidation=False, shuffle=False)
    # On error a batch is returned unchanged -> nothing is dropped
    assert len(out) == 50


@patch("catstack.collapse_themes.detect_provider", return_value="openai")
@patch("catstack.collapse_themes.UnifiedLLMClient")
def test_aggressive_routes_to_merge_prompt(mock_cls, mock_dp):
    inst = MagicMock()
    inst.complete.side_effect = _half_complete
    mock_cls.return_value = inst

    collapse_themes(["a", "b", "c"], api_key="k", passes=1, aggressive=True,
                    embedding_merge_threshold=None, shuffle=False)
    prompt = inst.complete.call_args.kwargs["messages"][0]["content"]
    assert "consolidating" in prompt.lower()


@patch("catstack.collapse_themes.detect_provider", return_value="openai")
@patch("catstack.collapse_themes.UnifiedLLMClient")
def test_unique_is_default_prompt(mock_cls, mock_dp):
    inst = MagicMock()
    inst.complete.side_effect = _half_complete
    mock_cls.return_value = inst

    collapse_themes(["a", "b", "c"], api_key="k", passes=1,
                    embedding_merge_threshold=None, shuffle=False)
    prompt = inst.complete.call_args.kwargs["messages"][0]["content"]
    assert "unique categories" in prompt.lower()


@patch("catstack.collapse_themes.detect_provider", return_value="openai")
@patch("catstack.collapse_themes.UnifiedLLMClient")
def test_unparseable_reply_keeps_batch(mock_cls, mock_dp):
    """A reply with no numbered list must not silently drop the batch — this
    previously zeroed out merge-mode batches (no subset fallback there)."""
    inst = MagicMock()
    inst.complete.return_value = ("Sure! The categories are alpha, beta and gamma.", None)
    mock_cls.return_value = inst

    themes = [f"category {i}" for i in range(10)]
    out = collapse_themes(themes, api_key="k", passes=1, aggressive=True,
                          embedding_merge_threshold=None, dedupe_threshold=1.0,
                          final_consolidation=False, shuffle=False)
    assert len(out) == 10


@patch("catstack.collapse_themes.detect_provider", return_value="openai")
@patch("catstack.collapse_themes.UnifiedLLMClient")
def test_fixed_point_stops_early(mock_cls, mock_dp):
    """Once two consecutive passes change nothing, remaining passes are skipped
    instead of burning one LLM call per batch per pass."""

    def echo(messages, **kwargs):
        blob = messages[0]["content"].split("```")[1]
        items = [x.strip() for x in blob.split(";") if x.strip()]
        return "\n".join(f"{i + 1}. {it}" for i, it in enumerate(items)), None

    inst = MagicMock()
    inst.complete.side_effect = echo
    mock_cls.return_value = inst

    themes = [f"category {i}" for i in range(10)]  # already distinct: echo = fixed point
    out = collapse_themes(themes, api_key="k", passes=10, batch_size=40,
                          embedding_merge_threshold=None, dedupe_threshold=1.0,
                          final_consolidation=False, shuffle=False)
    assert len(out) == 10
    # pass 0 establishes the baseline; passes 1-2 are the two unchanged passes
    assert inst.complete.call_count == 3


@patch("catstack.collapse_themes.detect_provider", return_value="openai")
@patch("catstack.collapse_themes.UnifiedLLMClient")
def test_top_n_consolidates_to_n(mock_cls, mock_dp):
    """top_n adds one final global call that reduces the result to <= n."""

    def responder(messages, **kwargs):
        content = messages[0]["content"]
        if "EXACTLY" in content:  # the top_n prompt
            return "\n".join(f"{i + 1}. theme {i}" for i in range(10)), None
        return _half_complete(messages, **kwargs)

    inst = MagicMock()
    inst.complete.side_effect = responder
    mock_cls.return_value = inst

    themes = [f"category {i}" for i in range(30)]
    out = collapse_themes(themes, api_key="k", passes=1, batch_size=40, top_n=10,
                          embedding_merge_threshold=None, dedupe_threshold=1.0,
                          final_consolidation=False, shuffle=False)
    assert len(out) == 10
    prompts = [c.kwargs["messages"][0]["content"] for c in inst.complete.call_args_list]
    assert any("EXACTLY 10" in p for p in prompts)


@patch("catstack.collapse_themes.detect_provider", return_value="openai")
@patch("catstack.collapse_themes.UnifiedLLMClient")
def test_top_n_fallback_is_top_by_count(mock_cls, mock_dp):
    """An unparseable top_n reply falls back to the n highest-count labels."""

    def responder(messages, **kwargs):
        content = messages[0]["content"]
        if "EXACTLY" in content:
            return "no numbered list here, sorry", None
        blob = content.split("```")[1]
        items = [x.strip() for x in blob.split(";") if x.strip()]
        return "\n".join(f"{i + 1}. {it}" for i, it in enumerate(items)), None

    inst = MagicMock()
    inst.complete.side_effect = responder
    mock_cls.return_value = inst

    themes = {f"category {i}": 100 - i for i in range(30)}  # counts descending
    out = collapse_themes(themes, api_key="k", passes=1, batch_size=40, top_n=5,
                          embedding_merge_threshold=None, dedupe_threshold=1.0,
                          final_consolidation=False, shuffle=False)
    assert out == [f"category {i}" for i in range(5)]


@patch("catstack.collapse_themes.detect_provider", return_value="openai")
@patch("catstack.collapse_themes.UnifiedLLMClient")
def test_top_n_noop_when_already_small(mock_cls, mock_dp):
    """top_n larger than the surviving list must not issue the extra call."""

    def responder(messages, **kwargs):
        blob = messages[0]["content"].split("```")[1]
        items = [x.strip() for x in blob.split(";") if x.strip()]
        return "\n".join(f"{i + 1}. {it}" for i, it in enumerate(items)), None

    inst = MagicMock()
    inst.complete.side_effect = responder
    mock_cls.return_value = inst

    out = collapse_themes(["a", "b", "c"], api_key="k", passes=1, top_n=10,
                          embedding_merge_threshold=None, dedupe_threshold=1.0,
                          final_consolidation=False, shuffle=False)
    assert len(out) == 3
    prompts = [c.kwargs["messages"][0]["content"] for c in inst.complete.call_args_list]
    assert not any("EXACTLY" in p for p in prompts)


@patch("catstack.collapse_themes.detect_provider", return_value="openai")
@patch("catstack.collapse_themes.UnifiedLLMClient")
def test_prune_path_parallel_with_top_n(mock_cls, mock_dp):
    """The prune path must survive parallel batches (max_workers > 1) and apply
    the top_n consolidation at its end. Covers the previously untested prune
    strategy: batched prune -> global master prune -> top_n."""

    def responder(messages, **kwargs):
        content = messages[0]["content"]
        if "EXACTLY" in content:  # the top_n prompt
            return "\n".join(f"{i + 1}. final {i}" for i in range(4)), None
        blob = content.split("```")[1]
        items = [x.strip() for x in blob.split(";") if x.strip()]
        return "\n".join(f"{i + 1}. {it}" for i, it in enumerate(items)), None

    inst = MagicMock()
    inst.complete.side_effect = responder
    mock_cls.return_value = inst

    themes = [f"category {i}" for i in range(60)]
    out = collapse_themes(themes, api_key="k", prune=True, prune_threshold=25,
                          top_n=4, max_workers=4,
                          embedding_merge_threshold=None, dedupe_threshold=1.0,
                          shuffle=False)
    assert out == [f"final {i}" for i in range(4)]


def test_requires_api_key():
    try:
        collapse_themes(["a", "b"], api_key=None)
        assert False, "expected ValueError without api_key"
    except ValueError:
        pass


def test_two_phase_routes_each_model_to_its_phase():
    """unique_model + merge_model must instantiate two distinct clients (each with its
    own provider) and route the unique-thinning passes to unique_model and the merge
    passes to merge_model."""
    created = []   # (model, provider) per client instantiation
    calls = []     # model used for each complete() call

    def factory(provider=None, api_key=None, model=None):
        created.append((model, provider))

        class C:
            def complete(self, *a, **k):
                calls.append(model)
                return ("1. alpha\n2. beta\n3. gamma", None)

        return C()

    with patch("catstack.collapse_themes.UnifiedLLMClient", side_effect=factory), \
         patch("catstack.collapse_themes.detect_provider",
               side_effect=lambda m, s: f"prov[{m}|{s}]"):
        collapse_themes(
            [f"x{i}" for i in range(8)], api_key="k", description="Q",
            aggressive=True, passes=1,
            unique_model="UNIQUE-MODEL", unique_model_source="huggingface", unique_passes=2,
            merge_model="MERGE-MODEL", merge_model_source="openai",
            embedding_merge_threshold=None, dedupe_threshold=1.0,
            final_consolidation=False, shuffle=False,
        )

    models_created = {m for m, _ in created}
    assert models_created == {"UNIQUE-MODEL", "MERGE-MODEL"}
    # each model's provider was resolved from its own *_model_source
    prov = dict(created)
    assert prov["UNIQUE-MODEL"] == "prov[UNIQUE-MODEL|huggingface]"
    assert prov["MERGE-MODEL"] == "prov[MERGE-MODEL|openai]"
    # unique phase ran on unique_model for unique_passes; merge phase on merge_model
    assert calls.count("UNIQUE-MODEL") == 2
    assert calls.count("MERGE-MODEL") == 1


def test_single_model_backward_compatible():
    """With no unique_model/merge_model, only one client is created from user_model."""
    created = []

    def factory(provider=None, api_key=None, model=None):
        created.append(model)
        inst = MagicMock()
        inst.complete.side_effect = _half_complete
        return inst

    with patch("catstack.collapse_themes.UnifiedLLMClient", side_effect=factory), \
         patch("catstack.collapse_themes.detect_provider", return_value="openai"):
        collapse_themes(["a", "b", "c", "d"], api_key="k", passes=1, user_model="solo",
                        embedding_merge_threshold=None, dedupe_threshold=1.0,
                        final_consolidation=False, shuffle=False)

    assert created == ["solo"]


@patch("catstack.collapse_themes.detect_provider", return_value="openai")
@patch("catstack.collapse_themes.UnifiedLLMClient")
def test_dict_input_topn_noop_returns_label_list(mock_cls, mock_dp):
    """Regression: a {label: count} dict reaching a no-op exit (passes=0,
    no final consolidation, top_n larger than the list) must come back as
    the documented list[str], not the input dict — the dict leak made
    downstream DataFrame writes save counts instead of labels."""
    inst = MagicMock()
    mock_cls.return_value = inst

    out = collapse_themes({"family bonds": 9, "trust": 4}, api_key="k",
                          passes=0, final_consolidation=False, top_n=12)
    assert out == ["family bonds", "trust"]
    inst.complete.assert_not_called()
