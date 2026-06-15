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
