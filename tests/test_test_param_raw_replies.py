"""
classify(test=True): keep each model's raw reply per row (cat-stack 2.6.6).

Output files keep only the parsed 0/1 matrix, so a row that came back
"success" with every category 0 could not be traced to what the model said.
The hidden `test` switch adds raw_reply / raw_attempts / parsed_json /
raw_error columns (suffixed per model in an ensemble). test=False must leave
the output exactly as before.

All clients are mocked; every input string is synthetic.
"""

import json
from unittest.mock import patch

import pandas as pd
import pytest

import importlib
C = importlib.import_module("catstack.classify")  # module, not the function
import catstack._batch as B


RAW_COLS = ["raw_reply", "raw_attempts", "parsed_json", "raw_error"]
CATS3 = ["Alpha", "Beta", "Gamma"]
TEXT_REPLY = '{"1": "the respondent mentions alpha", "3": "0"}'
GOOD_REPLY = '{"1":"1","2":"0","3":"0"}'

COMMON = dict(
    check_verbosity=False,
    add_other=False,
    json_formatter=False,
    use_json_schema=False,
)


def _is_validation_ping(messages):
    """prepare_model_configs() sends one connectivity check per model."""
    return "Reply with exactly" in str(messages[-1].get("content", ""))


def _script(replies_by_model):
    """Patch UnifiedLLMClient.complete to serve scripted replies per model.

    replies_by_model: {model_name: [reply, ...]} - the i-th call for that
    model gets reply i (the last one repeats). Returns (fake, log).
    """
    import threading
    lock = threading.Lock()
    log = {m: 0 for m in replies_by_model}

    def fake_complete(self, messages, json_schema=None, creativity=None,
                      thinking_budget=None, max_retries=5):
        if _is_validation_ping(messages):
            return '{"1":"0"}', None
        with lock:
            replies = replies_by_model[self.model]
            i = log[self.model]
            log[self.model] += 1
        return replies[min(i, len(replies) - 1)], None

    return fake_complete, log


def _model_suffixes(df):
    """Per-model suffixes exactly as the existing category_1_<model> columns
    spell them (sanitized name, creativity tag included)."""
    return [c[len("category_1_"):] for c in df.columns
            if c.startswith("category_1_")
            and c not in ("category_1_consensus", "category_1_agreement")]


def _suffix_for(df, model_prefix):
    (m,) = [s for s in _model_suffixes(df) if s.startswith(model_prefix)
            and not any(o != s and o.startswith(model_prefix) and len(o) < len(s)
                        for o in _model_suffixes(df))]
    return m


# ---------------------------------------------------------------------------
# (1) test=False: no new columns, identical output
# ---------------------------------------------------------------------------

def test_default_off_output_unchanged(monkeypatch):
    texts = ["synthetic reply one", "synthetic reply two"]
    kwargs = dict(user_model="gpt-4o", api_key="sk-fake", parallel=False,
                  batch_retries=0, **COMMON)

    fake, _ = _script({"gpt-4o": [GOOD_REPLY]})
    monkeypatch.setattr("catstack._providers.UnifiedLLMClient.complete", fake)
    base = C.classify(texts, CATS3, **kwargs)

    fake, _ = _script({"gpt-4o": [GOOD_REPLY]})
    monkeypatch.setattr("catstack._providers.UnifiedLLMClient.complete", fake)
    off = C.classify(texts, CATS3, test=False, **kwargs)

    fake, _ = _script({"gpt-4o": [GOOD_REPLY]})
    monkeypatch.setattr("catstack._providers.UnifiedLLMClient.complete", fake)
    on = C.classify(texts, CATS3, test=True, **kwargs)

    pd.testing.assert_frame_equal(base, off)
    assert not any(c in base.columns for c in RAW_COLS)
    # test=True only appends columns; the rest is unchanged and in order
    assert list(on.columns) == list(base.columns) + RAW_COLS
    pd.testing.assert_frame_equal(on[list(base.columns)], base)


def test_default_off_makes_identical_downstream_calls(monkeypatch):
    """test=False passes no new keyword to classify_ensemble / batch runners."""
    fake, _ = _script({"gpt-4o": [GOOD_REPLY]})  # answers the preflight ping
    monkeypatch.setattr("catstack._providers.UnifiedLLMClient.complete", fake)
    frame = pd.DataFrame({"input_data": ["x"]})
    with patch.object(C, "classify_ensemble", return_value=frame) as ens:
        C.classify(["x"], CATS3, user_model="gpt-4o", api_key="k", **COMMON)
    assert "test" not in ens.call_args.kwargs

    with patch("catstack._batch.run_batch_classify", return_value=frame) as rb:
        C.classify(["x"], CATS3, models=[("gpt-4o", "openai", "k")],
                   batch_mode=True, **COMMON)
    assert "test" not in rb.call_args.kwargs


# ---------------------------------------------------------------------------
# (2) single model, synchronous path
# ---------------------------------------------------------------------------

def test_single_model_sync_raw_columns(monkeypatch):
    texts = ["synthetic row with text in a value", "synthetic row with empty json"]
    calls = []

    def fake_complete(self, messages, **kw):
        if _is_validation_ping(messages):
            return '{"1":"0"}', None
        user = messages[-1]["content"]
        calls.append(user)
        if "text in a value" in user:
            return TEXT_REPLY, None
        return "{}", None

    monkeypatch.setattr("catstack._providers.UnifiedLLMClient.complete", fake_complete)
    out = C.classify(texts, CATS3, user_model="gpt-4o", api_key="sk-fake",
                     parallel=False, json_retries=2, batch_retries=0,
                     test=True, **COMMON)

    assert list(out.columns[-4:]) == RAW_COLS
    # Row 0: text under a category key leaves that code unknown, so the
    # reply is unusable (2.6.6): initial call + 2 JSON retries, reply kept
    # verbatim, parsed JSON recorded
    assert out.loc[0, "raw_reply"] == TEXT_REPLY
    assert out.loc[0, "raw_attempts"] == 3
    assert json.loads(out.loc[0, "parsed_json"]) == {
        "1": "the respondent mentions alpha", "3": "0"}
    assert out.loc[0, "raw_error"] is None
    # Row 1: '{}' is unusable -> initial call + 2 JSON retries
    assert out.loc[1, "raw_reply"] == "{}"
    assert out.loc[1, "raw_attempts"] == 3
    assert out.loc[1, "parsed_json"] == "{}"
    assert len(calls) == 6


def test_single_model_sync_attempts_include_batch_retries(monkeypatch):
    fake, log = _script({"gpt-4o": ["{}", "{}", "{}", GOOD_REPLY]})
    monkeypatch.setattr("catstack._providers.UnifiedLLMClient.complete", fake)
    out = C.classify(["synthetic text"], CATS3, user_model="gpt-4o",
                     api_key="sk-fake", parallel=False, json_retries=2,
                     batch_retries=1, retry_delay=0, test=True, **COMMON)
    # 3 calls in the first pass, 1 in the batch retry that succeeded
    assert out.loc[0, "raw_attempts"] == 4 == log["gpt-4o"]
    assert out.loc[0, "raw_reply"] == GOOD_REPLY
    assert out.loc[0, "parsed_json"] == GOOD_REPLY
    assert out.loc[0, "category_1"] == 1


def test_single_model_sync_records_error(monkeypatch):
    def fake_complete(self, messages, **kw):
        return None, "HTTP 500: synthetic outage"

    monkeypatch.setattr("catstack._providers.UnifiedLLMClient.complete", fake_complete)
    out = C.classify(["synthetic text"], CATS3, user_model="gpt-4o",
                     api_key="sk-fake", parallel=False, batch_retries=0,
                     test=True, **COMMON)
    assert out.loc[0, "raw_reply"] is None
    assert out.loc[0, "raw_attempts"] == 1
    assert out.loc[0, "raw_error"] == "HTTP 500: synthetic outage"


def test_nan_row_has_no_raw_reply(monkeypatch):
    fake, _ = _script({"gpt-4o": [GOOD_REPLY]})
    monkeypatch.setattr("catstack._providers.UnifiedLLMClient.complete", fake)
    out = C.classify(["synthetic text", float("nan")], CATS3, user_model="gpt-4o",
                     api_key="sk-fake", parallel=False, batch_retries=0,
                     test=True, **COMMON)
    assert out.loc[1, "processing_status"] == "skipped"
    assert out.loc[1, "raw_reply"] is None
    assert out.loc[1, "raw_attempts"] == 0


# ---------------------------------------------------------------------------
# (3) two-model ensemble: suffixed columns, parallel threads
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("parallel", [True, False])
def test_ensemble_suffixed_columns(monkeypatch, parallel):
    a_reply = '{"1":"1","2":"0","3":"0"}'
    b_reply = 'Here you go: {"1":"0","2":"1","3":"0"}'
    fake, log = _script({"gpt-4o": [a_reply], "claude-sonnet-4-5": [b_reply]})
    monkeypatch.setattr("catstack._providers.UnifiedLLMClient.complete", fake)

    texts = [f"synthetic ensemble row {i}" for i in range(4)]
    out = C.classify(
        texts, CATS3,
        models=[("gpt-4o", "openai", "k1"), ("claude-sonnet-4-5", "anthropic", "k2")],
        parallel=parallel, batch_retries=0, test=True, **COMMON,
    )

    suffixes = _model_suffixes(out)
    assert len(suffixes) == 2
    a, b = _suffix_for(out, "gpt_4o"), _suffix_for(out, "claude_sonnet_4_5")
    expected = [f"{c}_{m}" for m in suffixes for c in RAW_COLS]
    assert list(out.columns[-8:]) == expected
    assert not any(c in out.columns for c in RAW_COLS)  # no unsuffixed copies
    assert (out[f"raw_reply_{a}"] == a_reply).all()
    assert (out[f"raw_reply_{b}"] == b_reply).all()
    assert (out[f"parsed_json_{b}"] == '{"1":"0","2":"1","3":"0"}').all()
    assert (out[f"raw_attempts_{a}"] == 1).all()
    assert out[f"raw_error_{a}"].isna().all()
    assert log == {"gpt-4o": 4, "claude-sonnet-4-5": 4}


# ---------------------------------------------------------------------------
# (4) batch path, mocked at the provider-request level
# ---------------------------------------------------------------------------

def _openai_jsonl(replies):
    lines = []
    for i, content in enumerate(replies):
        lines.append(json.dumps({
            "custom_id": f"item-{i}",
            "response": {"status_code": 200, "body": {
                "choices": [{"message": {"role": "assistant", "content": content}}]}},
        }))
    return "\n".join(lines)


def _patch_batch_http(monkeypatch, replies_by_model):
    """Stand in for upload / create / poll / download; the downloaded JSONL
    then goes through the real _parse_batch_results. Also answers the
    per-model preflight ping so nothing touches the network."""
    def ping_only(self, messages, **kw):
        assert _is_validation_ping(messages), "unexpected sync call in batch test"
        return '{"1":"0"}', None

    monkeypatch.setattr("catstack._providers.UnifiedLLMClient.complete", ping_only)
    monkeypatch.setattr(B, "_upload_jsonl", lambda *a, **k: "file-synthetic")
    monkeypatch.setattr(B, "_create_batch_job",
                        lambda provider, api_key, model, **k: f"job-{model}")
    monkeypatch.setattr(B, "_poll_batch_job", lambda **k: {"status": "completed"})

    def download(provider, api_key, job_id, status_data):
        return _openai_jsonl(replies_by_model[job_id[len("job-"):]])

    monkeypatch.setattr(B, "_download_batch_results", download)


def test_batch_single_model_raw_columns(monkeypatch):
    replies = [TEXT_REPLY, "{}", GOOD_REPLY]
    _patch_batch_http(monkeypatch, {"gpt-4o": replies})
    texts = ["synthetic batch a", "synthetic batch b", "synthetic batch c"]
    out = C.classify(texts, CATS3, models=[("gpt-4o", "openai", "k")],
                     batch_mode=True, test=True, **COMMON)
    assert list(out.columns[-4:]) == RAW_COLS
    assert list(out["raw_reply"]) == replies
    assert list(out["raw_attempts"]) == [1, 1, 1]
    assert out.loc[0, "parsed_json"] == '{"1":"the respondent mentions alpha","3":"0"}'
    assert out.loc[2, "parsed_json"] == GOOD_REPLY

    # Off: same frame minus the raw columns
    _patch_batch_http(monkeypatch, {"gpt-4o": replies})
    off = C.classify(texts, CATS3, models=[("gpt-4o", "openai", "k")],
                     batch_mode=True, **COMMON)
    assert list(out.columns[:-4]) == list(off.columns)
    pd.testing.assert_frame_equal(out[list(off.columns)], off)


def test_batch_ensemble_suffixed_columns(monkeypatch):
    _patch_batch_http(monkeypatch, {
        "gpt-4o": [GOOD_REPLY, GOOD_REPLY],
        "gpt-4o-mini": ['{"1":"0","2":"1","3":"0"}', "{}"],
    })
    out = C.classify(
        ["synthetic batch a", "synthetic batch b"], CATS3,
        models=[("gpt-4o", "openai", "k"), ("gpt-4o-mini", "openai", "k")],
        batch_mode=True, test=True, **COMMON,
    )
    mini, big = _suffix_for(out, "gpt_4o_mini"), _suffix_for(out, "gpt_4o")
    assert mini != big
    assert list(out[f"raw_reply_{mini}"]) == ['{"1":"0","2":"1","3":"0"}', "{}"]
    assert list(out[f"raw_attempts_{big}"]) == [1, 1]
    assert list(out[f"raw_reply_{big}"]) == [GOOD_REPLY, GOOD_REPLY]
    assert "raw_reply" not in out.columns


def test_batch_ensemble_sync_fallback_model(monkeypatch):
    """A provider without a batch API runs synchronously inside the batch
    ensemble; its raw replies are captured too."""
    _patch_batch_http(monkeypatch, {"gpt-4o": [GOOD_REPLY]})
    fake, _ = _script({"Qwen/Qwen3-8B": ['```json\n{"1":"0","2":"0","3":"1"}\n```']})
    monkeypatch.setattr("catstack._providers.UnifiedLLMClient.complete", fake)
    out = C.classify(
        ["synthetic batch a"], CATS3,
        models=[("gpt-4o", "openai", "k"), ("Qwen/Qwen3-8B", "huggingface", "k")],
        batch_mode=True, test=True, **COMMON,
    )
    q, g = _suffix_for(out, "qwen_qwen3_8b"), _suffix_for(out, "gpt_4o")
    assert out.loc[0, f"raw_reply_{q}"] == '```json\n{"1":"0","2":"0","3":"1"}\n```'
    assert out.loc[0, f"raw_attempts_{q}"] == 1
    assert out.loc[0, f"raw_reply_{g}"] == GOOD_REPLY


# ---------------------------------------------------------------------------
# (5) chunked path (categories_per_call)
# ---------------------------------------------------------------------------

def test_chunked_concatenates_chunk_replies(monkeypatch):
    cats = ["Alpha", "Beta", "Gamma", "Delta"]
    # Two chunks of 2; each chunk gets a temporary "Other" (local key 3)
    chunk1 = '{"1":"1","2":"0","3":"0"}'
    chunk2 = '{"1":"0","2":"1","3":"0"}'
    fake, log = _script({"gpt-4o": [chunk1, chunk2]})
    monkeypatch.setattr("catstack._providers.UnifiedLLMClient.complete", fake)

    out = C.classify(["synthetic chunked row"], cats, user_model="gpt-4o",
                     api_key="sk-fake", parallel=False, batch_retries=0,
                     categories_per_call=2, test=True, **COMMON)

    raw = out.loc[0, "raw_reply"]
    assert raw == (
        "----- chunk 1/2 -----\n" + chunk1 + "\n----- chunk 2/2 -----\n" + chunk2
    )
    assert out.loc[0, "raw_attempts"] == 2 == log["gpt-4o"]
    merged = json.loads(out.loc[0, "parsed_json"])
    assert merged["1"] == "1" and merged["4"] == "1"
    assert out.loc[0, "category_1"] == 1 and out.loc[0, "category_4"] == 1
