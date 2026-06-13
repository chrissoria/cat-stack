"""v1.6.9: defensive formatter-tokenizer load.

`_load_formatter_tokenizer` must survive a `tokenizer_config.json` whose
`extra_special_tokens` is a LIST (transformers 4.56-4.57 crash) by snapshotting
the repo, normalizing the field to {}, and loading from the patched copy.
"""
import json
import os
import tempfile

import catstack._formatter as fmt


class _FakeTok:
    """Stand-in tokenizer that records where it was loaded from."""
    def __init__(self, src):
        self.src = src


def _make_auto_tokenizer(fail_on_repo):
    """Return a fake AutoTokenizer whose from_pretrained raises the
    list-keys error when called on the repo constant, but succeeds on a
    local dir."""
    class _AutoTok:
        @staticmethod
        def from_pretrained(path, **kw):
            if path == fmt._MERGED_MODEL_REPO and fail_on_repo:
                raise AttributeError("'list' object has no attribute 'keys'")
            return _FakeTok(path)
    return _AutoTok


def test_happy_path_returns_tokenizer():
    tok = fmt._load_formatter_tokenizer(_make_auto_tokenizer(fail_on_repo=False))
    assert isinstance(tok, _FakeTok)
    assert tok.src == fmt._MERGED_MODEL_REPO


def test_unrelated_error_reraises():
    class _AutoTok:
        @staticmethod
        def from_pretrained(path, **kw):
            raise ValueError("totally unrelated failure")
    try:
        fmt._load_formatter_tokenizer(_AutoTok)
        assert False, "should have re-raised"
    except ValueError as e:
        assert "unrelated" in str(e)


def test_list_extra_special_tokens_normalized(monkeypatch):
    # snapshot dir with a malformed (list) extra_special_tokens
    snap = tempfile.mkdtemp()
    cfg = {"tokenizer_class": "Qwen2Tokenizer",
           "extra_special_tokens": ["<|im_start|>", "<|im_end|>"]}
    with open(os.path.join(snap, "tokenizer_config.json"), "w") as f:
        json.dump(cfg, f)
    monkeypatch.setattr(fmt, "snapshot_download", lambda repo: snap, raising=False)
    # huggingface_hub.snapshot_download is imported inside the function; patch there
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: snap)

    tok = fmt._load_formatter_tokenizer(_make_auto_tokenizer(fail_on_repo=True))
    # loaded from a patched local dir, not the repo
    assert isinstance(tok, _FakeTok) and tok.src != fmt._MERGED_MODEL_REPO
    # the patched config has extra_special_tokens == {}
    with open(os.path.join(tok.src, "tokenizer_config.json")) as f:
        patched = json.load(f)
    assert patched["extra_special_tokens"] == {}
