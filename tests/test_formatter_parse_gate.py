"""
The JSON formatter runs only on replies that don't parse as JSON.

A reply such as {"1": "NA", "2": "NA", "3": "NA"} is valid JSON with no usable
0/1 codes. The formatter's prompt makes it emit all N categories and default
ambiguous ones to "0", so it turned that abstention into all zeros, which then
counted as a real vote (and blocked every "1" in a unanimous ensemble). Now a
reply that parses is left as a failure; the formatter still repairs prose,
markdown and cut-off JSON.

UnifiedLLMClient.complete and the formatter model are mocked; no network, no
torch.
"""

import pandas as pd
import pytest

import catstack._formatter as formatter_mod
from catstack._utils import reply_parses_as_json
from catstack.classify import classify


CATEGORIES = ["A", "B", "C"]
ALL_ZERO = '{"1":"0","2":"0","3":"0"}'


class TestReplyParsesAsJson:
    @pytest.mark.parametrize("reply", [
        '{"1":"NA","2":"NA","3":"NA"}',
        '{"1": null, "2": null}',
        'Here is my answer: {"1": "unknown"}',
        '```json\n{"category_2": 0}\n```',
    ])
    def test_valid_json_object(self, reply):
        assert reply_parses_as_json(reply)

    @pytest.mark.parametrize("reply", [
        None,
        "",
        "I can't classify this response.",
        "1: present, 2: absent, 3: absent",
        '{"1": "1", "2": ',
        "{1: yes, 2: no}",
        '["1", "0", "0"]',
    ])
    def test_not_a_json_object(self, reply):
        assert not reply_parses_as_json(reply)


@pytest.fixture
def fake_formatter(monkeypatch):
    """Formatter that is available, loads instantly, and answers all zeros."""
    calls = []

    def fake_run(raw_output, categories, model, tokenizer, device):
        calls.append(raw_output)
        return ALL_ZERO

    monkeypatch.setattr(formatter_mod, "ensure_formatter_available", lambda: True)
    monkeypatch.setattr(formatter_mod, "load_formatter", lambda: (None, None, "cpu"))
    monkeypatch.setattr(formatter_mod, "run_formatter", fake_run)
    return calls


def _run(monkeypatch, reply):
    def fake_complete(self, messages, json_schema=None, creativity=None,
                      thinking_budget=None, max_retries=5):
        return reply, None

    monkeypatch.setattr("catstack._providers.UnifiedLLMClient.complete", fake_complete)
    return classify(
        input_data=pd.DataFrame({"text": ["hello world"]}),
        categories=CATEGORIES,
        user_model="gpt-4o",
        api_key="sk-fake",
        json_formatter=True,
        json_retries=0,
        batch_retries=0,
        check_verbosity=False,
        add_other=False,
        parallel=False,
    )


def test_all_na_json_skips_formatter_and_fails(monkeypatch, fake_formatter):
    df = _run(monkeypatch, '{"1":"NA","2":"NA","3":"NA"}')
    assert fake_formatter == []
    assert df.loc[0, "processing_status"] == "error"
    for i in range(1, 4):
        assert pd.isna(df.loc[0, f"category_{i}"])


def test_prose_reply_still_goes_to_formatter(monkeypatch, fake_formatter):
    df = _run(monkeypatch, "Category A: present. B and C: absent.")
    assert fake_formatter == ["Category A: present. B and C: absent."]
    assert df.loc[0, "processing_status"] == "success"
    for i in range(1, 4):
        assert df.loc[0, f"category_{i}"] == 0
