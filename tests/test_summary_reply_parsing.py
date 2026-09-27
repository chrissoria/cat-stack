"""Tolerant summary parsing (_parse_summary_reply).

A reply that could not be read used to become an empty summary with NO error:
the row failed silently and batch retries never saw it (observed live on an
image summary through claude-agent). Now common near-misses are accepted and
anything still unreadable is an error with a snippet of the reply.
"""
import json

import pytest

from catstack.text_functions_ensemble import _parse_summary_reply


def _summary(json_str):
    return json.loads(json_str)["summary"]


@pytest.mark.parametrize("reply, expected", [
    ('{"summary": "El 42% raramente o nunca."}', "El 42% raramente o nunca."),
    ('```json\n{"summary": "Fenced."}\n```', "Fenced."),
    ('<think>reasoning {x}</think>{"summary": "After thinking."}', "After thinking."),
    ('Here you go: {"summary": "Preamble then JSON."} Thanks!', "Preamble then JSON."),
])
def test_valid_json_forms(reply, expected):
    json_str, err = _parse_summary_reply(reply)
    assert err is None and _summary(json_str) == expected


def test_unescaped_quotes_inside_the_summary():
    # Invalid JSON: inner double quotes are not escaped.
    reply = '{"summary": "El 41% responde "raramente/nunca" y el 24% "la mayoría"."}'
    json_str, err = _parse_summary_reply(reply)
    assert err is None
    assert _summary(json_str) == 'El 41% responde "raramente/nunca" y el 24% "la mayoría".'


def test_raw_newline_inside_the_summary():
    reply = '{"summary": "Primera frase.\nSegunda frase."}'
    json_str, err = _parse_summary_reply(reply)
    assert err is None and "Primera frase." in _summary(json_str)


def test_plain_prose_reply_is_the_summary():
    reply = "El 56% de los adultos de 65 años o más son mujeres."
    json_str, err = _parse_summary_reply(reply)
    assert err is None and _summary(json_str) == reply


@pytest.mark.parametrize("reply", [None, "", "   "])
def test_empty_reply_is_an_error(reply):
    json_str, err = _parse_summary_reply(reply)
    assert _summary(json_str) == "" and "empty reply" in err


def test_unreadable_reply_is_an_error_with_a_snippet():
    reply = '{"answer": "wrong key", "notes": {"a": 1}}'
    json_str, err = _parse_summary_reply(reply)
    assert _summary(json_str) == ""
    assert "could not read a summary" in err and "wrong key" in err


def test_empty_summary_value_is_an_error():
    json_str, err = _parse_summary_reply('{"summary": ""}')
    assert err is not None
