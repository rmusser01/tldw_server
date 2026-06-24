from __future__ import annotations

import json

import pytest

import tldw_Server_API.app.core.Chat.validate_dictionary as validate_module
from tldw_Server_API.app.core.Chat.validate_dictionary import validate_dictionary

pytestmark = pytest.mark.unit


def _mk(payload_entries):


    return {"name": "test", "entries": payload_entries}


def test_schema_invalid_missing_entries_warns():


    res = validate_dictionary({"name": "x"}, schema_version=1)
    assert any(w.get("code") == "schema_invalid" and w.get("field") == "entries" for w in res.warnings)


def test_regex_invalid_is_caught():


    payload = _mk([{"type": "regex", "pattern": "(unclosed", "replacement": "x"}])
    res = validate_dictionary(payload)
    assert any(e.get("code") == "regex_invalid" for e in res.errors)


def test_regex_unsafe_nested_quantifiers():


    payload = _mk([{"type": "regex", "pattern": "(a+)+$", "replacement": "x"}])
    res = validate_dictionary(payload)
    assert any(e.get("code") == "regex_unsafe" for e in res.errors)


def test_template_forbidden_construct():


    payload = _mk([{"type": "literal", "pattern": "hello", "replacement": "{% for x in y %}hi{% endfor %}"}])
    res = validate_dictionary(payload)
    assert any(e.get("code") == "template_forbidden_construct" for e in res.errors)


def test_template_expensive_operator_forbidden():


    payload = _mk([{"type": "literal", "pattern": "hello", "replacement": "{{ 'x' * 100 }}"}])
    res = validate_dictionary(payload)
    assert any(e.get("code") == "template_forbidden_construct" for e in res.errors)


def test_template_arbitrary_method_call_forbidden():


    payload = _mk([{"type": "literal", "pattern": "hello", "replacement": "{{ obj.public_method() }}"}])
    res = validate_dictionary(payload)
    assert any(e.get("code") == "template_forbidden_construct" for e in res.errors)


def test_template_unknown_function_weather_marked_external():


    payload = _mk([{"type": "literal", "pattern": "w", "replacement": "{{ weather('Boston') }}"}])
    res = validate_dictionary(payload)
    assert any(e.get("code") == "template_external_calls_disabled" for e in res.errors)


def test_probability_out_of_range():


    payload = _mk([{"type": "literal", "pattern": "x", "replacement": "y", "probability": 2.0}])
    res = validate_dictionary(payload)
    assert any(e.get("code") == "probability_out_of_range" for e in res.errors)


def test_duplicate_pattern_detected():


    payload = _mk([
        {"type": "literal", "pattern": "x", "replacement": "a"},
        {"type": "literal", "pattern": "x", "replacement": "b"},
    ])
    res = validate_dictionary(payload)
    assert any(e.get("code") == "duplicate_pattern" for e in res.errors)


def test_unknown_schema_version_is_schema_invalid_error():


    payload = _mk([{"type": "literal", "pattern": "x", "replacement": "y"}])
    res = validate_dictionary(payload, schema_version=999)
    assert res.ok is False
    assert any(e.get("code") == "schema_invalid" and e.get("field") == "schema_version" for e in res.errors)


def test_partial_reason_max_entries(monkeypatch):


    monkeypatch.setenv("CHAT_DICT_VALIDATE_MAX_ENTRIES", "1")
    payload = _mk([
        {"type": "literal", "pattern": "x1", "replacement": "a"},
        {"type": "literal", "pattern": "x2", "replacement": "b"},
    ])
    res = validate_dictionary(payload)
    assert res.partial is True
    assert res.partial_reason == "max_entries"


def test_partial_reason_timeout(monkeypatch):


    monkeypatch.setenv("CHAT_DICT_VALIDATE_TIMEOUT_MS", "1")
    payload = _mk([
        {"type": "literal", "pattern": "x1", "replacement": "a"},
        {"type": "literal", "pattern": "x2", "replacement": "b"},
    ])

    values = iter([0.0, 0.0, 2.0, 2.0, 2.0, 2.0])

    def fake_perf_counter():
        try:
            return next(values)
        except StopIteration:
            return 2.0

    monkeypatch.setattr(validate_module.time, "perf_counter", fake_perf_counter)
    res = validate_dictionary(payload)
    assert res.partial is True
    assert res.partial_reason == "timeout"


def test_strict_promotes_template_unknown_function_warning_to_error():


    payload = _mk([{"type": "literal", "pattern": "x", "replacement": "{{ custom_fn('v') }}"}])

    non_strict = validate_dictionary(payload, strict=False)
    assert any(w.get("code") == "template_unknown_function" for w in non_strict.warnings)
    assert not any(e.get("code") == "template_unknown_function" for e in non_strict.errors)

    strict = validate_dictionary(payload, strict=True)
    assert any(e.get("code") == "template_unknown_function" for e in strict.errors)
    assert not any(w.get("code") == "template_unknown_function" for w in strict.warnings)


def test_cli_strict_returns_1_for_fatal_errors(tmp_path):


    payload = _mk([{"type": "regex", "pattern": "(unclosed", "replacement": "x"}])
    p = tmp_path / "fatal.json"
    p.write_text(json.dumps(payload), encoding="utf-8")

    rc = validate_module.main(["--file", str(p), "--strict"])
    assert rc == 1


def test_cli_strict_returns_0_for_non_fatal_errors(tmp_path):


    payload = _mk([{"type": "literal", "pattern": "x", "replacement": "{{ custom_fn('v') }}"}])
    p = tmp_path / "non_fatal.json"
    p.write_text(json.dumps(payload), encoding="utf-8")

    rc = validate_module.main(["--file", str(p), "--strict"])
    assert rc == 0


def test_cli_returns_2_on_file_read_error():


    rc = validate_module.main(["--file", "/definitely/not/here.json"])
    assert rc == 2
