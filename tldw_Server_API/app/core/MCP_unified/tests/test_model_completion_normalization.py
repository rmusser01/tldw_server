"""Strict text-only response normalization without provider calls."""

from __future__ import annotations

import copy
import importlib
from dataclasses import FrozenInstanceError, asdict, fields
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tldw_Server_API.app.core.MCP_unified.interfaces.model_completion import (
    ModelCompletionFailure,
    ModelCompletionRequest,
    ModelFailureDomain,
)

pytestmark = pytest.mark.unit

_MAX_INT = 2**63 - 1
_FORBIDDEN_CONTROLS = tuple(value for value in range(0xA0) if value < 0x20 or value >= 0x7F)
_FORBIDDEN_CONTROLS = tuple(value for value in _FORBIDDEN_CONTROLS if value not in (9, 10, 13))
_PROPERTY_SETTINGS = settings(max_examples=200, derandomize=True, database=None)


def _request(*, chars: int = 1024, bytes_limit: int = 4096, tokens: int = 32) -> ModelCompletionRequest:
    """Construct the real, validated request contract."""
    return ModelCompletionRequest("system", "user", tokens, chars, bytes_limit, 16384)


def _envelope(content: object = "answer") -> dict[str, Any]:
    """Build one non-streaming, text-only provider choice."""
    return {"choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}]}


def _normalize(envelope: object, request: ModelCompletionRequest | None = None) -> Any:
    """Load the sidecar lazily so missing implementation fails a test, not collection."""
    module = importlib.import_module("tldw_Server_API.app.core.MCP_unified.adapters.model_completion.normalization")
    return module.normalize_model_completion_response(envelope, request or _request())


def _assert_invalid(envelope: object, request: ModelCompletionRequest | None = None) -> None:
    """Assert stable request-local failure without a retained exception graph."""
    with pytest.raises(ModelCompletionFailure) as captured:
        _normalize(envelope, request)
    error = captured.value
    assert (error.code, error.domain, error.args) == (
        "invalid_model_output",
        ModelFailureDomain.REQUEST,
        ("invalid_model_output",),
    )
    assert error.__cause__ is None
    assert error.__context__ is None


def test_normalizes_only_line_endings_in_one_text_choice() -> None:
    result = _normalize(_envelope(" \tCafe\u0301\r\n\u00e9\r\U0001f642 \n"))

    assert result.content == " \tCafe\u0301\n\u00e9\n\U0001f642 \n"


@pytest.mark.parametrize("envelope", [None, True, 1, 1.0, "private-body", b"{}", [], ()])
def test_rejects_non_object_envelope(envelope: object) -> None:
    _assert_invalid(envelope)


@pytest.mark.parametrize("choices", [None, True, 1, 1.0, "answer", {}, (), ({"message": {"content": "x"}},)])
def test_requires_choices_to_be_a_json_list(choices: object) -> None:
    _assert_invalid({"choices": choices})


@pytest.mark.parametrize("count", [0, 2, 3])
def test_requires_exactly_one_choice(count: int) -> None:
    _assert_invalid({"choices": [_envelope()["choices"][0] for _ in range(count)]})


@pytest.mark.parametrize("choice", [None, True, 1, 1.0, "answer", [], ()])
def test_requires_choice_to_be_a_json_object(choice: object) -> None:
    _assert_invalid({"choices": [choice]})


@pytest.mark.parametrize("message", [None, True, 1, 1.0, "answer", [], ()])
def test_requires_message_to_be_a_json_object(message: object) -> None:
    _assert_invalid({"choices": [{"message": message}]})


@pytest.mark.parametrize("content", [None, True, 1, 1.0, b"answer", [], {}, [{"type": "text", "text": "answer"}]])
def test_requires_message_content_to_be_text(content: object) -> None:
    _assert_invalid(_envelope(content))


@pytest.mark.parametrize(
    "envelope",
    [{}, {"choices": [{}]}, {"choices": [{"message": {}}]}],
)
def test_rejects_missing_required_fields(envelope: object) -> None:
    _assert_invalid(envelope)


@pytest.mark.parametrize("level", ["envelope", "choice", "message"])
@pytest.mark.parametrize("field", ["tool_calls", "function_call"])
@pytest.mark.parametrize("value", [None, [], {}, "", False, {"name": "private-tool"}, [{"id": "private-call"}]])
def test_rejects_tool_call_field_presence_even_when_empty(level: str, field: str, value: object) -> None:
    envelope = _envelope()
    target = envelope
    if level == "choice":
        target = envelope["choices"][0]
    elif level == "message":
        target = envelope["choices"][0]["message"]
    target[field] = value

    _assert_invalid(envelope)


@pytest.mark.parametrize("finish_reason", ["tool_calls", "function_call"])
def test_rejects_tool_or_function_finish_reason(finish_reason: str) -> None:
    envelope = _envelope()
    envelope["choices"][0]["finish_reason"] = finish_reason

    _assert_invalid(envelope)


@pytest.mark.parametrize("finish_reason", [True, 1, 1.5, [], {}])
def test_rejects_non_text_non_null_finish_reason(finish_reason: object) -> None:
    envelope = _envelope()
    envelope["choices"][0]["finish_reason"] = finish_reason

    _assert_invalid(envelope)


@pytest.mark.parametrize("finish_reason", [None, "stop", "length", "content_filter"])
def test_accepts_non_tool_finish_reasons(finish_reason: object) -> None:
    envelope = _envelope()
    envelope["choices"][0]["finish_reason"] = finish_reason

    assert _normalize(envelope).content == "answer"


def test_accepts_missing_finish_reason() -> None:
    envelope = _envelope()
    del envelope["choices"][0]["finish_reason"]

    assert _normalize(envelope).content == "answer"


@pytest.mark.parametrize("content", ["", " ", "\t\n", "\r\n\r", "\u00a0\u2003\u2028\u3000"])
def test_rejects_empty_or_unicode_whitespace_only_output(content: str) -> None:
    _assert_invalid(_envelope(content))


@pytest.mark.parametrize("surrogate", ["\ud800", "\udbff", "\udc00", "\udfff", "\ud83d\ude00"])
def test_rejects_surrogates_with_sanitized_failure(surrogate: str) -> None:
    _assert_invalid(_envelope("private-provider-body" + surrogate))


@pytest.mark.parametrize("codepoint", _FORBIDDEN_CONTROLS, ids=lambda value: f"U+{value:04X}")
def test_rejects_every_prohibited_c0_del_and_c1_control(codepoint: int) -> None:
    _assert_invalid(_envelope("a" + chr(codepoint) + "b"))


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("a\r\nb", "a\nb"),
        ("a\rb", "a\nb"),
        ("a\r\r\nb", "a\n\nb"),
        ("a\n\rb", "a\n\nb"),
        ("\ta\t\n", "\ta\t\n"),
        ("a\u2028b\u2029c", "a\u2028b\u2029c"),
    ],
)
def test_line_ending_conversion_does_not_change_other_whitespace(raw: str, expected: str) -> None:
    assert _normalize(_envelope(raw)).content == expected


@pytest.mark.parametrize(
    "content", [" leading and trailing \t\n", "e\u0301", "\u00e9", "\u00a0x\u3000", "\ufeff\u200bx"]
)
def test_never_trims_or_applies_unicode_normalization(content: str) -> None:
    assert _normalize(_envelope(content)).content == content


@pytest.mark.parametrize("content", ["abc", " \tx\n ", "\u00e9\u00e9", "\u20ac", "\U0001f642", "e\u0301"])
def test_accepts_exact_character_and_utf8_byte_limits(content: str) -> None:
    request = _request(chars=len(content), bytes_limit=len(content.encode("utf-8")))

    assert _normalize(_envelope(content), request).content == content


@pytest.mark.parametrize("content", ["ab", " x ", "\u00e9\u00e9", "\U0001f642x", "e\u0301"])
def test_rejects_entire_content_at_character_limit_plus_one(content: str) -> None:
    _assert_invalid(_envelope(content), _request(chars=len(content) - 1))


@pytest.mark.parametrize("content", ["ab", " x ", "\u00e9", "\u20ac", "\U0001f642", "e\u0301"])
def test_rejects_entire_content_at_utf8_byte_limit_plus_one(content: str) -> None:
    _assert_invalid(_envelope(content), _request(bytes_limit=len(content.encode("utf-8")) - 1))


def test_measures_limits_after_line_ending_normalization() -> None:
    assert _normalize(_envelope("a\r\nb\rc"), _request(chars=5, bytes_limit=5)).content == "a\nb\nc"


@pytest.mark.parametrize("suffix", ["x", "\u00e9", "\x00", "\ud800"])
def test_never_truncates_valid_prefix_to_hide_invalid_suffix(suffix: str) -> None:
    _assert_invalid(_envelope("ok" + suffix), _request(chars=2, bytes_limit=2))


def test_failure_detaches_preexisting_private_exception_context() -> None:
    try:
        raise RuntimeError("private-provider-exception")
    except RuntimeError:
        _assert_invalid(_envelope("private-body\ud800"))


def test_returns_frozen_slotted_minimized_internal_result_without_mutating_envelope() -> None:
    envelope = _envelope("answer\r\n")
    envelope.update(
        id="private-response-id",
        model="private-model",
        usage={"prompt_tokens": 7, "completion_tokens": 3, "private_usage": {"opaque": "private-value"}},
    )
    original = copy.deepcopy(envelope)
    result = _normalize(envelope)

    assert type(result).__name__ == "NormalizedModelCompletion"
    assert [field.name for field in fields(result)] == ["content", "input_tokens", "output_tokens"]
    assert asdict(result) == {"content": "answer\n", "input_tokens": 7, "output_tokens": 3}
    assert not hasattr(result, "__dict__")
    assert not hasattr(result, "usage")
    assert envelope == original
    for field_name in ("content", "input_tokens", "output_tokens"):
        with pytest.raises(FrozenInstanceError):
            setattr(result, field_name, "changed")


@pytest.mark.parametrize("usage", [None, True, 0, 1.0, "private-usage", [], (), {}])
def test_missing_or_non_object_usage_is_untrusted_not_a_content_failure(usage: object) -> None:
    envelope = _envelope()
    envelope["usage"] = usage
    result = _normalize(envelope)

    assert (result.content, result.input_tokens, result.output_tokens) == ("answer", None, None)


def test_absent_usage_counts_are_none() -> None:
    result = _normalize(_envelope())

    assert (result.input_tokens, result.output_tokens) == (None, None)


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        ({"prompt_tokens": 7}, (7, None)),
        ({"completion_tokens": 3}, (None, 3)),
        ({"prompt_tokens": 0, "completion_tokens": 0}, (0, 0)),
        ({"prompt_tokens": 7, "completion_tokens": 32}, (7, 32)),
        ({"prompt_tokens": 7, "completion_tokens": 33}, (7, None)),
        ({"input_tokens": 7, "output_tokens": 3, "total_tokens": 10}, (None, None)),
        ({"prompt_tokens": _MAX_INT, "completion_tokens": 3}, (_MAX_INT, 3)),
    ],
)
def test_minimizes_usage_counts_independently(usage: object, expected: tuple[int | None, int | None]) -> None:
    envelope = _envelope()
    envelope["usage"] = usage
    result = _normalize(envelope)

    assert (result.input_tokens, result.output_tokens) == expected


class _IntSubclass(int):
    """An integer-like object is not an exact provider JSON integer."""


@pytest.mark.parametrize("field", ["prompt_tokens", "completion_tokens"])
@pytest.mark.parametrize("invalid", [None, True, False, -1, 1.0, "7", [], {}, _MAX_INT + 1, _IntSubclass(7)])
def test_malformed_usage_count_is_none_without_discarding_valid_sibling(field: str, invalid: object) -> None:
    envelope = _envelope()
    envelope["usage"] = {"prompt_tokens": 7, "completion_tokens": 3, field: invalid}
    result = _normalize(envelope)

    expected = (None, 3) if field == "prompt_tokens" else (7, None)
    assert (result.content, result.input_tokens, result.output_tokens) == ("answer", *expected)


def test_accepts_max_integer_output_usage_only_with_sufficient_request_token_limit() -> None:
    envelope = _envelope()
    envelope["usage"] = {"prompt_tokens": _MAX_INT, "completion_tokens": _MAX_INT}
    result = _normalize(envelope, _request(tokens=_MAX_INT))

    assert (result.input_tokens, result.output_tokens) == (_MAX_INT, _MAX_INT)


_ALLOWED_CHARACTERS = st.characters(
    exclude_categories=("Cs",),
    exclude_characters="".join(chr(value) for value in _FORBIDDEN_CONTROLS),
)


@_PROPERTY_SETTINGS
@given(st.text(alphabet=_ALLOWED_CHARACTERS, max_size=128))
def test_property_allowed_unicode_is_preserved_except_line_endings(raw: str) -> None:
    content = "x" + raw
    expected = content.replace("\r\n", "\n").replace("\r", "\n")
    request = _request(chars=len(expected), bytes_limit=len(expected.encode("utf-8")))
    result = _normalize(_envelope(content), request)

    assert result.content == expected
    assert _normalize(_envelope(result.content), request) == result


@_PROPERTY_SETTINGS
@given(st.integers(min_value=0, max_value=0x10FFFF))
def test_property_unicode_codepoint_control_and_surrogate_policy(codepoint: int) -> None:
    content = "x" + chr(codepoint)
    if codepoint in _FORBIDDEN_CONTROLS or 0xD800 <= codepoint <= 0xDFFF:
        _assert_invalid(_envelope(content))
    else:
        expected = content.replace("\r", "\n")
        assert _normalize(_envelope(content)).content == expected


@_PROPERTY_SETTINGS
@given(st.sampled_from(_FORBIDDEN_CONTROLS), st.text(alphabet=_ALLOWED_CHARACTERS, max_size=64))
def test_property_prohibited_controls_are_never_removed_to_accept_output(codepoint: int, raw: str) -> None:
    _assert_invalid(_envelope("x" + raw + chr(codepoint)))


@_PROPERTY_SETTINGS
@given(st.integers(min_value=0xD800, max_value=0xDFFF), st.text(alphabet=_ALLOWED_CHARACTERS, max_size=64))
def test_property_surrogates_always_fail_strict_utf8(codepoint: int, raw: str) -> None:
    _assert_invalid(_envelope("x" + raw + chr(codepoint)))


@_PROPERTY_SETTINGS
@given(st.text(alphabet=_ALLOWED_CHARACTERS, min_size=1, max_size=64))
def test_property_normalized_entire_content_limits_are_atomic(raw: str) -> None:
    content = "x" + raw
    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    chars = len(normalized)
    byte_count = len(normalized.encode("utf-8"))

    assert _normalize(_envelope(content), _request(chars=chars, bytes_limit=byte_count)).content == normalized
    _assert_invalid(_envelope(content), _request(chars=chars - 1, bytes_limit=byte_count))
    _assert_invalid(_envelope(content), _request(chars=chars, bytes_limit=byte_count - 1))


@_PROPERTY_SETTINGS
@given(st.integers(min_value=-_MAX_INT, max_value=_MAX_INT + 1), st.integers(min_value=1, max_value=1000))
def test_property_usage_counts_require_bounded_integers_and_output_token_ceiling(count: int, limit: int) -> None:
    envelope = _envelope()
    envelope["usage"] = {"prompt_tokens": count, "completion_tokens": count}
    result = _normalize(envelope, _request(tokens=limit))

    assert result.input_tokens == (count if 0 <= count <= _MAX_INT else None)
    assert result.output_tokens == (count if 0 <= count <= min(limit, _MAX_INT) else None)
