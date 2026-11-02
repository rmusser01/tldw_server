"""Bounded, data-only moderation wire contract (no matching in the parent)."""

import json
import re
import struct
from dataclasses import replace

import pytest
from hypothesis import given
from hypothesis import strategies as st

from tldw_Server_API.app.core.exception_types import ModerationExecutionError
from tldw_Server_API.app.core.Moderation.execution_models import (
    ExecutionBounds,
    FailureMetadata,
    WorkerResponse,
    execution_scope,
)
from tldw_Server_API.app.core.Moderation.execution_protocol import (
    decode_request,
    decode_response,
    encode_request,
    encode_response,
    snapshot_request,
)
from tldw_Server_API.app.core.Moderation.models import ModerationPolicy, PatternRule
from tldw_Server_API.app.core.Moderation.policy_evaluator import EvaluationLimits


def request(text="hello", **kwargs):
    return snapshot_request(
        "redact_with_count",
        text,
        ModerationPolicy(),
        None,
        EvaluationLimits(100, 10, 1000, None),
        ExecutionBounds(**kwargs),
    )


def frame(data):
    payload = json.dumps(data, ensure_ascii=True).encode()
    return struct.pack(">I", len(payload)) + payload


def test_unknown_wire_version_is_rejected():
    with pytest.raises(ModerationExecutionError, match="invalid_response"):
        decode_request(b'{"version":999}')


def test_requester_defaults_are_fixed():
    bounds = ExecutionBounds()
    assert (bounds.timeout_s, bounds.max_workers, bounds.max_output_chars) == (2.0, 4, 1_000_000)
    assert (
        bounds.max_input_chars,
        bounds.max_rules,
        bounds.max_pattern_chars,
        bounds.max_replacement_chars,
        bounds.max_policy_bytes,
        bounds.max_request_bytes,
        bounds.max_response_bytes,
    ) == (1_000_000, 256, 4096, 4096, 256 * 1024, 32 * 1024 * 1024, 8 * 1024 * 1024)


@pytest.mark.parametrize("value", [True, 0, -1, float("nan"), float("inf"), 2.1])
def test_unsafe_bounds_are_rejected(value):
    with pytest.raises(ValueError):
        ExecutionBounds(timeout_s=value)


def test_raw_pattern_flags_round_trip(monkeypatch):
    raw = re.compile("hello", re.I | re.M)
    policy = ModerationPolicy(enabled=True, block_patterns=[raw, PatternRule(raw, action="warn")])
    monkeypatch.setattr(re, "compile", lambda *a, **k: pytest.fail("parent compiled a pattern"))
    original = snapshot_request(
        "evaluate", "HELLO", policy, "unknown", EvaluationLimits(100, 10, 1000, "2"), ExecutionBounds()
    )
    decoded = decode_request(encode_request(original))
    assert decoded == original
    assert [r["kind"] for r in decoded.policy["block_patterns"]] == ["raw", "rule"]
    assert decoded.policy["block_patterns"][0]["flags"] == int(raw.flags)


def test_oversized_policy_is_rejected_before_encoding(monkeypatch):
    policy = ModerationPolicy(block_patterns=[re.compile("x")] * 257)
    with pytest.raises(ModerationExecutionError, match="size_exceeded"):
        snapshot_request("evaluate", "x", policy, None, EvaluationLimits(1, 0, 1, None), ExecutionBounds())


@given(st.text(max_size=300))
def test_bounded_unicode_control_frames_round_trip(text):
    original = request(text)
    assert decode_request(encode_request(original)) == original


@given(st.binary(max_size=200))
def test_malformed_frames_fail_closed(data):
    with pytest.raises(ModerationExecutionError):
        decode_request(data)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d.update(extra="secret"),
        lambda d: d.update(version=True),
        lambda d: d["bounds"].update(max_rules=True),
        lambda d: d["limits"].update(max_scan_chars=float("nan")),
        lambda d: d.update(operation="unknown"),
    ],
)
def test_malformed_fields_are_rejected(mutation):
    data = json.loads(encode_request(request())[4:])
    mutation(data)
    with pytest.raises(ModerationExecutionError):
        decode_request(frame(data))


def test_exactly_one_frame_and_eof():
    encoded = encode_request(request())
    for bad in (encoded + b"\n", encoded + encoded, encoded[:-1], struct.pack(">I", 2**32 - 1)):
        with pytest.raises(ModerationExecutionError):
            decode_request(bad)


def test_response_span_and_count_are_validated():
    from tldw_Server_API.app.core.Moderation.models import ModerationEvaluationResult

    original = snapshot_request(
        "evaluate",
        "x",
        ModerationPolicy(block_patterns=[re.compile("x")]),
        None,
        EvaluationLimits(100, 0, 100, None),
        ExecutionBounds(),
    )
    result = ModerationEvaluationResult(action="warn", match_span=(0, 999), matched_pattern="x")
    with pytest.raises(ModerationExecutionError, match="invalid_response"):
        decode_response(encode_response(WorkerResponse("evaluate", result)), original)
    for count in (True, -1, 999999999):
        with pytest.raises(ModerationExecutionError, match="invalid_response"):
            decode_response(encode_response(WorkerResponse("redact_with_count", ("safe", count))), request())


def test_metadata_and_nested_scope_are_content_free_and_report_once():
    assert FailureMetadata("evaluate", "input", 2_000_000).input_chars == 2_000_000
    assert FailureMetadata("evaluate", "unknown", 0).phase is None
    with execution_scope() as outer:
        with execution_scope(timeout_s=1) as inner:
            assert inner is outer
            assert inner.claim_report() is True
            assert outer.claim_report() is False
    with pytest.raises(ValueError):
        FailureMetadata("evaluate", None, True)
    with pytest.raises(ValueError) as error:
        ModerationExecutionError("secret-content")
    assert "secret-content" not in repr(error.value)


@pytest.mark.parametrize(
    "field,value",
    [
        ("result", []),
        ("result", {"action": []}),
        ("operation", []),
        ("reason", {}),
    ],
)
def test_response_malformed_objects_raise_typed_error(field, value):
    original = replace(request(), operation="evaluate")
    from tldw_Server_API.app.core.Moderation.models import ModerationEvaluationResult

    data = json.loads(encode_response(WorkerResponse("evaluate", ModerationEvaluationResult()))[4:])
    data[field] = value
    with pytest.raises(ModerationExecutionError, match="invalid_response"):
        decode_response(frame(data), original)


@pytest.mark.parametrize("operation", ["evaluate", "redact", "redact_with_count"])
def test_raw_boolean_flags_are_rejected(operation):
    original = replace(request(), operation=operation)
    data = json.loads(encode_request(original)[4:])
    data["policy"]["block_patterns"] = [{"kind": "raw", "pattern": "x", "flags": True}]
    with pytest.raises(ModerationExecutionError):
        decode_request(frame(data))


def test_duplicate_json_keys_are_rejected():
    payload = b'{"version":1,"version":1}'
    with pytest.raises(ModerationExecutionError, match="invalid_response"):
        decode_request(struct.pack(">I", len(payload)) + payload)


@pytest.mark.parametrize("bad", [[], {}, True, 1])
def test_unhashable_and_mistyped_actions_fail_closed(bad):
    from tldw_Server_API.app.core.Moderation.models import ModerationEvaluationResult

    original = replace(request(), operation="evaluate")
    data = json.loads(encode_response(WorkerResponse("evaluate", ModerationEvaluationResult()))[4:])
    data["result"]["action"] = bad
    with pytest.raises(ModerationExecutionError, match="invalid_response"):
        decode_response(frame(data), original)


def test_policy_encoded_byte_cap_includes_categories():
    policy = ModerationPolicy(categories_enabled={"a" * 200, "b" * 200})
    with pytest.raises(ModerationExecutionError, match="size_exceeded"):
        snapshot_request(
            "evaluate", "x", policy, None, EvaluationLimits(1, 0, 1, None), ExecutionBounds(max_policy_bytes=300)
        )


def test_execution_scope_report_claim_is_thread_safe():
    from concurrent.futures import ThreadPoolExecutor

    with execution_scope() as scope, ThreadPoolExecutor(max_workers=4) as pool:
        assert sum(pool.map(lambda _: scope.claim_report(), range(30))) == 1


def test_request_frame_cap_counts_the_length_prefix():
    original = request()
    size = len(encode_request(original))
    # The bound changes its own encoded decimal width, so iterate to a fixed size.
    for _ in range(3):
        original = replace(original, bounds=replace(original.bounds, max_request_bytes=size))
        size = len(encode_request(original))
    original = replace(original, bounds=replace(original.bounds, max_request_bytes=size - 1))
    with pytest.raises(ModerationExecutionError, match="size_exceeded"):
        encode_request(original)


def test_response_frame_cap_counts_the_length_prefix():
    from tldw_Server_API.app.core.Moderation.execution_models import WorkerResponse

    original = request()
    encoded = encode_response(WorkerResponse("redact_with_count", ("hello", 0)))
    original = replace(original, bounds=replace(original.bounds, max_response_bytes=len(encoded) - 1))
    with pytest.raises(ModerationExecutionError, match="size_exceeded"):
        decode_response(encoded, original)


@pytest.mark.parametrize(
    "field",
    [
        "timeout_s",
        "max_workers",
        "max_input_chars",
        "max_output_chars",
        "max_rules",
        "max_pattern_chars",
        "max_replacement_chars",
        "max_policy_bytes",
        "max_request_bytes",
        "max_response_bytes",
    ],
)
def test_each_safety_bound_only_tightens_and_rejects_booleans(field):
    ceiling = getattr(ExecutionBounds(), field)
    for value in (True, False, 0, -1, ceiling + 1, 10**1000):
        with pytest.raises(ValueError):
            ExecutionBounds(**{field: value})


@pytest.mark.parametrize(
    "changes",
    [
        {"action": "pass", "sample": "unexpected"},
        {"action": "pass", "matched_pattern": "hello"},
        {"action": "warn", "match_span": [0, 1], "matched_pattern": None},
        {"action": "redact", "match_span": [0, 1], "matched_pattern": "hello", "redacted_text": None},
    ],
)
def test_evaluation_response_fields_are_consistent(changes):
    from tldw_Server_API.app.core.Moderation.models import ModerationEvaluationResult

    policy = ModerationPolicy(block_patterns=[re.compile("hello")])
    original = snapshot_request(
        "evaluate", "hello", policy, None, EvaluationLimits(100, 0, 100, None), ExecutionBounds()
    )
    data = json.loads(encode_response(WorkerResponse("evaluate", ModerationEvaluationResult()))[4:])
    data["result"].update(changes)
    with pytest.raises(ModerationExecutionError, match="invalid_response"):
        decode_response(frame(data), original)
