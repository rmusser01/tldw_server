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
    RegexRuleSnapshot,
    SubstitutionBatchResult,
    SubstitutionRuleSnapshot,
    SupervisedBatchResult,
    WorkerResponse,
    execution_scope,
)
from tldw_Server_API.app.core.Moderation.execution_protocol import (
    decode_request,
    decode_response,
    encode_request,
    encode_response,
    snapshot_request,
    snapshot_substitution_request,
    snapshot_supervised_request,
)
from tldw_Server_API.app.core.Moderation.models import ModerationEvaluationResult, ModerationPolicy, PatternRule
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
    original = snapshot_request(
        "evaluate",
        "",
        ModerationPolicy(),
        None,
        EvaluationLimits(100, 10, 1000, None),
        ExecutionBounds(),
    )
    data = json.loads(encode_request(original)[4:])
    assert decode_request(frame(data)) == original
    data["version"] = 999
    with pytest.raises(ModerationExecutionError, match="invalid_response"):
        decode_request(frame(data))


_OPERATIONS = ("evaluate", "redact", "redact_with_count", "supervised_evaluate", "custom_substitute")
_NON_INTEGER = st.one_of(
    st.none(),
    st.booleans(),
    st.floats(),
    st.text(max_size=8),
    st.lists(st.integers(min_value=0, max_value=3), max_size=2),
    st.dictionaries(st.text(max_size=3), st.integers(min_value=0, max_value=3), max_size=2),
)


def schema_request(operation, text="aa", phase=None):
    """Start every generated mutation at a valid bounded operation-specific schema."""
    bounds = ExecutionBounds(max_input_chars=64, max_output_chars=64, max_rules=4)
    if operation == "supervised_evaluate":
        return snapshot_supervised_request(text, [RegexRuleSnapshot("a", 0, "warn", "both")] * 2, phase, bounds)
    if operation == "custom_substitute":
        return snapshot_substitution_request(text, [SubstitutionRuleSnapshot("a", 0, "a")] * 2, phase, bounds)
    policy = ModerationPolicy(enabled=True, block_patterns=[re.compile("a"), PatternRule(re.compile("b"))])
    return snapshot_request(operation, text, policy, phase, EvaluationLimits(100, 10, 1000, None), bounds)


def schema_response(original):
    """A valid result shape for each variant, independent of regex execution."""
    results = {
        "evaluate": ModerationEvaluationResult(action="warn", matched_pattern="a", match_span=(0, 1)),
        "redact": "",
        "redact_with_count": ("", 0),
        "supervised_evaluate": SupervisedBatchResult(0, (0, 1), None),
        "custom_substitute": SubstitutionBatchResult("", 0, ()),
    }
    response = WorkerResponse(original.operation, results[original.operation])
    assert decode_response(encode_response(response), original) == response
    return json.loads(encode_response(response)[4:])


def invalid_integer(low, high):
    return st.one_of(
        _NON_INTEGER,
        st.integers(min_value=low - 20, max_value=low - 1),
        st.integers(min_value=high + 1, max_value=high + 20),
    )


@pytest.mark.parametrize("operation", _OPERATIONS)
@given(text=st.text(max_size=40), phase=st.one_of(st.none(), st.text(max_size=10)))
def test_schema_aware_bounded_unicode_requests_round_trip(operation, text, phase):
    original = schema_request(operation, text, phase)
    encoded = encode_request(original)
    assert len(encoded) <= original.bounds.max_request_bytes
    assert decode_request(encoded) == original


@pytest.mark.parametrize("operation", _OPERATIONS)
@pytest.mark.parametrize("field", ["version", "tag", "operation", "text", "phase", "rule_flags", "bounds_integer"])
@given(data=st.data())
def test_schema_aware_request_field_mutations_are_rejected(operation, field, data):
    original = schema_request(operation)
    payload = json.loads(encode_request(original)[4:])
    if field == "version":
        payload[field] = data.draw(invalid_integer(1, 1))
    elif field in {"tag", "operation"}:
        excluded = {"request"} if field == "tag" else set(_OPERATIONS)
        payload[field] = data.draw(
            st.one_of(
                _NON_INTEGER.filter(lambda value: type(value) is not str),
                st.text(max_size=12).filter(lambda value: value not in excluded),
            )
        )
    elif field == "text":
        payload[field] = data.draw(
            st.one_of(
                _NON_INTEGER.filter(lambda value: type(value) is not str),
                st.text(min_size=65, max_size=70),
            )
        )
    elif field == "phase":
        payload[field] = data.draw(_NON_INTEGER.filter(lambda value: value is not None and type(value) is not str))
    elif field == "rule_flags":
        rules = payload["policy"].get("rules", payload["policy"].get("block_patterns"))
        rules[0]["flags"] = data.draw(invalid_integer(0, 511))
    else:
        bound = data.draw(st.sampled_from(["max_workers", "max_rules", "max_input_chars", "max_output_chars"]))
        payload["bounds"][bound] = data.draw(invalid_integer(1, getattr(ExecutionBounds(), bound)))
    with pytest.raises(ModerationExecutionError):
        decode_request(frame(payload))


@pytest.mark.parametrize("operation", _OPERATIONS)
@pytest.mark.parametrize("target", ["envelope", "policy", "rule"])
@pytest.mark.parametrize("change", ["missing", "extra"])
@given(data=st.data())
def test_schema_aware_request_key_mutations_are_rejected(operation, target, change, data):
    payload = json.loads(encode_request(schema_request(operation))[4:])
    obj = payload
    if target == "policy":
        obj = payload["policy"]
    elif target == "rule":
        obj = payload["policy"].get("rules", payload["policy"].get("block_patterns"))[0]
    if change == "missing":
        del obj[data.draw(st.sampled_from(sorted(obj)))]
    else:
        obj["unexpected_field"] = data.draw(st.text(max_size=12))
    with pytest.raises(ModerationExecutionError):
        decode_request(frame(payload))


@pytest.mark.parametrize("operation", _OPERATIONS)
@pytest.mark.parametrize("field", ["version", "tag", "operation", "reason", "resource_limits"])
@given(data=st.data())
def test_schema_aware_response_envelope_mutations_are_rejected(operation, field, data):
    original = schema_request(operation)
    payload = schema_response(original)
    if field == "version":
        payload[field] = data.draw(invalid_integer(1, 1))
    elif field == "tag":
        payload[field] = data.draw(st.text(max_size=12).filter(lambda value: value != "response"))
    elif field == "operation":
        payload[field] = data.draw(st.sampled_from([value for value in _OPERATIONS if value != operation]))
    elif field == "reason":
        payload[field] = data.draw(st.text(max_size=12))
    else:
        payload[field] = {"address_space": False, "cpu": True, "core": data.draw(_NON_INTEGER)}
        # A posture flag must be a boolean, not an integer or other JSON value.
        if type(payload[field]["core"]) is bool:
            payload[field]["core"] = int(payload[field]["core"])
    with pytest.raises(ModerationExecutionError, match="invalid_response"):
        decode_response(frame(payload), original)


@pytest.mark.parametrize(
    "operation,target",
    [
        *[(operation, "envelope") for operation in _OPERATIONS],
        *[(operation, "result") for operation in ("evaluate", "supervised_evaluate", "custom_substitute")],
    ],
)
@pytest.mark.parametrize("change", ["missing", "extra"])
@given(data=st.data())
def test_schema_aware_response_key_mutations_are_rejected(operation, target, change, data):
    original = schema_request(operation)
    payload = schema_response(original)
    obj = payload if target == "envelope" else payload["result"]
    if change == "missing":
        del obj[data.draw(st.sampled_from(sorted(obj)))]
    else:
        obj["unexpected_field"] = data.draw(st.text(max_size=12))
    with pytest.raises(ModerationExecutionError, match="invalid_response"):
        decode_response(frame(payload), original)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
@given(data=st.data())
def test_schema_aware_nonfinite_request_and_response_numbers_are_rejected(value, data):
    original = schema_request(data.draw(st.sampled_from(_OPERATIONS[:3])))
    payload = json.loads(encode_request(original)[4:])
    payload["limits"][data.draw(st.sampled_from(sorted(payload["limits"])))] = value
    with pytest.raises(ModerationExecutionError, match="invalid_response"):
        decode_request(frame(payload))
    original = schema_request(data.draw(st.sampled_from(("redact_with_count", "custom_substitute"))))
    payload = schema_response(original)
    if original.operation == "redact_with_count":
        payload["result"][1] = value
    else:
        payload["result"]["count"] = value
    with pytest.raises(ModerationExecutionError, match="invalid_response"):
        decode_response(frame(payload), original)


@pytest.mark.parametrize("operation", ["redact_with_count", "custom_substitute"])
@given(
    length=st.integers(min_value=0, max_value=30),
    ceiling=st.integers(min_value=1, max_value=32),
    rule_count=st.integers(min_value=0, max_value=4),
    data=st.data(),
)
def test_schema_aware_count_types_and_conservative_bound(operation, length, ceiling, rule_count, data):
    original = schema_request(operation, "a" * length)
    policy = dict(original.policy)
    key = "rules" if operation == "custom_substitute" else "block_patterns"
    policy[key] = [policy[key][0]] * rule_count
    original = replace(original, policy=policy, bounds=replace(original.bounds, max_output_chars=ceiling))
    original = decode_request(encode_request(original))
    payload = schema_response(original)
    maximum = rule_count * (2 * max(length, ceiling) + 1)
    # Both endpoints and generated interior counts are admitted by the wire bound.
    for count in (0, maximum, data.draw(st.integers(min_value=0, max_value=maximum))):
        if operation == "custom_substitute":
            payload["result"].update(text="", count=count)
        else:
            payload["result"] = ["", count]
        assert decode_response(frame(payload), original).result is not None
    bad_count = data.draw(invalid_integer(0, maximum))
    if operation == "custom_substitute":
        payload["result"]["count"] = bad_count
    else:
        payload["result"][1] = bad_count
    with pytest.raises(ModerationExecutionError, match="invalid_response"):
        decode_response(frame(payload), original)


@pytest.mark.parametrize("operation", ["evaluate", "supervised_evaluate"])
@given(length=st.integers(min_value=1, max_value=32), data=st.data())
def test_schema_aware_span_types_and_boundaries(operation, length, data):
    original = schema_request(operation, "a" * length)
    payload = schema_response(original)
    start = data.draw(st.integers(min_value=0, max_value=length))
    end = data.draw(st.integers(min_value=start, max_value=length))
    for span in ([0, 0], [0, length], [length, length], [start, end]):
        payload["result"]["match_span"] = span
        assert decode_response(frame(payload), original).result.match_span == tuple(span)
    bad_span = data.draw(
        st.one_of(
            st.tuples(invalid_integer(0, length), st.just(length)).map(list),
            st.tuples(st.just(start), invalid_integer(start, length)).map(list),
            st.lists(st.integers(min_value=0, max_value=length), min_size=3, max_size=4),
            st.sampled_from([[], [0], "span", 0, True, {}]),
        )
    )
    payload["result"]["match_span"] = bad_span
    with pytest.raises(ModerationExecutionError, match="invalid_response"):
        decode_response(frame(payload), original)


@pytest.mark.parametrize("operation", ["supervised_evaluate", "custom_substitute"])
@given(rule_count=st.integers(min_value=1, max_value=4), data=st.data())
def test_schema_aware_rule_index_types_and_boundaries(operation, rule_count, data):
    original = schema_request(operation)
    original = replace(original, policy={"rules": [original.policy["rules"][0]] * rule_count})
    payload = schema_response(original)
    for index in (0, rule_count - 1, data.draw(st.integers(min_value=0, max_value=rule_count - 1))):
        if operation == "supervised_evaluate":
            payload["result"]["winner_index"] = index
        else:
            payload["result"]["skipped_rule_indices"] = [index]
        assert decode_response(frame(payload), original).result is not None
    bad_index = data.draw(invalid_integer(0, rule_count - 1))
    if operation == "supervised_evaluate":
        payload["result"]["winner_index"] = bad_index
    else:
        payload["result"]["skipped_rule_indices"] = [bad_index]
    with pytest.raises(ModerationExecutionError, match="invalid_response"):
        decode_response(frame(payload), original)
    if operation == "custom_substitute":
        index = data.draw(st.integers(min_value=0, max_value=rule_count - 1))
        for skipped in (
            [index, index],
            [rule_count - 1, 0],
            data.draw(_NON_INTEGER.filter(lambda v: type(v) is not list)),
        ):
            payload["result"]["skipped_rule_indices"] = skipped
            with pytest.raises(ModerationExecutionError, match="invalid_response"):
                decode_response(frame(payload), original)


@pytest.mark.parametrize("operation", ["supervised_evaluate", "custom_substitute"])
@given(index=st.one_of(st.integers(min_value=0, max_value=8), st.booleans()))
def test_schema_aware_empty_rule_snapshots_reject_returned_indices(operation, index):
    original = replace(schema_request(operation), policy={"rules": []})
    result = (
        SupervisedBatchResult(None, None, None)
        if operation == "supervised_evaluate"
        else SubstitutionBatchResult("", 0, ())
    )
    response = WorkerResponse(operation, result)
    assert decode_response(encode_response(response), original) == response
    payload = json.loads(encode_response(response)[4:])
    if operation == "supervised_evaluate":
        payload["result"].update(winner_index=index, match_span=[0, 0])
    else:
        payload["result"]["skipped_rule_indices"] = [index]
    with pytest.raises(ModerationExecutionError, match="invalid_response"):
        decode_response(frame(payload), original)


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
        {"action": "redact", "match_span": [0, 1], "matched_pattern": "hello", "redacted_text": "eager text"},
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


def test_redact_decision_without_redacted_text_is_valid():
    from tldw_Server_API.app.core.Moderation.models import ModerationEvaluationResult

    policy = ModerationPolicy(block_patterns=[re.compile("hello")])
    request = snapshot_request(
        "evaluate", "hello", policy, None, EvaluationLimits(100, 0, 100, None), ExecutionBounds()
    )
    result = ModerationEvaluationResult(action="redact", matched_pattern="hello", match_span=(0, 5))
    response = decode_response(encode_response(WorkerResponse("evaluate", result)), request)
    assert response.result == result
    assert response.result.redacted_text is None


@given(action=st.sampled_from(["pass", "warn", "block", "redact"]), eager_text=st.text(max_size=80))
def test_every_decision_rejects_non_null_redaction_text(action, eager_text):
    from tldw_Server_API.app.core.Moderation.models import ModerationEvaluationResult

    policy = ModerationPolicy(block_patterns=[re.compile("hello")])
    request = snapshot_request(
        "evaluate", "hello", policy, None, EvaluationLimits(100, 0, 100, None), ExecutionBounds()
    )
    result = ModerationEvaluationResult(
        action=action,
        redacted_text=eager_text,
        matched_pattern="hello" if action != "pass" else None,
        match_span=(0, 5) if action != "pass" else None,
    )
    with pytest.raises(ModerationExecutionError, match="invalid_response"):
        decode_response(encode_response(WorkerResponse("evaluate", result)), request)


def batch_request(texts, **bounds):
    from tldw_Server_API.app.core.Moderation.execution_protocol import snapshot_redaction_batch_request

    return snapshot_redaction_batch_request(
        texts, ModerationPolicy(), None, EvaluationLimits(100, 10, 1000, None), ExecutionBounds(**bounds)
    )


@given(st.lists(st.text(max_size=20), max_size=10))
def test_redact_many_unicode_empty_round_trip_and_aggregate_metadata(texts):
    from tldw_Server_API.app.core.Moderation.execution_models import request_input_chars

    original = batch_request(texts)
    assert original.text == tuple(texts)
    assert decode_request(encode_request(original)) == original
    assert request_input_chars(original) == sum(map(len, texts))
    response = WorkerResponse("redact_many", texts)
    assert decode_response(encode_response(response), original) == response


def test_redact_many_empty_field_count_and_aggregate_input_caps():
    assert len(batch_request([""] * 4096).text) == 4096
    assert batch_request(["aa", "b"], max_input_chars=3).text == ("aa", "b")
    for texts, bounds in [([""] * 4097, {}), (["aa", "bb"], {"max_input_chars": 3})]:
        with pytest.raises(ModerationExecutionError, match="size_exceeded"):
            batch_request(texts, **bounds)


@pytest.mark.parametrize("operation", _OPERATIONS + ("redact_many",))
@pytest.mark.parametrize("bad", [None, 1, True, {}, ["a", 1]])
def test_batch_and_single_text_variants_reject_incompatible_wire_text(operation, bad):
    original = batch_request(["a"]) if operation == "redact_many" else schema_request(operation)
    data = json.loads(encode_request(original)[4:])
    data["text"] = bad
    with pytest.raises(ModerationExecutionError, match="invalid_response"):
        decode_request(frame(data))


def test_redact_many_discriminant_and_raw_flags_stay_strict(monkeypatch):
    from tldw_Server_API.app.core.Moderation.execution_protocol import snapshot_redaction_batch_request

    raw = re.compile("^hello$", re.I | re.M)
    policy = ModerationPolicy(block_patterns=[raw, PatternRule(raw, replacement=r"\1")])
    with monkeypatch.context() as guarded:
        guarded.setattr(re, "compile", lambda *a, **k: pytest.fail("parent compilation"))
        original = snapshot_redaction_batch_request(["hello"], policy, None, EvaluationLimits(1, 0, 1, None), ExecutionBounds())
    assert decode_request(encode_request(original)) == original
    assert original.policy["block_patterns"][0]["flags"] == int(raw.flags)
    for changed in (replace(original, text="hello"), replace(original, text=["hello"]), replace(original, operation="redact")):
        with pytest.raises(ModerationExecutionError, match="invalid_response"):
            encode_request(changed)
    data = json.loads(encode_request(original)[4:])
    data["text"] = "hello"
    with pytest.raises(ModerationExecutionError, match="invalid_response"):
        decode_request(frame(data))


@pytest.mark.parametrize("result", [None, "safe", [], ["a"], ["a", "b", "c"], ["a", 1], ["aa", "bb"]])
def test_redact_many_rejects_wrong_count_type_or_aggregate_output(result):
    original = batch_request(["a", "b"], max_output_chars=3)
    with pytest.raises(ModerationExecutionError):
        decode_response(encode_response(WorkerResponse("redact_many", result)), original)


def test_redact_many_frame_limits_remain_independent_of_character_admission():
    original = batch_request(["\u00e9" * 100], max_request_bytes=200)
    with pytest.raises(ModerationExecutionError, match="size_exceeded"):
        encode_request(original)
    original = batch_request(["a"], max_response_bytes=100)
    with pytest.raises(ModerationExecutionError, match="size_exceeded"):
        decode_response(encode_response(WorkerResponse("redact_many", ["\u00e9" * 100])), original)
