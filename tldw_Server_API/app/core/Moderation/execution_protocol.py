"""Strict, bounded JSON snapshots and single-frame moderation IPC.

Framing includes one four-byte big-endian payload length and requires EOF.
Snapshotting only reads compiled pattern metadata; compilation is child-only.
"""

from __future__ import annotations

import json
import math
import struct
from collections.abc import Sequence
from dataclasses import asdict, fields

from tldw_Server_API.app.core.exception_types import MODERATION_EXECUTION_REASONS, ModerationExecutionError
from tldw_Server_API.app.core.Moderation.execution_models import (
    OPERATIONS,
    BaseOperation,
    ExecutionBounds,
    RegexRuleSnapshot,
    SubstitutionBatchResult,
    SubstitutionRuleSnapshot,
    SupervisedBatchResult,
    WorkerRequest,
    WorkerResponse,
)
from tldw_Server_API.app.core.Moderation.models import ModerationEvaluationResult, ModerationPolicy, PatternRule
from tldw_Server_API.app.core.Moderation.policy_evaluator import EvaluationLimits

VERSION = 1
_CEILINGS = ExecutionBounds()
_POLICY_FIELDS = {item.name for item in fields(ModerationPolicy)}
_LIMIT_FIELDS = {item.name for item in fields(EvaluationLimits)}
_BOUND_FIELDS = {item.name for item in fields(ExecutionBounds)}


def _invalid() -> None:
    raise ModerationExecutionError("invalid_response")


def _size(condition: bool) -> None:
    if condition:
        raise ModerationExecutionError("size_exceeded")


def _object(value: object, names: set[str]) -> dict:
    if type(value) is not dict or set(value) != names:
        _invalid()
    return value


def _string(value: object, maximum: int | None = None, *, nullable: bool = False) -> None:
    if value is None and nullable:
        return
    if type(value) is not str:
        _invalid()
    if maximum is not None:
        _size(len(value) > maximum)


def _integer(value: object, low: int, high: int) -> None:
    if type(value) is not int or not low <= value <= high:
        _invalid()


def _json_bytes(value: object, cap: int) -> bytes:
    parts = []
    size = 0
    try:
        for part in json.JSONEncoder(ensure_ascii=True, allow_nan=False, separators=(",", ":")).iterencode(value):
            encoded = part.encode("utf-8")
            size += len(encoded)
            _size(size > cap)
            parts.append(encoded)
    except (TypeError, ValueError, OverflowError):
        raise ModerationExecutionError("invalid_response") from None
    return b"".join(parts)


def _frame(value: object, cap: int) -> bytes:
    payload = _json_bytes(value, cap - 4)
    return struct.pack(">I", len(payload)) + payload


def _pairs(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            _invalid()
        result[key] = value
    return result


def _unframe(frame: bytes, cap: int) -> dict:
    if type(frame) is not bytes or len(frame) < 4:
        _invalid()
    declared = struct.unpack(">I", frame[:4])[0]
    if declared != len(frame) - 4:
        _invalid()
    _size(len(frame) > cap)
    try:
        return json.loads(frame[4:].decode("utf-8"), object_pairs_hook=_pairs, parse_constant=lambda _value: _invalid())
    except (ValueError, UnicodeError, RecursionError):
        raise ModerationExecutionError("invalid_response") from None


def _validate_pattern(rule: dict, bounds: ExecutionBounds) -> None:
    _string(rule["pattern"], bounds.max_pattern_chars)
    # All currently supported stdlib flags, including DEBUG (captured in child).
    _integer(rule["flags"], 0, 511)


def _validate_request(request: WorkerRequest) -> None:
    if type(request.operation) is not str or request.operation not in OPERATIONS:
        _invalid()
    bounds = request.bounds
    _string(request.text, bounds.max_input_chars)
    _string(request.phase, bounds.max_pattern_chars, nullable=True)
    policy = request.policy
    if request.operation in {"supervised_evaluate", "custom_substitute"}:
        _object(policy, {"rules"})
        rules = policy["rules"]
    else:
        _object(policy, _POLICY_FIELDS)
        for name in ("enabled", "input_enabled", "output_enabled", "per_user_overrides"):
            if type(policy[name]) is not bool:
                _invalid()
        for name in ("input_action", "output_action"):
            _string(policy[name], bounds.max_pattern_chars, nullable=True)
        _string(policy["redact_replacement"], bounds.max_replacement_chars)
        _categories(policy["categories_enabled"], bounds)
        rules = policy["block_patterns"]
    if type(rules) is not list:
        _invalid()
    _size(len(rules) > bounds.max_rules)
    for rule in rules:
        if request.operation == "supervised_evaluate":
            _object(rule, {"pattern", "flags", "action", "phase"})
            _string(rule["action"], bounds.max_pattern_chars, nullable=True)
            _string(rule["phase"], bounds.max_pattern_chars)
        elif request.operation == "custom_substitute":
            _object(rule, {"pattern", "flags", "replacement"})
            _string(rule["replacement"], bounds.max_replacement_chars)
        else:
            if type(rule) is not dict or type(rule.get("kind")) is not str or rule["kind"] not in {"raw", "rule"}:
                _invalid()
            names = {"kind", "pattern", "flags"}
            if rule["kind"] == "rule":
                names |= {"action", "replacement", "categories", "phase"}
            _object(rule, names)
            if rule["kind"] == "rule":
                _string(rule["action"], bounds.max_pattern_chars, nullable=True)
                _string(rule["replacement"], bounds.max_replacement_chars, nullable=True)
                _string(rule["phase"], bounds.max_pattern_chars, nullable=True)
                _categories(rule["categories"], bounds)
        _validate_pattern(rule, bounds)
    _json_bytes(policy, bounds.max_policy_bytes)
    expected_limits = set() if request.operation in {"supervised_evaluate", "custom_substitute"} else _LIMIT_FIELDS
    _object(request.limits, expected_limits)
    # Preserve legacy limit coercions (notably string counts), not arbitrary objects.
    for value in request.limits.values():
        if value is not None and type(value) not in (str, int, float, bool):
            _invalid()
        if type(value) is float and not math.isfinite(value):
            _invalid()
        if type(value) is str:
            _string(value, bounds.max_pattern_chars)


def _categories(value: object, bounds: ExecutionBounds) -> None:
    if value is None:
        return
    if type(value) is not list:
        _invalid()
    _size(len(value) > bounds.max_policy_bytes)
    for category in value:
        _string(category, bounds.max_policy_bytes)


def snapshot_request(
    operation: BaseOperation,
    text: str,
    policy: ModerationPolicy,
    phase: str | None,
    limits: EvaluationLimits,
    bounds: ExecutionBounds,
) -> WorkerRequest:
    """Copy raw and structured compiled rules without compiling in the caller."""
    if operation not in {"evaluate", "redact", "redact_with_count"}:
        _invalid()
    _size(len(text) > bounds.max_input_chars or len(policy.block_patterns or []) > bounds.max_rules)
    snapshot = {name: getattr(policy, name) for name in _POLICY_FIELDS - {"block_patterns", "categories_enabled"}}
    snapshot["categories_enabled"] = (
        sorted(policy.categories_enabled) if policy.categories_enabled is not None else None
    )
    rules = []
    for rule in policy.block_patterns or []:
        structured = isinstance(rule, PatternRule)
        regex = rule.regex if structured else rule
        item = {"kind": "rule" if structured else "raw", "pattern": regex.pattern, "flags": int(regex.flags)}
        if structured:
            item.update(
                action=rule.action,
                replacement=rule.replacement,
                phase=rule.phase,
                categories=sorted(rule.categories) if rule.categories is not None else None,
            )
        rules.append(item)
    snapshot["block_patterns"] = rules
    request = WorkerRequest(operation, text, phase, snapshot, asdict(limits), bounds)
    _validate_request(request)
    return request


def snapshot_supervised_request(
    text: str, rules: Sequence[RegexRuleSnapshot], phase: str | None, bounds: ExecutionBounds
) -> WorkerRequest:
    """Snapshot only ordered matching metadata, never Guardian DB objects."""
    _size(len(rules) > bounds.max_rules or len(text) > bounds.max_input_chars)
    request = WorkerRequest("supervised_evaluate", text, phase, {"rules": [asdict(rule) for rule in rules]}, {}, bounds)
    _validate_request(request)
    return request


def snapshot_substitution_request(
    text: str, rules: Sequence[SubstitutionRuleSnapshot], phase: str | None, bounds: ExecutionBounds
) -> WorkerRequest:
    """Snapshot an ordered Workflow substitution batch with template replacements."""
    _size(len(rules) > bounds.max_rules or len(text) > bounds.max_input_chars)
    request = WorkerRequest("custom_substitute", text, phase, {"rules": [asdict(rule) for rule in rules]}, {}, bounds)
    _validate_request(request)
    return request


def encode_request(request: WorkerRequest) -> bytes:
    """Validate before encoding a single request frame."""
    _validate_request(request)
    data = asdict(request)
    data.update(version=VERSION, tag="request")
    return _frame(data, request.bounds.max_request_bytes)


def decode_request(frame: bytes) -> WorkerRequest:
    """Reject unknown versions, duplicate/extra fields and unsafe configuration."""
    data = _unframe(frame, _CEILINGS.max_request_bytes)
    _object(data, {"version", "tag", "operation", "text", "phase", "policy", "limits", "bounds"})
    if type(data["version"]) is not int or data["version"] != VERSION or data["tag"] != "request":
        _invalid()
    _object(data["bounds"], _BOUND_FIELDS)
    try:
        bounds = ExecutionBounds(**data["bounds"])
    except (TypeError, ValueError, OverflowError):
        raise ModerationExecutionError("invalid_response") from None
    _size(len(frame) > bounds.max_request_bytes)
    request = WorkerRequest(data["operation"], data["text"], data["phase"], data["policy"], data["limits"], bounds)
    _validate_request(request)
    return request


def encode_response(response: WorkerResponse) -> bytes:
    """Encode one bounded response; content is never used as a diagnostic."""
    data = asdict(response)
    data.update(version=VERSION, tag="response")
    return _frame(data, _CEILINGS.max_response_bytes)


def _span(value: object, request: WorkerRequest) -> tuple[int, int] | None:
    if value is None:
        return None
    if type(value) is not list or len(value) != 2:
        _invalid()
    _integer(value[0], 0, len(request.text))
    _integer(value[1], value[0], len(request.text))
    return tuple(value)


def _count(value: object, request: WorkerRequest) -> None:
    rules = request.policy.get("rules", request.policy.get("block_patterns", []))
    _integer(value, 0, len(rules) * (max(len(request.text), request.bounds.max_output_chars) + 1))


def decode_response(frame: bytes, request: WorkerRequest) -> WorkerResponse:
    """Validate successful result geometry/types against the submitted snapshot."""
    data = _unframe(frame, request.bounds.max_response_bytes)
    _object(data, {"version", "tag", "operation", "result", "reason", "resource_limits"})
    if type(data["version"]) is not int or data["version"] != VERSION or data["tag"] != "response":
        _invalid()
    posture = data["resource_limits"]
    if posture is not None:
        _object(posture, {"address_space", "cpu", "core"})
        if any(type(value) is not bool for value in posture.values()):
            _invalid()
    reason = data["reason"]
    result = data["result"]
    if data["operation"] != request.operation:
        if not (
            data["operation"] is None
            and reason
            in (
                "worker_unavailable",
                "invalid_response",
                "size_exceeded",
                "worker_crashed",
            )
            and result is None
        ):
            _invalid()
    if reason is not None:
        if type(reason) is not str or reason not in MODERATION_EXECUTION_REASONS or result is not None:
            _invalid()
        return WorkerResponse(request.operation, reason=reason, resource_limits=posture)
    maximum = request.bounds.max_output_chars
    if request.operation == "evaluate":
        _object(result, {item.name for item in fields(ModerationEvaluationResult)})
        if type(result["action"]) is not str or result["action"] not in {"pass", "warn", "redact", "block"}:
            _invalid()
        for key, cap in (
            ("redacted_text", maximum),
            ("matched_pattern", request.bounds.max_pattern_chars),
            ("category", request.bounds.max_policy_bytes),
            ("sample", 80),
        ):
            _string(result[key], cap, nullable=True)
        result["match_span"] = _span(result["match_span"], request)
        patterns = {rule["pattern"] for rule in request.policy["block_patterns"]}
        if result["matched_pattern"] is not None and result["matched_pattern"] not in patterns:
            _invalid()
        if (result["action"] == "pass") != (result["match_span"] is None):
            _invalid()
        if result["action"] == "pass" and any(
            result[key] is not None for key in ("redacted_text", "matched_pattern", "category", "sample")
        ):
            _invalid()
        if result["action"] != "pass" and result["matched_pattern"] is None:
            _invalid()
        if (result["redacted_text"] is not None) != (result["action"] == "redact"):
            _invalid()
        result = ModerationEvaluationResult(**result)
    elif request.operation == "redact":
        _string(result, maximum)
    elif request.operation == "redact_with_count":
        if type(result) is not list or len(result) != 2:
            _invalid()
        _string(result[0], maximum)
        _count(result[1], request)
        result = tuple(result)
    elif request.operation == "supervised_evaluate":
        _object(result, {"winner_index", "match_span", "redacted_text"})
        index = result["winner_index"]
        span = _span(result["match_span"], request)
        _string(result["redacted_text"], maximum, nullable=True)
        if index is None:
            if span is not None or result["redacted_text"] is not None:
                _invalid()
        else:
            _integer(index, 0, len(request.policy["rules"]) - 1)
            rule = request.policy["rules"][index]
            if (
                span is None
                or rule["action"] not in {"block", "redact", "warn", "notify"}
                or rule["phase"] not in {"both", request.phase}
                or (rule["action"] == "redact") != (result["redacted_text"] is not None)
            ):
                _invalid()
        result = SupervisedBatchResult(index, span, result["redacted_text"])
    else:
        _object(result, {"text", "count", "skipped_rule_indices"})
        _string(result["text"], maximum)
        _count(result["count"], request)
        skipped = result["skipped_rule_indices"]
        if type(skipped) is not list:
            _invalid()
        previous = -1
        for index in skipped:
            _integer(index, previous + 1, len(request.policy["rules"]) - 1)
            previous = index
        result = SubstitutionBatchResult(result["text"], result["count"], tuple(skipped))
    return WorkerResponse(request.operation, result, resource_limits=posture)
