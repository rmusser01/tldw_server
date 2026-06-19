"""Bounded, redacted report models for MCP smoke runs."""

from __future__ import annotations

from collections.abc import Sized
from dataclasses import dataclass, field
from itertools import islice
import os
import re

MAX_TEXT_CHARS = 240
MAX_MAPPING_ITEMS = 16
MAX_SEQUENCE_ITEMS = 12
MAX_DEPTH = 5

_REDACTED = "[redacted]"
_REDACTED_ARGUMENTS = "[redacted tool arguments]"
_REDACTED_CONTENT = "[summarized content]"
_REDACTED_PATH = "[local-path]"

_SENSITIVE_KEY_PARTS = (
    "authorization",
    "api_key",
    "apikey",
    "bearer",
    "cookie",
    "credential",
    "password",
    "secret",
    "set_cookie",
    "token",
)
_ARGUMENT_KEYS = {"args", "arguments", "input", "tool_arguments", "tool_args"}
_CONTENT_KEYS = {
    "body",
    "bytes",
    "content",
    "content_bytes",
    "contents",
    "file_contents",
    "file_content",
    "payload",
    "raw",
    "structured_content",
}
_CONTENT_BLOCK_KEYS = {"blob", "text"}
_ENV_KEYS = {"env", "environment", "environ"}

_BEARER_RE = re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]+", re.IGNORECASE)
_FILE_URI_RE = re.compile(r"\bfile://[^\s'\"),\]}]+")
_UNIX_ABSOLUTE_PATH_RE = re.compile(r"(?<![:/\w.-])/(?!/)[^\s'\"),\]}]+")
_WINDOWS_LOCAL_PATH_RE = re.compile(
    r"(?<![\w.-])(?:[A-Za-z]:\\|\\\\)[^\s'\"),\]}]+"
)


@dataclass(slots=True)
class SmokeStepReport:
    """One smoke scenario step outcome."""

    name: str
    ok: bool
    method: str | None = None
    request_id: str | None = None
    elapsed_ms: float | None = None
    result_summary: dict[str, object] | None = None
    error_code: int | None = None
    reason_code: str | None = None
    detail: object | None = None


@dataclass(slots=True)
class SmokeTraceSummary:
    """Small redacted diagnostic trace for one transport exchange."""

    request_id: str | None = None
    method: str | None = None
    status: str | None = None
    elapsed_ms: float | None = None
    detail: object | None = None


@dataclass(slots=True)
class SmokeReport:
    """JSON-serializable smoke run report."""

    transport: str
    steps: list[SmokeStepReport] = field(default_factory=list)
    ok: bool | None = None
    started_at: str | None = None
    elapsed_ms: float | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    traces: list[SmokeTraceSummary] = field(default_factory=list)


def summarize_result(result: object, max_text_chars: int = MAX_TEXT_CHARS) -> dict[str, object]:
    """Return a bounded and redacted summary of a JSON-RPC result object."""

    if isinstance(result, dict):
        summary: dict[str, object] = {"kind": "dict"}
        for key, value in islice(result.items(), MAX_MAPPING_ITEMS):
            key_text = str(key)
            normalized = _normalize_key(key_text)
            redacted = _redact_keyed_value(
                normalized=normalized,
                value=value,
                max_text_chars=max_text_chars,
            )
            if redacted is not _USE_GENERIC_REDACTION:
                summary[key_text] = redacted
                continue
            if normalized in {"tools", "resources", "prompts"} and isinstance(value, list):
                summary[f"{normalized}_count"] = len(value)
                names = _safe_item_names(value, normalized)
                if names:
                    summary[f"{normalized}_sample"] = names
                continue
            summary[key_text] = _redact(value, max_text_chars=max_text_chars)
        omitted_keys = _omitted_count(result, MAX_MAPPING_ITEMS)
        if omitted_keys is not None:
            summary["omitted_keys"] = omitted_keys
        return summary
    return {
        "kind": type(result).__name__,
        "value": _redact(result, max_text_chars=max_text_chars),
    }


def redact_detail(value: object) -> object:
    """Return a recursively redacted, size-bounded diagnostic detail."""

    return _redact(value, max_text_chars=MAX_TEXT_CHARS)


def report_to_json(report: SmokeReport) -> dict[str, object]:
    """Convert a smoke report to a bounded JSON-compatible dictionary."""

    return {
        "ok": _report_ok(report),
        "transport": redact_detail(report.transport),
        "started_at": redact_detail(report.started_at),
        "elapsed_ms": report.elapsed_ms,
        "metadata": redact_detail(report.metadata),
        "steps": [_step_to_json(step) for step in report.steps],
        "traces": [_trace_to_json(trace) for trace in report.traces],
    }


def _step_to_json(step: SmokeStepReport) -> dict[str, object]:
    return {
        "name": redact_detail(step.name),
        "ok": step.ok,
        "method": redact_detail(step.method),
        "request_id": redact_detail(step.request_id),
        "elapsed_ms": step.elapsed_ms,
        "result_summary": redact_detail(step.result_summary),
        "error_code": step.error_code,
        "reason_code": redact_detail(step.reason_code),
        "detail": redact_detail(step.detail),
    }


def _trace_to_json(trace: SmokeTraceSummary) -> dict[str, object]:
    return {
        "request_id": redact_detail(trace.request_id),
        "method": redact_detail(trace.method),
        "status": redact_detail(trace.status),
        "elapsed_ms": trace.elapsed_ms,
        "detail": redact_detail(trace.detail),
    }


def _report_ok(report: SmokeReport) -> bool:
    if report.ok is not None:
        return report.ok
    return all(step.ok for step in report.steps)


def _redact(
    value: object,
    *,
    max_text_chars: int,
    depth: int = 0,
    in_env: bool = False,
    in_content_payload: bool = False,
) -> object:
    if depth > MAX_DEPTH:
        return "[max-depth]"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if in_env:
            return _REDACTED
        return _redact_string(value, max_text_chars=max_text_chars)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"kind": "bytes", "byte_count": len(value)}
    if isinstance(value, dict):
        output: dict[str, object] = {}
        for key, nested in islice(value.items(), MAX_MAPPING_ITEMS):
            key_text = str(key)
            normalized = _normalize_key(key_text)
            redacted = _redact_keyed_value(
                normalized=normalized,
                value=nested,
                max_text_chars=max_text_chars,
                in_content_payload=in_content_payload,
            )
            if redacted is not _USE_GENERIC_REDACTION:
                output[key_text] = redacted
                continue
            output[key_text] = _redact(
                nested,
                max_text_chars=max_text_chars,
                depth=depth + 1,
                in_env=in_env or normalized in _ENV_KEYS,
                in_content_payload=in_content_payload or normalized in _CONTENT_KEYS,
            )
        omitted_keys = _omitted_count(value, MAX_MAPPING_ITEMS)
        if omitted_keys is not None:
            output["omitted_keys"] = omitted_keys
        return output
    if isinstance(value, (list, tuple, set, frozenset)):
        output = [
            _redact(
                item,
                max_text_chars=max_text_chars,
                depth=depth + 1,
                in_env=in_env,
                in_content_payload=in_content_payload,
            )
            for item in islice(value, MAX_SEQUENCE_ITEMS)
        ]
        omitted_items = _omitted_count(value, MAX_SEQUENCE_ITEMS)
        if omitted_items is not None:
            output.append({"omitted_items": omitted_items})
        return output
    return _redact_string(repr(value), max_text_chars=max_text_chars)


def _redact_string(value: str, *, max_text_chars: int) -> str:
    text = _BEARER_RE.sub("[redacted bearer token]", value)
    for env_value in _sensitive_env_values():
        text = text.replace(env_value, _REDACTED)
    text = _FILE_URI_RE.sub(f"file://{_REDACTED_PATH}", text)
    text = _UNIX_ABSOLUTE_PATH_RE.sub(_REDACTED_PATH, text)
    text = _WINDOWS_LOCAL_PATH_RE.sub(_REDACTED_PATH, text)
    if len(text) > max_text_chars:
        omitted = len(text) - max_text_chars
        text = f"{text[:max_text_chars]}...[truncated {omitted} chars]"
    return text


def _sensitive_env_values() -> tuple[str, ...]:
    values: list[str] = []
    for key, value in os.environ.items():
        if len(value) < 8:
            continue
        if _is_sensitive_key(_normalize_key(key)):
            values.append(value)
    values.sort(key=len, reverse=True)
    return tuple(values)


def _normalize_key(key: str) -> str:
    text = key.strip().replace("-", "_").replace(".", "_")
    text = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", text)
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    return text.lower()


def _is_sensitive_key(key: str) -> bool:
    return any(part in key for part in _SENSITIVE_KEY_PARTS)


_USE_GENERIC_REDACTION = object()


def _redact_keyed_value(
    *,
    normalized: str,
    value: object,
    max_text_chars: int,
    in_content_payload: bool = False,
) -> object:
    if _is_sensitive_key(normalized):
        return _REDACTED
    if normalized in _ARGUMENT_KEYS:
        return _REDACTED_ARGUMENTS
    if normalized in _CONTENT_KEYS:
        return _summarize_content(value)
    if in_content_payload and normalized in _CONTENT_BLOCK_KEYS:
        return _summarize_content(value)
    if normalized in _ENV_KEYS:
        return _redact(value, max_text_chars=max_text_chars, in_env=True)
    return _USE_GENERIC_REDACTION


def _omitted_count(value: object, limit: int) -> int | None:
    if not isinstance(value, Sized):
        return None
    total = len(value)
    omitted = total - limit
    if omitted <= 0:
        return None
    return omitted


def _safe_item_names(value: list[object], kind: str) -> list[object]:
    field_name = "uri" if kind == "resources" else "name"
    names: list[object] = []
    for item in value[:MAX_SEQUENCE_ITEMS]:
        if isinstance(item, dict) and field_name in item:
            names.append(_redact(item[field_name], max_text_chars=MAX_TEXT_CHARS))
    return names


def _summarize_content(value: object) -> object:
    if isinstance(value, str):
        return {"summary": _REDACTED_CONTENT, "char_count": len(value)}
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"summary": _REDACTED_CONTENT, "byte_count": len(value)}
    if isinstance(value, (list, tuple, set, frozenset)):
        return {"summary": _REDACTED_CONTENT, "item_count": len(value)}
    if isinstance(value, dict):
        return {"summary": _REDACTED_CONTENT, "key_count": len(value)}
    return _REDACTED_CONTENT
