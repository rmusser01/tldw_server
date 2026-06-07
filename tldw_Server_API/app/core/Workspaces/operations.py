from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any
from urllib.parse import quote

WORKSPACE_OPERATION_ACTIVE_STATUSES = frozenset({"queued", "running"})
WORKSPACE_OPERATION_STATUSES = frozenset(
    {"queued", "running", "succeeded", "failed", "conflicted", "expired"}
)
_MAX_DIAGNOSTIC_KEYS = 16
_MAX_DIAGNOSTIC_STRING = 320
_SECRET_RE = re.compile(r"(api[_-]?key|secret|token|password|credential)", re.IGNORECASE)
_SECRET_VALUE_RE = re.compile(
    r"(Bearer\s+\S+|eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+|"
    r"(?:ghp|github_pat|sk|pk|rk|xox[baprs]?)-[A-Za-z0-9_\-]{16,}|"
    r"[A-Za-z0-9_\-]{32,}\.[A-Za-z0-9_\-]{16,})",
    re.IGNORECASE,
)
_HOST_PATH_RE = re.compile(
    r"(/Users/[^\s'\"<>]+|/home/[^\s'\"<>]+|/private/[^\s'\"<>]+|/var/folders/[^\s'\"<>]+|[A-Za-z]:\\[^\s'\"<>]+)"
)


def fingerprint_workspace_command(payload: Mapping[str, Any]) -> str:
    """Return a stable fingerprint for an idempotent Workspace command payload."""
    stable_payload = _stable_fingerprint_value(payload)
    encoded = json.dumps(stable_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def redact_operation_diagnostics(value: Mapping[str, Any] | None) -> dict[str, Any]:
    """Bound operation diagnostics and remove secrets/raw host-local paths."""
    if not isinstance(value, Mapping):
        return {}
    redacted: dict[str, Any] = {}
    for idx, (key, raw_value) in enumerate(value.items()):
        if idx >= _MAX_DIAGNOSTIC_KEYS:
            redacted["truncated"] = True
            break
        key_text = _bounded_text(key, 80) or "diagnostic"
        redacted[key_text] = _redact_diagnostic_value(key_text, raw_value)
    return redacted


def operation_poll_href(workspace_id: str, operation_id: str) -> str:
    workspace_segment = quote(str(workspace_id), safe="")
    operation_segment = quote(str(operation_id), safe="")
    return f"/api/v1/workspaces/{workspace_segment}/operations/{operation_segment}"


def workspace_operation_response_payload(operation: Mapping[str, Any]) -> dict[str, Any]:
    status = _operation_status(operation.get("status"))
    operation_id = str(operation.get("id") or operation.get("operation_id") or "")
    workspace_id = str(operation.get("workspace_id") or "")
    return {
        "operation_id": operation_id,
        "workspace_id": workspace_id,
        "command": str(operation.get("command") or ""),
        "status": status,
        "started_at": str(operation.get("created_at") or ""),
        "updated_at": str(operation.get("updated_at") or operation.get("created_at") or ""),
        "retryable": _operation_retryable(status, operation.get("diagnostics")),
        "diagnostics": redact_operation_diagnostics(
            operation.get("diagnostics") if isinstance(operation.get("diagnostics"), Mapping) else {}
        ),
        "poll_href": operation_poll_href(workspace_id, operation_id),
    }


def _operation_status(value: Any) -> str:
    status = str(value or "").strip().lower()
    return status if status in WORKSPACE_OPERATION_STATUSES else "failed"


def _operation_retryable(status: str, diagnostics: Any) -> bool:
    if isinstance(diagnostics, Mapping) and diagnostics.get("retryable") is not None:
        return bool(diagnostics["retryable"])
    return status in {"queued", "running", "failed", "conflicted"}


def _stable_fingerprint_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _stable_fingerprint_value(raw_value)
            for key, raw_value in sorted(value.items(), key=lambda item: str(item[0]))
            if not _SECRET_RE.search(str(key))
        }
    if isinstance(value, (list, tuple)):
        return [_stable_fingerprint_value(item) for item in value]
    if isinstance(value, str):
        return _HOST_PATH_RE.sub("[redacted-path]", value)
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    return _HOST_PATH_RE.sub("[redacted-path]", str(value))


def _redact_diagnostic_value(key: str, value: Any) -> Any:
    if _SECRET_RE.search(key):
        return "[redacted]"
    if isinstance(value, Mapping):
        return redact_operation_diagnostics(value)
    if isinstance(value, (list, tuple)):
        return [_redact_diagnostic_value(key, item) for item in value[:_MAX_DIAGNOSTIC_KEYS]]
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    raw_text = str(value)
    if _SECRET_VALUE_RE.search(raw_text):
        return "[redacted]"
    text = _HOST_PATH_RE.sub("[redacted-path]", raw_text)
    if _SECRET_RE.search(text):
        text = "[redacted]"
    return text[:_MAX_DIAGNOSTIC_STRING]


def _bounded_text(value: Any, limit: int) -> str | None:
    text = str(value).strip() if value is not None else ""
    if not text:
        return None
    return text[: max(1, int(limit))]
