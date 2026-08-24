"""Helpers for building metadata-only MCP tool-use reporting events."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from enum import Enum
from typing import Any

from mcp_unified.tool_use_reporting.models import ToolUseStatus
from mcp_unified.tool_use_reporting.sanitization import (
    sanitize_reason_code,
    sanitize_safe_id,
)

_GOVERNANCE_REASON_FALLBACK = "policy_denied"
_APPROVAL_REASON_FALLBACK = "approval_required"
_RATE_LIMIT_REASON = "rate_limited"
_INVALID_PARAMS_REASON = "invalid_params"
_PERMISSION_REASON = "permission_denied"
_UNAVAILABLE_REASON = "tool_unavailable"
_ERROR_REASON = "tool_execution_failed"

_DIMENSION_CANDIDATES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("profile_id", ("profile_id", "mcp_profile_id", "gateway_profile_id")),
    ("mode_id", ("mode_id", "mcp_mode_id")),
    ("model_id", ("model_id", "mcp_model_id")),
)

_MISSING = object()


def _guarded_getattr(value: Any, name: str, default: Any = None) -> Any:
    """Read an attribute while allowing only cancellation to escape."""

    try:
        return getattr(value, name, default)
    except asyncio.CancelledError:
        raise
    except BaseException:  # noqa: BLE001 - classifier inputs can expose hostile descriptors.
        return default


def _safe_isinstance(value: Any, class_or_tuple: Any) -> bool:
    """Return a type check that degrades safely for hostile runtime objects."""

    try:
        return issubclass(type(value), class_or_tuple)
    except asyncio.CancelledError:
        raise
    except BaseException:  # noqa: BLE001 - virtual type hooks are untrusted here.
        return False


def _safe_mapping_get(payload: Any, key: str, default: Any = None) -> Any:
    """Read one mapping key while allowing only cancellation to escape."""

    if not _safe_isinstance(payload, Mapping):
        return default
    try:
        return payload[key]
    except KeyError:
        return default
    except asyncio.CancelledError:
        raise
    except BaseException:  # noqa: BLE001 - mapping implementations may be hostile.
        return default


def _safe_type_name(exc: BaseException) -> str:
    """Return the real type name without consulting ``exc.__class__``."""

    try:
        name = type(exc).__name__
    except asyncio.CancelledError:
        raise
    except BaseException:  # noqa: BLE001 - hostile metaclasses degrade safely.
        return ""
    return name if type(name) is str else ""


def _mapping_reason_code(payload: Any) -> str | None:
    """Return a safe reason code from a structured exception payload."""

    if not _safe_isinstance(payload, Mapping):
        return None
    for key in ("reason_code", "reason", "code"):
        value = _safe_mapping_get(payload, key, _MISSING)
        if value is _MISSING:
            continue
        try:
            reason_code = sanitize_reason_code(value)
        except asyncio.CancelledError:
            raise
        except BaseException:  # noqa: BLE001 - sanitization input may be hostile.
            continue
        if reason_code:
            return reason_code
    return None


def _safe_exception_family(exc: BaseException) -> str:
    """Return a bounded exception-family reason without preserving messages."""

    reason_code = sanitize_reason_code(_safe_type_name(exc))
    return reason_code or _ERROR_REASON


def _expected_failure_reason_code(exc: BaseException) -> str | None:
    """Return a sanitized reason for a host-neutral expected-failure shape."""

    try:
        reason = exc.reason
        if not isinstance(reason, Enum):
            return None

        reason_code = exc.reason_code
        public_message = exc.public_message
        breaker_action = exc.breaker_action
        catalog_reason_code = reason.reason_code
        catalog_public_message = reason.public_message
        catalog_breaker_action = reason.breaker_action
        breaker_action_value = getattr(breaker_action, "value", breaker_action)
        catalog_breaker_action_value = getattr(
            catalog_breaker_action,
            "value",
            catalog_breaker_action,
        )
        sanitized_reason_code = sanitize_reason_code(reason_code)
        if (
            not isinstance(reason_code, str)
            or sanitized_reason_code is None
            or reason_code != catalog_reason_code
            or not isinstance(public_message, str)
            or not public_message.strip()
            or len(public_message) > 200
            or public_message != catalog_public_message
            or breaker_action_value not in {"ignore", "record_failure"}
            or breaker_action_value != catalog_breaker_action_value
        ):
            return None
    except asyncio.CancelledError:
        raise
    except BaseException:  # noqa: BLE001 - hostile descriptor access degrades safely.
        return None
    return sanitized_reason_code


def classify_tool_use_exception(exc: BaseException) -> tuple[ToolUseStatus, str]:
    """Classify a tool-call exception into reporting status and safe reason code.

    The classifier intentionally relies on exception shape rather than importing
    host-specific exception classes so standalone package imports remain light.
    """

    class_name = _safe_type_name(exc)

    expected_failure_reason = _expected_failure_reason_code(exc)
    if expected_failure_reason is not None:
        return "error", expected_failure_reason

    governance = _guarded_getattr(exc, "governance")
    if governance is not None or class_name == "GovernanceDeniedError":
        return (
            "denied",
            _mapping_reason_code(governance) or _GOVERNANCE_REASON_FALLBACK,
        )

    approval = _guarded_getattr(exc, "approval")
    if approval is not None or class_name == "ApprovalRequiredError":
        return (
            "approval_required",
            _mapping_reason_code(approval) or _APPROVAL_REASON_FALLBACK,
        )

    if class_name in {"RateLimitExceeded", "RateLimitError"}:
        return "rate_limited", _RATE_LIMIT_REASON

    if class_name in {"InvalidParamsException", "ValidationError"} or _safe_isinstance(
        exc,
        (TypeError, ValueError),
    ):
        return "invalid_params", _INVALID_PARAMS_REASON

    if _safe_isinstance(exc, PermissionError):
        return "denied", _PERMISSION_REASON

    if _safe_isinstance(exc, (FileNotFoundError, LookupError)):
        return "unavailable", _UNAVAILABLE_REASON

    return "error", _safe_exception_family(exc)


def extract_safe_context_dimensions(metadata: Mapping[str, Any] | None) -> dict[str, str]:
    """Extract allowlisted, sanitized reporting dimensions from request metadata."""

    if not _safe_isinstance(metadata, Mapping):
        return {}

    dimensions: dict[str, str] = {}
    for output_key, candidate_keys in _DIMENSION_CANDIDATES:
        for candidate_key in candidate_keys:
            value = sanitize_safe_id(
                _safe_mapping_get(metadata, candidate_key),
                field=output_key,
            )
            if value:
                dimensions[output_key] = value
                break

    if _safe_mapping_get(metadata, "mcp_tool_use_safe_correlation_id") is True:
        for candidate_key in ("correlation_id", "request_id"):
            value = sanitize_safe_id(
                _safe_mapping_get(metadata, candidate_key),
                field="correlation_id",
            )
            if value:
                dimensions["correlation_id"] = value
                break

    return dimensions
