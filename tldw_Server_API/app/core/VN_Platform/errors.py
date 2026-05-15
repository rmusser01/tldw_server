"""Stable error helpers for VN platform endpoints."""

from __future__ import annotations

from typing import Any

ERROR_IDEMPOTENCY_KEY_CONFLICT = "idempotency_key_conflict"
ERROR_INVALID_REQUEST = "invalid_request"
ERROR_NOT_FOUND = "not_found"
ERROR_PERMISSION_DENIED = "permission_denied"
ERROR_POLICY_BLOCKED = "policy_blocked"


def vn_error_detail(
    code: str,
    message: str,
    *,
    details: dict[str, Any] | None = None,
    retryable: bool = False,
) -> dict[str, Any]:
    """Build the stable FastAPI ``HTTPException.detail`` shape for VN errors."""
    return {
        "code": code,
        "message": message,
        "details": details or {},
        "retryable": retryable,
    }
