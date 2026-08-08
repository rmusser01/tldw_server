"""Safe application exceptions for strict MCP protocol projection."""

from __future__ import annotations

import re
from typing import Literal

GatewayApplicationErrorKind = Literal["application", "tool", "resource", "prompt"]

_REASON_CODE_PATTERN = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
_ERROR_KINDS = frozenset({"application", "tool", "resource", "prompt"})


def _positive_int(value: object, name: str) -> int:
    """Return a positive integer, explicitly rejecting booleans."""

    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


class GatewayApplicationError(Exception):
    """An application failure containing only bounded wire-safe fields."""

    def __init__(
        self,
        public_message: str,
        *,
        reason_code: str,
        kind: GatewayApplicationErrorKind = "application",
    ) -> None:
        if not isinstance(public_message, str) or not public_message:
            raise ValueError("public_message must be a non-empty string")
        if len(public_message) > 512:
            raise ValueError("public_message must not exceed 512 code points")
        if not isinstance(reason_code, str) or _REASON_CODE_PATTERN.fullmatch(reason_code) is None:
            raise ValueError("reason_code must match [a-z][a-z0-9_]{0,63}")
        if not isinstance(kind, str) or kind not in _ERROR_KINDS:
            raise ValueError("kind must be application, tool, resource, or prompt")

        super().__init__(public_message)
        self.public_message = public_message
        self.reason_code = reason_code
        self.kind = kind


class GatewayToolExecutionError(GatewayApplicationError):
    """A safe, actionable tool execution failure."""

    def __init__(self, public_message: str, *, reason_code: str) -> None:
        super().__init__(public_message, reason_code=reason_code, kind="tool")


class GatewayResourceNotFound(GatewayApplicationError):
    """A generic missing-resource failure without private URI details."""

    def __init__(self) -> None:
        super().__init__(
            "Resource not found",
            reason_code="resource_not_found",
            kind="resource",
        )


class GatewayResultTooLarge(GatewayApplicationError):
    """An application result that exceeds its configured public limit."""

    def __init__(self, *, limit_bytes: int) -> None:
        super().__init__(
            "Application result exceeds the configured limit",
            reason_code="result_too_large",
            kind="application",
        )
        self.limit_bytes = _positive_int(limit_bytes, "limit_bytes")


class GatewayInvalidApplicationResult(GatewayApplicationError):
    """A generic invalid-result failure without private result details."""

    def __init__(self) -> None:
        super().__init__(
            "Application returned an invalid result",
            reason_code="invalid_application_result",
            kind="application",
        )


__all__ = [
    "GatewayApplicationError",
    "GatewayInvalidApplicationResult",
    "GatewayResourceNotFound",
    "GatewayResultTooLarge",
    "GatewayToolExecutionError",
]
