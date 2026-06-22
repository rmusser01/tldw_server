"""Shared exceptions for MCP smoke clients and transports."""

from __future__ import annotations


class McpSmokeClientError(RuntimeError):
    """Raised when a smoke client receives an unusable JSON-RPC response."""

    def __init__(
        self,
        message: str,
        *,
        response: object | None = None,
        error: object | None = None,
    ) -> None:
        super().__init__(message)
        self.response = response
        self.error = error


class McpSmokeTransportError(RuntimeError):
    """Raised when a smoke transport cannot exchange JSON-RPC payloads."""

    def __init__(
        self,
        reason_code: str,
        message: str,
        *,
        method: str | None = None,
        status_code: int | None = None,
        cause: BaseException | None = None,
    ) -> None:
        parts = [reason_code, message]
        if method is not None:
            parts.append(f"method={method}")
        if status_code is not None:
            parts.append(f"status_code={status_code}")
        super().__init__(": ".join(parts))
        self.reason_code = reason_code
        self.method = method
        self.status_code = status_code
        self.__cause__ = cause


__all__ = ["McpSmokeClientError", "McpSmokeTransportError"]
