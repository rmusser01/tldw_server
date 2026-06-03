"""Standalone gateway admin-auth helpers."""

from __future__ import annotations

import inspect
import secrets
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast

if TYPE_CHECKING:
    from fastapi import Request
else:
    Request = Any

GatewayAdminVerifier = Callable[[str, Request], bool | Awaitable[bool]]

_DEFAULT_ADMIN_HEADER = "X-MCP-Gateway-Admin-Key"
_REQUIRED_PAYLOAD = {
    "ok": False,
    "error": "Gateway admin authentication required",
    "reason_code": "admin_auth_required",
}
_INVALID_PAYLOAD = {
    "ok": False,
    "error": "Gateway admin authentication failed",
    "reason_code": "admin_auth_invalid",
}


class GatewayAdminAuthConfigLike(Protocol):
    """Protocol for package-owned admin auth config values."""

    enabled: bool
    header_name: str
    api_key: str | None
    verifier: GatewayAdminVerifier | None


class GatewayAdminAuthError(Exception):
    """Raised when a management route receives invalid admin credentials."""

    def __init__(self, *, reason_code: str) -> None:
        """Build an auth failure with a stable public reason code."""

        if reason_code == "admin_auth_required":
            self.status_code = 401
            self.payload = dict(_REQUIRED_PAYLOAD)
        elif reason_code == "admin_auth_invalid":
            self.status_code = 403
            self.payload = dict(_INVALID_PAYLOAD)
        else:
            raise ValueError(f"Unsupported gateway admin auth reason: {reason_code!r}")
        self.reason_code = reason_code
        super().__init__(self.payload["error"])


@dataclass(frozen=True, slots=True)
class GatewayAdminAuthConfig:
    """Runtime admin-auth configuration for standalone gateway management routes."""

    enabled: bool = False
    header_name: str = _DEFAULT_ADMIN_HEADER
    api_key: str | None = None
    verifier: GatewayAdminVerifier | None = None

    def __post_init__(self) -> None:
        """Validate admin auth settings and normalize the header name."""

        if not isinstance(self.enabled, bool):
            raise ValueError("admin_auth.enabled must be a boolean")
        header_name = str(self.header_name).strip()
        if not header_name:
            raise ValueError("admin_auth.header_name must be non-blank")
        object.__setattr__(self, "header_name", header_name)

        api_key = self.api_key
        if api_key is not None:
            api_key = str(api_key)
            if not api_key:
                raise ValueError("admin_auth.api_key must be non-blank when supplied")
            object.__setattr__(self, "api_key", api_key)

        if self.enabled and self.api_key is None and self.verifier is None:
            raise ValueError(
                "admin auth requires api_key or verifier when enabled"
            )

    async def verify(self, credential: str, request: Request) -> bool:
        """Return whether the supplied credential is accepted for this request."""

        if self.verifier is not None:
            result = self.verifier(credential, request)
            if inspect.isawaitable(result):
                result = await cast(Awaitable[bool], result)
            return bool(result)
        if self.api_key is None:
            return False
        return secrets.compare_digest(credential, self.api_key)


def normalize_gateway_admin_auth_config(
    config: GatewayAdminAuthConfig | Mapping[str, Any] | None,
) -> GatewayAdminAuthConfig:
    """Return a runtime admin-auth config from a model, mapping, or empty value."""

    if config is None:
        return GatewayAdminAuthConfig()
    if isinstance(config, GatewayAdminAuthConfig):
        return config
    return GatewayAdminAuthConfig(**dict(config))


def gateway_admin_auth_dependencies(
    config: GatewayAdminAuthConfig | Mapping[str, Any] | None,
) -> list[Any]:
    """Return FastAPI dependencies that enforce optional admin authentication."""

    from fastapi import Depends

    resolved = normalize_gateway_admin_auth_config(config)
    if not resolved.enabled:
        return []
    return [Depends(_gateway_admin_auth_dependency(resolved))]


def gateway_admin_auth_error_response(
    request: Request,
    exc: GatewayAdminAuthError,
) -> Any:
    """Return the package-owned direct JSON error response for admin auth failures."""

    from fastapi.responses import JSONResponse

    del request
    return JSONResponse(status_code=exc.status_code, content=exc.payload)


def _gateway_admin_auth_dependency(
    config: GatewayAdminAuthConfig,
) -> Callable[[Request], Awaitable[None]]:
    """Build a request dependency for one immutable auth config."""

    from fastapi import Request as FastAPIRequest

    async def require_gateway_admin(request: Any) -> None:
        credential = request.headers.get(config.header_name)
        if credential is None or not credential.strip():
            raise GatewayAdminAuthError(reason_code="admin_auth_required")
        if not await config.verify(credential, request):
            raise GatewayAdminAuthError(reason_code="admin_auth_invalid")

    require_gateway_admin.__annotations__["request"] = FastAPIRequest
    return require_gateway_admin


__all__ = [
    "GatewayAdminAuthConfig",
    "GatewayAdminAuthConfigLike",
    "GatewayAdminAuthError",
    "GatewayAdminVerifier",
    "gateway_admin_auth_dependencies",
    "gateway_admin_auth_error_response",
    "normalize_gateway_admin_auth_config",
]
