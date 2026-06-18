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
_PERMISSION_DENIED_PAYLOAD = {
    "ok": False,
    "error": "Gateway admin permission denied",
    "reason_code": "admin_permission_denied",
}
_LOCAL_ADMIN_POLICY_EXPLAIN_PERMISSION = "mcp.policy.explain"


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


class GatewayAdminPermissionError(Exception):
    """Raised when an admin identity lacks a required gateway permission."""

    def __init__(self, *, reason_code: str) -> None:
        """Build a permission failure with a stable public reason code."""

        if reason_code != "admin_permission_denied":
            raise ValueError(
                f"Unsupported gateway admin permission reason: {reason_code!r}"
            )
        self.status_code = 403
        self.payload = dict(_PERMISSION_DENIED_PAYLOAD)
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


@dataclass(frozen=True, slots=True)
class GatewayAdminIdentity:
    """Authenticated gateway administrator identity and effective permissions."""

    actor_id: str
    permissions: frozenset[str]
    source: str = "gateway_admin_auth"

    def __post_init__(self) -> None:
        """Normalize identity fields for stable permission checks."""

        actor_id = str(self.actor_id).strip()
        if not actor_id:
            raise ValueError("gateway admin identity actor_id must be non-blank")
        object.__setattr__(self, "actor_id", actor_id)
        object.__setattr__(self, "permissions", frozenset(self.permissions))

        source = str(self.source).strip()
        if not source:
            raise ValueError("gateway admin identity source must be non-blank")
        object.__setattr__(self, "source", source)

    @classmethod
    def local_admin(cls) -> "GatewayAdminIdentity":
        """Return the default local administrator identity."""

        return cls(
            actor_id="local-admin",
            permissions=frozenset({_LOCAL_ADMIN_POLICY_EXPLAIN_PERMISSION}),
            source="local",
        )

    @classmethod
    def authenticated_admin(cls) -> "GatewayAdminIdentity":
        """Return the generic authenticated gateway administrator identity."""

        return cls(
            actor_id="gateway-admin",
            permissions=frozenset({_LOCAL_ADMIN_POLICY_EXPLAIN_PERMISSION}),
        )


class GatewayAdminPermissionChecker(Protocol):
    """Protocol for checking gateway admin effective permissions."""

    async def require_permission(
        self,
        identity: GatewayAdminIdentity,
        permission: str,
    ) -> None:
        """Raise when the identity lacks the required permission."""


class DefaultGatewayAdminPermissionChecker:
    """Default in-memory gateway admin permission checker."""

    async def require_permission(
        self,
        identity: GatewayAdminIdentity,
        permission: str,
    ) -> None:
        """Require one permission from the identity's effective permissions."""

        if permission not in identity.permissions:
            raise GatewayAdminPermissionError(reason_code="admin_permission_denied")


def default_gateway_admin_identity() -> GatewayAdminIdentity:
    """Return the default local gateway admin identity."""

    return GatewayAdminIdentity.local_admin()


def authenticated_gateway_admin_identity() -> GatewayAdminIdentity:
    """Return the default authenticated gateway admin identity."""

    return GatewayAdminIdentity.authenticated_admin()


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


def gateway_admin_identity_dependency(
    config: GatewayAdminAuthConfig | Mapping[str, Any] | None,
) -> Callable[[Request], Awaitable[GatewayAdminIdentity]]:
    """Return a FastAPI dependency that authenticates and yields an identity."""

    return _gateway_admin_identity_dependency(
        normalize_gateway_admin_auth_config(config)
    )


def gateway_admin_auth_error_response(
    request: Request,
    exc: GatewayAdminAuthError,
) -> Any:
    """Return the package-owned direct JSON error response for admin auth failures."""

    from fastapi.responses import JSONResponse

    del request
    return JSONResponse(status_code=exc.status_code, content=exc.payload)


def gateway_admin_permission_error_response(
    request: Request,
    exc: GatewayAdminPermissionError,
) -> Any:
    """Return the package-owned direct JSON error response for permission failures."""

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


def _gateway_admin_identity_dependency(
    config: GatewayAdminAuthConfig,
) -> Callable[[Request], Awaitable[GatewayAdminIdentity]]:
    """Build an identity-producing request dependency."""

    from fastapi import Request as FastAPIRequest

    async def require_gateway_admin_identity(request: Any) -> GatewayAdminIdentity:
        if not config.enabled:
            return default_gateway_admin_identity()
        credential = request.headers.get(config.header_name)
        if credential is None or not credential.strip():
            raise GatewayAdminAuthError(reason_code="admin_auth_required")
        if not await config.verify(credential, request):
            raise GatewayAdminAuthError(reason_code="admin_auth_invalid")
        return authenticated_gateway_admin_identity()

    require_gateway_admin_identity.__annotations__["request"] = FastAPIRequest
    return require_gateway_admin_identity


__all__ = [
    "DefaultGatewayAdminPermissionChecker",
    "GatewayAdminAuthConfig",
    "GatewayAdminAuthConfigLike",
    "GatewayAdminAuthError",
    "GatewayAdminIdentity",
    "GatewayAdminPermissionChecker",
    "GatewayAdminPermissionError",
    "GatewayAdminVerifier",
    "authenticated_gateway_admin_identity",
    "default_gateway_admin_identity",
    "gateway_admin_auth_dependencies",
    "gateway_admin_auth_error_response",
    "gateway_admin_identity_dependency",
    "gateway_admin_permission_error_response",
    "normalize_gateway_admin_auth_config",
]
