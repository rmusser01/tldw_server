"""Shared helpers for MCP Unified tests."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable
from contextlib import contextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.api.v1.endpoints.mcp_unified_endpoint import router as mcp_router
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.MCP_unified import get_config
from tldw_Server_API.app.core.MCP_unified.auth.authnz_rbac import reset_rbac_policy
from tldw_Server_API.app.core.MCP_unified.auth import jwt_manager as jwt_manager_module
from tldw_Server_API.app.core.MCP_unified.monitoring import metrics as metrics_module
from tldw_Server_API.app.core.MCP_unified.security.ip_filter import get_ip_access_controller
from tldw_Server_API.app.core.MCP_unified.server import reset_mcp_server
from tldw_Server_API.app.core.config import API_V1_PREFIX

SAFE_DEFAULT_ENV_VARS = (
    "MCP_ALLOWED_IPS",
    "MCP_WS_ALLOWED_ORIGINS",
    "MCP_CORS_ORIGINS",
    "MCP_WS_AUTH_REQUIRED",
    "MCP_TRUST_X_FORWARDED",
    "MCP_JWT_SECRET",
    "MCP_API_KEY_SALT",
)

ACCESS_TOKEN_TYPE = "access"  # nosec B105 - JWT token type literal, not a credential


def clear_mcp_singleton_state() -> None:
    """Clear MCP caches/singletons that commonly leak between tests."""
    with contextlib.suppress(Exception):
        get_config.cache_clear()  # type: ignore[attr-defined]
    with contextlib.suppress(Exception):
        get_ip_access_controller.cache_clear()  # type: ignore[attr-defined]
    with contextlib.suppress(Exception):
        reset_rbac_policy()
    jwt_manager_module._jwt_manager = None
    metrics_module._metrics_collector = None


def reset_mcp_test_state() -> None:
    """Reset shared MCP state before and after isolated tests."""
    clear_mcp_singleton_state()
    with contextlib.suppress(Exception):
        asyncio.run(reset_mcp_server())
    clear_mcp_singleton_state()


def build_mcp_test_app(
    *,
    auth_principal_override: Callable[[Request], AuthPrincipal] | None = None,
) -> FastAPI:
    """Create a fresh FastAPI app with the MCP router mounted."""
    app = FastAPI()
    app.include_router(mcp_router, prefix=API_V1_PREFIX)
    if auth_principal_override is not None:
        app.dependency_overrides[auth_deps.get_auth_principal] = auth_principal_override
    return app


@contextmanager
def build_mcp_test_client(
    *,
    auth_principal_override: Callable[[Request], AuthPrincipal] | None = None,
):
    """Yield a TestClient bound to a fresh MCP test app."""
    reset_mcp_test_state()
    app = build_mcp_test_app(auth_principal_override=auth_principal_override)
    client = TestClient(app)
    try:
        yield client
    finally:
        client.close()
        reset_mcp_test_state()


def build_mcp_admin_auth_override() -> Callable[[Request], AuthPrincipal]:
    """Return an auth override that allows authenticated admin requests."""

    async def _fake_get_auth_principal(request: Request) -> AuthPrincipal:  # type: ignore[override]
        auth = request.headers.get("Authorization")
        x_api_key = request.headers.get("X-API-KEY")
        if not auth and not x_api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required",
            )
        return AuthPrincipal(
            kind="user",
            user_id=1,
            api_key_id=None,
            subject=None,
            token_type=ACCESS_TOKEN_TYPE,
            jti=None,
            roles=["admin"],
            permissions=["system.logs"],
            is_admin=True,
            org_ids=[],
            team_ids=[],
        )

    return _fake_get_auth_principal
