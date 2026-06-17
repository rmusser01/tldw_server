from __future__ import annotations

import json

import pytest

from mcp_unified.gateway.admin_auth import (
    DefaultGatewayAdminPermissionChecker,
    GatewayAdminAuthConfig,
    GatewayAdminAuthError,
    GatewayAdminIdentity,
    GatewayAdminPermissionError,
    gateway_admin_identity_dependency,
    gateway_admin_permission_error_response,
)


class _RequestStub:
    def __init__(self, headers: dict[str, str] | None = None) -> None:
        self.headers = headers or {}


def test_default_admin_identity_has_policy_explain_permission_when_auth_disabled() -> None:
    identity = GatewayAdminIdentity.local_admin()

    assert identity.actor_id == "local-admin"
    assert "mcp.policy.explain" in identity.permissions


@pytest.mark.asyncio
async def test_gateway_admin_identity_dependency_returns_local_admin_when_auth_disabled() -> None:
    dependency = gateway_admin_identity_dependency(GatewayAdminAuthConfig())

    identity = await dependency(None)

    assert identity == GatewayAdminIdentity.local_admin()


@pytest.mark.asyncio
async def test_gateway_admin_identity_dependency_distinguishes_authenticated_admin() -> None:
    dependency = gateway_admin_identity_dependency(
        GatewayAdminAuthConfig(enabled=True, api_key="secret")
    )

    identity = await dependency(
        _RequestStub(headers={"X-MCP-Gateway-Admin-Key": "secret"})
    )

    assert identity.actor_id == "gateway-admin"
    assert identity.source == "gateway_admin_auth"
    assert "mcp.policy.explain" in identity.permissions

    with pytest.raises(GatewayAdminAuthError) as missing:
        await dependency(_RequestStub())
    assert missing.value.reason_code == "admin_auth_required"

    with pytest.raises(GatewayAdminAuthError) as invalid:
        await dependency(
            _RequestStub(headers={"X-MCP-Gateway-Admin-Key": "wrong"})
        )
    assert invalid.value.reason_code == "admin_auth_invalid"


@pytest.mark.asyncio
async def test_permission_checker_denies_missing_policy_explain_permission() -> None:
    checker = DefaultGatewayAdminPermissionChecker()
    identity = GatewayAdminIdentity(actor_id="viewer", permissions=frozenset())

    with pytest.raises(GatewayAdminPermissionError) as exc_info:
        await checker.require_permission(identity, "mcp.policy.explain")

    assert exc_info.value.reason_code == "admin_permission_denied"


def test_permission_error_response_uses_stable_error_envelope() -> None:
    response = gateway_admin_permission_error_response(
        None,
        GatewayAdminPermissionError(reason_code="admin_permission_denied"),
    )

    assert response.status_code == 403
    assert json.loads(response.body) == {
        "ok": False,
        "error": "Gateway admin permission denied",
        "reason_code": "admin_permission_denied",
    }
