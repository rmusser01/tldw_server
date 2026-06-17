from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from mcp_unified.gateway.admin_auth import (
    DefaultGatewayAdminPermissionChecker,
    GatewayAdminAuthConfig,
    GatewayAdminAuthError,
    GatewayAdminIdentity,
    GatewayAdminPermissionError,
    gateway_admin_identity_dependency,
    gateway_admin_permission_error_response,
)
from mcp_unified.gateway.bootstrap import bootstrap_profile_gateway
from mcp_unified.gateway.fastapi import create_gateway_app
from mcp_unified.gateway.policy_explain import GatewayPolicyExplainService
from mcp_unified.profiles import MCPProfile, ProfilePolicy
from mcp_unified.storage.models import AuditEvent


class _RequestStub:
    def __init__(self, headers: dict[str, str] | None = None) -> None:
        self.headers = headers or {}


class _MemoryAuditStore:
    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    async def append_event(self, event: AuditEvent) -> None:
        self.events.append(event)


class _PolicyExplainRuntime:
    name = "policy-explain-test"
    version = "0.0-test"

    async def list_tools(self, context: Any) -> list[dict[str, Any]]:
        assert context.client_id is None
        assert context.user_id is None
        return [
            {
                "name": "fs.patch",
                "description": "Patch files",
                "metadata": {"category": "filesystem"},
            },
            {
                "name": "shell.exec",
                "description": "Execute shell commands",
                "metadata": {"category": "shell"},
            },
        ]


class _DenyingPermissionChecker:
    async def require_permission(
        self,
        identity: GatewayAdminIdentity,
        permission: str,
    ) -> None:
        assert identity.actor_id == "local-admin"
        assert permission == "mcp.policy.explain"
        raise GatewayAdminPermissionError(reason_code="admin_permission_denied")


def _policy_profile() -> MCPProfile:
    return MCPProfile(
        id="backend-engineer",
        name="Backend Engineer",
        policy_document=ProfilePolicy(
            allowed_tools=["fs.patch"],
            denied_tools=["shell.exec"],
        ),
    )


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


def test_policy_explain_route_requires_admin_key_when_admin_auth_enabled() -> None:
    app = create_gateway_app(
        _PolicyExplainRuntime(),
        enable_policy_explain_management=True,
        policy_explain_profile_resolver=lambda profile_id: _policy_profile(),
        policy_explain_audit_store=_MemoryAuditStore(),
        admin_auth=GatewayAdminAuthConfig(enabled=True, api_key="secret"),
    )

    with TestClient(app) as client:
        response = client.post(
            "/mcp/policy/explain",
            json={"profile_id": "backend-engineer", "tool_name": "fs.patch"},
        )

    assert response.status_code == 401
    assert response.json() == {
        "ok": False,
        "error": "Gateway admin authentication required",
        "reason_code": "admin_auth_required",
    }


def test_policy_explain_route_succeeds_and_audits_with_valid_admin_key() -> None:
    audit = _MemoryAuditStore()
    app = create_gateway_app(
        _PolicyExplainRuntime(),
        enable_policy_explain_management=True,
        policy_explain_profile_resolver=lambda profile_id: _policy_profile(),
        policy_explain_audit_store=audit,
        admin_auth=GatewayAdminAuthConfig(enabled=True, api_key="secret"),
    )

    with TestClient(app) as client:
        response = client.post(
            "/mcp/policy/explain",
            headers={"X-MCP-Gateway-Admin-Key": "secret"},
            json={
                "profile_id": "backend-engineer",
                "tool_name": "fs.patch",
                "arguments": {"path": "workspace/example.py"},
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["profile_id"] == "backend-engineer"
    assert payload["tool_name"] == "fs.patch"
    assert payload["final_outcome"] == "allow"
    assert audit.events[0].event_type == "policy.explain.requested"
    assert audit.events[0].actor_id == "gateway-admin"
    assert audit.events[0].target_id == "fs.patch"


def test_profile_tool_preview_route_includes_denied_installed_runtime_tool() -> None:
    audit = _MemoryAuditStore()
    app = create_gateway_app(
        _PolicyExplainRuntime(),
        enable_policy_explain_management=True,
        policy_explain_profile_resolver=lambda profile_id: _policy_profile(),
        policy_explain_audit_store=audit,
    )

    with TestClient(app) as client:
        response = client.post(
            "/mcp/profiles/backend-engineer/tool-preview",
            json={},
        )

    assert response.status_code == 200
    payload = response.json()
    tools_by_name = {tool["tool_name"]: tool for tool in payload["tools"]}
    assert tools_by_name["fs.patch"]["outcome"] == "allow"
    assert tools_by_name["fs.patch"]["installation_status"] == "installed"
    assert tools_by_name["shell.exec"]["outcome"] == "deny"
    assert tools_by_name["shell.exec"]["visibility"] == "hidden"
    assert tools_by_name["shell.exec"]["installation_status"] == "installed"
    assert audit.events[0].event_type == "policy.preview_tools.requested"


@pytest.mark.asyncio
async def test_profile_tool_preview_route_uses_unfiltered_profile_runtime_catalog() -> None:
    audit = _MemoryAuditStore()
    bootstrap = await bootstrap_profile_gateway(
        _PolicyExplainRuntime(),
        profiles=[_policy_profile()],
        default_profile_id="backend-engineer",
    )
    app = create_gateway_app(
        bootstrap.runtime,
        enable_policy_explain_management=True,
        policy_explain_profile_resolver=lambda profile_id: _policy_profile(),
        policy_explain_audit_store=audit,
    )

    with TestClient(app) as client:
        response = client.post(
            "/mcp/profiles/backend-engineer/tool-preview",
            json={},
        )

    assert response.status_code == 200
    payload = response.json()
    tools_by_name = {tool["tool_name"]: tool for tool in payload["tools"]}
    assert tools_by_name["fs.patch"]["installation_status"] == "installed"
    assert tools_by_name["shell.exec"]["outcome"] == "deny"
    assert tools_by_name["shell.exec"]["visibility"] == "hidden"
    assert tools_by_name["shell.exec"]["installation_status"] == "installed"


def test_profile_tool_preview_route_rejects_conflicting_body_profile_id() -> None:
    audit = _MemoryAuditStore()
    app = create_gateway_app(
        _PolicyExplainRuntime(),
        enable_policy_explain_management=True,
        policy_explain_profile_resolver=lambda profile_id: _policy_profile(),
        policy_explain_audit_store=audit,
    )

    with TestClient(app) as client:
        response = client.post(
            "/mcp/profiles/backend-engineer/tool-preview",
            json={"profile_id": "frontend-engineer"},
        )

    assert response.status_code == 422
    assert response.json() == {
        "ok": False,
        "message": "Invalid policy preview request",
        "reason_code": "invalid_policy_preview_request",
        "details": {},
    }
    assert audit.events == []


def test_injected_policy_explain_service_uses_route_identity_and_runtime_catalog() -> None:
    audit = _MemoryAuditStore()
    service = GatewayPolicyExplainService(
        profile_resolver=lambda profile_id: _policy_profile(),
        audit_store=audit,
    )
    app = create_gateway_app(
        _PolicyExplainRuntime(),
        enable_policy_explain_management=True,
        policy_explain_service=service,
        admin_auth=GatewayAdminAuthConfig(enabled=True, api_key="secret"),
    )

    with TestClient(app) as client:
        explain_response = client.post(
            "/mcp/policy/explain",
            headers={"X-MCP-Gateway-Admin-Key": "secret"},
            json={"profile_id": "backend-engineer", "tool_name": "fs.patch"},
        )
        preview_response = client.post(
            "/mcp/profiles/backend-engineer/tool-preview",
            headers={"X-MCP-Gateway-Admin-Key": "secret"},
            json={},
        )

    assert explain_response.status_code == 200
    assert audit.events[0].event_type == "policy.explain.requested"
    assert audit.events[0].actor_id == "gateway-admin"
    assert preview_response.status_code == 200
    preview_payload = preview_response.json()
    tools_by_name = {tool["tool_name"]: tool for tool in preview_payload["tools"]}
    assert preview_payload["degraded"] is False
    assert tools_by_name["shell.exec"]["installation_status"] == "installed"
    assert audit.events[1].event_type == "policy.preview_tools.requested"
    assert audit.events[1].actor_id == "gateway-admin"


def test_policy_explain_route_maps_permission_denial_to_stable_json_envelope() -> None:
    audit = _MemoryAuditStore()
    app = create_gateway_app(
        _PolicyExplainRuntime(),
        enable_policy_explain_management=True,
        policy_explain_profile_resolver=lambda profile_id: _policy_profile(),
        policy_explain_audit_store=audit,
        policy_explain_permission_checker=_DenyingPermissionChecker(),
    )

    with TestClient(app) as client:
        response = client.post(
            "/mcp/policy/explain",
            json={"profile_id": "backend-engineer", "tool_name": "fs.patch"},
        )

    assert response.status_code == 403
    assert response.json() == {
        "ok": False,
        "message": "Gateway admin permission denied",
        "reason_code": "admin_permission_denied",
        "details": {},
    }
    assert audit.events == []


def test_policy_explain_route_maps_policy_errors_to_stable_json_envelope() -> None:
    audit = _MemoryAuditStore()
    app = create_gateway_app(
        _PolicyExplainRuntime(),
        enable_policy_explain_management=True,
        policy_explain_profile_resolver=lambda profile_id: None,
        policy_explain_audit_store=audit,
    )

    with TestClient(app) as client:
        response = client.post(
            "/mcp/policy/explain",
            json={"profile_id": "missing-profile", "tool_name": "fs.patch"},
        )

    assert response.status_code == 404
    assert response.json() == {
        "ok": False,
        "message": "Profile not found",
        "reason_code": "profile_not_found",
        "details": {},
    }
