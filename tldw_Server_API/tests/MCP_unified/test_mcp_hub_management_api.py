from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
import pytest
from starlette.requests import Request

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.api.v1.endpoints import mcp_hub_management
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.exceptions import BadRequestError, ResourceNotFoundError
from tldw_Server_API.app.services import mcp_hub_service


class _NeverDisconnectedRequest:
    async def is_disconnected(self) -> bool:
        return False


def _make_principal(
    *,
    roles: list[str] | None = None,
    permissions: list[str] | None = None,
    team_ids: list[int] | None = None,
) -> AuthPrincipal:
    return AuthPrincipal(
        kind="user",
        user_id=1,
        api_key_id=None,
        subject="1",
        token_type="access",
        jti=None,
        roles=roles or [],
        permissions=permissions or [],
        is_admin=False,
        org_ids=[],
        team_ids=team_ids or [],
    )


class _FakeService:
    async def list_acp_profiles(self, **_kwargs: Any) -> list[dict[str, Any]]:
        return []

    async def get_permission_profile(self, profile_id: int) -> dict[str, Any] | None:
        return {
            "id": profile_id,
            "name": "Docs Profile",
            "owner_scope_type": "global",
            "owner_scope_id": None,
            "mode": "custom",
            "policy_document": {"capabilities": ["network.external"]},
            "is_active": True,
        }

    async def get_policy_assignment(self, assignment_id: int) -> dict[str, Any] | None:
        return {
            "id": assignment_id,
            "target_type": "persona",
            "target_id": "researcher",
            "owner_scope_type": "global",
            "owner_scope_id": None,
            "profile_id": 7,
            "inline_policy_document": {"capabilities": ["network.external"]},
            "approval_policy_id": None,
            "is_active": True,
        }

    async def list_external_servers(self, **_kwargs: Any) -> list[dict[str, Any]]:
        return [
            {
                "id": "docs",
                "name": "Docs",
                "enabled": True,
                "owner_scope_type": "global",
                "owner_scope_id": None,
                "transport": "websocket",
                "config": {"url": "wss://docs.example/ws"},
                "secret_configured": True,
                "key_hint": "cdef",
                "server_source": "managed",
                "legacy_source_ref": None,
                "superseded_by_server_id": None,
                "binding_count": 2,
                "runtime_executable": True,
                "auth_template_present": True,
                "auth_template_valid": True,
                "auth_template_blocked_reason": None,
                "credential_slots": [
                    {
                        "server_id": "docs",
                        "slot_name": "token_readonly",
                        "display_name": "Read-only token",
                        "secret_kind": "bearer_token",
                        "privilege_class": "read",
                        "is_required": True,
                        "secret_configured": True,
                    }
                ],
                "created_by": 1,
                "updated_by": 1,
                "created_at": None,
                "updated_at": None,
            }
        ]

    async def get_external_server_auth_template(self, *, server_id: str) -> dict[str, Any]:
        assert server_id == "docs"
        return {
            "mode": "template",
            "mappings": [
                {
                    "slot_name": "token_readonly",
                    "target_type": "header",
                    "target_name": "Authorization",
                    "prefix": "Bearer ",
                    "suffix": "",
                    "required": True,
                }
            ],
        }

    async def update_external_server_auth_template(
        self,
        *,
        server_id: str,
        auth_template: dict[str, Any],
        actor_id: int | None,
    ) -> dict[str, Any]:
        assert server_id == "docs"
        assert actor_id == 1
        return auth_template

    async def import_legacy_external_server(self, *, server_id: str, actor_id: int | None):
        assert actor_id == 1
        assert server_id == "legacy-docs"
        return {
            "id": server_id,
            "name": "Legacy Docs",
            "enabled": True,
            "owner_scope_type": "global",
            "owner_scope_id": None,
            "transport": "websocket",
            "config": {"url": "wss://docs.example/ws"},
            "secret_configured": False,
            "key_hint": None,
            "server_source": "managed",
            "legacy_source_ref": "yaml:legacy-docs",
            "superseded_by_server_id": None,
            "binding_count": 0,
            "runtime_executable": True,
            "created_by": actor_id,
            "updated_by": actor_id,
            "created_at": None,
            "updated_at": None,
        }

    async def list_profile_credential_bindings(self, *, profile_id: int) -> list[dict[str, Any]]:
        assert profile_id == 7
        return [
            {
                "id": 1,
                "binding_target_type": "profile",
                "binding_target_id": "7",
                "external_server_id": "docs",
                "slot_name": "token_readonly",
                "credential_ref": "server",
                "binding_mode": "grant",
                "usage_rules": {},
                "created_by": 1,
                "updated_by": 1,
                "created_at": None,
                "updated_at": None,
            }
        ]

    async def upsert_profile_credential_binding(
        self,
        *,
        profile_id: int,
        external_server_id: str,
        slot_name: str | None = None,
        managed_secret_ref_id: int | None = None,
        actor_id: int | None,
    ) -> dict[str, Any]:
        assert profile_id == 7
        assert external_server_id == "docs"
        assert slot_name in {None, "token_readonly"}
        assert managed_secret_ref_id is None
        assert actor_id == 1
        return {
            "id": 1,
            "binding_target_type": "profile",
            "binding_target_id": "7",
            "external_server_id": "docs",
            "slot_name": slot_name or "token_readonly",
            "credential_ref": "server",
            "binding_mode": "grant",
            "usage_rules": {},
            "created_by": 1,
            "updated_by": 1,
            "created_at": None,
            "updated_at": None,
        }

    async def delete_profile_credential_binding(
        self,
        *,
        profile_id: int,
        external_server_id: str,
        slot_name: str | None = None,
        actor_id: int | None,
    ) -> bool:
        assert profile_id == 7
        assert external_server_id == "docs"
        assert slot_name in {None, "token_readonly"}
        assert actor_id == 1
        return True

    async def list_assignment_credential_bindings(self, *, assignment_id: int) -> list[dict[str, Any]]:
        assert assignment_id == 11
        return [
            {
                "id": 2,
                "binding_target_type": "assignment",
                "binding_target_id": "11",
                "external_server_id": "docs",
                "slot_name": "token_write",
                "credential_ref": "server",
                "binding_mode": "disable",
                "usage_rules": {},
                "created_by": 1,
                "updated_by": 1,
                "created_at": None,
                "updated_at": None,
            }
        ]

    async def upsert_assignment_credential_binding(
        self,
        *,
        assignment_id: int,
        external_server_id: str,
        slot_name: str | None = None,
        binding_mode: str,
        managed_secret_ref_id: int | None = None,
        actor_id: int | None,
    ) -> dict[str, Any]:
        assert assignment_id == 11
        assert external_server_id == "docs"
        assert slot_name in {None, "token_write"}
        assert binding_mode == "disable"
        assert managed_secret_ref_id is None
        assert actor_id == 1
        return {
            "id": 2,
            "binding_target_type": "assignment",
            "binding_target_id": "11",
            "external_server_id": "docs",
            "slot_name": slot_name or "token_write",
            "credential_ref": "server",
            "binding_mode": "disable",
            "usage_rules": {},
            "created_by": 1,
            "updated_by": 1,
            "created_at": None,
            "updated_at": None,
        }

    async def delete_assignment_credential_binding(
        self,
        *,
        assignment_id: int,
        external_server_id: str,
        slot_name: str | None = None,
        actor_id: int | None,
    ) -> bool:
        assert assignment_id == 11
        assert external_server_id == "docs"
        assert slot_name in {None, "token_write"}
        assert actor_id == 1
        return True

    async def resolve_effective_external_access(
        self,
        *,
        assignment_id: int,
        actor_id: int | None,
    ) -> dict[str, Any]:
        assert assignment_id == 11
        assert actor_id == 1
        return {
            "servers": [
                {
                    "server_id": "docs",
                    "server_name": "Docs",
                    "granted_by": "profile",
                    "disabled_by_assignment": True,
                    "server_source": "managed",
                    "superseded_by_server_id": None,
                    "secret_available": True,
                    "runtime_executable": False,
                    "blocked_reason": "disabled_by_assignment",
                    "slots": [
                        {
                            "slot_name": "token_readonly",
                            "display_name": "Read-only token",
                            "granted_by": "profile",
                            "disabled_by_assignment": False,
                            "secret_available": True,
                            "runtime_usable": True,
                            "blocked_reason": None,
                        },
                        {
                            "slot_name": "token_write",
                            "display_name": "Write token",
                            "granted_by": "assignment",
                            "disabled_by_assignment": True,
                            "secret_available": True,
                            "runtime_usable": False,
                            "blocked_reason": "disabled_by_assignment",
                        },
                    ],
                }
            ]
        }

    async def set_external_server_secret(self, *, server_id: str, secret_value: str, actor_id: int | None):
        assert actor_id == 1
        assert server_id == "docs"
        assert secret_value == "abc123secret"
        return {
            "server_id": server_id,
            "secret_configured": True,
            "key_hint": "cdef",
            "updated_at": None,
        }

    async def list_external_server_credential_slots(self, *, server_id: str) -> list[dict[str, Any]]:
        assert server_id == "docs"
        return [
            {
                "server_id": "docs",
                "slot_name": "token_readonly",
                "display_name": "Read-only token",
                "secret_kind": "bearer_token",
                "privilege_class": "read",
                "is_required": True,
                "secret_configured": True,
            }
        ]

    async def create_external_server_credential_slot(
        self,
        *,
        server_id: str,
        slot_name: str,
        display_name: str,
        secret_kind: str,
        privilege_class: str,
        is_required: bool,
        actor_id: int | None,
    ) -> dict[str, Any]:
        assert server_id == "docs"
        assert slot_name == "token_readonly"
        assert display_name == "Read-only token"
        assert secret_kind == "bearer_token"
        assert privilege_class == "read"
        assert is_required is True
        assert actor_id == 1
        return {
            "server_id": server_id,
            "slot_name": slot_name,
            "display_name": display_name,
            "secret_kind": secret_kind,
            "privilege_class": privilege_class,
            "is_required": is_required,
            "secret_configured": False,
        }

    async def update_external_server_credential_slot(
        self,
        *,
        server_id: str,
        slot_name: str,
        display_name: str | None = None,
        secret_kind: str | None = None,
        privilege_class: str | None = None,
        is_required: bool | None = None,
        actor_id: int | None,
    ) -> dict[str, Any]:
        assert server_id == "docs"
        assert slot_name == "token_readonly"
        assert display_name == "Updated read-only token"
        assert actor_id == 1
        return {
            "server_id": server_id,
            "slot_name": slot_name,
            "display_name": display_name,
            "secret_kind": secret_kind or "bearer_token",
            "privilege_class": privilege_class or "read",
            "is_required": True if is_required is None else is_required,
            "secret_configured": True,
        }

    async def delete_external_server_credential_slot(
        self,
        *,
        server_id: str,
        slot_name: str,
        actor_id: int | None,
    ) -> bool:
        assert server_id == "docs"
        assert slot_name == "token_readonly"
        assert actor_id == 1
        return True

    async def set_external_server_slot_secret(
        self,
        *,
        server_id: str,
        slot_name: str,
        secret_value: str,
        actor_id: int | None,
    ) -> dict[str, Any]:
        assert server_id == "docs"
        assert slot_name == "token_readonly"
        assert secret_value == "abc123secret"
        assert actor_id == 1
        return {
            "server_id": server_id,
            "slot_name": slot_name,
            "secret_configured": True,
            "key_hint": "cdef",
            "updated_at": None,
        }

    async def clear_external_server_slot_secret(
        self,
        *,
        server_id: str,
        slot_name: str,
        actor_id: int | None,
    ) -> bool:
        assert server_id == "docs"
        assert slot_name == "token_readonly"
        assert actor_id == 1
        return True


class _ScopeAuditRepo:
    async def create_permission_profile(self, **_kwargs: Any) -> dict[str, Any]:
        return {
            "id": 44,
            "name": "Team Profile",
            "owner_scope_type": "team",
            "owner_scope_id": 7,
            "mode": "custom",
            "path_scope_object_id": None,
            "policy_document": {},
            "is_active": True,
        }


class _FakeBrokerService:
    async def get_slot_status(
        self,
        *,
        server_id: str,
        slot_name: str,
        profile_id: int | None = None,
        assignment_id: int | None = None,
    ) -> dict[str, Any]:
        binding_target_type = "profile" if profile_id is not None else "assignment"
        binding_target_id = str(profile_id if profile_id is not None else assignment_id)
        return {
            "server_id": server_id,
            "slot_name": slot_name,
            "binding_target_type": binding_target_type,
            "binding_target_id": binding_target_id,
            "credential_ref": "server",
            "managed_secret_ref_id": None,
            "state": "ready",
            "blocked_reason": None,
            "backend_name": "local_encrypted_v1",
            "expires_at": None,
        }


class _FakeExternalFederationManager:
    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result
        self.calls: list[str | None] = []

    async def reconcile_servers(self, server_id: str | None = None) -> dict[str, Any]:
        self.calls.append(server_id)
        return dict(self.result)


class _FakeExternalFederationModule:
    def __init__(self, manager: _FakeExternalFederationManager) -> None:
        self._manager = manager
        self.invalidated = 0

    def invalidate_capability_caches(self) -> None:
        self.invalidated += 1


class _FakeModuleRegistry:
    def __init__(self, module: Any | None) -> None:
        self.module = module
        self.refresh_calls: list[str] = []

    async def get_module(self, module_id: str) -> Any | None:
        if module_id == "external_federation":
            return self.module
        return None

    async def get_all_modules(self) -> dict[str, Any]:
        return {"fallback": self.module} if self.module is not None else {}

    async def refresh_module_registries(self, module_id: str) -> None:
        self.refresh_calls.append(module_id)


class _FakeMcpServer:
    def __init__(self, module: Any | None, *, initialized: bool = True) -> None:
        self.initialized = initialized
        self.initialize_calls = 0
        self.module_registry = _FakeModuleRegistry(module)

    async def initialize(self) -> None:
        self.initialize_calls += 1
        self.initialized = True


def _build_app(
    *,
    principal: AuthPrincipal | None,
    fail_with_401: bool,
) -> FastAPI:
    app = FastAPI()
    app.include_router(mcp_hub_management.router, prefix="/api/v1")

    async def _fake_get_auth_principal(_request: Request) -> AuthPrincipal:  # type: ignore[override]
        if fail_with_401:
            raise HTTPException(
                status_code=401,
                detail="Authentication required",
                headers={"WWW-Authenticate": "Bearer"},
            )
        assert principal is not None
        return principal

    app.dependency_overrides[auth_deps.get_auth_principal] = _fake_get_auth_principal
    app.dependency_overrides[mcp_hub_management.get_mcp_hub_service] = lambda: _FakeService()
    app.dependency_overrides[mcp_hub_management.get_mcp_credential_broker_service] = lambda: _FakeBrokerService()
    return app


@pytest.mark.asyncio
async def test_refresh_external_server_discovery_requires_mutation_permission(monkeypatch) -> None:
    monkeypatch.setattr(
        mcp_hub_management,
        "get_mcp_server",
        lambda: _FakeMcpServer(module=None),
        raising=False,
    )
    app = _build_app(
        principal=_make_principal(permissions=[]),
        fail_with_401=False,
    )

    with TestClient(app) as client:
        resp = client.post("/api/v1/mcp/hub/external-servers/refresh-discovery")

    assert resp.status_code == 403
    assert "system.configure" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_refresh_external_server_discovery_reconciles_live_runtime(monkeypatch) -> None:
    manager = _FakeExternalFederationManager(
        {
            "server_id": "docs",
            "reconciled_servers": 1,
            "refreshed_servers": 1,
            "total_servers": 1,
            "virtual_tools": 2,
            "errors": {},
        }
    )
    module = _FakeExternalFederationModule(manager)
    server = _FakeMcpServer(module=module, initialized=False)
    monkeypatch.setattr(mcp_hub_management, "get_mcp_server", lambda: server, raising=False)

    app = _build_app(
        principal=_make_principal(roles=["admin"], permissions=[]),
        fail_with_401=False,
    )

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/mcp/hub/external-servers/refresh-discovery",
            json={"server_id": "docs"},
        )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["ok"] is True
    assert payload["server_id"] == "docs"
    assert payload["refreshed_servers"] == 1
    assert payload["virtual_tools"] == 2
    assert payload["requires_restart"] is False
    assert manager.calls == ["docs"]
    assert server.initialize_calls == 1
    assert module.invalidated == 1
    assert server.module_registry.refresh_calls == ["external_federation"]


@pytest.mark.asyncio
async def test_refresh_external_server_discovery_accepts_query_server_id(monkeypatch) -> None:
    manager = _FakeExternalFederationManager(
        {
            "server_id": "docs",
            "reconciled_servers": 1,
            "refreshed_servers": 1,
            "total_servers": 1,
            "virtual_tools": 2,
            "errors": {},
        }
    )
    module = _FakeExternalFederationModule(manager)
    monkeypatch.setattr(mcp_hub_management, "get_mcp_server", lambda: _FakeMcpServer(module=module), raising=False)
    app = _build_app(
        principal=_make_principal(roles=["admin"], permissions=[]),
        fail_with_401=False,
    )

    with TestClient(app) as client:
        resp = client.post("/api/v1/mcp/hub/external-servers/refresh-discovery?server_id=docs")

    assert resp.status_code == 200
    assert manager.calls == ["docs"]


@pytest.mark.asyncio
async def test_refresh_external_server_discovery_rejects_unknown_body_fields(monkeypatch) -> None:
    manager = _FakeExternalFederationManager(
        {
            "server_id": None,
            "reconciled_servers": 0,
            "refreshed_servers": 0,
            "total_servers": 0,
            "virtual_tools": 0,
            "errors": {},
        }
    )
    module = _FakeExternalFederationModule(manager)
    monkeypatch.setattr(mcp_hub_management, "get_mcp_server", lambda: _FakeMcpServer(module=module), raising=False)
    app = _build_app(
        principal=_make_principal(roles=["admin"], permissions=[]),
        fail_with_401=False,
    )

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/mcp/hub/external-servers/refresh-discovery",
            json={"server_id": "docs", "unexpected": True},
        )

    assert resp.status_code == 422
    assert manager.calls == []


@pytest.mark.asyncio
async def test_refresh_external_server_discovery_rejects_conflicting_query_and_body_server_id(monkeypatch) -> None:
    manager = _FakeExternalFederationManager(
        {
            "server_id": None,
            "reconciled_servers": 0,
            "refreshed_servers": 0,
            "total_servers": 0,
            "virtual_tools": 0,
            "errors": {},
        }
    )
    module = _FakeExternalFederationModule(manager)
    monkeypatch.setattr(mcp_hub_management, "get_mcp_server", lambda: _FakeMcpServer(module=module), raising=False)
    app = _build_app(
        principal=_make_principal(roles=["admin"], permissions=[]),
        fail_with_401=False,
    )

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/mcp/hub/external-servers/refresh-discovery?server_id=query-docs",
            json={"server_id": "body-docs"},
        )

    assert resp.status_code == 422
    assert manager.calls == []


@pytest.mark.asyncio
async def test_refresh_external_server_discovery_rejects_blank_body_server_id(monkeypatch) -> None:
    manager = _FakeExternalFederationManager(
        {
            "server_id": None,
            "reconciled_servers": 0,
            "refreshed_servers": 0,
            "total_servers": 0,
            "virtual_tools": 0,
            "errors": {},
        }
    )
    module = _FakeExternalFederationModule(manager)
    monkeypatch.setattr(mcp_hub_management, "get_mcp_server", lambda: _FakeMcpServer(module=module), raising=False)
    app = _build_app(
        principal=_make_principal(roles=["admin"], permissions=[]),
        fail_with_401=False,
    )

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/mcp/hub/external-servers/refresh-discovery",
            json={"server_id": "   "},
        )

    assert resp.status_code == 422
    assert manager.calls == []


@pytest.mark.asyncio
async def test_refresh_external_server_discovery_returns_503_when_runtime_module_unavailable(monkeypatch) -> None:
    server = _FakeMcpServer(module=None, initialized=True)
    monkeypatch.setattr(mcp_hub_management, "get_mcp_server", lambda: server, raising=False)
    app = _build_app(
        principal=_make_principal(roles=["admin"], permissions=[]),
        fail_with_401=False,
    )

    with TestClient(app) as client:
        resp = client.post("/api/v1/mcp/hub/external-servers/refresh-discovery")

    assert resp.status_code == 503
    payload = resp.json()
    assert payload["detail"]["requires_restart"] is True
    assert "external federation" in payload["detail"]["message"].lower()


@pytest.mark.asyncio
async def test_mcp_hub_events_stream_replays_governance_audit_events(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AUDIT_STORAGE_MODE", "shared")
    monkeypatch.setenv("AUDIT_SHARED_DB_PATH", str(tmp_path / "audit_shared.db"))
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "user_dbs"))
    monkeypatch.setattr(
        mcp_hub_service,
        "_mcp_hub_event_bus",
        mcp_hub_service.McpHubEventBus(max_events=32),
        raising=False,
    )

    from tldw_Server_API.app.api.v1.API_Deps.Audit_DB_Deps import shutdown_all_audit_services

    await shutdown_all_audit_services()
    try:
        event_id = await mcp_hub_service.publish_mcp_hub_event(
            event_type="mcp_hub.external_server.created",
            action="external_server.created",
            actor_id=1,
            resource_type="mcp_external_server",
            resource_id="docs",
            metadata={"owner_scope_type": "team", "owner_scope_id": 7},
        )
        app = _build_app(
            principal=_make_principal(roles=["admin"], permissions=[]),
            fail_with_401=False,
        )

        with TestClient(app) as client:
            resp = client.get(
                "/api/v1/mcp/hub/events/stream",
                params={"replay": "true", "limit": "1", "event_type": "mcp_hub.external_server.created"},
            )
    finally:
        await shutdown_all_audit_services()
        monkeypatch.setattr(mcp_hub_service, "_mcp_hub_event_bus", None, raising=False)

    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers.get("content-type", "")
    assert f"id: {event_id}" in resp.text
    assert "event: mcp_hub.external_server.created" in resp.text
    assert '"resource_type": "mcp_external_server"' in resp.text
    assert '"resource_id": "docs"' in resp.text


@pytest.mark.asyncio
async def test_mcp_hub_event_stream_permission_replay_is_tenant_scoped(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    async def _fake_replay(**kwargs: Any) -> list[dict[str, Any]]:
        captured.update(kwargs)
        return [
            {
                "event_id": "evt_scoped",
                "event_type": "mcp_hub.external_server.created",
                "action": "external_server.created",
                "actor_id": 1,
                "resource_type": "mcp_external_server",
                "resource_id": "docs",
                "metadata": {"owner_scope_type": "global", "owner_scope_id": None},
            }
        ]

    async def _fake_bus() -> mcp_hub_service.McpHubEventBus:
        return mcp_hub_service.McpHubEventBus(max_events=4)

    monkeypatch.setattr(mcp_hub_management, "replay_mcp_hub_audit_events", _fake_replay)
    monkeypatch.setattr(mcp_hub_management, "get_mcp_hub_event_bus", _fake_bus)

    response = await mcp_hub_management.stream_mcp_hub_events(
        request=_NeverDisconnectedRequest(),  # type: ignore[arg-type]
        after_event_id=None,
        event_type=None,
        owner_scope_type=None,
        owner_scope_id=None,
        replay=True,
        limit=1,
        principal=_make_principal(permissions=["system.configure"]),
    )
    chunk = await asyncio.wait_for(anext(response.body_iterator), timeout=0.5)
    text = chunk.decode() if isinstance(chunk, bytes) else str(chunk)

    assert "evt_scoped" in text
    assert captured["principal_user_id"] == 1
    assert captured["user_id"] == "1"
    assert captured["allow_cross_tenant"] is False


@pytest.mark.asyncio
async def test_mcp_hub_event_stream_subscribes_before_replay(monkeypatch) -> None:
    live_event = {
        "event_id": "evt_live_race",
        "event_type": "mcp_hub.external_server.updated",
        "action": "external_server.updated",
        "actor_id": 1,
        "resource_type": "mcp_external_server",
        "resource_id": "docs-race",
        "metadata": {"owner_scope_type": "global", "owner_scope_id": None},
    }

    class _RaceBus(mcp_hub_service.McpHubEventBus):
        async def replay(self, **_kwargs: Any) -> list[dict[str, Any]]:
            await self.publish(live_event)
            return []

    race_bus = _RaceBus(max_events=4)

    async def _fake_bus() -> _RaceBus:
        return race_bus

    async def _fake_replay(**_kwargs: Any) -> list[dict[str, Any]]:
        return []

    monkeypatch.setattr(mcp_hub_management, "get_mcp_hub_event_bus", _fake_bus)
    monkeypatch.setattr(mcp_hub_management, "replay_mcp_hub_audit_events", _fake_replay)

    response = await mcp_hub_management.stream_mcp_hub_events(
        request=_NeverDisconnectedRequest(),  # type: ignore[arg-type]
        after_event_id=None,
        event_type=["mcp_hub.external_server.updated"],
        owner_scope_type=None,
        owner_scope_id=None,
        replay=True,
        limit=1,
        principal=_make_principal(roles=["admin"], permissions=[]),
    )
    chunk = await asyncio.wait_for(anext(response.body_iterator), timeout=0.5)
    text = chunk.decode() if isinstance(chunk, bytes) else str(chunk)

    assert "evt_live_race" in text
    assert "docs-race" in text


def test_mcp_hub_audit_row_normalizes_prefixed_actions() -> None:
    event = mcp_hub_service._audit_row_to_mcp_hub_event(
        {
            "event_id": "evt_1",
            "timestamp": "2026-04-30T00:00:00Z",
            "metadata": json.dumps(
                {
                    "action": "mcp_hub.permission_profile.create",
                    "actor_id": 1,
                    "resource_type": "mcp_permission_profile",
                    "resource_id": "44",
                }
            ),
        }
    )

    assert event is not None
    assert event["event_type"] == "mcp_hub.permission_profile.create"


@pytest.mark.asyncio
async def test_mcp_hub_scoped_audit_metadata_includes_scope_id(monkeypatch) -> None:
    captured: list[dict[str, Any] | None] = []

    async def _capture_emit(**kwargs: Any) -> None:
        captured.append(kwargs.get("metadata"))

    monkeypatch.setattr(mcp_hub_service, "emit_mcp_hub_audit", _capture_emit)
    service = mcp_hub_service.McpHubService(repo=_ScopeAuditRepo())  # type: ignore[arg-type]

    await service.create_permission_profile(
        name="Team Profile",
        owner_scope_type="team",
        owner_scope_id=7,
        mode="custom",
        path_scope_object_id=None,
        policy_document={},
        actor_id=1,
    )

    assert captured == [{"name": "Team Profile", "owner_scope_type": "team", "owner_scope_id": 7}]
    assert mcp_hub_management._event_matches_visible_scope(
        {"metadata": captured[0]},
        [("team", 7)],
    )


@pytest.mark.asyncio
async def test_mcp_hub_durable_audit_replay_survives_event_ring_eviction(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AUDIT_STORAGE_MODE", "shared")
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "user_dbs"))
    monkeypatch.setattr(
        mcp_hub_service,
        "_mcp_hub_event_bus",
        mcp_hub_service.McpHubEventBus(max_events=1),
        raising=False,
    )

    from tldw_Server_API.app.api.v1.API_Deps.Audit_DB_Deps import shutdown_all_audit_services

    await shutdown_all_audit_services()
    try:
        await mcp_hub_service.emit_mcp_hub_audit(
            action="external_server.created",
            actor_id=1,
            resource_type="mcp_external_server",
            resource_id="docs-old",
            metadata={"owner_scope_type": "global", "owner_scope_id": None},
        )
        first_event = (await (await mcp_hub_service.get_mcp_hub_event_bus()).replay(limit=1))[0]

        await mcp_hub_service.emit_mcp_hub_audit(
            action="external_server.updated",
            actor_id=1,
            resource_type="mcp_external_server",
            resource_id="docs-new",
            metadata={"owner_scope_type": "global", "owner_scope_id": None},
        )

        ring_events = await (await mcp_hub_service.get_mcp_hub_event_bus()).replay()
        assert [event["resource_id"] for event in ring_events] == ["docs-new"]

        replayed = await mcp_hub_service.replay_mcp_hub_audit_events(
            principal_user_id=1,
            after_event_id=str(first_event["event_id"]),
            event_types={"mcp_hub.external_server.updated"},
            limit=10,
            allow_cross_tenant=True,
        )
    finally:
        await shutdown_all_audit_services()
        monkeypatch.setattr(mcp_hub_service, "_mcp_hub_event_bus", None, raising=False)

    assert [event["resource_id"] for event in replayed] == ["docs-new"]
    assert replayed[0]["event_type"] == "mcp_hub.external_server.updated"
    assert replayed[0]["source"] == "mcp_hub.audit"


@pytest.mark.asyncio
async def test_mcp_hub_durable_replay_scopes_non_admin_queries(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class _AuditService:
        async def query_events(self, **kwargs: Any) -> list[dict[str, Any]]:
            captured.update(kwargs)
            return [
                {
                    "event_id": "evt_user_1",
                    "timestamp": "2026-04-30T00:00:00Z",
                    "metadata": json.dumps(
                        {
                            "action": "external_server.updated",
                            "actor_id": 1,
                            "resource_type": "mcp_external_server",
                            "resource_id": "docs",
                            "owner_scope_type": "user",
                            "owner_scope_id": 1,
                        }
                    ),
                }
            ]

    async def _fake_audit_service(principal_user_id: int | None) -> _AuditService:
        assert principal_user_id == 1
        return _AuditService()

    monkeypatch.setattr(
        mcp_hub_service,
        "get_or_create_audit_service_for_user_id_optional",
        _fake_audit_service,
    )

    replayed = await mcp_hub_service.replay_mcp_hub_audit_events(principal_user_id=1, limit=10)

    assert captured["user_id"] == "1"
    assert captured["allow_cross_tenant"] is False
    assert replayed[0]["resource_id"] == "docs"


@pytest.mark.asyncio
async def test_mcp_hub_durable_replay_allows_admin_cross_tenant_queries(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class _AuditService:
        async def query_events(self, **kwargs: Any) -> list[dict[str, Any]]:
            captured.update(kwargs)
            return []

    async def _fake_audit_service(principal_user_id: int | None) -> _AuditService:
        assert principal_user_id == 1
        return _AuditService()

    monkeypatch.setattr(
        mcp_hub_service,
        "get_or_create_audit_service_for_user_id_optional",
        _fake_audit_service,
    )

    replayed = await mcp_hub_service.replay_mcp_hub_audit_events(
        principal_user_id=1,
        limit=10,
        allow_cross_tenant=True,
    )

    assert replayed == []
    assert captured["user_id"] is None
    assert captured["allow_cross_tenant"] is True


@pytest.mark.asyncio
async def test_get_mcp_hub_profiles_requires_auth() -> None:
    app = _build_app(principal=None, fail_with_401=True)
    with TestClient(app) as client:
        resp = client.get("/api/v1/mcp/hub/acp-profiles")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_set_external_secret_returns_masked_only() -> None:
    app = _build_app(
        principal=_make_principal(roles=["admin"], permissions=[]),
        fail_with_401=False,
    )
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/mcp/hub/external-servers/docs/secret",
            json={"secret": "abc123secret"},
        )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["secret_configured"] is True
    assert "abc123secret" not in json.dumps(payload)


@pytest.mark.asyncio
async def test_set_external_secret_not_found_maps_to_404() -> None:
    class _MissingService(_FakeService):
        async def set_external_server_secret(self, *, server_id: str, secret_value: str, actor_id: int | None):
            raise ResourceNotFoundError("mcp_external_server", identifier=server_id)

    app = _build_app(
        principal=_make_principal(roles=["admin"], permissions=[]),
        fail_with_401=False,
    )
    app.dependency_overrides[mcp_hub_management.get_mcp_hub_service] = lambda: _MissingService()
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/mcp/hub/external-servers/docs/secret",
            json={"secret": "abc123secret"},
        )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_set_external_secret_bad_request_maps_to_400() -> None:
    class _BadPayloadService(_FakeService):
        async def set_external_server_secret(self, *, server_id: str, secret_value: str, actor_id: int | None):
            raise BadRequestError("Secret value is required")

    app = _build_app(
        principal=_make_principal(roles=["admin"], permissions=[]),
        fail_with_401=False,
    )
    app.dependency_overrides[mcp_hub_management.get_mcp_hub_service] = lambda: _BadPayloadService()
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/mcp/hub/external-servers/docs/secret",
            json={"secret": "abc123secret"},
        )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_list_external_servers_includes_source_state_fields() -> None:
    app = _build_app(
        principal=_make_principal(roles=["admin"], permissions=[]),
        fail_with_401=False,
    )
    with TestClient(app) as client:
        resp = client.get("/api/v1/mcp/hub/external-servers")

    assert resp.status_code == 200
    payload = resp.json()
    assert payload[0]["server_source"] == "managed"
    assert payload[0]["binding_count"] == 2
    assert payload[0]["runtime_executable"] is True
    assert payload[0]["auth_template_present"] is True
    assert payload[0]["auth_template_valid"] is True
    assert payload[0]["auth_template_blocked_reason"] is None


@pytest.mark.asyncio
async def test_import_legacy_external_server_endpoint_returns_managed_row() -> None:
    app = _build_app(
        principal=_make_principal(roles=["admin"], permissions=[]),
        fail_with_401=False,
    )
    with TestClient(app) as client:
        resp = client.post("/api/v1/mcp/hub/external-servers/legacy-docs/import")

    assert resp.status_code == 201
    payload = resp.json()
    assert payload["id"] == "legacy-docs"
    assert payload["server_source"] == "managed"
    assert payload["legacy_source_ref"] == "yaml:legacy-docs"


@pytest.mark.asyncio
async def test_profile_credential_binding_endpoints_round_trip() -> None:
    app = _build_app(
        principal=_make_principal(roles=["admin"], permissions=[]),
        fail_with_401=False,
    )
    with TestClient(app) as client:
        list_resp = client.get("/api/v1/mcp/hub/permission-profiles/7/credential-bindings")
        put_resp = client.put("/api/v1/mcp/hub/permission-profiles/7/credential-bindings/docs")
        delete_resp = client.delete("/api/v1/mcp/hub/permission-profiles/7/credential-bindings/docs")

    assert list_resp.status_code == 200
    assert list_resp.json()[0]["binding_mode"] == "grant"
    assert put_resp.status_code == 200
    assert put_resp.json()["external_server_id"] == "docs"
    assert delete_resp.status_code == 200
    assert delete_resp.json()["ok"] is True


@pytest.mark.asyncio
async def test_profile_slot_write_binding_requires_credential_grant_authority() -> None:
    class _WriteSlotService(_FakeService):
        async def list_external_server_credential_slots(self, *, server_id: str) -> list[dict[str, Any]]:
            assert server_id == "docs"
            return [
                {
                    "server_id": "docs",
                    "slot_name": "token_write",
                    "display_name": "Write token",
                    "secret_kind": "api_key",
                    "privilege_class": "write",
                    "is_required": False,
                    "secret_configured": True,
                }
            ]

        async def upsert_profile_credential_binding(
            self,
            *,
            profile_id: int,
            external_server_id: str,
            slot_name: str | None = None,
            actor_id: int | None,
        ) -> dict[str, Any]:
            assert profile_id == 7
            assert external_server_id == "docs"
            assert slot_name == "token_write"
            assert actor_id == 1
            return {
                "id": 9,
                "binding_target_type": "profile",
                "binding_target_id": "7",
                "external_server_id": "docs",
                "slot_name": "token_write",
                "credential_ref": "slot",
                "binding_mode": "grant",
                "usage_rules": {},
                "created_by": 1,
                "updated_by": 1,
                "created_at": None,
                "updated_at": None,
            }

    app = _build_app(
        principal=_make_principal(permissions=["system.configure", "grant.credentials.read"]),
        fail_with_401=False,
    )
    app.dependency_overrides[mcp_hub_management.get_mcp_hub_service] = lambda: _WriteSlotService()
    with TestClient(app) as client:
        resp = client.put("/api/v1/mcp/hub/permission-profiles/7/credential-bindings/docs/token_write")

    assert resp.status_code == 403
    assert "grant.credentials.write" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_profile_server_binding_uses_default_slot_grant_authority() -> None:
    class _DefaultWriteSlotService(_FakeService):
        async def list_external_server_credential_slots(self, *, server_id: str) -> list[dict[str, Any]]:
            assert server_id == "docs"
            return [
                {
                    "server_id": "docs",
                    "slot_name": "bearer_token",
                    "display_name": "Bearer token",
                    "secret_kind": "bearer_token",
                    "privilege_class": "write",
                    "is_required": True,
                    "secret_configured": True,
                }
            ]

        async def upsert_profile_credential_binding(
            self,
            *,
            profile_id: int,
            external_server_id: str,
            slot_name: str | None = None,
            actor_id: int | None,
        ) -> dict[str, Any]:
            assert profile_id == 7
            assert external_server_id == "docs"
            assert slot_name is None
            assert actor_id == 1
            return {
                "id": 10,
                "binding_target_type": "profile",
                "binding_target_id": "7",
                "external_server_id": "docs",
                "slot_name": "bearer_token",
                "credential_ref": "server",
                "binding_mode": "grant",
                "usage_rules": {},
                "created_by": 1,
                "updated_by": 1,
                "created_at": None,
                "updated_at": None,
            }

    app = _build_app(
        principal=_make_principal(permissions=["system.configure", "grant.credentials.read"]),
        fail_with_401=False,
    )
    app.dependency_overrides[mcp_hub_management.get_mcp_hub_service] = lambda: _DefaultWriteSlotService()
    with TestClient(app) as client:
        resp = client.put("/api/v1/mcp/hub/permission-profiles/7/credential-bindings/docs")

    assert resp.status_code == 403
    assert "grant.credentials.write" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_assignment_credential_binding_and_external_access_endpoints_round_trip() -> None:
    app = _build_app(
        principal=_make_principal(roles=["admin"], permissions=[]),
        fail_with_401=False,
    )
    with TestClient(app) as client:
        list_resp = client.get("/api/v1/mcp/hub/policy-assignments/11/credential-bindings")
        put_resp = client.put(
            "/api/v1/mcp/hub/policy-assignments/11/credential-bindings/docs",
            json={"binding_mode": "disable"},
        )
        preview_resp = client.get("/api/v1/mcp/hub/policy-assignments/11/external-access")

    assert list_resp.status_code == 200
    assert list_resp.json()[0]["binding_mode"] == "disable"
    assert put_resp.status_code == 200
    assert put_resp.json()["binding_mode"] == "disable"
    assert preview_resp.status_code == 200
    assert preview_resp.json()["servers"][0]["blocked_reason"] == "disabled_by_assignment"


@pytest.mark.asyncio
async def test_assignment_disable_does_not_require_credential_grant_authority() -> None:
    app = _build_app(
        principal=_make_principal(permissions=["system.configure"]),
        fail_with_401=False,
    )
    with TestClient(app) as client:
        put_resp = client.put(
            "/api/v1/mcp/hub/policy-assignments/11/credential-bindings/docs/token_write",
            json={"binding_mode": "disable"},
        )

    assert put_resp.status_code == 200
    assert put_resp.json()["binding_mode"] == "disable"


@pytest.mark.asyncio
async def test_external_server_credential_slot_endpoints_round_trip() -> None:
    app = _build_app(
        principal=_make_principal(roles=["admin"], permissions=[]),
        fail_with_401=False,
    )
    with TestClient(app) as client:
        list_resp = client.get("/api/v1/mcp/hub/external-servers/docs/credential-slots")
        create_resp = client.post(
            "/api/v1/mcp/hub/external-servers/docs/credential-slots",
            json={
                "slot_name": "token_readonly",
                "display_name": "Read-only token",
                "secret_kind": "bearer_token",
                "privilege_class": "read",
                "is_required": True,
            },
        )
        update_resp = client.put(
            "/api/v1/mcp/hub/external-servers/docs/credential-slots/token_readonly",
            json={"display_name": "Updated read-only token"},
        )
        set_secret_resp = client.post(
            "/api/v1/mcp/hub/external-servers/docs/credential-slots/token_readonly/secret",
            json={"secret": "abc123secret"},
        )
        clear_secret_resp = client.delete(
            "/api/v1/mcp/hub/external-servers/docs/credential-slots/token_readonly/secret"
        )
        delete_resp = client.delete(
            "/api/v1/mcp/hub/external-servers/docs/credential-slots/token_readonly"
        )

    assert list_resp.status_code == 200
    assert list_resp.json()[0]["slot_name"] == "token_readonly"
    assert create_resp.status_code == 201
    assert create_resp.json()["secret_configured"] is False
    assert update_resp.status_code == 200
    assert update_resp.json()["display_name"] == "Updated read-only token"
    assert set_secret_resp.status_code == 200
    assert set_secret_resp.json()["slot_name"] == "token_readonly"
    assert clear_secret_resp.status_code == 200
    assert clear_secret_resp.json()["ok"] is True
    assert delete_resp.status_code == 200
    assert delete_resp.json()["ok"] is True


@pytest.mark.asyncio
async def test_create_external_server_credential_slot_admin_requires_grant_authority() -> None:
    class _AdminSlotService(_FakeService):
        async def create_external_server_credential_slot(
            self,
            *,
            server_id: str,
            slot_name: str,
            display_name: str,
            secret_kind: str,
            privilege_class: str,
            is_required: bool,
            actor_id: int | None,
        ) -> dict[str, Any]:
            assert server_id == "docs"
            assert slot_name == "token_admin"
            assert privilege_class == "admin"
            assert actor_id == 1
            return {
                "server_id": server_id,
                "slot_name": slot_name,
                "display_name": display_name,
                "secret_kind": secret_kind,
                "privilege_class": privilege_class,
                "is_required": is_required,
                "secret_configured": False,
            }

    app = _build_app(
        principal=_make_principal(permissions=["system.configure"]),
        fail_with_401=False,
    )
    app.dependency_overrides[mcp_hub_management.get_mcp_hub_service] = lambda: _AdminSlotService()
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/mcp/hub/external-servers/docs/credential-slots",
            json={
                "slot_name": "token_admin",
                "display_name": "Admin token",
                "secret_kind": "bearer_token",
                "privilege_class": "admin",
                "is_required": True,
            },
        )

    assert resp.status_code == 403
    assert "grant.credentials.admin" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_update_external_server_credential_slot_privilege_escalation_requires_grant_authority() -> None:
    class _EscalatingSlotService(_FakeService):
        async def list_external_server_credential_slots(self, *, server_id: str) -> list[dict[str, Any]]:
            assert server_id == "docs"
            return [
                {
                    "server_id": "docs",
                    "slot_name": "token_readonly",
                    "display_name": "Read-only token",
                    "secret_kind": "bearer_token",
                    "privilege_class": "read",
                    "is_required": True,
                    "secret_configured": True,
                }
            ]

        async def update_external_server_credential_slot(
            self,
            *,
            server_id: str,
            slot_name: str,
            display_name: str | None = None,
            secret_kind: str | None = None,
            privilege_class: str | None = None,
            is_required: bool | None = None,
            actor_id: int | None,
        ) -> dict[str, Any]:
            assert server_id == "docs"
            assert slot_name == "token_readonly"
            assert privilege_class == "write"
            assert actor_id == 1
            return {
                "server_id": server_id,
                "slot_name": slot_name,
                "display_name": display_name or "Read-only token",
                "secret_kind": secret_kind or "bearer_token",
                "privilege_class": "write",
                "is_required": True if is_required is None else is_required,
                "secret_configured": True,
            }

    app = _build_app(
        principal=_make_principal(permissions=["system.configure"]),
        fail_with_401=False,
    )
    app.dependency_overrides[mcp_hub_management.get_mcp_hub_service] = lambda: _EscalatingSlotService()
    with TestClient(app) as client:
        resp = client.put(
            "/api/v1/mcp/hub/external-servers/docs/credential-slots/token_readonly",
            json={"privilege_class": "write"},
        )

    assert resp.status_code == 403
    assert "grant.credentials.write" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_external_server_auth_template_endpoints_round_trip() -> None:
    app = _build_app(
        principal=_make_principal(roles=["admin"], permissions=[]),
        fail_with_401=False,
    )
    with TestClient(app) as client:
        get_resp = client.get("/api/v1/mcp/hub/external-servers/docs/auth-template")
        put_resp = client.put(
            "/api/v1/mcp/hub/external-servers/docs/auth-template",
            json={
                "mode": "template",
                "mappings": [
                    {
                        "slot_name": "token_readonly",
                        "target_type": "header",
                        "target_name": "Authorization",
                        "prefix": "Bearer ",
                        "suffix": "",
                        "required": True,
                    }
                ],
            },
        )

    assert get_resp.status_code == 200
    assert get_resp.json()["mappings"][0]["target_type"] == "header"
    assert put_resp.status_code == 200
    assert put_resp.json()["mappings"][0]["slot_name"] == "token_readonly"


@pytest.mark.asyncio
async def test_slot_binding_endpoints_round_trip() -> None:
    app = _build_app(
        principal=_make_principal(roles=["admin"], permissions=[]),
        fail_with_401=False,
    )
    with TestClient(app) as client:
        profile_put_resp = client.put(
            "/api/v1/mcp/hub/permission-profiles/7/credential-bindings/docs/token_readonly"
        )
        profile_delete_resp = client.delete(
            "/api/v1/mcp/hub/permission-profiles/7/credential-bindings/docs/token_readonly"
        )
        assignment_put_resp = client.put(
            "/api/v1/mcp/hub/policy-assignments/11/credential-bindings/docs/token_write",
            json={"binding_mode": "disable"},
        )
        assignment_delete_resp = client.delete(
            "/api/v1/mcp/hub/policy-assignments/11/credential-bindings/docs/token_write"
        )
        preview_resp = client.get("/api/v1/mcp/hub/policy-assignments/11/external-access")

    assert profile_put_resp.status_code == 200
    assert profile_put_resp.json()["slot_name"] == "token_readonly"
    assert profile_delete_resp.status_code == 200
    assert profile_delete_resp.json()["ok"] is True
    assert assignment_put_resp.status_code == 200
    assert assignment_put_resp.json()["slot_name"] == "token_write"
    assert assignment_delete_resp.status_code == 200
    assert assignment_delete_resp.json()["ok"] is True
    assert preview_resp.status_code == 200
    assert preview_resp.json()["servers"][0]["slots"][1]["blocked_reason"] == "disabled_by_assignment"


@pytest.mark.asyncio
async def test_slot_status_endpoints_support_action_style_routes() -> None:
    app = _build_app(
        principal=_make_principal(roles=["admin"], permissions=[]),
        fail_with_401=False,
    )
    with TestClient(app) as client:
        profile_status_resp = client.get(
            "/api/v1/mcp/hub/permission-profiles/7/credential-bindings/status/docs/token_readonly"
        )
        assignment_status_resp = client.get(
            "/api/v1/mcp/hub/policy-assignments/11/credential-bindings/status/docs/token_write"
        )

    assert profile_status_resp.status_code == 200
    assert profile_status_resp.json()["binding_target_type"] == "profile"
    assert profile_status_resp.json()["slot_name"] == "token_readonly"
    assert assignment_status_resp.status_code == 200
    assert assignment_status_resp.json()["binding_target_type"] == "assignment"
    assert assignment_status_resp.json()["slot_name"] == "token_write"


@pytest.mark.asyncio
async def test_set_external_secret_alias_bad_request_maps_to_400_for_multislot_server() -> None:
    class _AmbiguousSecretService(_FakeService):
        async def set_external_server_secret(self, *, server_id: str, secret_value: str, actor_id: int | None):
            raise BadRequestError("Server-level secret alias is only valid for default-slot servers")

    app = _build_app(
        principal=_make_principal(roles=["admin"], permissions=[]),
        fail_with_401=False,
    )
    app.dependency_overrides[mcp_hub_management.get_mcp_hub_service] = lambda: _AmbiguousSecretService()
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/mcp/hub/external-servers/docs/secret",
            json={"secret": "abc123secret"},
        )

    assert resp.status_code == 400
