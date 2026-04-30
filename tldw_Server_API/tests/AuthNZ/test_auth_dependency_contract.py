from __future__ import annotations

import importlib
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import Depends, FastAPI, Request
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
    AdminPrincipal,
    CurrentPrincipal,
    CurrentUserDict,
    RequireApiKeyScope,
    RequirePermission,
    RequireRole,
    ServicePrincipal,
    TokenScopeGuard,
)
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User


pytestmark = pytest.mark.unit


def _public_names(module: object) -> set[str]:
    return {name for name in dir(module) if not name.startswith("_")}


def _cached_user_id(cached_user: object) -> Any:
    if isinstance(cached_user, Mapping):
        return cached_user.get("id")
    return getattr(cached_user, "id", None)


def _principal(
    *,
    kind: str = "user",
    user_id: int | None = 42,
    api_key_id: int | None = None,
    roles: list[str] | None = None,
    permissions: list[str] | None = None,
    subject: str | None = None,
    token_type: str | None = "access",
    is_admin: bool = False,
) -> AuthPrincipal:
    return AuthPrincipal(
        kind=kind,
        user_id=user_id,
        api_key_id=api_key_id,
        username="phase34-user" if kind == "user" else None,
        subject=subject,
        token_type=token_type,
        roles=list(roles or []),
        permissions=list(permissions or []),
        is_admin=is_admin,
        org_ids=[7] if kind == "user" else [],
        team_ids=[11] if kind == "user" else [],
    )


async def _fake_get_auth_principal(request: Request) -> AuthPrincipal:
    principal = request.app.state.principal
    request.state.auth = AuthContext(
        principal=principal,
        ip="127.0.0.1",
        user_agent="phase34-test",
        request_id="phase34-request",
    )
    request.state._auth_user = {
        "id": principal.user_id,
        "roles": list(principal.roles),
        "permissions": list(principal.permissions),
        "is_active": True,
        "is_verified": True,
    }
    request.state.user_id = principal.user_id
    request.state.api_key_id = principal.api_key_id
    request.state.org_ids = list(principal.org_ids)
    request.state.team_ids = list(principal.team_ids)
    if hasattr(request.app.state, "api_key_scope"):
        request.state._api_key_scope = request.app.state.api_key_scope
    return principal


async def _fake_current_active_user() -> dict[str, Any]:
    return {
        "id": 99,
        "roles": ["legacy"],
        "permissions": ["legacy.read"],
        "is_active": True,
        "is_verified": True,
    }


def _build_app(principal: AuthPrincipal) -> FastAPI:
    app = FastAPI()
    app.state.principal = principal
    app.dependency_overrides[auth_deps.get_auth_principal] = _fake_get_auth_principal
    app.dependency_overrides[auth_deps.get_current_active_user] = _fake_current_active_user

    @app.get("/current-principal")
    async def current_principal(request: Request, principal: CurrentPrincipal):
        ctx = request.state.auth
        return {
            "principal_id": principal.principal_id,
            "state_principal_id": ctx.principal.principal_id,
            "user_id": principal.user_id,
            "state_user_id": request.state.user_id,
            "org_ids": request.state.org_ids,
            "team_ids": request.state.team_ids,
        }

    @app.get("/current-user-dict")
    async def current_user_dict(user: CurrentUserDict):
        return user

    @app.get("/admin")
    async def admin(principal: AdminPrincipal):
        return {"principal_id": principal.principal_id}

    @app.get("/service")
    async def service(principal: ServicePrincipal):
        return {"principal_id": principal.principal_id, "kind": principal.kind}

    @app.get("/role")
    async def role(principal: AuthPrincipal = Depends(RequireRole("editor", "admin"))):
        return {"roles": principal.roles}

    @app.get("/permission")
    async def permission(
        principal: AuthPrincipal = Depends(RequirePermission("media.read", "skills.read")),
    ):
        return {"permissions": principal.permissions}

    @app.get("/api-key-scope")
    async def api_key_scope(
        principal: AuthPrincipal = Depends(RequireApiKeyScope("write")),
    ):
        return {"kind": principal.kind, "api_key_id": principal.api_key_id}

    return app


def _build_real_principal_app() -> FastAPI:
    app = FastAPI()

    @app.get("/current-principal")
    async def current_principal(request: Request, principal: CurrentPrincipal):
        auth_ctx = getattr(request.state, "auth", None)
        state_principal = getattr(auth_ctx, "principal", None)
        cached_user = getattr(request.state, "_auth_user", None)
        return {
            "kind": principal.kind,
            "user_id": principal.user_id,
            "api_key_id": principal.api_key_id,
            "subject": principal.subject,
            "token_type": principal.token_type,
            "roles": principal.roles,
            "permissions": principal.permissions,
            "state_principal_id": getattr(state_principal, "principal_id", None),
            "principal_id": principal.principal_id,
            "cached_user_id": _cached_user_id(cached_user),
            "state_user_id": getattr(request.state, "user_id", None),
            "state_api_key_id": getattr(request.state, "api_key_id", None),
        }

    return app


class _FakeJwtService:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def decode_access_token(self, token: str) -> dict[str, Any]:
        assert token == "jwt.header.signature"
        return dict(self.payload)


class _FakeSessionManager:
    async def is_token_blacklisted(self, token: str, jti: str | None = None) -> bool:
        assert token == "jwt.header.signature"
        return False


def _build_token_scope_app(payload: dict[str, Any]) -> FastAPI:
    app = FastAPI()

    async def fake_jwt_service() -> _FakeJwtService:
        return _FakeJwtService(payload)

    async def fake_db_pool() -> object:
        return object()

    app.dependency_overrides[auth_deps.get_jwt_service_dep] = fake_jwt_service
    app.dependency_overrides[auth_deps.get_db_pool] = fake_db_pool

    @app.get("/token-scope")
    async def token_scope(
        request: Request,
        _: None = Depends(TokenScopeGuard("skills.run", allow_admin_bypass=False)),
    ):
        return {
            "scope_enforced": getattr(request.state, "_token_scope_enforced", None),
            "scope_claim": getattr(request.state, "_token_scope_claim", None),
            "scope_required": getattr(request.state, "_token_scope_required", None),
        }

    return app


def test_current_principal_alias_returns_principal_and_preserves_request_state() -> None:
    app = _build_app(
        _principal(roles=["user"], permissions=["media.read"], subject="user:42"),
    )

    response = TestClient(app).get("/current-principal")

    assert response.status_code == 200
    assert response.json() == {
        "principal_id": _principal(subject="user:42").principal_id,
        "state_principal_id": _principal(subject="user:42").principal_id,
        "user_id": 42,
        "state_user_id": 42,
        "org_ids": [7],
        "team_ids": [11],
    }


def test_current_user_dict_alias_honors_legacy_active_user_override() -> None:
    response = TestClient(_build_app(_principal())).get("/current-user-dict")

    assert response.status_code == 200
    assert response.json() == {
        "id": 99,
        "roles": ["legacy"],
        "permissions": ["legacy.read"],
        "is_active": True,
        "is_verified": True,
    }


def test_admin_principal_alias_preserves_admin_role_gate() -> None:
    admin_response = TestClient(
        _build_app(_principal(roles=["admin"], is_admin=True)),
    ).get("/admin")
    user_response = TestClient(
        _build_app(_principal(roles=["user"], permissions=[])),
    ).get("/admin")

    assert admin_response.status_code == 200
    assert user_response.status_code == 403


def test_service_principal_alias_preserves_service_gate() -> None:
    service_response = TestClient(
        _build_app(_principal(kind="service", user_id=None, subject="service:worker")),
    ).get("/service")
    user_response = TestClient(_build_app(_principal(kind="user"))).get("/service")

    assert service_response.status_code == 200
    assert service_response.json()["kind"] == "service"
    assert user_response.status_code == 403


def test_role_and_permission_factory_aliases_preserve_existing_semantics() -> None:
    role_response = TestClient(_build_app(_principal(roles=["editor"]))).get("/role")
    permission_response = TestClient(
        _build_app(_principal(permissions=["media.read", "skills.read"])),
    ).get("/permission")
    missing_permission_response = TestClient(
        _build_app(_principal(permissions=["media.read"])),
    ).get("/permission")

    assert role_response.status_code == 200
    assert permission_response.status_code == 200
    assert missing_permission_response.status_code == 403


def test_factory_aliases_are_documented_existing_factories() -> None:
    assert RequireRole is auth_deps.require_roles
    assert RequirePermission is auth_deps.require_permissions
    assert RequireApiKeyScope is auth_deps.require_api_key_scope
    assert TokenScopeGuard is auth_deps.require_token_scope


def test_config_admin_router_uses_standard_role_factory_alias() -> None:
    from tldw_Server_API.app.api.v1.endpoints import config_admin

    assert config_admin.RequireRole is auth_deps.RequireRole
    assert not hasattr(config_admin, "require_roles")


def test_metrics_router_uses_standard_role_factory_alias() -> None:
    from tldw_Server_API.app.api.v1.endpoints import metrics

    assert metrics.RequireRole is auth_deps.RequireRole
    assert not hasattr(metrics, "require_roles")


def test_telegram_router_uses_standard_role_factory_alias() -> None:
    from tldw_Server_API.app.api.v1.endpoints import telegram

    assert telegram.RequireRole is auth_deps.RequireRole
    assert not hasattr(telegram, "require_roles")


def test_slack_router_uses_standard_role_factory_alias() -> None:
    from tldw_Server_API.app.api.v1.endpoints import slack

    assert slack.RequireRole is auth_deps.RequireRole
    assert not hasattr(slack, "require_roles")


def test_discord_router_uses_standard_role_factory_alias() -> None:
    from tldw_Server_API.app.api.v1.endpoints import discord

    assert discord.RequireRole is auth_deps.RequireRole
    assert not hasattr(discord, "require_roles")


def test_integrations_control_plane_router_uses_standard_role_factory_alias() -> None:
    from tldw_Server_API.app.api.v1.endpoints import integrations_control_plane

    assert integrations_control_plane.RequireRole is auth_deps.RequireRole
    assert not hasattr(integrations_control_plane, "require_roles")


def test_claims_router_uses_standard_admin_principal_alias() -> None:
    from tldw_Server_API.app.api.v1.endpoints import claims

    assert claims.AdminPrincipal is auth_deps.AdminPrincipal
    assert claims.RequirePermission is auth_deps.RequirePermission
    assert not hasattr(claims, "require_roles")
    assert not hasattr(claims, "require_permissions")


def test_audio_jobs_router_uses_standard_admin_dependency_aliases() -> None:
    from tldw_Server_API.app.api.v1.endpoints.audio import audio_jobs

    assert audio_jobs.RequireRole is auth_deps.RequireRole
    assert audio_jobs.RequirePermission is auth_deps.RequirePermission
    assert not hasattr(audio_jobs, "require_roles")
    assert not hasattr(audio_jobs, "require_permissions")


@pytest.mark.parametrize(
    "module_name",
    [
        "audio_history",
        "audio_streaming",
        "audio_tokenizer",
        "audio_transcriptions",
        "audio_tts",
        "audio_voices",
    ],
)
def test_audio_leaf_routers_use_standard_token_scope_alias(module_name: str) -> None:
    module = importlib.import_module(
        f"tldw_Server_API.app.api.v1.endpoints.audio.{module_name}",
    )

    assert module.TokenScopeGuard is auth_deps.TokenScopeGuard
    assert not hasattr(module, "require_token_scope")


def test_connectors_router_uses_standard_admin_dependency_aliases() -> None:
    from tldw_Server_API.app.api.v1.endpoints import connectors

    assert connectors.RequireRole is auth_deps.RequireRole
    assert connectors.RequirePermission is auth_deps.RequirePermission
    assert not hasattr(connectors, "require_roles")
    assert not hasattr(connectors, "require_permissions")


def test_evaluations_unified_router_uses_standard_role_factory_alias() -> None:
    from tldw_Server_API.app.api.v1.endpoints.evaluations import evaluations_unified

    assert evaluations_unified.RequireRole is auth_deps.RequireRole
    assert not hasattr(evaluations_unified, "require_roles")


@pytest.mark.parametrize(
    "module_name",
    [
        "evaluations_crud",
        "evaluations_embeddings_abtest",
        "evaluations_rag_pipeline",
    ],
)
def test_evaluations_leaf_routers_use_standard_token_scope_alias(module_name: str) -> None:
    module = importlib.import_module(
        f"tldw_Server_API.app.api.v1.endpoints.evaluations.{module_name}",
    )

    assert module.TokenScopeGuard is auth_deps.TokenScopeGuard
    assert not hasattr(module, "require_token_scope")


def test_admin_tools_router_uses_standard_role_factory_alias() -> None:
    from tldw_Server_API.app.api.v1.endpoints.admin import admin_tools

    assert admin_tools.RequireRole is auth_deps.RequireRole
    assert not hasattr(admin_tools, "require_roles")


def test_admin_personalization_router_uses_standard_role_factory_alias() -> None:
    from tldw_Server_API.app.api.v1.endpoints.admin import admin_personalization

    assert admin_personalization.RequireRole is auth_deps.RequireRole
    assert not hasattr(admin_personalization, "require_roles")


def test_admin_byok_router_uses_standard_role_factory_alias() -> None:
    from tldw_Server_API.app.api.v1.endpoints.admin import admin_byok

    assert admin_byok.RequireRole is auth_deps.RequireRole
    assert not hasattr(admin_byok, "require_roles")


def test_admin_package_router_uses_standard_role_factory_alias() -> None:
    from tldw_Server_API.app.api.v1.endpoints import admin

    assert admin.RequireRole is auth_deps.RequireRole
    assert not hasattr(admin, "require_roles")


def test_resource_governor_router_uses_standard_role_factory_alias() -> None:
    from tldw_Server_API.app.api.v1.endpoints import resource_governor

    assert resource_governor.RequireRole is auth_deps.RequireRole
    assert not hasattr(resource_governor, "require_roles")


def test_setup_router_uses_standard_auth_factory_aliases() -> None:
    from tldw_Server_API.app.api.v1.endpoints import setup

    assert setup.RequireRole is auth_deps.RequireRole
    assert setup.RequirePermission is auth_deps.RequirePermission
    assert not hasattr(setup, "require_roles")
    assert not hasattr(setup, "require_permissions")


def test_admin_circuit_breakers_router_uses_standard_permission_factory_alias() -> None:
    from tldw_Server_API.app.api.v1.endpoints.admin import admin_circuit_breakers

    assert admin_circuit_breakers.RequirePermission is auth_deps.RequirePermission
    assert not hasattr(admin_circuit_breakers, "require_permissions")


def test_text2sql_router_uses_standard_permission_factory_alias() -> None:
    from tldw_Server_API.app.api.v1.endpoints import text2sql

    assert text2sql.RequirePermission is auth_deps.RequirePermission
    assert text2sql.TokenScopeGuard is auth_deps.TokenScopeGuard
    assert not hasattr(text2sql, "require_permissions")
    assert not hasattr(text2sql, "require_token_scope")


def test_rag_health_router_uses_standard_permission_factory_alias() -> None:
    from tldw_Server_API.app.api.v1.endpoints import rag_health

    assert rag_health.RequirePermission is auth_deps.RequirePermission
    assert not hasattr(rag_health, "require_permissions")


def test_monitoring_router_uses_standard_permission_factory_alias() -> None:
    from tldw_Server_API.app.api.v1.endpoints import monitoring

    assert monitoring.RequirePermission is auth_deps.RequirePermission
    assert not hasattr(monitoring, "require_permissions")


def test_audit_router_uses_standard_permission_factory_alias() -> None:
    from tldw_Server_API.app.api.v1.endpoints import audit

    assert audit.RequirePermission is auth_deps.RequirePermission
    assert not hasattr(audit, "require_permissions")


def test_scheduled_tasks_control_plane_router_uses_standard_permission_factory_alias() -> None:
    from tldw_Server_API.app.api.v1.endpoints import scheduled_tasks_control_plane

    assert scheduled_tasks_control_plane.RequirePermission is auth_deps.RequirePermission
    assert not hasattr(scheduled_tasks_control_plane, "require_permissions")


def test_reminders_router_uses_standard_permission_factory_alias() -> None:
    from tldw_Server_API.app.api.v1.endpoints import reminders

    assert reminders.RequirePermission is auth_deps.RequirePermission
    assert not hasattr(reminders, "require_permissions")


def test_notifications_router_uses_standard_permission_factory_alias() -> None:
    from tldw_Server_API.app.api.v1.endpoints import notifications

    assert notifications.RequirePermission is auth_deps.RequirePermission
    assert not hasattr(notifications, "require_permissions")


def test_mcp_unified_router_uses_standard_permission_factory_alias() -> None:
    from tldw_Server_API.app.api.v1.endpoints import mcp_unified_endpoint

    assert mcp_unified_endpoint.RequirePermission is auth_deps.RequirePermission
    assert not hasattr(mcp_unified_endpoint, "require_permissions")


def test_billing_router_uses_standard_role_factory_alias() -> None:
    from tldw_Server_API.app.api.v1.endpoints import billing

    assert billing.RequireRole is auth_deps.RequireRole
    assert not hasattr(billing, "require_roles")


def test_scheduler_workflows_router_uses_standard_permission_factory_alias() -> None:
    from tldw_Server_API.app.api.v1.endpoints import scheduler_workflows

    assert scheduler_workflows.RequirePermission is auth_deps.RequirePermission
    assert scheduler_workflows.TokenScopeGuard is auth_deps.TokenScopeGuard
    assert not hasattr(scheduler_workflows, "require_permissions")
    assert not hasattr(scheduler_workflows, "require_token_scope")


def test_tools_router_uses_standard_permission_factory_alias() -> None:
    from tldw_Server_API.app.api.v1.endpoints import tools

    assert tools.RequirePermission is auth_deps.RequirePermission
    assert callable(tools.RequirePermission("tools.execute:*"))
    assert "auth_deps" not in _public_names(tools)
    assert not hasattr(tools, "require_permissions")


@pytest.mark.parametrize(
    "module_name",
    [
        "add",
        "listing",
        "process_web_scraping",
        "process_videos",
        "reprocess",
        "ingest_jobs",
        "item",
    ],
)
def test_media_leaf_routers_use_standard_permission_factory_alias(module_name: str) -> None:
    module = importlib.import_module(
        f"tldw_Server_API.app.api.v1.endpoints.media.{module_name}",
    )

    assert module.RequirePermission is auth_deps.RequirePermission
    assert not hasattr(module, "require_permissions")


def test_media_ingest_web_content_router_uses_standard_token_scope_alias() -> None:
    from tldw_Server_API.app.api.v1.endpoints.media import ingest_web_content

    assert ingest_web_content.TokenScopeGuard is auth_deps.TokenScopeGuard
    assert not hasattr(ingest_web_content, "require_token_scope")


def test_notes_graph_router_uses_standard_permission_factory_alias() -> None:
    from tldw_Server_API.app.api.v1.endpoints import notes_graph

    assert notes_graph.RequirePermission is auth_deps.RequirePermission
    assert notes_graph.TokenScopeGuard is auth_deps.TokenScopeGuard
    assert not hasattr(notes_graph, "require_permissions")
    assert not hasattr(notes_graph, "require_token_scope")


def test_chat_router_uses_standard_permission_factory_alias() -> None:
    from tldw_Server_API.app.api.v1.endpoints import chat

    assert chat.RequirePermission is auth_deps.RequirePermission
    assert chat.TokenScopeGuard is auth_deps.TokenScopeGuard
    assert not hasattr(chat, "require_permissions")
    assert not hasattr(chat, "require_token_scope")


def test_agent_orchestration_router_uses_standard_token_scope_alias() -> None:
    from tldw_Server_API.app.api.v1.endpoints import agent_orchestration

    assert agent_orchestration.TokenScopeGuard is auth_deps.TokenScopeGuard
    assert not hasattr(agent_orchestration, "require_token_scope")


def test_agent_client_protocol_router_uses_standard_token_scope_alias() -> None:
    from tldw_Server_API.app.api.v1.endpoints import agent_client_protocol

    assert agent_client_protocol.TokenScopeGuard is auth_deps.TokenScopeGuard
    assert not hasattr(agent_client_protocol, "require_token_scope")


def test_vector_stores_router_uses_standard_auth_factory_aliases() -> None:
    from tldw_Server_API.app.api.v1.endpoints import vector_stores_openai

    assert vector_stores_openai.RequireRole is auth_deps.RequireRole
    assert vector_stores_openai.RequirePermission is auth_deps.RequirePermission
    assert not hasattr(vector_stores_openai, "require_roles")
    assert not hasattr(vector_stores_openai, "require_permissions")


def test_embeddings_v5_router_uses_standard_auth_factory_aliases() -> None:
    from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced

    assert embeddings_v5_production_enhanced.RequireRole is auth_deps.RequireRole
    assert embeddings_v5_production_enhanced.RequirePermission is auth_deps.RequirePermission
    assert not hasattr(embeddings_v5_production_enhanced, "require_roles")
    assert not hasattr(embeddings_v5_production_enhanced, "require_permissions")


def test_rag_unified_router_uses_standard_permission_factory_alias() -> None:
    from tldw_Server_API.app.api.v1.endpoints import rag_unified

    assert rag_unified.RequirePermission is auth_deps.RequirePermission
    assert rag_unified.TokenScopeGuard is auth_deps.TokenScopeGuard
    assert not hasattr(rag_unified, "require_permissions")
    assert not hasattr(rag_unified, "require_token_scope")


def test_data_tables_router_uses_standard_permission_factory_alias() -> None:
    from tldw_Server_API.app.api.v1.endpoints import data_tables

    assert data_tables.RequirePermission is auth_deps.RequirePermission
    assert not hasattr(data_tables, "require_permissions")


def test_chat_workflows_router_uses_standard_permission_factory_alias() -> None:
    from tldw_Server_API.app.api.v1.endpoints import chat_workflows

    assert chat_workflows.RequirePermission is auth_deps.RequirePermission
    assert callable(chat_workflows.RequirePermission("chat.workflows.read"))
    assert "auth_deps" not in _public_names(chat_workflows)
    assert not hasattr(chat_workflows, "require_permissions")


def test_sandbox_router_uses_standard_role_factory_alias() -> None:
    from tldw_Server_API.app.api.v1.endpoints import sandbox

    assert sandbox.RequireRole is auth_deps.RequireRole
    assert callable(sandbox.RequireRole("admin"))
    assert "auth_deps" not in _public_names(sandbox)
    assert not hasattr(sandbox, "require_roles")


def test_slides_router_uses_standard_permission_factory_alias() -> None:
    from tldw_Server_API.app.api.v1.endpoints import slides

    assert slides.RequirePermission is auth_deps.RequirePermission
    assert not hasattr(slides, "require_permissions")


def test_workflows_router_uses_standard_auth_factory_aliases() -> None:
    from tldw_Server_API.app.api.v1.endpoints import workflows

    assert workflows.RequireRole is auth_deps.RequireRole
    assert workflows.RequirePermission is auth_deps.RequirePermission
    assert workflows.TokenScopeGuard is auth_deps.TokenScopeGuard
    assert callable(workflows.RequireRole("admin"))
    assert callable(workflows.RequirePermission("workflows.runs.read"))
    assert callable(workflows.TokenScopeGuard("workflows.runs.read"))
    assert "auth_deps" not in _public_names(workflows)
    assert not hasattr(workflows, "require_roles")
    assert not hasattr(workflows, "require_permissions")
    assert not hasattr(workflows, "require_token_scope")


def test_api_key_scope_factory_alias_preserves_jwt_bypass_and_scope_checks() -> None:
    jwt_response = TestClient(_build_app(_principal(kind="user", api_key_id=None))).get(
        "/api-key-scope",
    )

    api_key_app = _build_app(_principal(kind="api_key", api_key_id=123))
    api_key_app.state.api_key_scope = "write"
    api_key_response = TestClient(api_key_app).get("/api-key-scope")

    limited_key_app = _build_app(_principal(kind="api_key", api_key_id=456))
    limited_key_app.state.api_key_scope = "read"
    limited_key_response = TestClient(limited_key_app).get("/api-key-scope")

    assert jwt_response.status_code == 200
    assert jwt_response.json() == {"kind": "user", "api_key_id": None}
    assert api_key_response.status_code == 200
    assert api_key_response.json() == {"kind": "api_key", "api_key_id": 123}
    assert limited_key_response.status_code == 403
    assert "API key lacks required scope" in limited_key_response.json()["detail"]


def test_token_scope_guard_alias_preserves_scoped_jwt_success_and_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.AuthNZ import session_manager

    async def fake_session_manager() -> _FakeSessionManager:
        return _FakeSessionManager()

    monkeypatch.setattr(session_manager, "get_session_manager", fake_session_manager)

    scoped_response = TestClient(
        _build_token_scope_app({"scope": "skills.run", "jti": "token-1"}),
    ).get(
        "/token-scope",
        headers={"Authorization": "Bearer jwt.header.signature"},
    )
    invalid_scope_response = TestClient(
        _build_token_scope_app({"scope": "wrong.scope", "jti": "token-2"}),
    ).get(
        "/token-scope",
        headers={"Authorization": "Bearer jwt.header.signature"},
    )
    missing_credentials_response = TestClient(
        _build_token_scope_app({"scope": "skills.run", "jti": "token-3"}),
    ).get("/token-scope")

    assert scoped_response.status_code == 200
    assert scoped_response.json() == {
        "scope_enforced": True,
        "scope_claim": "skills.run",
        "scope_required": "skills.run",
    }
    assert invalid_scope_response.status_code == 403
    assert invalid_scope_response.json()["detail"] == "Forbidden: invalid token scope"
    assert missing_credentials_response.status_code == 401
    assert missing_credentials_response.json()["detail"] == "Authentication required"
    assert missing_credentials_response.headers.get("WWW-Authenticate") == "Bearer"


def test_current_principal_alias_preserves_missing_credentials_401() -> None:
    response = TestClient(_build_real_principal_app()).get("/current-principal")

    assert response.status_code == 401
    assert response.json()["detail"] == "Not authenticated (provide Bearer token or X-API-KEY)"
    assert response.headers.get("WWW-Authenticate") == "Bearer"


def test_current_principal_alias_populates_state_for_single_user_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.AuthNZ.settings import get_settings, reset_settings

    with monkeypatch.context() as env:
        env.setenv("AUTH_MODE", "single_user")
        env.setenv("SINGLE_USER_API_KEY", "phase34-single-user-key")
        reset_settings()
        settings = get_settings()

        response = TestClient(_build_real_principal_app()).get(
            "/current-principal",
            headers={"X-API-KEY": "phase34-single-user-key"},
        )

    reset_settings()

    assert response.status_code == 200
    payload = response.json()
    assert payload["kind"] == "user"
    assert payload["user_id"] == settings.SINGLE_USER_FIXED_ID
    assert payload["api_key_id"] is None
    assert payload["subject"] == "single_user"
    assert payload["token_type"] == "api_key"
    assert "admin" in payload["roles"]
    assert payload["permissions"]
    assert payload["principal_id"] == payload["state_principal_id"]
    assert payload["cached_user_id"] == settings.SINGLE_USER_FIXED_ID
    assert payload["state_user_id"] == settings.SINGLE_USER_FIXED_ID
    assert payload["state_api_key_id"] is None


def test_current_principal_alias_populates_state_for_multi_user_jwt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.AuthNZ import User_DB_Handling as udh
    from tldw_Server_API.app.core.AuthNZ import auth_principal_resolver as resolver

    async def fake_verify_jwt_and_fetch_user(request: Request, token: str) -> User:
        assert token == "jwt.header.signature"
        request.state.user_id = 17
        request.state.org_ids = [3]
        request.state.team_ids = [5]
        return User(
            id=17,
            username="jwt-user",
            roles=["editor"],
            permissions=["skills.read"],
        )

    monkeypatch.setattr(
        resolver,
        "get_settings",
        lambda: SimpleNamespace(AUTH_MODE="multi_user", PII_REDACT_LOGS=False),
    )
    monkeypatch.setattr(udh, "verify_jwt_and_fetch_user", fake_verify_jwt_and_fetch_user)

    response = TestClient(_build_real_principal_app()).get(
        "/current-principal",
        headers={"Authorization": "Bearer jwt.header.signature"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["kind"] == "user"
    assert payload["user_id"] == 17
    assert payload["api_key_id"] is None
    assert set(payload["roles"]) == {"editor", "user"}
    assert payload["permissions"] == ["skills.read"]
    assert payload["principal_id"] == payload["state_principal_id"]
    assert payload["cached_user_id"] == 17
    assert payload["state_user_id"] == 17
    assert payload["state_api_key_id"] is None


def test_current_principal_alias_populates_state_for_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.AuthNZ import User_DB_Handling as udh
    from tldw_Server_API.app.core.AuthNZ import auth_principal_resolver as resolver

    async def fake_authenticate_api_key_user(request: Request, api_key: str) -> User:
        assert api_key == "test-api-key"
        request.state.user_id = 33
        request.state.api_key_id = 123
        request.state.org_ids = [7]
        request.state.team_ids = [11]
        return User(
            id=33,
            username="api-key-user",
            roles=["automation"],
            permissions=["skills.read", "skills.write"],
        )

    monkeypatch.setattr(
        resolver,
        "get_settings",
        lambda: SimpleNamespace(AUTH_MODE="multi_user", PII_REDACT_LOGS=False),
    )
    monkeypatch.setattr(udh, "authenticate_api_key_user", fake_authenticate_api_key_user)

    response = TestClient(_build_real_principal_app()).get(
        "/current-principal",
        headers={"X-API-KEY": "test-api-key"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["kind"] == "api_key"
    assert payload["user_id"] == 33
    assert payload["api_key_id"] == 123
    assert set(payload["roles"]) == {"automation", "user"}
    assert payload["permissions"] == ["skills.read", "skills.write"]
    assert payload["principal_id"] == payload["state_principal_id"]
    assert payload["cached_user_id"] == 33
    assert payload["state_user_id"] == 33
    assert payload["state_api_key_id"] == 123
