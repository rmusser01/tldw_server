"""Stage 5 (admin perf A): admin-serving reads must be offloaded off the event loop.

Recorder tests monkeypatch the offload primitives — ``run_in_threadpool`` for the
llama.cpp inventory endpoint (matching its assets sibling) and ``asyncio.to_thread``
for the admin system-ops reads (matching sibling admin endpoints) — and assert the
sync service calls route through them instead of running on the event loop.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.api.v1.endpoints import llamacpp as lp
from tldw_Server_API.app.api.v1.endpoints.admin import admin_api_keys as admin_api_keys_mod
from tldw_Server_API.app.api.v1.endpoints.admin import admin_ops as admin_ops_mod
from tldw_Server_API.app.api.v1.schemas.llamacpp_admin_schemas import LlamaCppInventoryResponse
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal


def _admin_principal() -> AuthPrincipal:
    return AuthPrincipal(
        kind="user",
        user_id=1,
        api_key_id=None,
        subject=None,
        token_type="access",
        jti=None,
        roles=["admin"],
        permissions=[],
        is_admin=True,
        org_ids=[],
        team_ids=[],
    )


async def _fake_get_auth_principal(request: Request) -> AuthPrincipal:  # type: ignore[override]
    principal = _admin_principal()
    ip = request.client.host if getattr(request, "client", None) else None
    ua = request.headers.get("User-Agent") if getattr(request, "headers", None) else None
    request_id = request.headers.get("X-Request-ID") if getattr(request, "headers", None) else None
    request.state.auth = AuthContext(
        principal=principal,
        ip=ip,
        user_agent=ua,
        request_id=request_id,
    )
    return principal


async def _fake_check_rate_limit() -> None:
    return


def _make_llamacpp_app() -> FastAPI:
    app = FastAPI()
    app.include_router(lp.router, prefix="/api/v1")
    app.state.llm_manager = object()  # patched services ignore the manager
    app.dependency_overrides[auth_deps.get_auth_principal] = _fake_get_auth_principal
    app.dependency_overrides[auth_deps.check_rate_limit] = _fake_check_rate_limit
    app.dependency_overrides[lp.check_rate_limit] = _fake_check_rate_limit
    return app


def _make_admin_app() -> FastAPI:
    app = FastAPI()
    app.include_router(admin_ops_mod.router, prefix="/api/v1/admin")
    app.include_router(admin_api_keys_mod.router, prefix="/api/v1/admin")
    app.dependency_overrides[auth_deps.get_auth_principal] = _fake_get_auth_principal
    return app


@pytest.mark.unit
def test_llamacpp_inventory_uses_threadpool(monkeypatch: pytest.MonkeyPatch):
    """GET /llamacpp/inventory must run both service calls via run_in_threadpool."""
    calls: list[tuple[Any, tuple, dict]] = []
    real_threadpool = lp.run_in_threadpool
    config_state = {"models_dir": "/tmp/sentinel-models"}

    async def _threadpool_recorder(func, /, *args, **kwargs):
        calls.append((func, args, kwargs))
        return await real_threadpool(func, *args, **kwargs)

    def _fake_get_config_state(llm_manager: Any) -> dict[str, Any]:
        return config_state

    def _fake_scan_inventory(state: Any) -> LlamaCppInventoryResponse:
        return LlamaCppInventoryResponse(models=[], warnings=[], scan_limited=False)

    monkeypatch.setattr(lp, "run_in_threadpool", _threadpool_recorder)
    monkeypatch.setattr(lp.llamacpp_config_service, "get_config_state", _fake_get_config_state)
    monkeypatch.setattr(lp.llamacpp_inventory_service, "scan_inventory", _fake_scan_inventory)

    app = _make_llamacpp_app()
    with TestClient(app) as client:
        response = client.get("/api/v1/llamacpp/inventory")

    assert response.status_code == 200, response.text
    assert response.json()["models"] == []
    assert [call[0] for call in calls] == [_fake_get_config_state, _fake_scan_inventory]
    # The config state produced by the first offloaded call feeds the second one.
    assert calls[0][1] == (app.state.llm_manager,)
    assert calls[1][1] == (config_state,)


@pytest.mark.unit
def test_system_ops_reads_offloaded(monkeypatch: pytest.MonkeyPatch):
    """The admin system-ops reads must wrap their sync service calls in asyncio.to_thread."""
    calls: list[tuple[Any, tuple, dict]] = []
    real_to_thread = asyncio.to_thread

    async def _to_thread_recorder(func, /, *args, **kwargs):
        calls.append((func, args, kwargs))
        return await real_to_thread(func, *args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", _to_thread_recorder)

    async def _no_org_scoping(principal: AuthPrincipal) -> list[int] | None:
        return None

    monkeypatch.setattr(admin_ops_mod, "_get_admin_org_ids", _no_org_scoping)

    def _fake_maintenance_state() -> dict[str, Any]:
        return {"enabled": False, "message": "", "allowlist_user_ids": [], "allowlist_emails": []}

    def _fake_list_feature_flags(**kwargs: Any) -> list[dict[str, Any]]:
        return []

    def _fake_list_incidents(**kwargs: Any) -> tuple[list[dict[str, Any]], int]:
        return [], 0

    def _fake_list_api_key_usage(*, limit: int = 10) -> list[dict[str, Any]]:
        return [
            {
                "key_id": "key-1",
                "request_count": 2,
                "total_tokens": 40,
                "estimated_cost_usd": 0.5,
                "last_used_at": None,
            }
        ]

    monkeypatch.setattr(admin_ops_mod, "svc_get_maintenance_state", _fake_maintenance_state)
    monkeypatch.setattr(admin_ops_mod, "svc_list_feature_flags", _fake_list_feature_flags)
    monkeypatch.setattr(admin_ops_mod, "svc_list_incidents", _fake_list_incidents)
    monkeypatch.setattr(admin_api_keys_mod, "svc_list_api_key_usage", _fake_list_api_key_usage)

    app = _make_admin_app()
    with TestClient(app) as client:
        maintenance_resp = client.get("/api/v1/admin/maintenance")
        flags_resp = client.get("/api/v1/admin/feature-flags")
        incidents_resp = client.get("/api/v1/admin/incidents")
        sla_resp = client.get("/api/v1/admin/incidents/metrics/sla")
        usage_resp = client.get("/api/v1/admin/api-keys/usage/top", params={"limit": 5})

    assert maintenance_resp.status_code == 200, maintenance_resp.text
    assert maintenance_resp.json()["enabled"] is False
    assert flags_resp.status_code == 200, flags_resp.text
    assert flags_resp.json() == {"items": [], "total": 0}
    assert incidents_resp.status_code == 200, incidents_resp.text
    assert incidents_resp.json()["total"] == 0
    assert sla_resp.status_code == 200, sla_resp.text
    assert sla_resp.json()["total_incidents"] == 0
    assert usage_resp.status_code == 200, usage_resp.text
    assert usage_resp.json()["items"][0]["key_id"] == "key-1"

    offloaded_funcs = [call[0] for call in calls]
    assert _fake_maintenance_state in offloaded_funcs
    assert _fake_list_feature_flags in offloaded_funcs
    assert _fake_list_incidents in offloaded_funcs
    assert _fake_list_api_key_usage in offloaded_funcs

    incident_calls = [call for call in calls if call[0] is _fake_list_incidents]
    assert incident_calls == [
        (_fake_list_incidents, (), {"status": None, "severity": None, "tag": None, "limit": 50, "offset": 0}),
        (_fake_list_incidents, (), {"status": None, "severity": None, "tag": None, "limit": 10000, "offset": 0}),
    ]

    usage_calls = [call for call in calls if call[0] is _fake_list_api_key_usage]
    assert usage_calls == [(_fake_list_api_key_usage, (), {"limit": 5})]

    flags_calls = [call for call in calls if call[0] is _fake_list_feature_flags]
    assert flags_calls == [(_fake_list_feature_flags, (), {"scope": None, "org_id": None, "user_id": None})]
