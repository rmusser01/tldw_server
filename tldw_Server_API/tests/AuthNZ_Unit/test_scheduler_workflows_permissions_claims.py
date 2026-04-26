from types import SimpleNamespace
from typing import Any, Dict

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.api.v1.endpoints import scheduler_workflows as sched_mod
from tldw_Server_API.app.core.AuthNZ.permissions import WORKFLOWS_ADMIN
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal


def _build_app_with_overrides(principal: AuthPrincipal) -> FastAPI:
    app = FastAPI()
    app.include_router(sched_mod.router)

    async def _fake_get_auth_principal(request: Request) -> AuthPrincipal:  # type: ignore[override]
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

    app.dependency_overrides[auth_deps.get_auth_principal] = _fake_get_auth_principal

    async def _fake_get_request_user():
        return SimpleNamespace(
            id=1,
            username="wf-admin",
            is_active=True,
            roles=list(principal.roles),
            permissions=list(principal.permissions),
            is_admin=principal.is_admin,
            tenant_id="default",
        )

    app.dependency_overrides[sched_mod.get_request_user] = _fake_get_request_user

    async def _fake_require_token_scope():
        return None

    app.dependency_overrides[sched_mod._ADMIN_RESCAN_SCOPE_DEP] = _fake_require_token_scope

    async def _allow_non_authz_dep() -> None:
        # Claim tests isolate scheduler claim checks and bypass unrelated
        # per-route token-scope/rate-limit enforcement dependencies.
        return None

    for route in app.routes:
        dependant = getattr(route, "dependant", None)
        if dependant is None:
            continue
        for dep in getattr(dependant, "dependencies", []):
            call = getattr(dep, "call", None)
            if call is None:
                continue
            if getattr(call, "_tldw_token_scope", False):
                app.dependency_overrides[call] = _allow_non_authz_dep
            if getattr(call, "_tldw_rate_limit_resource", None) is not None:
                app.dependency_overrides[call] = _allow_non_authz_dep

    class _FakeScheduler:
        def __init__(self) -> None:
            self.calls: Dict[str, Any] = {}
            self._aps = SimpleNamespace(get_jobs=lambda: [])
            self._schedule = SimpleNamespace(
                id="sched-1",
                workflow_id=123,
                name="test",
                cron="*/15 * * * *",
                timezone="UTC",
                inputs_json="{}",
                run_mode="async",
                validation_mode="block",
                enabled=True,
                tenant_id="default",
                user_id="2",
                concurrency_mode="skip",
                misfire_grace_sec=300,
                coalesce=True,
                require_online=False,
                last_run_at=None,
                next_run_at=None,
                last_status=None,
            )

        async def _rescan_once(self):
            self.calls["rescan"] = True

        def list(self, *, tenant_id: str, user_id: str | None, limit: int, offset: int):
            self.calls["list"] = {
                "tenant_id": tenant_id,
                "user_id": user_id,
                "limit": limit,
                "offset": offset,
            }
            return []

        def get(self, schedule_id: str):
            self.calls["get"] = {"schedule_id": schedule_id}
            if schedule_id == getattr(self._schedule, "id", ""):
                return self._schedule
            return None

    fake_scheduler = _FakeScheduler()

    def _get_workflows_scheduler():
        return fake_scheduler

    app.dependency_overrides[sched_mod.get_workflows_scheduler] = _get_workflows_scheduler

    # Attach for inspection in tests
    app.state._fake_scheduler = fake_scheduler
    return app


def _make_principal(
    *,
    kind: str = "user",
    is_admin: bool = False,
    roles: list[str] | None = None,
    permissions: list[str] | None = None,
) -> AuthPrincipal:
    return AuthPrincipal(
        kind=kind,
        user_id=1,
        api_key_id=None,
        subject=None,
        token_type="access",
        jti=None,
        roles=roles or [],
        permissions=permissions or [],
        is_admin=is_admin,
        org_ids=[],
        team_ids=[],
    )


class _LoggerStub:
    def __init__(self) -> None:
        self.warnings: list[str] = []

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.warnings.append(message.format(*args) if args else message)


@pytest.mark.asyncio
async def test_scheduler_admin_rescan_failure_log_is_sanitized(monkeypatch):
    class _FailingScheduler:
        async def _rescan_once(self) -> None:
            raise RuntimeError("scheduler backend exploded at /private/scheduler.db")

    logger = _LoggerStub()
    monkeypatch.setattr(sched_mod, "logger", logger)
    monkeypatch.setattr(sched_mod, "get_workflows_scheduler", lambda: _FailingScheduler())

    with pytest.raises(sched_mod.HTTPException) as exc_info:
        await sched_mod.admin_rescan(_principal=_make_principal())

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Rescan failed"
    assert logger.warnings == ["Admin rescan failed"]
    warning_text = "\n".join(logger.warnings)
    assert "scheduler backend exploded" not in warning_text
    assert "/private/scheduler.db" not in warning_text


@pytest.mark.asyncio
async def test_scheduler_admin_rescan_forbidden_for_non_admin_without_claims(monkeypatch):
    principal = _make_principal(
        roles=["user"],
        permissions=[],
        is_admin=False,
    )
    app = _build_app_with_overrides(principal)
    fake_scheduler = app.state._fake_scheduler
    monkeypatch.setattr(sched_mod, "get_workflows_scheduler", lambda: fake_scheduler)

    with TestClient(app) as client:
        resp = client.post("/api/v1/scheduler/workflows/admin/rescan")
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_scheduler_admin_rescan_allows_for_admin_principal(monkeypatch):
    principal = _make_principal(
        roles=["admin"],
        permissions=[],
        is_admin=True,
    )
    app = _build_app_with_overrides(principal)
    fake_scheduler = app.state._fake_scheduler
    monkeypatch.setattr(sched_mod, "get_workflows_scheduler", lambda: fake_scheduler)

    with TestClient(app) as client:
        resp = client.post("/api/v1/scheduler/workflows/admin/rescan")
    assert resp.status_code == 200
    assert resp.json().get("ok") is True
    assert getattr(app.state._fake_scheduler, "calls", {}).get("rescan") is True


@pytest.mark.asyncio
async def test_scheduler_admin_rescan_allows_service_admin_principal(monkeypatch):
    principal = _make_principal(
        kind="service",
        roles=["admin"],
        permissions=[],
        is_admin=False,
    )
    app = _build_app_with_overrides(principal)
    fake_scheduler = app.state._fake_scheduler
    monkeypatch.setattr(sched_mod, "get_workflows_scheduler", lambda: fake_scheduler)

    with TestClient(app) as client:
        resp = client.post("/api/v1/scheduler/workflows/admin/rescan")
    assert resp.status_code == 200
    assert resp.json().get("ok") is True
    assert getattr(app.state._fake_scheduler, "calls", {}).get("rescan") is True


@pytest.mark.asyncio
async def test_scheduler_admin_rescan_allows_non_admin_with_workflows_admin_permission(monkeypatch):
    principal = _make_principal(
        roles=["user"],
        permissions=[WORKFLOWS_ADMIN],
        is_admin=False,
    )
    app = _build_app_with_overrides(principal)
    fake_scheduler = app.state._fake_scheduler
    monkeypatch.setattr(sched_mod, "get_workflows_scheduler", lambda: fake_scheduler)

    with TestClient(app) as client:
        resp = client.post("/api/v1/scheduler/workflows/admin/rescan")
    assert resp.status_code == 200
    assert resp.json().get("ok") is True
    assert getattr(app.state._fake_scheduler, "calls", {}).get("rescan") is True


@pytest.mark.asyncio
async def test_scheduler_list_owner_filter_allows_admin_role_claim(monkeypatch):
    principal = _make_principal(
        roles=["admin"],
        permissions=[],
        is_admin=False,
    )
    app = _build_app_with_overrides(principal)
    fake_scheduler = app.state._fake_scheduler
    monkeypatch.setattr(sched_mod, "get_workflows_scheduler", lambda: fake_scheduler)

    with TestClient(app) as client:
        resp = client.get("/api/v1/scheduler/workflows", params={"owner": "2"})
    assert resp.status_code == 200
    assert getattr(app.state._fake_scheduler, "calls", {}).get("list", {}).get("user_id") == "2"


@pytest.mark.asyncio
async def test_scheduler_list_owner_filter_allows_workflows_admin_permission(monkeypatch):
    principal = _make_principal(
        roles=["user"],
        permissions=[WORKFLOWS_ADMIN],
        is_admin=False,
    )
    app = _build_app_with_overrides(principal)
    fake_scheduler = app.state._fake_scheduler
    monkeypatch.setattr(sched_mod, "get_workflows_scheduler", lambda: fake_scheduler)

    with TestClient(app) as client:
        resp = client.get("/api/v1/scheduler/workflows", params={"owner": "2"})
    assert resp.status_code == 200
    assert getattr(app.state._fake_scheduler, "calls", {}).get("list", {}).get("user_id") == "2"


@pytest.mark.asyncio
async def test_scheduler_list_owner_filter_forbidden_without_admin_claims(monkeypatch):
    principal = _make_principal(
        roles=["user"],
        permissions=[],
        is_admin=False,
    )
    app = _build_app_with_overrides(principal)
    fake_scheduler = app.state._fake_scheduler
    monkeypatch.setattr(sched_mod, "get_workflows_scheduler", lambda: fake_scheduler)

    with TestClient(app) as client:
        resp = client.get("/api/v1/scheduler/workflows", params={"owner": "2"})
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_scheduler_list_owner_filter_forbidden_for_is_admin_boolean_without_claims(monkeypatch):
    principal = _make_principal(
        roles=["user"],
        permissions=[],
        is_admin=True,
    )
    app = _build_app_with_overrides(principal)
    fake_scheduler = app.state._fake_scheduler
    monkeypatch.setattr(sched_mod, "get_workflows_scheduler", lambda: fake_scheduler)

    with TestClient(app) as client:
        resp = client.get("/api/v1/scheduler/workflows", params={"owner": "2"})
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_scheduler_get_schedule_allows_admin_role_cross_user(monkeypatch):
    principal = _make_principal(
        roles=["admin"],
        permissions=[],
        is_admin=False,
    )
    app = _build_app_with_overrides(principal)
    fake_scheduler = app.state._fake_scheduler
    monkeypatch.setattr(sched_mod, "get_workflows_scheduler", lambda: fake_scheduler)

    with TestClient(app) as client:
        resp = client.get("/api/v1/scheduler/workflows/sched-1")
    assert resp.status_code == 200
    assert resp.json().get("user_id") == "2"


@pytest.mark.asyncio
async def test_scheduler_get_schedule_forbidden_without_admin_claims(monkeypatch):
    principal = _make_principal(
        roles=["user"],
        permissions=[],
        is_admin=False,
    )
    app = _build_app_with_overrides(principal)
    fake_scheduler = app.state._fake_scheduler
    monkeypatch.setattr(sched_mod, "get_workflows_scheduler", lambda: fake_scheduler)

    with TestClient(app) as client:
        resp = client.get("/api/v1/scheduler/workflows/sched-1")
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_scheduler_get_schedule_forbidden_for_is_admin_boolean_without_claims(monkeypatch):
    principal = _make_principal(
        roles=["worker"],
        permissions=[],
        is_admin=True,
    )
    app = _build_app_with_overrides(principal)
    fake_scheduler = app.state._fake_scheduler
    monkeypatch.setattr(sched_mod, "get_workflows_scheduler", lambda: fake_scheduler)

    with TestClient(app) as client:
        resp = client.get("/api/v1/scheduler/workflows/sched-1")
    assert resp.status_code == 403
