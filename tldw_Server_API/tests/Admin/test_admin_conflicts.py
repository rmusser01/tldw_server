from importlib import import_module
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient
from starlette.requests import Request


pytestmark = pytest.mark.integration


async def _setup_isolated_authnz(monkeypatch, db_path: Path):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("AUTH_MODE", "single_user")
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables

    reset_settings()
    await reset_db_pool()
    ensure_authnz_tables(db_path)


def _admin_app():
    mod = import_module("tldw_Server_API.app.main")
    app = getattr(mod, "app")
    from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
    from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal, AuthContext

    async def _principal_override(request: Request) -> AuthPrincipal:  # type: ignore[override]
        principal = AuthPrincipal(
            kind="user",
            user_id=1,
            api_key_id=None,
            subject="admin",
            token_type="access",  # nosec B106
            jti=None,
            roles=["admin"],
            permissions=["system.configure"],
            is_admin=True,
            org_ids=[],
            team_ids=[],
        )
        try:
            request.state.auth = AuthContext(
                principal=principal,
                ip=None,
                user_agent=None,
                request_id=None,
            )
        except Exception:
            # Best-effort; not all test paths require request.state.auth
            _ = None
        return principal

    app.dependency_overrides[get_auth_principal] = _principal_override
    return app, get_auth_principal


def _reset_app_lifecycle(app) -> None:
    from tldw_Server_API.app.services.app_lifecycle import reset_lifecycle_state

    reset_lifecycle_state(app)


@pytest.mark.asyncio
async def test_admin_create_team_conflict_returns_409(monkeypatch, tmp_path):
    base_dir = tmp_path / "admin_conflict_team"
    base_dir.mkdir(parents=True, exist_ok=True)
    db_path = base_dir / "authnz_admin.db"
    await _setup_isolated_authnz(monkeypatch, db_path)

    app, dep = _admin_app()
    _reset_app_lifecycle(app)
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            # Create org
            r = await client.post("/api/v1/admin/orgs", json={"name": "Org A"})
            assert r.status_code == 200, r.text
            org_id = r.json()["id"]
            # Create team once OK
            r1 = await client.post(f"/api/v1/admin/orgs/{org_id}/teams", json={"name": "Core"})
            assert r1.status_code == 200, r1.text
            # Duplicate create -> 409
            r2 = await client.post(f"/api/v1/admin/orgs/{org_id}/teams", json={"name": "Core"})
            assert r2.status_code == 409, r2.text
    finally:
        app.dependency_overrides.pop(dep, None)


@pytest.mark.asyncio
async def test_admin_create_role_conflict_returns_409(monkeypatch, tmp_path):
    base_dir = tmp_path / "admin_conflict_role"
    base_dir.mkdir(parents=True, exist_ok=True)
    db_path = base_dir / "authnz_admin.db"
    await _setup_isolated_authnz(monkeypatch, db_path)

    app, dep = _admin_app()
    _reset_app_lifecycle(app)
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            r1 = await client.post("/api/v1/admin/roles", json={"name": "analyst"})
            assert r1.status_code == 200, r1.text
            r2 = await client.post("/api/v1/admin/roles", json={"name": "analyst"})
            assert r2.status_code == 409, r2.text
    finally:
        app.dependency_overrides.pop(dep, None)


@pytest.mark.asyncio
async def test_admin_create_permission_conflict_returns_409(monkeypatch, tmp_path):
    base_dir = tmp_path / "admin_conflict_perm"
    base_dir.mkdir(parents=True, exist_ok=True)
    db_path = base_dir / "authnz_admin.db"
    await _setup_isolated_authnz(monkeypatch, db_path)

    app, dep = _admin_app()
    _reset_app_lifecycle(app)
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            body = {"name": "tools.execute:test", "description": "ok", "category": "tools"}
            r1 = await client.post("/api/v1/admin/permissions", json=body)
            assert r1.status_code == 200, r1.text
            r2 = await client.post("/api/v1/admin/permissions", json=body)
            assert r2.status_code == 409, r2.text
    finally:
        app.dependency_overrides.pop(dep, None)
