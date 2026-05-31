from importlib import import_module
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient
from starlette.requests import Request


pytestmark = pytest.mark.integration


async def _setup_isolated_authnz(monkeypatch, db_path: Path):
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("AUTH_MODE", "single_user")

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
async def test_admin_org_slug_conflict_returns_409(monkeypatch, tmp_path):
    base_dir = tmp_path / "admin_edge_org_slug_conflict"
    base_dir.mkdir(parents=True, exist_ok=True)
    db_path = base_dir / "authnz_admin.db"
    await _setup_isolated_authnz(monkeypatch, db_path)

    app, dep = _admin_app()
    _reset_app_lifecycle(app)
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            r1 = await client.post("/api/v1/admin/orgs", json={"name": "Org A", "slug": "acme"})
            assert r1.status_code == 200, r1.text
            r2 = await client.post("/api/v1/admin/orgs", json={"name": "Org B", "slug": "acme"})
            assert r2.status_code == 409, r2.text
    finally:
        # cleanup dependency override
        app.dependency_overrides.pop(dep, None)


@pytest.mark.asyncio
async def test_admin_team_slug_duplicate_allowed_same_org(monkeypatch, tmp_path):
    base_dir = tmp_path / "admin_edge_team_slug_dup"
    base_dir.mkdir(parents=True, exist_ok=True)
    db_path = base_dir / "authnz_admin.db"
    await _setup_isolated_authnz(monkeypatch, db_path)

    app, dep = _admin_app()
    _reset_app_lifecycle(app)
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            r_org = await client.post("/api/v1/admin/orgs", json={"name": "Org A"})
            assert r_org.status_code == 200, r_org.text
            org_id = r_org.json()["id"]

            r1 = await client.post(f"/api/v1/admin/orgs/{org_id}/teams", json={"name": "Team One", "slug": "same"})
            assert r1.status_code == 200, r1.text
            r2 = await client.post(f"/api/v1/admin/orgs/{org_id}/teams", json={"name": "Team Two", "slug": "same"})
            # Slug is not unique for teams; same slug is allowed
            assert r2.status_code == 200, r2.text
    finally:
        app.dependency_overrides.pop(dep, None)


@pytest.mark.asyncio
async def test_admin_team_name_unique_per_org(monkeypatch, tmp_path):
    base_dir = tmp_path / "admin_edge_team_name_per_org"
    base_dir.mkdir(parents=True, exist_ok=True)
    db_path = base_dir / "authnz_admin.db"
    await _setup_isolated_authnz(monkeypatch, db_path)

    app, dep = _admin_app()
    _reset_app_lifecycle(app)
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            r_org1 = await client.post("/api/v1/admin/orgs", json={"name": "Org A"})
            r_org2 = await client.post("/api/v1/admin/orgs", json={"name": "Org B"})
            assert r_org1.status_code == 200 and r_org2.status_code == 200
            org1 = r_org1.json()["id"]
            org2 = r_org2.json()["id"]
            # Same team name in different orgs should be allowed
            r_t1 = await client.post(f"/api/v1/admin/orgs/{org1}/teams", json={"name": "Core"})
            r_t2 = await client.post(f"/api/v1/admin/orgs/{org2}/teams", json={"name": "Core"})
            assert r_t1.status_code == 200, r_t1.text
            assert r_t2.status_code == 200, r_t2.text
    finally:
        app.dependency_overrides.pop(dep, None)


@pytest.mark.asyncio
async def test_case_insensitive_permissions_names(monkeypatch, tmp_path):
    base_dir = tmp_path / "admin_edge_perm_case"
    base_dir.mkdir(parents=True, exist_ok=True)
    db_path = base_dir / "authnz_admin.db"
    await _setup_isolated_authnz(monkeypatch, db_path)

    app, dep = _admin_app()
    _reset_app_lifecycle(app)
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            body1 = {"name": "Tools.Execute:Test", "description": "A", "category": "tools"}
            body2 = {"name": "tools.execute:test", "description": "B", "category": "tools"}
            r1 = await client.post("/api/v1/admin/permissions", json=body1)
            assert r1.status_code == 200, r1.text
            r2 = await client.post("/api/v1/admin/permissions", json=body2)
            # Now case-insensitive uniqueness: second should conflict
            assert r2.status_code == 409, r2.text
    finally:
        app.dependency_overrides.pop(dep, None)


@pytest.mark.asyncio
async def test_case_insensitive_org_and_team_names(monkeypatch, tmp_path):
    base_dir = tmp_path / "admin_edge_org_team_case"
    base_dir.mkdir(parents=True, exist_ok=True)
    db_path = base_dir / "authnz_admin.db"
    await _setup_isolated_authnz(monkeypatch, db_path)

    app, dep = _admin_app()
    _reset_app_lifecycle(app)
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            # Org names: case-insensitive uniqueness -> second conflicts
            r1 = await client.post("/api/v1/admin/orgs", json={"name": "Acme"})
            assert r1.status_code == 200, r1.text
            r2 = await client.post("/api/v1/admin/orgs", json={"name": "acme"})
            assert r2.status_code == 409, r2.text
            org_id = r1.json()["id"]
            # Team names within an org: case-insensitive uniqueness -> second conflicts
            t1 = await client.post(f"/api/v1/admin/orgs/{org_id}/teams", json={"name": "Core"})
            assert t1.status_code == 200, t1.text
            t2 = await client.post(f"/api/v1/admin/orgs/{org_id}/teams", json={"name": "core"})
            assert t2.status_code == 409, t2.text
    finally:
        app.dependency_overrides.pop(dep, None)


@pytest.mark.asyncio
async def test_org_slug_case_insensitive(monkeypatch, tmp_path):
    base_dir = tmp_path / "admin_edge_org_slug_case"
    base_dir.mkdir(parents=True, exist_ok=True)
    db_path = base_dir / "authnz_admin.db"
    await _setup_isolated_authnz(monkeypatch, db_path)

    app, dep = _admin_app()
    _reset_app_lifecycle(app)
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            # Slug unique and case-insensitive: second should conflict
            r1 = await client.post("/api/v1/admin/orgs", json={"name": "Org A", "slug": "acme"})
            assert r1.status_code == 200, r1.text
            r2 = await client.post("/api/v1/admin/orgs", json={"name": "Org B", "slug": "Acme"})
            assert r2.status_code == 409, r2.text
    finally:
        app.dependency_overrides.pop(dep, None)
