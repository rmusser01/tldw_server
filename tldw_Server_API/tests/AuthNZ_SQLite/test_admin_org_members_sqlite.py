import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.mark.asyncio
async def test_admin_org_members_endpoints_sqlite(tmp_path):
    # Configure SQLite
    os.environ['AUTH_MODE'] = 'single_user'
    db_path = tmp_path / 'users.db'
    os.environ['DATABASE_URL'] = f'sqlite:///{db_path}'

    # Reset singletons and init schema
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool, get_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    reset_settings()
    await reset_db_pool()

    pool = await get_db_pool()
    ensure_authnz_tables(Path(pool.db_path))

    # Create users
    async with pool.transaction() as conn:
        await conn.execute(
            "INSERT INTO users (username, email, password_hash, is_active) VALUES (?, ?, ?, 1)",
            ("admin", "admin@example.com", "x"),
        )
        await conn.execute(
            "INSERT INTO users (username, email, password_hash, is_active) VALUES (?, ?, ?, 1)",
            ("bob", "bob@example.com", "x"),
        )
        await conn.execute(
            "INSERT INTO users (username, email, password_hash, is_active) VALUES (?, ?, ?, 1)",
            ("charlie", "charlie@example.com", "x"),
        )
    admin_id = await pool.fetchval("SELECT id FROM users WHERE username = ?", "admin")
    bob_id = await pool.fetchval("SELECT id FROM users WHERE username = ?", "bob")
    charlie_id = await pool.fetchval("SELECT id FROM users WHERE username = ?", "charlie")

    # Prepare app client and override principal to simulate admin claims
    from tldw_Server_API.app.main import app
    from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
    from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal, AuthContext
    from starlette.requests import Request
    from tldw_Server_API.app.core.config import settings as app_settings
    app_settings['CSRF_ENABLED'] = False

    async def _principal_override(request: Request):  # type: ignore[override]
        principal = AuthPrincipal(
            kind="user",
            user_id=admin_id,
            api_key_id=None,
            subject="admin",
            token_type="access",
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
            # Best-effort only; do not fail test setup on state assignment issues
            _ = None
        return principal

    app.dependency_overrides[get_auth_principal] = _principal_override

    with TestClient(app) as client:
        # Create org
        r = client.post("/api/v1/admin/orgs", json={"name": "Gamma Org"})
        assert r.status_code == 200, r.text
        org = r.json()

        # Add member (idempotent)
        r = client.post(f"/api/v1/admin/orgs/{org['id']}/members", json={"user_id": bob_id, "role": "member"})
        assert r.status_code == 200, r.text
        r2 = client.post(f"/api/v1/admin/orgs/{org['id']}/members", json={"user_id": bob_id, "role": "member"})
        assert r2.status_code == 200

        # Default-Base team auto-created and enrollment applied
        team_id = await pool.fetchval(
            "SELECT id FROM teams WHERE org_id = ? AND name = ?",
            (org['id'], "Default-Base"),
        )
        assert team_id is not None
        member_count = await pool.fetchval(
            "SELECT COUNT(*) FROM team_members WHERE team_id = ? AND user_id = ?",
            (team_id, bob_id),
        )
        assert member_count == 1

        # List members (with pagination + filters)
        r = client.get(f"/api/v1/admin/orgs/{org['id']}/members", params={"limit": 100, "offset": 0})
        assert r.status_code == 200
        arr = r.json()
        assert any(m['user_id'] == bob_id for m in arr)

        # Patch role
        r = client.patch(f"/api/v1/admin/orgs/{org['id']}/members/{bob_id}", json={"role": "admin"})
        assert r.status_code == 200
        assert r.json()['role'] == 'admin'

        # Filter by role
        r = client.get(f"/api/v1/admin/orgs/{org['id']}/members", params={"role": "admin"})
        assert r.status_code == 200
        assert any(m['user_id'] == bob_id for m in r.json())

        # Remove member
        r = client.delete(f"/api/v1/admin/orgs/{org['id']}/members/{bob_id}")
        assert r.status_code == 200
        assert "Org member removed" in r.text

        # Removing again returns friendly message
        r = client.delete(f"/api/v1/admin/orgs/{org['id']}/members/{bob_id}")
        assert r.status_code == 200
        assert "No membership found" in r.text

        # Default team membership removed
        remaining = await pool.fetchval(
            "SELECT COUNT(*) FROM team_members WHERE team_id = ? AND user_id = ?",
            (team_id, bob_id),
        )
        assert remaining == 0

        # Add sole owner and ensure enforcement prevents demotion/removal
        r = client.post(
            f"/api/v1/admin/orgs/{org['id']}/members",
            json={"user_id": charlie_id, "role": "owner"},
        )
        assert r.status_code == 200, r.text

        r = client.patch(
            f"/api/v1/admin/orgs/{org['id']}/members/{charlie_id}",
            json={"role": "admin"},
        )
        assert r.status_code == 400
        assert "retain at least one owner" in r.text.lower()

        r = client.delete(f"/api/v1/admin/orgs/{org['id']}/members/{charlie_id}")
        assert r.status_code == 400
        assert "retain at least one owner" in r.text.lower()

    # Cleanup overrides
    app.dependency_overrides.pop(get_auth_principal, None)
