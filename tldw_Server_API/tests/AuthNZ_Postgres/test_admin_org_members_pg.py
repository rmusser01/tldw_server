import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal
from tldw_Server_API.tests.helpers.authnz_seed import ensure_test_user


@pytest.mark.integration
@pytest.mark.asyncio
async def test_admin_org_members_endpoints_postgres(test_db_pool):
    # App and overrides
    from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal

    # Disable CSRF for test client
    from tldw_Server_API.app.core.config import settings as app_settings
    from tldw_Server_API.app.main import app
    app_settings['CSRF_ENABLED'] = False

    pool = test_db_pool

    # Ensure org tables exist
    await pool.execute(
        """
        CREATE TABLE IF NOT EXISTS organizations (
            id SERIAL PRIMARY KEY,
            uuid VARCHAR(64) UNIQUE,
            name VARCHAR(255) UNIQUE NOT NULL,
            slug VARCHAR(255) UNIQUE,
            owner_user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
            is_active BOOLEAN DEFAULT TRUE,
            metadata JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    await pool.execute("CREATE INDEX IF NOT EXISTS idx_orgs_owner ON organizations(owner_user_id)")
    await pool.execute(
        """
        CREATE TABLE IF NOT EXISTS org_members (
            org_id INTEGER NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            role VARCHAR(32) DEFAULT 'member',
            status VARCHAR(32) DEFAULT 'active',
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (org_id, user_id)
        )
        """
    )
    await pool.execute("CREATE INDEX IF NOT EXISTS idx_org_members_user ON org_members(user_id)")
    await pool.execute(
        """
        CREATE TABLE IF NOT EXISTS teams (
            id SERIAL PRIMARY KEY,
            org_id INTEGER NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
            name VARCHAR(255) NOT NULL,
            slug VARCHAR(255),
            description TEXT,
            is_active BOOLEAN DEFAULT TRUE,
            metadata JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (org_id, name)
        )
        """
    )
    await pool.execute(
        """
        CREATE TABLE IF NOT EXISTS team_members (
            team_id INTEGER NOT NULL REFERENCES teams(id) ON DELETE CASCADE,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            role VARCHAR(32) DEFAULT 'member',
            status VARCHAR(32) DEFAULT 'active',
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (team_id, user_id)
        )
        """
    )

    # Insert admin and standard user
    admin_id = await ensure_test_user(
        pool, "pgadmin2", "pgadmin2@example.com"
    )
    bob_id = await ensure_test_user(
        pool, "pgbob", "pgbob@example.com"
    )
    charlie_id = await ensure_test_user(
        pool, "pgcharlie", "pgcharlie@example.com"
    )

    # Override auth principal with an admin user for claim-first RBAC
    async def _principal_override(request: Request) -> AuthPrincipal:  # type: ignore[override]
        principal = AuthPrincipal(
            kind="user",
            user_id=admin_id,
            api_key_id=None,
            subject="pgadmin2",
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
            # Best-effort; not all code paths require request.state.auth
            _ = None
        return principal

    app.dependency_overrides[get_auth_principal] = _principal_override

    with TestClient(app) as client:
        # Create org
        r = client.post("/api/v1/admin/orgs", json={"name": "Sigma Org"})
        assert r.status_code == 200, r.text
        org = r.json()

        # Add member (idempotent)
        r = client.post(f"/api/v1/admin/orgs/{org['id']}/members", json={"user_id": bob_id, "role": "member"})
        assert r.status_code == 200, r.text
        r2 = client.post(f"/api/v1/admin/orgs/{org['id']}/members", json={"user_id": bob_id, "role": "member"})
        assert r2.status_code == 200

        team_id = await pool.fetchval(
            "SELECT id FROM teams WHERE org_id = $1 AND name = $2",
            org['id'],
            "Default-Base",
        )
        assert team_id is not None
        member_count = await pool.fetchval(
            "SELECT COUNT(*) FROM team_members WHERE team_id = $1 AND user_id = $2",
            team_id,
            bob_id,
        )
        assert member_count == 1

        # User-centric listing
        r = client.get(f"/api/v1/admin/users/{bob_id}/org-memberships")
        assert r.status_code == 200
        assert any(m['org_id'] == org['id'] for m in r.json())

        # List members (filters)
        r = client.get(f"/api/v1/admin/orgs/{org['id']}/members", params={"role": "member"})
        assert r.status_code == 200
        assert any(m['user_id'] == bob_id for m in r.json())

        # Patch role
        r = client.patch(f"/api/v1/admin/orgs/{org['id']}/members/{bob_id}", json={"role": "admin"})
        assert r.status_code == 200
        assert r.json()['role'] == 'admin'

        # Filter by new role
        r = client.get(f"/api/v1/admin/orgs/{org['id']}/members", params={"role": "admin"})
        assert r.status_code == 200
        assert any(m['user_id'] == bob_id for m in r.json())

        # Remove member
        r = client.delete(f"/api/v1/admin/orgs/{org['id']}/members/{bob_id}")
        assert r.status_code == 200
        assert "Org member removed" in r.text
        # Removing again yields friendly message
        r = client.delete(f"/api/v1/admin/orgs/{org['id']}/members/{bob_id}")
        assert r.status_code == 200
        assert "No membership found" in r.text

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

        remaining = await pool.fetchval(
            "SELECT COUNT(*) FROM team_members WHERE team_id = $1 AND user_id = $2",
            team_id,
            bob_id,
        )
        assert remaining == 0

    app.dependency_overrides.pop(get_auth_principal, None)


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("is_verified", [None, False, True], ids=["default", "unverified", "verified"])
async def test_canonical_user_seed_preserves_flags_and_write_guard_pg(test_db_pool, is_verified):
    from tldw_Server_API.app.core.AuthNZ.profile_user_write_guard import ProfileUserWriteRejected

    verification = {} if is_verified is None else {"is_verified": is_verified}
    user_id = await ensure_test_user(
        test_db_pool, "pg_seed_flags", role="admin", **verification
    )
    query = "SELECT role, is_active, is_verified, profile_version FROM users WHERE id = $1"
    row = await test_db_pool.fetchrow(query, user_id)
    assert row["role"] == "admin"
    assert row["is_active"] is True
    assert row["is_verified"] is (False if is_verified is None else is_verified)
    assert row["profile_version"] is not None

    # Reusing a seed must not change the existing user's role or verification.
    assert await ensure_test_user(test_db_pool, "pg_seed_flags") == user_id
    assert dict(await test_db_pool.fetchrow(query, user_id)) == dict(row)

    with pytest.raises(ProfileUserWriteRejected):
        await test_db_pool.execute("UPDATE users SET is_active = FALSE WHERE id = $1", user_id)
    assert dict(await test_db_pool.fetchrow(query, user_id)) == dict(row)
