from datetime import datetime, timedelta
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient


async def _execute_membership_fixture_sql(pool, query: str, *args) -> None:
    from tldw_Server_API.app.core.AuthNZ.profile_user_write_guard import (
        _execute_membership_scope_sql,
    )

    async with pool.transaction() as conn:
        await _execute_membership_scope_sql(
            conn,
            query,
            *args,
            backend="postgres",
        )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_admin_endpoints_pg(test_db_pool):
    # App and overrides
    from starlette.requests import Request

    from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
    from tldw_Server_API.app.core.AuthNZ.api_key_manager import APIKeyManager
    from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal

    # Disable CSRF for test client
    from tldw_Server_API.app.core.config import settings as app_settings
    from tldw_Server_API.app.main import app

    app_settings['CSRF_ENABLED'] = False

    # Ensure Postgres pool from fixture
    pool = test_db_pool

    # Ensure org/team/api_keys/usage tables exist
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
    await pool.execute(
        """
        CREATE TABLE IF NOT EXISTS llm_usage_log (
            id SERIAL PRIMARY KEY,
            ts TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
            key_id INTEGER REFERENCES api_keys(id) ON DELETE SET NULL,
            endpoint TEXT,
            operation TEXT,
            provider TEXT,
            model TEXT,
            status INTEGER,
            latency_ms INTEGER,
            prompt_tokens INTEGER,
            completion_tokens INTEGER,
            total_tokens INTEGER,
            prompt_cost_usd DOUBLE PRECISION,
            completion_cost_usd DOUBLE PRECISION,
            total_cost_usd DOUBLE PRECISION,
            currency TEXT DEFAULT 'USD',
            estimated BOOLEAN DEFAULT FALSE,
            request_id TEXT
        )
        """
    )

    # Ensure api_keys and virtual columns via manager
    mgr = APIKeyManager(pool)
    await mgr.initialize()

    # Insert admin user
    await pool.execute(
        "INSERT INTO users (uuid, username, email, password_hash, is_active) VALUES ($1, $2, $3, $4, TRUE)",
        str(uuid4()), "pgadmin", "pgadmin@example.com", "x",
    )
    user_id = await pool.fetchval("SELECT id FROM users WHERE username = $1", "pgadmin")

    # Override AuthPrincipal to treat this user as admin for claim-first gates
    async def _principal_override(request: Request):  # type: ignore[override]
        principal = AuthPrincipal(
            kind="user",
            user_id=user_id,
            api_key_id=None,
            subject="pgadmin",
            token_type="access",
            jti=None,
            roles=["admin"],
            permissions=["system.configure"],
            is_admin=True,
            org_ids=[],
            team_ids=[],
        )
        if request is not None:
            try:
                request.state.auth = AuthContext(
                    principal=principal,
                    ip=None,
                    user_agent=None,
                    request_id=None,
                )
            except Exception:
                # Best-effort; do not fail tests if state attachment fails
                _ = None
        return principal

    app.dependency_overrides[get_auth_principal] = _principal_override

    with TestClient(app) as client:
        # Create org
        r = client.post("/api/v1/admin/orgs", json={"name": "Omega Org"})
        assert r.status_code == 200, r.text
        org = r.json()
        assert org['id'] > 0

        # Create team
        r = client.post(f"/api/v1/admin/orgs/{org['id']}/teams", json={"name": "Ops"})
        assert r.status_code == 200
        team = r.json()
        assert team['name'] == 'Ops'

        # Create virtual key
        r = client.post(
            f"/api/v1/admin/users/{user_id}/virtual-keys",
            json={
                "name": "pg-vk",
                "allowed_endpoints": ["chat.completions"],
                "budget_day_tokens": 300
            }
        )
        assert r.status_code == 200, r.text
        vk = r.json()
        assert 'key' in vk and vk['id'] > 0

        # Create a scoped virtual key tied to org/team
        r = client.post(
            f"/api/v1/admin/users/{user_id}/virtual-keys",
            json={
                "name": "pg-vk-team",
                "allowed_endpoints": ["chat.completions"],
                "budget_day_tokens": 200,
                "org_id": org["id"],
                "team_id": team["id"],
            }
        )
        assert r.status_code == 200, r.text
        vk_scoped = r.json()
        assert 'key' in vk_scoped and vk_scoped['id'] > 0

        base_ts = datetime.utcnow().replace(microsecond=0)
        older_ts = base_ts - timedelta(days=2)
        newer_ts = base_ts - timedelta(days=1)
        await pool.execute(
            "UPDATE api_keys SET status = $1, created_at = $2 WHERE id = $3",
            "revoked",
            older_ts,
            vk["id"],
        )
        await pool.execute(
            "UPDATE api_keys SET created_at = $1 WHERE id = $2",
            newer_ts,
            vk_scoped["id"],
        )

        # List virtual keys
        r = client.get(f"/api/v1/admin/users/{user_id}/virtual-keys")
        assert r.status_code == 200
        arr = r.json()
        assert any(k['id'] == vk['id'] for k in arr)
        assert any(k['id'] == vk_scoped['id'] for k in arr)

        # Filter by name
        r = client.get(f"/api/v1/admin/users/{user_id}/virtual-keys", params={"name": "pg-vk-team"})
        assert r.status_code == 200
        arr = r.json()
        assert len(arr) == 1 and arr[0]["id"] == vk_scoped["id"]

        # Filter by status
        r = client.get(f"/api/v1/admin/users/{user_id}/virtual-keys", params={"status": "revoked"})
        assert r.status_code == 200
        arr = r.json()
        assert len(arr) == 1 and arr[0]["id"] == vk["id"]

        # Filter by org_id/team_id
        r = client.get(f"/api/v1/admin/users/{user_id}/virtual-keys", params={"org_id": org["id"]})
        assert r.status_code == 200
        arr = r.json()
        assert len(arr) == 1 and arr[0]["id"] == vk_scoped["id"]

        r = client.get(f"/api/v1/admin/users/{user_id}/virtual-keys", params={"team_id": team["id"]})
        assert r.status_code == 200
        arr = r.json()
        assert len(arr) == 1 and arr[0]["id"] == vk_scoped["id"]

        # Filter by created_at window
        r = client.get(
            f"/api/v1/admin/users/{user_id}/virtual-keys",
            params={"created_after": (older_ts + timedelta(hours=12)).isoformat() + "Z"},
        )
        assert r.status_code == 200
        arr = r.json()
        assert len(arr) == 1 and arr[0]["id"] == vk_scoped["id"]

        r = client.get(
            f"/api/v1/admin/users/{user_id}/virtual-keys",
            params={"created_before": (older_ts + timedelta(hours=12)).isoformat() + "Z"},
        )
        assert r.status_code == 200
        arr = r.json()
        assert len(arr) == 1 and arr[0]["id"] == vk["id"]

        # Fetch user details via admin endpoint (AuthnzUsersRepo-backed)
        r = client.get(f"/api/v1/admin/users/{user_id}")
        assert r.status_code == 200, r.text
        detail = r.json()
        assert detail.get("id") == user_id
        assert detail.get("username") == "pgadmin"
        assert "password_hash" not in detail

    app.dependency_overrides.pop(get_auth_principal, None)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_org_member_list_pagination_filters_pg(test_db_pool):
    from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
    from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal
    from tldw_Server_API.app.core.config import settings as app_settings
    from tldw_Server_API.app.main import app

    pool = test_db_pool
    app_settings['CSRF_ENABLED'] = False

    # Ensure hierarchy tables exist
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

    # Insert admin user and override principal for claim-first gates
    admin_id = await pool.fetchval(
        "INSERT INTO users (uuid, username, email, password_hash, is_active) VALUES ($1, $2, $3, $4, TRUE) RETURNING id",
        str(uuid4()), "pg-root-admin", "pg-root-admin@example.com", "x",
    )

    async def _principal_override(request=None):  # type: ignore[override]
        principal = AuthPrincipal(
            kind="user",
            user_id=admin_id,
            api_key_id=None,
            subject="pg-root-admin",
            token_type="access",
            jti=None,
            roles=["admin"],
            permissions=["system.configure"],
            is_admin=True,
            org_ids=[],
            team_ids=[],
        )
        if request is not None:
            try:
                request.state.auth = AuthContext(
                    principal=principal,
                    ip=None,
                    user_agent=None,
                    request_id=None,
                )
            except Exception:
                _ = None
        return principal

    app.dependency_overrides[get_auth_principal] = _principal_override

    total_members = 140
    user_ids: list[int] = []
    admin_ids: set[int] = set()
    suspended_ids: set[int] = set()
    lead_invited_ids: set[int] = set()

    with TestClient(app) as client:
        # Create org through API to mimic real flow
        r = client.post("/api/v1/admin/orgs", json={"name": "PG Paginated Org"})
        assert r.status_code == 200, r.text
        org = r.json()
        org_id = org['id']

        base_ts = datetime.utcnow().replace(microsecond=0)
        for idx in range(total_members):
            username = f"pg-member{idx}"
            user_id = await pool.fetchval(
                """
                INSERT INTO users (uuid, username, email, password_hash, is_active)
                VALUES ($1, $2, $3, $4, TRUE)
                RETURNING id
                """,
                str(uuid4()), username, f"{username}@example.com", "x",
            )
            user_ids.append(user_id)

            if idx % 10 == 0:
                role = 'admin'
            elif idx % 7 == 0:
                role = 'lead'
            else:
                role = 'member'

            status = 'suspended' if idx % 17 == 0 else ('invited' if idx % 5 == 0 else 'active')

            if role == 'admin':
                admin_ids.add(user_id)
            if status == 'suspended':
                suspended_ids.add(user_id)
            if role == 'lead' and status == 'invited':
                lead_invited_ids.add(user_id)

            added_at = base_ts + timedelta(seconds=idx)
            await _execute_membership_fixture_sql(
                pool,
                """
                INSERT INTO public.org_members (
                    org_id, user_id, role, status, added_at
                )
                VALUES ($1, $2, $3, $4, $5)
                """,
                org_id, user_id, role, status, added_at,
            )

        expected_order = list(reversed(user_ids))

        r = client.get(f"/api/v1/admin/orgs/{org_id}/members", params={"limit": 30, "offset": 0})
        assert r.status_code == 200, r.text
        first_page = r.json()
        assert len(first_page) == 30
        assert [item['user_id'] for item in first_page] == expected_order[:30]

        r = client.get(f"/api/v1/admin/orgs/{org_id}/members", params={"limit": 40, "offset": 60})
        assert r.status_code == 200, r.text
        mid_page = r.json()
        assert len(mid_page) == 40
        assert [item['user_id'] for item in mid_page] == expected_order[60:100]

        r = client.get(f"/api/v1/admin/orgs/{org_id}/members", params={"role": "admin", "limit": 200})
        assert r.status_code == 200, r.text
        admins = r.json()
        assert all(item['role'] == 'admin' for item in admins)
        assert {item['user_id'] for item in admins} == admin_ids

        r = client.get(f"/api/v1/admin/orgs/{org_id}/members", params={"status": "suspended", "limit": 200})
        assert r.status_code == 200, r.text
        suspended = r.json()
        assert all(item['status'] == 'suspended' for item in suspended)
        assert {item['user_id'] for item in suspended} == suspended_ids

        r = client.get(
            f"/api/v1/admin/orgs/{org_id}/members",
            params={"role": "lead", "status": "invited", "limit": 200},
        )
        assert r.status_code == 200, r.text
        combined = r.json()
        assert all(item['role'] == 'lead' and item['status'] == 'invited' for item in combined)
        assert {item['user_id'] for item in combined} == lead_invited_ids

    app.dependency_overrides.pop(get_auth_principal, None)
