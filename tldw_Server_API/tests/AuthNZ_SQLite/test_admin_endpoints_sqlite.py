import os
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from loguru import logger


@pytest.mark.asyncio
async def test_admin_endpoints_basic_sqlite(tmp_path):
    # Configure SQLite
    os.environ['AUTH_MODE'] = 'single_user'
    db_path = tmp_path / 'users.db'
    os.environ['DATABASE_URL'] = f'sqlite:///{db_path}'

    # Reset singletons
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool, get_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    reset_settings()
    await reset_db_pool()

    # Disable CSRF for test client
    from tldw_Server_API.app.core.config import settings as app_settings
    app_settings['CSRF_ENABLED'] = False

    pool = await get_db_pool()
    ensure_authnz_tables(Path(pool.db_path))

    # Create a user to satisfy FK and for membership
    async with pool.transaction() as conn:
        await conn.execute(
            "INSERT INTO users (username, email, password_hash, is_active) VALUES (?, ?, ?, 1)",
            ("adminuser", "admin@example.com", "x"),
        )
    user_id = await pool.fetchval("SELECT id FROM users WHERE username = ?", "adminuser")

    # Create TestClient and override admin/principal dependencies to bypass auth
    from tldw_Server_API.app.main import app
    from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
    from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal, AuthContext

    async def _principal_override(request=None):  # type: ignore[override]
        principal = AuthPrincipal(
            kind="user",
            user_id=user_id,
            api_key_id=None,
            subject="adminuser",
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
            except Exception as e:
                logger.debug(f"Could not set request.state.auth in test override: {e}")
        return principal

    app.dependency_overrides[get_auth_principal] = _principal_override

    with TestClient(app) as client:
        # Create org
        r = client.post("/api/v1/admin/orgs", json={"name": "Zeta Org"})
        assert r.status_code == 200, r.text
        org = r.json()
        assert org['id'] > 0 and org['name'] == 'Zeta Org'

        # List orgs
        r = client.get("/api/v1/admin/orgs")
        assert r.status_code == 200
        data = r.json()
        items = data.get("items", [])
        assert isinstance(items, list)
        assert any(o["name"] == "Zeta Org" for o in items)

        # Create a team
        r = client.post(f"/api/v1/admin/orgs/{org['id']}/teams", json={"name": "Infra"})
        assert r.status_code == 200
        team = r.json()
        assert team['name'] == 'Infra'

        # Create a virtual key for admin user with small budget
        r = client.post(
            f"/api/v1/admin/users/{user_id}/virtual-keys",
            json={
                "name": "vk-admin",
                "allowed_endpoints": ["chat.completions"],
                "budget_day_tokens": 500
            }
        )
        assert r.status_code == 200
        vk = r.json()
        assert 'key' in vk and vk['id'] > 0

        # Create a scoped virtual key tied to org/team
        r = client.post(
            f"/api/v1/admin/users/{user_id}/virtual-keys",
            json={
                "name": "vk-team",
                "allowed_endpoints": ["chat.completions"],
                "budget_day_tokens": 300,
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
        # Mark first key revoked and set created_at values for filter testing
        async with pool.transaction() as conn:
            await conn.execute(
                "UPDATE api_keys SET status = ?, created_at = ? WHERE id = ?",
                ("revoked", older_ts.strftime("%Y-%m-%d %H:%M:%S"), vk["id"]),
            )
            await conn.execute(
                "UPDATE api_keys SET created_at = ? WHERE id = ?",
                (newer_ts.strftime("%Y-%m-%d %H:%M:%S"), vk_scoped["id"]),
            )

        # List virtual keys for user
        r = client.get(f"/api/v1/admin/users/{user_id}/virtual-keys")
        assert r.status_code == 200
        arr = r.json()
        assert any(k['id'] == vk['id'] for k in arr)
        assert any(k['id'] == vk_scoped['id'] for k in arr)

        # Filter by name
        r = client.get(f"/api/v1/admin/users/{user_id}/virtual-keys", params={"name": "vk-team"})
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
        assert detail.get("username") == "adminuser"
        assert "password_hash" not in detail

    # Cleanup overrides
    app.dependency_overrides.pop(get_auth_principal, None)


@pytest.mark.asyncio
async def test_org_member_list_pagination_filters_sqlite(tmp_path):
    # Configure SQLite and reset singletons
    os.environ['AUTH_MODE'] = 'single_user'
    db_path = tmp_path / 'users_members.db'
    os.environ['DATABASE_URL'] = f'sqlite:///{db_path}'

    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool, get_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    reset_settings()
    await reset_db_pool()

    pool = await get_db_pool()
    ensure_authnz_tables(Path(pool.db_path))

    # Seed an admin user for auth overrides
    async with pool.transaction() as conn:
        cursor = await conn.execute(
            "INSERT INTO users (username, email, password_hash, is_active) VALUES (?, ?, ?, 1)",
            ("rootadmin", "rootadmin@example.com", "x"),
        )
        admin_id = cursor.lastrowid

    from tldw_Server_API.app.main import app
    from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
    from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal, AuthContext
    from starlette.requests import Request

    async def _principal_override(request: Request):  # type: ignore[override]
        principal = AuthPrincipal(
            kind="user",
            user_id=admin_id,
            api_key_id=None,
            subject="rootadmin",
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
        except Exception as e:
            logger.debug(f"Could not set request.state.auth in test override: {e}")
        return principal

    app.dependency_overrides[get_auth_principal] = _principal_override

    # Build a roster of members with varying roles/statuses and deterministic timestamps
    total_members = 120
    base_ts = datetime.utcnow().replace(microsecond=0)
    user_ids: list[int] = []
    admin_ids: set[int] = set()
    suspended_ids: set[int] = set()
    lead_invited_ids: set[int] = set()

    async with pool.transaction() as conn:
        for idx in range(total_members):
            username = f"member{idx}"
            cursor = await conn.execute(
                "INSERT INTO users (username, email, password_hash, is_active) VALUES (?, ?, ?, 1)",
                (username, f"{username}@example.com", "x"),
            )
            user_id = cursor.lastrowid
            user_ids.append(user_id)

        org_cursor = await conn.execute(
            "INSERT INTO organizations (name, slug, owner_user_id) VALUES (?, ?, ?)",
            ("Paginated Org", "paginated-org", admin_id),
        )
        org_id = org_cursor.lastrowid

        for idx, user_id in enumerate(user_ids):
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
            await conn.execute(
                """
                INSERT INTO org_members (org_id, user_id, role, status, added_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (org_id, user_id, role, status, added_at.isoformat()),
            )

    expected_order = list(reversed(user_ids))

    from tldw_Server_API.app.core.config import settings as app_settings
    app_settings['CSRF_ENABLED'] = False

    with TestClient(app) as client:
        # First page
        r = client.get(f"/api/v1/admin/orgs/{org_id}/members", params={"limit": 25, "offset": 0})
        assert r.status_code == 200, r.text
        payload = r.json()
        assert len(payload) == 25
        assert [item['user_id'] for item in payload] == expected_order[:25]

        # Third page (offset 50, limit 25)
        r = client.get(f"/api/v1/admin/orgs/{org_id}/members", params={"limit": 25, "offset": 50})
        assert r.status_code == 200, r.text
        page = r.json()
        assert len(page) == 25
        assert [item['user_id'] for item in page] == expected_order[50:75]

        # Filter by role=admin
        r = client.get(f"/api/v1/admin/orgs/{org_id}/members", params={"role": "admin", "limit": 200})
        assert r.status_code == 200, r.text
        admins = r.json()
        assert all(item['role'] == 'admin' for item in admins)
        assert {item['user_id'] for item in admins} == admin_ids

        # Filter by status=suspended
        r = client.get(f"/api/v1/admin/orgs/{org_id}/members", params={"status": "suspended", "limit": 200})
        assert r.status_code == 200, r.text
        suspended = r.json()
        assert all(item['status'] == 'suspended' for item in suspended)
        assert {item['user_id'] for item in suspended} == suspended_ids

        # Combined filters: role=lead & status=invited
        r = client.get(
            f"/api/v1/admin/orgs/{org_id}/members",
            params={"role": "lead", "status": "invited", "limit": 200},
        )
        assert r.status_code == 200, r.text
        combined = r.json()
        assert all(item['role'] == 'lead' and item['status'] == 'invited' for item in combined)
        assert {item['user_id'] for item in combined} == lead_invited_ids

    app.dependency_overrides.pop(get_auth_principal, None)
