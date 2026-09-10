"""Endpoint-level tests for the /diag/coverage route-list limit (#2890).

The core audit function is covered in test_coverage_audit.py; these tests
exercise the FastAPI layer - query validation bounds and forwarding of the
``limit`` parameter into ``audit_governor_coverage``.
"""

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.rate_limit


async def _init_authnz_sqlite(db_path, monkeypatch) -> None:
    """Point AuthNZ at a throwaway SQLite DB and reset cached pools."""
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    await reset_db_pool()
    reset_settings()
    from tldw_Server_API.app.core.AuthNZ.initialize import (
        ensure_authnz_schema_ready_once,
    )

    await ensure_authnz_schema_ready_once()


async def _create_admin_api_key(*, username: str, email: str) -> str:
    """Create an admin user plus API key and return the raw key value."""
    from uuid import uuid4

    from tldw_Server_API.app.core.AuthNZ.api_key_manager import APIKeyManager
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.AuthNZ.repos.users_repo import AuthnzUsersRepo
    from tldw_Server_API.app.core.DB_Management.Users_DB import UsersDB

    pool = await get_db_pool()
    users_db = UsersDB(pool)
    await users_db.initialize()
    created_user = await users_db.create_user(
        username=username,
        email=email,
        password_hash="x",
        role="admin",
        is_active=True,
        is_superuser=True,
        storage_quota_mb=5120,
        uuid_value=uuid4(),
    )
    user_id = int(created_user["id"])
    await AuthnzUsersRepo(db_pool=pool).assign_role_if_missing(
        user_id=user_id,
        role_name="admin",
    )
    mgr = APIKeyManager(pool)
    await mgr.initialize()
    key_rec = await mgr.create_api_key(user_id=user_id, name=f"{username}-key")
    return str(key_rec["key"])


@pytest.mark.asyncio
async def test_coverage_endpoint_forwards_limit(monkeypatch, tmp_path):
    """limit is forwarded: larger limits return longer (uncapped) lists."""
    db_path = tmp_path / "authnz_rg_coverage_limit.db"
    await _init_authnz_sqlite(db_path, monkeypatch)
    api_key = await _create_admin_api_key(
        username="rg-coverage-admin", email="rg-coverage-admin@example.com"
    )
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.main import app

    with TestClient(app) as client:
        headers = {"X-API-KEY": api_key}

        default_resp = client.get("/api/v1/diag/coverage", headers=headers)
        assert default_resp.status_code == 200
        default_body = default_resp.json()
        assert default_body["route_list_limit"] == 50
        assert len(default_body["unprotected_routes"]) <= 50

        full_resp = client.get(
            "/api/v1/diag/coverage", params={"limit": 5000}, headers=headers
        )
        assert full_resp.status_code == 200
        full_body = full_resp.json()
        assert full_body["route_list_limit"] == 5000
        # With the cap lifted, the lists match the reported totals.
        assert (
            len(full_body["unprotected_routes"])
            == full_body["unprotected_count"]
        )
        assert len(full_body["protected_routes"]) == full_body["protected_count"]


@pytest.mark.asyncio
async def test_coverage_endpoint_rejects_out_of_bounds_limit(
    monkeypatch, tmp_path
):
    """The FastAPI query bounds (1-5000) reject invalid limits."""
    db_path = tmp_path / "authnz_rg_coverage_bounds.db"
    await _init_authnz_sqlite(db_path, monkeypatch)
    api_key = await _create_admin_api_key(
        username="rg-coverage-bounds", email="rg-coverage-bounds@example.com"
    )
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.main import app

    with TestClient(app) as client:
        headers = {"X-API-KEY": api_key}
        for bad_limit in (0, 5001):
            resp = client.get(
                "/api/v1/diag/coverage",
                params={"limit": bad_limit},
                headers=headers,
            )
            assert resp.status_code == 422
