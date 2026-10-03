import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.rate_limit


async def _init_authnz_sqlite(db_path, monkeypatch) -> None:
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    try:
        from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool
        from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

        await reset_db_pool()
        reset_settings()
    except Exception:
        _ = None
    try:
        from tldw_Server_API.app.core.AuthNZ.initialize import ensure_authnz_schema_ready_once

        await ensure_authnz_schema_ready_once()
    except Exception:
        _ = None
    # Reset cached workflows daily ledger between tests when DATABASE_URL changes.
    try:
        import tldw_Server_API.app.core.Workflows.daily_ledger as _dl

        _dl._workflows_daily_ledger = None  # type: ignore[attr-defined]
    except Exception:
        _ = None


@pytest.mark.asyncio
async def test_e2e_workflows_daily_cap_denies_with_headers(monkeypatch, tmp_path):
    """A user-level ``limits.workflows_runs_per_day`` override is enforced by the real endpoint."""
    db_path = tmp_path / "authnz_wf_e2e.db"
    await _init_authnz_sqlite(db_path, monkeypatch)

    # Create a fresh user + API key so other tests' runs do not affect the cap.
    from uuid import uuid4
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.DB_Management.Users_DB import UsersDB
    from tldw_Server_API.app.core.AuthNZ.api_key_manager import APIKeyManager
    from tldw_Server_API.app.core.Usage import quota_resolver
    from tldw_Server_API.app.core.UserProfiles.overrides_repo import UserProfileOverridesRepo

    pool = await get_db_pool()
    users_db = UsersDB(pool)
    await users_db.initialize()
    created_user = await users_db.create_user(
        username="wf-cap-user",
        email="wf-cap-user@example.com",
        password_hash="x",
        role="user",
        is_active=True,
        is_superuser=False,
        storage_quota_mb=5120,
        uuid_value=uuid4(),
    )
    user_id = int(created_user["id"])
    mgr = APIKeyManager(pool)
    await mgr.initialize()
    key_rec = await mgr.create_api_key(user_id=user_id, name="wf-cap-key", scope="write")
    api_key = key_rec["key"]

    # A user-level override of exactly one run/day (spec 2 §3: platform-admin plain override).
    overrides = UserProfileOverridesRepo(pool)
    await overrides.ensure_tables()
    await overrides.upsert_override(user_id=user_id, key="limits.workflows_runs_per_day", value=1, updated_by=None)
    quota_resolver.invalidate_user(user_id)

    # Isolate workflows content DB under a temporary user DB base dir so other
    # tests' runs do not pollute this user's daily count.
    user_db_base = tmp_path / "user_dbs"
    monkeypatch.setenv("USER_DB_BASE_DIR", str(user_db_base))

    # Minimal app; the daily cap is enforced by the usage-quotas resolver, not RG.
    monkeypatch.setenv("MINIMAL_TEST_APP", "1")
    monkeypatch.setenv("USAGE_QUOTAS_ENABLED", "1")
    monkeypatch.delenv("WORKFLOWS_DISABLE_QUOTAS", raising=False)

    # Auth is multi-user (API key) and test-mode stability.
    monkeypatch.setenv("TEST_MODE", "true")

    from tldw_Server_API.app.main import app

    try:
        import configparser
        from tldw_Server_API.app.core.DB_Management.DB_Manager import reset_content_backend

        cfg = configparser.ConfigParser()
        cfg["Database"] = {
            "type": "sqlite",
            "workflows_path": str(tmp_path / "workflows.db"),
        }
        reset_content_backend(config=cfg, reload=False)
    except Exception:
        _ = None

    body = {
        "definition": {
            "name": "wf-small",
            "version": 1,
            "steps": [{"id": "log", "type": "log", "config": {"message": "hi"}}],
        },
        "inputs": {},
    }

    with TestClient(app) as c:
        r1 = c.post(
            "/api/v1/workflows/run",
            headers={"X-API-KEY": api_key},
            json=body,
        )
        assert r1.status_code == 200, r1.text

        r2 = c.post(
            "/api/v1/workflows/run",
            headers={"X-API-KEY": api_key},
            json=body,
        )
        assert r2.status_code == 429, r2.text
        assert r2.headers.get("X-RateLimit-Limit") == "1"
        assert r2.headers.get("Retry-After") is not None
