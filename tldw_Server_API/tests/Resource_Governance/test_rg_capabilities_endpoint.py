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


async def _create_admin_user_and_key(*, username: str, email: str) -> str:
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
async def test_rg_capabilities_endpoint_admin(monkeypatch, tmp_path):
    db_path = tmp_path / "authnz_rg_caps_admin.db"
    await _init_authnz_sqlite(db_path, monkeypatch)
    api_key = await _create_admin_user_and_key(username="rg-caps-admin", email="rg-caps-admin@example.com")

    # Ensure lightweight app behavior in tests and memory backend
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("RG_BACKEND", "memory")

    from tldw_Server_API.app.main import app

    # Ensure a governor instance exists even if startup skipped it
    try:
        from tldw_Server_API.app.core.Resource_Governance.governor import MemoryResourceGovernor
        if getattr(app.state, "rg_governor", None) is None:
            app.state.rg_governor = MemoryResourceGovernor()
    except Exception:
        _ = None

    with TestClient(app) as c:
        headers = {"X-API-KEY": api_key}
        r = c.get("/api/v1/resource-governor/diag/capabilities", headers=headers)
        assert r.status_code == 200
        body = r.json()
        assert body.get("status") == "ok"
        caps = body.get("capabilities") or {}
        assert isinstance(caps, dict) and "backend" in caps


@pytest.mark.asyncio
async def test_rg_capabilities_endpoint_redis_stub(monkeypatch, tmp_path):
    db_path = tmp_path / "authnz_rg_caps_redis.db"
    await _init_authnz_sqlite(db_path, monkeypatch)
    api_key = await _create_admin_user_and_key(username="rg-caps-redis", email="rg-caps-redis@example.com")

    # Keep lightweight app; we will inject a Redis governor stub instance
    monkeypatch.setenv("TEST_MODE", "1")
    # Ensure we use the in-memory Redis stub
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.delenv("RG_REAL_REDIS_URL", raising=False)

    from tldw_Server_API.app.main import app

    # Prepare a RedisResourceGovernor using the in-memory Redis stub
    from tldw_Server_API.app.core.Resource_Governance.governor_redis import RedisResourceGovernor

    class _DummyLoader:
        def get_policy(self, _pid):
                     # Allow the diag capabilities request even if RG middleware is active
            # and reuses a cached route_map from earlier tests.
            return {"requests": {"rpm": 120, "burst": 1.0}}

    gov = RedisResourceGovernor(policy_loader=_DummyLoader())

    # Preload tokens Lua so capabilities reflect loaded state
    await gov._ensure_tokens_lua()  # type: ignore[attr-defined]

    with TestClient(app) as c:
        # Override any startup-initialized governor with our Redis instance
        app.state.rg_governor = gov
        headers = {"X-API-KEY": api_key}
        r = c.get("/api/v1/resource-governor/diag/capabilities", headers=headers)
        assert r.status_code == 200
        caps = (r.json().get("capabilities") or {})
        assert caps.get("backend") == "redis"
        # Depending on stub implementation, real_redis may report either False or True;
        # focus on script capability signal for this test.
        assert caps.get("tokens_lua_loaded") is True
