import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.Resource_Governance import MemoryResourceGovernor
from tldw_Server_API.app.core.Resource_Governance.policy_loader import PolicyLoader, PolicyReloadConfig


pytestmark = pytest.mark.rate_limit


@pytest.mark.asyncio
async def test_diag_peek_with_policy_id(monkeypatch, tmp_path):
    async def _init_authnz_sqlite(db_path) -> None:
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

    db_path = tmp_path / "authnz_rg_diag_peek.db"
    await _init_authnz_sqlite(db_path)
    api_key = await _create_admin_user_and_key(username="rg-diag-admin", email="rg-diag-admin@example.com")

    # Run app in minimal mode.
    monkeypatch.setenv("MINIMAL_TEST_APP", "1")

    # Minimal policy file (not strictly required for this test)
    yaml_path = tmp_path / "rg.yaml"
    yaml_path.write_text(
        (
            "version: 1\n"
            "policies:\n"
            "  chat.default:\n"
            "    requests: { rpm: 2 }\n"
            "    tokens: { per_min: 5 }\n"
            "    scopes: [user]\n"
        ),
        encoding="utf-8",
    )
    loader = PolicyLoader(str(yaml_path), PolicyReloadConfig(enabled=False))
    await loader.load_once()

    pols = {"chat.default": {"requests": {"rpm": 2}, "tokens": {"per_min": 5}, "scopes": ["user"]}}
    gov = MemoryResourceGovernor(policies=pols)

    from tldw_Server_API.app.main import app as main_app
    # Attach governor (loader not required for peek_with_policy here)
    main_app.state.rg_governor = gov

    with TestClient(main_app) as c:
        r = c.get(
            "/api/v1/resource-governor/diag/peek",
            params={"entity": "user:diag", "categories": "requests,tokens", "policy_id": "chat.default"},
            headers={"X-API-KEY": api_key},
        )
        assert r.status_code == 200
        data = r.json()
        assert data.get("status") == "ok"
        d = data.get("data") or {}
        assert "requests" in d and "tokens" in d
