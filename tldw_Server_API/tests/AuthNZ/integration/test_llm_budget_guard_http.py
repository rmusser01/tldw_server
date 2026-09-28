import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.mark.asyncio
async def test_chat_budget_guard_dependency_returns_principal(tmp_path):
    """
    Drive the enforce_llm_budget dependency (not middleware) through a real HTTP request
    and assert the returned 402 includes AuthGovernor principal details.
    """
    os.environ["AUTH_MODE"] = "multi_user"
    os.environ["JWT_SECRET_KEY"] = "test-secret-key-for-budget-guard-1234567890"
    os.environ["VIRTUAL_KEYS_ENABLED"] = "true"
    os.environ["LLM_BUDGET_ENFORCE"] = "true"
    db_path = tmp_path / "users.db"
    os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"

    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool, get_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables

    reset_settings()
    await reset_db_pool()

    pool = await get_db_pool()
    ensure_authnz_tables(Path(pool.db_path))

    # Seeded through UsersDB: profile_user_write_guard rejects raw users writes.
    from tldw_Server_API.tests.helpers.authnz_seed import ensure_test_user

    user_id = await ensure_test_user(pool, "budget_guard_user", "budget_guard@example.com", password_hash="x")

    # Create a virtual key with a zero budget so it is immediately over limit
    from tldw_Server_API.app.core.AuthNZ.api_key_manager import APIKeyManager

    mgr = APIKeyManager()
    await mgr.initialize()
    vk = await mgr.create_virtual_key(
        user_id=user_id,
        name="vk-budget-guard",
        allowed_endpoints=["chat.completions"],
        budget_day_tokens=0,
    )
    key_id = vk["id"]
    vkey = vk["key"]

    # Virtual keys default to read scope, but POST chat completions now
    # requires write scope even under scope="any".
    async with pool.transaction() as conn:
        if getattr(pool, "pool", None):
            await conn.execute("UPDATE api_keys SET scope = $1 WHERE id = $2", "write", key_id)
        else:
            await conn.execute("UPDATE api_keys SET scope = ? WHERE id = ?", ("write", key_id))

    # Remove LLMBudgetMiddleware so the dependency path handles the 402
    from tldw_Server_API.app.main import app
    from tldw_Server_API.app.core.AuthNZ.llm_budget_middleware import LLMBudgetMiddleware

    original_middleware = list(getattr(app, "user_middleware", []))
    app.user_middleware = [m for m in original_middleware if getattr(m, "cls", None) is not LLMBudgetMiddleware]
    app.middleware_stack = app.build_middleware_stack()

    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/chat/completions",
                headers={"X-API-KEY": vkey, "Content-Type": "application/json"},
                json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
            )

        assert response.status_code == 402, response.text
        body = response.json()
        detail = body.get("detail", {})
        assert detail.get("error") == "budget_exceeded"
        principal = (detail.get("details") or {}).get("principal") or {}
        assert principal.get("api_key_id") == key_id
        assert principal.get("user_id") == user_id
        assert principal.get("principal_id")
    finally:
        app.user_middleware = original_middleware
        app.middleware_stack = app.build_middleware_stack()
        try:
            await pool.close()
        except Exception:
            _ = None
        await reset_db_pool()
        reset_settings()
