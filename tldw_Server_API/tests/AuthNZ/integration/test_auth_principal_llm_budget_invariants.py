import os
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal
from tldw_Server_API.app.core.DB_Management.Users_DB import UsersDB

pytestmark = pytest.mark.integration


def _install_auth_capture(app: FastAPI) -> tuple[dict[str, Any], Any]:
    """
    Install a lightweight wrapper around get_auth_principal that records
    principal/state alignment for the last request, in the context of LLM
    budget enforcement.
    """
    captured: dict[str, Any] = {}
    original_get_auth_principal = auth_deps.get_auth_principal

    async def _capturing_get_auth_principal(request: Request) -> AuthPrincipal:  # type: ignore[override]
        principal = await original_get_auth_principal(request)

        captured["principal"] = {
            "principal_id": principal.principal_id,
            "kind": principal.kind,
            "user_id": principal.user_id,
            "api_key_id": principal.api_key_id,
            "roles": list(principal.roles),
            "permissions": list(principal.permissions),
            "org_ids": list(principal.org_ids),
            "team_ids": list(principal.team_ids),
        }
        captured["state"] = {
            "user_id": getattr(request.state, "user_id", None),
            "api_key_id": getattr(request.state, "api_key_id", None),
            "org_ids": getattr(request.state, "org_ids", None),
            "team_ids": getattr(request.state, "team_ids", None),
        }

        ctx = getattr(request.state, "auth", None)
        if isinstance(ctx, AuthContext):
            cp = ctx.principal
            captured["state_auth_principal"] = {
                "principal_id": cp.principal_id,
                "kind": cp.kind,
                "user_id": cp.user_id,
                "api_key_id": cp.api_key_id,
                "roles": list(cp.roles),
                "permissions": list(cp.permissions),
                "org_ids": list(cp.org_ids),
                "team_ids": list(cp.team_ids),
            }
        else:
            captured["state_auth_principal"] = None

        return principal

    app.dependency_overrides[auth_deps.get_auth_principal] = _capturing_get_auth_principal
    return captured, original_get_auth_principal


def _restore_auth_capture(app: FastAPI, original_get_auth_principal: Any) -> None:
    try:
        app.dependency_overrides.pop(auth_deps.get_auth_principal, None)
    finally:
        auth_deps.get_auth_principal = original_get_auth_principal  # type: ignore[assignment]


@pytest.mark.asyncio
async def test_llm_budget_guard_overage_preserves_principal_state_alignment(tmp_path):
    """
    Drive enforce_llm_budget via the chat completions endpoint and assert that,
    when the virtual key is over budget (402), the AuthPrincipal and request
    state remain aligned for the request.
    """
    os.environ["AUTH_MODE"] = "multi_user"
    os.environ["JWT_SECRET_KEY"] = "test-secret-key-for-budget-guard-invariants"
    os.environ["VIRTUAL_KEYS_ENABLED"] = "true"
    os.environ["LLM_BUDGET_ENFORCE"] = "true"
    db_path = tmp_path / "users_budget_invariants.db"
    os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"

    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    reset_settings()
    await reset_db_pool()

    pool = await get_db_pool()
    ensure_authnz_tables(Path(pool.db_path))

    # Seed through the canonical writer for profile-visible user fields.
    users = UsersDB(pool)
    await users.initialize(ensure_schema=False)
    await users.create_user(  # nosec B106 # Inert fixture hash, never used for login
        username="budget_guard_invariants_user",
        email="bg_invariants@example.com",
        password_hash="x",
        is_active=True,
    )
    user_id = await pool.fetchval("SELECT id FROM users WHERE username = ?", "budget_guard_invariants_user")

    # Create a virtual key with a zero budget so it is immediately over limit
    from tldw_Server_API.app.core.AuthNZ.api_key_manager import APIKeyManager

    mgr = APIKeyManager()
    await mgr.initialize()
    vk = await mgr.create_virtual_key(
        user_id=user_id,
        name="vk-budget-guard-invariants",
        scope="write",
        allowed_endpoints=["chat.completions"],
        budget_day_tokens=0,
    )
    key_id = vk["id"]
    vkey = vk["key"]

    # The endpoint requires write scope, so verify the canonical key writer stored it.
    assert await pool.fetchval("SELECT scope FROM api_keys WHERE id = ?", key_id) == "write"

    # Remove LLMBudgetMiddleware so the dependency path handles the 402
    from tldw_Server_API.app.core.AuthNZ.llm_budget_middleware import LLMBudgetMiddleware
    from tldw_Server_API.app.main import app

    original_middleware = list(getattr(app, "user_middleware", []))
    app.user_middleware = [m for m in original_middleware if getattr(m, "cls", None) is not LLMBudgetMiddleware]
    app.middleware_stack = app.build_middleware_stack()

    captured, original = _install_auth_capture(app)
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/chat/completions",
                headers={"X-API-KEY": vkey, "Content-Type": "application/json"},
                json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
            )

        assert response.status_code == 402, response.text

        principal = captured.get("principal")
        state = captured.get("state")
        state_auth_principal = captured.get("state_auth_principal")

        assert principal is not None
        assert state is not None
        assert state_auth_principal is not None

        # AuthPrincipal kind/user identity reflects API-key path.
        assert principal["kind"] == "api_key"
        assert principal["user_id"] is not None
        assert principal["api_key_id"] == key_id

        # request.state mirrors principal identity (user_id and api_key_id).
        assert str(state["user_id"]) == str(principal["user_id"])
        assert state["api_key_id"] == key_id

        # request.state.auth.principal mirrors both principal and state.
        assert state_auth_principal["kind"] == principal["kind"]
        assert str(state_auth_principal["user_id"]) == str(principal["user_id"])
        assert state_auth_principal["api_key_id"] == principal["api_key_id"]
        assert state_auth_principal["api_key_id"] == state["api_key_id"]
    finally:
        _restore_auth_capture(app, original)
        app.user_middleware = original_middleware
        app.middleware_stack = app.build_middleware_stack()
        try:
            await pool.close()
        except Exception:
            _ = None
        await reset_db_pool()
        reset_settings()
