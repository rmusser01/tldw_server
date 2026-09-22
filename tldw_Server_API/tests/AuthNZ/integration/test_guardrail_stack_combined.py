"""
End-to-end guardrail stack checks combining login lockouts and LLM budget enforcement.

This test drives `/api/v1/auth/login` into lockout via AuthGovernor + a stubbed
rate limiter, then exercises the chat completions endpoint with an over-budget
virtual key on the same FastAPI app, ensuring both guardrails operate together.
"""

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.AuthNZ.password_service import PasswordService
from tldw_Server_API.app.core.DB_Management.Users_DB import UsersDB


@pytest.mark.asyncio
async def test_guardrail_stack_login_lockout_and_chat_budget(tmp_path):
    os.environ["AUTH_MODE"] = "multi_user"
    os.environ["JWT_SECRET_KEY"] = "test-secret-key-guardrail-stack-1234567890"
    db_path = tmp_path / "users.db"
    os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"
    os.environ["VIRTUAL_KEYS_ENABLED"] = "true"
    os.environ["LLM_BUDGET_ENFORCE"] = "true"

    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    reset_settings()
    await reset_db_pool()

    pool = await get_db_pool()
    ensure_authnz_tables(Path(pool.db_path))

    password_service = PasswordService()
    password_hash = password_service.hash_password("GuardrailStack!2024")

    users = UsersDB(pool)
    await users.initialize(ensure_schema=False)
    await users.create_user(
        username="guardrail_user",
        email="guardrail@example.com",
        password_hash=password_hash,
        is_active=True,
    )

    user_id = await pool.fetchval("SELECT id FROM users WHERE username = ?", "guardrail_user")

    from tldw_Server_API.app.core.AuthNZ.api_key_manager import APIKeyManager

    mgr = APIKeyManager()
    await mgr.initialize()
    vk = await mgr.create_virtual_key(
        user_id=user_id,
        name="vk-guardrail-stack",
        allowed_endpoints=["chat.completions"],
        budget_day_tokens=0,
    )
    vkey = vk["key"]

    class _StubLimiter:
        def __init__(self, threshold: int = 3) -> None:
            self.enabled = True
            self.threshold = threshold
            self._attempts: dict[str, int] = {}
            self._locked_ids: set[str] = set()

        async def check_lockout(self, identifier: str):
            from datetime import datetime, timedelta

            if identifier in self._locked_ids:
                return True, datetime.utcnow() + timedelta(minutes=15)
            return False, None

        async def record_failed_attempt(self, *, identifier: str, attempt_type: str):
            _ = attempt_type
            count = self._attempts.get(identifier, 0) + 1
            self._attempts[identifier] = count
            is_locked = count >= self.threshold
            if is_locked:
                self._locked_ids.add(identifier)
            remaining = 0 if is_locked else max(self.threshold - count, 0)
            return {
                "attempt_count": count,
                "remaining_attempts": remaining,
                "is_locked": is_locked,
            }

    limiter = _StubLimiter(threshold=3)

    from tldw_Server_API.app.api.v1.API_Deps import auth_deps
    from tldw_Server_API.app.main import app

    async def _get_stub_limiter():
        return limiter

    app.dependency_overrides[auth_deps.get_rate_limiter_dep] = _get_stub_limiter

    try:
        with TestClient(app) as client:
            # Drive login into lockout via AuthGovernor + stub limiter
            for _ in range(3):
                resp = client.post(
                    "/api/v1/auth/login",
                    data={"username": "guardrail_user", "password": "WrongPassword!"},
                )
            assert resp.status_code == 429

            # Drive chat over-budget via virtual key on the same app
            response = client.post(
                "/api/v1/chat/completions",
                headers={"X-API-KEY": vkey, "Content-Type": "application/json"},
                json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
            )
            assert response.status_code == 402
            body = response.json()
            detail = body.get("detail") if isinstance(body.get("detail"), dict) else body
            assert detail.get("error") == "budget_exceeded"
    finally:
        app.dependency_overrides.pop(auth_deps.get_rate_limiter_dep, None)
        try:
            await pool.close()
        except Exception:
            _ = None
        await reset_db_pool()
        reset_settings()
