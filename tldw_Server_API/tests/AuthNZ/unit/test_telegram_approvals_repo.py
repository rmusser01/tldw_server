from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

pytest_plugins = ("tldw_Server_API.tests._plugins.authnz_full_fixtures",)


@pytest.mark.asyncio
async def test_telegram_approvals_repo_persists_and_consumes_pending_approvals(tmp_path, monkeypatch) -> None:
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.repos.telegram_approvals_repo import TelegramApprovalsRepo
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    db_path = tmp_path / "users.db"
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

    reset_settings()
    await reset_db_pool()

    pool = await get_db_pool()
    ensure_authnz_tables(Path(str(db_path)))

    repo = TelegramApprovalsRepo(pool)
    await repo.ensure_tables()

    expires_at = datetime.now(timezone.utc) + timedelta(minutes=5)
    created = await repo.create_pending_approval(
        approval_token="tok-123",
        scope_type="group",
        scope_id=88,
        approval_policy_id=17,
        context_key="user:202|group:88|persona:researcher",
        conversation_id="conv-1",
        tool_name="Bash",
        scope_key="tool:Bash|args:abc123",
        initiating_auth_user_id=202,
        expires_at=expires_at,
    )
    assert created["approval_token"] == "tok-123"

    fetched = await repo.get_pending_approval_by_token("tok-123")
    assert fetched is not None
    assert fetched["scope_key"] == "tool:Bash|args:abc123"

    consumed = await repo.consume_pending_approval("tok-123")
    assert consumed is not None
    assert consumed["consumed_at"] is not None

    consumed_again = await repo.consume_pending_approval("tok-123")
    assert consumed_again is None

    missing = await repo.get_pending_approval_by_token("tok-123")
    assert missing is None


@pytest.mark.asyncio
async def test_telegram_approvals_repo_ignores_expired_approvals(tmp_path, monkeypatch) -> None:
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.repos.telegram_approvals_repo import TelegramApprovalsRepo
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    db_path = tmp_path / "users.db"
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

    reset_settings()
    await reset_db_pool()

    pool = await get_db_pool()
    ensure_authnz_tables(Path(str(db_path)))

    repo = TelegramApprovalsRepo(pool)
    await repo.ensure_tables()

    expired_at = datetime.now(timezone.utc) - timedelta(minutes=1)
    await repo.create_pending_approval(
        approval_token="tok-expired",
        scope_type="group",
        scope_id=88,
        approval_policy_id=17,
        context_key="user:202|group:88|persona:researcher",
        conversation_id="conv-1",
        tool_name="Bash",
        scope_key="tool:Bash|args:abc123",
        initiating_auth_user_id=202,
        expires_at=expired_at,
    )

    assert await repo.get_pending_approval_by_token("tok-expired") is None
    assert await repo.consume_pending_approval("tok-expired") is None
