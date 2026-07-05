from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import uuid

import pytest

pytest_plugins = ("tldw_Server_API.tests._plugins.authnz_full_fixtures",)


@pytest.mark.asyncio
async def test_byok_oauth_state_repo_sqlite_roundtrip(tmp_path, monkeypatch) -> None:
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.repos.byok_oauth_state_repo import (
        AuthnzByokOAuthStateRepo,
    )
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.DB_Management.Users_DB import UsersDB

    db_path = tmp_path / "users.db"
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

    reset_settings()
    await reset_db_pool()

    pool = await get_db_pool()
    ensure_authnz_tables(Path(str(db_path)))

    users_db = UsersDB(pool)
    await users_db.initialize()
    created_user = await users_db.create_user(
        username="oauth-state-user",
        email="oauth-state@example.com",
        password_hash="hashed-password",
        role="user",
        is_active=True,
        is_superuser=False,
        storage_quota_mb=5120,
        uuid_value=uuid.uuid4(),
    )
    user_id = int(created_user["id"])

    repo = AuthnzByokOAuthStateRepo(pool)
    await repo.ensure_tables()

    now = datetime.now(timezone.utc)
    active_state = f"state-active-{uuid.uuid4().hex}"
    consumed_state = f"state-consumed-{uuid.uuid4().hex}"
    expired_state = f"state-expired-{uuid.uuid4().hex}"

    created_active = await repo.create_state(
        state=active_state,
        user_id=user_id,
        provider="openai",
        auth_session_id="session-active",
        redirect_uri="https://example.com/oauth/callback",
        pkce_verifier_encrypted="enc-pkce-active",
        expires_at=now + timedelta(minutes=60),
        return_path="/settings/providers",
    )
    assert created_active["state"] == active_state
    assert created_active["provider"] == "openai"

    fetched_active = await repo.fetch_state(
        state=active_state,
        provider="openai",
        now=now,
    )
    assert fetched_active is not None
    assert fetched_active["consumed_at"] is None
    assert fetched_active["return_path"] == "/settings/providers"

    await repo.create_state(
        state=consumed_state,
        user_id=user_id,
        provider="openai",
        auth_session_id="session-consumed",
        redirect_uri="https://example.com/oauth/callback",
        pkce_verifier_encrypted="enc-pkce-consumed",
        expires_at=now + timedelta(minutes=10),
    )
    consumed = await repo.consume_state(
        state=consumed_state,
        provider="openai",
        consumed_at=now + timedelta(seconds=5),
    )
    assert consumed is not None
    assert consumed["state"] == consumed_state
    assert consumed["consumed_at"] is not None

    replay = await repo.consume_state(
        state=consumed_state,
        provider="openai",
        consumed_at=now + timedelta(seconds=10),
    )
    assert replay is None

    await repo.create_state(
        state=expired_state,
        user_id=user_id,
        provider="openai",
        auth_session_id="session-expired",
        redirect_uri="https://example.com/oauth/callback",
        pkce_verifier_encrypted="enc-pkce-expired",
        expires_at=now - timedelta(minutes=1),
    )
    expired_consume = await repo.consume_state(
        state=expired_state,
        provider="openai",
        consumed_at=now,
    )
    assert expired_consume is None

    purged = await repo.purge_expired(now=now + timedelta(minutes=30))
    assert purged >= 2

    still_active = await repo.fetch_state(
        state=active_state,
        provider="openai",
        now=now + timedelta(minutes=5),
    )
    assert still_active is not None
    assert still_active["state"] == active_state


@pytest.mark.asyncio
async def test_byok_oauth_state_repo_enforces_outstanding_cap_sqlite(tmp_path, monkeypatch) -> None:
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.repos.byok_oauth_state_repo import (
        AuthnzByokOAuthStateRepo,
    )
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.DB_Management.Users_DB import UsersDB

    db_path = tmp_path / "users.db"
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

    reset_settings()
    await reset_db_pool()

    pool = await get_db_pool()
    ensure_authnz_tables(Path(str(db_path)))

    users_db = UsersDB(pool)
    await users_db.initialize()
    created_user = await users_db.create_user(
        username="oauth-state-cap-user",
        email="oauth-state-cap@example.com",
        password_hash="hashed-password",
        role="user",
        is_active=True,
        is_superuser=False,
        storage_quota_mb=5120,
        uuid_value=uuid.uuid4(),
    )
    user_id = int(created_user["id"])

    repo = AuthnzByokOAuthStateRepo(pool)
    await repo.ensure_tables()

    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(minutes=30)
    states = [
        f"state-cap-{idx}-{uuid.uuid4().hex}"
        for idx in range(3)
    ]

    for idx, state in enumerate(states):
        await repo.create_state(
            state=state,
            user_id=user_id,
            provider="openai",
            auth_session_id=f"session-cap-{idx}",
            redirect_uri="https://example.com/oauth/callback",
            pkce_verifier_encrypted=f"enc-pkce-cap-{idx}",
            created_at=now + timedelta(seconds=idx),
            expires_at=expires_at,
        )

    outstanding_before = await repo.count_outstanding(
        user_id=user_id,
        provider="openai",
        now=now + timedelta(seconds=5),
    )
    assert outstanding_before == 3

    deleted = await repo.enforce_outstanding_cap(
        user_id=user_id,
        provider="openai",
        max_outstanding=2,
        now=now + timedelta(seconds=5),
    )
    assert deleted == 2

    outstanding_after = await repo.count_outstanding(
        user_id=user_id,
        provider="openai",
        now=now + timedelta(seconds=5),
    )
    assert outstanding_after == 1

    oldest = await repo.fetch_state(
        state=states[0],
        provider="openai",
        now=now + timedelta(seconds=5),
    )
    middle = await repo.fetch_state(
        state=states[1],
        provider="openai",
        now=now + timedelta(seconds=5),
    )
    newest = await repo.fetch_state(
        state=states[2],
        provider="openai",
        now=now + timedelta(seconds=5),
    )
    assert oldest is None
    assert middle is None
    assert newest is not None
    assert newest["state"] == states[2]
