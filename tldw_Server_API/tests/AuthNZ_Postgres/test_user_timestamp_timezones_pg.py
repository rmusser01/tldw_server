from __future__ import annotations

import uuid
from datetime import datetime, timezone

import asyncpg
import pytest

from tldw_Server_API.app.core.DB_Management.Users_DB import UsersDB


@pytest.mark.integration
@pytest.mark.asyncio
async def test_user_timestamp_repair_allows_aware_setup_self_verify(test_db_pool) -> None:
    from tldw_Server_API.app.core.AuthNZ.pg_migrations_extra import (
        ensure_user_timestamp_timezones_pg,
    )
    from tldw_Server_API.app.services.auth_service import (
        mark_user_verified,
        update_user_last_login,
    )

    username = f"pg-setup-{uuid.uuid4().hex[:8]}"
    users = UsersDB(test_db_pool)
    await users.initialize(ensure_schema=False)
    seed_hash = "hash"
    user = await users.create_user(
        username=username,
        email=f"{username}@example.com",
        password_hash=seed_hash,
        uuid_value=uuid.uuid4(),
    )
    user_id = int(user["id"])
    legacy_connection = await asyncpg.connect(test_db_pool.settings.DATABASE_URL)
    try:
        await legacy_connection.execute(
            "ALTER TABLE users ALTER COLUMN updated_at TYPE TIMESTAMP WITHOUT TIME ZONE "
            "USING updated_at AT TIME ZONE 'UTC'"
        )
    finally:
        await legacy_connection.close()
    before_type = await test_db_pool.fetchval(
        """
        SELECT data_type
        FROM information_schema.columns
        WHERE table_schema = current_schema()
          AND table_name = 'users'
          AND column_name = 'updated_at'
        """
    )

    assert before_type == "timestamp without time zone"

    assert await ensure_user_timestamp_timezones_pg(test_db_pool) is True
    after_type = await test_db_pool.fetchval(
        """
        SELECT data_type
        FROM information_schema.columns
        WHERE table_schema = current_schema()
          AND table_name = 'users'
          AND column_name = 'updated_at'
        """
    )
    assert after_type == "timestamp with time zone"

    await mark_user_verified(
        test_db_pool,
        user_id=int(user_id),
        now_utc=datetime(2026, 7, 5, 5, 25, 52, tzinfo=timezone.utc),
    )
    await update_user_last_login(
        test_db_pool,
        user_id=int(user_id),
        now=datetime(2026, 7, 5, 6, 25, 52),
    )

    row = await test_db_pool.fetchrow(
        "SELECT is_verified, updated_at, last_login FROM users WHERE id = $1",
        user_id,
    )
    assert row["is_verified"] is True
    assert row["updated_at"].tzinfo is not None
    assert row["last_login"].tzinfo is not None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_auth_service_pool_profile_writes_share_one_connection(test_db_pool) -> None:
    from tldw_Server_API.app.core.AuthNZ.pg_migrations_extra import (
        ensure_user_timestamp_timezones_pg,
    )
    from tldw_Server_API.app.services.auth_service import (
        update_user_last_login,
        verify_user_email_once,
    )

    username = f"pg-profile-service-{uuid.uuid4().hex[:8]}"
    assert await ensure_user_timestamp_timezones_pg(test_db_pool) is True
    users = UsersDB(test_db_pool)
    await users.initialize(ensure_schema=False)
    seed_hash = "hash"
    user = await users.create_user(
        username=username,
        email=f"{username}@example.com",
        password_hash=seed_hash,
    )
    user_id = int(user["id"])
    now = datetime(2026, 7, 5, 6, 25, 52, tzinfo=timezone.utc)

    changed = await verify_user_email_once(
        test_db_pool,
        user_id=user_id,
        email=f"{username}@example.com",
        now_utc=now,
    )
    assert changed == 1
    await update_user_last_login(test_db_pool, user_id=user_id, now=now)

    row = await test_db_pool.fetchrow(
        "SELECT is_verified, last_login, profile_version FROM users WHERE id = $1",
        user_id,
    )
    assert row["is_verified"] is True
    assert row["last_login"] == now
    assert row["profile_version"] > user["profile_version"]
