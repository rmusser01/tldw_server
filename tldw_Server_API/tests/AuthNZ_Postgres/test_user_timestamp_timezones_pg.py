from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from tldw_Server_API.tests.helpers.authnz_seed import (
    ensure_test_user,
    unmanaged_authnz_pg_connection,
)


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
    user_id = await ensure_test_user(test_db_pool, username, f"{username}@example.com")
    # Seeding through UsersDB already repairs the schema, so put the legacy
    # naive timestamp columns back the way an older release left them.
    async with unmanaged_authnz_pg_connection() as conn:
        for column in (
            "created_at",
            "updated_at",
            "last_login",
            "locked_until",
            "email_verified_at",
            "password_changed_at",
        ):
            await conn.execute(
                f"ALTER TABLE users ALTER COLUMN {column} "
                f"TYPE TIMESTAMP WITHOUT TIME ZONE USING {column} AT TIME ZONE 'UTC'"
            )
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

    # Versioned users writes run on a transaction connection, as the
    # endpoints' get_db_transaction dependency provides in production.
    async with test_db_pool.transaction() as conn:
        await mark_user_verified(
            conn,
            user_id=int(user_id),
            now_utc=datetime(2026, 7, 5, 5, 25, 52, tzinfo=timezone.utc),
        )
    async with test_db_pool.transaction() as conn:
        await update_user_last_login(
            conn,
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
