from __future__ import annotations

from datetime import datetime, timezone
import uuid

import pytest


@pytest.mark.integration
@pytest.mark.asyncio
async def test_user_timestamp_repair_allows_aware_setup_self_verify(test_db_pool) -> None:
    from tldw_Server_API.app.core.AuthNZ.pg_migrations_extra import (
        ensure_user_timestamp_timezones_pg,
    )
    from tldw_Server_API.app.services.auth_service import mark_user_verified

    username = f"pg-setup-{uuid.uuid4().hex[:8]}"
    await test_db_pool.execute(
        """
        INSERT INTO users (uuid, username, email, password_hash, is_active)
        VALUES ($1, $2, $3, $4, TRUE)
        """,
        str(uuid.uuid4()),
        username,
        f"{username}@example.com",
        "hash",
    )
    user_id = await test_db_pool.fetchval("SELECT id FROM users WHERE username = $1", username)
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

    row = await test_db_pool.fetchrow(
        "SELECT is_verified, updated_at FROM users WHERE id = $1",
        user_id,
    )
    assert row["is_verified"] is True
    assert row["updated_at"].tzinfo is not None
