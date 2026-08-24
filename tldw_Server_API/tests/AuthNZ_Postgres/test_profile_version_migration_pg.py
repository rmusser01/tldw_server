from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.postgres]

_POSTGRES_SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "Databases"
    / "Postgres"
    / "Schema"
    / "postgresql_users.sql"
)


async def _insert_legacy_user(test_db_pool, *, updated_at: datetime | None) -> int:  # noqa: ANN001
    username = f"profile-version-{uuid.uuid4().hex[:8]}"
    return int(
        await test_db_pool.fetchval(
            """
            INSERT INTO users (uuid, username, email, password_hash, updated_at)
            VALUES ($1, $2, $3, 'hash', $4)
            RETURNING id
            """,
            str(uuid.uuid4()),
            username,
            f"{username}@example.com",
            updated_at,
        )
    )


async def _reset_to_legacy_users_schema(test_db_pool) -> None:  # noqa: ANN001
    await test_db_pool.execute("ALTER TABLE users DROP COLUMN IF EXISTS profile_version")
    await test_db_pool.execute(
        """
        ALTER TABLE users
        ALTER COLUMN updated_at TYPE TIMESTAMP WITHOUT TIME ZONE
        USING updated_at AT TIME ZONE 'UTC'
        """
    )


def test_fresh_postgres_schema_declares_durable_profile_version() -> None:
    schema_sql = _POSTGRES_SCHEMA_PATH.read_text(encoding="utf-8").upper()

    assert "PROFILE_VERSION TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP" in schema_sql


@pytest.mark.asyncio
async def test_postgres_upgrade_backfills_naive_updated_at_without_version_jump(
    test_db_pool,
) -> None:  # noqa: ANN001
    from tldw_Server_API.app.core.AuthNZ.pg_migrations_extra import (
        ensure_user_profile_version_pg,
    )

    await _reset_to_legacy_users_schema(test_db_pool)
    legacy_value = datetime(2026, 1, 2, 3, 4, 5, 123456)
    user_id = await _insert_legacy_user(test_db_pool, updated_at=legacy_value)

    assert await ensure_user_profile_version_pg(test_db_pool) is True

    row = await test_db_pool.fetchrow(
        "SELECT updated_at, profile_version FROM users WHERE id = $1",
        user_id,
    )
    metadata = await test_db_pool.fetchrow(
        """
        SELECT data_type, is_nullable, column_default
        FROM information_schema.columns
        WHERE table_schema = current_schema()
          AND table_name = 'users'
          AND column_name = 'profile_version'
        """
    )
    expected = legacy_value.replace(tzinfo=timezone.utc)
    assert row["updated_at"] == expected
    assert row["profile_version"] == expected
    assert metadata["data_type"] == "timestamp with time zone"
    assert metadata["is_nullable"] == "NO"
    assert "CURRENT_TIMESTAMP" in metadata["column_default"].upper()


@pytest.mark.asyncio
async def test_postgres_upgrade_preserves_aware_updated_at_and_is_idempotent(
    test_db_pool,
) -> None:  # noqa: ANN001
    from tldw_Server_API.app.core.AuthNZ.pg_migrations_extra import (
        ensure_user_profile_version_pg,
    )

    await test_db_pool.execute("ALTER TABLE users DROP COLUMN IF EXISTS profile_version")
    await test_db_pool.execute(
        """
        ALTER TABLE users
        ALTER COLUMN updated_at TYPE TIMESTAMPTZ
        USING updated_at AT TIME ZONE 'UTC'
        """
    )
    aware_value = datetime(2026, 1, 2, 3, 4, 5, 654321, tzinfo=timezone.utc)
    user_id = await _insert_legacy_user(test_db_pool, updated_at=aware_value)

    assert await ensure_user_profile_version_pg(test_db_pool) is True
    first_value = await test_db_pool.fetchval(
        "SELECT profile_version FROM users WHERE id = $1",
        user_id,
    )
    assert await ensure_user_profile_version_pg(test_db_pool) is True

    assert first_value == aware_value
    assert await test_db_pool.fetchval(
        "SELECT profile_version FROM users WHERE id = $1",
        user_id,
    ) == aware_value


@pytest.mark.asyncio
async def test_postgres_upgrade_rejects_null_updated_at_and_rolls_back(
    test_db_pool,
) -> None:  # noqa: ANN001
    from tldw_Server_API.app.core.AuthNZ.pg_migrations_extra import (
        ensure_user_profile_version_pg,
    )

    await _reset_to_legacy_users_schema(test_db_pool)
    user_id = await _insert_legacy_user(test_db_pool, updated_at=None)

    with pytest.raises(RuntimeError, match="profile_version"):
        await ensure_user_profile_version_pg(test_db_pool)

    exists = await test_db_pool.fetchval(
        """
        SELECT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = 'users'
              AND column_name = 'profile_version'
        )
        """
    )
    assert exists is False

    await test_db_pool.execute(
        "UPDATE users SET updated_at = $2 WHERE id = $1",
        user_id,
        datetime(2026, 1, 2, 3, 4, 5),
    )
    assert await ensure_user_profile_version_pg(test_db_pool) is True


@pytest.mark.asyncio
async def test_postgres_current_schema_corruption_fails_closed_at_startup(
    test_db_pool,
) -> None:  # noqa: ANN001
    from tldw_Server_API.app.core.AuthNZ.pg_migrations_extra import (
        ensure_authnz_core_tables_pg,
        ensure_user_profile_version_pg,
    )

    assert await ensure_user_profile_version_pg(test_db_pool) is True
    user_id = await _insert_legacy_user(
        test_db_pool,
        updated_at=datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
    )
    await test_db_pool.execute(
        "ALTER TABLE users ALTER COLUMN profile_version DROP NOT NULL"
    )
    await test_db_pool.execute(
        "UPDATE users SET profile_version = NULL WHERE id = $1",
        user_id,
    )

    with pytest.raises(RuntimeError, match="profile_version"):
        await ensure_authnz_core_tables_pg(test_db_pool)

    await test_db_pool.execute(
        "UPDATE users SET profile_version = updated_at WHERE id = $1",
        user_id,
    )
    assert await ensure_user_profile_version_pg(test_db_pool) is True


@pytest.mark.asyncio
async def test_postgres_readiness_rejects_updatable_view_alias_to_users(
    test_db_pool,
) -> None:  # noqa: ANN001
    from tldw_Server_API.app.core.AuthNZ.pg_migrations_extra import (
        ensure_user_profile_version_pg,
    )

    view_name = f"user_alias_{uuid.uuid4().hex[:8]}"
    nested_view_name = f"nested_user_alias_{uuid.uuid4().hex[:8]}"
    async with test_db_pool.pool.acquire() as raw_conn:
        await raw_conn.execute(
            f"CREATE VIEW public.{view_name} AS "
            "SELECT id, email FROM public.users"
        )
        await raw_conn.execute(
            f"CREATE VIEW public.{nested_view_name} AS "
            f"SELECT id, email FROM public.{view_name}"
        )
    try:
        with pytest.raises(RuntimeError, match="indirect.*users write"):
            await ensure_user_profile_version_pg(test_db_pool)
    finally:
        async with test_db_pool.pool.acquire() as raw_conn:
            await raw_conn.execute(f"DROP VIEW IF EXISTS public.{nested_view_name}")
            await raw_conn.execute(f"DROP VIEW IF EXISTS public.{view_name}")


@pytest.mark.asyncio
async def test_postgres_readiness_rejects_inherited_users_descendant(
    test_db_pool,
) -> None:  # noqa: ANN001
    from tldw_Server_API.app.core.AuthNZ.pg_migrations_extra import (
        ensure_user_profile_version_pg,
    )

    table_name = f"user_child_{uuid.uuid4().hex[:8]}"
    async with test_db_pool.pool.acquire() as raw_conn:
        await raw_conn.execute(
            f"CREATE TABLE public.{table_name} () INHERITS (public.users)"
        )
    try:
        with pytest.raises(RuntimeError, match="indirect.*users write"):
            await ensure_user_profile_version_pg(test_db_pool)
    finally:
        async with test_db_pool.pool.acquire() as raw_conn:
            await raw_conn.execute(f"DROP TABLE IF EXISTS public.{table_name}")


@pytest.mark.asyncio
async def test_postgres_readiness_serializes_concurrent_legacy_upgrades(
    test_db_pool,
) -> None:  # noqa: ANN001
    from tldw_Server_API.app.core.AuthNZ.postgres_profile_version_schema import (
        ensure_postgres_profile_version_on_connection,
    )

    await _reset_to_legacy_users_schema(test_db_pool)

    async def _upgrade() -> None:
        async with test_db_pool.pool.acquire() as raw_conn:
            async with raw_conn.transaction():
                await ensure_postgres_profile_version_on_connection(raw_conn)

    await asyncio.gather(_upgrade(), _upgrade())

    assert await test_db_pool.fetchval(
        "SELECT COUNT(*) FROM information_schema.columns "
        "WHERE table_schema = 'public' AND table_name = 'users' "
        "AND column_name = 'profile_version'"
    ) == 1
