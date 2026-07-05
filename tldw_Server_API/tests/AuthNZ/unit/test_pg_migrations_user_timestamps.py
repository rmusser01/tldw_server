from __future__ import annotations

import pytest


pytestmark = pytest.mark.unit


class _StubPostgresPool:
    def __init__(self) -> None:
        self.pool = object()
        self.executed_sql: list[str] = []

    async def execute(self, query: str, *args):  # noqa: ANN001, ANN002
        self.executed_sql.append(query)
        return None


class _StubNonPostgresPool:
    pool = None


@pytest.mark.asyncio
async def test_ensure_user_timestamp_timezones_pg_skips_non_postgres() -> None:
    from tldw_Server_API.app.core.AuthNZ.pg_migrations_extra import (
        ensure_user_timestamp_timezones_pg,
    )

    ok = await ensure_user_timestamp_timezones_pg(_StubNonPostgresPool())

    assert ok is False


@pytest.mark.asyncio
async def test_ensure_user_timestamp_timezones_pg_emits_utc_repairs() -> None:
    from tldw_Server_API.app.core.AuthNZ.pg_migrations_extra import (
        ensure_user_timestamp_timezones_pg,
    )

    pool = _StubPostgresPool()

    ok = await ensure_user_timestamp_timezones_pg(pool)

    assert ok is True
    ddl = "\n".join(pool.executed_sql)
    assert "ALTER TABLE users" in ddl
    assert "updated_at" in ddl
    assert "TYPE TIMESTAMPTZ" in ddl
    assert "AT TIME ZONE" in ddl
    assert "UTC" in ddl
