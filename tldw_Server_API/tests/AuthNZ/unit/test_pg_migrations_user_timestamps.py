from __future__ import annotations

from typing import Any

import pytest

from tldw_Server_API.app.core.AuthNZ.profile_user_write_guard import _guard_sql

pytestmark = pytest.mark.unit


class _StubPostgresPool:
    def __init__(self) -> None:
        self.pool = object()
        self.executed_sql: list[str] = []
        self.connection = _StubPostgresConnection(self.executed_sql)

    def transaction(self) -> _Transaction:
        return _Transaction(self.connection)


class _Transaction:
    def __init__(self, connection: _StubPostgresConnection) -> None:
        self.connection = connection

    async def __aenter__(self) -> _StubPostgresConnection:
        return self.connection

    async def __aexit__(self, exc_type, exc, traceback) -> bool:  # noqa: ANN001
        del exc_type, exc, traceback
        return False


class _StubPostgresConnection:
    _authnz_profile_user_backend = "postgres"

    def __init__(self, executed_sql: list[str]) -> None:
        self._authnz_profile_user_guard_identity = self
        self.executed_sql = executed_sql

    async def fetchval(self, query: str, key: int) -> None:
        assert "pg_advisory_xact_lock" in query
        assert key == 0x544C44575F505631

    async def fetch(self, query: str, columns: list[str]) -> list[dict[str, Any]]:
        assert "information_schema.columns" in query
        assert "updated_at" in columns
        return [
            {
                "column_name": "updated_at",
                "data_type": "timestamp without time zone",
            }
        ]

    async def execute(self, query: object) -> None:
        concrete = _guard_sql(
            query,
            backend="postgres",
            connection_identity=self,
            operation="execute",
        )
        self.executed_sql.append(concrete)


class _StubNonPostgresPool:
    pool = None


class _FailingPostgresPool:
    pool = object()

    def transaction(self) -> _Transaction:
        raise RuntimeError("private timestamp migration failure")


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
    assert "ALTER TABLE public.users" in ddl
    assert "updated_at" in ddl
    assert "TYPE TIMESTAMPTZ" in ddl
    assert "AT TIME ZONE" in ddl
    assert "UTC" in ddl


@pytest.mark.asyncio
async def test_ensure_user_timestamp_timezones_pg_propagates_migration_failure() -> None:
    from tldw_Server_API.app.core.AuthNZ.pg_migrations_extra import (
        ensure_user_timestamp_timezones_pg,
    )

    with pytest.raises(RuntimeError, match="private timestamp migration failure"):
        await ensure_user_timestamp_timezones_pg(_FailingPostgresPool())
