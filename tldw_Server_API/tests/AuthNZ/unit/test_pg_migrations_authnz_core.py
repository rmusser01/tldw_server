"""Unit tests for AuthNZ PostgreSQL core bootstrap migrations."""

import asyncio
from unittest.mock import AsyncMock

import pytest

pytestmark = pytest.mark.unit


class _StubPostgresPool:
    """Capture SQL statements executed by the PostgreSQL migration helper."""

    def __init__(self) -> None:
        self.pool = object()
        self.executed_sql: list[str] = []

    async def execute(self, query: str, *args: object) -> None:
        """Record a migration SQL statement without touching a real database."""
        self.executed_sql.append(query)


def test_ensure_authnz_core_tables_pg_emits_password_history_ddl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify the core Postgres bootstrap emits password history DDL."""
    from tldw_Server_API.app.core.AuthNZ import pg_migrations_extra

    ensure_profile_ready = AsyncMock(return_value=True)
    monkeypatch.setattr(
        pg_migrations_extra,
        "ensure_user_profile_version_pg",
        ensure_profile_ready,
    )

    async def _run() -> None:
        """Run the async migration helper inside this synchronous unit test."""
        pool = _StubPostgresPool()
        ok = await pg_migrations_extra.ensure_authnz_core_tables_pg(pool)

        assert ok is True
        ensure_profile_ready.assert_awaited_once_with(pool)
        assert any("CREATE TABLE IF NOT EXISTS password_history" in sql for sql in pool.executed_sql)
        assert any("created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP" in sql for sql in pool.executed_sql)
        assert any("idx_password_history_user_created_at" in sql for sql in pool.executed_sql)
        assert any("ON password_history(user_id, created_at DESC, id DESC)" in sql for sql in pool.executed_sql)

    asyncio.run(_run())


@pytest.mark.asyncio
async def test_ensure_authnz_core_tables_pg_never_succeeds_without_profile_readiness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.AuthNZ import pg_migrations_extra

    pool = _StubPostgresPool()
    ensure_profile_ready = AsyncMock(return_value=False)
    monkeypatch.setattr(
        pg_migrations_extra,
        "ensure_user_profile_version_pg",
        ensure_profile_ready,
    )

    result = await pg_migrations_extra.ensure_authnz_core_tables_pg(pool)

    assert result is False
    ensure_profile_ready.assert_awaited_once_with(pool)


@pytest.mark.asyncio
async def test_ensure_authnz_core_tables_pg_propagates_profile_readiness_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.AuthNZ import pg_migrations_extra

    pool = _StubPostgresPool()
    readiness_failure = RuntimeError("profile_version readiness failed")
    ensure_profile_ready = AsyncMock(side_effect=readiness_failure)
    monkeypatch.setattr(
        pg_migrations_extra,
        "ensure_user_profile_version_pg",
        ensure_profile_ready,
    )

    with pytest.raises(RuntimeError, match="profile_version") as exc_info:
        await pg_migrations_extra.ensure_authnz_core_tables_pg(pool)

    assert exc_info.value is readiness_failure
    ensure_profile_ready.assert_awaited_once_with(pool)
