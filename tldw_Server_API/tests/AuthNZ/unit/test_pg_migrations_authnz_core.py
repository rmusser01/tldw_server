"""Unit tests for AuthNZ PostgreSQL core bootstrap migrations."""

import asyncio

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


def test_ensure_authnz_core_tables_pg_emits_password_history_ddl() -> None:
    """Verify the core Postgres bootstrap emits password history DDL."""
    from tldw_Server_API.app.core.AuthNZ.pg_migrations_extra import ensure_authnz_core_tables_pg

    async def _run() -> None:
        """Run the async migration helper inside this synchronous unit test."""
        pool = _StubPostgresPool()
        ok = await ensure_authnz_core_tables_pg(pool)

        assert ok is True
        assert any("CREATE TABLE IF NOT EXISTS password_history" in sql for sql in pool.executed_sql)
        assert any("created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP" in sql for sql in pool.executed_sql)
        assert any("idx_password_history_user_created_at" in sql for sql in pool.executed_sql)
        assert any("ON password_history(user_id, created_at DESC, id DESC)" in sql for sql in pool.executed_sql)

    asyncio.run(_run())
