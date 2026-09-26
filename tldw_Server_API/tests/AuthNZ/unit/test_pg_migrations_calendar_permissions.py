"""Regression coverage for the Calendar PostgreSQL permission migration boundary."""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, get_type_hints
from unittest.mock import AsyncMock

import pytest

from tldw_Server_API.app.core.AuthNZ import pg_migrations_extra, rbac_seed

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


class _Connection:
    """Expose only the asynchronous table lookup used by this migration."""

    def __init__(self, tables: tuple[str, ...] = ("roles", "permissions", "role_permissions")) -> None:
        """Configure returned table names and record query provenance."""
        self.tables = tables
        self.calls: list[tuple[str, list[str]]] = []
        self.query_modules: list[str] = []
        self.in_transaction = False

    async def fetch(self, query: str, tables: list[str]) -> list[dict[str, str]]:
        """Record the actual query caller without replacing the DB helper."""
        frame = inspect.currentframe()
        assert frame is not None and frame.f_back is not None
        self.query_modules.append(str(frame.f_back.f_globals["__name__"]))
        del frame
        self.calls.append((query, tables))
        return [{"table_name": table} for table in self.tables]


class _Pool:
    """Model the caller-owned transaction and its exception propagation."""

    def __init__(self, connection: _Connection, *, is_postgres: bool = True) -> None:
        """Keep connection acquisition separate from table-query execution."""
        self.pool = object() if is_postgres else None
        self.connection = connection
        self.events: list[str] = []

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[_Connection]:
        """Record whether migration work exits normally or with a failure."""
        self.events.append("begin")
        self.connection.in_transaction = True
        try:
            yield self.connection
        except RuntimeError:
            self.events.append("rollback")
            raise
        else:
            self.events.append("commit")
        finally:
            self.connection.in_transaction = False


async def test_calendar_table_lookup_executes_in_db_management(monkeypatch: pytest.MonkeyPatch) -> None:
    """Catch moving the new existence query back across the DB ownership boundary."""
    connection = _Connection()
    monkeypatch.setattr(rbac_seed, "ensure_baseline_rbac_seed", AsyncMock())

    assert await pg_migrations_extra.ensure_calendar_permissions_pg(_Pool(connection)) is True
    assert connection.query_modules == ["tldw_Server_API.app.core.DB_Management.calendar_permission_schema"]


@pytest.mark.parametrize("missing", ["roles", "permissions", "role_permissions"])
async def test_calendar_missing_rbac_table_skips_seed(monkeypatch: pytest.MonkeyPatch, missing: str) -> None:
    """Require every RBAC table before attempting the baseline backfill."""
    connection = _Connection(tuple(table for table in ("roles", "permissions", "role_permissions") if table != missing))
    pool = _Pool(connection)
    seed = AsyncMock()
    monkeypatch.setattr(rbac_seed, "ensure_baseline_rbac_seed", seed)

    assert await pg_migrations_extra.ensure_calendar_permissions_pg(pool) is False
    seed.assert_not_awaited()
    assert pool.events == ["begin", "commit"]
    assert connection.in_transaction is False


async def test_calendar_seed_uses_lookup_connection_inside_transaction(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the seed on the lookup transaction with the original PostgreSQL/MCP flags."""
    connection = _Connection()
    pool = _Pool(connection)

    async def seed(conn: Any, *, include_mcp_permissions: bool, is_postgres: bool) -> None:
        """Check the seeder's boundary inputs while its transaction is active."""
        assert conn is connection
        assert conn.in_transaction is True
        assert len(conn.calls) == 1
        assert include_mcp_permissions is True
        assert is_postgres is True
        pool.events.append("seed")

    seed_spy = AsyncMock(side_effect=seed)
    monkeypatch.setattr(rbac_seed, "ensure_baseline_rbac_seed", seed_spy)

    assert await pg_migrations_extra.ensure_calendar_permissions_pg(pool) is True
    seed_spy.assert_awaited_once_with(connection, include_mcp_permissions=True, is_postgres=True)
    assert pool.events == ["begin", "seed", "commit"]
    assert connection.in_transaction is False


async def test_calendar_seed_failure_exits_transaction_and_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Do not turn failed backfills into successful migration results."""
    pool = _Pool(_Connection())
    monkeypatch.setattr(rbac_seed, "ensure_baseline_rbac_seed", AsyncMock(side_effect=RuntimeError("seed failed")))

    with pytest.raises(RuntimeError, match="seed failed"):
        await pg_migrations_extra.ensure_calendar_permissions_pg(pool)

    assert pool.events == ["begin", "rollback"]
    assert pool.connection.in_transaction is False


async def test_calendar_lookup_failure_exits_transaction_without_seeding(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep table-query failures visible and let the owner roll back."""
    connection = _Connection()
    pool = _Pool(connection)
    monkeypatch.setattr(connection, "fetch", AsyncMock(side_effect=RuntimeError("lookup failed")))
    seed = AsyncMock()
    monkeypatch.setattr(rbac_seed, "ensure_baseline_rbac_seed", seed)

    with pytest.raises(RuntimeError, match="lookup failed"):
        await pg_migrations_extra.ensure_calendar_permissions_pg(pool)

    seed.assert_not_awaited()
    assert pool.events == ["begin", "rollback"]


async def test_calendar_non_postgres_skips_transaction_and_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    """SQLite pools must not acquire a PostgreSQL transaction or run its seed."""
    pool = _Pool(_Connection(), is_postgres=False)
    seed = AsyncMock()
    monkeypatch.setattr(rbac_seed, "ensure_baseline_rbac_seed", seed)

    assert await pg_migrations_extra.ensure_calendar_permissions_pg(pool) is False
    seed.assert_not_awaited()
    assert pool.events == []
    assert pool.connection.calls == []


async def test_calendar_resolves_default_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retain the default-pool path when the migration is called at startup."""
    pool = _Pool(_Connection())
    get_pool = AsyncMock(return_value=pool)
    monkeypatch.setattr(pg_migrations_extra, "get_db_pool", get_pool)
    monkeypatch.setattr(rbac_seed, "ensure_baseline_rbac_seed", AsyncMock())

    assert await pg_migrations_extra.ensure_calendar_permissions_pg() is True
    get_pool.assert_awaited_once_with()


async def test_calendar_migration_has_boolean_return_annotation_and_docstring() -> None:
    """Keep the migration's documented public return contract explicit."""
    migration = pg_migrations_extra.ensure_calendar_permissions_pg

    assert get_type_hints(migration)["return"] is bool
    assert inspect.getdoc(migration)


@pytest.mark.parametrize(
    ("tables", "expected"),
    [
        (("role_permissions", "roles", "permissions"), True),
        (("roles", "permissions"), False),
        (("roles", "role_permissions"), False),
        (("permissions", "role_permissions"), False),
        (("roles", "roles", "permissions"), False),
        ((), False),
    ],
)
async def test_calendar_rbac_table_helper_checks_all_required_names(tables: tuple[str, ...], expected: bool) -> None:
    """Table ordering and duplicate rows must not bypass missing-table detection."""
    from tldw_Server_API.app.core.DB_Management.calendar_permission_schema import (
        postgres_calendar_rbac_tables_exist,
    )

    connection = _Connection(tables)

    assert await postgres_calendar_rbac_tables_exist(connection) is expected


async def test_calendar_rbac_table_helper_parameterizes_lookup_without_transaction() -> None:
    """Keep names as bound text-array values and leave transaction ownership to callers."""
    from tldw_Server_API.app.core.DB_Management.calendar_permission_schema import (
        postgres_calendar_rbac_tables_exist,
    )

    connection = _Connection()

    assert await postgres_calendar_rbac_tables_exist(connection) is True
    assert len(connection.calls) == 1
    query, parameters = connection.calls[0]
    assert "information_schema.tables" in query
    assert "table_schema = current_schema()" in query
    assert "table_name = ANY($1::text[])" in query
    assert parameters == ["roles", "permissions", "role_permissions"]
    assert all(table not in query for table in parameters)
    assert connection.in_transaction is False


async def test_calendar_rbac_table_helper_has_boolean_return_annotation_and_docstrings() -> None:
    """Document the DB helper's result and caller-owned connection contract."""
    from tldw_Server_API.app.core.DB_Management import calendar_permission_schema

    helper = calendar_permission_schema.postgres_calendar_rbac_tables_exist

    assert get_type_hints(helper)["return"] is bool
    assert inspect.getdoc(helper)
    assert inspect.getdoc(calendar_permission_schema)
