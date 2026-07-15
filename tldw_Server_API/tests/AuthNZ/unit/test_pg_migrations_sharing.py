"""Tests for PostgreSQL sharing schema bootstrap."""

from __future__ import annotations

import sqlite3

import asyncpg
import pytest

from tldw_Server_API.app.core.AuthNZ import pg_migrations_extra
from tldw_Server_API.app.core.AuthNZ.exceptions import DatabaseError
from tldw_Server_API.app.core.AuthNZ.repos.shared_workspace_repo import SharedWorkspaceRepo

pytestmark = pytest.mark.unit

_SHARING_TABLES = {
    "shared_workspaces",
    "share_tokens",
    "share_audit_log",
    "sharing_config",
}


class _StubPostgresPool:
    """Capture PostgreSQL migration statements."""

    def __init__(self) -> None:
        self.pool = object()
        self.executed_sql: list[str] = []
        self.queries: list[str] = []

    async def execute(self, query: str, *args: object) -> None:
        self.executed_sql.append(query)

    async def fetchall(self, query: str, args: tuple[object, ...]) -> list[dict[str, str]]:
        self.queries.append(query)
        if "sharing_schema_issues" in query:
            return []
        return [{"name": table} for table in sorted(_SHARING_TABLES)]


class _StubNonPostgresPool:
    pool = None


class _BrokenPostgresPool:
    pool = object()

    async def execute(self, query: str, *args: object) -> None:
        raise DatabaseError("forced DDL failure")


class _RawBrokenPostgresPool:
    pool = object()

    async def execute(self, query: str, *args: object) -> None:
        raise asyncpg.InsufficientPrivilegeError("forced raw PostgreSQL failure")


class _RawInterfaceBrokenPostgresPool:
    pool = object()

    async def execute(self, query: str, *args: object) -> None:
        raise asyncpg.InterfaceError("forced PostgreSQL interface failure")


class _RawInternalBrokenPostgresPool:
    pool = object()

    async def execute(self, query: str, *args: object) -> None:
        raise asyncpg.InternalClientError("forced PostgreSQL client failure")


@pytest.mark.asyncio
async def test_ensure_sharing_tables_pg_emits_complete_schema() -> None:
    ensure = getattr(pg_migrations_extra, "ensure_sharing_tables_pg", None)
    assert callable(ensure), "PostgreSQL sharing bootstrap is missing"

    pool = _StubPostgresPool()
    assert await ensure(pool) is True

    ddl = "\n".join(pool.executed_sql)
    for table in _SHARING_TABLES:
        assert f"CREATE TABLE IF NOT EXISTS {table}" in ddl
    assert "prototype_workspace" in ddl
    assert "pg_get_constraintdef" in ddl
    assert "VALIDATE CONSTRAINT ck_share_tokens_resource_type" in ddl
    assert "BOOLEAN NOT NULL DEFAULT TRUE" in ddl
    assert "uq_shared_workspaces_scope" in ddl
    assert "uq_sharing_config_scope_key" in ddl
    assert "uq_sharing_config_global_key" in ddl
    assert "WHERE scope_id IS NULL" in ddl
    assert "idx_share_tokens_resource" in ddl
    assert "idx_share_audit_owner" in ddl


@pytest.mark.asyncio
async def test_ensure_sharing_tables_pg_skips_non_postgres() -> None:
    ensure = getattr(pg_migrations_extra, "ensure_sharing_tables_pg", None)
    assert callable(ensure), "PostgreSQL sharing bootstrap is missing"
    assert await ensure(_StubNonPostgresPool()) is False


@pytest.mark.asyncio
async def test_ensure_sharing_tables_pg_reports_backend_failure() -> None:
    ensure = getattr(pg_migrations_extra, "ensure_sharing_tables_pg", None)
    assert callable(ensure), "PostgreSQL sharing bootstrap is missing"
    assert await ensure(_BrokenPostgresPool()) is False


@pytest.mark.asyncio
async def test_ensure_sharing_tables_pg_reports_raw_asyncpg_failure() -> None:
    ensure = getattr(pg_migrations_extra, "ensure_sharing_tables_pg", None)
    assert callable(ensure), "PostgreSQL sharing bootstrap is missing"
    assert await ensure(_RawBrokenPostgresPool()) is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pool",
    [_RawInterfaceBrokenPostgresPool(), _RawInternalBrokenPostgresPool()],
)
async def test_ensure_sharing_tables_pg_reports_asyncpg_client_failure(pool: object) -> None:
    ensure = getattr(pg_migrations_extra, "ensure_sharing_tables_pg", None)
    assert callable(ensure), "PostgreSQL sharing bootstrap is missing"
    assert await ensure(pool) is False


@pytest.mark.asyncio
async def test_shared_workspace_repo_checks_postgres_catalog() -> None:
    pool = _StubPostgresPool()
    await SharedWorkspaceRepo(pool).ensure_tables()
    assert len(pool.queries) == 2
    assert "information_schema.tables" in pool.queries[0]
    assert "sharing_schema_issues" in pool.queries[1]
    assert "sqlite_master" not in pool.queries[0]


@pytest.mark.asyncio
async def test_shared_workspace_repo_reports_missing_postgres_table() -> None:
    pool = _StubPostgresPool()

    async def _missing_table(
        query: str,
        args: tuple[object, ...],
    ) -> list[dict[str, str]]:
        pool.queries.append(query)
        return [{"name": table} for table in sorted(_SHARING_TABLES - {"share_tokens"})]

    pool.fetchall = _missing_table  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="share_tokens"):
        await SharedWorkspaceRepo(pool).ensure_tables()


@pytest.mark.asyncio
async def test_shared_workspace_repo_reports_postgres_contract_drift() -> None:
    pool = _StubPostgresPool()

    async def _schema_drift(
        query: str,
        args: tuple[object, ...],
    ) -> list[dict[str, str]]:
        pool.queries.append(query)
        if "sharing_schema_issues" in query:
            return [{"issue": "missing constraint shared_workspaces.ck_shared_workspaces_access_level"}]
        return [{"name": table} for table in sorted(_SHARING_TABLES)]

    pool.fetchall = _schema_drift  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="ck_shared_workspaces_access_level"):
        await SharedWorkspaceRepo(pool).ensure_tables()


def test_duplicate_share_mapping_requires_scope_constraint() -> None:
    from tldw_Server_API.app.api.v1.endpoints.sharing import (
        _is_duplicate_share_error,
    )

    duplicate = asyncpg.UniqueViolationError("duplicate share")
    duplicate.constraint_name = "uq_shared_workspaces_scope"
    unrelated = asyncpg.UniqueViolationError("stale primary-key sequence")
    unrelated.constraint_name = "shared_workspaces_pkey"
    sqlite_duplicate = sqlite3.IntegrityError(
        "UNIQUE constraint failed: shared_workspaces.workspace_id, "
        "shared_workspaces.owner_user_id, shared_workspaces.share_scope_type, "
        "shared_workspaces.share_scope_id"
    )
    sqlite_unrelated = sqlite3.IntegrityError(
        "UNIQUE constraint failed: shared_workspaces.id"
    )

    assert _is_duplicate_share_error(duplicate) is True
    assert _is_duplicate_share_error(unrelated) is False
    assert _is_duplicate_share_error(sqlite_duplicate) is True
    assert _is_duplicate_share_error(sqlite_unrelated) is False
