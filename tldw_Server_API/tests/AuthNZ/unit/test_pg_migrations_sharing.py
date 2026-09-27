"""Tests for PostgreSQL sharing schema bootstrap."""

from __future__ import annotations

import sqlite3
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import asyncpg
import pytest

from tldw_Server_API.app.core.AuthNZ import pg_migrations_extra
from tldw_Server_API.app.core.AuthNZ.exceptions import DatabaseError
from tldw_Server_API.app.core.AuthNZ.profile_user_write_guard import _guard_sql
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

    def __init__(
        self,
        *,
        constraint_rows: list[dict[str, object]] | None = None,
    ) -> None:
        self.pool = object()
        self.executed_sql: list[str] = []
        self.queries: list[str] = []
        self.constraint_rows = constraint_rows or []
        self.operations: list[str] = []
        self.transaction_entries = 0
        self.transaction_commits = 0
        self.transaction_rollbacks = 0

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[_StubPostgresPool]:
        self.operations.append("acquire")
        yield self

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[_StubPostgresPool]:
        self.transaction_entries += 1
        self.operations.append("begin")
        try:
            yield self
        except BaseException:
            self.transaction_rollbacks += 1
            self.operations.append("rollback")
            raise
        else:
            self.transaction_commits += 1
            self.operations.append("commit")

    async def execute(self, query: str, *args: object) -> None:
        self.operations.append(f"execute:{query}")
        self.executed_sql.append(query)

    async def fetch(self, query: str, *args: object) -> list[dict[str, object]]:
        self.operations.append(f"fetch:{query}")
        return await self.fetchall(query, args)

    async def fetchval(self, query: str, *args: object) -> None:
        self.operations.append(f"fetchval:{query}")
        self.queries.append(query)

    async def fetchall(
        self,
        query: str,
        args: tuple[object, ...],
    ) -> list[dict[str, object]]:
        self.queries.append(query)
        if "share_token_resource_type_constraints" in query:
            return self.constraint_rows
        if "sharing_schema_issues" in query:
            return []
        return [{"name": table} for table in sorted(_SHARING_TABLES)]


class _GuardedStubPostgresPool(_StubPostgresPool):
    """Exercise migration statements through the managed PostgreSQL guard."""

    def __init__(
        self,
        *,
        constraint_rows: list[dict[str, object]] | None = None,
    ) -> None:
        super().__init__(constraint_rows=constraint_rows)
        self._connection_identity = object()

    async def execute(self, query: str, *args: object) -> None:
        guarded = _guard_sql(
            query,
            backend="postgres",
            connection_identity=self._connection_identity,
            operation="execute",
        )
        await super().execute(guarded, *args)

    async def fetchall(
        self,
        query: str,
        args: tuple[object, ...],
    ) -> list[dict[str, object]]:
        guarded = _guard_sql(
            query,
            backend="postgres",
            connection_identity=self._connection_identity,
            operation="fetchall",
        )
        return await super().fetchall(guarded, args)

    async def fetch(self, query: str, *args: object) -> list[dict[str, object]]:
        guarded = _guard_sql(
            query,
            backend="postgres",
            connection_identity=self._connection_identity,
            operation="fetch",
        )
        return await super().fetch(guarded, *args)

    async def fetchval(self, query: str, *args: object) -> None:
        guarded = _guard_sql(
            query,
            backend="postgres",
            connection_identity=self._connection_identity,
            operation="fetchval",
        )
        await super().fetchval(guarded, *args)


class _FailingCanonicalAddPool(_GuardedStubPostgresPool):
    def __init__(self) -> None:
        super().__init__(
            constraint_rows=[
                {
                    "constraint_name": "legacy_resource_type",
                    "is_validated": True,
                    "normalized_structure": "checkresource_type=anyarray[?,?]",
                    "literal_values": ["chatbook", "workspace"],
                }
            ]
        )
        self.legacy_constraint_present = True

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[_StubPostgresPool]:
        legacy_constraint_present = self.legacy_constraint_present
        try:
            async with super().transaction():
                yield self
        except BaseException:
            self.legacy_constraint_present = legacy_constraint_present
            raise

    async def execute(self, query: str, *args: object) -> None:
        if "DROP CONSTRAINT" in query:
            self.legacy_constraint_present = False
        if "ADD CONSTRAINT ck_share_tokens_resource_type" in query:
            raise DatabaseError("forced canonical constraint failure")
        await super().execute(query, *args)


class _FailingCanonicalValidationPool(_GuardedStubPostgresPool):
    def __init__(self) -> None:
        super().__init__(
            constraint_rows=[
                {
                    "constraint_name": "legacy_resource_type",
                    "is_validated": True,
                    "normalized_structure": "checkresource_type=anyarray[?,?]",
                    "literal_values": ["chatbook", "workspace"],
                }
            ]
        )
        self.legacy_constraint_present = True
        self.canonical_constraint_present = False

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[_StubPostgresPool]:
        state = (
            self.legacy_constraint_present,
            self.canonical_constraint_present,
        )
        try:
            async with super().transaction():
                yield self
        except BaseException:
            (
                self.legacy_constraint_present,
                self.canonical_constraint_present,
            ) = state
            raise

    async def execute(self, query: str, *args: object) -> None:
        await super().execute(query, *args)
        if "DROP CONSTRAINT" in query:
            self.legacy_constraint_present = False
        elif "ADD CONSTRAINT ck_share_tokens_resource_type" in query:
            self.canonical_constraint_present = True
        elif "VALIDATE CONSTRAINT ck_share_tokens_resource_type" in query:
            raise DatabaseError("forced canonical validation failure")


class _StubNonPostgresPool:
    pool = None


class _BrokenPostgresPool(_StubPostgresPool):
    async def execute(self, query: str, *args: object) -> None:
        raise DatabaseError("forced DDL failure")


class _RawBrokenPostgresPool(_StubPostgresPool):
    async def execute(self, query: str, *args: object) -> None:
        raise asyncpg.InsufficientPrivilegeError("forced raw PostgreSQL failure")


class _RawInterfaceBrokenPostgresPool(_StubPostgresPool):
    async def execute(self, query: str, *args: object) -> None:
        raise asyncpg.InterfaceError("forced PostgreSQL interface failure")


class _RawInternalBrokenPostgresPool(_StubPostgresPool):
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
    assert "ADD CONSTRAINT ck_share_tokens_resource_type" in ddl
    assert "VALIDATE CONSTRAINT ck_share_tokens_resource_type" in ddl
    assert any("pg_get_constraintdef" in query for query in pool.queries)
    assert "BOOLEAN NOT NULL DEFAULT TRUE" in ddl
    assert "uq_shared_workspaces_scope" in ddl
    assert "uq_sharing_config_scope_key" in ddl
    assert "uq_sharing_config_global_key" in ddl
    assert "WHERE scope_id IS NULL" in ddl
    assert "idx_share_tokens_resource" in ddl
    assert "idx_share_audit_owner" in ddl


@pytest.mark.asyncio
async def test_ensure_sharing_tables_pg_uses_guard_verifiable_sql() -> None:
    """Startup DDL must pass the managed PostgreSQL users-write firewall."""
    pool = _GuardedStubPostgresPool()

    assert await pg_migrations_extra.ensure_sharing_tables_pg(pool) is True


@pytest.mark.asyncio
async def test_ensure_sharing_tables_pg_serializes_one_transaction() -> None:
    pool = _GuardedStubPostgresPool()

    assert await pg_migrations_extra.ensure_sharing_tables_pg(pool) is True
    assert pool.transaction_entries == 1
    assert pool.transaction_commits == 1
    assert pool.transaction_rollbacks == 0
    lock_index = next(
        index
        for index, operation in enumerate(pool.operations)
        if "pg_advisory_xact_lock" in operation
    )
    ddl_index = next(
        index
        for index, operation in enumerate(pool.operations)
        if "CREATE TABLE IF NOT EXISTS shared_workspaces" in operation
    )
    assert lock_index < ddl_index


@pytest.mark.asyncio
async def test_ensure_sharing_tables_pg_rolls_back_failed_constraint_repair() -> None:
    pool = _FailingCanonicalAddPool()

    assert await pg_migrations_extra.ensure_sharing_tables_pg(pool) is False
    assert pool.transaction_entries == 1
    assert pool.transaction_commits == 0
    assert pool.transaction_rollbacks == 1
    assert any("DROP CONSTRAINT" in sql for sql in pool.executed_sql)
    assert pool.legacy_constraint_present is True


@pytest.mark.asyncio
async def test_ensure_sharing_tables_pg_rolls_back_failed_constraint_validation() -> None:
    pool = _FailingCanonicalValidationPool()

    assert await pg_migrations_extra.ensure_sharing_tables_pg(pool) is False
    assert pool.transaction_entries == 1
    assert pool.transaction_commits == 0
    assert pool.transaction_rollbacks == 1
    assert pool.legacy_constraint_present is True
    assert pool.canonical_constraint_present is False
    drop_index = next(
        index
        for index, sql in enumerate(pool.executed_sql)
        if "DROP CONSTRAINT" in sql
    )
    add_index = next(
        index
        for index, sql in enumerate(pool.executed_sql)
        if "ADD CONSTRAINT ck_share_tokens_resource_type" in sql
    )
    validate_index = next(
        index
        for index, sql in enumerate(pool.executed_sql)
        if "VALIDATE CONSTRAINT ck_share_tokens_resource_type" in sql
    )
    assert drop_index < add_index < validate_index


@pytest.mark.asyncio
async def test_ensure_sharing_tables_pg_preserves_ready_canonical_constraint() -> None:
    pool = _GuardedStubPostgresPool(
        constraint_rows=[
            {
                "constraint_name": "ck_share_tokens_resource_type",
                "is_validated": True,
                "normalized_structure": "checkresource_type=anyarray[?,?,?]",
                "literal_values": [
                    "chatbook",
                    "workspace",
                    "prototype_workspace",
                ],
            }
        ]
    )

    assert await pg_migrations_extra.ensure_sharing_tables_pg(pool) is True
    assert not any(
        "ALTER TABLE share_tokens ADD CONSTRAINT" in sql
        or "ALTER TABLE share_tokens DROP CONSTRAINT" in sql
        for sql in pool.executed_sql
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("normalized_structure", "literal_values", "is_validated"),
    [
        ("checkother_column=?", ("custom",), True),
        (
            "checkresource_type=anyarray[?,?,?]",
            ("chatbook", "workspace", "prototype_workspace"),
            False,
        ),
    ],
)
async def test_ensure_sharing_tables_pg_preserves_canonical_drift(
    normalized_structure: str,
    literal_values: tuple[str, ...],
    is_validated: bool,
) -> None:
    pool = _GuardedStubPostgresPool(
        constraint_rows=[
            {
                "constraint_name": "ck_share_tokens_resource_type",
                "is_validated": is_validated,
                "normalized_structure": normalized_structure,
                "literal_values": list(literal_values),
            }
        ]
    )

    assert await pg_migrations_extra.ensure_sharing_tables_pg(pool) is True
    assert not any(
        "ALTER TABLE share_tokens ADD CONSTRAINT" in sql
        or "ALTER TABLE share_tokens DROP CONSTRAINT" in sql
        for sql in pool.executed_sql
    )
    assert any(
        "c.conname = 'ck_share_tokens_resource_type'" in query
        for query in pool.queries
    )


@pytest.mark.asyncio
async def test_ensure_sharing_tables_pg_replaces_legacy_quoted_constraint() -> None:
    pool = _GuardedStubPostgresPool(
        constraint_rows=[
            {
                "constraint_name": 'legacy "resource type"',
                "is_validated": True,
                "normalized_structure": "checkresource_type=anyarray[?,?]",
                "literal_values": ["chatbook", "workspace"],
            }
        ]
    )

    assert await pg_migrations_extra.ensure_sharing_tables_pg(pool) is True
    assert any(
        'DROP CONSTRAINT "legacy ""resource type"""' in sql
        for sql in pool.executed_sql
    )
    assert any(
        "ALTER TABLE share_tokens\nADD CONSTRAINT ck_share_tokens_resource_type" in sql
        for sql in pool.executed_sql
    )
    assert any("NOT VALID" in sql for sql in pool.executed_sql)
    assert any(
        "VALIDATE CONSTRAINT ck_share_tokens_resource_type" in sql
        for sql in pool.executed_sql
    )


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
