"""Guard the VN SQL boundary and caller-owned transaction/runtime isolation."""

from __future__ import annotations

import contextlib
import inspect
import sqlite3
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import aiosqlite
import pytest

from tldw_Server_API.app.core.AuthNZ.repos.generated_files_repo import AuthnzGeneratedFilesRepo
from tldw_Server_API.tests.AuthNZ.integration import test_vn_generated_file_idempotency as durability

pytestmark = pytest.mark.unit


class _DBOwnedConnection:
    """Reject VN SQL emitted outside DB_Management on the borrowed connection."""

    def __init__(self) -> None:
        """Record statements and their bound values for boundary assertions."""
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    async def fetchrow(self, query: str, *params: Any) -> dict[str, Any] | None:
        """Check SQL ownership and return a representative generated-file row."""
        caller = inspect.currentframe().f_back
        assert ".DB_Management." in caller.f_globals["__name__"], "VN SQL must be DB-owned"
        self.calls.append((" ".join(query.split()), params))
        if "FROM generated_files" in query:
            return {"id": 7, "user_id": 5, "tags": '["vn"]', "is_deleted": False}
        return None


class _OwningPool:
    """Only the outer owner may enter a transaction or borrow from the pool."""

    def __init__(self, conn: _DBOwnedConnection) -> None:
        """Expose a PostgreSQL owner while counting outer transaction entries."""
        self.conn = conn
        self.pool = object()
        self.transactions = 0

    @contextlib.asynccontextmanager
    async def transaction(self) -> AsyncIterator[_DBOwnedConnection]:
        """Yield the one owning connection without simulating nested commits."""
        self.transactions += 1
        yield self.conn

    @contextlib.asynccontextmanager
    async def acquire(self) -> AsyncIterator[_DBOwnedConnection]:
        """Reject any attempt to bypass the transaction-bound pool."""
        raise AssertionError("VN work must borrow the bound transaction, not the outer pool")
        yield self.conn


@pytest.mark.asyncio
async def test_vn_locks_and_replay_queries_are_db_owned_on_one_transaction() -> None:
    """Catch repository-owned SQL and independent connection/transaction use."""
    conn = _DBOwnedConnection()
    pool = _OwningPool(conn)
    repo = AuthnzGeneratedFilesRepo(pool)
    async with repo.vn_item_transaction(user_id=5, source_ref="vn_asset_item:42") as bound:
        await bound.lock_quota_scopes(user_id=5, org_id=51, team_id=52)
        replay = await bound.create_file(
            user_id=5, filename="loser.png", storage_path="vn_assets/loser.png",
            file_category="image", source_feature="vn_assets", source_ref="vn_asset_item:42",
        )
        by_ref = await bound.get_file_by_source_ref(
            user_id=5, source_feature="vn_assets", source_ref="vn_asset_item:42",
        )
        by_path = await bound.get_live_file_by_storage_path(user_id=5, storage_path="vn_assets/live.png")

    assert pool.transactions == 1
    assert replay == {**by_ref, "_idempotent_replay": True}
    assert by_path == by_ref == {"id": 7, "user_id": 5, "tags": ["vn"], "is_deleted": False}
    assert [query for query, _params in conn.calls[:4]] == [
        "SELECT pg_advisory_xact_lock($1)",
        "SELECT id FROM users WHERE id = $1 FOR UPDATE",
        "SELECT id FROM storage_quotas WHERE org_id = $1 FOR UPDATE",
        "SELECT id FROM storage_quotas WHERE team_id = $1 FOR UPDATE",
    ]
    assert [params for _query, params in conn.calls[1:4]] == [(5,), (51,), (52,)]
    assert conn.calls[0] == conn.calls[4]
    assert conn.calls[-2][1] == (5, "vn_assets", "vn_asset_item:42")
    assert conn.calls[-1][1] == (5, "vn_assets/live.png")


@pytest.mark.asyncio
async def test_db_lookup_scopes_real_sqlite_rows_without_committing() -> None:
    """Catch missing live/owner/feature filters, interpolation and hidden commits."""
    from tldw_Server_API.app.core.DB_Management.vn_generated_files_queries import VNGeneratedFilesQueries

    async with aiosqlite.connect(":memory:") as conn:
        await conn.execute("""
            CREATE TABLE generated_files (
                id INTEGER PRIMARY KEY, user_id INTEGER, source_feature TEXT,
                source_ref TEXT, storage_path TEXT, is_deleted INTEGER
            )
        """)
        await conn.executemany("INSERT INTO generated_files VALUES (?, ?, ?, ?, ?, ?)", [
            (1, 5, "vn_assets", "vn_asset_item:42", "vn_assets/live.png", 0),
            (2, 5, "vn_assets", "vn_asset_item:42", "vn_assets/live.png", 0),
            (3, 6, "vn_assets", "vn_asset_item:42", "vn_assets/live.png", 0),
            (4, 5, "image_gen", "vn_asset_item:42", "images/live.png", 0),
            (5, 5, "vn_assets", "vn_asset_item:42", "vn_assets/live.png", 1),
        ])
        await conn.commit()
        await conn.execute("BEGIN IMMEDIATE")
        queries = VNGeneratedFilesQueries(conn, postgres=False)
        await queries.lock_item(user_id=5, source_ref="vn_asset_item:42")
        await queries.lock_quota_scopes(user_id=5, org_id=51, team_id=52)
        by_ref = await queries.find_live_by_source_ref(
            user_id=5, source_feature="vn_assets", source_ref="vn_asset_item:42",
        )
        by_path = await queries.find_live_by_storage_path(user_id=5, storage_path="vn_assets/live.png")
        assert by_ref["id"] == by_path["id"] == 2
        assert await queries.find_live_by_source_ref(
            user_id=5, source_feature="vn_assets", source_ref="vn_asset_item:42' OR 1=1 --",
        ) is None
        assert await queries.find_live_by_storage_path(user_id=5, storage_path="missing' OR 1=1 --") is None
        assert conn.in_transaction
        await conn.rollback()


@pytest.mark.asyncio
@pytest.mark.parametrize("lookup", ["source_ref", "storage_path"])
@pytest.mark.parametrize("outcome", ["hit", "miss", "read_error"])
async def test_sqlite_lookup_closes_cursor_without_owning_transaction(
    lookup: str, outcome: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch leaked real cursors and stolen caller transaction ownership.

    Args:
        lookup (str): Source-reference or storage-path lookup to exercise.
        outcome (str): Matching row, missing row, or native row-read failure.
        monkeypatch (pytest.MonkeyPatch): Capture the real lookup cursor only.

    Returns:
        None: The cursor is unusable but the caller can read, write and roll back.
    """
    from tldw_Server_API.app.core.DB_Management.vn_generated_files_queries import VNGeneratedFilesQueries

    cursors: list[aiosqlite.Cursor] = []
    fetchone = aiosqlite.Cursor.fetchone
    read_error = sqlite3.OperationalError("injected row-read failure")

    async def capture_fetchone(cursor: aiosqlite.Cursor) -> Any:
        """Observe the actual resource while retaining its native read operation.

        Args:
            cursor (aiosqlite.Cursor): Real cursor returned by SQLite execution.

        Returns:
            Any: The unchanged native row, or None when the lookup misses.
        """
        cursors.append(cursor)
        return await fetchone(cursor)

    def fail_read(cursor: sqlite3.Cursor, row: tuple[Any, ...]) -> Any:
        """Raise at SQLite's row-materialization boundary, keeping its cursor real.

        Args:
            cursor (sqlite3.Cursor): Native SQLite cursor performing the read.
            row (tuple[Any, ...]): Native row that would otherwise be returned.

        Returns:
            Any: Never returns; propagates the original native read error.
        """
        raise read_error

    async with aiosqlite.connect(":memory:") as conn:
        async with conn.execute("""
            CREATE TABLE generated_files (
                id INTEGER PRIMARY KEY, user_id INTEGER, source_feature TEXT,
                source_ref TEXT, storage_path TEXT, is_deleted INTEGER
            )
        """):
            pass
        await conn.commit()
        async with conn.execute("BEGIN IMMEDIATE"):
            pass
        async with conn.execute(
            "INSERT INTO generated_files VALUES (?, ?, ?, ?, ?, ?)",
            (7, 5, "vn_assets", "vn_asset_item:42", "vn_assets/live.png", 0),
        ):
            pass

        queries = VNGeneratedFilesQueries(conn, postgres=False)
        method = getattr(queries, f"find_live_by_{lookup}")
        params = {"user_id": 6 if outcome == "miss" else 5}
        if lookup == "source_ref":
            params.update(source_feature="vn_assets", source_ref="vn_asset_item:42")
        else:
            params.update(storage_path="vn_assets/live.png")
        with monkeypatch.context() as patch:
            patch.setattr(aiosqlite.Cursor, "fetchone", capture_fetchone)
            if outcome == "read_error":
                conn.row_factory = fail_read
            try:
                if outcome == "read_error":
                    with pytest.raises(sqlite3.OperationalError) as caught:
                        await method(**params)
                    assert caught.value is read_error
                else:
                    result = await method(**params)
                    assert result == (None if outcome == "miss" else {
                        "id": 7, "user_id": 5, "source_feature": "vn_assets",
                        "source_ref": "vn_asset_item:42", "storage_path": "vn_assets/live.png",
                        "is_deleted": 0,
                    })
            finally:
                conn.row_factory = None

        assert conn.in_transaction
        async with conn.execute("SELECT id FROM generated_files") as caller_cursor:
            assert await caller_cursor.fetchone() == (7,)
        async with conn.execute(
            "INSERT INTO generated_files VALUES (?, ?, ?, ?, ?, ?)",
            (8, 5, "vn_assets", "vn_asset_item:43", "vn_assets/other.png", 0),
        ):
            pass
        await conn.rollback()
        async with conn.execute("SELECT COUNT(*) FROM generated_files") as caller_cursor:
            assert await caller_cursor.fetchone() == (0,)
        lookup_cursor, = cursors
        with pytest.raises(sqlite3.ProgrammingError, match="closed cursor"):
            await lookup_cursor.execute("SELECT 1")


def test_vn_runtime_derives_pg_config_from_required_isolation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch direct DSN fixtures and leakage of parent pytest runtime flags."""
    database_url = "postgresql://fixture.invalid/tldw_test_owned"
    monkeypatch.setenv("DATABASE_URL", "sqlite:///unrelated.db")
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "parent test")

    class _IsolationRequest:
        """Provide the official fixture's environment and client/name result."""

        def getfixturevalue(self, name: str) -> tuple[None, str]:
            """Reject alternative lifecycle fixtures before setting the owned DSN."""
            assert name == "isolated_test_environment", "Use the required shared fixture"
            monkeypatch.setenv("DATABASE_URL", database_url)
            return None, "tldw_test_owned"

    env = durability._vn_runtime_env(tmp_path, _IsolationRequest(), "postgres")
    assert env["DATABASE_URL"] == database_url
    assert env["TLDW_USER_DB_BACKEND"] == "postgres"
    assert env["AUTH_MODE"] == "single_user"
    assert "TEST_MODE" not in env and "PYTEST_CURRENT_TEST" not in env


def test_vn_sqlite_runtime_does_not_require_postgres(tmp_path: Path) -> None:
    """Keep real SQLite coverage runnable without any PostgreSQL lifecycle."""
    class _NoPostgresRequest:
        """Reject fixture requests that would unnecessarily provision PostgreSQL."""

        def getfixturevalue(self, name: str) -> None:
            """Fail if the SQLite runtime asks for a PostgreSQL fixture."""
            raise AssertionError(f"SQLite must not provision PostgreSQL: {name}")

    env = durability._vn_runtime_env(tmp_path, _NoPostgresRequest(), "sqlite")
    assert env["DATABASE_URL"] == f"sqlite:///{tmp_path / 'users.db'}"
