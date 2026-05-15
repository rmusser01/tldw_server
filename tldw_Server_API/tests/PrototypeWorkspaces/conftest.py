"""Shared fixtures for PrototypeWorkspaces repository tests."""
from __future__ import annotations

import sqlite3
from contextlib import asynccontextmanager

import pytest

from tldw_Server_API.app.core.AuthNZ.migrations import (
    migration_001_create_users_table,
    migration_086_create_prototype_workspace_tables,
)


class _FakePool:
    """Minimal DatabasePool stand-in backed by an in-memory SQLite connection."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    async def execute(self, sql: str, params: tuple = ()) -> None:
        self._conn.execute(sql, params)
        self._conn.commit()

    async def fetchone(self, sql: str, params: tuple = ()) -> dict | None:
        cur = self._conn.execute(sql, params)
        row = cur.fetchone()
        if row is None:
            return None
        cols = [d[0] for d in cur.description]
        return dict(zip(cols, row, strict=True))

    async def fetchall(self, sql: str, params: tuple = ()) -> list[dict]:
        cur = self._conn.execute(sql, params)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row, strict=True)) for row in rows]

    @asynccontextmanager
    async def transaction(self):
        """Yield a no-autocommit adapter over the in-memory SQLite connection."""

        class _TxConn:
            def __init__(self, conn: sqlite3.Connection) -> None:
                self._conn = conn

            async def execute(self, sql: str, params: tuple = ()):
                return self._conn.execute(sql, params)

            async def fetchone(self, sql: str, params: tuple = ()) -> dict | None:
                cur = self._conn.execute(sql, params)
                row = cur.fetchone()
                if row is None:
                    return None
                cols = [d[0] for d in cur.description]
                return dict(zip(cols, row, strict=True))

            async def fetchall(self, sql: str, params: tuple = ()) -> list[dict]:
                cur = self._conn.execute(sql, params)
                rows = cur.fetchall()
                cols = [d[0] for d in cur.description]
                return [dict(zip(cols, row, strict=True)) for row in rows]

        self._conn.execute("BEGIN")
        try:
            yield _TxConn(self._conn)
        except Exception:
            self._conn.rollback()
            raise
        else:
            self._conn.commit()


@pytest.fixture
def prototype_db():
    """In-memory SQLite database with users + prototype metadata tables."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    migration_001_create_users_table(conn)
    conn.execute(
        "INSERT INTO users (id, username, email, password_hash) VALUES (1, 'owner', 'owner@test.com', 'hash')"
    )
    conn.execute(
        "INSERT INTO users (id, username, email, password_hash) VALUES (2, 'collab', 'collab@test.com', 'hash')"
    )
    conn.commit()
    migration_086_create_prototype_workspace_tables(conn)
    yield conn
    conn.close()


@pytest.fixture
def fake_pool(prototype_db):
    """FakePool wrapping the in-memory prototype DB."""
    return _FakePool(prototype_db)


@pytest.fixture
def repo(fake_pool):
    """PrototypeWorkspacesRepo using the fake pool."""
    from tldw_Server_API.app.core.AuthNZ.repos.prototype_workspaces_repo import (
        PrototypeWorkspacesRepo,
    )

    return PrototypeWorkspacesRepo(db_pool=fake_pool)
