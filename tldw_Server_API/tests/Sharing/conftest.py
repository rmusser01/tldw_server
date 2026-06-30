"""Shared fixtures for Sharing module tests."""
from __future__ import annotations

import sqlite3

import pytest

from tldw_Server_API.app.core.AuthNZ.migrations import (
    migration_001_create_users_table,
    migration_077_create_sharing_tables,
    migration_087_expand_share_tokens_resource_type_for_prototypes,
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


@pytest.fixture
def sharing_db():
    """In-memory SQLite database with sharing + users tables."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    migration_001_create_users_table(conn)
    # Seed a test user
    conn.execute(
        "INSERT INTO users (id, username, email, password_hash) VALUES (1, 'alice', 'alice@test.com', 'hash')"
    )
    conn.execute(
        "INSERT INTO users (id, username, email, password_hash) VALUES (2, 'bob', 'bob@test.com', 'hash')"
    )
    conn.commit()
    migration_077_create_sharing_tables(conn)
    migration_087_expand_share_tokens_resource_type_for_prototypes(conn)
    yield conn
    conn.close()


@pytest.fixture
def fake_pool(sharing_db):
    """FakePool wrapping the in-memory sharing DB."""
    return _FakePool(sharing_db)


@pytest.fixture
def repo(fake_pool):
    """SharedWorkspaceRepo using the fake pool."""
    from tldw_Server_API.app.core.AuthNZ.repos.shared_workspace_repo import SharedWorkspaceRepo
    return SharedWorkspaceRepo(db_pool=fake_pool)
