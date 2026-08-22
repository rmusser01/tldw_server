from __future__ import annotations

import sqlite3
from typing import Any

import pytest

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.api.v1.endpoints import auth
from tldw_Server_API.app.core.AuthNZ.database import DatabasePool


class _AcquireContext:
    def __init__(self, connection: Any) -> None:
        self.connection = connection
        self.entered = 0
        self.exited = 0

    async def __aenter__(self) -> Any:
        self.entered += 1
        return self.connection

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        self.exited += 1
        return False


class _Pool:
    def __init__(self) -> None:
        self.connection = object()
        self.acquire_context = _AcquireContext(self.connection)

    def acquire_statement_autocommit(self) -> _AcquireContext:
        return self.acquire_context

    def transaction(self) -> Any:
        raise AssertionError("login must not hold a request-wide write transaction")


@pytest.mark.asyncio
async def test_login_db_connection_uses_statement_autocommit_without_request_transaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = _Pool()

    async def _get_pool() -> _Pool:
        return pool

    monkeypatch.setattr(auth_deps, "get_db_pool", _get_pool)

    dependency = auth_deps.get_login_db_connection()
    connection = await dependency.__anext__()
    assert connection is pool.connection
    await dependency.aclose()
    assert pool.acquire_context.entered == 1
    assert pool.acquire_context.exited == 1


def _sqlite_pool(db_path: str) -> DatabasePool:
    pool = DatabasePool.__new__(DatabasePool)
    pool.pool = None
    pool.db_path = db_path
    pool._sqlite_uri = False
    pool._initialized = True
    return pool


@pytest.mark.asyncio
async def test_sqlite_statement_autocommit_persists_login_write_without_explicit_commit(
    tmp_path,
) -> None:
    db_path = str(tmp_path / "login-autocommit.db")
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, password_hash TEXT)")
        conn.execute("INSERT INTO users (id, password_hash) VALUES (1, 'old-hash')")

    pool = _sqlite_pool(db_path)
    async with pool.acquire_statement_autocommit() as login_conn:
        await login_conn.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?",
            "new-hash",
            1,
        )

    with sqlite3.connect(db_path) as conn:
        stored_hash = conn.execute(
            "SELECT password_hash FROM users WHERE id = 1"
        ).fetchone()[0]
    assert stored_hash == "new-hash"


@pytest.mark.asyncio
async def test_sqlite_rehash_write_does_not_lock_separate_session_transaction(
    tmp_path,
) -> None:
    db_path = str(tmp_path / "login-rehash.db")
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE users (id INTEGER PRIMARY KEY, password_hash TEXT);
            CREATE TABLE sessions (id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL);
            INSERT INTO users (id, password_hash) VALUES (1, 'legacy-hash');
            """
        )

    pool = _sqlite_pool(db_path)
    async with pool.acquire_statement_autocommit() as login_conn:
        await login_conn.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?",
            "rehash-result",
            1,
        )
        async with pool.transaction() as session_conn:
            await session_conn.execute(
                "INSERT INTO sessions (id, user_id) VALUES (?, ?)",
                7,
                1,
            )

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT users.password_hash, sessions.user_id "
            "FROM users JOIN sessions ON sessions.user_id = users.id"
        ).fetchone()
    assert row == ("rehash-result", 1)


def test_login_route_uses_nontransactional_connection_dependency() -> None:
    dependency_calls = {
        dependency.call
        for dependency in auth.router.routes
        if getattr(dependency, "path", None) == "/auth/login"
        for dependency in dependency.dependant.dependencies
    }

    assert auth_deps.get_login_db_connection in dependency_calls
    assert auth_deps.get_db_transaction not in dependency_calls
