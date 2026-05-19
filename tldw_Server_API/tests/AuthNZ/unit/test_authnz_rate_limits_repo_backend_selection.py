from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from tldw_Server_API.app.core.AuthNZ.repos.rate_limits_repo import AuthnzRateLimitsRepo


class _Tx:
    def __init__(self, conn: Any) -> None:
        self._conn = conn

    async def __aenter__(self) -> Any:
        return self._conn

    async def __aexit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001, ARG002
        return False


class _Acquire:
    def __init__(self, conn: Any) -> None:
        self._conn = conn

    async def __aenter__(self) -> Any:
        return self._conn

    async def __aexit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001, ARG002
        return False


class _PoolStub:
    def __init__(self, conn: Any, *, postgres: bool) -> None:
        self._conn = conn
        self.pool = object() if postgres else None

    def transaction(self) -> _Tx:
        return _Tx(self._conn)

    def acquire(self) -> _Acquire:
        return _Acquire(self._conn)


class _Cursor:
    def __init__(self, *, row: Any = None, rowcount: int = 1) -> None:
        self._row = row
        self.rowcount = rowcount

    async def fetchone(self) -> Any:
        return self._row


class _SqliteConnWithPgTrap:
    def __init__(self) -> None:
        self.execute_calls: list[tuple[str, Any]] = []
        self.commit_calls = 0

    async def fetchrow(self, *args, **kwargs):  # noqa: ANN001, ANN002, ARG002
        raise AssertionError("SQLite backend path should not call conn.fetchrow")

    async def fetchval(self, *args, **kwargs):  # noqa: ANN001, ANN002, ARG002
        raise AssertionError("SQLite backend path should not call conn.fetchval")

    async def execute(self, query: str, params: Any) -> _Cursor:
        self.execute_calls.append((str(query), params))
        lower_q = str(query).lower()
        if "select request_count" in lower_q:
            return _Cursor(row=(1,))
        if "select attempt_count" in lower_q:
            return _Cursor(row=(1,))
        if "select locked_until" in lower_q:
            return _Cursor(row=None)
        return _Cursor(rowcount=1)

    async def commit(self) -> None:
        self.commit_calls += 1


class _PostgresConnWithSqliteTrap:
    def __init__(self) -> None:
        self.fetchval_calls: list[tuple[str, tuple[Any, ...]]] = []
        self.fetchrow_calls: list[tuple[str, tuple[Any, ...]]] = []
        self.execute_calls: list[tuple[str, tuple[Any, ...]]] = []

    async def fetchval(self, query: str, *params: Any) -> int:
        lower_q = str(query).lower()
        if "?" in lower_q:
            raise AssertionError("Postgres backend path should not use SQLite placeholders")
        self.fetchval_calls.append((str(query), tuple(params)))
        return 3

    async def fetchrow(self, query: str, *params: Any) -> dict[str, Any] | None:
        lower_q = str(query).lower()
        if "?" in lower_q:
            raise AssertionError("Postgres backend path should not use SQLite placeholders")
        self.fetchrow_calls.append((str(query), tuple(params)))
        if "select locked_until" in lower_q:
            return {"locked_until": datetime(2026, 2, 8, 12, 0, 0)}
        return None

    async def execute(self, query: str, *params: Any) -> str:
        lower_q = str(query).lower()
        if "?" in lower_q:
            raise AssertionError("Postgres backend path should not use SQLite placeholders")
        self.execute_calls.append((str(query), tuple(params)))
        return "DELETE 0"


@pytest.mark.asyncio
async def test_increment_rate_limit_window_sqlite_backend_selection_uses_execute():
    conn = _SqliteConnWithPgTrap()
    repo = AuthnzRateLimitsRepo(db_pool=_PoolStub(conn, postgres=False))
    now = datetime.now(timezone.utc).replace(microsecond=0)

    count = await repo.increment_rate_limit_window(
        identifier="id-1",
        endpoint="/api/test",
        window_start=now,
    )

    assert count == 2
    assert conn.execute_calls
    assert "select request_count" in conn.execute_calls[0][0].lower()
    assert "update rate_limits" in conn.execute_calls[1][0].lower()


@pytest.mark.asyncio
async def test_increment_rate_limit_window_postgres_backend_selection_uses_fetchval():
    conn = _PostgresConnWithSqliteTrap()
    repo = AuthnzRateLimitsRepo(db_pool=_PoolStub(conn, postgres=True))
    now = datetime.now(timezone.utc).replace(microsecond=0)

    count = await repo.increment_rate_limit_window(
        identifier="id-1",
        endpoint="/api/test",
        window_start=now,
    )

    assert count == 3
    assert conn.fetchval_calls
    query, _ = conn.fetchval_calls[0]
    assert "on conflict (identifier, endpoint, window_start)" in query.lower()
    assert "returning request_count" in query.lower()


@pytest.mark.asyncio
async def test_record_failed_attempt_and_lockout_sqlite_backend_selection_uses_execute():
    conn = _SqliteConnWithPgTrap()
    repo = AuthnzRateLimitsRepo(db_pool=_PoolStub(conn, postgres=False))
    now = datetime.now(timezone.utc).replace(microsecond=0)

    result = await repo.record_failed_attempt_and_lockout(
        identifier="id-2",
        attempt_type="login",
        now=now,
        lockout_threshold=1,
        lockout_duration_minutes=5,
    )

    assert result["attempt_count"] == 1
    assert result["is_locked"] is True
    lockout_queries = [
        q for q, _ in conn.execute_calls if "insert into account_lockouts" in q.lower()
    ]
    assert lockout_queries
    assert "attempt_type" in lockout_queries[0].lower()
    assert "values (?, ?, ?, ?)" in lockout_queries[0].lower()


@pytest.mark.asyncio
async def test_get_active_lockout_postgres_backend_selection_uses_fetchrow():
    conn = _PostgresConnWithSqliteTrap()
    repo = AuthnzRateLimitsRepo(db_pool=_PoolStub(conn, postgres=True))
    now = datetime.now(timezone.utc).replace(microsecond=0)

    locked_until = await repo.get_active_lockout(identifier="id-3", now=now)

    assert isinstance(locked_until, datetime)
    assert conn.fetchrow_calls
    query, params = conn.fetchrow_calls[0]
    assert "where identifier = $1 and attempt_type = $2 and locked_until > $3" in query.lower()
    assert params[:2] == ("id-3", "login")
