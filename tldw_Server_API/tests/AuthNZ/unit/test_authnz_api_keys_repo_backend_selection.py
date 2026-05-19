from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from tldw_Server_API.app.core.AuthNZ.repos.api_keys_repo import AuthnzApiKeysRepo


class _Tx:
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


class _SqliteCursor:
    def __init__(self, rowcount: int) -> None:
        self.rowcount = rowcount


class _SqliteConnWithPgTrap:
    def __init__(self, *, rowcount: int = 1) -> None:
        self.rowcount = rowcount
        self.execute_calls: list[tuple[str, Any]] = []

    async def fetchrow(self, *args, **kwargs):  # noqa: ANN001, ANN002, ARG002
        raise AssertionError("SQLite backend path should not call conn.fetchrow")

    async def execute(self, query: str, params: Any) -> _SqliteCursor:
        self.execute_calls.append((str(query), params))
        return _SqliteCursor(self.rowcount)


class _PostgresConnWithSqliteTrap:
    def __init__(self, *, revoke_status: str = "UPDATE 1") -> None:
        self.revoke_status = revoke_status
        self.execute_calls: list[tuple[str, tuple[Any, ...]]] = []

    async def execute(self, query: str, *params: Any) -> str:
        lower_q = str(query).lower()
        if "?" in lower_q:
            raise AssertionError("Postgres backend path should not use SQLite placeholders")
        self.execute_calls.append((str(query), tuple(params)))
        if "where id = $5 and user_id = $6" in lower_q:
            return self.revoke_status
        if "insert into api_key_audit_log" in lower_q:
            return "INSERT 0 1"
        return "UPDATE 1"


@pytest.mark.asyncio
async def test_revoke_api_key_for_user_sqlite_backend_selection_uses_execute():
    conn = _SqliteConnWithPgTrap(rowcount=1)
    repo = AuthnzApiKeysRepo(db_pool=_PoolStub(conn, postgres=False))

    revoked = await repo.revoke_api_key_for_user(
        key_id=10,
        user_id=7,
        revoked_status="revoked",
        active_status="active",
        reason="unit-test",
        revoked_at=datetime.now(timezone.utc),
    )

    assert revoked is True
    assert conn.execute_calls
    query, params = conn.execute_calls[0]
    assert "where id = ? and user_id = ? and status = ?" in query.lower()
    assert isinstance(params, tuple)


@pytest.mark.asyncio
async def test_revoke_api_key_for_user_postgres_backend_selection_uses_dollar_params():
    conn = _PostgresConnWithSqliteTrap(revoke_status="UPDATE 1")
    repo = AuthnzApiKeysRepo(db_pool=_PoolStub(conn, postgres=True))

    revoked = await repo.revoke_api_key_for_user(
        key_id=10,
        user_id=7,
        revoked_status="revoked",
        active_status="active",
        reason="unit-test",
        revoked_at=datetime.now(timezone.utc),
    )

    assert revoked is True
    assert conn.execute_calls
    query, params = conn.execute_calls[0]
    assert "where id = $5 and user_id = $6 and status = $7" in query.lower()
    assert len(params) == 7


@pytest.mark.asyncio
async def test_mark_key_expired_postgres_backend_selection_uses_dollar_params():
    conn = _PostgresConnWithSqliteTrap()
    repo = AuthnzApiKeysRepo(db_pool=_PoolStub(conn, postgres=True))

    await repo.mark_key_expired(key_id=3, expired_status="expired")

    assert conn.execute_calls
    query, params = conn.execute_calls[0]
    assert "update api_keys set status = $1 where id = $2" in query.lower()
    assert params == ("expired", 3)


@pytest.mark.asyncio
async def test_upsert_primary_key_sqlite_uses_stable_key_identifier_when_available():
    conn = _SqliteConnWithPgTrap(rowcount=1)
    repo = AuthnzApiKeysRepo(db_pool=_PoolStub(conn, postgres=False))

    await repo.upsert_primary_key(
        user_id=7,
        key_hash="salted-hash",
        key_identifier="abcdef123456",
        key_prefix="tldw_abc...",
        name="primary",
        description="primary key",
        scope="admin",
    )

    query, params = conn.execute_calls[0]
    assert "where key_id = ?" in query.lower()
    assert params[0] == "abcdef123456"
    assert params[3] == "abcdef123456"


@pytest.mark.asyncio
async def test_upsert_primary_key_postgres_uses_stable_key_identifier_when_available():
    conn = _PostgresConnWithSqliteTrap()
    repo = AuthnzApiKeysRepo(db_pool=_PoolStub(conn, postgres=True))

    await repo.upsert_primary_key(
        user_id=7,
        key_hash="salted-hash",
        key_identifier="abcdef123456",
        key_prefix="tldw_abc...",
        name="primary",
        description="primary key",
        scope="admin",
    )

    query, params = conn.execute_calls[0]
    assert "where key_id = $8" in query.lower()
    assert "on conflict" not in query.lower()
    assert params[-1] == "abcdef123456"


@pytest.mark.asyncio
async def test_insert_audit_log_sqlite_backend_selection_uses_execute():
    conn = _SqliteConnWithPgTrap(rowcount=1)
    repo = AuthnzApiKeysRepo(db_pool=_PoolStub(conn, postgres=False))

    await repo.insert_audit_log(
        key_id=5,
        action="rotate",
        user_id=7,
        details={"reason": "unit"},
    )

    assert conn.execute_calls
    query, params = conn.execute_calls[0]
    assert "insert into api_key_audit_log" in query.lower()
    assert "values (?, ?, ?, ?)" in query.lower()
    assert isinstance(params, tuple)


@pytest.mark.asyncio
async def test_insert_audit_log_postgres_backend_selection_uses_dollar_params():
    conn = _PostgresConnWithSqliteTrap()
    repo = AuthnzApiKeysRepo(db_pool=_PoolStub(conn, postgres=True))

    await repo.insert_audit_log(
        key_id=5,
        action="rotate",
        user_id=7,
        details={"reason": "unit"},
    )

    assert conn.execute_calls
    query, params = conn.execute_calls[0]
    assert "insert into api_key_audit_log" in query.lower()
    assert "values ($1, $2, $3, $4::jsonb)" in query.lower()
    assert params[:3] == (5, "rotate", 7)
