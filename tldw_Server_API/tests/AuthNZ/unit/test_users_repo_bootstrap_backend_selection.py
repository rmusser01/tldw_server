from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from tldw_Server_API.app.core.AuthNZ.repos.users_repo import AuthnzUsersRepo


class _Tx:
    def __init__(self, conn: Any, *, exit_error: Exception | None = None) -> None:
        self._conn = conn
        self._exit_error = exit_error

    async def __aenter__(self) -> Any:
        return self._conn

    async def __aexit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001, ARG002
        if exc_type is None and self._exit_error is not None:
            raise self._exit_error
        return False


class _PoolStub:
    def __init__(
        self,
        conn: Any,
        *,
        postgres: bool,
        exit_error: Exception | None = None,
    ) -> None:
        self._conn = conn
        self.pool = object() if postgres else None
        self._exit_error = exit_error

    def transaction(self) -> _Tx:
        return _Tx(self._conn, exit_error=self._exit_error)


class _CursorStub:
    def __init__(
        self,
        row: Any = None,
        *,
        rows: list[Any] | None = None,
        rowcount: int | None = None,
        lastrowid: int | None = None,
    ) -> None:
        self._row = row
        self._rows = rows or []
        self.rowcount = rowcount
        self.lastrowid = lastrowid

    async def fetchone(self) -> Any:
        return self._row

    async def fetchall(self) -> list[Any]:
        return self._rows


class _SQLiteConnWithFetchvalTrap:
    def __init__(self, *, role_row: Any = None) -> None:
        self.role_row = role_row
        self.execute_calls: list[tuple[str, Any]] = []
        self.committed = False

    async def execute(self, query: str, params: Any) -> _CursorStub:
        self.execute_calls.append((str(query), params))
        q = str(query).lower()
        if "select id from roles" in q:
            return _CursorStub(self.role_row)
        if q.lstrip().startswith("with target_user as"):
            user_id = int(params[0])
            return _CursorStub(
                rows=[
                    (
                        "user",
                        user_id,
                        "2026-07-26T12:00:00.000000Z",
                    )
                ]
            )
        if "insert or ignore into main.users" in q:
            return _CursorStub(rowcount=1, lastrowid=int(params[0]))
        return _CursorStub(rowcount=1)

    async def commit(self) -> None:
        self.committed = True

    async def fetchval(self, *args, **kwargs):  # noqa: ANN001, ANN002, ARG002
        raise AssertionError("SQLite backend path should not call conn.fetchval")


class _PostgresConnWithSqliteTrap:
    def __init__(self, *, role_id: int | None = None) -> None:
        self.role_id = role_id
        self.execute_calls: list[tuple[str, tuple[Any, ...]]] = []
        self.fetchval_calls: list[tuple[str, tuple[Any, ...]]] = []

    async def execute(self, query: str, *params: Any) -> str:
        lower_q = str(query).lower()
        if "insert or ignore" in lower_q:
            raise AssertionError("Postgres backend path should not use SQLite INSERT OR IGNORE SQL")
        self.execute_calls.append((str(query), tuple(params)))
        return "UPDATE 1"

    async def fetchval(self, query: str, *params: Any) -> int | None:
        self.fetchval_calls.append((str(query), tuple(params)))
        if "insert into public.users" in str(query).lower():
            return int(params[0])
        return self.role_id

    async def fetch(self, query: str, *params: Any) -> list[dict[str, Any]]:
        assert "locked_user" in str(query).lower()
        return [
            {
                "source_tag": "user",
                "source_id": int(params[0]),
                "candidate_value": datetime(2026, 7, 26, 12, tzinfo=timezone.utc),
            }
        ]


@pytest.mark.asyncio
async def test_ensure_single_user_admin_user_sqlite_backend_selection_uses_execute():
    conn = _SQLiteConnWithFetchvalTrap()
    repo = AuthnzUsersRepo(db_pool=_PoolStub(conn, postgres=False))

    await repo.ensure_single_user_admin_user(user_id=321)

    assert conn.committed is False
    assert conn.execute_calls
    first_q = conn.execute_calls[0][0].lower()
    update_q = next(
        query.lower()
        for query, _ in conn.execute_calls
        if "update main.users set role = 'admin'" in query.lower()
    )
    assert "insert or ignore into main.users" in first_q
    assert "update main.users set role = 'admin'" in update_q


@pytest.mark.asyncio
async def test_ensure_single_user_admin_user_postgres_backend_selection_uses_pg_sql():
    conn = _PostgresConnWithSqliteTrap()
    repo = AuthnzUsersRepo(db_pool=_PoolStub(conn, postgres=True))

    await repo.ensure_single_user_admin_user(user_id=654)

    assert conn.execute_calls
    assert "on conflict do nothing" in conn.fetchval_calls[0][0].lower()
    update_query = next(
        query.lower()
        for query, _ in conn.execute_calls
        if "update public.users set role = 'admin'" in query.lower()
    )
    assert "where id = $1" in update_query


@pytest.mark.asyncio
async def test_assign_role_if_missing_sqlite_backend_selection_uses_execute():
    conn = _SQLiteConnWithFetchvalTrap(role_row=(7,))
    repo = AuthnzUsersRepo(db_pool=_PoolStub(conn, postgres=False))

    await repo.assign_role_if_missing(user_id=11, role_name="admin")

    assert conn.committed is False
    all_queries = " ".join(q.lower() for q, _ in conn.execute_calls)
    assert "select id from roles where name = ?" in all_queries
    assert "insert or ignore into user_roles" in all_queries


@pytest.mark.asyncio
async def test_assign_role_if_missing_sqlite_propagates_transaction_exit_failure():
    conn = _SQLiteConnWithFetchvalTrap(role_row=(7,))
    repo = AuthnzUsersRepo(
        db_pool=_PoolStub(
            conn,
            postgres=False,
            exit_error=RuntimeError("commit failed"),
        )
    )

    with pytest.raises(RuntimeError, match="commit failed"):
        await repo.assign_role_if_missing(user_id=11, role_name="admin")


@pytest.mark.asyncio
async def test_assign_role_if_missing_postgres_backend_selection_uses_fetchval():
    conn = _PostgresConnWithSqliteTrap(role_id=9)
    repo = AuthnzUsersRepo(db_pool=_PoolStub(conn, postgres=True))

    await repo.assign_role_if_missing(user_id=22, role_name="admin")

    assert conn.fetchval_calls, "expected Postgres fetchval path to be used"
    assert "select id from roles where name = $1" in conn.fetchval_calls[0][0].lower()
    assert conn.execute_calls
    assert "insert into user_roles" in conn.execute_calls[0][0].lower()


@pytest.mark.asyncio
async def test_ensure_single_user_admin_user_sqlite_propagates_transaction_exit_failure():
    conn = _SQLiteConnWithFetchvalTrap()
    repo = AuthnzUsersRepo(
        db_pool=_PoolStub(
            conn,
            postgres=False,
            exit_error=RuntimeError("commit failed"),
        )
    )

    with pytest.raises(RuntimeError, match="commit failed"):
        await repo.ensure_single_user_admin_user(user_id=321)


@pytest.mark.asyncio
async def test_remove_role_if_present_sqlite_propagates_transaction_exit_failure():
    class _DeleteConn(_SQLiteConnWithFetchvalTrap):
        async def execute(self, query: str, params: Any) -> _CursorStub:
            self.execute_calls.append((str(query), params))
            q = str(query).lower()
            if "select id from roles" in q:
                return _CursorStub((7,))
            if "delete from user_roles" in q:
                return _CursorStub(rowcount=1)
            return _CursorStub()

    conn = _DeleteConn(role_row=(7,))
    repo = AuthnzUsersRepo(
        db_pool=_PoolStub(
            conn,
            postgres=False,
            exit_error=RuntimeError("commit failed"),
        )
    )

    with pytest.raises(RuntimeError, match="commit failed"):
        await repo.remove_role_if_present(user_id=11, role_name="admin")


@pytest.mark.asyncio
async def test_remove_role_if_present_postgres_locks_user_before_revocation():
    class _Connection:
        def __init__(self) -> None:
            self.calls: list[tuple[str, tuple[Any, ...]]] = []

        async def fetchrow(self, query: str, *params: Any) -> dict[str, int] | None:
            statement = str(query)
            self.calls.append((statement, tuple(params)))
            if statement == "SELECT id FROM public.users WHERE id = $1 FOR UPDATE":
                return {"id": int(params[0])}
            if "DELETE FROM public.user_roles" in statement:
                return {"user_id": int(params[0])}
            raise AssertionError(statement)

    conn = _Connection()
    repo = AuthnzUsersRepo(db_pool=_PoolStub(conn, postgres=True))

    assert await repo.remove_role_if_present(user_id=11, role_name="admin")
    assert conn.calls == [
        ("SELECT id FROM public.users WHERE id = $1 FOR UPDATE", (11,)),
        (
            "DELETE FROM public.user_roles ur "
            "USING public.roles r "
            "WHERE ur.role_id = r.id "
            "AND ur.user_id = $1 "
            "AND r.name = $2 "
            "RETURNING ur.user_id",
            (11, "admin"),
        ),
    ]
