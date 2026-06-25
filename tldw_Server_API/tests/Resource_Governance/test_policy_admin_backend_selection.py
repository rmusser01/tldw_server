from __future__ import annotations

from typing import Any

import pytest

from tldw_Server_API.app.core.Resource_Governance.policy_admin import AuthNZPolicyAdmin


class _TxConn:
    def __init__(self) -> None:
        self.execute_calls: list[tuple[str, tuple[Any, ...]]] = []

    async def execute(self, query: str, *args: Any) -> Any:
        self.execute_calls.append((str(query), tuple(args)))
        return "OK"


class _TxCtx:
    def __init__(self, conn: _TxConn) -> None:
        self._conn = conn

    async def __aenter__(self) -> _TxConn:
        return self._conn

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


class _FakeDbPool:
    def __init__(self, *, is_postgres: bool) -> None:
        self.pool = object() if is_postgres else None
        self._is_sqlite = not is_postgres
        self.tx_conn = _TxConn()
        self.execute_calls: list[tuple[str, tuple[Any, ...]]] = []
        self.fetchone_calls: list[tuple[str, tuple[Any, ...]]] = []

    def transaction(self) -> _TxCtx:
        return _TxCtx(self.tx_conn)

    async def execute(self, query: str, *args: Any) -> Any:
        self.execute_calls.append((str(query), tuple(args)))
        return "OK"

    async def fetchone(self, query: str, *args: Any) -> Any:
        self.fetchone_calls.append((str(query), tuple(args)))
        return None


class _ExistingPolicyDbPool(_FakeDbPool):
    async def fetchone(self, query: str, *args: Any) -> Any:
        self.fetchone_calls.append((str(query), tuple(args)))
        return {"version": 1}


@pytest.mark.asyncio
@pytest.mark.unit
async def test_initialize_sqlite_backend_selection_uses_sqlite_schema() -> None:
    db = _FakeDbPool(is_postgres=False)
    admin = AuthNZPolicyAdmin(db_pool=db)

    await admin.initialize()

    assert admin._initialized is True
    assert db.tx_conn.execute_calls
    assert any("payload text not null" in q.lower() for q, _ in db.tx_conn.execute_calls)
    assert all("jsonb" not in q.lower() for q, _ in db.tx_conn.execute_calls)


@pytest.mark.asyncio
@pytest.mark.unit
async def test_initialize_postgres_backend_selection_uses_postgres_schema() -> None:
    db = _FakeDbPool(is_postgres=True)
    admin = AuthNZPolicyAdmin(db_pool=db)

    await admin.initialize()

    assert admin._initialized is True
    assert db.tx_conn.execute_calls
    assert any("payload jsonb not null" in q.lower() for q, _ in db.tx_conn.execute_calls)
    assert all("autoincrement" not in q.lower() for q, _ in db.tx_conn.execute_calls)


@pytest.mark.asyncio
@pytest.mark.unit
async def test_upsert_policy_sqlite_backend_selection_uses_sqlite_upsert() -> None:
    db = _FakeDbPool(is_postgres=False)
    admin = AuthNZPolicyAdmin(db_pool=db)
    admin._initialized = True

    await admin.upsert_policy("p.sqlite", {"requests": {"rpm": 10}})

    assert db.execute_calls
    query, args = db.execute_calls[0]
    assert "insert or replace into rg_policies" in query.lower()
    assert "?" in query
    assert args and args[0] == "p.sqlite"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_upsert_policy_postgres_backend_selection_uses_postgres_upsert() -> None:
    db = _FakeDbPool(is_postgres=True)
    admin = AuthNZPolicyAdmin(db_pool=db)
    admin._initialized = True

    await admin.upsert_policy("p.pg", {"requests": {"rpm": 11}})

    assert db.execute_calls
    query, args = db.execute_calls[0]
    assert "on conflict (id) do update" in query.lower()
    assert "$1" in query and "$2::jsonb" in query
    assert args and args[0] == "p.pg"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_upsert_policy_sqlite_expected_version_uses_atomic_version_guard() -> None:
    db = _ExistingPolicyDbPool(is_postgres=False)
    admin = AuthNZPolicyAdmin(db_pool=db)
    admin._initialized = True

    await admin.upsert_policy("p.sqlite", {"requests": {"rpm": 12}}, version=1)

    assert db.execute_calls
    query, args = db.execute_calls[0]
    q_lower = query.lower()
    assert "where id = ?" in q_lower
    assert "version = ?" in q_lower
    assert args[-2:] == ("p.sqlite", 1)


@pytest.mark.asyncio
@pytest.mark.unit
async def test_upsert_policy_sqlite_expected_version_create_does_not_replace() -> None:
    db = _FakeDbPool(is_postgres=False)
    admin = AuthNZPolicyAdmin(db_pool=db)
    admin._initialized = True

    await admin.upsert_policy("p.sqlite.create", {"requests": {"rpm": 12}}, version=1)

    assert db.execute_calls
    query, args = db.execute_calls[0]
    q_lower = query.lower()
    assert "insert or ignore into rg_policies" in q_lower
    assert "insert or replace" not in q_lower
    assert args[0] == "p.sqlite.create"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_upsert_policy_postgres_expected_version_uses_atomic_version_guard() -> None:
    db = _ExistingPolicyDbPool(is_postgres=True)
    admin = AuthNZPolicyAdmin(db_pool=db)
    admin._initialized = True

    await admin.upsert_policy("p.pg", {"requests": {"rpm": 13}}, version=1)

    assert db.execute_calls
    query, args = db.execute_calls[0]
    q_lower = query.lower()
    assert "on conflict (id) do update" in q_lower
    assert "where rg_policies.version = $5" in q_lower
    assert args[-1] == 1
