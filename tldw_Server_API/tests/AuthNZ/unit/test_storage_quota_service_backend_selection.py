from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from tldw_Server_API.app.core.AuthNZ.profile_user_write_guard import _guard_sql
from tldw_Server_API.app.services.storage_quota_service import StorageQuotaService


class _Ctx:
    def __init__(self, conn: Any) -> None:
        self._conn = conn

    async def __aenter__(self) -> Any:
        return self._conn

    async def __aexit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001, ARG002
        return False


class _PoolStub:
    def __init__(self, *, postgres: bool, tx_conn: Any | None = None, acquire_conn: Any | None = None) -> None:
        self.pool = object() if postgres else None
        self._tx_conn = tx_conn
        self._acquire_conn = acquire_conn

    def transaction(self) -> _Ctx:
        assert self._tx_conn is not None
        return _Ctx(self._tx_conn)

    def acquire(self) -> _Ctx:
        assert self._acquire_conn is not None
        return _Ctx(self._acquire_conn)


class _CursorStub:
    def __init__(
        self,
        *,
        row: Any = None,
        rows: list[Any] | None = None,
        description: list[tuple[Any, ...]] | None = None,
        rowcount: int = 1,
    ) -> None:
        self._row = row
        self._rows = rows if rows is not None else ([] if row is None else [row])
        self.description = description
        self.rowcount = rowcount

    async def fetchone(self) -> Any:
        return self._row

    async def fetchall(self) -> list[Any]:
        return list(self._rows)


class _SQLiteUpdateConnWithFetchrowTrap:
    _authnz_profile_user_backend = "sqlite"

    def __init__(self) -> None:
        self._authnz_profile_user_guard_identity = self
        self.update_calls = 0
        self.select_calls = 0
        self.committed = False

    async def fetchrow(self, *args, **kwargs):  # noqa: ANN001, ANN002, ARG002
        raise AssertionError("SQLite backend path should not call conn.fetchrow")

    async def execute(self, query: object, params: Any) -> _CursorStub:
        concrete = _guard_sql(
            query,
            backend="sqlite",
            connection_identity=self,
            operation="execute",
        )
        q = concrete.lower()
        if "source_tag" in q:
            return _CursorStub(
                rows=[("user", 1, "2026-07-26T12:00:00.000000Z")]
            )
        if "update main.users" in q and "set storage_used_mb" in q:
            self.update_calls += 1
            return _CursorStub()
        if "select storage_used_mb, storage_quota_mb from users" in q:
            self.select_calls += 1
            return _CursorStub(row=(5.5, 100))
        if "set profile_version" in q:
            return _CursorStub()
        raise AssertionError(f"Unexpected SQLite query: {concrete!r}")

    async def commit(self) -> None:
        self.committed = True


class _PostgresFetchvalConn:
    def __init__(self, bytes_value: int) -> None:
        self.bytes_value = bytes_value
        self.fetchval_calls: list[tuple[str, tuple[Any, ...]]] = []

    async def fetchval(self, query: str, *params: Any) -> int:
        self.fetchval_calls.append((str(query), tuple(params)))
        return self.bytes_value

    async def execute(self, query: str, *params: Any):  # noqa: ARG002
        raise AssertionError(f"Postgres backend path should not call conn.execute: {query!r}")


class _SQLiteSumConnWithFetchvalTrap:
    def __init__(self, bytes_value: int) -> None:
        self.bytes_value = bytes_value
        self.execute_calls: list[tuple[str, Any]] = []

    async def fetchval(self, *args, **kwargs):  # noqa: ANN001, ANN002, ARG002
        raise AssertionError("SQLite backend path should not call conn.fetchval")

    async def execute(self, query: str, params: Any) -> _CursorStub:
        self.execute_calls.append((str(query), params))
        return _CursorStub(row=(self.bytes_value,))


class _PostgresUsersConnWithExecuteTrap:
    def __init__(self) -> None:
        self.fetch_calls: list[str] = []

    async def fetch(self, query: str) -> list[dict[str, Any]]:
        self.fetch_calls.append(str(query))
        return [
            {"id": 1, "username": "alice", "storage_used_mb": 8.0, "storage_quota_mb": 10.0},
            {"id": 2, "username": "bob", "storage_used_mb": 2.0, "storage_quota_mb": 10.0},
        ]

    async def execute(self, query: str, *params: Any):  # noqa: ARG002
        raise AssertionError(f"Postgres backend path should not call conn.execute: {query!r}")


class _SQLiteUsersConnWithFetchTrap:
    def __init__(self) -> None:
        self.execute_calls: list[str] = []

    async def fetch(self, *args, **kwargs):  # noqa: ANN001, ANN002, ARG002
        raise AssertionError("SQLite backend path should not call conn.fetch")

    async def execute(self, query: str, params: Any | None = None) -> _CursorStub:  # noqa: ARG002
        self.execute_calls.append(str(query))
        return _CursorStub(
            rows=[
                (1, "alice", 8.0, 10.0),
                (2, "bob", 2.0, 10.0),
            ],
            description=[
                ("id",),
                ("username",),
                ("storage_used_mb",),
                ("storage_quota_mb",),
            ],
        )


class _QuotasRepoStub:
    def __init__(self) -> None:
        self.org_updates: list[tuple[int, float]] = []
        self.team_updates: list[tuple[int, float]] = []

    async def update_org_used_mb(self, org_id: int, used_mb: float) -> None:
        self.org_updates.append((org_id, used_mb))

    async def update_team_used_mb(self, team_id: int, used_mb: float) -> None:
        self.team_updates.append((team_id, used_mb))

    async def check_quota_status(
        self,
        *,
        org_id: int | None = None,
        team_id: int | None = None,
    ) -> dict[str, Any]:
        if org_id is not None:
            return {"scope": "org", "org_id": org_id}
        return {"scope": "team", "team_id": team_id}


def _make_service(pool: _PoolStub, tmp_path: Path) -> StorageQuotaService:
    service = StorageQuotaService(
        db_pool=pool,
        settings=SimpleNamespace(USER_DATA_BASE_PATH=tmp_path.as_posix(), CHROMADB_BASE_PATH=""),
    )
    service._initialized = True
    return service


@pytest.mark.asyncio
async def test_update_usage_sqlite_backend_selection_ignores_conn_fetchrow(tmp_path: Path):
    conn = _SQLiteUpdateConnWithFetchrowTrap()
    service = _make_service(_PoolStub(postgres=False, tx_conn=conn), tmp_path)

    result = await service.update_usage(user_id=1, bytes_delta=1024 * 1024, operation="add")

    assert result["storage_used_mb"] == 5.5
    assert conn.update_calls == 1
    assert conn.select_calls == 1
    assert conn.committed is False


@pytest.mark.asyncio
async def test_recalculate_org_usage_postgres_backend_selection_uses_fetchval(tmp_path: Path):
    conn = _PostgresFetchvalConn(bytes_value=10 * 1024 * 1024)
    service = _make_service(_PoolStub(postgres=True, acquire_conn=conn), tmp_path)
    repo = _QuotasRepoStub()
    service.get_storage_quotas_repo = AsyncMock(return_value=repo)

    status = await service.recalculate_org_usage(org_id=7)

    assert status == {"scope": "org", "org_id": 7}
    assert conn.fetchval_calls, "expected Postgres fetchval path to be used"
    assert "where org_id = $1" in conn.fetchval_calls[0][0].lower()
    assert repo.org_updates and repo.org_updates[0][0] == 7


@pytest.mark.asyncio
async def test_recalculate_team_usage_sqlite_backend_selection_ignores_conn_fetchval(tmp_path: Path):
    conn = _SQLiteSumConnWithFetchvalTrap(bytes_value=3 * 1024 * 1024)
    service = _make_service(_PoolStub(postgres=False, acquire_conn=conn), tmp_path)
    repo = _QuotasRepoStub()
    service.get_storage_quotas_repo = AsyncMock(return_value=repo)

    status = await service.recalculate_team_usage(team_id=9)

    assert status == {"scope": "team", "team_id": 9}
    assert conn.execute_calls, "expected SQLite execute path to be used"
    assert "where team_id = ?" in conn.execute_calls[0][0].lower()
    assert repo.team_updates and repo.team_updates[0][0] == 9


@pytest.mark.asyncio
async def test_get_all_users_storage_postgres_backend_selection_uses_fetch(tmp_path: Path):
    conn = _PostgresUsersConnWithExecuteTrap()
    service = _make_service(_PoolStub(postgres=True, acquire_conn=conn), tmp_path)

    result = await service.get_all_users_storage()

    assert conn.fetch_calls, "expected Postgres fetch path to be used"
    assert result[0]["user_id"] == 1
    assert result[0]["usage_percentage"] == 80.0


@pytest.mark.asyncio
async def test_get_all_users_storage_sqlite_backend_selection_ignores_conn_fetch(tmp_path: Path):
    conn = _SQLiteUsersConnWithFetchTrap()
    service = _make_service(_PoolStub(postgres=False, acquire_conn=conn), tmp_path)

    result = await service.get_all_users_storage()

    assert conn.execute_calls, "expected SQLite execute path to be used"
    assert "where is_active = 1" in conn.execute_calls[0].lower()
    assert [item["user_id"] for item in result] == [1, 2]
