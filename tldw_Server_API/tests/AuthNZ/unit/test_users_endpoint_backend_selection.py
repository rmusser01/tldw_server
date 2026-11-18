from __future__ import annotations

from typing import Any

import pytest

from tldw_Server_API.app.api.v1.endpoints import users as users_endpoint


class _CursorStub:
    def __init__(self, row: Any) -> None:
        self._row = row

    async def fetchone(self) -> Any:
        return self._row


class _SQLiteConnWithFetchvalTrap:
    def __init__(self) -> None:
        self.fetchval_called = False
        self.execute_calls: list[tuple[str, Any]] = []

    async def fetchval(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - should never run
        self.fetchval_called = True
        raise AssertionError("sqlite path should not call fetchval()")

    async def execute(self, query: str, params: Any) -> _CursorStub:
        self.execute_calls.append((str(query), params))
        return _CursorStub(("hash-value",))


class _PostgresConnWithExecuteTrap:
    def __init__(self) -> None:
        self.fetchval_calls: list[tuple[str, Any]] = []
        self.execute_called = False

    async def fetchval(self, query: str, user_id: int) -> Any:
        self.fetchval_calls.append((query, user_id))
        return "hash-value"

    async def execute(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - should never run
        self.execute_called = True
        raise AssertionError("postgres path should not call execute() for password hash read")


@pytest.mark.asyncio
async def test_fetch_password_hash_sqlite_path_uses_execute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_is_postgres_backend() -> bool:
        return False

    monkeypatch.setattr("tldw_Server_API.app.core.AuthNZ.database.is_postgres_backend", _fake_is_postgres_backend)
    db = _SQLiteConnWithFetchvalTrap()

    value = await users_endpoint._fetch_password_hash_for_user(db, 7)

    assert value == "hash-value"
    assert db.fetchval_called is False
    assert db.execute_calls
    assert "select password_hash from users where id = ?" in db.execute_calls[0][0].lower()


@pytest.mark.asyncio
async def test_fetch_password_hash_postgres_path_uses_fetchval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_is_postgres_backend() -> bool:
        return True

    monkeypatch.setattr("tldw_Server_API.app.core.AuthNZ.database.is_postgres_backend", _fake_is_postgres_backend)
    db = _PostgresConnWithExecuteTrap()

    value = await users_endpoint._fetch_password_hash_for_user(db, 42)

    assert value == "hash-value"
    assert db.execute_called is False
    assert db.fetchval_calls
    assert "select password_hash from users where id = $1" in db.fetchval_calls[0][0].lower()
