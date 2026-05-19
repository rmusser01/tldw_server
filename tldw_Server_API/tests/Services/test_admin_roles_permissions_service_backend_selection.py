from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from tldw_Server_API.app.services import admin_roles_permissions_service as svc


class _CursorStub:
    def __init__(self, *, rows: list[Any] | None = None, lastrowid: int | None = None) -> None:
        self._rows = list(rows or [])
        self.lastrowid = lastrowid

    async def fetchall(self) -> list[Any]:
        return list(self._rows)

    async def fetchone(self) -> Any:
        return self._rows[0] if self._rows else None


class _SqliteDbWithPgTraps:
    def __init__(self) -> None:
        self._is_sqlite = True
        self.execute_calls: list[tuple[str, Any]] = []
        self.fetch_calls: list[tuple[str, tuple[Any, ...]]] = []
        self.fetchrow_calls: list[tuple[str, tuple[Any, ...]]] = []
        self.commit_calls = 0

    async def fetch(self, query: str, *args: Any) -> list[Any]:  # pragma: no cover - trap
        self.fetch_calls.append((str(query), tuple(args)))
        raise AssertionError("SQLite backend selection should not use fetch()")

    async def fetchrow(self, query: str, *args: Any) -> Any:  # pragma: no cover - trap
        self.fetchrow_calls.append((str(query), tuple(args)))
        raise AssertionError("SQLite backend selection should not use fetchrow()")

    async def execute(self, query: str, params: Any = ()) -> _CursorStub:
        self.execute_calls.append((str(query), params))
        q = str(query).lower()
        if "select id, name, description, coalesce(is_system, 0)" in q:
            return _CursorStub(rows=[(1, "role-sqlite", "desc", 0)])
        if "select 1 from roles where lower(name) = lower(?)" in q:
            return _CursorStub(rows=[])
        if "insert into roles" in q:
            return _CursorStub(rows=[], lastrowid=2)
        if "select id, name, description, coalesce(is_system,0) from roles where id =" in q:
            return _CursorStub(rows=[(2, "new-role", "desc", 1)])
        return _CursorStub(rows=[])

    async def commit(self) -> None:
        self.commit_calls += 1


class _PostgresDbWithSqliteTraps:
    def __init__(self) -> None:
        self._is_sqlite = False
        self.fetch_calls: list[tuple[str, tuple[Any, ...]]] = []
        self.fetchrow_calls: list[tuple[str, tuple[Any, ...]]] = []
        self.execute_calls: list[tuple[str, tuple[Any, ...]]] = []

    async def execute(self, query: str, *args: Any) -> str:
        self.execute_calls.append((str(query), tuple(args)))
        if "?" in query:
            raise AssertionError("Postgres path should not use sqlite placeholders")
        return "OK"

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        self.fetch_calls.append((str(query), tuple(args)))
        if "?" in query:
            raise AssertionError("Postgres path should not use sqlite placeholders")
        return [{"id": 5, "name": "role-pg", "description": "desc", "is_system": False}]

    async def fetchrow(self, query: str, *args: Any) -> dict[str, Any] | None:
        self.fetchrow_calls.append((str(query), tuple(args)))
        if "?" in query:
            raise AssertionError("Postgres path should not use sqlite placeholders")
        if "select 1 from roles" in query.lower():
            return None
        if "returning id, name, description, is_system" in query.lower():
            return {"id": 6, "name": "new-role", "description": "desc", "is_system": True}
        return {"id": 6, "name": "new-role", "description": "desc", "is_system": True}


class _ExplodingSqliteDb:
    _is_sqlite = True

    def __init__(self, message: str) -> None:
        self.message = message

    async def execute(self, *_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError(self.message)


async def _assert_role_permission_log_sanitized(
    call: Callable[[], Awaitable[Any]],
    *,
    expected_log: str,
    raw_marker: str,
) -> None:
    messages: list[str] = []
    sink_id = svc.logger.add(lambda message: messages.append(str(message)), level="ERROR")
    try:
        with pytest.raises(RuntimeError):
            await call()
    finally:
        svc.logger.remove(sink_id)

    joined = "\n".join(messages)
    assert expected_log in joined
    assert raw_marker not in joined
    assert "/private/" not in joined


@pytest.mark.asyncio
@pytest.mark.unit
async def test_list_roles_sqlite_backend_selection_uses_execute() -> None:
    db = _SqliteDbWithPgTraps()

    rows = await svc.list_roles(db)

    assert rows and rows[0]["name"] == "role-sqlite"
    assert db.execute_calls
    assert not db.fetch_calls
    assert not db.fetchrow_calls


@pytest.mark.asyncio
@pytest.mark.unit
async def test_list_roles_postgres_backend_selection_uses_fetch() -> None:
    db = _PostgresDbWithSqliteTraps()

    rows = await svc.list_roles(db)

    assert rows and rows[0]["name"] == "role-pg"
    assert db.fetch_calls
    assert not db.execute_calls


@pytest.mark.asyncio
@pytest.mark.unit
async def test_create_role_sqlite_backend_selection_uses_sqlite_queries() -> None:
    db = _SqliteDbWithPgTraps()

    row = await svc.create_role(db, "new-role", "desc", True)

    assert row["name"] == "new-role"
    assert any("lower(name) = lower(?)" in q.lower() for q, _ in db.execute_calls)
    assert any("insert into roles" in q.lower() and "?" in q for q, _ in db.execute_calls)
    assert db.commit_calls >= 1


@pytest.mark.asyncio
@pytest.mark.unit
async def test_create_role_postgres_backend_selection_uses_postgres_queries() -> None:
    db = _PostgresDbWithSqliteTraps()

    row = await svc.create_role(db, "new-role", "desc", True)

    assert row["name"] == "new-role"
    assert db.fetchrow_calls
    assert any("$1" in q for q, _ in db.fetchrow_calls)


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize(
    ("call_factory", "expected_log", "raw_marker"),
    [
        (
            lambda db: svc.list_roles(db),
            "Failed to list roles",
            "roles list failed",
        ),
        (
            lambda db: svc.create_role(db, "new-role", "desc", False),
            "Failed to create role",
            "role create failed",
        ),
        (
            lambda db: svc.delete_role(db, 42),
            "Failed to delete role",
            "role delete failed",
        ),
        (
            lambda db: svc.list_role_permissions(db, 42),
            "Failed to list role permissions",
            "role permissions list failed",
        ),
        (
            lambda db: svc.list_tool_permissions(db),
            "Failed to list tool permissions",
            "tool permissions list failed",
        ),
        (
            lambda db: svc.delete_tool_permission(db, "tools.execute:test"),
            "Failed to delete tool permission",
            "tool permission delete failed",
        ),
        (
            lambda db: svc.revoke_tool_permission_from_role(db, 42, "tools.execute:test"),
            "Failed to revoke tool permission from role",
            "tool permission revoke failed",
        ),
    ],
)
async def test_role_permission_service_sanitizes_backend_failure_logs(
    call_factory: Callable[[Any], Awaitable[Any]],
    expected_log: str,
    raw_marker: str,
) -> None:
    db = _ExplodingSqliteDb(f"{raw_marker} at /private/rbac.db")

    await _assert_role_permission_log_sanitized(
        lambda: call_factory(db),
        expected_log=expected_log,
        raw_marker=raw_marker,
    )


@pytest.mark.asyncio
@pytest.mark.unit
async def test_grant_tool_permission_to_role_sanitizes_backend_failure_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_ensure_permission(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {"id": 7, "name": "tools.execute:test", "description": "desc", "category": "tools"}

    db = _ExplodingSqliteDb("tool permission grant failed at /private/rbac.db")
    monkeypatch.setattr(svc, "ensure_permission", fake_ensure_permission)

    await _assert_role_permission_log_sanitized(
        lambda: svc.grant_tool_permission_to_role(db, 42, "tools.execute:test", "desc"),
        expected_log="Failed to grant tool permission to role",
        raw_marker="tool permission grant failed",
    )
