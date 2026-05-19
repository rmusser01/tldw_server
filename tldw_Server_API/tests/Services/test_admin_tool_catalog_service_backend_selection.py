from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from tldw_Server_API.app.services import admin_tool_catalog_service as svc


class _CursorStub:
    def __init__(self, *, rows: list[Any] | None = None) -> None:
        self._rows = list(rows or [])

    async def fetchall(self) -> list[Any]:
        return list(self._rows)


class _SqliteDbWithPgTraps:
    def __init__(self) -> None:
        self._is_sqlite = True
        self.execute_calls: list[tuple[str, Any]] = []
        self.fetch_calls: list[tuple[str, tuple[Any, ...]]] = []

    async def fetch(self, query: str, *args: Any) -> list[Any]:  # pragma: no cover - trap
        self.fetch_calls.append((str(query), tuple(args)))
        raise AssertionError("SQLite backend selection should not use fetch()")

    async def execute(self, query: str, params: Any) -> _CursorStub:
        self.execute_calls.append((str(query), params))
        return _CursorStub(
            rows=[
                (1, "sqlite-cat", "desc", None, None, 1, "2026-01-01", "2026-01-01"),
            ]
        )


class _PostgresDbWithSqliteTraps:
    def __init__(self) -> None:
        self._is_sqlite = False
        self.fetch_calls: list[tuple[str, tuple[Any, ...]]] = []

    async def execute(self, query: str, *args: Any) -> str:  # pragma: no cover - trap
        raise AssertionError("Postgres backend selection should not use sqlite execute() in this path")

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        self.fetch_calls.append((str(query), tuple(args)))
        if "?" in query:
            raise AssertionError("Postgres path should not use sqlite placeholders")
        return [
            {
                "id": 2,
                "name": "pg-cat",
                "description": "desc",
                "org_id": None,
                "team_id": None,
                "is_active": True,
                "created_at": "2026-01-01",
                "updated_at": "2026-01-01",
            }
        ]


class _ExplodingSqliteDb:
    _is_sqlite = True

    def __init__(self, message: str) -> None:
        self.message = message

    async def execute(self, *_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError(self.message)


async def _assert_tool_catalog_log_sanitized(
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
async def test_list_tool_catalogs_sqlite_backend_selection_uses_execute() -> None:
    db = _SqliteDbWithPgTraps()

    rows = await svc.list_tool_catalogs(db, org_id=None, team_id=None, limit=10, offset=0)

    assert db.execute_calls
    assert not db.fetch_calls
    assert rows and rows[0]["name"] == "sqlite-cat"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_list_tool_catalogs_postgres_backend_selection_uses_fetch() -> None:
    db = _PostgresDbWithSqliteTraps()

    rows = await svc.list_tool_catalogs(db, org_id=None, team_id=None, limit=10, offset=0)

    assert db.fetch_calls
    query, params = db.fetch_calls[0]
    assert "$1" in query and "$2" in query
    assert params[-2:] == (10, 0)
    assert rows and rows[0]["name"] == "pg-cat"


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize(
    ("call_factory", "expected_log", "raw_marker"),
    [
        (
            lambda db: svc.list_tool_catalogs(db, org_id=None, team_id=None, limit=10, offset=0),
            "Failed to list tool catalogs",
            "tool catalogs list failed",
        ),
        (
            lambda db: svc.list_visible_tool_catalogs(db, scope_norm="global", admin_all=True),
            "Failed to list visible tool catalogs",
            "visible tool catalogs list failed",
        ),
        (
            lambda db: svc.create_tool_catalog(
                db,
                name="new-cat",
                description="desc",
                org_id=None,
                team_id=None,
                is_active=True,
            ),
            "Failed to create tool catalog",
            "tool catalog create failed",
        ),
        (
            lambda db: svc.get_tool_catalog(db, 42),
            "Failed to get tool catalog",
            "tool catalog get failed",
        ),
        (
            lambda db: svc.delete_tool_catalog(db, 42),
            "Failed to delete tool catalog",
            "tool catalog delete failed",
        ),
        (
            lambda db: svc.list_tool_catalog_entries(db, 42),
            "Failed to list tool catalog entries",
            "tool catalog entries list failed",
        ),
        (
            lambda db: svc.add_tool_catalog_entry(db, 42, "media.search", None),
            "Failed to add tool catalog entry",
            "tool catalog entry add failed",
        ),
        (
            lambda db: svc.delete_tool_catalog_entry(db, 42, "media.search"),
            "Failed to delete tool catalog entry",
            "tool catalog entry delete failed",
        ),
    ],
)
async def test_tool_catalog_service_sanitizes_backend_failure_logs(
    call_factory: Callable[[Any], Awaitable[Any]],
    expected_log: str,
    raw_marker: str,
) -> None:
    db = _ExplodingSqliteDb(f"{raw_marker} at /private/tool-catalogs.db")

    await _assert_tool_catalog_log_sanitized(
        lambda: call_factory(db),
        expected_log=expected_log,
        raw_marker=raw_marker,
    )
