from __future__ import annotations

from typing import Any

import pytest

from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.services import admin_system_service as svc


class _CursorStub:
    def __init__(self, *, row: Any = None, rows: list[Any] | None = None) -> None:
        self._row = row
        self._rows = list(rows or ([] if row is None else [row]))

    async def fetchone(self) -> Any:
        return self._row

    async def fetchall(self) -> list[Any]:
        return list(self._rows)


class _SqliteRowLike:
    def __init__(self, keys: list[str], values: tuple[Any, ...]) -> None:
        self._keys = list(keys)
        self._values = tuple(values)

    def keys(self) -> list[str]:
        return list(self._keys)

    def __iter__(self):
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, int):
            return self._values[key]
        idx = self._keys.index(str(key))
        return self._values[idx]


class _SqliteDbWithPgTraps:
    def __init__(self) -> None:
        self._is_sqlite = True
        self.execute_calls: list[tuple[str, Any]] = []
        self.fetchrow_calls: list[tuple[str, tuple[Any, ...]]] = []
        self.fetchval_calls: list[tuple[str, tuple[Any, ...]]] = []
        self.fetch_calls: list[tuple[str, tuple[Any, ...]]] = []

    async def fetchrow(self, query: str, *args: Any) -> Any:  # pragma: no cover - trap
        self.fetchrow_calls.append((str(query), tuple(args)))
        raise AssertionError("sqlite path should not use fetchrow()")

    async def fetchval(self, query: str, *args: Any) -> Any:  # pragma: no cover - trap
        self.fetchval_calls.append((str(query), tuple(args)))
        raise AssertionError("sqlite path should not use fetchval()")

    async def fetch(self, query: str, *args: Any) -> list[Any]:  # pragma: no cover - trap
        self.fetch_calls.append((str(query), tuple(args)))
        raise AssertionError("sqlite path should not use fetch()")

    async def execute(self, query: str, params: Any = ()) -> _CursorStub:
        self.execute_calls.append((str(query), params))
        q = str(query).lower()
        if "from users" in q and "count(*) as total_users" in q:
            return _CursorStub(row=(10, 8, 7, 1, 2))
        if "sum(storage_used_mb) as total_used_mb" in q:
            return _CursorStub(row=(100.0, 1000.0, 12.5, 50.0))
        if "from sessions" in q and "count(distinct user_id) as unique_users" in q:
            return _CursorStub(row=(4, 3))
        if "select count(*)" in q and "from audit_logs a" in q:
            return _CursorStub(row=(1,))
        if "select a.id, a.user_id" in q and "from audit_logs a" in q:
            return _CursorStub(
                rows=[
                    (
                        99,
                        1,
                        "alice",
                        "login",
                        "session",
                        123,
                        {"ok": True},
                        "127.0.0.1",
                        "2026-02-09T00:00:00+00:00",
                    )
                ]
            )
        raise AssertionError(f"Unexpected query: {query!r}")


class _PostgresDbWithSqliteTraps:
    def __init__(self) -> None:
        self._is_sqlite = False
        self.fetchrow_calls: list[tuple[str, tuple[Any, ...]]] = []
        self.fetchval_calls: list[tuple[str, tuple[Any, ...]]] = []
        self.fetch_calls: list[tuple[str, tuple[Any, ...]]] = []
        self.execute_calls: list[tuple[str, Any]] = []

    async def execute(self, query: str, params: Any = ()) -> Any:  # pragma: no cover - trap
        self.execute_calls.append((str(query), params))
        raise AssertionError("postgres path should not use sqlite execute()")

    async def fetchrow(self, query: str, *args: Any) -> dict[str, Any]:
        self.fetchrow_calls.append((str(query), tuple(args)))
        if "?" in query:
            raise AssertionError("postgres path should not use sqlite placeholders")
        q = str(query).lower()
        if "from users" in q and "count(*) as total_users" in q:
            return {
                "total_users": 12,
                "active_users": 9,
                "verified_users": 8,
                "admin_users": 2,
                "new_users_30d": 3,
            }
        if "sum(storage_used_mb) as total_used_mb" in q:
            return {
                "total_used_mb": 200.0,
                "total_quota_mb": 1500.0,
                "avg_used_mb": 16.5,
                "max_used_mb": 80.0,
            }
        if "from sessions" in q and "count(distinct user_id) as unique_users" in q:
            return {"active_sessions": 5, "unique_users": 4}
        return {}

    async def fetchval(self, query: str, *args: Any) -> int:
        self.fetchval_calls.append((str(query), tuple(args)))
        if "?" in query:
            raise AssertionError("postgres path should not use sqlite placeholders")
        return 1

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        self.fetch_calls.append((str(query), tuple(args)))
        if "?" in query:
            raise AssertionError("postgres path should not use sqlite placeholders")
        return [
            {
                "id": 77,
                "user_id": 1,
                "username": "alice",
                "action": "login",
                "resource_type": "session",
                "resource_id": 456,
                "details": {"ok": True},
                "ip_address": "127.0.0.1",
                "created_at": "2026-02-09T00:00:00+00:00",
            }
        ]


class _FailingStatsDb:
    _is_sqlite = True

    async def execute(self, query: str, params: Any = ()) -> _CursorStub:
        raise RuntimeError("stats unavailable")


def _admin_principal() -> AuthPrincipal:
    return AuthPrincipal(
        kind="user",
        user_id=1,
        roles=["admin"],
        permissions=[],
        is_admin=True,
        org_ids=[],
        team_ids=[],
    )


@pytest.mark.asyncio
@pytest.mark.unit
async def test_get_system_stats_sqlite_backend_selection_uses_execute() -> None:
    db = _SqliteDbWithPgTraps()

    response = await svc.get_system_stats(db)

    assert response.users.total == 10
    assert response.storage.total_used_mb == 100.0
    assert response.sessions.active == 4
    assert db.execute_calls
    assert not db.fetchrow_calls
    assert not db.fetchval_calls
    assert not db.fetch_calls


@pytest.mark.asyncio
@pytest.mark.unit
async def test_get_system_stats_sqlite_row_objects_use_row_keys() -> None:
    class _SqliteDbWithRowObjects(_SqliteDbWithPgTraps):
        async def execute(self, query: str, params: Any = ()) -> _CursorStub:
            self.execute_calls.append((str(query), params))
            q = str(query).lower()
            if "from users" in q and "count(*) as total_users" in q:
                return _CursorStub(
                    row=_SqliteRowLike(
                        ["total_users", "active_users", "verified_users", "admin_users", "new_users_30d"],
                        (10, 8, 7, 1, 2),
                    )
                )
            if "sum(storage_used_mb) as total_used_mb" in q:
                return _CursorStub(
                    row=_SqliteRowLike(
                        ["total_used_mb", "total_quota_mb", "avg_used_mb", "max_used_mb"],
                        (100.0, 1000.0, 12.5, 50.0),
                    )
                )
            if "from sessions" in q and "count(distinct user_id) as unique_users" in q:
                return _CursorStub(
                    row=_SqliteRowLike(
                        ["active_sessions", "unique_users"],
                        (4, 3),
                    )
                )
            raise AssertionError(f"Unexpected query: {query!r}")

    db = _SqliteDbWithRowObjects()

    response = await svc.get_system_stats(db)

    assert response.users.total == 10
    assert response.storage.total_quota_mb == 1000.0
    assert response.sessions.unique_users == 3


@pytest.mark.asyncio
@pytest.mark.unit
async def test_debug_decode_token_reports_decoded_without_claiming_signature_validation() -> None:
    result = await svc.debug_decode_token(
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjMiLCJpc3MiOiJ0ZXN0In0.signature"
    )

    assert result["decoded"] is True
    assert result["signature_verified"] is False
    assert result["subject"] == "123"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_debug_decode_token_sanitizes_decode_failures() -> None:
    result = await svc.debug_decode_token("header.payload.signature")

    assert result["decoded"] is False
    assert result["signature_verified"] is False
    assert result["error"] == "Failed to decode token."
    assert "Expecting value" not in result["error"]
    assert "header" not in result["error"]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_debug_resolve_permissions_sanitizes_backend_failures() -> None:
    class _FailingDebugDb:
        async def execute(self, *_args, **_kwargs):
            raise RuntimeError("permission DB failed at /private/admin-permissions.db")

    result = await svc.debug_resolve_permissions(42, _FailingDebugDb())

    assert result["user_id"] == 42
    assert result["roles"] == []
    assert result["effective_permissions"] == []
    assert result["error"] == "Failed to resolve permissions."
    assert "permission DB failed" not in result["error"]
    assert "/private/admin-permissions.db" not in result["error"]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_get_system_stats_postgres_backend_selection_uses_fetchrow() -> None:
    db = _PostgresDbWithSqliteTraps()

    response = await svc.get_system_stats(db)

    assert response.users.total == 12
    assert response.storage.max_used_mb == 80.0
    assert response.sessions.unique_users == 4
    assert len(db.fetchrow_calls) >= 3
    assert not db.execute_calls


@pytest.mark.asyncio
@pytest.mark.unit
async def test_get_system_stats_returns_empty_snapshot_when_queries_fail() -> None:
    response = await svc.get_system_stats(_FailingStatsDb())

    assert response.users.total == 0
    assert response.users.active == 0
    assert response.storage.total_used_mb == 0.0
    assert response.sessions.active == 0


@pytest.mark.asyncio
@pytest.mark.unit
async def test_get_audit_log_sqlite_backend_selection_uses_execute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _org_ids(_principal: AuthPrincipal) -> None:
        return None

    monkeypatch.setattr(svc.admin_scope_service, "get_admin_org_ids", _org_ids)
    db = _SqliteDbWithPgTraps()

    response = await svc.get_audit_log(
        user_id=None,
        action=None,
        resource=None,
        start=None,
        end=None,
        days=7,
        limit=10,
        offset=0,
        org_id=None,
        principal=_admin_principal(),
        db=db,
    )

    assert response.total == 1
    assert response.entries and response.entries[0].resource == "session:123"
    assert db.execute_calls
    assert not db.fetchval_calls
    assert not db.fetch_calls


@pytest.mark.asyncio
@pytest.mark.unit
async def test_get_audit_log_postgres_backend_selection_uses_fetch_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _org_ids(_principal: AuthPrincipal) -> None:
        return None

    monkeypatch.setattr(svc.admin_scope_service, "get_admin_org_ids", _org_ids)
    db = _PostgresDbWithSqliteTraps()

    response = await svc.get_audit_log(
        user_id=None,
        action=None,
        resource=None,
        start=None,
        end=None,
        days=7,
        limit=10,
        offset=0,
        org_id=None,
        principal=_admin_principal(),
        db=db,
    )

    assert response.total == 1
    assert response.entries and response.entries[0].resource == "session:456"
    assert db.fetchval_calls
    assert db.fetch_calls
    assert not db.execute_calls
