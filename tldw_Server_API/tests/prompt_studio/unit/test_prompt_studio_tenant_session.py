from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.app.core.DB_Management import PromptStudioDatabase as PromptStudioDatabaseModule
from tldw_Server_API.app.core.DB_Management.PromptStudioDatabase import (
    BackendPromptStudioDatabaseBase,
    DatabaseError,
    PromptStudioBackendManagedTransaction,
)


class _Cursor:
    def __init__(self, *, error: BaseException | None = None) -> None:
        self.error = error
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def execute(self, statement: str, params: tuple[Any, ...]) -> None:
        self.calls.append((statement, params))
        if self.error is not None:
            raise self.error


class _Connection:
    def __init__(self, cursor: _Cursor) -> None:
        self._cursor = cursor
        self.commits = 0
        self.rollbacks = 0

    def cursor(self) -> _Cursor:
        return self._cursor

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1


def _database(*, tenant_user_id: str) -> BackendPromptStudioDatabaseBase:
    database = object.__new__(BackendPromptStudioDatabaseBase)
    database.client_id = "request-audit-client"
    database.tenant_user_id = tenant_user_id
    return database


def test_tenant_session_uses_owner_identity_not_audit_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(PromptStudioDatabaseModule, "psycopg_sql", None)
    cursor = _Cursor()
    connection = _Connection(cursor)

    _database(tenant_user_id="tenant-42")._apply_tenant_session(connection)

    assert cursor.calls == [
        (
            "SELECT set_config('app.current_user_id', %s, false)",
            ("tenant-42",),
        )
    ]
    assert connection.commits == 1


def test_tenant_session_setup_failure_is_bounded_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(PromptStudioDatabaseModule, "psycopg_sql", None)
    sentinel = "tenant-session-secret"
    connection = _Connection(_Cursor(error=RuntimeError(sentinel)))

    with pytest.raises(DatabaseError) as exc_info:
        _database(tenant_user_id="tenant-42")._apply_tenant_session(connection)

    assert str(exc_info.value) == "Failed to apply Prompt Studio tenant session"
    assert sentinel not in repr(exc_info.value)
    assert connection.commits == 0
    assert connection.rollbacks == 1


def test_thread_connection_releases_borrow_when_tenant_setup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(PromptStudioDatabaseModule, "psycopg_sql", None)
    connection = _Connection(_Cursor(error=RuntimeError("setup failed")))
    returned: list[Any] = []
    pool = SimpleNamespace(return_connection=returned.append)
    database = _database(tenant_user_id="tenant-42")
    database._local = threading.local()
    database.backend = SimpleNamespace(get_pool=lambda: pool)
    database.backend_type = SimpleNamespace(value="postgresql")
    database._open_new_connection = lambda: connection

    with pytest.raises(DatabaseError):
        database._get_thread_connection()

    assert returned == [connection]
    assert getattr(database._local, "conn", None) is None


def test_managed_transaction_exits_context_when_tenant_setup_fails() -> None:
    connection = _Connection(_Cursor())
    exits: list[tuple[Any, Any, Any]] = []

    class _Transaction:
        def __enter__(self) -> _Connection:
            return connection

        def __exit__(self, *exc_info: Any) -> bool:
            exits.append(exc_info)
            return False

    database = SimpleNamespace(
        backend=SimpleNamespace(transaction=_Transaction),
        _apply_tenant_session=lambda _connection: (_ for _ in ()).throw(
            DatabaseError("setup failed")
        ),
    )

    with pytest.raises(DatabaseError):
        PromptStudioBackendManagedTransaction(database).__enter__()

    assert len(exits) == 1
    assert exits[0][0] is DatabaseError
