"""A connection whose tenant scope cannot be set must never be handed out.

Tenant GUCs are session-scoped, so a pooled checkout keeps the previous
borrower's ``app.current_user_id`` until something overwrites it. When
re-scoping at checkout failed, the failure used to be logged at debug and the
connection returned anyway -- still carrying the last request's identity.
"""

from __future__ import annotations

from types import SimpleNamespace

import psycopg
import pytest

import tldw_Server_API.app.core.DB_Management.backends.postgresql_backend as pg_backend
from tldw_Server_API.app.core.DB_Management.backends.base import (
    BackendType,
    DatabaseConfig,
    DatabaseError,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime import connection_lifecycle

pytestmark = pytest.mark.unit


class _Cursor:
    def __init__(self, conn: _Conn) -> None:
        self.conn = conn

    def __enter__(self) -> _Cursor:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def execute(self, sql: str, params: tuple | None = None) -> None:
        if self.conn.fail_set_config and "set_config" in sql:
            raise psycopg.OperationalError("injected set_config failure")
        self.conn.executed.append((sql, params))


class _Conn:
    def __init__(self) -> None:
        self.fail_set_config = False
        self.closed = False
        self.row_factory = None
        self.executed: list[tuple[str, tuple | None]] = []

    def cursor(self) -> _Cursor:
        return _Cursor(self)

    def rollback(self) -> None:
        return None

    def commit(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


class _Delegate:
    """Stands in for psycopg_pool.ConnectionPool: one checkout, returns recorded."""

    def __init__(self, conn: _Conn) -> None:
        self.conn = conn
        self.returned: list[_Conn] = []

    def getconn(self) -> _Conn:
        return self.conn

    def putconn(self, conn: _Conn) -> None:
        self.returned.append(conn)

    def close(self) -> None:
        return None


@pytest.fixture()
def wired(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    conn = _Conn()
    delegate = _Delegate(conn)
    monkeypatch.setattr(pg_backend, "PSYCOPG2_AVAILABLE", True)
    monkeypatch.setattr(
        pg_backend, "psycopg_pool", SimpleNamespace(ConnectionPool=lambda *_a, **_k: delegate)
    )
    backend = pg_backend.PostgreSQLBackend(
        DatabaseConfig(
            backend_type=BackendType.POSTGRESQL,
            connection_string="postgresql://unit/scope",
            pool_size=1,
        )
    )
    return SimpleNamespace(backend=backend, pool=backend.get_pool(), conn=conn, delegate=delegate)


def _assert_discarded(wired: SimpleNamespace) -> None:
    assert wired.conn.closed is True
    assert wired.delegate.returned == [wired.conn]


def test_checkout_scope_failure_discards_and_raises(wired: SimpleNamespace) -> None:
    wired.conn.fail_set_config = True

    with pytest.raises(DatabaseError):
        wired.pool.get_connection()

    _assert_discarded(wired)


def test_apply_scope_failure_discards_and_raises(wired: SimpleNamespace) -> None:
    conn = wired.pool.get_connection()
    conn.fail_set_config = True

    with pytest.raises(DatabaseError):
        wired.backend.apply_scope(conn)

    _assert_discarded(wired)


def test_media_connection_lifecycle_scope_failure_discards_and_raises(
    wired: SimpleNamespace,
) -> None:
    host = SimpleNamespace(
        backend=wired.backend,
        backend_type=BackendType.POSTGRESQL,
        persistent=None,
        _get_txn_conn=lambda: None,
    )
    host._get_persistent_conn = lambda: host.persistent
    host._set_persistent_conn = lambda conn: setattr(host, "persistent", conn)
    assert connection_lifecycle.get_connection(host) is wired.conn
    wired.conn.fail_set_config = True

    with pytest.raises(DatabaseError):
        connection_lifecycle.get_connection(host)

    _assert_discarded(wired)
    assert host.persistent is None


def test_direct_connect_scope_failure_closes_and_raises(
    wired: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = _Conn()
    raw.fail_set_config = True
    monkeypatch.setattr(pg_backend.psycopg, "connect", lambda *_a, **_k: raw)

    with pytest.raises(DatabaseError):
        wired.backend.connect()

    assert raw.closed is True


def test_no_scope_background_checkout_is_still_usable(wired: SimpleNamespace) -> None:
    """Background work runs with no request scope and must still get a connection."""
    conn = wired.pool.get_connection()

    assert conn is wired.conn
    assert conn.closed is False
    assert wired.delegate.returned == []
    assert any(
        "set_config('app.current_user_id'" in sql and params == ("", "", "", "", "0")
        for sql, params in conn.executed
    )
