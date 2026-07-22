import threading
from types import SimpleNamespace

import tldw_Server_API.app.core.DB_Management.backends.postgresql_backend as pg_backend
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType, DatabaseConfig


class _DummyConn:
    def __init__(self) -> None:
        self.closed = False
        self.close_calls = 0
        self.rollback_calls = 0
        self.row_factory = None
        self.info = SimpleNamespace(
            transaction_status=SimpleNamespace(name="IDLE")
        )

    def close(self) -> None:
        self.close_calls += 1
        self.closed = True

    def rollback(self) -> None:
        self.rollback_calls += 1
        self.info.transaction_status.name = "IDLE"


def _configure_fake_psycopg(monkeypatch) -> None:
    def _connect(_dsn: str):
        return _DummyConn()

    monkeypatch.setattr(pg_backend, "PSYCOPG2_AVAILABLE", True, raising=True)
    monkeypatch.setattr(pg_backend, "psycopg_pool", None, raising=False)
    monkeypatch.setattr(pg_backend, "psycopg", SimpleNamespace(connect=_connect), raising=False)
    monkeypatch.setattr(pg_backend, "dict_row", object(), raising=False)


def test_fallback_pool_closes_overflow_connections_on_return(monkeypatch) -> None:
    _configure_fake_psycopg(monkeypatch)
    cfg = DatabaseConfig(backend_type=BackendType.POSTGRESQL, pool_size=1)
    pool = pg_backend.PostgreSQLConnectionPool(cfg)

    managed_conn = pool.get_connection()
    overflow_conn = pool.get_connection()

    assert managed_conn in pool._connections
    assert overflow_conn not in pool._connections

    pool.return_connection(overflow_conn)
    assert overflow_conn.closed is True
    assert overflow_conn not in pool._free

    pool.return_connection(managed_conn)
    assert managed_conn.closed is False
    assert managed_conn in pool._free

    pool.close_all()


def test_fallback_pool_discard_replaces_poisoned_managed_connection(
    monkeypatch,
) -> None:
    _configure_fake_psycopg(monkeypatch)
    cfg = DatabaseConfig(backend_type=BackendType.POSTGRESQL, pool_size=1)
    pool = pg_backend.PostgreSQLConnectionPool(cfg)

    poisoned = pool.get_connection()
    pool.discard_connection(poisoned)

    assert poisoned.closed is True
    assert poisoned not in pool._connections
    assert poisoned not in pool._free

    replacement = pool.get_connection()
    assert replacement is not poisoned
    assert replacement.closed is False
    pool.return_connection(replacement)
    pool.close_all()


def test_fallback_pool_return_discards_connection_when_reset_fails(
    monkeypatch,
) -> None:
    _configure_fake_psycopg(monkeypatch)
    poisoned = _DummyConn()
    poisoned.info.transaction_status.name = "INERROR"
    replacement = _DummyConn()
    connections = iter((poisoned, replacement))

    def _connect(_dsn: str) -> _DummyConn:
        return next(connections)

    monkeypatch.setattr(
        pg_backend,
        "psycopg",
        SimpleNamespace(connect=_connect),
        raising=False,
    )

    def _rollback_failure() -> None:
        poisoned.rollback_calls += 1
        raise RuntimeError("driver rollback failed")

    poisoned.rollback = _rollback_failure  # type: ignore[method-assign]
    cfg = DatabaseConfig(backend_type=BackendType.POSTGRESQL, pool_size=1)
    pool = pg_backend.PostgreSQLConnectionPool(cfg)

    borrowed = pool.get_connection()
    assert borrowed is poisoned
    pool.return_connection(borrowed)

    assert poisoned.rollback_calls == 1
    assert poisoned.closed is True
    assert poisoned not in pool._connections
    assert poisoned not in pool._free
    assert pool.get_connection() is replacement
    pool.close_all()


def test_fallback_pool_reserves_managed_slot_before_concurrent_connect(
    monkeypatch,
) -> None:
    """Concurrent connects must not overfill or poison managed bookkeeping."""

    monkeypatch.setattr(pg_backend, "PSYCOPG2_AVAILABLE", True, raising=False)
    monkeypatch.setattr(pg_backend, "psycopg_pool", None, raising=False)
    first_connect_started = threading.Event()
    release_first_connect = threading.Event()
    connect_lock = threading.Lock()
    connections: list[_DummyConn] = []
    connect_calls = 0

    def _connect(_dsn: str) -> _DummyConn:
        nonlocal connect_calls
        with connect_lock:
            connect_calls += 1
            call_number = connect_calls
        if call_number == 1:
            first_connect_started.set()
            assert release_first_connect.wait(timeout=5)
        connection = _DummyConn()
        connections.append(connection)
        return connection

    monkeypatch.setattr(
        pg_backend,
        "psycopg",
        SimpleNamespace(connect=_connect),
        raising=False,
    )
    cfg = DatabaseConfig(backend_type=BackendType.POSTGRESQL, pool_size=1)
    pool = pg_backend.PostgreSQLConnectionPool(cfg)
    borrowed: list[_DummyConn] = []
    failures: list[BaseException] = []

    def _borrow() -> None:
        try:
            borrowed.append(pool.get_connection())
        except BaseException as exc:  # noqa: BLE001 - report thread failures below
            failures.append(exc)

    first = threading.Thread(target=_borrow)
    second = threading.Thread(target=_borrow)
    first.start()
    assert first_connect_started.wait(timeout=5)
    second.start()
    second.join(timeout=5)
    assert second.is_alive() is False
    release_first_connect.set()
    first.join(timeout=5)

    assert first.is_alive() is False
    assert failures == []
    assert len(borrowed) == 2
    assert len(pool._connections) == 1
    assert all(connection.closed is False for connection in pool._connections)

    for connection in borrowed:
        pool.return_connection(connection)

    assert len(pool._connections) == 1
    assert len(pool._free) == 1
    assert len({id(connection) for connection in pool._free}) == 1
    assert sum(connection.closed for connection in connections) == 1
    pool.close_all()


def test_psycopg_pool_constructor_opens_primary_and_compatibility_paths(
    monkeypatch,
) -> None:
    """Both supported constructor paths must explicitly open the delegate."""

    calls: list[dict[str, object]] = []
    delegate = SimpleNamespace()

    def _pool_factory(_dsn: str, **kwargs: object) -> object:
        calls.append(kwargs)
        if len(calls) == 1:
            raise TypeError("simulate unsupported tuning arguments")
        return delegate

    monkeypatch.setattr(pg_backend, "PSYCOPG2_AVAILABLE", True, raising=False)
    monkeypatch.setattr(
        pg_backend,
        "psycopg_pool",
        SimpleNamespace(ConnectionPool=_pool_factory),
        raising=False,
    )
    cfg = DatabaseConfig(backend_type=BackendType.POSTGRESQL, pool_size=1)

    pool = pg_backend.PostgreSQLConnectionPool(cfg)

    assert pool._pool is delegate
    assert [call.get("open") for call in calls] == [True, True]


def test_psycopg_pool_discard_closes_and_returns_checkout_for_bookkeeping(
    monkeypatch,
) -> None:
    """A poisoned delegate checkout must still be paired with putconn()."""

    returned: list[_DummyConn] = []

    class _Delegate:
        def putconn(self, connection: _DummyConn) -> None:
            returned.append(connection)

    delegate = _Delegate()
    monkeypatch.setattr(pg_backend, "PSYCOPG2_AVAILABLE", True, raising=False)
    monkeypatch.setattr(
        pg_backend,
        "psycopg_pool",
        SimpleNamespace(ConnectionPool=lambda *_args, **_kwargs: delegate),
        raising=False,
    )
    cfg = DatabaseConfig(backend_type=BackendType.POSTGRESQL, pool_size=1)
    pool = pg_backend.PostgreSQLConnectionPool(cfg)
    poisoned = _DummyConn()

    pool.discard_connection(poisoned)

    assert poisoned.closed is True
    assert poisoned.close_calls == 1
    assert returned == [poisoned]


def test_fallback_pool_close_all_closes_free_connections(monkeypatch) -> None:
    _configure_fake_psycopg(monkeypatch)
    cfg = DatabaseConfig(backend_type=BackendType.POSTGRESQL, pool_size=1)
    pool = pg_backend.PostgreSQLConnectionPool(cfg)

    conn = pool.get_connection()
    pool.return_connection(conn)
    assert conn in pool._free

    pool.close_all()
    assert conn.closed is True


def test_fallback_pool_close_all_deduplicates_connection_close(monkeypatch) -> None:
    _configure_fake_psycopg(monkeypatch)
    cfg = DatabaseConfig(backend_type=BackendType.POSTGRESQL, pool_size=1)
    pool = pg_backend.PostgreSQLConnectionPool(cfg)

    conn = pool.get_connection()
    # Managed connection is tracked and can also appear in free list.
    pool.return_connection(conn)
    assert conn in pool._connections
    assert conn in pool._free

    pool.close_all()
    assert conn.closed is True
    assert conn.close_calls == 1
