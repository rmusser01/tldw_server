"""Regression tests for PostgreSQL pool checkout cleanup and capacity."""

import threading
from types import SimpleNamespace

import pytest

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
    def _connect(_dsn: str, *, row_factory):
        connection = _DummyConn()
        connection.row_factory = row_factory
        return connection

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


@pytest.mark.unit
def test_fallback_pool_close_preserves_borrowers_until_return(monkeypatch) -> None:
    """Closing a pool must not free a connection while its borrower still uses it."""
    _configure_fake_psycopg(monkeypatch)
    pool = pg_backend.PostgreSQLConnectionPool(
        DatabaseConfig(backend_type=BackendType.POSTGRESQL, pool_size=2)
    )
    idle = pool.get_connection()
    borrowed = pool.get_connection()
    pool.return_connection(idle)

    pool.close_all()

    assert (idle.closed, borrowed.closed) == (True, False)  # nosec B101 - Pytest pool-lifetime regression assertion.
    with pytest.raises(pg_backend.DatabaseError, match="Connection pool is closed"):
        pool.get_connection()
    pool.return_connection(borrowed)
    assert borrowed.close_calls == 1  # nosec B101 - Pytest pool-lifetime regression assertion.


@pytest.mark.unit
@pytest.mark.parametrize("operation", ["discard_connection", "invalidate_connection"], ids=[
    "discard-poisoned-checkout", "invalidate-poisoned-checkout",
])
def test_fallback_pool_discard_replaces_poisoned_managed_connection(
    monkeypatch: pytest.MonkeyPatch, operation: str,
) -> None:
    """Both removal APIs release managed capacity and permit a healthy replacement."""
    _configure_fake_psycopg(monkeypatch)
    cfg = DatabaseConfig(backend_type=BackendType.POSTGRESQL, pool_size=1)
    pool = pg_backend.PostgreSQLConnectionPool(cfg)

    poisoned = pool.get_connection()
    getattr(pool, operation)(poisoned)

    assert poisoned.closed is True

    replacement = pool.get_connection()
    assert replacement is not poisoned
    assert replacement.closed is False
    pool.return_connection(replacement)
    assert replacement.closed is False
    reborrowed = pool.get_connection()
    assert reborrowed.closed is False
    pool.return_connection(reborrowed)
    pool.close_all()


def test_fallback_pool_return_discards_connection_when_reset_fails(
    monkeypatch,
) -> None:
    _configure_fake_psycopg(monkeypatch)
    poisoned = _DummyConn()
    poisoned.info.transaction_status.name = "INERROR"
    replacement = _DummyConn()
    connections = iter((poisoned, replacement))

    def _connect(_dsn: str, *, row_factory) -> _DummyConn:
        connection = next(connections)
        connection.row_factory = row_factory
        return connection

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

    def _connect(_dsn: str, *, row_factory) -> _DummyConn:
        nonlocal connect_calls
        with connect_lock:
            connect_calls += 1
            call_number = connect_calls
        if call_number == 1:
            first_connect_started.set()
            assert release_first_connect.wait(timeout=5)
        connection = _DummyConn()
        connection.row_factory = row_factory
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


@pytest.mark.unit
@pytest.mark.parametrize("operation", ["discard_connection", "invalidate_connection"], ids=[
    "discard-poisoned-checkout", "invalidate-poisoned-checkout",
])
def test_psycopg_pool_discard_closes_and_returns_checkout_for_bookkeeping(
    monkeypatch: pytest.MonkeyPatch, operation: str,
) -> None:
    """A poisoned delegate checkout must still be paired with putconn()."""

    returned: list[_DummyConn] = []

    class _Delegate:
        """Record checkouts returned to the psycopg pool for replacement."""

        def putconn(self, connection: _DummyConn) -> None:
            """Retain the returned checkout as observable pool bookkeeping."""
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

    getattr(pool, operation)(poisoned)

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


@pytest.mark.unit
@pytest.mark.parametrize("operation", ["return_connection", "discard_connection"])
@pytest.mark.parametrize("failure_type", [RuntimeError, Exception], ids=["classified", "unclassified"])
def test_pool_cleanup_preserves_exception_classification_and_return_ownership(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    failure_type: type[Exception],
) -> None:
    """Known close failures are suppressed; unknown ones preserve cleanup ownership."""
    events: list[tuple[str, _DummyConn]] = []
    failure = failure_type("private cleanup detail")
    poisoned = _DummyConn()

    def _close() -> None:
        events.append(("close", poisoned))
        raise failure

    def _putconn(connection: _DummyConn) -> None:
        events.append(("putconn", connection))
        if operation == "return_connection":
            raise RuntimeError("delegate return failed")

    monkeypatch.setattr(poisoned, "close", _close)
    monkeypatch.setattr(pg_backend, "PSYCOPG2_AVAILABLE", True, raising=False)
    monkeypatch.setattr(
        pg_backend,
        "psycopg_pool",
        SimpleNamespace(ConnectionPool=lambda *_args, **_kwargs: SimpleNamespace(putconn=_putconn)),
        raising=False,
    )
    pool = pg_backend.PostgreSQLConnectionPool(DatabaseConfig(backend_type=BackendType.POSTGRESQL, pool_size=1))

    if failure_type is RuntimeError:
        getattr(pool, operation)(poisoned)
    else:
        with pytest.raises(Exception) as raised:
            getattr(pool, operation)(poisoned)
        assert raised.value is failure

    expected = ["putconn", "close"] if operation == "return_connection" else ["close", "putconn"]
    assert events == [(event, poisoned) for event in expected]


@pytest.mark.parametrize("delegate", [False, True], ids=["fallback", "native-delegate"])
@pytest.mark.parametrize(
    "failure_type",
    [None, RuntimeError, Exception],
    ids=["success", "classified", "unclassified"],
)
def test_checkout_scope_preserves_connection_and_error_policy(monkeypatch, delegate, failure_type):
    _configure_fake_psycopg(monkeypatch)
    raw = _DummyConn()
    if delegate:
        monkeypatch.setattr(
            pg_backend,
            "psycopg_pool",
            SimpleNamespace(
                ConnectionPool=lambda *_args, **_kwargs: SimpleNamespace(getconn=lambda: raw, close=lambda: None)
            ),
        )
    pool = pg_backend.PostgreSQLConnectionPool(DatabaseConfig(backend_type=BackendType.POSTGRESQL))
    calls = []

    def apply_scope(connection):
        calls.append(connection)
        if failure_type is not None:
            raise failure_type("private scope detail")

    try:
        assert not hasattr(pool, "_apply_scope_settings")
        unscoped = pool.get_connection()
        assert unscoped is raw if delegate else isinstance(unscoped, _DummyConn)
        if not delegate:
            pool.return_connection(unscoped)
        pool._apply_scope_settings = apply_scope
        if failure_type is Exception:
            with pytest.raises(Exception, match="private scope detail"):
                pool.get_connection()
        else:
            assert pool.get_connection() is unscoped
        assert calls == [unscoped]
    finally:
        pool.close_all()


@pytest.mark.parametrize("direct", [False, True], ids=["fallback-native-connect", "direct-native-connect"])
def test_native_connect_retains_dsn_factory_identity_and_scope(monkeypatch, direct):
    _configure_fake_psycopg(monkeypatch)
    raw = _DummyConn()
    calls = []
    scoped = []

    def _connect(dsn, *, row_factory):
        calls.append((dsn, row_factory))
        raw.row_factory = row_factory
        return raw

    def _apply_scope(self, connection):
        scoped.append(connection)

    monkeypatch.setattr(pg_backend.psycopg, "connect", _connect)
    monkeypatch.setattr(pg_backend.PostgreSQLBackend, "_apply_scope_settings", _apply_scope)
    cfg = DatabaseConfig(
        backend_type=BackendType.POSTGRESQL,
        connection_string="postgresql://contract/owned",
        pool_size=1,
    )
    backend = pg_backend.PostgreSQLBackend(cfg)
    pool = None
    try:
        if direct:
            connection = backend.connect()
        else:
            pool = backend.get_pool()
            connection = pool.get_connection()
        assert connection is raw
        assert calls == [(cfg.connection_string, pg_backend.dict_row)]
        assert raw.row_factory is pg_backend.dict_row
        assert scoped == [raw]
    finally:
        if pool is not None:
            pool.close_all()
        else:
            backend.disconnect(raw)
