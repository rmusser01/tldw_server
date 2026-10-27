"""Regression tests for PostgreSQL pool checkout cleanup and scoping."""

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


@pytest.mark.parametrize(
    "failure_type",
    [None, RuntimeError, Exception],
    ids=["success", "classified", "unclassified"],
)
def test_checkout_scope_failure_discards_the_checkout(monkeypatch, failure_type):
    """No scope, no connection: a checkout that cannot be scoped is never handed out."""
    _configure_fake_psycopg(monkeypatch)
    raw = _DummyConn()
    returned: list[_DummyConn] = []
    monkeypatch.setattr(
        pg_backend,
        "psycopg_pool",
        SimpleNamespace(
            ConnectionPool=lambda *_args, **_kwargs: SimpleNamespace(
                getconn=lambda: raw, putconn=returned.append, close=lambda: None
            )
        ),
    )
    pool = pg_backend.PostgreSQLConnectionPool(DatabaseConfig(backend_type=BackendType.POSTGRESQL))

    def apply_scope(connection):
        if failure_type is not None:
            raise failure_type("private scope detail")

    pool._apply_scope_settings = apply_scope
    try:
        if failure_type is None:
            assert pool.get_connection() is raw
            assert (raw.closed, returned) == (False, [])
        else:
            with pytest.raises(failure_type, match="private scope detail"):
                pool.get_connection()
            assert (raw.closed, returned) == (True, [raw])
    finally:
        pool.close_all()


def test_pool_refuses_to_start_without_psycopg_pool(monkeypatch):
    """There is no hand-rolled fallback pool: it never reset tenant state on return."""
    _configure_fake_psycopg(monkeypatch)

    with pytest.raises(pg_backend.DatabaseError, match="psycopg-pool is not installed"):
        pg_backend.PostgreSQLConnectionPool(DatabaseConfig(backend_type=BackendType.POSTGRESQL))


def test_native_connect_retains_dsn_factory_identity_and_scope(monkeypatch):
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
    connection = backend.connect()
    try:
        assert connection is raw
        assert calls == [(cfg.connection_string, pg_backend.dict_row)]
        assert raw.row_factory is pg_backend.dict_row
        assert scoped == [raw]
    finally:
        backend.disconnect(raw)
