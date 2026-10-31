import io

import pytest
from loguru import logger

import tldw_Server_API.app.core.DB_Management.backends.postgresql_backend as postgresql_backend
from tldw_Server_API.app.core.DB_Management.backends.base import (
    BackendType,
    DatabaseConfig,
    DatabaseError,
)
from tldw_Server_API.app.core.DB_Management.backends.postgresql_backend import (
    PostgreSQLBackend,
)


class DummyConn:
    def __init__(self):
        self.commits = 0
        self.rollbacks = 0
        self.closed = False

    def commit(self):

        self.commits += 1

    def rollback(self):

        self.rollbacks += 1

    def close(self):

        self.closed = True


class _CursorNoStatus:
    description = None

    def __init__(self, rowcount: int = 0):
        self.rowcount = rowcount

    def executemany(self, _query, _params):
        return None


class _CursorRaises:
    description = None
    rowcount = -1

    def __init__(self, exc: Exception):
        self._exc = exc

    def execute(self, _query, _params=None):
        raise self._exc


class _CursorManyRaises(_CursorRaises):
    def executemany(self, _query, _params):
        raise self._exc


class _ConnWithCursor(DummyConn):
    def __init__(self, cursor):
        super().__init__()
        self._cursor = cursor

    def cursor(self):
        return self._cursor


def _pg_backend():

    # Construct without touching real pool/psycopg; tests pass a connection explicitly
    return PostgreSQLBackend(DatabaseConfig(backend_type=BackendType.POSTGRESQL))


def test_transaction_outermost_commits_with_external_connection():

    backend = _pg_backend()
    conn = DummyConn()

    # Single outermost transaction should commit exactly once
    with backend.transaction(connection=conn):
        pass

    assert conn.commits == 1
    assert conn.rollbacks == 0


def test_transaction_nested_commits_once_with_external_connection():

    backend = _pg_backend()
    conn = DummyConn()

    # Nested transactions on same connection should only commit at outermost
    with backend.transaction(connection=conn):
        with backend.transaction(connection=conn):
            pass

    assert conn.commits == 1
    assert conn.rollbacks == 0


def test_transaction_rollback_on_exception_with_external_connection():

    backend = _pg_backend()
    conn = DummyConn()

    with pytest.raises(RuntimeError):
        with backend.transaction(connection=conn):
            raise RuntimeError("boom")

    assert conn.commits == 0
    assert conn.rollbacks == 1


def test_transaction_nested_rollback_only_once_on_exception_with_external_connection():

    backend = _pg_backend()
    conn = DummyConn()

    with pytest.raises(ValueError):
        with backend.transaction(connection=conn):
            with backend.transaction(connection=conn):
                raise ValueError("fail nested")

    # Only outermost rollback should occur
    assert conn.commits == 0
    assert conn.rollbacks == 1


def test_transaction_rolls_back_unclassified_body_exception():
    class UnclassifiedFailure(Exception):
        pass

    backend = _pg_backend()
    conn = DummyConn()

    with pytest.raises(UnclassifiedFailure):
        with backend.transaction(connection=conn):
            raise UnclassifiedFailure("private transaction detail")

    assert conn.commits == 0
    assert conn.rollbacks == 1


def test_transaction_preserves_base_exception_and_rolls_back():
    class ControlSignal(BaseException):
        pass

    backend = _pg_backend()
    conn = DummyConn()
    signal = ControlSignal()

    with pytest.raises(ControlSignal) as raised:
        with backend.transaction(connection=conn):
            raise signal

    assert raised.value is signal
    assert conn.commits == 0
    assert conn.rollbacks == 1


def test_transaction_sanitizes_commit_failure_and_rolls_back(monkeypatch):
    class DriverBoom(Exception):
        pass

    class CommitFailingConn(DummyConn):
        def commit(self):
            raise DriverBoom("private transaction detail")

    monkeypatch.setattr(
        postgresql_backend,
        "_POSTGRES_BACKEND_NONCRITICAL_EXCEPTIONS",
        tuple(postgresql_backend._POSTGRES_BACKEND_NONCRITICAL_EXCEPTIONS) + (DriverBoom,),
        raising=True,
    )
    backend = _pg_backend()
    conn = CommitFailingConn()
    output = io.StringIO()
    sink = logger.add(output, format="{message} {extra}")
    try:
        with pytest.raises(DatabaseError) as raised:
            with backend.transaction(connection=conn):
                pass
    finally:
        logger.remove(sink)

    assert str(raised.value) == "PostgreSQL transaction commit failed"
    assert raised.value.__cause__ is None
    assert conn.rollbacks == 1
    assert "private transaction detail" not in output.getvalue()


def test_execute_many_handles_cursor_without_statusmessage():

    backend = _pg_backend()
    conn = _ConnWithCursor(_CursorNoStatus(rowcount=2))

    result = backend.execute_many(
        "INSERT INTO demo(value) VALUES (?)",
        [(1,), (2,)],
        connection=conn,
    )

    assert result.rowcount == 2


def test_execute_sanitizes_classified_driver_exceptions(monkeypatch):

    class DriverBoom(Exception):
        pass

    backend = _pg_backend()
    sentinel = "private@example.com constraint=users_email_key"
    conn = _ConnWithCursor(_CursorRaises(DriverBoom(sentinel)))

    monkeypatch.setattr(
        postgresql_backend,
        "_POSTGRES_BACKEND_NONCRITICAL_EXCEPTIONS",
        tuple(postgresql_backend._POSTGRES_BACKEND_NONCRITICAL_EXCEPTIONS) + (DriverBoom,),
        raising=True,
    )

    output = io.StringIO()
    sink = logger.add(output, format="{message} {extra}")
    try:
        with pytest.raises(DatabaseError) as raised:
            backend.execute("SELECT 1", connection=conn)
    finally:
        logger.remove(sink)

    assert str(raised.value) == "PostgreSQL query execution failed"
    assert raised.value.__cause__ is None
    assert sentinel not in output.getvalue()


def test_execute_many_sanitizes_classified_driver_exceptions(monkeypatch):
    class DriverBoom(Exception):
        pass

    backend = _pg_backend()
    sentinel = "private@example.com constraint=users_email_key"
    conn = _ConnWithCursor(_CursorManyRaises(DriverBoom(sentinel)))
    monkeypatch.setattr(
        postgresql_backend,
        "_POSTGRES_BACKEND_NONCRITICAL_EXCEPTIONS",
        tuple(postgresql_backend._POSTGRES_BACKEND_NONCRITICAL_EXCEPTIONS) + (DriverBoom,),
        raising=True,
    )

    output = io.StringIO()
    sink = logger.add(output, format="{message} {extra}")
    try:
        with pytest.raises(DatabaseError) as raised:
            backend.execute_many(
                "INSERT INTO demo(value) VALUES (?)",
                [(1,)],
                connection=conn,
            )
    finally:
        logger.remove(sink)

    assert str(raised.value) == "PostgreSQL batch execution failed"
    assert raised.value.__cause__ is None
    assert sentinel not in output.getvalue()


def test_psycopg_error_classified_when_driver_available():

    if not postgresql_backend.PSYCOPG2_AVAILABLE:
        pytest.skip("psycopg not installed in test environment")

    psycopg_error = getattr(postgresql_backend.psycopg, "Error", None)
    assert psycopg_error in postgresql_backend._POSTGRES_BACKEND_NONCRITICAL_EXCEPTIONS
