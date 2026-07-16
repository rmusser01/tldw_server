"""Regression coverage for backend failures on redacted source-query paths."""

from __future__ import annotations

from typing import Any

import pytest
from loguru import logger

from tldw_Server_API.app.core.DB_Management.backends.base import (
    BackendType,
    DatabaseConfig,
    QueryResult,
)
from tldw_Server_API.app.core.DB_Management.backends.base import (
    DatabaseError as BackendDatabaseError,
)
from tldw_Server_API.app.core.DB_Management.backends.postgresql_backend import (
    PostgreSQLBackend,
)
from tldw_Server_API.app.core.DB_Management.backends.sqlite_backend import SQLiteBackend
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    BackendConnectionWrapper,
    CharactersRAGDB,
    CharactersRAGDBError,
)
from tldw_Server_API.app.core.DB_Management.media_db.errors import DatabaseError
from tldw_Server_API.app.core.DB_Management.media_db.runtime.execution_ops import (
    execute_query as execute_media_query,
)

_QUERY_SECRET = "PRIVATE_POSTGRES_SOURCE_FRAGMENT"
_ROLLBACK_SECRET = "PRIVATE_POSTGRES_ROLLBACK_FRAGMENT"
_COMMIT_SECRET = "PRIVATE_POSTGRES_COMMIT_FRAGMENT"


class _FailingCursor:
    description = None
    rowcount = -1
    statusmessage = None

    @staticmethod
    def execute(_query: str, _params: Any = None) -> None:
        raise RuntimeError(f"driver could not decode {_QUERY_SECRET}")


class _FailingConnection:
    closed = False

    @staticmethod
    def cursor() -> _FailingCursor:
        return _FailingCursor()

    @staticmethod
    def rollback() -> None:
        raise RuntimeError(f"rollback failed near {_ROLLBACK_SECRET}")


class _Pool:
    def __init__(self, connection: _FailingConnection) -> None:
        self.connection = connection
        self.returned: list[_FailingConnection] = []

    def get_connection(self) -> _FailingConnection:
        return self.connection

    def return_connection(self, connection: _FailingConnection) -> None:
        self.returned.append(connection)


def _backend(connection: _FailingConnection | None = None) -> tuple[PostgreSQLBackend, _Pool]:
    backend = PostgreSQLBackend(DatabaseConfig(backend_type=BackendType.POSTGRESQL))
    pool = _Pool(connection or _FailingConnection())
    backend._pool = pool
    return backend, pool


def _exception_chain_text(exc: BaseException) -> str:
    pending = [exc]
    seen: set[int] = set()
    values: list[str] = []
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        values.append(repr(current))
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)
    return "\n".join(values)


def _capture_messages() -> tuple[list[str], int]:
    messages: list[str] = []
    sink_id = logger.add(messages.append, level="DEBUG", format="{message}")
    return messages, sink_id


@pytest.mark.unit
def test_postgres_backend_redacts_query_and_rollback_failures() -> None:
    backend, pool = _backend()
    messages, sink_id = _capture_messages()
    try:
        with pytest.raises(BackendDatabaseError) as exc_info:
            backend.execute("SELECT source_text FROM private_sources", log_errors=False)
    finally:
        logger.remove(sink_id)

    combined = "\n".join(messages) + _exception_chain_text(exc_info.value)
    assert _QUERY_SECRET not in combined
    assert _ROLLBACK_SECRET not in combined
    assert str(exc_info.value) == "PostgreSQL query execution failed"
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert pool.returned == [pool.connection]


@pytest.mark.unit
def test_sqlite_backend_honors_shared_error_redaction_contract() -> None:
    backend = SQLiteBackend(
        DatabaseConfig(
            backend_type=BackendType.SQLITE,
            sqlite_path=":memory:",
        )
    )
    messages, sink_id = _capture_messages()
    try:
        with pytest.raises(BackendDatabaseError) as exc_info:
            backend.execute(
                f"SELECT * FROM {_QUERY_SECRET}",
                log_errors=False,
            )
    finally:
        logger.remove(sink_id)
        backend.get_pool().close_all()

    combined = "\n".join(messages) + _exception_chain_text(exc_info.value)
    assert _QUERY_SECRET not in combined
    assert str(exc_info.value) == "SQLite query execution failed"
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


@pytest.mark.unit
def test_chacha_redacted_query_reaches_postgres_backend() -> None:
    backend, _pool = _backend()
    connection = _FailingConnection()

    class _ChaChaHarness:
        backend_type = BackendType.POSTGRESQL

        @staticmethod
        def _prepare_backend_statement(query: str, params: Any = None) -> tuple[str, Any]:
            return query, params

        def get_connection(self) -> BackendConnectionWrapper:
            return BackendConnectionWrapper(self, connection, backend)

    messages, sink_id = _capture_messages()
    try:
        with pytest.raises(CharactersRAGDBError) as exc_info:
            CharactersRAGDB.execute_query(
                _ChaChaHarness(),
                "SELECT content FROM messages",
                log_params=False,
                log_errors=False,
            )
    finally:
        logger.remove(sink_id)

    combined = "\n".join(messages) + _exception_chain_text(exc_info.value)
    assert _QUERY_SECRET not in combined
    assert str(exc_info.value) == "Query execution failed."


@pytest.mark.unit
def test_media_redacted_query_reaches_postgres_backend() -> None:
    backend, pool = _backend()

    class _MediaHarness:
        backend_type = BackendType.POSTGRESQL
        is_memory_db = False
        db_path_str = "unused"

        def __init__(self) -> None:
            self.backend = backend

        @staticmethod
        def _prepare_backend_statement(query: str, params: Any = None) -> tuple[str, Any]:
            return query, params

        @staticmethod
        def _get_txn_conn() -> None:
            return None

    messages, sink_id = _capture_messages()
    try:
        with pytest.raises(DatabaseError) as exc_info:
            execute_media_query(
                _MediaHarness(),
                "SELECT source_text FROM media",
                log_errors=False,
            )
    finally:
        logger.remove(sink_id)

    combined = "\n".join(messages) + _exception_chain_text(exc_info.value)
    assert _QUERY_SECRET not in combined
    assert _ROLLBACK_SECRET not in combined
    assert str(exc_info.value) == "Backend query execution failed."
    assert pool.returned == [pool.connection]


@pytest.mark.unit
def test_media_redacts_external_postgres_commit_failure() -> None:
    class _SuccessfulBackend:
        backend_type = BackendType.POSTGRESQL

        @staticmethod
        def execute(_query: str, _params: Any = None, **kwargs: Any) -> QueryResult:
            assert kwargs["connection"] is connection
            assert kwargs["log_errors"] is False
            return QueryResult(rows=[], rowcount=0)

    class _CommitConnection:
        @staticmethod
        def commit() -> None:
            raise RuntimeError(f"commit failed near {_COMMIT_SECRET}")

    connection = _CommitConnection()

    class _MediaHarness:
        backend_type = BackendType.POSTGRESQL
        is_memory_db = False
        db_path_str = "unused"
        backend = _SuccessfulBackend()

        @staticmethod
        def _prepare_backend_statement(query: str, params: Any = None) -> tuple[str, Any]:
            return query, params

        @staticmethod
        def _get_txn_conn() -> _CommitConnection:
            return connection

    messages, sink_id = _capture_messages()
    try:
        with pytest.raises(DatabaseError) as exc_info:
            execute_media_query(
                _MediaHarness(),
                "UPDATE media SET source_text = NULL",
                commit=True,
                log_errors=False,
            )
    finally:
        logger.remove(sink_id)

    combined = "\n".join(messages) + _exception_chain_text(exc_info.value)
    assert _COMMIT_SECRET not in combined
    assert str(exc_info.value) == "Backend commit failed."
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
