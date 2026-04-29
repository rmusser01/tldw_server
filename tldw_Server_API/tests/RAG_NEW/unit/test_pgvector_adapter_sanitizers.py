"""Sanitizer coverage for PGVector adapter fallback logs."""

import pytest

from tldw_Server_API.app.core.RAG.rag_service.vector_stores import pgvector_adapter
from tldw_Server_API.app.core.RAG.rag_service.vector_stores.base import (
    VectorStoreConfig,
    VectorStoreType,
)
from tldw_Server_API.app.core.RAG.rag_service.vector_stores.pgvector_adapter import (
    PGVectorAdapter,
)


pytestmark = pytest.mark.unit


class _LoggerStub:
    def __init__(self) -> None:
        self.debug_records: list[tuple[tuple[object, ...], dict[str, object]]] = []
        self.error_records: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def debug(self, *args: object, **kwargs: object) -> None:
        self.debug_records.append((args, kwargs))

    def error(self, *args: object, **kwargs: object) -> None:
        self.error_records.append((args, kwargs))

    def info(self, *args: object, **kwargs: object) -> None:
        pass


def _adapter() -> PGVectorAdapter:
    return PGVectorAdapter(
        VectorStoreConfig(
            store_type=VectorStoreType.PGVECTOR,
            connection_params={},
            embedding_dim=3,
            user_id="sanitizer-test",
        )
    )


def _assert_no_sensitive_fragments(rendered_log: str) -> None:
    assert "topsecret" not in rendered_log
    assert "pgvector.sqlite" not in rendered_log
    assert "/var/lib/tldw/private" not in rendered_log
    assert "token=secret-token" not in rendered_log
    assert "postgresql://user:" not in rendered_log


def _sensitive_error(action: str) -> RuntimeError:
    return RuntimeError(
        f"{action} failed for postgresql://user:topsecret@db.example/app "
        "using /var/lib/tldw/private/pgvector.sqlite?token=secret-token"
    )


class _BorrowedConnection:
    def __init__(self, conn: object) -> None:
        self.conn = conn

    def __enter__(self) -> object:
        return self.conn

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        return False


class _FakeCursor:
    def __init__(
        self,
        *,
        set_error: Exception | None = None,
        sql_error: Exception | None = None,
        close_error: Exception | None = None,
        rows: list[tuple[object, ...]] | None = None,
    ) -> None:
        self.set_error = set_error
        self.sql_error = sql_error
        self.close_error = close_error
        self.rows = rows or []
        self.executed: list[tuple[str, tuple[object, ...]]] = []

    def execute(self, sql: str, params: tuple[object, ...] = ()) -> None:
        self.executed.append((sql, params))
        if sql.startswith("SET hnsw.ef_search") and self.set_error is not None:
            raise self.set_error
        if not sql.startswith("SET hnsw.ef_search") and self.sql_error is not None:
            raise self.sql_error

    def fetchall(self) -> list[tuple[object, ...]]:
        return self.rows

    def close(self) -> None:
        if self.close_error is not None:
            raise self.close_error


class _FakeConnection:
    def __init__(
        self,
        cursor: _FakeCursor,
        *,
        rollback_error: Exception | None = None,
    ) -> None:
        self._cursor = cursor
        self.rollback_error = rollback_error
        self.committed = False
        self.rollback_called = False

    def cursor(self) -> _FakeCursor:
        return self._cursor

    def commit(self) -> None:
        self.committed = True

    def rollback(self) -> None:
        self.rollback_called = True
        if self.rollback_error is not None:
            raise self.rollback_error


def _borrow_adapter_conn(monkeypatch: pytest.MonkeyPatch, adapter: PGVectorAdapter, conn: object) -> None:
    monkeypatch.setattr(adapter, "_borrow_conn", lambda: _BorrowedConnection(conn))


def _assert_debug_records_sanitized(logger_stub: _LoggerStub) -> None:
    assert logger_stub.debug_records
    rendered_log = repr(logger_stub.debug_records)
    assert "exc_info" not in rendered_log
    _assert_no_sensitive_fragments(rendered_log)


@pytest.mark.asyncio
async def test_initialize_failure_log_omits_raw_exception_and_resets_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()
    adapter = _adapter()
    adapter._conn = object()
    adapter._pool = object()

    def _raise_sensitive_dsn_error(_params: dict[str, object]) -> str:
        raise RuntimeError(
            "failed for postgresql://user:topsecret@db.example/app "
            "using /var/lib/tldw/private/pgvector.sqlite?token=secret-token"
        )

    monkeypatch.setattr(pgvector_adapter, "logger", logger_stub)
    monkeypatch.setattr(adapter, "_build_dsn", _raise_sensitive_dsn_error)

    await adapter.initialize()

    assert adapter._conn is None
    assert adapter._pool is None
    assert adapter._initialized is False
    assert logger_stub.error_records
    _assert_no_sensitive_fragments(repr(logger_stub.error_records))


@pytest.mark.asyncio
async def test_register_vector_failure_log_omits_raw_exception_and_stays_best_effort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()
    adapter = _adapter()
    adapter._conn = object()

    class _FakeVector:
        pass

    def _raise_sensitive_registration_error(_conn: object) -> None:
        raise RuntimeError(
            "registration failed for postgresql://user:topsecret@db.example/app "
            "using /var/lib/tldw/private/pgvector.sqlite?token=secret-token"
        )

    monkeypatch.setattr(pgvector_adapter, "logger", logger_stub)
    monkeypatch.setattr(pgvector_adapter, "_PgVector", _FakeVector)
    monkeypatch.setattr(pgvector_adapter, "_register_pgvector", _raise_sensitive_registration_error)

    await adapter._register_vector_support()

    assert adapter._vector_cls is None
    assert logger_stub.debug_records
    _assert_no_sensitive_fragments(repr(logger_stub.debug_records))


@pytest.mark.asyncio
async def test_exec_set_hnsw_failure_log_omits_raw_exception_and_is_swallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()
    adapter = _adapter()
    cursor = _FakeCursor(set_error=_sensitive_error("SET hnsw"))
    conn = _FakeConnection(cursor)

    monkeypatch.setattr(pgvector_adapter, "logger", logger_stub)
    _borrow_adapter_conn(monkeypatch, adapter, conn)

    await adapter._exec("SELECT 1", ())

    assert conn.committed is True
    assert cursor.executed == [
        ("SET hnsw.ef_search = 64", ()),
        ("SELECT 1", ()),
    ]
    _assert_debug_records_sanitized(logger_stub)


@pytest.mark.asyncio
async def test_query_rollback_failure_log_omits_raw_exception_and_reraises_original_query_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()
    adapter = _adapter()
    original_error = ValueError("original query failed without private details")
    cursor = _FakeCursor(sql_error=original_error)
    conn = _FakeConnection(cursor, rollback_error=_sensitive_error("rollback"))

    monkeypatch.setattr(pgvector_adapter, "logger", logger_stub)
    _borrow_adapter_conn(monkeypatch, adapter, conn)

    with pytest.raises(ValueError, match="original query failed") as exc_info:
        await adapter._query("SELECT * FROM vectors", ())

    assert exc_info.value is original_error
    assert conn.rollback_called is True
    _assert_debug_records_sanitized(logger_stub)


@pytest.mark.asyncio
async def test_exec_rollback_failure_log_omits_raw_exception_and_reraises_original_sql_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()
    adapter = _adapter()
    original_error = ValueError("original exec failed without private details")
    cursor = _FakeCursor(sql_error=original_error)
    conn = _FakeConnection(cursor, rollback_error=_sensitive_error("rollback"))

    monkeypatch.setattr(pgvector_adapter, "logger", logger_stub)
    _borrow_adapter_conn(monkeypatch, adapter, conn)

    with pytest.raises(ValueError, match="original exec failed") as exc_info:
        await adapter._exec("SELECT 1", ())

    assert exc_info.value is original_error
    assert conn.rollback_called is True
    _assert_debug_records_sanitized(logger_stub)


@pytest.mark.asyncio
async def test_query_set_hnsw_failure_log_omits_raw_exception_and_is_swallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()
    adapter = _adapter()
    rows = [("row-1",)]
    cursor = _FakeCursor(set_error=_sensitive_error("SET hnsw"), rows=rows)
    conn = _FakeConnection(cursor)

    monkeypatch.setattr(pgvector_adapter, "logger", logger_stub)
    _borrow_adapter_conn(monkeypatch, adapter, conn)

    result = await adapter._query("SELECT * FROM vectors", ())

    assert result == rows
    assert cursor.executed == [
        ("SET hnsw.ef_search = 64", ()),
        ("SELECT * FROM vectors", ()),
    ]
    _assert_debug_records_sanitized(logger_stub)


@pytest.mark.asyncio
async def test_query_cursor_close_failure_log_omits_raw_exception_and_is_swallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()
    adapter = _adapter()
    rows = [("row-1",)]
    cursor = _FakeCursor(close_error=_sensitive_error("cursor close"), rows=rows)
    conn = _FakeConnection(cursor)

    monkeypatch.setattr(pgvector_adapter, "logger", logger_stub)
    _borrow_adapter_conn(monkeypatch, adapter, conn)

    result = await adapter._query("SELECT * FROM vectors", ())

    assert result == rows
    _assert_debug_records_sanitized(logger_stub)


@pytest.mark.asyncio
async def test_exec_cursor_close_failure_log_omits_raw_exception_and_is_swallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()
    adapter = _adapter()
    cursor = _FakeCursor(close_error=_sensitive_error("cursor close"))
    conn = _FakeConnection(cursor)

    monkeypatch.setattr(pgvector_adapter, "logger", logger_stub)
    _borrow_adapter_conn(monkeypatch, adapter, conn)

    await adapter._exec("SELECT 1", ())

    assert conn.committed is True
    _assert_debug_records_sanitized(logger_stub)
