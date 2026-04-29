"""Sanitizer coverage for PGVector adapter fallback logs."""

import sys
import types

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
    assert "RAW_EXCEPTION_MARKER" not in rendered_log
    assert "topsecret" not in rendered_log
    assert "pgvector.sqlite" not in rendered_log
    assert "/var/lib/tldw/private" not in rendered_log
    assert "token=secret-token" not in rendered_log
    assert "postgresql://user:" not in rendered_log


def _sensitive_error(action: str) -> RuntimeError:
    return RuntimeError(
        f"RAW_EXCEPTION_MARKER {action} failed for postgresql://user:topsecret@db.example/app "
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
        rowcount: int = 0,
    ) -> None:
        self.set_error = set_error
        self.sql_error = sql_error
        self.close_error = close_error
        self.rows = rows or []
        self.rowcount = rowcount
        self.executed: list[tuple[str, tuple[object, ...]]] = []
        self.executemany_calls: list[tuple[str, list[tuple[object, ...]]]] = []

    def execute(self, sql: str, params: tuple[object, ...] = ()) -> None:
        self.executed.append((sql, params))
        if sql.startswith("SET hnsw.ef_search") and self.set_error is not None:
            raise self.set_error
        if not sql.startswith("SET hnsw.ef_search") and self.sql_error is not None:
            raise self.sql_error

    def executemany(self, sql: str, args: list[tuple[object, ...]]) -> None:
        self.executemany_calls.append((sql, args))
        if self.sql_error is not None:
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


class _NoopTimer:
    def __enter__(self) -> "_NoopTimer":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        return False


class _NoopHistogram:
    def labels(self, **_kwargs: object) -> "_NoopHistogram":
        return self

    def time(self) -> _NoopTimer:
        return _NoopTimer()


class _FailingCounter:
    def labels(self, **_kwargs: object) -> "_FailingCounter":
        return self

    def inc(self, _amount: int) -> None:
        raise _sensitive_error("metrics increment")


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
async def test_initialize_pool_fallback_log_omits_raw_exception_and_uses_single_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()
    adapter = _adapter()
    connection = object()

    psycopg = types.ModuleType("psycopg")
    psycopg.connect = lambda _dsn: connection

    psycopg_pool = types.ModuleType("psycopg_pool")

    class _FailingConnectionPool:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise _sensitive_error("pool init")

    psycopg_pool.ConnectionPool = _FailingConnectionPool

    async def _noop_register_vector_support() -> None:
        pass

    async def _noop_exec(_sql: str, _params: tuple[object, ...] | None = None) -> None:
        pass

    monkeypatch.setattr(pgvector_adapter, "logger", logger_stub)
    monkeypatch.setitem(sys.modules, "psycopg", psycopg)
    monkeypatch.setitem(sys.modules, "psycopg_pool", psycopg_pool)
    monkeypatch.setattr(adapter, "_register_vector_support", _noop_register_vector_support)
    monkeypatch.setattr(adapter, "_exec", _noop_exec)

    await adapter.initialize()

    assert adapter._pool is None
    assert adapter._conn is connection
    assert adapter._driver == "psycopg"
    assert adapter._initialized is True
    _assert_debug_records_sanitized(logger_stub)


@pytest.mark.asyncio
async def test_initialize_psycopg_connect_fallback_log_omits_raw_exception_and_uses_psycopg2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()
    adapter = _adapter()
    connection = object()

    psycopg = types.ModuleType("psycopg")

    def _fail_psycopg_connect(_dsn: str) -> object:
        raise _sensitive_error("psycopg connect")

    psycopg.connect = _fail_psycopg_connect
    psycopg2 = types.ModuleType("psycopg2")
    psycopg2.connect = lambda _dsn: connection

    async def _noop_register_vector_support() -> None:
        pass

    async def _noop_exec(_sql: str, _params: tuple[object, ...] | None = None) -> None:
        pass

    monkeypatch.setattr(pgvector_adapter, "logger", logger_stub)
    monkeypatch.setitem(sys.modules, "psycopg", psycopg)
    monkeypatch.delitem(sys.modules, "psycopg_pool", raising=False)
    monkeypatch.setitem(sys.modules, "psycopg2", psycopg2)
    monkeypatch.setattr(adapter, "_register_vector_support", _noop_register_vector_support)
    monkeypatch.setattr(adapter, "_exec", _noop_exec)

    await adapter.initialize()

    assert adapter._pool is None
    assert adapter._conn is connection
    assert adapter._driver == "psycopg2"
    assert adapter._initialized is True
    _assert_debug_records_sanitized(logger_stub)


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


@pytest.mark.asyncio
async def test_create_collection_index_failure_log_omits_raw_exception_and_is_swallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()
    adapter = _adapter()
    calls: list[str] = []

    async def _fake_exec(sql: str, params: tuple[object, ...] | None = None) -> None:
        del params
        calls.append(sql)
        if "CREATE INDEX" in sql:
            raise _sensitive_error("create index")

    monkeypatch.setattr(pgvector_adapter, "logger", logger_stub)
    monkeypatch.setattr(adapter, "_exec", _fake_exec)

    await adapter.create_collection("private collection")

    assert any("CREATE TABLE" in call for call in calls)
    assert any("USING hnsw" in call for call in calls)
    assert any("USING ivfflat" in call for call in calls)
    assert any(call.startswith("ANALYZE") for call in calls)
    _assert_debug_records_sanitized(logger_stub)


@pytest.mark.asyncio
async def test_create_collection_analyze_failure_log_omits_raw_exception_and_is_swallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()
    adapter = _adapter()

    async def _fake_exec(sql: str, params: tuple[object, ...] | None = None) -> None:
        del params
        if sql.startswith("ANALYZE"):
            raise _sensitive_error("analyze")

    monkeypatch.setattr(pgvector_adapter, "logger", logger_stub)
    monkeypatch.setattr(adapter, "_exec", _fake_exec)

    await adapter.create_collection("private collection")

    _assert_debug_records_sanitized(logger_stub)


@pytest.mark.asyncio
async def test_rebuild_index_drop_fallback_logs_omit_raw_exception_and_return_index_info(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()
    adapter = _adapter()
    expected_info = {"backend": "pgvector", "index_type": "none"}

    async def _fake_query(sql: str, params: tuple[object, ...] | None = None) -> list[tuple[str]]:
        if "tablename" in sql:
            return [("private_embedding_hnsw",)]
        if "indexname" in sql:
            return [("CREATE INDEX private_embedding_hnsw ON vs_private USING hnsw (embedding)",)]
        return []

    async def _fake_exec(sql: str, params: tuple[object, ...] | None = None) -> None:
        del params
        if sql.startswith("DROP INDEX"):
            raise _sensitive_error("drop index")
        if sql.startswith("ANALYZE"):
            raise _sensitive_error("analyze after drop")

    async def _fake_index_info(collection_name: str) -> dict[str, str]:
        assert collection_name == "private collection"
        return expected_info

    monkeypatch.setattr(pgvector_adapter, "logger", logger_stub)
    monkeypatch.setattr(adapter, "_query", _fake_query)
    monkeypatch.setattr(adapter, "_exec", _fake_exec)
    monkeypatch.setattr(adapter, "get_index_info", _fake_index_info)

    result = await adapter.rebuild_index("private collection", index_type="drop")

    assert result == expected_info
    assert len(logger_stub.debug_records) == 2
    _assert_debug_records_sanitized(logger_stub)


@pytest.mark.asyncio
async def test_rebuild_index_optimize_analyze_fallback_log_omits_raw_exception_and_returns_index_info(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()
    adapter = _adapter()
    expected_info = {"backend": "pgvector", "index_type": "hnsw"}

    async def _fake_query(sql: str, params: tuple[object, ...] | None = None) -> list[tuple[str]]:
        del sql, params
        return []

    async def _fake_exec(sql: str, params: tuple[object, ...] | None = None) -> None:
        del params
        if sql.startswith("ANALYZE"):
            raise _sensitive_error("analyze after optimize")

    async def _fake_index_info(collection_name: str) -> dict[str, str]:
        assert collection_name == "private collection"
        return expected_info

    monkeypatch.setattr(pgvector_adapter, "logger", logger_stub)
    monkeypatch.setattr(adapter, "_query", _fake_query)
    monkeypatch.setattr(adapter, "_exec", _fake_exec)
    monkeypatch.setattr(adapter, "get_index_info", _fake_index_info)

    result = await adapter.rebuild_index("private collection", index_type="hnsw")

    assert result == expected_info
    _assert_debug_records_sanitized(logger_stub)


@pytest.mark.asyncio
async def test_optimize_collection_analyze_fallback_log_omits_raw_exception_and_is_swallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()
    adapter = _adapter()

    async def _fake_exec(sql: str, params: tuple[object, ...] | None = None) -> None:
        del params
        assert sql.startswith("ANALYZE")
        raise _sensitive_error("optimize collection analyze")

    monkeypatch.setattr(pgvector_adapter, "logger", logger_stub)
    monkeypatch.setattr(adapter, "_exec", _fake_exec)

    result = await adapter.optimize_collection("private collection")

    assert result is None
    _assert_debug_records_sanitized(logger_stub)


@pytest.mark.asyncio
async def test_close_fallback_logs_omit_raw_exception_and_clear_connections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()
    adapter = _adapter()
    adapter._initialized = True

    class _FailingPool:
        def close(self) -> None:
            raise _sensitive_error("pool close")

    class _FailingConnection:
        def close(self) -> None:
            raise _sensitive_error("connection close")

    adapter._pool = _FailingPool()
    adapter._conn = _FailingConnection()

    monkeypatch.setattr(pgvector_adapter, "logger", logger_stub)

    result = await adapter.close()

    assert result is None
    assert adapter._pool is None
    assert adapter._conn is None
    assert adapter._initialized is False
    assert len(logger_stub.debug_records) == 2
    _assert_debug_records_sanitized(logger_stub)


@pytest.mark.asyncio
async def test_health_pool_stats_fallback_log_omits_raw_exception_and_preserves_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()
    adapter = _adapter()
    adapter._driver = "psycopg_pool"

    class _FailingStatsPool:
        def __getattribute__(self, name: str) -> object:
            if name == "min_size":
                raise _sensitive_error("pool stats")
            return super().__getattribute__(name)

    async def _fake_query(sql: str, params: tuple[object, ...] | None = None) -> list[tuple[int]]:
        assert sql == "SELECT 1"
        assert params is None
        return [(1,)]

    adapter._pool = _FailingStatsPool()

    monkeypatch.setattr(pgvector_adapter, "logger", logger_stub)
    monkeypatch.setattr(adapter, "_query", _fake_query)

    result = await adapter.health()

    assert result == {"driver": "psycopg_pool", "ok": True}
    _assert_debug_records_sanitized(logger_stub)


@pytest.mark.asyncio
async def test_upsert_operation_fallback_logs_omit_raw_exception_and_are_swallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()
    adapter = _adapter()
    cursor = _FakeCursor(
        set_error=_sensitive_error("upsert SET hnsw"),
        close_error=_sensitive_error("upsert cursor close"),
    )
    conn = _FakeConnection(cursor)

    monkeypatch.setattr(pgvector_adapter, "logger", logger_stub)
    monkeypatch.setattr(adapter, "_H_UPSERT_LAT", _NoopHistogram())
    monkeypatch.setattr(adapter, "_C_ROWS_UPSERTED", _FailingCounter())
    _borrow_adapter_conn(monkeypatch, adapter, conn)

    await adapter.upsert_vectors(
        "private collection",
        ["id-1"],
        [[1.0, 2.0, 3.0]],
        ["document"],
        [{"source": "test"}],
    )

    assert conn.committed is True
    assert cursor.executemany_calls
    assert len(logger_stub.debug_records) == 3
    _assert_debug_records_sanitized(logger_stub)


@pytest.mark.asyncio
async def test_delete_vectors_operation_fallback_logs_omit_raw_exception_and_are_swallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()
    adapter = _adapter()
    cursor = _FakeCursor(close_error=_sensitive_error("delete cursor close"), rowcount=1)
    conn = _FakeConnection(cursor)

    monkeypatch.setattr(pgvector_adapter, "logger", logger_stub)
    monkeypatch.setattr(adapter, "_H_DELETE_LAT", _NoopHistogram())
    monkeypatch.setattr(adapter, "_C_ROWS_DELETED", _FailingCounter())
    _borrow_adapter_conn(monkeypatch, adapter, conn)

    await adapter.delete_vectors("private collection", ["id-1"])

    assert conn.committed is True
    assert cursor.executemany_calls
    assert len(logger_stub.debug_records) == 2
    _assert_debug_records_sanitized(logger_stub)


@pytest.mark.asyncio
async def test_delete_by_filter_operation_fallback_logs_omit_raw_exception_and_are_swallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()
    adapter = _adapter()
    cursor = _FakeCursor(
        set_error=_sensitive_error("delete_by_filter SET hnsw"),
        close_error=_sensitive_error("delete_by_filter cursor close"),
        rowcount=1,
    )
    conn = _FakeConnection(cursor)

    monkeypatch.setattr(pgvector_adapter, "logger", logger_stub)
    monkeypatch.setattr(adapter, "_H_DELETE_LAT", _NoopHistogram())
    monkeypatch.setattr(adapter, "_C_ROWS_DELETED", _FailingCounter())
    _borrow_adapter_conn(monkeypatch, adapter, conn)

    result = await adapter.delete_by_filter("private collection", {"source": "test"})

    assert result == 1
    assert conn.committed is True
    assert len(logger_stub.debug_records) == 3
    _assert_debug_records_sanitized(logger_stub)
