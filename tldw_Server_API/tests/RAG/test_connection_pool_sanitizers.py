"""Sanitizer coverage for RAG connection pool fallback logs."""

from tldw_Server_API.app.core.RAG.rag_service import connection_pool
from tldw_Server_API.app.core.RAG.rag_service.connection_pool import ConnectionPool


def _capture_logs(level: str = "DEBUG") -> tuple[list[str], int]:
    messages: list[str] = []
    sink_id = connection_pool.logger.add(
        lambda message: messages.append(str(message)),
        level=level,
    )
    return messages, sink_id


def test_create_connection_sanitizes_failure_log(monkeypatch, tmp_path):
    """Connection creation fallback should not expose sqlite exception details."""

    def broken_connect(*_args, **_kwargs):
        raise RuntimeError("sqlite open failed for /private/rag-create.db?token=secret")

    monkeypatch.setattr(connection_pool.sqlite3, "connect", broken_connect)
    pool = ConnectionPool(str(tmp_path / "safe.db"), min_connections=0)

    messages, sink_id = _capture_logs(level="ERROR")
    try:
        assert pool._create_connection() is None
    finally:
        connection_pool.logger.remove(sink_id)

    joined = "\n".join(messages)
    assert "Failed to create database connection" in joined
    assert "rag-create.db" not in joined
    assert "secret" not in joined


def test_connection_validation_sanitizes_failure_log(tmp_path):
    """Connection validation fallback should not expose sqlite exception details."""

    class BrokenConnection:
        def execute(self, _sql):
            raise RuntimeError("validation failed for /private/rag-validate.db?token=secret")

    pool = ConnectionPool(str(tmp_path / "safe.db"), min_connections=0)

    messages, sink_id = _capture_logs()
    try:
        assert pool._is_connection_valid(BrokenConnection()) is False
    finally:
        connection_pool.logger.remove(sink_id)

    joined = "\n".join(messages)
    assert "Connection validation failed" in joined
    assert "rag-validate.db" not in joined
    assert "secret" not in joined


def test_close_connection_sanitizes_failure_log(tmp_path):
    """Connection close fallback should not expose sqlite exception details."""

    class BrokenConnection:
        def close(self):
            raise RuntimeError("close failed for /private/rag-close.db?token=secret")

    pool = ConnectionPool(str(tmp_path / "safe.db"), min_connections=0)

    messages, sink_id = _capture_logs(level="ERROR")
    try:
        pool._close_connection(BrokenConnection())
    finally:
        connection_pool.logger.remove(sink_id)

    joined = "\n".join(messages)
    assert "Error closing database connection" in joined
    assert "rag-close.db" not in joined
    assert "secret" not in joined


def test_shutdown_close_sanitizes_remaining_connection_failure_log(tmp_path):
    """Shutdown fallback should not expose raw close exception details."""

    class BrokenConnection:
        def close(self):
            raise RuntimeError("shutdown failed for /private/rag-shutdown.db?token=secret")

    pool = ConnectionPool(str(tmp_path / "safe.db"), min_connections=0)
    pool._all_connections[id(BrokenConnection)] = {"connection": BrokenConnection()}

    messages, sink_id = _capture_logs()
    try:
        pool.close()
    finally:
        connection_pool.logger.remove(sink_id)

    joined = "\n".join(messages)
    assert "Error closing pooled connection during shutdown" in joined
    assert "rag-shutdown.db" not in joined
    assert "secret" not in joined


def test_close_idle_connections_sanitizes_close_failure_log(monkeypatch, tmp_path):
    """Idle close fallback should not expose raw close exception details."""

    class FakeConnection:
        pass

    pool = ConnectionPool(str(tmp_path / "safe.db"), min_connections=0)
    pool._all_connections[id(FakeConnection)] = {
        "connection": FakeConnection(),
        "last_used": 0,
        "in_use": False,
    }
    pool.max_idle_time = 0

    def broken_close(_conn):
        raise RuntimeError("idle close failed for /private/rag-idle.db?token=secret")

    monkeypatch.setattr(pool, "_close_connection", broken_close)

    messages, sink_id = _capture_logs(level="ERROR")
    try:
        pool.close_idle_connections()
    finally:
        connection_pool.logger.remove(sink_id)

    joined = "\n".join(messages)
    assert "Error closing idle connection" in joined
    assert "rag-idle.db" not in joined
    assert "secret" not in joined
