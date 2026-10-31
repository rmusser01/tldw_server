import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType, DatabaseConfig, DatabaseError
from tldw_Server_API.app.core.DB_Management.backends.sqlite_backend import SQLiteConnectionPool


def test_sqlite_connection_pool_rejects_after_close() -> None:
    config = DatabaseConfig(backend_type=BackendType.SQLITE, sqlite_path=":memory:")
    pool = SQLiteConnectionPool(config.sqlite_path or ":memory:", config)

    conn = pool.get_connection()
    assert conn is not None

    pool.close_all()
    with pytest.raises(DatabaseError):
        pool.get_connection()


def test_close_all_does_not_close_another_threads_connection_mid_statement(tmp_path) -> None:
    """Closing the pool must not pull a connection out from under another thread.

    CI segfaulted when app shutdown closed every pooled connection while a
    background loop on another thread was mid-query on its own connection.
    """
    import threading

    config = DatabaseConfig(backend_type=BackendType.SQLITE, sqlite_path=str(tmp_path / "pool.db"))
    pool = SQLiteConnectionPool(config.sqlite_path, config)
    inside = threading.Event()
    release = threading.Event()
    outcome: dict[str, object] = {}

    def block() -> int:
        inside.set()
        release.wait(5)
        return 1

    def worker() -> None:
        conn = pool.get_connection()
        conn.create_function("block", 0, block)
        try:
            # The pool closes while this statement is in flight; the owner's
            # connection must stay usable for the work it already holds it for.
            outcome["value"] = conn.execute("SELECT block()").fetchone()[0]
            outcome["after"] = conn.execute("SELECT 2").fetchone()[0]
        except Exception as exc:  # noqa: BLE001 - the assertion reports it
            outcome["error"] = exc

    thread = threading.Thread(target=worker)
    thread.start()
    assert inside.wait(5)
    pool.close_all()
    release.set()
    thread.join(5)

    assert outcome == {"value": 1, "after": 2}
    with pytest.raises(DatabaseError):
        pool.get_connection()
