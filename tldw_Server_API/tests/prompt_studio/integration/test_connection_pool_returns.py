import threading

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseConfig
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.PromptStudioDatabase import (
    BackendPromptStudioDatabaseBase,
    PromptStudioDatabase,
)
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.services import (
    jobs_worker,
)

pytestmark = pytest.mark.integration


def test_prompt_studio_pool_returns_connections(pg_database_config: DatabaseConfig, tmp_path):
    pg_database_config.pool_size = 1
    pg_database_config.max_overflow = 0
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = PromptStudioDatabase(
        db_path=str(tmp_path / "prompt_studio_pool.sqlite"),
        client_id="pool-ps",
        backend=backend,
    )

    pool = backend.get_pool()
    counts = {"get": 0, "return": 0}
    orig_get = pool.get_connection
    orig_return = pool.return_connection

    def tracked_get_connection():
        counts["get"] += 1
        return orig_get()

    def tracked_return_connection(conn):
        counts["return"] += 1
        return orig_return(conn)

    try:
        # Ensure any init-time connection is cleared before tracking.
        try:
            db.close_connection()
        except Exception:
            _ = None

        pool.get_connection = tracked_get_connection  # type: ignore[assignment]
        pool.return_connection = tracked_return_connection  # type: ignore[assignment]

        for _ in range(5):
            db.get_connection()
            db.close_connection()

        assert counts["get"] == 5
        assert counts["return"] == 5
    finally:
        try:
            pool.get_connection = orig_get  # type: ignore[assignment]
            pool.return_connection = orig_return  # type: ignore[assignment]
        except Exception:
            _ = None
        try:
            db.close()
        except Exception:
            _ = None
        try:
            backend.get_pool().close_all()
        except Exception:
            _ = None


def test_prompt_studio_pool_return_rolls_back_implicit_read_transaction(
    pg_database_config: DatabaseConfig,
    tmp_path,
):
    """Returning a psycopg connection must release read locks first."""

    pg_database_config.pool_size = 1
    pg_database_config.max_overflow = 0
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = PromptStudioDatabase(
        db_path=str(tmp_path / "prompt_studio_pool_read.sqlite"),
        client_id="pool-ps-read",
        backend=backend,
    )

    try:
        db.close_connection()
        conn = db.get_connection()
        raw_connection = conn.raw_connection
        assert conn.execute("SELECT 1 AS value").fetchone()["value"] == 1
        assert raw_connection.info.transaction_status.name == "INTRANS"

        db.close_connection()

        assert raw_connection.info.transaction_status.name == "IDLE"
    finally:
        db.close_connection()
        backend.get_pool().close_all()


def test_prompt_studio_pool_return_survives_driver_rollback_failure() -> None:
    """A broken rollback must discard, not recycle, the poisoned checkout."""

    raw_connection = object()
    returned: list[object] = []
    discarded: list[object] = []

    class _Pool:
        def return_connection(self, connection: object) -> None:
            returned.append(connection)

        def discard_connection(self, connection: object) -> None:
            discarded.append(connection)

    class _Backend:
        def get_pool(self) -> _Pool:
            return _Pool()

    class _Wrapper:
        def __init__(self) -> None:
            self.raw_connection = raw_connection

        def rollback(self) -> None:
            raise Exception("driver rollback failed")

    db = object.__new__(BackendPromptStudioDatabaseBase)
    db.backend = _Backend()  # type: ignore[assignment]
    db._local = threading.local()
    db._local.conn = _Wrapper()

    db.close_connection()

    assert returned == []
    assert discarded == [raw_connection]
    assert getattr(db._local, "conn", None) is None


def test_prompt_studio_pool_return_reuses_connection_after_successful_rollback() -> None:
    """A successfully reset checkout must return once and never be discarded."""

    raw_connection = object()
    returned: list[object] = []
    discarded: list[object] = []

    class _Pool:
        def return_connection(self, connection: object) -> None:
            returned.append(connection)

        def discard_connection(self, connection: object) -> None:
            discarded.append(connection)

    class _Backend:
        def get_pool(self) -> _Pool:
            return _Pool()

    class _Wrapper:
        def __init__(self) -> None:
            self.raw_connection = raw_connection

        def rollback(self) -> None:
            return None

    db = object.__new__(BackendPromptStudioDatabaseBase)
    db.backend = _Backend()  # type: ignore[assignment]
    db._local = threading.local()
    db._local.conn = _Wrapper()

    db.close_connection()

    assert returned == [raw_connection]
    assert discarded == []
    assert getattr(db._local, "conn", None) is None


def test_worker_scope_releases_postgres_reads_between_tenants(
    pg_database_config: DatabaseConfig,
    tmp_path,
) -> None:
    """Sequential tenant scopes must not consume the shared pool permanently."""

    pg_database_config.pool_size = 1
    pg_database_config.max_overflow = 0
    pg_database_config.pool_timeout = 0.2
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    databases = [
        PromptStudioDatabase(
            db_path=str(tmp_path / f"prompt-studio-scope-{owner}.sqlite"),
            client_id=f"pool-scope-{owner}",
            tenant_user_id=owner,
            backend=backend,
        )
        for owner in ("7", "8")
    ]
    for db in databases:
        db.close_connection()

    with jobs_worker._CACHE_LOCK:
        saved_db_cache = jobs_worker._DB_CACHE.copy()
        saved_processor_cache = jobs_worker._PROCESSOR_CACHE.copy()
        jobs_worker._DB_CACHE.clear()
        jobs_worker._PROCESSOR_CACHE.clear()
        jobs_worker._DB_CACHE.update(zip(("7", "8"), databases, strict=True))

    try:
        for owner, db in zip(("7", "8"), databases, strict=True):
            with jobs_worker._active_user_cache_scope(owner):
                connection = db.get_connection()
                assert connection.execute("SELECT 1 AS value").fetchone()["value"] == 1
                raw_connection = connection.raw_connection
                assert raw_connection.info.transaction_status.name == "INTRANS"

            assert raw_connection.info.transaction_status.name == "IDLE"
            assert jobs_worker._DB_CACHE[owner] is db

        borrowed = backend.get_pool().get_connection()
        backend.get_pool().return_connection(borrowed)
    finally:
        with jobs_worker._CACHE_LOCK:
            jobs_worker._DB_CACHE.clear()
            jobs_worker._DB_CACHE.update(saved_db_cache)
            jobs_worker._PROCESSOR_CACHE.clear()
            jobs_worker._PROCESSOR_CACHE.update(saved_processor_cache)
        for db in databases:
            db.close_connection()
        backend.get_pool().close_all()
