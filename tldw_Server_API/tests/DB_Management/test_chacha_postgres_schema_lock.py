"""Real PostgreSQL lock lifetime across resumable schema transactions."""


from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
import time
from typing import Any

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import (
    DatabaseConfig,
    DatabaseError,
    TransientContentionError,
)
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.chacha.schema_bootstrap import postgres_schema_migration
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("deadline", ["bootstrap", "operator-statement"])
def test_initializer_acquisition_deadline_discards_checkout_and_allows_later_retry(
    pg_database_config: DatabaseConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    deadline: str,
) -> None:
    """Both native acquisition deadlines discard failed initializer sessions."""
    from psycopg.conninfo import make_conninfo

    owner_backend = DatabaseBackendFactory.create_backend(pg_database_config)
    waiter_config = pg_database_config
    if deadline == "operator-statement":
        waiter_config = replace(pg_database_config, connection_string=make_conninfo(
            host=pg_database_config.pg_host, port=str(pg_database_config.pg_port),
            dbname=pg_database_config.pg_database, user=pg_database_config.pg_user,
            password=pg_database_config.pg_password, options="-c statement_timeout=100ms",
        ))
    waiter_backend = DatabaseBackendFactory.create_backend(waiter_config)
    pool = waiter_backend.get_pool()
    original_get_connection = pool.get_connection
    checkouts = []

    def observe_checkout(*args: object, **kwargs: object):
        """Observe the real connection without replacing PostgreSQL behavior."""
        connection = original_get_connection(*args, **kwargs)
        checkouts.append((connection, connection.info.backend_pid))
        return connection

    try:
        with ThreadPoolExecutor(max_workers=1) as workers:
            with postgres_schema_migration(owner_backend, "100ms"):
                with monkeypatch.context() as patch:
                    patch.setattr(pool, "get_connection", observe_checkout)
                    waiter = workers.submit(
                        CharactersRAGDB, tmp_path / "blocked.db", client_id="2", backend=waiter_backend,
                    )
                    with pytest.raises(CharactersRAGDBError) as failure:
                        waiter.result(timeout=35 if deadline == "bootstrap" else 5)
                cause = failure.value.__cause__
                if deadline == "bootstrap":
                    assert isinstance(cause, TransientContentionError)
                else:
                    assert type(cause) is DatabaseError
                connection, process_id = checkouts[-1]
                assert connection.closed
                assert owner_backend.execute(
                    "SELECT count(*) FROM pg_locks WHERE pid=%s AND locktype='advisory'",
                    (process_id,),
                ).scalar == 0
        replacement = CharactersRAGDB(tmp_path / "retry.db", client_id="2", backend=waiter_backend)
        try:
            note = replacement.add_note(title="Recovered", content="Acquisition can retry")
            assert replacement.get_note_by_id(note)["content"] == "Acquisition can retry"
        finally:
            replacement.close_connection()
    finally:
        for backend in (owner_backend, waiter_backend):
            backend.get_pool().close_all()


@pytest.mark.parametrize("failure", [None, "body", "unlock"])
def test_schema_lock_survives_commit_and_is_released_before_checkout_reuse(
    pg_database_config: DatabaseConfig, monkeypatch: pytest.MonkeyPatch, failure: str | None,
) -> None:
    """Success and failed bodies/cleanup leave no session lock in the pool."""
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    connection = None
    process_id = None
    original_execute = backend.execute

    def fail_unlock(query: str, *args: object, **kwargs: object):
        """Fail the release query at the public SQL boundary."""
        if "pg_advisory_unlock(" in query:
            raise DatabaseError("planned unlock failure")
        return original_execute(query, *args, **kwargs)

    def migrate() -> None:
        """Retain the migration lock across a deliberate durable checkpoint."""
        nonlocal connection, process_id
        with postgres_schema_migration(backend, "1s") as connection:
            process_id = connection.info.backend_pid
            connection.commit()
            assert original_execute(
                "SELECT count(*) FROM pg_locks WHERE pid=%s AND locktype='advisory' AND granted",
                (process_id,),
            ).scalar == 1
            if failure == "body":
                raise RuntimeError("planned body failure")

    try:
        with monkeypatch.context() as patch:
            if failure == "unlock":
                patch.setattr(backend, "execute", fail_unlock)
            if failure:
                with pytest.raises((RuntimeError, DatabaseError), match=f"planned {failure} failure"):
                    migrate()
            else:
                migrate()
        assert original_execute(
            "SELECT count(*) FROM pg_locks WHERE pid=%s AND locktype='advisory' AND granted",
            (process_id,),
        ).scalar == 0
        if failure:
            assert connection.closed
        # Another migration can acquire the same key after success or failure.
        with postgres_schema_migration(backend, "100ms"):
            pass
    finally:
        backend.get_pool().close_all()


def test_schema_lock_acquisition_deadline_invalidates_only_waiter_then_allows_reuse(
    pg_database_config: DatabaseConfig, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A short acquisition deadline closes the waiter while preserving the lock owner."""
    blocker_backend = DatabaseBackendFactory.create_backend(pg_database_config)
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    pool = backend.get_pool()
    invalidated: list[tuple[Any, int]] = []
    yielded: list[Any] = []
    original_invalidate = pool.invalidate_connection

    def record_invalidation(connection: Any) -> None:
        invalidated.append((connection, connection.info.backend_pid))
        original_invalidate(connection)

    monkeypatch.setattr(pool, "invalidate_connection", record_invalidation)
    try:
        with blocker_backend.transaction() as blocker:
            blocker_backend.execute(
                "SELECT pg_advisory_lock(hashtext(%s), hashtext(current_schema()))",
                ("chacha_schema_bootstrap",), connection=blocker,
            )
            try:
                started = time.monotonic()
                with pytest.raises(TransientContentionError):
                    with postgres_schema_migration(backend, "100ms") as connection:
                        yielded.append(connection)
                assert time.monotonic() - started < 2
                assert yielded == []
                assert len(invalidated) == 1
                failed, failed_pid = invalidated[0]
                assert failed.closed
                assert blocker_backend.execute(
                    "SELECT count(*) FROM pg_locks WHERE pid=%s AND locktype='advisory' AND granted",
                    (blocker.info.backend_pid,), connection=blocker,
                ).scalar == 1
                assert blocker_backend.execute("SELECT 1 AS value", connection=blocker).rows == [{"value": 1}]
            finally:
                blocker_backend.execute(
                    "SELECT pg_advisory_unlock(hashtext(%s), hashtext(current_schema()))",
                    ("chacha_schema_bootstrap",), connection=blocker,
                )
        with postgres_schema_migration(backend, "100ms") as fresh:
            assert fresh is not failed
            assert backend.execute("SELECT 1 AS value", connection=fresh).rows == [{"value": 1}]
            assert backend.execute(
                "SELECT count(*) FROM pg_locks WHERE pid=%s AND locktype='advisory' AND granted",
                (failed_pid,), connection=fresh,
            ).scalar == 0
    finally:
        backend.get_pool().close_all()
        blocker_backend.get_pool().close_all()
