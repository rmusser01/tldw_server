"""Real PostgreSQL lock lifetime across resumable schema transactions."""


import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseConfig, DatabaseError
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.chacha.schema_bootstrap import postgres_schema_migration

pytestmark = pytest.mark.integration


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
