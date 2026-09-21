"""PostgreSQL bootstrap coordination across resumable migration commits."""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseBackend


@contextmanager
def postgres_schema_migration(backend: DatabaseBackend, lock_timeout: str) -> Iterator[Any]:
    """Own a checkout and schema lock until all migration transactions finish.

    The legacy v63 migration deliberately commits durable pages. A transaction
    advisory lock would be released mid-bootstrap, so use a session lock here.
    Release it only after commit/rollback; discard the checkout if cleanup fails.
    """
    pool = backend.get_pool()
    connection = pool.get_connection()
    try:
        with backend.transaction(connection=connection):
            backend.execute("SELECT set_config('lock_timeout',%s,true)", (lock_timeout,), connection=connection)
            backend.execute(
                "SELECT pg_advisory_lock(hashtext(%s), hashtext(current_schema()))",
                ("chacha_schema_bootstrap",), connection=connection,
            )
        try:
            with backend.transaction(connection=connection):
                yield connection
        finally:
            with backend.transaction(connection=connection):
                backend.execute(
                    "SELECT pg_advisory_unlock(hashtext(%s), hashtext(current_schema()))",
                    ("chacha_schema_bootstrap",), connection=connection,
                )
    except BaseException:
        # An uncertain acquisition/release must never return a session lock to
        # the pool. Closing this independently owned session releases its locks.
        pool.invalidate_connection(connection)
        raise
    else:
        pool.return_connection(connection)
