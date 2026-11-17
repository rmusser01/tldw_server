"""Live scope settings must replace prior authorization on pooled reuse."""

from contextlib import AbstractContextManager, nullcontext
from dataclasses import replace
from threading import Event
from typing import cast

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseConfig
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.backends.postgresql_backend import PostgreSQLConnectionPool
from tldw_Server_API.app.core.DB_Management.scope_context import get_scope, scoped_context


@pytest.mark.integration
@pytest.mark.postgres
def test_pool_shutdown_closes_returned_connection_before_worker_reset(
    pg_database_config: DatabaseConfig, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A returned socket must close even when shutdown beats its reset worker."""
    from psycopg_pool.pool import ReturnConnection

    backend = DatabaseBackendFactory.create_backend(replace(pg_database_config, pool_size=1))
    pool = cast(PostgreSQLConnectionPool, backend.get_pool())
    delegate = pool._pool
    entered = Event()
    resume = Event()
    run = ReturnConnection.run
    run_task = delegate.run_task
    signal_stop = delegate._signal_stop_worker

    def delayed_return(task):
        entered.set()
        assert resume.wait(5)
        run(task)

    def schedule(task):
        run_task(task)
        if isinstance(task, ReturnConnection):
            assert entered.wait(5)

    def stop_workers():
        workers = signal_stop()
        resume.set()
        return workers

    monkeypatch.setattr(ReturnConnection, "run", delayed_return)
    monkeypatch.setattr(delegate, "run_task", schedule)
    monkeypatch.setattr(delegate, "_signal_stop_worker", stop_workers)
    connection = pool.get_connection()
    try:
        pool.return_connection(connection)
        pool.close_all()
        assert connection.closed
    finally:
        resume.set()
        connection.close()
        pool.close_all()


@pytest.mark.integration
@pytest.mark.postgres
def test_pool_return_rolls_back_uncommitted_writes_before_reset(pg_database_config: DatabaseConfig) -> None:
    """Reset commits its session changes, never a borrower's unfinished work."""
    backend = DatabaseBackendFactory.create_backend(replace(pg_database_config, pool_size=1))
    pool = backend.get_pool()
    try:
        with backend.transaction() as connection:
            connection.execute("CREATE TABLE pool_return_probe (id integer)")
        with pool.connection() as connection:
            original_pid = connection.info.backend_pid
            connection.execute("INSERT INTO pool_return_probe VALUES (1)")
        with pool.connection() as reused:
            assert reused.info.backend_pid == original_pid
            assert reused.execute("SELECT count(*) AS count FROM pool_return_probe").fetchone()["count"] == 0
    finally:
        pool.close_all()


@pytest.mark.integration
@pytest.mark.postgres
@pytest.mark.timeout(120)
def test_pooled_scope_installation_replaces_memberships_admin_and_clears_owner(
    pg_database_config: DatabaseConfig,
) -> None:
    """Catch swapped GUC parameters and stale authorization across requests.

    Commit every installation before returning the checkout: a pool rollback
    must not erase the old values and conceal an incomplete replacement. The
    backend PID proves that all transitions reuse one PostgreSQL session. Scope
    installation runs through the real pool checkout, with no scope mocks.
    """
    assert get_scope() is None
    backend = DatabaseBackendFactory.create_backend(
        replace(pg_database_config, pool_size=1)
    )
    pool = backend.get_pool()
    contexts = (
        {
            "user_id": 42,
            "org_ids": [7, 8],
            "team_ids": [9],
            "is_admin": True,
        },
        {"user_id": 43, "is_admin": False},
        None,
    )
    observed_scopes = []
    backend_pids = set()
    try:
        for scope in contexts:
            with cast(AbstractContextManager[object], scoped_context(**scope) if scope is not None else nullcontext()):
                with pool.connection() as connection:
                    result = backend.execute(
                        "SELECT current_setting('app.current_user_id') AS current_user_id, "
                        "current_setting('app.user_id') AS user_id, "
                        "current_setting('app.org_ids') AS org_ids, "
                        "current_setting('app.team_ids') AS team_ids, "
                        "current_setting('app.is_admin') AS is_admin, "
                        "current_setting('row_security') AS row_security, "
                        "pg_backend_pid() AS backend_pid",
                        connection=connection,
                    )
                    observed = dict(result.rows[0])
                    backend_pids.add(observed.pop("backend_pid"))
                    observed_scopes.append(observed)
                    connection.commit()
    finally:
        pool.close_all()

    assert len(backend_pids) == 1
    assert observed_scopes == [
        {
            "current_user_id": "42",
            "user_id": "42",
            "org_ids": "7,8",
            "team_ids": "9",
            "is_admin": "1",
            "row_security": "on",
        },
        {
            "current_user_id": "43",
            "user_id": "43",
            "org_ids": "",
            "team_ids": "",
            "is_admin": "0",
            "row_security": "on",
        },
        {
            "current_user_id": "",
            "user_id": "",
            "org_ids": "",
            "team_ids": "",
            "is_admin": "0",
            "row_security": "on",
        },
    ]
    assert get_scope() is None
