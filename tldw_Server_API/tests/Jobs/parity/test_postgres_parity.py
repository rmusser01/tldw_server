"""Postgres wrappers for shared Jobs backend parity scenarios."""

from __future__ import annotations

from collections.abc import Callable

import pytest

psycopg = pytest.importorskip("psycopg")
pytestmark = pytest.mark.pg_jobs

from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.tests.Jobs.parity.scenarios import (
    run_acquire_complete_lifecycle_scenario,
    run_cancel_terminal_noop_scenario,
    run_events_outbox_create_complete_scenario,
    run_idempotent_create_replay_event_uses_current_request_ids_scenario,
    run_idempotent_create_preserves_original_request_ids_scenario,
    run_idempotent_create_scope_scenario,
    run_renew_stale_lease_noop_scenario,
)


@pytest.fixture()
def postgres_manager_factory(jobs_pg_dsn: str, monkeypatch: pytest.MonkeyPatch) -> Callable[[], JobManager]:
    """Create a Postgres-backed JobManager factory from the test DSN."""

    monkeypatch.setenv("JOBS_DB_URL", jobs_pg_dsn)
    monkeypatch.setenv("JOBS_DISABLE_LEASE_ENFORCEMENT", "true")
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "true")
    return lambda: JobManager(None, backend="postgres", db_url=jobs_pg_dsn)


def test_postgres_idempotent_create_scope(postgres_manager_factory: Callable[[], JobManager]) -> None:
    """Run the idempotent-create scope scenario against Postgres."""

    run_idempotent_create_scope_scenario(postgres_manager_factory)


def test_postgres_idempotent_create_preserves_request_ids(postgres_manager_factory: Callable[[], JobManager]) -> None:
    """Run the request-id preservation scenario against Postgres."""

    run_idempotent_create_preserves_original_request_ids_scenario(postgres_manager_factory)


def test_postgres_idempotent_create_replay_event_uses_current_request_ids(
    postgres_manager_factory: Callable[[], JobManager],
) -> None:
    """Run the idempotent replay event context scenario against Postgres."""

    run_idempotent_create_replay_event_uses_current_request_ids_scenario(postgres_manager_factory)


def test_postgres_acquire_complete_lifecycle(postgres_manager_factory: Callable[[], JobManager]) -> None:
    """Run the acquire-complete lifecycle scenario against Postgres."""

    run_acquire_complete_lifecycle_scenario(postgres_manager_factory)


def test_postgres_renew_stale_lease_noop(postgres_manager_factory: Callable[[], JobManager]) -> None:
    """Run the stale lease renewal scenario against Postgres."""

    run_renew_stale_lease_noop_scenario(postgres_manager_factory)


def test_postgres_cancel_terminal_noop(postgres_manager_factory: Callable[[], JobManager]) -> None:
    """Run the terminal cancellation scenario against Postgres."""

    run_cancel_terminal_noop_scenario(postgres_manager_factory)


def test_postgres_events_outbox_create_complete(postgres_manager_factory: Callable[[], JobManager]) -> None:
    """Run the create-complete outbox scenario against Postgres."""

    run_events_outbox_create_complete_scenario(postgres_manager_factory)
