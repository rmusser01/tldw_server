"""SQLite wrappers for shared Jobs backend parity scenarios."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.migrations import ensure_jobs_tables
from tldw_Server_API.tests.Jobs.parity.scenarios import (
    FUTURE_NOW_EPOCH,
    run_acquire_complete_lifecycle_scenario,
    run_acquire_contention_scenario,
    run_cancel_terminal_noop_scenario,
    run_events_outbox_create_complete_scenario,
    run_expired_lease_reclaim_scenario,
    run_idempotent_create_preserves_original_request_ids_scenario,
    run_idempotent_create_replay_event_uses_current_request_ids_scenario,
    run_idempotent_create_scope_scenario,
    run_release_lease_ownership_scenario,
    run_renew_lease_characterization_scenario,
    run_renew_stale_lease_noop_scenario,
)

pytestmark = pytest.mark.jobs


@pytest.fixture()
def sqlite_manager_factory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[[], JobManager]:
    """Create an isolated SQLite-backed JobManager factory."""

    db_path = tmp_path / "jobs.db"
    ensure_jobs_tables(db_path)
    monkeypatch.setenv("JOBS_DB_PATH", str(db_path))
    monkeypatch.setenv("JOBS_DISABLE_LEASE_ENFORCEMENT", "true")
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "true")
    return lambda: JobManager(db_path)


def _expire_sqlite_lease(manager: JobManager, job_id: int) -> None:
    conn = manager._connect()
    try:
        with conn:
            conn.execute(
                "UPDATE jobs SET leased_until = DATETIME('now', '-10 seconds') WHERE id = ?",
                (job_id,),
            )
    finally:
        conn.close()


def test_sqlite_idempotent_create_scope(sqlite_manager_factory: Callable[[], JobManager]) -> None:
    """Run the idempotent-create scope scenario against SQLite."""

    run_idempotent_create_scope_scenario(sqlite_manager_factory)


def test_sqlite_idempotent_create_preserves_request_ids(sqlite_manager_factory: Callable[[], JobManager]) -> None:
    """Run the request-id preservation scenario against SQLite."""

    run_idempotent_create_preserves_original_request_ids_scenario(sqlite_manager_factory)


def test_sqlite_idempotent_create_replay_event_uses_current_request_ids(
    sqlite_manager_factory: Callable[[], JobManager],
) -> None:
    """Run the idempotent replay event context scenario against SQLite."""

    run_idempotent_create_replay_event_uses_current_request_ids_scenario(sqlite_manager_factory)


def test_sqlite_acquire_complete_lifecycle(sqlite_manager_factory: Callable[[], JobManager]) -> None:
    """Run the acquire-complete lifecycle scenario against SQLite."""

    run_acquire_complete_lifecycle_scenario(sqlite_manager_factory)


def test_sqlite_acquire_contention(sqlite_manager_factory: Callable[[], JobManager]) -> None:
    """Run the concurrent acquisition scenario against SQLite."""

    run_acquire_contention_scenario(sqlite_manager_factory)


def test_sqlite_expired_lease_reclaim(sqlite_manager_factory: Callable[[], JobManager]) -> None:
    """Run the expired lease reclaim scenario against SQLite."""

    run_expired_lease_reclaim_scenario(sqlite_manager_factory, _expire_sqlite_lease)


def test_sqlite_renew_stale_lease_noop(sqlite_manager_factory: Callable[[], JobManager]) -> None:
    """Run the stale lease renewal scenario against SQLite."""

    run_renew_stale_lease_noop_scenario(sqlite_manager_factory)


def test_sqlite_renew_lease_characterization(
    sqlite_manager_factory: Callable[[], JobManager],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run future-clock renewal characterization against SQLite."""

    monkeypatch.setenv("JOBS_TEST_NOW_EPOCH", FUTURE_NOW_EPOCH)
    run_renew_lease_characterization_scenario(sqlite_manager_factory)


def test_sqlite_release_lease_ownership(
    sqlite_manager_factory: Callable[[], JobManager],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run release ownership and compatibility characterization against SQLite."""

    monkeypatch.setenv("JOBS_TEST_NOW_EPOCH", FUTURE_NOW_EPOCH)
    run_release_lease_ownership_scenario(sqlite_manager_factory)


def test_sqlite_cancel_terminal_noop(sqlite_manager_factory: Callable[[], JobManager]) -> None:
    """Run the terminal cancellation scenario against SQLite."""

    run_cancel_terminal_noop_scenario(sqlite_manager_factory)


def test_sqlite_events_outbox_create_complete(sqlite_manager_factory: Callable[[], JobManager]) -> None:
    """Run the create-complete outbox scenario against SQLite."""

    run_events_outbox_create_complete_scenario(sqlite_manager_factory)
