"""SQLite wrappers for shared Jobs backend parity scenarios."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.migrations import ensure_jobs_tables
from tldw_Server_API.tests.Jobs.parity.scenarios import (
    run_acquire_complete_lifecycle_scenario,
    run_cancel_terminal_noop_scenario,
    run_conditional_cancel_binding_scenario,
    run_events_outbox_create_complete_scenario,
    run_idempotent_create_preserves_original_request_ids_scenario,
    run_idempotent_create_replay_event_uses_current_request_ids_scenario,
    run_idempotent_create_scope_scenario,
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


def test_sqlite_renew_stale_lease_noop(sqlite_manager_factory: Callable[[], JobManager]) -> None:
    """Run the stale lease renewal scenario against SQLite."""

    run_renew_stale_lease_noop_scenario(sqlite_manager_factory)


def test_sqlite_cancel_terminal_noop(sqlite_manager_factory: Callable[[], JobManager]) -> None:
    """Run the terminal cancellation scenario against SQLite."""

    run_cancel_terminal_noop_scenario(sqlite_manager_factory)


def test_sqlite_conditional_cancel_binding(sqlite_manager_factory: Callable[[], JobManager]) -> None:
    """Run exact-binding cancellation against SQLite."""

    run_conditional_cancel_binding_scenario(sqlite_manager_factory)


def test_sqlite_events_outbox_create_complete(sqlite_manager_factory: Callable[[], JobManager]) -> None:
    """Run the create-complete outbox scenario against SQLite."""

    run_events_outbox_create_complete_scenario(sqlite_manager_factory)
