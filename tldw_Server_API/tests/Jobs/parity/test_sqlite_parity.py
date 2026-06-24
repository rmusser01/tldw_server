from __future__ import annotations

import pytest

from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.migrations import ensure_jobs_tables
from tldw_Server_API.tests.Jobs.parity.scenarios import (
    run_acquire_complete_lifecycle_scenario,
    run_cancel_terminal_noop_scenario,
    run_events_outbox_create_complete_scenario,
    run_idempotent_create_preserves_original_request_ids_scenario,
    run_idempotent_create_scope_scenario,
    run_renew_stale_lease_noop_scenario,
)


@pytest.fixture()
def sqlite_manager_factory(tmp_path, monkeypatch):
    db_path = tmp_path / "jobs.db"
    ensure_jobs_tables(db_path)
    monkeypatch.setenv("JOBS_DB_PATH", str(db_path))
    monkeypatch.setenv("JOBS_DISABLE_LEASE_ENFORCEMENT", "true")
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "true")
    return lambda: JobManager(db_path)


def test_sqlite_idempotent_create_scope(sqlite_manager_factory):
    run_idempotent_create_scope_scenario(sqlite_manager_factory)


def test_sqlite_idempotent_create_preserves_request_ids(sqlite_manager_factory):
    run_idempotent_create_preserves_original_request_ids_scenario(sqlite_manager_factory)


def test_sqlite_acquire_complete_lifecycle(sqlite_manager_factory):
    run_acquire_complete_lifecycle_scenario(sqlite_manager_factory)


def test_sqlite_renew_stale_lease_noop(sqlite_manager_factory):
    run_renew_stale_lease_noop_scenario(sqlite_manager_factory)


def test_sqlite_cancel_terminal_noop(sqlite_manager_factory):
    run_cancel_terminal_noop_scenario(sqlite_manager_factory)


def test_sqlite_events_outbox_create_complete(sqlite_manager_factory):
    run_events_outbox_create_complete_scenario(sqlite_manager_factory)
