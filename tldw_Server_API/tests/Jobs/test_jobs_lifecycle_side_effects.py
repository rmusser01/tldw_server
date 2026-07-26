"""Facade side-effect ordering tests for extracted Jobs lifecycle operations."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

import tldw_Server_API.app.core.Jobs.manager as manager_module
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.operations.contracts import (
    LifecycleResult,
    NoTransitionReason,
)


def _manager_without_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    observed: list[str],
) -> JobManager:
    manager = JobManager(tmp_path / "jobs.db")
    monkeypatch.setattr(manager, "_get_queue_flags", lambda *_args: {"paused": False, "drain": False})
    monkeypatch.setattr(manager, "_reconcile_terminal_dependents", lambda **_kwargs: 0)
    monkeypatch.setattr(manager, "_recover_expired_processing_jobs", lambda **_kwargs: 0)
    monkeypatch.setattr(manager, "_quota_get", lambda *_args: 0)
    monkeypatch.setattr(
        manager_module,
        "observe_queue_latency",
        lambda *_args, **_kwargs: observed.append("latency"),
    )
    monkeypatch.setattr(
        manager_module,
        "emit_job_event",
        lambda *_args, **_kwargs: observed.append("event"),
    )
    monkeypatch.setattr(manager, "_update_gauges", lambda **_kwargs: observed.append("gauge"))
    return manager


def test_sqlite_acquire_runs_success_observers_after_operation_returns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[str] = []
    captured: dict[str, Any] = {}
    manager = _manager_without_preflight(tmp_path, monkeypatch, observed)

    def acquire_stub(
        _conn: sqlite3.Connection,
        *,
        command: Any,
        counters_enabled: bool,
        now: Any,
    ) -> LifecycleResult:
        captured.update(command=command, counters_enabled=counters_enabled, now=now)
        observed.append("operation-returned")
        return LifecycleResult.applied(
            row={
                "id": 1,
                "uuid": "job-1",
                "domain": command.domain,
                "queue": command.queue,
                "job_type": "work",
                "owner_user_id": "owner",
                "status": "processing",
                "worker_id": command.worker_id,
                "lease_id": command.lease_id,
                "leased_until": "2026-01-01 00:00:30",
                "created_at": "2026-01-01 00:00:00",
                "acquired_at": "2026-01-01 00:00:00",
                "retry_count": 0,
                "payload": '{"value": 1}',
            }
        )

    monkeypatch.setattr(manager_module, "_sqlite_acquire_job", acquire_stub, raising=False)

    acquired = manager.acquire_next_job(
        domain="facade",
        queue="default",
        job_type="work",
        lease_seconds=30,
        worker_id="worker-1",
        owner_user_id="owner",
    )

    assert acquired is not None
    assert acquired["payload"] == {"value": 1}
    assert acquired["lease_id"] == captured["command"].lease_id
    assert observed == ["operation-returned", "latency", "gauge", "event"]


def test_sqlite_acquire_no_transition_runs_no_success_observers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[str] = []
    manager = _manager_without_preflight(tmp_path, monkeypatch, observed)

    def acquire_stub(*_args: Any, **_kwargs: Any) -> LifecycleResult:
        observed.append("operation-returned")
        return LifecycleResult.no_transition(NoTransitionReason.NO_ELIGIBLE_JOB)

    monkeypatch.setattr(manager_module, "_sqlite_acquire_job", acquire_stub, raising=False)

    acquired = manager.acquire_next_job(
        domain="facade",
        queue="default",
        lease_seconds=30,
        worker_id="worker-1",
    )

    assert acquired is None
    assert observed == ["operation-returned"]


def test_sqlite_acquire_backend_error_runs_no_success_observers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[str] = []
    manager = _manager_without_preflight(tmp_path, monkeypatch, observed)

    def acquire_stub(*_args: Any, **_kwargs: Any) -> LifecycleResult:
        observed.append("operation-returned")
        raise sqlite3.OperationalError("forced acquisition failure")

    monkeypatch.setattr(manager_module, "_sqlite_acquire_job", acquire_stub, raising=False)

    with pytest.raises(sqlite3.OperationalError, match="forced acquisition failure"):
        manager.acquire_next_job(
            domain="facade",
            queue="default",
            lease_seconds=30,
            worker_id="worker-1",
        )

    assert observed == ["operation-returned"]


class _FakePostgresConnection:
    def close(self) -> None:
        return None


def _postgres_manager_without_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    observed: list[str],
) -> JobManager:
    manager = _manager_without_preflight(tmp_path, monkeypatch, observed)
    manager.backend = "postgres"
    monkeypatch.setattr(manager, "_connect", lambda: _FakePostgresConnection())
    return manager


@pytest.mark.parametrize("single_update", [False, True])
def test_postgres_acquire_runs_success_observers_after_operation_returns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    single_update: bool,
) -> None:
    observed: list[str] = []
    captured: dict[str, Any] = {}
    manager = _postgres_manager_without_preflight(tmp_path, monkeypatch, observed)
    monkeypatch.setenv("JOBS_PG_ACQUIRE_PRIORITY_DESC_DOMAINS", "facade")
    monkeypatch.setenv("JOBS_PG_ACQUIRE_TIE_BREAK_FACADE", "lifo")
    monkeypatch.setenv("JOBS_PG_SINGLE_UPDATE_ACQUIRE", str(single_update))

    def acquire_stub(
        _conn: Any,
        cursor_factory: Any,
        *,
        command: Any,
        counters_enabled: bool,
        now: Any,
    ) -> LifecycleResult:
        captured.update(
            command=command,
            counters_enabled=counters_enabled,
            cursor_factory=cursor_factory,
            now=now,
        )
        observed.append("operation-returned")
        return LifecycleResult.applied(
            row={
                "id": 1,
                "uuid": "job-1",
                "domain": command.domain,
                "queue": command.queue,
                "job_type": "work",
                "owner_user_id": "owner",
                "status": "processing",
                "worker_id": command.worker_id,
                "lease_id": command.lease_id,
                "leased_until": "2026-01-01 00:00:30",
                "created_at": "2026-01-01 00:00:00",
                "acquired_at": "2026-01-01 00:00:00",
                "retry_count": 0,
                "payload": '{"value": 1}',
            }
        )

    monkeypatch.setattr(manager_module, "_postgres_acquire_job", acquire_stub, raising=False)

    acquired = manager.acquire_next_job(
        domain="facade",
        queue="default",
        job_type="work",
        lease_seconds=30,
        worker_id="worker-1",
        owner_user_id="owner",
    )

    assert acquired is not None
    assert acquired["payload"] == {"value": 1}
    assert acquired["lease_id"] == captured["command"].lease_id
    assert captured["cursor_factory"] == manager._pg_cursor
    assert captured["command"].priority_direction == "DESC"
    assert captured["command"].tie_break == "lifo"
    assert captured["command"].single_update is single_update
    assert observed == ["operation-returned", "latency", "gauge", "event"]


def test_postgres_acquire_no_transition_runs_no_success_observers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[str] = []
    manager = _postgres_manager_without_preflight(tmp_path, monkeypatch, observed)

    def acquire_stub(*_args: Any, **_kwargs: Any) -> LifecycleResult:
        observed.append("operation-returned")
        return LifecycleResult.no_transition(NoTransitionReason.NO_ELIGIBLE_JOB)

    monkeypatch.setattr(manager_module, "_postgres_acquire_job", acquire_stub, raising=False)

    acquired = manager.acquire_next_job(
        domain="facade",
        queue="default",
        lease_seconds=30,
        worker_id="worker-1",
    )

    assert acquired is None
    assert observed == ["operation-returned"]


def test_postgres_acquire_backend_error_runs_no_success_observers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[str] = []
    manager = _postgres_manager_without_preflight(tmp_path, monkeypatch, observed)

    def acquire_stub(*_args: Any, **_kwargs: Any) -> LifecycleResult:
        observed.append("operation-returned")
        raise RuntimeError("forced acquisition failure")

    monkeypatch.setattr(manager_module, "_postgres_acquire_job", acquire_stub, raising=False)

    with pytest.raises(RuntimeError, match="forced acquisition failure"):
        manager.acquire_next_job(
            domain="facade",
            queue="default",
            lease_seconds=30,
            worker_id="worker-1",
        )

    assert observed == ["operation-returned"]
