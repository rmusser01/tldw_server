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

pytestmark = pytest.mark.integration


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


def test_sqlite_renew_runs_exact_event_after_operation_returns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[str] = []
    captured: dict[str, Any] = {}
    events: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    manager = _manager_without_preflight(tmp_path, monkeypatch, observed)
    monkeypatch.setenv("JOBS_LEASE_MAX_SECONDS", "45")
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "true")
    monkeypatch.setattr(
        manager_module,
        "submit_job_audit_event",
        lambda *_args, **_kwargs: pytest.fail("renewal must not use the durable outbox"),
    )

    def renew_stub(
        _conn: sqlite3.Connection,
        *,
        command: Any,
        now: Any,
    ) -> LifecycleResult:
        captured.update(command=command, now=now)
        observed.append("operation-returned")
        return LifecycleResult.applied(row={"id": command.job_id, "status": "processing"})

    def capture_event(
        event_type: str,
        *,
        job: dict[str, Any],
        attrs: dict[str, Any],
    ) -> None:
        observed.append("event")
        events.append((event_type, job, attrs))

    monkeypatch.setattr(manager_module, "_sqlite_renew_lease", renew_stub, raising=False)
    monkeypatch.setattr(manager_module, "emit_job_event", capture_event)

    renewed = manager.renew_job_lease(
        42,
        seconds=999,
        worker_id="worker-1",
        lease_id="lease-1",
        progress_percent=62.5,
        progress_message="indexing",
        enforce=True,
    )

    assert renewed is True
    assert captured["command"].job_id == 42
    assert captured["command"].seconds == 45
    assert captured["command"].enforce is True
    assert captured["command"].worker_id == "worker-1"
    assert captured["command"].lease_id == "lease-1"
    assert captured["command"].progress_percent == 62.5
    assert captured["command"].progress_message == "indexing"
    assert captured["now"] is not None
    assert events == [("job.lease_renewed", {"id": 42}, {"seconds": 45})]
    assert observed == ["operation-returned", "event"]


def test_sqlite_renew_no_transition_runs_no_success_observers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[str] = []
    manager = _manager_without_preflight(tmp_path, monkeypatch, observed)

    def renew_stub(*_args: Any, **_kwargs: Any) -> LifecycleResult:
        observed.append("operation-returned")
        return LifecycleResult.no_transition(NoTransitionReason.STALE_LEASE)

    monkeypatch.setattr(manager_module, "_sqlite_renew_lease", renew_stub, raising=False)

    renewed = manager.renew_job_lease(42, seconds=30, enforce=False)

    assert renewed is False
    assert observed == ["operation-returned"]


def test_sqlite_renew_backend_error_runs_no_success_observers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[str] = []
    manager = _manager_without_preflight(tmp_path, monkeypatch, observed)

    def renew_stub(*_args: Any, **_kwargs: Any) -> LifecycleResult:
        observed.append("operation-returned")
        raise sqlite3.OperationalError("forced renewal failure")

    monkeypatch.setattr(manager_module, "_sqlite_renew_lease", renew_stub, raising=False)

    with pytest.raises(sqlite3.OperationalError, match="forced renewal failure"):
        manager.renew_job_lease(42, seconds=30, enforce=False)

    assert observed == ["operation-returned"]


class _RollbackInsteadOfRenewalCommit:
    def __init__(self, inner: sqlite3.Connection) -> None:
        self._inner = inner

    def __enter__(self) -> _RollbackInsteadOfRenewalCommit:
        self._inner.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
        if exc_type is not None:
            return self._inner.__exit__(exc_type, exc, tb)
        self._inner.rollback()
        raise RuntimeError("forced renewal commit failure")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def test_sqlite_renew_commit_failure_rolls_back_and_suppresses_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = JobManager(tmp_path / "renew-commit.db")
    job = manager.create_job(
        domain="renew-commit",
        queue="default",
        job_type="work",
        payload={},
        owner_user_id="owner",
    )
    acquired = manager.acquire_next_job(
        domain="renew-commit",
        queue="default",
        job_type="work",
        lease_seconds=30,
        worker_id="worker-1",
    )
    assert acquired is not None
    before = manager.get_job(int(job["id"]))
    assert before is not None

    observed: list[str] = []
    monkeypatch.setattr(
        manager_module,
        "emit_job_event",
        lambda *_args, **_kwargs: observed.append("event"),
    )
    original_connect = manager._connect
    monkeypatch.setattr(
        manager,
        "_connect",
        lambda: _RollbackInsteadOfRenewalCommit(original_connect()),
    )

    with pytest.raises(RuntimeError, match="forced renewal commit failure"):
        manager.renew_job_lease(
            int(job["id"]),
            seconds=600,
            worker_id="worker-1",
            lease_id=str(acquired["lease_id"]),
            progress_percent=75.0,
            progress_message="should roll back",
            enforce=True,
        )

    monkeypatch.setattr(manager, "_connect", original_connect)
    after = manager.get_job(int(job["id"]))
    assert after is not None
    assert after["leased_until"] == before["leased_until"]
    assert after["progress_percent"] == before["progress_percent"]
    assert after["progress_message"] == before["progress_message"]
    assert observed == []


@pytest.mark.parametrize(
    ("reason", "expected_observed", "expected_events"),
    [
        (
            "yield",
            ["operation-returned", "gauge", "event"],
            [("job.released", {"id": 42, "domain": "facade", "queue": "default", "job_type": "work"}, {"reason": "yield"})],
        ),
        (None, ["operation-returned", "gauge"], []),
    ],
    ids=["truthy-reason", "no-reason"],
)
def test_sqlite_release_runs_post_commit_observers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reason: str | None,
    expected_observed: list[str],
    expected_events: list[tuple[str, dict[str, Any], dict[str, Any]]],
) -> None:
    observed: list[str] = []
    captured: dict[str, Any] = {}
    events: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    manager = _manager_without_preflight(tmp_path, monkeypatch, observed)
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "true")
    monkeypatch.setattr(
        manager_module,
        "submit_job_audit_event",
        lambda *_args, **_kwargs: pytest.fail("release must not use the durable outbox"),
    )

    def release_stub(
        _conn: sqlite3.Connection,
        *,
        command: Any,
        counters_enabled: bool,
    ) -> LifecycleResult:
        captured.update(command=command, counters_enabled=counters_enabled)
        observed.append("operation-returned")
        return LifecycleResult.applied(
            row={
                "id": command.job_id,
                "domain": "facade",
                "queue": "default",
                "job_type": "work",
                "status": "queued",
            }
        )

    def capture_event(
        event_type: str,
        *,
        job: dict[str, Any],
        attrs: dict[str, Any],
    ) -> None:
        observed.append("event")
        events.append((event_type, job, attrs))

    monkeypatch.setattr(manager_module, "_sqlite_release_job", release_stub, raising=False)
    monkeypatch.setattr(manager_module, "emit_job_event", capture_event)

    released = manager.release_job(
        42,
        worker_id="worker-1",
        lease_id="lease-1",
        reason=reason,
        enforce=True,
    )

    assert released is True
    assert captured["command"].job_id == 42
    assert captured["command"].enforce is True
    assert captured["command"].worker_id == "worker-1"
    assert captured["command"].lease_id == "lease-1"
    assert captured["command"].reason == reason
    assert captured["counters_enabled"] is True
    assert events == expected_events
    assert observed == expected_observed


def test_sqlite_release_no_transition_runs_no_success_observers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[str] = []
    manager = _manager_without_preflight(tmp_path, monkeypatch, observed)

    def release_stub(*_args: Any, **_kwargs: Any) -> LifecycleResult:
        observed.append("operation-returned")
        return LifecycleResult.no_transition(NoTransitionReason.WRONG_STATUS)

    monkeypatch.setattr(manager_module, "_sqlite_release_job", release_stub, raising=False)

    released = manager.release_job(42, reason="yield", enforce=False)

    assert released is False
    assert observed == ["operation-returned"]


def test_sqlite_release_backend_error_runs_no_success_observers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[str] = []
    manager = _manager_without_preflight(tmp_path, monkeypatch, observed)

    def release_stub(*_args: Any, **_kwargs: Any) -> LifecycleResult:
        observed.append("operation-returned")
        raise sqlite3.OperationalError("forced release failure")

    monkeypatch.setattr(manager_module, "_sqlite_release_job", release_stub, raising=False)

    with pytest.raises(sqlite3.OperationalError, match="forced release failure"):
        manager.release_job(42, reason="yield", enforce=False)

    assert observed == ["operation-returned"]


def test_sqlite_release_rejects_missing_enforced_credentials_before_connect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = JobManager(tmp_path / "release-precheck.db")
    monkeypatch.setattr(
        manager,
        "_connect",
        lambda: pytest.fail("release opened a connection before credential validation"),
    )

    assert manager.release_job(42, worker_id="worker-1", enforce=True) is False
    assert manager.release_job(42, lease_id="lease-1", enforce=True) is False


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


def test_postgres_renew_runs_exact_event_after_operation_returns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[str] = []
    captured: dict[str, Any] = {}
    events: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    manager = _postgres_manager_without_preflight(tmp_path, monkeypatch, observed)
    monkeypatch.setenv("JOBS_LEASE_MAX_SECONDS", "45")
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "true")
    monkeypatch.setattr(
        manager_module,
        "submit_job_audit_event",
        lambda *_args, **_kwargs: pytest.fail("renewal must not use the durable outbox"),
    )

    def renew_stub(
        _conn: Any,
        cursor_factory: Any,
        *,
        command: Any,
        now: Any,
    ) -> LifecycleResult:
        captured.update(command=command, cursor_factory=cursor_factory, now=now)
        observed.append("operation-returned")
        return LifecycleResult.applied(row={"id": command.job_id, "status": "processing"})

    def capture_event(
        event_type: str,
        *,
        job: dict[str, Any],
        attrs: dict[str, Any],
    ) -> None:
        observed.append("event")
        events.append((event_type, job, attrs))

    monkeypatch.setattr(manager_module, "_postgres_renew_lease", renew_stub, raising=False)
    monkeypatch.setattr(manager_module, "emit_job_event", capture_event)

    renewed = manager.renew_job_lease(
        42,
        seconds=999,
        worker_id="worker-1",
        lease_id="lease-1",
        progress_percent=62.5,
        progress_message="indexing",
        enforce=True,
    )

    assert renewed is True
    assert captured["cursor_factory"] == manager._pg_cursor
    assert captured["command"].job_id == 42
    assert captured["command"].seconds == 45
    assert captured["command"].enforce is True
    assert captured["command"].worker_id == "worker-1"
    assert captured["command"].lease_id == "lease-1"
    assert captured["command"].progress_percent == 62.5
    assert captured["command"].progress_message == "indexing"
    assert captured["now"] is not None
    assert events == [("job.lease_renewed", {"id": 42}, {"seconds": 45})]
    assert observed == ["operation-returned", "event"]


def test_postgres_renew_no_transition_runs_no_success_observers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[str] = []
    manager = _postgres_manager_without_preflight(tmp_path, monkeypatch, observed)

    def renew_stub(*_args: Any, **_kwargs: Any) -> LifecycleResult:
        observed.append("operation-returned")
        return LifecycleResult.no_transition(NoTransitionReason.STALE_LEASE)

    monkeypatch.setattr(manager_module, "_postgres_renew_lease", renew_stub, raising=False)

    renewed = manager.renew_job_lease(42, seconds=30, enforce=False)

    assert renewed is False
    assert observed == ["operation-returned"]


def test_postgres_renew_backend_error_runs_no_success_observers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[str] = []
    manager = _postgres_manager_without_preflight(tmp_path, monkeypatch, observed)

    def renew_stub(*_args: Any, **_kwargs: Any) -> LifecycleResult:
        observed.append("operation-returned")
        raise RuntimeError("forced renewal failure")

    monkeypatch.setattr(manager_module, "_postgres_renew_lease", renew_stub, raising=False)

    with pytest.raises(RuntimeError, match="forced renewal failure"):
        manager.renew_job_lease(42, seconds=30, enforce=False)

    assert observed == ["operation-returned"]


@pytest.mark.pg_jobs
def test_postgres_renew_commit_failure_rolls_back_and_suppresses_event(
    jobs_pg_dsn: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = JobManager(None, backend="postgres", db_url=jobs_pg_dsn)
    job = manager.create_job(
        domain="renew-commit",
        queue="default",
        job_type="work",
        payload={},
        owner_user_id="owner",
    )
    acquired = manager.acquire_next_job(
        domain="renew-commit",
        queue="default",
        job_type="work",
        lease_seconds=30,
        worker_id="worker-1",
    )
    assert acquired is not None
    before = manager.get_job(int(job["id"]))
    assert before is not None

    observed: list[str] = []
    monkeypatch.setattr(
        manager_module,
        "emit_job_event",
        lambda *_args, **_kwargs: observed.append("event"),
    )
    original_connect = manager._connect
    monkeypatch.setattr(
        manager,
        "_connect",
        lambda: _RollbackInsteadOfRenewalCommit(original_connect()),
    )

    with pytest.raises(RuntimeError, match="forced renewal commit failure"):
        manager.renew_job_lease(
            int(job["id"]),
            seconds=600,
            worker_id="worker-1",
            lease_id=str(acquired["lease_id"]),
            progress_percent=75.0,
            progress_message="should roll back",
            enforce=True,
        )

    monkeypatch.setattr(manager, "_connect", original_connect)
    after = manager.get_job(int(job["id"]))
    assert after is not None
    assert after["leased_until"] == before["leased_until"]
    assert after["progress_percent"] == before["progress_percent"]
    assert after["progress_message"] == before["progress_message"]
    assert observed == []


@pytest.mark.parametrize(
    ("reason", "expected_observed", "expected_events"),
    [
        (
            "yield",
            ["operation-returned", "gauge", "event"],
            [
                (
                    "job.released",
                    {
                        "id": 42,
                        "domain": "facade",
                        "queue": "default",
                        "job_type": "work",
                    },
                    {"reason": "yield"},
                )
            ],
        ),
        (None, ["operation-returned", "gauge"], []),
    ],
    ids=["truthy-reason", "no-reason"],
)
def test_postgres_release_runs_post_commit_observers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reason: str | None,
    expected_observed: list[str],
    expected_events: list[tuple[str, dict[str, Any], dict[str, Any]]],
) -> None:
    observed: list[str] = []
    captured: dict[str, Any] = {}
    events: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    manager = _postgres_manager_without_preflight(tmp_path, monkeypatch, observed)
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "true")
    monkeypatch.setattr(
        manager_module,
        "submit_job_audit_event",
        lambda *_args, **_kwargs: pytest.fail("release must not use the durable outbox"),
    )

    def release_stub(
        _conn: Any,
        cursor_factory: Any,
        *,
        command: Any,
        counters_enabled: bool,
    ) -> LifecycleResult:
        captured.update(
            command=command,
            counters_enabled=counters_enabled,
            cursor_factory=cursor_factory,
        )
        observed.append("operation-returned")
        return LifecycleResult.applied(
            row={
                "id": command.job_id,
                "domain": "facade",
                "queue": "default",
                "job_type": "work",
                "status": "queued",
            }
        )

    def capture_event(
        event_type: str,
        *,
        job: dict[str, Any],
        attrs: dict[str, Any],
    ) -> None:
        observed.append("event")
        events.append((event_type, job, attrs))

    monkeypatch.setattr(manager_module, "_postgres_release_job", release_stub, raising=False)
    monkeypatch.setattr(manager_module, "emit_job_event", capture_event)

    released = manager.release_job(
        42,
        worker_id="worker-1",
        lease_id="lease-1",
        reason=reason,
        enforce=True,
    )

    assert released is True
    assert captured["cursor_factory"] == manager._pg_cursor
    assert captured["command"].job_id == 42
    assert captured["command"].enforce is True
    assert captured["command"].worker_id == "worker-1"
    assert captured["command"].lease_id == "lease-1"
    assert captured["command"].reason == reason
    assert captured["counters_enabled"] is True
    assert events == expected_events
    assert observed == expected_observed


def test_postgres_release_no_transition_runs_no_success_observers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[str] = []
    manager = _postgres_manager_without_preflight(tmp_path, monkeypatch, observed)

    def release_stub(*_args: Any, **_kwargs: Any) -> LifecycleResult:
        observed.append("operation-returned")
        return LifecycleResult.no_transition(NoTransitionReason.WRONG_STATUS)

    monkeypatch.setattr(manager_module, "_postgres_release_job", release_stub, raising=False)

    released = manager.release_job(42, reason="yield", enforce=False)

    assert released is False
    assert observed == ["operation-returned"]


def test_postgres_release_backend_error_runs_no_success_observers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[str] = []
    manager = _postgres_manager_without_preflight(tmp_path, monkeypatch, observed)

    def release_stub(*_args: Any, **_kwargs: Any) -> LifecycleResult:
        observed.append("operation-returned")
        raise RuntimeError("forced release failure")

    monkeypatch.setattr(manager_module, "_postgres_release_job", release_stub, raising=False)

    with pytest.raises(RuntimeError, match="forced release failure"):
        manager.release_job(42, reason="yield", enforce=False)

    assert observed == ["operation-returned"]


def test_postgres_release_rejects_missing_enforced_credentials_before_connect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[str] = []
    manager = _postgres_manager_without_preflight(tmp_path, monkeypatch, observed)
    monkeypatch.setattr(
        manager,
        "_connect",
        lambda: pytest.fail("release opened a connection before credential validation"),
    )

    assert manager.release_job(42, worker_id="worker-1", enforce=True) is False
    assert manager.release_job(42, lease_id="lease-1", enforce=True) is False
