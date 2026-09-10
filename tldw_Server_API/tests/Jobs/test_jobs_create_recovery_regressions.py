"""Regressions for Jobs create-event routing and recovery lock ordering."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from tldw_Server_API.app.core.Jobs.manager import JobManager


class _RecoveryConnection:
    """Minimal Postgres connection boundary for recovery ordering tests."""

    def __enter__(self) -> _RecoveryConnection:
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: Any,
    ) -> bool:
        return False

    def close(self) -> None:
        return None


class _RecoveryCursor:
    """Capture counter mutations while returning lease-ordered job rows."""

    def __init__(self) -> None:
        self.rowcount = 0
        self._rows: list[dict[str, Any]] = []
        self.counter_updates: list[tuple[Any, ...]] = []

    def execute(self, sql: Any, params: Any = None) -> None:
        normalized = " ".join(str(sql).split())
        self.rowcount = 0
        self._rows = []
        if normalized.startswith("SELECT id, uuid, domain, queue, job_type"):
            self._rows = [
                {
                    "id": 1,
                    "uuid": "job-zeta-1",
                    "domain": "recovery",
                    "queue": "default",
                    "job_type": "zeta",
                    "owner_user_id": "owner",
                    "request_id": None,
                    "trace_id": None,
                    "effective_retry_count": 0,
                    "effective_max_retries": 3,
                    "effective_expired_lease_policy": "consume_retry",
                },
                {
                    "id": 2,
                    "uuid": "job-alpha",
                    "domain": "recovery",
                    "queue": "default",
                    "job_type": "alpha",
                    "owner_user_id": "owner",
                    "request_id": None,
                    "trace_id": None,
                    "effective_retry_count": 0,
                    "effective_max_retries": 3,
                    "effective_expired_lease_policy": "consume_retry",
                },
                {
                    "id": 3,
                    "uuid": "job-zeta-2",
                    "domain": "recovery",
                    "queue": "default",
                    "job_type": "zeta",
                    "owner_user_id": "owner",
                    "request_id": None,
                    "trace_id": None,
                    "effective_retry_count": 0,
                    "effective_max_retries": 3,
                    "effective_expired_lease_policy": "consume_retry",
                },
            ]
            return
        if normalized.startswith("UPDATE jobs SET status='queued'"):
            self.rowcount = 1
            return
        if normalized.startswith("INSERT INTO job_counters"):
            self.counter_updates.append(tuple(params or ()))
            self.rowcount = 1
            return
        raise AssertionError(f"unexpected recovery SQL: {normalized}")

    def fetchall(self) -> list[dict[str, Any]]:
        return list(self._rows)


@pytest.mark.unit
def test_postgres_expired_recovery_aggregates_and_sorts_counter_updates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Counter locks use a global key order, independent of leased job order."""

    import tldw_Server_API.app.core.Jobs.manager as jobs_manager_module
    import tldw_Server_API.app.core.Jobs.pg_util as pg_util_module

    monkeypatch.setenv("JOBS_PG_SKIP_SCHEMA_INIT", "true")
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "false")
    monkeypatch.setattr(pg_util_module, "negotiate_pg_dsn", lambda dsn: dsn)
    cursor = _RecoveryCursor()
    manager = JobManager(None, backend="postgres", db_url="postgresql://fake")
    monkeypatch.setattr(manager, "_connect", _RecoveryConnection)

    @contextmanager
    def cursor_factory(_connection: Any) -> Any:
        yield cursor

    monkeypatch.setattr(manager, "_pg_cursor", cursor_factory)
    monkeypatch.setattr(manager, "_update_gauges", lambda **_kwargs: None)
    monkeypatch.setattr(jobs_manager_module, "emit_job_event", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(jobs_manager_module, "increment_retries", lambda *_args, **_kwargs: None)

    assert manager._recover_expired_processing_jobs() == 3
    assert cursor.counter_updates == [
        ("recovery", "default", "alpha", 1, 1),
        ("recovery", "default", "zeta", 2, 2),
    ]


@pytest.mark.unit
@pytest.mark.parametrize("outbox_enabled", [False, True], ids=["direct", "outbox"])
@pytest.mark.parametrize("events_enabled", [False, True], ids=["events-off", "events-on"])
def test_non_idempotent_create_routes_one_audit_and_one_durable_event(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    outbox_enabled: bool,
    events_enabled: bool,
) -> None:
    """A transactional job.created fact is never audited or persisted twice."""

    import tldw_Server_API.app.core.Jobs.event_stream as event_stream_module
    import tldw_Server_API.app.core.Jobs.manager as jobs_manager_module

    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "true" if outbox_enabled else "false")
    monkeypatch.setenv("JOBS_EVENTS_ENABLED", "true" if events_enabled else "false")
    audit_calls: list[str] = []

    def record_audit(event_type: str, **_kwargs: Any) -> None:
        audit_calls.append(event_type)

    monkeypatch.setattr(jobs_manager_module, "submit_job_audit_event", record_audit)
    monkeypatch.setattr(event_stream_module, "submit_job_audit_event", record_audit)

    manager = JobManager(tmp_path / "jobs.db")
    created = manager.create_job(
        domain="events",
        queue="default",
        job_type="create-once",
        payload={},
        owner_user_id="owner",
    )

    connection = manager._connect()
    try:
        event_count = connection.execute(
            "SELECT COUNT(*) FROM job_events WHERE job_id=? AND event_type='job.created'",
            (int(created["id"]),),
        ).fetchone()[0]
    finally:
        connection.close()

    assert audit_calls == ["job.created"]
    assert int(event_count) == 1
