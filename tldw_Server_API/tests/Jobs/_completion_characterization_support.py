"""Small delegating probes for the real completion transaction, never SQL fakes."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

import tldw_Server_API.app.core.Jobs.manager as manager_module
from tldw_Server_API.app.core.Jobs.manager import JobManager

_BINDING_COMPLETION = "binding-token"


class CompletionConnection:
    """Record the owner's explicit boundaries while retaining native DB behavior."""

    def __init__(
        self,
        inner: Any,
        calls: list[str],
        on_commit: Callable[[], None] | None,
        fail_at: str | None,
    ) -> None:
        self.inner = inner
        self.calls = calls
        self.on_commit = on_commit
        self.fail_at = fail_at

    def __enter__(self) -> CompletionConnection:
        self.inner.__enter__()
        return self

    def __exit__(self, *exc: Any) -> Any:
        try:
            return self.inner.__exit__(*exc)
        finally:
            self.calls.append("context_exit")
            if self.fail_at == "context_exit":
                raise RuntimeError("context exit probe")

    def execute(self, sql: Any, parameters: Any = ()) -> Any:
        return self.inner.execute(sql, parameters)

    def commit(self) -> None:
        self.calls.append("commit")
        if self.fail_at == "commit":
            raise RuntimeError("commit probe")
        self.inner.commit()
        if self.on_commit is not None:
            self.on_commit()

    def rollback(self) -> None:
        self.calls.append("rollback")
        self.inner.rollback()

    def close(self) -> None:
        self.calls.append("close")
        self.inner.close()
        if self.fail_at == "close":
            raise RuntimeError("close probe")
        if self.fail_at == "close_assertion":
            raise AssertionError("close probe")

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


class CompletionCursor:
    """Keep PostgreSQL's RLS cursor and its distinct context exit observable."""

    def __init__(self, inner: Any, calls: list[str]) -> None:
        self.inner = inner
        self.calls = calls

    def __enter__(self) -> CompletionCursor:
        self.inner.__enter__()
        return self

    def __exit__(self, *exc: Any) -> Any:
        try:
            return self.inner.__exit__(*exc)
        finally:
            self.calls.append("cursor_exit")

    def execute(self, sql: Any, parameters: Any = None) -> Any:
        return self.inner.execute(sql, parameters)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


def record_completion_boundaries(
    manager: JobManager,
    monkeypatch: pytest.MonkeyPatch,
    calls: list[str],
    *,
    on_commit: Callable[[], None] | None = None,
    fail_at: str | None = None,
) -> None:
    """Wrap only the first connection; SLA policy reads keep their own lifetime."""
    original_connect = manager._connect
    original_cursor = manager._pg_cursor
    owner: CompletionConnection | None = None

    def connect() -> Any:
        nonlocal owner
        connection = original_connect()
        if owner is None:
            owner = CompletionConnection(connection, calls, on_commit, fail_at)
            return owner
        return connection

    def cursor(connection: Any) -> Any:
        native = original_cursor(connection)
        return CompletionCursor(native, calls) if connection is owner else native

    monkeypatch.setattr(manager, "_connect", connect)
    monkeypatch.setattr(manager, "_pg_cursor", cursor)


def record_completion_observers(manager: JobManager, monkeypatch: pytest.MonkeyPatch, calls: list[str]) -> None:
    """Replace external telemetry only, after creation/acquisition have finished."""
    for name, label in (
        ("increment_json_truncated", "truncated"),
        ("observe_duration", "duration"),
        ("increment_completed", "completed"),
        ("increment_sla_breach", "sla"),
        ("_record_job_span", "trace"),
        ("observe_job_event", "event"),
    ):
        monkeypatch.setattr(
            manager_module,
            name,
            lambda *args, _label=label, **kwargs: calls.append(_label),
        )
    monkeypatch.setattr(manager, "_update_gauges", lambda **kwargs: calls.append("original_gauges"))


def seed_job(manager: JobManager, *, processing: bool = True, **kwargs: Any) -> dict[str, Any]:
    """Create the scoped row and, by default, acquire a real lease."""
    created = manager.create_job(
        domain="chatbooks",
        queue="default",
        job_type="completion-characterization",
        payload={},
        owner_user_id="owner-1",
        request_id="request-1",
        trace_id="trace-1",
        **kwargs,
    )
    if not processing:
        return created
    acquired = manager.acquire_next_job(domain="chatbooks", queue="default", worker_id="worker-1", lease_seconds=60)
    assert acquired is not None and acquired["id"] == created["id"]
    assert acquired["lease_id"]
    return acquired


def completion_arguments(job: dict[str, Any]) -> dict[str, Any]:
    """Return the valid identity used by binding and failure-order probes."""
    return {
        "worker_id": job.get("worker_id"),
        "lease_id": job.get("lease_id"),
        "expected_uuid": job["uuid"],
        "completion_token": _BINDING_COMPLETION,
        "enforce": True,
    }


def independent_reader(manager: JobManager) -> JobManager:
    """Observe persistence without using the instrumented manager instance."""
    return JobManager(manager.db_path, backend=manager.backend, db_url=manager.db_url)


def rows(manager: JobManager, sql: str, parameters: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    """Read native rows, or commit test setup, on a separate connection."""
    connection = manager._connect()
    try:
        with connection:
            if manager.backend == "postgres":
                with manager._pg_cursor(connection) as cursor:
                    cursor.execute(sql.replace("?", "%s"), parameters)
                    return [dict(row) for row in cursor.fetchall()] if cursor.description else []
            cursor = connection.execute(sql, parameters)
            return [dict(row) for row in cursor.fetchall()] if cursor.description else []
    finally:
        connection.close()


def durable_state(reader: JobManager, job_id: int) -> tuple[Any, ...]:
    """Capture raw job, all scoped counters, and outbox for no-mutation checks."""
    return (
        rows(reader, "SELECT * FROM jobs WHERE id=?", (job_id,)),
        rows(reader, "SELECT * FROM job_counters ORDER BY domain, queue, job_type"),
        rows(reader, "SELECT * FROM job_events WHERE job_id=? ORDER BY id", (job_id,)),
    )


def assert_completed(reader: JobManager, job_id: int) -> None:
    """Assert committed completion, balanced counters and exactly one durable event."""
    job, counters, events = durable_state(reader, job_id)
    assert job[0]["status"] == "completed"
    assert job[0]["completion_token"] == "binding-token"
    assert (job[0]["worker_id"], job[0]["lease_id"], job[0]["leased_until"]) == (None, None, None)
    assert len(counters) == 1
    assert tuple(
        counters[0][name] for name in ("ready_count", "scheduled_count", "processing_count", "quarantined_count")
    ) == (0, 0, 0, 0)
    assert [event["event_type"] for event in events] == ["job.created", "job.acquired", "job.completed"]
