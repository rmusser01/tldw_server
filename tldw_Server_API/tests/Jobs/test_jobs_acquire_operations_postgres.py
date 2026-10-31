"""Direct Postgres single-job acquisition operation tests."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from typing import Any

import pytest

from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.operations.contracts import (
    AcquireJobCommand,
    NoTransitionReason,
    OperationOutcome,
)
from tldw_Server_API.app.core.Jobs.operations.postgres import lifecycle
from tldw_Server_API.app.core.Jobs.operations.postgres.lifecycle import acquire_job

psycopg = pytest.importorskip("psycopg")
pytestmark = pytest.mark.pg_jobs

NOW = datetime(2026, 1, 2, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture()
def manager(jobs_pg_dsn: str, monkeypatch: pytest.MonkeyPatch) -> JobManager:
    monkeypatch.setenv("JOBS_DB_URL", jobs_pg_dsn)
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "false")
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "false")
    return JobManager(None, backend="postgres", db_url=jobs_pg_dsn)


@pytest.fixture()
def conn(manager: JobManager):
    connection = manager._connect()
    try:
        yield connection
    finally:
        connection.close()


def _create_job(
    manager: JobManager,
    *,
    domain: str = "acquire",
    job_type: str = "work",
    owner_user_id: str = "owner",
    priority: int = 5,
) -> dict[str, Any]:
    return manager.create_job(
        domain=domain,
        queue="default",
        job_type=job_type,
        payload={},
        owner_user_id=owner_user_id,
        priority=priority,
    )


def _execute(manager: JobManager, sql: str, params: tuple[Any, ...] = ()) -> None:
    connection = manager._connect()
    try:
        with connection, manager._pg_cursor(connection) as cur:
            cur.execute(sql, params)
    finally:
        connection.close()


def _command(
    *,
    domain: str = "acquire",
    lease_id: str = "lease-exact",
    owner_user_id: str | None = None,
    max_inflight_quota: int = 0,
    priority_direction: str = "ASC",
    tie_break: str | None = None,
    single_update: bool = False,
) -> AcquireJobCommand:
    return AcquireJobCommand(
        domain=domain,
        queue="default",
        lease_seconds=30,
        worker_id="worker-1",
        lease_id=lease_id,
        owner_user_id=owner_user_id,
        job_type="work",
        max_inflight_quota=max_inflight_quota,
        priority_direction=priority_direction,
        tie_break=tie_break,
        single_update=single_update,
    )


@pytest.mark.parametrize("single_update", [False, True], ids=["two-step", "single-update"])
def test_postgres_acquire_applies_exact_command_lease_identity(
    manager: JobManager,
    conn: Any,
    single_update: bool,
) -> None:
    job = _create_job(manager)

    result = acquire_job(
        conn,
        manager._pg_cursor,
        command=_command(single_update=single_update),
        counters_enabled=False,
        now=NOW,
    )

    assert result.outcome is OperationOutcome.APPLIED
    assert result.row is not None
    assert int(result.row["id"]) == int(job["id"])
    assert result.row["status"] == "processing"
    assert result.row["worker_id"] == "worker-1"
    assert result.row["lease_id"] == "lease-exact"


def test_postgres_acquire_returns_no_transition_without_eligible_row(
    manager: JobManager,
    conn: Any,
) -> None:
    job = _create_job(manager)
    _execute(
        manager,
        "UPDATE jobs SET available_at = NOW() + interval '1 day' WHERE id = %s",
        (int(job["id"]),),
    )

    result = acquire_job(
        conn,
        manager._pg_cursor,
        command=_command(),
        counters_enabled=False,
        now=NOW,
    )

    assert result.outcome is OperationOutcome.NO_TRANSITION
    assert result.no_transition_reason is NoTransitionReason.NO_ELIGIBLE_JOB


@pytest.mark.parametrize("single_update", [False, True], ids=["two-step", "single-update"])
def test_postgres_acquire_skips_locked_candidate(
    manager: JobManager,
    conn: Any,
    single_update: bool,
) -> None:
    locked = _create_job(manager, priority=1)
    eligible = _create_job(manager, priority=2)
    locking_conn = manager._connect()
    try:
        with manager._pg_cursor(locking_conn) as cur:
            cur.execute("SELECT id FROM jobs WHERE id = %s FOR UPDATE", (int(locked["id"]),))

        result = acquire_job(
            conn,
            manager._pg_cursor,
            command=_command(single_update=single_update),
            counters_enabled=False,
            now=NOW,
        )
    finally:
        locking_conn.rollback()
        locking_conn.close()

    assert result.row is not None
    assert int(result.row["id"]) == int(eligible["id"])


@pytest.mark.parametrize("single_update", [False, True], ids=["two-step", "single-update"])
def test_postgres_acquire_honors_resolved_priority_direction(
    manager: JobManager,
    conn: Any,
    single_update: bool,
) -> None:
    _create_job(manager, priority=1)
    lower_priority = _create_job(manager, priority=10)

    result = acquire_job(
        conn,
        manager._pg_cursor,
        command=_command(priority_direction="DESC", single_update=single_update),
        counters_enabled=False,
        now=NOW,
    )

    assert result.row is not None
    assert int(result.row["id"]) == int(lower_priority["id"])


@pytest.mark.parametrize(
    ("tie_break", "expected_offset"),
    [("fifo", "-1 hour"), ("lifo", "-1 minute"), (None, "-1 hour")],
    ids=["fifo", "lifo", "default-fifo"],
)
def test_postgres_acquire_honors_resolved_tie_ordering(
    manager: JobManager,
    conn: Any,
    tie_break: str | None,
    expected_offset: str,
) -> None:
    older = _create_job(manager)
    newer = _create_job(manager)
    _execute(
        manager,
        "UPDATE jobs SET created_at = NOW() - (%s)::interval WHERE id = %s",
        ("1 hour", int(older["id"])),
    )
    _execute(
        manager,
        "UPDATE jobs SET created_at = NOW() - (%s)::interval WHERE id = %s",
        ("1 minute", int(newer["id"])),
    )

    result = acquire_job(
        conn,
        manager._pg_cursor,
        command=_command(tie_break=tie_break),
        counters_enabled=False,
        now=NOW,
    )

    expected = older if expected_offset == "-1 hour" else newer
    assert result.row is not None
    assert int(result.row["id"]) == int(expected["id"])


def test_postgres_acquire_skips_dependency_blocked_job(
    manager: JobManager,
    conn: Any,
) -> None:
    parent = _create_job(manager, job_type="parent", priority=1)
    blocked = _create_job(manager, priority=1)
    eligible = _create_job(manager, priority=2)
    assert manager.add_job_dependency(str(blocked["uuid"]), str(parent["uuid"]))

    result = acquire_job(
        conn,
        manager._pg_cursor,
        command=_command(),
        counters_enabled=False,
        now=NOW,
    )

    assert result.row is not None
    assert int(result.row["id"]) == int(eligible["id"])


def test_postgres_acquire_serializes_max_inflight_quota(manager: JobManager) -> None:
    _create_job(manager)
    _create_job(manager)
    barrier = threading.Barrier(2)

    def run(index: int):
        connection = manager._connect()
        try:
            barrier.wait(timeout=5)
            return acquire_job(
                connection,
                manager._pg_cursor,
                command=_command(
                    lease_id=f"lease-{index}",
                    owner_user_id="owner",
                    max_inflight_quota=1,
                ),
                counters_enabled=False,
                now=NOW,
            )
        finally:
            connection.close()

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(run, range(2)))

    assert [result.outcome for result in results].count(OperationOutcome.APPLIED) == 1
    assert [result.outcome for result in results].count(OperationOutcome.NO_TRANSITION) == 1
    assert {result.no_transition_reason for result in results if not result.transition_applied} == {
        NoTransitionReason.NO_ELIGIBLE_JOB
    }
    connection = manager._connect()
    try:
        with manager._pg_cursor(connection) as cur:
            cur.execute("SELECT status, COUNT(*) AS c FROM jobs GROUP BY status ORDER BY status")
            counts = {row["status"]: int(row["c"]) for row in cur.fetchall()}
    finally:
        connection.close()
    assert counts == {"processing": 1, "queued": 1}


def test_postgres_acquire_ignores_expired_processing_lease_for_quota(
    manager: JobManager,
    conn: Any,
) -> None:
    expired = _create_job(manager)
    queued = _create_job(manager)
    _execute(
        manager,
        "UPDATE jobs SET status = 'processing', leased_until = NOW() - interval '1 hour' WHERE id = %s",
        (int(expired["id"]),),
    )

    result = acquire_job(
        conn,
        manager._pg_cursor,
        command=_command(owner_user_id="owner", max_inflight_quota=1),
        counters_enabled=False,
        now=NOW,
    )

    assert result.row is not None
    assert int(result.row["id"]) == int(queued["id"])


@pytest.mark.parametrize(
    ("scheduled", "initial_counts"),
    [(False, (1, 0, 0)), (True, (0, 1, 0))],
    ids=["ready", "scheduled"],
)
def test_postgres_acquire_moves_counter_to_processing(
    manager: JobManager,
    conn: Any,
    scheduled: bool,
    initial_counts: tuple[int, int, int],
) -> None:
    job = _create_job(manager)
    if scheduled:
        _execute(
            manager,
            "UPDATE jobs SET available_at = NOW() - interval '1 second' WHERE id = %s",
            (int(job["id"]),),
        )
    _execute(
        manager,
        (
            "INSERT INTO job_counters(domain, queue, job_type, ready_count, scheduled_count, "
            "processing_count, quarantined_count) VALUES('acquire', 'default', 'work', %s, %s, %s, 0)"
        ),
        initial_counts,
    )

    result = acquire_job(
        conn,
        manager._pg_cursor,
        command=_command(),
        counters_enabled=True,
        now=NOW,
    )

    assert result.outcome is OperationOutcome.APPLIED
    connection = manager._connect()
    try:
        with manager._pg_cursor(connection) as cur:
            cur.execute(
                "SELECT ready_count, scheduled_count, processing_count FROM job_counters "
                "WHERE domain = 'acquire' AND queue = 'default' AND job_type = 'work'"
            )
            counter = cur.fetchone()
    finally:
        connection.close()
    assert tuple(counter.values()) == (0, 0, 1)


def test_postgres_counter_failure_rolls_back_to_savepoint_and_commits_lease(
    manager: JobManager,
    conn: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _create_job(manager)

    def fail_counter(cur: Any, *, acquired: dict[str, Any]) -> None:
        del acquired
        cur.execute("SELECT definitely_missing_column FROM job_counters")

    monkeypatch.setattr(lifecycle, "_bump_acquired_counters", fail_counter)

    result = acquire_job(
        conn,
        manager._pg_cursor,
        command=_command(),
        counters_enabled=True,
        now=NOW,
    )

    assert result.outcome is OperationOutcome.APPLIED
    stored = manager.get_job(int(job["id"]))
    assert stored is not None
    assert stored["status"] == "processing"
    assert stored["lease_id"] == "lease-exact"


class _SavepointFailureCursor(AbstractContextManager[Any]):
    def __init__(self, inner: AbstractContextManager[Any]) -> None:
        self._inner = inner
        self._cursor: Any = None

    def __enter__(self) -> _SavepointFailureCursor:
        self._cursor = self._inner.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> Any:
        return self._inner.__exit__(exc_type, exc, traceback)

    def execute(self, sql: Any, params: Any = None) -> Any:
        if str(sql).startswith("SAVEPOINT"):
            raise psycopg.OperationalError("savepoint infrastructure failed")
        return self._cursor.execute(sql, params)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cursor, name)


def test_postgres_savepoint_infrastructure_failure_propagates_and_rolls_back_lease(
    manager: JobManager,
    conn: Any,
) -> None:
    job = _create_job(manager)

    def cursor_factory(connection: Any) -> _SavepointFailureCursor:
        return _SavepointFailureCursor(manager._pg_cursor(connection))

    with pytest.raises(psycopg.OperationalError, match="savepoint infrastructure failed"):
        acquire_job(
            conn,
            cursor_factory,
            command=_command(),
            counters_enabled=True,
            now=NOW,
        )

    stored = manager.get_job(int(job["id"]))
    assert stored is not None
    assert stored["status"] == "queued"
    assert stored["lease_id"] is None


def test_postgres_max_inflight_advisory_key_matches_existing_workers() -> None:
    expected = JobManager._pg_advisory_key(
        object(),
        "max-inflight",
        "chatbooks",
        "owner-1",
    )

    assert lifecycle._pg_advisory_key("max-inflight", "chatbooks", "owner-1") == expected
