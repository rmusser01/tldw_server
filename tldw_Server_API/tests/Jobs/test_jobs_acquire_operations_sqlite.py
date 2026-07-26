"""Direct SQLite single-job acquisition operation tests."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tldw_Server_API.app.core.Jobs.migrations import ensure_jobs_tables
from tldw_Server_API.app.core.Jobs.operations.contracts import (
    AcquireJobCommand,
    NoTransitionReason,
    OperationOutcome,
)
from tldw_Server_API.app.core.Jobs.operations.sqlite.lifecycle import acquire_job

NOW = datetime(2026, 1, 2, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture()
def conn(tmp_path: Path) -> sqlite3.Connection:
    db_path = ensure_jobs_tables(tmp_path / "jobs.db")
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
    finally:
        connection.close()


def _insert_job(
    conn: sqlite3.Connection,
    *,
    uuid: str,
    domain: str = "acquire",
    job_type: str = "work",
    owner_user_id: str = "owner",
    status: str = "queued",
    priority: int = 5,
    available_at: str | None = None,
    created_at: str = "2026-01-01 00:00:00",
    leased_until: str | None = None,
) -> int:
    cursor = conn.execute(
        (
            "INSERT INTO jobs(uuid, domain, queue, job_type, owner_user_id, payload, status, priority, "
            "available_at, leased_until, created_at, updated_at) "
            "VALUES(?, ?, 'default', ?, ?, '{}', ?, ?, ?, ?, ?, ?)"
        ),
        (
            uuid,
            domain,
            job_type,
            owner_user_id,
            status,
            priority,
            available_at,
            leased_until,
            created_at,
            created_at,
        ),
    )
    conn.commit()
    return int(cursor.lastrowid)


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
def test_sqlite_acquire_applies_exact_command_lease_identity(
    conn: sqlite3.Connection,
    single_update: bool,
) -> None:
    job_id = _insert_job(conn, uuid=f"job-{single_update}")

    result = acquire_job(
        conn,
        command=_command(single_update=single_update),
        counters_enabled=False,
        now=NOW,
    )

    assert result.outcome is OperationOutcome.APPLIED
    assert result.row is not None
    assert int(result.row["id"]) == job_id
    assert result.row["status"] == "processing"
    assert result.row["worker_id"] == "worker-1"
    assert result.row["lease_id"] == "lease-exact"


def test_sqlite_acquire_returns_no_transition_without_eligible_row(conn: sqlite3.Connection) -> None:
    _insert_job(conn, uuid="future", available_at="2026-01-03 00:00:00")

    result = acquire_job(
        conn,
        command=_command(),
        counters_enabled=False,
        now=NOW,
    )

    assert result.outcome is OperationOutcome.NO_TRANSITION
    assert result.no_transition_reason is NoTransitionReason.NO_ELIGIBLE_JOB


@pytest.mark.parametrize(
    ("tie_break", "expected_uuid"),
    [("fifo", "older"), ("lifo", "newer"), (None, "older")],
    ids=["fifo", "lifo", "default-fifo"],
)
def test_sqlite_acquire_honors_resolved_ordering(
    conn: sqlite3.Connection,
    tie_break: str | None,
    expected_uuid: str,
) -> None:
    _insert_job(conn, uuid="older", created_at="2026-01-01 00:00:00")
    _insert_job(conn, uuid="newer", created_at="2026-01-01 01:00:00")

    result = acquire_job(
        conn,
        command=_command(tie_break=tie_break),
        counters_enabled=False,
        now=NOW,
    )

    assert result.row is not None
    assert result.row["uuid"] == expected_uuid


def test_sqlite_two_step_chatbooks_default_uses_lifo_without_scheduled_work(
    conn: sqlite3.Connection,
) -> None:
    _insert_job(conn, uuid="older", domain="chatbooks", created_at="2026-01-01 00:00:00")
    _insert_job(conn, uuid="newer", domain="chatbooks", created_at="2026-01-01 01:00:00")

    result = acquire_job(
        conn,
        command=_command(domain="chatbooks"),
        counters_enabled=False,
        now=NOW,
    )

    assert result.row is not None
    assert result.row["uuid"] == "newer"


def test_sqlite_two_step_chatbooks_default_uses_fifo_with_scheduled_work(
    conn: sqlite3.Connection,
) -> None:
    _insert_job(conn, uuid="older", domain="chatbooks", created_at="2026-01-01 00:00:00")
    _insert_job(conn, uuid="newer", domain="chatbooks", created_at="2026-01-01 01:00:00")
    _insert_job(
        conn,
        uuid="scheduled",
        domain="chatbooks",
        available_at="2026-01-03 00:00:00",
        created_at="2026-01-01 02:00:00",
    )

    result = acquire_job(
        conn,
        command=_command(domain="chatbooks"),
        counters_enabled=False,
        now=NOW,
    )

    assert result.row is not None
    assert result.row["uuid"] == "older"


def test_sqlite_acquire_skips_dependency_blocked_job(conn: sqlite3.Connection) -> None:
    _insert_job(conn, uuid="parent", job_type="parent", priority=1)
    _insert_job(conn, uuid="blocked", priority=1)
    eligible_id = _insert_job(conn, uuid="eligible", priority=2)
    conn.execute(
        "INSERT INTO job_dependencies(job_uuid, depends_on_job_uuid) VALUES(?, ?)",
        ("blocked", "parent"),
    )
    conn.commit()

    result = acquire_job(
        conn,
        command=_command(),
        counters_enabled=False,
        now=NOW,
    )

    assert result.row is not None
    assert int(result.row["id"]) == eligible_id


def test_sqlite_acquire_enforces_max_inflight_quota(conn: sqlite3.Connection) -> None:
    _insert_job(
        conn,
        uuid="active",
        status="processing",
        leased_until="2026-01-02 13:00:00",
    )
    queued_id = _insert_job(conn, uuid="queued")

    result = acquire_job(
        conn,
        command=_command(owner_user_id="owner", max_inflight_quota=1),
        counters_enabled=False,
        now=NOW,
    )

    assert result.no_transition_reason is NoTransitionReason.NO_ELIGIBLE_JOB
    assert conn.execute("SELECT status FROM jobs WHERE id = ?", (queued_id,)).fetchone()[0] == "queued"


def test_sqlite_acquire_ignores_expired_processing_lease_for_quota(conn: sqlite3.Connection) -> None:
    _insert_job(
        conn,
        uuid="expired",
        status="processing",
        leased_until="2026-01-02 11:00:00",
    )
    queued_id = _insert_job(conn, uuid="queued")

    result = acquire_job(
        conn,
        command=_command(owner_user_id="owner", max_inflight_quota=1),
        counters_enabled=False,
        now=NOW,
    )

    assert result.row is not None
    assert int(result.row["id"]) == queued_id


@pytest.mark.parametrize("single_update", [False, True], ids=["two-step", "single-update"])
def test_sqlite_acquire_moves_ready_counter_to_processing(
    conn: sqlite3.Connection,
    single_update: bool,
) -> None:
    _insert_job(conn, uuid=f"counter-{single_update}")
    conn.execute(
        "INSERT INTO job_counters(domain, queue, job_type, ready_count, scheduled_count, "
        "processing_count, quarantined_count) VALUES('acquire', 'default', 'work', 1, 0, 0, 0)"
    )
    conn.commit()

    acquire_job(
        conn,
        command=_command(single_update=single_update),
        counters_enabled=True,
        now=NOW,
    )

    counter = conn.execute(
        "SELECT ready_count, scheduled_count, processing_count FROM job_counters "
        "WHERE domain='acquire' AND queue='default' AND job_type='work'"
    ).fetchone()
    assert tuple(counter) == (0, 0, 1)
