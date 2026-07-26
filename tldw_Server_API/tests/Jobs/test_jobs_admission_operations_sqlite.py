"""Direct SQLite admission operation tests."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from tldw_Server_API.app.core.Jobs.migrations import ensure_jobs_tables
from tldw_Server_API.app.core.Jobs.operations.contracts import (
    AdmissionRejectionReason,
    CreateJobCommand,
    OperationOutcome,
)
from tldw_Server_API.app.core.Jobs.operations.sqlite.admission import create_job_admission


class _FailJobEventsInsertConnection:
    """Connection wrapper that fails transactional job_events inserts."""

    def __init__(self, inner: sqlite3.Connection):
        self._inner = inner

    def __enter__(self) -> _FailJobEventsInsertConnection:
        self._inner.__enter__()
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: Any) -> bool | None:
        return self._inner.__exit__(exc_type, exc, tb)

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Cursor:
        if "INSERT INTO job_events" in str(sql):
            raise sqlite3.OperationalError("forced job_events insert failure")
        return self._inner.execute(sql, params)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _FailJobCountersConnection:
    """Connection wrapper that fails transactional job_counters upserts."""

    def __init__(self, inner: sqlite3.Connection):
        self._inner = inner

    def __enter__(self) -> _FailJobCountersConnection:
        self._inner.__enter__()
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: Any) -> bool | None:
        return self._inner.__exit__(exc_type, exc, tb)

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Cursor:
        if "INSERT INTO job_counters" in str(sql):
            raise sqlite3.OperationalError("forced job_counters upsert failure")
        return self._inner.execute(sql, params)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _FailQuotaSelectConnection:
    """Connection wrapper that fails quota count queries before insert."""

    def __init__(self, inner: sqlite3.Connection):
        self._inner = inner

    def __enter__(self) -> _FailQuotaSelectConnection:
        self._inner.__enter__()
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: Any) -> bool | None:
        return self._inner.__exit__(exc_type, exc, tb)

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Cursor:
        if "SELECT COUNT(*) FROM jobs WHERE domain=?" in str(sql):
            raise sqlite3.OperationalError("quota read failed")
        return self._inner.execute(sql, params)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _open_jobs_db(tmp_path: Path, name: str = "jobs.db") -> tuple[Path, sqlite3.Connection]:
    db_path = ensure_jobs_tables(tmp_path / name)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return db_path, conn


def _command(
    *,
    job_type: str = "insert",
    idempotency_key: str | None = None,
    request_id: str = "req-1",
    trace_id: str = "trace-1",
) -> CreateJobCommand:
    return CreateJobCommand(
        domain="admission",
        queue="default",
        job_type=job_type,
        payload={"x": 1},
        owner_user_id="u1",
        idempotency_key=idempotency_key,
        priority=5,
        max_retries=3,
        request_id=request_id,
        trace_id=trace_id,
    )


def _created_events(conn: sqlite3.Connection, job_type: str) -> list[sqlite3.Row]:
    return list(
        conn.execute(
            (
                "SELECT * FROM job_events "
                "WHERE domain = ? AND queue = ? AND job_type = ? AND event_type = 'job.created' "
                "ORDER BY id"
            ),
            ("admission", "default", job_type),
        )
    )


def test_sqlite_admission_inserts_job_event_and_counter(tmp_path: Path) -> None:
    _db_path, conn = _open_jobs_db(tmp_path)
    with conn:
        result = create_job_admission(
            conn,
            command=_command(),
            uuid_value="uuid-insert",
            now=datetime(2026, 1, 1, tzinfo=timezone.utc),
            max_queued_quota=0,
            submits_per_minute_quota=0,
            counters_enabled=True,
        )

    assert result.outcome is OperationOutcome.APPLIED
    assert result.inserted is True
    assert result.row is not None
    assert result.row["status"] == "queued"
    assert result.row["request_id"] == "req-1"
    assert result.row["trace_id"] == "trace-1"

    events = _created_events(conn, "insert")
    assert len(events) == 1
    assert events[0]["request_id"] == "req-1"
    assert events[0]["trace_id"] == "trace-1"
    assert json.loads(events[0]["attrs_json"])["idempotent"] is False

    counter = conn.execute(
        "SELECT ready_count, scheduled_count FROM job_counters WHERE domain = ? AND queue = ? AND job_type = ?",
        ("admission", "default", "insert"),
    ).fetchone()
    assert dict(counter) == {"ready_count": 1, "scheduled_count": 0}


@pytest.mark.parametrize("idempotency_key", [None, "same"], ids=["plain", "idempotent"])
def test_sqlite_admission_counter_failure_rolls_back_job_and_event(
    tmp_path: Path,
    idempotency_key: str | None,
) -> None:
    _db_path, inner = _open_jobs_db(tmp_path)
    wrapped = _FailJobCountersConnection(inner)

    with pytest.raises(sqlite3.OperationalError, match="forced job_counters upsert failure"):
        create_job_admission(
            wrapped,
            command=_command(job_type="counter-fail", idempotency_key=idempotency_key),
            uuid_value="uuid-counter-fail",
            now=datetime(2026, 1, 1, tzinfo=timezone.utc),
            max_queued_quota=0,
            submits_per_minute_quota=0,
            counters_enabled=True,
        )

    job_count = inner.execute("SELECT COUNT(*) FROM jobs WHERE job_type = ?", ("counter-fail",)).fetchone()[0]
    assert job_count == 0
    assert _created_events(inner, "counter-fail") == []
    counter_count = inner.execute(
        "SELECT COUNT(*) FROM job_counters WHERE domain = ? AND queue = ? AND job_type = ?",
        ("admission", "default", "counter-fail"),
    ).fetchone()[0]
    assert counter_count == 0


def test_sqlite_admission_idempotent_existing_writes_replay_event_with_current_context(tmp_path: Path) -> None:
    _db_path, conn = _open_jobs_db(tmp_path)
    first = create_job_admission(
        conn,
        command=_command(job_type="idem", idempotency_key="same"),
        uuid_value="uuid-idem-1",
        now=datetime(2026, 1, 1, tzinfo=timezone.utc),
        max_queued_quota=0,
        submits_per_minute_quota=0,
        counters_enabled=True,
    )
    replay = create_job_admission(
        conn,
        command=_command(
            job_type="idem",
            idempotency_key="same",
            request_id="req-replay",
            trace_id="trace-replay",
        ),
        uuid_value="uuid-idem-2",
        now=datetime(2026, 1, 1, 0, 0, 5, tzinfo=timezone.utc),
        max_queued_quota=0,
        submits_per_minute_quota=0,
        counters_enabled=True,
    )

    assert first.outcome is OperationOutcome.APPLIED
    assert replay.outcome is OperationOutcome.NO_TRANSITION
    assert replay.inserted is False
    assert replay.row is not None
    assert replay.row["uuid"] == "uuid-idem-1"
    assert replay.row["request_id"] == "req-1"
    assert replay.row["trace_id"] == "trace-1"

    events = _created_events(conn, "idem")
    assert len(events) == 2
    assert events[0]["request_id"] == "req-1"
    assert events[0]["trace_id"] == "trace-1"
    assert json.loads(events[0]["attrs_json"])["idempotent"] is False
    assert events[1]["request_id"] == "req-replay"
    assert events[1]["trace_id"] == "trace-replay"
    assert json.loads(events[1]["attrs_json"])["idempotent"] is True
    assert replay.durable_events == (
        {
            "event_type": "job.created",
            "attrs": {"idempotent": True, "owner_user_id": "u1", "retry_count": 0},
            "request_id": "req-replay",
            "trace_id": "trace-replay",
            "job_id": first.row["id"],
            "domain": "admission",
            "queue": "default",
            "job_type": "idem",
            "owner_user_id": "u1",
        },
    )


def test_sqlite_admission_rejects_max_queued_quota(tmp_path: Path) -> None:
    _db_path, conn = _open_jobs_db(tmp_path)
    create_job_admission(
        conn,
        command=_command(job_type="quota"),
        uuid_value="uuid-quota-1",
        now=datetime(2026, 1, 1, tzinfo=timezone.utc),
        max_queued_quota=0,
        submits_per_minute_quota=0,
        counters_enabled=False,
    )

    result = create_job_admission(
        conn,
        command=_command(job_type="quota-two"),
        uuid_value="uuid-quota-2",
        now=datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
        max_queued_quota=1,
        submits_per_minute_quota=0,
        counters_enabled=False,
    )

    assert result.outcome is OperationOutcome.ADMISSION_REJECTED
    assert result.admission_rejection_reason is AdmissionRejectionReason.QUOTA_EXCEEDED
    assert result.message == "Quota exceeded: max queued per user/domain"


def test_sqlite_admission_rejects_submits_per_minute_quota(tmp_path: Path) -> None:
    _db_path, conn = _open_jobs_db(tmp_path)
    create_job_admission(
        conn,
        command=_command(job_type="spm"),
        uuid_value="uuid-spm-1",
        now=datetime(2026, 1, 1, tzinfo=timezone.utc),
        max_queued_quota=0,
        submits_per_minute_quota=0,
        counters_enabled=False,
    )

    result = create_job_admission(
        conn,
        command=_command(job_type="spm-two"),
        uuid_value="uuid-spm-2",
        now=datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
        max_queued_quota=0,
        submits_per_minute_quota=1,
        counters_enabled=False,
    )

    assert result.outcome is OperationOutcome.ADMISSION_REJECTED
    assert result.admission_rejection_reason is AdmissionRejectionReason.QUOTA_EXCEEDED
    assert result.message == "Quota exceeded: submits per minute"


def test_sqlite_admission_rolls_back_job_when_created_event_insert_fails(tmp_path: Path) -> None:
    db_path, inner = _open_jobs_db(tmp_path)
    wrapped = _FailJobEventsInsertConnection(inner)

    with pytest.raises(sqlite3.OperationalError, match="job_events insert failure"):
        create_job_admission(
            wrapped,
            command=_command(job_type="event-fail"),
            uuid_value="uuid-event-fail",
            now=datetime(2026, 1, 1, tzinfo=timezone.utc),
            max_queued_quota=0,
            submits_per_minute_quota=0,
            counters_enabled=True,
        )

    with sqlite3.connect(db_path) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE domain = ? AND queue = ? AND job_type = ?",
            ("admission", "default", "event-fail"),
        ).fetchone()[0]
        counter_count = conn.execute(
            "SELECT COUNT(*) FROM job_counters WHERE domain = ? AND queue = ? AND job_type = ?",
            ("admission", "default", "event-fail"),
        ).fetchone()[0]
    assert count == 0
    assert counter_count == 0


def test_sqlite_admission_rolls_back_when_quota_query_fails(tmp_path: Path) -> None:
    db_path, inner = _open_jobs_db(tmp_path)
    wrapped = _FailQuotaSelectConnection(inner)

    with pytest.raises(sqlite3.OperationalError, match="quota read failed"):
        create_job_admission(
            wrapped,
            command=_command(job_type="quota-fail"),
            uuid_value="uuid-quota-fail",
            now=datetime(2026, 1, 1, tzinfo=timezone.utc),
            max_queued_quota=1,
            submits_per_minute_quota=0,
            counters_enabled=False,
        )

    with sqlite3.connect(db_path) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE domain = ? AND queue = ? AND job_type = ?",
            ("admission", "default", "quota-fail"),
        ).fetchone()[0]
    assert count == 0
