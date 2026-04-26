import os
import sqlite3
from datetime import datetime

import pytest

from tldw_Server_API.app.core.Jobs.migrations import ensure_jobs_tables
from tldw_Server_API.app.core.Jobs.manager import JobManager


def _parse_sqlite_ts(s: str) -> datetime:
    try:
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    except Exception:
        return datetime.fromisoformat(s)


class _FailJobEventsInsertConnection:
    def __init__(self, inner: sqlite3.Connection):
        self._inner = inner

    def __enter__(self):
        self._inner.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        return self._inner.__exit__(exc_type, exc, tb)

    def execute(self, sql: str, params=()):
        if "INSERT INTO job_events" in str(sql):
            raise sqlite3.OperationalError("forced job_events insert failure")
        return self._inner.execute(sql, params)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _count_jobs(db_path, domain: str, queue: str, job_type: str) -> int:
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE domain = ? AND queue = ? AND job_type = ?",
            (domain, queue, job_type),
        ).fetchone()
    return int(row[0] if row else 0)


def _count_job_created_events(db_path, domain: str, queue: str, job_type: str) -> int:
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            (
                "SELECT COUNT(*) FROM job_events "
                "WHERE domain = ? AND queue = ? AND job_type = ? AND event_type = 'job.created'"
            ),
            (domain, queue, job_type),
        ).fetchone()
    return int(row[0] if row else 0)


def test_acquire_with_transient_db_timeout_then_retry_sqlite(monkeypatch, tmp_path):


    db_path = tmp_path / "jobs.db"
    ensure_jobs_tables(db_path)
    jm = JobManager(db_path)
    jm.create_job(domain="chatbooks", queue="default", job_type="t", payload={}, owner_user_id="u")

    # Patch _connect to raise once to simulate transient timeout/lock
    orig = jm._connect
    called = {"n": 0}

    def flaky_connect():

        if called["n"] == 0:
            called["n"] += 1
            raise sqlite3.OperationalError("database is locked")
        return orig()

    jm._connect = flaky_connect  # type: ignore

    with pytest.raises(sqlite3.OperationalError):
        jm.acquire_next_job(domain="chatbooks", queue="default", lease_seconds=5, worker_id="w1")

    # Restore and retry
    jm._connect = orig  # type: ignore
    acq = jm.acquire_next_job(domain="chatbooks", queue="default", lease_seconds=5, worker_id="w1")
    assert acq and str(acq.get("status")) == "processing"


def test_complete_transient_error_then_idempotent_finalize_sqlite(monkeypatch, tmp_path):


    db_path = tmp_path / "jobs2.db"
    ensure_jobs_tables(db_path)
    jm = JobManager(db_path)
    j = jm.create_job(domain="ps", queue="default", job_type="t", payload={}, owner_user_id="u")
    acq = jm.acquire_next_job(domain="ps", queue="default", lease_seconds=5, worker_id="w1")
    lease_id = str(acq.get("lease_id"))

    # Fail once on complete
    orig = jm._connect
    called = {"n": 0}

    def flaky_connect():

        if called["n"] == 0:
            called["n"] += 1
            raise sqlite3.OperationalError("transient")
        return orig()

    jm._connect = flaky_connect  # type: ignore
    with pytest.raises(sqlite3.OperationalError):
        jm.complete_job(int(acq["id"]), result={"ok": True}, worker_id="w1", lease_id=lease_id, completion_token=lease_id)
    # Restore and finalize
    jm._connect = orig  # type: ignore
    ok = jm.complete_job(int(acq["id"]), result={"ok": True}, worker_id="w1", lease_id=lease_id, completion_token=lease_id)
    assert ok is True
    # Idempotent retry with same token returns True
    ok2 = jm.complete_job(int(acq["id"]), result={"ok": True}, worker_id="w1", lease_id=lease_id, completion_token=lease_id)
    assert ok2 is True


def test_renew_with_clock_skew_does_not_shrink_lease_sqlite(monkeypatch, tmp_path):


    db_path = tmp_path / "jobs3.db"
    ensure_jobs_tables(db_path)
    jm = JobManager(db_path)
    jm.create_job(domain="ps", queue="default", job_type="t", payload={}, owner_user_id="u")
    acq = jm.acquire_next_job(domain="ps", queue="default", lease_seconds=20, worker_id="w")
    row = jm.get_job(int(acq["id"]))
    before = _parse_sqlite_ts(row["leased_until"]) if isinstance(row["leased_until"], str) else row["leased_until"]

    # Move clock backwards and renew; leased_until should not move back
    # Capture current epoch from manager clock and subtract skew
    from time import time as _now
    skewed = int(_now()) - 3600
    monkeypatch.setenv("JOBS_TEST_NOW_EPOCH", str(skewed))
    ok = jm.renew_job_lease(int(acq["id"]), seconds=5)
    assert ok is True
    row2 = jm.get_job(int(acq["id"]))
    after = _parse_sqlite_ts(row2["leased_until"]) if isinstance(row2["leased_until"], str) else row2["leased_until"]
    assert after >= before


@pytest.mark.parametrize("idempotency_key", [None, "idem-k"])
def test_create_rolls_back_when_job_created_event_insert_fails_sqlite(tmp_path, idempotency_key):
    db_path = tmp_path / "jobs_create_fail.db"
    ensure_jobs_tables(db_path)
    jm = JobManager(db_path)

    orig_connect = jm._connect
    jm._connect = lambda: _FailJobEventsInsertConnection(orig_connect())  # type: ignore[method-assign]

    with pytest.raises(sqlite3.OperationalError, match="job_events insert failure"):
        jm.create_job(
            domain="ps",
            queue="default",
            job_type="event-fail",
            payload={"x": 1},
            owner_user_id="u",
            idempotency_key=idempotency_key,
        )

    if _count_jobs(db_path, "ps", "default", "event-fail") != 0:
        raise AssertionError("create_job should not commit a new row when job.created event insert fails")


def test_idempotent_existing_create_fails_when_job_created_event_insert_fails_sqlite(tmp_path):
    db_path = tmp_path / "jobs_create_fail_existing.db"
    ensure_jobs_tables(db_path)
    jm = JobManager(db_path)

    first = jm.create_job(
        domain="ps",
        queue="default",
        job_type="event-fail-existing",
        payload={"x": 1},
        owner_user_id="u",
        idempotency_key="stable-key",
    )
    if not first.get("id"):
        raise AssertionError("expected first idempotent create call to return a persisted job id")
    if _count_jobs(db_path, "ps", "default", "event-fail-existing") != 1:
        raise AssertionError("expected exactly one persisted row after initial idempotent create")

    orig_connect = jm._connect
    jm._connect = lambda: _FailJobEventsInsertConnection(orig_connect())  # type: ignore[method-assign]

    with pytest.raises(sqlite3.OperationalError, match="job_events insert failure"):
        jm.create_job(
            domain="ps",
            queue="default",
            job_type="event-fail-existing",
            payload={"x": 2},
            owner_user_id="u",
            idempotency_key="stable-key",
        )

    if _count_jobs(db_path, "ps", "default", "event-fail-existing") != 1:
        raise AssertionError("failing idempotent create-existing call must not add a new job row")


def test_create_failure_does_not_increment_created_metric_sqlite(monkeypatch, tmp_path):
    import tldw_Server_API.app.core.Jobs.manager as mgr

    calls = {"n": 0}

    def _inc(_labels):
        calls["n"] += 1

    monkeypatch.setattr(mgr, "increment_created", _inc)

    db_path = tmp_path / "jobs_create_fail_metrics.db"
    ensure_jobs_tables(db_path)
    jm = JobManager(db_path)

    orig_connect = jm._connect
    jm._connect = lambda: _FailJobEventsInsertConnection(orig_connect())  # type: ignore[method-assign]

    with pytest.raises(sqlite3.OperationalError, match="job_events insert failure"):
        jm.create_job(
            domain="ps",
            queue="default",
            job_type="event-fail-metrics",
            payload={"x": 1},
            owner_user_id="u",
        )

    if calls["n"] != 0:
        raise AssertionError("increment_created must not be called when create rolls back due to job_events failure")


@pytest.mark.parametrize(
    ("outbox_enabled", "expected_emit_calls", "expected_direct_audit_calls"),
    [
        (False, 1, 0),
        (True, 0, 1),
    ],
)
def test_idempotent_create_uses_exactly_one_audit_bridge_path_sqlite(
    monkeypatch,
    tmp_path,
    outbox_enabled,
    expected_emit_calls,
    expected_direct_audit_calls,
):
    import tldw_Server_API.app.core.Jobs.manager as mgr

    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "true" if outbox_enabled else "false")
    calls = {"emit_n": 0, "audit_n": 0}

    def _emit(_event_type, **_kwargs):
        calls["emit_n"] += 1

    def _audit(_event_type, **_kwargs):
        calls["audit_n"] += 1

    monkeypatch.setattr(mgr, "emit_job_event", _emit)
    monkeypatch.setattr(mgr, "submit_job_audit_event", _audit)

    mode = "outbox" if outbox_enabled else "non_outbox"
    db_path = tmp_path / f"jobs_create_idem_{mode}.db"
    ensure_jobs_tables(db_path)
    jm = JobManager(db_path)

    created = jm.create_job(
        domain="ps",
        queue="default",
        job_type=f"event-idem-{mode}",
        payload={"x": 1},
        owner_user_id="u",
        idempotency_key=f"stable-idem-{mode}",
    )

    if not created.get("id"):
        raise AssertionError("expected idempotent create to return a job row")
    if calls["emit_n"] != expected_emit_calls:
        raise AssertionError(
            f"expected emit_job_event calls={expected_emit_calls}, got {calls['emit_n']} for mode={mode}"
        )
    if calls["audit_n"] != expected_direct_audit_calls:
        raise AssertionError(
            f"expected direct submit_job_audit_event calls={expected_direct_audit_calls}, got {calls['audit_n']} for mode={mode}"
        )
    if _count_job_created_events(db_path, "ps", "default", f"event-idem-{mode}") != 1:
        raise AssertionError("expected exactly one transactional job.created record for idempotent create")
