from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tldw_Server_API.app.core.Jobs.migrations import ensure_jobs_tables
from tldw_Server_API.app.core.Jobs.manager import JobManager


@pytest.fixture()
def jobs_db(tmp_path):
    db_path = tmp_path / "jobs.db"
    ensure_jobs_tables(db_path)
    return db_path


def test_create_job_normalizes_timezone_aware_available_at_to_utc_sqlite(jobs_db):
    jm = JobManager(jobs_db)
    target_utc = (datetime.now(timezone.utc) + timedelta(minutes=30)).replace(microsecond=0)
    user_tz = timezone(timedelta(hours=2))
    available_at = target_utc.astimezone(user_tz)

    job = jm.create_job(
        domain="chatbooks",
        queue="default",
        job_type="export",
        payload={},
        owner_user_id="1",
        available_at=available_at,
    )

    stored = jm.get_job(int(job["id"]))
    assert stored is not None
    stored_available_at = datetime.fromisoformat(str(stored["available_at"]))
    expected_utc_naive = target_utc.replace(tzinfo=None)
    assert stored_available_at == expected_utc_naive


@pytest.mark.parametrize("terminal_status", ["completed", "failed", "quarantined"])
def test_finalize_cancelled_does_not_overwrite_terminal_states_sqlite(jobs_db, terminal_status):
    jm = JobManager(jobs_db)
    job = jm.create_job(
        domain="chatbooks",
        queue="default",
        job_type="export",
        payload={},
        owner_user_id="1",
    )

    if terminal_status == "completed":
        acquired = jm.acquire_next_job(domain="chatbooks", queue="default", lease_seconds=30, worker_id="w")
        assert acquired is not None
        assert jm.complete_job(int(acquired["id"]), worker_id="w", lease_id=acquired.get("lease_id"), enforce=False)
    elif terminal_status == "failed":
        acquired = jm.acquire_next_job(domain="chatbooks", queue="default", lease_seconds=30, worker_id="w")
        assert acquired is not None
        assert jm.fail_job(
            int(acquired["id"]),
            error="boom",
            retryable=False,
            worker_id="w",
            lease_id=acquired.get("lease_id"),
            enforce=False,
        )
    else:
        conn = jm._connect()
        try:
            with conn:
                conn.execute(
                    "UPDATE jobs SET status='quarantined', quarantined_at=DATETIME('now') WHERE id=?",
                    (int(job["id"]),),
                )
        finally:
            conn.close()

    before = jm.get_job(int(job["id"]))
    assert before is not None
    assert before["status"] == terminal_status

    changed = jm.finalize_cancelled(
        int(job["id"]),
        reason="forced",
        expected_uuid=str(job["uuid"]),
    )
    assert changed is False

    after = jm.get_job(int(job["id"]))
    assert after is not None
    assert after["status"] == terminal_status


def test_finalize_cancelled_processing_updates_counters_sqlite(jobs_db, monkeypatch):
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    jm = JobManager(jobs_db)
    job = jm.create_job(
        domain="chatbooks",
        queue="default",
        job_type="export",
        payload={},
        owner_user_id="1",
    )
    acquired = jm.acquire_next_job(domain="chatbooks", queue="default", lease_seconds=30, worker_id="w")
    assert acquired is not None
    assert int(acquired["id"]) == int(job["id"])

    assert jm.finalize_cancelled(
        int(job["id"]),
        reason="cancel requested during processing",
        expected_uuid=str(acquired["uuid"]),
        worker_id=str(acquired["worker_id"]),
        lease_id=str(acquired["lease_id"]),
    )

    conn = jm._connect()
    try:
        row = conn.execute(
            "SELECT ready_count, scheduled_count, processing_count, quarantined_count "
            "FROM job_counters WHERE domain=? AND queue=? AND job_type=?",
            ("chatbooks", "default", "export"),
        ).fetchone()
    finally:
        conn.close()
    assert row is not None
    assert int(row[2]) == 0


def test_retry_now_failed_updates_ready_counter_sqlite(jobs_db, monkeypatch):
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    jm = JobManager(jobs_db)
    job = jm.create_job(
        domain="chatbooks",
        queue="default",
        job_type="export",
        payload={},
        owner_user_id="1",
        max_retries=3,
    )
    acquired = jm.acquire_next_job(domain="chatbooks", queue="default", lease_seconds=30, worker_id="w")
    assert acquired is not None
    assert jm.fail_job(
        int(acquired["id"]),
        error="terminal",
        retryable=False,
        worker_id="w",
        lease_id=acquired.get("lease_id"),
        enforce=False,
    )

    moved = jm.retry_now_jobs(job_id=int(job["id"]), only_failed=True, dry_run=False)
    assert moved == 1

    conn = jm._connect()
    try:
        row = conn.execute(
            "SELECT ready_count, scheduled_count, processing_count, quarantined_count "
            "FROM job_counters WHERE domain=? AND queue=? AND job_type=?",
            ("chatbooks", "default", "export"),
        ).fetchone()
    finally:
        conn.close()
    assert row is not None
    assert int(row[0]) == 1
    assert int(row[1]) == 0
    assert int(row[2]) == 0


def test_complete_job_sla_duration_breach_records_attachment_sqlite(jobs_db):
    jm = JobManager(jobs_db)
    jm.upsert_sla_policy(
        domain="chatbooks",
        queue="default",
        job_type="slow",
        max_duration_seconds=0,
        enabled=True,
    )
    job = jm.create_job(
        domain="chatbooks",
        queue="default",
        job_type="slow",
        payload={},
        owner_user_id="1",
    )
    acquired = jm.acquire_next_job(domain="chatbooks", queue="default", lease_seconds=30, worker_id="w")
    assert acquired is not None

    conn = jm._connect()
    try:
        with conn:
            conn.execute(
                "UPDATE jobs SET started_at = DATETIME('now','-3600 seconds') WHERE id = ?",
                (int(acquired["id"]),),
            )
    finally:
        conn.close()

    assert jm.complete_job(
        int(acquired["id"]),
        worker_id="w",
        lease_id=acquired.get("lease_id"),
        enforce=False,
    )

    attachments = jm.list_job_attachments(int(acquired["id"]))
    assert any(
        str(item.get("kind")) == "tag" and "SLA breach: duration" in str(item.get("content_text") or "")
        for item in attachments
    )
