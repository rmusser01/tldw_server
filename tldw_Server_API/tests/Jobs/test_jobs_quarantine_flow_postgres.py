import os
import pytest

psycopg = pytest.importorskip("psycopg")

from tldw_Server_API.app.core.Jobs.manager import JobManager


pytestmark = pytest.mark.pg_jobs


def _require_pg(monkeypatch):


    db_url = os.getenv("JOBS_DB_URL", "")
    if not db_url or not db_url.startswith("postgres"):
        pytest.skip("JOBS_DB_URL not configured for Postgres tests")
    monkeypatch.setenv("JOBS_ENFORCE_LEASE_ACK", "true")
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    monkeypatch.setenv("JOBS_QUARANTINE_THRESHOLD", "2")
    monkeypatch.setenv("JOBS_PG_SINGLE_UPDATE_ACQUIRE", "true")


def test_pg_quarantine_and_requeue_updates_counters(monkeypatch):


    _require_pg(monkeypatch)
    jm = JobManager(backend="postgres", db_url=os.getenv("JOBS_DB_URL"))
    j = jm.create_job(domain="chatbooks", queue="default", job_type="export", payload={}, owner_user_id="1")
    acq = jm.acquire_next_job(domain="chatbooks", queue="default", lease_seconds=30, worker_id="wpg")
    assert acq and acq["id"] == j["id"]
    token1 = str(acq.get("lease_id"))
    ok1 = jm.fail_job(int(j["id"]), error="boom", retryable=True, worker_id="wpg", lease_id=str(acq.get("lease_id")), completion_token=token1)
    assert ok1 is True
    # Re-acquire after backoff or immediate in TEST_MODE (PG path doesn't force TEST_MODE here but delay is small with low retry_count)
    acq2 = jm.acquire_next_job(domain="chatbooks", queue="default", lease_seconds=30, worker_id="wpg")
    assert acq2 and acq2["id"] == j["id"]
    token2 = str(acq2.get("lease_id"))
    ok2 = jm.fail_job(int(j["id"]), error="boom", retryable=True, worker_id="wpg", lease_id=str(acq2.get("lease_id")), completion_token=token2)
    assert ok2 is True
    # Should now be quarantined
    rows = jm.get_queue_stats(domain="chatbooks", queue="default", job_type="export")
    assert rows and rows[0]["quarantined"] >= 1
    # Requeue quarantined (PG path mirrors admin endpoint logic minimal)
    conn = jm._connect()
    try:
        with conn:
            with jm._pg_cursor(conn) as cur:
                cur.execute(
                    (
                        "UPDATE jobs SET status='queued', failure_streak_count = 0, failure_streak_code = NULL, quarantined_at = NULL, "
                        "available_at = NULL, leased_until = NULL, worker_id = NULL, lease_id = NULL, completion_token = NULL WHERE domain=%s AND queue=%s AND job_type=%s AND status='quarantined'"
                    ),
                    ("chatbooks", "default", "export"),
                )
                cur.execute(
                    (
                        "INSERT INTO job_counters(domain,queue,job_type,ready_count,scheduled_count,processing_count,quarantined_count) VALUES(%s,%s,%s,0,0,0,0) "
                        "ON CONFLICT(domain,queue,job_type) DO UPDATE SET ready_count = job_counters.ready_count + 1, quarantined_count = GREATEST(job_counters.quarantined_count - 1, 0), updated_at = NOW()"
                    ),
                    ("chatbooks", "default", "export"),
                )
    finally:
        try:
            conn.close()
        except Exception:
            _ = None
    rows2 = jm.get_queue_stats(domain="chatbooks", queue="default", job_type="export")
    assert rows2 and rows2[0]["queued"] >= 1 and rows2[0]["quarantined"] == 0
