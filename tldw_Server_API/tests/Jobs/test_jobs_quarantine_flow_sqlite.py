import os

from tldw_Server_API.app.core.Jobs.manager import JobManager


def _set_env(monkeypatch, tmp_path):


    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("AUTH_MODE", "single_user")
    db_path = os.path.join(os.getcwd(), "Databases", "jobs.db")
    monkeypatch.setenv("JOBS_DB_PATH", db_path)
    monkeypatch.setenv("JOBS_ENFORCE_LEASE_ACK", "true")
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    monkeypatch.setenv("JOBS_QUARANTINE_THRESHOLD", "2")


def test_quarantine_and_requeue_updates_counters(monkeypatch, tmp_path):


    _set_env(monkeypatch, tmp_path)
    jm = JobManager()
    # Seed one job and acquire
    j = jm.create_job(domain="chatbooks", queue="default", job_type="export", payload={}, owner_user_id="1")
    acq = jm.acquire_next_job(domain="chatbooks", queue="default", lease_seconds=30, worker_id="wq")
    assert acq and acq["id"] == j["id"]
    # Fail twice to trigger quarantine threshold
    token = str(acq.get("lease_id"))
    ok1 = jm.fail_job(int(j["id"]), error="e1", retryable=True, worker_id="wq", lease_id=str(acq.get("lease_id")), completion_token=token)
    assert ok1 is True
    # Re-acquire after backoff (TEST_MODE forces short/no delay)
    acq2 = jm.acquire_next_job(domain="chatbooks", queue="default", lease_seconds=30, worker_id="wq")
    assert acq2 and acq2["id"] == j["id"]
    token2 = str(acq2.get("lease_id"))
    ok2 = jm.fail_job(int(j["id"]), error="e1", retryable=True, worker_id="wq", lease_id=str(acq2.get("lease_id")), completion_token=token2)
    assert ok2 is True
    # Now job should be quarantined
    rows = jm.get_queue_stats(domain="chatbooks", queue="default", job_type="export")
    assert rows and rows[0]["quarantined"] >= 1
    # Requeue quarantined via admin helper: reuse endpoint logic by calling manager-side
    # emulate minimal batch requeue behavior: update status and counters
    conn = jm._connect()
    try:
        with conn:
            conn.execute(
                (
                    "UPDATE jobs SET status='queued', failure_streak_count=0, failure_streak_code=NULL, quarantined_at=NULL, "
                    "available_at=NULL, leased_until=NULL, worker_id=NULL, lease_id=NULL WHERE domain=? AND queue=? AND job_type=? AND status='quarantined'"
                ),
                ("chatbooks", "default", "export"),
            )
            conn.execute(
                (
                    "INSERT INTO job_counters(domain,queue,job_type,ready_count,scheduled_count,processing_count,quarantined_count) VALUES(?,?,?,?,0,0,0) "
                    "ON CONFLICT(domain,queue,job_type) DO UPDATE SET ready_count = ready_count + 1, quarantined_count = CASE WHEN (quarantined_count - 1) < 0 THEN 0 ELSE quarantined_count - 1 END, updated_at = DATETIME('now')"
                ),
                ("chatbooks", "default", "export", 0),
            )
    finally:
        try:
            conn.close()
        except Exception:
            _ = None
    rows2 = jm.get_queue_stats(domain="chatbooks", queue="default", job_type="export")
    assert rows2 and rows2[0]["queued"] >= 1 and rows2[0]["quarantined"] == 0
