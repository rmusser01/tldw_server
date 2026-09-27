import pytest
from fastapi.testclient import TestClient

psycopg = pytest.importorskip("psycopg")
pytestmark = pytest.mark.pg_jobs

from tldw_Server_API.app.core.Jobs.manager import JobManager


def test_endpoint_requeue_quarantined_postgres(monkeypatch, jobs_pg_dsn):


    monkeypatch.setenv("JOBS_DB_URL", jobs_pg_dsn)
    monkeypatch.setenv("JOBS_QUARANTINE_THRESHOLD", "1")
    from tldw_Server_API.app.core.Jobs.pg_migrations import ensure_jobs_tables_pg
    ensure_jobs_tables_pg(jobs_pg_dsn)
    jm = JobManager(None, backend="postgres", db_url=jobs_pg_dsn)

    # Seed and quarantine
    j = jm.create_job(domain="chatbooks", queue="default", job_type="export", payload={}, owner_user_id="1")
    acq = jm.acquire_next_job(domain="chatbooks", queue="default", lease_seconds=5, worker_id="w")
    assert acq and acq.get("id") == j["id"]
    jm.fail_job(int(j["id"]), error="boom", retryable=True, backoff_seconds=1, worker_id="w", lease_id=str(acq.get("lease_id")), error_code="E1")
    row = jm.get_job(int(j["id"]))
    assert row and row.get("status") == "quarantined"

    from tldw_Server_API.app.core.AuthNZ.settings import get_settings, reset_settings
    reset_settings()
    from tldw_Server_API.app.main import app
    headers = {"X-API-KEY": get_settings().SINGLE_USER_API_KEY}
    with TestClient(app, headers=headers) as client:
        # Dry run
        r = client.post("/api/v1/jobs/batch/requeue_quarantined", json={"domain": "chatbooks", "queue": "default", "job_type": "export", "dry_run": True})
        assert r.status_code == 200, r.text
        assert r.json()["affected"] >= 1
        # Real run with confirm header
        r2 = client.post(
            "/api/v1/jobs/batch/requeue_quarantined",
            headers={**headers, "X-Confirm": "true"},
            json={"domain": "chatbooks", "queue": "default", "job_type": "export", "dry_run": False},
        )
        assert r2.status_code == 200, r2.text
        assert r2.json()["affected"] >= 1
        row2 = jm.get_job(int(j["id"]))
        assert row2 and row2.get("status") == "queued"
        assert row2.get("available_at") is None
        assert (row2.get("failure_streak_count") or 0) == 0


def test_requeue_quarantined_updates_counters_postgres(monkeypatch, jobs_pg_dsn):


    monkeypatch.setenv("JOBS_DB_URL", jobs_pg_dsn)
    monkeypatch.setenv("JOBS_QUARANTINE_THRESHOLD", "1")
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    from tldw_Server_API.app.core.Jobs.pg_migrations import ensure_jobs_tables_pg
    ensure_jobs_tables_pg(jobs_pg_dsn)
    jm = JobManager(None, backend="postgres", db_url=jobs_pg_dsn)

    # Quarantine a job
    j = jm.create_job(domain="chatbooks", queue="default", job_type="export", payload={}, owner_user_id="1")
    acq = jm.acquire_next_job(domain="chatbooks", queue="default", lease_seconds=1, worker_id="w")
    assert acq and acq.get("id") == j["id"]
    jm.fail_job(int(j["id"]), error="boom", retryable=True, backoff_seconds=1, worker_id="w", lease_id=str(acq.get("lease_id")), error_code="E1")

    # Pre-check counters: quarantined_count should be >=1
    conn = jm._connect()
    try:
        with jm._pg_cursor(conn) as cur:
            cur.execute(
                "SELECT ready_count, scheduled_count, processing_count, quarantined_count FROM job_counters WHERE domain=%s AND queue=%s AND job_type=%s",
                ("chatbooks", "default", "export"),
            )
            row = cur.fetchone()
            if row:
                assert int(row["quarantined_count"] or 0) >= 1
    finally:
        conn.close()

    from tldw_Server_API.app.core.AuthNZ.settings import get_settings, reset_settings
    reset_settings()
    from tldw_Server_API.app.main import app
    headers = {"X-API-KEY": get_settings().SINGLE_USER_API_KEY}
    from fastapi.testclient import TestClient
    with TestClient(app, headers=headers) as client:
        r = client.post(
            "/api/v1/jobs/batch/requeue_quarantined",
            headers={**headers, "X-Confirm": "true"},
            json={"domain": "chatbooks", "queue": "default", "job_type": "export", "dry_run": False},
        )
        assert r.status_code == 200, r.text
        assert r.json()["affected"] >= 1

    conn2 = jm._connect()
    try:
        with jm._pg_cursor(conn2) as cur:
            cur.execute(
                "SELECT ready_count, scheduled_count, processing_count, quarantined_count FROM job_counters WHERE domain=%s AND queue=%s AND job_type=%s",
                ("chatbooks", "default", "export"),
            )
            row2 = cur.fetchone()
            if row2:
                assert int(row2["ready_count"] or 0) >= 1
                assert int(row2["quarantined_count"] or 0) == 0
    finally:
        conn2.close()
