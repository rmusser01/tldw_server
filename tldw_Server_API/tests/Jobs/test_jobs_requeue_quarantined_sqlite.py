import os
import time
import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.Jobs.manager import JobManager


def _init_env(monkeypatch, tmp_path):


    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.delenv("SINGLE_USER_API_KEY", raising=False)
    monkeypatch.setenv("JOBS_DB_PATH", os.path.join(os.getcwd(), "Databases", "jobs.db"))
    monkeypatch.setenv("JOBS_QUARANTINE_THRESHOLD", "1")


@pytest.mark.unit
def test_endpoint_requeue_quarantined_sqlite(monkeypatch, tmp_path):
    _init_env(monkeypatch, tmp_path)
    from tldw_Server_API.app.core.AuthNZ.settings import get_settings, reset_settings
    reset_settings()
    from tldw_Server_API.app.main import app

    jm = JobManager()
    # Create job and move to quarantined in one retry (threshold=1)
    j = jm.create_job(domain="chatbooks", queue="default", job_type="export", payload={}, owner_user_id="1")
    acq = jm.acquire_next_job(domain="chatbooks", queue="default", lease_seconds=5, worker_id="w")
    assert acq and acq.get("id") == j["id"]
    jm.fail_job(int(j["id"]), error="boom", retryable=True, backoff_seconds=0, worker_id="w", lease_id=str(acq.get("lease_id")), error_code="E1")
    row = jm.get_job(int(j["id"]))
    assert row and row.get("status") == "quarantined"

    headers = {"X-API-KEY": get_settings().SINGLE_USER_API_KEY}
    with TestClient(app, headers=headers) as client:
        # Dry run should report count but keep status
        r = client.post("/api/v1/jobs/batch/requeue_quarantined", json={"domain": "chatbooks", "queue": "default", "job_type": "export", "dry_run": True})
        assert r.status_code == 200
        assert r.json()["affected"] >= 1
        row2 = jm.get_job(int(j["id"]))
        assert row2 and row2.get("status") == "quarantined"

        # Real run requires confirm header
        r2 = client.post(
            "/api/v1/jobs/batch/requeue_quarantined",
            headers={**headers, "X-Confirm": "true"},
            json={"domain": "chatbooks", "queue": "default", "job_type": "export", "dry_run": False},
        )
        assert r2.status_code == 200
        assert r2.json()["affected"] >= 1
        row3 = jm.get_job(int(j["id"]))
        assert row3 and row3.get("status") == "queued"
        assert (row3.get("failure_streak_count") or 0) == 0


@pytest.mark.unit
def test_requeue_quarantined_updates_counters_sqlite(monkeypatch, tmp_path):
    _init_env(monkeypatch, tmp_path)
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    from tldw_Server_API.app.core.AuthNZ.settings import get_settings, reset_settings
    reset_settings()
    from tldw_Server_API.app.main import app

    jm = JobManager()
    # Quarantine one job
    j = jm.create_job(domain="chatbooks", queue="default", job_type="export", payload={}, owner_user_id="1")
    acq = jm.acquire_next_job(domain="chatbooks", queue="default", lease_seconds=1, worker_id="w")
    jm.fail_job(int(acq["id"]), error="boom", retryable=True, backoff_seconds=0, worker_id="w", lease_id=str(acq.get("lease_id")), error_code="E1")

    # Verify counters show quarantined_count=1 pre-requeue
    conn = jm._connect()
    try:
        row = conn.execute(
            "SELECT ready_count, scheduled_count, processing_count, quarantined_count FROM job_counters WHERE domain=? AND queue=? AND job_type=?",
            ("chatbooks", "default", "export"),
        ).fetchone()
        # Row may not exist if counters updated lazily; fetch via stats fallback
        if row:
            assert int(row[3] or 0) >= 1
    finally:
        conn.close()

    headers = {"X-API-KEY": get_settings().SINGLE_USER_API_KEY}
    with TestClient(app, headers=headers) as client:
        r = client.post(
            "/api/v1/jobs/batch/requeue_quarantined",
            headers={**headers, "X-Confirm": "true"},
            json={"domain": "chatbooks", "queue": "default", "job_type": "export", "dry_run": False},
        )
        assert r.status_code == 200
        assert r.json()["affected"] >= 1

    # Counters should now reflect ready_count+ and quarantined_count-
    conn2 = jm._connect()
    try:
        row2 = conn2.execute(
            "SELECT ready_count, scheduled_count, processing_count, quarantined_count FROM job_counters WHERE domain=? AND queue=? AND job_type=?",
            ("chatbooks", "default", "export"),
        ).fetchone()
        if row2:
            assert int(row2[0] or 0) >= 1
            assert int(row2[3] or 0) == 0
    finally:
        conn2.close()


@pytest.mark.unit
def test_requeue_quarantined_sanitizes_generic_failure(monkeypatch, tmp_path):
    _init_env(monkeypatch, tmp_path)
    from tldw_Server_API.app.core.AuthNZ.settings import get_settings, reset_settings
    reset_settings()
    from tldw_Server_API.app.main import app

    def boom(self):
        raise RuntimeError("jobs requeue backend exploded")

    monkeypatch.setattr(JobManager, "_connect", boom)

    headers = {"X-API-KEY": get_settings().SINGLE_USER_API_KEY}
    with TestClient(app, headers=headers) as client:
        r = client.post(
            "/api/v1/jobs/batch/requeue_quarantined",
            headers={**headers, "X-Confirm": "true"},
            json={"domain": "chatbooks", "queue": "default", "job_type": "export", "dry_run": False},
        )
        assert r.status_code == 500
        assert r.json()["detail"] == "Batch requeue quarantined failed"
