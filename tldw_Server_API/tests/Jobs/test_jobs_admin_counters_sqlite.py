import os
from datetime import datetime, timedelta

from fastapi.testclient import TestClient

from tldw_Server_API.app.core.Jobs.manager import JobManager


def _env(monkeypatch, tmp_path):


    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("JOBS_DB_PATH", os.path.join(os.getcwd(), "Databases", "jobs.db"))
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    # Debounce off for deterministic gauges
    monkeypatch.setenv("JOBS_GAUGES_DEBOUNCE_MS", "0")
    # Disable background workers that can race with tests
    monkeypatch.setenv("CHATBOOKS_CORE_WORKER_ENABLED", "false")
    monkeypatch.setenv("JOBS_METRICS_GAUGES_ENABLED", "false")
    monkeypatch.setenv("JOBS_METRICS_RECONCILE_ENABLE", "false")
    monkeypatch.setenv("AUDIO_JOBS_WORKER_ENABLED", "false")
    monkeypatch.setenv("EMBEDDINGS_REEMBED_WORKER_ENABLED", "false")
    monkeypatch.setenv("JOBS_WEBHOOKS_ENABLED", "false")
    monkeypatch.setenv("WORKFLOWS_WEBHOOK_DLQ_ENABLED", "false")
    monkeypatch.setenv("WORKFLOWS_ARTIFACT_GC_ENABLED", "false")
    monkeypatch.setenv("WORKFLOWS_DB_MAINTENANCE_ENABLED", "false")
    # Skip privilege catalog validation to avoid requiring config file in unit tests
    monkeypatch.setenv("PRIVILEGE_METADATA_VALIDATE_ON_STARTUP", "0")
    # Minimize startup
    monkeypatch.setenv("MINIMAL_TEST_APP", "1")


def _get_api(app):


    from tldw_Server_API.app.core.AuthNZ.settings import get_settings
    return {"X-API-KEY": get_settings().SINGLE_USER_API_KEY}


def _stats(client, domain="chatbooks", queue="default", job_type="export"):


    r = client.get("/api/v1/jobs/stats", params={"domain": domain, "queue": queue, "job_type": job_type})
    assert r.status_code == 200
    rows = r.json(); assert len(rows) == 1
    return rows[0]


def test_batch_cancel_updates_counters_and_gauges(monkeypatch, tmp_path):


    _env(monkeypatch, tmp_path)
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    reset_settings()
    from tldw_Server_API.app.main import app
    jm = JobManager()
    domain = "chatbooks"; queue = "default"; jt = "export"
    # Create one ready, one scheduled, and one processing
    first_ready = jm.create_job(domain=domain, queue=queue, job_type=jt, payload={}, owner_user_id="1")
    jm.create_job(domain=domain, queue=queue, job_type=jt, payload={}, owner_user_id="1", available_at=datetime.utcnow() + timedelta(seconds=60))
    acq_target = jm.create_job(domain=domain, queue=queue, job_type=jt, payload={}, owner_user_id="1")
    acq = jm.acquire_next_job(domain=domain, queue=queue, lease_seconds=30, worker_id="w")
    # FIFO by created_at: first ready job is acquired first
    assert acq and acq["id"] == first_ready["id"]
    headers = _get_api(app)
    with TestClient(app, headers=headers) as client:
        # Sanity stats before cancel
        s0 = _stats(client, domain, queue, jt)
        assert s0["queued"] >= 1 and s0["processing"] == 1
        r = client.post("/api/v1/jobs/batch/cancel", json={"domain": domain, "queue": queue, "job_type": jt, "dry_run": False}, headers={**headers, "X-Confirm": "true"})
        assert r.status_code == 200
        s1 = _stats(client, domain, queue, jt)
        assert s1["queued"] == 0 and s1["processing"] == 0
    # Verify counters table reflects zeros
    conn = jm._connect()
    try:
        row = conn.execute("SELECT ready_count, scheduled_count, processing_count FROM job_counters WHERE domain=? AND queue=? AND job_type=?", (domain, queue, jt)).fetchone()
        assert row is not None
        assert int(row[0]) == 0 and int(row[1]) == 0 and int(row[2]) == 0
    finally:
        conn.close()


def test_complete_queued_updates_counters(monkeypatch, tmp_path):


    _env(monkeypatch, tmp_path)
    jm = JobManager()
    domain = "chatbooks"; queue = "default"; jt = "export"

    ready = jm.create_job(domain=domain, queue=queue, job_type=jt, payload={}, owner_user_id="1")
    scheduled = jm.create_job(
        domain=domain,
        queue=queue,
        job_type=jt,
        payload={},
        owner_user_id="1",
        available_at=datetime.utcnow() + timedelta(seconds=60),
    )

    conn = jm._connect()
    try:
        row = conn.execute(
            "SELECT ready_count, scheduled_count, processing_count FROM job_counters WHERE domain=? AND queue=? AND job_type=?",
            (domain, queue, jt),
        ).fetchone()
        assert row is not None
        assert int(row[0]) == 1 and int(row[1]) == 1 and int(row[2]) == 0
    finally:
        conn.close()

    assert jm.complete_job(int(ready["id"]), result={}, enforce=False)
    assert jm.complete_job(int(scheduled["id"]), result={}, enforce=False)

    conn = jm._connect()
    try:
        row = conn.execute(
            "SELECT ready_count, scheduled_count, processing_count FROM job_counters WHERE domain=? AND queue=? AND job_type=?",
            (domain, queue, jt),
        ).fetchone()
        assert row is not None
        assert int(row[0]) == 0 and int(row[1]) == 0 and int(row[2]) == 0
    finally:
        conn.close()


def test_batch_reschedule_moves_ready_to_scheduled(monkeypatch, tmp_path):


    _env(monkeypatch, tmp_path)
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    reset_settings()
    from tldw_Server_API.app.main import app
    jm = JobManager()
    domain = "chatbooks"; queue = "default"; jt = "export"
    for _ in range(3):
        jm.create_job(domain=domain, queue=queue, job_type=jt, payload={}, owner_user_id="1")
    headers = _get_api(app)
    with TestClient(app, headers=headers) as client:
        r = client.post("/api/v1/jobs/batch/reschedule", json={"domain": domain, "queue": queue, "job_type": jt, "delay_seconds": 30, "dry_run": False}, headers={**headers, "X-Confirm": "true"})
        assert r.status_code == 200
        s = _stats(client, domain, queue, jt)
        # All moved to scheduled from ready, so queued immediate is 0
        assert s["queued"] == 0
    # Check counters scheduled increased
    conn = jm._connect()
    try:
        row = conn.execute("SELECT ready_count, scheduled_count FROM job_counters WHERE domain=? AND queue=? AND job_type=?", (domain, queue, jt)).fetchone()
        assert row is not None
        assert int(row[0]) == 0 and int(row[1]) >= 3
    finally:
        conn.close()


def test_batch_cancel_sanitizes_generic_failure(monkeypatch, tmp_path):
    _env(monkeypatch, tmp_path)
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    reset_settings()
    from tldw_Server_API.app.main import app

    def boom(self):
        raise RuntimeError("jobs batch cancel backend exploded")

    monkeypatch.setattr(JobManager, "_connect", boom)

    headers = _get_api(app)
    with TestClient(app, headers=headers) as client:
        r = client.post(
            "/api/v1/jobs/batch/cancel",
            json={"domain": "chatbooks", "queue": "default", "job_type": "export", "dry_run": False},
            headers={**headers, "X-Confirm": "true"},
        )
        assert r.status_code == 500
        assert r.json()["detail"] == "Batch cancel failed"


def test_batch_reschedule_sanitizes_generic_failure(monkeypatch, tmp_path):
    _env(monkeypatch, tmp_path)
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    reset_settings()
    from tldw_Server_API.app.main import app

    def boom(self):
        raise RuntimeError("jobs batch reschedule backend exploded")

    monkeypatch.setattr(JobManager, "_connect", boom)

    headers = _get_api(app)
    with TestClient(app, headers=headers) as client:
        r = client.post(
            "/api/v1/jobs/batch/reschedule",
            json={"domain": "chatbooks", "queue": "default", "job_type": "export", "delay_seconds": 30, "dry_run": False},
            headers={**headers, "X-Confirm": "true"},
        )
        assert r.status_code == 500
        assert r.json()["detail"] == "Batch reschedule failed"


def test_batch_requeue_quarantined_adjusts_counters(monkeypatch, tmp_path):


    _env(monkeypatch, tmp_path)
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    reset_settings()
    from tldw_Server_API.app.main import app
    jm = JobManager()
    domain = "chatbooks"; queue = "default"; jt = "export"
    # Quarantine one job
    j = jm.create_job(domain=domain, queue=queue, job_type=jt, payload={}, owner_user_id="1")
    acq = jm.acquire_next_job(domain=domain, queue=queue, lease_seconds=30, worker_id="w")
    token = str(acq.get("lease_id"))
    jm.fail_job(int(j["id"]), error="e1", retryable=True, worker_id="w", lease_id=str(acq.get("lease_id")), completion_token=token)
    acq2 = jm.acquire_next_job(domain=domain, queue=queue, lease_seconds=30, worker_id="w")
    token2 = str(acq2.get("lease_id"))
    jm.fail_job(int(j["id"]), error="e1", retryable=True, worker_id="w", lease_id=str(acq2.get("lease_id")), completion_token=token2)
    headers = _get_api(app)
    with TestClient(app, headers=headers) as client:
        r = client.post("/api/v1/jobs/batch/requeue-quarantined", json={"domain": domain, "queue": queue, "job_type": jt, "dry_run": False}, headers={**headers, "X-Confirm": "true"})
        assert r.status_code == 200
        s = _stats(client, domain, queue, jt)
        assert s["quarantined"] == 0 and s["queued"] >= 1
    conn = jm._connect()
    try:
        row = conn.execute("SELECT ready_count, quarantined_count FROM job_counters WHERE domain=? AND queue=? AND job_type=?", (domain, queue, jt)).fetchone()
        assert row is not None
        assert int(row[0]) >= 1 and int(row[1]) == 0
    finally:
        conn.close()
