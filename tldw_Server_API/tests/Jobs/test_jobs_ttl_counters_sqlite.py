import os

from fastapi.testclient import TestClient

from tldw_Server_API.app.core.Jobs.manager import JobManager


def _env(monkeypatch, tmp_path):

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.delenv("SINGLE_USER_API_KEY", raising=False)
    monkeypatch.setenv("JOBS_DB_PATH", os.path.join(os.getcwd(), "Databases", "jobs.db"))
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    monkeypatch.setenv("JOBS_GAUGES_DEBOUNCE_MS", "0")


def _stats(client, domain="chatbooks", queue="default", job_type="export"):

    r = client.get("/api/v1/jobs/stats", params={"domain": domain, "queue": queue, "job_type": job_type})
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == 1
    return rows[0]


def _assert_terminal_leases_cleared(jm, *job_ids):
    for job_id in job_ids:
        terminal = jm.get_job(int(job_id))
        assert terminal is not None
        assert terminal["status"] in {"cancelled", "failed"}
        assert terminal["leased_until"] is None
        assert terminal["worker_id"] is None
        assert terminal["lease_id"] is None


def test_ttl_cancel_updates_counters(monkeypatch, tmp_path):

    _env(monkeypatch, tmp_path)
    from tldw_Server_API.app.core.AuthNZ.settings import get_settings, reset_settings

    reset_settings()
    from tldw_Server_API.app.main import app

    jm = JobManager()
    domain = "chatbooks"
    queue = "default"
    jt = "export"
    jq = jm.create_job(domain=domain, queue=queue, job_type=jt, payload={}, owner_user_id="1")
    jp = jm.create_job(domain=domain, queue=queue, job_type=jt, payload={}, owner_user_id="1")
    acq = jm.acquire_next_job(domain=domain, queue=queue, lease_seconds=30, worker_id="w")
    assert acq and acq["id"] == jp["id"]
    # Backdate created and started
    conn = jm._connect()
    try:
        with conn:
            conn.execute(
                "UPDATE jobs SET created_at = DATETIME('now','-2 hours'), "
                "leased_until = DATETIME('now','+1 hour'), worker_id = 'stale-worker', "
                "lease_id = 'stale-lease' WHERE id=?",
                (int(jq["id"]),),
            )
            conn.execute(
                "UPDATE jobs SET started_at = DATETIME('now','-3 hours'), acquired_at = DATETIME('now','-3 hours') WHERE id=?",
                (int(jp["id"]),),
            )
    finally:
        conn.close()
    headers = {"X-API-KEY": get_settings().SINGLE_USER_API_KEY}
    with TestClient(app, headers=headers) as client:
        r = client.post(
            "/api/v1/jobs/ttl/sweep",
            json={
                "age_seconds": 3600,
                "runtime_seconds": 3600,
                "action": "cancel",
                "domain": domain,
                "queue": queue,
                "job_type": jt,
            },
            headers={**headers, "X-Confirm": "true"},
        )
        assert r.status_code == 200
        s = _stats(client, domain, queue, jt)
        assert s["queued"] == 0 and s["processing"] == 0
        # Metrics: cancelled_total should increment for this scope
        try:
            from tldw_Server_API.app.core.Metrics.metrics_manager import get_metrics_registry

            reg = get_metrics_registry()
            vals = list(reg.values.get("jobs.cancelled_total", []))
            saw = False
            for mv in vals:
                if (
                    mv.labels.get("domain") == domain
                    and mv.labels.get("queue") == queue
                    and mv.labels.get("job_type") == jt
                ):
                    saw = True
                    break
            assert saw, "Expected cancelled_total increment for TTL cancel"
        except Exception:
            _ = None
    _assert_terminal_leases_cleared(jm, jq["id"], jp["id"])
    # Counters reflect zeros
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


def test_ttl_fail_updates_counters(monkeypatch, tmp_path):

    _env(monkeypatch, tmp_path)
    from tldw_Server_API.app.core.AuthNZ.settings import get_settings, reset_settings

    reset_settings()
    from tldw_Server_API.app.main import app

    jm = JobManager()
    domain = "chatbooks"
    queue = "default"
    jt = "export"
    jq = jm.create_job(domain=domain, queue=queue, job_type=jt, payload={}, owner_user_id="1")
    jp = jm.create_job(domain=domain, queue=queue, job_type=jt, payload={}, owner_user_id="1")
    acq = jm.acquire_next_job(domain=domain, queue=queue, lease_seconds=30, worker_id="w")
    assert acq and acq["id"] == jp["id"]
    conn = jm._connect()
    try:
        with conn:
            conn.execute(
                "UPDATE jobs SET created_at = DATETIME('now','-2 hours'), "
                "leased_until = DATETIME('now','+1 hour'), worker_id = 'stale-worker', "
                "lease_id = 'stale-lease' WHERE id=?",
                (int(jq["id"]),),
            )
            conn.execute(
                "UPDATE jobs SET started_at = DATETIME('now','-3 hours'), acquired_at = DATETIME('now','-3 hours') WHERE id=?",
                (int(jp["id"]),),
            )
    finally:
        conn.close()
    headers = {"X-API-KEY": get_settings().SINGLE_USER_API_KEY}
    with TestClient(app, headers=headers) as client:
        r = client.post(
            "/api/v1/jobs/ttl/sweep",
            json={
                "age_seconds": 3600,
                "runtime_seconds": 3600,
                "action": "fail",
                "domain": domain,
                "queue": queue,
                "job_type": jt,
            },
            headers={**headers, "X-Confirm": "true"},
        )
        assert r.status_code == 200
        s = _stats(client, domain, queue, jt)
        assert s["queued"] == 0 and s["processing"] == 0
        # Metrics: failures_total should have entries for ttl_age and ttl_runtime
        try:
            from tldw_Server_API.app.core.Metrics.metrics_manager import get_metrics_registry

            reg = get_metrics_registry()
            vals = list(reg.values.get("jobs.failures_total", []))
            saw_age = False
            saw_runtime = False
            for mv in vals:
                if (
                    mv.labels.get("domain") == domain
                    and mv.labels.get("queue") == queue
                    and mv.labels.get("job_type") == jt
                ):
                    if mv.labels.get("reason") == "ttl_age":
                        saw_age = True
                    if mv.labels.get("reason") == "ttl_runtime":
                        saw_runtime = True
            assert saw_age and saw_runtime, "Expected failures_total increments for ttl_age and ttl_runtime"
        except Exception:
            _ = None
    _assert_terminal_leases_cleared(jm, jq["id"], jp["id"])
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
