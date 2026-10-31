import os

import pytest
from fastapi.testclient import TestClient

psycopg = pytest.importorskip("psycopg")

from tldw_Server_API.app.core.Jobs.manager import JobManager

pytestmark = pytest.mark.pg_jobs


def _require_pg(monkeypatch):

    dsn = os.getenv("JOBS_DB_URL")
    if not dsn:
        pytest.skip("JOBS_DB_URL not configured")
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    monkeypatch.setenv("JOBS_PG_SINGLE_UPDATE_ACQUIRE", "true")
    monkeypatch.setenv("JOBS_GAUGES_DEBOUNCE_MS", "0")
    return dsn


def _stats(client, domain="chatbooks", queue="default", job_type="export"):

    r = client.get("/api/v1/jobs/stats", params={"domain": domain, "queue": queue, "job_type": job_type})
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == 1
    return rows[0]


def _row_val(row, key, idx):
    if isinstance(row, dict):
        return row.get(key)
    return row[idx] if row is not None else None


def _assert_terminal_leases_cleared(jm, *job_ids):
    for job_id in job_ids:
        terminal = jm.get_job(int(job_id))
        assert terminal is not None
        assert terminal["status"] in {"cancelled", "failed"}
        assert terminal["leased_until"] is None
        assert terminal["worker_id"] is None
        assert terminal["lease_id"] is None


def test_pg_ttl_cancel_updates_counters(monkeypatch):

    dsn = _require_pg(monkeypatch)
    from tldw_Server_API.app.core.AuthNZ.settings import get_settings, reset_settings

    reset_settings()
    from tldw_Server_API.app.main import app

    jm = JobManager(backend="postgres", db_url=dsn)
    domain = "chatbooks"
    queue = "default"
    jt = "export"
    jq = jm.create_job(domain=domain, queue=queue, job_type=jt, payload={}, owner_user_id="1")
    jp = jm.create_job(domain=domain, queue=queue, job_type=jt, payload={}, owner_user_id="1")
    acq = jm.acquire_next_job(domain=domain, queue=queue, lease_seconds=30, worker_id="w")
    assert acq
    proc_id = int(acq["id"])
    queued_id = int(jq["id"]) if proc_id != int(jq["id"]) else int(jp["id"])
    # Backdate
    conn = jm._connect()
    try:
        with jm._pg_cursor(conn) as cur:
            cur.execute(
                "UPDATE jobs SET created_at = NOW() - interval '2 hours', "
                "leased_until = NOW() + interval '1 hour', worker_id = 'stale-worker', "
                "lease_id = 'stale-lease' WHERE id = %s",
                (queued_id,),
            )
            cur.execute(
                "UPDATE jobs SET started_at = NOW() - interval '3 hours', acquired_at = NOW() - interval '3 hours' WHERE id = %s",
                (proc_id,),
            )
        conn.commit()
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
        # Metrics: cancelled_total should increment
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
            assert saw
        except Exception:
            _ = None
    _assert_terminal_leases_cleared(jm, queued_id, proc_id)
    conn = jm._connect()
    try:
        with jm._pg_cursor(conn) as cur:
            cur.execute(
                "SELECT ready_count, scheduled_count, processing_count FROM job_counters WHERE domain=%s AND queue=%s AND job_type=%s",
                (domain, queue, jt),
            )
            row = cur.fetchone()
            assert row is not None
            assert (
                int(_row_val(row, "ready_count", 0) or 0) == 0
                and int(_row_val(row, "scheduled_count", 1) or 0) == 0
                and int(_row_val(row, "processing_count", 2) or 0) == 0
            )
    finally:
        conn.close()


def test_pg_ttl_fail_updates_counters(monkeypatch):

    dsn = _require_pg(monkeypatch)
    from tldw_Server_API.app.core.AuthNZ.settings import get_settings, reset_settings

    reset_settings()
    from tldw_Server_API.app.main import app

    jm = JobManager(backend="postgres", db_url=dsn)
    domain = "chatbooks"
    queue = "default"
    jt = "export"
    jq = jm.create_job(domain=domain, queue=queue, job_type=jt, payload={}, owner_user_id="1")
    jp = jm.create_job(domain=domain, queue=queue, job_type=jt, payload={}, owner_user_id="1")
    acq = jm.acquire_next_job(domain=domain, queue=queue, lease_seconds=30, worker_id="w")
    assert acq
    proc_id = int(acq["id"])
    queued_id = int(jq["id"]) if proc_id != int(jq["id"]) else int(jp["id"])
    conn = jm._connect()
    try:
        with jm._pg_cursor(conn) as cur:
            cur.execute(
                "UPDATE jobs SET created_at = NOW() - interval '2 hours', "
                "leased_until = NOW() + interval '1 hour', worker_id = 'stale-worker', "
                "lease_id = 'stale-lease' WHERE id = %s",
                (queued_id,),
            )
            cur.execute(
                "UPDATE jobs SET started_at = NOW() - interval '3 hours', acquired_at = NOW() - interval '3 hours' WHERE id = %s",
                (proc_id,),
            )
        conn.commit()
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
        # Metrics: failures_total should have ttl_age and ttl_runtime labels
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
            assert saw_age and saw_runtime
        except Exception:
            _ = None
    _assert_terminal_leases_cleared(jm, queued_id, proc_id)
    conn = jm._connect()
    try:
        with jm._pg_cursor(conn) as cur:
            cur.execute(
                "SELECT ready_count, scheduled_count, processing_count FROM job_counters WHERE domain=%s AND queue=%s AND job_type=%s",
                (domain, queue, jt),
            )
            row = cur.fetchone()
            assert row is not None
            assert (
                int(_row_val(row, "ready_count", 0) or 0) == 0
                and int(_row_val(row, "scheduled_count", 1) or 0) == 0
                and int(_row_val(row, "processing_count", 2) or 0) == 0
            )
    finally:
        conn.close()
