import os
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

psycopg = pytest.importorskip("psycopg")

from tldw_Server_API.app.core.Jobs.manager import JobManager

pytestmark = pytest.mark.pg_jobs


def _headers(app):

    from tldw_Server_API.app.core.AuthNZ.settings import get_settings

    return {"X-API-KEY": get_settings().SINGLE_USER_API_KEY}


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


def _require_pg(monkeypatch):

    dsn = os.getenv("JOBS_DB_URL")
    if not dsn:
        pytest.skip("JOBS_DB_URL not configured")
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    monkeypatch.setenv("JOBS_PG_SINGLE_UPDATE_ACQUIRE", "true")
    monkeypatch.setenv("JOBS_GAUGES_DEBOUNCE_MS", "0")
    return dsn


def test_pg_batch_cancel_updates_counters(monkeypatch):

    dsn = _require_pg(monkeypatch)
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    reset_settings()
    from tldw_Server_API.app.main import app

    jm = JobManager(backend="postgres", db_url=dsn)
    domain = "chatbooks"
    queue = "default"
    jt = "export"
    first_ready = jm.create_job(domain=domain, queue=queue, job_type=jt, payload={}, owner_user_id="1")
    scheduled = jm.create_job(
        domain=domain,
        queue=queue,
        job_type=jt,
        payload={},
        owner_user_id="1",
        available_at=datetime.now(tz=timezone.utc) + timedelta(seconds=60),
    )
    queued = jm.create_job(domain=domain, queue=queue, job_type=jt, payload={}, owner_user_id="1")
    acq = jm.acquire_next_job(domain=domain, queue=queue, lease_seconds=30, worker_id="w")
    assert acq and acq["id"] == first_ready["id"]
    conn = jm._connect()
    try:
        with jm._pg_cursor(conn) as cur:
            cur.execute(
                "UPDATE jobs SET leased_until=NOW() + interval '1 hour', "
                "worker_id='stale-worker', lease_id='stale-lease' WHERE id IN (%s, %s)",
                (int(scheduled["id"]), int(queued["id"])),
            )
        conn.commit()
    finally:
        conn.close()
    headers = _headers(app)
    with TestClient(app, headers=headers) as client:
        s0 = _stats(client, domain, queue, jt)
        assert s0["processing"] == 1
        r = client.post(
            "/api/v1/jobs/batch/cancel",
            json={"domain": domain, "queue": queue, "job_type": jt, "dry_run": False},
            headers={**headers, "X-Confirm": "true"},
        )
        assert r.status_code == 200
        s1 = _stats(client, domain, queue, jt)
        assert s1["queued"] == 0 and s1["processing"] == 0
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
            cur.execute(
                "SELECT status, leased_until, worker_id, lease_id FROM jobs "
                "WHERE domain=%s AND queue=%s AND job_type=%s",
                (domain, queue, jt),
            )
            terminal_rows = cur.fetchall() or []
            assert terminal_rows
            assert all(
                (
                    _row_val(terminal, "status", 0),
                    _row_val(terminal, "leased_until", 1),
                    _row_val(terminal, "worker_id", 2),
                    _row_val(terminal, "lease_id", 3),
                )
                == ("cancelled", None, None, None)
                for terminal in terminal_rows
            )
    finally:
        conn.close()


def test_pg_complete_queued_updates_counters(monkeypatch):

    dsn = _require_pg(monkeypatch)
    jm = JobManager(backend="postgres", db_url=dsn)
    domain = "chatbooks"
    queue = "default"
    jt = "export"

    ready = jm.create_job(domain=domain, queue=queue, job_type=jt, payload={}, owner_user_id="1")
    scheduled = jm.create_job(
        domain=domain,
        queue=queue,
        job_type=jt,
        payload={},
        owner_user_id="1",
        available_at=datetime.now(tz=timezone.utc) + timedelta(seconds=60),
    )

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
                int(_row_val(row, "ready_count", 0) or 0) == 1
                and int(_row_val(row, "scheduled_count", 1) or 0) == 1
                and int(_row_val(row, "processing_count", 2) or 0) == 0
            )
    finally:
        conn.close()

    assert jm.complete_job(int(ready["id"]), result={}, enforce=False)
    assert jm.complete_job(int(scheduled["id"]), result={}, enforce=False)

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


def test_pg_batch_reschedule_moves_ready_to_scheduled(monkeypatch):

    dsn = _require_pg(monkeypatch)
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    reset_settings()
    from tldw_Server_API.app.main import app

    jm = JobManager(backend="postgres", db_url=dsn)
    domain = "chatbooks"
    queue = "default"
    jt = "export"
    for _ in range(4):
        jm.create_job(domain=domain, queue=queue, job_type=jt, payload={}, owner_user_id="1")
    headers = _headers(app)
    with TestClient(app, headers=headers) as client:
        r = client.post(
            "/api/v1/jobs/batch/reschedule",
            json={"domain": domain, "queue": queue, "job_type": jt, "delay_seconds": 30, "dry_run": False},
            headers={**headers, "X-Confirm": "true"},
        )
        assert r.status_code == 200
        s = _stats(client, domain, queue, jt)
        assert s["queued"] == 0
    conn = jm._connect()
    try:
        with jm._pg_cursor(conn) as cur:
            cur.execute(
                "SELECT ready_count, scheduled_count FROM job_counters WHERE domain=%s AND queue=%s AND job_type=%s",
                (domain, queue, jt),
            )
            row = cur.fetchone()
            assert row is not None
            assert int(_row_val(row, "ready_count", 0) or 0) == 0 and int(_row_val(row, "scheduled_count", 1) or 0) >= 4
    finally:
        conn.close()


def test_pg_batch_requeue_quarantined_adjusts_counters(monkeypatch):

    dsn = _require_pg(monkeypatch)
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    reset_settings()
    from tldw_Server_API.app.main import app

    jm = JobManager(backend="postgres", db_url=dsn)
    domain = "chatbooks"
    queue = "default"
    jt = "export"
    j = jm.create_job(domain=domain, queue=queue, job_type=jt, payload={}, owner_user_id="1")
    acq = jm.acquire_next_job(domain=domain, queue=queue, lease_seconds=30, worker_id="w")
    token1 = str(acq.get("lease_id"))
    jm.fail_job(
        int(j["id"]),
        error="boom",
        retryable=True,
        worker_id="w",
        lease_id=str(acq.get("lease_id")),
        completion_token=token1,
    )
    acq2 = jm.acquire_next_job(domain=domain, queue=queue, lease_seconds=30, worker_id="w")
    token2 = str(acq2.get("lease_id"))
    jm.fail_job(
        int(j["id"]),
        error="boom",
        retryable=True,
        worker_id="w",
        lease_id=str(acq2.get("lease_id")),
        completion_token=token2,
    )
    headers = _headers(app)
    with TestClient(app, headers=headers) as client:
        r = client.post(
            "/api/v1/jobs/batch/requeue-quarantined",
            json={"domain": domain, "queue": queue, "job_type": jt, "dry_run": False},
            headers={**headers, "X-Confirm": "true"},
        )
        assert r.status_code == 200
        s = _stats(client, domain, queue, jt)
        assert s["quarantined"] == 0 and s["queued"] >= 1
    conn = jm._connect()
    try:
        with jm._pg_cursor(conn) as cur:
            cur.execute(
                "SELECT ready_count, quarantined_count FROM job_counters WHERE domain=%s AND queue=%s AND job_type=%s",
                (domain, queue, jt),
            )
            row = cur.fetchone()
            assert row is not None
            assert (
                int(_row_val(row, "ready_count", 0) or 0) >= 1 and int(_row_val(row, "quarantined_count", 1) or 0) == 0
            )
    finally:
        conn.close()
