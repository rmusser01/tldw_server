import os

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

psycopg = pytest.importorskip("psycopg")
from tldw_Server_API.app.core.Utils.fastapi_routes import iter_served_routes
from tldw_Server_API.app.core.Jobs.manager import JobManager


pytestmark = [pytest.mark.pg_jobs]


@pytest.fixture(autouse=True)
def _setup(jobs_pg_dsn):
     # jobs_pg_dsn ensures a fresh temp DB and schema per test and exports JOBS_DB_URL
    return


def _client_pg(monkeypatch):


     # DSN already set via jobs_pg_dsn fixture
    from tldw_Server_API.app.core.AuthNZ.settings import get_settings, reset_settings
    reset_settings()
    from tldw_Server_API.app.main import app
    try:
        app.dependency_overrides.clear()
    except Exception:
        _ = None
    # Force-include jobs admin router only when missing.
    # Some app boot paths already register these routes, and re-including
    # the same router triggers duplicate-route guards in startup.
    try:
        has_queue_control = any(
            isinstance(route.route, APIRoute)
            and route.path == "/api/v1/jobs/queue/control"
            and "POST" in (route.methods or set())
            for route in iter_served_routes(app.routes)
        )
        if not has_queue_control:
            from tldw_Server_API.app.api.v1.endpoints.jobs_admin import (
                router as jobs_admin_router,
            )

            app.include_router(jobs_admin_router, prefix="/api/v1", tags=["jobs"])
    except Exception:
        _ = None
    headers = {"X-API-KEY": get_settings().SINGLE_USER_API_KEY}
    return app, headers


def test_queue_control_and_status_postgres(monkeypatch):


    app, headers = _client_pg(monkeypatch)
    with TestClient(app, headers=headers) as client:
        r = client.post("/api/v1/jobs/queue/control", json={"domain": "ps", "queue": "default", "action": "pause"})
        assert r.status_code == 200
        r2 = client.get("/api/v1/jobs/queue/status", params={"domain": "ps", "queue": "default"})
        assert r2.status_code == 200 and r2.json()["paused"] is True


def test_attachments_and_sla_postgres(monkeypatch, jobs_pg_dsn):


    app, headers = _client_pg(monkeypatch)
    jm = JobManager(None, backend="postgres", db_url=jobs_pg_dsn)
    j = jm.create_job(domain="ps", queue="default", job_type="exp", payload={}, owner_user_id="u")
    with TestClient(app, headers=headers) as client:
        r = client.post(
            f"/api/v1/jobs/{int(j['id'])}/attachments",
            params={"domain": "ps"},
            json={"kind": "log", "content_text": "hello", "url": "http://example.invalid/log"},
        )
        assert r.status_code == 200, r.text
        r2 = client.get(f"/api/v1/jobs/{int(j['id'])}/attachments", params={"domain": "ps"})
        assert r2.status_code == 200, r2.text
        r3 = client.post("/api/v1/jobs/sla/policy", json={"domain": "ps", "queue": "default", "job_type": "exp", "max_queue_latency_seconds": 0, "max_duration_seconds": 0, "enabled": True})
        assert r3.status_code == 200
        r4 = client.get("/api/v1/jobs/sla/policies", params={"domain": "ps", "queue": "default", "job_type": "exp"})
        assert r4.status_code == 200


def test_reschedule_and_retry_now_postgres(monkeypatch, jobs_pg_dsn):


    app, headers = _client_pg(monkeypatch)
    jm = JobManager(None, backend="postgres", db_url=jobs_pg_dsn)
    # Seed a scheduled queued job
    from datetime import datetime, timedelta
    future = datetime.utcnow() + timedelta(hours=1)
    jm.create_job(domain="ps", queue="default", job_type="t", payload={}, owner_user_id="u", available_at=future)
    with TestClient(app, headers=headers) as client:
        r = client.post("/api/v1/jobs/reschedule", json={"domain": "ps", "queue": "default", "job_type": "t", "status": "queued", "set_now": True, "dry_run": True})
        assert r.status_code == 200 and int(r.json().get("affected", 0)) >= 1
        r2 = client.post("/api/v1/jobs/reschedule", json={"domain": "ps", "queue": "default", "job_type": "t", "status": "queued", "set_now": True, "dry_run": False})
        assert r2.status_code == 200 and int(r2.json().get("affected", 0)) >= 1
    # Seed a failed job with retries
    j2 = jm.create_job(domain="ps", queue="default", job_type="t2", payload={}, owner_user_id="u")
    acq = jm.acquire_next_job(domain="ps", queue="default", lease_seconds=10, worker_id="w")
    assert acq
    jm.fail_job(int(acq["id"]), error="x", retryable=True)
    with TestClient(app, headers=headers) as client:
        rr = client.post("/api/v1/jobs/retry-now", json={"domain": "ps", "queue": "default", "only_failed": True, "dry_run": True})
        assert rr.status_code == 200
        rr2 = client.post("/api/v1/jobs/retry-now", json={"domain": "ps", "queue": "default", "only_failed": True, "dry_run": False})
        assert rr2.status_code == 200


def test_crypto_rotate_postgres(monkeypatch, jobs_pg_dsn):


    app, headers = _client_pg(monkeypatch)
    # Configure encryption for domain and set ENV key (old)
    monkeypatch.setenv("JOBS_ENCRYPT", "true")
    old_key = "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVo0NTY3ODkwMTIzNDU2Nzg5MDEy"[:44]
    monkeypatch.setenv("WORKFLOWS_ARTIFACT_ENC_KEY", old_key)
    jm = JobManager(None, backend="postgres", db_url=jobs_pg_dsn)
    # Create a job so payload is stored encrypted with old key
    jm.create_job(domain="ps", queue="default", job_type="cipher", payload={"x": 1}, owner_user_id="u")
    new_key = "MDEyMzQ1Njc4OTAxMjM0NTY3ODkwQUJDREVGR0hJSktMTU5PUFFSU1RVVldY"[:44]
    with TestClient(app, headers=headers) as client:
        body = {"old_key_b64": old_key, "new_key_b64": new_key, "domain": "ps", "fields": ["payload"], "dry_run": True}
        r = client.post("/api/v1/jobs/crypto/rotate", json=body)
        assert r.status_code == 200
        # Execute rotation
        r2 = client.post("/api/v1/jobs/crypto/rotate", headers={**headers, "X-Confirm": "true"}, json={**body, "dry_run": False})
        assert r2.status_code == 200
