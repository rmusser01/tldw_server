import os
import json
import time
import pytest
from fastapi.routing import APIRoute

# FIXME: These Postgres outbox tests intermittently time out in some envs.
# Disable by default; set RUN_PG_JOBS_TESTS=1 to enable locally.
_RUN = str(os.getenv("RUN_PG_JOBS_TESTS", "")).strip().lower() in {"1", "true", "yes", "y", "on"}
pytestmark = [pytest.mark.pg_jobs]
if not _RUN:
    pytestmark.append(pytest.mark.skip(reason="FIXME: Postgres outbox tests disabled by default; set RUN_PG_JOBS_TESTS=1 to enable"))
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.Jobs.manager import JobManager


def _ensure_jobs_router_registered(app) -> None:
    has_events_route = any(
        isinstance(route, APIRoute)
        and route.path == "/api/v1/jobs/events"
        and "GET" in (route.methods or set())
        for route in app.routes
    )
    if has_events_route:
        return

    from tldw_Server_API.app.api.v1.endpoints.jobs_admin import router as jobs_admin_router
    from tldw_Server_API.app.core.config import API_V1_PREFIX

    app.include_router(jobs_admin_router, prefix=f"{API_V1_PREFIX}", tags=["jobs"])





@pytest.mark.integration
def test_outbox_list_and_sse_postgres(monkeypatch, jobs_pg_dsn, route_debugger):
     # Ensure outbox is enabled and polling is snappy for the test
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "true")
    monkeypatch.setenv("JOBS_EVENTS_POLL_INTERVAL", "0.05")
    os.environ["JOBS_DB_URL"] = jobs_pg_dsn
    from tldw_Server_API.app.core.AuthNZ.settings import get_settings, reset_settings
    reset_settings()
    from tldw_Server_API.app.main import app
    # Ensure jobs router is mounted even if route policy disabled it
    try:
        _ensure_jobs_router_registered(app)
    except Exception:
        _ = None

    jm = JobManager(backend="postgres", db_url=os.getenv("JOBS_DB_URL"))
    j = jm.create_job(domain="chatbooks", queue="default", job_type="export", payload={}, owner_user_id="u1")
    acq = jm.acquire_next_job(domain="chatbooks", queue="default", lease_seconds=5, worker_id="w1")
    assert acq is not None
    jm.fail_job(
        int(acq["id"]),
        error="boom",
        retryable=True,
        backoff_seconds=1,
        worker_id="w1",
        lease_id=str(acq.get("lease_id")),
    )

    headers = {"X-API-KEY": get_settings().SINGLE_USER_API_KEY}
    with TestClient(app, headers=headers) as client:
        r = client.get("/api/v1/jobs/events", params={"after_id": 0, "domain": "chatbooks"})
        if r.status_code == 404:
            route_debugger(app)
        assert r.status_code == 200
        rows = r.json()
        assert any(ev["event_type"].startswith("job.") for ev in rows)

        with client.stream("GET", "/api/v1/jobs/events/stream", params={"after_id": 0}) as s:
            jm.create_job(domain="chatbooks", queue="default", job_type="export", payload={}, owner_user_id="u2")
            deadline = time.time() + 3.0
            ok = False
            for line in s.iter_lines():
                if not line:
                    if time.time() > deadline:
                        break
                    continue
                if isinstance(line, bytes):
                    line = line.decode()
                if line.startswith("data:"):
                    try:
                        obj = json.loads(line[len("data:"):].strip())
                        if obj.get("event") == "job.created":
                            ok = True
                            break
                    except Exception:
                        _ = None
                if time.time() > deadline:
                    break
            assert ok

        # Create and complete a job and assert exactly one job.completed event
        j3 = jm.create_job(domain="chatbooks", queue="default", job_type="export", payload={}, owner_user_id="u3", priority=1)
        acq3 = jm.acquire_next_job(domain="chatbooks", queue="default", lease_seconds=5, worker_id="w3")
        assert acq3 is not None
        completed_job_id = int(acq3["id"])
        jm.complete_job(completed_job_id, worker_id="w3", lease_id=str(acq3.get("lease_id")))

        # Poll list endpoint to find job.completed for this job
        deadline2 = time.time() + 3.0
        count = 0
        while time.time() < deadline2 and count == 0:
            r = client.get("/api/v1/jobs/events", params={"after_id": 0, "domain": "chatbooks"})
            assert r.status_code == 200
            rows = r.json()
            count = sum(1 for ev in rows if ev.get("event_type") == "job.completed" and int(ev.get("job_id") or 0) == completed_job_id)
            if count == 0:
                time.sleep(0.05)
        assert count == 1, f"expected 1 job.completed event for job {completed_job_id}, found {count}"


@pytest.mark.integration
def test_outbox_after_id_and_filters_postgres(monkeypatch, jobs_pg_dsn, route_debugger):
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "true")
    monkeypatch.setenv("JOBS_EVENTS_POLL_INTERVAL", "0.05")
    os.environ["JOBS_DB_URL"] = jobs_pg_dsn
    pg_dsn_local = jobs_pg_dsn
    from tldw_Server_API.app.core.AuthNZ.settings import get_settings, reset_settings
    reset_settings()
    from tldw_Server_API.app.core.Jobs.pg_migrations import ensure_jobs_tables_pg
    ensure_jobs_tables_pg(pg_dsn_local)
    jm = JobManager(backend="postgres", db_url=pg_dsn_local)
    j1 = jm.create_job(domain="chatbooks", queue="default", job_type="export", payload={}, owner_user_id="u1")
    j2 = jm.create_job(domain="other", queue="default", job_type="import", payload={}, owner_user_id="u2")

    from tldw_Server_API.app.core.Jobs.event_stream import emit_job_event
    emit_job_event("jobs.filter_test", job={"id": int(j1["id"]), "domain": "chatbooks", "queue": "default", "job_type": "export"}, attrs={})
    emit_job_event("jobs.filter_test", job={"id": int(j2["id"]), "domain": "other", "queue": "default", "job_type": "import"}, attrs={})

    from fastapi.testclient import TestClient
    from tldw_Server_API.app.main import app
    # Ensure jobs router is mounted even if route policy disabled it
    try:
        _ensure_jobs_router_registered(app)
    except Exception:
        _ = None
    headers = {"X-API-KEY": get_settings().SINGLE_USER_API_KEY}
    with TestClient(app, headers=headers) as client:
        r = client.get("/api/v1/jobs/events", params={"after_id": 0, "domain": "chatbooks"})
        if r.status_code == 404:
            route_debugger(app)
        assert r.status_code == 200
        rows = r.json()
        assert all(ev.get("domain") == "chatbooks" for ev in rows)
        last_id = rows[-1]["id"] if rows else 0
        emit_job_event("jobs.paging_test", job={"id": int(j1["id"]), "domain": "chatbooks", "queue": "default", "job_type": "export"}, attrs={})
        r2 = client.get("/api/v1/jobs/events", params={"after_id": int(last_id)})
        if r2.status_code == 404:
            route_debugger(app)
        assert r2.status_code == 200
        rows2 = r2.json()
        assert all(ev["id"] > int(last_id) for ev in rows2)
        with client.stream("GET", "/api/v1/jobs/events/stream", params={"after_id": 0}):
            pass
