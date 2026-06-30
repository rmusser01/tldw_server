import os
from pathlib import Path
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.Jobs.migrations import ensure_jobs_tables
from tldw_Server_API.app.core.Jobs.manager import JobManager


def _setup_env(monkeypatch, tmp_path):


    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.delenv("SINGLE_USER_API_KEY", raising=False)
    monkeypatch.setenv("JOBS_DB_PATH", os.path.join(os.getcwd(), "Databases", "jobs.db"))
    # Disable background workers to avoid races during tests
    monkeypatch.setenv("CHATBOOKS_CORE_WORKER_ENABLED", "false")
    monkeypatch.setenv("JOBS_METRICS_GAUGES_ENABLED", "false")
    monkeypatch.setenv("JOBS_METRICS_RECONCILE_ENABLE", "false")
    monkeypatch.setenv("AUDIO_JOBS_WORKER_ENABLED", "false")
    monkeypatch.setenv("EMBEDDINGS_REEMBED_WORKER_ENABLED", "false")
    monkeypatch.setenv("JOBS_WEBHOOKS_ENABLED", "false")
    monkeypatch.setenv("WORKFLOWS_WEBHOOK_DLQ_ENABLED", "false")
    monkeypatch.setenv("WORKFLOWS_ARTIFACT_GC_ENABLED", "false")
    monkeypatch.setenv("WORKFLOWS_DB_MAINTENANCE_ENABLED", "false")
    # Skip privilege catalog validation to avoid requiring config file in these tests
    monkeypatch.setenv("PRIVILEGE_METADATA_VALIDATE_ON_STARTUP", "0")
    # Minimize startup
    monkeypatch.setenv("MINIMAL_TEST_APP", "1")


def _client(monkeypatch):


    from tldw_Server_API.app.core.AuthNZ.settings import get_settings, reset_settings
    reset_settings()
    from tldw_Server_API.app.main import app
    try:
        app.dependency_overrides.clear()
    except Exception:
        _ = None
    headers = {"X-API-KEY": get_settings().SINGLE_USER_API_KEY}
    return app, headers


def test_queue_control_and_status_sqlite(monkeypatch, tmp_path):


    _setup_env(monkeypatch, tmp_path)
    ensure_jobs_tables(Path(os.environ["JOBS_DB_PATH"]))
    app, headers = _client(monkeypatch)

    with TestClient(app, headers=headers) as client:
        # Pause
        r = client.post("/api/v1/jobs/queue/control", json={"domain": "ps", "queue": "default", "action": "pause"})
        assert r.status_code == 200
        assert r.json()["paused"] is True
        # Check status
        r2 = client.get("/api/v1/jobs/queue/status", params={"domain": "ps", "queue": "default"})
        assert r2.status_code == 200
        assert r2.json()["paused"] is True
        # Resume
        r3 = client.post("/api/v1/jobs/queue/control", json={"domain": "ps", "queue": "default", "action": "resume"})
        assert r3.status_code == 200
        assert r3.json()["paused"] is False


def test_reschedule_and_retry_now_endpoints_sqlite(monkeypatch, tmp_path):


    _setup_env(monkeypatch, tmp_path)
    ensure_jobs_tables(Path(os.environ["JOBS_DB_PATH"]))
    app, headers = _client(monkeypatch)
    jm = JobManager()
    future = datetime.utcnow() + timedelta(hours=1)
    jm.create_job(domain="ps", queue="default", job_type="t", payload={}, owner_user_id="u", available_at=future)

    with TestClient(app, headers=headers) as client:
        # Dry-run reschedule
        r = client.post("/api/v1/jobs/reschedule", json={"domain": "ps", "queue": "default", "job_type": "t", "status": "queued", "set_now": True, "dry_run": True})
        assert r.status_code == 200 and r.json()["affected"] >= 1
        # Execute reschedule
        r2 = client.post("/api/v1/jobs/reschedule", json={"domain": "ps", "queue": "default", "job_type": "t", "status": "queued", "set_now": True, "dry_run": False})
        assert r2.status_code == 200 and r2.json()["affected"] >= 1
        # Retry-now dry-run (include scheduled queued as well)
        r3 = client.post("/api/v1/jobs/retry-now", json={"domain": "ps", "queue": "default", "only_failed": False, "dry_run": True})
        assert r3.status_code == 200


def test_job_detail_endpoint_sqlite(monkeypatch, tmp_path):


    _setup_env(monkeypatch, tmp_path)
    ensure_jobs_tables(Path(os.environ["JOBS_DB_PATH"]))
    app, headers = _client(monkeypatch)
    jm = JobManager()
    job = jm.create_job(domain="ps", queue="default", job_type="detail", payload={"hello": "world"}, owner_user_id="u")

    with TestClient(app, headers=headers) as client:
        r = client.get(f"/api/v1/jobs/{int(job['id'])}", params={"domain": "ps"})
        assert r.status_code == 200
        body = r.json()
        assert body["id"] == int(job["id"])
        assert body["domain"] == "ps"
        assert body.get("payload", {}).get("hello") == "world"
        assert body.get("archived") is False


def test_attachments_and_sla_endpoints_sqlite(monkeypatch, tmp_path):


    _setup_env(monkeypatch, tmp_path)
    ensure_jobs_tables(Path(os.environ["JOBS_DB_PATH"]))
    app, headers = _client(monkeypatch)
    jm = JobManager()
    j = jm.create_job(domain="ps", queue="default", job_type="exp", payload={}, owner_user_id="u")

    with TestClient(app, headers=headers) as client:
        # Add attachment
        r = client.post(f"/api/v1/jobs/{int(j['id'])}/attachments", json={"kind": "log", "content_text": "hello"})
        assert r.status_code == 200
        # List attachments
        r2 = client.get(f"/api/v1/jobs/{int(j['id'])}/attachments")
        assert r2.status_code == 200 and isinstance(r2.json(), list) and len(r2.json()) >= 1
        # Upsert SLA policy
        r3 = client.post("/api/v1/jobs/sla/policy", json={"domain": "ps", "queue": "default", "job_type": "exp", "max_queue_latency_seconds": 0, "max_duration_seconds": 0, "enabled": True})
        assert r3.status_code == 200
        # List SLA policies
        r4 = client.get("/api/v1/jobs/sla/policies", params={"domain": "ps", "queue": "default", "job_type": "exp"})
        assert r4.status_code == 200 and isinstance(r4.json(), list) and len(r4.json()) == 1


def test_crypto_rotate_endpoint_sqlite(monkeypatch, tmp_path):


    _setup_env(monkeypatch, tmp_path)
    ensure_jobs_tables(Path(os.environ["JOBS_DB_PATH"]))
    app, headers = _client(monkeypatch)
    jm = JobManager()
    # Set encryption for domain
    monkeypatch.setenv("JOBS_ENCRYPT", "true")
    # Seed a job with payload to be encrypted
    j = jm.create_job(domain="ps", queue="default", job_type="t", payload={"a": 1}, owner_user_id="u")
    # Rotate keys: use two random-ish fixed keys (base64 32-byte strings)
    old_key = "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVo0NTY3ODkwMTIzNDU2Nzg5MDEy"[:44]
    new_key = "MDEyMzQ1Njc4OTAxMjM0NTY3ODkwQUJDREVGR0hJSktMTU5PUFFSU1RVVldY"[:44]
    with TestClient(app, headers=headers) as client:
        # Dry-run (no-op if key doesn't match envelopes yet)
        body = {"old_key_b64": old_key, "new_key_b64": new_key, "domain": "ps", "fields": ["payload"], "dry_run": True}
        r = client.post("/api/v1/jobs/crypto/rotate", json=body)
        assert r.status_code == 200
        # Execute requires X-Confirm; still safe as envelopes may not match provided keys
        r2 = client.post("/api/v1/jobs/crypto/rotate", headers={**headers, "X-Confirm": "true"}, json={**body, "dry_run": False})
        assert r2.status_code == 200


def test_job_events_list_returns_public_attrs_shape_sqlite(monkeypatch, tmp_path):
    _setup_env(monkeypatch, tmp_path)
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "true")
    ensure_jobs_tables(Path(os.environ["JOBS_DB_PATH"]))
    app, headers = _client(monkeypatch)
    jm = JobManager()
    jm.create_job(
        domain="media_ingest",
        queue="default",
        job_type="download",
        payload={},
        owner_user_id="u1",
    )
    jm.create_job(
        domain="chatbooks",
        queue="default",
        job_type="download",
        payload={},
        owner_user_id="u1",
    )

    with TestClient(app, headers=headers) as client:
        response = client.get(
            "/api/v1/jobs/events",
            params={
                "after_id": 0,
                "domain": "media_ingest",
                "queue": "default",
                "job_type": "download",
            },
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body) == 1
    assert body[0]["domain"] == "media_ingest"
    assert body[0]["event_type"] == "job.created"
    assert isinstance(body[0]["attrs"], dict)
    assert body[0]["attrs"]["owner_user_id"] == "u1"
    assert "attrs_json" not in body[0]
