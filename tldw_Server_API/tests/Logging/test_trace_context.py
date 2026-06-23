import types
from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.main import app
from tldw_Server_API.app.core.Logging.log_context import ensure_traceparent
from tldw_Server_API.app.services.app_lifecycle import reset_lifecycle_state


def _route_exists(path: str, method: str) -> bool:
    wanted_method = method.upper()
    for route in app.routes:
        route_path = getattr(route, "path", None)
        route_methods = getattr(route, "methods", set()) or set()
        if route_path == path and wanted_method in route_methods:
            return True
    return False


def _fresh_test_client() -> TestClient:
    """Return a TestClient after clearing any leaked shared-app drain state."""
    reset_lifecycle_state(app)
    client = TestClient(app)
    reset_lifecycle_state(app)
    return client


def test_ensure_traceparent_sets_state_and_returns_value():


    class _State:  # minimal stand-in for request.state
        pass

    class _Req:
        def __init__(self, headers: Dict[str, str]):
            self.headers = headers
            self.state = _State()

    tp = "00-0123456789abcdef0123456789abcdef-0123456789abcdef-01"
    req = _Req({"traceparent": tp})
    out = ensure_traceparent(req)
    assert out == tp
    assert getattr(req.state, "traceparent") == tp

    # Case-insensitive header
    req2 = _Req({"Traceparent": tp})
    out2 = ensure_traceparent(req2)
    assert out2 == tp
    assert getattr(req2.state, "traceparent") == tp


def test_audio_jobs_submit_propagates_request_id(monkeypatch):


    if not _route_exists("/api/v1/audio/jobs/submit", "POST"):
        pytest.skip("audio-jobs submit route is disabled in this test app configuration")

    # Capture kwargs sent to JobManager.create_job
    captured: Dict[str, Any] = {}

    from tldw_Server_API.app.core.Jobs import manager as jobs_manager

    def fake_create_job(self, *, domain, queue, job_type, payload, owner_user_id, project_id=None,
                        priority=5, max_retries=3, available_at=None, idempotency_key=None,
                        request_id=None, trace_id=None, **kwargs):  # signature-compatible
        captured.update({
            "domain": domain,
            "queue": queue,
            "job_type": job_type,
            "payload": payload,
            "owner_user_id": owner_user_id,
            "request_id": request_id,
        })
        return {"id": 42, "uuid": "u", "domain": domain, "queue": queue, "job_type": job_type, "status": "queued"}

    monkeypatch.setattr(jobs_manager.JobManager, "create_job", fake_create_job, raising=True)

    client = _fresh_test_client()
    resp = client.post(
        "/api/v1/audio/jobs/submit",
        json={"url": "https://example.com/a.mp3"},
        headers={
            "X-API-KEY": "test-api-key-12345",
            "X-Request-ID": "req-123",
            "traceparent": "00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-01",
        },
    )
    assert resp.status_code == 200, resp.text
    assert captured.get("request_id") == "req-123"


def test_media_ingest_jobs_submit_propagates_request_id(monkeypatch, tmp_path):


    captured: Dict[str, Any] = {}

    from tldw_Server_API.app.core.Jobs import manager as jobs_manager

    def fake_create_job(self, *, domain, queue, job_type, payload, owner_user_id, project_id=None,
                        priority=5, max_retries=3, available_at=None, idempotency_key=None,
                        request_id=None, trace_id=None, batch_group=None, **kwargs):
        captured.update({
            "domain": domain,
            "queue": queue,
            "job_type": job_type,
            "payload": payload,
            "owner_user_id": owner_user_id,
            "request_id": request_id,
        })
        return {"id": 77, "uuid": "u77", "domain": domain, "queue": queue, "job_type": job_type, "status": "queued"}

    monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / "jobs_media_ingest.db"))
    monkeypatch.delenv("JOBS_DB_URL", raising=False)
    monkeypatch.setattr(jobs_manager.JobManager, "create_job", fake_create_job, raising=True)

    client = _fresh_test_client()
    resp = client.post(
        "/api/v1/media/ingest/jobs",
        data={"media_type": "audio", "urls": "https://example.com/a.mp3"},
        headers={
            "X-API-KEY": "test-api-key-12345",
            "X-Request-ID": "req-789",
            "traceparent": "00-eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee-ffffffffffffffff-01",
        },
    )
    assert resp.status_code == 200, resp.text
    assert captured.get("request_id") == "req-789"


def test_reembed_schedule_propagates_request_id(monkeypatch):


    captured: Dict[str, Any] = {}

    from tldw_Server_API.app.core.Jobs import manager as jobs_manager
    from tldw_Server_API.app.core.Embeddings import redis_pipeline

    if not _route_exists("/api/v1/embeddings/reembed/schedule", "POST"):
        pytest.skip("re-embed schedule route is disabled in this test app configuration")

    def fake_create_job(self, *, domain, queue, job_type, payload, owner_user_id, project_id=None,
                        priority=5, max_retries=3, available_at=None, idempotency_key=None,
                        request_id=None, trace_id=None, **kwargs):
        captured.update({
            "domain": domain,
            "queue": queue,
            "job_type": job_type,
            "payload": payload,
            "owner_user_id": owner_user_id,
            "request_id": request_id,
        })
        return {"id": 99, "uuid": "u2", "domain": domain, "queue": queue, "job_type": job_type, "status": "queued"}

    def fake_enqueue_chunking_job(*, payload, root_job_uuid, force_regenerate=False, require_redis=True):
        captured.update({
            "redis_root_job_uuid": root_job_uuid,
            "redis_request_id": payload.get("request_id"),
        })
        return "1-0"

    monkeypatch.setattr(jobs_manager.JobManager, "create_job", fake_create_job, raising=True)
    monkeypatch.setattr(redis_pipeline, "enqueue_chunking_job", fake_enqueue_chunking_job, raising=True)

    client = _fresh_test_client()
    resp = client.post(
        "/api/v1/embeddings/reembed/schedule",
        json={"media_id": 1},
        headers={
            "X-API-KEY": "test-api-key-12345",
            "X-Request-ID": "req-456",
            "traceparent": "00-cccccccccccccccccccccccccccccccc-dddddddddddddddd-01",
        },
    )
    assert resp.status_code == 200, resp.text
    assert captured.get("request_id") == "req-456"
