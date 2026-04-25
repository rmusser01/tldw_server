import shutil
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.exceptions import BadRequestError


pytestmark = pytest.mark.unit


@pytest.fixture
def media_ingest_jobs_client(monkeypatch, tmp_path):
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", "test-api-key-12345")
    monkeypatch.setenv("SANDBOX_WS_REDIS_FANOUT", "0")
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.delenv("SANDBOX_REDIS_URL", raising=False)
    monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / "jobs.db"))
    monkeypatch.delenv("JOBS_DB_URL", raising=False)

    from tldw_Server_API.app.api.v1.endpoints.media.ingest_jobs import router as ingest_jobs_router

    app = FastAPI()
    app.include_router(ingest_jobs_router, prefix="/api/v1/media", tags=["media"])
    with TestClient(app) as client:
        yield client


def test_submit_media_ingest_jobs_creates_one_job_per_item(
    media_ingest_jobs_client,
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / "jobs.db"))

    captured = []

    from tldw_Server_API.app.core.Jobs import manager as jobs_manager

    def fake_create_job(
        self,
        *,
        domain,
        queue,
        job_type,
        payload,
        owner_user_id,
        project_id=None,
        batch_group=None,
        priority=5,
        max_retries=3,
        available_at=None,
        idempotency_key=None,
        request_id=None,
        trace_id=None,
        **_kwargs,
    ):
        captured.append(
            {
                "domain": domain,
                "queue": queue,
                "job_type": job_type,
                "payload": payload,
                "owner_user_id": owner_user_id,
                "batch_group": batch_group,
                "request_id": request_id,
                "trace_id": trace_id,
            }
        )
        return {"id": len(captured), "uuid": f"u{len(captured)}", "status": "queued"}

    monkeypatch.setattr(jobs_manager.JobManager, "create_job", fake_create_job, raising=True)

    upload_path = tmp_path / "sample.txt"
    upload_path.write_text("hello ingest job", encoding="utf-8")

    data = {
        "media_type": "document",
        "urls": "https://example.com/doc1",
    }
    files = [
        ("files", ("sample.txt", upload_path.read_bytes(), "text/plain")),
    ]

    resp = media_ingest_jobs_client.post(
        "/api/v1/media/ingest/jobs",
        data=data,
        files=files,
        headers={"X-API-KEY": "test-api-key-12345"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body.get("batch_id")
    assert len(body.get("jobs", [])) == 2

    payloads = [item["payload"] for item in captured]
    url_payload = next(item for item in payloads if item.get("source_kind") == "url")
    file_payload = next(item for item in payloads if item.get("source_kind") == "file")

    assert url_payload["source"] == "https://example.com/doc1"
    assert file_payload["original_filename"] == "sample.txt"
    assert file_payload.get("temp_dir")
    assert Path(file_payload["source"]).exists()

    shutil.rmtree(file_payload["temp_dir"], ignore_errors=True)


def test_get_media_ingest_job_includes_result_media_id(
    media_ingest_jobs_client,
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / "jobs.db"))

    from tldw_Server_API.app.core.Jobs import manager as jobs_manager

    def fake_get_job(self, job_id):
        return {
            "id": int(job_id),
            "domain": "media_ingest",
            "job_type": "media_ingest_item",
            "owner_user_id": "1",
            "status": "completed",
            "created_at": "2026-01-01T00:00:00Z",
            "started_at": "2026-01-01T00:00:01Z",
            "completed_at": "2026-01-01T00:00:05Z",
            "cancelled_at": None,
            "cancellation_reason": None,
            "progress_percent": 100.0,
            "progress_message": "completed",
            "payload": {
                "media_type": "video",
                "source": "https://example.com/video",
                "source_kind": "url",
                "batch_id": "batch-1",
            },
            "result": {
                "status": "Success",
                "media_id": 321,
            },
            "error_message": None,
        }

    monkeypatch.setattr(jobs_manager.JobManager, "get_job", fake_get_job, raising=True)

    resp = media_ingest_jobs_client.get(
        "/api/v1/media/ingest/jobs/99",
        headers={"X-API-KEY": "test-api-key-12345"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "completed"
    assert body["media_type"] == "video"
    assert body["source_kind"] == "url"
    assert body["result"]["media_id"] == 321


def test_get_media_ingest_job_rejects_boolean_admin_without_claims(monkeypatch, tmp_path):
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", "test-api-key-12345")
    monkeypatch.setenv("SANDBOX_WS_REDIS_FANOUT", "0")
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.delenv("SANDBOX_REDIS_URL", raising=False)
    monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / "jobs.db"))
    monkeypatch.delenv("JOBS_DB_URL", raising=False)

    from tldw_Server_API.app.api.v1.endpoints.media.ingest_jobs import (
        get_job_manager,
        router as ingest_jobs_router,
    )

    app = FastAPI()
    app.include_router(ingest_jobs_router, prefix="/api/v1/media", tags=["media"])

    class _StubJobManager:
        def get_job(self, _job_id: int):
            return {
                "id": 7,
                "domain": "media_ingest",
                "job_type": "media_ingest_item",
                "owner_user_id": "2",
                "status": "queued",
                "payload": {},
            }

    async def _principal_override():
        return AuthPrincipal(
            kind="user",
            user_id=1,
            api_key_id=None,
            subject=None,
            token_type="access",
            jti=None,
            roles=["user"],
            permissions=[],
            is_admin=True,
            org_ids=[],
            team_ids=[],
        )

    async def _user_override():
        return User(id=1, username="tester", email=None, is_active=True, is_admin=False)

    app.dependency_overrides[get_job_manager] = lambda: _StubJobManager()
    app.dependency_overrides[get_auth_principal] = _principal_override
    app.dependency_overrides[get_request_user] = _user_override
    try:
        with TestClient(app) as client:
            resp = client.get(
                "/api/v1/media/ingest/jobs/7",
                headers={"X-API-KEY": "test-api-key-12345"},
            )
            assert resp.status_code == 403
    finally:
        app.dependency_overrides.clear()


def test_submit_media_ingest_jobs_returns_429_for_concurrent_job_limit(
    media_ingest_jobs_client,
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / "jobs.db"))

    from tldw_Server_API.app.core.Jobs import manager as jobs_manager

    def fake_create_job(self, **_kwargs):
        raise BadRequestError("User 1 has reached the maximum concurrent job limit (5)")

    monkeypatch.setattr(jobs_manager.JobManager, "create_job", fake_create_job, raising=True)

    resp = media_ingest_jobs_client.post(
        "/api/v1/media/ingest/jobs",
        data={
            "media_type": "document",
            "urls": "https://example.com/too-many",
        },
        headers={"X-API-KEY": "test-api-key-12345"},
    )

    assert resp.status_code == 429, resp.text
    assert resp.json()["detail"] == "User 1 has reached the maximum concurrent job limit (5)"


def test_submit_media_ingest_jobs_routes_heavy_request_to_default_queue_when_heavy_worker_unavailable(
    media_ingest_jobs_client,
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / "jobs.db"))
    monkeypatch.setenv("MEDIA_INGEST_JOBS_DEFAULT_QUEUE", "default")
    monkeypatch.setenv("MEDIA_INGEST_JOBS_HEAVY_QUEUE", "media-heavy")
    monkeypatch.setenv("MEDIA_INGEST_JOBS_ROUTE_HEAVY", "true")
    monkeypatch.delenv("MEDIA_INGEST_HEAVY_JOBS_WORKER_ENABLED", raising=False)
    monkeypatch.delenv("ROUTES_ENABLE", raising=False)
    monkeypatch.delenv("TLDW_WORKERS_SIDECAR_MODE", raising=False)

    captured = []

    from tldw_Server_API.app.core.Jobs import manager as jobs_manager

    def fake_create_job(self, *, queue, **_kwargs):
        captured.append(queue)
        return {"id": 1, "uuid": "u1", "status": "queued"}

    monkeypatch.setattr(jobs_manager.JobManager, "create_job", fake_create_job, raising=True)

    resp = media_ingest_jobs_client.post(
        "/api/v1/media/ingest/jobs",
        data={
            "media_type": "video",
            "urls": "https://example.com/video-default.mp4",
        },
        headers={"X-API-KEY": "test-api-key-12345"},
    )

    assert resp.status_code == 200, resp.text
    assert captured == ["default"]


def test_submit_media_ingest_jobs_routes_heavy_request_to_default_queue_in_sidecar_mode_when_heavy_worker_disabled(
    media_ingest_jobs_client,
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / "jobs.db"))
    monkeypatch.setenv("MEDIA_INGEST_JOBS_DEFAULT_QUEUE", "default")
    monkeypatch.setenv("MEDIA_INGEST_JOBS_HEAVY_QUEUE", "media-heavy")
    monkeypatch.setenv("MEDIA_INGEST_JOBS_ROUTE_HEAVY", "true")
    monkeypatch.setenv("TLDW_WORKERS_SIDECAR_MODE", "true")
    monkeypatch.delenv("MEDIA_INGEST_HEAVY_JOBS_WORKER_ENABLED", raising=False)
    monkeypatch.delenv("ROUTES_ENABLE", raising=False)

    captured = []

    from tldw_Server_API.app.core.Jobs import manager as jobs_manager

    def fake_create_job(self, *, queue, **_kwargs):
        captured.append(queue)
        return {"id": 1, "uuid": "u1", "status": "queued"}

    monkeypatch.setattr(jobs_manager.JobManager, "create_job", fake_create_job, raising=True)

    resp = media_ingest_jobs_client.post(
        "/api/v1/media/ingest/jobs",
        data={
            "media_type": "video",
            "urls": "https://example.com/video-sidecar.mp4",
        },
        headers={"X-API-KEY": "test-api-key-12345"},
    )

    assert resp.status_code == 200, resp.text
    assert captured == ["default"]


def test_submit_media_ingest_jobs_routes_heavy_request_to_heavy_queue_in_sidecar_mode_when_enabled(
    media_ingest_jobs_client,
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / "jobs.db"))
    monkeypatch.setenv("MEDIA_INGEST_JOBS_DEFAULT_QUEUE", "default")
    monkeypatch.setenv("MEDIA_INGEST_JOBS_HEAVY_QUEUE", "media-heavy")
    monkeypatch.setenv("MEDIA_INGEST_JOBS_ROUTE_HEAVY", "true")
    monkeypatch.setenv("TLDW_WORKERS_SIDECAR_MODE", "true")
    monkeypatch.setenv("MEDIA_INGEST_HEAVY_JOBS_WORKER_ENABLED", "true")
    monkeypatch.delenv("ROUTES_ENABLE", raising=False)

    captured = []

    from tldw_Server_API.app.core.Jobs import manager as jobs_manager

    def fake_create_job(self, *, queue, **_kwargs):
        captured.append(queue)
        return {"id": 1, "uuid": "u1", "status": "queued"}

    monkeypatch.setattr(jobs_manager.JobManager, "create_job", fake_create_job, raising=True)

    resp = media_ingest_jobs_client.post(
        "/api/v1/media/ingest/jobs",
        data={
            "media_type": "video",
            "urls": "https://example.com/video-sidecar.mp4",
        },
        headers={"X-API-KEY": "test-api-key-12345"},
    )

    assert resp.status_code == 200, resp.text
    assert captured == ["media-heavy"]


def test_submit_media_ingest_jobs_routes_heavy_request_to_heavy_queue_when_route_enabled_in_test_mode(
    media_ingest_jobs_client,
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / "jobs.db"))
    monkeypatch.setenv("MEDIA_INGEST_JOBS_DEFAULT_QUEUE", "default")
    monkeypatch.setenv("MEDIA_INGEST_JOBS_HEAVY_QUEUE", "media-heavy")
    monkeypatch.setenv("MEDIA_INGEST_JOBS_ROUTE_HEAVY", "true")
    monkeypatch.setenv("ROUTES_ENABLE", "media-ingest-heavy-jobs")
    monkeypatch.delenv("MEDIA_INGEST_HEAVY_JOBS_WORKER_ENABLED", raising=False)
    monkeypatch.delenv("TLDW_WORKERS_SIDECAR_MODE", raising=False)

    captured = []

    from tldw_Server_API.app.core.Jobs import manager as jobs_manager

    def fake_create_job(self, *, queue, **_kwargs):
        captured.append(queue)
        return {"id": 1, "uuid": "u1", "status": "queued"}

    monkeypatch.setattr(jobs_manager.JobManager, "create_job", fake_create_job, raising=True)

    resp = media_ingest_jobs_client.post(
        "/api/v1/media/ingest/jobs",
        data={
            "media_type": "audio",
            "urls": "https://example.com/audio-heavy.mp3",
        },
        headers={"X-API-KEY": "test-api-key-12345"},
    )

    assert resp.status_code == 200, resp.text
    assert captured == ["media-heavy"]
