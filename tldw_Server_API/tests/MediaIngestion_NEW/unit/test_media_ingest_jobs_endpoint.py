import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, Lock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.exceptions import BadRequestError, JobSubmissionLimitError

pytestmark = pytest.mark.unit


class _LoggerStub:
    def __init__(self) -> None:
        self.warnings: list[str] = []

    def warning(self, message: str, *args: object, **kwargs: object) -> None:
        self.warnings.append(message.format(*args, **kwargs) if args or kwargs else message)


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


def test_submit_media_ingest_jobs_preserves_collection_item_binding(
    media_ingest_jobs_client,
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / "jobs.db"))

    captured: list[dict] = []

    from tldw_Server_API.app.core.Jobs import manager as jobs_manager

    def fake_create_job(
        self,
        *,
        payload,
        batch_group=None,
        **_kwargs,
    ):
        captured.append({"payload": payload, "batch_group": batch_group})
        return {"id": len(captured), "uuid": f"u{len(captured)}", "status": "queued"}

    def fake_get_job(self, job_id):
        payload = captured[int(job_id) - 1]["payload"]
        return {
            "id": int(job_id),
            "domain": "media_ingest",
            "job_type": "media_ingest_item",
            "owner_user_id": "1",
            "status": "queued",
            "created_at": "2026-01-01T00:00:00Z",
            "started_at": None,
            "completed_at": None,
            "cancelled_at": None,
            "cancellation_reason": None,
            "progress_percent": 0.0,
            "progress_message": "queued",
            "payload": payload,
            "result": None,
            "error_message": None,
        }

    monkeypatch.setattr(jobs_manager.JobManager, "create_job", fake_create_job, raising=True)
    monkeypatch.setattr(jobs_manager.JobManager, "get_job", fake_get_job, raising=True)

    resp = media_ingest_jobs_client.post(
        "/api/v1/media/ingest/jobs",
        data={
            "media_type": "video",
            "urls": "https://www.youtube.com/watch?v=talk-1",
            "media_collection_id": "42",
            "planned_item_ids": json.dumps(["101"]),
            "idempotency_keys": json.dumps(["conference-42-101-0"]),
        },
        headers={"X-API-KEY": "test-api-key-12345"},
    )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    job = body["jobs"][0]
    assert job["collection_id"] == "42"
    assert job["planned_item_id"] == "101"
    assert job["idempotency_key"] == "conference-42-101-0"
    assert captured[0]["payload"]["collection_id"] == "42"
    assert captured[0]["payload"]["planned_item_id"] == "101"
    assert captured[0]["payload"]["idempotency_key"] == "conference-42-101-0"

    status_resp = media_ingest_jobs_client.get(
        "/api/v1/media/ingest/jobs/1",
        headers={"X-API-KEY": "test-api-key-12345"},
    )

    assert status_resp.status_code == 200, status_resp.text
    status_body = status_resp.json()
    assert status_body["collection_id"] == "42"
    assert status_body["planned_item_id"] == "101"
    assert status_body["idempotency_key"] == "conference-42-101-0"


def test_submit_media_ingest_jobs_rejects_mismatched_collection_item_bindings(
    media_ingest_jobs_client,
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / "jobs.db"))

    from tldw_Server_API.app.core.Jobs import manager as jobs_manager

    def fake_create_job(self, **_kwargs):
        raise AssertionError("job creation should not run when binding arrays do not match urls")

    monkeypatch.setattr(jobs_manager.JobManager, "create_job", fake_create_job, raising=True)

    resp = media_ingest_jobs_client.post(
        "/api/v1/media/ingest/jobs",
        data={
            "media_type": "video",
            "urls": [
                "https://www.youtube.com/watch?v=talk-1",
                "https://www.youtube.com/watch?v=talk-2",
            ],
            "media_collection_id": "42",
            "planned_item_ids": json.dumps(["101"]),
        },
        headers={"X-API-KEY": "test-api-key-12345"},
    )

    assert resp.status_code == 422, resp.text
    assert "planned_item_ids must match the number of URL items" in resp.text


def test_submit_media_ingest_jobs_sanitizes_upload_staging_failure(
    media_ingest_jobs_client,
    monkeypatch,
    tmp_path,
):
    from tldw_Server_API.app.api.v1.endpoints.media import ingest_jobs

    logger_stub = _LoggerStub()

    async def fake_save_uploaded_files(*_args, **_kwargs):
        raise RuntimeError("staging backend exploded at /private/cache/upload.txt")

    monkeypatch.setattr(ingest_jobs, "logger", logger_stub)
    monkeypatch.setattr(
        ingest_jobs,
        "save_uploaded_files",
        fake_save_uploaded_files,
        raising=True,
    )

    upload_path = tmp_path / "sample.txt"
    upload_path.write_text("hello ingest job", encoding="utf-8")

    resp = media_ingest_jobs_client.post(
        "/api/v1/media/ingest/jobs",
        data={"media_type": "document"},
        files=[("files", ("sample.txt", upload_path.read_bytes(), "text/plain"))],
        headers={"X-API-KEY": "test-api-key-12345"},
    )

    assert resp.status_code == 207, resp.text
    body = resp.json()
    assert body["jobs"] == []
    assert body["errors"] == ["Upload staging failed"]
    assert logger_stub.warnings == ["Failed to stage upload for ingest jobs"]
    assert "staging backend exploded" not in str(logger_stub.warnings)
    assert "/private/cache/upload.txt" not in str(logger_stub.warnings)
    assert "staging backend exploded" not in resp.text
    assert "/private/cache/upload.txt" not in resp.text


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
    )
    from tldw_Server_API.app.api.v1.endpoints.media.ingest_jobs import (
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
        raise JobSubmissionLimitError(
            "User 1 has reached the maximum concurrent job limit (5)",
            code="jobs_concurrent_limit",
        )

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


def test_submit_media_ingest_jobs_marks_planned_item_submit_failed_on_job_error(
    media_ingest_jobs_client,
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / "jobs.db"))

    from tldw_Server_API.app.api.v1.API_Deps.Collections_DB_Deps import (
        try_get_collections_db_for_user,
    )
    from tldw_Server_API.app.core.Jobs import manager as jobs_manager

    status_updates: list[dict] = []

    class _FakeCollectionsDatabase:
        def update_media_collection_item_status(self, item_id, **kwargs):
            status_updates.append({"item_id": item_id, **kwargs})

    async def _collections_db_override():
        return _FakeCollectionsDatabase()

    def fake_create_job(self, **_kwargs):
        raise BadRequestError("queue unavailable")

    monkeypatch.setattr(jobs_manager.JobManager, "create_job", fake_create_job, raising=True)
    media_ingest_jobs_client.app.dependency_overrides[try_get_collections_db_for_user] = _collections_db_override

    try:
        resp = media_ingest_jobs_client.post(
            "/api/v1/media/ingest/jobs",
            data={
                "media_type": "video",
                "urls": "https://example.com/talk-submit-fails",
                "media_collection_id": "42",
                "planned_item_ids": json.dumps(["101"]),
            },
            headers={"X-API-KEY": "test-api-key-12345"},
        )

        assert resp.status_code == 400, resp.text
        assert resp.json()["detail"] == "queue unavailable"
        assert status_updates == [
            {
                "item_id": 101,
                "status": "submit_failed",
                "latest_job_id": None,
                "error_summary": "queue unavailable",
            }
        ]
    finally:
        media_ingest_jobs_client.app.dependency_overrides.pop(
            try_get_collections_db_for_user,
            None,
        )


def test_submit_media_ingest_jobs_keeps_created_url_jobs_when_later_url_fails(
    media_ingest_jobs_client,
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / "jobs.db"))

    from tldw_Server_API.app.api.v1.API_Deps.Collections_DB_Deps import (
        try_get_collections_db_for_user,
    )
    from tldw_Server_API.app.core.Jobs import manager as jobs_manager

    captured_payloads: list[dict] = []
    status_updates: list[dict] = []

    class _FakeCollectionsDatabase:
        def update_media_collection_item_status(self, item_id, **kwargs):
            status_updates.append({"item_id": item_id, **kwargs})

    async def _collections_db_override():
        return _FakeCollectionsDatabase()

    def fake_create_job(self, *, payload, **_kwargs):
        captured_payloads.append(payload)
        if payload["source"].endswith("talk-2"):
            raise BadRequestError("queue unavailable")
        return {"id": len(captured_payloads), "uuid": f"u{len(captured_payloads)}", "status": "queued"}

    monkeypatch.setattr(jobs_manager.JobManager, "create_job", fake_create_job, raising=True)
    media_ingest_jobs_client.app.dependency_overrides[try_get_collections_db_for_user] = _collections_db_override

    try:
        resp = media_ingest_jobs_client.post(
            "/api/v1/media/ingest/jobs",
            data={
                "media_type": "video",
                "urls": [
                    "https://example.com/talk-1",
                    "https://example.com/talk-2",
                ],
                "media_collection_id": "42",
                "planned_item_ids": json.dumps(["101", "102"]),
            },
            headers={"X-API-KEY": "test-api-key-12345"},
        )

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["jobs"] == [
            {
                "id": 1,
                "uuid": "u1",
                "source": "https://example.com/talk-1",
                "source_kind": "url",
                "status": "queued",
                "collection_id": "42",
                "planned_item_id": "101",
                "idempotency_key": None,
            }
        ]
        assert body["errors"] == ["https://example.com/talk-2: queue unavailable"]
        assert status_updates == [
            {
                "item_id": 102,
                "status": "submit_failed",
                "latest_job_id": None,
                "error_summary": "queue unavailable",
            }
        ]
    finally:
        media_ingest_jobs_client.app.dependency_overrides.pop(
            try_get_collections_db_for_user,
            None,
        )


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


def _seed_occurrence_run(*, owner_id="1", include_file=False, planned=False):
    from tldw_Server_API.app.api.v1.endpoints.media import ingest_jobs
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_store import (
        PlaylistIngestStore,
    )

    manager = ingest_jobs.get_job_manager()
    store = PlaylistIngestStore(manager)
    items = [
        {
            "occurrence_id": "occ-url-1",
            "input_kind": "direct_url",
            "source_url": "https://www.youtube.com/watch?v=alpha123456",
            "normalized_source_id": "youtube:video:alpha123456",
            "source_kind": "youtube_video",
            "display_metadata": {"title": "Alpha"},
            "state": "staged",
            "action": "ingest",
            "metadata_patch": None,
        },
        {
            "occurrence_id": "occ-url-2",
            "input_kind": "direct_url",
            "source_url": "https://www.youtube.com/watch?v=alpha123456",
            "normalized_source_id": "youtube:video:alpha123456",
            "source_kind": "youtube_video",
            "display_metadata": {"title": "Alpha repeated"},
            "state": "staged",
            "action": "overwrite",
            "metadata_patch": {"title": "Reviewed Alpha"},
        },
    ]
    if include_file:
        items.append(
            {
                "occurrence_id": "occ-file-1",
                "input_kind": "file_stub",
                "source_url": None,
                "normalized_source_id": None,
                "source_kind": "file",
                "display_metadata": {"name": "clip.mp3", "size_bytes": 4},
                "state": "awaiting_upload",
                "action": "ingest",
                "metadata_patch": None,
            }
        )
    run = store.create_validated_run(owner_id, items=items)
    if planned:
        planned_ids = {"occ-url-1": 101, "occ-url-2": 102}
        if include_file:
            planned_ids["occ-file-1"] = 103
        store.attach_collection_plan(owner_id, run.run_id, collection_id=55, planned_item_ids=planned_ids)
    return manager, store, run


def _run_submit_data(run_id, *, urls=None, occurrence_ids=None, attempts=None, planned_item_ids=None):
    data = {
        "media_type": "video",
        "run_id": run_id,
        "urls": urls or ["https://www.youtube.com/watch?v=alpha123456"],
        "occurrence_ids": json.dumps(occurrence_ids or ["occ-url-1"]),
        "attempts": json.dumps(attempts or [1]),
    }
    if planned_item_ids is not None:
        data["planned_item_ids"] = json.dumps(planned_item_ids)
    return data


def test_run_bound_url_submit_uses_server_authority_and_derived_idempotency(
    media_ingest_jobs_client,
):
    manager, store, run = _seed_occurrence_run(planned=True)
    data = _run_submit_data(
        run.run_id,
        urls=[
            "https://www.youtube.com/watch?v=alpha123456",
            "https://www.youtube.com/watch?v=alpha123456",
        ],
        occurrence_ids=["occ-url-1", "occ-url-2"],
        attempts=[1, 1],
        planned_item_ids=[101, 102],
    )
    data["idempotency_keys"] = json.dumps(["attacker-key-1", "attacker-key-2"])

    response = media_ingest_jobs_client.post(
        "/api/v1/media/ingest/jobs",
        data=data,
        headers={"X-API-KEY": "test-api-key-12345"},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert [record["status"] for record in body["submissions"]] == ["accepted", "accepted"]
    assert [record["occurrence_id"] for record in body["submissions"]] == ["occ-url-1", "occ-url-2"]
    assert all(record["accepted"] is True for record in body["submissions"])
    assert all(record["attempt"] == 1 for record in body["submissions"])
    assert all(record["batch_id"] == body["batch_id"] for record in body["submissions"])

    jobs = manager.list_jobs(domain="media_ingest", owner_user_id="1", limit=10)
    assert len(jobs) == 2
    payloads = [json.loads(job["payload"]) if isinstance(job["payload"], str) else job["payload"] for job in jobs]
    assert {payload["occurrence_id"] for payload in payloads} == {"occ-url-1", "occ-url-2"}
    assert {payload["source"] for payload in payloads} == {"https://www.youtube.com/watch?v=alpha123456"}
    assert {payload["run_id"] for payload in payloads} == {run.run_id}
    assert {payload["attempt"] for payload in payloads} == {1}
    assert {payload["planned_item_id"] for payload in payloads} == {101, 102}
    derived = {payload["idempotency_key"] for payload in payloads}
    assert len(derived) == 2
    assert not derived & {"attacker-key-1", "attacker-key-2"}
    assert all(key.startswith("playlist-ingest-v1:") for key in derived)
    assert [item.state for item in store.list_run_items("1", run.run_id)] == ["queued", "queued"]


def test_run_bound_submit_isolates_source_mismatch_without_blocking_valid_occurrence(
    media_ingest_jobs_client,
):
    manager, store, run = _seed_occurrence_run()

    response = media_ingest_jobs_client.post(
        "/api/v1/media/ingest/jobs",
        data=_run_submit_data(
            run.run_id,
            urls=["https://attacker.invalid/wrong", "https://www.youtube.com/watch?v=alpha123456"],
            occurrence_ids=["occ-url-1", "occ-url-2"],
            attempts=[1, 1],
        ),
        headers={"X-API-KEY": "test-api-key-12345"},
    )

    assert response.status_code == 207, response.text
    submissions = response.json()["submissions"]
    assert submissions[0] == {
        "occurrence_id": "occ-url-1",
        "status": "rejected",
        "accepted": False,
        "job_id": None,
        "batch_id": response.json()["batch_id"],
        "error_code": "occurrence_source_mismatch",
        "message": "Submitted source does not match the run occurrence.",
        "retryable": False,
        "attempt": 1,
    }
    assert submissions[1]["status"] == "accepted"
    assert len(manager.list_jobs(domain="media_ingest", owner_user_id="1", limit=10)) == 1
    assert store.get_run_item("1", run.run_id, "occ-url-1").state == "staged"
    assert store.get_run_item("1", run.run_id, "occ-url-2").state == "queued"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("occurrence_ids", json.dumps(["occ-url-1"])),
        ("attempts", json.dumps([1])),
        ("planned_item_ids", json.dumps([101])),
    ],
)
def test_run_bound_url_submit_rejects_misaligned_arrays_before_jobs(
    media_ingest_jobs_client,
    field,
    value,
):
    manager, _store, run = _seed_occurrence_run(planned=True)
    data = _run_submit_data(
        run.run_id,
        urls=[
            "https://www.youtube.com/watch?v=alpha123456",
            "https://www.youtube.com/watch?v=alpha123456",
        ],
        occurrence_ids=["occ-url-1", "occ-url-2"],
        attempts=[1, 1],
        planned_item_ids=[101, 102],
    )
    data[field] = value

    response = media_ingest_jobs_client.post(
        "/api/v1/media/ingest/jobs",
        data=data,
        headers={"X-API-KEY": "test-api-key-12345"},
    )

    assert response.status_code == 422, response.text
    assert len(manager.list_jobs(domain="media_ingest", owner_user_id="1", limit=10)) == 0


@pytest.mark.parametrize("attempt", ["true", "1.0", "0", "-1", "2"])
def test_run_bound_submit_rejects_invalid_or_stale_attempt_strictly(
    media_ingest_jobs_client,
    attempt,
):
    manager, store, run = _seed_occurrence_run()

    response = media_ingest_jobs_client.post(
        "/api/v1/media/ingest/jobs",
        data=_run_submit_data(run.run_id, attempts=[attempt]),
        headers={"X-API-KEY": "test-api-key-12345"},
    )

    assert response.status_code in {207, 422}, response.text
    assert len(manager.list_jobs(domain="media_ingest", owner_user_id="1", limit=10)) == 0
    assert store.get_run_item("1", run.run_id, "occ-url-1").state == "staged"


def test_run_bound_submit_rejects_cross_owner_run_without_leaking_it(
    media_ingest_jobs_client,
):
    manager, _store, run = _seed_occurrence_run(owner_id="other-owner")

    response = media_ingest_jobs_client.post(
        "/api/v1/media/ingest/jobs",
        data=_run_submit_data(run.run_id),
        headers={"X-API-KEY": "test-api-key-12345"},
    )

    assert response.status_code == 404, response.text
    assert response.json()["detail"] == "Playlist ingest run not found."
    assert manager.list_jobs(domain="media_ingest", owner_user_id=None, limit=10) == []


def test_run_bound_submit_treats_server_draining_as_global_before_jobs(
    media_ingest_jobs_client,
    monkeypatch,
):
    manager, store, run = _seed_occurrence_run()
    from fastapi import HTTPException

    from tldw_Server_API.app.api.v1.endpoints.media import ingest_jobs

    def reject_during_drain(_app, _kind):
        raise HTTPException(status_code=503, detail={"code": "server_draining"})

    monkeypatch.setattr(ingest_jobs, "assert_may_start_work", reject_during_drain, raising=True)

    response = media_ingest_jobs_client.post(
        "/api/v1/media/ingest/jobs",
        data=_run_submit_data(run.run_id),
        headers={"X-API-KEY": "test-api-key-12345"},
    )

    assert response.status_code == 503, response.text
    assert manager.list_jobs(domain="media_ingest", owner_user_id="1", limit=10) == []
    assert store.get_run_item("1", run.run_id, "occ-url-1").state == "staged"


@pytest.mark.parametrize(("state", "action"), [("preparing", "skip"), ("terminal", "skip")])
def test_run_bound_submit_rejects_nonprocessing_task6_states(
    media_ingest_jobs_client,
    state,
    action,
):
    manager, store, run = _seed_occurrence_run()
    with store._connection(write=True) as db:
        if state == "terminal":
            store._query(
                db,
                "UPDATE media_ingest_run_items SET state = ?, duplicate_policy = ?, outcome = 'skipped_existing' "
                "WHERE run_id = ? AND occurrence_id = ?",
                (state, action, run.run_id, "occ-url-1"),
            )
        else:
            store._query(
                db,
                "UPDATE media_ingest_run_items SET state = ?, duplicate_policy = ? "
                "WHERE run_id = ? AND occurrence_id = ?",
                (state, action, run.run_id, "occ-url-1"),
            )

    response = media_ingest_jobs_client.post(
        "/api/v1/media/ingest/jobs",
        data=_run_submit_data(run.run_id),
        headers={"X-API-KEY": "test-api-key-12345"},
    )

    assert response.status_code == 207, response.text
    assert response.json()["submissions"][0]["error_code"] == "occurrence_not_processable"
    assert manager.list_jobs(domain="media_ingest", owner_user_id="1", limit=10) == []


def test_run_bound_submit_validates_planned_id_against_stored_authority(
    media_ingest_jobs_client,
):
    manager, store, run = _seed_occurrence_run(planned=True)

    response = media_ingest_jobs_client.post(
        "/api/v1/media/ingest/jobs",
        data=_run_submit_data(run.run_id, planned_item_ids=[999]),
        headers={"X-API-KEY": "test-api-key-12345"},
    )

    assert response.status_code == 207, response.text
    assert response.json()["submissions"][0]["error_code"] == "planned_item_mismatch"
    assert manager.list_jobs(domain="media_ingest", owner_user_id="1", limit=10) == []
    assert store.get_run_item("1", run.run_id, "occ-url-1").state == "staged"


def test_run_bound_submit_rejects_bool_like_planned_id_strictly(media_ingest_jobs_client):
    manager, store, run = _seed_occurrence_run(planned=True)
    data = _run_submit_data(run.run_id)
    data["planned_item_ids"] = json.dumps([True])

    response = media_ingest_jobs_client.post(
        "/api/v1/media/ingest/jobs",
        data=data,
        headers={"X-API-KEY": "test-api-key-12345"},
    )

    assert response.status_code == 422, response.text
    assert manager.list_jobs(domain="media_ingest", owner_user_id="1", limit=10) == []
    assert store.get_run_item("1", run.run_id, "occ-url-1").state == "staged"


def test_run_bound_submit_rejects_more_than_500_items_before_run_lookup(media_ingest_jobs_client):
    urls = [f"https://example.com/video/{index}" for index in range(501)]

    response = media_ingest_jobs_client.post(
        "/api/v1/media/ingest/jobs",
        data={
            "media_type": "video",
            "run_id": "does-not-exist",
            "urls": urls,
            "occurrence_ids": json.dumps([f"occ-{index}" for index in range(501)]),
            "attempts": json.dumps([1] * 501),
        },
        headers={"X-API-KEY": "test-api-key-12345"},
    )

    assert response.status_code == 422, response.text
    assert "between 1 and 500" in response.text


def test_run_bound_file_submit_aligns_identity_before_staging(
    media_ingest_jobs_client,
    tmp_path,
):
    manager, store, run = _seed_occurrence_run(include_file=True, planned=True)

    response = media_ingest_jobs_client.post(
        "/api/v1/media/ingest/jobs",
        data={
            "media_type": "audio",
            "run_id": run.run_id,
            "file_occurrence_ids": json.dumps(["occ-file-1"]),
            "file_attempts": json.dumps([1]),
            "file_planned_item_ids": json.dumps([103]),
        },
        files=[("files", ("clip.mp3", b"test", "audio/mpeg"))],
        headers={"X-API-KEY": "test-api-key-12345"},
    )

    assert response.status_code == 200, response.text
    record = response.json()["submissions"][0]
    assert record["occurrence_id"] == "occ-file-1"
    assert record["status"] == "accepted"
    job = manager.get_job(record["job_id"])
    payload = json.loads(job["payload"]) if isinstance(job["payload"], str) else job["payload"]
    assert payload["run_id"] == run.run_id
    assert payload["occurrence_id"] == "occ-file-1"
    assert payload["attempt"] == 1
    assert payload["planned_item_id"] == 103
    assert store.get_run_item("1", run.run_id, "occ-file-1").state == "queued"
    shutil.rmtree(payload["temp_dir"], ignore_errors=True)


def test_run_bound_file_length_mismatch_does_not_stage_or_leak(
    media_ingest_jobs_client,
    monkeypatch,
):
    manager, store, run = _seed_occurrence_run(include_file=True)
    from tldw_Server_API.app.api.v1.endpoints.media import ingest_jobs

    stage_calls = 0

    async def fail_if_staged(*_args, **_kwargs):
        nonlocal stage_calls
        stage_calls += 1
        raise AssertionError("length validation must happen before upload staging")

    monkeypatch.setattr(ingest_jobs, "save_uploaded_files", fail_if_staged, raising=True)

    response = media_ingest_jobs_client.post(
        "/api/v1/media/ingest/jobs",
        data={
            "media_type": "audio",
            "run_id": run.run_id,
            "file_occurrence_ids": json.dumps([]),
            "file_attempts": json.dumps([1]),
        },
        files=[("files", ("clip.mp3", b"test", "audio/mpeg"))],
        headers={"X-API-KEY": "test-api-key-12345"},
    )

    assert response.status_code == 422, response.text
    assert stage_calls == 0
    assert manager.list_jobs(domain="media_ingest", owner_user_id="1", limit=10) == []
    assert store.get_run_item("1", run.run_id, "occ-file-1").state == "awaiting_upload"


def test_run_bound_file_job_failure_cleans_own_staging_and_preserves_reservation_for_retry(
    media_ingest_jobs_client,
    monkeypatch,
):
    manager, store, run = _seed_occurrence_run(include_file=True)
    from tldw_Server_API.app.api.v1.endpoints.media import ingest_jobs

    cleaned: list[str] = []
    create_calls: list[dict] = []
    original_cleanup = ingest_jobs._cleanup_dir
    original_create = ingest_jobs._create_media_ingest_job

    def record_cleanup(path):
        cleaned.append(path)
        original_cleanup(path)

    def fail_once(**kwargs):
        create_calls.append(kwargs)
        if len(create_calls) == 1:
            raise RuntimeError("private token=do-not-leak")
        return original_create(**kwargs)

    monkeypatch.setattr(ingest_jobs, "_cleanup_dir", record_cleanup, raising=True)
    monkeypatch.setattr(ingest_jobs, "_create_media_ingest_job", fail_once, raising=True)

    data = {
        "media_type": "audio",
        "run_id": run.run_id,
        "file_occurrence_ids": json.dumps(["occ-file-1"]),
        "file_attempts": json.dumps([1]),
    }

    first = media_ingest_jobs_client.post(
        "/api/v1/media/ingest/jobs",
        data=data,
        files=[("files", ("clip.mp3", b"test", "audio/mpeg"))],
        headers={"X-API-KEY": "test-api-key-12345"},
    )
    pending = store.get_run_item("1", run.run_id, "occ-file-1")
    second = media_ingest_jobs_client.post(
        "/api/v1/media/ingest/jobs",
        data=data,
        files=[("files", ("clip.mp3", b"next", "audio/mpeg"))],
        headers={"X-API-KEY": "test-api-key-12345"},
    )

    assert first.status_code == 207, first.text
    assert first.json()["submissions"][0]["error_code"] == "job_submission_failed"
    assert "do-not-leak" not in first.text
    assert pending.state == "submit_pending"
    assert pending.batch_id == first.json()["submissions"][0]["batch_id"]
    assert second.status_code == 200, second.text
    assert len(create_calls) == 2
    assert {call["batch_id"] for call in create_calls} == {pending.batch_id}
    assert cleaned and not Path(cleaned[0]).exists()
    jobs = manager.list_jobs(domain="media_ingest", owner_user_id="1", limit=10)
    assert len(jobs) == 1
    payload = json.loads(jobs[0]["payload"]) if isinstance(jobs[0]["payload"], str) else jobs[0]["payload"]
    assert Path(payload["source"]).read_bytes() == b"next"
    assert store.get_run_item("1", run.run_id, "occ-file-1").job_id == jobs[0]["id"]
    shutil.rmtree(payload["temp_dir"], ignore_errors=True)


def test_run_bound_upload_staging_failure_preserves_reservation_for_retry(
    media_ingest_jobs_client,
    monkeypatch,
):
    manager, store, run = _seed_occurrence_run(include_file=True)
    from tldw_Server_API.app.api.v1.endpoints.media import ingest_jobs

    original_save = ingest_jobs.save_uploaded_files
    staged_dirs: list[Path] = []
    save_calls = 0

    async def fail_once(*args, temp_dir, **kwargs):
        nonlocal save_calls
        save_calls += 1
        staged_dirs.append(Path(temp_dir))
        if save_calls == 1:
            raise RuntimeError("private staging failure")
        return await original_save(*args, temp_dir=temp_dir, **kwargs)

    monkeypatch.setattr(ingest_jobs, "save_uploaded_files", fail_once, raising=True)
    data = {
        "media_type": "audio",
        "run_id": run.run_id,
        "file_occurrence_ids": json.dumps(["occ-file-1"]),
        "file_attempts": json.dumps([1]),
    }

    first = media_ingest_jobs_client.post(
        "/api/v1/media/ingest/jobs",
        data=data,
        files=[("files", ("clip.mp3", b"old!", "audio/mpeg"))],
        headers={"X-API-KEY": "test-api-key-12345"},
    )
    pending = store.get_run_item("1", run.run_id, "occ-file-1")
    second = media_ingest_jobs_client.post(
        "/api/v1/media/ingest/jobs",
        data=data,
        files=[("files", ("clip.mp3", b"new!", "audio/mpeg"))],
        headers={"X-API-KEY": "test-api-key-12345"},
    )

    assert first.status_code == 207, first.text
    assert first.json()["submissions"][0]["error_code"] == "upload_staging_failed"
    assert "private" not in first.text
    assert pending.state == "submit_pending"
    assert pending.batch_id == first.json()["submissions"][0]["batch_id"]
    assert len(staged_dirs) == 2
    assert not staged_dirs[0].exists()
    assert second.status_code == 200, second.text
    assert second.json()["submissions"][0]["batch_id"] == pending.batch_id
    job = manager.get_job(second.json()["submissions"][0]["job_id"])
    payload = json.loads(job["payload"]) if isinstance(job["payload"], str) else job["payload"]
    assert Path(payload["source"]).read_bytes() == b"new!"
    assert Path(payload["temp_dir"]) == staged_dirs[1]
    shutil.rmtree(payload["temp_dir"], ignore_errors=True)


@pytest.mark.parametrize(
    "url",
    [
        "https://www.youtube.com/playlist?list=PLopaque",
        "https://youtu.be/alpha123456?list=PLopaque",
    ],
)
def test_submit_media_ingest_jobs_rejects_opaque_playlist_with_safe_422(
    media_ingest_jobs_client,
    url,
):
    from tldw_Server_API.app.api.v1.endpoints.media import ingest_jobs

    response = media_ingest_jobs_client.post(
        "/api/v1/media/ingest/jobs",
        data={"media_type": "video", "urls": url},
        headers={"X-API-KEY": "test-api-key-12345"},
    )

    assert response.status_code == 422, response.text
    assert response.json()["detail"] == {
        "code": "playlist_preflight_required",
        "message": "Playlist URLs must be inspected before job submission.",
    }
    assert ingest_jobs.get_job_manager().list_jobs(domain="media_ingest", owner_user_id="1", limit=10) == []


def test_run_bound_ambiguous_repeat_reconciles_original_job(
    media_ingest_jobs_client,
    monkeypatch,
):
    manager, store, run = _seed_occurrence_run()
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_store import (
        PlaylistIngestStore,
    )

    original_bind = PlaylistIngestStore.bind_run_item_job
    calls = 0

    def fail_first_bind(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("commit outcome unknown: token=private")
        return original_bind(self, *args, **kwargs)

    monkeypatch.setattr(PlaylistIngestStore, "bind_run_item_job", fail_first_bind, raising=True)
    data = _run_submit_data(run.run_id)

    first = media_ingest_jobs_client.post(
        "/api/v1/media/ingest/jobs",
        data=data,
        headers={"X-API-KEY": "test-api-key-12345"},
    )
    second = media_ingest_jobs_client.post(
        "/api/v1/media/ingest/jobs",
        data=data,
        headers={"X-API-KEY": "test-api-key-12345"},
    )

    assert first.status_code == 207, first.text
    assert first.json()["submissions"][0]["error_code"] == "occurrence_binding_pending"
    assert "private" not in first.text
    assert second.status_code == 200, second.text
    jobs = manager.list_jobs(domain="media_ingest", owner_user_id="1", limit=10)
    assert len(jobs) == 1
    assert second.json()["submissions"][0]["job_id"] == jobs[0]["id"]
    assert store.get_run_item("1", run.run_id, "occ-url-1").job_id == jobs[0]["id"]


@pytest.mark.parametrize(
    "occurrences",
    [
        [1],
        [True],
        [1.5],
        [{}],
        [[]],
        [None],
        [""],
        [" occ-url-1"],
        ["occ-url-1 "],
        ["x" * 256],
    ],
)
def test_run_bound_url_identity_array_rejects_noncanonical_elements_before_run_lookup(
    media_ingest_jobs_client,
    occurrences,
):
    response = media_ingest_jobs_client.post(
        "/api/v1/media/ingest/jobs",
        data={
            "media_type": "video",
            "run_id": "missing-run",
            "urls": "https://www.youtube.com/watch?v=alpha123456",
            "occurrence_ids": json.dumps(occurrences),
            "attempts": json.dumps([1]),
        },
        headers={"X-API-KEY": "test-api-key-12345"},
    )

    assert response.status_code == 422, response.text
    assert response.json()["detail"] == "occurrence_ids contains an invalid identifier."


@pytest.mark.parametrize(
    "attempts",
    [
        [True],
        [1.0],
        [{}],
        [[]],
        [None],
        ["+1"],
        ["-1"],
        [" 1"],
        ["1 "],
        ["01"],
        ["9" * 5000],
    ],
)
def test_run_bound_url_integer_array_rejects_ambiguous_values_before_run_lookup(
    media_ingest_jobs_client,
    attempts,
):
    response = media_ingest_jobs_client.post(
        "/api/v1/media/ingest/jobs",
        data={
            "media_type": "video",
            "run_id": "missing-run",
            "urls": "https://www.youtube.com/watch?v=alpha123456",
            "occurrence_ids": json.dumps(["occ-url-1"]),
            "attempts": json.dumps(attempts),
        },
        headers={"X-API-KEY": "test-api-key-12345"},
    )

    assert response.status_code == 422, response.text
    assert response.json()["detail"] == "attempts must contain positive integers."


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("file_occurrence_ids", [1]),
        ("file_occurrence_ids", [" occ-file-1"]),
        ("file_occurrence_ids", [{}]),
        ("file_attempts", [" 1"]),
        ("file_attempts", [True]),
        ("file_attempts", ["9" * 5000]),
        ("file_planned_item_ids", [1.0]),
        ("file_planned_item_ids", [{}]),
    ],
)
def test_run_bound_file_arrays_reject_invalid_elements_before_staging(
    media_ingest_jobs_client,
    monkeypatch,
    field,
    value,
):
    from tldw_Server_API.app.api.v1.endpoints.media import ingest_jobs

    stage_calls = 0

    async def fail_if_staged(*_args, **_kwargs):
        nonlocal stage_calls
        stage_calls += 1
        raise AssertionError("strict array validation must happen before staging")

    monkeypatch.setattr(ingest_jobs, "save_uploaded_files", fail_if_staged, raising=True)
    data = {
        "media_type": "audio",
        "run_id": "missing-run",
        "file_occurrence_ids": json.dumps(["occ-file-1"]),
        "file_attempts": json.dumps([1]),
    }
    data[field] = json.dumps(value)

    response = media_ingest_jobs_client.post(
        "/api/v1/media/ingest/jobs",
        data=data,
        files=[("files", ("clip.mp3", b"test", "audio/mpeg"))],
        headers={"X-API-KEY": "test-api-key-12345"},
    )

    assert response.status_code == 422, response.text
    assert stage_calls == 0


def test_run_bound_url_commit_then_throw_reconciles_original_job(
    media_ingest_jobs_client,
    monkeypatch,
):
    manager, store, run = _seed_occurrence_run()
    from tldw_Server_API.app.api.v1.endpoints.media import ingest_jobs

    original_create = ingest_jobs._create_media_ingest_job

    def commit_then_throw(**kwargs):
        original_create(**kwargs)
        raise RuntimeError("commit outcome unknown: token=private")

    monkeypatch.setattr(ingest_jobs, "_create_media_ingest_job", commit_then_throw, raising=True)

    response = media_ingest_jobs_client.post(
        "/api/v1/media/ingest/jobs",
        data=_run_submit_data(run.run_id),
        headers={"X-API-KEY": "test-api-key-12345"},
    )

    assert response.status_code == 200, response.text
    jobs = manager.list_jobs(domain="media_ingest", owner_user_id="1", limit=10)
    assert len(jobs) == 1
    assert response.json()["submissions"][0]["job_id"] == jobs[0]["id"]
    assert store.get_run_item("1", run.run_id, "occ-url-1").job_id == jobs[0]["id"]
    assert "private" not in response.text


def test_run_bound_url_commit_then_http_503_reconciles_original_job(
    media_ingest_jobs_client,
    monkeypatch,
):
    manager, store, run = _seed_occurrence_run()
    from tldw_Server_API.app.api.v1.endpoints.media import ingest_jobs

    original_create = ingest_jobs._create_media_ingest_job

    def commit_then_unavailable(**kwargs):
        original_create(**kwargs)
        raise HTTPException(status_code=503, detail="job store unavailable")

    monkeypatch.setattr(ingest_jobs, "_create_media_ingest_job", commit_then_unavailable, raising=True)

    response = media_ingest_jobs_client.post(
        "/api/v1/media/ingest/jobs",
        data=_run_submit_data(run.run_id),
        headers={"X-API-KEY": "test-api-key-12345"},
    )

    assert response.status_code == 200, response.text
    jobs = manager.list_jobs(domain="media_ingest", owner_user_id="1", limit=10)
    assert len(jobs) == 1
    assert response.json()["submissions"][0]["job_id"] == jobs[0]["id"]
    assert store.get_run_item("1", run.run_id, "occ-url-1").job_id == jobs[0]["id"]


def test_confirmed_url_create_failure_preserves_reservation_for_retry(
    media_ingest_jobs_client,
    monkeypatch,
):
    manager, store, run = _seed_occurrence_run()
    from tldw_Server_API.app.api.v1.endpoints.media import ingest_jobs

    original_create = ingest_jobs._create_media_ingest_job
    create_calls: list[dict] = []

    def fail_once(**kwargs):
        create_calls.append(kwargs)
        if len(create_calls) == 1:
            raise RuntimeError("job was not created")
        return original_create(**kwargs)

    monkeypatch.setattr(ingest_jobs, "_create_media_ingest_job", fail_once, raising=True)
    data = _run_submit_data(run.run_id)

    first = media_ingest_jobs_client.post(
        "/api/v1/media/ingest/jobs",
        data=data,
        headers={"X-API-KEY": "test-api-key-12345"},
    )
    pending = store.get_run_item("1", run.run_id, "occ-url-1")
    second = media_ingest_jobs_client.post(
        "/api/v1/media/ingest/jobs",
        data=data,
        headers={"X-API-KEY": "test-api-key-12345"},
    )

    assert first.status_code == 207, first.text
    assert first.json()["submissions"][0]["error_code"] == "job_submission_failed"
    assert pending.state == "submit_pending"
    assert pending.batch_id == first.json()["submissions"][0]["batch_id"]
    assert second.status_code == 200, second.text
    assert len(create_calls) == 2
    assert {call["batch_id"] for call in create_calls} == {pending.batch_id}
    assert len({call["payload"]["idempotency_key"] for call in create_calls}) == 1
    jobs = manager.list_jobs(domain="media_ingest", owner_user_id="1", limit=10)
    assert len(jobs) == 1
    assert store.get_run_item("1", run.run_id, "occ-url-1").job_id == jobs[0]["id"]


def test_ambiguous_create_and_lookup_failure_preserves_pending_reservation(
    media_ingest_jobs_client,
    monkeypatch,
):
    manager, store, run = _seed_occurrence_run()
    from tldw_Server_API.app.api.v1.endpoints.media import ingest_jobs
    from tldw_Server_API.app.core.Jobs import manager as jobs_manager

    def ambiguous_create(**_kwargs):
        raise RuntimeError("job commit outcome unavailable")

    def unavailable_lookup(self, **_kwargs):
        raise RuntimeError("jobs lookup unavailable")

    monkeypatch.setattr(ingest_jobs, "_create_media_ingest_job", ambiguous_create, raising=True)
    monkeypatch.setattr(jobs_manager.JobManager, "get_job_by_idempotency", unavailable_lookup, raising=True)

    response = media_ingest_jobs_client.post(
        "/api/v1/media/ingest/jobs",
        data=_run_submit_data(run.run_id),
        headers={"X-API-KEY": "test-api-key-12345"},
    )

    assert response.status_code == 207, response.text
    assert response.json()["submissions"][0]["error_code"] == "occurrence_binding_pending"
    pending = store.get_run_item("1", run.run_id, "occ-url-1")
    assert pending.state == "submit_pending"
    assert pending.batch_id == response.json()["submissions"][0]["batch_id"]
    assert manager.list_jobs(domain="media_ingest", owner_user_id="1", limit=10) == []


def test_pending_url_retry_reuses_stored_reservation_and_creates_idempotently(
    media_ingest_jobs_client,
    monkeypatch,
):
    manager, store, run = _seed_occurrence_run()
    from tldw_Server_API.app.api.v1.endpoints.media import ingest_jobs
    from tldw_Server_API.app.core.Jobs import manager as jobs_manager

    original_create = ingest_jobs._create_media_ingest_job
    original_lookup = jobs_manager.JobManager.get_job_by_idempotency
    create_calls: list[dict] = []
    lookup_calls = 0

    def create_after_ambiguous_first_request(**kwargs):
        create_calls.append(kwargs)
        if len(create_calls) == 1:
            raise RuntimeError("job commit outcome unavailable")
        return original_create(**kwargs)

    def lookup_unavailable_then_empty(self, **kwargs):
        nonlocal lookup_calls
        lookup_calls += 1
        if lookup_calls == 1:
            raise RuntimeError("jobs lookup unavailable")
        return original_lookup(self, **kwargs)

    monkeypatch.setattr(ingest_jobs, "_create_media_ingest_job", create_after_ambiguous_first_request, raising=True)
    monkeypatch.setattr(
        jobs_manager.JobManager,
        "get_job_by_idempotency",
        lookup_unavailable_then_empty,
        raising=True,
    )
    data = _run_submit_data(run.run_id)

    first = media_ingest_jobs_client.post(
        "/api/v1/media/ingest/jobs",
        data=data,
        headers={"X-API-KEY": "test-api-key-12345"},
    )
    pending = store.get_run_item("1", run.run_id, "occ-url-1")
    second = media_ingest_jobs_client.post(
        "/api/v1/media/ingest/jobs",
        data=data,
        headers={"X-API-KEY": "test-api-key-12345"},
    )

    assert first.status_code == 207, first.text
    assert first.json()["submissions"][0]["error_code"] == "occurrence_binding_pending"
    assert pending.state == "submit_pending"
    assert second.status_code == 200, second.text
    assert len(create_calls) == 2
    assert create_calls[1]["batch_id"] == pending.batch_id
    assert create_calls[1]["payload"]["batch_id"] == pending.batch_id
    assert create_calls[1]["payload"]["idempotency_key"] == create_calls[0]["payload"]["idempotency_key"]
    jobs = manager.list_jobs(domain="media_ingest", owner_user_id="1", limit=10)
    assert len(jobs) == 1
    assert second.json()["submissions"][0]["batch_id"] == pending.batch_id
    assert store.get_run_item("1", run.run_id, "occ-url-1").job_id == jobs[0]["id"]


def test_pending_url_retry_remains_pending_when_create_is_ambiguous_again(
    media_ingest_jobs_client,
    monkeypatch,
):
    manager, store, run = _seed_occurrence_run()
    from tldw_Server_API.app.api.v1.endpoints.media import ingest_jobs
    from tldw_Server_API.app.core.Jobs import manager as jobs_manager

    original_lookup = jobs_manager.JobManager.get_job_by_idempotency
    create_calls: list[dict] = []
    lookup_calls = 0

    def ambiguous_create(**kwargs):
        create_calls.append(kwargs)
        raise RuntimeError("job commit outcome unavailable")

    def unavailable_empty_unavailable(self, **kwargs):
        nonlocal lookup_calls
        lookup_calls += 1
        if lookup_calls in {1, 3}:
            raise RuntimeError("jobs lookup unavailable")
        return original_lookup(self, **kwargs)

    monkeypatch.setattr(ingest_jobs, "_create_media_ingest_job", ambiguous_create, raising=True)
    monkeypatch.setattr(
        jobs_manager.JobManager,
        "get_job_by_idempotency",
        unavailable_empty_unavailable,
        raising=True,
    )
    data = _run_submit_data(run.run_id)

    first = media_ingest_jobs_client.post(
        "/api/v1/media/ingest/jobs",
        data=data,
        headers={"X-API-KEY": "test-api-key-12345"},
    )
    pending_batch = store.get_run_item("1", run.run_id, "occ-url-1").batch_id
    second = media_ingest_jobs_client.post(
        "/api/v1/media/ingest/jobs",
        data=data,
        headers={"X-API-KEY": "test-api-key-12345"},
    )

    assert first.status_code == 207, first.text
    assert second.status_code == 207, second.text
    assert second.json()["submissions"][0]["error_code"] == "occurrence_binding_pending"
    assert len(create_calls) == 2
    assert {call["batch_id"] for call in create_calls} == {pending_batch}
    assert len({call["payload"]["idempotency_key"] for call in create_calls}) == 1
    pending = store.get_run_item("1", run.run_id, "occ-url-1")
    assert pending.state == "submit_pending"
    assert pending.batch_id == pending_batch
    assert manager.list_jobs(domain="media_ingest", owner_user_id="1", limit=10) == []


def test_run_bound_file_commit_then_throw_preserves_accepted_staging(
    media_ingest_jobs_client,
    monkeypatch,
):
    manager, store, run = _seed_occurrence_run(include_file=True)
    from tldw_Server_API.app.api.v1.endpoints.media import ingest_jobs

    original_create = ingest_jobs._create_media_ingest_job

    def commit_then_throw(**kwargs):
        original_create(**kwargs)
        raise RuntimeError("commit outcome unknown: token=private")

    monkeypatch.setattr(ingest_jobs, "_create_media_ingest_job", commit_then_throw, raising=True)

    response = media_ingest_jobs_client.post(
        "/api/v1/media/ingest/jobs",
        data={
            "media_type": "audio",
            "run_id": run.run_id,
            "file_occurrence_ids": json.dumps(["occ-file-1"]),
            "file_attempts": json.dumps([1]),
        },
        files=[("files", ("clip.mp3", b"test", "audio/mpeg"))],
        headers={"X-API-KEY": "test-api-key-12345"},
    )

    assert response.status_code == 200, response.text
    record = response.json()["submissions"][0]
    job = manager.get_job(record["job_id"])
    payload = json.loads(job["payload"]) if isinstance(job["payload"], str) else job["payload"]
    assert Path(payload["source"]).exists()
    assert store.get_run_item("1", run.run_id, "occ-file-1").job_id == record["job_id"]
    assert "private" not in response.text
    shutil.rmtree(payload["temp_dir"], ignore_errors=True)


def test_pending_file_retry_uses_current_validated_bytes_and_cleans_old_staging(
    media_ingest_jobs_client,
    monkeypatch,
):
    manager, store, run = _seed_occurrence_run(include_file=True)
    from tldw_Server_API.app.api.v1.endpoints.media import ingest_jobs
    from tldw_Server_API.app.core.Jobs import manager as jobs_manager

    original_create = ingest_jobs._create_media_ingest_job
    original_lookup = jobs_manager.JobManager.get_job_by_idempotency
    original_save = ingest_jobs.save_uploaded_files
    create_calls: list[dict] = []
    staged_dirs: list[Path] = []
    lookup_calls = 0

    def create_after_ambiguous_first_request(**kwargs):
        create_calls.append(kwargs)
        if len(create_calls) == 1:
            raise RuntimeError("job commit outcome unavailable")
        return original_create(**kwargs)

    def lookup_unavailable_then_empty(self, **kwargs):
        nonlocal lookup_calls
        lookup_calls += 1
        if lookup_calls == 1:
            raise RuntimeError("jobs lookup unavailable")
        return original_lookup(self, **kwargs)

    async def capture_staging(*args, temp_dir, **kwargs):
        staged_dirs.append(Path(temp_dir))
        return await original_save(*args, temp_dir=temp_dir, **kwargs)

    monkeypatch.setattr(ingest_jobs, "_create_media_ingest_job", create_after_ambiguous_first_request, raising=True)
    monkeypatch.setattr(ingest_jobs, "save_uploaded_files", capture_staging, raising=True)
    monkeypatch.setattr(
        jobs_manager.JobManager,
        "get_job_by_idempotency",
        lookup_unavailable_then_empty,
        raising=True,
    )
    data = {
        "media_type": "audio",
        "run_id": run.run_id,
        "file_occurrence_ids": json.dumps(["occ-file-1"]),
        "file_attempts": json.dumps([1]),
    }

    first = media_ingest_jobs_client.post(
        "/api/v1/media/ingest/jobs",
        data=data,
        files=[("files", ("clip.mp3", b"old!", "audio/mpeg"))],
        headers={"X-API-KEY": "test-api-key-12345"},
    )
    pending_batch = store.get_run_item("1", run.run_id, "occ-file-1").batch_id
    assert first.status_code == 207, first.text
    assert len(staged_dirs) == 1
    assert staged_dirs[0].exists()

    second = media_ingest_jobs_client.post(
        "/api/v1/media/ingest/jobs",
        data=data,
        files=[("files", ("clip.mp3", b"new!", "audio/mpeg"))],
        headers={"X-API-KEY": "test-api-key-12345"},
    )

    assert second.status_code == 200, second.text
    assert len(create_calls) == 2
    assert len(staged_dirs) == 2
    job = manager.get_job(second.json()["submissions"][0]["job_id"])
    payload = json.loads(job["payload"]) if isinstance(job["payload"], str) else job["payload"]
    assert create_calls[1]["batch_id"] == pending_batch
    assert Path(payload["source"]).read_bytes() == b"new!"
    assert Path(payload["temp_dir"]) == staged_dirs[1]
    assert not staged_dirs[0].exists()
    assert staged_dirs[1].exists()
    assert store.get_run_item("1", run.run_id, "occ-file-1").job_id == job["id"]
    shutil.rmtree(payload["temp_dir"], ignore_errors=True)


def test_pending_file_retry_preserves_staging_of_job_found_during_reconciliation(
    media_ingest_jobs_client,
    monkeypatch,
):
    manager, store, run = _seed_occurrence_run(include_file=True)
    from tldw_Server_API.app.api.v1.endpoints.media import ingest_jobs
    from tldw_Server_API.app.core.Jobs import manager as jobs_manager

    original_create = ingest_jobs._create_media_ingest_job
    original_lookup = jobs_manager.JobManager.get_job_by_idempotency
    original_save = ingest_jobs.save_uploaded_files
    staged_dirs: list[Path] = []
    lookup_calls = 0

    def commit_then_throw(**kwargs):
        original_create(**kwargs)
        raise RuntimeError("job commit outcome unavailable")

    def unavailable_then_reconcile(self, **kwargs):
        nonlocal lookup_calls
        lookup_calls += 1
        if lookup_calls == 1:
            raise RuntimeError("jobs lookup unavailable")
        return original_lookup(self, **kwargs)

    async def capture_staging(*args, temp_dir, **kwargs):
        staged_dirs.append(Path(temp_dir))
        return await original_save(*args, temp_dir=temp_dir, **kwargs)

    monkeypatch.setattr(ingest_jobs, "_create_media_ingest_job", commit_then_throw, raising=True)
    monkeypatch.setattr(ingest_jobs, "save_uploaded_files", capture_staging, raising=True)
    monkeypatch.setattr(
        jobs_manager.JobManager,
        "get_job_by_idempotency",
        unavailable_then_reconcile,
        raising=True,
    )
    data = {
        "media_type": "audio",
        "run_id": run.run_id,
        "file_occurrence_ids": json.dumps(["occ-file-1"]),
        "file_attempts": json.dumps([1]),
    }

    first = media_ingest_jobs_client.post(
        "/api/v1/media/ingest/jobs",
        data=data,
        files=[("files", ("clip.mp3", b"kept", "audio/mpeg"))],
        headers={"X-API-KEY": "test-api-key-12345"},
    )
    second = media_ingest_jobs_client.post(
        "/api/v1/media/ingest/jobs",
        data=data,
        files=[("files", ("clip.mp3", b"unused", "audio/mpeg"))],
        headers={"X-API-KEY": "test-api-key-12345"},
    )

    assert first.status_code == 207, first.text
    assert second.status_code == 200, second.text
    assert len(staged_dirs) == 1
    job = manager.get_job(second.json()["submissions"][0]["job_id"])
    payload = json.loads(job["payload"]) if isinstance(job["payload"], str) else job["payload"]
    assert Path(payload["temp_dir"]) == staged_dirs[0]
    assert Path(payload["source"]).read_bytes() == b"kept"
    assert staged_dirs[0].exists()
    assert store.get_run_item("1", run.run_id, "occ-file-1").job_id == job["id"]
    shutil.rmtree(payload["temp_dir"], ignore_errors=True)


def test_concurrent_retry_cooperatively_creates_without_replacing_reservation(
    media_ingest_jobs_client,
    monkeypatch,
):
    manager, store, run = _seed_occurrence_run()
    from tldw_Server_API.app.api.v1.endpoints.media import ingest_jobs

    original_create = ingest_jobs._create_media_ingest_job
    first_entered = Event()
    release_first = Event()
    calls_lock = Lock()
    create_calls = 0
    create_inputs: list[tuple[str, str]] = []

    def controlled_create(**kwargs):
        nonlocal create_calls
        with calls_lock:
            create_calls += 1
            call_number = create_calls
            create_inputs.append((kwargs["batch_id"], kwargs["payload"]["idempotency_key"]))
        if call_number == 1:
            first_entered.set()
            assert release_first.wait(10)
        return original_create(**kwargs)

    monkeypatch.setattr(ingest_jobs, "_create_media_ingest_job", controlled_create, raising=True)
    data = _run_submit_data(run.run_id)

    with ThreadPoolExecutor(max_workers=1) as pool:
        first_future = pool.submit(
            media_ingest_jobs_client.post,
            "/api/v1/media/ingest/jobs",
            data=data,
            headers={"X-API-KEY": "test-api-key-12345"},
        )
        assert first_entered.wait(10)
        try:
            with TestClient(media_ingest_jobs_client.app) as second_client:
                second = second_client.post(
                    "/api/v1/media/ingest/jobs",
                    data=data,
                    headers={"X-API-KEY": "test-api-key-12345"},
                )
        finally:
            release_first.set()
        first = first_future.result(timeout=10)

    assert second.status_code == 200, second.text
    assert first.status_code == 200, first.text
    assert create_calls == 2
    assert len(set(create_inputs)) == 1
    jobs = manager.list_jobs(domain="media_ingest", owner_user_id="1", limit=10)
    assert len(jobs) == 1
    assert first.json()["submissions"][0]["job_id"] == jobs[0]["id"]
    assert second.json()["submissions"][0]["job_id"] == jobs[0]["id"]
    assert first.json()["submissions"][0]["batch_id"] == create_inputs[0][0]
    assert second.json()["submissions"][0]["batch_id"] == create_inputs[0][0]
    assert store.get_run_item("1", run.run_id, "occ-url-1").job_id == jobs[0]["id"]


def test_original_failure_cannot_release_reservation_after_retry_creates_job(
    media_ingest_jobs_client,
    monkeypatch,
):
    manager, store, run = _seed_occurrence_run()
    from tldw_Server_API.app.api.v1.endpoints.media import ingest_jobs

    original_create = ingest_jobs._create_media_ingest_job
    first_entered = Event()
    retry_created = Event()
    release_retry_bind = Event()
    calls_lock = Lock()
    create_calls = 0
    lookup_calls = 0
    create_inputs: list[tuple[str, str]] = []

    def controlled_create(**kwargs):
        nonlocal create_calls
        with calls_lock:
            create_calls += 1
            call_number = create_calls
            create_inputs.append((kwargs["batch_id"], kwargs["payload"]["idempotency_key"]))
        if call_number == 1:
            first_entered.set()
            assert retry_created.wait(10)
            raise RuntimeError("original submission did not create a job")
        row = original_create(**kwargs)
        retry_created.set()
        assert release_retry_bind.wait(10)
        return row

    def controlled_lookup(*_args, **_kwargs):
        nonlocal lookup_calls
        with calls_lock:
            lookup_calls += 1
            call_number = lookup_calls
        assert call_number <= 2
        return None

    monkeypatch.setattr(ingest_jobs, "_create_media_ingest_job", controlled_create, raising=True)
    monkeypatch.setattr(ingest_jobs, "_find_exact_occurrence_job", controlled_lookup, raising=True)
    data = _run_submit_data(run.run_id)

    def post_retry():
        with TestClient(media_ingest_jobs_client.app) as client:
            return client.post(
                "/api/v1/media/ingest/jobs",
                data=data,
                headers={"X-API-KEY": "test-api-key-12345"},
            )

    with ThreadPoolExecutor(max_workers=2) as pool:
        original_future = pool.submit(
            media_ingest_jobs_client.post,
            "/api/v1/media/ingest/jobs",
            data=data,
            headers={"X-API-KEY": "test-api-key-12345"},
        )
        assert first_entered.wait(10)
        retry_future = pool.submit(post_retry)
        assert retry_created.wait(10)
        original_response = original_future.result(timeout=10)
        release_retry_bind.set()
        retry_response = retry_future.result(timeout=10)

    assert original_response.status_code == 207, original_response.text
    assert original_response.json()["submissions"][0]["error_code"] == "job_submission_failed"
    assert retry_response.status_code == 200, retry_response.text
    assert create_calls == 2
    assert lookup_calls == 2
    assert len(set(create_inputs)) == 1
    jobs = manager.list_jobs(domain="media_ingest", owner_user_id="1", limit=10)
    assert len(jobs) == 1
    assert retry_response.json()["submissions"][0]["job_id"] == jobs[0]["id"]
    bound = store.get_run_item("1", run.run_id, "occ-url-1")
    assert bound.state == "queued"
    assert bound.job_id == jobs[0]["id"]


def test_pending_retry_rate_limit_stops_globally_without_resetting_stored_reservation(
    media_ingest_jobs_client,
    monkeypatch,
):
    manager, store, run = _seed_occurrence_run()
    from tldw_Server_API.app.api.v1.endpoints.media import ingest_jobs
    from tldw_Server_API.app.core.Jobs import manager as jobs_manager

    original_lookup = jobs_manager.JobManager.get_job_by_idempotency
    create_calls = 0
    lookup_calls = 0

    def ambiguous_then_limited(**_kwargs):
        nonlocal create_calls
        create_calls += 1
        if create_calls == 1:
            raise RuntimeError("job commit outcome unavailable")
        limit = JobSubmissionLimitError(
            "Quota exceeded: submits per minute",
            code="jobs_submit_rate_limited",
            retry_after=19,
        )
        raise HTTPException(
            status_code=429,
            detail=str(limit),
            headers={"Retry-After": "19"},
        ) from limit

    def unavailable_then_empty(self, **kwargs):
        nonlocal lookup_calls
        lookup_calls += 1
        if lookup_calls == 1:
            raise RuntimeError("jobs lookup unavailable")
        return original_lookup(self, **kwargs)

    monkeypatch.setattr(ingest_jobs, "_create_media_ingest_job", ambiguous_then_limited, raising=True)
    monkeypatch.setattr(
        jobs_manager.JobManager,
        "get_job_by_idempotency",
        unavailable_then_empty,
        raising=True,
    )
    first = media_ingest_jobs_client.post(
        "/api/v1/media/ingest/jobs",
        data=_run_submit_data(run.run_id),
        headers={"X-API-KEY": "test-api-key-12345"},
    )
    pending_batch = store.get_run_item("1", run.run_id, "occ-url-1").batch_id

    second = media_ingest_jobs_client.post(
        "/api/v1/media/ingest/jobs",
        data=_run_submit_data(
            run.run_id,
            urls=[
                "https://www.youtube.com/watch?v=alpha123456",
                "https://www.youtube.com/watch?v=alpha123456",
            ],
            occurrence_ids=["occ-url-1", "occ-url-2"],
            attempts=[1, 1],
        ),
        headers={"X-API-KEY": "test-api-key-12345"},
    )

    assert first.status_code == 207, first.text
    assert second.status_code == 429, second.text
    assert second.headers["Retry-After"] == "19"
    assert create_calls == 2
    pending = store.get_run_item("1", run.run_id, "occ-url-1")
    assert pending.state == "submit_pending"
    assert pending.batch_id == pending_batch
    assert store.get_run_item("1", run.run_id, "occ-url-2").state == "staged"
    assert manager.list_jobs(domain="media_ingest", owner_user_id="1", limit=10) == []


@pytest.mark.parametrize(("status_code", "retry_after"), [(429, "23"), (503, "29")])
def test_pending_retry_generic_http_failure_stays_global_after_confirmed_empty_lookup(
    media_ingest_jobs_client,
    monkeypatch,
    status_code,
    retry_after,
):
    manager, store, run = _seed_occurrence_run()
    from tldw_Server_API.app.api.v1.endpoints.media import ingest_jobs
    from tldw_Server_API.app.core.Jobs import manager as jobs_manager

    original_create = ingest_jobs._create_media_ingest_job
    original_lookup = jobs_manager.JobManager.get_job_by_idempotency
    create_occurrences: list[str] = []
    lookup_calls = 0

    def ambiguous_then_http_failure(**kwargs):
        create_occurrences.append(kwargs["payload"]["occurrence_id"])
        if len(create_occurrences) == 1:
            raise RuntimeError("job commit outcome unavailable")
        if len(create_occurrences) == 2:
            raise HTTPException(
                status_code=status_code,
                detail="job submission temporarily unavailable",
                headers={"Retry-After": retry_after},
            )
        return original_create(**kwargs)

    def unavailable_then_empty(self, **kwargs):
        nonlocal lookup_calls
        lookup_calls += 1
        if lookup_calls == 1:
            raise RuntimeError("jobs lookup unavailable")
        return original_lookup(self, **kwargs)

    monkeypatch.setattr(ingest_jobs, "_create_media_ingest_job", ambiguous_then_http_failure, raising=True)
    monkeypatch.setattr(
        jobs_manager.JobManager,
        "get_job_by_idempotency",
        unavailable_then_empty,
        raising=True,
    )
    first = media_ingest_jobs_client.post(
        "/api/v1/media/ingest/jobs",
        data=_run_submit_data(run.run_id),
        headers={"X-API-KEY": "test-api-key-12345"},
    )
    pending_batch = store.get_run_item("1", run.run_id, "occ-url-1").batch_id

    second = media_ingest_jobs_client.post(
        "/api/v1/media/ingest/jobs",
        data=_run_submit_data(
            run.run_id,
            urls=[
                "https://www.youtube.com/watch?v=alpha123456",
                "https://www.youtube.com/watch?v=alpha123456",
            ],
            occurrence_ids=["occ-url-1", "occ-url-2"],
            attempts=[1, 1],
        ),
        headers={"X-API-KEY": "test-api-key-12345"},
    )

    assert first.status_code == 207, first.text
    assert second.status_code == status_code, second.text
    assert second.headers["Retry-After"] == retry_after
    assert create_occurrences == ["occ-url-1", "occ-url-1"]
    pending = store.get_run_item("1", run.run_id, "occ-url-1")
    assert pending.state == "submit_pending"
    assert pending.batch_id == pending_batch
    assert store.get_run_item("1", run.run_id, "occ-url-2").state == "staged"
    assert manager.list_jobs(domain="media_ingest", owner_user_id="1", limit=10) == []


def test_pending_retry_accepts_exact_job_before_propagating_prior_http_failure(
    media_ingest_jobs_client,
    monkeypatch,
):
    manager, store, run = _seed_occurrence_run()
    from tldw_Server_API.app.api.v1.endpoints.media import ingest_jobs
    from tldw_Server_API.app.core.Jobs import manager as jobs_manager

    original_create = ingest_jobs._create_media_ingest_job
    original_lookup = jobs_manager.JobManager.get_job_by_idempotency
    create_calls = 0
    lookup_calls = 0

    def commit_then_http_failure(**kwargs):
        nonlocal create_calls
        create_calls += 1
        original_create(**kwargs)
        raise HTTPException(
            status_code=503,
            detail="job submission temporarily unavailable",
            headers={"Retry-After": "31"},
        )

    def unavailable_then_committed(self, **kwargs):
        nonlocal lookup_calls
        lookup_calls += 1
        if lookup_calls == 1:
            raise RuntimeError("jobs lookup unavailable")
        return original_lookup(self, **kwargs)

    monkeypatch.setattr(ingest_jobs, "_create_media_ingest_job", commit_then_http_failure, raising=True)
    monkeypatch.setattr(
        jobs_manager.JobManager,
        "get_job_by_idempotency",
        unavailable_then_committed,
        raising=True,
    )
    data = _run_submit_data(run.run_id)

    first = media_ingest_jobs_client.post(
        "/api/v1/media/ingest/jobs",
        data=data,
        headers={"X-API-KEY": "test-api-key-12345"},
    )
    pending_batch = store.get_run_item("1", run.run_id, "occ-url-1").batch_id
    second = media_ingest_jobs_client.post(
        "/api/v1/media/ingest/jobs",
        data=data,
        headers={"X-API-KEY": "test-api-key-12345"},
    )

    assert first.status_code == 207, first.text
    assert first.json()["submissions"][0]["error_code"] == "occurrence_binding_pending"
    assert second.status_code == 200, second.text
    assert create_calls == 1
    jobs = manager.list_jobs(domain="media_ingest", owner_user_id="1", limit=10)
    assert len(jobs) == 1
    assert second.json()["submissions"][0]["batch_id"] == pending_batch
    assert second.json()["submissions"][0]["job_id"] == jobs[0]["id"]
    assert store.get_run_item("1", run.run_id, "occ-url-1").job_id == jobs[0]["id"]


def test_run_bound_rate_limit_preserves_reservation_for_retry_and_stops_later_entries(
    media_ingest_jobs_client,
    monkeypatch,
):
    manager, store, run = _seed_occurrence_run()
    from tldw_Server_API.app.core.Jobs import manager as jobs_manager

    original_create = jobs_manager.JobManager.create_job
    create_calls: list[dict] = []

    def rate_limited_once(self, **kwargs):
        create_calls.append(kwargs)
        if len(create_calls) == 1:
            raise JobSubmissionLimitError(
                "Quota exceeded: submits per minute",
                code="jobs_submit_rate_limited",
                retry_after=17,
            )
        return original_create(self, **kwargs)

    monkeypatch.setattr(jobs_manager.JobManager, "create_job", rate_limited_once, raising=True)
    first = media_ingest_jobs_client.post(
        "/api/v1/media/ingest/jobs",
        data=_run_submit_data(
            run.run_id,
            urls=[
                "https://www.youtube.com/watch?v=alpha123456",
                "https://www.youtube.com/watch?v=alpha123456",
            ],
            occurrence_ids=["occ-url-1", "occ-url-2"],
            attempts=[1, 1],
        ),
        headers={"X-API-KEY": "test-api-key-12345"},
    )
    pending = store.get_run_item("1", run.run_id, "occ-url-1")
    second = media_ingest_jobs_client.post(
        "/api/v1/media/ingest/jobs",
        data=_run_submit_data(run.run_id),
        headers={"X-API-KEY": "test-api-key-12345"},
    )

    assert first.status_code == 429, first.text
    assert first.headers["Retry-After"] == "17"
    assert first.json()["detail"] == "Quota exceeded: submits per minute"
    assert pending.state == "submit_pending"
    assert pending.batch_id is not None
    assert store.get_run_item("1", run.run_id, "occ-url-2").state == "staged"
    assert second.status_code == 200, second.text
    assert second.json()["submissions"][0]["batch_id"] == pending.batch_id
    assert len(create_calls) == 2
    jobs = manager.list_jobs(domain="media_ingest", owner_user_id="1", limit=10)
    assert len(jobs) == 1
    assert store.get_run_item("1", run.run_id, "occ-url-1").job_id == jobs[0]["id"]
