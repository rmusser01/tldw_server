import json
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
        "planned_item_ids": json.dumps(["101"]),
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
    assert url_payload["planned_item_id"] == "101"
    assert "planned_item_id" not in file_payload
    assert file_payload["original_filename"] == "sample.txt"
    assert file_payload.get("temp_dir")
    assert Path(file_payload["source"]).exists()

    shutil.rmtree(file_payload["temp_dir"], ignore_errors=True)


@pytest.mark.asyncio
async def test_uploaded_txt_worker_and_detail_preserve_exact_source(
    media_ingest_jobs_client,
    auth_headers,
    monkeypatch,
    tmp_path,
):
    """Keep uploaded bytes and the collection binding through retries, Jobs, and storage."""
    import tldw_Server_API.app.services.media_ingest_jobs_worker as worker
    from tldw_Server_API.app.api.v1.schemas.media_response_models import (
        MediaDetailResponse,
    )
    from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase
    from tldw_Server_API.app.core.DB_Management.media_db.api import (
        get_full_media_details_rich,
        get_media_by_id,
    )
    from tldw_Server_API.app.core.DB_Management.scope_context import scoped_context
    from tldw_Server_API.app.core.Jobs.manager import JobManager

    source = (
        b"Rowan Observatory\n\n"
        b"Rowan Observatory opened in 2019 in Cedar Ridge. Its director is Dr. Mira Vale.\n"
        b"The observatory's main telescope is named Selene.\n\n"
        b"Public tours begin every Friday at 18:00. Visitors must reserve a place before\n"
        b"arriving. The tour reference code is ORBIT-742.\n\n"
        b"The observatory studies variable stars and shares a monthly public report.\n"
        b"The source provides no ticket price and no current weather forecast.\n"
    )
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "user_databases"))

    with CollectionsDatabase.for_user("1") as db:
        collection = db.create_media_collection(name="Talks", kind="conference")
        item = db.add_media_collection_item(
            collection_id=collection.id, source_url="local://F-SOURCE.txt", source_kind="file",
            title="Observatory", speaker="Mira", track="Astronomy", tags=["stars"],
        )
    data = {
        "media_type": "document", "perform_analysis": "false", "overwrite_existing": "false",
        "perform_chunking": "true", "chunking_mode": "manual", "generate_embeddings": "false",
        "media_collection_id": str(collection.id), "media_collection_item_id": str(item.id),
        "idempotency_key": f"conference-retry-{item.id}-1",
    }
    files = [("files", ("F-SOURCE.txt", source, "text/plain"))]
    response = media_ingest_jobs_client.post("/api/v1/media/ingest/jobs", data=data, files=files, headers=auth_headers)
    assert response.status_code == 200, response.text  # nosec B101 # pytest regression assertion.
    jm = JobManager()
    job = jm.get_job(response.json()["jobs"][0]["id"])
    try:
        replay = media_ingest_jobs_client.post("/api/v1/media/ingest/jobs", data=data, files=files, headers=auth_headers)
        assert replay.json() == response.json()
        assert Path(job["payload"]["source"]).read_bytes() == source  # nosec B101 # pytest regression assertion.
        result = await worker._handle_job(job, jm, worker._ProgressState())
        assert result["status"] == "Success", result  # nosec B101 # pytest regression assertion.
        with CollectionsDatabase.for_user(job["owner_user_id"]) as db:
            persisted = db.get_media_collection_item(item.id)
            assert (persisted.status, persisted.media_id) == ("completed", result["media_id"])
            assert (persisted.title, persisted.speaker, persisted.track, persisted.tags) == ("Observatory", "Mira", "Astronomy", ["stars"])
        with scoped_context(user_id=int(job["owner_user_id"]), is_admin=False):
            db = worker._create_db(job["owner_user_id"])
            try:
                stored = get_media_by_id(db, result["media_id"])
                detail = MediaDetailResponse(**get_full_media_details_rich(
                    db,
                    media_id=result["media_id"],
                    include_content=True,
                    include_versions=True,
                    include_version_content=True,
                )).model_dump()
                assert stored["content"].encode("utf-8") == source  # nosec B101 # pytest regression assertion.
                assert stored["chunking_status"] == "completed"  # nosec B101 # pytest regression assertion.
                assert detail["content"]["text"].encode("utf-8") == source  # nosec B101 # pytest regression assertion.
            finally:
                db.close_connection()
    finally:
        shutil.rmtree(job["payload"]["temp_dir"], ignore_errors=True)


@pytest.mark.parametrize("source_kind", ["url", "file"])
def test_submit_media_ingest_jobs_preserves_collection_item_binding(
    media_ingest_jobs_client,
    monkeypatch,
    tmp_path,
    source_kind,
):
    monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / "jobs.db"))

    captured: list[dict] = []

    from tldw_Server_API.app.core.Jobs import manager as jobs_manager

    def fake_create_job(
        self,
        *,
        payload,
        batch_group=None,
        owner_user_id,
        **_kwargs,
    ):
        captured.append({"payload": payload, "batch_group": batch_group})
        return {"id": len(captured), "uuid": f"u{len(captured)}", "status": "queued",
                "payload": payload, "owner_user_id": owner_user_id}

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

    data = {"media_type": "document", "media_collection_id": "42"}
    files = None
    if source_kind == "url":
        data.update(urls="https://example.com/talk.pdf", planned_item_ids=json.dumps(["101"]),
                    idempotency_keys=json.dumps(["conference-42-101-0"]))
    else:
        data.update(media_collection_item_id="101", idempotency_key="conference-42-101-0")
        files = [("files", ("talk.txt", b"conference talk", "text/plain"))]
    resp = media_ingest_jobs_client.post(
        "/api/v1/media/ingest/jobs",
        data=data,
        files=files,
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
    if source_kind == "file":
        shutil.rmtree(captured[0]["payload"]["temp_dir"])


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


def test_collection_url_retries_respect_owner_cancellation_batch_and_legacy_rows(media_ingest_jobs_client, auth_headers):
    from tldw_Server_API.app.core.Jobs.manager import JobManager

    jm = JobManager()
    client = media_ingest_jobs_client
    owner = 1
    client.app.dependency_overrides[get_request_user] = lambda: User(id=owner, username=f"owner-{owner}")
    urls = ["https://example.test/one.pdf", "https://example.test/two.pdf"]
    keys = ["conference-retry-11-1", "conference-retry-12-1"]

    def submit(sources, item_ids, item_keys, files=None):
        return client.post("/api/v1/media/ingest/jobs", headers=auth_headers, files=files, data={
            "media_type": "pdf", "urls": sources, "media_collection_id": "7",
            "planned_item_ids": json.dumps(item_ids), "idempotency_keys": json.dumps(item_keys),
        })

    first = submit(urls, ["11", "12"], keys)
    assert first.status_code == 200, first.text
    original = first.json()
    assert submit(urls, ["11", "12"], keys).json() == original
    owner = 2
    other = submit(urls, ["11", "12"], keys)
    assert other.status_code == 200, other.text
    assert {job["id"] for job in original["jobs"]}.isdisjoint(job["id"] for job in other.json()["jobs"])
    assert all(jm.get_job(job["id"])["owner_user_id"] == "2" for job in other.json()["jobs"])
    assert submit(urls, ["11", "12"], keys).json() == other.json()

    before = len(jm.list_jobs(domain="media_ingest", owner_user_id="2"))
    mixed = submit(["https://example.test/new.pdf", urls[0]], ["13", "11"], ["new-13", keys[0]])
    assert mixed.status_code == 409, mixed.text
    assert len(jm.list_jobs(domain="media_ingest", owner_user_id="2")) == before
    cancelled = client.post("/api/v1/media/ingest/jobs/cancel", params={"batch_id": other.json()["batch_id"]}, headers=auth_headers)
    assert cancelled.json()["cancelled"] == 2
    assert all(jm.get_job(job["id"])["status"] == "queued" for job in original["jobs"])

    legacy_key = "conference-retry-21-1"
    legacy = jm.create_job(
        domain="media_ingest", queue="default", job_type="media_ingest_item", owner_user_id="1",
        idempotency_key=legacy_key, batch_group="older-batch",
        payload={"source_kind": "url", "source": urls[0], "batch_id": "older-batch", "collection_id": "7",
                 "planned_item_id": "21", "idempotency_key": legacy_key},
    )
    owner = 1
    replay = submit([urls[0]], ["21"], [legacy_key]).json()
    assert (replay["batch_id"], replay["jobs"][0]["id"]) == ("older-batch", legacy["id"])
    owner = 2
    assert submit([urls[0]], ["21"], [legacy_key]).json()["jobs"][0]["id"] != legacy["id"]

    unbound = client.post("/api/v1/media/ingest/jobs", headers=auth_headers, data={
        "media_type": "pdf", "urls": urls[0], "idempotency_key": "unbound-client-key",
    })
    assert unbound.status_code == 200, unbound.text
    assert jm.get_job(unbound.json()["jobs"][0]["id"])["idempotency_key"] == "unbound-client-key"


def test_collection_url_admission_race_keeps_each_acknowledged_job_in_its_batch(media_ingest_jobs_client, auth_headers, monkeypatch):
    from tldw_Server_API.app.core.Jobs.manager import JobManager

    jm = JobManager()
    client = media_ingest_jobs_client
    urls = ["https://example.test/first.pdf", "https://example.test/second.pdf"]
    data = {"media_type": "pdf", "media_collection_id": "7", "urls": urls[1],
            "planned_item_ids": json.dumps(["12"]), "idempotency_keys": json.dumps(["second"])}
    original = client.post("/api/v1/media/ingest/jobs", headers=auth_headers, data=data).json()
    # Model another request winning admission after our lookup completed.
    monkeypatch.setattr(JobManager, "get_job_or_archived_by_idempotency_key", lambda *_args, **_kwargs: None)
    assert client.post("/api/v1/media/ingest/jobs", headers=auth_headers, data=data).json() == original
    data.update(urls=urls, planned_item_ids=json.dumps(["11", "12"]), idempotency_keys=json.dumps(["first", "second"]))
    raced = client.post("/api/v1/media/ingest/jobs", headers=auth_headers, data=data).json()
    assert [job["source"] for job in raced["jobs"]] == urls[:1]
    assert len(raced["errors"]) == 1
    assert jm.get_job(raced["jobs"][0]["id"])["batch_group"] == raced["batch_id"]
    assert raced["batch_id"] != original["batch_id"]

    def mismatched_admission(*_args, payload, **_kwargs):
        return {"id": 999, "uuid": "another-owner", "owner_user_id": "2", "status": "queued", "payload": payload}
    monkeypatch.setattr(JobManager, "create_job", mismatched_admission)
    conflict = client.post("/api/v1/media/ingest/jobs", headers=auth_headers, data=data)
    assert conflict.status_code == 409
    assert "another-owner" not in conflict.text


@pytest.mark.parametrize("kind", ["url", "file"])
def test_collection_retry_survives_heavy_worker_routing_changes(kind, media_ingest_jobs_client, auth_headers, monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints.media import ingest_jobs
    from tldw_Server_API.app.core.Jobs.manager import JobManager

    jm = JobManager()
    heavy_available = False
    monkeypatch.setattr(ingest_jobs, "_heavy_media_ingest_worker_available", lambda: heavy_available)
    monkeypatch.setenv("MEDIA_INGEST_JOBS_ROUTE_HEAVY", "true")
    data = {"media_type": "document", "enable_ocr": "true", "media_collection_id": "7",
            "media_collection_item_id": "11", "idempotency_key": "conference-retry-11-1"}
    files = None
    if kind == "url":
        data["urls"] = "https://example.test/talk.txt"
    else:
        files = [("files", ("talk.txt", b"conference talk", "text/plain"))]

    def submit():
        result = media_ingest_jobs_client.post("/api/v1/media/ingest/jobs", headers=auth_headers, data=data, files=files)
        assert result.status_code == 200, result.text
        return result.json()

    first = submit()
    first_job = jm.get_job(first["jobs"][0]["id"])
    try:
        heavy_available = True
        assert submit() == first
        with monkeypatch.context() as concurrent:
            # Both contenders saw an empty receipt slot before either insert committed.
            concurrent.setattr(JobManager, "get_job_or_archived_by_idempotency_key", lambda *_args, **_kwargs: None)
            assert submit() == first
        # A pre-cutover row can live on a queue that this deployment no longer selects.
        data["media_collection_item_id"] = "22"
        data["idempotency_key"] = legacy_key = "conference-retry-22-1"
        payload = {**first_job["payload"], "batch_id": "older-heavy-batch", "planned_item_id": "22",
                   "idempotency_key": legacy_key}
        legacy = jm.create_job(
            domain="media_ingest", queue="low", job_type="media_ingest_item", owner_user_id="1",
            idempotency_key=legacy_key if kind == "url" else f"media-upload:1:{legacy_key}",
            batch_group="older-heavy-batch", payload=payload,
        )
        heavy_available = False
        restored = submit()
        assert (restored["batch_id"], restored["jobs"][0]["id"]) == ("older-heavy-batch", legacy["id"])
    finally:
        if kind == "file":
            shutil.rmtree(first_job["payload"]["temp_dir"], ignore_errors=True)


def test_conflicting_native_receipt_preserves_existing_collection_status(media_ingest_jobs_client, auth_headers, monkeypatch):
    from unittest.mock import Mock

    from tldw_Server_API.app.api.v1.endpoints.media import ingest_jobs

    mark_failed = Mock()
    monkeypatch.setattr(ingest_jobs, "_mark_collection_item_submit_failed", mark_failed)
    monkeypatch.setattr(ingest_jobs, "_find_collection_retry_job", lambda *_args: {"archived": True})
    response = media_ingest_jobs_client.post("/api/v1/media/ingest/jobs", headers=auth_headers,
        data={"media_type": "document", "media_collection_id": "7", "media_collection_item_id": "11", "idempotency_key": "retry"},
        files=[("files", ("talk.txt", b"conference talk", "text/plain"))])
    assert response.status_code == 409
    mark_failed.assert_not_called()


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
        "tldw_Server_API.app.core.Ingestion_Media_Processing.input_sourcing.save_uploaded_files",
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


@pytest.mark.parametrize("job_status", ("queued", "completed", "cancelled"))
def test_get_media_ingest_job_serializes_postgres_timestamps(
    media_ingest_jobs_client,
    monkeypatch,
    job_status,
):
    """Real PG timestamps must not turn a valid status read into HTTP500."""
    from datetime import datetime, timezone

    from tldw_Server_API.app.core.Jobs import manager as jobs_manager

    timestamp = datetime(2026, 10, 14, 13, 0, 0, tzinfo=timezone.utc)
    dates = {
        "created_at": timestamp,
        "started_at": timestamp if job_status != "queued" else None,
        "completed_at": timestamp if job_status == "completed" else None,
        "cancelled_at": timestamp if job_status == "cancelled" else None,
    }
    row = {
        "id": 99,
        "uuid": "postgres-job",
        "domain": "media_ingest",
        "job_type": "media_ingest_item",
        "owner_user_id": "1",
        "status": job_status,
        **dates,
        "result": {"media_id": 321} if job_status == "completed" else None,
        "payload": {"batch_id": "postgres-batch"},
    }
    monkeypatch.setattr(jobs_manager.JobManager, "get_job", lambda self, job_id: row)
    response = media_ingest_jobs_client.get(
        "/api/v1/media/ingest/jobs/99",
        headers={"X-API-KEY": "test-api-key-12345"},
    )
    assert response.status_code == 200, response.text  # nosec B101 - Actual endpoint timestamp regression.
    body = response.json()
    assert {key: body[key] for key in dates} == {
        key: value.isoformat() if value else None for key, value in dates.items()
    }  # nosec B101 - PG serialization and nullable fields.
    assert (body["owner_user_id"], body["batch_id"], body["result"]) == ("1", "postgres-batch", row["result"])  # nosec B101 - Status identity/result preserved.


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
        def is_quick_ingest_plan_item(self, item_id):
            return False

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
        def is_quick_ingest_plan_item(self, item_id):
            return False

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
    monkeypatch.setenv("MEDIA_INGEST_HEAVY_JOBS_WORKER_ENABLED", "false")
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


@pytest.mark.parametrize(
    ("media_type", "analysis", "provider", "settings", "expected_queue"),
    [
        ("pdf", "true", "deepseek", {}, "low"),
        ("document", "true", "deepseek", {}, "low"),
        ("ebook", "true", "deepseek", {}, "low"),
        ("document", "false", "deepseek", {}, "default"),
        ("document", "true", "none", {}, "default"),
        ("pdf", "true", "deepseek", {"MEDIA_INGEST_HEAVY_JOBS_WORKER_ENABLED": "false"}, "default"),
        ("pdf", "true", "deepseek", {"MEDIA_INGEST_JOBS_ROUTE_HEAVY": "false"}, "default"),
        ("pdf", "true", "deepseek", {"ROUTES_DISABLE": "media-ingest-heavy-jobs"}, "default"),
        ("pdf", "true", "deepseek", {"TLDW_WORKERS_SIDECAR_MODE": "true"}, "default"),
        ("pdf", "true", "deepseek", {"MEDIA_INGEST_JOBS_WORKER_ENABLED": "false"}, "default"),
        (
            "pdf", "true", "deepseek",
            {
                "TLDW_WORKERS_SIDECAR_MODE": "true",
                "MEDIA_INGEST_HEAVY_JOBS_WORKER_ENABLED": "true",
                "MEDIA_INGEST_JOBS_HEAVY_QUEUE": "media-heavy",
            },
            "media-heavy",
        ),
    ],
)
def test_submit_media_ingest_routes_provider_analysis(
    media_ingest_jobs_client, monkeypatch, media_type, analysis, provider, settings, expected_queue,
):
    from tldw_Server_API.app.core.Jobs.manager import JobManager

    for key in (
        "ROUTES_ENABLE", "ROUTES_DISABLE", "TLDW_WORKERS_SIDECAR_MODE",
        "MEDIA_INGEST_JOBS_ROUTE_HEAVY", "MEDIA_INGEST_JOBS_DEFAULT_QUEUE",
        "MEDIA_INGEST_JOBS_HEAVY_QUEUE", "MEDIA_INGEST_JOBS_WORKER_ENABLED",
        "MEDIA_INGEST_HEAVY_JOBS_WORKER_ENABLED",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("JOBS_ALLOWED_QUEUES", "default,low,media-heavy")
    for key, value in settings.items():
        monkeypatch.setenv(key, value)

    response = media_ingest_jobs_client.post(
        "/api/v1/media/ingest/jobs",
        data={
            "media_type": media_type,
            "urls": "https://example.com/document",
            "perform_analysis": analysis,
            "api_name": provider,
        },
        headers={"X-API-KEY": "test-api-key-12345"},
    )

    assert response.status_code == 200, response.text
    job_id = response.json()["jobs"][0]["id"]
    assert JobManager().get_job(job_id)["queue"] == expected_queue
