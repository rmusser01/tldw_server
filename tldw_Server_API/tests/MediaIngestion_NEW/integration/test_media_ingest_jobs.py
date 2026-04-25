import shutil
from pathlib import Path

import pytest


pytestmark = pytest.mark.integration


def _set_jobs_db(monkeypatch, tmp_path):
    monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / "jobs.db"))
    monkeypatch.delenv("JOBS_DB_URL", raising=False)


def _fetch_job_queue(job_id: int) -> str | None:
    from tldw_Server_API.app.core.Jobs.manager import JobManager

    jm = JobManager()
    row = jm.get_job(int(job_id)) or {}
    queue = row.get("queue")
    return str(queue) if queue is not None else None


def test_media_ingest_jobs_submit_status_cancel(
    test_client,
    auth_headers,
    monkeypatch,
    tmp_path,
):
    _set_jobs_db(monkeypatch, tmp_path)

    data = {
        "media_type": "audio",
        "urls": "https://example.com/audio.mp3",
    }
    resp = test_client.post(
        "/api/v1/media/ingest/jobs",
        data=data,
        headers=auth_headers,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body.get("jobs")
    job_id = body["jobs"][0]["id"]

    status_resp = test_client.get(
        f"/api/v1/media/ingest/jobs/{job_id}",
        headers=auth_headers,
    )
    assert status_resp.status_code == 200, status_resp.text
    status_body = status_resp.json()
    assert status_body["media_type"] == "audio"
    assert status_body["source"] == "https://example.com/audio.mp3"
    assert status_body["status"] in {"queued", "processing"}

    cancel_resp = test_client.delete(
        f"/api/v1/media/ingest/jobs/{job_id}",
        headers=auth_headers,
    )
    assert cancel_resp.status_code == 200, cancel_resp.text
    assert cancel_resp.json().get("status") == "cancelled"

    status_resp = test_client.get(
        f"/api/v1/media/ingest/jobs/{job_id}",
        headers=auth_headers,
    )
    assert status_resp.status_code == 200, status_resp.text
    assert status_resp.json().get("status") == "cancelled"


def test_media_ingest_jobs_file_staging_payload(
    test_client,
    auth_headers,
    monkeypatch,
    tmp_path,
):
    _set_jobs_db(monkeypatch, tmp_path)

    upload_path = tmp_path / "sample.txt"
    upload_path.write_text("hello ingest staging", encoding="utf-8")

    data = {"media_type": "document"}
    files = [("files", ("sample.txt", upload_path.read_bytes(), "text/plain"))]
    resp = test_client.post(
        "/api/v1/media/ingest/jobs",
        data=data,
        files=files,
        headers=auth_headers,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    job_id = body["jobs"][0]["id"]

    from tldw_Server_API.app.core.Jobs.manager import JobManager

    jm = JobManager()
    job = jm.get_job(job_id)
    assert job is not None
    payload = job.get("payload") or {}
    assert payload.get("source_kind") == "file"
    assert payload.get("original_filename") == "sample.txt"
    assert payload.get("temp_dir")
    assert Path(payload["source"]).exists()

    shutil.rmtree(payload["temp_dir"], ignore_errors=True)


def test_media_ingest_jobs_list_by_batch(
    test_client,
    auth_headers,
    monkeypatch,
    tmp_path,
):
    _set_jobs_db(monkeypatch, tmp_path)

    upload_path = tmp_path / "batch.txt"
    upload_path.write_text("batch ingest", encoding="utf-8")

    data = {
        "media_type": "document",
        "urls": "https://example.com/doc-1",
    }
    files = [("files", ("batch.txt", upload_path.read_bytes(), "text/plain"))]
    resp = test_client.post(
        "/api/v1/media/ingest/jobs",
        data=data,
        files=files,
        headers=auth_headers,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    batch_id = body.get("batch_id")
    assert batch_id
    job_ids = {job["id"] for job in body.get("jobs", [])}
    assert len(job_ids) == 2

    list_resp = test_client.get(
        f"/api/v1/media/ingest/jobs?batch_id={batch_id}",
        headers=auth_headers,
    )
    assert list_resp.status_code == 200, list_resp.text
    list_body = list_resp.json()
    assert list_body.get("batch_id") == batch_id
    listed_ids = {job["id"] for job in list_body.get("jobs", [])}
    assert listed_ids == job_ids


def test_media_ingest_jobs_routes_audio_to_heavy_queue(
    test_client,
    auth_headers,
    monkeypatch,
    tmp_path,
):
    _set_jobs_db(monkeypatch, tmp_path)
    monkeypatch.setenv("MEDIA_INGEST_JOBS_DEFAULT_QUEUE", "default")
    monkeypatch.setenv("MEDIA_INGEST_JOBS_HEAVY_QUEUE", "media-heavy")
    monkeypatch.setenv("MEDIA_INGEST_JOBS_ROUTE_HEAVY", "true")
    monkeypatch.setenv("ROUTES_ENABLE", "media-ingest-heavy-jobs")
    monkeypatch.setenv("JOBS_ALLOWED_QUEUES", "media-heavy")

    resp = test_client.post(
        "/api/v1/media/ingest/jobs",
        data={
            "media_type": "audio",
            "urls": "https://example.com/audio-heavy.mp3",
        },
        headers=auth_headers,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body.get("jobs")
    job_id = int(body["jobs"][0]["id"])
    assert _fetch_job_queue(job_id) == "media-heavy"


def test_media_ingest_jobs_routes_audio_to_default_queue_when_heavy_worker_unavailable(
    test_client,
    auth_headers,
    monkeypatch,
    tmp_path,
):
    _set_jobs_db(monkeypatch, tmp_path)
    monkeypatch.setenv("MEDIA_INGEST_JOBS_DEFAULT_QUEUE", "default")
    monkeypatch.setenv("MEDIA_INGEST_JOBS_HEAVY_QUEUE", "media-heavy")
    monkeypatch.setenv("MEDIA_INGEST_JOBS_ROUTE_HEAVY", "true")
    monkeypatch.setenv("JOBS_ALLOWED_QUEUES", "default,media-heavy")
    monkeypatch.delenv("ROUTES_ENABLE", raising=False)
    monkeypatch.delenv("MEDIA_INGEST_HEAVY_JOBS_WORKER_ENABLED", raising=False)
    monkeypatch.delenv("TLDW_WORKERS_SIDECAR_MODE", raising=False)

    resp = test_client.post(
        "/api/v1/media/ingest/jobs",
        data={
            "media_type": "video",
            "urls": "https://example.com/video-default.mp4",
        },
        headers=auth_headers,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body.get("jobs")
    job_id = int(body["jobs"][0]["id"])
    assert _fetch_job_queue(job_id) == "default"


def test_media_ingest_jobs_keeps_heavy_queue_routing_in_sidecar_mode(
    test_client,
    auth_headers,
    monkeypatch,
    tmp_path,
):
    _set_jobs_db(monkeypatch, tmp_path)
    monkeypatch.setenv("MEDIA_INGEST_JOBS_DEFAULT_QUEUE", "default")
    monkeypatch.setenv("MEDIA_INGEST_JOBS_HEAVY_QUEUE", "media-heavy")
    monkeypatch.setenv("MEDIA_INGEST_JOBS_ROUTE_HEAVY", "true")
    monkeypatch.setenv("TLDW_WORKERS_SIDECAR_MODE", "true")
    monkeypatch.setenv("MEDIA_INGEST_HEAVY_JOBS_WORKER_ENABLED", "true")
    monkeypatch.setenv("JOBS_ALLOWED_QUEUES", "default,media-heavy")
    monkeypatch.delenv("ROUTES_ENABLE", raising=False)

    resp = test_client.post(
        "/api/v1/media/ingest/jobs",
        data={
            "media_type": "video",
            "urls": "https://example.com/video-sidecar.mp4",
        },
        headers=auth_headers,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body.get("jobs")
    job_id = int(body["jobs"][0]["id"])
    assert _fetch_job_queue(job_id) == "media-heavy"


def test_media_ingest_jobs_routes_ocr_documents_to_heavy_queue_and_respects_disable_flag(
    test_client,
    auth_headers,
    monkeypatch,
    tmp_path,
):
    _set_jobs_db(monkeypatch, tmp_path)
    monkeypatch.setenv("MEDIA_INGEST_JOBS_DEFAULT_QUEUE", "default")
    monkeypatch.setenv("MEDIA_INGEST_JOBS_HEAVY_QUEUE", "media-heavy")
    monkeypatch.setenv("MEDIA_INGEST_JOBS_ROUTE_HEAVY", "true")
    monkeypatch.setenv("ROUTES_ENABLE", "media-ingest-heavy-jobs")
    monkeypatch.setenv("JOBS_ALLOWED_QUEUES", "media-heavy")

    heavy_resp = test_client.post(
        "/api/v1/media/ingest/jobs",
        data={
            "media_type": "pdf",
            "enable_ocr": "true",
            "urls": "https://example.com/scanned-doc.pdf",
        },
        headers=auth_headers,
    )
    assert heavy_resp.status_code == 200, heavy_resp.text
    heavy_job_id = int(heavy_resp.json()["jobs"][0]["id"])
    assert _fetch_job_queue(heavy_job_id) == "media-heavy"

    monkeypatch.setenv("MEDIA_INGEST_JOBS_ROUTE_HEAVY", "false")
    default_resp = test_client.post(
        "/api/v1/media/ingest/jobs",
        data={
            "media_type": "audio",
            "urls": "https://example.com/audio-default.mp3",
        },
        headers=auth_headers,
    )
    assert default_resp.status_code == 200, default_resp.text
    default_job_id = int(default_resp.json()["jobs"][0]["id"])
    assert _fetch_job_queue(default_job_id) == "default"
