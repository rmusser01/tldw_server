import asyncio
import contextlib
import importlib
import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.Collections.reading_import_jobs import (
    READING_IMPORT_DOMAIN,
    READING_IMPORT_JOB_TYPE,
    ReadingImportJobError,
    handle_reading_import_job,
    reading_import_queue,
    resolve_reading_import_file,
    stage_reading_import_file,
)
import tldw_Server_API.app.core.Collections.reading_import_jobs as reading_import_jobs_module
from tldw_Server_API.app.core.Collections.reading_importers import (
    parse_instapaper_export,
    parse_pocket_export,
)
from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.Jobs.manager import JobManager

pytestmark = pytest.mark.unit


def test_parse_pocket_export():
    payload = {
        "list": {
            "1": {
                "resolved_url": "https://example.com/story",
                "resolved_title": "Example Story",
                "tags": {"research": {}, "ai": {}},
                "status": "0",
                "favorite": "1",
                "time_added": "1700000000",
                "excerpt": "Pocket note",
            }
        }
    }
    items = parse_pocket_export(json.dumps(payload).encode("utf-8"))
    assert len(items) == 1
    item = items[0]
    assert item.url == "https://example.com/story"
    assert item.title == "Example Story"
    assert set(item.tags) == {"ai", "research"}
    assert item.status == "saved"
    assert item.favorite is True
    assert item.notes == "Pocket note"


def test_parse_instapaper_export():
    csv_payload = "URL,Title,Tags,Folder,Notes\nhttps://example.com/a,Example A,tag1;tag2,Archive,Note A\n"
    items = parse_instapaper_export(csv_payload.encode("utf-8"))
    assert len(items) == 1
    item = items[0]
    assert item.url == "https://example.com/a"
    assert item.title == "Example A"
    assert set(item.tags) == {"tag1", "tag2"}
    assert item.status == "archived"
    assert item.notes == "Note A"


def test_reading_import_max_bytes_invalid_env_uses_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid READING_IMPORT_MAX_BYTES falls back without import-time failure."""
    monkeypatch.setenv("READING_IMPORT_MAX_BYTES", "not-an-int")
    reloaded = importlib.reload(reading_import_jobs_module)
    try:
        assert reloaded.MAX_READING_IMPORT_BYTES == 10 * 1024 * 1024
    finally:
        monkeypatch.delenv("READING_IMPORT_MAX_BYTES", raising=False)
        importlib.reload(reading_import_jobs_module)


@pytest.mark.usefixtures("client_with_user")
def test_stage_and_resolve_import_file():
    path = stage_reading_import_file(
        user_id=222,
        filename="pocket.json",
        raw_bytes=b"payload",
    )
    try:
        assert path.exists()
        resolved = resolve_reading_import_file(222, path.name)
        assert resolved == path
    finally:
        path.unlink(missing_ok=True)


@pytest.mark.usefixtures("client_with_user")
def test_resolve_import_file_rejects_invalid_token():
    with pytest.raises(ReadingImportJobError):
        resolve_reading_import_file(222, "../evil.json")


@pytest.fixture()
def client_with_user(monkeypatch):
    async def override_user():
        return User(id=222, username="reader", email=None, is_active=True)

    monkeypatch.setenv("MINIMAL_TEST_APP", "0")
    monkeypatch.setenv("ULTRA_MINIMAL_APP", "0")
    monkeypatch.setenv("ROUTES_ENABLE", "reading")

    base_dir = Path.cwd() / "Databases" / "test_reading_import"
    shutil.rmtree(base_dir, ignore_errors=True)
    base_dir.mkdir(parents=True, exist_ok=True)
    prev_base_dir = settings.get("USER_DB_BASE_DIR")
    settings.USER_DB_BASE_DIR = str(base_dir)
    monkeypatch.setenv("USER_DB_BASE_DIR", str(base_dir))
    jobs_db_path = base_dir / "jobs.db"
    monkeypatch.setenv("JOBS_DB_PATH", str(jobs_db_path))

    from tldw_Server_API.app import main as app_main

    importlib.reload(app_main)
    fastapi_app = app_main.app

    fastapi_app.dependency_overrides[get_request_user] = override_user
    try:
        with TestClient(fastapi_app) as client:
            yield client
    finally:
        fastapi_app.dependency_overrides.clear()
        if prev_base_dir is not None:
            settings.USER_DB_BASE_DIR = prev_base_dir
        else:
            with contextlib.suppress(AttributeError):
                del settings.USER_DB_BASE_DIR


def _run_import_job(job_id: int) -> None:
    async def _runner() -> None:
        jm = JobManager()
        job = jm.get_job(job_id)
        assert job is not None
        worker_id = "test-reading-import-worker"
        owner_user_id = str(job.get("owner_user_id") or "")
        queue = reading_import_queue()

        acquired = None
        for _attempt in range(5):
            acquired = jm.acquire_next_job(
                domain=READING_IMPORT_DOMAIN,
                queue=queue,
                lease_seconds=60,
                worker_id=worker_id,
                owner_user_id=owner_user_id or None,
            )
            if not acquired:
                continue
            acquired_id = int(acquired.get("id") or 0)
            acquired_type = str(acquired.get("job_type") or "")
            if acquired_id == int(job_id) and acquired_type == READING_IMPORT_JOB_TYPE:
                break
            jm.release_job(
                acquired_id,
                worker_id=worker_id,
                lease_id=acquired.get("lease_id"),
                reason="test_acquired_non_target_job",
                enforce=False,
            )
            acquired = None

        assert acquired is not None, "failed to acquire reading import job for processing"
        assert int(acquired.get("id") or 0) == int(job_id), "acquired wrong job id"
        assert str(acquired.get("job_type") or "") == READING_IMPORT_JOB_TYPE, "acquired wrong job type"

        result = await handle_reading_import_job(acquired)
        jm.complete_job(
            job_id,
            result=result,
            worker_id=worker_id,
            lease_id=acquired.get("lease_id"),
            enforce=False,
        )

    asyncio.run(_runner())


def test_reading_import_and_export(client_with_user):
    client = client_with_user
    payload = {
        "list": {
            "1": {
                "resolved_url": "https://example.com/story",
                "resolved_title": "Example Story",
                "tags": {"research": {}, "ai": {}},
                "status": "0",
                "favorite": "1",
                "excerpt": "Pocket note",
            }
        }
    }
    files = {
        "file": ("pocket.json", json.dumps(payload).encode("utf-8"), "application/json"),
    }
    data = {"source": "pocket", "merge_tags": "true"}
    r = client.post("/api/v1/reading/import", files=files, data=data)
    assert r.status_code == 202, r.text
    body = r.json()
    job_id = body["job_id"]
    _run_import_job(job_id)

    job_resp = client.get(f"/api/v1/reading/import/jobs/{job_id}")
    assert job_resp.status_code == 200
    job_body = job_resp.json()
    assert job_body["status"] == "completed"
    assert job_body["result"]["imported"] == 1

    jobs_resp = client.get("/api/v1/reading/import/jobs")
    assert jobs_resp.status_code == 200
    jobs_body = jobs_resp.json()
    assert any(j["job_id"] == job_id for j in jobs_body["jobs"])

    r = client.get("/api/v1/reading/items")
    assert r.status_code == 200
    items = r.json()["items"]
    assert items
    assert items[0]["title"] == "Example Story"

    r = client.get("/api/v1/reading/export", params={"format": "jsonl"})
    assert r.status_code == 200
    lines = [line for line in r.text.splitlines() if line]
    assert lines
    exported = json.loads(lines[0])
    assert exported["title"] == "Example Story"
    assert "notes" in exported


def test_reading_import_preserves_existing_fields(client_with_user):
    client = client_with_user
    url = "https://example.com/preserve"
    save_payload = {
        "url": url,
        "title": "Original Title",
        "content": "Original content body",
        "summary": "Original summary",
        "notes": "Original notes",
    }
    r = client.post("/api/v1/reading/save", json=save_payload)
    assert r.status_code == 200, r.text

    db = CollectionsDatabase.for_user(user_id=222)
    existing = db.get_content_item_by_url(url)
    assert existing is not None
    original_hash = existing.content_hash
    original_word_count = existing.word_count
    original_summary = existing.summary
    original_notes = existing.notes
    original_meta = json.loads(existing.metadata_json or "{}")

    import_payload = {
        "list": {
            "1": {
                "resolved_url": url,
                "resolved_title": "Imported Title",
                "status": "0",
            }
        }
    }
    files = {
        "file": ("pocket.json", json.dumps(import_payload).encode("utf-8"), "application/json"),
    }
    data = {"source": "pocket", "merge_tags": "true"}
    r = client.post("/api/v1/reading/import", files=files, data=data)
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    _run_import_job(job_id)

    updated = db.get_content_item_by_url(url)
    assert updated is not None
    assert updated.summary == original_summary
    assert updated.notes == original_notes
    assert updated.content_hash == original_hash
    assert updated.word_count == original_word_count
    updated_meta = json.loads(updated.metadata_json or "{}")
    assert updated_meta.get("text") == original_meta.get("text")


def test_reading_export_includes_highlights(client_with_user):
    client = client_with_user
    content = "Hello world. Highlight me."
    save_payload = {
        "url": "https://example.com/highlight",
        "title": "Highlight Article",
        "content": content,
    }
    r = client.post("/api/v1/reading/save", json=save_payload)
    assert r.status_code == 200, r.text
    item_id = r.json()["id"]

    quote = "Highlight me"
    start_offset = content.index(quote)
    end_offset = start_offset + len(quote)
    highlight_payload = {
        "item_id": item_id,
        "quote": quote,
        "start_offset": start_offset,
        "end_offset": end_offset,
        "color": "yellow",
        "note": "Important",
    }
    r = client.post(f"/api/v1/reading/items/{item_id}/highlight", json=highlight_payload)
    assert r.status_code == 200, r.text

    r = client.get("/api/v1/reading/export", params={"format": "jsonl", "include_highlights": "true"})
    assert r.status_code == 200, r.text
    rows = [json.loads(line) for line in r.text.splitlines() if line.strip()]
    exported = next(row for row in rows if row["id"] == item_id)
    highlights = exported.get("highlights") or []
    assert highlights
    assert highlights[0]["quote"] == quote


def test_reading_export_can_omit_notes(client_with_user):
    client = client_with_user
    save_payload = {
        "url": "https://example.com/no-notes",
        "title": "No Notes Export",
        "content": "Export body",
        "notes": "Should be omitted",
    }
    r = client.post("/api/v1/reading/save", json=save_payload)
    assert r.status_code == 200, r.text
    item_id = r.json()["id"]

    r = client.get("/api/v1/reading/export", params={"format": "jsonl", "include_notes": "false"})
    assert r.status_code == 200, r.text
    rows = [json.loads(line) for line in r.text.splitlines() if line.strip()]
    exported = next(row for row in rows if row["id"] == item_id)
    assert "notes" not in exported


def test_reading_archive_creates_output(client_with_user):
    client = client_with_user
    save_payload = {
        "url": "https://example.com/archive",
        "title": "Archive Article",
        "content": "Archive content.",
    }
    r = client.post("/api/v1/reading/save", json=save_payload)
    assert r.status_code == 200, r.text
    item_id = r.json()["id"]

    r = client.post(
        f"/api/v1/reading/items/{item_id}/archive",
        json={"format": "html", "retention_days": 1},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["output_id"]
    assert body["download_url"].endswith(f"/api/v1/outputs/{body['output_id']}/download")
    assert body["retention_until"] is not None

    output_path = DatabasePaths.get_user_outputs_dir(222) / body["storage_path"]
    assert output_path.exists()

    download = client.get(body["download_url"])
    assert download.status_code == 200
    assert "Archive Article" in download.text
