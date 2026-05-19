import os
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.main import app
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.api.v1.endpoints import chatbooks as chatbooks_mod


class _DummyConn:
    def execute(self, *_args, **_kwargs):
        return None

    def close(self):
        return None


class _DummyCursor:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return list(self._rows)


class FakeDB:
    """Minimal in-memory DB to satisfy ChatbookService job storage during tests."""

    def __init__(self):
        self._export_jobs = {}
        self._import_jobs = {}

    def get_connection(self):
        return _DummyConn()

    def execute_query(self, sql, params=(), commit=False):
        sql_norm = " ".join(str(sql).strip().split()).lower()
        # Create tables: no-op
        if sql_norm.startswith("create table if not exists export_jobs"):
            return _DummyCursor([])
        if sql_norm.startswith("create table if not exists import_jobs"):
            return _DummyCursor([])

        # Upsert export job
        if sql_norm.startswith("insert or replace into export_jobs"):
            (
                job_id, user_id, status, chatbook_name, output_path,
                created_at, started_at, completed_at, error_message,
                progress_percentage, total_items, processed_items,
                file_size_bytes, download_url, expires_at,
            ) = params
            self._export_jobs[job_id] = {
                "job_id": job_id,
                "user_id": user_id,
                "status": status,
                "chatbook_name": chatbook_name,
                "output_path": output_path,
                "created_at": created_at,
                "started_at": started_at,
                "completed_at": completed_at,
                "error_message": error_message,
                "progress_percentage": progress_percentage,
                "total_items": total_items,
                "processed_items": processed_items,
                "file_size_bytes": file_size_bytes,
                "download_url": download_url,
                "expires_at": expires_at,
            }
            return _DummyCursor([])

        # Select export job by id + user
        if sql_norm.startswith("select * from export_jobs where job_id"):
            job_id, user_id = params
            row = self._export_jobs.get(job_id)
            if not row or row.get("user_id") != user_id:
                return _DummyCursor([])
            return _DummyCursor([row])

        # Default: empty
        return _DummyCursor([])


@pytest.fixture()
def client_override(tmp_path, monkeypatch):
    # Ensure test mode to avoid global rate limiter
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path))

    async def override_user():
        return User(id=1, username="tester", email="t@e.com", is_active=True)

    # Use a single FakeDB instance across requests to persist jobs
    _shared_db = FakeDB()

    def override_db():
        return _shared_db

    app.dependency_overrides[get_request_user] = override_user
    app.dependency_overrides[chatbooks_mod.get_chacha_db] = override_db

    # Stub audit logger to avoid strict signature requirements in tests
    class _DummyAudit:
        def log_event(self, *args, **kwargs):
            return None
        def log_security_event(self, *args, **kwargs):
            return None

    chatbooks_mod.audit_logger = _DummyAudit()

    with TestClient(app) as client:
        client._chatbooks_db = _shared_db
        yield client

    app.dependency_overrides.clear()


def _make_export_payload(async_mode: bool = False):
    return {
        "name": "Test Export",
        "description": "Test run",
        "content_selections": {},
        "async_mode": async_mode,
    }


@pytest.mark.unit
def test_chatbooks_export_sync_persists_job_and_downloads(client_override: TestClient):
    client = client_override
    # Create a minimal sync export with no content
    r = client.post("/api/v1/chatbooks/export", json=_make_export_payload(async_mode=False))
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["success"] is True
    assert isinstance(data.get("job_id"), str) and len(data["job_id"]) > 0
    assert "file_path" not in data
    assert data.get("download_url", "").endswith(data["job_id"])  # URL is job-based

    job_id = data["job_id"]

    # Fetch the job and validate completed state and URL
    r2 = client.get(f"/api/v1/chatbooks/export/jobs/{job_id}")
    assert r2.status_code == 200, r2.text
    job = r2.json()
    assert job["status"] == "completed"
    assert job.get("download_url", "").endswith(job_id)

    # Download the archive by job_id
    r3 = client.get(f"/api/v1/chatbooks/download/{job_id}")
    assert r3.status_code == 200, r3.text
    assert r3.headers.get("content-type") == "application/zip"
    assert "attachment; filename=" in r3.headers.get("content-disposition", "")


@pytest.mark.unit
def test_chatbooks_download_blocks_export_path_outside_exports_dir(client_override: TestClient):
    client = client_override
    r = client.post("/api/v1/chatbooks/export", json=_make_export_payload(async_mode=False))
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]

    db = getattr(client, "_chatbooks_db", None)
    assert db is not None
    export_dir = DatabasePaths.get_user_chatbooks_exports_dir(1).resolve()
    outside_path = (export_dir.parent / "outside.zip").resolve()
    db._export_jobs[job_id]["output_path"] = str(outside_path)

    resp = client.get(f"/api/v1/chatbooks/download/{job_id}")
    assert resp.status_code == 403, resp.text
    assert resp.json().get("detail") == "Access denied"


@pytest.mark.unit
def test_chatbooks_export_sync_rejects_result_outside_exports_dir(
    client_override: TestClient,
    monkeypatch,
):
    client = client_override
    export_dir = DatabasePaths.get_user_chatbooks_exports_dir(1).resolve()
    outside_path = (export_dir.parent / "outside.zip").resolve()

    async def _fake_create_chatbook(*_args, **_kwargs):
        return True, "ok", str(outside_path)

    monkeypatch.setattr(chatbooks_mod.ChatbookService, "create_chatbook", _fake_create_chatbook)
    resp = client.post("/api/v1/chatbooks/export", json=_make_export_payload(async_mode=False))
    assert resp.status_code == 500, resp.text
    assert resp.json().get("detail") == "Export path validation failed"


@pytest.mark.unit
def test_chatbooks_export_sync_rejects_missing_result_file(
    client_override: TestClient,
    monkeypatch,
):
    client = client_override
    export_dir = DatabasePaths.get_user_chatbooks_exports_dir(1).resolve()
    missing_path = export_dir / "missing.chatbook"

    async def _fake_create_chatbook(*_args, **_kwargs):
        return True, "ok", str(missing_path)

    monkeypatch.setattr(chatbooks_mod.ChatbookService, "create_chatbook", _fake_create_chatbook)
    resp = client.post("/api/v1/chatbooks/export", json=_make_export_payload(async_mode=False))
    assert resp.status_code == 500, resp.text
    assert resp.json().get("detail") == "Export archive was not created"

    db = getattr(client, "_chatbooks_db", None)
    assert db is not None
    assert db._export_jobs == {}


@pytest.mark.unit
def test_chatbooks_continue_export_sync_persists_job_and_downloads(
    client_override: TestClient,
    monkeypatch,
):
    client = client_override
    export_dir = DatabasePaths.get_user_chatbooks_exports_dir(1).resolve()
    continuation_path = export_dir / "continuation.chatbook"
    continuation_path.write_bytes(b"continuation-bytes")

    async def _fake_continue_export(*_args, **_kwargs):
        return True, "Continuation created", str(continuation_path)

    monkeypatch.setattr(chatbooks_mod.ChatbookService, "continue_chatbook_export", _fake_continue_export)
    opaque_cursor = "-".join(("cursor", "1"))

    response = client.post(
        "/api/v1/chatbooks/export/continue",
        json={
            "export_id": "export-123",
            "continuations": [{"evaluation_id": "eval-1", "continuation_token": opaque_cursor}],
            "async_mode": False,
        },
    )

    assert response.status_code == 200, response.text
    data = response.json()
    assert data == {
        "success": True,
        "message": "Continuation created",
        "job_id": data["job_id"],
        "download_url": data["download_url"],
    }
    assert isinstance(data["job_id"], str) and data["job_id"]
    assert data["download_url"].endswith(data["job_id"])
    assert "file_path" not in data

    job_response = client.get(f"/api/v1/chatbooks/export/jobs/{data['job_id']}")
    assert job_response.status_code == 200, job_response.text
    job = job_response.json()
    assert job["status"] == "completed"
    assert job["download_url"].endswith(data["job_id"])

    download_response = client.get(f"/api/v1/chatbooks/download/{data['job_id']}")
    assert download_response.status_code == 200, download_response.text
    assert download_response.headers.get("content-type") == "application/zip"


@pytest.mark.unit
def test_chatbooks_continue_export_sync_rejects_missing_result_file(
    client_override: TestClient,
    monkeypatch,
):
    client = client_override
    export_dir = DatabasePaths.get_user_chatbooks_exports_dir(1).resolve()
    missing_path = export_dir / "missing-continuation.chatbook"

    async def _fake_continue_export(*_args, **_kwargs):
        return True, "Continuation created", str(missing_path)

    monkeypatch.setattr(chatbooks_mod.ChatbookService, "continue_chatbook_export", _fake_continue_export)
    opaque_cursor = "-".join(("cursor", "1"))

    response = client.post(
        "/api/v1/chatbooks/export/continue",
        json={
            "export_id": "export-123",
            "continuations": [{"evaluation_id": "eval-1", "continuation_token": opaque_cursor}],
            "async_mode": False,
        },
    )

    assert response.status_code == 500, response.text
    assert response.json().get("detail") == "Export archive was not created"

    db = getattr(client, "_chatbooks_db", None)
    assert db is not None
    assert db._export_jobs == {}
