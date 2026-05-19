import io
import json
import os
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.main import app
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.api.v1.API_Deps.Audit_DB_Deps import get_audit_service_for_user
from tldw_Server_API.app.api.v1.endpoints import chatbooks as chatbooks_mod
from tldw_Server_API.app.core.Chatbooks.chatbook_models import ExportJob, ExportStatus, ImportJob, ImportStatus
from tldw_Server_API.app.core.Chatbooks.chatbook_service import ChatbookService


@pytest.fixture()
def client(tmp_path_factory, monkeypatch):
    """Provide a TestClient with deterministic chatbooks job state."""
    monkeypatch.setenv("TEST_MODE", "true")
    tmp_dir = tmp_path_factory.mktemp("chatbooks_cancel")

    class _DummyCursor:
        def __init__(self, rows):
            self._rows = rows

        def fetchall(self):
            return list(self._rows)

    class _JobDB:
        def __init__(self):
            self.export_jobs: dict[str, dict[str, object]] = {}
            self.import_jobs: dict[str, dict[str, object]] = {}

        def get_connection(self):
            return MagicMock()

        def execute_query(self, sql, params=(), commit=False):
            sql_norm = " ".join(str(sql).strip().split()).lower()
            if sql_norm.startswith("create table if not exists export_jobs"):
                return _DummyCursor([])
            if sql_norm.startswith("create table if not exists import_jobs"):
                return _DummyCursor([])

            if sql_norm.startswith("insert or replace into export_jobs"):
                (
                    job_id, user_id, status, chatbook_name, output_path,
                    created_at, started_at, completed_at, error_message,
                    progress_percentage, total_items, processed_items,
                    file_size_bytes, download_url, expires_at,
                ) = params
                self.export_jobs[job_id] = {
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

            if sql_norm.startswith("insert or replace into import_jobs"):
                (
                    job_id, user_id, status, chatbook_path,
                    created_at, started_at, completed_at, error_message,
                    progress_percentage, total_items, processed_items,
                    successful_items, failed_items, skipped_items,
                    conflicts, warnings,
                ) = params
                self.import_jobs[job_id] = {
                    "job_id": job_id,
                    "user_id": user_id,
                    "status": status,
                    "chatbook_path": chatbook_path,
                    "created_at": created_at,
                    "started_at": started_at,
                    "completed_at": completed_at,
                    "error_message": error_message,
                    "progress_percentage": progress_percentage,
                    "total_items": total_items,
                    "processed_items": processed_items,
                    "successful_items": successful_items,
                    "failed_items": failed_items,
                    "skipped_items": skipped_items,
                    "conflicts": conflicts,
                    "warnings": warnings,
                }
                return _DummyCursor([])

            if sql_norm.startswith("select * from export_jobs where job_id"):
                job_id, user_id = params
                row = self.export_jobs.get(job_id)
                if not row or row.get("user_id") != user_id:
                    return _DummyCursor([])
                return _DummyCursor([row])

            if sql_norm.startswith("select * from import_jobs where job_id"):
                job_id, user_id = params
                row = self.import_jobs.get(job_id)
                if not row or row.get("user_id") != user_id:
                    return _DummyCursor([])
                return _DummyCursor([row])

            if sql_norm.startswith("update export_jobs set"):
                status, completed_at, error_message, progress_percentage, job_id, user_id = params
                row = self.export_jobs[job_id]
                if row.get("user_id") == user_id:
                    row["status"] = status
                    row["completed_at"] = completed_at
                    row["error_message"] = error_message
                    row["progress_percentage"] = progress_percentage
                return _DummyCursor([])

            if sql_norm.startswith("update import_jobs set"):
                status, completed_at, error_message, progress_percentage, job_id, user_id = params
                row = self.import_jobs[job_id]
                if row.get("user_id") == user_id:
                    row["status"] = status
                    row["completed_at"] = completed_at
                    row["error_message"] = error_message
                    row["progress_percentage"] = progress_percentage
                return _DummyCursor([])


            return _DummyCursor([])

    async def override_user():
        return User(id=1, username="tester", is_active=True)

    class _AuditService:
        async def log_event(self, *args, **kwargs):
            return None

    db = _JobDB()
    service = ChatbookService(user_id="1", db=db)
    service.temp_dir = tmp_dir / "temp"
    service.import_dir = tmp_dir / "imports"
    service.export_dir = tmp_dir / "exports"
    for directory in (service.temp_dir, service.import_dir, service.export_dir):
        directory.mkdir(parents=True, exist_ok=True)

    lagging_rows = {
        "export-job-1": {"status": "processing"},
        "import-job-1": {"status": "queued"},
    }

    def _lagging_get_job(job_id: str, job_type: str):
        return lagging_rows.get(job_id)

    if service._jobs_adapter is None:
        from tldw_Server_API.app.core.Chatbooks.jobs_adapter import ChatbooksJobsAdapter

        service._jobs_adapter = ChatbooksJobsAdapter(owner_user_id=service.user_id)
    service._jobs_adapter._get_job = _lagging_get_job  # type: ignore[method-assign]

    async def _fake_create_chatbook(**kwargs):
        job_id = "export-job-1"
        job = ExportJob(
            job_id=job_id,
            user_id=service.user_id,
            status=ExportStatus.PENDING,
            chatbook_name=kwargs["name"],
            created_at=datetime.now(timezone.utc),
        )
        service._save_export_job(job)
        return True, "queued", job_id

    async def _fake_import_chatbook(**kwargs):
        job_id = "import-job-1"
        job = ImportJob(
            job_id=job_id,
            user_id=service.user_id,
            status=ImportStatus.PENDING,
            chatbook_path=str(kwargs["file_path"]),
            created_at=datetime.now(timezone.utc),
        )
        service._save_import_job(job)
        return True, "queued", job_id

    service.create_chatbook = _fake_create_chatbook
    service.import_chatbook = _fake_import_chatbook

    app.dependency_overrides[get_request_user] = override_user
    app.dependency_overrides[chatbooks_mod.get_chatbook_service] = lambda: service
    app.dependency_overrides[get_audit_service_for_user] = lambda: _AuditService()

    try:
        with TestClient(app) as c:
            yield c
    finally:
        app.dependency_overrides.pop(get_request_user, None)
        app.dependency_overrides.pop(chatbooks_mod.get_chatbook_service, None)
        app.dependency_overrides.pop(get_audit_service_for_user, None)


def _make_export_payload(async_mode: bool = True):
    return {
        "name": "Cancel Test",
        "description": "Testing cancellation",
        "content_selections": {},
        "async_mode": async_mode,
    }


def _make_chatbook_bytes() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w") as zf:
        manifest = {
            "version": "1.0.0",
            "name": "Cancel Import",
            "description": "Test",
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
            "content_items": [],
            "configuration": {},
            "statistics": {},
            "metadata": {},
            "user_info": {"user_id": "test"},
        }
        zf.writestr("manifest.json", json.dumps(manifest))
    return buf.getvalue()


def test_cancel_export_job_flow(client):
    # Start async export job
    resp = client.post("/api/v1/chatbooks/export", json=_make_export_payload(async_mode=True))
    assert resp.status_code == 200, f"unexpected export status {resp.status_code}: {resp.text}"
    job_id = resp.json().get("job_id")
    assert job_id

    # Try to cancel
    cresp = client.delete(f"/api/v1/chatbooks/export/jobs/{job_id}")
    assert cresp.status_code == 200

    # Check job status using the lagging jobs row to exercise reconciliation.
    sresp = client.get(f"/api/v1/chatbooks/export/jobs/{job_id}")
    assert sresp.status_code == 200
    status = sresp.json().get("status")
    assert status == "cancelled"


def test_cancel_import_job_flow(client):
    # Prepare small chatbook upload
    data = _make_chatbook_bytes()
    files = {"file": ("test.chatbook", data, "application/zip")}

    # async_mode via query params (schema uses Depends to parse)
    resp = client.post("/api/v1/chatbooks/import?async_mode=true", files=files)
    assert resp.status_code == 200, f"unexpected import status {resp.status_code}: {resp.text}"
    job_id = resp.json().get("job_id")
    assert job_id

    cresp = client.delete(f"/api/v1/chatbooks/import/jobs/{job_id}")
    assert cresp.status_code == 200

    sresp = client.get(f"/api/v1/chatbooks/import/jobs/{job_id}")
    assert sresp.status_code == 200
    status = sresp.json().get("status")
    assert status == "cancelled"
