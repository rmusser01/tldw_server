import io
import json
import zipfile
from datetime import datetime

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints import chatbooks as chatbooks_endpoints
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.core.Chatbooks.chatbook_models import (
    ChatbookManifest,
    ChatbookVersion,
    ExportJob,
    ExportStatus,
    ImportJob,
    ImportStatus,
)
from tldw_Server_API.app.core.Chatbooks.exceptions import JobError


class _DummyAuditService:
    async def log_event(self, *args, **kwargs) -> None:
        return None


async def _override_user() -> User:
    return User(id=1, username="tester", email=None, is_active=True)


def _make_app(service) -> FastAPI:
    app = FastAPI()
    app.include_router(chatbooks_endpoints.router, prefix="/api/v1")
    app.dependency_overrides[chatbooks_endpoints.get_chatbook_service] = lambda: service
    app.dependency_overrides[chatbooks_endpoints.get_request_user] = _override_user
    app.dependency_overrides[chatbooks_endpoints.get_audit_service_for_user] = lambda: _DummyAuditService()
    return app


def _make_chatbook_bytes() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w") as zf:
        manifest = {
            "version": "1.0.0",
            "name": "Preview Test",
            "description": "Test manifest",
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


class _MissingJobService:
    db = None

    def cancel_export_job(self, job_id: str):
        raise JobError(f"Export job {job_id} not found", job_id=job_id)

    def cancel_import_job(self, job_id: str):
        raise JobError(f"Import job {job_id} not found", job_id=job_id)

    def delete_export_job(self, job_id: str):
        raise JobError(f"Export job {job_id} not found", job_id=job_id)

    def delete_import_job(self, job_id: str):
        raise JobError(f"Import job {job_id} not found", job_id=job_id)


class _InvalidTransitionService:
    db = None

    def cancel_export_job(self, _job_id: str):
        return False

    def cancel_import_job(self, _job_id: str):
        return False

    def delete_export_job(self, _job_id: str):
        return False

    def delete_import_job(self, _job_id: str):
        return False


class _PreviewStatsService:
    db = None

    def preview_chatbook(self, _file_path: str):
        manifest = ChatbookManifest(
            version=ChatbookVersion.V1,
            name="stats",
            description="preview",
            created_at=datetime(2024, 1, 1),
            updated_at=datetime(2024, 1, 1),
            total_prompts=7,
            total_evaluations=5,
            total_embeddings=9,
        )
        return manifest, None


class _PreviewInvalidManifestService:
    db = None

    def preview_chatbook(self, _file_path: str):
        return None, "Invalid chatbook manifest"


class _PreviewExplodingService:
    db = None

    def preview_chatbook(self, _file_path: str):
        raise RuntimeError("preview exploded")


class _ListJobsService:
    db = None

    def __init__(self) -> None:
        created = datetime(2024, 1, 1)
        self._export_jobs = [
            ExportJob(
                job_id=f"export-{index}",
                user_id="1",
                status=ExportStatus.PENDING,
                chatbook_name=f"Export {index}",
                created_at=created,
            )
            for index in range(3)
        ]
        self._import_jobs = [
            ImportJob(
                job_id=f"import-{index}",
                user_id="1",
                status=ImportStatus.PENDING,
                chatbook_path=f"/tmp/import-{index}.chatbook",
                created_at=created,
            )
            for index in range(3)
        ]

    def count_export_jobs(self) -> int:
        return len(self._export_jobs)

    def list_export_jobs(self, *, limit: int, offset: int):
        return self._export_jobs[offset:offset + limit]

    def count_import_jobs(self) -> int:
        return len(self._import_jobs)

    def list_import_jobs(self, *, limit: int, offset: int):
        return self._import_jobs[offset:offset + limit]


class _ContinuationFailedService:
    db = None

    async def continue_chatbook_export(self, **kwargs):
        _ = kwargs
        return False, "continuation backend exploded at /private/db/path", None


@pytest.mark.parametrize(
    "path",
    [
        "/api/v1/chatbooks/export/jobs/missing",
        "/api/v1/chatbooks/import/jobs/missing",
        "/api/v1/chatbooks/export/jobs/missing/remove",
        "/api/v1/chatbooks/import/jobs/missing/remove",
    ],
)
def test_job_endpoints_map_joberror_to_404(path):
    app = _make_app(_MissingJobService())

    with TestClient(app) as client:
        response = client.delete(path)

    assert response.status_code == 404
    assert "not found" in response.json().get("detail", "").lower()


@pytest.mark.parametrize(
    "path",
    [
        "/api/v1/chatbooks/export/jobs/test-job",
        "/api/v1/chatbooks/import/jobs/test-job",
        "/api/v1/chatbooks/export/jobs/test-job/remove",
        "/api/v1/chatbooks/import/jobs/test-job/remove",
    ],
)
def test_job_endpoints_keep_invalid_transition_as_400(path):
    app = _make_app(_InvalidTransitionService())

    with TestClient(app) as client:
        response = client.delete(path)

    assert response.status_code == 400


def test_preview_preserves_prompt_eval_embedding_stats():
    app = _make_app(_PreviewStatsService())
    files = {"file": ("preview.chatbook", _make_chatbook_bytes(), "application/zip")}

    with TestClient(app) as client:
        response = client.post("/api/v1/chatbooks/preview", files=files)

    assert response.status_code == 200, response.text
    manifest = response.json().get("manifest", {})
    assert manifest.get("total_prompts") == 7
    assert manifest.get("total_evaluations") == 5
    assert manifest.get("total_embeddings") == 9


def test_preview_maps_service_validation_errors_to_400():
    app = _make_app(_PreviewInvalidManifestService())
    files = {"file": ("preview.chatbook", _make_chatbook_bytes(), "application/zip")}

    with TestClient(app) as client:
        response = client.post("/api/v1/chatbooks/preview", files=files)

    assert response.status_code == 400
    assert response.json().get("detail") == "Invalid chatbook manifest"


def test_preview_maps_unexpected_service_failures_to_500():
    app = _make_app(_PreviewExplodingService())
    files = {"file": ("preview.chatbook", _make_chatbook_bytes(), "application/zip")}

    with TestClient(app) as client:
        response = client.post("/api/v1/chatbooks/preview", files=files)

    assert response.status_code == 500
    assert response.json().get("detail") == "An error occurred while previewing the chatbook"


def test_job_list_routes_include_canonical_pagination():
    app = _make_app(_ListJobsService())

    with TestClient(app) as client:
        export_response = client.get("/api/v1/chatbooks/export/jobs?limit=2&offset=0")
        import_response = client.get("/api/v1/chatbooks/import/jobs?limit=1&offset=1")

    assert export_response.status_code == 200, export_response.text
    export_payload = export_response.json()
    assert len(export_payload["jobs"]) == 2
    assert export_payload["total"] == 3
    assert export_payload["has_more"] is True
    assert export_payload["next_offset"] == 2
    assert export_payload["pagination"] == {
        "mode": "offset",
        "total": 3,
        "limit": 2,
        "offset": 0,
        "has_more": True,
        "next_offset": 2,
    }

    assert import_response.status_code == 200, import_response.text
    import_payload = import_response.json()
    assert len(import_payload["jobs"]) == 1
    assert import_payload["total"] == 3
    assert import_payload["has_more"] is True
    assert import_payload["next_offset"] == 2
    assert import_payload["pagination"] == {
        "mode": "offset",
        "total": 3,
        "limit": 1,
        "offset": 1,
        "has_more": True,
        "next_offset": 2,
    }


def test_continue_export_sanitizes_service_failure_message():
    app = _make_app(_ContinuationFailedService())

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/chatbooks/export/continue",
            json={
                "export_id": "export-1",
                "continuations": [{"kind": "evaluations", "offset": 100}],
            },
        )

    assert response.status_code == 500
    assert response.json().get("detail") == "An error occurred while continuing the chatbook export"


def test_health_sanitizes_storage_base_failure(monkeypatch):
    def _explode_storage_base():
        raise RuntimeError("chatbook storage exploded at /private/chatbooks")

    monkeypatch.setattr(
        chatbooks_endpoints.DatabasePaths,
        "get_user_db_base_dir",
        _explode_storage_base,
    )
    app = _make_app(object())

    with TestClient(app) as client:
        response = client.get("/api/v1/chatbooks/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "unhealthy"
    assert body["error"] == "Chatbooks health check failed"
