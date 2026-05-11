from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal, get_request_user
from tldw_Server_API.app.api.v1.API_Deps.jobs_deps import get_job_manager
from tldw_Server_API.app.api.v1.endpoints import chatbooks as chatbooks_mod
from tldw_Server_API.app.api.v1.schemas import chatbook_schemas
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal


pytestmark = pytest.mark.unit


class _DummyChatbookService:
    db = None

    def __init__(self) -> None:
        self.preview_calls: list[dict[str, Any]] = []

    def preview_openwebui_attachment_hydration(self, **kwargs) -> dict[str, Any]:
        self.preview_calls.append(kwargs)
        return {
            "summary": {
                "referenced_files": 1,
                "resolved_files": 1,
                "image_files": 1,
                "media_files": 0,
                "missing_files": 0,
                "unsupported_files": 0,
                "failed_files": 0,
                "warning_count": 1,
            },
            "items": [
                {
                    "conversation_id": "conv-a",
                    "message_id": "msg-a",
                    "file_id": "file-a",
                    "status": "resolved",
                    "source": "message_metadata",
                    "file_kind": "image",
                    "mime_type": "image/png",
                    "source_path": "/private/openwebui/uploads/file-a.png",
                }
            ],
            "warnings": ["Resolved source at /private/openwebui/uploads/file-a.png"],
        }


class _FakeJobsManager:
    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []
        self.rows: dict[int, dict[str, Any]] = {}
        self.rows_by_uuid: dict[str, dict[str, Any]] = {}

    def create_job(self, **kwargs) -> dict[str, Any]:
        self.created.append(kwargs)
        row = {
            "id": len(self.created),
            "uuid": f"job-uuid-{len(self.created)}",
            "domain": kwargs["domain"],
            "queue": kwargs["queue"],
            "job_type": kwargs["job_type"],
            "status": "queued",
            "owner_user_id": kwargs["owner_user_id"],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "payload": kwargs["payload"],
            "result": None,
            "error": None,
        }
        self.rows[int(row["id"])] = row
        self.rows_by_uuid[str(row["uuid"])] = row
        return row

    def get_job(self, job_id: int) -> dict[str, Any] | None:
        return self.rows.get(int(job_id))

    def get_job_by_uuid(self, job_uuid: str) -> dict[str, Any] | None:
        return self.rows_by_uuid.get(str(job_uuid))


async def _user() -> User:
    return User(id=1, username="tester", email=None, is_active=True)


async def _other_user() -> User:
    return User(id=2, username="other", email=None, is_active=True)


async def _single_user_principal() -> AuthPrincipal:
    return AuthPrincipal(kind="user", user_id=1, subject="single_user", roles=["admin"], permissions=["*"])


async def _admin_principal() -> AuthPrincipal:
    return AuthPrincipal(kind="user", user_id=1, roles=["admin"], permissions=[])


async def _admin_flag_principal() -> AuthPrincipal:
    return AuthPrincipal(kind="user", user_id=1, roles=[], permissions=[], is_admin=True)


async def _non_admin_principal() -> AuthPrincipal:
    return AuthPrincipal(kind="user", user_id=2, roles=[], permissions=[])


def _make_app(
    *,
    service: _DummyChatbookService | None = None,
    jobs: _FakeJobsManager | None = None,
    principal_override=_single_user_principal,
    user_override=_user,
) -> FastAPI:
    app = FastAPI()
    app.include_router(chatbooks_mod.router, prefix="/api/v1")
    app.dependency_overrides[chatbooks_mod.get_chatbook_service] = lambda: service or _DummyChatbookService()
    app.dependency_overrides[get_request_user] = user_override
    app.dependency_overrides[get_auth_principal] = principal_override
    if jobs is not None:
        app.dependency_overrides[get_job_manager] = lambda: jobs
    return app


def _hydration_payload() -> dict[str, Any]:
    return {
        "openwebui_data_root": "/srv/openwebui",
        "scope": {
            "conversation_ids": ["conv-a"],
            "source_user_id": "ow-user",
        },
        "process_supported_files": False,
    }


def test_hydration_schema_rejects_empty_conversation_ids():
    with pytest.raises(ValidationError):
        chatbook_schemas.OpenWebUIHydrationPreviewRequest(
            openwebui_data_root="/srv/openwebui",
            scope={"conversation_ids": [""]},
        )


def test_hydration_schema_rejects_blank_source_user_id():
    with pytest.raises(ValidationError):
        chatbook_schemas.OpenWebUIHydrationPreviewRequest(
            openwebui_data_root="/srv/openwebui",
            scope={"source_user_id": "   "},
        )


def test_preview_allows_single_user_and_redacts_source_paths():
    service = _DummyChatbookService()
    app = _make_app(service=service, principal_override=_single_user_principal)

    with TestClient(app) as client:
        resp = client.post("/api/v1/chatbooks/openwebui/hydration/preview", json=_hydration_payload())

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["summary"]["referenced_files"] == 1
    assert body["items"][0]["mime_type"] == "image/png"
    assert "/private/openwebui" not in resp.text
    assert service.preview_calls[0]["scope"]["conversation_ids"] == ["conv-a"]
    assert service.preview_calls[0]["scope"]["source_user_id"] == "ow-user"
    assert service.preview_calls[0]["process_supported_files"] is False


def test_preview_rejects_multi_user_non_admin():
    app = _make_app(principal_override=_non_admin_principal)

    with TestClient(app) as client:
        resp = client.post("/api/v1/chatbooks/openwebui/hydration/preview", json=_hydration_payload())

    assert resp.status_code == 403, resp.text


def test_hydration_job_creation_enqueues_core_job():
    jobs = _FakeJobsManager()
    app = _make_app(jobs=jobs, principal_override=_admin_flag_principal)

    with TestClient(app) as client:
        resp = client.post("/api/v1/chatbooks/openwebui/hydration/jobs", json=_hydration_payload())

    assert resp.status_code == 200, resp.text
    assert resp.json()["job_type"] == "openwebui_attachment_hydration"
    assert jobs.created[0]["domain"] == "chatbooks"
    assert jobs.created[0]["job_type"] == "openwebui_attachment_hydration"
    assert jobs.created[0]["owner_user_id"] == "1"
    assert jobs.created[0]["payload"]["scope"]["conversation_ids"] == ["conv-a"]


def test_hydration_job_status_rejects_multi_user_non_admin():
    jobs = _FakeJobsManager()
    jobs.create_job(
        domain="chatbooks",
        queue="default",
        job_type="openwebui_attachment_hydration",
        payload=_hydration_payload(),
        owner_user_id="1",
    )
    app = _make_app(jobs=jobs, principal_override=_non_admin_principal, user_override=_other_user)

    with TestClient(app) as client:
        resp = client.get("/api/v1/chatbooks/openwebui/hydration/jobs/1")

    assert resp.status_code == 403, resp.text


def test_hydration_job_status_returns_admin_visible_job():
    jobs = _FakeJobsManager()
    job = jobs.create_job(
        domain="chatbooks",
        queue="default",
        job_type="openwebui_attachment_hydration",
        payload=_hydration_payload(),
        owner_user_id="2",
    )
    job["result"] = {"warnings": ["Copied /private/openwebui/uploads/file-a.png"]}
    job["error"] = "Failed at /private/openwebui/uploads/file-a.png"
    app = _make_app(jobs=jobs, principal_override=_admin_principal, user_override=_user)

    with TestClient(app) as client:
        resp = client.get("/api/v1/chatbooks/openwebui/hydration/jobs/job-uuid-1")

    assert resp.status_code == 200, resp.text
    assert resp.json()["job_uuid"] == "job-uuid-1"
    assert resp.json()["owner_user_id"] == "2"
    assert "/private/openwebui" not in resp.text
