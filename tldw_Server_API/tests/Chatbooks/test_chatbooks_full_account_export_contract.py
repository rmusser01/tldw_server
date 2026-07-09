from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints import chatbooks as chatbooks_endpoints
from tldw_Server_API.app.api.v1.schemas.chatbook_schemas import CreateChatbookRequest
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.core.Chatbooks.chatbook_models import (
    FULL_ACCOUNT_EXPORT_MODE,
    ChatbookVersion,
    ContentType,
    ExportJob,
    ExportStatus,
)
from tldw_Server_API.app.core.Chatbooks.chatbook_service import ChatbookService
from tldw_Server_API.app.core.Chatbooks.services.jobs_worker import ChatbooksJobError, _handle_export
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


class _DummyAuditService:
    async def log_event(self, *args, **kwargs) -> None:
        return None


class _PassingQuotaManager:
    def __init__(self, *args, **kwargs) -> None:
        return None

    async def check_export_quota(self):
        return True, "ok"

    async def check_concurrent_jobs(self):
        return True, "ok"


async def _override_user() -> User:
    return User(id=1, username="tester", email=None, is_active=True)


class _CaptureExportService:
    db = None

    def __init__(self) -> None:
        self.calls = []

    async def create_chatbook(self, **kwargs):
        self.calls.append(kwargs)
        return True, "queued", "job-1"


@pytest.fixture
def export_client(monkeypatch):
    service = _CaptureExportService()
    app = FastAPI()
    app.include_router(chatbooks_endpoints.router, prefix="/api/v1")
    app.dependency_overrides[chatbooks_endpoints.get_chatbook_service] = lambda: service
    app.dependency_overrides[chatbooks_endpoints.get_request_user] = _override_user
    app.dependency_overrides[chatbooks_endpoints.get_audit_service_for_user] = lambda: _DummyAuditService()
    monkeypatch.setattr(chatbooks_endpoints, "QuotaManager", _PassingQuotaManager)
    return TestClient(app), service


def _payload(**overrides):
    payload = {
        "name": "Account export",
        "description": "Contract check",
        "async_mode": True,
    }
    payload.update(overrides)
    return payload


def test_create_chatbook_schema_allows_omitted_content_selections():
    request = CreateChatbookRequest(name="backup", description="all")

    assert request.content_selections is None


def test_create_chatbook_schema_treats_empty_content_selections_as_full_account():
    request = CreateChatbookRequest(name="backup", description="all", content_selections={})

    assert request.content_selections == {}


@pytest.mark.parametrize("body", [_payload(), _payload(content_selections={})])
def test_export_endpoint_passes_full_account_marker_for_omitted_or_empty_selections(export_client, body):
    client, service = export_client

    response = client.post("/api/v1/chatbooks/export", json=body)

    assert response.status_code == 200
    assert service.calls[-1]["content_selections"] is None


def test_export_endpoint_preserves_explicit_allowlist_with_empty_arrays(export_client):
    client, service = export_client

    response = client.post(
        "/api/v1/chatbooks/export",
        json=_payload(content_selections={"conversation": ["conv-1"], "note": []}),
    )

    assert response.status_code == 200
    assert service.calls[-1]["content_selections"] == {
        ContentType.CONVERSATION: ["conv-1"],
        ContentType.NOTE: [],
    }


@pytest.mark.asyncio
async def test_service_rejects_zero_item_allowlist_before_creating_archive(tmp_path, monkeypatch):
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path))
    db = CharactersRAGDB(db_path=str(tmp_path / "chatbooks.db"), client_id="chatbooks-contract")
    service = ChatbookService(user_id=1, db=db, user_id_int=1)

    called = False

    async def _unexpected_archive_creation(*args, **kwargs):
        nonlocal called
        called = True
        return True, "created", str(Path("empty.chatbook"))

    monkeypatch.setattr(service, "_create_chatbook_sync_wrapper", _unexpected_archive_creation)

    success, message, result = await service.create_chatbook(
        name="empty",
        description="empty allowlist",
        content_selections={ContentType.CONVERSATION: []},
        async_mode=False,
    )

    assert success is False
    assert result is None
    assert called is False
    assert "allowlist" in message.lower()


@pytest.mark.asyncio
async def test_service_async_payload_preserves_full_account_selection_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path))
    db = CharactersRAGDB(db_path=str(tmp_path / "chatbooks.db"), client_id="chatbooks-contract")
    service = ChatbookService(user_id=1, db=db, user_id_int=1)
    captured_payload = {}

    def _capture_create_job(_self, **kwargs):
        captured_payload.update(kwargs["payload"])
        return {"uuid": "core-job-123"}

    monkeypatch.setattr(service, "_save_export_job_with_quota", lambda _job: None)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Jobs.manager.JobManager.create_job",
        _capture_create_job,
    )

    success, _message, _job_id = await service.create_chatbook(
        name="full account",
        description="async marker",
        content_selections=None,
        format_version=ChatbookVersion.V1,
        async_mode=True,
    )

    assert success is True
    assert captured_payload["selection_mode"] == FULL_ACCOUNT_EXPORT_MODE
    assert captured_payload["content_selections"] is None


@pytest.mark.asyncio
async def test_jobs_worker_preserves_full_account_mode_for_archive_creation():
    captured_kwargs = {}

    class _Service:
        def _claim_export_job(self, _job_id):
            return True

        async def _create_chatbook_sync_wrapper(self, **kwargs):
            captured_kwargs.update(kwargs)
            return True, "ok", "export.chatbook"

        def _get_export_job(self, _job_id):
            return None

    await _handle_export(
        _Service(),
        {
            "name": "full account queued",
            "description": "queued",
            "selection_mode": FULL_ACCOUNT_EXPORT_MODE,
            "content_selections": None,
        },
        "job-1",
    )

    assert captured_kwargs["selection_mode"] == FULL_ACCOUNT_EXPORT_MODE
    assert captured_kwargs["content_selections"] is None


@pytest.mark.asyncio
async def test_jobs_worker_treats_legacy_empty_mapping_payload_as_full_account():
    captured_kwargs = {}

    class _Service:
        def _claim_export_job(self, _job_id):
            return True

        async def _create_chatbook_sync_wrapper(self, **kwargs):
            captured_kwargs.update(kwargs)
            return True, "ok", "export.chatbook"

        def _get_export_job(self, _job_id):
            return None

    await _handle_export(
        _Service(),
        {
            "name": "legacy full account queued",
            "description": "queued",
            "content_selections": {},
        },
        "job-1",
    )

    assert captured_kwargs["selection_mode"] == FULL_ACCOUNT_EXPORT_MODE
    assert captured_kwargs["content_selections"] is None


@pytest.mark.asyncio
async def test_jobs_worker_marks_export_job_failed_for_zero_item_allowlist():
    saved_jobs = []
    export_job = ExportJob(
        job_id="job-1",
        user_id="user-1",
        status=ExportStatus.IN_PROGRESS,
        chatbook_name="empty allowlist",
    )

    class _Service:
        def _claim_export_job(self, _job_id):
            return True

        def _get_export_job(self, _job_id):
            return export_job

        def _save_export_job(self, job):
            saved_jobs.append(job)

    with pytest.raises(ChatbooksJobError, match="allowlist") as exc_info:
        await _handle_export(
            _Service(),
            {
                "name": "empty allowlist",
                "description": "queued",
                "selection_mode": "allowlist",
                "content_selections": {"conversation": []},
            },
            "job-1",
        )

    assert exc_info.value.retryable is False
    assert len(saved_jobs) == 1
    assert saved_jobs[0].status is ExportStatus.FAILED
    assert saved_jobs[0].completed_at is not None
    assert "allowlist" in saved_jobs[0].error_message
