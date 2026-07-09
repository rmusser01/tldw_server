import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints import chatbooks as chatbooks_endpoints
from tldw_Server_API.app.api.v1.schemas.chatbook_schemas import CreateChatbookRequest
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.core.Chat.document_generator import (
    DocumentGeneratorService,
    DocumentType as GeneratedDocumentType,
)
from tldw_Server_API.app.core.Chatbooks.chatbook_models import (
    FULL_ACCOUNT_EXPORT_MODE,
    ChatbookVersion,
    ContentType,
    ExportJob,
    ExportStatus,
)
from tldw_Server_API.app.core.Chatbooks.chatbook_account_inventory import ACCOUNT_DATA_INVENTORY
from tldw_Server_API.app.core.Chatbooks.chatbook_service import ChatbookService
from tldw_Server_API.app.core.Chatbooks.services.jobs_worker import ChatbooksJobError, _handle_export
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.DB_Management.media_db.api import create_media_database


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


def _seed_owned_media_artifact(tmp_path, monkeypatch):
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path))
    user_root = DatabasePaths.resolve_user_base_directory(1)
    user_root.mkdir(parents=True, exist_ok=True)
    media_db = create_media_database("1", db_path=str(user_root / DatabasePaths.MEDIA_DB_NAME))
    media_id, _media_uuid, _message = media_db.add_media_with_keywords(
        url="https://example.test/source.txt",
        title="Stored media",
        media_type="document",
        content="stored media body",
        keywords=[],
    )
    artifact_bytes = b"account-owned stored media bytes"
    artifact_path = user_root / "stored_media" / "source.txt"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_bytes(artifact_bytes)
    media_db.insert_media_file(
        media_id=int(media_id),
        file_type="original",
        storage_path="stored_media/source.txt",
        original_filename="source.txt",
        file_size=len(artifact_bytes),
        mime_type="text/plain",
        checksum="sha256:test",
    )
    return media_id, artifact_bytes


def _seed_unresolved_media_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path))
    user_root = DatabasePaths.resolve_user_base_directory(1)
    user_root.mkdir(parents=True, exist_ok=True)
    media_db = create_media_database("1", db_path=str(user_root / DatabasePaths.MEDIA_DB_NAME))
    media_id, _media_uuid, _message = media_db.add_media_with_keywords(
        url="https://example.test/pointer-only",
        title="Pointer media",
        media_type="document",
        content="pointer body",
        keywords=[],
    )
    for index in range(2):
        media_db.insert_media_file(
            media_id=int(media_id),
            file_type=f"derived-{index}",
            storage_path=f"missing_media/artifact-{index}.bin",
            original_filename=f"artifact-{index}.bin",
            file_size=32 + index,
            mime_type="application/octet-stream",
            checksum=f"sha256:missing-{index}",
        )
    return media_id


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
async def test_full_account_export_expands_to_existing_content_and_manifest_summary(tmp_path, monkeypatch):
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path))
    secret_user_id = "SECRET_USER_ID_SHOULD_NOT_APPEAR"
    db = CharactersRAGDB(db_path=str(tmp_path / "chatbooks.db"), client_id="chatbooks-contract")
    note_id = db.add_note(title="Full Account Note", content="exported body")
    service = ChatbookService(user_id=secret_user_id, db=db, user_id_int=1)

    success, message, file_path = await service.create_chatbook(
        name="full account",
        description="full account archive",
        content_selections=None,
        format_version=ChatbookVersion.V1_1,
        async_mode=False,
    )

    assert success is True, message
    assert file_path is not None
    with zipfile.ZipFile(file_path) as zf:
        names = set(zf.namelist())
        manifest = json.loads(zf.read("manifest.json"))

    note_path = f"content/notes/note_{note_id}.md"
    assert note_path in names
    assert any(item["type"] == "note" and item["file_path"] == note_path for item in manifest["content_items"])
    assert {row["category"] for row in manifest["account_inventory"]} == {
        row.category for row in ACCOUNT_DATA_INVENTORY
    }
    summary = manifest["account_inventory_summary"]
    assert summary["counts"]["notes"] == 1
    assert summary["counts"]["media_pointers"] == 0
    assert summary["pointer_only_count"] == 0
    assert summary["sensitive_category_count"] >= 1
    assert summary["warning_count"] >= 1
    assert summary["archive_size_bytes"] == Path(file_path).stat().st_size
    assert summary["post_write_verification"] is True
    assert manifest["statistics"]["notes"] == 1
    assert manifest["statistics"]["account_profiles"] == 1
    assert secret_user_id not in json.dumps(manifest)


@pytest.mark.asyncio
async def test_full_account_export_includes_generated_documents_even_when_option_is_false(tmp_path, monkeypatch):
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path))
    db = CharactersRAGDB(db_path=str(tmp_path / "chatbooks.db"), client_id="chatbooks-contract")
    doc_service = DocumentGeneratorService(db, user_id="1")
    document_id = doc_service._save_generated_document(
        conversation_id=None,
        document_type=GeneratedDocumentType.SUMMARY,
        title="Generated summary",
        content="Generated account-owned content",
        provider="test-provider",
        model="test-model",
        generation_time_ms=25,
        token_count=8,
    )
    service = ChatbookService(user_id="doc-user", db=db, user_id_int=1)

    success, message, file_path = await service.create_chatbook(
        name="full generated account",
        description="full account archive with generated documents",
        content_selections=None,
        include_generated_content=False,
        format_version=ChatbookVersion.V1_1,
        async_mode=False,
    )

    assert success is True, message
    assert file_path is not None
    with zipfile.ZipFile(file_path) as zf:
        names = set(zf.namelist())
        manifest = json.loads(zf.read("manifest.json"))

    document_path = f"content/generated_documents/document_{document_id}.json"
    assert document_path in names
    assert manifest["configuration"]["include_generated_content"] is True
    assert manifest["account_inventory_summary"]["counts"]["generated_documents"] == 1


@pytest.mark.asyncio
async def test_full_account_export_bundles_owned_media_artifact_bytes(tmp_path, monkeypatch):
    media_id, artifact_bytes = _seed_owned_media_artifact(tmp_path, monkeypatch)
    db = CharactersRAGDB(db_path=str(tmp_path / "chatbooks.db"), client_id="chatbooks-contract")
    service = ChatbookService(user_id="media-user", db=db, user_id_int=1)

    success, message, file_path = await service.create_chatbook(
        name="full media account",
        description="full account archive with stored media",
        content_selections=None,
        format_version=ChatbookVersion.V1_1,
        async_mode=False,
    )

    assert success is True, message
    assert file_path is not None
    with zipfile.ZipFile(file_path) as zf:
        names = set(zf.namelist())
        manifest = json.loads(zf.read("manifest.json"))
        media_payload = json.loads(zf.read(f"content/media/media_{media_id}.json"))
        stored_artifact = media_payload["stored_artifacts"][0]
        archived_artifact_path = stored_artifact["archive_path"]
        archived_artifact_bytes = zf.read(archived_artifact_path)

    assert f"content/media/media_{media_id}.json" in names
    assert archived_artifact_path in names
    assert archived_artifact_bytes == artifact_bytes
    assert stored_artifact["bundled"] is True
    assert stored_artifact["pointer_only"] is False
    assert "storage_path" not in stored_artifact
    summary = manifest["account_inventory_summary"]
    assert summary["counts"]["media_records"] == 1
    assert summary["counts"]["media_stored_artifacts"] == 1
    assert summary["counts"]["media_pointers"] == 1


@pytest.mark.asyncio
async def test_full_account_export_reports_each_unresolved_media_artifact_pointer(tmp_path, monkeypatch):
    media_id = _seed_unresolved_media_artifacts(tmp_path, monkeypatch)
    db = CharactersRAGDB(db_path=str(tmp_path / "chatbooks.db"), client_id="chatbooks-contract")
    service = ChatbookService(user_id="media-user", db=db, user_id_int=1)

    success, message, file_path = await service.create_chatbook(
        name="full pointer account",
        description="full account archive with unresolved stored artifacts",
        content_selections=None,
        format_version=ChatbookVersion.V1_1,
        async_mode=False,
    )

    assert success is True, message
    assert file_path is not None
    with zipfile.ZipFile(file_path) as zf:
        manifest = json.loads(zf.read("manifest.json"))
        media_payload = json.loads(zf.read(f"content/media/media_{media_id}.json"))

    assert len(media_payload["stored_artifacts"]) == 2
    assert all(artifact["pointer_only"] is True for artifact in media_payload["stored_artifacts"])
    assert all("archive_path" not in artifact for artifact in media_payload["stored_artifacts"])
    assert all("storage_path" not in artifact for artifact in media_payload["stored_artifacts"])
    summary = manifest["account_inventory_summary"]
    assert summary["counts"]["media_pointers"] == 3
    assert summary["pointer_only_count"] == 3
    assert any(
        "2 stored media artifact pointer(s)" in warning
        for warning in summary["warnings"]
    )
    assert summary["warning_count"] == len(summary["warnings"])


@pytest.mark.asyncio
async def test_explicit_media_export_respects_include_media_false(tmp_path, monkeypatch):
    media_id, _artifact_bytes = _seed_owned_media_artifact(tmp_path, monkeypatch)
    db = CharactersRAGDB(db_path=str(tmp_path / "chatbooks.db"), client_id="chatbooks-contract")
    service = ChatbookService(user_id="media-user", db=db, user_id_int=1)

    success, message, file_path = await service.create_chatbook(
        name="metadata only media",
        description="explicit media archive without stored bytes",
        content_selections={ContentType.MEDIA: [str(media_id)]},
        include_media=False,
        format_version=ChatbookVersion.V1_1,
        async_mode=False,
    )

    assert success is True, message
    assert file_path is not None
    with zipfile.ZipFile(file_path) as zf:
        names = set(zf.namelist())
        manifest = json.loads(zf.read("manifest.json"))
        media_payload = json.loads(zf.read(f"content/media/media_{media_id}.json"))

    assert not any(name.startswith("content/media/files/") for name in names)
    assert media_payload["stored_artifacts"] == []
    assert manifest["account_inventory_summary"]["counts"]["media_stored_artifacts"] == 0


def test_full_account_explainer_session_listing_pages_all_sessions(tmp_path, monkeypatch):
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path))
    user_root = DatabasePaths.resolve_user_base_directory(1)
    user_root.mkdir(parents=True, exist_ok=True)
    (user_root / DatabasePaths.EXPLAINER_DB_NAME).touch()
    db = CharactersRAGDB(db_path=str(tmp_path / "chatbooks.db"), client_id="chatbooks-contract")
    service = ChatbookService(user_id="1", db=db, user_id_int=1)
    offsets_seen = []

    class _Repo:
        def list_session_summaries(self, *, owner_user_id, limit, offset, include_archived=False):
            del owner_user_id, include_archived
            offsets_seen.append(offset)
            total = 250
            end = min(offset + limit, total)
            return [SimpleNamespace(id=f"session-{index}") for index in range(offset, end)], total

    monkeypatch.setattr(service, "_get_explainer_repo", lambda: _Repo())

    session_ids = service._list_all_explainer_session_ids()

    assert len(session_ids) == 250
    assert session_ids[0] == "session-0"
    assert session_ids[-1] == "session-249"
    assert offsets_seen == [0, 100, 200]


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
