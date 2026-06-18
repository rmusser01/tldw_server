import json
import shutil
import zipfile
from pathlib import Path

import jsonschema
import pytest

from tldw_Server_API.app.api.v1.schemas.chatbook_schemas import CreateChatbookRequest
from tldw_Server_API.app.core.Chatbooks.chatbook_models import ChatbookVersion
from tldw_Server_API.app.core.Chatbooks.chatbook_service import ChatbookService
from tldw_Server_API.app.core.Chatbooks.services.jobs_worker import _handle_export
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


@pytest.fixture
def chatbook_v1_1_service(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "test")
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path))

    db_path = tmp_path / "chatbook_v1_1.db"
    db = CharactersRAGDB(db_path=str(db_path), client_id="chatbook-v1-1")
    service = ChatbookService(user_id="test_user", db=db)

    yield service

    if hasattr(service, "temp_dir") and service.temp_dir.exists():
        shutil.rmtree(service.temp_dir, ignore_errors=True)


def _load_v1_1_schema() -> dict:
    schema_path = Path(__file__).resolve().parents[3] / "Docs" / "Schemas" / "chatbooks_manifest_v1_1.json"
    return json.loads(schema_path.read_text(encoding="utf-8"))


def _minimal_v1_1_manifest() -> dict:
    return {
        "version": "1.1.0",
        "name": "v1.1 contract",
        "description": "contract",
        "author": None,
        "created_at": "2026-06-18T12:00:00+00:00",
        "updated_at": "2026-06-18T12:00:00+00:00",
        "export_id": "contract-export",
        "content_items": [],
        "relationships": [],
        "configuration": {
            "include_media": False,
            "include_embeddings": False,
            "include_generated_content": True,
            "media_quality": "compressed",
            "max_file_size_mb": 100,
        },
        "statistics": {
            "total_conversations": 0,
            "total_notes": 0,
            "total_characters": 0,
            "total_media_items": 0,
            "total_prompts": 0,
            "total_evaluations": 0,
            "total_embeddings": 0,
            "total_world_books": 0,
            "total_dictionaries": 0,
            "total_documents": 0,
            "total_explainer_sessions": 0,
            "total_size_bytes": 0,
        },
        "metadata": {
            "tags": [],
            "categories": [],
            "language": "en",
            "license": None,
        },
        "user_info": {"user_id": None},
        "features_used": [],
        "producer": {"name": "tldw_server"},
        "source_instance": {},
        "compatibility": {"min_reader_version": "1.0.0"},
        "file_inventory": [],
    }


def test_chatbook_version_accepts_v1_1():
    assert ChatbookVersion("1.1.0") is ChatbookVersion.V1_1


def test_create_chatbook_request_accepts_format_version_v1_1():
    request = CreateChatbookRequest(
        name="v1.1",
        description="v1.1",
        content_selections={},
        format_version="1.1.0",
    )

    assert request.format_version is ChatbookVersion.V1_1


@pytest.mark.asyncio
async def test_create_chatbook_export_writes_requested_v1_1_manifest_version(chatbook_v1_1_service):
    success, _message, archive_path = await chatbook_v1_1_service.create_chatbook(
        name="v1.1",
        description="v1.1",
        content_selections={},
        format_version=ChatbookVersion.V1_1,
    )

    assert success is True
    assert archive_path is not None

    with zipfile.ZipFile(archive_path, "r") as zf:
        manifest = json.loads(zf.read("manifest.json"))

    assert manifest["version"] == "1.1.0"


@pytest.mark.asyncio
async def test_prompt_studio_export_payload_includes_format_version(chatbook_v1_1_service):
    captured_payload = {}

    class _PromptStudioAdapter:
        def create_export_job(self, payload, request_id=None):
            captured_payload.update(payload)
            return {"id": 123}

    chatbook_v1_1_service._jobs_backend = "prompt_studio"
    chatbook_v1_1_service._ps_job_adapter = _PromptStudioAdapter()

    success, _message, job_id = await chatbook_v1_1_service.create_chatbook(
        name="v1.1 async",
        description="v1.1 async",
        content_selections={},
        format_version=ChatbookVersion.V1_1,
        async_mode=True,
    )

    assert success is True
    assert job_id == "123"
    assert captured_payload["format_version"] == "1.1.0"


@pytest.mark.asyncio
async def test_core_jobs_worker_forwards_format_version_to_archive_creation():
    captured_kwargs = {}

    class _Service:
        def _claim_export_job(self, job_id):
            return True

        async def _create_chatbook_sync_wrapper(self, **kwargs):
            captured_kwargs.update(kwargs)
            return True, "ok", "/tmp/export.chatbook"

        def _get_export_job(self, job_id):
            return None

    result = await _handle_export(
        _Service(),
        {
            "name": "v1.1 queued",
            "description": "v1.1 queued",
            "content_selections": {},
            "format_version": "1.1.0",
        },
        "job-1",
    )

    assert result == {"path": "/tmp/export.chatbook", "download_url": None}
    assert captured_kwargs["format_version"] == ChatbookVersion.V1_1


def test_minimal_v1_1_manifest_matches_schema():
    jsonschema.validate(_minimal_v1_1_manifest(), _load_v1_1_schema())


def test_v1_1_manifest_allows_content_item_metadata_envelope():
    manifest = _minimal_v1_1_manifest()
    manifest["features_used"] = ["content_envelopes"]
    manifest["content_items"] = [
        {
            "id": "exp_123",
            "type": "explainer_session",
            "title": "Learn attention",
            "description": None,
            "created_at": None,
            "updated_at": None,
            "tags": [],
            "metadata": {
                "format": "tldw.explainer_session.v1",
                "envelope": {
                    "format": "tldw.explainer_session.v1",
                    "schema_version": 1,
                    "representations": [],
                    "integrity": {},
                    "lossiness": {},
                    "provenance": {},
                    "source_refs": [],
                },
            },
            "file_path": "content/explainer_sessions/session_exp_123.json",
            "checksum": "sha256:example",
        }
    ]

    jsonschema.validate(manifest, _load_v1_1_schema())
