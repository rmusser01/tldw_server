from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tldw_Server_API.app.core.Chatbooks.chatbook_models import (
    ChatbookContent,
    ChatbookManifest,
    ChatbookVersion,
    ConflictResolution,
    ContentItem,
    ContentType,
)
from tldw_Server_API.app.core.Chatbooks.chatbook_service import ChatbookService
from tldw_Server_API.app.core.DB_Management.Explainer_DB import ExplainerDatabase
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.Explainer.chatbook_adapter import build_explainer_chatbook_payload
from tldw_Server_API.app.core.Explainer.repository import ExplainerRepository

pytestmark = pytest.mark.integration


@pytest.fixture()
def chatbook_env(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "test")
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path))
    yield tmp_path


def _mock_chacha_db():
    db = MagicMock()
    db.execute_query.return_value = []
    connection = MagicMock()
    connection.execute = MagicMock()
    connection.close = MagicMock()
    db.get_connection.return_value = connection
    return db


def _service(user_id: int) -> ChatbookService:
    return ChatbookService(user_id=str(user_id), user_id_int=user_id, db=_mock_chacha_db())


def _create_session_for_user(user_id: int):
    db = ExplainerDatabase(DatabasePaths.get_explainer_db_path(user_id), client_id=f"user-{user_id}")
    repo = ExplainerRepository(db)
    session = repo.create_session(
        owner_user_id=str(user_id),
        title="Learn attention",
        mode="goal",
        output_intent="explain",
        grounding="source_led",
        depth_preset="standard",
        selected_sources=[
            {
                "source_id": "media-42",
                "source_type": "media",
                "title": "Attention notes",
                "metadata": {"snapshotHash": "sha256:sourcehash"},
            }
        ],
        root_prompt="Explain transformer attention",
    )
    root_id = session.root_node_ids[0]
    repo.update_node(
        session.id,
        root_id,
        owner_user_id=str(user_id),
        body="Attention lets tokens share context.",
        status="complete",
        evidence_state="supported",
        citations=[
            {
                "source_id": "media-42",
                "source_type": "media",
                "title": "Attention notes",
                "excerpt": "Attention weights are computed from query-key similarity.",
                "location_label": "chunk 3",
                "snapshot_hash": "sha256:citationhash",
            }
        ],
    )
    loaded = repo.get_session(session.id, owner_user_id=str(user_id))
    assert loaded is not None
    return db, repo, loaded


def _stage_for_import(import_service: ChatbookService, archive_path: str) -> Path:
    staged = Path(import_service.import_dir) / Path(archive_path).name
    shutil.copyfile(archive_path, staged)
    return staged


def test_content_type_and_content_container_include_explainer_sessions():
    assert ContentType.EXPLAINER_SESSION.value == "explainer_session"
    content = ChatbookContent(explainer_sessions={"session-1": {"title": "Session"}})

    assert "session-1" in content.get_all_ids()


@pytest.mark.asyncio
async def test_create_chatbook_writes_single_explainer_session_item(chatbook_env):
    _db, _repo, session = _create_session_for_user(7)
    service = _service(7)

    success, message, archive_path = await service.create_chatbook(
        name="Explainer Export",
        description="Full Explainer session",
        content_selections={ContentType.EXPLAINER_SESSION: [session.id]},
        include_media=False,
        include_embeddings=False,
        include_generated_content=True,
        async_mode=False,
    )

    assert success is True, message
    assert archive_path is not None
    with zipfile.ZipFile(archive_path, "r") as zf:
        manifest = json.loads(zf.read("manifest.json"))
        names = set(zf.namelist())
        item = manifest["content_items"][0]
        payload = json.loads(zf.read(item["file_path"]))

    assert item["type"] == "explainer_session"
    assert item["id"] == session.id
    assert item["file_path"] == f"content/explainer_sessions/session_{session.id}.json"
    assert item["metadata"]["format"] == "tldw.explainer_session.v1"
    assert manifest["statistics"]["total_explainer_sessions"] == 1
    assert manifest["statistics"]["total_documents"] == 0
    assert "content/notes" not in " ".join(names)
    assert "content/generated_documents" not in " ".join(names)
    assert payload["type"] == "explainer_session"
    assert payload["structured"]["session"]["id"] == session.id
    assert payload["rendered"]["markdown"].startswith("# Learn attention")


@pytest.mark.asyncio
async def test_import_chatbook_restores_explainer_session_content_type(chatbook_env):
    _db, _repo, session = _create_session_for_user(7)
    export_service = _service(7)
    success, _message, archive_path = await export_service.create_chatbook(
        name="Explainer Export",
        description="Full Explainer session",
        content_selections={ContentType.EXPLAINER_SESSION: [session.id]},
        include_media=False,
        include_embeddings=False,
        include_generated_content=True,
        async_mode=False,
    )
    assert success is True
    assert archive_path is not None
    import_service = _service(8)
    staged = _stage_for_import(import_service, archive_path)

    success, message, details = import_service._import_chatbook_sync(
        file_path=str(staged),
        content_selections=None,
        conflict_resolution=ConflictResolution.SKIP,
        prefix_imported=False,
        import_media=False,
        import_embeddings=False,
    )

    assert success is True, message
    assert details is not None
    assert details["imported_items"]["explainer_session"] == 1
    restored_repo = ExplainerRepository(
        ExplainerDatabase(DatabasePaths.get_explainer_db_path(8), client_id="user-8")
    )
    sessions = restored_repo.list_sessions(owner_user_id="8")
    assert len(sessions) == 1
    restored = sessions[0]
    assert restored.id != session.id
    assert restored.title == "Learn attention"
    assert restored.selected_sources[0].metadata["resolutionStatus"] == "unresolved"
    assert restored.nodes[restored.root_node_ids[0]].citations[0].excerpt.startswith("Attention weights")


def test_import_generated_document_subtype_fallback_restores_explainer_session(chatbook_env):
    _db, repo, session = _create_session_for_user(7)
    payload = build_explainer_chatbook_payload(
        repo=repo,
        session_id=session.id,
        owner_user_id="7",
    )
    import_service = _service(8)
    archive_path = Path(import_service.import_dir) / "fallback.chatbook"
    manifest = ChatbookManifest(
        version=ChatbookVersion.V1,
        name="Fallback",
        description="Generated document fallback",
        content_items=[
            ContentItem(
                id=session.id,
                type=ContentType.GENERATED_DOCUMENT,
                title=session.title,
                file_path=f"content/generated_documents/document_{session.id}.json",
                metadata={
                    "subtype": "explainer_session",
                    "format": "tldw.explainer_session.v1",
                },
            )
        ],
    )
    generated_doc = {
        "type": "generated_document",
        "metadata": {"subtype": "explainer_session"},
        "content": payload,
    }
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr("manifest.json", json.dumps(manifest.to_dict()))
        zf.writestr(f"content/generated_documents/document_{session.id}.json", json.dumps(generated_doc))

    success, message, details = import_service._import_chatbook_sync(
        file_path=str(archive_path),
        content_selections=None,
        conflict_resolution=ConflictResolution.SKIP,
        prefix_imported=False,
        import_media=False,
        import_embeddings=False,
    )

    assert success is True, message
    assert details is not None
    assert details["imported_items"]["explainer_session"] == 1
    restored_repo = ExplainerRepository(
        ExplainerDatabase(DatabasePaths.get_explainer_db_path(8), client_id="user-8")
    )
    [restored] = restored_repo.list_sessions(owner_user_id="8")
    assert restored.title == "Learn attention"
