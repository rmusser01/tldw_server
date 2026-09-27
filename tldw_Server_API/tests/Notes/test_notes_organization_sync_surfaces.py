"""Regression coverage for non-Notes API organization mutation surfaces."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.endpoints import chat as chat_endpoint
from tldw_Server_API.app.api.v1.endpoints import notes_tasks as notes_tasks_endpoint
from tldw_Server_API.app.api.v1.endpoints.notes_sync_errors import notes_sync_http_error
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user
from tldw_Server_API.app.core.Chat.conversation_enrichment import auto_tag_conversation
from tldw_Server_API.app.core.Chatbooks.chatbook_models import (
    ChatbookManifest,
    ChatbookVersion,
    ConflictResolution,
    ImportJob,
    ImportStatus,
)
from tldw_Server_API.app.core.Chatbooks.chatbook_service import ChatbookService
from tldw_Server_API.app.core.Chatbooks.openwebui_folders import mirror_openwebui_folder_for_conversation
from tldw_Server_API.app.core.MCP_unified.modules.base import ModuleConfig
from tldw_Server_API.app.core.MCP_unified.modules.implementations.notes_module import NotesModule
from tldw_Server_API.app.core.Notes.Notes_Library import NotesInteropService
from tldw_Server_API.app.core.Notes.organization_capture import capture_note_upsert, capture_plan
from tldw_Server_API.app.core.Notes.studio_service import NotesStudioService
from tldw_Server_API.app.core.Notes_Tasks.models import TaskActor
from tldw_Server_API.app.core.Notes_Tasks.service import NotesTaskService
from tldw_Server_API.app.core.Sync.v2 import server_origin
from tldw_Server_API.app.core.Sync.v2.errors import SyncStoreError
from tldw_Server_API.app.core.Sync.v2.notes_organization_coordinator import (
    NotesOrganizationCoordinator,
    NotesOrganizationDomainsIncompleteError,
    NotesOrganizationNotReadyError,
    NotesOrganizationVersionConflictError,
)
from tldw_Server_API.app.core.Sync.v2.server_origin import capture_server_origin_mutation
from tldw_Server_API.app.core.Sync.v2.server_origin_batch import (
    SyncServerOriginBatchAppendError,
)
from tldw_Server_API.app.core.WebClipper.schemas import WebClipperSaveRequest
from tldw_Server_API.app.core.WebClipper.service import WebClipperService
from tldw_Server_API.tests.Sync.notes_organization_test_support import build_ready_notes_sync_stack

pytestmark = pytest.mark.unit


def _patch_active_service(monkeypatch: pytest.MonkeyPatch, service) -> None:
    from tldw_Server_API.app.core.Sync.v2 import server_origin

    monkeypatch.setattr(
        server_origin,
        "get_active_server_origin_sync_service_for_user",
        lambda _user_id: service,
    )


def _domains(sync_store) -> list[str]:
    dataset_id = sync_store.list_datasets_for_user("user-1")[0].dataset_id
    return [
        envelope.domain
        for envelope in sync_store.list_envelopes_after(dataset_id, 0, limit=100)
        if envelope.domain.startswith("notes.") and envelope.routing_metadata.get("source") != "test-prerequisite"
    ]


def _conversation(db, service, *, title: str = "Conversation") -> str:
    conversation_id = str(uuid4())
    capture_server_origin_mutation(
        service,
        user_id="user-1",
        domain="chat.conversation",
        operation="upsert",
        object_id=conversation_id,
        payload={"title": title, "client_id": "user-1", "scope_type": "global"},
        source="test-prerequisite",
    )
    assert db.get_conversation_by_id(conversation_id) is not None
    return conversation_id


def _make_notes_sync_not_ready(service, monkeypatch, state: str) -> None:
    dataset = service.store.list_datasets_for_user("user-1")[0]
    if state == "partial":
        not_ready = replace(
            dataset,
            domains=[domain for domain in dataset.domains if domain != "notes.folder_link"],
        )
    else:
        metadata = dict(dataset.metadata)
        organization = dict(metadata["notes_organization_v1"])
        organization["state"] = state
        organization["error_code"] = "safe_repair_code" if state == "failed" else None
        metadata["notes_organization_v1"] = organization
        not_ready = replace(dataset, metadata=metadata)
    monkeypatch.setattr(service.store, "list_datasets_for_user", lambda _user_id: [not_ready])


def test_conversation_keyword_replace_uses_active_sync_authority(tmp_path, monkeypatch) -> None:
    db, sync_store, service = build_ready_notes_sync_stack(tmp_path)
    _patch_active_service(monkeypatch, service)
    conversation_id = _conversation(db, service)

    chat_endpoint._replace_conversation_keywords(
        db,
        conversation_id,
        ["alpha", "beta"],
        owner_user_id="user-1",
    )

    assert _domains(sync_store) == [
        "notes.keyword",
        "notes.keyword",
        "notes.keyword_link",
        "notes.keyword_link",
    ]
    assert sorted(row["keyword"] for row in db.get_keywords_for_conversation(conversation_id)) == [
        "alpha",
        "beta",
    ]


def test_conversation_keyword_replace_exact_retry_is_stable(tmp_path, monkeypatch) -> None:
    db, sync_store, service = build_ready_notes_sync_stack(tmp_path)
    _patch_active_service(monkeypatch, service)
    conversation_id = _conversation(db, service)

    chat_endpoint._replace_conversation_keywords(db, conversation_id, ["alpha"], owner_user_id="user-1")
    first = _domains(sync_store)
    chat_endpoint._replace_conversation_keywords(db, conversation_id, ["alpha"], owner_user_id="user-1")

    assert _domains(sync_store) == first == ["notes.keyword", "notes.keyword_link"]


@pytest.mark.parametrize(
    ("state", "error_type"),
    [
        ("partial", NotesOrganizationDomainsIncompleteError),
        ("initializing", NotesOrganizationNotReadyError),
        ("failed", NotesOrganizationNotReadyError),
    ],
)
def test_conversation_keyword_replace_active_not_ready_fails_closed(
    tmp_path,
    monkeypatch,
    state,
    error_type,
) -> None:
    db, sync_store, service = build_ready_notes_sync_stack(tmp_path)
    _patch_active_service(monkeypatch, service)
    conversation_id = _conversation(db, service)
    _make_notes_sync_not_ready(service, monkeypatch, state)

    with pytest.raises(error_type):
        chat_endpoint._replace_conversation_keywords(
            db,
            conversation_id,
            ["must-not-write"],
            owner_user_id="user-1",
        )

    assert db.get_keyword_by_text("must-not-write") is None
    assert _domains(sync_store) == []


def test_conversation_keyword_replace_inactive_keeps_legacy_path(tmp_path, monkeypatch) -> None:
    db, sync_store, service = build_ready_notes_sync_stack(tmp_path)
    _patch_active_service(monkeypatch, None)
    conversation_id = _conversation(db, service)

    chat_endpoint._replace_conversation_keywords(db, conversation_id, ["legacy"], owner_user_id="user-1")

    assert [row["keyword"] for row in db.get_keywords_for_conversation(conversation_id)] == ["legacy"]
    assert _domains(sync_store) == []


@pytest.mark.parametrize(
    ("state", "error_code"),
    [
        ("partial", "notes_organization_sync_domains_incomplete"),
        ("initializing", "notes_organization_sync_not_ready"),
        ("failed", "notes_organization_sync_not_ready"),
    ],
)
def test_conversation_patch_preflights_sync_before_product_update(
    tmp_path,
    monkeypatch,
    state,
    error_code,
) -> None:
    db, sync_store, service = build_ready_notes_sync_stack(tmp_path)
    _patch_active_service(monkeypatch, service)
    conversation_id = _conversation(db, service)
    before = db.get_conversation_by_id(conversation_id)
    _make_notes_sync_not_ready(service, monkeypatch, state)
    app = FastAPI()
    app.include_router(chat_endpoint.router, prefix="/api/v1/chat")
    app.dependency_overrides[get_chacha_db_for_user] = lambda: db
    app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id="user-1")

    response = TestClient(app).patch(
        f"/api/v1/chat/conversations/{conversation_id}",
        json={"version": int(before["version"]), "state": "backlog", "keywords": ["blocked"]},
    )

    assert response.status_code == 409, response.text
    assert response.json()["detail"]["error_code"] == error_code
    after = db.get_conversation_by_id(conversation_id)
    assert after["state"] == before["state"]
    assert after["version"] == before["version"]
    assert db.get_keyword_by_text("blocked") is None
    assert _domains(sync_store) == []


@pytest.mark.parametrize("state", ["partial", "initializing", "failed"])
def test_auto_tag_preflights_sync_before_conversation_update(
    tmp_path,
    monkeypatch,
    state,
) -> None:
    db, sync_store, service = build_ready_notes_sync_stack(tmp_path)
    _patch_active_service(monkeypatch, service)
    conversation_id = _conversation(db, service, title="Billing")
    for content in ("invoice payment", "invoice charge", "payment receipt"):
        db.add_message({"conversation_id": conversation_id, "sender": "user", "content": content})
    before = db.get_conversation_by_id(conversation_id)
    _make_notes_sync_not_ready(service, monkeypatch, state)

    with pytest.raises((NotesOrganizationDomainsIncompleteError, NotesOrganizationNotReadyError)):
        auto_tag_conversation(
            db,
            conversation_id,
            owner_user_id="user-1",
            trigger_clustering=False,
        )

    after = db.get_conversation_by_id(conversation_id)
    assert after["topic_label"] == before["topic_label"]
    assert after["version"] == before["version"]
    assert db.get_keywords_for_conversation(conversation_id) == []
    assert _domains(sync_store) == []


def test_conversation_patch_maps_sync_append_failure_to_503(tmp_path, monkeypatch) -> None:
    db, _sync_store, service = build_ready_notes_sync_stack(tmp_path)
    _patch_active_service(monkeypatch, service)
    conversation_id = _conversation(db, service)
    before = db.get_conversation_by_id(conversation_id)

    def fail_append(_envelopes):
        raise SyncServerOriginBatchAppendError("injected")

    monkeypatch.setattr(service.store, "insert_envelopes_atomic", fail_append)
    app = FastAPI()
    app.include_router(chat_endpoint.router, prefix="/api/v1/chat")
    app.dependency_overrides[get_chacha_db_for_user] = lambda: db
    app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id="user-1")

    response = TestClient(app).patch(
        f"/api/v1/chat/conversations/{conversation_id}",
        json={"version": int(before["version"]), "keywords": ["blocked"]},
    )

    assert response.status_code == 503, response.text
    assert response.json()["detail"]["error_code"] == "sync_server_origin_batch_append_failed"


def test_knowledge_save_captures_note_and_tags_before_product_projection(tmp_path, monkeypatch) -> None:
    db, sync_store, service = build_ready_notes_sync_stack(tmp_path)
    _patch_active_service(monkeypatch, service)
    conversation_id = _conversation(db, service, title="Knowledge")
    app = FastAPI()
    app.include_router(chat_endpoint.router, prefix="/api/v1/chat")
    app.dependency_overrides[get_chacha_db_for_user] = lambda: db
    app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id="user-1")
    client = TestClient(app)

    response = client.post(
        "/api/v1/chat/knowledge/save",
        json={
            "conversation_id": conversation_id,
            "snippet": "Captured evidence",
            "tags": ["alpha", "beta"],
            "make_flashcard": False,
        },
    )

    assert response.status_code == 201, response.text
    assert _domains(sync_store) == [
        "notes.note",
        "notes.keyword",
        "notes.keyword",
        "notes.keyword_link",
        "notes.keyword_link",
    ]
    note_id = str(response.json()["note_id"])
    assert sorted(row["keyword"] for row in db.get_keywords_for_note(note_id)) == ["alpha", "beta"]


def test_web_clipper_save_captures_note_keywords_and_folder(tmp_path, monkeypatch) -> None:
    db, sync_store, service = build_ready_notes_sync_stack(tmp_path)
    _patch_active_service(monkeypatch, service)
    coordinator = NotesOrganizationCoordinator(service=service, note_db=db, user_id="user-1")
    folder = capture_plan(
        coordinator,
        coordinator.plan_folder_path("Clips", idempotency_key="clips-folder"),
        source="test-prerequisite",
        key="clips-folder",
    )
    assert isinstance(folder, dict)
    clipper = WebClipperService(db=db, user_id="user-1")
    clip_id = str(uuid4())
    request = WebClipperSaveRequest(
        clip_id=clip_id,
        clip_type="article",
        source_url="https://example.com/story",
        source_title="Example Story",
        destination_mode="note",
        note=WebClipperSaveRequest.NotePayload(
            title="Example Story",
            keywords=["alpha"],
            folder_id=int(folder["id"]),
        ),
        content=WebClipperSaveRequest.ContentPayload(visible_body="Body", full_extract="Body"),
    )

    result = clipper.save_clip(request)

    assert result.status == "saved"
    assert _domains(sync_store) == [
        "notes.note",
        "notes.keyword",
        "notes.keyword_link",
        "notes.folder_link",
    ]
    assert [row["path"] for row in db.get_note_folders_for_note(result.note.id)] == ["Clips"]


@pytest.mark.asyncio
async def test_mcp_note_create_and_tag_set_use_active_sync_authority(tmp_path, monkeypatch) -> None:
    db, sync_store, service = build_ready_notes_sync_stack(tmp_path)
    _patch_active_service(monkeypatch, service)
    module = NotesModule(ModuleConfig(name="notes"))
    monkeypatch.setattr(module, "_open_db", lambda _context: db)
    monkeypatch.setattr(db, "close_all_connections", lambda: None)
    context = SimpleNamespace(metadata={"roles": []}, user_id="user-1")

    created = await module.execute_tool(
        "notes.create",
        {"title": "MCP note", "content": "Body", "tags": ["alpha"]},
        context=context,
    )
    await module.execute_tool(
        "notes.tags.set",
        {"note_id": created["note_id"], "tags": ["beta"]},
        context=context,
    )

    assert _domains(sync_store) == [
        "notes.note",
        "notes.keyword",
        "notes.keyword_link",
        "notes.keyword",
        "notes.keyword_link",
        "notes.keyword_link",
    ]
    assert [row["keyword"] for row in db.get_keywords_for_note(created["note_id"])] == ["beta"]


@pytest.mark.asyncio
async def test_mcp_note_write_without_authenticated_owner_fails_closed(tmp_path, monkeypatch) -> None:
    db, _sync_store, service = build_ready_notes_sync_stack(tmp_path)
    _patch_active_service(monkeypatch, service)
    module = NotesModule(ModuleConfig(name="notes"))
    monkeypatch.setattr(module, "_open_db", lambda _context: db)
    monkeypatch.setattr(db, "close_all_connections", lambda: None)
    context = SimpleNamespace(metadata={"roles": []}, user_id=None)

    with pytest.raises(SyncStoreError, match="authenticated owner"):
        await module.execute_tool(
            "notes.create",
            {"title": "Blocked", "content": "Body", "tags": ["alpha"]},
            context=context,
        )

    assert db.search_notes(search_term="Blocked") == []


def test_auto_tag_background_worker_uses_active_sync_authority(tmp_path, monkeypatch) -> None:
    db, sync_store, service = build_ready_notes_sync_stack(tmp_path)
    _patch_active_service(monkeypatch, service)
    conversation_id = _conversation(db, service, title="Billing")
    for content in ("invoice payment", "invoice charge", "payment receipt"):
        db.add_message({"conversation_id": conversation_id, "sender": "user", "content": content})

    result = auto_tag_conversation(
        db,
        conversation_id,
        owner_user_id="user-1",
        trigger_clustering=False,
    )

    assert result.updated is True
    assert "notes.keyword_link" in _domains(sync_store)


def test_openwebui_folder_import_uses_active_sync_authority(tmp_path, monkeypatch) -> None:
    db, sync_store, service = build_ready_notes_sync_stack(tmp_path)
    _patch_active_service(monkeypatch, service)
    conversation_id = _conversation(db, service, title="Imported")

    result = mirror_openwebui_folder_for_conversation(
        db,
        conversation_id=conversation_id,
        namespace_segments=["OpenWebUI", "Alice"],
        source_path_segments=["Research"],
        source_folder_id="folder-1",
        metadata={"source_user_id": "alice"},
        owner_user_id="user-1",
    )

    assert result.final_collection_id is not None
    assert _domains(sync_store) == [
        "notes.keyword_collection",
        "notes.keyword_collection",
        "notes.keyword_collection",
        "notes.keyword",
        "notes.keyword_collection_link",
        "notes.keyword_link",
    ]


def test_openwebui_folder_import_exact_retry_is_stable(tmp_path, monkeypatch) -> None:
    db, sync_store, service = build_ready_notes_sync_stack(tmp_path)
    _patch_active_service(monkeypatch, service)
    conversation_id = _conversation(db, service, title="Imported")
    arguments = {
        "conversation_id": conversation_id,
        "namespace_segments": ["OpenWebUI", "Alice"],
        "source_path_segments": ["Research"],
        "source_folder_id": "folder-1",
        "metadata": {"source_user_id": "alice"},
        "owner_user_id": "user-1",
    }

    first = mirror_openwebui_folder_for_conversation(db, **arguments)
    first_domains = _domains(sync_store)
    second = mirror_openwebui_folder_for_conversation(db, **arguments)

    assert second.collection_ids == first.collection_ids
    assert second.keyword_id == first.keyword_id
    assert _domains(sync_store) == first_domains


def test_openwebui_folder_replay_preserves_original_response_after_collection_rename(
    tmp_path,
    monkeypatch,
) -> None:
    db, _sync_store, service = build_ready_notes_sync_stack(tmp_path)
    _patch_active_service(monkeypatch, service)
    coordinator = NotesOrganizationCoordinator(service=service, note_db=db, user_id="user-1")
    root = capture_plan(
        coordinator,
        coordinator.plan_collection_change(None, "OpenWebUI", None, idempotency_key="owui-root"),
        source="test-prerequisite",
        key="owui-root",
    )
    conversation_id = _conversation(db, service, title="Imported")
    arguments = {
        "conversation_id": conversation_id,
        "namespace_segments": ["OpenWebUI", "Alice"],
        "source_path_segments": ["Research"],
        "source_folder_id": "folder-1",
        "metadata": {"source_user_id": "alice"},
        "owner_user_id": "user-1",
    }

    first = mirror_openwebui_folder_for_conversation(db, **arguments)
    capture_plan(
        coordinator,
        coordinator.plan_collection_change(
            int(root["id"]),
            "Renamed OpenWebUI",
            None,
            expected_version=int(root["version"]),
        ),
        source="test-prerequisite",
        key="rename-owui-root",
    )
    second = mirror_openwebui_folder_for_conversation(db, **arguments)

    assert second == first
    assert first.created_collections == 2
    assert first.reused_collections == 1


def test_openwebui_folder_import_append_failure_writes_no_product_state(
    tmp_path,
    monkeypatch,
) -> None:
    db, sync_store, service = build_ready_notes_sync_stack(tmp_path)
    _patch_active_service(monkeypatch, service)
    conversation_id = _conversation(db, service, title="Imported")

    def fail_append(_envelopes):
        raise SyncServerOriginBatchAppendError("injected")

    monkeypatch.setattr(service.store, "insert_envelopes_atomic", fail_append)

    with pytest.raises(SyncServerOriginBatchAppendError):
        mirror_openwebui_folder_for_conversation(
            db,
            conversation_id=conversation_id,
            namespace_segments=["OpenWebUI", "Alice"],
            source_path_segments=["Research"],
            source_folder_id="folder-1",
            metadata={"source_user_id": "alice"},
            owner_user_id="user-1",
        )

    assert _domains(sync_store) == []
    assert db.get_keyword_collection_by_name("OpenWebUI") is None
    assert db.get_keyword_collection_by_name("Alice") is None
    assert db.get_keyword_collection_by_name("Research") is None
    assert db.get_keywords_for_conversation(conversation_id) == []


def test_notes_interop_workflow_path_uses_active_sync_authority(tmp_path, monkeypatch) -> None:
    db, sync_store, service = build_ready_notes_sync_stack(tmp_path)
    _patch_active_service(monkeypatch, service)
    interop = NotesInteropService(tmp_path / "interop", "workflow-engine")
    monkeypatch.setattr(interop, "_get_db", lambda _user_id: db)

    note_id = interop.add_note("user-1", "Workflow note", "Body")

    assert db.get_note_by_id(note_id) is not None
    assert _domains(sync_store) == ["notes.note"]


def test_notes_interop_delete_note_uses_active_sync_authority(tmp_path, monkeypatch) -> None:
    db, sync_store, service = build_ready_notes_sync_stack(tmp_path)
    _patch_active_service(monkeypatch, service)
    coordinator = NotesOrganizationCoordinator(service=service, note_db=db, user_id="user-1")
    note_id = str(uuid4())
    capture_note_upsert(
        coordinator,
        note_id=note_id,
        title="Delete me",
        content="Body",
        source="test-prerequisite",
    )
    interop = NotesInteropService(tmp_path / "interop-delete-note", "workflow-engine")
    interop._db_instances["user-1"] = db

    assert interop.delete_note(note_id=note_id, user_id="user-1") is True
    first_domains = _domains(sync_store)
    assert interop.delete_note(note_id=note_id, user_id="user-1") is True

    assert db.get_note_by_id(note_id) is None
    assert _domains(sync_store) == first_domains == ["notes.note"]


def test_notes_interop_delete_note_keeps_inactive_legacy_path(tmp_path, monkeypatch) -> None:
    db, sync_store, _service = build_ready_notes_sync_stack(tmp_path)
    monkeypatch.setattr(
        server_origin,
        "get_active_server_origin_sync_service_for_user",
        lambda _user_id: None,
    )
    note_id = db.add_note(title="Delete me", content="Body")
    assert note_id is not None
    interop = NotesInteropService(tmp_path / "interop-delete-note-inactive", "workflow-engine")
    interop._db_instances["user-1"] = db

    assert interop.delete_note(note_id=str(note_id), user_id="user-1") is True

    assert db.get_note_by_id(str(note_id)) is None
    assert _domains(sync_store) == []


@pytest.mark.parametrize("state", ["partial", "initializing", "failed"])
def test_notes_interop_delete_note_fails_closed_when_active_not_ready(
    tmp_path,
    monkeypatch,
    state,
) -> None:
    db, sync_store, service = build_ready_notes_sync_stack(tmp_path)
    _patch_active_service(monkeypatch, service)
    coordinator = NotesOrganizationCoordinator(service=service, note_db=db, user_id="user-1")
    note_id = str(uuid4())
    note = capture_note_upsert(
        coordinator,
        note_id=note_id,
        title="Keep me",
        content="Body",
        source="test-prerequisite",
    )
    _make_notes_sync_not_ready(service, monkeypatch, state)
    interop = NotesInteropService(tmp_path / f"interop-delete-note-{state}", "workflow-engine")
    interop._db_instances["user-1"] = db

    with pytest.raises((NotesOrganizationDomainsIncompleteError, NotesOrganizationNotReadyError)):
        interop.delete_note(note_id=note_id, user_id="user-1")

    current = db.get_note_by_id(note_id)
    assert current is not None
    assert current["version"] == note["version"]
    assert _domains(sync_store) == []


def test_active_coordinator_requires_explicit_owner(tmp_path) -> None:
    db, _sync_store, _service = build_ready_notes_sync_stack(tmp_path)

    with pytest.raises(SyncStoreError, match="authenticated owner"):
        chat_endpoint._replace_conversation_keywords(
            db,
            str(uuid4()),
            ["alpha"],
            owner_user_id=None,
        )


def test_active_service_lookup_propagates_store_failure(monkeypatch) -> None:
    from tldw_Server_API.app.core.Sync.v2 import factory

    failing_service = SimpleNamespace(
        store=SimpleNamespace(
            list_datasets_for_user=lambda _user_id: (_ for _ in ()).throw(SyncStoreError("lookup failed"))
        )
    )
    monkeypatch.setattr(factory, "sync_v2_storage_exists_for_user", lambda _user_id: True)
    monkeypatch.setattr(factory, "sync_v2_service_for_user", lambda _user_id: failing_service)

    with pytest.raises(SyncStoreError, match="lookup failed"):
        server_origin.get_active_server_origin_sync_service_for_user("user-1")


@pytest.mark.parametrize("storage_exists", [False, True])
def test_active_service_lookup_returns_none_only_for_authoritative_inactive_state(
    monkeypatch,
    storage_exists,
) -> None:
    from tldw_Server_API.app.core.Sync.v2 import factory

    monkeypatch.setattr(factory, "sync_v2_storage_exists_for_user", lambda _user_id: storage_exists)
    monkeypatch.setattr(
        factory,
        "sync_v2_service_for_user",
        lambda _user_id: SimpleNamespace(store=SimpleNamespace(list_datasets_for_user=lambda _owner: [])),
    )

    assert server_origin.get_active_server_origin_sync_service_for_user("user-1") is None


@pytest.mark.parametrize(
    ("exc", "expected_status", "error_code"),
    [
        (
            NotesOrganizationNotReadyError(state="initializing"),
            409,
            "notes_organization_sync_not_ready",
        ),
        (
            SyncServerOriginBatchAppendError("group-1"),
            503,
            "sync_server_origin_batch_append_failed",
        ),
        (
            NotesOrganizationVersionConflictError(),
            409,
            "notes_organization_version_conflict",
        ),
    ],
)
def test_tasks_and_studio_share_exact_safe_sync_http_mapping(
    exc,
    expected_status,
    error_code,
) -> None:
    mapped = notes_sync_http_error(exc)
    assert mapped.status_code == expected_status
    assert mapped.detail["error_code"] == error_code

    with pytest.raises(HTTPException) as task_error:
        notes_tasks_endpoint._handle_task_error(exc)
    assert task_error.value.status_code == expected_status
    assert task_error.value.detail["error_code"] == error_code


@pytest.mark.parametrize("method_name", ["create_keyword", "add_keyword"])
def test_notes_interop_keyword_create_wrappers_use_active_sync(
    tmp_path,
    monkeypatch,
    method_name,
) -> None:
    db, sync_store, service = build_ready_notes_sync_stack(tmp_path)
    _patch_active_service(monkeypatch, service)
    interop = NotesInteropService(tmp_path / "interop-keyword", "workflow-engine")
    monkeypatch.setattr(interop, "_get_db", lambda _user_id: db)

    if method_name == "create_keyword":
        keyword_id = interop.create_keyword(keyword="alpha", user_id="user-1")
    else:
        keyword_id = interop.add_keyword("user-1", "alpha")

    assert keyword_id is not None
    assert _domains(sync_store) == ["notes.keyword"]


@pytest.mark.parametrize(
    ("link_method", "unlink_method"),
    [
        ("link_note_keyword", "unlink_note_keyword"),
        ("link_note_to_keyword", "unlink_note_from_keyword"),
    ],
)
def test_notes_interop_keyword_link_wrappers_use_active_sync(
    tmp_path,
    monkeypatch,
    link_method,
    unlink_method,
) -> None:
    db, sync_store, service = build_ready_notes_sync_stack(tmp_path)
    _patch_active_service(monkeypatch, service)
    coordinator = NotesOrganizationCoordinator(service=service, note_db=db, user_id="user-1")
    note_id = str(uuid4())
    capture_note_upsert(
        coordinator,
        note_id=note_id,
        title="Linked",
        content="Body",
        source="test-prerequisite",
    )
    keyword = capture_plan(
        coordinator,
        coordinator.plan_keyword_create("alpha", idempotency_key="interop-keyword"),
        source="test-prerequisite",
        key="interop-keyword",
    )
    interop = NotesInteropService(tmp_path / "interop-link", "workflow-engine")
    monkeypatch.setattr(interop, "_get_db", lambda _user_id: db)
    keyword_id = int(keyword["id"])

    if link_method == "link_note_keyword":
        assert interop.link_note_keyword(note_id=note_id, keyword_id=keyword_id, user_id="user-1")
        assert interop.unlink_note_keyword(note_id=note_id, keyword_id=keyword_id, user_id="user-1")
    else:
        assert interop.link_note_to_keyword("user-1", note_id, keyword_id)
        assert interop.unlink_note_from_keyword("user-1", note_id, keyword_id)

    assert _domains(sync_store) == ["notes.keyword_link", "notes.keyword_link"]


@pytest.mark.parametrize("method_name", ["delete_keyword", "soft_delete_keyword"])
def test_notes_interop_keyword_delete_uses_active_sync(tmp_path, monkeypatch, method_name) -> None:
    db, sync_store, service = build_ready_notes_sync_stack(tmp_path)
    _patch_active_service(monkeypatch, service)
    coordinator = NotesOrganizationCoordinator(service=service, note_db=db, user_id="user-1")
    keyword = capture_plan(
        coordinator,
        coordinator.plan_keyword_create("alpha", idempotency_key="delete-keyword"),
        source="test-prerequisite",
        key="delete-keyword",
    )
    interop = NotesInteropService(tmp_path / "interop-delete", "workflow-engine")
    monkeypatch.setattr(interop, "_get_db", lambda _user_id: db)

    if method_name == "delete_keyword":
        assert interop.delete_keyword(keyword_id=int(keyword["id"]), user_id="user-1")
    else:
        assert interop.soft_delete_keyword("user-1", int(keyword["id"]), int(keyword["version"]))

    assert db.get_keyword_by_text("alpha") is None
    assert _domains(sync_store) == ["notes.keyword"]


@pytest.mark.parametrize("family", ["create", "link", "delete"])
def test_notes_interop_keyword_families_keep_inactive_legacy_path(
    tmp_path,
    monkeypatch,
    family,
) -> None:
    db, sync_store, service = build_ready_notes_sync_stack(tmp_path)
    _patch_active_service(monkeypatch, None)
    interop = NotesInteropService(tmp_path / f"interop-inactive-{family}", "workflow-engine")
    monkeypatch.setattr(interop, "_get_db", lambda _user_id: db)

    if family == "create":
        assert interop.add_keyword("user-1", "legacy") is not None
    elif family == "link":
        note_id = db.add_note(title="Legacy", content="Body")
        keyword_id = db.add_keyword("legacy")
        assert interop.link_note_to_keyword("user-1", str(note_id), int(keyword_id))
    else:
        keyword_id = db.add_keyword("legacy")
        keyword = db.get_keyword_by_id(int(keyword_id))
        assert interop.soft_delete_keyword("user-1", int(keyword_id), int(keyword["version"]))

    assert _domains(sync_store) == []


@pytest.mark.parametrize("family", ["create", "link", "delete"])
@pytest.mark.parametrize("state", ["partial", "initializing", "failed"])
def test_notes_interop_keyword_families_fail_closed_when_active_not_ready(
    tmp_path,
    monkeypatch,
    family,
    state,
) -> None:
    db, sync_store, service = build_ready_notes_sync_stack(tmp_path)
    _patch_active_service(monkeypatch, service)
    coordinator = NotesOrganizationCoordinator(service=service, note_db=db, user_id="user-1")
    note_id = str(uuid4())
    capture_note_upsert(
        coordinator,
        note_id=note_id,
        title="Linked",
        content="Body",
        source="test-prerequisite",
    )
    keyword = capture_plan(
        coordinator,
        coordinator.plan_keyword_create("existing", idempotency_key=f"not-ready-{family}"),
        source="test-prerequisite",
        key=f"not-ready-{family}",
    )
    _make_notes_sync_not_ready(service, monkeypatch, state)
    interop = NotesInteropService(tmp_path / f"interop-not-ready-{family}", "workflow-engine")
    monkeypatch.setattr(interop, "_get_db", lambda _user_id: db)

    with pytest.raises((NotesOrganizationDomainsIncompleteError, NotesOrganizationNotReadyError)):
        if family == "create":
            interop.add_keyword("user-1", "blocked")
        elif family == "link":
            interop.link_note_to_keyword("user-1", note_id, int(keyword["id"]))
        else:
            interop.soft_delete_keyword("user-1", int(keyword["id"]), int(keyword["version"]))

    assert db.get_keyword_by_text("blocked") is None
    assert db.get_keywords_for_note(note_id) == []
    assert db.get_keyword_by_text("existing") is not None
    assert _domains(sync_store) == []


@pytest.mark.asyncio
async def test_notes_studio_derive_uses_active_sync_authority(tmp_path, monkeypatch) -> None:
    db, sync_store, service = build_ready_notes_sync_stack(tmp_path)
    _patch_active_service(monkeypatch, service)
    coordinator = NotesOrganizationCoordinator(service=service, note_db=db, user_id="user-1")
    source_id = str(uuid4())
    capture_note_upsert(
        coordinator,
        note_id=source_id,
        title="Source",
        content="Cells need energy to function.",
        source="test-prerequisite",
    )

    async def generate(request, _context):
        excerpt = str(request["excerpt_text"])
        return {
            "payload": {
                "meta": {"title": "Study Notes", "source_note_id": source_id},
                "sections": [{"id": "notes-1", "kind": "notes", "title": "Notes", "content": excerpt}],
            }
        }

    result = await NotesStudioService(
        db=db,
        user_id="user-1",
        generation_adapter=generate,
    ).derive_from_excerpt(
        source_note_id=source_id,
        excerpt_text="Cells need energy",
        template_type="lined",
        handwriting_mode="off",
    )

    assert result["note"]["title"] == "Study Notes"
    assert _domains(sync_store) == ["notes.note"]


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["partial", "initializing", "failed"])
async def test_notes_studio_preflights_readiness_before_generation(
    tmp_path,
    monkeypatch,
    state,
) -> None:
    db, _sync_store, service = build_ready_notes_sync_stack(tmp_path)
    _patch_active_service(monkeypatch, service)
    coordinator = NotesOrganizationCoordinator(service=service, note_db=db, user_id="user-1")
    source_id = str(uuid4())
    capture_note_upsert(
        coordinator,
        note_id=source_id,
        title="Source",
        content="Cells need energy to function.",
        source="test-prerequisite",
    )
    _make_notes_sync_not_ready(service, monkeypatch, state)
    generation_calls = 0

    async def generate(_request, _context):
        nonlocal generation_calls
        generation_calls += 1
        return {"payload": {}}

    with pytest.raises((NotesOrganizationDomainsIncompleteError, NotesOrganizationNotReadyError)):
        await NotesStudioService(
            db=db,
            user_id="user-1",
            generation_adapter=generate,
        ).derive_from_excerpt(
            source_note_id=source_id,
            excerpt_text="Cells need energy",
            template_type="lined",
            handwriting_mode="off",
        )

    assert generation_calls == 0


@pytest.mark.asyncio
async def test_notes_studio_derive_exact_retry_skips_generation_and_sidecar_insert(
    tmp_path,
    monkeypatch,
) -> None:
    db, _sync_store, service = build_ready_notes_sync_stack(tmp_path)
    _patch_active_service(monkeypatch, service)
    coordinator = NotesOrganizationCoordinator(service=service, note_db=db, user_id="user-1")
    source_id = str(uuid4())
    capture_note_upsert(
        coordinator,
        note_id=source_id,
        title="Source",
        content="Cells need energy to function.",
        source="test-prerequisite",
    )
    generation_calls = 0

    async def generate(request, _context):
        nonlocal generation_calls
        generation_calls += 1
        content = str(request["excerpt_text"]) if generation_calls == 1 else "Different retry output"
        return {
            "payload": {
                "meta": {"title": "Study Notes", "source_note_id": source_id},
                "sections": [
                    {
                        "id": "notes-1",
                        "kind": "notes",
                        "title": "Notes",
                        "content": content,
                    }
                ],
            }
        }

    studio = NotesStudioService(db=db, user_id="user-1", generation_adapter=generate)
    arguments = {
        "source_note_id": source_id,
        "excerpt_text": "Cells need energy",
        "template_type": "lined",
        "handwriting_mode": "off",
    }
    first = await studio.derive_from_excerpt(**arguments)
    second = await studio.derive_from_excerpt(**arguments)

    assert second == first
    assert generation_calls == 1


@pytest.mark.asyncio
async def test_notes_studio_derive_retry_repairs_missing_sidecar_after_capture(
    tmp_path,
    monkeypatch,
) -> None:
    db, sync_store, service = build_ready_notes_sync_stack(tmp_path)
    _patch_active_service(monkeypatch, service)
    coordinator = NotesOrganizationCoordinator(service=service, note_db=db, user_id="user-1")
    source_id = str(uuid4())
    capture_note_upsert(
        coordinator,
        note_id=source_id,
        title="Source",
        content="Cells need energy to function.",
        source="test-prerequisite",
    )
    generation_calls = 0

    async def generate(request, _context):
        nonlocal generation_calls
        generation_calls += 1
        content = str(request["excerpt_text"]) if generation_calls == 1 else "Different retry output"
        return {
            "payload": {
                "meta": {"title": "Study Notes", "source_note_id": source_id},
                "sections": [
                    {
                        "id": "notes-1",
                        "kind": "notes",
                        "title": "Notes",
                        "content": content,
                    }
                ],
            }
        }

    original_create = db.create_note_studio_document
    create_calls = 0

    def fail_first_sidecar_create(**fields):
        nonlocal create_calls
        create_calls += 1
        if create_calls == 1:
            raise RuntimeError("simulated sidecar write failure")
        return original_create(**fields)

    monkeypatch.setattr(db, "create_note_studio_document", fail_first_sidecar_create)
    studio = NotesStudioService(db=db, user_id="user-1", generation_adapter=generate)
    arguments = {
        "source_note_id": source_id,
        "excerpt_text": "Cells need energy",
        "template_type": "lined",
        "handwriting_mode": "off",
    }

    with pytest.raises(RuntimeError, match="sidecar write failure"):
        await studio.derive_from_excerpt(**arguments)
    first_domains = _domains(sync_store)

    repaired = await studio.derive_from_excerpt(**arguments)

    assert repaired["studio_document"]["note_id"] == repaired["note"]["id"]
    assert repaired["is_stale"] is False
    assert _domains(sync_store) == first_domains == ["notes.note"]
    assert generation_calls == 1


@pytest.mark.asyncio
async def test_notes_studio_regeneration_replays_manifest_before_version_check(
    tmp_path,
    monkeypatch,
) -> None:
    db, _sync_store, service = build_ready_notes_sync_stack(tmp_path)
    _patch_active_service(monkeypatch, service)
    coordinator = NotesOrganizationCoordinator(service=service, note_db=db, user_id="user-1")
    source_id = str(uuid4())
    capture_note_upsert(
        coordinator,
        note_id=source_id,
        title="Source",
        content="Cells need energy to function.",
        source="test-prerequisite",
    )

    async def generate(_request, _context):
        return {
            "payload": {
                "meta": {"title": "Study Notes", "source_note_id": source_id},
                "sections": [{"id": "n1", "kind": "notes", "title": "Notes", "content": "Cells"}],
            }
        }

    studio = NotesStudioService(db=db, user_id="user-1", generation_adapter=generate)
    derived = await studio.derive_from_excerpt(
        source_note_id=source_id,
        excerpt_text="Cells need energy",
        template_type="lined",
        handwriting_mode="off",
    )
    expected_version = int(derived["note"]["version"])
    current_markdown = str(derived["note"]["content"]) + "\nExtra"
    first = await studio.regenerate_note_markdown(
        note_id=str(derived["note"]["id"]),
        expected_version=expected_version,
        current_markdown=current_markdown,
    )
    second = await studio.regenerate_note_markdown(
        note_id=str(derived["note"]["id"]),
        expected_version=expected_version,
        current_markdown=current_markdown,
    )

    assert second == first


@pytest.mark.asyncio
async def test_notes_studio_regeneration_retry_repairs_stale_sidecar_after_capture(
    tmp_path,
    monkeypatch,
) -> None:
    db, sync_store, service = build_ready_notes_sync_stack(tmp_path)
    _patch_active_service(monkeypatch, service)
    coordinator = NotesOrganizationCoordinator(service=service, note_db=db, user_id="user-1")
    source_id = str(uuid4())
    capture_note_upsert(
        coordinator,
        note_id=source_id,
        title="Source",
        content="Cells need energy to function.",
        source="test-prerequisite",
    )

    async def generate(_request, _context):
        return {
            "payload": {
                "meta": {"title": "Study Notes", "source_note_id": source_id},
                "sections": [{"id": "n1", "kind": "notes", "title": "Notes", "content": "Cells"}],
            }
        }

    studio = NotesStudioService(db=db, user_id="user-1", generation_adapter=generate)
    derived = await studio.derive_from_excerpt(
        source_note_id=source_id,
        excerpt_text="Cells need energy",
        template_type="lined",
        handwriting_mode="off",
    )
    original_upsert = db.upsert_note_studio_document
    upsert_calls = 0

    def fail_first_sidecar_upsert(**fields):
        nonlocal upsert_calls
        upsert_calls += 1
        if upsert_calls == 1:
            raise RuntimeError("simulated sidecar write failure")
        return original_upsert(**fields)

    monkeypatch.setattr(db, "upsert_note_studio_document", fail_first_sidecar_upsert)
    arguments = {
        "note_id": str(derived["note"]["id"]),
        "expected_version": int(derived["note"]["version"]),
        "current_markdown": str(derived["note"]["content"]) + "\nExtra",
    }

    with pytest.raises(RuntimeError, match="sidecar write failure"):
        await studio.regenerate_note_markdown(**arguments)
    first_domains = _domains(sync_store)

    repaired = await studio.regenerate_note_markdown(**arguments)

    assert repaired["is_stale"] is False
    assert _domains(sync_store) == first_domains == ["notes.note", "notes.note"]
    assert upsert_calls == 2


def test_notes_tasks_projection_uses_active_sync_authority(tmp_path, monkeypatch) -> None:
    db, sync_store, service = build_ready_notes_sync_stack(tmp_path)
    _patch_active_service(monkeypatch, service)
    coordinator = NotesOrganizationCoordinator(service=service, note_db=db, user_id="user-1")
    note_id = str(uuid4())
    note = capture_note_upsert(
        coordinator,
        note_id=note_id,
        title="Tasks",
        content="Intro\n",
        source="test-prerequisite",
    )

    task = NotesTaskService().create_task_for_note(
        db=db,
        note_id=note_id,
        text="Review",
        status="open",
        metadata={},
        expected_note_version=int(note["version"]),
        actor=TaskActor(actor_type="user", actor_id="user-1"),
    )

    assert task["text"] == "Review"
    assert _domains(sync_store) == ["notes.note"]


def test_chatbook_note_import_uses_active_sync_authority(tmp_path, monkeypatch) -> None:
    db, sync_store, service = build_ready_notes_sync_stack(tmp_path)
    _patch_active_service(monkeypatch, service)
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path))
    chatbooks = ChatbookService(user_id="user-1", db=db)
    extract_dir = tmp_path / "chatbook"
    notes_dir = extract_dir / "content" / "notes"
    notes_dir.mkdir(parents=True)
    (notes_dir / "note_1.md").write_text(
        "---\ntitle: Imported Note\n---\n\nBody",
        encoding="utf-8",
    )
    status = ImportJob(
        job_id="job-1",
        user_id="user-1",
        status=ImportStatus.IN_PROGRESS,
        chatbook_path="fixture",
    )

    chatbooks._import_notes(
        extract_dir,
        ChatbookManifest(
            version=ChatbookVersion.V1,
            name="Fixture",
            description="Fixture",
        ),
        ["1"],
        ConflictResolution.SKIP,
        prefix_imported=False,
        status=status,
    )

    assert status.successful_items == 1
    assert _domains(sync_store) == ["notes.note"]
