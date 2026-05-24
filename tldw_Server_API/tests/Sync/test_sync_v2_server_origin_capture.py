from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import User
from tldw_Server_API.app.api.v1.endpoints import character_chat_sessions as chat_sessions_endpoint
from tldw_Server_API.app.api.v1.endpoints import character_messages as messages_endpoint
from tldw_Server_API.app.api.v1.endpoints import notes as notes_endpoint
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
from tldw_Server_API.app.core.Sync.v2.adapters import StaticSyncAdapter, SyncAdapterRegistry
from tldw_Server_API.app.core.Sync.v2.errors import SyncStoreError
from tldw_Server_API.app.core.Sync.v2.materializers import (
    ChatConversationMaterializer,
    ChatMessageMaterializer,
    MaterializationResult,
    NotesMaterializer,
)
from tldw_Server_API.app.core.Sync.v2.models import M1_SYNC_DOMAINS, SyncEnvelope
from tldw_Server_API.app.core.Sync.v2.security import (
    server_trusted_encryption_status_from_config,
)
from tldw_Server_API.app.core.Sync.v2.server_origin import (
    CLIENT_PRIVATE_SERVER_FRONTEND_LIMITATION_CODE,
    SERVER_ORIGIN_DEVICE_ID,
    SyncServerOriginIdempotencyConflictError,
    SyncServerOriginMaterializationError,
    SyncServerOriginMutationNotSupportedError,
    capture_server_origin_mutation,
    server_origin_object_id,
)
from tldw_Server_API.app.core.Sync.v2.service import SyncV2Service, SyncV2Settings
from tldw_Server_API.app.core.Sync.v2.store import SyncV2Store


def _ready_encryption():
    return server_trusted_encryption_status_from_config(
        mode="managed_storage",
        server_trusted_enabled=True,
        auth_mode="multi_user",
    )


class _OrderAssertingNotesMaterializer(NotesMaterializer):
    def apply(self, envelope: SyncEnvelope, *, store: SyncV2Store) -> MaterializationResult:
        stored = store.list_envelopes_after(
            envelope.dataset_id,
            0,
            domains=[envelope.domain],
            limit=10,
        )
        assert stored[-1].client_envelope_id == envelope.client_envelope_id
        if envelope.base_server_cursor is None:
            assert self.note_db.get_note_by_id(envelope.object_id) is None
        return super().apply(envelope, store=store)


class _FailingApplyMaterializer:
    domain = "notes.note"

    def apply(self, envelope: SyncEnvelope, *, store: SyncV2Store) -> MaterializationResult:
        raise RuntimeError("raw content must not leak")


class _ConflictingApplyMaterializer:
    domain = "notes.note"

    def apply(self, envelope: SyncEnvelope, *, store: SyncV2Store) -> MaterializationResult:
        assert envelope.server_cursor is not None
        store.mark_envelope_apply_status(
            envelope.server_cursor,
            apply_status="conflict",
            apply_error_code="whole_object_conflict",
            apply_error_message="server-origin conflict after append",
        )
        return MaterializationResult(
            status="conflict",
            error_code="whole_object_conflict",
            message="server-origin conflict after append",
        )


class _FailingInsertStore(SyncV2Store):
    def insert_envelope(self, envelope):
        raise SyncStoreError("append is unavailable")


class _NoopRateLimiter:
    async def check_user_rate_limit(self, user_id: int, endpoint: str, role: str = "user"):
        return True, {}


class _NoopCharacterRateLimiter:
    async def check_rate_limit(self, user_id, operation):
        return None

    async def check_chat_limit(self, user_id, current_chat_count):
        return None

    async def check_message_send_rate(self, user_id):
        return None

    async def check_message_limit(self, chat_id, message_count):
        return None


def _notes_app(
    monkeypatch: pytest.MonkeyPatch,
    *,
    chacha_db: CharactersRAGDB,
    sync_service: SyncV2Service | None,
) -> TestClient:
    app = FastAPI()
    app.include_router(notes_endpoint.router, prefix="/api/v1/notes")

    async def _db_override():
        return chacha_db

    async def _user_override():
        return User(id="user-1", username="user-1", is_admin=True)

    app.dependency_overrides[notes_endpoint.get_chacha_db_for_user] = _db_override
    app.dependency_overrides[notes_endpoint.get_request_user] = _user_override
    app.dependency_overrides[notes_endpoint.get_rate_limiter_dep] = lambda: _NoopRateLimiter()
    monkeypatch.setattr(
        notes_endpoint,
        "get_active_server_origin_sync_service_for_user",
        lambda user_id: sync_service,
        raising=False,
    )
    return TestClient(app)


def _chat_messages_app(
    monkeypatch: pytest.MonkeyPatch,
    *,
    chacha_db: CharactersRAGDB,
    sync_service: SyncV2Service | None,
) -> TestClient:
    app = FastAPI()
    app.include_router(chat_sessions_endpoint.router, prefix="/api/v1/chats")
    app.include_router(messages_endpoint.router, prefix="/api/v1")

    async def _db_override():
        return chacha_db

    async def _user_override():
        return User(id="user-1", username="user-1", is_admin=True)

    app.dependency_overrides[chat_sessions_endpoint.get_chacha_db_for_user] = _db_override
    app.dependency_overrides[messages_endpoint.get_chacha_db_for_user] = _db_override
    app.dependency_overrides[chat_sessions_endpoint.get_request_user] = _user_override
    app.dependency_overrides[messages_endpoint.get_request_user] = _user_override
    monkeypatch.setattr(
        chat_sessions_endpoint,
        "get_active_server_origin_sync_service_for_user",
        lambda user_id: sync_service,
        raising=False,
    )
    monkeypatch.setattr(
        messages_endpoint,
        "get_active_server_origin_sync_service_for_user",
        lambda user_id: sync_service,
        raising=False,
    )
    monkeypatch.setattr(
        chat_sessions_endpoint,
        "get_character_rate_limiter",
        lambda: _NoopCharacterRateLimiter(),
    )
    monkeypatch.setattr(
        messages_endpoint,
        "get_character_rate_limiter",
        lambda: _NoopCharacterRateLimiter(),
    )
    return TestClient(app)


@pytest.fixture()
def chacha_db(tmp_path: Path) -> CharactersRAGDB:
    return CharactersRAGDB(
        db_path=str(tmp_path / "ChaChaNotes.db"),
        client_id="user-1",
    )


@pytest.fixture()
def sync_service(tmp_path: Path, chacha_db: CharactersRAGDB) -> SyncV2Service:
    registry = SyncAdapterRegistry(
        [StaticSyncAdapter(domain=domain, supported_adapter_versions={1}) for domain in M1_SYNC_DOMAINS]
    )
    service = SyncV2Service(
        store=SyncV2Store(SyncDatabase(sqlite_path=tmp_path / "Sync_v2.db")),
        adapters=registry,
        materializers={
            "chat.conversation": ChatConversationMaterializer(chacha_db),
            "chat.message": ChatMessageMaterializer(chacha_db),
            "notes.note": _OrderAssertingNotesMaterializer(chacha_db),
        },
        clock=lambda: "2026-05-23T18:12:00+00:00",
        id_factory=lambda prefix: f"{prefix}-generated",
        settings=SyncV2Settings(
            server_trusted_encryption=_ready_encryption(),
        ),
    )
    service.bootstrap_profile(
        user_id="user-1",
        mode="server_frontend",
        device_id="frontend-device",
        device_name="Server frontend",
    )
    service.register_device(
        user_id="user-1",
        display_name="Offline laptop",
        client_type="chatbook",
        device_id="offline-device",
    )
    return service


def test_note_create_appends_server_origin_envelope_before_projection_and_pulls(
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    result = capture_server_origin_mutation(
        sync_service,
        user_id="user-1",
        domain="notes.note",
        operation="upsert",
        object_id="note-server-1",
        payload={"title": "Server note", "content": "Created through normal Notes API."},
        source="server_api",
    )

    note = chacha_db.get_note_by_id("note-server-1")
    assert note is not None
    assert note["title"] == "Server note"
    assert result.envelope.domain == "notes.note"
    assert result.envelope.device_id == SERVER_ORIGIN_DEVICE_ID
    assert result.envelope.encryption_metadata["policy"] == "server_trusted_v1"
    assert result.envelope.routing_metadata["source"] == "server_api"
    assert "Server note" not in str(result.envelope.routing_metadata)

    pulled = sync_service.pull(
        user_id="user-1",
        dataset_id=result.dataset.dataset_id,
        device_id="offline-device",
        cursor="0",
        domains=["notes.note"],
    )
    assert [item.client_envelope_id for item in pulled.envelopes] == [
        result.envelope.client_envelope_id
    ]


def test_server_origin_capture_rejects_client_private_dataset_before_append(
    monkeypatch: pytest.MonkeyPatch,
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    dataset = sync_service.store.list_datasets_for_user("user-1")[0]
    private_dataset = replace(dataset, encryption_policy="client_private_v1")
    monkeypatch.setattr(
        sync_service.store,
        "list_datasets_for_user",
        lambda user_id: [private_dataset] if user_id == "user-1" else [],
    )

    def fail_insert(envelope):
        raise AssertionError("client-private server-origin mutation must not append")

    monkeypatch.setattr(sync_service.store, "insert_envelope", fail_insert)

    with pytest.raises(SyncServerOriginMutationNotSupportedError) as exc_info:
        capture_server_origin_mutation(
            sync_service,
            user_id="user-1",
            domain="notes.note",
            operation="upsert",
            object_id="note-client-private",
            payload={
                "title": "Private",
                "content": "The server cannot re-encrypt this opaque field.",
            },
            source="server_frontend",
        )

    assert exc_info.value.dataset.dataset_id == dataset.dataset_id
    assert exc_info.value.error_code == CLIENT_PRIVATE_SERVER_FRONTEND_LIMITATION_CODE
    assert chacha_db.get_note_by_id("note-client-private") is None
    assert sync_service.store.list_envelopes_after(
        dataset.dataset_id,
        0,
        domains=["notes.note"],
        limit=10,
    ) == []


def test_chat_conversation_and_message_server_origin_envelopes_are_materialized(
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    conversation = capture_server_origin_mutation(
        sync_service,
        user_id="user-1",
        domain="chat.conversation",
        operation="upsert",
        object_id="chat-server-1",
        payload={
            "title": "Server chat",
            "assistant_kind": "persona",
            "assistant_id": "assistant-1",
            "scope_type": "global",
        },
        source="server_frontend",
    )
    message = capture_server_origin_mutation(
        sync_service,
        user_id="user-1",
        domain="chat.message",
        operation="append",
        object_id="message-server-1",
        parent_id="chat-server-1",
        payload={
            "conversation_id": "chat-server-1",
            "sender": "user",
            "content": "Hello from the server API.",
        },
        source="server_frontend",
    )

    assert chacha_db.get_conversation_by_id("chat-server-1") is not None
    assert chacha_db.get_message_by_id("message-server-1") is not None
    pulled = sync_service.pull(
        user_id="user-1",
        dataset_id=conversation.dataset.dataset_id,
        device_id="offline-device",
        cursor="0",
        domains=["chat.conversation", "chat.message"],
    )
    assert [(item.domain, item.object_id) for item in pulled.envelopes] == [
        ("chat.conversation", "chat-server-1"),
        ("chat.message", "message-server-1"),
    ]
    assert message.envelope.parent_id == "chat-server-1"


def test_chat_message_idempotency_replays_when_only_payload_timestamp_drifts(
    sync_service: SyncV2Service,
) -> None:
    capture_server_origin_mutation(
        sync_service,
        user_id="user-1",
        domain="chat.conversation",
        operation="upsert",
        object_id="chat-idempotent-timestamp",
        payload={
            "title": "Timestamp retry chat",
            "assistant_kind": "persona",
            "assistant_id": "assistant-1",
            "scope_type": "global",
        },
        source="server_api",
    )
    stable_key = "server_api:chat.message:append:timestamp-retry"
    first = capture_server_origin_mutation(
        sync_service,
        user_id="user-1",
        domain="chat.message",
        operation="append",
        object_id="message-idempotent-timestamp",
        parent_id="chat-idempotent-timestamp",
        payload={
            "conversation_id": "chat-idempotent-timestamp",
            "parent_message_id": None,
            "sender": "user",
            "content": "Same message.",
            "timestamp": "2026-05-23T18:12:00+00:00",
            "client_id": "user-1",
        },
        source="server_api",
        stable_key=stable_key,
    )

    second = capture_server_origin_mutation(
        sync_service,
        user_id="user-1",
        domain="chat.message",
        operation="append",
        object_id="message-idempotent-timestamp",
        parent_id="chat-idempotent-timestamp",
        payload={
            "conversation_id": "chat-idempotent-timestamp",
            "parent_message_id": None,
            "sender": "user",
            "content": "Same message.",
            "timestamp": "2026-05-23T18:13:00+00:00",
            "client_id": "user-1",
        },
        source="server_api",
        stable_key=stable_key,
    )

    assert second.envelope.server_cursor == first.envelope.server_cursor
    dataset_id = first.dataset.dataset_id
    envelopes = sync_service.store.list_envelopes_after(
        dataset_id,
        0,
        domains=["chat.message"],
        limit=10,
    )
    assert len(envelopes) == 1
    assert envelopes[0].payload["timestamp"] == "2026-05-23T18:12:00+00:00"

    for changed_payload in (
        {
            "conversation_id": "chat-idempotent-timestamp",
            "parent_message_id": None,
            "sender": "user",
            "content": "Different message.",
            "timestamp": "2026-05-23T18:14:00+00:00",
            "client_id": "user-1",
        },
        {
            "conversation_id": "chat-idempotent-timestamp",
            "parent_message_id": None,
            "sender": "assistant",
            "content": "Same message.",
            "timestamp": "2026-05-23T18:14:00+00:00",
            "client_id": "user-1",
        },
        {
            "conversation_id": "other-chat",
            "parent_message_id": None,
            "sender": "user",
            "content": "Same message.",
            "timestamp": "2026-05-23T18:14:00+00:00",
            "client_id": "user-1",
        },
    ):
        with pytest.raises(SyncServerOriginIdempotencyConflictError):
            capture_server_origin_mutation(
                sync_service,
                user_id="user-1",
                domain="chat.message",
                operation="append",
                object_id="message-idempotent-timestamp",
                parent_id=str(changed_payload["conversation_id"]),
                payload=changed_payload,
                source="server_api",
                stable_key=stable_key,
            )


def test_append_failure_prevents_server_projection(tmp_path: Path, chacha_db: CharactersRAGDB) -> None:
    service = SyncV2Service(
        store=_FailingInsertStore(SyncDatabase(sqlite_path=tmp_path / "Sync_v2.db")),
        adapters=SyncAdapterRegistry([StaticSyncAdapter(domain="notes.note", supported_adapter_versions={1})]),
        materializers={"notes.note": NotesMaterializer(chacha_db)},
        settings=SyncV2Settings(
            supported_domains=["notes.note"],
            operations={"notes.note": ["upsert", "tombstone"]},
            server_trusted_encryption=_ready_encryption(),
        ),
    )
    service.bootstrap_profile(user_id="user-1", mode="server_frontend", device_id="frontend-device")

    with pytest.raises(SyncStoreError):
        capture_server_origin_mutation(
            service,
            user_id="user-1",
            domain="notes.note",
            operation="upsert",
            object_id="note-no-log",
            payload={"title": "No log", "content": "Should not project."},
            source="server_api",
        )

    assert chacha_db.get_note_by_id("note-no-log") is None


def test_materialization_failure_leaves_replayable_failed_envelope_and_profile_status(
    tmp_path: Path,
    chacha_db: CharactersRAGDB,
) -> None:
    service = SyncV2Service(
        store=SyncV2Store(SyncDatabase(sqlite_path=tmp_path / "Sync_v2.db")),
        adapters=SyncAdapterRegistry([StaticSyncAdapter(domain="notes.note", supported_adapter_versions={1})]),
        materializers={"notes.note": _FailingApplyMaterializer()},
        clock=lambda: "2026-05-23T18:12:00+00:00",
        settings=SyncV2Settings(
            supported_domains=["notes.note"],
            operations={"notes.note": ["upsert", "tombstone"]},
            server_trusted_encryption=_ready_encryption(),
        ),
    )
    profile = service.bootstrap_profile(
        user_id="user-1",
        mode="server_frontend",
        device_id="frontend-device",
        requested_domains=["notes.note"],
    )

    with pytest.raises(SyncServerOriginMaterializationError):
        capture_server_origin_mutation(
            service,
            user_id="user-1",
            domain="notes.note",
            operation="upsert",
            object_id="note-failed-apply",
            payload={"title": "Failed", "content": "Projection fails."},
            source="server_api",
        )

    envelopes = service.store.list_envelopes_after(
        profile.active_dataset_id or "",
        0,
        domains=["notes.note"],
        limit=10,
    )
    assert [(item.status, item.apply_status, item.apply_error_code) for item in envelopes] == [
        ("accepted", "failed", "sync_projection_failed")
    ]
    assert "RuntimeError" in (envelopes[0].apply_error_message or "")
    assert chacha_db.get_note_by_id("note-failed-apply") is None

    status = service.profile_status(user_id="user-1", dataset_id=profile.active_dataset_id or "")
    domains = {item.domain: item for item in status.domain_status}
    assert domains["notes.note"].failed_apply_count == 1
    assert domains["notes.note"].last_apply_status == "failed"


def test_materialization_conflict_reports_accepted_conflict_envelope(tmp_path: Path) -> None:
    service = SyncV2Service(
        store=SyncV2Store(SyncDatabase(sqlite_path=tmp_path / "Sync_v2.db")),
        adapters=SyncAdapterRegistry([StaticSyncAdapter(domain="notes.note", supported_adapter_versions={1})]),
        materializers={"notes.note": _ConflictingApplyMaterializer()},
        settings=SyncV2Settings(
            supported_domains=["notes.note"],
            operations={"notes.note": ["upsert", "tombstone"]},
            server_trusted_encryption=_ready_encryption(),
        ),
    )
    profile = service.bootstrap_profile(
        user_id="user-1",
        mode="server_frontend",
        device_id="frontend-device",
        requested_domains=["notes.note"],
    )

    with pytest.raises(SyncServerOriginMaterializationError) as exc_info:
        capture_server_origin_mutation(
            service,
            user_id="user-1",
            domain="notes.note",
            operation="upsert",
            object_id="note-conflict",
            payload={"title": "Conflict", "content": "Projection conflicts."},
            source="server_api",
        )

    assert exc_info.value.envelope.status == "accepted"
    assert exc_info.value.envelope.apply_status == "conflict"
    assert exc_info.value.envelope.server_cursor is not None
    envelopes = service.store.list_envelopes_after(
        profile.active_dataset_id or "",
        0,
        domains=["notes.note"],
        limit=10,
    )
    assert [(item.status, item.apply_status, item.apply_error_code) for item in envelopes] == [
        ("accepted", "conflict", "whole_object_conflict")
    ]


def test_normal_notes_create_api_routes_personal_write_through_sync_when_active(
    monkeypatch: pytest.MonkeyPatch,
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    client = _notes_app(monkeypatch, chacha_db=chacha_db, sync_service=sync_service)

    response = client.post(
        "/api/v1/notes/",
        json={
            "id": "note-api-1",
            "title": "API note",
            "content": "Created through the normal Notes API.",
        },
    )

    assert response.status_code == 201
    assert chacha_db.get_note_by_id("note-api-1")["client_id"] == "user-1"
    update_response = client.put(
        "/api/v1/notes/note-api-1",
        headers={"expected-version": str(response.json()["version"])},
        json={"title": "API note renamed", "content": "Updated content."},
    )
    assert update_response.status_code == 200
    assert chacha_db.get_note_by_id("note-api-1")["client_id"] == "user-1"
    patch_response = client.patch(
        "/api/v1/notes/note-api-1",
        headers={"expected-version": str(update_response.json()["version"])},
        json={"content": "Patched content."},
    )
    assert patch_response.status_code == 200
    assert chacha_db.get_note_by_id("note-api-1")["client_id"] == "user-1"
    delete_response = client.delete(
        "/api/v1/notes/note-api-1",
        headers={"expected-version": str(patch_response.json()["version"])},
    )
    assert delete_response.status_code == 204
    assert chacha_db.get_note_by_id("note-api-1", include_deleted=True)["client_id"] == "user-1"
    envelopes = sync_service.store.list_envelopes_after(
        sync_service.profile(user_id="user-1").active_dataset_id or "",
        0,
        domains=["notes.note"],
        limit=10,
    )
    assert [(item.domain, item.object_id, item.device_id, item.apply_status) for item in envelopes] == [
        ("notes.note", "note-api-1", SERVER_ORIGIN_DEVICE_ID, "applied"),
        ("notes.note", "note-api-1", SERVER_ORIGIN_DEVICE_ID, "applied"),
        ("notes.note", "note-api-1", SERVER_ORIGIN_DEVICE_ID, "applied"),
        ("notes.note", "note-api-1", SERVER_ORIGIN_DEVICE_ID, "applied"),
    ]


def test_normal_notes_create_api_reports_client_private_server_frontend_limitation(
    monkeypatch: pytest.MonkeyPatch,
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    dataset = sync_service.store.list_datasets_for_user("user-1")[0]
    private_dataset = replace(dataset, encryption_policy="client_private_v1")
    monkeypatch.setattr(
        sync_service.store,
        "list_datasets_for_user",
        lambda user_id: [private_dataset] if user_id == "user-1" else [],
    )
    client = _notes_app(monkeypatch, chacha_db=chacha_db, sync_service=sync_service)

    response = client.post(
        "/api/v1/notes/",
        json={
            "id": "note-api-client-private",
            "title": "API private",
            "content": "This cannot be accepted through server-origin capture.",
        },
    )

    assert response.status_code == 409
    assert response.json()["detail"]["error_code"] == CLIENT_PRIVATE_SERVER_FRONTEND_LIMITATION_CODE
    assert chacha_db.get_note_by_id("note-api-client-private") is None
    assert sync_service.store.list_envelopes_after(
        dataset.dataset_id,
        0,
        domains=["notes.note"],
        limit=10,
    ) == []


def test_normal_chat_create_api_reports_client_private_server_frontend_limitation(
    monkeypatch: pytest.MonkeyPatch,
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    character_id = chacha_db.add_character_card({"name": "Assistant"})
    assert character_id is not None
    dataset = sync_service.store.list_datasets_for_user("user-1")[0]
    private_dataset = replace(dataset, encryption_policy="client_private_v1")
    monkeypatch.setattr(
        sync_service.store,
        "list_datasets_for_user",
        lambda user_id: [private_dataset] if user_id == "user-1" else [],
    )
    client = _chat_messages_app(monkeypatch, chacha_db=chacha_db, sync_service=sync_service)
    idempotency_key = "client-private-chat-create"
    expected_chat_id = server_origin_object_id("chat.conversation", idempotency_key)
    assert expected_chat_id is not None

    response = client.post(
        "/api/v1/chats/",
        headers={"Idempotency-Key": idempotency_key},
        json={"character_id": character_id, "title": "API private chat"},
    )

    assert response.status_code == 409
    assert response.json()["detail"]["error_code"] == CLIENT_PRIVATE_SERVER_FRONTEND_LIMITATION_CODE
    assert chacha_db.get_conversation_by_id(expected_chat_id) is None
    assert sync_service.store.list_envelopes_after(
        dataset.dataset_id,
        0,
        domains=["chat.conversation"],
        limit=10,
    ) == []


def test_normal_chat_message_api_reports_client_private_server_frontend_limitation(
    monkeypatch: pytest.MonkeyPatch,
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    character_id = chacha_db.add_character_card({"name": "Assistant"})
    assert character_id is not None
    conversation_id = chacha_db.add_conversation(
        {
            "id": "chat-existing-private",
            "character_id": character_id,
            "title": "Existing direct chat",
            "client_id": "user-1",
        }
    )
    dataset = sync_service.store.list_datasets_for_user("user-1")[0]
    private_dataset = replace(dataset, encryption_policy="client_private_v1")
    monkeypatch.setattr(
        sync_service.store,
        "list_datasets_for_user",
        lambda user_id: [private_dataset] if user_id == "user-1" else [],
    )
    client = _chat_messages_app(monkeypatch, chacha_db=chacha_db, sync_service=sync_service)
    idempotency_key = "client-private-message-create"
    expected_message_id = server_origin_object_id("chat.message", idempotency_key)
    assert expected_message_id is not None

    response = client.post(
        f"/api/v1/chats/{conversation_id}/messages",
        headers={"Idempotency-Key": idempotency_key},
        json={"role": "user", "content": "This message cannot be server-origin captured."},
    )

    assert response.status_code == 409
    assert response.json()["detail"]["error_code"] == CLIENT_PRIVATE_SERVER_FRONTEND_LIMITATION_CODE
    assert chacha_db.get_message_by_id(expected_message_id) is None
    assert chacha_db.get_messages_for_conversation(conversation_id) == []
    assert sync_service.store.list_envelopes_after(
        dataset.dataset_id,
        0,
        domains=["chat.message"],
        limit=10,
    ) == []


def test_active_sync_note_keywords_are_rejected_without_direct_mutation(
    monkeypatch: pytest.MonkeyPatch,
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    client = _notes_app(monkeypatch, chacha_db=chacha_db, sync_service=sync_service)

    create_with_keywords = client.post(
        "/api/v1/notes/",
        json={
            "id": "note-keywords-create",
            "title": "Keywords",
            "content": "Rejected while sync is active.",
            "keywords": ["alpha"],
        },
    )
    assert create_with_keywords.status_code == 400
    assert create_with_keywords.json()["detail"]["error_code"] == "sync_v2_keywords_not_supported"
    assert chacha_db.get_note_by_id("note-keywords-create") is None
    assert sync_service.store.list_envelopes_after(
        sync_service.profile(user_id="user-1").active_dataset_id or "",
        0,
        domains=["notes.note"],
        limit=10,
    ) == []

    create_response = client.post(
        "/api/v1/notes/",
        json={
            "id": "note-keywords-update",
            "title": "Plain note",
            "content": "Created without keywords.",
        },
    )
    assert create_response.status_code == 201
    update_keywords_only = client.put(
        "/api/v1/notes/note-keywords-update",
        headers={"expected-version": str(create_response.json()["version"])},
        json={"keywords": ["beta"]},
    )
    assert update_keywords_only.status_code == 400
    assert update_keywords_only.json()["detail"]["error_code"] == "sync_v2_keywords_not_supported"
    assert chacha_db.get_keywords_for_note("note-keywords-update") == []

    patch_keywords = client.patch(
        "/api/v1/notes/note-keywords-update",
        headers={"expected-version": str(create_response.json()["version"])},
        json={"keywords": ["gamma"]},
    )
    assert patch_keywords.status_code == 400
    assert patch_keywords.json()["detail"]["error_code"] == "sync_v2_keywords_not_supported"
    assert chacha_db.get_keywords_for_note("note-keywords-update") == []
    envelopes = sync_service.store.list_envelopes_after(
        sync_service.profile(user_id="user-1").active_dataset_id or "",
        0,
        domains=["notes.note"],
        limit=10,
    )
    assert [(item.operation, item.object_id) for item in envelopes] == [
        ("upsert", "note-keywords-update")
    ]


def test_inactive_sync_note_keywords_keep_existing_direct_behavior(
    monkeypatch: pytest.MonkeyPatch,
    chacha_db: CharactersRAGDB,
) -> None:
    client = _notes_app(monkeypatch, chacha_db=chacha_db, sync_service=None)

    response = client.post(
        "/api/v1/notes/",
        json={
            "id": "note-keywords-direct",
            "title": "Direct keywords",
            "content": "Inactive sync still supports keywords.",
            "keywords": ["direct"],
        },
    )

    assert response.status_code == 201
    assert [item["keyword"] for item in chacha_db.get_keywords_for_note("note-keywords-direct")] == ["direct"]


def test_inactive_sync_note_create_ignores_idempotency_key_for_direct_id_generation(
    monkeypatch: pytest.MonkeyPatch,
    chacha_db: CharactersRAGDB,
) -> None:
    client = _notes_app(monkeypatch, chacha_db=chacha_db, sync_service=None)
    idempotency_key = "inactive-note-retry"

    response = client.post(
        "/api/v1/notes/",
        headers={"Idempotency-Key": idempotency_key},
        json={
            "title": "Direct note",
            "content": "Inactive Sync keeps direct create behavior.",
        },
    )

    assert response.status_code == 201
    assert response.json()["id"] != server_origin_object_id("notes.note", idempotency_key)
    assert chacha_db.get_note_by_id(response.json()["id"]) is not None


def test_active_sync_note_restore_is_rejected_without_direct_projection_write(
    monkeypatch: pytest.MonkeyPatch,
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    client = _notes_app(monkeypatch, chacha_db=chacha_db, sync_service=sync_service)
    create_response = client.post(
        "/api/v1/notes/",
        json={"id": "note-restore-active", "title": "Restore", "content": "Delete me."},
    )
    assert create_response.status_code == 201
    delete_response = client.delete(
        "/api/v1/notes/note-restore-active",
        headers={"expected-version": str(create_response.json()["version"])},
    )
    assert delete_response.status_code == 204
    deleted_note = chacha_db.get_note_by_id("note-restore-active", include_deleted=True)
    assert deleted_note["deleted"] in (1, True)

    restore_response = client.post(
        "/api/v1/notes/note-restore-active/restore",
        params={"expected_version": deleted_note["version"]},
    )

    assert restore_response.status_code == 400
    assert restore_response.json()["detail"]["error_code"] == "sync_v2_note_restore_not_supported"
    assert chacha_db.get_note_by_id("note-restore-active", include_deleted=True)["deleted"] in (1, True)
    dataset_id = sync_service.profile(user_id="user-1").active_dataset_id or ""
    envelopes = sync_service.store.list_envelopes_after(
        dataset_id,
        0,
        domains=["notes.note"],
        limit=10,
    )
    assert [(item.operation, item.object_id) for item in envelopes] == [
        ("upsert", "note-restore-active"),
        ("tombstone", "note-restore-active"),
    ]


def test_inactive_sync_note_restore_keeps_existing_direct_behavior(
    monkeypatch: pytest.MonkeyPatch,
    chacha_db: CharactersRAGDB,
) -> None:
    client = _notes_app(monkeypatch, chacha_db=chacha_db, sync_service=None)
    create_response = client.post(
        "/api/v1/notes/",
        json={"id": "note-restore-direct", "title": "Restore", "content": "Direct restore."},
    )
    assert create_response.status_code == 201
    delete_response = client.delete(
        "/api/v1/notes/note-restore-direct",
        headers={"expected-version": str(create_response.json()["version"])},
    )
    assert delete_response.status_code == 204
    deleted_note = chacha_db.get_note_by_id("note-restore-direct", include_deleted=True)
    assert deleted_note["deleted"] in (1, True)

    restore_response = client.post(
        "/api/v1/notes/note-restore-direct/restore",
        params={"expected_version": deleted_note["version"]},
    )

    assert restore_response.status_code == 200
    assert restore_response.json()["deleted"] is False
    assert chacha_db.get_note_by_id("note-restore-direct", include_deleted=True)["deleted"] in (0, False)


def test_active_sync_note_create_idempotency_key_replays_and_rejects_conflicts(
    monkeypatch: pytest.MonkeyPatch,
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    client = _notes_app(monkeypatch, chacha_db=chacha_db, sync_service=sync_service)
    payload = {"title": "Retryable note", "content": "Created once."}
    headers = {"Idempotency-Key": "note-create-retry"}

    first = client.post("/api/v1/notes/", headers=headers, json=payload)
    second = client.post("/api/v1/notes/", headers=headers, json=payload)

    assert first.status_code == 201
    assert second.status_code == 201
    assert second.json()["id"] == first.json()["id"]
    dataset_id = sync_service.profile(user_id="user-1").active_dataset_id or ""
    envelopes = sync_service.store.list_envelopes_after(
        dataset_id,
        0,
        domains=["notes.note"],
        limit=10,
    )
    assert [(item.operation, item.object_id) for item in envelopes] == [
        ("upsert", first.json()["id"])
    ]

    conflict = client.post(
        "/api/v1/notes/",
        headers=headers,
        json={"title": "Retryable note", "content": "Different content."},
    )

    assert conflict.status_code == 409
    assert conflict.json()["detail"]["error_code"] == "sync_server_origin_idempotency_conflict"
    assert len(
        sync_service.store.list_envelopes_after(dataset_id, 0, domains=["notes.note"], limit=10)
    ) == 1


def test_active_sync_note_import_bulk_and_keyword_links_do_not_bypass_sync(
    monkeypatch: pytest.MonkeyPatch,
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    client = _notes_app(monkeypatch, chacha_db=chacha_db, sync_service=sync_service)

    import_response = client.post(
        "/api/v1/notes/import",
        json={
            "duplicate_strategy": "overwrite",
            "items": [
                {
                    "format": "json",
                    "content": json.dumps(
                        [{"id": "import-sync-1", "title": "Imported", "content": "Via import."}]
                    ),
                }
            ],
        },
    )
    assert import_response.status_code == 200
    assert import_response.json()["created_count"] == 1
    assert chacha_db.get_note_by_id("import-sync-1")["client_id"] == "user-1"

    import_keywords = client.post(
        "/api/v1/notes/import",
        json={
            "items": [
                {
                    "format": "json",
                    "content": json.dumps(
                        [
                            {
                                "id": "import-keyword-reject",
                                "title": "Keywords",
                                "content": "Rejected.",
                                "keywords": ["alpha"],
                            }
                        ]
                    ),
                }
            ],
        },
    )
    assert import_keywords.status_code == 400
    assert import_keywords.json()["detail"]["error_code"] == "sync_v2_keywords_not_supported"
    assert chacha_db.get_note_by_id("import-keyword-reject") is None

    bulk_response = client.post(
        "/api/v1/notes/bulk",
        json={"notes": [{"id": "bulk-sync-1", "title": "Bulk", "content": "Via bulk."}]},
    )
    assert bulk_response.status_code == 200
    assert bulk_response.json()["created_count"] == 1
    assert chacha_db.get_note_by_id("bulk-sync-1")["client_id"] == "user-1"

    bulk_keywords = client.post(
        "/api/v1/notes/bulk",
        json={
            "notes": [
                {
                    "id": "bulk-keyword-reject",
                    "title": "Bulk keywords",
                    "content": "Rejected.",
                    "keywords": ["beta"],
                }
            ]
        },
    )
    assert bulk_keywords.status_code == 400
    assert bulk_keywords.json()["detail"]["error_code"] == "sync_v2_keywords_not_supported"
    assert chacha_db.get_note_by_id("bulk-keyword-reject") is None

    keyword_id = chacha_db.add_keyword("linked")
    assert keyword_id is not None
    link_response = client.post(f"/api/v1/notes/import-sync-1/keywords/{keyword_id}")
    assert link_response.status_code == 400
    assert link_response.json()["detail"]["error_code"] == "sync_v2_keywords_not_supported"
    assert chacha_db.get_keywords_for_note("import-sync-1") == []
    chacha_db.link_note_to_keyword("import-sync-1", keyword_id)
    unlink_response = client.delete(f"/api/v1/notes/import-sync-1/keywords/{keyword_id}")
    assert unlink_response.status_code == 400
    assert unlink_response.json()["detail"]["error_code"] == "sync_v2_keywords_not_supported"
    assert [item["keyword"] for item in chacha_db.get_keywords_for_note("import-sync-1")] == ["linked"]

    dataset_id = sync_service.profile(user_id="user-1").active_dataset_id or ""
    envelopes = sync_service.store.list_envelopes_after(
        dataset_id,
        0,
        domains=["notes.note"],
        limit=10,
    )
    assert [(item.operation, item.object_id) for item in envelopes] == [
        ("upsert", "import-sync-1"),
        ("upsert", "bulk-sync-1"),
    ]


def test_normal_chat_and_message_apis_route_personal_writes_through_sync_when_active(
    monkeypatch: pytest.MonkeyPatch,
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    character_id = chacha_db.add_character_card({"name": "Assistant"})
    assert character_id is not None
    app = FastAPI()
    app.include_router(chat_sessions_endpoint.router, prefix="/api/v1/chats")
    app.include_router(messages_endpoint.router, prefix="/api/v1")

    async def _db_override():
        return chacha_db

    async def _user_override():
        return User(id="user-1", username="user-1", is_admin=True)

    app.dependency_overrides[chat_sessions_endpoint.get_chacha_db_for_user] = _db_override
    app.dependency_overrides[messages_endpoint.get_chacha_db_for_user] = _db_override
    app.dependency_overrides[chat_sessions_endpoint.get_request_user] = _user_override
    app.dependency_overrides[messages_endpoint.get_request_user] = _user_override
    monkeypatch.setattr(
        chat_sessions_endpoint,
        "get_active_server_origin_sync_service_for_user",
        lambda user_id: sync_service,
        raising=False,
    )
    monkeypatch.setattr(
        messages_endpoint,
        "get_active_server_origin_sync_service_for_user",
        lambda user_id: sync_service,
        raising=False,
    )
    monkeypatch.setattr(
        chat_sessions_endpoint,
        "get_character_rate_limiter",
        lambda: _NoopCharacterRateLimiter(),
    )
    monkeypatch.setattr(
        messages_endpoint,
        "get_character_rate_limiter",
        lambda: _NoopCharacterRateLimiter(),
    )
    client = TestClient(app)

    chat_response = client.post(
        "/api/v1/chats/",
        json={"character_id": character_id, "title": "API chat"},
    )
    assert chat_response.status_code == 201
    chat_id = chat_response.json()["id"]
    update_response = client.put(
        f"/api/v1/chats/{chat_id}",
        params={"expected_version": chat_response.json()["version"]},
        json={"title": "API chat renamed"},
    )
    assert update_response.status_code == 200
    message_response = client.post(
        f"/api/v1/chats/{chat_id}/messages",
        json={"role": "user", "content": "Hello from the normal message API."},
    )

    assert message_response.status_code == 201
    message_id = message_response.json()["id"]
    message_delete = client.delete(
        f"/api/v1/messages/{message_id}",
        params={"expected_version": message_response.json()["version"]},
    )
    assert message_delete.status_code == 204
    chat_delete = client.delete(
        f"/api/v1/chats/{chat_id}",
        params={"expected_version": update_response.json()["version"]},
    )
    assert chat_delete.status_code == 204
    envelopes = sync_service.store.list_envelopes_after(
        sync_service.profile(user_id="user-1").active_dataset_id or "",
        0,
        domains=["chat.conversation", "chat.message"],
        limit=10,
    )
    assert [
        (item.domain, item.operation, item.object_id, item.device_id, item.apply_status)
        for item in envelopes
    ] == [
        ("chat.conversation", "upsert", chat_id, SERVER_ORIGIN_DEVICE_ID, "applied"),
        ("chat.conversation", "upsert", chat_id, SERVER_ORIGIN_DEVICE_ID, "applied"),
        ("chat.message", "append", message_id, SERVER_ORIGIN_DEVICE_ID, "applied"),
        ("chat.message", "tombstone", message_id, SERVER_ORIGIN_DEVICE_ID, "applied"),
        ("chat.conversation", "tombstone", chat_id, SERVER_ORIGIN_DEVICE_ID, "applied"),
    ]


def test_workspace_chat_api_write_stays_direct_when_sync_active(
    monkeypatch: pytest.MonkeyPatch,
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    character_id = chacha_db.add_character_card({"name": "Workspace Assistant"})
    assert character_id is not None
    app = FastAPI()
    app.include_router(chat_sessions_endpoint.router, prefix="/api/v1/chats")

    async def _db_override():
        return chacha_db

    async def _user_override():
        return User(id="user-1", username="user-1", is_admin=True)

    app.dependency_overrides[chat_sessions_endpoint.get_chacha_db_for_user] = _db_override
    app.dependency_overrides[chat_sessions_endpoint.get_request_user] = _user_override
    created_conversations: dict[str, dict] = {}

    def _fake_add_conversation(data: dict) -> str:
        row = dict(data)
        row["state"] = row.get("state") or "active"
        row["version"] = row.get("version") or 1
        row["deleted"] = row.get("deleted") or 0
        row["assistant_display_name"] = row.get("assistant_display_name") or "Workspace Assistant"
        created_conversations[data["id"]] = row
        return data["id"]

    def _fake_get_conversation_by_id(conversation_id: str, include_deleted: bool = False):
        return created_conversations.get(conversation_id)

    monkeypatch.setattr(chacha_db, "add_conversation", _fake_add_conversation)
    monkeypatch.setattr(chacha_db, "get_conversation_by_id", _fake_get_conversation_by_id)
    monkeypatch.setattr(chacha_db, "upsert_conversation_settings", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        chat_sessions_endpoint,
        "get_active_server_origin_sync_service_for_user",
        lambda user_id: sync_service,
        raising=False,
    )
    monkeypatch.setattr(
        chat_sessions_endpoint,
        "get_character_rate_limiter",
        lambda: _NoopCharacterRateLimiter(),
    )

    response = TestClient(app).post(
        "/api/v1/chats/",
        json={
            "character_id": character_id,
            "title": "Workspace chat",
            "scope_type": "workspace",
            "workspace_id": "workspace-1",
        },
    )

    assert response.status_code == 201
    envelopes = sync_service.store.list_envelopes_after(
        sync_service.profile(user_id="user-1").active_dataset_id or "",
        0,
        domains=["chat.conversation"],
        limit=10,
    )
    assert envelopes == []
    assert chacha_db.get_conversation_by_id(response.json()["id"]) is not None


def test_active_sync_chat_delete_tombstones_child_messages_before_conversation(
    monkeypatch: pytest.MonkeyPatch,
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    character_id = chacha_db.add_character_card({"name": "Assistant"})
    assert character_id is not None
    app = FastAPI()
    app.include_router(chat_sessions_endpoint.router, prefix="/api/v1/chats")
    app.include_router(messages_endpoint.router, prefix="/api/v1")

    async def _db_override():
        return chacha_db

    async def _user_override():
        return User(id="user-1", username="user-1", is_admin=True)

    app.dependency_overrides[chat_sessions_endpoint.get_chacha_db_for_user] = _db_override
    app.dependency_overrides[messages_endpoint.get_chacha_db_for_user] = _db_override
    app.dependency_overrides[chat_sessions_endpoint.get_request_user] = _user_override
    app.dependency_overrides[messages_endpoint.get_request_user] = _user_override
    monkeypatch.setattr(
        chat_sessions_endpoint,
        "get_active_server_origin_sync_service_for_user",
        lambda user_id: sync_service,
        raising=False,
    )
    monkeypatch.setattr(
        messages_endpoint,
        "get_active_server_origin_sync_service_for_user",
        lambda user_id: sync_service,
        raising=False,
    )
    monkeypatch.setattr(
        chat_sessions_endpoint,
        "get_character_rate_limiter",
        lambda: _NoopCharacterRateLimiter(),
    )
    monkeypatch.setattr(
        messages_endpoint,
        "get_character_rate_limiter",
        lambda: _NoopCharacterRateLimiter(),
    )
    client = TestClient(app)

    chat_response = client.post(
        "/api/v1/chats/",
        json={"character_id": character_id, "title": "Delete chat"},
    )
    assert chat_response.status_code == 201
    chat_id = chat_response.json()["id"]
    first_message = client.post(
        f"/api/v1/chats/{chat_id}/messages",
        json={"role": "user", "content": "First message."},
    )
    second_message = client.post(
        f"/api/v1/chats/{chat_id}/messages",
        json={"role": "assistant", "content": "Second message."},
    )
    assert first_message.status_code == 201
    assert second_message.status_code == 201
    message_ids = [first_message.json()["id"], second_message.json()["id"]]

    delete_response = client.delete(
        f"/api/v1/chats/{chat_id}",
        params={"expected_version": chat_response.json()["version"]},
    )

    assert delete_response.status_code == 204
    dataset_id = sync_service.profile(user_id="user-1").active_dataset_id or ""
    for message_id in message_ids:
        assert chacha_db.get_message_by_id(message_id, include_deleted=True)["deleted"] in (1, True)
        state = sync_service.store.get_object_state(dataset_id, "chat.message", message_id)
        assert state is not None
        assert state.deleted is True
    envelopes = sync_service.store.list_envelopes_after(
        dataset_id,
        0,
        domains=["chat.message", "chat.conversation"],
        limit=20,
    )
    assert [
        (item.domain, item.operation, item.object_id)
        for item in envelopes
    ] == [
        ("chat.conversation", "upsert", chat_id),
        ("chat.message", "append", message_ids[0]),
        ("chat.message", "append", message_ids[1]),
        ("chat.message", "tombstone", message_ids[0]),
        ("chat.message", "tombstone", message_ids[1]),
        ("chat.conversation", "tombstone", chat_id),
    ]
    pulled = sync_service.pull(
        user_id="user-1",
        dataset_id=dataset_id,
        device_id="offline-device",
        cursor="0",
        domains=["chat.message"],
    )
    assert [
        (item.operation, item.object_id)
        for item in pulled.envelopes
        if item.operation == "tombstone"
    ] == [
        ("tombstone", message_ids[0]),
        ("tombstone", message_ids[1]),
    ]


def test_active_sync_chat_restore_is_rejected_without_direct_projection_write(
    monkeypatch: pytest.MonkeyPatch,
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    character_id = chacha_db.add_character_card({"name": "Assistant"})
    assert character_id is not None
    client = _chat_messages_app(monkeypatch, chacha_db=chacha_db, sync_service=sync_service)
    chat_response = client.post(
        "/api/v1/chats/",
        json={"character_id": character_id, "title": "Restore blocked"},
    )
    assert chat_response.status_code == 201
    chat_id = chat_response.json()["id"]
    delete_response = client.delete(
        f"/api/v1/chats/{chat_id}",
        params={"expected_version": chat_response.json()["version"]},
    )
    assert delete_response.status_code == 204
    deleted_conversation = chacha_db.get_conversation_by_id(chat_id, include_deleted=True)
    assert deleted_conversation["deleted"] in (1, True)

    restore_response = client.post(
        f"/api/v1/chats/{chat_id}/restore",
        params={"expected_version": deleted_conversation["version"]},
    )

    assert restore_response.status_code == 400
    assert restore_response.json()["detail"]["error_code"] == "sync_v2_chat_restore_not_supported"
    assert chacha_db.get_conversation_by_id(chat_id, include_deleted=True)["deleted"] in (1, True)
    dataset_id = sync_service.profile(user_id="user-1").active_dataset_id or ""
    envelopes = sync_service.store.list_envelopes_after(
        dataset_id,
        0,
        domains=["chat.conversation"],
        limit=10,
    )
    assert [(item.operation, item.object_id) for item in envelopes] == [
        ("upsert", chat_id),
        ("tombstone", chat_id),
    ]


def test_inactive_sync_chat_restore_keeps_existing_direct_behavior(
    monkeypatch: pytest.MonkeyPatch,
    chacha_db: CharactersRAGDB,
) -> None:
    character_id = chacha_db.add_character_card({"name": "Assistant"})
    assert character_id is not None
    client = _chat_messages_app(monkeypatch, chacha_db=chacha_db, sync_service=None)
    chat_response = client.post(
        "/api/v1/chats/",
        json={"character_id": character_id, "title": "Direct restore"},
    )
    assert chat_response.status_code == 201
    chat_id = chat_response.json()["id"]
    delete_response = client.delete(
        f"/api/v1/chats/{chat_id}",
        params={"expected_version": chat_response.json()["version"]},
    )
    assert delete_response.status_code == 204
    deleted_conversation = chacha_db.get_conversation_by_id(chat_id, include_deleted=True)
    assert deleted_conversation["deleted"] in (1, True)

    restore_response = client.post(
        f"/api/v1/chats/{chat_id}/restore",
        params={"expected_version": deleted_conversation["version"]},
    )

    assert restore_response.status_code == 200
    assert restore_response.json()["id"] == chat_id
    assert chacha_db.get_conversation_by_id(chat_id, include_deleted=True)["deleted"] in (0, False)


def test_active_sync_chat_hard_delete_is_rejected_without_removing_projection(
    monkeypatch: pytest.MonkeyPatch,
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    character_id = chacha_db.add_character_card({"name": "Assistant"})
    assert character_id is not None
    client = _chat_messages_app(monkeypatch, chacha_db=chacha_db, sync_service=sync_service)
    chat_response = client.post(
        "/api/v1/chats/",
        json={"character_id": character_id, "title": "Hard delete blocked"},
    )
    assert chat_response.status_code == 201
    chat_id = chat_response.json()["id"]
    delete_response = client.delete(
        f"/api/v1/chats/{chat_id}",
        params={"expected_version": chat_response.json()["version"]},
    )
    assert delete_response.status_code == 204
    deleted_conversation = chacha_db.get_conversation_by_id(chat_id, include_deleted=True)
    assert deleted_conversation["deleted"] in (1, True)

    stale_version_response = client.delete(
        f"/api/v1/chats/{chat_id}",
        params={"hard_delete": True, "expected_version": deleted_conversation["version"] + 1},
    )
    assert stale_version_response.status_code == 409

    hard_delete_response = client.delete(
        f"/api/v1/chats/{chat_id}",
        params={"hard_delete": True, "expected_version": deleted_conversation["version"]},
    )

    assert hard_delete_response.status_code == 400
    assert hard_delete_response.json()["detail"]["error_code"] == "sync_v2_chat_hard_delete_not_supported"
    assert chacha_db.get_conversation_by_id(chat_id, include_deleted=True)["deleted"] in (1, True)
    dataset_id = sync_service.profile(user_id="user-1").active_dataset_id or ""
    envelopes = sync_service.store.list_envelopes_after(
        dataset_id,
        0,
        domains=["chat.conversation"],
        limit=10,
    )
    assert [(item.operation, item.object_id) for item in envelopes] == [
        ("upsert", chat_id),
        ("tombstone", chat_id),
    ]


def test_inactive_sync_chat_hard_delete_keeps_existing_direct_behavior(
    monkeypatch: pytest.MonkeyPatch,
    chacha_db: CharactersRAGDB,
) -> None:
    character_id = chacha_db.add_character_card({"name": "Assistant"})
    assert character_id is not None
    client = _chat_messages_app(monkeypatch, chacha_db=chacha_db, sync_service=None)
    chat_response = client.post(
        "/api/v1/chats/",
        json={"character_id": character_id, "title": "Direct hard delete"},
    )
    assert chat_response.status_code == 201
    chat_id = chat_response.json()["id"]
    delete_response = client.delete(
        f"/api/v1/chats/{chat_id}",
        params={"expected_version": chat_response.json()["version"]},
    )
    assert delete_response.status_code == 204
    deleted_conversation = chacha_db.get_conversation_by_id(chat_id, include_deleted=True)
    assert deleted_conversation["deleted"] in (1, True)

    hard_delete_response = client.delete(
        f"/api/v1/chats/{chat_id}",
        params={"hard_delete": True, "expected_version": deleted_conversation["version"]},
    )

    assert hard_delete_response.status_code == 204
    assert chacha_db.get_conversation_by_id(chat_id, include_deleted=True) is None


def test_active_sync_chat_and_message_create_idempotency_keys_are_replayable(
    monkeypatch: pytest.MonkeyPatch,
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    character_id = chacha_db.add_character_card({"name": "Assistant"})
    assert character_id is not None
    client = _chat_messages_app(monkeypatch, chacha_db=chacha_db, sync_service=sync_service)

    chat_payload = {"character_id": character_id, "title": "Retryable chat"}
    chat_headers = {"Idempotency-Key": "chat-create-retry"}
    first_chat = client.post("/api/v1/chats/", headers=chat_headers, json=chat_payload)
    second_chat = client.post("/api/v1/chats/", headers=chat_headers, json=chat_payload)

    assert first_chat.status_code == 201
    assert second_chat.status_code == 201
    assert second_chat.json()["id"] == first_chat.json()["id"]
    chat_conflict = client.post(
        "/api/v1/chats/",
        headers=chat_headers,
        json={"character_id": character_id, "title": "Different title"},
    )
    assert chat_conflict.status_code == 409
    assert chat_conflict.json()["detail"]["error_code"] == "sync_server_origin_idempotency_conflict"

    chat_id = first_chat.json()["id"]
    message_headers = {"Idempotency-Key": "message-create-retry"}
    message_payload = {"role": "user", "content": "Created once."}
    first_message = client.post(
        f"/api/v1/chats/{chat_id}/messages",
        headers=message_headers,
        json=message_payload,
    )
    second_message = client.post(
        f"/api/v1/chats/{chat_id}/messages",
        headers=message_headers,
        json=message_payload,
    )
    assert first_message.status_code == 201
    assert second_message.status_code == 201
    assert second_message.json()["id"] == first_message.json()["id"]
    message_conflict = client.post(
        f"/api/v1/chats/{chat_id}/messages",
        headers=message_headers,
        json={"role": "user", "content": "Different content."},
    )
    assert message_conflict.status_code == 409
    assert message_conflict.json()["detail"]["error_code"] == "sync_server_origin_idempotency_conflict"

    dataset_id = sync_service.profile(user_id="user-1").active_dataset_id or ""
    envelopes = sync_service.store.list_envelopes_after(
        dataset_id,
        0,
        domains=["chat.conversation", "chat.message"],
        limit=10,
    )
    assert [(item.domain, item.operation, item.object_id) for item in envelopes] == [
        ("chat.conversation", "upsert", chat_id),
        ("chat.message", "append", first_message.json()["id"]),
    ]


def test_active_sync_chat_completion_persist_flows_are_rejected_before_direct_writes(
    monkeypatch: pytest.MonkeyPatch,
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    character_id = chacha_db.add_character_card({"name": "Assistant"})
    assert character_id is not None
    client = _chat_messages_app(monkeypatch, chacha_db=chacha_db, sync_service=sync_service)
    chat_response = client.post(
        "/api/v1/chats/",
        json={"character_id": character_id, "title": "Persist blocked"},
    )
    assert chat_response.status_code == 201
    chat_id = chat_response.json()["id"]

    completion_response = client.post(
        f"/api/v1/chats/{chat_id}/complete-v2",
        json={"append_user_message": "Hello", "save_to_db": True, "stream": False},
    )
    assert completion_response.status_code == 400
    assert completion_response.json()["detail"]["error_code"] == "sync_v2_chat_completion_persist_not_supported"

    persist_response = client.post(
        f"/api/v1/chats/{chat_id}/completions/persist",
        json={"assistant_content": "Streamed reply."},
    )
    assert persist_response.status_code == 400
    assert persist_response.json()["detail"]["error_code"] == "sync_v2_chat_completion_persist_not_supported"

    dataset_id = sync_service.profile(user_id="user-1").active_dataset_id or ""
    envelopes = sync_service.store.list_envelopes_after(
        dataset_id,
        0,
        domains=["chat.conversation", "chat.message"],
        limit=10,
    )
    assert [(item.domain, item.operation) for item in envelopes] == [
        ("chat.conversation", "upsert")
    ]
    assert chacha_db.get_messages_for_conversation(chat_id) == []


def test_active_sync_message_edit_is_rejected_before_direct_write(
    monkeypatch: pytest.MonkeyPatch,
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    character_id = chacha_db.add_character_card({"name": "Assistant"})
    assert character_id is not None
    client = _chat_messages_app(monkeypatch, chacha_db=chacha_db, sync_service=sync_service)
    chat_response = client.post(
        "/api/v1/chats/",
        json={"character_id": character_id, "title": "Edit blocked"},
    )
    assert chat_response.status_code == 201
    chat_id = chat_response.json()["id"]
    message_response = client.post(
        f"/api/v1/chats/{chat_id}/messages",
        json={"role": "user", "content": "Original."},
    )
    assert message_response.status_code == 201
    message_id = message_response.json()["id"]

    edit_response = client.put(
        f"/api/v1/messages/{message_id}",
        params={"expected_version": message_response.json()["version"]},
        json={"content": "Edited."},
    )

    assert edit_response.status_code == 400
    assert edit_response.json()["detail"]["error_code"] == "sync_v2_message_edit_not_supported"
    assert chacha_db.get_message_by_id(message_id)["content"] == "Original."
