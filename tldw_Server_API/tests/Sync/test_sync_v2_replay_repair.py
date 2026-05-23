from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import User, get_request_user
from tldw_Server_API.app.api.v1.endpoints import sync as sync_endpoint
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
from tldw_Server_API.app.core.Sync.v2.adapters import StaticSyncAdapter, SyncAdapterRegistry
from tldw_Server_API.app.core.Sync.v2.materializers.chat import (
    ChatConversationMaterializer,
    ChatMessageMaterializer,
)
from tldw_Server_API.app.core.Sync.v2.materializers.notes import NotesMaterializer
from tldw_Server_API.app.core.Sync.v2.models import M1_SYNC_DOMAINS, SyncEnvelopeCreate
from tldw_Server_API.app.core.Sync.v2.security import server_trusted_encryption_status_from_config
from tldw_Server_API.app.core.Sync.v2.service import SyncV2Service, SyncV2Settings
from tldw_Server_API.app.core.Sync.v2.store import SyncV2Store


def _clock() -> str:
    return "2026-05-23T18:12:00+00:00"


def _ready_encryption():
    return server_trusted_encryption_status_from_config(
        mode="managed_storage",
        server_trusted_enabled=True,
        auth_mode="multi_user",
    )


def _test_user() -> User:
    return User(id="user-1", username="user-1")


@pytest.fixture()
def chacha_db(tmp_path: Path) -> CharactersRAGDB:
    return CharactersRAGDB(
        db_path=str(tmp_path / "ChaChaNotes.db"),
        client_id="server-user-1",
    )


@pytest.fixture()
def sync_store(tmp_path: Path) -> SyncV2Store:
    return SyncV2Store(SyncDatabase(sqlite_path=tmp_path / "Sync_v2.db"))


@pytest.fixture()
def log_service(sync_store: SyncV2Store) -> SyncV2Service:
    service = _service(sync_store, materializers={})
    _register_and_enroll(service)
    return service


@pytest.fixture()
def repair_service(sync_store: SyncV2Store, chacha_db: CharactersRAGDB) -> SyncV2Service:
    return _service(
        sync_store,
        materializers={
            "notes.note": NotesMaterializer(chacha_db),
            "chat.conversation": ChatConversationMaterializer(chacha_db),
            "chat.message": ChatMessageMaterializer(chacha_db),
        },
    )


@pytest.fixture()
def repair_client(repair_service: SyncV2Service) -> TestClient:
    app = FastAPI()
    app.include_router(sync_endpoint.router, prefix="/api/v1/sync")
    app.dependency_overrides[get_request_user] = _test_user
    app.dependency_overrides[sync_endpoint.get_sync_v2_service] = lambda: repair_service
    if hasattr(sync_endpoint, "get_sync_v2_profile_service"):
        app.dependency_overrides[sync_endpoint.get_sync_v2_profile_service] = lambda: repair_service
    return TestClient(app)


def _registry() -> SyncAdapterRegistry:
    return SyncAdapterRegistry(
        [StaticSyncAdapter(domain=domain, supported_adapter_versions={1}) for domain in M1_SYNC_DOMAINS]
    )


def _service(sync_store: SyncV2Store, *, materializers: dict[str, Any]) -> SyncV2Service:
    return SyncV2Service(
        store=sync_store,
        adapters=_registry(),
        materializers=materializers,
        clock=_clock,
        id_factory=lambda prefix: f"{prefix}-generated",
        settings=SyncV2Settings(
            server_trusted_encryption=_ready_encryption(),
            restore_manifest_scan_limit=100,
        ),
    )


def _register_and_enroll(service: SyncV2Service) -> None:
    service.register_device(
        user_id="user-1",
        display_name="Laptop",
        client_type="chatbook",
        device_id="device-1",
    )
    service.enroll_dataset(
        user_id="user-1",
        dataset_id="dataset-1",
        domains=list(M1_SYNC_DOMAINS),
    )


def _note_envelope(**overrides: Any) -> SyncEnvelopeCreate:
    payload: dict[str, Any] = {
        "dataset_id": "dataset-1",
        "client_envelope_id": "env-note-create",
        "domain": "notes.note",
        "operation": "upsert",
        "object_id": "note-1",
        "device_id": "device-1",
        "client_sequence": 1,
        "object_revision": 1,
        "payload": {"title": "Repair note", "content": "Rebuilt from log."},
        "payload_hash": "sha256:note-v1",
        "payload_size_bytes": 64,
        "created_at_client": "2026-05-23T18:00:00+00:00",
        "encryption_metadata": {"policy": "server_trusted_v1"},
        "stable_key": "note:note-1",
    }
    payload.update(overrides)
    return SyncEnvelopeCreate(**payload)


def _conversation_envelope(**overrides: Any) -> SyncEnvelopeCreate:
    payload: dict[str, Any] = {
        "dataset_id": "dataset-1",
        "client_envelope_id": "env-conversation-create",
        "domain": "chat.conversation",
        "operation": "upsert",
        "object_id": "conversation-1",
        "device_id": "device-1",
        "client_sequence": 10,
        "object_revision": 1,
        "payload": {
            "title": "Repair chat",
            "assistant_kind": "persona",
            "assistant_id": "sync-assistant",
        },
        "payload_hash": "sha256:conversation-v1",
        "payload_size_bytes": 96,
        "created_at_client": "2026-05-23T18:01:00+00:00",
        "encryption_metadata": {"policy": "server_trusted_v1"},
        "stable_key": "chat:conversation-1",
    }
    payload.update(overrides)
    return SyncEnvelopeCreate(**payload)


def _message_envelope(**overrides: Any) -> SyncEnvelopeCreate:
    payload: dict[str, Any] = {
        "dataset_id": "dataset-1",
        "client_envelope_id": "env-message-create",
        "domain": "chat.message",
        "operation": "append",
        "object_id": "message-1",
        "parent_id": "conversation-1",
        "device_id": "device-1",
        "client_sequence": 20,
        "object_revision": 1,
        "payload": {
            "conversation_id": "conversation-1",
            "sender": "user",
            "content": "Replay this message.",
            "timestamp": "2026-05-23T18:02:00+00:00",
        },
        "payload_hash": "sha256:message-v1",
        "payload_size_bytes": 80,
        "created_at_client": "2026-05-23T18:02:00+00:00",
        "encryption_metadata": {"policy": "server_trusted_v1"},
        "stable_key": "chat:message-1",
    }
    payload.update(overrides)
    return SyncEnvelopeCreate(**payload)


def _push(service: SyncV2Service, *envelopes: SyncEnvelopeCreate) -> None:
    result = service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=list(envelopes),
    )
    assert result.rejected == []
    assert result.conflicts == []
    assert [item.client_envelope_id for item in result.accepted] == [
        envelope.client_envelope_id for envelope in envelopes
    ]


def test_repair_rebuilds_note_projection_from_accepted_envelopes(
    log_service: SyncV2Service,
    repair_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    _push(log_service, _note_envelope())
    assert chacha_db.get_note_by_id("note-1") is None

    result = repair_service.repair(
        user_id="user-1",
        dataset_id="dataset-1",
        domains=["notes.note"],
    )

    note = chacha_db.get_note_by_id("note-1")
    assert note is not None
    assert note["title"] == "Repair note"
    assert result.applied_count == 1
    assert result.failed_count == 0
    assert result.domain_results[0].domain == "notes.note"
    assert result.domain_results[0].applied_count == 1


def test_repair_rebuilds_chat_conversation_and_messages(
    log_service: SyncV2Service,
    repair_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    _push(log_service, _conversation_envelope(), _message_envelope())

    result = repair_service.repair(
        user_id="user-1",
        dataset_id="dataset-1",
        domains=["chat.conversation", "chat.message"],
    )

    conversation = chacha_db.get_conversation_by_id("conversation-1")
    message = chacha_db.get_message_by_id("message-1")
    assert conversation is not None
    assert conversation["title"] == "Repair chat"
    assert message is not None
    assert message["conversation_id"] == "conversation-1"
    assert message["content"] == "Replay this message."
    assert result.applied_count == 2
    assert [item.domain for item in result.domain_results] == ["chat.conversation", "chat.message"]


def test_repair_retries_failed_apply_after_projection_issue_is_fixed(
    sync_store: SyncV2Store,
    chacha_db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(
        sync_store,
        materializers={"notes.note": NotesMaterializer(chacha_db)},
    )
    _register_and_enroll(service)
    original_upsert = chacha_db.upsert_note_from_sync
    projection_available = False

    def _maybe_fail_projection(*args: Any, **kwargs: Any):
        if not projection_available:
            raise RuntimeError("projection unavailable")
        return original_upsert(*args, **kwargs)

    monkeypatch.setattr(chacha_db, "upsert_note_from_sync", _maybe_fail_projection)
    _push(service, _note_envelope())
    before = service.profile_status(user_id="user-1", dataset_id="dataset-1")
    before_domains = {item.domain: item for item in before.domain_status}
    assert before_domains["notes.note"].failed_apply_count == 1
    assert before_domains["notes.note"].repair_status["status"] == "repair_needed"

    projection_available = True
    result = service.repair(
        user_id="user-1",
        dataset_id="dataset-1",
        domains=["notes.note"],
        failed_only=True,
    )

    note = chacha_db.get_note_by_id("note-1")
    after = service.profile_status(user_id="user-1", dataset_id="dataset-1")
    after_domains = {item.domain: item for item in after.domain_status}
    assert note is not None
    assert result.applied_count == 1
    assert result.failed_count == 0
    assert after_domains["notes.note"].failed_apply_count == 0
    assert after_domains["notes.note"].repair_status["status"] == "healthy"
    assert after_domains["notes.note"].last_apply_result["status"] == "applied"


def test_repair_preserves_tombstones(
    log_service: SyncV2Service,
    repair_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    _push(log_service, _note_envelope())
    _push(
        log_service,
        _note_envelope(
            client_envelope_id="env-note-delete",
            operation="tombstone",
            client_sequence=2,
            object_revision=2,
            payload={"deleted": True},
            payload_hash="sha256:note-deleted",
            base_server_cursor=1,
            base_object_revision=1,
            base_object_hash="sha256:note-v1",
        ),
    )

    result = repair_service.repair(
        user_id="user-1",
        dataset_id="dataset-1",
        domains=["notes.note"],
    )

    assert chacha_db.get_note_by_id("note-1") is None
    deleted = chacha_db.get_note_by_id("note-1", include_deleted=True)
    assert deleted is not None
    assert bool(deleted["deleted"]) is True
    assert result.applied_count == 2


def test_repair_never_replays_conflict_envelopes_as_accepted_changes(
    log_service: SyncV2Service,
    repair_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    _push(log_service, _note_envelope())
    log_service.store.insert_envelope(
        _note_envelope(
            client_envelope_id="env-conflict",
            object_id="note-conflict",
            client_sequence=2,
            payload={"title": "Conflict note", "content": "Must not project."},
            payload_hash="sha256:conflict",
            status="conflict",
            apply_status="conflict",
        )
    )

    result = repair_service.repair(user_id="user-1", dataset_id="dataset-1")

    assert chacha_db.get_note_by_id("note-1") is not None
    assert chacha_db.get_note_by_id("note-conflict") is None
    assert result.applied_count == 1
    assert result.conflict_count == 0


def test_repair_endpoint_requires_owned_dataset_and_returns_status(
    log_service: SyncV2Service,
    repair_client: TestClient,
    chacha_db: CharactersRAGDB,
) -> None:
    _push(log_service, _note_envelope())

    repaired = repair_client.post(
        "/api/v1/sync/repair",
        json={"dataset_id": "dataset-1", "domains": ["notes.note"]},
    )
    forbidden = repair_client.post(
        "/api/v1/sync/repair",
        json={"dataset_id": "dataset-other", "domains": ["notes.note"]},
    )

    assert repaired.status_code == 200
    assert repaired.json()["applied_count"] == 1
    assert repaired.json()["domain_results"][0]["domain"] == "notes.note"
    assert chacha_db.get_note_by_id("note-1") is not None
    assert forbidden.status_code == 404
    assert forbidden.json()["detail"]["error_code"] == "sync_resource_not_found"
