from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import User, get_request_user
from tldw_Server_API.app.api.v1.endpoints import sync as sync_endpoint
from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
from tldw_Server_API.app.core.Sync.v2.factory import default_sync_v2_registry
from tldw_Server_API.app.core.Sync.v2.models import M1_SYNC_DOMAINS
from tldw_Server_API.app.core.Sync.v2.security import server_trusted_encryption_status_from_config
from tldw_Server_API.app.core.Sync.v2.service import SyncV2Service, SyncV2Settings
from tldw_Server_API.app.core.Sync.v2.store import SyncV2Store

PRIVATE_NOTE_BODY = "Never expose this note body in restore metadata"
PRIVATE_CHAT_BODY = "Never expose this chat body in restore metadata"


def _clock() -> str:
    return "2026-05-10T12:00:00+00:00"


def _test_user() -> User:
    return User(id="user-1", username="user-1")


def _ready_encryption():
    return server_trusted_encryption_status_from_config(
        mode="managed_storage",
        server_trusted_enabled=True,
        auth_mode="multi_user",
    )


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    default_sync_v2_registry.cache_clear()
    service = SyncV2Service(
        store=SyncV2Store(SyncDatabase(sqlite_path=tmp_path / "sync_restore_e2e.db")),
        adapters=default_sync_v2_registry(),
        clock=_clock,
        id_factory=lambda prefix: f"{prefix}-generated",
        settings=SyncV2Settings(
            max_batch_size=20,
            max_pull_page_size=20,
            restore_manifest_scan_limit=100,
            server_trusted_encryption=_ready_encryption(),
        ),
    )
    app = FastAPI()
    app.include_router(sync_endpoint.router, prefix="/api/v1/sync")
    app.dependency_overrides[get_request_user] = _test_user
    app.dependency_overrides[sync_endpoint.get_sync_v2_service] = lambda: service
    if hasattr(sync_endpoint, "get_sync_v2_profile_service"):
        app.dependency_overrides[sync_endpoint.get_sync_v2_profile_service] = lambda: service
    return TestClient(app)


def _register_device(client: TestClient, device_id: str, display_name: str) -> None:
    response = client.post(
        "/api/v1/sync/devices/register",
        json={
            "device_id": device_id,
            "display_name": display_name,
            "client_type": "chatbook",
            "client_version": "0.1.0",
            "capabilities": {"domains": list(M1_SYNC_DOMAINS)},
        },
    )
    assert response.status_code == 200


def _enroll_dataset(client: TestClient) -> None:
    response = client.post(
        "/api/v1/sync/datasets/enroll",
        json={
            "dataset_id": "dataset-1",
            "device_id": "device-a",
            "scope_type": "personal",
            "domains": list(M1_SYNC_DOMAINS),
            "encryption_policy": "server_trusted_v1",
            "metadata": {"label": "Chatbook personal profile"},
        },
    )
    assert response.status_code == 200


def _note_envelope(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "client_envelope_id": "env-note-1",
        "dataset_id": "dataset-1",
        "domain": "notes.note",
        "object_id": "note-1",
        "operation": "upsert",
        "device_id": "device-a",
        "client_sequence": 1,
        "object_revision": 1,
        "stable_key": "note:note-1",
        "created_at_client": "2026-05-10T00:00:00+00:00",
        "routing_metadata": {
            "entity_kind": "note",
            "update_kind": "title_body",
            "content_fields": ["title", "content"],
        },
        "payload": {"title": "Private note", "content": PRIVATE_NOTE_BODY},
        "payload_hash": "sha256:note-1",
        "payload_size_bytes": 128,
        "encryption_metadata": {"policy": "server_trusted_v1"},
    }
    payload.update(overrides)
    return payload


def _conversation_envelope(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "client_envelope_id": "env-conversation-1",
        "dataset_id": "dataset-1",
        "domain": "chat.conversation",
        "object_id": "conversation-1",
        "operation": "upsert",
        "device_id": "device-a",
        "client_sequence": 2,
        "object_revision": 1,
        "stable_key": "chat:conversation-1",
        "created_at_client": "2026-05-10T00:01:00+00:00",
        "payload": {"title": "Private chat", "character_id": "character-1"},
        "payload_hash": "sha256:conversation-1",
        "payload_size_bytes": 96,
        "encryption_metadata": {"policy": "server_trusted_v1"},
    }
    payload.update(overrides)
    return payload


def _message_envelope(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "client_envelope_id": "env-message-1",
        "dataset_id": "dataset-1",
        "domain": "chat.message",
        "object_id": "message-1",
        "parent_id": "conversation-1",
        "operation": "append",
        "device_id": "device-a",
        "client_sequence": 3,
        "stable_key": "chat:message-1",
        "created_at_client": "2026-05-10T00:02:00+00:00",
        "payload": {"conversation_id": "conversation-1", "role": "user", "content": PRIVATE_CHAT_BODY},
        "payload_hash": "sha256:message-1",
        "payload_size_bytes": 160,
        "encryption_metadata": {"policy": "server_trusted_v1"},
    }
    payload.update(overrides)
    return payload


def _attachment_ref_envelope(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "client_envelope_id": "env-attachment-ref-1",
        "dataset_id": "dataset-1",
        "domain": "attachment.ref",
        "object_id": "attachment-1",
        "operation": "upsert",
        "device_id": "device-a",
        "client_sequence": 4,
        "stable_key": "attachment:attachment-1",
        "created_at_client": "2026-05-10T00:03:00+00:00",
        "payload": {
            "attachment_id": "attachment-1",
            "parent_domain": "notes.note",
            "parent_object_id": "note-1",
            "content_type": "image/png",
            "size_bytes": 512,
            "payload_hash": "sha256:attachment-1",
            "availability": "client_local",
        },
        "payload_hash": "sha256:attachment-1",
        "payload_size_bytes": 96,
        "encryption_metadata": {"policy": "server_trusted_v1"},
    }
    payload.update(overrides)
    return payload


def _push(client: TestClient, envelopes: list[dict[str, Any]]) -> dict[str, Any]:
    response = client.post(
        "/api/v1/sync/push",
        json={"dataset_id": "dataset-1", "device_id": "device-a", "envelopes": envelopes},
    )
    assert response.status_code == 200
    return response.json()


def test_chatbook_sync_v2_restore_preview_and_pull_roundtrip(
    client: TestClient,
) -> None:
    _register_device(client, "device-a", "Laptop A")
    _register_device(client, "device-b", "Laptop B")
    _enroll_dataset(client)
    recovery = client.post(
        "/api/v1/sync/keys/recovery-bundle",
        json={
            "dataset_id": "dataset-1",
            "device_id": "device-a",
            "key_purpose": "dataset_recovery",
            "wrapped_key_blob": "wrapped:opaque-dataset-key",
            "kdf_metadata": {"algorithm": "scrypt", "salt": "opaque-salt"},
            "recovery_hint": "laptop a",
        },
    )
    assert recovery.status_code == 200

    note = _note_envelope()
    conversation = _conversation_envelope()
    message = _message_envelope()
    attachment_ref = _attachment_ref_envelope()

    pushed = _push(client, [note, conversation, message, attachment_ref])
    duplicate = _push(client, [note])

    manifest = client.get("/api/v1/sync/restore-manifest", params={"dataset_id": "dataset-1"})
    preview = client.post(
        "/api/v1/sync/restore/preview",
        json={
            "dataset_ids": ["dataset-1"],
            "local_inventory": [
                {
                    "domain": "notes.note",
                    "object_id": "note-1",
                    "object_revision": 1,
                    "object_hash": "sha256:note-1",
                    "deleted": False,
                }
            ],
        },
    )
    pulled = client.get(
        "/api/v1/sync/pull",
        params=[
            ("dataset_id", "dataset-1"),
            ("device_id", "device-b"),
            ("cursor", "0"),
            ("domain", "notes.note"),
            ("domain", "chat.conversation"),
            ("domain", "chat.message"),
            ("domain", "attachment.ref"),
            ("page_size", "20"),
            ("include_own_changes", "true"),
        ],
    )

    assert pushed["rejected"] == []
    assert pushed["conflicts"] == []
    assert [item["client_envelope_id"] for item in pushed["accepted"]] == [
        "env-note-1",
        "env-conversation-1",
        "env-message-1",
        "env-attachment-ref-1",
    ]
    assert duplicate["accepted"][0]["server_cursor"] == pushed["accepted"][0]["server_cursor"]
    assert manifest.status_code == 200
    assert preview.status_code == 200
    assert pulled.status_code == 200

    manifest_body = manifest.json()
    preview_body = preview.json()
    dataset = manifest_body["datasets"][0]
    assert dataset["key_recovery_available"] is True
    assert dataset["approximate_counts"] == {
        "attachment.ref": 1,
        "chat.conversation": 1,
        "chat.message": 1,
        "notes.note": 1,
    }

    assert preview_body["total_counts"] == dataset["approximate_counts"]
    assert preview_body["key_status"] == {"dataset-1": {"key_recovery_available": True}}
    assert [(item["domain"], item["object_id"], item["action"]) for item in preview_body["safe_applies"]] == [
        ("notes.note", "note-1", "noop"),
        ("chat.conversation", "conversation-1", "apply"),
        ("chat.message", "message-1", "append"),
    ]
    assert [item["domain"] for item in preview_body["envelope_ranges"]] == [
        "notes.note",
        "chat.conversation",
        "chat.message",
        "attachment.ref",
    ]
    assert [item["attachment_id"] for item in preview_body["attachment_refs"]] == ["attachment-1"]
    assert [item["attachment_id"] for item in preview_body["missing_blobs"]] == ["attachment-1"]

    rendered_restore_metadata = f"{manifest_body}\n{preview_body}"
    for private_marker in [PRIVATE_NOTE_BODY, PRIVATE_CHAT_BODY, "wrapped:opaque-dataset-key"]:
        assert private_marker not in rendered_restore_metadata

    pulled_ids = [item["client_envelope_id"] for item in pulled.json()["envelopes"]]
    assert pulled_ids == [
        "env-note-1",
        "env-conversation-1",
        "env-message-1",
        "env-attachment-ref-1",
    ]
