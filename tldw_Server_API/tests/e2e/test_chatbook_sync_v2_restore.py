from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import User, get_request_user
from tldw_Server_API.app.api.v1.endpoints import sync as sync_endpoint
from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
from tldw_Server_API.app.core.Sync.v2.domain_adapters import (
    ChatDomainAdapter,
    NotesDomainAdapter,
    SourceCacheAdapter,
    WorkspacesDomainAdapter,
)
from tldw_Server_API.app.core.Sync.v2.adapters import SyncAdapterRegistry
from tldw_Server_API.app.core.Sync.v2.service import SyncV2Service, SyncV2Settings
from tldw_Server_API.app.core.Sync.v2.store import SyncV2Store


PRIVATE_NOTE_BODY = "Never sync private note body"
PRIVATE_CHAT_BODY = "Never sync private chat body"
PRIVATE_SOURCE_BODY = "Never sync private source cache body"
PRIVATE_DATASET_LABEL = "Never sync private dataset label"


def _clock() -> str:
    return "2026-05-10T12:00:00+00:00"


def _test_user() -> User:
    return User(id="user-1", username="user-1")


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    registry = SyncAdapterRegistry(
        [
            NotesDomainAdapter(),
            ChatDomainAdapter(),
            WorkspacesDomainAdapter(),
            SourceCacheAdapter(),
        ]
    )
    service = SyncV2Service(
        store=SyncV2Store(SyncDatabase(sqlite_path=tmp_path / "sync_restore_e2e.db")),
        adapters=registry,
        clock=_clock,
        id_factory=lambda prefix: f"{prefix}-generated",
        settings=SyncV2Settings(max_batch_size=20, max_pull_page_size=20),
    )
    app = FastAPI()
    app.include_router(sync_endpoint.router, prefix="/api/v1/sync")
    app.dependency_overrides[get_request_user] = _test_user
    app.dependency_overrides[sync_endpoint.get_sync_v2_service] = lambda: service
    return TestClient(app)


def _register_device(client: TestClient, device_id: str, display_name: str) -> None:
    response = client.post(
        "/api/v1/sync/devices/register",
        json={
            "device_id": device_id,
            "display_name": display_name,
            "client_type": "chatbook",
            "client_version": "0.1.0",
            "capabilities": {"domains": ["notes", "chat", "workspaces", "source_cache"]},
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
            "domains": ["notes", "chat", "workspaces", "source_cache"],
            "encryption_policy": "client_private_v1",
            "metadata": {"label": PRIVATE_DATASET_LABEL},
        },
    )
    assert response.status_code == 200


def _envelope(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "client_envelope_id": "env-note-1",
        "dataset_id": "dataset-1",
        "domain": "notes",
        "entity_id": "note-1",
        "operation": "upsert",
        "adapter_version": 1,
        "device_id": "device-a",
        "stable_key": "note:note-1",
        "client_timestamp": "2026-05-10T00:00:00+00:00",
        "routing_metadata": {
            "entity_kind": "note",
            "update_kind": "title_body",
            "content_fields": ["title", "body"],
        },
        "payload_ciphertext": "ciphertext:note-envelope",
        "payload_clear": {
            "status": "active",
            "attachment_id": "attachment-1",
            "availability": "available",
            "size_bytes": 128,
        },
        "payload_hash": "sha256:note-1",
        "payload_size_bytes": 128,
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


def test_chatbook_sync_v2_restore_roundtrip_keeps_private_payload_metadata_only(
    client: TestClient,
) -> None:
    _register_device(client, "device-a", "Laptop A")
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

    note = _envelope()
    chat = _envelope(
        client_envelope_id="env-chat-1",
        domain="chat",
        entity_id="conversation-1:message-1",
        stable_key="chat:conversation-1:message-1",
        routing_metadata={"entity_kind": "chat_message", "message_id": "message-1"},
        payload_ciphertext="ciphertext:chat-envelope",
        payload_clear={"entity_kind": "chat_message", "status": "active"},
        payload_hash="sha256:chat-1",
        payload_size_bytes=96,
    )
    workspace_ref = _envelope(
        client_envelope_id="env-workspace-ref-1",
        domain="workspaces",
        entity_id="workspace-1:source-1",
        operation="link",
        stable_key="workspace:workspace-1:source:source-1",
        routing_metadata={"entity_kind": "workspace_source_ref", "workspace_id": "workspace-1"},
        payload_ciphertext="ciphertext:workspace-ref-envelope",
        payload_clear={
            "entity_kind": "workspace_source_ref",
            "workspace_id": "workspace-1",
            "source_id": "source-1",
            "link_type": "source_ref",
        },
        payload_hash="sha256:workspace-ref-1",
        payload_size_bytes=64,
    )
    source_cache = _envelope(
        client_envelope_id="env-source-cache-1",
        domain="source_cache",
        entity_id="source-1:cache-1",
        stable_key="source-cache:source-1:hash-1",
        routing_metadata={"source_id": "source-1", "content_hash": "sha256:source-content"},
        payload_ciphertext="ciphertext:source-cache-envelope",
        payload_clear={"source_id": "source-1", "payload_hash": "sha256:source-content"},
        payload_hash="sha256:source-cache-1",
        payload_size_bytes=512,
    )

    pushed = _push(client, [note, chat, workspace_ref, source_cache])
    duplicate = _push(client, [note])
    _register_device(client, "device-b", "Laptop B")

    manifest = client.get("/api/v1/sync/restore-manifest", params={"dataset_id": "dataset-1"})
    pulled = client.get(
        "/api/v1/sync/pull",
        params=[
            ("dataset_id", "dataset-1"),
            ("device_id", "device-b"),
            ("cursor", "0"),
            ("domain", "notes"),
            ("domain", "chat"),
            ("domain", "workspaces"),
            ("domain", "source_cache"),
            ("page_size", "20"),
        ],
    )

    assert pushed["rejected"] == []
    assert pushed["conflicts"] == []
    assert [item["client_envelope_id"] for item in pushed["accepted"]] == [
        "env-note-1",
        "env-chat-1",
        "env-workspace-ref-1",
        "env-source-cache-1",
    ]
    assert duplicate["accepted"][0]["server_sequence"] == pushed["accepted"][0]["server_sequence"]
    assert manifest.status_code == 200
    assert pulled.status_code == 200

    manifest_body = manifest.json()
    dataset = manifest_body["datasets"][0]
    rendered_server_metadata = f"{manifest_body}\n{recovery.json()}"
    assert dataset["key_recovery_available"] is True
    assert dataset["approximate_counts"] == {
        "chat": 1,
        "notes": 1,
        "source_cache": 1,
        "workspaces": 1,
    }
    assert dataset["attachment_availability"] == {"available": 1}
    assert dataset["attachment_size_classes"] == {"small": 1}
    for private_marker in [
        PRIVATE_NOTE_BODY,
        PRIVATE_CHAT_BODY,
        PRIVATE_SOURCE_BODY,
        PRIVATE_DATASET_LABEL,
        "wrapped:opaque-dataset-key",
    ]:
        assert private_marker not in rendered_server_metadata

    pulled_ids = [item["client_envelope_id"] for item in pulled.json()["envelopes"]]
    assert pulled_ids == [
        "env-note-1",
        "env-chat-1",
        "env-workspace-ref-1",
        "env-source-cache-1",
    ]
    assert PRIVATE_NOTE_BODY not in str(pulled.json())
    assert PRIVATE_CHAT_BODY not in str(pulled.json())
    assert PRIVATE_SOURCE_BODY not in str(pulled.json())
