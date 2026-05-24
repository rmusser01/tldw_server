from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import User, get_request_user
from tldw_Server_API.app.api.v1.endpoints import notes as notes_endpoint
from tldw_Server_API.app.api.v1.endpoints import sync as sync_endpoint
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
from tldw_Server_API.app.core.Sync.v2.blob_store import LocalSyncBlobStore
from tldw_Server_API.app.core.Sync.v2.factory import default_sync_v2_registry
from tldw_Server_API.app.core.Sync.v2.materializers import (
    AttachmentRefMaterializer,
    ChatConversationMaterializer,
    ChatMessageMaterializer,
    NotesMaterializer,
)
from tldw_Server_API.app.core.Sync.v2.models import M1_SYNC_DOMAINS, WORKSPACE_SYNC_DOMAINS
from tldw_Server_API.app.core.Sync.v2.security import server_trusted_encryption_status_from_config
from tldw_Server_API.app.core.Sync.v2.server_origin import SERVER_ORIGIN_DEVICE_ID
from tldw_Server_API.app.core.Sync.v2.service import SyncV2Service, SyncV2Settings
from tldw_Server_API.app.core.Sync.v2.store import SyncV2Store

PRIVATE_NOTE_BODY = "Never expose this note body in restore metadata"
PRIVATE_CHAT_BODY = "Never expose this chat body in restore metadata"


def _clock() -> str:
    return "2026-05-10T12:00:00+00:00"


def _sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _test_user() -> User:
    return User(id="user-1", username="user-1")


def _user(user_id: str) -> User:
    return User(id=user_id, username=user_id)


def _ready_encryption():
    return server_trusted_encryption_status_from_config(
        mode="managed_storage",
        server_trusted_enabled=True,
        auth_mode="multi_user",
    )


@dataclass(slots=True)
class SyncE2EHarness:
    client: TestClient
    service: SyncV2Service
    chacha_db: CharactersRAGDB
    workspace_permissions: set[tuple[str, str, str]]


class _NoopRateLimiter:
    async def check_user_rate_limit(self, user_id: int, endpoint: str, role: str = "user"):
        return True, {}


@pytest.fixture()
def harness(tmp_path: Path) -> SyncE2EHarness:
    default_sync_v2_registry.cache_clear()
    chacha_db = CharactersRAGDB(
        db_path=str(tmp_path / "ChaChaNotes.db"),
        client_id="server-user-1",
    )
    workspace_permissions = {("user-1", "workspace-1", "sync")}
    service = SyncV2Service(
        store=SyncV2Store(SyncDatabase(sqlite_path=tmp_path / "sync_restore_e2e.db")),
        adapters=default_sync_v2_registry(),
        materializers={
            "attachment.ref": AttachmentRefMaterializer(),
            "chat.conversation": ChatConversationMaterializer(chacha_db),
            "chat.message": ChatMessageMaterializer(chacha_db),
            "notes.note": NotesMaterializer(chacha_db),
        },
        clock=_clock,
        id_factory=lambda prefix: f"{prefix}-generated",
        workspace_access_checker=(
            lambda user_id, workspace_id, permission: (
                user_id,
                workspace_id,
                permission,
            )
            in workspace_permissions
        ),
        settings=SyncV2Settings(
            max_batch_size=20,
            max_pull_page_size=20,
            restore_manifest_scan_limit=100,
            server_trusted_encryption=_ready_encryption(),
        ),
    )
    return SyncE2EHarness(
        client=_sync_client(service, _test_user()),
        service=service,
        chacha_db=chacha_db,
        workspace_permissions=workspace_permissions,
    )


@pytest.fixture()
def m2_harness(tmp_path: Path) -> SyncE2EHarness:
    default_sync_v2_registry.cache_clear()
    chacha_db = CharactersRAGDB(
        db_path=str(tmp_path / "ChaChaNotes.db"),
        client_id="server-user-1",
    )
    workspace_permissions = {("user-1", "workspace-1", "sync")}
    service = SyncV2Service(
        store=SyncV2Store(SyncDatabase(sqlite_path=tmp_path / "sync_restore_m2_e2e.db")),
        adapters=default_sync_v2_registry(),
        materializers={
            "attachment.ref": AttachmentRefMaterializer(),
            "chat.conversation": ChatConversationMaterializer(chacha_db),
            "chat.message": ChatMessageMaterializer(chacha_db),
            "notes.note": NotesMaterializer(chacha_db),
        },
        clock=_clock,
        id_factory=lambda prefix: f"{prefix}-generated",
        blob_store=LocalSyncBlobStore(tmp_path / "sync_blobs"),
        workspace_access_checker=(
            lambda user_id, workspace_id, permission: (
                user_id,
                workspace_id,
                permission,
            )
            in workspace_permissions
        ),
        settings=SyncV2Settings(
            max_batch_size=20,
            max_pull_page_size=20,
            restore_manifest_scan_limit=100,
            supports_attachments=True,
            max_attachment_bytes=4096,
            max_blob_bytes=4096,
            max_chunk_bytes=1024,
            user_blob_quota_bytes=8192,
            server_trusted_encryption=_ready_encryption(),
        ),
    )
    return SyncE2EHarness(
        client=_sync_client(service, _test_user()),
        service=service,
        chacha_db=chacha_db,
        workspace_permissions=workspace_permissions,
    )


@pytest.fixture()
def client(harness: SyncE2EHarness) -> TestClient:
    return harness.client


def _sync_client(service: SyncV2Service, user: User) -> TestClient:
    app = FastAPI()
    app.include_router(sync_endpoint.router, prefix="/api/v1/sync")
    app.dependency_overrides[get_request_user] = lambda: user
    app.dependency_overrides[sync_endpoint.get_sync_v2_service] = lambda: service
    if hasattr(sync_endpoint, "get_sync_v2_profile_service"):
        app.dependency_overrides[sync_endpoint.get_sync_v2_profile_service] = lambda: service
    return TestClient(app)


def _sync_and_notes_client(
    service: SyncV2Service,
    chacha_db: CharactersRAGDB,
    user: User,
    monkeypatch: pytest.MonkeyPatch,
) -> TestClient:
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_TEST_API_KEY", "THIS-IS-A-SECURE-KEY-123-FAKE-KEY")
    app = FastAPI()
    app.include_router(sync_endpoint.router, prefix="/api/v1/sync")
    app.include_router(notes_endpoint.router, prefix="/api/v1/notes")

    async def _db_override():
        return chacha_db

    async def _user_override():
        return user

    app.dependency_overrides[get_request_user] = lambda: user
    app.dependency_overrides[sync_endpoint.get_sync_v2_service] = lambda: service
    if hasattr(sync_endpoint, "get_sync_v2_profile_service"):
        app.dependency_overrides[sync_endpoint.get_sync_v2_profile_service] = lambda: service
    app.dependency_overrides[notes_endpoint.get_chacha_db_for_user] = _db_override
    app.dependency_overrides[notes_endpoint.get_request_user] = _user_override
    app.dependency_overrides[notes_endpoint.get_rate_limiter_dep] = lambda: _NoopRateLimiter()
    monkeypatch.setattr(
        notes_endpoint,
        "get_active_server_origin_sync_service_for_user",
        lambda user_id: service,
        raising=False,
    )
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
        "payload": {
            "title": "Private chat",
            "assistant_kind": "persona",
            "assistant_id": "sync-assistant",
        },
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
        "payload": {
            "conversation_id": "conversation-1",
            "role": "user",
            "sender": "user",
            "content": PRIVATE_CHAT_BODY,
        },
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


def _workspace_envelope(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "client_envelope_id": "env-workspace-1",
        "dataset_id": "workspace-dataset",
        "domain": "workspaces.workspace",
        "object_id": "workspace-1",
        "operation": "upsert",
        "device_id": "device-a",
        "client_sequence": 1,
        "object_revision": 1,
        "stable_key": "workspace:workspace-1",
        "created_at_client": "2026-05-10T00:04:00+00:00",
        "routing_metadata": {"entity_kind": "workspace", "workspace_id": "workspace-1"},
        "payload": {
            "entity_kind": "workspace",
            "workspace_id": "workspace-1",
            "name": "Shared research",
        },
        "payload_hash": "sha256:workspace-1",
        "payload_size_bytes": 96,
        "encryption_metadata": {"policy": "server_trusted_v1"},
    }
    payload.update(overrides)
    return payload


def _workspace_source_ref_envelope(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "client_envelope_id": "env-workspace-source-1",
        "dataset_id": "workspace-dataset",
        "domain": "workspaces.source_ref",
        "object_id": "workspace-1:source-1",
        "operation": "upsert",
        "device_id": "device-a",
        "client_sequence": 2,
        "object_revision": 1,
        "stable_key": "workspace_source:workspace-1:source-1",
        "created_at_client": "2026-05-10T00:05:00+00:00",
        "routing_metadata": {
            "entity_kind": "workspace_source_ref",
            "workspace_id": "workspace-1",
            "source_id": "source-1",
        },
        "payload": {
            "entity_kind": "workspace_source_ref",
            "workspace_id": "workspace-1",
            "source_id": "source-1",
            "title": "Reference source",
        },
        "payload_hash": "sha256:workspace-source-1",
        "payload_size_bytes": 128,
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


def _push_dataset(
    client: TestClient,
    *,
    dataset_id: str,
    device_id: str,
    envelopes: list[dict[str, Any]],
) -> dict[str, Any]:
    response = client.post(
        "/api/v1/sync/push",
        json={"dataset_id": dataset_id, "device_id": device_id, "envelopes": envelopes},
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
    metadata_only_preview = client.post(
        "/api/v1/sync/restore/preview",
        json={
            "dataset_ids": ["dataset-1"],
            "selected_attachment_ids": ["attachment-1"],
            "metadata_only": True,
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
    assert metadata_only_preview.status_code == 200
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
    metadata_only_body = metadata_only_preview.json()
    assert metadata_only_body["restore_status"] == "metadata_ready"
    assert metadata_only_body["blob_details"][0]["required_for_restore"] is False

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


def test_chatbook_sync_v2_m2_uploaded_blob_restore_completeness_roundtrip(
    m2_harness: SyncE2EHarness,
) -> None:
    client = m2_harness.client
    payload = b"uploaded attachment payload"
    payload_hash = _sha256(payload)
    attachment_ref = _attachment_ref_envelope(
        payload={
            "attachment_id": "attachment-1",
            "parent_domain": "notes.note",
            "parent_object_id": "note-1",
            "content_type": "application/octet-stream",
            "size_bytes": len(payload),
            "payload_hash": payload_hash,
            "availability": "client_local",
        },
        payload_hash=payload_hash,
        payload_size_bytes=len(payload),
    )

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
        },
    )
    pushed = _push(client, [_note_envelope(), attachment_ref])
    blob_incomplete = client.post(
        "/api/v1/sync/restore/preview",
        json={"dataset_ids": ["dataset-1"], "selected_attachment_ids": ["attachment-1"]},
    )

    create_upload = client.post(
        "/api/v1/sync/blob-uploads",
        json={
            "dataset_id": "dataset-1",
            "device_id": "device-a",
            "domain": "notes.note",
            "object_id": "note-1",
            "attachment_id": "attachment-1",
            "content_type": "application/octet-stream",
            "size_bytes": len(payload),
            "payload_hash": payload_hash,
            "chunk_size": len(payload),
            "chunk_count": 1,
            "idempotency_key": "upload-attachment-1",
        },
    )
    upload_id = create_upload.json()["upload_id"]
    chunk = client.put(
        f"/api/v1/sync/blob-uploads/{upload_id}/chunks/0",
        params={
            "dataset_id": "dataset-1",
            "offset_bytes": 0,
            "chunk_hash": payload_hash,
        },
        content=payload,
        headers={"content-type": "application/octet-stream"},
    )
    complete = client.post(
        f"/api/v1/sync/blob-uploads/{upload_id}/complete",
        params={"dataset_id": "dataset-1"},
    )
    content_complete = client.post(
        "/api/v1/sync/restore/preview",
        json={"dataset_ids": ["dataset-1"], "selected_attachment_ids": ["attachment-1"]},
    )
    verified_complete = client.post(
        "/api/v1/sync/restore/preview",
        json={
            "dataset_ids": ["dataset-1"],
            "selected_attachment_ids": ["attachment-1"],
            "attachment_availability": {"attachment-1": "verified"},
        },
    )
    download_manifest = client.get(
        "/api/v1/sync/attachments/attachment-1/manifest",
        params={"dataset_id": "dataset-1"},
    )
    downloaded = client.get(
        "/api/v1/sync/attachments/attachment-1",
        params={"dataset_id": "dataset-1"},
    )

    assert recovery.status_code == 200
    assert pushed["rejected"] == []
    assert pushed["conflicts"] == []
    assert blob_incomplete.status_code == 200
    blob_incomplete_body = blob_incomplete.json()
    assert blob_incomplete_body["restore_status"] == "blob_incomplete"
    assert blob_incomplete_body["blob_details"][0]["server_availability"] == "metadata_only"
    assert [item["attachment_id"] for item in blob_incomplete_body["missing_blobs"]] == [
        "attachment-1"
    ]

    assert create_upload.status_code == 200
    assert chunk.status_code == 200
    assert complete.status_code == 200
    assert complete.json()["status"] == "available"
    assert complete.json()["payload_hash"] == payload_hash

    assert content_complete.status_code == 200
    content_complete_body = content_complete.json()
    assert content_complete_body["restore_status"] == "content_complete"
    assert content_complete_body["blob_details"][0]["server_availability"] == "available"
    assert content_complete_body["domain_details"][0]["domain"] == "notes.note"
    assert any(
        detail["domain"] == "attachment.ref" and detail["status"] == "content_complete"
        for detail in content_complete_body["domain_details"]
    )

    assert verified_complete.status_code == 200
    assert verified_complete.json()["restore_status"] == "verified_complete"
    assert download_manifest.status_code == 200
    assert download_manifest.json()["payload_hash"] == payload_hash
    assert downloaded.status_code == 200
    assert downloaded.content == payload


def test_chatbook_sync_v2_two_device_pagination_echo_and_cross_user_isolation(
    harness: SyncE2EHarness,
) -> None:
    client = harness.client
    _register_device(client, "device-a", "Laptop A")
    _register_device(client, "device-b", "Laptop B")
    _enroll_dataset(client)
    pushed = _push(
        client,
        [
            _note_envelope(),
            _conversation_envelope(),
            _message_envelope(),
            _attachment_ref_envelope(),
        ],
    )
    assert pushed["conflicts"] == []

    same_device_echo = client.get(
        "/api/v1/sync/pull",
        params={"dataset_id": "dataset-1", "device_id": "device-a", "cursor": "0", "page_size": "20"},
    )
    first_page = client.get(
        "/api/v1/sync/pull",
        params={"dataset_id": "dataset-1", "device_id": "device-b", "cursor": "0", "page_size": "2"},
    )
    second_page = client.get(
        "/api/v1/sync/pull",
        params={
            "dataset_id": "dataset-1",
            "device_id": "device-b",
            "cursor": first_page.json()["next_cursor"],
            "page_size": "2",
        },
    )

    assert same_device_echo.status_code == 200
    assert same_device_echo.json()["envelopes"] == []
    assert first_page.status_code == 200
    assert first_page.json()["has_more"] is True
    assert [item["client_envelope_id"] for item in first_page.json()["envelopes"]] == [
        "env-note-1",
        "env-conversation-1",
    ]
    assert second_page.status_code == 200
    assert second_page.json()["has_more"] is False
    assert [item["client_envelope_id"] for item in second_page.json()["envelopes"]] == [
        "env-message-1",
        "env-attachment-ref-1",
    ]

    stale_update = _note_envelope(
        client_envelope_id="env-note-stale",
        client_sequence=5,
        object_revision=2,
        payload={"title": "Stale", "content": "Should conflict."},
        payload_hash="sha256:note-stale",
        base_server_cursor=999,
        base_object_revision=1,
        base_object_hash="sha256:note-1",
    )
    conflict_push = _push(client, [stale_update])
    assert conflict_push["accepted"] == []
    assert [item["client_envelope_id"] for item in conflict_push["conflicts"]] == ["env-note-stale"]
    conflict_id = conflict_push["conflicts"][0]["conflict_id"]

    other_client = _sync_client(harness.service, _user("user-2"))
    _register_device(other_client, "device-c", "Laptop C")
    forbidden_pull = other_client.get(
        "/api/v1/sync/pull",
        params={"dataset_id": "dataset-1", "device_id": "device-c", "cursor": "0"},
    )
    hidden_manifest = other_client.get("/api/v1/sync/restore-manifest", params={"dataset_id": "dataset-1"})
    forbidden_preview = other_client.post(
        "/api/v1/sync/restore/preview",
        json={"dataset_ids": ["dataset-1"]},
    )
    forbidden_conflicts = other_client.get("/api/v1/sync/conflicts", params={"dataset_id": "dataset-1"})
    rejected_resolution = other_client.post(
        "/api/v1/sync/conflicts/resolve",
        json={
            "dataset_id": "dataset-1",
            "device_id": "device-c",
            "resolutions": [{"conflict_id": conflict_id, "action": "skip"}],
        },
    )
    forbidden_attachment = other_client.post(
        "/api/v1/sync/attachments",
        json={
            "dataset_id": "dataset-1",
            "domain": "notes.note",
            "object_id": "note-1",
            "attachment_id": "attachment-1",
            "content_type": "image/png",
            "size_bytes": 12,
            "payload_ciphertext": "ciphertext",
            "payload_hash": "sha256:attachment-1",
        },
    )

    assert forbidden_pull.status_code == 404
    assert hidden_manifest.status_code == 404
    assert forbidden_preview.status_code == 404
    assert forbidden_conflicts.status_code == 404
    assert rejected_resolution.status_code == 200
    assert rejected_resolution.json()["resolved"] == []
    assert rejected_resolution.json()["rejected"][0]["conflict_id"] == conflict_id
    assert forbidden_attachment.status_code == 501


def test_chatbook_sync_v2_restore_conflicts_tombstones_and_message_stable_ids(
    harness: SyncE2EHarness,
) -> None:
    client = harness.client
    _register_device(client, "device-a", "Laptop A")
    _register_device(client, "device-b", "Laptop B")
    _enroll_dataset(client)
    _push(
        client,
        [
            _note_envelope(),
            _note_envelope(
                client_envelope_id="env-note-conflict-base",
                object_id="note-conflict",
                stable_key="note:note-conflict",
                client_sequence=2,
                payload={"title": "Server conflict note", "content": "Server version."},
                payload_hash="sha256:note-conflict-server",
            ),
            _conversation_envelope(client_sequence=3),
            _conversation_envelope(
                client_envelope_id="env-conversation-conflict-base",
                object_id="conversation-conflict",
                stable_key="chat:conversation-conflict",
                client_sequence=4,
                payload={
                    "title": "Server conflict chat",
                    "assistant_kind": "persona",
                    "assistant_id": "sync-assistant",
                },
                payload_hash="sha256:conversation-conflict-server",
            ),
            _message_envelope(client_sequence=5),
            _attachment_ref_envelope(client_sequence=6),
        ],
    )
    duplicate_message = _push(
        client,
        [
            _message_envelope(
                client_envelope_id="env-message-duplicate",
                client_sequence=7,
            )
        ],
    )
    divergent_message = _push(
        client,
        [
            _message_envelope(
                client_envelope_id="env-message-divergent",
                client_sequence=8,
                payload={
                    "conversation_id": "conversation-1",
                    "role": "user",
                    "sender": "user",
                    "content": "Different body for same stable ID.",
                },
                payload_hash="sha256:message-divergent",
            )
        ],
    )
    assert duplicate_message["conflicts"] == []
    assert divergent_message["accepted"] == []
    assert divergent_message["conflicts"][0]["domain"] == "chat.message"

    dataset_id = "dataset-1"
    note_state = harness.service.store.get_object_state(dataset_id, "notes.note", "note-1")
    message_state = harness.service.store.get_object_state(dataset_id, "chat.message", "message-1")
    assert note_state is not None
    assert message_state is not None
    _push(
        client,
        [
            _note_envelope(
                client_envelope_id="env-note-tombstone",
                operation="tombstone",
                client_sequence=9,
                object_revision=note_state.object_revision + 1,
                payload={"deleted": True},
                payload_hash="sha256:note-tombstone",
                base_server_cursor=note_state.latest_server_cursor,
                base_object_revision=note_state.object_revision,
                base_object_hash=note_state.object_hash,
            ),
            _message_envelope(
                client_envelope_id="env-message-tombstone",
                operation="tombstone",
                client_sequence=10,
                object_revision=message_state.object_revision + 1,
                payload={"conversation_id": "conversation-1", "deleted": True},
                payload_hash="sha256:message-tombstone",
                base_server_cursor=message_state.latest_server_cursor,
                base_object_revision=message_state.object_revision,
                base_object_hash=message_state.object_hash,
            ),
        ],
    )

    preview = client.post(
        "/api/v1/sync/restore/preview",
        json={
            "dataset_ids": ["dataset-1"],
            "local_inventory": [
                {
                    "domain": "notes.note",
                    "object_id": "note-conflict",
                    "object_revision": 1,
                    "object_hash": "sha256:local-note-conflict",
                    "deleted": False,
                },
                {
                    "domain": "chat.conversation",
                    "object_id": "conversation-conflict",
                    "object_revision": 1,
                    "object_hash": "sha256:local-conversation-conflict",
                    "deleted": False,
                },
                {
                    "domain": "notes.note",
                    "object_id": "note-1",
                    "object_revision": note_state.object_revision,
                    "object_hash": note_state.object_hash,
                    "deleted": False,
                },
            ],
        },
    )

    assert preview.status_code == 200
    body = preview.json()
    assert {(item["domain"], item["object_id"]) for item in body["object_conflicts"]} == {
        ("notes.note", "note-conflict"),
        ("chat.conversation", "conversation-conflict"),
    }
    assert [(item["domain"], item["object_id"], item["deleted"]) for item in body["tombstones"]] == [
        ("notes.note", "note-1", True),
        ("chat.message", "message-1", True),
    ]
    assert [item["attachment_id"] for item in body["missing_blobs"]] == ["attachment-1"]
    assert harness.chacha_db.get_note_by_id("note-1") is None
    assert harness.chacha_db.get_message_by_id("message-1") is None
    stored_messages = harness.chacha_db.get_messages_for_conversation("conversation-1", include_deleted=True)
    assert {item["id"] for item in stored_messages} == {
        "message-1",
        "message-1__sync_conflict__8",
    }
    assert all(item["deleted"] in (1, True) for item in stored_messages)


def test_chatbook_sync_v2_server_frontend_note_write_pulls_as_server_origin(
    harness: SyncE2EHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _sync_and_notes_client(harness.service, harness.chacha_db, _test_user(), monkeypatch)
    _register_device(client, "device-b", "Laptop B")
    bootstrap = client.post(
        "/api/v1/sync/profile/bootstrap",
        json={
            "client_family": "chatbook",
            "mode": "server_frontend",
            "device_id": "frontend-device",
            "device_name": "Server frontend",
            "requested_domains": list(M1_SYNC_DOMAINS),
        },
    )
    assert bootstrap.status_code == 200
    dataset_id = bootstrap.json()["active_dataset_id"]

    created = client.post(
        "/api/v1/notes/",
        json={
            "id": "note-server-api",
            "title": "Server API note",
            "content": "Created through normal Notes API while sync is active.",
        },
    )
    pulled = client.get(
        "/api/v1/sync/pull",
        params={"dataset_id": dataset_id, "device_id": "device-b", "cursor": "0", "domain": "notes.note"},
    )

    assert created.status_code == 201
    assert harness.chacha_db.get_note_by_id("note-server-api") is not None
    assert pulled.status_code == 200
    envelopes = pulled.json()["envelopes"]
    assert [(item["domain"], item["object_id"], item["device_id"]) for item in envelopes] == [
        ("notes.note", "note-server-api", SERVER_ORIGIN_DEVICE_ID)
    ]


def test_chatbook_sync_v2_m3_multi_device_workspace_key_retention_and_diagnostics_flow(
    harness: SyncE2EHarness,
) -> None:
    client = harness.client
    secret = "wrapped:m3-passphrase-secret"
    salt = "m3-secret-salt"

    _register_device(client, "device-a", "Laptop A")
    _register_device(client, "device-b", "Laptop B")
    _enroll_dataset(client)

    first_note = _push(client, [_note_envelope()])
    first_cursor = first_note["accepted"][0]["server_cursor"]
    second_note = _push(
        client,
        [
            _note_envelope(
                client_envelope_id="env-note-2",
                client_sequence=2,
                object_revision=2,
                payload={"title": "Private note", "content": "Updated private note"},
                payload_hash="sha256:note-2",
                base_server_cursor=first_cursor,
                base_object_revision=1,
                base_object_hash="sha256:note-1",
            )
        ],
    )
    latest_cursor = second_note["accepted"][0]["server_cursor"]

    stale_note = _push(
        client,
        [
            _note_envelope(
                client_envelope_id="env-note-stale-m3",
                client_sequence=3,
                object_revision=3,
                payload={"title": "Stale note", "content": "Should be reviewed."},
                payload_hash="sha256:note-stale-m3",
                base_server_cursor=first_cursor,
                base_object_revision=1,
                base_object_hash="sha256:note-1",
            )
        ],
    )
    assert stale_note["accepted"] == []
    assert stale_note["conflicts"][0]["domain"] == "notes.note"
    conflict_id = stale_note["conflicts"][0]["conflict_id"]

    resolved = client.post(
        "/api/v1/sync/conflicts/resolve",
        json={
            "dataset_id": "dataset-1",
            "device_id": "device-a",
            "resolutions": [{"conflict_id": conflict_id, "action": "skip"}],
        },
    )
    assert resolved.status_code == 200
    assert resolved.json()["resolved"][0]["conflict_id"] == conflict_id
    assert resolved.json()["rejected"] == []

    recovery = client.post(
        "/api/v1/sync/keys/recovery-bundle",
        json={
            "dataset_id": "dataset-1",
            "device_id": "device-a",
            "key_purpose": "dataset_recovery",
            "wrapped_key_blob": "wrapped:m3-current-secret",
            "kdf_metadata": {"algorithm": "scrypt", "salt": "m3-current-secret-salt"},
        },
    )
    assert recovery.status_code == 200
    recovery_key_id = recovery.json()["key_record_id"]

    rotation_preview = client.post(
        "/api/v1/sync/key-rotation/preview",
        json={
            "dataset_id": "dataset-1",
            "target_encryption_policy": "passphrase_wrapped_v1",
            "source_key_record_ids": [recovery_key_id],
        },
    )
    rotation_commit = client.post(
        "/api/v1/sync/key-rotation/commit",
        json={
            "dataset_id": "dataset-1",
            "rotation_id": "rotation-m3-e2e",
            "target_encryption_policy": "passphrase_wrapped_v1",
            "wrapped_key_blob": secret,
            "kdf_metadata": {"algorithm": "scrypt", "salt": salt},
            "source_key_record_ids": [recovery_key_id],
            "wrapped_for": "passphrase",
        },
    )
    assert rotation_preview.status_code == 200
    assert rotation_preview.json()["can_commit"] is True
    assert rotation_commit.status_code == 200
    assert rotation_commit.json()["committed"] is True
    assert secret not in rotation_preview.text
    assert salt not in rotation_preview.text
    assert secret not in rotation_commit.text
    assert salt not in rotation_commit.text

    policy = client.patch(
        "/api/v1/sync/background-policy",
        json={
            "dataset_id": "dataset-1",
            "device_id": "device-b",
            "enabled": True,
            "pending_local_changes": False,
        },
    )
    lease = client.post(
        "/api/v1/sync/background-leases",
        json={"dataset_id": "dataset-1", "device_id": "device-b", "lease_id": "lease-m3"},
    )
    pre_pull_status = client.get(
        "/api/v1/sync/background-status",
        params={"dataset_id": "dataset-1", "device_id": "device-b"},
    )
    assert policy.status_code == 200
    assert lease.status_code == 200
    assert lease.json()["status"] == "acquired"
    assert pre_pull_status.status_code == 200
    notes_status = {
        item["domain"]: item for item in pre_pull_status.json()["domains"]
    }["notes.note"]
    assert notes_status["cursor_lag_count"] >= 2
    assert pre_pull_status.json()["restore_completeness"] == "content_complete"

    pulled = client.get(
        "/api/v1/sync/pull",
        params={
            "dataset_id": "dataset-1",
            "device_id": "device-b",
            "cursor": "0",
            "domain": "notes.note",
            "page_size": "20",
        },
    )
    assert pulled.status_code == 200
    assert [item["client_envelope_id"] for item in pulled.json()["envelopes"]] == [
        "env-note-1",
        "env-note-2",
    ]

    for device_id in ("device-a", "device-b"):
        acked = client.post(
            "/api/v1/sync/device-acknowledgments",
            json={
                "dataset_id": "dataset-1",
                "device_id": device_id,
                "domain_acks": [
                    {
                        "domain": "notes.note",
                        "through_server_sequence": latest_cursor,
                        "applied_at": _clock(),
                    }
                ],
            },
        )
        assert acked.status_code == 200

    dry_run = client.post(
        "/api/v1/sync/retention/dry-run",
        json={
            "dataset_id": "dataset-1",
            "device_id": "device-a",
            "domains": ["notes.note"],
            "audit_mode": False,
            "minimum_envelope_age_seconds": 0,
            "offline_restore_window_seconds": 0,
        },
    )
    compact_without_confirmation = client.post(
        "/api/v1/sync/retention/compact",
        json={
            "dataset_id": "dataset-1",
            "device_id": "device-a",
            "domains": ["notes.note"],
            "confirm": False,
            "minimum_envelope_age_seconds": 0,
            "offline_restore_window_seconds": 0,
        },
    )
    assert dry_run.status_code == 200
    assert dry_run.json()["mutation_performed"] is False
    assert any(
        candidate["candidate_type"] == "envelope_compaction"
        and candidate["object_id"] == "note-1"
        and candidate["blockers"] == []
        for candidate in dry_run.json()["candidates"]
    )
    assert compact_without_confirmation.status_code == 200
    assert compact_without_confirmation.json()["confirmation_required"] is True
    assert compact_without_confirmation.json()["mutation_performed"] is False

    diagnostics = client.get(
        "/api/v1/sync/diagnostics",
        params={"dataset_id": "dataset-1", "device_id": "device-a", "retention_limit": "20"},
    )
    assert diagnostics.status_code == 200
    assert diagnostics.json()["retention"]["dry_run"] is True
    assert diagnostics.json()["key_summary"]["key_record_count"] == 2
    assert diagnostics.json()["key_summary"]["superseded_key_record_count"] == 1
    diagnostics_text = diagnostics.text
    for private_marker in [PRIVATE_NOTE_BODY, secret, salt, "m3-current-secret-salt"]:
        assert private_marker not in diagnostics_text

    workspace = client.post(
        "/api/v1/sync/datasets/enroll",
        json={
            "dataset_id": "workspace-dataset",
            "device_id": "device-a",
            "scope_type": "workspace",
            "workspace_id": "workspace-1",
            "domains": list(WORKSPACE_SYNC_DOMAINS),
            "encryption_policy": "server_trusted_v1",
            "metadata": {"label": "Shared research"},
        },
    )
    workspace_push = _push_dataset(
        client,
        dataset_id="workspace-dataset",
        device_id="device-a",
        envelopes=[_workspace_envelope(), _workspace_source_ref_envelope()],
    )
    workspace_status = client.get(
        "/api/v1/sync/background-status",
        params={"dataset_id": "workspace-dataset", "device_id": "device-a"},
    )
    assert workspace.status_code == 200
    assert workspace_push["rejected"] == []
    assert [item["client_envelope_id"] for item in workspace_push["accepted"]] == [
        "env-workspace-1",
        "env-workspace-source-1",
    ]
    assert workspace_status.status_code == 200
    assert {item["domain"] for item in workspace_status.json()["domains"]} == set(WORKSPACE_SYNC_DOMAINS)

    harness.workspace_permissions.clear()
    workspace_denied = client.get(
        "/api/v1/sync/background-status",
        params={"dataset_id": "workspace-dataset", "device_id": "device-a"},
    )
    assert workspace_denied.status_code == 404

    revoked = client.post(
        "/api/v1/sync/devices/device-b/revoke",
        json={"reason": "lost_device", "revoke_key_records": False},
    )
    revoked_pull = client.get(
        "/api/v1/sync/pull",
        params={"dataset_id": "dataset-1", "device_id": "device-b", "cursor": "0"},
    )
    revoked_status = client.get(
        "/api/v1/sync/background-status",
        params={"dataset_id": "dataset-1", "device_id": "device-b"},
    )
    assert revoked.status_code == 200
    assert revoked.json()["status"] == "revoked"
    assert revoked_pull.status_code == 404
    assert revoked_status.status_code == 404
