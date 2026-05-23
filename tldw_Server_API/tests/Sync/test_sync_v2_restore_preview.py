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
def sync_service(tmp_path: Path) -> SyncV2Service:
    default_sync_v2_registry.cache_clear()
    service = SyncV2Service(
        store=SyncV2Store(SyncDatabase(sqlite_path=tmp_path / "sync_v2_restore_preview.db")),
        adapters=default_sync_v2_registry(),
        clock=_clock,
        id_factory=lambda prefix: f"{prefix}-generated",
        settings=SyncV2Settings(
            server_trusted_encryption=_ready_encryption(),
            restore_manifest_scan_limit=100,
        ),
    )
    for user_id, device_id in (
        ("user-1", "device-1"),
        ("user-1", "device-2"),
        ("user-2", "other-device"),
    ):
        service.register_device(
            user_id=user_id,
            display_name=device_id,
            client_type="chatbook",
            device_id=device_id,
        )
    service.enroll_dataset(
        user_id="user-1",
        dataset_id="dataset-1",
        domains=list(M1_SYNC_DOMAINS),
    )
    return service


@pytest.fixture()
def client(sync_service: SyncV2Service) -> TestClient:
    app = FastAPI()
    app.include_router(sync_endpoint.router, prefix="/api/v1/sync")
    app.dependency_overrides[get_request_user] = _test_user
    app.dependency_overrides[sync_endpoint.get_sync_v2_service] = lambda: sync_service
    if hasattr(sync_endpoint, "get_sync_v2_profile_service"):
        app.dependency_overrides[sync_endpoint.get_sync_v2_profile_service] = lambda: sync_service
    return TestClient(app)


def _note_envelope(**overrides: Any) -> SyncEnvelopeCreate:
    payload: dict[str, Any] = {
        "dataset_id": "dataset-1",
        "client_envelope_id": "env-note-1",
        "domain": "notes.note",
        "operation": "upsert",
        "object_id": "note-1",
        "device_id": "device-1",
        "client_sequence": 1,
        "object_revision": 1,
        "payload": {"title": "Research note", "content": "Body"},
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
        "client_envelope_id": "env-chat-1",
        "domain": "chat.conversation",
        "operation": "upsert",
        "object_id": "conversation-1",
        "device_id": "device-1",
        "client_sequence": 10,
        "object_revision": 1,
        "payload": {"title": "Research thread", "character_id": "character-1"},
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
        "client_envelope_id": "env-message-1",
        "domain": "chat.message",
        "operation": "append",
        "object_id": "message-1",
        "parent_id": "conversation-1",
        "device_id": "device-1",
        "client_sequence": 20,
        "payload": {"conversation_id": "conversation-1", "role": "user", "content": "Hello"},
        "payload_hash": "sha256:message-v1",
        "payload_size_bytes": 80,
        "created_at_client": "2026-05-23T18:02:00+00:00",
        "encryption_metadata": {"policy": "server_trusted_v1"},
        "stable_key": "chat:message-1",
    }
    payload.update(overrides)
    return SyncEnvelopeCreate(**payload)


def _attachment_ref_envelope(**overrides: Any) -> SyncEnvelopeCreate:
    payload: dict[str, Any] = {
        "dataset_id": "dataset-1",
        "client_envelope_id": "env-attachment-1",
        "domain": "attachment.ref",
        "operation": "upsert",
        "object_id": "attachment-1",
        "device_id": "device-1",
        "client_sequence": 30,
        "payload": {
            "attachment_id": "attachment-1",
            "parent_domain": "notes.note",
            "parent_object_id": "note-1",
            "content_type": "image/png",
            "size_bytes": 512,
            "payload_hash": "sha256:attachment-v1",
            "availability": "client_local",
        },
        "payload_hash": "sha256:attachment-v1",
        "payload_size_bytes": 128,
        "created_at_client": "2026-05-23T18:03:00+00:00",
        "encryption_metadata": {"policy": "server_trusted_v1"},
        "stable_key": "attachment:attachment-1",
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


def test_restore_preview_empty_inventory_returns_safe_applies_ranges_counts_and_key_status(
    sync_service: SyncV2Service,
) -> None:
    _push(sync_service, _note_envelope(), _conversation_envelope(), _message_envelope())

    preview = sync_service.restore_preview(
        user_id="user-1",
        dataset_ids=["dataset-1"],
        local_inventory=[],
    )

    assert preview.total_counts == {
        "chat.conversation": 1,
        "chat.message": 1,
        "notes.note": 1,
    }
    assert preview.datasets[0].latest_cursors == {
        "chat.conversation": 2,
        "chat.message": 3,
        "notes.note": 1,
    }
    assert [(item.domain, item.object_id, item.action) for item in preview.safe_applies] == [
        ("notes.note", "note-1", "apply"),
        ("chat.conversation", "conversation-1", "apply"),
        ("chat.message", "message-1", "append"),
    ]
    ranges = [(item.domain, item.from_cursor, item.to_cursor, item.envelope_count) for item in preview.envelope_ranges]
    assert ranges == [
        ("notes.note", 1, 1, 1),
        ("chat.conversation", 2, 2, 1),
        ("chat.message", 3, 3, 1),
    ]
    assert preview.object_conflicts == []
    assert preview.tombstones == []
    assert preview.encryption["policy"] == "server_trusted_v1"
    assert preview.encryption["ready"] is True
    assert preview.key_status == {"dataset-1": {"key_recovery_available": False}}


def test_restore_preview_matching_local_inventory_is_safe_noop(
    sync_service: SyncV2Service,
) -> None:
    _push(sync_service, _note_envelope())

    preview = sync_service.restore_preview(
        user_id="user-1",
        dataset_ids=["dataset-1"],
        local_inventory=[
            {
                "domain": "notes.note",
                "object_id": "note-1",
                "object_revision": 1,
                "object_hash": "sha256:note-v1",
                "deleted": False,
            }
        ],
    )

    assert [(item.domain, item.object_id, item.action) for item in preview.safe_applies] == [
        ("notes.note", "note-1", "noop")
    ]
    assert preview.object_conflicts == []


def test_restore_preview_reports_whole_object_note_and_conversation_conflicts(
    sync_service: SyncV2Service,
) -> None:
    _push(sync_service, _note_envelope(), _conversation_envelope())

    preview = sync_service.restore_preview(
        user_id="user-1",
        dataset_ids=["dataset-1"],
        local_inventory=[
            {
                "domain": "notes.note",
                "object_id": "note-1",
                "object_revision": 1,
                "object_hash": "sha256:local-note",
                "deleted": False,
            },
            {
                "domain": "chat.conversation",
                "object_id": "conversation-1",
                "object_revision": 1,
                "object_hash": "sha256:local-conversation",
                "deleted": False,
            },
        ],
    )

    assert [(item.domain, item.object_id, item.conflict_type) for item in preview.object_conflicts] == [
        ("notes.note", "note-1", "whole_object_conflict"),
        ("chat.conversation", "conversation-1", "whole_object_conflict"),
    ]
    assert [item.server_hash for item in preview.object_conflicts] == [
        "sha256:note-v1",
        "sha256:conversation-v1",
    ]
    assert preview.safe_applies == []


def test_restore_preview_surfaces_tombstones_as_delete_actions(
    sync_service: SyncV2Service,
) -> None:
    _push(sync_service, _note_envelope())
    _push(
        sync_service,
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

    preview = sync_service.restore_preview(
        user_id="user-1",
        dataset_ids=["dataset-1"],
        local_inventory=[
            {
                "domain": "notes.note",
                "object_id": "note-1",
                "object_revision": 1,
                "object_hash": "sha256:note-v1",
                "deleted": False,
            }
        ],
    )

    assert [(item.domain, item.object_id, item.action, item.server_cursor) for item in preview.tombstones] == [
        ("notes.note", "note-1", "delete", 2)
    ]
    assert preview.safe_applies == []
    assert preview.object_conflicts == []


def test_restore_preview_includes_attachment_refs_and_missing_blob_warning(
    sync_service: SyncV2Service,
) -> None:
    _push(sync_service, _attachment_ref_envelope())

    preview = sync_service.restore_preview(
        user_id="user-1",
        dataset_ids=["dataset-1"],
        local_inventory=[],
    )

    assert [(item.attachment_id, item.parent_domain, item.parent_object_id) for item in preview.attachment_refs] == [
        ("attachment-1", "notes.note", "note-1")
    ]
    assert [item.attachment_id for item in preview.missing_blobs] == ["attachment-1"]
    assert [warning.code for warning in preview.warnings] == [
        "sync_key_recovery_missing",
        "sync_attachment_blob_missing",
    ]


def test_restore_preview_endpoint_blocks_requested_cross_user_dataset(
    client: TestClient,
    sync_service: SyncV2Service,
) -> None:
    sync_service.enroll_dataset(
        user_id="user-2",
        dataset_id="dataset-other",
        domains=list(M1_SYNC_DOMAINS),
    )
    result = sync_service.push(
        user_id="user-2",
        dataset_id="dataset-other",
        device_id="other-device",
        envelopes=[
            _note_envelope(
                dataset_id="dataset-other",
                device_id="other-device",
                object_id="other-note",
                payload_hash="sha256:other-user",
            )
        ],
    )
    assert [item.client_envelope_id for item in result.accepted] == ["env-note-1"]

    response = client.post(
        "/api/v1/sync/restore/preview",
        json={"dataset_ids": ["dataset-other"], "local_inventory": []},
    )
    broad_response = client.post(
        "/api/v1/sync/restore/preview",
        json={"local_inventory": []},
    )

    assert response.status_code == 404
    assert response.json()["detail"]["error_code"] == "sync_resource_not_found"
    assert [dataset["dataset_id"] for dataset in broad_response.json()["datasets"]] == ["dataset-1"]
    assert "other-note" not in str(broad_response.json())
