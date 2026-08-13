from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
from tldw_Server_API.app.core.Sync.v2.adapters import StaticSyncAdapter, SyncAdapterRegistry
from tldw_Server_API.app.core.Sync.v2.blob_store import LocalSyncBlobStore
from tldw_Server_API.app.core.Sync.v2.domain_adapters.attachment_refs import (
    AttachmentRefDomainAdapter,
)
from tldw_Server_API.app.core.Sync.v2.errors import (
    SyncIdempotencyConflictError,
    SyncStoreError,
)
from tldw_Server_API.app.core.Sync.v2.models import SyncDatasetCreate
from tldw_Server_API.app.core.Sync.v2.security import server_trusted_encryption_status_from_config
from tldw_Server_API.app.core.Sync.v2.service import SyncV2Service, SyncV2Settings
from tldw_Server_API.app.core.Sync.v2.store import SyncV2Store


def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _ready_encryption():
    return server_trusted_encryption_status_from_config(
        mode="managed_storage",
        server_trusted_enabled=True,
        auth_mode="multi_user",
    )


NOTE_ID = "a1677eb1-1f41-4c86-a8dd-1eaa14b014e2"
OTHER_NOTE_ID = "c213e645-0dc2-44d4-b720-c5281cfdf3d6"
ATTACHMENT_ID = "2c4cb609-c4db-44f9-8e35-f078bd36d6b2"


def _attachment_blob_service(tmp_path: Path) -> SyncV2Service:
    store = SyncV2Store(SyncDatabase(sqlite_path=tmp_path / "attachment-sync.sqlite"))
    store.enroll_dataset(
        SyncDatasetCreate(
            dataset_id="dataset-1",
            owner_user_id="user-1",
            domains=["notes.note", "attachment.ref"],
            metadata={
                "default_personal": True,
                "client_family": "chatbook",
                "notes_attachment_v2": {"state": "ready"},
            },
        )
    )
    return SyncV2Service(
        store=store,
        adapters=SyncAdapterRegistry(
            [AttachmentRefDomainAdapter(v2_writes_enabled=True)]
        ),
        blob_store=LocalSyncBlobStore(tmp_path / "attachment-blobs"),
        settings=SyncV2Settings(
            supports_attachments=True,
            max_blob_bytes=1024,
            max_chunk_bytes=128,
            server_trusted_encryption=_ready_encryption(),
        ),
    )


def _attachment_intent(*, note_id: str = NOTE_ID) -> dict[str, object]:
    return {
        "notes_attachment_intent": {
            "intent": "create",
            "note_id": note_id,
            "attachment_id": ATTACHMENT_ID,
            "file_name": "report.pdf",
        }
    }


def _create_attachment_session(
    service: SyncV2Service,
    payload: bytes,
    *,
    metadata: dict[str, object] | None,
    idempotency_key: str = "upload-key",
):
    return service.create_blob_upload_session(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id=None,
        domain="attachment.ref",
        entity_id=ATTACHMENT_ID,
        attachment_id=ATTACHMENT_ID,
        content_type="application/pdf",
        size_bytes=len(payload),
        payload_hash=_sha256(payload),
        chunk_size=len(payload),
        chunk_count=1,
        idempotency_key=idempotency_key,
        metadata=metadata,
    )


def test_attachment_upload_intent_is_required_immutable_and_owner_bound(
    tmp_path: Path,
) -> None:
    service = _attachment_blob_service(tmp_path)
    payload = b"attachment payload"

    with pytest.raises(SyncStoreError, match="intent"):
        _create_attachment_session(service, payload, metadata=None)

    session = _create_attachment_session(
        service,
        payload,
        metadata=_attachment_intent(),
    )
    replay = _create_attachment_session(
        service,
        payload,
        metadata=_attachment_intent(),
    )

    assert replay.upload_id == session.upload_id
    assert session.owner_user_id == "user-1"
    assert session.domain == "attachment.ref"
    assert session.object_id == ATTACHMENT_ID
    assert session.metadata["notes_attachment_intent"] == _attachment_intent()[
        "notes_attachment_intent"
    ]
    assert session.metadata["_notes_attachment_binding"] == {
        "intent": "create",
        "note_id": NOTE_ID,
        "attachment_id": ATTACHMENT_ID,
        "file_name": "report.pdf",
        "original_file_name": "report.pdf",
    }

    with pytest.raises(SyncIdempotencyConflictError):
        _create_attachment_session(
            service,
            payload,
            metadata=_attachment_intent(note_id=OTHER_NOTE_ID),
        )


def test_attachment_upload_completion_is_namespaced_retryable_and_not_reassignable(
    tmp_path: Path,
) -> None:
    service = _attachment_blob_service(tmp_path)
    payload = b"attachment payload"
    session = _create_attachment_session(
        service,
        payload,
        metadata=_attachment_intent(),
    )
    service.upload_blob_chunk(
        user_id="user-1",
        dataset_id="dataset-1",
        upload_id=session.upload_id,
        chunk_index=0,
        offset_bytes=0,
        chunk_payload=payload,
        chunk_hash=_sha256(payload),
    )

    first = service.complete_blob_upload(
        user_id="user-1",
        dataset_id="dataset-1",
        upload_id=session.upload_id,
    )
    replay = service.complete_blob_upload(
        user_id="user-1",
        dataset_id="dataset-1",
        upload_id=session.upload_id,
    )

    assert replay.blob_id == first.blob_id
    assert first.storage_key.startswith("blobs/v2/")
    assert "user-1" not in first.storage_key
    assert "dataset-1" not in first.storage_key
    assert service.require_completed_notes_attachment_upload(
        user_id="user-1",
        dataset_id="dataset-1",
        upload_id=session.upload_id,
        note_id=NOTE_ID,
        attachment_id=ATTACHMENT_ID,
    )[1].blob_id == first.blob_id

    with pytest.raises(SyncStoreError, match="intent"):
        service.require_completed_notes_attachment_upload(
            user_id="user-1",
            dataset_id="dataset-1",
            upload_id=session.upload_id,
            note_id=OTHER_NOTE_ID,
            attachment_id=ATTACHMENT_ID,
        )


def test_workspace_blob_download_is_dataset_scoped_not_uploader_scoped(tmp_path: Path) -> None:
    service = SyncV2Service(
        store=SyncV2Store(SyncDatabase(sqlite_path=tmp_path / "sync_v2_workspace_blobs.db")),
        adapters=SyncAdapterRegistry(
            [
                StaticSyncAdapter(domain="workspaces.source_ref", supported_adapter_versions={1}),
            ]
        ),
        blob_store=LocalSyncBlobStore(tmp_path / "sync_blobs"),
        settings=SyncV2Settings(
            supports_attachments=True,
            max_blob_bytes=1024,
            max_chunk_bytes=128,
            server_trusted_encryption=_ready_encryption(),
        ),
        workspace_access_checker=lambda _user_id, workspace_id, permission: (
            workspace_id == "workspace-1" and permission == "sync"
        ),
    )
    service.register_device(
        user_id="user-1",
        device_id="owner-device",
        display_name="Owner",
        client_type="chatbook",
    )
    service.register_device(
        user_id="user-2",
        device_id="member-device",
        display_name="Member",
        client_type="chatbook",
    )
    service.enroll_dataset(
        user_id="user-1",
        dataset_id="workspace-dataset",
        scope_type="workspace",
        workspace_id="workspace-1",
        domains=["workspaces.source_ref"],
    )
    payload = b"workspace payload"
    session = service.create_blob_upload_session(
        user_id="user-1",
        dataset_id="workspace-dataset",
        device_id="owner-device",
        domain="workspaces.source_ref",
        entity_id="source-1",
        attachment_id="attachment-1",
        content_type="application/octet-stream",
        size_bytes=len(payload),
        payload_hash=_sha256(payload),
        chunk_size=len(payload),
        chunk_count=1,
    )
    service.upload_blob_chunk(
        user_id="user-1",
        dataset_id="workspace-dataset",
        upload_id=session.upload_id,
        chunk_index=0,
        offset_bytes=0,
        chunk_payload=payload,
        chunk_hash=_sha256(payload),
    )
    service.complete_blob_upload(
        user_id="user-1",
        dataset_id="workspace-dataset",
        upload_id=session.upload_id,
    )

    manifest = service.blob_download_manifest(
        user_id="user-2",
        dataset_id="workspace-dataset",
        attachment_id="attachment-1",
    )
    downloaded = b"".join(
        service.iter_blob_bytes(
            user_id="user-2",
            dataset_id="workspace-dataset",
            attachment_id="attachment-1",
        )
    )

    assert manifest.availability == "available"
    assert manifest.payload_hash == _sha256(payload)
    assert downloaded == payload
