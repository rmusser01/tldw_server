from __future__ import annotations

import hashlib
from pathlib import Path

from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
from tldw_Server_API.app.core.Sync.v2.adapters import StaticSyncAdapter, SyncAdapterRegistry
from tldw_Server_API.app.core.Sync.v2.blob_store import LocalSyncBlobStore
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
