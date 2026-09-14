from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
from tldw_Server_API.app.core.Sync.v2.adapters import StaticSyncAdapter, SyncAdapterRegistry
from tldw_Server_API.app.core.Sync.v2.blob_store import (
    LocalSyncBlobStore,
    SyncBlobStoreError,
)
from tldw_Server_API.app.core.Sync.v2.domain_adapters.attachment_refs import (
    AttachmentRefDomainAdapter,
)
from tldw_Server_API.app.core.Sync.v2.errors import (
    SyncIdempotencyConflictError,
    SyncStoreError,
)
from tldw_Server_API.app.core.Sync.v2.models import (
    SyncAttachmentRevisionBindingCreate,
    SyncBlobObjectCreate,
    SyncDatasetCreate,
    SyncEnvelope,
)
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


def _complete_attachment_blob(
    service: SyncV2Service,
    payload: bytes,
    *,
    idempotency_key: str,
):
    session = _create_attachment_session(
        service,
        payload,
        metadata=_attachment_intent(),
        idempotency_key=idempotency_key,
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
    return service.complete_blob_upload(
        user_id="user-1",
        dataset_id="dataset-1",
        upload_id=session.upload_id,
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


def test_physical_gc_fences_before_unlink_and_finalizes_deleted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _attachment_blob_service(tmp_path)
    payload = b"physical gc payload"
    blob = _complete_attachment_blob(service, payload, idempotency_key="gc-upload")
    assert service.blob_store is not None
    target = service.blob_store.resolve_storage_key(blob.storage_key)
    original_delete = service.blob_store.delete_namespace_blob
    observed_statuses: list[str] = []

    def observe_fence(**kwargs):
        current = service.store.get_blob_object(
            "dataset-1",
            blob_id=blob.blob_id,
            owner_user_id="user-1",
            include_unavailable=True,
        )
        assert current is not None
        observed_statuses.append(current.status)
        return original_delete(**kwargs)

    monkeypatch.setattr(service.blob_store, "delete_namespace_blob", observe_fence)

    result = service.retention_compact(
        user_id="user-1",
        dataset_id="dataset-1",
        confirm=True,
        apply_envelope_compaction=False,
        apply_tombstone_prune=False,
        apply_binding_release=False,
        apply_blob_gc=True,
    )

    assert observed_statuses == ["deleting"]
    assert result.applied_count == 1
    assert not target.exists()
    deleted = service.store.get_blob_object(
        "dataset-1",
        blob_id=blob.blob_id,
        owner_user_id="user-1",
        include_unavailable=True,
    )
    assert deleted is not None and deleted.status == "deleted"


def test_physical_gc_retries_deleting_after_transient_unlink_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _attachment_blob_service(tmp_path)
    payload = b"retry physical gc"
    blob = _complete_attachment_blob(service, payload, idempotency_key="retry-upload")
    assert service.blob_store is not None
    target = service.blob_store.resolve_storage_key(blob.storage_key)
    original_delete = service.blob_store.delete_namespace_blob

    def fail_delete(**_kwargs):
        raise SyncBlobStoreError("transient unlink failure")

    monkeypatch.setattr(service.blob_store, "delete_namespace_blob", fail_delete)
    first = service.retention_compact(
        user_id="user-1",
        dataset_id="dataset-1",
        confirm=True,
        apply_envelope_compaction=False,
        apply_tombstone_prune=False,
        apply_binding_release=False,
        apply_blob_gc=True,
    )

    fenced = service.store.get_blob_object(
        "dataset-1",
        blob_id=blob.blob_id,
        owner_user_id="user-1",
        include_unavailable=True,
    )
    assert first.applied_count == 0
    assert first.mutation_performed is True
    assert first.blocker_counts == {"retention_blob_delete_retry": 1}
    assert fenced is not None and fenced.status == "deleting"
    assert target.read_bytes() == payload

    monkeypatch.setattr(service.blob_store, "delete_namespace_blob", original_delete)
    retry = service.retention_compact(
        user_id="user-1",
        dataset_id="dataset-1",
        confirm=True,
        apply_envelope_compaction=False,
        apply_tombstone_prune=False,
        apply_binding_release=False,
        apply_blob_gc=True,
    )

    deleted = service.store.get_blob_object(
        "dataset-1",
        blob_id=blob.blob_id,
        owner_user_id="user-1",
        include_unavailable=True,
    )
    assert retry.applied_count == 1
    assert deleted is not None and deleted.status == "deleted"
    assert not target.exists()


def test_physical_gc_rejects_upload_while_deleting_then_allows_deleted_repair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _attachment_blob_service(tmp_path)
    payload = b"same digest repair"
    blob = _complete_attachment_blob(service, payload, idempotency_key="original-upload")
    assert service.blob_store is not None
    original_delete = service.blob_store.delete_namespace_blob
    repair_session = _create_attachment_session(
        service,
        payload,
        metadata=_attachment_intent(),
        idempotency_key="repair-upload",
    )
    service.upload_blob_chunk(
        user_id="user-1",
        dataset_id="dataset-1",
        upload_id=repair_session.upload_id,
        chunk_index=0,
        offset_bytes=0,
        chunk_payload=payload,
        chunk_hash=_sha256(payload),
    )
    with service.store.db.materialization_transaction(
        [("dataset-1", "attachment.ref", OTHER_NOTE_ID)]
    ) as connection:
        service.store.db._create_attachment_revision_binding(
            SyncAttachmentRevisionBindingCreate(
                dataset_id="dataset-1",
                attachment_id=OTHER_NOTE_ID,
                attachment_revision=1,
                blob_hash=_sha256(payload),
                size_bytes=len(payload),
                establishing_server_cursor=999,
                availability_at_acceptance="metadata_only",
            ),
            connection=connection,
        )

    def assert_deleting_then_fail(**_kwargs):
        current = service.store.get_blob_object(
            "dataset-1",
            blob_id=blob.blob_id,
            owner_user_id="user-1",
            include_unavailable=True,
        )
        assert current is not None and current.status == "deleting"
        original_commit = service.blob_store.commit_upload

        def fail_if_storage_is_touched(**_commit_kwargs):
            raise AssertionError("deleting blob completion touched storage")

        monkeypatch.setattr(service.blob_store, "commit_upload", fail_if_storage_is_touched)
        try:
            with pytest.raises(SyncStoreError, match="deleting"):
                service.complete_blob_upload(
                    user_id="user-1",
                    dataset_id="dataset-1",
                    upload_id=repair_session.upload_id,
                )
        finally:
            monkeypatch.setattr(service.blob_store, "commit_upload", original_commit)
        with pytest.raises(SyncStoreError, match="exact available blob"):
            service.store.db.resolve_attachment_revision_binding(
                "dataset-1",
                OTHER_NOTE_ID,
                1,
                blob_id=blob.blob_id,
                owner_user_id="user-1",
            )
        envelope = SyncEnvelope(
            dataset_id="dataset-1",
            client_envelope_id="deleting-binding",
            domain="attachment.ref",
            operation="upsert",
            server_cursor=1000,
            object_id="f3333333-3333-4333-8333-333333333333",
            object_revision=1,
            adapter_version=2,
            payload={
                "attachment_id": "f3333333-3333-4333-8333-333333333333",
                "blob_hash": _sha256(payload),
                "size_bytes": len(payload),
            },
        )
        with service.store.db.materialization_transaction(
            [("dataset-1", "attachment.ref", envelope.object_id)]
        ) as connection:
            with pytest.raises(SyncStoreError, match="deleting"):
                service.store.db._create_attachment_binding_for_envelope(
                    envelope,
                    connection=connection,
                )
        raise SyncBlobStoreError("leave the fence durable")

    monkeypatch.setattr(
        service.blob_store,
        "delete_namespace_blob",
        assert_deleting_then_fail,
    )
    service.retention_compact(
        user_id="user-1",
        dataset_id="dataset-1",
        confirm=True,
        apply_envelope_compaction=False,
        apply_tombstone_prune=False,
        apply_binding_release=False,
        apply_blob_gc=True,
    )

    monkeypatch.setattr(service.blob_store, "delete_namespace_blob", original_delete)
    service.retention_compact(
        user_id="user-1",
        dataset_id="dataset-1",
        confirm=True,
        apply_envelope_compaction=False,
        apply_tombstone_prune=False,
        apply_binding_release=False,
        apply_blob_gc=True,
    )
    repaired = service.complete_blob_upload(
        user_id="user-1",
        dataset_id="dataset-1",
        upload_id=repair_session.upload_id,
    )

    assert repaired.blob_id == blob.blob_id
    assert repaired.status == "available"
    assert service.blob_store.read_blob(repaired.storage_key) == payload


@pytest.mark.parametrize("unavailable_status", ["verify_failed", "quarantined"])
def test_attachment_upload_completion_rejects_unavailable_blob_before_storage_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unavailable_status: str,
) -> None:
    service = _attachment_blob_service(tmp_path)
    payload = b"unavailable exact blob"
    _complete_attachment_blob(service, payload, idempotency_key="original-upload")
    service.store.db.execute(
        "UPDATE sync_blob_objects SET status = ? WHERE dataset_id = ? AND payload_hash = ?",
        (unavailable_status, "dataset-1", _sha256(payload)),
    )
    retry = _create_attachment_session(
        service,
        payload,
        metadata=_attachment_intent(),
        idempotency_key=f"retry-{unavailable_status}",
    )
    service.upload_blob_chunk(
        user_id="user-1",
        dataset_id="dataset-1",
        upload_id=retry.upload_id,
        chunk_index=0,
        offset_bytes=0,
        chunk_payload=payload,
        chunk_hash=_sha256(payload),
    )
    assert service.blob_store is not None

    def fail_if_storage_is_touched(**_kwargs):
        raise AssertionError("unavailable blob completion touched storage")

    monkeypatch.setattr(service.blob_store, "commit_upload", fail_if_storage_is_touched)

    with pytest.raises(SyncStoreError, match="not available"):
        service.complete_blob_upload(
            user_id="user-1",
            dataset_id="dataset-1",
            upload_id=retry.upload_id,
        )


def test_physical_gc_finalizes_when_fenced_namespace_blob_is_already_absent(
    tmp_path: Path,
) -> None:
    service = _attachment_blob_service(tmp_path)
    payload = b"already absent physical gc"
    blob = _complete_attachment_blob(service, payload, idempotency_key="absent-upload")
    assert service.blob_store is not None
    with service.store.retention_guard("dataset-1", blob.blob_id) as guarded:
        fenced = guarded.fence_blob_object_deleting("dataset-1", blob.blob_id)
        assert fenced is not None and fenced.status == "deleting"
    service.blob_store.resolve_storage_key(blob.storage_key).unlink()

    result = service.retention_compact(
        user_id="user-1",
        dataset_id="dataset-1",
        confirm=True,
        apply_envelope_compaction=False,
        apply_tombstone_prune=False,
        apply_binding_release=False,
        apply_blob_gc=True,
    )

    deleted = service.store.get_blob_object(
        "dataset-1",
        blob_id=blob.blob_id,
        owner_user_id="user-1",
        include_unavailable=True,
    )
    assert result.applied_count == 1
    assert deleted is not None and deleted.status == "deleted"


def test_physical_gc_blocks_legacy_global_storage_key(tmp_path: Path) -> None:
    service = _attachment_blob_service(tmp_path)
    payload = b"legacy global payload"
    payload_hash = _sha256(payload)
    assert service.blob_store is not None
    storage_key = service.blob_store.legacy_storage_key(payload_hash)
    target = service.blob_store.resolve_storage_key(storage_key)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    blob = service.store.complete_blob_upload(
        SyncBlobObjectCreate(
            blob_id="legacy-global-blob",
            dataset_id="dataset-1",
            owner_user_id="user-1",
            attachment_id=ATTACHMENT_ID,
            payload_hash=payload_hash,
            content_type="application/pdf",
            size_bytes=len(payload),
            storage_backend="local_fs",
            storage_key=storage_key,
        )
    )

    dry_run = service.retention_dry_run(
        user_id="user-1",
        dataset_id="dataset-1",
        audit_mode=False,
    )
    candidate = next(item for item in dry_run.candidates if item.blob_id == blob.blob_id)
    result = service.retention_compact(
        user_id="user-1",
        dataset_id="dataset-1",
        confirm=True,
        apply_envelope_compaction=False,
        apply_tombstone_prune=False,
        apply_binding_release=False,
        apply_blob_gc=True,
    )

    assert candidate.blockers == ["retention_blob_storage_key_not_namespaced"]
    assert result.applied_count == 0
    assert target.read_bytes() == payload
    current = service.store.get_blob_object(
        "dataset-1",
        blob_id=blob.blob_id,
        owner_user_id="user-1",
        include_unavailable=True,
    )
    assert current is not None and current.status == "available"


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
