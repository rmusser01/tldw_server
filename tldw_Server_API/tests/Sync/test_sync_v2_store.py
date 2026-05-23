from __future__ import annotations

import inspect
import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import tldw_Server_API.app.core.Sync.v2.store as store_module
from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase, utcnow_iso
from tldw_Server_API.app.core.Sync.v2.errors import (
    SyncDatasetNotFoundError,
    SyncIdempotencyConflictError,
    SyncInvalidDomainError,
    SyncStoreError,
)
from tldw_Server_API.app.core.Sync.v2.models import (
    SyncAttachmentCreate,
    SyncBlobChunkCreate,
    SyncBlobObjectCreate,
    SyncBlobUploadSessionCreate,
    SyncConflictCreate,
    SyncDatasetCreate,
    SyncDeviceCursor,
    SyncDeviceUpsert,
    SyncEnvelopeCreate,
    SyncKeyRecordCreate,
)
from tldw_Server_API.app.core.Sync.v2.store import SyncV2Store


@pytest.fixture()
def sync_store(tmp_path: Path) -> SyncV2Store:
    db = SyncDatabase(sqlite_path=tmp_path / "sync_v2.db")
    return SyncV2Store(db)


def _device(**overrides) -> SyncDeviceUpsert:
    payload = {
        "device_id": "device-1",
        "user_id": "user-1",
        "display_name": "Laptop",
        "client_type": "chatbook",
        "client_version": "0.1.0",
        "capabilities": {"domains": ["notes.note"]},
    }
    payload.update(overrides)
    return SyncDeviceUpsert(**payload)


def _dataset(**overrides) -> SyncDatasetCreate:
    payload = {
        "dataset_id": "dataset-1",
        "owner_user_id": "user-1",
        "scope_type": "personal",
        "encryption_policy": "server_trusted_v1",
        "domains": [
            "notes.note",
            "chat.conversation",
            "chat.message",
            "attachment.ref",
        ],
        "metadata": {"label": "Personal research"},
    }
    payload.update(overrides)
    return SyncDatasetCreate(**payload)


def _envelope(**overrides) -> SyncEnvelopeCreate:
    payload = {
        "dataset_id": "dataset-1",
        "client_envelope_id": "env-1",
        "domain": "notes.note",
        "object_id": "note-1",
        "operation": "upsert",
        "device_id": "device-1",
        "client_profile_id": "chatbook-profile-1",
        "client_sequence": None,
        "base_server_cursor": None,
        "base_object_revision": None,
        "base_object_hash": None,
        "object_revision": None,
        "parent_id": None,
        "schema_version": 1,
        "payload": {"status": "active"},
        "payload_hash": "sha256:note-1",
        "payload_size_bytes": 24,
        "created_at_client": "2026-05-10T00:00:00+00:00",
        "deleted": False,
        "encryption_metadata": {"policy": "server_trusted_v1"},
        "status": "accepted",
        "apply_status": "pending",
    }
    payload.update(overrides)
    return SyncEnvelopeCreate(**payload)


def _conflict(**overrides) -> SyncConflictCreate:
    payload = {
        "conflict_id": "conflict-1",
        "dataset_id": "dataset-1",
        "domain": "notes.note",
        "object_id": "note-1",
        "conflict_type": "version_divergence",
        "base_envelope_id": "env-base",
        "local_envelope_id": "env-local",
        "remote_envelope_id": "env-remote",
        "server_cursor": 3,
        "metadata": {"reason": "same entity changed on two devices"},
    }
    payload.update(overrides)
    return SyncConflictCreate(**payload)


def _key_record(**overrides) -> SyncKeyRecordCreate:
    payload = {
        "key_record_id": "key-1",
        "dataset_id": "dataset-1",
        "user_id": "user-1",
        "device_id": "device-1",
        "key_purpose": "dataset_recovery",
        "wrapped_key_blob": "wrapped:opaque",
        "kdf_metadata": {"algorithm": "argon2id"},
        "recovery_hint": "personal laptop",
    }
    payload.update(overrides)
    return SyncKeyRecordCreate(**payload)


def _attachment(**overrides) -> SyncAttachmentCreate:
    payload = {
        "attachment_id": "attachment-1",
        "dataset_id": "dataset-1",
        "domain": "attachment.ref",
        "object_id": "attachment-1",
        "content_type": "application/octet-stream",
        "size_bytes": 512,
        "payload_ciphertext": "ciphertext:attachment",
        "payload_hash": "sha256:attachment",
        "encryption_policy": "server_trusted_v1",
        "metadata": {
            "parent_domain": "notes.note",
            "parent_object_id": "note-1",
            "availability": "available",
        },
    }
    payload.update(overrides)
    return SyncAttachmentCreate(**payload)


def test_sync_database_rejects_unsupported_database_url_scheme(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("SYNC_V2_DATABASE_URL", "mysql://sync.example/sync_v2")
    monkeypatch.delenv("SYNC_V2_SQLITE_PATH", raising=False)

    with pytest.raises(SyncStoreError, match="Unsupported SYNC_V2_DATABASE_URL scheme"):
        SyncDatabase(sqlite_path=tmp_path / "ignored.db")


def test_sync_database_bootstrap_creates_required_tables(sync_store: SyncV2Store):
    required_tables = {
        "sync_devices",
        "sync_datasets",
        "sync_domain_state",
        "sync_envelopes",
        "sync_object_state",
        "sync_device_cursors",
        "sync_conflicts",
        "sync_key_records",
        "sync_attachments",
        "sync_blob_objects",
        "sync_blob_upload_sessions",
        "sync_blob_chunks",
    }

    for table_name in required_tables:
        assert sync_store.db.backend.table_exists(table_name)


def test_blob_upload_sessions_are_idempotent_and_release_reserved_quota(
    sync_store: SyncV2Store,
):
    sync_store.upsert_device(_device())
    sync_store.enroll_dataset(_dataset())

    session = sync_store.create_blob_upload_session(
        SyncBlobUploadSessionCreate(
            upload_id="upload-1",
            dataset_id="dataset-1",
            owner_user_id="user-1",
            device_id="device-1",
            attachment_id="attachment-1",
            domain="attachment.ref",
            object_id="attachment-1",
            content_type="application/octet-stream",
            size_bytes=2048,
            payload_hash="sha256:" + "a" * 64,
            chunk_size=1024,
            chunk_count=2,
            reserved_quota_bytes=2048,
            idempotency_key="same-upload",
        )
    )
    retry = sync_store.create_blob_upload_session(
        SyncBlobUploadSessionCreate(
            upload_id="upload-retry",
            dataset_id="dataset-1",
            owner_user_id="user-1",
            device_id="device-1",
            attachment_id="attachment-1",
            domain="attachment.ref",
            object_id="attachment-1",
            content_type="application/octet-stream",
            size_bytes=2048,
            payload_hash="sha256:" + "a" * 64,
            chunk_size=1024,
            chunk_count=2,
            reserved_quota_bytes=2048,
            idempotency_key="same-upload",
        )
    )

    assert retry.upload_id == session.upload_id
    assert retry.missing_chunks == [0, 1]
    assert sync_store.summarize_blob_quota("user-1").reserved_blob_bytes == 2048

    with pytest.raises(SyncIdempotencyConflictError):
        sync_store.create_blob_upload_session(
            SyncBlobUploadSessionCreate(
                upload_id="upload-drift",
                dataset_id="dataset-1",
                owner_user_id="user-1",
                device_id="device-1",
                attachment_id="attachment-1",
                domain="attachment.ref",
                object_id="attachment-1",
                content_type="application/octet-stream",
                size_bytes=2048,
                payload_hash="sha256:" + "b" * 64,
                chunk_size=1024,
                chunk_count=2,
                reserved_quota_bytes=2048,
                idempotency_key="same-upload",
            )
        )

    cancelled = sync_store.cancel_blob_upload_session("upload-1", dataset_id="dataset-1")

    assert cancelled.status == "cancelled"
    assert sync_store.summarize_blob_quota("user-1").reserved_blob_bytes == 0


def test_blob_chunks_and_completion_validate_idempotency_and_dedupe(
    sync_store: SyncV2Store,
):
    sync_store.upsert_device(_device())
    sync_store.enroll_dataset(_dataset())
    payload_hash = "sha256:" + "a" * 64

    sync_store.create_blob_upload_session(
        SyncBlobUploadSessionCreate(
            upload_id="upload-1",
            dataset_id="dataset-1",
            owner_user_id="user-1",
            device_id="device-1",
            attachment_id="attachment-1",
            domain="attachment.ref",
            object_id="attachment-1",
            content_type="application/octet-stream",
            size_bytes=2048,
            payload_hash=payload_hash,
            chunk_size=1024,
            chunk_count=2,
            reserved_quota_bytes=2048,
        )
    )
    first_chunk = sync_store.record_blob_chunk(
        SyncBlobChunkCreate(
            upload_id="upload-1",
            dataset_id="dataset-1",
            chunk_index=0,
            offset_bytes=0,
            size_bytes=1024,
            chunk_hash="sha256:" + "1" * 64,
            storage_key="uploads/upload-1/0.part",
        )
    )
    retry_chunk = sync_store.record_blob_chunk(
        SyncBlobChunkCreate(
            upload_id="upload-1",
            dataset_id="dataset-1",
            chunk_index=0,
            offset_bytes=0,
            size_bytes=1024,
            chunk_hash="sha256:" + "1" * 64,
            storage_key="uploads/upload-1/0.part",
        )
    )

    assert first_chunk.chunk_hash == retry_chunk.chunk_hash
    assert sync_store.get_blob_upload_session("upload-1").missing_chunks == [1]

    with pytest.raises(SyncIdempotencyConflictError):
        sync_store.record_blob_chunk(
            SyncBlobChunkCreate(
                upload_id="upload-1",
                dataset_id="dataset-1",
                chunk_index=0,
                offset_bytes=0,
                size_bytes=1024,
                chunk_hash="sha256:" + "2" * 64,
                storage_key="uploads/upload-1/0.part",
            )
        )

    sync_store.record_blob_chunk(
        SyncBlobChunkCreate(
            upload_id="upload-1",
            dataset_id="dataset-1",
            chunk_index=1,
            offset_bytes=1024,
            size_bytes=1024,
            chunk_hash="sha256:" + "3" * 64,
            storage_key="uploads/upload-1/1.part",
        )
    )
    blob = sync_store.complete_blob_upload(
        SyncBlobObjectCreate(
            blob_id="blob-1",
            dataset_id="dataset-1",
            owner_user_id="user-1",
            attachment_id="attachment-1",
            payload_hash=payload_hash,
            content_type="application/octet-stream",
            size_bytes=2048,
            storage_backend="local_fs",
            storage_key="blobs/sha256/aa/blob.bin",
        )
    )
    duplicate = sync_store.complete_blob_upload(
        SyncBlobObjectCreate(
            blob_id="blob-duplicate",
            dataset_id="dataset-1",
            owner_user_id="user-1",
            attachment_id="attachment-2",
            payload_hash=payload_hash,
            content_type="application/octet-stream",
            size_bytes=2048,
            storage_backend="local_fs",
            storage_key="blobs/sha256/aa/blob.bin",
        )
    )

    quota = sync_store.summarize_blob_quota("user-1")

    assert blob.status == "available"
    assert duplicate.blob_id == blob.blob_id
    assert quota.used_blob_bytes == 2048
    assert quota.reserved_blob_bytes == 0


def test_sync_envelope_schema_contains_m1_columns_and_indexes(sync_store: SyncV2Store):
    envelope_columns = {
        column["name"]
        for column in sync_store.db.backend.get_table_info("sync_envelopes")
    }
    object_state_columns = {
        column["name"]
        for column in sync_store.db.backend.get_table_info("sync_object_state")
    }
    indexes = {
        row["name"]
        for row in sync_store.db.execute("PRAGMA index_list(sync_envelopes)").rows
    }

    assert {
        "client_sequence",
        "base_server_cursor",
        "base_object_revision",
        "base_object_hash",
        "object_revision",
        "parent_id",
        "schema_version",
        "payload_json",
        "payload_hash",
        "created_at_client",
        "received_at_server",
        "deleted",
        "encryption_metadata_json",
        "apply_status",
        "apply_error_code",
        "apply_error_message",
        "applied_at",
        "client_profile_id",
    }.issubset(envelope_columns)
    assert {
        "dataset_id",
        "domain",
        "object_id",
        "object_revision",
        "object_hash",
        "latest_server_cursor",
        "deleted",
        "updated_at",
    }.issubset(object_state_columns)
    assert {
        "idx_sync_envelopes_dataset_sequence",
        "idx_sync_envelopes_dataset_domain_object",
        "idx_sync_envelopes_dataset_device_client_sequence",
        "idx_sync_envelopes_payload_hash",
        "idx_sync_envelopes_failed_apply",
    }.issubset(indexes)


def test_sync_database_migrates_pre_m1_sqlite_schema_before_index_creation(
    tmp_path: Path,
):
    db_path = tmp_path / "pre_m1_sync_v2.db"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE sync_envelopes (
                server_sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_id TEXT NOT NULL,
                domain TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                stable_key TEXT,
                operation TEXT NOT NULL,
                client_envelope_id TEXT NOT NULL,
                device_id TEXT,
                client_timestamp TEXT,
                server_timestamp TEXT NOT NULL,
                base_version TEXT,
                entity_version TEXT,
                dependency_json TEXT NOT NULL DEFAULT '[]',
                routing_metadata_json TEXT NOT NULL DEFAULT '{}',
                payload_ciphertext TEXT,
                payload_clear_json TEXT NOT NULL DEFAULT '{}',
                payload_hash TEXT,
                payload_size_bytes INTEGER,
                adapter_version INTEGER NOT NULL,
                status TEXT NOT NULL,
                UNIQUE (dataset_id, client_envelope_id)
            );
            """
        )

    db = SyncDatabase(sqlite_path=db_path)
    envelope_columns = {
        column["name"]
        for column in db.backend.get_table_info("sync_envelopes")
    }
    indexes = {
        row["name"]
        for row in db.execute("PRAGMA index_list(sync_envelopes)").rows
    }

    assert "client_sequence" in envelope_columns
    assert "apply_status" in envelope_columns
    assert "idx_sync_envelopes_dataset_device_client_sequence" in indexes
    assert "idx_sync_envelopes_failed_apply" in indexes


def test_sync_timestamps_are_timezone_aware_utc():
    timestamp = utcnow_iso()

    parsed = datetime.fromisoformat(timestamp)

    assert parsed.tzinfo is not None
    assert parsed.utcoffset() == timedelta(0)
    assert parsed.tzinfo == timezone.utc


def test_sync_store_facade_does_not_embed_sql_statements():
    source = inspect.getsource(store_module.SyncV2Store)

    assert "SELECT " not in source
    assert "INSERT " not in source
    assert "UPDATE " not in source
    assert "DELETE " not in source
    assert "CREATE TABLE" not in source
    assert "ALTER TABLE" not in source


def test_core_models_import_without_api_schema_module():
    code = """
import sys
from tldw_Server_API.app.core.Sync.v2.models import SyncKeyRecordCreate

assert "tldw_Server_API.app.api.v1.schemas.sync_v2_models" not in sys.modules
record = SyncKeyRecordCreate(
    key_record_id="key-1",
    dataset_id="dataset-1",
    user_id="user-1",
    key_purpose="dataset_recovery",
    wrapped_key_blob="wrapped:opaque",
)
assert record.user_id == "user-1"
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_device_upsert_is_idempotent(sync_store: SyncV2Store):
    first = sync_store.upsert_device(_device())
    second = sync_store.upsert_device(
        _device(
            display_name="Renamed Laptop",
            capabilities={"domains": ["notes.note", "chat.conversation"]},
        )
    )

    assert second.device_id == first.device_id
    assert second.registered_at == first.registered_at
    assert second.display_name == "Renamed Laptop"
    assert second.capabilities == {"domains": ["notes.note", "chat.conversation"]}
    assert second.last_seen_at >= first.last_seen_at


def test_device_upsert_rejects_cross_user_takeover(sync_store: SyncV2Store):
    sync_store.upsert_device(_device(device_id="device-shared", user_id="user-1"))

    with pytest.raises(SyncStoreError):
        sync_store.upsert_device(_device(device_id="device-shared", user_id="user-2"))

    assert [device.user_id for device in sync_store.list_devices_for_user("user-1")] == ["user-1"]
    assert sync_store.list_devices_for_user("user-2") == []


def test_dataset_enrollment_is_idempotent(sync_store: SyncV2Store):
    first = sync_store.enroll_dataset(_dataset())
    second = sync_store.enroll_dataset(
        _dataset(
            domains=["notes.note", "chat.conversation", "chat.message"],
            metadata={"label": "Updated"},
        )
    )
    fetched = sync_store.get_dataset("dataset-1")

    assert second.dataset_id == first.dataset_id
    assert second.created_at == first.created_at
    assert second.domains == ["notes.note", "chat.conversation", "chat.message"]
    assert second.metadata == {"label": "Updated"}
    assert fetched == second


def test_dataset_enrollment_rejects_cross_user_takeover(sync_store: SyncV2Store):
    sync_store.enroll_dataset(_dataset(dataset_id="dataset-shared", owner_user_id="user-1"))

    with pytest.raises(SyncStoreError):
        sync_store.enroll_dataset(_dataset(dataset_id="dataset-shared", owner_user_id="user-2"))

    dataset = sync_store.get_dataset("dataset-shared")
    assert dataset is not None
    assert dataset.owner_user_id == "user-1"


def test_dataset_enrollment_rejects_non_m1_domain_and_encryption_policy(
    sync_store: SyncV2Store,
):
    with pytest.raises(SyncInvalidDomainError):
        sync_store.enroll_dataset(_dataset(domains=["media"]))

    with pytest.raises(SyncStoreError):
        sync_store.enroll_dataset(
            _dataset(dataset_id="dataset-2", encryption_policy="client_private_v1")
        )


def test_get_dataset_can_be_scoped_by_owner(sync_store: SyncV2Store):
    dataset = sync_store.enroll_dataset(_dataset())

    assert sync_store.get_dataset("dataset-1", owner_user_id="user-1") == dataset
    assert sync_store.get_dataset("dataset-1", owner_user_id="user-2") is None


def test_get_or_create_default_personal_dataset_is_idempotent(sync_store: SyncV2Store):
    first = sync_store.get_or_create_default_personal_dataset("user-1")
    second = sync_store.get_or_create_default_personal_dataset("user-1")

    assert second.dataset_id == first.dataset_id
    assert second.created_at == first.created_at
    assert second.owner_user_id == "user-1"
    assert second.scope_type == "personal"
    assert second.encryption_policy == "server_trusted_v1"
    assert second.domains == [
        "notes.note",
        "chat.conversation",
        "chat.message",
        "attachment.ref",
    ]
    assert second.metadata["default_personal"] is True
    assert second.metadata["client_family"] == "chatbook"
    assert sync_store.list_datasets_for_user("user-1") == [second]


def test_insert_envelope_is_idempotent_by_dataset_and_client_envelope(sync_store: SyncV2Store):
    sync_store.enroll_dataset(_dataset())
    envelope = _envelope(client_envelope_id="env-1")

    first = sync_store.insert_envelope(envelope)
    second = sync_store.insert_envelope(envelope)

    assert second.server_cursor == first.server_cursor
    assert second.received_at_server == first.received_at_server
    assert sync_store.list_envelopes_after("dataset-1", 0) == [first]


def test_insert_envelope_idempotent_retry_ignores_mutable_apply_state(
    sync_store: SyncV2Store,
):
    sync_store.enroll_dataset(_dataset())
    envelope = _envelope(client_envelope_id="env-1")

    first = sync_store.insert_envelope(envelope)
    applied = sync_store.mark_envelope_apply_status(
        first.server_cursor,
        apply_status="applied",
    )
    retried = sync_store.insert_envelope(envelope)

    assert applied.apply_status == "applied"
    assert retried.server_cursor == first.server_cursor
    assert retried.apply_status == "applied"


def test_insert_envelope_idempotency_uses_envelope_key_after_other_insert(sync_store: SyncV2Store):
    sync_store.enroll_dataset(_dataset())
    envelope = _envelope(client_envelope_id="env-1")

    first = sync_store.insert_envelope(envelope)
    sync_store.insert_envelope(
        _envelope(
            client_envelope_id="env-2",
            object_id="note-2",
            payload_hash="sha256:note-2",
        )
    )
    duplicate = sync_store.insert_envelope(envelope)

    assert duplicate.server_cursor == first.server_cursor


def test_insert_envelope_persists_m1_fields_and_aliases(sync_store: SyncV2Store):
    sync_store.enroll_dataset(_dataset())

    stored = sync_store.insert_envelope(
        _envelope(
            client_sequence=1,
            base_server_cursor=8,
            base_object_revision=2,
            base_object_hash="sha256:note-v2",
            object_revision=3,
            parent_id="folder-1",
            schema_version=2,
            payload={"title": "Changed"},
            payload_hash="sha256:note-v3",
            deleted=True,
            encryption_metadata={"policy": "server_trusted_v1", "key": "server"},
        )
    )

    assert stored.server_cursor >= 1
    assert stored.server_sequence == stored.server_cursor
    assert stored.object_id == "note-1"
    assert stored.entity_id == "note-1"
    assert stored.base_server_cursor == 8
    assert stored.base_object_revision == 2
    assert stored.base_object_hash == "sha256:note-v2"
    assert stored.object_revision == 3
    assert stored.parent_id == "folder-1"
    assert stored.schema_version == 2
    assert stored.payload == {"title": "Changed"}
    assert stored.payload_clear == {"title": "Changed"}
    assert stored.payload_hash == "sha256:note-v3"
    assert stored.created_at_client == "2026-05-10T00:00:00+00:00"
    assert stored.received_at_server is not None
    assert stored.deleted is True
    assert stored.encryption_metadata == {"policy": "server_trusted_v1", "key": "server"}
    assert stored.apply_status == "pending"


def test_insert_envelope_is_idempotent_by_dataset_device_and_client_sequence(
    sync_store: SyncV2Store,
):
    sync_store.enroll_dataset(_dataset())

    first = sync_store.insert_envelope(_envelope(client_sequence=11))
    duplicate = sync_store.insert_envelope(
        _envelope(client_envelope_id="env-1-retry", client_sequence=11)
    )

    assert duplicate.server_cursor == first.server_cursor
    assert duplicate.client_envelope_id == first.client_envelope_id

    with pytest.raises(SyncIdempotencyConflictError):
        sync_store.insert_envelope(
            _envelope(
                client_envelope_id="env-1-conflict",
                client_sequence=11,
                payload_hash="sha256:changed",
            )
        )


def test_insert_envelope_rejects_duplicate_drift(sync_store: SyncV2Store):
    sync_store.enroll_dataset(_dataset())
    envelope = _envelope(client_envelope_id="env-1")

    sync_store.insert_envelope(envelope)

    with pytest.raises(SyncIdempotencyConflictError):
        sync_store.insert_envelope(
            _envelope(client_envelope_id="env-1", payload_hash="sha256:changed")
        )


def test_insert_envelope_rejects_domain_not_enrolled(sync_store: SyncV2Store):
    sync_store.enroll_dataset(_dataset(domains=["notes.note"]))

    with pytest.raises(SyncInvalidDomainError):
        sync_store.insert_envelope(
            _envelope(
                domain="chat.conversation",
                object_id="conversation-1",
                payload_hash="sha256:chat-1",
            )
        )


def test_insert_envelope_rejects_unsupported_operation_and_encryption_policy(
    sync_store: SyncV2Store,
):
    sync_store.enroll_dataset(_dataset())

    with pytest.raises(SyncStoreError):
        sync_store.insert_envelope(_envelope(operation="delete"))

    with pytest.raises(SyncStoreError):
        sync_store.insert_envelope(
            _envelope(encryption_metadata={"policy": "client_private_v1"})
        )


def test_insert_envelope_rejects_direct_core_payload_hash_bypass(
    sync_store: SyncV2Store,
):
    sync_store.enroll_dataset(_dataset())

    with pytest.raises(SyncStoreError):
        sync_store.insert_envelope(_envelope(payload_hash=None))

    with pytest.raises(SyncStoreError):
        sync_store.insert_envelope(
            _envelope(client_envelope_id="env-blank-hash", payload_hash="   ")
        )


def test_insert_envelope_rejects_direct_core_whole_object_base_metadata_bypass(
    sync_store: SyncV2Store,
):
    sync_store.enroll_dataset(_dataset())

    with pytest.raises(SyncStoreError):
        sync_store.insert_envelope(
            _envelope(
                client_envelope_id="env-update-no-base",
                object_revision=2,
            )
        )

    with pytest.raises(SyncStoreError):
        sync_store.insert_envelope(
            _envelope(
                client_envelope_id="env-partial-base",
                base_server_cursor=1,
            )
        )

    with pytest.raises(SyncStoreError):
        sync_store.insert_envelope(
            _envelope(
                client_envelope_id="env-tombstone-no-base",
                operation="tombstone",
                deleted=True,
                payload_hash="sha256:tombstone",
            )
        )


def test_insert_envelope_rejects_direct_core_chat_message_identity_and_hash_bypass(
    sync_store: SyncV2Store,
):
    sync_store.enroll_dataset(_dataset())

    with pytest.raises(SyncStoreError):
        sync_store.insert_envelope(
            _envelope(
                client_envelope_id="env-message-blank-id",
                domain="chat.message",
                operation="append",
                object_id="   ",
                parent_id="conversation-1",
                payload_hash="sha256:message",
            )
        )

    with pytest.raises(SyncStoreError):
        sync_store.insert_envelope(
            _envelope(
                client_envelope_id="env-message-blank-hash",
                domain="chat.message",
                operation="append",
                object_id="message-1",
                parent_id="conversation-1",
                payload_hash="",
            )
        )


def test_insert_envelope_rejects_direct_core_attachment_ref_metadata_bypass(
    sync_store: SyncV2Store,
):
    sync_store.enroll_dataset(_dataset())

    with pytest.raises(SyncStoreError):
        sync_store.insert_envelope(
            _envelope(
                client_envelope_id="env-attachment-ref",
                domain="attachment.ref",
                object_id="attachment-1",
                payload={"attachment_id": "attachment-1"},
                payload_hash="sha256:attachment-ref",
            )
        )


def test_list_envelopes_after_cursor_is_ordered_and_domain_filterable(sync_store: SyncV2Store):
    sync_store.enroll_dataset(_dataset())
    first = sync_store.insert_envelope(_envelope(client_envelope_id="env-1", object_id="note-1"))
    second = sync_store.insert_envelope(
        _envelope(
            client_envelope_id="env-2",
            domain="chat.conversation",
            object_id="conversation-1",
            payload_hash="sha256:chat-1",
        )
    )
    third = sync_store.insert_envelope(
        _envelope(client_envelope_id="env-3", object_id="note-2", payload_hash="sha256:note-2")
    )

    assert [row.server_cursor for row in sync_store.list_envelopes_after("dataset-1", first.server_cursor)] == [
        second.server_cursor,
        third.server_cursor,
    ]
    assert sync_store.list_envelopes_after("dataset-1", 0, domains=["notes.note"]) == [first, third]


def test_list_envelopes_after_filters_status_and_excluded_device_in_sql(
    sync_store: SyncV2Store,
):
    sync_store.enroll_dataset(_dataset())
    own = sync_store.insert_envelope(_envelope(client_envelope_id="own-accepted"))
    remote = sync_store.insert_envelope(
        _envelope(
            client_envelope_id="remote-accepted",
            object_id="note-remote",
            device_id="device-2",
            payload_hash="sha256:remote",
        )
    )
    sync_store.insert_envelope(
        _envelope(
            client_envelope_id="remote-conflict",
            object_id="note-conflict",
            device_id="device-2",
            payload_hash="sha256:conflict",
            status="conflict",
        )
    )

    visible = sync_store.list_envelopes_after(
        "dataset-1",
        0,
        status="accepted",
        exclude_device_id="device-1",
    )

    assert own not in visible
    assert remote in visible
    assert [envelope.client_envelope_id for envelope in visible] == ["remote-accepted"]


def test_apply_status_lifecycle_and_replay_listing(sync_store: SyncV2Store):
    sync_store.enroll_dataset(_dataset())
    first = sync_store.insert_envelope(_envelope(client_envelope_id="env-1"))
    second = sync_store.insert_envelope(
        _envelope(
            client_envelope_id="env-2",
            object_id="note-2",
            client_sequence=2,
            payload_hash="sha256:note-2",
        )
    )
    sync_store.insert_envelope(
        _envelope(
            client_envelope_id="env-conflict",
            object_id="note-conflict",
            client_sequence=3,
            payload_hash="sha256:note-conflict",
            status="conflict",
            apply_status="conflict",
        )
    )

    applied = sync_store.mark_envelope_apply_status(
        first.server_cursor,
        apply_status="applied",
    )
    failed = sync_store.mark_envelope_apply_status(
        second.server_cursor,
        apply_status="failed",
        apply_error_code="projection_write_failed",
        apply_error_message="projection database is locked",
    )

    assert applied.apply_status == "applied"
    assert applied.applied_at is not None
    assert failed.apply_status == "failed"
    assert failed.apply_error_code == "projection_write_failed"
    assert failed.apply_error_message == "projection database is locked"
    assert sync_store.list_failed_applies("dataset-1") == [failed]
    assert sync_store.list_accepted_envelopes_for_replay("dataset-1", since_cursor=0) == [
        applied,
        failed,
    ]


def test_device_cursor_upsert_and_fetch(sync_store: SyncV2Store):
    sync_store.enroll_dataset(_dataset())

    cursor = sync_store.update_device_cursor(
        SyncDeviceCursor(
            dataset_id="dataset-1",
            device_id="device-1",
            domain="notes.note",
            last_pulled_sequence=42,
        )
    )
    fetched = sync_store.get_device_cursor("dataset-1", "device-1", "notes.note")

    assert fetched == cursor
    assert fetched.last_pulled_sequence == 42


def test_device_cursor_rejects_missing_dataset_and_unenrolled_domain(sync_store: SyncV2Store):
    with pytest.raises(SyncDatasetNotFoundError):
        sync_store.update_device_cursor(
            SyncDeviceCursor(
                dataset_id="missing-dataset",
                device_id="device-1",
                domain="notes.note",
                last_pulled_sequence=1,
            )
        )

    sync_store.enroll_dataset(_dataset(domains=["notes.note"]))

    with pytest.raises(SyncInvalidDomainError):
        sync_store.update_device_cursor(
            SyncDeviceCursor(
                dataset_id="dataset-1",
                device_id="device-1",
                domain="chat.conversation",
                last_pulled_sequence=1,
            )
        )


def test_conflict_insert_list_and_resolve_lifecycle(sync_store: SyncV2Store):
    sync_store.enroll_dataset(_dataset())

    inserted = sync_store.insert_conflict(_conflict())

    assert inserted.status == "unresolved"
    assert sync_store.get_conflict("conflict-1") == inserted
    assert sync_store.get_conflict("missing-conflict") is None
    assert sync_store.list_conflicts("dataset-1") == [inserted]
    assert sync_store.list_conflicts("dataset-1", status="resolved") == []

    resolved = sync_store.resolve_conflict(
        "conflict-1",
        status="resolved",
        resolved_by_envelope_id="env-resolution",
        resolved_by_device_id="device-1",
        resolution_action="merge",
        resolution_notes="Merged locally",
    )

    assert resolved.status == "resolved"
    assert resolved.resolved_by_envelope_id == "env-resolution"
    assert resolved.resolved_at is not None
    assert sync_store.list_conflicts("dataset-1", status="unresolved") == []
    assert sync_store.list_conflicts("dataset-1", status="resolved") == [resolved]


def test_resolve_conflict_preserves_existing_resolution_metadata(sync_store: SyncV2Store):
    sync_store.enroll_dataset(_dataset())
    sync_store.insert_conflict(_conflict())

    first = sync_store.resolve_conflict(
        "conflict-1",
        server_cursor=10,
        status="resolved",
        resolved_by_envelope_id="srv_env_first",
        resolved_by_device_id="device-1",
        resolution_action="overwrite",
        resolution_notes="first decision",
    )
    replayed = sync_store.resolve_conflict(
        "conflict-1",
        server_cursor=10,
        status="resolved",
        resolved_by_envelope_id="srv_env_first",
        resolved_by_device_id="device-1",
        resolution_action="overwrite",
        resolution_notes="first decision",
    )

    assert replayed == first

    with pytest.raises(SyncStoreError, match="already resolved"):
        sync_store.resolve_conflict(
            "conflict-1",
            server_cursor=11,
            status="resolved",
            resolved_by_envelope_id="srv_env_second",
            resolved_by_device_id="device-2",
            resolution_action="duplicate_rename",
            resolution_notes="second decision",
        )

    assert sync_store.get_conflict("conflict-1") == first


def test_conflict_resolution_claim_and_finalize_lifecycle(sync_store: SyncV2Store):
    sync_store.enroll_dataset(_dataset())
    sync_store.insert_conflict(_conflict())

    claimed = sync_store.claim_conflict_resolution(
        "conflict-1",
        dataset_id="dataset-1",
        resolved_by_device_id="device-1",
        resolution_action="overwrite",
        resolution_notes="first decision",
    )

    assert claimed.status == "unresolved"
    assert claimed.resolution_action == "overwrite"
    assert claimed.resolved_by_device_id == "device-1"
    assert claimed.resolution_notes == "first decision"
    assert claimed.resolved_by_envelope_id is None

    with pytest.raises(SyncStoreError, match="already claimed"):
        sync_store.claim_conflict_resolution(
            "conflict-1",
            dataset_id="dataset-1",
            resolved_by_device_id="device-2",
            resolution_action="duplicate_rename",
            resolution_notes="second decision",
        )

    assert sync_store.get_conflict("conflict-1") == claimed

    resolved = sync_store.resolve_conflict(
        "conflict-1",
        dataset_id="dataset-1",
        server_cursor=10,
        status="resolved",
        resolved_by_envelope_id="srv_env_first",
        resolved_by_device_id="device-1",
        resolution_action="overwrite",
        resolution_notes="first decision",
    )
    replayed = sync_store.resolve_conflict(
        "conflict-1",
        dataset_id="dataset-1",
        server_cursor=10,
        status="resolved",
        resolved_by_envelope_id="srv_env_first",
        resolved_by_device_id="device-1",
        resolution_action="overwrite",
        resolution_notes="first decision",
    )

    assert resolved.status == "resolved"
    assert resolved.resolution_action == "overwrite"
    assert resolved.resolved_by_envelope_id == "srv_env_first"
    assert replayed == resolved

    with pytest.raises(SyncStoreError, match="already resolved"):
        sync_store.resolve_conflict(
            "conflict-1",
            dataset_id="dataset-1",
            server_cursor=11,
            status="resolved",
            resolved_by_envelope_id="srv_env_second",
            resolved_by_device_id="device-2",
            resolution_action="duplicate_rename",
            resolution_notes="second decision",
        )

    assert sync_store.get_conflict("conflict-1") == resolved


def test_conflict_rejects_missing_dataset_and_unenrolled_domain(sync_store: SyncV2Store):
    with pytest.raises(SyncDatasetNotFoundError):
        sync_store.insert_conflict(_conflict(dataset_id="missing-dataset"))

    sync_store.enroll_dataset(_dataset(domains=["notes.note"]))

    with pytest.raises(SyncInvalidDomainError):
        sync_store.insert_conflict(_conflict(domain="chat.conversation"))


def test_key_records_store_wrapped_blobs_without_plaintext_keys(sync_store: SyncV2Store):
    sync_store.enroll_dataset(_dataset())

    stored = sync_store.store_key_record(_key_record())
    duplicate = sync_store.store_key_record(_key_record())
    records = sync_store.list_key_records("dataset-1", user_id="user-1")
    columns = {column["name"] for column in sync_store.db.backend.get_table_info("sync_key_records")}

    assert duplicate.key_record_id == stored.key_record_id
    assert duplicate.created_at == stored.created_at
    assert duplicate.recovery_hint == "personal laptop"
    assert duplicate.user_id == "user-1"
    assert records == [duplicate]
    assert records[0].user_id == "user-1"
    assert records[0].wrapped_key_blob == "wrapped:opaque"
    assert not hasattr(records[0], "plaintext_key")
    assert "user_id" in columns
    assert all("plaintext" not in column_name.lower() for column_name in columns)

    with pytest.raises(TypeError):
        sync_store.list_key_records("dataset-1")

    with pytest.raises(SyncStoreError):
        sync_store.list_key_records("dataset-1", user_id="")


def test_key_records_are_scoped_by_user(sync_store: SyncV2Store):
    sync_store.enroll_dataset(_dataset())

    sync_store.store_key_record(_key_record(key_record_id="key-user-1", user_id="user-1"))
    sync_store.store_key_record(_key_record(key_record_id="key-user-2", user_id="user-2"))

    user_1_records = sync_store.list_key_records("dataset-1", user_id="user-1")
    user_2_records = sync_store.list_key_records("dataset-1", user_id="user-2")

    assert [record.key_record_id for record in user_1_records] == ["key-user-1"]
    assert [record.key_record_id for record in user_2_records] == ["key-user-2"]


def test_key_record_rejects_missing_dataset(sync_store: SyncV2Store):
    with pytest.raises(SyncDatasetNotFoundError):
        sync_store.store_key_record(_key_record(dataset_id="missing-dataset"))


def test_key_record_duplicate_drift_raises(sync_store: SyncV2Store):
    sync_store.enroll_dataset(_dataset())

    sync_store.store_key_record(_key_record())

    with pytest.raises(SyncIdempotencyConflictError):
        sync_store.store_key_record(_key_record(wrapped_key_blob="wrapped:changed"))

    with pytest.raises(SyncIdempotencyConflictError):
        sync_store.store_key_record(_key_record(user_id="user-2"))


def test_attachment_store_is_idempotent_and_keeps_ciphertext_out_of_manifest(
    sync_store: SyncV2Store,
):
    sync_store.enroll_dataset(_dataset())

    stored = sync_store.store_attachment(_attachment())
    duplicate = sync_store.store_attachment(_attachment())
    stats = sync_store.summarize_restore_manifest_dataset(
        "dataset-1",
        user_id="user-1",
        domains=["attachment.ref"],
    )

    assert stored.stored is True
    assert duplicate.stored is False
    assert duplicate.attachment_id == stored.attachment_id
    assert duplicate.payload_hash == "sha256:attachment"
    assert duplicate.payload_ciphertext == "ciphertext:attachment"
    assert stats.attachment_availability == {"available": 1}
    assert stats.attachment_size_classes == {"small": 1}
    assert "ciphertext:attachment" not in repr(stats)


def test_attachment_store_rejects_duplicate_drift_and_unenrolled_domain(
    sync_store: SyncV2Store,
):
    sync_store.enroll_dataset(_dataset(domains=["attachment.ref"]))
    sync_store.store_attachment(_attachment())

    with pytest.raises(SyncIdempotencyConflictError):
        sync_store.store_attachment(_attachment(payload_hash="sha256:changed"))

    with pytest.raises(SyncInvalidDomainError):
        sync_store.store_attachment(
            _attachment(
                attachment_id="attachment-chat",
                domain="chat.conversation",
                object_id="conversation-1",
                payload_hash="sha256:chat-attachment",
            )
        )


def test_attachment_store_restore_summary_respects_domain_filter(
    sync_store: SyncV2Store,
):
    sync_store.enroll_dataset(_dataset(domains=["attachment.ref", "chat.conversation"]))
    sync_store.store_attachment(_attachment(domain="attachment.ref", size_bytes=512))
    sync_store.store_attachment(
        _attachment(
            attachment_id="attachment-chat",
            domain="chat.conversation",
            object_id="conversation-1",
            size_bytes=2_097_152,
            payload_hash="sha256:chat-attachment",
        )
    )

    notes_stats = sync_store.summarize_restore_manifest_dataset(
        "dataset-1",
        user_id="user-1",
        domains=["attachment.ref"],
    )
    all_stats = sync_store.summarize_restore_manifest_dataset(
        "dataset-1",
        user_id="user-1",
        domains=["attachment.ref", "chat.conversation"],
    )

    assert notes_stats.attachment_availability == {"available": 1}
    assert notes_stats.attachment_size_classes == {"small": 1}
    assert all_stats.attachment_availability == {"available": 2}
    assert all_stats.attachment_size_classes == {"medium": 1, "small": 1}
