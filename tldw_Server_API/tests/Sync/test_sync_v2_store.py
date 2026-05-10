from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path

import pytest

import tldw_Server_API.app.core.Sync.v2.store as store_module
from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
from tldw_Server_API.app.core.Sync.v2.errors import (
    SyncDatasetNotFoundError,
    SyncIdempotencyConflictError,
    SyncInvalidDomainError,
    SyncStoreError,
)
from tldw_Server_API.app.core.Sync.v2.models import (
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
        "capabilities": {"domains": ["notes"]},
    }
    payload.update(overrides)
    return SyncDeviceUpsert(**payload)


def _dataset(**overrides) -> SyncDatasetCreate:
    payload = {
        "dataset_id": "dataset-1",
        "owner_user_id": "user-1",
        "scope_type": "personal",
        "encryption_policy": "client_private_v1",
        "domains": ["notes", "chat"],
        "metadata": {"label": "Personal research"},
    }
    payload.update(overrides)
    return SyncDatasetCreate(**payload)


def _envelope(**overrides) -> SyncEnvelopeCreate:
    payload = {
        "dataset_id": "dataset-1",
        "client_envelope_id": "env-1",
        "domain": "notes",
        "entity_id": "note-1",
        "stable_key": "note:note-1",
        "operation": "upsert",
        "device_id": "device-1",
        "client_timestamp": "2026-05-10T00:00:00+00:00",
        "base_version": None,
        "entity_version": "v1",
        "dependencies": [{"entity_id": "source-1"}],
        "routing_metadata": {"entity_kind": "note"},
        "payload_ciphertext": "ciphertext:opaque",
        "payload_clear": {"status": "active"},
        "payload_hash": "sha256:note-1",
        "payload_size_bytes": 24,
        "adapter_version": 1,
        "status": "accepted",
    }
    payload.update(overrides)
    return SyncEnvelopeCreate(**payload)


def _conflict(**overrides) -> SyncConflictCreate:
    payload = {
        "conflict_id": "conflict-1",
        "dataset_id": "dataset-1",
        "domain": "notes",
        "entity_id": "note-1",
        "conflict_type": "version_divergence",
        "base_envelope_id": "env-base",
        "local_envelope_id": "env-local",
        "remote_envelope_id": "env-remote",
        "server_sequence": 3,
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


def test_sync_database_bootstrap_creates_required_tables(sync_store: SyncV2Store):
    required_tables = {
        "sync_devices",
        "sync_datasets",
        "sync_domain_state",
        "sync_envelopes",
        "sync_device_cursors",
        "sync_conflicts",
        "sync_key_records",
    }

    for table_name in required_tables:
        assert sync_store.db.backend.table_exists(table_name)


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
        _device(display_name="Renamed Laptop", capabilities={"domains": ["notes", "chat"]})
    )

    assert second.device_id == first.device_id
    assert second.registered_at == first.registered_at
    assert second.display_name == "Renamed Laptop"
    assert second.capabilities == {"domains": ["notes", "chat"]}
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
        _dataset(domains=["notes", "chat", "workspaces"], metadata={"label": "Updated"})
    )
    fetched = sync_store.get_dataset("dataset-1")

    assert second.dataset_id == first.dataset_id
    assert second.created_at == first.created_at
    assert second.domains == ["notes", "chat", "workspaces"]
    assert second.metadata == {"label": "Updated"}
    assert fetched == second


def test_dataset_enrollment_rejects_cross_user_takeover(sync_store: SyncV2Store):
    sync_store.enroll_dataset(_dataset(dataset_id="dataset-shared", owner_user_id="user-1"))

    with pytest.raises(SyncStoreError):
        sync_store.enroll_dataset(_dataset(dataset_id="dataset-shared", owner_user_id="user-2"))

    dataset = sync_store.get_dataset("dataset-shared")
    assert dataset is not None
    assert dataset.owner_user_id == "user-1"


def test_get_dataset_can_be_scoped_by_owner(sync_store: SyncV2Store):
    dataset = sync_store.enroll_dataset(_dataset())

    assert sync_store.get_dataset("dataset-1", owner_user_id="user-1") == dataset
    assert sync_store.get_dataset("dataset-1", owner_user_id="user-2") is None


def test_insert_envelope_is_idempotent_by_dataset_and_client_envelope(sync_store: SyncV2Store):
    sync_store.enroll_dataset(_dataset())
    envelope = _envelope(client_envelope_id="env-1")

    first = sync_store.insert_envelope(envelope)
    second = sync_store.insert_envelope(envelope)

    assert second.server_sequence == first.server_sequence
    assert second.server_timestamp == first.server_timestamp
    assert sync_store.list_envelopes_after("dataset-1", 0) == [first]


def test_insert_envelope_idempotency_uses_envelope_key_after_other_insert(sync_store: SyncV2Store):
    sync_store.enroll_dataset(_dataset())
    envelope = _envelope(client_envelope_id="env-1")

    first = sync_store.insert_envelope(envelope)
    sync_store.insert_envelope(
        _envelope(
            client_envelope_id="env-2",
            entity_id="note-2",
            payload_hash="sha256:note-2",
        )
    )
    duplicate = sync_store.insert_envelope(envelope)

    assert duplicate.server_sequence == first.server_sequence


def test_insert_envelope_rejects_duplicate_drift(sync_store: SyncV2Store):
    sync_store.enroll_dataset(_dataset())
    envelope = _envelope(client_envelope_id="env-1")

    sync_store.insert_envelope(envelope)

    with pytest.raises(SyncIdempotencyConflictError):
        sync_store.insert_envelope(
            _envelope(client_envelope_id="env-1", payload_hash="sha256:changed")
        )


def test_insert_envelope_rejects_domain_not_enrolled(sync_store: SyncV2Store):
    sync_store.enroll_dataset(_dataset(domains=["notes"]))

    with pytest.raises(SyncInvalidDomainError):
        sync_store.insert_envelope(
            _envelope(
                domain="chat",
                stable_key="chat:conversation-1",
                payload_hash="sha256:chat-1",
            )
        )


def test_list_envelopes_after_cursor_is_ordered_and_domain_filterable(sync_store: SyncV2Store):
    sync_store.enroll_dataset(_dataset())
    first = sync_store.insert_envelope(_envelope(client_envelope_id="env-1", entity_id="note-1"))
    second = sync_store.insert_envelope(
        _envelope(
            client_envelope_id="env-2",
            domain="chat",
            entity_id="conversation-1",
            stable_key="chat:conversation-1",
            payload_hash="sha256:chat-1",
        )
    )
    third = sync_store.insert_envelope(
        _envelope(client_envelope_id="env-3", entity_id="note-2", payload_hash="sha256:note-2")
    )

    assert [row.server_sequence for row in sync_store.list_envelopes_after("dataset-1", first.server_sequence)] == [
        second.server_sequence,
        third.server_sequence,
    ]
    assert sync_store.list_envelopes_after("dataset-1", 0, domains=["notes"]) == [first, third]


def test_device_cursor_upsert_and_fetch(sync_store: SyncV2Store):
    sync_store.enroll_dataset(_dataset())

    cursor = sync_store.update_device_cursor(
        SyncDeviceCursor(
            dataset_id="dataset-1",
            device_id="device-1",
            domain="notes",
            last_pulled_sequence=42,
        )
    )
    fetched = sync_store.get_device_cursor("dataset-1", "device-1", "notes")

    assert fetched == cursor
    assert fetched.last_pulled_sequence == 42


def test_device_cursor_rejects_missing_dataset_and_unenrolled_domain(sync_store: SyncV2Store):
    with pytest.raises(SyncDatasetNotFoundError):
        sync_store.update_device_cursor(
            SyncDeviceCursor(
                dataset_id="missing-dataset",
                device_id="device-1",
                domain="notes",
                last_pulled_sequence=1,
            )
        )

    sync_store.enroll_dataset(_dataset(domains=["notes"]))

    with pytest.raises(SyncInvalidDomainError):
        sync_store.update_device_cursor(
            SyncDeviceCursor(
                dataset_id="dataset-1",
                device_id="device-1",
                domain="chat",
                last_pulled_sequence=1,
            )
        )


def test_conflict_insert_list_and_resolve_lifecycle(sync_store: SyncV2Store):
    sync_store.enroll_dataset(_dataset())

    inserted = sync_store.insert_conflict(_conflict())

    assert inserted.status == "unresolved"
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


def test_conflict_rejects_missing_dataset_and_unenrolled_domain(sync_store: SyncV2Store):
    with pytest.raises(SyncDatasetNotFoundError):
        sync_store.insert_conflict(_conflict(dataset_id="missing-dataset"))

    sync_store.enroll_dataset(_dataset(domains=["notes"]))

    with pytest.raises(SyncInvalidDomainError):
        sync_store.insert_conflict(_conflict(domain="chat"))


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
