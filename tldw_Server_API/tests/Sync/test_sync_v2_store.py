from __future__ import annotations

from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
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


def test_insert_envelope_is_idempotent_by_dataset_and_client_envelope(sync_store: SyncV2Store):
    sync_store.enroll_dataset(_dataset())
    envelope = _envelope(client_envelope_id="env-1")

    first = sync_store.insert_envelope(envelope)
    second = sync_store.insert_envelope(envelope)

    assert second.server_sequence == first.server_sequence
    assert second.server_timestamp == first.server_timestamp
    assert sync_store.list_envelopes_after("dataset-1", 0) == [first]


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


def test_conflict_insert_list_and_resolve_lifecycle(sync_store: SyncV2Store):
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


def test_key_records_store_wrapped_blobs_without_plaintext_keys(sync_store: SyncV2Store):
    stored = sync_store.store_key_record(_key_record())
    duplicate = sync_store.store_key_record(_key_record(recovery_hint="updated hint"))
    records = sync_store.list_key_records("dataset-1")
    columns = {column["name"] for column in sync_store.db.backend.get_table_info("sync_key_records")}

    assert duplicate.key_record_id == stored.key_record_id
    assert duplicate.created_at == stored.created_at
    assert duplicate.recovery_hint == "updated hint"
    assert records == [duplicate]
    assert records[0].wrapped_key_blob == "wrapped:opaque"
    assert not hasattr(records[0], "plaintext_key")
    assert all("plaintext" not in column_name.lower() for column_name in columns)
