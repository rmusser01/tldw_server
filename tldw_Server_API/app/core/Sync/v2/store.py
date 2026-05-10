from __future__ import annotations

"""Core-facing Sync v2 store facade."""

from collections.abc import Sequence

from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase

from .models import (
    ConflictStatus,
    SyncConflict,
    SyncConflictCreate,
    SyncDataset,
    SyncDatasetCreate,
    SyncDevice,
    SyncDeviceCursor,
    SyncDeviceUpsert,
    SyncDomain,
    SyncEnvelope,
    SyncEnvelopeCreate,
    SyncKeyRecord,
    SyncKeyRecordCreate,
)


class SyncV2Store:
    """Core Sync v2 persistence interface backed by DB_Management."""

    def __init__(self, db: SyncDatabase) -> None:
        self.db = db

    def upsert_device(self, device: SyncDeviceUpsert) -> SyncDevice:
        return self.db.upsert_device(device)

    def enroll_dataset(self, dataset: SyncDatasetCreate) -> SyncDataset:
        return self.db.enroll_dataset(dataset)

    def get_dataset(
        self,
        dataset_id: str,
        *,
        owner_user_id: str | None = None,
    ) -> SyncDataset | None:
        return self.db.get_dataset(dataset_id, owner_user_id=owner_user_id)

    def list_datasets_for_user(self, user_id: str) -> list[SyncDataset]:
        return self.db.list_datasets_for_user(user_id)

    def list_devices_for_user(self, user_id: str) -> list[SyncDevice]:
        return self.db.list_devices_for_user(user_id)

    def insert_envelope(self, envelope: SyncEnvelopeCreate) -> SyncEnvelope:
        return self.db.insert_envelope(envelope)

    def list_envelopes_after(
        self,
        dataset_id: str,
        since_sequence: int,
        *,
        limit: int = 100,
        domains: Sequence[SyncDomain] | None = None,
    ) -> list[SyncEnvelope]:
        return self.db.list_envelopes_after(
            dataset_id,
            since_sequence,
            limit=limit,
            domains=domains,
        )

    def list_envelopes_for_entity(
        self,
        dataset_id: str,
        domain: SyncDomain,
        *,
        entity_id: str | None = None,
        stable_key: str | None = None,
        limit: int = 100,
    ) -> list[SyncEnvelope]:
        return self.db.list_envelopes_for_entity(
            dataset_id,
            domain,
            entity_id=entity_id,
            stable_key=stable_key,
            limit=limit,
        )

    def update_device_cursor(self, cursor: SyncDeviceCursor) -> SyncDeviceCursor:
        return self.db.update_device_cursor(cursor)

    def get_device_cursor(
        self,
        dataset_id: str,
        device_id: str,
        domain: SyncDomain,
    ) -> SyncDeviceCursor | None:
        return self.db.get_device_cursor(dataset_id, device_id, domain)

    def insert_conflict(self, conflict: SyncConflictCreate) -> SyncConflict:
        return self.db.insert_conflict(conflict)

    def list_conflicts(
        self,
        dataset_id: str,
        *,
        status: ConflictStatus | None = None,
    ) -> list[SyncConflict]:
        return self.db.list_conflicts(dataset_id, status=status)

    def get_unresolved_conflict_for_envelope(
        self,
        dataset_id: str,
        *,
        local_envelope_id: str,
        server_sequence: int | None = None,
    ) -> SyncConflict | None:
        return self.db.get_unresolved_conflict_for_envelope(
            dataset_id,
            local_envelope_id=local_envelope_id,
            server_sequence=server_sequence,
        )

    def resolve_conflict(
        self,
        conflict_id: str,
        *,
        status: ConflictStatus = "resolved",
        resolved_by_envelope_id: str | None = None,
        resolved_by_device_id: str | None = None,
        resolution_action: str | None = None,
        resolution_notes: str | None = None,
    ) -> SyncConflict:
        return self.db.resolve_conflict(
            conflict_id,
            status=status,
            resolved_by_envelope_id=resolved_by_envelope_id,
            resolved_by_device_id=resolved_by_device_id,
            resolution_action=resolution_action,
            resolution_notes=resolution_notes,
        )

    def store_key_record(self, record: SyncKeyRecordCreate) -> SyncKeyRecord:
        return self.db.store_key_record(record)

    def list_key_records(
        self,
        dataset_id: str,
        *,
        user_id: str,
        device_id: str | None = None,
        key_purpose: str | None = None,
    ) -> list[SyncKeyRecord]:
        return self.db.list_key_records(
            dataset_id,
            user_id=user_id,
            device_id=device_id,
            key_purpose=key_purpose,
        )


__all__ = ["SyncV2Store"]
