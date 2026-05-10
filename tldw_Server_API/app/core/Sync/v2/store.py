from __future__ import annotations

"""Core-facing Sync v2 store facade."""

import json
from collections.abc import Sequence
from typing import Any

from tldw_Server_API.app.api.v1.schemas.sync_v2_models import ConflictStatus, SyncDomain
from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase, decode_json, encode_json, utcnow_iso

from .errors import SyncConflictNotFoundError, SyncDatasetNotFoundError
from .models import (
    SyncConflict,
    SyncConflictCreate,
    SyncDataset,
    SyncDatasetCreate,
    SyncDevice,
    SyncDeviceCursor,
    SyncDeviceUpsert,
    SyncEnvelope,
    SyncEnvelopeCreate,
    SyncKeyRecord,
    SyncKeyRecordCreate,
)


def _first(result: Any) -> dict[str, Any] | None:
    rows = getattr(result, "rows", None) or []
    return rows[0] if rows else None


def _version_to_storage(value: str | int | None) -> str | None:
    if value is None:
        return None
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _version_from_storage(value: str | None) -> str | int | None:
    if value is None:
        return None
    try:
        decoded = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return value
    if isinstance(decoded, (str, int)) or decoded is None:
        return decoded
    return str(decoded)


def _device_from_row(row: dict[str, Any]) -> SyncDevice:
    return SyncDevice(
        device_id=row["device_id"],
        user_id=row["user_id"],
        display_name=row["display_name"],
        client_type=row["client_type"],
        client_version=row.get("client_version"),
        capabilities=decode_json(row.get("capabilities_json"), default={}),
        registered_at=row["registered_at"],
        last_seen_at=row["last_seen_at"],
        revoked_at=row.get("revoked_at"),
    )


def _dataset_from_row(row: dict[str, Any]) -> SyncDataset:
    return SyncDataset(
        dataset_id=row["dataset_id"],
        owner_user_id=row["owner_user_id"],
        scope_type=row["scope_type"],
        encryption_policy=row["encryption_policy"],
        domains=decode_json(row.get("domain_set_json"), default=[]),
        workspace_id=row.get("workspace_id"),
        metadata=decode_json(row.get("metadata_json"), default={}),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        archived_at=row.get("archived_at"),
    )


def _envelope_from_row(row: dict[str, Any]) -> SyncEnvelope:
    return SyncEnvelope(
        server_sequence=int(row["server_sequence"]),
        dataset_id=row["dataset_id"],
        client_envelope_id=row["client_envelope_id"],
        domain=row["domain"],
        entity_id=row["entity_id"],
        operation=row["operation"],
        adapter_version=int(row["adapter_version"]),
        server_timestamp=row["server_timestamp"],
        device_id=row.get("device_id"),
        stable_key=row.get("stable_key"),
        client_timestamp=row.get("client_timestamp"),
        base_version=_version_from_storage(row.get("base_version")),
        entity_version=_version_from_storage(row.get("entity_version")),
        dependencies=decode_json(row.get("dependency_json"), default=[]),
        routing_metadata=decode_json(row.get("routing_metadata_json"), default={}),
        payload_ciphertext=row.get("payload_ciphertext"),
        payload_clear=decode_json(row.get("payload_clear_json"), default={}),
        payload_hash=row.get("payload_hash"),
        payload_size_bytes=(
            int(row["payload_size_bytes"])
            if row.get("payload_size_bytes") is not None
            else None
        ),
        status=row["status"],
    )


def _cursor_from_row(row: dict[str, Any]) -> SyncDeviceCursor:
    return SyncDeviceCursor(
        dataset_id=row["dataset_id"],
        device_id=row["device_id"],
        domain=row["domain"],
        last_pulled_sequence=int(row["last_pulled_sequence"]),
        updated_at=row["updated_at"],
    )


def _conflict_from_row(row: dict[str, Any]) -> SyncConflict:
    return SyncConflict(
        conflict_id=row["conflict_id"],
        dataset_id=row["dataset_id"],
        domain=row["domain"],
        entity_id=row["entity_id"],
        conflict_type=row["conflict_type"],
        status=row["status"],
        base_envelope_id=row.get("base_envelope_id"),
        local_envelope_id=row.get("local_envelope_id"),
        remote_envelope_id=row.get("remote_envelope_id"),
        server_sequence=(
            int(row["server_sequence"])
            if row.get("server_sequence") is not None
            else None
        ),
        metadata=decode_json(row.get("metadata_json"), default={}),
        created_at=row["created_at"],
        resolved_at=row.get("resolved_at"),
        resolved_by_envelope_id=row.get("resolved_by_envelope_id"),
        resolved_by_device_id=row.get("resolved_by_device_id"),
        resolution_action=row.get("resolution_action"),
        resolution_notes=row.get("resolution_notes"),
    )


def _key_record_from_row(row: dict[str, Any]) -> SyncKeyRecord:
    return SyncKeyRecord(
        key_record_id=row["key_record_id"],
        dataset_id=row["dataset_id"],
        device_id=row.get("device_id"),
        key_purpose=row["key_purpose"],
        wrapped_key_blob=row["wrapped_key_blob"],
        kdf_metadata=decode_json(row.get("kdf_metadata_json"), default={}),
        recovery_hint=row.get("recovery_hint"),
        rotation_of_key_record_id=row.get("rotation_of_key_record_id"),
        created_at=row["created_at"],
        revoked_at=row.get("revoked_at"),
    )


class SyncV2Store:
    """Core Sync v2 persistence interface backed by DB_Management."""

    def __init__(self, db: SyncDatabase) -> None:
        self.db = db

    def upsert_device(self, device: SyncDeviceUpsert) -> SyncDevice:
        now = utcnow_iso()
        with self.db.backend.transaction() as conn:
            existing = _first(
                self.db.execute(
                    "SELECT * FROM sync_devices WHERE device_id = ?",
                    (device.device_id,),
                    connection=conn,
                )
            )
            if existing:
                self.db.execute(
                    """
                    UPDATE sync_devices
                       SET user_id = ?,
                           display_name = ?,
                           client_type = ?,
                           client_version = ?,
                           capabilities_json = ?,
                           last_seen_at = ?,
                           revoked_at = ?
                     WHERE device_id = ?
                    """,
                    (
                        device.user_id,
                        device.display_name,
                        device.client_type,
                        device.client_version,
                        encode_json(device.capabilities, default={}),
                        now,
                        device.revoked_at,
                        device.device_id,
                    ),
                    connection=conn,
                )
            else:
                self.db.execute(
                    """
                    INSERT INTO sync_devices (
                        device_id, user_id, display_name, client_type, client_version,
                        capabilities_json, registered_at, last_seen_at, revoked_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        device.device_id,
                        device.user_id,
                        device.display_name,
                        device.client_type,
                        device.client_version,
                        encode_json(device.capabilities, default={}),
                        now,
                        now,
                        device.revoked_at,
                    ),
                    connection=conn,
                )
            row = _first(
                self.db.execute(
                    "SELECT * FROM sync_devices WHERE device_id = ?",
                    (device.device_id,),
                    connection=conn,
                )
            )
        return _device_from_row(row)

    def enroll_dataset(self, dataset: SyncDatasetCreate) -> SyncDataset:
        now = utcnow_iso()
        domains_json = encode_json(dataset.domains, default=[])
        metadata_json = encode_json(dataset.metadata, default={})
        with self.db.backend.transaction() as conn:
            existing = _first(
                self.db.execute(
                    "SELECT * FROM sync_datasets WHERE dataset_id = ?",
                    (dataset.dataset_id,),
                    connection=conn,
                )
            )
            if existing:
                self.db.execute(
                    """
                    UPDATE sync_datasets
                       SET owner_user_id = ?,
                           workspace_id = ?,
                           scope_type = ?,
                           encryption_policy = ?,
                           domain_set_json = ?,
                           metadata_json = ?,
                           updated_at = ?,
                           archived_at = ?
                     WHERE dataset_id = ?
                    """,
                    (
                        dataset.owner_user_id,
                        dataset.workspace_id,
                        dataset.scope_type,
                        dataset.encryption_policy,
                        domains_json,
                        metadata_json,
                        now,
                        dataset.archived_at,
                        dataset.dataset_id,
                    ),
                    connection=conn,
                )
            else:
                self.db.execute(
                    """
                    INSERT INTO sync_datasets (
                        dataset_id, owner_user_id, workspace_id, scope_type,
                        encryption_policy, domain_set_json, metadata_json,
                        created_at, updated_at, archived_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        dataset.dataset_id,
                        dataset.owner_user_id,
                        dataset.workspace_id,
                        dataset.scope_type,
                        dataset.encryption_policy,
                        domains_json,
                        metadata_json,
                        now,
                        now,
                        dataset.archived_at,
                    ),
                    connection=conn,
                )
            for domain in dataset.domains:
                self._ensure_domain_state(
                    dataset_id=dataset.dataset_id,
                    domain=domain,
                    adapter_version=1,
                    server_sequence=0,
                    connection=conn,
                )
            row = _first(
                self.db.execute(
                    "SELECT * FROM sync_datasets WHERE dataset_id = ?",
                    (dataset.dataset_id,),
                    connection=conn,
                )
            )
        return _dataset_from_row(row)

    def get_dataset(self, dataset_id: str) -> SyncDataset | None:
        row = _first(
            self.db.execute(
                "SELECT * FROM sync_datasets WHERE dataset_id = ?",
                (dataset_id,),
            )
        )
        if row is None:
            return None
        return _dataset_from_row(row)

    def insert_envelope(self, envelope: SyncEnvelopeCreate) -> SyncEnvelope:
        with self.db.backend.transaction() as conn:
            dataset = _first(
                self.db.execute(
                    "SELECT dataset_id FROM sync_datasets WHERE dataset_id = ?",
                    (envelope.dataset_id,),
                    connection=conn,
                )
            )
            if dataset is None:
                raise SyncDatasetNotFoundError(f"Sync dataset not found: {envelope.dataset_id}")

            existing = _first(
                self.db.execute(
                    """
                    SELECT * FROM sync_envelopes
                     WHERE dataset_id = ? AND client_envelope_id = ?
                    """,
                    (envelope.dataset_id, envelope.client_envelope_id),
                    connection=conn,
                )
            )
            if existing:
                return _envelope_from_row(existing)

            now = utcnow_iso()
            self.db.execute(
                """
                INSERT INTO sync_envelopes (
                    dataset_id, domain, entity_id, stable_key, operation,
                    client_envelope_id, device_id, client_timestamp, server_timestamp,
                    base_version, entity_version, dependency_json, routing_metadata_json,
                    payload_ciphertext, payload_clear_json, payload_hash,
                    payload_size_bytes, adapter_version, status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    envelope.dataset_id,
                    envelope.domain,
                    envelope.entity_id,
                    envelope.stable_key,
                    envelope.operation,
                    envelope.client_envelope_id,
                    envelope.device_id,
                    envelope.client_timestamp,
                    now,
                    _version_to_storage(envelope.base_version),
                    _version_to_storage(envelope.entity_version),
                    encode_json(envelope.dependencies, default=[]),
                    encode_json(envelope.routing_metadata, default={}),
                    envelope.payload_ciphertext,
                    encode_json(envelope.payload_clear, default={}),
                    envelope.payload_hash,
                    envelope.payload_size_bytes,
                    envelope.adapter_version,
                    envelope.status,
                ),
                connection=conn,
            )
            sequence = self.db.backend.get_last_insert_id(connection=conn)
            if sequence is None:
                row = _first(
                    self.db.execute(
                        """
                        SELECT * FROM sync_envelopes
                         WHERE dataset_id = ? AND client_envelope_id = ?
                        """,
                        (envelope.dataset_id, envelope.client_envelope_id),
                        connection=conn,
                    )
                )
            else:
                row = _first(
                    self.db.execute(
                        "SELECT * FROM sync_envelopes WHERE server_sequence = ?",
                        (sequence,),
                        connection=conn,
                    )
                )
            inserted = _envelope_from_row(row)
            self._ensure_domain_state(
                dataset_id=inserted.dataset_id,
                domain=inserted.domain,
                adapter_version=inserted.adapter_version,
                server_sequence=inserted.server_sequence,
                connection=conn,
            )
            return inserted

    def list_envelopes_after(
        self,
        dataset_id: str,
        since_sequence: int,
        *,
        limit: int = 100,
        domains: Sequence[SyncDomain] | None = None,
    ) -> list[SyncEnvelope]:
        if limit < 1:
            return []
        params: list[Any] = [dataset_id, since_sequence]
        sql = """
            SELECT * FROM sync_envelopes
             WHERE dataset_id = ? AND server_sequence > ?
        """
        if domains is not None:
            if not domains:
                return []
            placeholders = ", ".join("?" for _ in domains)
            sql += f" AND domain IN ({placeholders})"
            params.extend(domains)
        sql += " ORDER BY server_sequence ASC LIMIT ?"
        params.append(limit)
        result = self.db.execute(sql, tuple(params))
        return [_envelope_from_row(row) for row in result.rows]

    def update_device_cursor(self, cursor: SyncDeviceCursor) -> SyncDeviceCursor:
        now = utcnow_iso()
        with self.db.backend.transaction() as conn:
            existing = _first(
                self.db.execute(
                    """
                    SELECT * FROM sync_device_cursors
                     WHERE dataset_id = ? AND device_id = ? AND domain = ?
                    """,
                    (cursor.dataset_id, cursor.device_id, cursor.domain),
                    connection=conn,
                )
            )
            if existing:
                self.db.execute(
                    """
                    UPDATE sync_device_cursors
                       SET last_pulled_sequence = ?, updated_at = ?
                     WHERE dataset_id = ? AND device_id = ? AND domain = ?
                    """,
                    (
                        cursor.last_pulled_sequence,
                        now,
                        cursor.dataset_id,
                        cursor.device_id,
                        cursor.domain,
                    ),
                    connection=conn,
                )
            else:
                self.db.execute(
                    """
                    INSERT INTO sync_device_cursors (
                        dataset_id, device_id, domain, last_pulled_sequence, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        cursor.dataset_id,
                        cursor.device_id,
                        cursor.domain,
                        cursor.last_pulled_sequence,
                        now,
                    ),
                    connection=conn,
                )
            row = _first(
                self.db.execute(
                    """
                    SELECT * FROM sync_device_cursors
                     WHERE dataset_id = ? AND device_id = ? AND domain = ?
                    """,
                    (cursor.dataset_id, cursor.device_id, cursor.domain),
                    connection=conn,
                )
            )
        return _cursor_from_row(row)

    def get_device_cursor(
        self,
        dataset_id: str,
        device_id: str,
        domain: SyncDomain,
    ) -> SyncDeviceCursor | None:
        row = _first(
            self.db.execute(
                """
                SELECT * FROM sync_device_cursors
                 WHERE dataset_id = ? AND device_id = ? AND domain = ?
                """,
                (dataset_id, device_id, domain),
            )
        )
        if row is None:
            return None
        return _cursor_from_row(row)

    def insert_conflict(self, conflict: SyncConflictCreate) -> SyncConflict:
        now = utcnow_iso()
        with self.db.backend.transaction() as conn:
            existing = _first(
                self.db.execute(
                    "SELECT * FROM sync_conflicts WHERE conflict_id = ?",
                    (conflict.conflict_id,),
                    connection=conn,
                )
            )
            if existing:
                return _conflict_from_row(existing)
            self.db.execute(
                """
                INSERT INTO sync_conflicts (
                    conflict_id, dataset_id, domain, entity_id, conflict_type,
                    status, base_envelope_id, local_envelope_id, remote_envelope_id,
                    server_sequence, metadata_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    conflict.conflict_id,
                    conflict.dataset_id,
                    conflict.domain,
                    conflict.entity_id,
                    conflict.conflict_type,
                    "unresolved",
                    conflict.base_envelope_id,
                    conflict.local_envelope_id,
                    conflict.remote_envelope_id,
                    conflict.server_sequence,
                    encode_json(conflict.metadata, default={}),
                    now,
                ),
                connection=conn,
            )
            row = _first(
                self.db.execute(
                    "SELECT * FROM sync_conflicts WHERE conflict_id = ?",
                    (conflict.conflict_id,),
                    connection=conn,
                )
            )
        return _conflict_from_row(row)

    def list_conflicts(
        self,
        dataset_id: str,
        *,
        status: ConflictStatus | None = None,
    ) -> list[SyncConflict]:
        params: list[Any] = [dataset_id]
        sql = "SELECT * FROM sync_conflicts WHERE dataset_id = ?"
        if status is not None:
            sql += " AND status = ?"
            params.append(status)
        sql += " ORDER BY created_at ASC, conflict_id ASC"
        result = self.db.execute(sql, tuple(params))
        return [_conflict_from_row(row) for row in result.rows]

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
        now = utcnow_iso()
        with self.db.backend.transaction() as conn:
            existing = _first(
                self.db.execute(
                    "SELECT * FROM sync_conflicts WHERE conflict_id = ?",
                    (conflict_id,),
                    connection=conn,
                )
            )
            if existing is None:
                raise SyncConflictNotFoundError(f"Sync conflict not found: {conflict_id}")
            self.db.execute(
                """
                UPDATE sync_conflicts
                   SET status = ?,
                       resolved_at = ?,
                       resolved_by_envelope_id = ?,
                       resolved_by_device_id = ?,
                       resolution_action = ?,
                       resolution_notes = ?
                 WHERE conflict_id = ?
                """,
                (
                    status,
                    now,
                    resolved_by_envelope_id,
                    resolved_by_device_id,
                    resolution_action,
                    resolution_notes,
                    conflict_id,
                ),
                connection=conn,
            )
            row = _first(
                self.db.execute(
                    "SELECT * FROM sync_conflicts WHERE conflict_id = ?",
                    (conflict_id,),
                    connection=conn,
                )
            )
        return _conflict_from_row(row)

    def store_key_record(self, record: SyncKeyRecordCreate) -> SyncKeyRecord:
        now = utcnow_iso()
        with self.db.backend.transaction() as conn:
            existing = _first(
                self.db.execute(
                    "SELECT * FROM sync_key_records WHERE key_record_id = ?",
                    (record.key_record_id,),
                    connection=conn,
                )
            )
            if existing:
                self.db.execute(
                    """
                    UPDATE sync_key_records
                       SET dataset_id = ?,
                           device_id = ?,
                           key_purpose = ?,
                           wrapped_key_blob = ?,
                           kdf_metadata_json = ?,
                           recovery_hint = ?,
                           rotation_of_key_record_id = ?,
                           revoked_at = ?
                     WHERE key_record_id = ?
                    """,
                    (
                        record.dataset_id,
                        record.device_id,
                        record.key_purpose,
                        record.wrapped_key_blob,
                        encode_json(record.kdf_metadata, default={}),
                        record.recovery_hint,
                        record.rotation_of_key_record_id,
                        record.revoked_at,
                        record.key_record_id,
                    ),
                    connection=conn,
                )
            else:
                self.db.execute(
                    """
                    INSERT INTO sync_key_records (
                        key_record_id, dataset_id, device_id, key_purpose,
                        wrapped_key_blob, kdf_metadata_json, recovery_hint,
                        rotation_of_key_record_id, created_at, revoked_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.key_record_id,
                        record.dataset_id,
                        record.device_id,
                        record.key_purpose,
                        record.wrapped_key_blob,
                        encode_json(record.kdf_metadata, default={}),
                        record.recovery_hint,
                        record.rotation_of_key_record_id,
                        now,
                        record.revoked_at,
                    ),
                    connection=conn,
                )
            row = _first(
                self.db.execute(
                    "SELECT * FROM sync_key_records WHERE key_record_id = ?",
                    (record.key_record_id,),
                    connection=conn,
                )
            )
        return _key_record_from_row(row)

    def list_key_records(
        self,
        dataset_id: str,
        *,
        device_id: str | None = None,
        key_purpose: str | None = None,
    ) -> list[SyncKeyRecord]:
        params: list[Any] = [dataset_id]
        sql = "SELECT * FROM sync_key_records WHERE dataset_id = ?"
        if device_id is not None:
            sql += " AND device_id = ?"
            params.append(device_id)
        if key_purpose is not None:
            sql += " AND key_purpose = ?"
            params.append(key_purpose)
        sql += " ORDER BY created_at ASC, key_record_id ASC"
        result = self.db.execute(sql, tuple(params))
        return [_key_record_from_row(row) for row in result.rows]

    def _ensure_domain_state(
        self,
        *,
        dataset_id: str,
        domain: SyncDomain,
        adapter_version: int,
        server_sequence: int,
        connection: Any,
    ) -> None:
        now = utcnow_iso()
        existing = _first(
            self.db.execute(
                """
                SELECT * FROM sync_domain_state
                 WHERE dataset_id = ? AND domain = ?
                """,
                (dataset_id, domain),
                connection=connection,
            )
        )
        if existing:
            self.db.execute(
                """
                UPDATE sync_domain_state
                   SET adapter_version = ?,
                       server_sequence = CASE
                           WHEN server_sequence > ? THEN server_sequence
                           ELSE ?
                       END,
                       updated_at = ?
                 WHERE dataset_id = ? AND domain = ?
                """,
                (
                    adapter_version,
                    server_sequence,
                    server_sequence,
                    now,
                    dataset_id,
                    domain,
                ),
                connection=connection,
            )
            return
        self.db.execute(
            """
            INSERT INTO sync_domain_state (
                dataset_id, domain, adapter_version, server_sequence,
                last_compacted_sequence, state_json, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                dataset_id,
                domain,
                adapter_version,
                server_sequence,
                0,
                encode_json({}, default={}),
                now,
            ),
            connection=connection,
        )


__all__ = ["SyncV2Store"]
