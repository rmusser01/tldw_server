from __future__ import annotations

"""Database helper for per-user Sync v2 storage."""

import json
import os
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.Sync.v2.errors import (
    SyncConflictNotFoundError,
    SyncDatasetNotFoundError,
    SyncIdempotencyConflictError,
    SyncInvalidDomainError,
    SyncStoreError,
)
from tldw_Server_API.app.core.Sync.v2.models import (
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

from .backends.base import BackendType, DatabaseBackend, DatabaseConfig, QueryResult
from .backends.factory import DatabaseBackendFactory

SYNC_DB_FILENAME = "Sync_v2.db"

SYNC_SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS sync_devices (
    device_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    display_name TEXT NOT NULL,
    client_type TEXT NOT NULL,
    client_version TEXT,
    capabilities_json TEXT NOT NULL DEFAULT '{}',
    registered_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    revoked_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_sync_devices_user ON sync_devices(user_id);

CREATE TABLE IF NOT EXISTS sync_datasets (
    dataset_id TEXT PRIMARY KEY,
    owner_user_id TEXT NOT NULL,
    workspace_id TEXT,
    scope_type TEXT NOT NULL,
    encryption_policy TEXT NOT NULL,
    domain_set_json TEXT NOT NULL DEFAULT '[]',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    archived_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_sync_datasets_owner ON sync_datasets(owner_user_id);
CREATE INDEX IF NOT EXISTS idx_sync_datasets_workspace ON sync_datasets(workspace_id);

CREATE TABLE IF NOT EXISTS sync_domain_state (
    dataset_id TEXT NOT NULL,
    domain TEXT NOT NULL,
    adapter_version INTEGER NOT NULL,
    server_sequence INTEGER NOT NULL DEFAULT 0,
    last_compacted_sequence INTEGER NOT NULL DEFAULT 0,
    state_json TEXT NOT NULL DEFAULT '{}',
    updated_at TEXT NOT NULL,
    PRIMARY KEY (dataset_id, domain)
);

CREATE TABLE IF NOT EXISTS sync_envelopes (
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
CREATE INDEX IF NOT EXISTS idx_sync_envelopes_dataset_sequence
    ON sync_envelopes(dataset_id, server_sequence);
CREATE INDEX IF NOT EXISTS idx_sync_envelopes_dataset_domain_sequence
    ON sync_envelopes(dataset_id, domain, server_sequence);
CREATE INDEX IF NOT EXISTS idx_sync_envelopes_entity
    ON sync_envelopes(dataset_id, domain, entity_id);

CREATE TABLE IF NOT EXISTS sync_device_cursors (
    dataset_id TEXT NOT NULL,
    device_id TEXT NOT NULL,
    domain TEXT NOT NULL,
    last_pulled_sequence INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (dataset_id, device_id, domain)
);

CREATE TABLE IF NOT EXISTS sync_conflicts (
    conflict_id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL,
    domain TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    conflict_type TEXT NOT NULL,
    status TEXT NOT NULL,
    base_envelope_id TEXT,
    local_envelope_id TEXT,
    remote_envelope_id TEXT,
    server_sequence INTEGER,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    resolved_by_envelope_id TEXT,
    resolved_by_device_id TEXT,
    resolution_action TEXT,
    resolution_notes TEXT,
    created_at TEXT NOT NULL,
    resolved_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_sync_conflicts_dataset_status
    ON sync_conflicts(dataset_id, status, created_at);
CREATE INDEX IF NOT EXISTS idx_sync_conflicts_entity
    ON sync_conflicts(dataset_id, domain, entity_id);

CREATE TABLE IF NOT EXISTS sync_key_records (
    key_record_id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    device_id TEXT,
    key_purpose TEXT NOT NULL,
    wrapped_key_blob TEXT NOT NULL,
    kdf_metadata_json TEXT NOT NULL DEFAULT '{}',
    recovery_hint TEXT,
    rotation_of_key_record_id TEXT,
    created_at TEXT NOT NULL,
    revoked_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_sync_key_records_dataset
    ON sync_key_records(dataset_id, key_purpose, created_at);
CREATE INDEX IF NOT EXISTS idx_sync_key_records_device
    ON sync_key_records(dataset_id, device_id);
"""

SYNC_POSTGRES_SCHEMA = """
CREATE TABLE IF NOT EXISTS sync_devices (
    device_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    display_name TEXT NOT NULL,
    client_type TEXT NOT NULL,
    client_version TEXT,
    capabilities_json TEXT NOT NULL DEFAULT '{}',
    registered_at TIMESTAMPTZ NOT NULL,
    last_seen_at TIMESTAMPTZ NOT NULL,
    revoked_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_sync_devices_user ON sync_devices(user_id);

CREATE TABLE IF NOT EXISTS sync_datasets (
    dataset_id TEXT PRIMARY KEY,
    owner_user_id TEXT NOT NULL,
    workspace_id TEXT,
    scope_type TEXT NOT NULL,
    encryption_policy TEXT NOT NULL,
    domain_set_json TEXT NOT NULL DEFAULT '[]',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    archived_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_sync_datasets_owner ON sync_datasets(owner_user_id);
CREATE INDEX IF NOT EXISTS idx_sync_datasets_workspace ON sync_datasets(workspace_id);

CREATE TABLE IF NOT EXISTS sync_domain_state (
    dataset_id TEXT NOT NULL,
    domain TEXT NOT NULL,
    adapter_version INTEGER NOT NULL,
    server_sequence BIGINT NOT NULL DEFAULT 0,
    last_compacted_sequence BIGINT NOT NULL DEFAULT 0,
    state_json TEXT NOT NULL DEFAULT '{}',
    updated_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (dataset_id, domain)
);

CREATE TABLE IF NOT EXISTS sync_envelopes (
    server_sequence BIGSERIAL PRIMARY KEY,
    dataset_id TEXT NOT NULL,
    domain TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    stable_key TEXT,
    operation TEXT NOT NULL,
    client_envelope_id TEXT NOT NULL,
    device_id TEXT,
    client_timestamp TIMESTAMPTZ,
    server_timestamp TIMESTAMPTZ NOT NULL,
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
CREATE INDEX IF NOT EXISTS idx_sync_envelopes_dataset_sequence
    ON sync_envelopes(dataset_id, server_sequence);
CREATE INDEX IF NOT EXISTS idx_sync_envelopes_dataset_domain_sequence
    ON sync_envelopes(dataset_id, domain, server_sequence);
CREATE INDEX IF NOT EXISTS idx_sync_envelopes_entity
    ON sync_envelopes(dataset_id, domain, entity_id);

CREATE TABLE IF NOT EXISTS sync_device_cursors (
    dataset_id TEXT NOT NULL,
    device_id TEXT NOT NULL,
    domain TEXT NOT NULL,
    last_pulled_sequence BIGINT NOT NULL DEFAULT 0,
    updated_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (dataset_id, device_id, domain)
);

CREATE TABLE IF NOT EXISTS sync_conflicts (
    conflict_id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL,
    domain TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    conflict_type TEXT NOT NULL,
    status TEXT NOT NULL,
    base_envelope_id TEXT,
    local_envelope_id TEXT,
    remote_envelope_id TEXT,
    server_sequence BIGINT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    resolved_by_envelope_id TEXT,
    resolved_by_device_id TEXT,
    resolution_action TEXT,
    resolution_notes TEXT,
    created_at TIMESTAMPTZ NOT NULL,
    resolved_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_sync_conflicts_dataset_status
    ON sync_conflicts(dataset_id, status, created_at);
CREATE INDEX IF NOT EXISTS idx_sync_conflicts_entity
    ON sync_conflicts(dataset_id, domain, entity_id);

CREATE TABLE IF NOT EXISTS sync_key_records (
    key_record_id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    device_id TEXT,
    key_purpose TEXT NOT NULL,
    wrapped_key_blob TEXT NOT NULL,
    kdf_metadata_json TEXT NOT NULL DEFAULT '{}',
    recovery_hint TEXT,
    rotation_of_key_record_id TEXT,
    created_at TIMESTAMPTZ NOT NULL,
    revoked_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_sync_key_records_dataset
    ON sync_key_records(dataset_id, key_purpose, created_at);
CREATE INDEX IF NOT EXISTS idx_sync_key_records_device
    ON sync_key_records(dataset_id, device_id);
"""


def utcnow_iso() -> str:
    """Return an ISO-8601 UTC timestamp for Sync v2 rows."""

    return datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()


def encode_json(value: Any, *, default: Any) -> str:
    """Serialize storage JSON deterministically."""

    if value is None:
        value = default
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def decode_json(value: str | None, *, default: Any) -> Any:
    """Deserialize storage JSON with a defensive default."""

    if value is None or value == "":
        return default
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


def _sqlite_path_from_url(database_url: str, default_path: Path) -> Path | str:
    parsed = urlparse(database_url)
    raw_path = parsed.path or ""
    if raw_path in {"/:memory:", ":memory:"}:
        return ":memory:"
    if raw_path.startswith("/./"):
        raw_path = raw_path[1:]
    if raw_path.startswith("/") and raw_path != "/:memory:":
        return Path(raw_path)
    return default_path.parent / (raw_path or default_path.name)


def _default_sync_db_path(user_id: int | str | None) -> Path:
    user_dir = DatabasePaths.get_user_base_directory(user_id)
    return user_dir / SYNC_DB_FILENAME


def _first(result: QueryResult) -> dict[str, Any] | None:
    rows = result.rows or []
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
        user_id=row["user_id"],
        device_id=row.get("device_id"),
        key_purpose=row["key_purpose"],
        wrapped_key_blob=row["wrapped_key_blob"],
        kdf_metadata=decode_json(row.get("kdf_metadata_json"), default={}),
        recovery_hint=row.get("recovery_hint"),
        rotation_of_key_record_id=row.get("rotation_of_key_record_id"),
        created_at=row["created_at"],
        revoked_at=row.get("revoked_at"),
    )


def _dataset_domains_from_row(row: dict[str, Any]) -> set[str]:
    domains = decode_json(row.get("domain_set_json"), default=[])
    return {str(domain) for domain in domains}


def _envelope_fingerprint_from_create(envelope: SyncEnvelopeCreate) -> dict[str, Any]:
    return {
        "dataset_id": envelope.dataset_id,
        "domain": envelope.domain,
        "entity_id": envelope.entity_id,
        "stable_key": envelope.stable_key,
        "operation": envelope.operation,
        "client_envelope_id": envelope.client_envelope_id,
        "device_id": envelope.device_id,
        "client_timestamp": envelope.client_timestamp,
        "base_version": envelope.base_version,
        "entity_version": envelope.entity_version,
        "dependencies": envelope.dependencies,
        "routing_metadata": envelope.routing_metadata,
        "payload_ciphertext": envelope.payload_ciphertext,
        "payload_clear": envelope.payload_clear,
        "payload_hash": envelope.payload_hash,
        "payload_size_bytes": envelope.payload_size_bytes,
        "adapter_version": envelope.adapter_version,
        "status": envelope.status,
    }


def _envelope_fingerprint_from_row(row: dict[str, Any]) -> dict[str, Any]:
    payload_size_bytes = row.get("payload_size_bytes")
    return {
        "dataset_id": row["dataset_id"],
        "domain": row["domain"],
        "entity_id": row["entity_id"],
        "stable_key": row.get("stable_key"),
        "operation": row["operation"],
        "client_envelope_id": row["client_envelope_id"],
        "device_id": row.get("device_id"),
        "client_timestamp": row.get("client_timestamp"),
        "base_version": _version_from_storage(row.get("base_version")),
        "entity_version": _version_from_storage(row.get("entity_version")),
        "dependencies": decode_json(row.get("dependency_json"), default=[]),
        "routing_metadata": decode_json(row.get("routing_metadata_json"), default={}),
        "payload_ciphertext": row.get("payload_ciphertext"),
        "payload_clear": decode_json(row.get("payload_clear_json"), default={}),
        "payload_hash": row.get("payload_hash"),
        "payload_size_bytes": int(payload_size_bytes) if payload_size_bytes is not None else None,
        "adapter_version": int(row["adapter_version"]),
        "status": row["status"],
    }


def _key_record_fingerprint_from_create(record: SyncKeyRecordCreate) -> dict[str, Any]:
    return {
        "key_record_id": record.key_record_id,
        "dataset_id": record.dataset_id,
        "user_id": record.user_id,
        "device_id": record.device_id,
        "key_purpose": record.key_purpose,
        "wrapped_key_blob": record.wrapped_key_blob,
        "kdf_metadata": record.kdf_metadata,
        "recovery_hint": record.recovery_hint,
        "rotation_of_key_record_id": record.rotation_of_key_record_id,
        "revoked_at": record.revoked_at,
    }


def _key_record_fingerprint_from_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "key_record_id": row["key_record_id"],
        "dataset_id": row["dataset_id"],
        "user_id": row["user_id"],
        "device_id": row.get("device_id"),
        "key_purpose": row["key_purpose"],
        "wrapped_key_blob": row["wrapped_key_blob"],
        "kdf_metadata": decode_json(row.get("kdf_metadata_json"), default={}),
        "recovery_hint": row.get("recovery_hint"),
        "rotation_of_key_record_id": row.get("rotation_of_key_record_id"),
        "revoked_at": row.get("revoked_at"),
    }


class SyncDatabase:
    """Focused DB_Management helper for Sync v2 per-user storage."""

    def __init__(
        self,
        backend: DatabaseBackend | None = None,
        *,
        sqlite_path: str | Path | None = None,
        user_id: int | str | None = None,
    ) -> None:
        if backend is not None:
            self.backend = backend
        else:
            self.backend = DatabaseBackendFactory.create_backend(
                self._build_config(sqlite_path=sqlite_path, user_id=user_id)
            )
        self.ensure_schema()

    def _build_config(
        self,
        *,
        sqlite_path: str | Path | None,
        user_id: int | str | None,
    ) -> DatabaseConfig:
        default_path = _default_sync_db_path(user_id)
        custom_url = os.getenv("SYNC_V2_DATABASE_URL", "").strip()
        custom_path = sqlite_path or os.getenv("SYNC_V2_SQLITE_PATH", "").strip()

        if custom_url:
            parsed = urlparse(custom_url)
            scheme = (parsed.scheme or "").lower().split("+", 1)[0]
            if scheme in {"postgres", "postgresql"}:
                return DatabaseConfig(
                    backend_type=BackendType.POSTGRESQL,
                    connection_string=custom_url,
                    pg_host=parsed.hostname or "localhost",
                    pg_port=int(parsed.port or 5432),
                    pg_database=(parsed.path or "/").lstrip("/") or None,
                    pg_user=parsed.username or None,
                    pg_password=parsed.password or None,
                )
            if scheme in {"sqlite", "file", ""}:
                sqlite_target = _sqlite_path_from_url(custom_url, default_path)
                return DatabaseConfig(
                    backend_type=BackendType.SQLITE,
                    sqlite_path=str(sqlite_target),
                )

        if custom_path:
            return DatabaseConfig(
                backend_type=BackendType.SQLITE,
                sqlite_path=str(custom_path),
            )

        default_path.parent.mkdir(parents=True, exist_ok=True)
        return DatabaseConfig(
            backend_type=BackendType.SQLITE,
            sqlite_path=str(default_path),
        )

    @property
    def backend_type(self) -> BackendType | None:
        return getattr(getattr(self.backend, "config", None), "backend_type", None)

    def ensure_schema(self) -> None:
        """Create Sync v2 tables and indexes if they do not exist."""

        schema = (
            SYNC_POSTGRES_SCHEMA
            if self.backend_type == BackendType.POSTGRESQL
            else SYNC_SQLITE_SCHEMA
        )
        with self.backend.transaction() as conn:
            self.backend.create_tables(schema, connection=conn)
            self._ensure_key_record_user_id_column(connection=conn)
            self._ensure_key_record_user_id_index(connection=conn)

    def execute(
        self,
        query: str,
        params: tuple[Any, ...] | dict[str, Any] | None = None,
        *,
        connection: Any | None = None,
    ) -> QueryResult:
        """Execute a parameterized SQL statement through the configured backend."""

        return self.backend.execute(query, params, connection=connection)

    def _get_dataset_row(
        self,
        dataset_id: str,
        *,
        owner_user_id: str | None = None,
        connection: Any | None = None,
    ) -> dict[str, Any] | None:
        if owner_user_id is None:
            return _first(
                self.execute(
                    "SELECT * FROM sync_datasets WHERE dataset_id = ?",
                    (dataset_id,),
                    connection=connection,
                )
            )
        return _first(
            self.execute(
                """
                SELECT * FROM sync_datasets
                 WHERE dataset_id = ? AND owner_user_id = ?
                """,
                (dataset_id, owner_user_id),
                connection=connection,
            )
        )

    def _require_dataset(
        self,
        dataset_id: str,
        *,
        connection: Any | None = None,
    ) -> dict[str, Any]:
        row = self._get_dataset_row(dataset_id, connection=connection)
        if row is None:
            raise SyncDatasetNotFoundError(f"Sync dataset not found: {dataset_id}")
        return row

    def _require_dataset_domain(
        self,
        dataset_id: str,
        domain: SyncDomain,
        *,
        connection: Any | None = None,
    ) -> dict[str, Any]:
        row = self._require_dataset(dataset_id, connection=connection)
        if domain not in _dataset_domains_from_row(row):
            raise SyncInvalidDomainError(
                f"Sync domain is not enrolled for dataset {dataset_id}: {domain}"
            )
        return row

    def upsert_device(self, device: SyncDeviceUpsert) -> SyncDevice:
        now = utcnow_iso()
        with self.backend.transaction() as conn:
            existing = _first(
                self.execute(
                    "SELECT * FROM sync_devices WHERE device_id = ?",
                    (device.device_id,),
                    connection=conn,
                )
            )
            if existing:
                self.execute(
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
                self.execute(
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
                self.execute(
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
        with self.backend.transaction() as conn:
            existing = _first(
                self.execute(
                    "SELECT * FROM sync_datasets WHERE dataset_id = ?",
                    (dataset.dataset_id,),
                    connection=conn,
                )
            )
            if existing:
                self.execute(
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
                self.execute(
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
                self.execute(
                    "SELECT * FROM sync_datasets WHERE dataset_id = ?",
                    (dataset.dataset_id,),
                    connection=conn,
                )
            )
        return _dataset_from_row(row)

    def get_dataset(
        self,
        dataset_id: str,
        *,
        owner_user_id: str | None = None,
    ) -> SyncDataset | None:
        row = self._get_dataset_row(dataset_id, owner_user_id=owner_user_id)
        if row is None:
            return None
        return _dataset_from_row(row)

    def insert_envelope(self, envelope: SyncEnvelopeCreate) -> SyncEnvelope:
        with self.backend.transaction() as conn:
            self._require_dataset_domain(
                envelope.dataset_id,
                envelope.domain,
                connection=conn,
            )

            existing = _first(
                self.execute(
                    """
                    SELECT * FROM sync_envelopes
                     WHERE dataset_id = ? AND client_envelope_id = ?
                    """,
                    (envelope.dataset_id, envelope.client_envelope_id),
                    connection=conn,
                )
            )
            if existing:
                if (
                    _envelope_fingerprint_from_row(existing)
                    != _envelope_fingerprint_from_create(envelope)
                ):
                    raise SyncIdempotencyConflictError(
                        "Sync envelope idempotency key was reused with different content"
                    )
                return _envelope_from_row(existing)

            now = utcnow_iso()
            self.execute(
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
            sequence = self.backend.get_last_insert_id(connection=conn)
            if sequence is None:
                row = _first(
                    self.execute(
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
                    self.execute(
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
        result = self.execute(sql, tuple(params))
        return [_envelope_from_row(row) for row in result.rows]

    def update_device_cursor(self, cursor: SyncDeviceCursor) -> SyncDeviceCursor:
        now = utcnow_iso()
        with self.backend.transaction() as conn:
            self._require_dataset_domain(cursor.dataset_id, cursor.domain, connection=conn)
            existing = _first(
                self.execute(
                    """
                    SELECT * FROM sync_device_cursors
                     WHERE dataset_id = ? AND device_id = ? AND domain = ?
                    """,
                    (cursor.dataset_id, cursor.device_id, cursor.domain),
                    connection=conn,
                )
            )
            if existing:
                self.execute(
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
                self.execute(
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
                self.execute(
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
            self.execute(
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
        with self.backend.transaction() as conn:
            self._require_dataset_domain(conflict.dataset_id, conflict.domain, connection=conn)
            existing = _first(
                self.execute(
                    "SELECT * FROM sync_conflicts WHERE conflict_id = ?",
                    (conflict.conflict_id,),
                    connection=conn,
                )
            )
            if existing:
                return _conflict_from_row(existing)
            self.execute(
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
                self.execute(
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
        result = self.execute(sql, tuple(params))
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
        with self.backend.transaction() as conn:
            existing = _first(
                self.execute(
                    "SELECT * FROM sync_conflicts WHERE conflict_id = ?",
                    (conflict_id,),
                    connection=conn,
                )
            )
            if existing is None:
                raise SyncConflictNotFoundError(f"Sync conflict not found: {conflict_id}")
            self.execute(
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
                self.execute(
                    "SELECT * FROM sync_conflicts WHERE conflict_id = ?",
                    (conflict_id,),
                    connection=conn,
                )
            )
        return _conflict_from_row(row)

    def store_key_record(self, record: SyncKeyRecordCreate) -> SyncKeyRecord:
        now = utcnow_iso()
        with self.backend.transaction() as conn:
            self._require_dataset(record.dataset_id, connection=conn)
            existing = _first(
                self.execute(
                    "SELECT * FROM sync_key_records WHERE key_record_id = ?",
                    (record.key_record_id,),
                    connection=conn,
                )
            )
            if existing:
                if (
                    _key_record_fingerprint_from_row(existing)
                    != _key_record_fingerprint_from_create(record)
                ):
                    raise SyncIdempotencyConflictError(
                        "Sync key record ID was reused with different key material"
                    )
                row = existing
            else:
                self.execute(
                    """
                    INSERT INTO sync_key_records (
                        key_record_id, dataset_id, user_id, device_id, key_purpose,
                        wrapped_key_blob, kdf_metadata_json, recovery_hint,
                        rotation_of_key_record_id, created_at, revoked_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.key_record_id,
                        record.dataset_id,
                        record.user_id,
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
                    self.execute(
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
        user_id: str,
        device_id: str | None = None,
        key_purpose: str | None = None,
    ) -> list[SyncKeyRecord]:
        if not user_id:
            raise SyncStoreError("user_id is required when listing Sync key records")
        self._require_dataset(dataset_id)
        params: list[Any] = [dataset_id, user_id]
        sql = "SELECT * FROM sync_key_records WHERE dataset_id = ? AND user_id = ?"
        if device_id is not None:
            sql += " AND device_id = ?"
            params.append(device_id)
        if key_purpose is not None:
            sql += " AND key_purpose = ?"
            params.append(key_purpose)
        sql += " ORDER BY created_at ASC, key_record_id ASC"
        result = self.execute(sql, tuple(params))
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
            self.execute(
                """
                SELECT * FROM sync_domain_state
                 WHERE dataset_id = ? AND domain = ?
                """,
                (dataset_id, domain),
                connection=connection,
            )
        )
        if existing:
            self.execute(
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
        self.execute(
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

    def _ensure_key_record_user_id_column(self, *, connection: Any) -> None:
        columns = {
            column.get("name")
            for column in self.backend.get_table_info("sync_key_records", connection=connection)
            if isinstance(column, dict)
        }
        if "user_id" in columns:
            return
        if self.backend_type == BackendType.POSTGRESQL:
            self.execute(
                "ALTER TABLE sync_key_records ADD COLUMN IF NOT EXISTS user_id TEXT NOT NULL DEFAULT ''",
                connection=connection,
            )
        else:
            self.execute(
                "ALTER TABLE sync_key_records ADD COLUMN user_id TEXT NOT NULL DEFAULT ''",
                connection=connection,
            )

    def _ensure_key_record_user_id_index(self, *, connection: Any) -> None:
        self.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_sync_key_records_user
                ON sync_key_records(user_id, dataset_id)
            """,
            connection=connection,
        )


__all__ = [
    "SYNC_DB_FILENAME",
    "SyncDatabase",
    "decode_json",
    "encode_json",
    "utcnow_iso",
]
