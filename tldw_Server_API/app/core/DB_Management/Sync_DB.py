from __future__ import annotations

"""Database helper for per-user Sync v2 storage."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths

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

    def execute(
        self,
        query: str,
        params: tuple[Any, ...] | dict[str, Any] | None = None,
        *,
        connection: Any | None = None,
    ) -> QueryResult:
        """Execute a parameterized SQL statement through the configured backend."""

        return self.backend.execute(query, params, connection=connection)


__all__ = [
    "SYNC_DB_FILENAME",
    "SyncDatabase",
    "decode_json",
    "encode_json",
    "utcnow_iso",
]
