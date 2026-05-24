from __future__ import annotations

"""Database helper for per-user Sync v2 storage."""

import json
import os
from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime, timedelta, timezone
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
    DEFAULT_M1_ENCRYPTION_POLICY,
    M1_SYNC_DOMAINS,
    MEDIA_SYNC_DOMAINS,
    SOURCE_CACHE_SYNC_DOMAINS,
    SYNC_V2_SUPPORTED_OPERATIONS,
    WORKSPACE_SYNC_DOMAINS,
    ConflictStatus,
    SyncApplyStatus,
    SyncAttachment,
    SyncAttachmentCreate,
    SyncBackgroundDomainStatus,
    SyncBackgroundLease,
    SyncBackgroundLeaseCreate,
    SyncBackgroundPolicy,
    SyncBackgroundPolicyUpsert,
    SyncBlobChunk,
    SyncBlobChunkCreate,
    SyncBlobObject,
    SyncBlobObjectCreate,
    SyncBlobQuotaUsage,
    SyncBlobUploadSession,
    SyncBlobUploadSessionCreate,
    SyncConflict,
    SyncConflictCreate,
    SyncDataset,
    SyncDatasetCreate,
    SyncDevice,
    SyncDeviceAcknowledgmentSummary,
    SyncDeviceAuthorization,
    SyncDeviceAuthorizationCreate,
    SyncDeviceBlobAck,
    SyncDeviceBlobAckCreate,
    SyncDeviceCursor,
    SyncDeviceDomainAck,
    SyncDeviceDomainAckCreate,
    SyncDeviceUpsert,
    SyncDomain,
    SyncEnvelope,
    SyncEnvelopeCreate,
    SyncKeyRecord,
    SyncKeyRecordCreate,
    SyncKeyRotationEnvelopeRange,
    SyncObjectState,
    SyncRestoreManifestStats,
)

from .backends.base import BackendType, DatabaseBackend, DatabaseConfig, QueryResult
from .backends.factory import DatabaseBackendFactory

SYNC_DB_FILENAME = "Sync_v2.db"
SYNC_APPLY_STATUSES: set[str] = {"pending", "applied", "failed", "conflict"}
_WHOLE_OBJECT_DOMAINS = {"notes.note", "chat.conversation"}
_ATTACHMENT_REF_REQUIRED_PAYLOAD_KEYS = {
    "attachment_id",
    "parent_domain",
    "parent_object_id",
    "content_type",
    "size_bytes",
    "payload_hash",
    "availability",
}

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
    status TEXT NOT NULL DEFAULT 'active',
    user_label TEXT,
    authorized_at TEXT,
    revoked_at TEXT,
    revoked_reason TEXT
);
CREATE INDEX IF NOT EXISTS idx_sync_devices_user ON sync_devices(user_id);
CREATE INDEX IF NOT EXISTS idx_sync_devices_user_status
    ON sync_devices(user_id, status, last_seen_at);

CREATE TABLE IF NOT EXISTS sync_device_authorizations (
    authorization_id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    device_id TEXT NOT NULL,
    authorization_method TEXT NOT NULL,
    status TEXT NOT NULL,
    requested_at TEXT NOT NULL,
    approved_at TEXT,
    approving_device_id TEXT,
    idempotency_key TEXT,
    approval_idempotency_key TEXT,
    UNIQUE(dataset_id, device_id, idempotency_key)
);
CREATE INDEX IF NOT EXISTS idx_sync_device_authorizations_dataset_device
    ON sync_device_authorizations(dataset_id, device_id, requested_at);
CREATE INDEX IF NOT EXISTS idx_sync_device_authorizations_user_status
    ON sync_device_authorizations(user_id, status, requested_at);

CREATE TABLE IF NOT EXISTS sync_device_domain_acks (
    dataset_id TEXT NOT NULL,
    device_id TEXT NOT NULL,
    domain TEXT NOT NULL,
    through_server_sequence INTEGER NOT NULL DEFAULT 0,
    applied_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    idempotency_key TEXT,
    PRIMARY KEY(dataset_id, device_id, domain)
);
CREATE INDEX IF NOT EXISTS idx_sync_device_domain_acks_device
    ON sync_device_domain_acks(device_id, dataset_id);

CREATE TABLE IF NOT EXISTS sync_device_blob_acks (
    dataset_id TEXT NOT NULL,
    device_id TEXT NOT NULL,
    attachment_id TEXT NOT NULL,
    payload_hash TEXT NOT NULL,
    verified_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    idempotency_key TEXT,
    PRIMARY KEY(dataset_id, device_id, attachment_id)
);
CREATE INDEX IF NOT EXISTS idx_sync_device_blob_acks_device
    ON sync_device_blob_acks(device_id, dataset_id);

CREATE TABLE IF NOT EXISTS sync_background_policies (
    dataset_id TEXT NOT NULL,
    device_id TEXT NOT NULL,
    enabled INTEGER NOT NULL DEFAULT 1,
    minimum_interval_seconds INTEGER NOT NULL DEFAULT 300,
    backoff_floor_seconds INTEGER NOT NULL DEFAULT 60,
    max_batch_size INTEGER NOT NULL DEFAULT 100,
    max_blob_bytes_per_run INTEGER,
    respect_metered_networks INTEGER NOT NULL DEFAULT 1,
    maintenance_window_json TEXT,
    paused_reason TEXT,
    pending_local_changes INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(dataset_id, device_id)
);
CREATE INDEX IF NOT EXISTS idx_sync_background_policies_device
    ON sync_background_policies(device_id, dataset_id);

CREATE TABLE IF NOT EXISTS sync_background_leases (
    dataset_id TEXT NOT NULL,
    device_id TEXT NOT NULL,
    lease_id TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(dataset_id, device_id)
);
CREATE INDEX IF NOT EXISTS idx_sync_background_leases_expiry
    ON sync_background_leases(expires_at);

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
    client_profile_id TEXT,
    client_sequence INTEGER,
    client_timestamp TEXT,
    server_timestamp TEXT NOT NULL,
    base_server_cursor INTEGER,
    base_object_revision INTEGER,
    base_object_hash TEXT,
    object_revision INTEGER,
    parent_id TEXT,
    schema_version INTEGER NOT NULL DEFAULT 1,
    base_version TEXT,
    entity_version TEXT,
    dependency_json TEXT NOT NULL DEFAULT '[]',
    routing_metadata_json TEXT NOT NULL DEFAULT '{}',
    payload_ciphertext TEXT,
    payload_json TEXT NOT NULL DEFAULT '{}',
    payload_clear_json TEXT NOT NULL DEFAULT '{}',
    payload_hash TEXT,
    payload_size_bytes INTEGER,
    created_at_client TEXT,
    received_at_server TEXT,
    deleted INTEGER NOT NULL DEFAULT 0,
    encryption_metadata_json TEXT NOT NULL DEFAULT '{}',
    adapter_version INTEGER NOT NULL,
    status TEXT NOT NULL,
    apply_status TEXT NOT NULL DEFAULT 'pending',
    apply_error_code TEXT,
    apply_error_message TEXT,
    applied_at TEXT,
    UNIQUE (dataset_id, client_envelope_id)
);
CREATE INDEX IF NOT EXISTS idx_sync_envelopes_dataset_sequence
    ON sync_envelopes(dataset_id, server_sequence);
CREATE INDEX IF NOT EXISTS idx_sync_envelopes_dataset_domain_sequence
    ON sync_envelopes(dataset_id, domain, server_sequence);
CREATE INDEX IF NOT EXISTS idx_sync_envelopes_dataset_domain_object
    ON sync_envelopes(dataset_id, domain, entity_id);
CREATE INDEX IF NOT EXISTS idx_sync_envelopes_dataset_status_sequence
    ON sync_envelopes(dataset_id, status, server_sequence);
CREATE INDEX IF NOT EXISTS idx_sync_envelopes_dataset_device_sequence
    ON sync_envelopes(dataset_id, device_id, server_sequence);
CREATE UNIQUE INDEX IF NOT EXISTS idx_sync_envelopes_dataset_device_client_sequence
    ON sync_envelopes(dataset_id, device_id, client_sequence)
    WHERE device_id IS NOT NULL AND client_sequence IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_sync_envelopes_payload_hash
    ON sync_envelopes(dataset_id, payload_hash);
CREATE INDEX IF NOT EXISTS idx_sync_envelopes_failed_apply
    ON sync_envelopes(dataset_id, apply_status, server_sequence)
    WHERE apply_status = 'failed';
CREATE INDEX IF NOT EXISTS idx_sync_envelopes_entity
    ON sync_envelopes(dataset_id, domain, entity_id);
CREATE INDEX IF NOT EXISTS idx_sync_envelopes_stable_key
    ON sync_envelopes(dataset_id, domain, stable_key);

CREATE TABLE IF NOT EXISTS sync_object_state (
    dataset_id TEXT NOT NULL,
    domain TEXT NOT NULL,
    object_id TEXT NOT NULL,
    object_revision INTEGER NOT NULL,
    object_hash TEXT NOT NULL,
    latest_server_cursor INTEGER NOT NULL,
    deleted INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (dataset_id, domain, object_id)
);
CREATE INDEX IF NOT EXISTS idx_sync_object_state_dataset_domain_object
    ON sync_object_state(dataset_id, domain, object_id);

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
    rotation_source_key_record_ids_json TEXT NOT NULL DEFAULT '[]',
    encryption_policy TEXT NOT NULL DEFAULT 'server_trusted_v1',
    key_epoch INTEGER NOT NULL DEFAULT 1,
    active_from_server_sequence INTEGER,
    superseded_at TEXT,
    wrapped_for TEXT NOT NULL DEFAULT 'recovery',
    rewrap_status TEXT NOT NULL DEFAULT 'not_required',
    created_at TEXT NOT NULL,
    revoked_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_sync_key_records_dataset
    ON sync_key_records(dataset_id, key_purpose, created_at);
CREATE INDEX IF NOT EXISTS idx_sync_key_records_device
    ON sync_key_records(dataset_id, device_id);
CREATE INDEX IF NOT EXISTS idx_sync_key_records_epoch
    ON sync_key_records(dataset_id, encryption_policy, key_epoch);
CREATE INDEX IF NOT EXISTS idx_sync_key_records_rewrap
    ON sync_key_records(dataset_id, rewrap_status);

CREATE TABLE IF NOT EXISTS sync_attachments (
    attachment_id TEXT NOT NULL,
    dataset_id TEXT NOT NULL,
    domain TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    content_type TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    payload_ciphertext TEXT NOT NULL,
    payload_hash TEXT NOT NULL,
    encryption_policy TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    PRIMARY KEY (dataset_id, attachment_id)
);
CREATE INDEX IF NOT EXISTS idx_sync_attachments_dataset_domain
    ON sync_attachments(dataset_id, domain, created_at);
CREATE INDEX IF NOT EXISTS idx_sync_attachments_dataset_hash
    ON sync_attachments(dataset_id, payload_hash);

CREATE TABLE IF NOT EXISTS sync_blob_objects (
    blob_id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL,
    owner_user_id TEXT NOT NULL,
    attachment_id TEXT NOT NULL,
    payload_hash TEXT NOT NULL,
    content_type TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    encryption_policy TEXT NOT NULL,
    storage_backend TEXT NOT NULL,
    storage_key TEXT NOT NULL,
    status TEXT NOT NULL,
    ref_count INTEGER NOT NULL DEFAULT 1,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    deleted_at TEXT,
    UNIQUE(dataset_id, payload_hash)
);
CREATE INDEX IF NOT EXISTS idx_sync_blob_objects_owner
    ON sync_blob_objects(owner_user_id, dataset_id, status);
CREATE INDEX IF NOT EXISTS idx_sync_blob_objects_attachment
    ON sync_blob_objects(dataset_id, attachment_id);

CREATE TABLE IF NOT EXISTS sync_blob_upload_sessions (
    upload_id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL,
    owner_user_id TEXT NOT NULL,
    device_id TEXT,
    attachment_id TEXT NOT NULL,
    domain TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    content_type TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    payload_hash TEXT NOT NULL,
    chunk_size INTEGER NOT NULL,
    chunk_count INTEGER NOT NULL,
    reserved_quota_bytes INTEGER NOT NULL,
    status TEXT NOT NULL,
    idempotency_key TEXT,
    expires_at TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    blob_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(dataset_id, device_id, idempotency_key)
);
CREATE INDEX IF NOT EXISTS idx_sync_blob_upload_sessions_owner
    ON sync_blob_upload_sessions(owner_user_id, dataset_id, status);
CREATE INDEX IF NOT EXISTS idx_sync_blob_upload_sessions_hash
    ON sync_blob_upload_sessions(dataset_id, payload_hash);

CREATE TABLE IF NOT EXISTS sync_blob_chunks (
    upload_id TEXT NOT NULL,
    dataset_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    offset_bytes INTEGER NOT NULL,
    size_bytes INTEGER NOT NULL,
    chunk_hash TEXT NOT NULL,
    storage_key TEXT NOT NULL,
    received_at TEXT NOT NULL,
    PRIMARY KEY(upload_id, chunk_index)
);
CREATE INDEX IF NOT EXISTS idx_sync_blob_chunks_dataset
    ON sync_blob_chunks(dataset_id, upload_id);
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
    status TEXT NOT NULL DEFAULT 'active',
    user_label TEXT,
    authorized_at TIMESTAMPTZ,
    revoked_at TIMESTAMPTZ,
    revoked_reason TEXT
);
CREATE INDEX IF NOT EXISTS idx_sync_devices_user ON sync_devices(user_id);
CREATE INDEX IF NOT EXISTS idx_sync_devices_user_status
    ON sync_devices(user_id, status, last_seen_at);

CREATE TABLE IF NOT EXISTS sync_device_authorizations (
    authorization_id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    device_id TEXT NOT NULL,
    authorization_method TEXT NOT NULL,
    status TEXT NOT NULL,
    requested_at TIMESTAMPTZ NOT NULL,
    approved_at TIMESTAMPTZ,
    approving_device_id TEXT,
    idempotency_key TEXT,
    approval_idempotency_key TEXT,
    UNIQUE(dataset_id, device_id, idempotency_key)
);
CREATE INDEX IF NOT EXISTS idx_sync_device_authorizations_dataset_device
    ON sync_device_authorizations(dataset_id, device_id, requested_at);
CREATE INDEX IF NOT EXISTS idx_sync_device_authorizations_user_status
    ON sync_device_authorizations(user_id, status, requested_at);

CREATE TABLE IF NOT EXISTS sync_device_domain_acks (
    dataset_id TEXT NOT NULL,
    device_id TEXT NOT NULL,
    domain TEXT NOT NULL,
    through_server_sequence BIGINT NOT NULL DEFAULT 0,
    applied_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    idempotency_key TEXT,
    PRIMARY KEY(dataset_id, device_id, domain)
);
CREATE INDEX IF NOT EXISTS idx_sync_device_domain_acks_device
    ON sync_device_domain_acks(device_id, dataset_id);

CREATE TABLE IF NOT EXISTS sync_device_blob_acks (
    dataset_id TEXT NOT NULL,
    device_id TEXT NOT NULL,
    attachment_id TEXT NOT NULL,
    payload_hash TEXT NOT NULL,
    verified_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    idempotency_key TEXT,
    PRIMARY KEY(dataset_id, device_id, attachment_id)
);
CREATE INDEX IF NOT EXISTS idx_sync_device_blob_acks_device
    ON sync_device_blob_acks(device_id, dataset_id);

CREATE TABLE IF NOT EXISTS sync_background_policies (
    dataset_id TEXT NOT NULL,
    device_id TEXT NOT NULL,
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    minimum_interval_seconds INTEGER NOT NULL DEFAULT 300,
    backoff_floor_seconds INTEGER NOT NULL DEFAULT 60,
    max_batch_size INTEGER NOT NULL DEFAULT 100,
    max_blob_bytes_per_run BIGINT,
    respect_metered_networks BOOLEAN NOT NULL DEFAULT TRUE,
    maintenance_window_json TEXT,
    paused_reason TEXT,
    pending_local_changes BOOLEAN NOT NULL DEFAULT FALSE,
    updated_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY(dataset_id, device_id)
);
CREATE INDEX IF NOT EXISTS idx_sync_background_policies_device
    ON sync_background_policies(device_id, dataset_id);

CREATE TABLE IF NOT EXISTS sync_background_leases (
    dataset_id TEXT NOT NULL,
    device_id TEXT NOT NULL,
    lease_id TEXT NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY(dataset_id, device_id)
);
CREATE INDEX IF NOT EXISTS idx_sync_background_leases_expiry
    ON sync_background_leases(expires_at);

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
    client_profile_id TEXT,
    client_sequence BIGINT,
    client_timestamp TIMESTAMPTZ,
    server_timestamp TIMESTAMPTZ NOT NULL,
    base_server_cursor BIGINT,
    base_object_revision BIGINT,
    base_object_hash TEXT,
    object_revision BIGINT,
    parent_id TEXT,
    schema_version INTEGER NOT NULL DEFAULT 1,
    base_version TEXT,
    entity_version TEXT,
    dependency_json TEXT NOT NULL DEFAULT '[]',
    routing_metadata_json TEXT NOT NULL DEFAULT '{}',
    payload_ciphertext TEXT,
    payload_json TEXT NOT NULL DEFAULT '{}',
    payload_clear_json TEXT NOT NULL DEFAULT '{}',
    payload_hash TEXT,
    payload_size_bytes INTEGER,
    created_at_client TIMESTAMPTZ,
    received_at_server TIMESTAMPTZ,
    deleted BOOLEAN NOT NULL DEFAULT FALSE,
    encryption_metadata_json TEXT NOT NULL DEFAULT '{}',
    adapter_version INTEGER NOT NULL,
    status TEXT NOT NULL,
    apply_status TEXT NOT NULL DEFAULT 'pending',
    apply_error_code TEXT,
    apply_error_message TEXT,
    applied_at TIMESTAMPTZ,
    UNIQUE (dataset_id, client_envelope_id)
);
CREATE INDEX IF NOT EXISTS idx_sync_envelopes_dataset_sequence
    ON sync_envelopes(dataset_id, server_sequence);
CREATE INDEX IF NOT EXISTS idx_sync_envelopes_dataset_domain_sequence
    ON sync_envelopes(dataset_id, domain, server_sequence);
CREATE INDEX IF NOT EXISTS idx_sync_envelopes_dataset_domain_object
    ON sync_envelopes(dataset_id, domain, entity_id);
CREATE INDEX IF NOT EXISTS idx_sync_envelopes_dataset_status_sequence
    ON sync_envelopes(dataset_id, status, server_sequence);
CREATE INDEX IF NOT EXISTS idx_sync_envelopes_dataset_device_sequence
    ON sync_envelopes(dataset_id, device_id, server_sequence);
CREATE UNIQUE INDEX IF NOT EXISTS idx_sync_envelopes_dataset_device_client_sequence
    ON sync_envelopes(dataset_id, device_id, client_sequence)
    WHERE device_id IS NOT NULL AND client_sequence IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_sync_envelopes_payload_hash
    ON sync_envelopes(dataset_id, payload_hash);
CREATE INDEX IF NOT EXISTS idx_sync_envelopes_failed_apply
    ON sync_envelopes(dataset_id, apply_status, server_sequence)
    WHERE apply_status = 'failed';
CREATE INDEX IF NOT EXISTS idx_sync_envelopes_entity
    ON sync_envelopes(dataset_id, domain, entity_id);
CREATE INDEX IF NOT EXISTS idx_sync_envelopes_stable_key
    ON sync_envelopes(dataset_id, domain, stable_key);

CREATE TABLE IF NOT EXISTS sync_object_state (
    dataset_id TEXT NOT NULL,
    domain TEXT NOT NULL,
    object_id TEXT NOT NULL,
    object_revision BIGINT NOT NULL,
    object_hash TEXT NOT NULL,
    latest_server_cursor BIGINT NOT NULL,
    deleted BOOLEAN NOT NULL DEFAULT FALSE,
    updated_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (dataset_id, domain, object_id)
);
CREATE INDEX IF NOT EXISTS idx_sync_object_state_dataset_domain_object
    ON sync_object_state(dataset_id, domain, object_id);

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
    rotation_source_key_record_ids_json TEXT NOT NULL DEFAULT '[]',
    encryption_policy TEXT NOT NULL DEFAULT 'server_trusted_v1',
    key_epoch INTEGER NOT NULL DEFAULT 1,
    active_from_server_sequence INTEGER,
    superseded_at TIMESTAMPTZ,
    wrapped_for TEXT NOT NULL DEFAULT 'recovery',
    rewrap_status TEXT NOT NULL DEFAULT 'not_required',
    created_at TIMESTAMPTZ NOT NULL,
    revoked_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_sync_key_records_dataset
    ON sync_key_records(dataset_id, key_purpose, created_at);
CREATE INDEX IF NOT EXISTS idx_sync_key_records_device
    ON sync_key_records(dataset_id, device_id);
CREATE INDEX IF NOT EXISTS idx_sync_key_records_epoch
    ON sync_key_records(dataset_id, encryption_policy, key_epoch);
CREATE INDEX IF NOT EXISTS idx_sync_key_records_rewrap
    ON sync_key_records(dataset_id, rewrap_status);

CREATE TABLE IF NOT EXISTS sync_attachments (
    attachment_id TEXT NOT NULL,
    dataset_id TEXT NOT NULL,
    domain TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    content_type TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    payload_ciphertext TEXT NOT NULL,
    payload_hash TEXT NOT NULL,
    encryption_policy TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (dataset_id, attachment_id)
);
CREATE INDEX IF NOT EXISTS idx_sync_attachments_dataset_domain
    ON sync_attachments(dataset_id, domain, created_at);
CREATE INDEX IF NOT EXISTS idx_sync_attachments_dataset_hash
    ON sync_attachments(dataset_id, payload_hash);

CREATE TABLE IF NOT EXISTS sync_blob_objects (
    blob_id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL,
    owner_user_id TEXT NOT NULL,
    attachment_id TEXT NOT NULL,
    payload_hash TEXT NOT NULL,
    content_type TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    encryption_policy TEXT NOT NULL,
    storage_backend TEXT NOT NULL,
    storage_key TEXT NOT NULL,
    status TEXT NOT NULL,
    ref_count INTEGER NOT NULL DEFAULT 1,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    deleted_at TIMESTAMPTZ,
    UNIQUE(dataset_id, payload_hash)
);
CREATE INDEX IF NOT EXISTS idx_sync_blob_objects_owner
    ON sync_blob_objects(owner_user_id, dataset_id, status);
CREATE INDEX IF NOT EXISTS idx_sync_blob_objects_attachment
    ON sync_blob_objects(dataset_id, attachment_id);

CREATE TABLE IF NOT EXISTS sync_blob_upload_sessions (
    upload_id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL,
    owner_user_id TEXT NOT NULL,
    device_id TEXT,
    attachment_id TEXT NOT NULL,
    domain TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    content_type TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    payload_hash TEXT NOT NULL,
    chunk_size INTEGER NOT NULL,
    chunk_count INTEGER NOT NULL,
    reserved_quota_bytes INTEGER NOT NULL,
    status TEXT NOT NULL,
    idempotency_key TEXT,
    expires_at TIMESTAMPTZ,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    blob_id TEXT,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    UNIQUE(dataset_id, device_id, idempotency_key)
);
CREATE INDEX IF NOT EXISTS idx_sync_blob_upload_sessions_owner
    ON sync_blob_upload_sessions(owner_user_id, dataset_id, status);
CREATE INDEX IF NOT EXISTS idx_sync_blob_upload_sessions_hash
    ON sync_blob_upload_sessions(dataset_id, payload_hash);

CREATE TABLE IF NOT EXISTS sync_blob_chunks (
    upload_id TEXT NOT NULL,
    dataset_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    offset_bytes INTEGER NOT NULL,
    size_bytes INTEGER NOT NULL,
    chunk_hash TEXT NOT NULL,
    storage_key TEXT NOT NULL,
    received_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY(upload_id, chunk_index)
);
CREATE INDEX IF NOT EXISTS idx_sync_blob_chunks_dataset
    ON sync_blob_chunks(dataset_id, upload_id);
"""


def utcnow_iso() -> str:
    """Return an ISO-8601 UTC timestamp for Sync v2 rows."""

    return datetime.now(timezone.utc).isoformat()


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


def _domain_filter_sql(domains: Sequence[SyncDomain] | None, params: list[Any]) -> str:
    """Append a portable domain IN predicate and return its SQL fragment."""

    if domains is None:
        return ""
    placeholders = ", ".join("?" for _ in domains)
    params.extend(domains)
    return f" AND domain IN ({placeholders})"


def _manifest_attachment_size_class(size_bytes: int) -> str:
    if size_bytes <= 1_048_576:
        return "small"
    if size_bytes <= 16_777_216:
        return "medium"
    return "large"


def _timestamp_to_string(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _optional_int_from_storage(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _key_rotation_source_ids_from_storage(value: str | None) -> tuple[str, ...]:
    decoded = decode_json(value, default=[])
    if not isinstance(decoded, list):
        return ()
    return tuple(
        sorted(
            {
                str(source_id).strip()
                for source_id in decoded
                if str(source_id).strip()
            }
        )
    )


def _canonical_key_rotation_source_ids(
    source_key_record_ids: Sequence[str] | None,
) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                str(source_id).strip()
                for source_id in source_key_record_ids or ()
                if str(source_id).strip()
            }
        )
    )


def _parse_iso_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _add_seconds_iso(value: str, seconds: int) -> str:
    return (_parse_iso_datetime(value) + timedelta(seconds=seconds)).isoformat()


def _bool_from_storage(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y"}
    return bool(value)


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
    status = row.get("status") or ("revoked" if row.get("revoked_at") else "active")
    return SyncDevice(
        device_id=row["device_id"],
        user_id=row["user_id"],
        display_name=row["display_name"],
        client_type=row["client_type"],
        client_version=row.get("client_version"),
        capabilities=decode_json(row.get("capabilities_json"), default={}),
        registered_at=_timestamp_to_string(row.get("registered_at")) or "",
        last_seen_at=_timestamp_to_string(row.get("last_seen_at")) or "",
        status=status,
        user_label=row.get("user_label"),
        authorized_at=_timestamp_to_string(row.get("authorized_at")),
        revoked_at=_timestamp_to_string(row.get("revoked_at")),
        revoked_reason=row.get("revoked_reason"),
    )


def _device_authorization_from_row(row: dict[str, Any]) -> SyncDeviceAuthorization:
    return SyncDeviceAuthorization(
        authorization_id=row["authorization_id"],
        dataset_id=row["dataset_id"],
        user_id=row["user_id"],
        device_id=row["device_id"],
        authorization_method=row["authorization_method"],
        status=row["status"],
        requested_at=_timestamp_to_string(row.get("requested_at")) or "",
        approved_at=_timestamp_to_string(row.get("approved_at")),
        approving_device_id=row.get("approving_device_id"),
        idempotency_key=row.get("idempotency_key"),
    )


def _device_domain_ack_from_row(row: dict[str, Any]) -> SyncDeviceDomainAck:
    return SyncDeviceDomainAck(
        dataset_id=row["dataset_id"],
        device_id=row["device_id"],
        domain=row["domain"],
        through_server_sequence=int(row["through_server_sequence"]),
        applied_at=_timestamp_to_string(row.get("applied_at")) or "",
        updated_at=_timestamp_to_string(row.get("updated_at")) or "",
        idempotency_key=row.get("idempotency_key"),
    )


def _device_blob_ack_from_row(row: dict[str, Any]) -> SyncDeviceBlobAck:
    return SyncDeviceBlobAck(
        dataset_id=row["dataset_id"],
        device_id=row["device_id"],
        attachment_id=row["attachment_id"],
        payload_hash=row["payload_hash"],
        verified_at=_timestamp_to_string(row.get("verified_at")) or "",
        updated_at=_timestamp_to_string(row.get("updated_at")) or "",
        idempotency_key=row.get("idempotency_key"),
    )


def _background_policy_from_row(row: dict[str, Any]) -> SyncBackgroundPolicy:
    maintenance_window = decode_json(row.get("maintenance_window_json"), default=None)
    if maintenance_window is not None and not isinstance(maintenance_window, dict):
        maintenance_window = None
    return SyncBackgroundPolicy(
        dataset_id=row["dataset_id"],
        device_id=row["device_id"],
        enabled=_bool_from_storage(row.get("enabled")),
        minimum_interval_seconds=int(row["minimum_interval_seconds"]),
        backoff_floor_seconds=int(row["backoff_floor_seconds"]),
        max_batch_size=int(row["max_batch_size"]),
        max_blob_bytes_per_run=(
            int(row["max_blob_bytes_per_run"])
            if row.get("max_blob_bytes_per_run") is not None
            else None
        ),
        respect_metered_networks=_bool_from_storage(row.get("respect_metered_networks")),
        maintenance_window=maintenance_window,
        paused_reason=row.get("paused_reason"),
        pending_local_changes=_bool_from_storage(row.get("pending_local_changes")),
        updated_at=_timestamp_to_string(row.get("updated_at")) or "",
    )


def _background_lease_from_row(
    row: dict[str, Any],
    *,
    status: str,
    acquired: bool,
) -> SyncBackgroundLease:
    return SyncBackgroundLease(
        dataset_id=row["dataset_id"],
        device_id=row["device_id"],
        lease_id=row["lease_id"],
        status=status,
        acquired=acquired,
        expires_at=_timestamp_to_string(row.get("expires_at")) or "",
        updated_at=_timestamp_to_string(row.get("updated_at")) or "",
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
        server_cursor=int(row["server_sequence"]),
        dataset_id=row["dataset_id"],
        client_envelope_id=row["client_envelope_id"],
        domain=row["domain"],
        object_id=row["entity_id"],
        operation=row["operation"],
        envelope_id=f"srv_env_{int(row['server_sequence']):012d}",
        device_id=row.get("device_id"),
        client_profile_id=row.get("client_profile_id"),
        client_sequence=(
            int(row["client_sequence"])
            if row.get("client_sequence") is not None
            else None
        ),
        stable_key=row.get("stable_key"),
        created_at_client=row.get("created_at_client") or row.get("client_timestamp"),
        received_at_server=row.get("received_at_server") or row.get("server_timestamp"),
        client_timestamp=row.get("client_timestamp") or row.get("created_at_client"),
        server_timestamp=row.get("server_timestamp") or row.get("received_at_server"),
        base_server_cursor=(
            int(row["base_server_cursor"])
            if row.get("base_server_cursor") is not None
            else None
        ),
        base_object_revision=(
            int(row["base_object_revision"])
            if row.get("base_object_revision") is not None
            else None
        ),
        base_object_hash=row.get("base_object_hash"),
        object_revision=(
            int(row["object_revision"])
            if row.get("object_revision") is not None
            else None
        ),
        parent_id=row.get("parent_id"),
        schema_version=int(row.get("schema_version") or row.get("adapter_version") or 1),
        base_version=_version_from_storage(row.get("base_version")),
        entity_version=_version_from_storage(row.get("entity_version")),
        dependencies=decode_json(row.get("dependency_json"), default=[]),
        routing_metadata=decode_json(row.get("routing_metadata_json"), default={}),
        payload_ciphertext=row.get("payload_ciphertext"),
        payload=decode_json(
            row.get("payload_json") or row.get("payload_clear_json"),
            default={},
        ),
        payload_clear=decode_json(
            row.get("payload_clear_json") or row.get("payload_json"),
            default={},
        ),
        payload_hash=row.get("payload_hash"),
        payload_size_bytes=(
            int(row["payload_size_bytes"])
            if row.get("payload_size_bytes") is not None
            else None
        ),
        deleted=_bool_from_storage(row.get("deleted")),
        encryption_metadata=decode_json(row.get("encryption_metadata_json"), default={}),
        adapter_version=int(row.get("adapter_version") or row.get("schema_version") or 1),
        status=row["status"],
        apply_status=row.get("apply_status") or "pending",
        apply_error_code=row.get("apply_error_code"),
        apply_error_message=row.get("apply_error_message"),
        applied_at=row.get("applied_at"),
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
        object_id=row["entity_id"],
        conflict_type=row["conflict_type"],
        status=row["status"],
        base_envelope_id=row.get("base_envelope_id"),
        local_envelope_id=row.get("local_envelope_id"),
        remote_envelope_id=row.get("remote_envelope_id"),
        server_cursor=(
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
        rotation_source_key_record_ids=_key_rotation_source_ids_from_storage(
            row.get("rotation_source_key_record_ids_json")
        ),
        created_at=row["created_at"],
        revoked_at=row.get("revoked_at"),
        encryption_policy=row.get("encryption_policy") or DEFAULT_M1_ENCRYPTION_POLICY,
        key_epoch=int(row.get("key_epoch") or 1),
        active_from_server_sequence=_optional_int_from_storage(
            row.get("active_from_server_sequence")
        ),
        superseded_at=_timestamp_to_string(row.get("superseded_at")),
        wrapped_for=row.get("wrapped_for") or "recovery",
        rewrap_status=row.get("rewrap_status") or "not_required",
    )


def _attachment_from_row(row: dict[str, Any], *, stored: bool = True) -> SyncAttachment:
    """Convert a sync_attachments row into a core attachment model."""

    return SyncAttachment(
        attachment_id=row["attachment_id"],
        dataset_id=row["dataset_id"],
        domain=row["domain"],
        object_id=row["entity_id"],
        content_type=row["content_type"],
        size_bytes=int(row["size_bytes"]),
        payload_ciphertext=row["payload_ciphertext"],
        payload_hash=row["payload_hash"],
        encryption_policy=row["encryption_policy"],
        metadata=decode_json(row.get("metadata_json"), default={}),
        created_at=row["created_at"],
        stored=stored,
    )


def _blob_upload_session_from_row(
    row: dict[str, Any],
    *,
    uploaded_chunks: Sequence[int] | None = None,
) -> SyncBlobUploadSession:
    uploaded = sorted(int(index) for index in uploaded_chunks or [])
    chunk_count = int(row["chunk_count"])
    missing = [index for index in range(chunk_count) if index not in set(uploaded)]
    return SyncBlobUploadSession(
        upload_id=row["upload_id"],
        dataset_id=row["dataset_id"],
        attachment_id=row["attachment_id"],
        status=row["status"],
        chunk_size=int(row["chunk_size"]),
        chunk_count=chunk_count,
        size_bytes=int(row["size_bytes"]),
        payload_hash=row["payload_hash"],
        content_type=row["content_type"],
        device_id=row.get("device_id"),
        uploaded_chunks=uploaded,
        missing_chunks=missing,
        quota={"reserved_blob_bytes": int(row["reserved_quota_bytes"])},
        expires_at=_timestamp_to_string(row.get("expires_at")),
        blob_id=row.get("blob_id"),
    )


def _blob_chunk_from_row(row: dict[str, Any]) -> SyncBlobChunk:
    return SyncBlobChunk(
        upload_id=row["upload_id"],
        dataset_id=row["dataset_id"],
        chunk_index=int(row["chunk_index"]),
        offset_bytes=int(row["offset_bytes"]),
        size_bytes=int(row["size_bytes"]),
        chunk_hash=row["chunk_hash"],
        storage_key=row["storage_key"],
        received_at=_timestamp_to_string(row.get("received_at")) or "",
    )


def _blob_object_from_row(row: dict[str, Any]) -> SyncBlobObject:
    return SyncBlobObject(
        blob_id=row["blob_id"],
        dataset_id=row["dataset_id"],
        owner_user_id=row["owner_user_id"],
        attachment_id=row["attachment_id"],
        payload_hash=row["payload_hash"],
        content_type=row["content_type"],
        size_bytes=int(row["size_bytes"]),
        encryption_policy=row["encryption_policy"],
        storage_backend=row["storage_backend"],
        storage_key=row["storage_key"],
        status=row["status"],
        ref_count=int(row["ref_count"]),
        metadata=decode_json(row.get("metadata_json"), default={}),
        created_at=_timestamp_to_string(row.get("created_at")) or "",
        updated_at=_timestamp_to_string(row.get("updated_at")) or "",
        deleted_at=_timestamp_to_string(row.get("deleted_at")),
    )


def _object_state_from_row(row: dict[str, Any]) -> SyncObjectState:
    return SyncObjectState(
        dataset_id=row["dataset_id"],
        domain=row["domain"],
        object_id=row["object_id"],
        object_revision=int(row["object_revision"]),
        object_hash=row["object_hash"],
        latest_server_cursor=int(row["latest_server_cursor"]),
        deleted=_bool_from_storage(row.get("deleted")),
        updated_at=row.get("updated_at"),
    )


def _conflict_row_has_resolution_claim(row: dict[str, Any]) -> bool:
    return (
        row["status"] == "unresolved"
        and row.get("resolved_at") is None
        and row.get("resolved_by_envelope_id") is None
        and (
            row.get("resolution_action") is not None
            or row.get("resolved_by_device_id") is not None
            or row.get("resolution_notes") is not None
        )
    )


def _conflict_row_matches_resolution_claim(
    row: dict[str, Any],
    *,
    resolved_by_device_id: str | None,
    resolution_action: str | None,
    resolution_notes: str | None,
) -> bool:
    return (
        _conflict_row_has_resolution_claim(row)
        and row.get("resolution_action") == resolution_action
        and row.get("resolved_by_device_id") == resolved_by_device_id
        and row.get("resolution_notes") == resolution_notes
    )


def _dataset_domains_from_row(row: dict[str, Any]) -> set[str]:
    domains = decode_json(row.get("domain_set_json"), default=[])
    return {str(domain) for domain in domains}


def _envelope_fingerprint_from_create(envelope: SyncEnvelopeCreate) -> dict[str, Any]:
    return {
        "dataset_id": envelope.dataset_id,
        "domain": envelope.domain,
        "object_id": envelope.object_id,
        "stable_key": envelope.stable_key,
        "operation": envelope.operation,
        "client_envelope_id": envelope.client_envelope_id,
        "device_id": envelope.device_id,
        "client_profile_id": envelope.client_profile_id,
        "client_sequence": envelope.client_sequence,
        "created_at_client": envelope.created_at_client,
        "base_server_cursor": envelope.base_server_cursor,
        "base_object_revision": envelope.base_object_revision,
        "base_object_hash": envelope.base_object_hash,
        "object_revision": envelope.object_revision,
        "parent_id": envelope.parent_id,
        "schema_version": envelope.schema_version,
        "base_version": envelope.base_version,
        "entity_version": envelope.entity_version,
        "dependencies": envelope.dependencies,
        "routing_metadata": envelope.routing_metadata,
        "payload_ciphertext": envelope.payload_ciphertext,
        "payload": envelope.payload,
        "payload_hash": envelope.payload_hash,
        "payload_size_bytes": envelope.payload_size_bytes,
        "deleted": envelope.deleted,
        "encryption_metadata": envelope.encryption_metadata,
        "adapter_version": envelope.adapter_version,
        "status": envelope.status,
    }


def _envelope_fingerprint_from_row(
    row: dict[str, Any],
    *,
    ignore_client_envelope_id: bool = False,
) -> dict[str, Any]:
    payload_size_bytes = row.get("payload_size_bytes")
    fingerprint = {
        "dataset_id": row["dataset_id"],
        "domain": row["domain"],
        "object_id": row["entity_id"],
        "stable_key": row.get("stable_key"),
        "operation": row["operation"],
        "device_id": row.get("device_id"),
        "client_profile_id": row.get("client_profile_id"),
        "client_sequence": (
            int(row["client_sequence"])
            if row.get("client_sequence") is not None
            else None
        ),
        "created_at_client": row.get("created_at_client") or row.get("client_timestamp"),
        "base_server_cursor": (
            int(row["base_server_cursor"])
            if row.get("base_server_cursor") is not None
            else None
        ),
        "base_object_revision": (
            int(row["base_object_revision"])
            if row.get("base_object_revision") is not None
            else None
        ),
        "base_object_hash": row.get("base_object_hash"),
        "object_revision": (
            int(row["object_revision"])
            if row.get("object_revision") is not None
            else None
        ),
        "parent_id": row.get("parent_id"),
        "schema_version": int(row.get("schema_version") or row.get("adapter_version") or 1),
        "base_version": _version_from_storage(row.get("base_version")),
        "entity_version": _version_from_storage(row.get("entity_version")),
        "dependencies": decode_json(row.get("dependency_json"), default=[]),
        "routing_metadata": decode_json(row.get("routing_metadata_json"), default={}),
        "payload_ciphertext": row.get("payload_ciphertext"),
        "payload": decode_json(
            row.get("payload_json") or row.get("payload_clear_json"),
            default={},
        ),
        "payload_hash": row.get("payload_hash"),
        "payload_size_bytes": int(payload_size_bytes) if payload_size_bytes is not None else None,
        "deleted": _bool_from_storage(row.get("deleted")),
        "encryption_metadata": decode_json(row.get("encryption_metadata_json"), default={}),
        "adapter_version": int(row.get("adapter_version") or row.get("schema_version") or 1),
        "status": row["status"],
    }
    if not ignore_client_envelope_id:
        fingerprint["client_envelope_id"] = row["client_envelope_id"]
    return fingerprint


def _envelope_sequence_fingerprint_from_create(
    envelope: SyncEnvelopeCreate,
) -> dict[str, Any]:
    fingerprint = _envelope_fingerprint_from_create(envelope)
    fingerprint.pop("client_envelope_id", None)
    return fingerprint


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
        "rotation_source_key_record_ids": record.rotation_source_key_record_ids,
        "revoked_at": record.revoked_at,
        "encryption_policy": record.encryption_policy,
        "key_epoch": record.key_epoch,
        "active_from_server_sequence": record.active_from_server_sequence,
        "superseded_at": record.superseded_at,
        "wrapped_for": record.wrapped_for,
        "rewrap_status": record.rewrap_status,
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
        "rotation_source_key_record_ids": _key_rotation_source_ids_from_storage(
            row.get("rotation_source_key_record_ids_json")
        ),
        "revoked_at": row.get("revoked_at"),
        "encryption_policy": row.get("encryption_policy") or DEFAULT_M1_ENCRYPTION_POLICY,
        "key_epoch": int(row.get("key_epoch") or 1),
        "active_from_server_sequence": _optional_int_from_storage(
            row.get("active_from_server_sequence")
        ),
        "superseded_at": _timestamp_to_string(row.get("superseded_at")),
        "wrapped_for": row.get("wrapped_for") or "recovery",
        "rewrap_status": row.get("rewrap_status") or "not_required",
    }


def _key_rotation_record_matches_request(
    existing: SyncKeyRecord,
    requested: SyncKeyRecordCreate,
) -> bool:
    return (
        existing.key_record_id == requested.key_record_id
        and existing.dataset_id == requested.dataset_id
        and existing.user_id == requested.user_id
        and existing.device_id == requested.device_id
        and existing.key_purpose == requested.key_purpose
        and existing.wrapped_key_blob == requested.wrapped_key_blob
        and existing.kdf_metadata == requested.kdf_metadata
        and existing.recovery_hint == requested.recovery_hint
        and existing.encryption_policy == requested.encryption_policy
        and existing.wrapped_for == requested.wrapped_for
        and existing.rewrap_status == requested.rewrap_status
        and existing.revoked_at == requested.revoked_at
    )


def _device_authorization_fingerprint_from_create(
    authorization: SyncDeviceAuthorizationCreate,
) -> dict[str, Any]:
    return {
        "dataset_id": authorization.dataset_id,
        "user_id": authorization.user_id,
        "device_id": authorization.device_id,
        "authorization_method": authorization.authorization_method,
    }


def _device_authorization_fingerprint_from_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset_id": row["dataset_id"],
        "user_id": row["user_id"],
        "device_id": row["device_id"],
        "authorization_method": row["authorization_method"],
    }


def _attachment_fingerprint_from_create(attachment: SyncAttachmentCreate) -> dict[str, Any]:
    """Return idempotency-comparable fields from an attachment create model."""

    return {
        "attachment_id": attachment.attachment_id,
        "dataset_id": attachment.dataset_id,
        "domain": attachment.domain,
        "object_id": attachment.object_id,
        "content_type": attachment.content_type,
        "size_bytes": attachment.size_bytes,
        "payload_ciphertext": attachment.payload_ciphertext,
        "payload_hash": attachment.payload_hash,
        "encryption_policy": attachment.encryption_policy,
        "metadata": attachment.metadata,
    }


def _attachment_fingerprint_from_row(row: dict[str, Any]) -> dict[str, Any]:
    """Return idempotency-comparable fields from a stored attachment row."""

    return {
        "attachment_id": row["attachment_id"],
        "dataset_id": row["dataset_id"],
        "domain": row["domain"],
        "object_id": row["entity_id"],
        "content_type": row["content_type"],
        "size_bytes": int(row["size_bytes"]),
        "payload_ciphertext": row["payload_ciphertext"],
        "payload_hash": row["payload_hash"],
        "encryption_policy": row["encryption_policy"],
        "metadata": decode_json(row.get("metadata_json"), default={}),
    }


def _blob_session_fingerprint_from_create(
    session: SyncBlobUploadSessionCreate,
) -> dict[str, Any]:
    return {
        "dataset_id": session.dataset_id,
        "owner_user_id": session.owner_user_id,
        "device_id": session.device_id,
        "attachment_id": session.attachment_id,
        "domain": session.domain,
        "object_id": session.object_id,
        "content_type": session.content_type,
        "size_bytes": session.size_bytes,
        "payload_hash": session.payload_hash,
        "chunk_size": session.chunk_size,
        "chunk_count": session.chunk_count,
        "reserved_quota_bytes": session.reserved_quota_bytes,
        "metadata": session.metadata,
    }


def _blob_session_fingerprint_from_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset_id": row["dataset_id"],
        "owner_user_id": row["owner_user_id"],
        "device_id": row.get("device_id"),
        "attachment_id": row["attachment_id"],
        "domain": row["domain"],
        "object_id": row["entity_id"],
        "content_type": row["content_type"],
        "size_bytes": int(row["size_bytes"]),
        "payload_hash": row["payload_hash"],
        "chunk_size": int(row["chunk_size"]),
        "chunk_count": int(row["chunk_count"]),
        "reserved_quota_bytes": int(row["reserved_quota_bytes"]),
        "metadata": decode_json(row.get("metadata_json"), default={}),
    }


def _blob_chunk_fingerprint_from_create(chunk: SyncBlobChunkCreate) -> dict[str, Any]:
    return {
        "upload_id": chunk.upload_id,
        "dataset_id": chunk.dataset_id,
        "chunk_index": chunk.chunk_index,
        "offset_bytes": chunk.offset_bytes,
        "size_bytes": chunk.size_bytes,
        "chunk_hash": chunk.chunk_hash,
        "storage_key": chunk.storage_key,
    }


def _blob_chunk_fingerprint_from_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "upload_id": row["upload_id"],
        "dataset_id": row["dataset_id"],
        "chunk_index": int(row["chunk_index"]),
        "offset_bytes": int(row["offset_bytes"]),
        "size_bytes": int(row["size_bytes"]),
        "chunk_hash": row["chunk_hash"],
        "storage_key": row["storage_key"],
    }


def _blob_object_fingerprint_from_create(blob: SyncBlobObjectCreate) -> dict[str, Any]:
    return {
        "dataset_id": blob.dataset_id,
        "owner_user_id": blob.owner_user_id,
        "payload_hash": blob.payload_hash,
        "content_type": blob.content_type,
        "size_bytes": blob.size_bytes,
        "encryption_policy": blob.encryption_policy,
        "storage_backend": blob.storage_backend,
        "storage_key": blob.storage_key,
        "status": blob.status,
        "metadata": blob.metadata,
    }


def _blob_object_fingerprint_from_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset_id": row["dataset_id"],
        "owner_user_id": row["owner_user_id"],
        "payload_hash": row["payload_hash"],
        "content_type": row["content_type"],
        "size_bytes": int(row["size_bytes"]),
        "encryption_policy": row["encryption_policy"],
        "storage_backend": row["storage_backend"],
        "storage_key": row["storage_key"],
        "status": row["status"],
        "metadata": decode_json(row.get("metadata_json"), default={}),
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
            raise SyncStoreError(
                f"Unsupported SYNC_V2_DATABASE_URL scheme for Sync v2: {scheme}"
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
            if self.backend.table_exists("sync_envelopes", connection=conn):
                self._ensure_envelope_m1_columns(connection=conn)
            if self.backend.table_exists("sync_key_records", connection=conn):
                self._ensure_key_record_user_id_column(connection=conn)
                self._ensure_key_record_rotation_columns(connection=conn)
            self.backend.create_tables(schema, connection=conn)
            self._ensure_device_lifecycle_columns(connection=conn)
            self._ensure_device_lifecycle_tables(connection=conn)
            self._ensure_background_sync_tables(connection=conn)
            self._ensure_envelope_m1_columns(connection=conn)
            self._ensure_sync_object_state_table(connection=conn)
            self._ensure_envelope_m1_indexes(connection=conn)
            self._ensure_key_record_user_id_column(connection=conn)
            self._ensure_key_record_rotation_columns(connection=conn)
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

    def _get_device_row(
        self,
        user_id: str,
        device_id: str,
        *,
        connection: Any | None = None,
    ) -> dict[str, Any] | None:
        return _first(
            self.execute(
                """
                SELECT * FROM sync_devices
                 WHERE user_id = ? AND device_id = ?
                """,
                (user_id, device_id),
                connection=connection,
            )
        )

    def _require_device_for_dataset(
        self,
        dataset_id: str,
        device_id: str,
        *,
        connection: Any,
    ) -> dict[str, Any]:
        dataset_row = self._require_dataset(dataset_id, connection=connection)
        if str(dataset_row["scope_type"]) == "workspace":
            row = _first(
                self.execute(
                    "SELECT * FROM sync_devices WHERE device_id = ?",
                    (device_id,),
                    connection=connection,
                )
            )
        else:
            row = self._get_device_row(
                str(dataset_row["owner_user_id"]),
                device_id,
                connection=connection,
            )
        if row is None:
            raise SyncStoreError(
                f"Sync device is not registered for dataset {dataset_id}: {device_id}"
            )
        return row

    def _validate_dataset_contract(self, dataset: SyncDatasetCreate) -> None:
        if dataset.encryption_policy != DEFAULT_M1_ENCRYPTION_POLICY:
            raise SyncStoreError(
                f"Sync v2 M1 datasets require {DEFAULT_M1_ENCRYPTION_POLICY}"
            )
        if dataset.scope_type == "personal":
            if dataset.workspace_id is not None:
                raise SyncStoreError("Personal sync datasets must not include workspace_id")
            allowed_domains = set(M1_SYNC_DOMAINS).union(SOURCE_CACHE_SYNC_DOMAINS, MEDIA_SYNC_DOMAINS)
        elif dataset.scope_type == "workspace":
            if not dataset.workspace_id or not dataset.workspace_id.strip():
                raise SyncStoreError("Workspace sync datasets require workspace_id")
            allowed_domains = set(WORKSPACE_SYNC_DOMAINS).union(SOURCE_CACHE_SYNC_DOMAINS, MEDIA_SYNC_DOMAINS)
        else:
            raise SyncStoreError(f"Unsupported sync dataset scope: {dataset.scope_type}")
        invalid_domains = sorted(set(dataset.domains).difference(allowed_domains))
        if invalid_domains:
            raise SyncInvalidDomainError(
                f"Sync v2 dataset scope {dataset.scope_type} contains unsupported domains: "
                + ", ".join(invalid_domains)
            )

    def _validate_envelope_contract(self, envelope: SyncEnvelopeCreate) -> None:
        if envelope.domain not in SYNC_V2_SUPPORTED_OPERATIONS:
            raise SyncInvalidDomainError(f"Sync v2 M1 domain is not supported: {envelope.domain}")
        if envelope.operation not in SYNC_V2_SUPPORTED_OPERATIONS[envelope.domain]:
            raise SyncStoreError(
                f"Sync v2 M1 operation {envelope.operation} is not supported for {envelope.domain}"
            )
        if not envelope.object_id or not envelope.object_id.strip():
            raise SyncStoreError("Sync v2 M1 envelopes require a non-empty object_id")
        if not envelope.payload_hash or not envelope.payload_hash.strip():
            raise SyncStoreError("Sync v2 M1 envelopes require a non-empty payload_hash")
        policy = envelope.encryption_metadata.get("policy", DEFAULT_M1_ENCRYPTION_POLICY)
        if policy != DEFAULT_M1_ENCRYPTION_POLICY:
            raise SyncStoreError(
                f"Sync v2 M1 envelopes require {DEFAULT_M1_ENCRYPTION_POLICY}"
            )
        base_values = (
            envelope.base_server_cursor,
            envelope.base_object_revision,
            envelope.base_object_hash,
        )
        has_any_base = any(value is not None for value in base_values)
        has_all_base = all(value is not None for value in base_values)
        if has_any_base and not has_all_base:
            raise SyncStoreError(
                "Sync v2 M1 base metadata must be supplied as a complete set"
            )
        if envelope.domain in _WHOLE_OBJECT_DOMAINS:
            if envelope.operation == "tombstone" and not has_all_base:
                raise SyncStoreError(
                    f"Sync v2 M1 {envelope.domain} tombstones require base metadata"
                )
            if (
                envelope.operation == "upsert"
                and envelope.object_revision is not None
                and envelope.object_revision > 1
                and not has_all_base
            ):
                raise SyncStoreError(
                    f"Sync v2 M1 {envelope.domain} updates require base metadata"
                )
        if envelope.domain == "chat.message" and envelope.operation == "append":
            if not envelope.object_id.strip() or not envelope.payload_hash.strip():
                raise SyncStoreError(
                    "Sync v2 M1 chat.message append envelopes require object_id and payload_hash"
                )
        if envelope.domain == "attachment.ref":
            missing = _ATTACHMENT_REF_REQUIRED_PAYLOAD_KEYS.difference(envelope.payload)
            if missing:
                raise SyncStoreError(
                    "Sync v2 M1 attachment.ref envelopes require payload metadata fields: "
                    + ", ".join(sorted(missing))
                )

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
                if existing.get("user_id") != device.user_id:
                    raise SyncStoreError(
                        f"Sync device already belongs to another user: {device.device_id}"
                    )
                existing_status = existing.get("status") or (
                    "revoked" if existing.get("revoked_at") else "active"
                )
                if existing_status == "revoked" and device.status != "revoked":
                    status = "revoked"
                    authorized_at = existing.get("authorized_at") or device.authorized_at
                    revoked_at = existing.get("revoked_at") or now
                    revoked_reason = existing.get("revoked_reason") or device.revoked_reason
                elif device.status == "revoked" or device.revoked_at is not None:
                    status = "revoked"
                    authorized_at = device.authorized_at or existing.get("authorized_at")
                    revoked_at = device.revoked_at or now
                    revoked_reason = device.revoked_reason or existing.get("revoked_reason")
                else:
                    status = device.status
                    authorized_at = device.authorized_at or existing.get("authorized_at")
                    revoked_at = None
                    revoked_reason = None
                self.execute(
                    """
                    UPDATE sync_devices
                       SET user_id = ?,
                           display_name = ?,
                           client_type = ?,
                           client_version = ?,
                           capabilities_json = ?,
                           last_seen_at = ?,
                           status = ?,
                           user_label = ?,
                           authorized_at = ?,
                           revoked_at = ?,
                           revoked_reason = ?
                     WHERE device_id = ?
                    """,
                    (
                        device.user_id,
                        device.display_name,
                        device.client_type,
                        device.client_version,
                        encode_json(device.capabilities, default={}),
                        now,
                        status,
                        device.user_label or existing.get("user_label"),
                        authorized_at,
                        revoked_at,
                        revoked_reason,
                        device.device_id,
                    ),
                    connection=conn,
                )
            else:
                status = (
                    "revoked"
                    if device.status == "revoked" or device.revoked_at is not None
                    else device.status
                )
                revoked_at = device.revoked_at or (now if status == "revoked" else None)
                self.execute(
                    """
                    INSERT INTO sync_devices (
                        device_id, user_id, display_name, client_type, client_version,
                        capabilities_json, registered_at, last_seen_at, status,
                        user_label, authorized_at, revoked_at, revoked_reason
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        status,
                        device.user_label,
                        device.authorized_at,
                        revoked_at,
                        device.revoked_reason,
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
        self._validate_dataset_contract(dataset)
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
                if existing.get("owner_user_id") != dataset.owner_user_id:
                    raise SyncStoreError(
                        f"Sync dataset already belongs to another user: {dataset.dataset_id}"
                    )
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

    def list_datasets_for_user(self, user_id: str) -> list[SyncDataset]:
        """List active Sync v2 datasets owned by a user."""

        result = self.execute(
            """
            SELECT * FROM sync_datasets
             WHERE owner_user_id = ? AND archived_at IS NULL
             ORDER BY created_at ASC, dataset_id ASC
            """,
            (user_id,),
        )
        return [_dataset_from_row(row) for row in result.rows]

    def get_device(self, user_id: str, device_id: str) -> SyncDevice | None:
        """Return one Sync v2 device for a user."""

        row = self._get_device_row(user_id, device_id)
        if row is None:
            return None
        return _device_from_row(row)

    def list_devices_for_user(
        self,
        user_id: str,
        *,
        include_revoked: bool = False,
    ) -> list[SyncDevice]:
        """List Sync v2 devices registered by a user."""

        sql = """
            SELECT * FROM sync_devices
             WHERE user_id = ?
        """
        params: list[Any] = [user_id]
        if not include_revoked:
            sql += " AND status <> 'revoked' AND revoked_at IS NULL"
        sql += " ORDER BY last_seen_at DESC, device_id ASC"
        result = self.execute(sql, tuple(params))
        return [_device_from_row(row) for row in result.rows]

    def create_device_authorization(
        self,
        authorization: SyncDeviceAuthorizationCreate,
    ) -> SyncDeviceAuthorization:
        """Create or idempotently return a device authorization request."""

        now = utcnow_iso()
        with self.backend.transaction() as conn:
            dataset_row = self._require_dataset(
                authorization.dataset_id,
                connection=conn,
            )
            if (
                str(dataset_row["scope_type"]) != "workspace"
                and str(dataset_row["owner_user_id"]) != str(authorization.user_id)
            ):
                raise SyncDatasetNotFoundError(
                    f"Sync dataset not found: {authorization.dataset_id}"
                )
            if (
                self._get_device_row(
                    authorization.user_id,
                    authorization.device_id,
                    connection=conn,
                )
                is None
            ):
                raise SyncStoreError(
                    f"Sync device is not registered: {authorization.device_id}"
                )
            if authorization.idempotency_key:
                existing = _first(
                    self.execute(
                        """
                        SELECT * FROM sync_device_authorizations
                         WHERE dataset_id = ? AND device_id = ? AND idempotency_key = ?
                        """,
                        (
                            authorization.dataset_id,
                            authorization.device_id,
                            authorization.idempotency_key,
                        ),
                        connection=conn,
                    )
                )
                if existing is not None:
                    if (
                        _device_authorization_fingerprint_from_row(existing)
                        != _device_authorization_fingerprint_from_create(authorization)
                    ):
                        raise SyncIdempotencyConflictError(
                            "Sync device authorization idempotency key was reused"
                        )
                    return _device_authorization_from_row(existing)
            self.execute(
                """
                INSERT INTO sync_device_authorizations (
                    authorization_id, dataset_id, user_id, device_id,
                    authorization_method, status, requested_at, idempotency_key
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (authorization_id) DO NOTHING
                """,
                (
                    authorization.authorization_id,
                    authorization.dataset_id,
                    authorization.user_id,
                    authorization.device_id,
                    authorization.authorization_method,
                    "pending",
                    now,
                    authorization.idempotency_key,
                ),
                connection=conn,
            )
            row = _first(
                self.execute(
                    """
                    SELECT * FROM sync_device_authorizations
                     WHERE authorization_id = ?
                    """,
                    (authorization.authorization_id,),
                    connection=conn,
                )
            )
            if row is None:
                raise SyncStoreError(
                    "Sync device authorization insert did not produce a retrievable record"
                )
            if (
                _device_authorization_fingerprint_from_row(row)
                != _device_authorization_fingerprint_from_create(authorization)
            ):
                raise SyncIdempotencyConflictError(
                    "Sync device authorization ID was reused"
                )
        return _device_authorization_from_row(row)

    def approve_device_authorization(
        self,
        authorization_id: str,
        *,
        user_id: str,
        dataset_id: str,
        approving_device_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> SyncDeviceAuthorization:
        """Approve a pending device authorization and activate the device."""

        now = utcnow_iso()
        with self.backend.transaction() as conn:
            row = _first(
                self.execute(
                    """
                    SELECT * FROM sync_device_authorizations
                     WHERE authorization_id = ?
                       AND dataset_id = ?
                       AND user_id = ?
                    """,
                    (authorization_id, dataset_id, user_id),
                    connection=conn,
                )
            )
            if row is None:
                raise SyncStoreError(f"Sync device authorization not found: {authorization_id}")
            if row["status"] == "rejected":
                raise SyncStoreError("Sync device authorization has been rejected")
            if row["status"] == "pending":
                self.execute(
                    """
                    UPDATE sync_device_authorizations
                       SET status = 'approved',
                           approved_at = ?,
                           approving_device_id = ?,
                           approval_idempotency_key = ?
                     WHERE authorization_id = ?
                       AND status = 'pending'
                    """,
                    (now, approving_device_id, idempotency_key, authorization_id),
                    connection=conn,
                )
            device_row = self._get_device_row(user_id, row["device_id"], connection=conn)
            if device_row is None:
                raise SyncStoreError(f"Sync device is not registered: {row['device_id']}")
            self.execute(
                """
                UPDATE sync_devices
                   SET status = 'active',
                       authorized_at = COALESCE(authorized_at, ?),
                       revoked_at = NULL,
                       revoked_reason = NULL,
                       last_seen_at = ?
                 WHERE user_id = ? AND device_id = ?
                """,
                (now, now, user_id, row["device_id"]),
                connection=conn,
            )
            approved = _first(
                self.execute(
                    """
                    SELECT * FROM sync_device_authorizations
                     WHERE authorization_id = ?
                    """,
                    (authorization_id,),
                    connection=conn,
                )
            )
        return _device_authorization_from_row(approved)

    def revoke_device(
        self,
        *,
        user_id: str,
        device_id: str,
        reason: str | None = None,
        revoke_key_records: bool = False,
    ) -> SyncDevice:
        """Revoke a Sync v2 device and optionally revoke its key records."""

        now = utcnow_iso()
        with self.backend.transaction() as conn:
            existing = self._get_device_row(user_id, device_id, connection=conn)
            if existing is None:
                raise SyncStoreError(f"Sync device is not registered: {device_id}")
            self.execute(
                """
                UPDATE sync_devices
                   SET status = 'revoked',
                       revoked_at = COALESCE(revoked_at, ?),
                       revoked_reason = COALESCE(?, revoked_reason),
                       last_seen_at = ?
                 WHERE user_id = ? AND device_id = ?
                """,
                (now, reason, now, user_id, device_id),
                connection=conn,
            )
            if revoke_key_records:
                self.execute(
                    """
                    UPDATE sync_key_records
                       SET revoked_at = COALESCE(revoked_at, ?)
                     WHERE user_id = ?
                       AND device_id = ?
                    """,
                    (now, user_id, device_id),
                    connection=conn,
                )
            row = self._get_device_row(user_id, device_id, connection=conn)
        return _device_from_row(row)

    def upsert_device_domain_ack(
        self,
        acknowledgment: SyncDeviceDomainAckCreate,
    ) -> SyncDeviceDomainAck:
        """Record the highest accepted sequence a device has applied for a domain."""

        now = utcnow_iso()
        with self.backend.transaction() as conn:
            self._require_dataset_domain(
                acknowledgment.dataset_id,
                acknowledgment.domain,
                connection=conn,
            )
            self._require_device_for_dataset(
                acknowledgment.dataset_id,
                acknowledgment.device_id,
                connection=conn,
            )
            existing = _first(
                self.execute(
                    """
                    SELECT * FROM sync_device_domain_acks
                     WHERE dataset_id = ? AND device_id = ? AND domain = ?
                    """,
                    (
                        acknowledgment.dataset_id,
                        acknowledgment.device_id,
                        acknowledgment.domain,
                    ),
                    connection=conn,
                )
            )
            if existing is None:
                self.execute(
                    """
                    INSERT INTO sync_device_domain_acks (
                        dataset_id, device_id, domain, through_server_sequence,
                        applied_at, updated_at, idempotency_key
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        acknowledgment.dataset_id,
                        acknowledgment.device_id,
                        acknowledgment.domain,
                        acknowledgment.through_server_sequence,
                        acknowledgment.applied_at,
                        now,
                        acknowledgment.idempotency_key,
                    ),
                    connection=conn,
                )
            elif acknowledgment.through_server_sequence >= int(
                existing["through_server_sequence"]
            ):
                self.execute(
                    """
                    UPDATE sync_device_domain_acks
                       SET through_server_sequence = ?,
                           applied_at = ?,
                           updated_at = ?,
                           idempotency_key = ?
                     WHERE dataset_id = ? AND device_id = ? AND domain = ?
                    """,
                    (
                        acknowledgment.through_server_sequence,
                        acknowledgment.applied_at,
                        now,
                        acknowledgment.idempotency_key,
                        acknowledgment.dataset_id,
                        acknowledgment.device_id,
                        acknowledgment.domain,
                    ),
                    connection=conn,
                )
            row = _first(
                self.execute(
                    """
                    SELECT * FROM sync_device_domain_acks
                     WHERE dataset_id = ? AND device_id = ? AND domain = ?
                    """,
                    (
                        acknowledgment.dataset_id,
                        acknowledgment.device_id,
                        acknowledgment.domain,
                    ),
                    connection=conn,
                )
            )
        return _device_domain_ack_from_row(row)

    def upsert_device_blob_ack(
        self,
        acknowledgment: SyncDeviceBlobAckCreate,
    ) -> SyncDeviceBlobAck:
        """Record a device-level blob verification acknowledgment."""

        now = utcnow_iso()
        with self.backend.transaction() as conn:
            self._require_device_for_dataset(
                acknowledgment.dataset_id,
                acknowledgment.device_id,
                connection=conn,
            )
            self.execute(
                """
                INSERT INTO sync_device_blob_acks (
                    dataset_id, device_id, attachment_id, payload_hash,
                    verified_at, updated_at, idempotency_key
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (dataset_id, device_id, attachment_id)
                DO UPDATE SET
                    payload_hash = excluded.payload_hash,
                    verified_at = excluded.verified_at,
                    updated_at = excluded.updated_at,
                    idempotency_key = excluded.idempotency_key
                """,
                (
                    acknowledgment.dataset_id,
                    acknowledgment.device_id,
                    acknowledgment.attachment_id,
                    acknowledgment.payload_hash,
                    acknowledgment.verified_at,
                    now,
                    acknowledgment.idempotency_key,
                ),
                connection=conn,
            )
            row = _first(
                self.execute(
                    """
                    SELECT * FROM sync_device_blob_acks
                     WHERE dataset_id = ? AND device_id = ? AND attachment_id = ?
                    """,
                    (
                        acknowledgment.dataset_id,
                        acknowledgment.device_id,
                        acknowledgment.attachment_id,
                    ),
                    connection=conn,
                )
            )
        return _device_blob_ack_from_row(row)

    def list_device_acknowledgments(
        self,
        dataset_id: str,
        device_id: str,
    ) -> SyncDeviceAcknowledgmentSummary:
        """Return all domain and blob acknowledgments for one device in a dataset."""

        with self.backend.transaction() as conn:
            self._require_device_for_dataset(dataset_id, device_id, connection=conn)
            domain_rows = self.execute(
                """
                SELECT * FROM sync_device_domain_acks
                 WHERE dataset_id = ? AND device_id = ?
                 ORDER BY domain ASC
                """,
                (dataset_id, device_id),
                connection=conn,
            ).rows
            blob_rows = self.execute(
                """
                SELECT * FROM sync_device_blob_acks
                 WHERE dataset_id = ? AND device_id = ?
                 ORDER BY updated_at ASC, attachment_id ASC
                """,
                (dataset_id, device_id),
                connection=conn,
            ).rows
        domain_acks = {
            row["domain"]: _device_domain_ack_from_row(row)
            for row in domain_rows
        }
        return SyncDeviceAcknowledgmentSummary(
            dataset_id=dataset_id,
            device_id=device_id,
            domain_acks=domain_acks,
            blob_acks=[_device_blob_ack_from_row(row) for row in blob_rows],
        )

    def get_background_policy(
        self,
        dataset_id: str,
        device_id: str,
    ) -> SyncBackgroundPolicy | None:
        """Return stored background sync policy for one dataset/device."""

        row = _first(
            self.execute(
                """
                SELECT * FROM sync_background_policies
                 WHERE dataset_id = ? AND device_id = ?
                """,
                (dataset_id, device_id),
            )
        )
        if row is None:
            return None
        return _background_policy_from_row(row)

    def upsert_background_policy(
        self,
        policy: SyncBackgroundPolicyUpsert,
    ) -> SyncBackgroundPolicy:
        """Store background sync policy hints and user intent."""

        now = utcnow_iso()
        with self.backend.transaction() as conn:
            self._require_device_for_dataset(
                policy.dataset_id,
                policy.device_id,
                connection=conn,
            )
            self.execute(
                """
                INSERT INTO sync_background_policies (
                    dataset_id, device_id, enabled, minimum_interval_seconds,
                    backoff_floor_seconds, max_batch_size, max_blob_bytes_per_run,
                    respect_metered_networks, maintenance_window_json, paused_reason,
                    pending_local_changes, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (dataset_id, device_id)
                DO UPDATE SET
                    enabled = excluded.enabled,
                    minimum_interval_seconds = excluded.minimum_interval_seconds,
                    backoff_floor_seconds = excluded.backoff_floor_seconds,
                    max_batch_size = excluded.max_batch_size,
                    max_blob_bytes_per_run = excluded.max_blob_bytes_per_run,
                    respect_metered_networks = excluded.respect_metered_networks,
                    maintenance_window_json = excluded.maintenance_window_json,
                    paused_reason = excluded.paused_reason,
                    pending_local_changes = excluded.pending_local_changes,
                    updated_at = excluded.updated_at
                """,
                (
                    policy.dataset_id,
                    policy.device_id,
                    1 if policy.enabled else 0,
                    policy.minimum_interval_seconds,
                    policy.backoff_floor_seconds,
                    policy.max_batch_size,
                    policy.max_blob_bytes_per_run,
                    1 if policy.respect_metered_networks else 0,
                    encode_json(policy.maintenance_window, default=None),
                    policy.paused_reason,
                    1 if policy.pending_local_changes else 0,
                    now,
                ),
                connection=conn,
            )
            row = _first(
                self.execute(
                    """
                    SELECT * FROM sync_background_policies
                     WHERE dataset_id = ? AND device_id = ?
                    """,
                    (policy.dataset_id, policy.device_id),
                    connection=conn,
                )
            )
        return _background_policy_from_row(row)

    def get_background_lease(
        self,
        dataset_id: str,
        device_id: str,
    ) -> SyncBackgroundLease | None:
        """Return the current advisory background lease for one dataset/device."""

        row = _first(
            self.execute(
                """
                SELECT * FROM sync_background_leases
                 WHERE dataset_id = ? AND device_id = ?
                """,
                (dataset_id, device_id),
            )
        )
        if row is None:
            return None
        return _background_lease_from_row(row, status="acquired", acquired=True)

    def acquire_background_lease(
        self,
        lease: SyncBackgroundLeaseCreate,
    ) -> SyncBackgroundLease:
        """Create or refresh a short-lived advisory background sync lease."""

        if lease.ttl_seconds < 1:
            raise SyncStoreError("Sync background lease ttl_seconds must be greater than zero")
        now = lease.requested_at or utcnow_iso()
        expires_at = _add_seconds_iso(now, lease.ttl_seconds)
        with self.backend.transaction() as conn:
            self._require_device_for_dataset(
                lease.dataset_id,
                lease.device_id,
                connection=conn,
            )
            existing = _first(
                self.execute(
                    """
                    SELECT * FROM sync_background_leases
                     WHERE dataset_id = ? AND device_id = ?
                    """,
                    (lease.dataset_id, lease.device_id),
                    connection=conn,
                )
            )
            if existing is not None:
                existing_expires_at = _timestamp_to_string(existing.get("expires_at")) or ""
                active = _parse_iso_datetime(existing_expires_at) > _parse_iso_datetime(now)
                if active and existing["lease_id"] != lease.lease_id:
                    return _background_lease_from_row(
                        existing,
                        status="held_by_other",
                        acquired=False,
                    )
                status = "refreshed" if active else "acquired"
                self.execute(
                    """
                    UPDATE sync_background_leases
                       SET lease_id = ?, expires_at = ?, updated_at = ?
                     WHERE dataset_id = ? AND device_id = ?
                    """,
                    (
                        lease.lease_id,
                        expires_at,
                        now,
                        lease.dataset_id,
                        lease.device_id,
                    ),
                    connection=conn,
                )
            else:
                status = "acquired"
                self.execute(
                    """
                    INSERT INTO sync_background_leases (
                        dataset_id, device_id, lease_id, expires_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        lease.dataset_id,
                        lease.device_id,
                        lease.lease_id,
                        expires_at,
                        now,
                    ),
                    connection=conn,
                )
            row = _first(
                self.execute(
                    """
                    SELECT * FROM sync_background_leases
                     WHERE dataset_id = ? AND device_id = ?
                    """,
                    (lease.dataset_id, lease.device_id),
                    connection=conn,
                )
            )
        return _background_lease_from_row(row, status=status, acquired=True)

    def summarize_background_domains(
        self,
        dataset_id: str,
        device_id: str,
        *,
        domains: Sequence[SyncDomain] | None = None,
    ) -> list[SyncBackgroundDomainStatus]:
        """Return per-domain background sync status counters."""

        with self.backend.transaction() as conn:
            dataset_row = self._require_dataset(dataset_id, connection=conn)
            self._require_device_for_dataset(
                dataset_id,
                device_id,
                connection=conn,
            )
            selected_domains = [
                domain
                for domain in _dataset_domains_from_row(dataset_row)
                if domains is None or domain in domains
            ]
            statuses: list[SyncBackgroundDomainStatus] = []
            for domain in selected_domains:
                cursor_row = _first(
                    self.execute(
                        """
                        SELECT * FROM sync_device_cursors
                         WHERE dataset_id = ? AND device_id = ? AND domain = ?
                        """,
                        (dataset_id, device_id, domain),
                        connection=conn,
                    )
                )
                last_pulled = int(cursor_row["last_pulled_sequence"]) if cursor_row else 0
                last_pull_at = (
                    _timestamp_to_string(cursor_row.get("updated_at"))
                    if cursor_row
                    else None
                )
                envelope_row = _first(
                    self.execute(
                        """
                        SELECT MAX(server_sequence) AS last_server_sequence,
                               MAX(CASE WHEN device_id = ? THEN server_timestamp END)
                                   AS last_successful_push_at
                          FROM sync_envelopes
                         WHERE dataset_id = ?
                           AND domain = ?
                           AND status = 'accepted'
                        """,
                        (device_id, dataset_id, domain),
                        connection=conn,
                    )
                ) or {}
                lag_row = _first(
                    self.execute(
                        """
                        SELECT COUNT(*) AS lag_count
                          FROM sync_envelopes
                         WHERE dataset_id = ?
                           AND domain = ?
                           AND status = 'accepted'
                           AND server_sequence > ?
                        """,
                        (dataset_id, domain, last_pulled),
                        connection=conn,
                    )
                ) or {}
                conflict_row = _first(
                    self.execute(
                        """
                        SELECT COUNT(*) AS conflict_count
                          FROM sync_conflicts
                         WHERE dataset_id = ?
                           AND domain = ?
                           AND status = 'unresolved'
                        """,
                        (dataset_id, domain),
                        connection=conn,
                    )
                ) or {}
                failure_row = _first(
                    self.execute(
                        """
                        SELECT COUNT(*) AS failure_count
                          FROM sync_envelopes
                         WHERE dataset_id = ?
                           AND domain = ?
                           AND apply_status = 'failed'
                        """,
                        (dataset_id, domain),
                        connection=conn,
                    )
                ) or {}
                blob_completeness: dict[str, int] = {}
                if domain == "attachment.ref":
                    attachment_row = _first(
                        self.execute(
                            """
                            SELECT COUNT(*) AS required_blob_count
                              FROM sync_attachments
                             WHERE dataset_id = ?
                            """,
                            (dataset_id,),
                            connection=conn,
                        )
                    ) or {}
                    available_row = _first(
                        self.execute(
                            """
                            SELECT COUNT(*) AS available_blob_count
                              FROM sync_blob_objects
                             WHERE dataset_id = ?
                               AND status = 'available'
                               AND deleted_at IS NULL
                            """,
                            (dataset_id,),
                            connection=conn,
                        )
                    ) or {}
                    required = int(attachment_row.get("required_blob_count") or 0)
                    available = int(available_row.get("available_blob_count") or 0)
                    if required or available:
                        blob_completeness = {
                            "required_blob_count": required,
                            "available_blob_count": available,
                            "missing_blob_count": max(required - available, 0),
                        }
                statuses.append(
                    SyncBackgroundDomainStatus(
                        domain=domain,
                        last_server_sequence=int(envelope_row.get("last_server_sequence") or 0),
                        last_pulled_sequence=last_pulled,
                        cursor_lag_count=int(lag_row.get("lag_count") or 0),
                        unresolved_conflicts=int(conflict_row.get("conflict_count") or 0),
                        replayable_failures=int(failure_row.get("failure_count") or 0),
                        last_successful_push_at=_timestamp_to_string(
                            envelope_row.get("last_successful_push_at")
                        ),
                        last_successful_pull_at=last_pull_at,
                        blob_completeness=blob_completeness,
                    )
                )
        return statuses

    def get_or_create_default_personal_dataset(self, user_id: str) -> SyncDataset:
        """Return the user's default Chatbook personal dataset, creating it if needed."""

        if not user_id:
            raise SyncStoreError("user_id is required for default personal dataset lookup")
        for dataset in self.list_datasets_for_user(user_id):
            if (
                dataset.scope_type == "personal"
                and dataset.metadata.get("default_personal") is True
                and dataset.metadata.get("client_family") == "chatbook"
            ):
                return dataset

        dataset_id = f"ds_personal_{str(user_id).replace('/', '_').replace(':', '_')}"
        return self.enroll_dataset(
            SyncDatasetCreate(
                dataset_id=dataset_id,
                owner_user_id=user_id,
                scope_type="personal",
                encryption_policy=DEFAULT_M1_ENCRYPTION_POLICY,
                domains=list(M1_SYNC_DOMAINS),
                metadata={"default_personal": True, "client_family": "chatbook"},
            )
        )

    def _find_existing_envelope_for_idempotency(
        self,
        envelope: SyncEnvelopeCreate,
        *,
        connection: Any,
    ) -> dict[str, Any] | None:
        client_row = _first(
            self.execute(
                """
                SELECT * FROM sync_envelopes
                 WHERE dataset_id = ? AND client_envelope_id = ?
                """,
                (envelope.dataset_id, envelope.client_envelope_id),
                connection=connection,
            )
        )
        sequence_row: dict[str, Any] | None = None
        if envelope.device_id is not None and envelope.client_sequence is not None:
            sequence_row = _first(
                self.execute(
                    """
                    SELECT * FROM sync_envelopes
                     WHERE dataset_id = ? AND device_id = ? AND client_sequence = ?
                    """,
                    (envelope.dataset_id, envelope.device_id, envelope.client_sequence),
                    connection=connection,
                )
            )

        if (
            client_row is not None
            and sequence_row is not None
            and client_row["server_sequence"] != sequence_row["server_sequence"]
        ):
            raise SyncIdempotencyConflictError(
                "Sync envelope idempotency keys refer to different envelopes"
            )

        if client_row is not None:
            if (
                _envelope_fingerprint_from_row(client_row)
                != _envelope_fingerprint_from_create(envelope)
            ):
                raise SyncIdempotencyConflictError(
                    "Sync envelope idempotency key was reused with different content"
                )
            return client_row

        if sequence_row is not None:
            if (
                _envelope_fingerprint_from_row(
                    sequence_row,
                    ignore_client_envelope_id=True,
                )
                != _envelope_sequence_fingerprint_from_create(envelope)
            ):
                raise SyncIdempotencyConflictError(
                    "Sync envelope client sequence was reused with different content"
                )
            return sequence_row

        return None

    def get_existing_envelope_for_idempotency(
        self,
        envelope: SyncEnvelopeCreate,
    ) -> SyncEnvelope | None:
        self._validate_envelope_contract(envelope)
        with self.backend.transaction() as conn:
            self._require_dataset_domain(
                envelope.dataset_id,
                envelope.domain,
                connection=conn,
            )
            existing = self._find_existing_envelope_for_idempotency(
                envelope,
                connection=conn,
            )
        if existing is None:
            return None
        return _envelope_from_row(existing)

    def insert_envelope(self, envelope: SyncEnvelopeCreate) -> SyncEnvelope:
        self._validate_envelope_contract(envelope)
        with self.backend.transaction() as conn:
            self._require_dataset_domain(
                envelope.dataset_id,
                envelope.domain,
                connection=conn,
            )

            existing = self._find_existing_envelope_for_idempotency(
                envelope,
                connection=conn,
            )
            if existing is not None:
                return _envelope_from_row(existing)

            now = utcnow_iso()
            self.execute(
                """
                INSERT INTO sync_envelopes (
                    dataset_id, domain, entity_id, stable_key, operation,
                    client_envelope_id, device_id, client_profile_id, client_sequence,
                    client_timestamp, server_timestamp, base_server_cursor,
                    base_object_revision, base_object_hash, object_revision, parent_id,
                    schema_version, base_version, entity_version, dependency_json,
                    routing_metadata_json, payload_ciphertext, payload_json,
                    payload_clear_json, payload_hash, payload_size_bytes,
                    created_at_client, received_at_server, deleted,
                    encryption_metadata_json, adapter_version, status, apply_status,
                    apply_error_code, apply_error_message, applied_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    envelope.dataset_id,
                    envelope.domain,
                    envelope.object_id,
                    envelope.stable_key,
                    envelope.operation,
                    envelope.client_envelope_id,
                    envelope.device_id,
                    envelope.client_profile_id,
                    envelope.client_sequence,
                    envelope.client_timestamp,
                    now,
                    envelope.base_server_cursor,
                    envelope.base_object_revision,
                    envelope.base_object_hash,
                    envelope.object_revision,
                    envelope.parent_id,
                    envelope.schema_version,
                    _version_to_storage(envelope.base_version),
                    _version_to_storage(envelope.entity_version),
                    encode_json(envelope.dependencies, default=[]),
                    encode_json(envelope.routing_metadata, default={}),
                    envelope.payload_ciphertext,
                    encode_json(envelope.payload, default={}),
                    encode_json(envelope.payload_clear, default={}),
                    envelope.payload_hash,
                    envelope.payload_size_bytes,
                    envelope.created_at_client,
                    now,
                    1 if envelope.deleted else 0,
                    encode_json(envelope.encryption_metadata, default={}),
                    envelope.adapter_version,
                    envelope.status,
                    envelope.apply_status,
                    envelope.apply_error_code,
                    envelope.apply_error_message,
                    envelope.applied_at,
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
                if (
                    row is None
                    or row.get("dataset_id") != envelope.dataset_id
                    or row.get("client_envelope_id") != envelope.client_envelope_id
                ):
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
            if row is None:
                raise SyncStoreError(
                    "Sync envelope insert did not produce a retrievable record"
                )
            if (
                _envelope_fingerprint_from_row(row)
                != _envelope_fingerprint_from_create(envelope)
            ):
                raise SyncIdempotencyConflictError(
                    "Sync envelope idempotency key was reused with different content"
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
        status: str | Sequence[str] | None = None,
        exclude_device_id: str | None = None,
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
            sql += _domain_filter_sql(domains, params)
        if status is not None:
            statuses = [status] if isinstance(status, str) else list(status)
            if not statuses:
                return []
            if len(statuses) == 1:
                sql += " AND status = ?"
                params.append(statuses[0])
            else:
                placeholders = ", ".join("?" for _ in statuses)
                sql += f" AND status IN ({placeholders})"
                params.extend(statuses)
        if exclude_device_id is not None:
            sql += " AND (device_id IS NULL OR device_id <> ?)"
            params.append(exclude_device_id)
        sql += " ORDER BY server_sequence ASC LIMIT ?"
        params.append(limit)
        result = self.execute(sql, tuple(params))
        return [_envelope_from_row(row) for row in result.rows]

    def list_envelopes_for_entity(
        self,
        dataset_id: str,
        domain: SyncDomain,
        *,
        entity_id: str | None = None,
        stable_key: str | None = None,
        limit: int = 100,
    ) -> list[SyncEnvelope]:
        """List accepted envelopes for one entity identity or stable key."""

        if limit < 1 or (entity_id is None and stable_key is None):
            return []
        params: list[Any] = [dataset_id, domain]
        if entity_id is not None and stable_key is not None:
            sql = """
                SELECT * FROM sync_envelopes
                 WHERE dataset_id = ?
                   AND domain = ?
                   AND status = 'accepted'
                   AND (entity_id = ? OR stable_key = ?)
                 ORDER BY server_sequence DESC
                 LIMIT ?
            """
            params.extend([entity_id, stable_key])
        elif entity_id is not None:
            sql = """
                SELECT * FROM sync_envelopes
                 WHERE dataset_id = ?
                   AND domain = ?
                   AND status = 'accepted'
                   AND entity_id = ?
                 ORDER BY server_sequence DESC
                 LIMIT ?
            """
            params.append(entity_id)
        else:
            sql = """
                SELECT * FROM sync_envelopes
                 WHERE dataset_id = ?
                   AND domain = ?
                   AND status = 'accepted'
                   AND stable_key = ?
                 ORDER BY server_sequence DESC
                 LIMIT ?
            """
            params.append(stable_key)
        params.append(limit)
        result = self.execute(sql, tuple(params))
        return [_envelope_from_row(row) for row in result.rows]

    def get_object_state(
        self,
        dataset_id: str,
        domain: SyncDomain,
        object_id: str,
    ) -> SyncObjectState | None:
        row = _first(
            self.execute(
                """
                SELECT * FROM sync_object_state
                 WHERE dataset_id = ? AND domain = ? AND object_id = ?
                """,
                (dataset_id, domain, object_id),
            )
        )
        if row is None:
            return None
        return _object_state_from_row(row)

    def upsert_object_state(self, state: SyncObjectState) -> SyncObjectState:
        now = utcnow_iso()
        with self.backend.transaction() as conn:
            self._require_dataset_domain(state.dataset_id, state.domain, connection=conn)
            self.execute(
                """
                INSERT INTO sync_object_state (
                    dataset_id, domain, object_id, object_revision, object_hash,
                    latest_server_cursor, deleted, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (dataset_id, domain, object_id)
                DO UPDATE SET
                    object_revision = excluded.object_revision,
                    object_hash = excluded.object_hash,
                    latest_server_cursor = excluded.latest_server_cursor,
                    deleted = excluded.deleted,
                    updated_at = excluded.updated_at
                """,
                (
                    state.dataset_id,
                    state.domain,
                    state.object_id,
                    state.object_revision,
                    state.object_hash,
                    state.latest_server_cursor,
                    1 if state.deleted else 0,
                    now,
                ),
                connection=conn,
            )
            row = _first(
                self.execute(
                    """
                    SELECT * FROM sync_object_state
                     WHERE dataset_id = ? AND domain = ? AND object_id = ?
                    """,
                    (state.dataset_id, state.domain, state.object_id),
                    connection=conn,
                )
            )
        return _object_state_from_row(row)

    def mark_envelope_apply_status(
        self,
        server_cursor: int,
        *,
        apply_status: SyncApplyStatus,
        apply_error_code: str | None = None,
        apply_error_message: str | None = None,
    ) -> SyncEnvelope:
        if apply_status not in SYNC_APPLY_STATUSES:
            raise SyncStoreError(f"Invalid Sync envelope apply status: {apply_status}")
        now = utcnow_iso()
        applied_at = now if apply_status == "applied" else None
        with self.backend.transaction() as conn:
            self.execute(
                """
                UPDATE sync_envelopes
                   SET apply_status = ?,
                       apply_error_code = ?,
                       apply_error_message = ?,
                       applied_at = ?
                 WHERE server_sequence = ?
                """,
                (
                    apply_status,
                    apply_error_code,
                    apply_error_message,
                    applied_at,
                    server_cursor,
                ),
                connection=conn,
            )
            row = _first(
                self.execute(
                    "SELECT * FROM sync_envelopes WHERE server_sequence = ?",
                    (server_cursor,),
                    connection=conn,
                )
            )
        if row is None:
            raise SyncStoreError(f"Sync envelope not found for server cursor: {server_cursor}")
        return _envelope_from_row(row)

    def list_failed_applies(
        self,
        dataset_id: str,
        *,
        limit: int = 100,
    ) -> list[SyncEnvelope]:
        if limit < 1:
            return []
        result = self.execute(
            """
            SELECT * FROM sync_envelopes
             WHERE dataset_id = ? AND apply_status = 'failed'
             ORDER BY server_sequence ASC
             LIMIT ?
            """,
            (dataset_id, limit),
        )
        return [_envelope_from_row(row) for row in result.rows]

    def list_accepted_envelopes_for_replay(
        self,
        dataset_id: str,
        *,
        since_cursor: int = 0,
        limit: int = 1000,
    ) -> list[SyncEnvelope]:
        if limit < 1:
            return []
        result = self.execute(
            """
            SELECT * FROM sync_envelopes
             WHERE dataset_id = ?
               AND server_sequence > ?
               AND status = 'accepted'
             ORDER BY server_sequence ASC
             LIMIT ?
            """,
            (dataset_id, since_cursor, limit),
        )
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

    def get_conflict(self, conflict_id: str) -> SyncConflict | None:
        """Return a conflict by ID without scanning dataset conflict lists."""

        row = _first(
            self.execute(
                "SELECT * FROM sync_conflicts WHERE conflict_id = ?",
                (conflict_id,),
            )
        )
        if row is None:
            return None
        return _conflict_from_row(row)

    def get_unresolved_conflict_for_envelope(
        self,
        dataset_id: str,
        *,
        local_envelope_id: str,
        server_sequence: int | None = None,
    ) -> SyncConflict | None:
        """Return an unresolved conflict already recorded for a local envelope."""

        params: list[Any] = [dataset_id, local_envelope_id]
        sql = """
            SELECT * FROM sync_conflicts
             WHERE dataset_id = ?
               AND local_envelope_id = ?
               AND status = 'unresolved'
        """
        if server_sequence is not None:
            sql += " AND server_sequence = ?"
            params.append(server_sequence)
        sql += " ORDER BY created_at ASC, conflict_id ASC LIMIT 1"
        row = _first(self.execute(sql, tuple(params)))
        if row is None:
            return None
        return _conflict_from_row(row)

    def claim_conflict_resolution(
        self,
        conflict_id: str,
        *,
        dataset_id: str | None = None,
        resolved_by_device_id: str | None = None,
        resolution_action: str | None = None,
        resolution_notes: str | None = None,
    ) -> SyncConflict:
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
            if dataset_id is not None and existing["dataset_id"] != dataset_id:
                raise SyncConflictNotFoundError(
                    "Sync conflict was not found or is not accessible"
                )
            if existing["status"] != "unresolved":
                raise SyncStoreError("Sync conflict is already resolved")
            if _conflict_row_has_resolution_claim(existing):
                raise SyncStoreError("Sync conflict resolution is already claimed")

            result = self.execute(
                """
                UPDATE sync_conflicts
                   SET resolution_action = ?,
                       resolved_by_device_id = ?,
                       resolution_notes = ?
                 WHERE conflict_id = ?
                   AND status = 'unresolved'
                   AND resolved_at IS NULL
                   AND resolved_by_envelope_id IS NULL
                   AND resolution_action IS NULL
                   AND resolved_by_device_id IS NULL
                   AND resolution_notes IS NULL
                """,
                (
                    resolution_action,
                    resolved_by_device_id,
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
            if row is None:
                raise SyncConflictNotFoundError(f"Sync conflict not found: {conflict_id}")
            if result.rowcount == 0:
                if row["status"] != "unresolved":
                    raise SyncStoreError("Sync conflict is already resolved")
                raise SyncStoreError("Sync conflict resolution is already claimed")
        return _conflict_from_row(row)

    def release_conflict_resolution_claim(
        self,
        conflict_id: str,
        *,
        dataset_id: str | None = None,
        resolved_by_device_id: str | None = None,
        resolution_action: str | None = None,
        resolution_notes: str | None = None,
    ) -> SyncConflict:
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
            if dataset_id is not None and existing["dataset_id"] != dataset_id:
                raise SyncConflictNotFoundError(
                    "Sync conflict was not found or is not accessible"
                )
            result = self.execute(
                """
                UPDATE sync_conflicts
                   SET resolution_action = NULL,
                       resolved_by_device_id = NULL,
                       resolution_notes = NULL
                 WHERE conflict_id = ?
                   AND status = 'unresolved'
                   AND resolved_at IS NULL
                   AND resolved_by_envelope_id IS NULL
                   AND resolution_action = ?
                   AND (
                        resolved_by_device_id = ?
                        OR (resolved_by_device_id IS NULL AND ? IS NULL)
                   )
                   AND (
                        resolution_notes = ?
                        OR (resolution_notes IS NULL AND ? IS NULL)
                   )
                """,
                (
                    conflict_id,
                    resolution_action,
                    resolved_by_device_id,
                    resolved_by_device_id,
                    resolution_notes,
                    resolution_notes,
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
            if row is None:
                raise SyncConflictNotFoundError(f"Sync conflict not found: {conflict_id}")
            if result.rowcount == 0 and _conflict_row_matches_resolution_claim(
                row,
                resolved_by_device_id=resolved_by_device_id,
                resolution_action=resolution_action,
                resolution_notes=resolution_notes,
            ):
                raise SyncStoreError("Sync conflict resolution claim could not be released")
        return _conflict_from_row(row)

    def resolve_conflict(
        self,
        conflict_id: str,
        *,
        dataset_id: str | None = None,
        server_cursor: int | None = None,
        status: ConflictStatus = "resolved",
        resolved_by_envelope_id: str | None = None,
        resolved_by_device_id: str | None = None,
        resolution_action: str | None = None,
        resolution_notes: str | None = None,
    ) -> SyncConflict:
        def _matches_resolution(row: dict[str, Any]) -> bool:
            if row["status"] != status:
                return False
            if server_cursor is not None:
                stored_cursor = row.get("server_sequence")
                if stored_cursor is None or int(stored_cursor) != int(server_cursor):
                    return False
            return (
                row.get("resolved_by_envelope_id") == resolved_by_envelope_id
                and row.get("resolved_by_device_id") == resolved_by_device_id
                and row.get("resolution_action") == resolution_action
                and row.get("resolution_notes") == resolution_notes
            )

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
            if dataset_id is not None and existing["dataset_id"] != dataset_id:
                raise SyncConflictNotFoundError(
                    "Sync conflict was not found or is not accessible"
                )
            if existing["status"] != "unresolved":
                if _matches_resolution(existing):
                    return _conflict_from_row(existing)
                raise SyncStoreError("Sync conflict is already resolved")
            if _conflict_row_has_resolution_claim(
                existing
            ) and not _conflict_row_matches_resolution_claim(
                existing,
                resolved_by_device_id=resolved_by_device_id,
                resolution_action=resolution_action,
                resolution_notes=resolution_notes,
            ):
                raise SyncStoreError("Sync conflict resolution is already claimed")
            result = self.execute(
                """
                UPDATE sync_conflicts
                   SET status = ?,
                       server_sequence = COALESCE(?, server_sequence),
                       resolved_at = ?,
                       resolved_by_envelope_id = ?,
                       resolved_by_device_id = ?,
                       resolution_action = ?,
                       resolution_notes = ?
                 WHERE conflict_id = ?
                   AND status = 'unresolved'
                   AND (
                        (
                            resolved_at IS NULL
                            AND resolved_by_envelope_id IS NULL
                            AND resolution_action IS NULL
                            AND resolved_by_device_id IS NULL
                            AND resolution_notes IS NULL
                        )
                        OR (
                            resolved_at IS NULL
                            AND resolved_by_envelope_id IS NULL
                            AND resolution_action = ?
                            AND (
                                resolved_by_device_id = ?
                                OR (resolved_by_device_id IS NULL AND ? IS NULL)
                            )
                            AND (
                                resolution_notes = ?
                                OR (resolution_notes IS NULL AND ? IS NULL)
                            )
                        )
                   )
                """,
                (
                    status,
                    server_cursor,
                    now,
                    resolved_by_envelope_id,
                    resolved_by_device_id,
                    resolution_action,
                    resolution_notes,
                    conflict_id,
                    resolution_action,
                    resolved_by_device_id,
                    resolved_by_device_id,
                    resolution_notes,
                    resolution_notes,
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
            if row is None:
                raise SyncConflictNotFoundError(f"Sync conflict not found: {conflict_id}")
            if result.rowcount == 0:
                if _matches_resolution(row):
                    return _conflict_from_row(row)
                if _conflict_row_has_resolution_claim(row):
                    raise SyncStoreError("Sync conflict resolution is already claimed")
                raise SyncStoreError("Sync conflict is already resolved")
        return _conflict_from_row(row)

    def store_key_record(self, record: SyncKeyRecordCreate) -> SyncKeyRecord:
        now = utcnow_iso()
        with self.backend.transaction() as conn:
            self._require_dataset(record.dataset_id, connection=conn)
            self.execute(
                """
                INSERT INTO sync_key_records (
                    key_record_id, dataset_id, user_id, device_id, key_purpose,
                    wrapped_key_blob, kdf_metadata_json, recovery_hint,
                    rotation_of_key_record_id, rotation_source_key_record_ids_json,
                    encryption_policy, key_epoch, active_from_server_sequence,
                    superseded_at, wrapped_for, rewrap_status, created_at, revoked_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (key_record_id) DO NOTHING
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
                    encode_json(record.rotation_source_key_record_ids, default=[]),
                    record.encryption_policy,
                    record.key_epoch,
                    record.active_from_server_sequence,
                    record.superseded_at,
                    record.wrapped_for,
                    record.rewrap_status,
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
            if row is None:
                raise SyncStoreError(
                    "Sync key record insert did not produce a retrievable record"
                )
            if (
                _key_record_fingerprint_from_row(row)
                != _key_record_fingerprint_from_create(record)
            ):
                raise SyncIdempotencyConflictError(
                    "Sync key record ID was reused with different key material"
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

    def _dataset_envelope_range(
        self,
        dataset_id: str,
        *,
        connection: Any,
        through_server_sequence: int | None = None,
    ) -> SyncKeyRotationEnvelopeRange:
        params: list[Any] = [dataset_id]
        if through_server_sequence is not None:
            params.append(through_server_sequence)
            sql = """
                SELECT MIN(server_sequence) AS from_server_sequence,
                       MAX(server_sequence) AS through_server_sequence,
                       COUNT(*) AS envelope_count
                  FROM sync_envelopes
                 WHERE dataset_id = ?
                   AND status = 'accepted'
                   AND server_sequence <= ?
            """
        else:
            sql = """
                SELECT MIN(server_sequence) AS from_server_sequence,
                       MAX(server_sequence) AS through_server_sequence,
                       COUNT(*) AS envelope_count
                  FROM sync_envelopes
                 WHERE dataset_id = ?
                   AND status = 'accepted'
            """
        row = _first(
            self.execute(
                sql,
                tuple(params),
                connection=connection,
            )
        ) or {}
        return SyncKeyRotationEnvelopeRange(
            from_server_sequence=_optional_int_from_storage(
                row.get("from_server_sequence")
            ),
            through_server_sequence=_optional_int_from_storage(
                row.get("through_server_sequence")
            ),
            envelope_count=int(row.get("envelope_count") or 0),
        )

    def get_dataset_envelope_range(self, dataset_id: str) -> SyncKeyRotationEnvelopeRange:
        with self.backend.transaction() as conn:
            self._require_dataset(dataset_id, connection=conn)
            return self._dataset_envelope_range(dataset_id, connection=conn)

    def commit_key_rotation(
        self,
        record: SyncKeyRecordCreate,
        *,
        source_key_record_ids: Sequence[str],
        superseded_at: str,
    ) -> tuple[SyncKeyRecord, list[SyncKeyRecord], SyncKeyRotationEnvelopeRange]:
        requested_source_ids = _canonical_key_rotation_source_ids(source_key_record_ids)

        def source_rows_for_ids(source_ids: Sequence[str], *, connection: Any) -> list[dict[str, Any]]:
            rows: list[dict[str, Any]] = []
            for source_id in source_ids:
                row = _first(
                    self.execute(
                        """
                        SELECT * FROM sync_key_records
                         WHERE dataset_id = ?
                           AND user_id = ?
                           AND key_purpose = ?
                           AND key_record_id = ?
                        """,
                        (
                            record.dataset_id,
                            record.user_id,
                            record.key_purpose,
                            source_id,
                        ),
                        connection=connection,
                    )
                )
                if row is not None:
                    rows.append(row)
            return rows

        with self.backend.transaction() as conn:
            self._require_dataset(record.dataset_id, connection=conn)
            if self.backend_type == BackendType.POSTGRESQL:
                self.execute(
                    "LOCK TABLE sync_envelopes, sync_key_records IN SHARE ROW EXCLUSIVE MODE",
                    connection=conn,
                )

            existing = _first(
                self.execute(
                    "SELECT * FROM sync_key_records WHERE key_record_id = ?",
                    (record.key_record_id,),
                    connection=conn,
                )
            )
            if existing is not None:
                existing_record = _key_record_from_row(existing)
                if not _key_rotation_record_matches_request(existing_record, record):
                    raise SyncIdempotencyConflictError(
                        "Sync key rotation ID was reused with different key material"
                    )
                manifest_source_ids = existing_record.rotation_source_key_record_ids
                if (
                    not manifest_source_ids
                    and existing_record.rotation_of_key_record_id is not None
                ):
                    manifest_source_ids = (existing_record.rotation_of_key_record_id,)
                if not manifest_source_ids:
                    raise SyncStoreError("Sync key rotation is invalid")
                if requested_source_ids and requested_source_ids != manifest_source_ids:
                    requested_rows = source_rows_for_ids(requested_source_ids, connection=conn)
                    if len(requested_rows) != len(requested_source_ids):
                        raise SyncStoreError("Sync key rotation is invalid")
                    raise SyncIdempotencyConflictError(
                        "Sync key rotation source set changed for an existing rotation"
                    )
                source_rows = source_rows_for_ids(manifest_source_ids, connection=conn)
                if len(source_rows) != len(manifest_source_ids):
                    raise SyncStoreError("Sync key rotation is invalid")
                retained_range = self._dataset_envelope_range(
                    record.dataset_id,
                    connection=conn,
                    through_server_sequence=(existing_record.active_from_server_sequence or 1) - 1,
                )
                return (
                    existing_record,
                    [_key_record_from_row(row) for row in source_rows],
                    retained_range,
                )

            if requested_source_ids:
                source_ids = requested_source_ids
                source_rows = source_rows_for_ids(source_ids, connection=conn)
            else:
                source_rows = list(
                    self.execute(
                        """
                        SELECT * FROM sync_key_records
                         WHERE dataset_id = ?
                           AND user_id = ?
                           AND key_purpose = ?
                           AND revoked_at IS NULL
                           AND superseded_at IS NULL
                         ORDER BY key_record_id ASC
                        """,
                        (
                            record.dataset_id,
                            record.user_id,
                            record.key_purpose,
                        ),
                        connection=conn,
                    ).rows
                )
                source_ids = tuple(row["key_record_id"] for row in source_rows)
            if not source_ids or len(source_rows) != len(source_ids):
                raise SyncStoreError("Sync key rotation is invalid")
            if any(
                row.get("revoked_at") is not None or row.get("superseded_at") is not None
                for row in source_rows
            ):
                raise SyncStoreError("Sync key rotation is invalid")

            retained_range = self._dataset_envelope_range(record.dataset_id, connection=conn)
            active_from = (retained_range.through_server_sequence or 0) + 1
            epoch_row = _first(
                self.execute(
                    """
                    SELECT MAX(key_epoch) AS highest_epoch
                      FROM sync_key_records
                     WHERE dataset_id = ?
                       AND user_id = ?
                       AND key_purpose = ?
                    """,
                    (record.dataset_id, record.user_id, record.key_purpose),
                    connection=conn,
                )
            ) or {}
            record_to_store = replace(
                record,
                rotation_of_key_record_id=source_ids[0],
                rotation_source_key_record_ids=tuple(source_ids),
                key_epoch=max(int(epoch_row.get("highest_epoch") or 0) + 1, 1),
                active_from_server_sequence=active_from,
            )
            self.execute(
                """
                INSERT INTO sync_key_records (
                    key_record_id, dataset_id, user_id, device_id, key_purpose,
                    wrapped_key_blob, kdf_metadata_json, recovery_hint,
                    rotation_of_key_record_id, rotation_source_key_record_ids_json,
                    encryption_policy, key_epoch, active_from_server_sequence,
                    superseded_at, wrapped_for, rewrap_status, created_at, revoked_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record_to_store.key_record_id,
                    record_to_store.dataset_id,
                    record_to_store.user_id,
                    record_to_store.device_id,
                    record_to_store.key_purpose,
                    record_to_store.wrapped_key_blob,
                    encode_json(record_to_store.kdf_metadata, default={}),
                    record_to_store.recovery_hint,
                    record_to_store.rotation_of_key_record_id,
                    encode_json(record_to_store.rotation_source_key_record_ids, default=[]),
                    record_to_store.encryption_policy,
                    record_to_store.key_epoch,
                    record_to_store.active_from_server_sequence,
                    record_to_store.superseded_at,
                    record_to_store.wrapped_for,
                    record_to_store.rewrap_status,
                    utcnow_iso(),
                    record_to_store.revoked_at,
                ),
                connection=conn,
            )
            for source_id in source_ids:
                self.execute(
                    """
                UPDATE sync_key_records
                   SET superseded_at = ?
                 WHERE dataset_id = ?
                   AND user_id = ?
                   AND key_record_id = ?
                """,
                    (superseded_at, record.dataset_id, record.user_id, source_id),
                    connection=conn,
                )
            existing = _first(
                self.execute(
                    "SELECT * FROM sync_key_records WHERE key_record_id = ?",
                    (record.key_record_id,),
                    connection=conn,
                )
            )
            if existing is None:
                raise SyncStoreError(
                    "Sync key rotation insert did not produce a retrievable record"
                )
            source_rows = source_rows_for_ids(source_ids, connection=conn)
            if len(source_rows) != len(source_ids):
                raise SyncStoreError("Sync key rotation is invalid")

        return (
            _key_record_from_row(existing),
            [_key_record_from_row(row) for row in source_rows],
            retained_range,
        )

    def store_attachment(self, attachment: SyncAttachmentCreate) -> SyncAttachment:
        """Store or idempotently deduplicate an encrypted Sync v2 attachment."""

        now = utcnow_iso()
        with self.backend.transaction() as conn:
            self._require_dataset_domain(
                attachment.dataset_id,
                attachment.domain,
                connection=conn,
            )
            insert_result = self.execute(
                """
                INSERT INTO sync_attachments (
                    attachment_id, dataset_id, domain, entity_id, content_type,
                    size_bytes, payload_ciphertext, payload_hash, encryption_policy,
                    metadata_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (dataset_id, attachment_id) DO NOTHING
                """,
                (
                    attachment.attachment_id,
                    attachment.dataset_id,
                    attachment.domain,
                    attachment.entity_id,
                    attachment.content_type,
                    attachment.size_bytes,
                    attachment.payload_ciphertext,
                    attachment.payload_hash,
                    attachment.encryption_policy,
                    encode_json(attachment.metadata, default={}),
                    now,
                ),
                connection=conn,
            )
            inserted = insert_result.rowcount > 0
            row = _first(
                self.execute(
                    """
                    SELECT * FROM sync_attachments
                     WHERE dataset_id = ? AND attachment_id = ?
                    """,
                    (attachment.dataset_id, attachment.attachment_id),
                    connection=conn,
                )
            )
            if row is None:
                raise SyncStoreError(
                    "Sync attachment insert did not produce a retrievable record"
                )
            if (
                _attachment_fingerprint_from_row(row)
                != _attachment_fingerprint_from_create(attachment)
            ):
                raise SyncIdempotencyConflictError(
                    "Sync attachment ID was reused with different content"
                )
        return _attachment_from_row(row, stored=inserted)

    def create_blob_upload_session(
        self,
        session: SyncBlobUploadSessionCreate,
    ) -> SyncBlobUploadSession:
        """Create or idempotently return a resumable blob upload session."""

        now = utcnow_iso()
        with self.backend.transaction() as conn:
            dataset_row = self._require_dataset_domain(
                session.dataset_id,
                session.domain,
                connection=conn,
            )
            if (
                str(dataset_row["scope_type"]) != "workspace"
                and str(dataset_row["owner_user_id"]) != str(session.owner_user_id)
            ):
                raise SyncDatasetNotFoundError(
                    f"Sync dataset not found: {session.dataset_id}"
                )
            if session.idempotency_key:
                existing = _first(
                    self.execute(
                        """
                        SELECT * FROM sync_blob_upload_sessions
                         WHERE dataset_id = ? AND device_id = ? AND idempotency_key = ?
                        """,
                        (
                            session.dataset_id,
                            session.device_id,
                            session.idempotency_key,
                        ),
                        connection=conn,
                    )
                )
                if existing is not None:
                    if _blob_session_fingerprint_from_row(
                        existing
                    ) != _blob_session_fingerprint_from_create(session):
                        raise SyncIdempotencyConflictError(
                            "Sync blob upload idempotency key was reused with different content"
                        )
                    return self._blob_upload_session_with_chunks(
                        existing["upload_id"],
                        connection=conn,
                    )
            self.execute(
                """
                INSERT INTO sync_blob_upload_sessions (
                    upload_id, dataset_id, owner_user_id, device_id, attachment_id,
                    domain, entity_id, content_type, size_bytes, payload_hash,
                    chunk_size, chunk_count, reserved_quota_bytes, status,
                    idempotency_key, expires_at, metadata_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session.upload_id,
                    session.dataset_id,
                    session.owner_user_id,
                    session.device_id,
                    session.attachment_id,
                    session.domain,
                    session.object_id,
                    session.content_type,
                    session.size_bytes,
                    session.payload_hash,
                    session.chunk_size,
                    session.chunk_count,
                    session.reserved_quota_bytes,
                    session.status,
                    session.idempotency_key,
                    session.expires_at,
                    encode_json(session.metadata, default={}),
                    now,
                    now,
                ),
                connection=conn,
            )
            return self._blob_upload_session_with_chunks(
                session.upload_id,
                connection=conn,
            )

    def get_blob_upload_session(
        self,
        upload_id: str,
        *,
        dataset_id: str | None = None,
    ) -> SyncBlobUploadSession | None:
        """Return one blob upload session with uploaded/missing chunk detail."""

        if dataset_id is None:
            row = _first(
                self.execute(
                    "SELECT * FROM sync_blob_upload_sessions WHERE upload_id = ?",
                    (upload_id,),
                )
            )
        else:
            row = _first(
                self.execute(
                    """
                    SELECT * FROM sync_blob_upload_sessions
                     WHERE upload_id = ? AND dataset_id = ?
                    """,
                    (upload_id, dataset_id),
                )
            )
        if row is None:
            return None
        return self._blob_upload_session_with_chunks(row["upload_id"])

    def cancel_blob_upload_session(
        self,
        upload_id: str,
        *,
        dataset_id: str | None = None,
    ) -> SyncBlobUploadSession:
        """Cancel an incomplete blob upload session and release reserved quota."""

        now = utcnow_iso()
        with self.backend.transaction() as conn:
            row = self._require_blob_upload_session(
                upload_id,
                dataset_id=dataset_id,
                connection=conn,
            )
            if row["status"] not in {"complete", "cancelled", "expired"}:
                self.execute(
                    """
                    UPDATE sync_blob_upload_sessions
                       SET status = ?, updated_at = ?
                     WHERE upload_id = ?
                    """,
                    ("cancelled", now, upload_id),
                    connection=conn,
                )
            return self._blob_upload_session_with_chunks(
                upload_id,
                connection=conn,
            )

    def record_blob_chunk(self, chunk: SyncBlobChunkCreate) -> SyncBlobChunk:
        """Record one uploaded blob chunk idempotently."""

        now = utcnow_iso()
        with self.backend.transaction() as conn:
            session = self._require_blob_upload_session(
                chunk.upload_id,
                dataset_id=chunk.dataset_id,
                connection=conn,
            )
            if session["status"] not in {"created", "uploading"}:
                raise SyncStoreError("Sync blob upload session is not accepting chunks")
            if chunk.chunk_index < 0 or chunk.chunk_index >= int(session["chunk_count"]):
                raise SyncStoreError("Sync blob chunk index is outside the upload session")
            expected_offset = int(session["chunk_size"]) * chunk.chunk_index
            if chunk.offset_bytes != expected_offset:
                raise SyncStoreError("Sync blob chunk offset does not match the upload session")
            existing = _first(
                self.execute(
                    """
                    SELECT * FROM sync_blob_chunks
                     WHERE upload_id = ? AND chunk_index = ?
                    """,
                    (chunk.upload_id, chunk.chunk_index),
                    connection=conn,
                )
            )
            if existing is not None:
                if _blob_chunk_fingerprint_from_row(existing) != _blob_chunk_fingerprint_from_create(chunk):
                    raise SyncIdempotencyConflictError(
                        "Sync blob chunk was reused with different content"
                    )
                return _blob_chunk_from_row(existing)
            self.execute(
                """
                INSERT INTO sync_blob_chunks (
                    upload_id, dataset_id, chunk_index, offset_bytes, size_bytes,
                    chunk_hash, storage_key, received_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk.upload_id,
                    chunk.dataset_id,
                    chunk.chunk_index,
                    chunk.offset_bytes,
                    chunk.size_bytes,
                    chunk.chunk_hash,
                    chunk.storage_key,
                    now,
                ),
                connection=conn,
            )
            self.execute(
                """
                UPDATE sync_blob_upload_sessions
                   SET status = CASE WHEN status = 'created' THEN 'uploading' ELSE status END,
                       updated_at = ?
                 WHERE upload_id = ?
                """,
                (now, chunk.upload_id),
                connection=conn,
            )
            row = _first(
                self.execute(
                    """
                    SELECT * FROM sync_blob_chunks
                     WHERE upload_id = ? AND chunk_index = ?
                    """,
                    (chunk.upload_id, chunk.chunk_index),
                    connection=conn,
                )
            )
            if row is None:
                raise SyncStoreError("Sync blob chunk insert did not produce a retrievable record")
            return _blob_chunk_from_row(row)

    def complete_blob_upload(self, blob: SyncBlobObjectCreate) -> SyncBlobObject:
        """Commit a verified blob and deduplicate by dataset plus payload hash."""

        now = utcnow_iso()
        with self.backend.transaction() as conn:
            dataset_row = self._require_dataset(blob.dataset_id, connection=conn)
            if (
                str(dataset_row["scope_type"]) != "workspace"
                and str(dataset_row["owner_user_id"]) != str(blob.owner_user_id)
            ):
                raise SyncDatasetNotFoundError(f"Sync dataset not found: {blob.dataset_id}")
            session = self._find_active_blob_session_for_blob(blob, connection=conn)
            if session is not None:
                uploaded_chunks = self._blob_chunk_indexes(
                    session["upload_id"],
                    connection=conn,
                )
                expected = list(range(int(session["chunk_count"])))
                if uploaded_chunks != expected:
                    raise SyncStoreError("Sync blob upload session is missing chunks")
            self.execute(
                """
                INSERT INTO sync_blob_objects (
                    blob_id, dataset_id, owner_user_id, attachment_id, payload_hash,
                    content_type, size_bytes, encryption_policy, storage_backend,
                    storage_key, status, ref_count, metadata_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (dataset_id, payload_hash) DO NOTHING
                """,
                (
                    blob.blob_id,
                    blob.dataset_id,
                    blob.owner_user_id,
                    blob.attachment_id,
                    blob.payload_hash,
                    blob.content_type,
                    blob.size_bytes,
                    blob.encryption_policy,
                    blob.storage_backend,
                    blob.storage_key,
                    blob.status,
                    1,
                    encode_json(blob.metadata, default={}),
                    now,
                    now,
                ),
                connection=conn,
            )
            row = _first(
                self.execute(
                    """
                    SELECT * FROM sync_blob_objects
                     WHERE dataset_id = ? AND payload_hash = ?
                    """,
                    (blob.dataset_id, blob.payload_hash),
                    connection=conn,
                )
            )
            if row is None:
                raise SyncStoreError("Sync blob object insert did not produce a retrievable record")
            if _blob_object_fingerprint_from_row(row) != _blob_object_fingerprint_from_create(blob):
                raise SyncIdempotencyConflictError(
                    "Sync blob payload hash was reused with different metadata"
                )
            if session is not None:
                self.execute(
                    """
                    UPDATE sync_blob_upload_sessions
                       SET status = ?, blob_id = ?, updated_at = ?
                     WHERE upload_id = ?
                    """,
                    ("complete", row["blob_id"], now, session["upload_id"]),
                    connection=conn,
                )
            return _blob_object_from_row(row)

    def get_blob_object(
        self,
        dataset_id: str,
        *,
        attachment_id: str | None = None,
        blob_id: str | None = None,
        payload_hash: str | None = None,
        owner_user_id: str | None = None,
    ) -> SyncBlobObject | None:
        """Return an available blob object scoped by dataset and optional identity filters."""

        self._require_dataset(dataset_id)
        row = _first(
            self.execute(
                """
                SELECT *
                  FROM sync_blob_objects
                 WHERE dataset_id = ?
                   AND status = 'available'
                   AND (? IS NULL OR owner_user_id = ?)
                   AND (? IS NULL OR blob_id = ?)
                   AND (? IS NULL OR payload_hash = ?)
                   AND (
                        ? IS NULL
                        OR attachment_id = ?
                        OR payload_hash IN (
                            SELECT payload_hash
                              FROM sync_attachments
                             WHERE dataset_id = ?
                               AND attachment_id = ?
                        )
                   )
                 ORDER BY updated_at DESC, blob_id ASC
                 LIMIT 1
                """,
                (
                    dataset_id,
                    owner_user_id,
                    owner_user_id,
                    blob_id,
                    blob_id,
                    payload_hash,
                    payload_hash,
                    attachment_id,
                    attachment_id,
                    dataset_id,
                    attachment_id,
                ),
            )
        )
        if row is None:
            return None
        return _blob_object_from_row(row)

    def list_blob_objects_for_dataset(
        self,
        dataset_id: str,
        *,
        status: str | None = "available",
    ) -> list[SyncBlobObject]:
        """Return committed blob metadata for a dataset without payload bytes."""

        self._require_dataset(dataset_id)
        rows = self.execute(
            """
            SELECT *
              FROM sync_blob_objects
             WHERE dataset_id = ?
               AND (? IS NULL OR status = ?)
             ORDER BY updated_at ASC, blob_id ASC
            """,
            (dataset_id, status, status),
        ).rows
        return [_blob_object_from_row(row) for row in rows]

    def summarize_blob_quota(
        self,
        owner_user_id: str,
        *,
        dataset_id: str | None = None,
    ) -> SyncBlobQuotaUsage:
        """Return committed and pending blob quota usage for one user."""

        if dataset_id is None:
            reserved_row = _first(
                self.execute(
                    """
                    SELECT COALESCE(SUM(reserved_quota_bytes), 0) AS bytes,
                           COUNT(*) AS active_upload_count
                      FROM sync_blob_upload_sessions
                     WHERE owner_user_id = ?
                       AND status IN ('created', 'uploading')
                    """,
                    (owner_user_id,),
                )
            )
            used_row = _first(
                self.execute(
                    """
                    SELECT COALESCE(SUM(size_bytes), 0) AS bytes
                      FROM sync_blob_objects
                     WHERE owner_user_id = ?
                       AND status = 'available'
                    """,
                    (owner_user_id,),
                )
            )
        else:
            reserved_row = _first(
                self.execute(
                    """
                    SELECT COALESCE(SUM(reserved_quota_bytes), 0) AS bytes,
                           COUNT(*) AS active_upload_count
                      FROM sync_blob_upload_sessions
                     WHERE owner_user_id = ?
                       AND dataset_id = ?
                       AND status IN ('created', 'uploading')
                    """,
                    (owner_user_id, dataset_id),
                )
            )
            used_row = _first(
                self.execute(
                    """
                    SELECT COALESCE(SUM(size_bytes), 0) AS bytes
                      FROM sync_blob_objects
                     WHERE owner_user_id = ?
                       AND dataset_id = ?
                       AND status = 'available'
                    """,
                    (owner_user_id, dataset_id),
                )
            )
        return SyncBlobQuotaUsage(
            owner_user_id=owner_user_id,
            dataset_id=dataset_id,
            reserved_blob_bytes=int(reserved_row["bytes"] if reserved_row else 0),
            used_blob_bytes=int(used_row["bytes"] if used_row else 0),
            active_upload_count=int(reserved_row["active_upload_count"] if reserved_row else 0),
        )

    def _require_blob_upload_session(
        self,
        upload_id: str,
        *,
        dataset_id: str | None = None,
        connection: Any,
    ) -> dict[str, Any]:
        if dataset_id is None:
            row = _first(
                self.execute(
                    "SELECT * FROM sync_blob_upload_sessions WHERE upload_id = ?",
                    (upload_id,),
                    connection=connection,
                )
            )
        else:
            row = _first(
                self.execute(
                    """
                    SELECT * FROM sync_blob_upload_sessions
                     WHERE upload_id = ? AND dataset_id = ?
                    """,
                    (upload_id, dataset_id),
                    connection=connection,
                )
            )
        if row is None:
            raise SyncStoreError(f"Sync blob upload session not found: {upload_id}")
        return row

    def _blob_chunk_indexes(
        self,
        upload_id: str,
        *,
        connection: Any,
    ) -> list[int]:
        result = self.execute(
            """
            SELECT chunk_index
              FROM sync_blob_chunks
             WHERE upload_id = ?
             ORDER BY chunk_index ASC
            """,
            (upload_id,),
            connection=connection,
        )
        return [int(row["chunk_index"]) for row in result.rows]

    def _blob_upload_session_with_chunks(
        self,
        upload_id: str,
        *,
        connection: Any | None = None,
    ) -> SyncBlobUploadSession:
        row = _first(
            self.execute(
                "SELECT * FROM sync_blob_upload_sessions WHERE upload_id = ?",
                (upload_id,),
                connection=connection,
            )
        )
        if row is None:
            raise SyncStoreError(f"Sync blob upload session not found: {upload_id}")
        return _blob_upload_session_from_row(
            row,
            uploaded_chunks=self._blob_chunk_indexes(upload_id, connection=connection),
        )

    def _find_active_blob_session_for_blob(
        self,
        blob: SyncBlobObjectCreate,
        *,
        connection: Any,
    ) -> dict[str, Any] | None:
        return _first(
            self.execute(
                """
                SELECT * FROM sync_blob_upload_sessions
                 WHERE dataset_id = ?
                   AND attachment_id = ?
                   AND payload_hash = ?
                   AND status IN ('created', 'uploading')
                 ORDER BY created_at ASC
                 LIMIT 1
                """,
                (blob.dataset_id, blob.attachment_id, blob.payload_hash),
                connection=connection,
            )
        )

    def summarize_restore_manifest_dataset(
        self,
        dataset_id: str,
        *,
        user_id: str,
        domains: Sequence[SyncDomain] | None = None,
    ) -> SyncRestoreManifestStats:
        """Return aggregate restore-manifest statistics for a dataset."""

        if not user_id:
            raise SyncStoreError("user_id is required for Sync restore manifest summary")
        self._require_dataset(dataset_id)
        domain_filter_enabled = domains is not None
        domain_list = list(domains or [])

        counts: dict[str, int] = {}
        byte_estimates: dict[str, int] = {}
        last_updated_at: str | None = None
        unresolved_conflicts = 0
        attachment_availability: dict[str, int] = {}
        attachment_size_classes: dict[str, int] = {}

        if not domain_filter_enabled or domain_list:
            params: list[Any] = [dataset_id]
            sql = """
                SELECT domain,
                       COUNT(*) AS envelope_count,
                       COALESCE(SUM(COALESCE(payload_size_bytes, 0)), 0) AS byte_estimate,
                       MAX(server_timestamp) AS last_updated_at
                  FROM sync_envelopes
                 WHERE dataset_id = ?
            """
            sql += _domain_filter_sql(domain_list if domain_filter_enabled else None, params)
            sql += " GROUP BY domain"
            for row in self.execute(sql, tuple(params)).rows:
                domain = str(row["domain"])
                counts[domain] = int(row["envelope_count"] or 0)
                byte_estimates[domain] = int(row["byte_estimate"] or 0)
                row_last_updated = _timestamp_to_string(row.get("last_updated_at"))
                if row_last_updated and (
                    last_updated_at is None or row_last_updated > last_updated_at
                ):
                    last_updated_at = row_last_updated

            params = [dataset_id]
            sql = """
                SELECT COUNT(*) AS conflict_count
                  FROM sync_conflicts
                 WHERE dataset_id = ?
                   AND status = 'unresolved'
            """
            sql += _domain_filter_sql(domain_list if domain_filter_enabled else None, params)
            conflict_row = _first(self.execute(sql, tuple(params)))
            unresolved_conflicts = int((conflict_row or {}).get("conflict_count") or 0)

            params = [dataset_id]
            sql = """
                SELECT COUNT(*) AS attachment_count,
                       COALESCE(SUM(CASE WHEN size_bytes <= 1048576 THEN 1 ELSE 0 END), 0)
                           AS small_count,
                       COALESCE(SUM(
                           CASE
                               WHEN size_bytes > 1048576 AND size_bytes <= 16777216
                               THEN 1 ELSE 0
                           END
                       ), 0) AS medium_count,
                       COALESCE(SUM(CASE WHEN size_bytes > 16777216 THEN 1 ELSE 0 END), 0)
                           AS large_count
                  FROM sync_attachments
                 WHERE dataset_id = ?
            """
            sql += _domain_filter_sql(domain_list if domain_filter_enabled else None, params)
            attachment_row = _first(self.execute(sql, tuple(params)))
            attachment_count = int((attachment_row or {}).get("attachment_count") or 0)
            if attachment_count:
                attachment_availability["available"] = attachment_count
                size_counts = {
                    "small": int((attachment_row or {}).get("small_count") or 0),
                    "medium": int((attachment_row or {}).get("medium_count") or 0),
                    "large": int((attachment_row or {}).get("large_count") or 0),
                }
                attachment_size_classes.update(
                    {
                        size_class: count
                        for size_class, count in size_counts.items()
                        if count > 0
                    }
                )

        key_row = _first(
            self.execute(
                """
                SELECT 1 AS key_available
                  FROM sync_key_records
                 WHERE dataset_id = ?
                   AND user_id = ?
                   AND key_purpose = 'dataset_recovery'
                   AND revoked_at IS NULL
                 LIMIT 1
                """,
                (dataset_id, user_id),
            )
        )

        return SyncRestoreManifestStats(
            approximate_counts=dict(sorted(counts.items())),
            byte_estimates=dict(sorted(byte_estimates.items())),
            last_updated_at=last_updated_at,
            unresolved_conflicts=unresolved_conflicts,
            attachment_availability=dict(sorted(attachment_availability.items())),
            attachment_size_classes=dict(sorted(attachment_size_classes.items())),
            key_recovery_available=key_row is not None,
        )

    def _ensure_device_lifecycle_columns(self, *, connection: Any) -> None:
        existing = {
            column.get("name")
            for column in self.backend.get_table_info("sync_devices", connection=connection)
            if isinstance(column, dict)
        }
        if self.backend_type == BackendType.POSTGRESQL:
            column_specs = {
                "status": "TEXT NOT NULL DEFAULT 'active'",
                "user_label": "TEXT",
                "authorized_at": "TIMESTAMPTZ",
                "revoked_reason": "TEXT",
            }
        else:
            column_specs = {
                "status": "TEXT NOT NULL DEFAULT 'active'",
                "user_label": "TEXT",
                "authorized_at": "TEXT",
                "revoked_reason": "TEXT",
            }
        for column_name, column_spec in column_specs.items():
            if column_name in existing:
                continue
            self.execute(
                f"ALTER TABLE sync_devices ADD COLUMN {column_name} {column_spec}",
                connection=connection,
            )
        self.execute(
            """
            UPDATE sync_devices
               SET status = 'revoked'
             WHERE revoked_at IS NOT NULL
               AND (status IS NULL OR status <> 'revoked')
            """,
            connection=connection,
        )
        self.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_sync_devices_user_status
                ON sync_devices(user_id, status, last_seen_at)
            """,
            connection=connection,
        )

    def _ensure_device_lifecycle_tables(self, *, connection: Any) -> None:
        if self.backend_type == BackendType.POSTGRESQL:
            schema = """
            CREATE TABLE IF NOT EXISTS sync_device_authorizations (
                authorization_id TEXT PRIMARY KEY,
                dataset_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                device_id TEXT NOT NULL,
                authorization_method TEXT NOT NULL,
                status TEXT NOT NULL,
                requested_at TIMESTAMPTZ NOT NULL,
                approved_at TIMESTAMPTZ,
                approving_device_id TEXT,
                idempotency_key TEXT,
                approval_idempotency_key TEXT,
                UNIQUE(dataset_id, device_id, idempotency_key)
            );
            CREATE TABLE IF NOT EXISTS sync_device_domain_acks (
                dataset_id TEXT NOT NULL,
                device_id TEXT NOT NULL,
                domain TEXT NOT NULL,
                through_server_sequence BIGINT NOT NULL DEFAULT 0,
                applied_at TIMESTAMPTZ NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL,
                idempotency_key TEXT,
                PRIMARY KEY(dataset_id, device_id, domain)
            );
            CREATE TABLE IF NOT EXISTS sync_device_blob_acks (
                dataset_id TEXT NOT NULL,
                device_id TEXT NOT NULL,
                attachment_id TEXT NOT NULL,
                payload_hash TEXT NOT NULL,
                verified_at TIMESTAMPTZ NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL,
                idempotency_key TEXT,
                PRIMARY KEY(dataset_id, device_id, attachment_id)
            );
            """
        else:
            schema = """
            CREATE TABLE IF NOT EXISTS sync_device_authorizations (
                authorization_id TEXT PRIMARY KEY,
                dataset_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                device_id TEXT NOT NULL,
                authorization_method TEXT NOT NULL,
                status TEXT NOT NULL,
                requested_at TEXT NOT NULL,
                approved_at TEXT,
                approving_device_id TEXT,
                idempotency_key TEXT,
                approval_idempotency_key TEXT,
                UNIQUE(dataset_id, device_id, idempotency_key)
            );
            CREATE TABLE IF NOT EXISTS sync_device_domain_acks (
                dataset_id TEXT NOT NULL,
                device_id TEXT NOT NULL,
                domain TEXT NOT NULL,
                through_server_sequence INTEGER NOT NULL DEFAULT 0,
                applied_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                idempotency_key TEXT,
                PRIMARY KEY(dataset_id, device_id, domain)
            );
            CREATE TABLE IF NOT EXISTS sync_device_blob_acks (
                dataset_id TEXT NOT NULL,
                device_id TEXT NOT NULL,
                attachment_id TEXT NOT NULL,
                payload_hash TEXT NOT NULL,
                verified_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                idempotency_key TEXT,
                PRIMARY KEY(dataset_id, device_id, attachment_id)
            );
            """
        self.backend.create_tables(schema, connection=connection)
        statements = [
            """
            CREATE INDEX IF NOT EXISTS idx_sync_device_authorizations_dataset_device
                ON sync_device_authorizations(dataset_id, device_id, requested_at)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_sync_device_authorizations_user_status
                ON sync_device_authorizations(user_id, status, requested_at)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_sync_device_domain_acks_device
                ON sync_device_domain_acks(device_id, dataset_id)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_sync_device_blob_acks_device
                ON sync_device_blob_acks(device_id, dataset_id)
            """,
        ]
        for statement in statements:
            self.execute(statement, connection=connection)

    def _ensure_background_sync_tables(self, *, connection: Any) -> None:
        if self.backend_type == BackendType.POSTGRESQL:
            schema = """
            CREATE TABLE IF NOT EXISTS sync_background_policies (
                dataset_id TEXT NOT NULL,
                device_id TEXT NOT NULL,
                enabled BOOLEAN NOT NULL DEFAULT TRUE,
                minimum_interval_seconds INTEGER NOT NULL DEFAULT 300,
                backoff_floor_seconds INTEGER NOT NULL DEFAULT 60,
                max_batch_size INTEGER NOT NULL DEFAULT 100,
                max_blob_bytes_per_run BIGINT,
                respect_metered_networks BOOLEAN NOT NULL DEFAULT TRUE,
                maintenance_window_json TEXT,
                paused_reason TEXT,
                pending_local_changes BOOLEAN NOT NULL DEFAULT FALSE,
                updated_at TIMESTAMPTZ NOT NULL,
                PRIMARY KEY(dataset_id, device_id)
            );
            CREATE TABLE IF NOT EXISTS sync_background_leases (
                dataset_id TEXT NOT NULL,
                device_id TEXT NOT NULL,
                lease_id TEXT NOT NULL,
                expires_at TIMESTAMPTZ NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL,
                PRIMARY KEY(dataset_id, device_id)
            );
            """
        else:
            schema = """
            CREATE TABLE IF NOT EXISTS sync_background_policies (
                dataset_id TEXT NOT NULL,
                device_id TEXT NOT NULL,
                enabled INTEGER NOT NULL DEFAULT 1,
                minimum_interval_seconds INTEGER NOT NULL DEFAULT 300,
                backoff_floor_seconds INTEGER NOT NULL DEFAULT 60,
                max_batch_size INTEGER NOT NULL DEFAULT 100,
                max_blob_bytes_per_run INTEGER,
                respect_metered_networks INTEGER NOT NULL DEFAULT 1,
                maintenance_window_json TEXT,
                paused_reason TEXT,
                pending_local_changes INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(dataset_id, device_id)
            );
            CREATE TABLE IF NOT EXISTS sync_background_leases (
                dataset_id TEXT NOT NULL,
                device_id TEXT NOT NULL,
                lease_id TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(dataset_id, device_id)
            );
            """
        self.backend.create_tables(schema, connection=connection)
        for statement in (
            """
            CREATE INDEX IF NOT EXISTS idx_sync_background_policies_device
                ON sync_background_policies(device_id, dataset_id)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_sync_background_leases_expiry
                ON sync_background_leases(expires_at)
            """,
        ):
            self.execute(statement, connection=connection)

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

    def _ensure_envelope_m1_columns(self, *, connection: Any) -> None:
        existing = {
            column.get("name")
            for column in self.backend.get_table_info("sync_envelopes", connection=connection)
            if isinstance(column, dict)
        }
        if self.backend_type == BackendType.POSTGRESQL:
            column_specs = {
                "client_profile_id": "TEXT",
                "client_sequence": "BIGINT",
                "base_server_cursor": "BIGINT",
                "base_object_revision": "BIGINT",
                "base_object_hash": "TEXT",
                "object_revision": "BIGINT",
                "parent_id": "TEXT",
                "schema_version": "INTEGER NOT NULL DEFAULT 1",
                "payload_json": "TEXT NOT NULL DEFAULT '{}'",
                "payload_hash": "TEXT",
                "created_at_client": "TIMESTAMPTZ",
                "received_at_server": "TIMESTAMPTZ",
                "deleted": "BOOLEAN NOT NULL DEFAULT FALSE",
                "encryption_metadata_json": "TEXT NOT NULL DEFAULT '{}'",
                "apply_status": "TEXT NOT NULL DEFAULT 'pending'",
                "apply_error_code": "TEXT",
                "apply_error_message": "TEXT",
                "applied_at": "TIMESTAMPTZ",
            }
        else:
            column_specs = {
                "client_profile_id": "TEXT",
                "client_sequence": "INTEGER",
                "base_server_cursor": "INTEGER",
                "base_object_revision": "INTEGER",
                "base_object_hash": "TEXT",
                "object_revision": "INTEGER",
                "parent_id": "TEXT",
                "schema_version": "INTEGER NOT NULL DEFAULT 1",
                "payload_json": "TEXT NOT NULL DEFAULT '{}'",
                "payload_hash": "TEXT",
                "created_at_client": "TEXT",
                "received_at_server": "TEXT",
                "deleted": "INTEGER NOT NULL DEFAULT 0",
                "encryption_metadata_json": "TEXT NOT NULL DEFAULT '{}'",
                "apply_status": "TEXT NOT NULL DEFAULT 'pending'",
                "apply_error_code": "TEXT",
                "apply_error_message": "TEXT",
                "applied_at": "TEXT",
            }
        for column_name, column_spec in column_specs.items():
            if column_name in existing:
                continue
            self.execute(
                f"ALTER TABLE sync_envelopes ADD COLUMN {column_name} {column_spec}",
                connection=connection,
            )

    def _ensure_sync_object_state_table(self, *, connection: Any) -> None:
        if self.backend_type == BackendType.POSTGRESQL:
            schema = """
            CREATE TABLE IF NOT EXISTS sync_object_state (
                dataset_id TEXT NOT NULL,
                domain TEXT NOT NULL,
                object_id TEXT NOT NULL,
                object_revision BIGINT NOT NULL,
                object_hash TEXT NOT NULL,
                latest_server_cursor BIGINT NOT NULL,
                deleted BOOLEAN NOT NULL DEFAULT FALSE,
                updated_at TIMESTAMPTZ NOT NULL,
                PRIMARY KEY (dataset_id, domain, object_id)
            )
            """
        else:
            schema = """
            CREATE TABLE IF NOT EXISTS sync_object_state (
                dataset_id TEXT NOT NULL,
                domain TEXT NOT NULL,
                object_id TEXT NOT NULL,
                object_revision INTEGER NOT NULL,
                object_hash TEXT NOT NULL,
                latest_server_cursor INTEGER NOT NULL,
                deleted INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (dataset_id, domain, object_id)
            )
            """
        self.backend.create_tables(schema, connection=connection)
        self.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_sync_object_state_dataset_domain_object
                ON sync_object_state(dataset_id, domain, object_id)
            """,
            connection=connection,
        )

    def _ensure_envelope_m1_indexes(self, *, connection: Any) -> None:
        statements = [
            """
            CREATE INDEX IF NOT EXISTS idx_sync_envelopes_dataset_domain_object
                ON sync_envelopes(dataset_id, domain, entity_id)
            """,
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_sync_envelopes_dataset_device_client_sequence
                ON sync_envelopes(dataset_id, device_id, client_sequence)
                WHERE device_id IS NOT NULL AND client_sequence IS NOT NULL
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_sync_envelopes_payload_hash
                ON sync_envelopes(dataset_id, payload_hash)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_sync_envelopes_failed_apply
                ON sync_envelopes(dataset_id, apply_status, server_sequence)
                WHERE apply_status = 'failed'
            """,
        ]
        for statement in statements:
            self.execute(statement, connection=connection)

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

    def _ensure_key_record_rotation_columns(self, *, connection: Any) -> None:
        columns = {
            column.get("name")
            for column in self.backend.get_table_info("sync_key_records", connection=connection)
            if isinstance(column, dict)
        }
        superseded_type = (
            "TIMESTAMPTZ"
            if self.backend_type == BackendType.POSTGRESQL
            else "TEXT"
        )
        column_specs = [
            (
                "encryption_policy",
                "TEXT NOT NULL DEFAULT 'server_trusted_v1'",
            ),
            ("key_epoch", "INTEGER NOT NULL DEFAULT 1"),
            ("active_from_server_sequence", "INTEGER"),
            ("superseded_at", superseded_type),
            ("rotation_source_key_record_ids_json", "TEXT NOT NULL DEFAULT '[]'"),
            ("wrapped_for", "TEXT NOT NULL DEFAULT 'recovery'"),
            ("rewrap_status", "TEXT NOT NULL DEFAULT 'not_required'"),
        ]
        for column_name, column_type in column_specs:
            if column_name in columns:
                continue
            if self.backend_type == BackendType.POSTGRESQL:
                self.execute(
                    f"ALTER TABLE sync_key_records ADD COLUMN IF NOT EXISTS {column_name} {column_type}",
                    connection=connection,
                )
            else:
                self.execute(
                    f"ALTER TABLE sync_key_records ADD COLUMN {column_name} {column_type}",
                    connection=connection,
                )

        self.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_sync_key_records_epoch
                ON sync_key_records(dataset_id, encryption_policy, key_epoch)
            """,
            connection=connection,
        )
        self.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_sync_key_records_rewrap
                ON sync_key_records(dataset_id, rewrap_status)
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
