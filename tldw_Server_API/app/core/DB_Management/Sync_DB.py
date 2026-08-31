from __future__ import annotations

"""Database helper for per-user Sync v2 storage."""

import hashlib
import json
import os
import re
import typing
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse
from uuid import UUID, uuid4

from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.Sync.v2.errors import (
    SyncConflictNotFoundError,
    SyncDatasetNotFoundError,
    SyncHeadConflictError,
    SyncIdempotencyConflictError,
    SyncInvalidDomainError,
    SyncMaterializationBusyError,
    SyncMaterializationPredecessorError,
    SyncStoreError,
)
from tldw_Server_API.app.core.Sync.v2.models import (
    DEFAULT_M1_ENCRYPTION_POLICY,
    M1_SYNC_DOMAINS,
    MEDIA_SYNC_DOMAINS,
    NOTES_LINK_DOMAINS,
    NOTES_ORGANIZATION_DOMAINS,
    NOTES_TASK_SYNC_DOMAINS,
    PERSONAL_CONTEXT_SYNC_DOMAINS,
    SOURCE_CACHE_SYNC_DOMAINS,
    SYNC_REBASE_REQUIRED_AFTER_CONFLICT_RESOLUTION,
    SYNC_V2_INTERNAL_OPERATIONS,
    WORKSPACE_SYNC_DOMAINS,
    ConflictStatus,
    SyncApplyStatus,
    SyncAttachment,
    SyncAttachmentCreate,
    SyncAttachmentRevisionBinding,
    SyncAttachmentRevisionBindingCreate,
    SyncBackgroundDomainStatus,
    SyncBackgroundLease,
    SyncBackgroundLeaseCreate,
    SyncBackgroundPolicy,
    SyncBackgroundPolicyUpsert,
    SyncBlobAvailabilityStatus,
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
    SyncDatasetStorageNamespace,
    SyncDevice,
    SyncDeviceAcknowledgmentSummary,
    SyncDeviceAuthorization,
    SyncDeviceAuthorizationCreate,
    SyncDeviceBlobAck,
    SyncDeviceBlobAckCreate,
    SyncDeviceBlobIdAck,
    SyncDeviceBlobIdAckCreate,
    SyncDeviceCursor,
    SyncDeviceDomainAck,
    SyncDeviceDomainAckCreate,
    SyncDeviceUpsert,
    SyncDomain,
    SyncDomainEnvelopeSummary,
    SyncEnvelope,
    SyncEnvelopeCreate,
    SyncKeyRecord,
    SyncKeyRecordCreate,
    SyncKeyRotationEnvelopeRange,
    SyncNotesAttachmentCleanupCandidate,
    SyncNotesAttachmentSourceMap,
    SyncObjectState,
    SyncRestoreManifestStats,
    normalize_sync_timestamp,
)
from tldw_Server_API.app.core.Sync.v2.mutation_group_validation import (
    SYNC_MUTATION_GROUP_MAX_SIZE,
)
from tldw_Server_API.app.core.Sync.v2.notes_moodboard_studio_readiness import (
    NOTES_MOODBOARD_STUDIO_READINESS_REASON_CODES_BY_KEY,
    NOTES_MOODBOARD_STUDIO_READINESS_STATES,
    NOTES_MOODBOARD_STUDIO_SERVER_METADATA_KEYS,
    NotesMoodboardStudioReadinessRecord,
    default_notes_moodboard_studio_readiness_record,
    parse_notes_moodboard_studio_readiness_record,
)
from tldw_Server_API.app.core.Sync.v2.notes_task_readiness import (
    NOTES_TASK_READINESS_REASON_CODES_BY_KEY,
    NOTES_TASK_READINESS_STATES,
    NOTES_TASK_SERVER_METADATA_KEYS,
    NotesTaskReadinessRecord,
    default_notes_task_readiness_record,
    notes_task_capture_is_active,
    notes_task_sync_is_ready,
    parse_notes_task_readiness_record,
)
from tldw_Server_API.app.core.Utils.path_utils import safe_join

from .backends.base import (
    BackendType,
    DatabaseBackend,
    DatabaseConfig,
    QueryResult,
)
from .backends.base import DatabaseError as BackendDatabaseError
from .backends.factory import DatabaseBackendFactory

SYNC_DB_FILENAME = "Sync_v2.db"
SYNC_APPLY_STATUSES: set[str] = {
    "pending",
    "applied",
    "failed",
    "conflict",
    "superseded",
}
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
_NOTES_TASK_READINESS_TRANSITIONS = {
    "not_enrolled": frozenset({"not_enrolled", "enrolling"}),
    "enrolling": frozenset(
        {"enrolling", "bootstrapping", "blocked", "not_enrolled"}
    ),
    "bootstrapping": frozenset(
        {"bootstrapping", "verifying", "blocked", "not_enrolled"}
    ),
    "verifying": frozenset({"verifying", "ready", "blocked", "not_enrolled"}),
    "blocked": frozenset(
        {"blocked", "bootstrapping", "verifying", "not_enrolled"}
    ),
    "ready": frozenset({"ready"}),
}
_NOTES_TASK_LOCAL_UNBOUND_DATASET_ID = "local-unbound"


def _moodboard_studio_cursor_regressed(
    readiness_key: str,
    current: object,
    requested: object,
) -> bool:
    """Return whether a canonical moodboard or Studio cursor moved backward.

    Args:
        readiness_key: Domain-specific readiness metadata key.
        current: Current parsed UUID cursor or placement UUID pair.
        requested: Requested parsed UUID cursor or placement UUID pair.

    Returns:
        ``True`` when the requested cursor sorts before the current cursor.

    Raises:
        SyncStoreError: If the domain or either cursor shape is invalid.
    """
    if readiness_key in {"notes_moodboard_v1", "notes_studio_document_v1"}:
        if not isinstance(current, UUID) or not isinstance(requested, UUID):
            raise SyncStoreError("notes_moodboard_studio_readiness_cursor_invalid")
        return requested.int < current.int
    if readiness_key != "notes_moodboard_note_v1":
        raise SyncStoreError("notes_moodboard_studio_readiness_cursor_invalid")
    if not isinstance(current, tuple) or not isinstance(requested, tuple):
        raise SyncStoreError("notes_moodboard_studio_readiness_cursor_invalid")
    current_board, current_note = current
    requested_board, requested_note = requested
    if not all(
        isinstance(item, UUID)
        for item in (current_board, current_note, requested_board, requested_note)
    ):
        raise SyncStoreError("notes_moodboard_studio_readiness_cursor_invalid")
    return requested_board.int < current_board.int or (
        requested_board.int == current_board.int
        and requested_note.int < current_note.int
    )


def _is_rebase_required_conflict_source(
    conflict_row: Mapping[str, Any],
    envelope_row: Mapping[str, Any],
) -> bool:
    """Validate and identify one generated rebase-required conflict source."""

    conflict_marked = (
        conflict_row.get("conflict_type")
        == SYNC_REBASE_REQUIRED_AFTER_CONFLICT_RESOLUTION
    )
    envelope_marked = (
        envelope_row.get("apply_error_code")
        == SYNC_REBASE_REQUIRED_AFTER_CONFLICT_RESOLUTION
    )
    if conflict_marked != envelope_marked:
        raise SyncStoreError("Sync rebase conflict marker does not match its source")
    return conflict_marked


def _is_mutation_group_step_unique_error(exc: BaseException) -> bool:
    """Match PostgreSQL's named mutation-group step uniqueness violation."""

    current: BaseException | None = exc
    while current is not None:
        sqlstate = getattr(current, "sqlstate", None) or getattr(
            current, "pgcode", None
        )
        diagnostics = getattr(current, "diag", None)
        if (
            sqlstate == "23505"
            and getattr(diagnostics, "constraint_name", None)
            == "uq_sync_envelopes_dataset_mutation_group_step"
        ):
            return True
        current = current.__cause__
    return False


def _is_materialization_lock_error(exc: BaseException) -> bool:
    """Match bounded PostgreSQL/SQLite lock acquisition failures."""

    current: BaseException | None = exc
    while current is not None:
        sqlstate = getattr(current, "sqlstate", None) or getattr(
            current, "pgcode", None
        )
        if sqlstate in {"40P01", "55P03"}:
            return True
        message = str(current).lower()
        if "database is locked" in message or "database table is locked" in message:
            return True
        current = current.__cause__
    return False

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

CREATE TABLE IF NOT EXISTS sync_personal_context_link_receipts (
    user_id TEXT NOT NULL,
    dataset_id TEXT NOT NULL,
    device_id TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    integrity_key_id TEXT NOT NULL,
    purge_generation INTEGER NOT NULL,
    bootstrap_cursor TEXT NOT NULL,
    PRIMARY KEY (user_id, dataset_id, device_id)
);

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
    mutation_group_id TEXT,
    mutation_step INTEGER,
    mutation_step_count INTEGER,
    mutation_plan_hash TEXT,
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
CREATE INDEX IF NOT EXISTS idx_sync_envelopes_dataset_domain_entity_status_sequence
    ON sync_envelopes(dataset_id, domain, entity_id, status, server_sequence);
CREATE INDEX IF NOT EXISTS idx_sync_envelopes_dataset_status_sequence
    ON sync_envelopes(dataset_id, status, server_sequence);
CREATE INDEX IF NOT EXISTS idx_sync_envelopes_dataset_device_sequence
    ON sync_envelopes(dataset_id, device_id, server_sequence);
CREATE UNIQUE INDEX IF NOT EXISTS uq_sync_envelopes_dataset_mutation_group_step
    ON sync_envelopes(dataset_id, mutation_group_id, mutation_step)
    WHERE mutation_group_id IS NOT NULL AND mutation_step IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_sync_envelopes_dataset_mutation_group_step
    ON sync_envelopes(dataset_id, mutation_group_id, mutation_step);
CREATE UNIQUE INDEX IF NOT EXISTS idx_sync_envelopes_dataset_device_client_sequence
    ON sync_envelopes(dataset_id, device_id, client_sequence)
    WHERE device_id IS NOT NULL AND client_sequence IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_sync_envelopes_payload_hash
    ON sync_envelopes(dataset_id, payload_hash);
CREATE INDEX IF NOT EXISTS idx_sync_envelopes_failed_apply
    ON sync_envelopes(dataset_id, apply_status, server_sequence)
    WHERE apply_status = 'failed';
CREATE INDEX IF NOT EXISTS idx_sync_envelopes_outstanding_apply
    ON sync_envelopes(dataset_id, server_sequence)
    WHERE status = 'accepted' AND apply_status NOT IN ('applied', 'superseded');
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

CREATE TABLE IF NOT EXISTS sync_current_heads (
    dataset_id TEXT NOT NULL,
    domain TEXT NOT NULL,
    object_id TEXT NOT NULL,
    latest_server_cursor INTEGER NOT NULL,
    PRIMARY KEY (dataset_id, domain, object_id)
);
CREATE INDEX IF NOT EXISTS idx_sync_current_heads_dataset_domain_cursor
    ON sync_current_heads(dataset_id, domain, latest_server_cursor, object_id);

CREATE TABLE IF NOT EXISTS sync_materialization_locks (
    dataset_id TEXT NOT NULL,
    domain TEXT NOT NULL,
    object_id TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (dataset_id, domain, object_id)
);

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
CREATE INDEX IF NOT EXISTS idx_sync_blob_objects_retention
    ON sync_blob_objects(dataset_id, status, updated_at, blob_id);

CREATE TABLE IF NOT EXISTS sync_attachment_revision_bindings (
    dataset_id TEXT NOT NULL,
    attachment_id TEXT NOT NULL,
    attachment_revision INTEGER NOT NULL,
    blob_hash TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    establishing_server_cursor INTEGER NOT NULL,
    availability_at_acceptance TEXT NOT NULL,
    resolved_blob_id TEXT,
    retention_released_at TEXT,
    created_at TEXT NOT NULL,
    PRIMARY KEY (dataset_id, attachment_id, attachment_revision),
    CHECK (length(dataset_id) > 0),
    CHECK (
        length(attachment_id) = 36
        AND lower(attachment_id) = attachment_id
        AND substr(attachment_id, 9, 1) = '-'
        AND substr(attachment_id, 14, 1) = '-'
        AND substr(attachment_id, 15, 1) = '4'
        AND substr(attachment_id, 19, 1) = '-'
        AND substr(attachment_id, 20, 1) IN ('8', '9', 'a', 'b')
        AND substr(attachment_id, 24, 1) = '-'
        AND length(replace(attachment_id, '-', '')) = 32
        AND replace(attachment_id, '-', '') NOT GLOB '*[^0-9a-f]*'
    ),
    CHECK (attachment_revision > 0),
    CHECK (length(blob_hash) = 71),
    CHECK (substr(blob_hash, 1, 7) = 'sha256:'),
    CHECK (substr(blob_hash, 8) NOT GLOB '*[^0-9a-f]*'),
    CHECK (size_bytes > 0),
    CHECK (establishing_server_cursor > 0),
    CHECK (availability_at_acceptance IN ('available', 'metadata_only')),
    CHECK (resolved_blob_id IS NULL OR length(resolved_blob_id) > 0)
);
CREATE INDEX IF NOT EXISTS idx_sync_attachment_bindings_unresolved
    ON sync_attachment_revision_bindings(
        dataset_id, establishing_server_cursor, attachment_id, attachment_revision
    )
    WHERE resolved_blob_id IS NULL AND retention_released_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_sync_attachment_bindings_blob
    ON sync_attachment_revision_bindings(dataset_id, resolved_blob_id);
CREATE INDEX IF NOT EXISTS idx_sync_attachment_bindings_blob_retention
    ON sync_attachment_revision_bindings(
        dataset_id, resolved_blob_id, establishing_server_cursor,
        attachment_id, attachment_revision
    )
    WHERE retention_released_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_sync_attachment_bindings_retention_release
    ON sync_attachment_revision_bindings(dataset_id, establishing_server_cursor, attachment_id, attachment_revision)
    WHERE retention_released_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_sync_attachment_bindings_pending_digest
    ON sync_attachment_revision_bindings(dataset_id, blob_hash, size_bytes, establishing_server_cursor, attachment_id, attachment_revision)
    WHERE resolved_blob_id IS NULL AND retention_released_at IS NULL;

CREATE TABLE IF NOT EXISTS sync_dataset_storage_namespaces (
    dataset_id TEXT NOT NULL PRIMARY KEY,
    owner_user_id TEXT NOT NULL,
    storage_namespace_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    CHECK (length(dataset_id) > 0),
    CHECK (length(owner_user_id) > 0),
    CHECK (length(storage_namespace_id) = 32),
    CHECK (storage_namespace_id NOT GLOB '*[^0-9a-f]*')
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_sync_dataset_storage_namespace_id
    ON sync_dataset_storage_namespaces(storage_namespace_id);
CREATE INDEX IF NOT EXISTS idx_sync_dataset_storage_namespaces_owner
    ON sync_dataset_storage_namespaces(owner_user_id, dataset_id);

CREATE TABLE IF NOT EXISTS sync_notes_attachment_source_map (
    dataset_id TEXT NOT NULL,
    bootstrap_id TEXT NOT NULL,
    source_key_hash TEXT NOT NULL,
    note_id TEXT NOT NULL,
    attachment_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (dataset_id, bootstrap_id, source_key_hash),
    CHECK (length(dataset_id) > 0),
    CHECK (length(bootstrap_id) BETWEEN 1 AND 128),
    CHECK (length(source_key_hash) = 71),
    CHECK (substr(source_key_hash, 1, 7) = 'sha256:'),
    CHECK (substr(source_key_hash, 8) NOT GLOB '*[^0-9a-f]*'),
    CHECK (length(note_id) > 0),
    CHECK (
        length(attachment_id) = 36
        AND lower(attachment_id) = attachment_id
        AND substr(attachment_id, 9, 1) = '-'
        AND substr(attachment_id, 14, 1) = '-'
        AND substr(attachment_id, 15, 1) = '4'
        AND substr(attachment_id, 19, 1) = '-'
        AND substr(attachment_id, 20, 1) IN ('8', '9', 'a', 'b')
        AND substr(attachment_id, 24, 1) = '-'
        AND length(replace(attachment_id, '-', '')) = 32
        AND replace(attachment_id, '-', '') NOT GLOB '*[^0-9a-f]*'
    )
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_sync_notes_attachment_source_id
    ON sync_notes_attachment_source_map(dataset_id, attachment_id);

CREATE TABLE IF NOT EXISTS sync_notes_attachment_cleanup_candidates (
    dataset_id TEXT NOT NULL,
    bootstrap_id TEXT NOT NULL,
    source_key_hash TEXT NOT NULL,
    attachment_id TEXT NOT NULL,
    source_relative_path TEXT NOT NULL,
    source_path_hash TEXT NOT NULL,
    source_blob_hash TEXT NOT NULL,
    source_size_bytes INTEGER NOT NULL,
    source_modified_ns INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (dataset_id, bootstrap_id, source_key_hash),
    CHECK (length(dataset_id) > 0),
    CHECK (length(bootstrap_id) BETWEEN 1 AND 128),
    CHECK (length(source_key_hash) = 71),
    CHECK (substr(source_key_hash, 1, 7) = 'sha256:'),
    CHECK (substr(source_key_hash, 8) NOT GLOB '*[^0-9a-f]*'),
    CHECK (
        length(attachment_id) = 36
        AND lower(attachment_id) = attachment_id
        AND substr(attachment_id, 9, 1) = '-'
        AND substr(attachment_id, 14, 1) = '-'
        AND substr(attachment_id, 15, 1) = '4'
        AND substr(attachment_id, 19, 1) = '-'
        AND substr(attachment_id, 20, 1) IN ('8', '9', 'a', 'b')
        AND substr(attachment_id, 24, 1) = '-'
        AND length(replace(attachment_id, '-', '')) = 32
        AND replace(attachment_id, '-', '') NOT GLOB '*[^0-9a-f]*'
    ),
    CHECK (length(source_relative_path) BETWEEN 1 AND 4096),
    CHECK (length(source_path_hash) = 71),
    CHECK (substr(source_path_hash, 1, 7) = 'sha256:'),
    CHECK (substr(source_path_hash, 8) NOT GLOB '*[^0-9a-f]*'),
    CHECK (source_path_hash = source_key_hash),
    CHECK (length(source_blob_hash) = 71),
    CHECK (substr(source_blob_hash, 1, 7) = 'sha256:'),
    CHECK (substr(source_blob_hash, 8) NOT GLOB '*[^0-9a-f]*'),
    CHECK (typeof(source_size_bytes) = 'integer'),
    CHECK (typeof(source_modified_ns) = 'integer'),
    CHECK (source_size_bytes > 0),
    CHECK (source_modified_ns >= 0)
);
CREATE INDEX IF NOT EXISTS idx_sync_notes_attachment_cleanup_page
    ON sync_notes_attachment_cleanup_candidates(
        dataset_id, bootstrap_id, source_key_hash
    );

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
CREATE UNIQUE INDEX IF NOT EXISTS uq_sync_blob_upload_sessions_owner_key_without_device
    ON sync_blob_upload_sessions(dataset_id, owner_user_id, idempotency_key)
    WHERE device_id IS NULL AND idempotency_key IS NOT NULL;

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

SYNC_VERSIONED_DEVICE_STATE_MIGRATION_ID = "adapter_cursor_ack_blob_id_v1"
# Stable, process-independent signed 64-bit key reserved for this Sync migration.
SYNC_VERSIONED_DEVICE_STATE_MIGRATION_LOCK_KEY = 5_465_866_052_944_881_777

SYNC_VERSIONED_DEVICE_STATE_SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS sync_device_adapter_cursors (
    dataset_id TEXT NOT NULL,
    device_id TEXT NOT NULL,
    domain TEXT NOT NULL,
    adapter_version INTEGER NOT NULL CHECK (adapter_version > 0),
    last_pulled_sequence INTEGER NOT NULL DEFAULT 0 CHECK (last_pulled_sequence >= 0),
    max_delivered_sequence INTEGER NOT NULL DEFAULT 0 CHECK (max_delivered_sequence >= 0),
    updated_at TEXT NOT NULL,
    PRIMARY KEY(dataset_id, device_id, domain, adapter_version),
    CHECK (max_delivered_sequence <= last_pulled_sequence)
);
CREATE INDEX IF NOT EXISTS idx_sync_device_adapter_cursors_device
    ON sync_device_adapter_cursors(device_id, dataset_id);
CREATE TABLE IF NOT EXISTS sync_device_adapter_domain_acks (
    dataset_id TEXT NOT NULL,
    device_id TEXT NOT NULL,
    domain TEXT NOT NULL,
    adapter_version INTEGER NOT NULL CHECK (adapter_version > 0),
    through_server_sequence INTEGER NOT NULL DEFAULT 0 CHECK (through_server_sequence >= 0),
    applied_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    idempotency_key TEXT,
    PRIMARY KEY(dataset_id, device_id, domain, adapter_version)
);
CREATE INDEX IF NOT EXISTS idx_sync_device_adapter_domain_acks_device
    ON sync_device_adapter_domain_acks(device_id, dataset_id);
CREATE TABLE IF NOT EXISTS sync_device_blob_id_acks (
    dataset_id TEXT NOT NULL,
    device_id TEXT NOT NULL,
    blob_id TEXT NOT NULL,
    payload_hash TEXT NOT NULL,
    verified_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    idempotency_key TEXT,
    PRIMARY KEY(dataset_id, device_id, blob_id)
);
CREATE INDEX IF NOT EXISTS idx_sync_device_blob_id_acks_device
    ON sync_device_blob_id_acks(device_id, dataset_id);
"""

SYNC_VERSIONED_DEVICE_STATE_POSTGRES_SCHEMA = (
    SYNC_VERSIONED_DEVICE_STATE_SQLITE_SCHEMA
    .replace("last_pulled_sequence INTEGER", "last_pulled_sequence BIGINT")
    .replace("max_delivered_sequence INTEGER", "max_delivered_sequence BIGINT")
    .replace("through_server_sequence INTEGER", "through_server_sequence BIGINT")
    .replace("updated_at TEXT", "updated_at TIMESTAMPTZ")
    .replace("applied_at TEXT", "applied_at TIMESTAMPTZ")
    .replace("verified_at TEXT", "verified_at TIMESTAMPTZ")
)

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

CREATE TABLE IF NOT EXISTS sync_personal_context_link_receipts (
    user_id TEXT NOT NULL,
    dataset_id TEXT NOT NULL,
    device_id TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    integrity_key_id TEXT NOT NULL,
    purge_generation INTEGER NOT NULL,
    bootstrap_cursor TEXT NOT NULL,
    PRIMARY KEY (user_id, dataset_id, device_id)
);

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
    mutation_group_id TEXT,
    mutation_step INTEGER,
    mutation_step_count INTEGER,
    mutation_plan_hash TEXT,
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
CREATE INDEX IF NOT EXISTS idx_sync_envelopes_dataset_domain_entity_status_sequence
    ON sync_envelopes(dataset_id, domain, entity_id, status, server_sequence);
CREATE INDEX IF NOT EXISTS idx_sync_envelopes_dataset_status_sequence
    ON sync_envelopes(dataset_id, status, server_sequence);
CREATE INDEX IF NOT EXISTS idx_sync_envelopes_dataset_device_sequence
    ON sync_envelopes(dataset_id, device_id, server_sequence);
CREATE UNIQUE INDEX IF NOT EXISTS uq_sync_envelopes_dataset_mutation_group_step
    ON sync_envelopes(dataset_id, mutation_group_id, mutation_step)
    WHERE mutation_group_id IS NOT NULL AND mutation_step IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_sync_envelopes_dataset_mutation_group_step
    ON sync_envelopes(dataset_id, mutation_group_id, mutation_step);
CREATE UNIQUE INDEX IF NOT EXISTS idx_sync_envelopes_dataset_device_client_sequence
    ON sync_envelopes(dataset_id, device_id, client_sequence)
    WHERE device_id IS NOT NULL AND client_sequence IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_sync_envelopes_payload_hash
    ON sync_envelopes(dataset_id, payload_hash);
CREATE INDEX IF NOT EXISTS idx_sync_envelopes_failed_apply
    ON sync_envelopes(dataset_id, apply_status, server_sequence)
    WHERE apply_status = 'failed';
CREATE INDEX IF NOT EXISTS idx_sync_envelopes_outstanding_apply
    ON sync_envelopes(dataset_id, server_sequence)
    WHERE status = 'accepted' AND apply_status NOT IN ('applied', 'superseded');
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

CREATE TABLE IF NOT EXISTS sync_current_heads (
    dataset_id TEXT NOT NULL,
    domain TEXT NOT NULL,
    object_id TEXT NOT NULL,
    latest_server_cursor BIGINT NOT NULL,
    PRIMARY KEY (dataset_id, domain, object_id)
);
CREATE INDEX IF NOT EXISTS idx_sync_current_heads_dataset_domain_cursor
    ON sync_current_heads(dataset_id, domain, latest_server_cursor, object_id);

CREATE TABLE IF NOT EXISTS sync_materialization_locks (
    dataset_id TEXT NOT NULL,
    domain TEXT NOT NULL,
    object_id TEXT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (dataset_id, domain, object_id)
);

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
CREATE INDEX IF NOT EXISTS idx_sync_blob_objects_retention
    ON sync_blob_objects(dataset_id, status, updated_at, blob_id);

CREATE TABLE IF NOT EXISTS sync_attachment_revision_bindings (
    dataset_id TEXT NOT NULL,
    attachment_id TEXT NOT NULL,
    attachment_revision BIGINT NOT NULL,
    blob_hash TEXT NOT NULL,
    size_bytes BIGINT NOT NULL,
    establishing_server_cursor BIGINT NOT NULL,
    availability_at_acceptance TEXT NOT NULL,
    resolved_blob_id TEXT,
    retention_released_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (dataset_id, attachment_id, attachment_revision),
    CHECK (length(dataset_id) > 0),
    CHECK (attachment_id ~ '^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$'),
    CHECK (attachment_revision > 0),
    CHECK (blob_hash ~ '^sha256:[0-9a-f]{64}$'),
    CHECK (size_bytes > 0),
    CHECK (establishing_server_cursor > 0),
    CHECK (availability_at_acceptance IN ('available', 'metadata_only')),
    CHECK (resolved_blob_id IS NULL OR length(resolved_blob_id) > 0)
);
CREATE INDEX IF NOT EXISTS idx_sync_attachment_bindings_unresolved
    ON sync_attachment_revision_bindings(dataset_id, establishing_server_cursor, attachment_id, attachment_revision)
    WHERE resolved_blob_id IS NULL AND retention_released_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_sync_attachment_bindings_blob
    ON sync_attachment_revision_bindings(dataset_id, resolved_blob_id);
CREATE INDEX IF NOT EXISTS idx_sync_attachment_bindings_blob_retention
    ON sync_attachment_revision_bindings(
        dataset_id, resolved_blob_id, establishing_server_cursor,
        attachment_id, attachment_revision
    )
    WHERE retention_released_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_sync_attachment_bindings_retention_release
    ON sync_attachment_revision_bindings(dataset_id, establishing_server_cursor, attachment_id, attachment_revision)
    WHERE retention_released_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_sync_attachment_bindings_pending_digest
    ON sync_attachment_revision_bindings(dataset_id, blob_hash, size_bytes, establishing_server_cursor, attachment_id, attachment_revision)
    WHERE resolved_blob_id IS NULL AND retention_released_at IS NULL;

CREATE TABLE IF NOT EXISTS sync_dataset_storage_namespaces (
    dataset_id TEXT PRIMARY KEY,
    owner_user_id TEXT NOT NULL,
    storage_namespace_id TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    CHECK (length(dataset_id) > 0),
    CHECK (length(owner_user_id) > 0),
    CHECK (storage_namespace_id ~ '^[0-9a-f]{32}$')
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_sync_dataset_storage_namespace_id
    ON sync_dataset_storage_namespaces(storage_namespace_id);
CREATE INDEX IF NOT EXISTS idx_sync_dataset_storage_namespaces_owner
    ON sync_dataset_storage_namespaces(owner_user_id, dataset_id);

CREATE TABLE IF NOT EXISTS sync_notes_attachment_source_map (
    dataset_id TEXT NOT NULL,
    bootstrap_id TEXT NOT NULL,
    source_key_hash TEXT NOT NULL,
    note_id TEXT NOT NULL,
    attachment_id TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (dataset_id, bootstrap_id, source_key_hash),
    CHECK (length(dataset_id) > 0),
    CHECK (length(bootstrap_id) BETWEEN 1 AND 128),
    CHECK (source_key_hash ~ '^sha256:[0-9a-f]{64}$'),
    CHECK (length(note_id) > 0),
    CHECK (attachment_id ~ '^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$')
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_sync_notes_attachment_source_id
    ON sync_notes_attachment_source_map(dataset_id, attachment_id);

CREATE TABLE IF NOT EXISTS sync_notes_attachment_cleanup_candidates (
    dataset_id TEXT NOT NULL,
    bootstrap_id TEXT NOT NULL,
    source_key_hash TEXT NOT NULL,
    attachment_id TEXT NOT NULL,
    source_relative_path TEXT NOT NULL,
    source_path_hash TEXT NOT NULL,
    source_blob_hash TEXT NOT NULL,
    source_size_bytes BIGINT NOT NULL,
    source_modified_ns BIGINT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (dataset_id, bootstrap_id, source_key_hash),
    CHECK (length(dataset_id) > 0),
    CHECK (length(bootstrap_id) BETWEEN 1 AND 128),
    CHECK (source_key_hash ~ '^sha256:[0-9a-f]{64}$'),
    CHECK (attachment_id ~ '^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$'),
    CHECK (length(source_relative_path) BETWEEN 1 AND 4096),
    CHECK (source_path_hash ~ '^sha256:[0-9a-f]{64}$'),
    CHECK (source_path_hash = source_key_hash),
    CHECK (source_blob_hash ~ '^sha256:[0-9a-f]{64}$'),
    CHECK (source_size_bytes > 0),
    CHECK (source_modified_ns >= 0)
);
CREATE INDEX IF NOT EXISTS idx_sync_notes_attachment_cleanup_page
    ON sync_notes_attachment_cleanup_candidates(
        dataset_id, bootstrap_id, source_key_hash
    );

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
CREATE UNIQUE INDEX IF NOT EXISTS uq_sync_blob_upload_sessions_owner_key_without_device
    ON sync_blob_upload_sessions(dataset_id, owner_user_id, idempotency_key)
    WHERE device_id IS NULL AND idempotency_key IS NOT NULL;

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
    return normalize_sync_timestamp(value)


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
    resolved = safe_join(
        str(default_path.parent),
        raw_path or default_path.name,
        error_factory=lambda _exc: SyncStoreError("Sync v2 SQLite URL path escapes default directory"),
    )
    return Path(resolved)


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
        adapter_version=int(row.get("adapter_version") or 1),
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


def _device_blob_id_ack_from_row(row: dict[str, Any]) -> SyncDeviceBlobIdAck:
    return SyncDeviceBlobIdAck(
        dataset_id=row["dataset_id"],
        device_id=row["device_id"],
        blob_id=row["blob_id"],
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
    raw_created_at_client = row.get("created_at_client") or row.get("client_timestamp")
    raw_client_timestamp = row.get("client_timestamp") or row.get("created_at_client")
    created_at_client = (
        raw_created_at_client
        if isinstance(raw_created_at_client, str)
        else _timestamp_to_string(raw_created_at_client)
    )
    client_timestamp = (
        raw_client_timestamp
        if isinstance(raw_client_timestamp, str)
        else _timestamp_to_string(raw_client_timestamp)
    )
    envelope = SyncEnvelope(
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
        mutation_group_id=row.get("mutation_group_id"),
        mutation_step=_optional_int_from_storage(row.get("mutation_step")),
        mutation_step_count=_optional_int_from_storage(row.get("mutation_step_count")),
        mutation_plan_hash=row.get("mutation_plan_hash"),
        stable_key=row.get("stable_key"),
        created_at_client=created_at_client,
        received_at_server=_timestamp_to_string(
            row.get("received_at_server") or row.get("server_timestamp")
        ),
        client_timestamp=client_timestamp,
        server_timestamp=_timestamp_to_string(
            row.get("server_timestamp") or row.get("received_at_server")
        ),
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
    if isinstance(raw_created_at_client, str):
        object.__setattr__(envelope, "created_at_client", created_at_client)
        object.__setattr__(envelope, "client_timestamp", client_timestamp)
    return envelope


def _cursor_from_row(row: dict[str, Any]) -> SyncDeviceCursor:
    return SyncDeviceCursor(
        dataset_id=row["dataset_id"],
        device_id=row["device_id"],
        domain=row["domain"],
        last_pulled_sequence=int(row["last_pulled_sequence"]),
        adapter_version=int(row.get("adapter_version") or 1),
        max_delivered_sequence=int(row.get("max_delivered_sequence") or 0),
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
        owner_user_id=row["owner_user_id"],
        attachment_id=row["attachment_id"],
        domain=row["domain"],
        object_id=row["entity_id"],
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
        metadata=decode_json(row.get("metadata_json"), default={}),
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


def _attachment_revision_binding_from_row(
    row: dict[str, Any],
) -> SyncAttachmentRevisionBinding:
    return SyncAttachmentRevisionBinding(
        dataset_id=row["dataset_id"],
        attachment_id=row["attachment_id"],
        attachment_revision=int(row["attachment_revision"]),
        blob_hash=row["blob_hash"],
        size_bytes=int(row["size_bytes"]),
        establishing_server_cursor=int(row["establishing_server_cursor"]),
        availability_at_acceptance=row["availability_at_acceptance"],
        resolved_blob_id=row.get("resolved_blob_id"),
        retention_released_at=_timestamp_to_string(row.get("retention_released_at")),
        created_at=_timestamp_to_string(row.get("created_at")) or "",
    )


def _storage_namespace_from_row(row: dict[str, Any]) -> SyncDatasetStorageNamespace:
    return SyncDatasetStorageNamespace(
        dataset_id=row["dataset_id"],
        owner_user_id=row["owner_user_id"],
        storage_namespace_id=row["storage_namespace_id"],
        created_at=_timestamp_to_string(row.get("created_at")) or "",
    )


def _notes_attachment_source_map_from_row(
    row: dict[str, Any],
) -> SyncNotesAttachmentSourceMap:
    return SyncNotesAttachmentSourceMap(
        dataset_id=row["dataset_id"],
        bootstrap_id=row["bootstrap_id"],
        source_key_hash=row["source_key_hash"],
        note_id=row["note_id"],
        attachment_id=row["attachment_id"],
        created_at=_timestamp_to_string(row.get("created_at")) or "",
    )


def _notes_attachment_cleanup_candidate_from_row(
    row: dict[str, Any],
) -> SyncNotesAttachmentCleanupCandidate:
    return SyncNotesAttachmentCleanupCandidate(
        dataset_id=row["dataset_id"],
        bootstrap_id=row["bootstrap_id"],
        source_key_hash=row["source_key_hash"],
        attachment_id=row["attachment_id"],
        source_relative_path=row["source_relative_path"],
        source_path_hash=row["source_path_hash"],
        source_blob_hash=row["source_blob_hash"],
        source_size_bytes=int(row["source_size_bytes"]),
        source_modified_ns=int(row["source_modified_ns"]),
        created_at=_timestamp_to_string(row.get("created_at")) or "",
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
    fingerprint = {
        "dataset_id": envelope.dataset_id,
        "domain": envelope.domain,
        "object_id": envelope.object_id,
        "stable_key": envelope.stable_key,
        "operation": envelope.operation,
        "client_envelope_id": envelope.client_envelope_id,
        "device_id": envelope.device_id,
        "client_profile_id": envelope.client_profile_id,
        "client_sequence": envelope.client_sequence,
        "created_at_client": normalize_sync_timestamp(envelope.created_at_client),
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
        **_mutation_group_fingerprint(
            mutation_group_id=envelope.mutation_group_id,
            mutation_step=envelope.mutation_step,
            mutation_step_count=envelope.mutation_step_count,
            mutation_plan_hash=envelope.mutation_plan_hash,
        ),
    }
    if envelope.domain.startswith("personal_context."):
        fingerprint["payload_ciphertext"] = None
        fingerprint["payload"] = {}
        fingerprint["encryption_metadata"] = _without_personal_context_at_rest(
            fingerprint["encryption_metadata"]
        )
    return fingerprint


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
        "created_at_client": normalize_sync_timestamp(
            row.get("created_at_client") or row.get("client_timestamp")
        ),
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
        **_mutation_group_fingerprint(
            mutation_group_id=row.get("mutation_group_id"),
            mutation_step=_optional_int_from_storage(row.get("mutation_step")),
            mutation_step_count=_optional_int_from_storage(row.get("mutation_step_count")),
            mutation_plan_hash=row.get("mutation_plan_hash"),
        ),
    }
    if not ignore_client_envelope_id:
        fingerprint["client_envelope_id"] = row["client_envelope_id"]
    if str(row["domain"]).startswith("personal_context."):
        fingerprint["payload_ciphertext"] = None
        fingerprint["payload"] = {}
        fingerprint["encryption_metadata"] = _without_personal_context_at_rest(
            fingerprint["encryption_metadata"]
        )
    return fingerprint


def _without_personal_context_at_rest(value: Any) -> dict[str, Any]:
    metadata = dict(value) if isinstance(value, dict) else {}
    metadata.pop("personal_context_at_rest", None)
    return metadata


def _mutation_group_fingerprint(
    *,
    mutation_group_id: Any,
    mutation_step: Any,
    mutation_step_count: Any,
    mutation_plan_hash: Any,
) -> dict[str, Any]:
    if mutation_group_id is None:
        return {}
    return {
        "mutation_group_id": mutation_group_id,
        "mutation_step": mutation_step,
        "mutation_step_count": mutation_step_count,
        "mutation_plan_hash": mutation_plan_hash,
    }


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

        # lgtm[py/path-injection] default_path is built under the normalized per-user database root.
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
            if self.backend_type == BackendType.POSTGRESQL:
                self.execute(
                    "SELECT pg_advisory_xact_lock(?)",
                    (SYNC_VERSIONED_DEVICE_STATE_MIGRATION_LOCK_KEY,),
                    connection=conn,
                )
            current_heads_existed = self.backend.table_exists(
                "sync_current_heads", connection=conn
            )
            if self.backend.table_exists("sync_envelopes", connection=conn):
                self._ensure_envelope_m1_columns(connection=conn)
            if self.backend.table_exists("sync_key_records", connection=conn):
                self._ensure_key_record_user_id_column(connection=conn)
                self._ensure_key_record_rotation_columns(connection=conn)
            self._preflight_notes_attachment_bootstrap_tables(connection=conn)
            self.backend.create_tables(schema, connection=conn)
            self._ensure_device_lifecycle_columns(connection=conn)
            self._ensure_device_lifecycle_tables(connection=conn)
            self._ensure_background_sync_tables(connection=conn)
            self._ensure_envelope_m1_columns(connection=conn)
            self._ensure_sync_object_state_table(connection=conn)
            self._ensure_sync_current_heads_table(
                connection=conn, projection_exists=current_heads_existed
            )
            self._ensure_sync_materialization_locks_table(connection=conn)
            self._ensure_attachment_binding_tables(connection=conn)
            self._ensure_notes_attachment_bootstrap_tables(connection=conn)
            self._ensure_envelope_m1_indexes(connection=conn)
            self._ensure_conflict_indexes(connection=conn)
            self._ensure_key_record_user_id_column(connection=conn)
            self._ensure_key_record_rotation_columns(connection=conn)
            self._ensure_key_record_user_id_index(connection=conn)
        with self.backend.transaction() as conn:
            self._migrate_versioned_device_state(connection=conn)

    def execute(
        self,
        query: str,
        params: tuple[Any, ...] | dict[str, Any] | None = None,
        *,
        connection: Any | None = None,
    ) -> QueryResult:
        """Execute a parameterized SQL statement through the configured backend."""

        return self.backend.execute(query, params, connection=connection)

    @contextmanager
    def materialization_transaction(
        self,
        keys: Sequence[tuple[str, SyncDomain, str]],
        *,
        trusted_notes_task_bootstrap_id: str | None = None,
        trusted_notes_task_coordinator: bool = False,
    ) -> Iterator[Any]:
        """Serialize product projection and Sync bookkeeping by dataset."""

        ordered_keys = sorted(set(keys))
        if not ordered_keys:
            raise SyncStoreError("Sync materialization requires at least one object")
        try:
            with self.backend.transaction() as conn:
                domains_by_dataset: dict[str, set[SyncDomain]] = {}
                for dataset_id, domain, _object_id in ordered_keys:
                    domains_by_dataset.setdefault(dataset_id, set()).add(domain)
                for dataset_id, domains in sorted(domains_by_dataset.items()):
                    row = self._get_dataset_row_for_update(
                        dataset_id,
                        connection=conn,
                    )
                    if row is None:
                        raise SyncDatasetNotFoundError(
                            f"Sync dataset not found: {dataset_id}"
                        )
                    enrolled = _dataset_domains_from_row(row)
                    metadata = decode_json(row.get("metadata_json"), default={})
                    for domain in sorted(domains):
                        trusted_task = (
                            domain in {"notes.task", "notes.task_activity"}
                            and trusted_notes_task_bootstrap_id is not None
                        )
                        if trusted_task:
                            readiness_key = (
                                "notes_task_v1"
                                if domain == "notes.task"
                                else "notes_task_activity_v1"
                            )
                            readiness = metadata.get(readiness_key)
                            if (
                                not isinstance(readiness, Mapping)
                                or readiness.get("state") != "bootstrapping"
                                or metadata.get("task_activity_capture_enabled") is not True
                            ):
                                raise SyncStoreError("notes_task_sync_not_ready")
                        elif (
                            domain in {"notes.task", "notes.task_activity"}
                            and trusted_notes_task_coordinator
                        ):
                            if not notes_task_capture_is_active(metadata):
                                raise SyncStoreError("notes_task_sync_not_ready")
                        elif domain not in enrolled:
                            raise SyncInvalidDomainError(
                                "Sync domain is not enrolled for dataset "
                                f"{dataset_id}: {domain}"
                            )
                for dataset_id in sorted(domains_by_dataset):
                    self._lock_materialization_dataset(dataset_id, connection=conn)
                yield conn
        except Exception as exc:
            if _is_materialization_lock_error(exc):
                raise SyncMaterializationBusyError() from exc
            raise

    def _lock_materialization_dataset(
        self,
        dataset_id: str,
        *,
        connection: Any,
    ) -> None:
        """Acquire one reusable durable dataset projection lock in the caller's tx."""

        if self.backend_type == BackendType.POSTGRESQL:
            self.execute("SET LOCAL lock_timeout = '10s'", connection=connection)
        self.execute(
            """
            INSERT INTO sync_materialization_locks (
                dataset_id, domain, object_id, updated_at
            ) VALUES (?, ?, ?, ?)
            ON CONFLICT (dataset_id, domain, object_id)
            DO UPDATE SET updated_at = excluded.updated_at
            """,
            (dataset_id, "*", "*", utcnow_iso()),
            connection=connection,
        )
        suffix = " FOR UPDATE" if self.backend_type == BackendType.POSTGRESQL else ""
        row = _first(
            self.execute(
                """
                SELECT dataset_id FROM sync_materialization_locks
                 WHERE dataset_id = ? AND domain = ? AND object_id = ?
                """
                + suffix,  # nosec B608 - suffix is a backend-controlled SQL literal.
                (dataset_id, "*", "*"),
                connection=connection,
            )
        )
        if row is None:
            raise SyncStoreError("sync_materialization_lock_unavailable")

    def require_materialization_predecessors_applied(
        self,
        envelopes: Sequence[SyncEnvelope],
        *,
        connection: Any,
    ) -> None:
        """Prevent a projection unit from advancing past earlier unresolved work."""

        cursors_by_dataset: dict[str, list[int]] = {}
        for envelope in envelopes:
            if envelope.status != "accepted" or envelope.server_cursor is None:
                raise SyncStoreError("Sync materialization requires stored accepted envelopes")
            cursors_by_dataset.setdefault(envelope.dataset_id, []).append(
                envelope.server_cursor
            )
        for dataset_id, cursors in sorted(cursors_by_dataset.items()):
            predecessor = _first(
                self.execute(
                    """
                    SELECT server_sequence, apply_status
                      FROM sync_envelopes
                     WHERE dataset_id = ?
                       AND status = 'accepted'
                       AND server_sequence < ?
                       AND apply_status NOT IN ('applied', 'superseded')
                     ORDER BY server_sequence ASC
                     LIMIT 1
                    """,
                    (dataset_id, min(cursors)),
                    connection=connection,
                )
            )
            if predecessor is not None:
                raise SyncMaterializationPredecessorError(
                    apply_status=str(predecessor.get("apply_status") or "pending")
                )

    def require_conflict_resolution_predecessors_applied(
        self,
        conflict_id: str,
        *,
        dataset_id: str,
        resolved_by_device_id: str | None,
        resolution_action: str,
        resolution_notes: str | None,
        connection: Any,
    ) -> SyncEnvelope:
        """Evaluate readiness at the claimed conflict's logical cursor."""

        conflict_row, envelope_row = self._require_claimed_conflict_source(
            conflict_id,
            dataset_id=dataset_id,
            resolved_by_device_id=resolved_by_device_id,
            resolution_action=resolution_action,
            resolution_notes=resolution_notes,
            connection=connection,
        )
        source_cursor = int(conflict_row["server_sequence"])
        predecessor = _first(
            self.execute(
                """
                SELECT server_sequence, apply_status
                  FROM sync_envelopes
                 WHERE dataset_id = ?
                   AND status = 'accepted'
                   AND server_sequence < ?
                   AND apply_status NOT IN ('applied', 'superseded')
                 ORDER BY server_sequence ASC
                 LIMIT 1
                """,
                (dataset_id, source_cursor),
                connection=connection,
            )
        )
        if predecessor is not None:
            raise SyncMaterializationPredecessorError(
                apply_status=str(predecessor.get("apply_status") or "pending")
            )
        return _envelope_from_row(envelope_row)

    def terminalize_claimed_conflict_envelope(
        self,
        conflict_id: str,
        *,
        dataset_id: str,
        resolved_by_device_id: str | None,
        resolution_action: str,
        resolution_notes: str | None,
        apply_error_code: str,
        connection: Any,
    ) -> SyncEnvelope:
        """Mark exactly the claimed, unprojected conflict source superseded."""

        _conflict_row, envelope_row = self._require_claimed_conflict_source(
            conflict_id,
            dataset_id=dataset_id,
            resolved_by_device_id=resolved_by_device_id,
            resolution_action=resolution_action,
            resolution_notes=resolution_notes,
            connection=connection,
        )
        cursor = int(envelope_row["server_sequence"])
        result = self.execute(
            """
            UPDATE sync_envelopes
               SET apply_status = 'superseded',
                   apply_error_code = ?,
                   apply_error_message = NULL,
                   applied_at = NULL
             WHERE server_sequence = ?
               AND apply_status IN ('pending', 'failed', 'conflict')
            """,
            (apply_error_code, cursor),
            connection=connection,
        )
        if result.rowcount == 0:
            raise SyncStoreError("Sync conflict source could not be terminalized")
        self._repoint_current_head_from_unprojected_row(
            envelope_row,
            connection=connection,
        )
        updated = _first(
            self.execute(
                "SELECT * FROM sync_envelopes WHERE server_sequence = ?",
                (cursor,),
                connection=connection,
            )
        )
        if updated is None:
            raise SyncStoreError("Sync conflict source envelope was not found")
        return _envelope_from_row(updated)

    def stage_later_claimed_conflict_rebase_plan(
        self,
        conflict_id: str,
        *,
        dataset_id: str,
        resolved_by_device_id: str | None,
        resolution_action: str,
        resolution_notes: str | None,
        connection: Any,
    ) -> tuple[int, ...]:
        """Validate and freeze the bounded later-row rebase plan."""

        conflict_row, _source_row = self._require_claimed_conflict_source(
            conflict_id,
            dataset_id=dataset_id,
            resolved_by_device_id=resolved_by_device_id,
            resolution_action=resolution_action,
            resolution_notes=resolution_notes,
            connection=connection,
        )
        rows = self._validated_later_conflict_rebase_rows(
            dataset_id=dataset_id,
            source_cursor=int(conflict_row["server_sequence"]),
            connection=connection,
        )
        return tuple(int(row["server_sequence"]) for row in rows)

    def rebase_later_claimed_conflict_envelopes(
        self,
        conflict_id: str,
        *,
        dataset_id: str,
        resolved_by_device_id: str | None,
        resolution_action: str,
        resolution_notes: str | None,
        expected_server_cursors: Sequence[int] | None = None,
        connection: Any,
    ) -> list[SyncConflict]:
        """Convert bounded legacy work queued after a claimed source into conflicts."""

        conflict_row, _source_row = self._require_claimed_conflict_source(
            conflict_id,
            dataset_id=dataset_id,
            resolved_by_device_id=resolved_by_device_id,
            resolution_action=resolution_action,
            resolution_notes=resolution_notes,
            connection=connection,
        )
        source_cursor = int(conflict_row["server_sequence"])
        rows = self._validated_later_conflict_rebase_rows(
            dataset_id=dataset_id,
            source_cursor=source_cursor,
            connection=connection,
        )
        planned_cursors = tuple(int(row["server_sequence"]) for row in rows)
        if (
            expected_server_cursors is not None
            and planned_cursors != tuple(expected_server_cursors)
        ):
            raise SyncStoreError("sync_conflict_resolution_rebase_plan_changed")

        conflicts: list[SyncConflict] = []
        for row in rows:
            previous_apply_status = str(row.get("apply_status") or "pending")
            cursor = int(row["server_sequence"])
            self.execute(
                """
                UPDATE sync_envelopes
                   SET apply_status = 'conflict',
                       apply_error_code = ?,
                       apply_error_message = ?,
                       applied_at = NULL
                 WHERE server_sequence = ?
                   AND status = 'accepted'
                   AND apply_status NOT IN ('applied', 'superseded')
                """,
                (
                    SYNC_REBASE_REQUIRED_AFTER_CONFLICT_RESOLUTION,
                    "Queued change requires review after conflict resolution",
                    cursor,
                ),
                connection=connection,
            )
            conflicts.append(
                self._upsert_rebase_required_conflict(
                    row,
                    source_conflict_id=conflict_id,
                    source_cursor=source_cursor,
                    previous_apply_status=previous_apply_status,
                    connection=connection,
                )
            )
            self._repoint_current_head_from_unprojected_row(
                row,
                connection=connection,
            )
        return conflicts

    def _validated_later_conflict_rebase_rows(
        self,
        *,
        dataset_id: str,
        source_cursor: int,
        connection: Any,
    ) -> list[dict[str, Any]]:
        """Return the bounded plan after validating every existing conflict row."""

        rows = self.execute(
            """
            SELECT *
              FROM sync_envelopes
             WHERE dataset_id = ?
               AND status = 'accepted'
               AND server_sequence > ?
               AND apply_status NOT IN ('applied', 'superseded')
             ORDER BY server_sequence ASC
             LIMIT ?
            """,
            (dataset_id, source_cursor, SYNC_MUTATION_GROUP_MAX_SIZE + 1),
            connection=connection,
        ).rows
        if len(rows) > SYNC_MUTATION_GROUP_MAX_SIZE:
            raise SyncStoreError("sync_conflict_resolution_rebase_limit_exceeded")
        for row in rows:
            existing = self._get_conflict_for_rebase_envelope(
                row,
                connection=connection,
            )
            if existing is not None:
                self._require_compatible_rebase_conflict_record(existing, row)
        return rows

    def _get_conflict_for_rebase_envelope(
        self,
        envelope_row: Mapping[str, Any],
        *,
        connection: Any,
    ) -> dict[str, Any] | None:
        return _first(
            self.execute(
                """
                SELECT * FROM sync_conflicts
                 WHERE dataset_id = ?
                   AND local_envelope_id = ?
                   AND server_sequence = ?
                """,
                (
                    envelope_row["dataset_id"],
                    envelope_row["client_envelope_id"],
                    int(envelope_row["server_sequence"]),
                ),
                connection=connection,
            )
        )

    @staticmethod
    def _require_compatible_rebase_conflict_record(
        conflict_row: Mapping[str, Any],
        envelope_row: Mapping[str, Any],
    ) -> None:
        identity_matches = (
            conflict_row.get("dataset_id") == envelope_row.get("dataset_id")
            and conflict_row.get("local_envelope_id")
            == envelope_row.get("client_envelope_id")
            and conflict_row.get("server_sequence")
            == envelope_row.get("server_sequence")
            and conflict_row.get("domain") == envelope_row.get("domain")
            and conflict_row.get("entity_id") == envelope_row.get("entity_id")
        )
        if (
            not identity_matches
            or conflict_row.get("status") != "unresolved"
            or _conflict_row_has_resolution_claim(conflict_row)
        ):
            raise SyncStoreError(
                "sync_conflict_resolution_rebase_record_incompatible"
            )

    def _upsert_rebase_required_conflict(
        self,
        envelope_row: dict[str, Any],
        *,
        source_conflict_id: str,
        source_cursor: int,
        previous_apply_status: str,
        connection: Any,
    ) -> SyncConflict:
        cursor = int(envelope_row["server_sequence"])
        existing = self._get_conflict_for_rebase_envelope(
            envelope_row,
            connection=connection,
        )
        metadata: dict[str, Any] = {
            "source_conflict_id": source_conflict_id,
            "source_server_cursor": source_cursor,
            "previous_apply_status": previous_apply_status,
        }
        if existing is not None:
            self._require_compatible_rebase_conflict_record(existing, envelope_row)
            if existing.get("conflict_type") != (
                SYNC_REBASE_REQUIRED_AFTER_CONFLICT_RESOLUTION
            ):
                metadata["previous_conflict_type"] = existing.get("conflict_type")
                metadata["previous_conflict_metadata"] = decode_json(
                    existing.get("metadata_json"),
                    default={},
                )
            else:
                return _conflict_from_row(existing)
            self.execute(
                """
                UPDATE sync_conflicts
                   SET conflict_type = ?,
                       metadata_json = ?
                 WHERE conflict_id = ? AND status = 'unresolved'
                """,
                (
                    SYNC_REBASE_REQUIRED_AFTER_CONFLICT_RESOLUTION,
                    encode_json(metadata, default={}),
                    existing["conflict_id"],
                ),
                connection=connection,
            )
            updated = _first(
                self.execute(
                    "SELECT * FROM sync_conflicts WHERE conflict_id = ?",
                    (existing["conflict_id"],),
                    connection=connection,
                )
            )
            if updated is None:
                raise SyncStoreError("Sync rebase conflict could not be stored")
            return _conflict_from_row(updated)

        digest = hashlib.sha256(
            (
                f"{envelope_row['dataset_id']}\0"
                f"{envelope_row['client_envelope_id']}\0{cursor}"
            ).encode()
        ).hexdigest()
        return self.insert_conflict(
            SyncConflictCreate(
                conflict_id=f"conflict-rebase-{digest}",
                dataset_id=str(envelope_row["dataset_id"]),
                domain=str(envelope_row["domain"]),  # type: ignore[arg-type]
                entity_id=str(envelope_row["entity_id"]),
                conflict_type=SYNC_REBASE_REQUIRED_AFTER_CONFLICT_RESOLUTION,
                local_envelope_id=str(envelope_row["client_envelope_id"]),
                server_sequence=cursor,
                metadata=metadata,
            ),
            connection=connection,
        )

    def _repoint_current_head_from_unprojected_row(
        self,
        envelope_row: dict[str, Any],
        *,
        connection: Any,
    ) -> None:
        """Conditionally restore the latest applied head hidden by one row."""

        cursor = int(envelope_row["server_sequence"])
        head = _first(
            self.execute(
                """
                SELECT latest_server_cursor
                  FROM sync_current_heads
                 WHERE dataset_id = ? AND domain = ? AND object_id = ?
                """,
                (
                    envelope_row["dataset_id"],
                    envelope_row["domain"],
                    envelope_row["entity_id"],
                ),
                connection=connection,
            )
        )
        if head is None or int(head["latest_server_cursor"]) != cursor:
            return
        projected = _first(
            self.execute(
                """
                SELECT server_sequence
                  FROM sync_envelopes
                 WHERE dataset_id = ? AND domain = ? AND entity_id = ?
                   AND status = 'accepted' AND apply_status = 'applied'
                   AND server_sequence < ?
                 ORDER BY server_sequence DESC
                 LIMIT 1
                """,
                (
                    envelope_row["dataset_id"],
                    envelope_row["domain"],
                    envelope_row["entity_id"],
                    cursor,
                ),
                connection=connection,
            )
        )
        if projected is None:
            self.execute(
                """
                DELETE FROM sync_current_heads
                 WHERE dataset_id = ? AND domain = ? AND object_id = ?
                   AND latest_server_cursor = ?
                """,
                (
                    envelope_row["dataset_id"],
                    envelope_row["domain"],
                    envelope_row["entity_id"],
                    cursor,
                ),
                connection=connection,
            )
            return
        self.execute(
            """
            UPDATE sync_current_heads
               SET latest_server_cursor = ?
             WHERE dataset_id = ? AND domain = ? AND object_id = ?
               AND latest_server_cursor = ?
            """,
            (
                projected["server_sequence"],
                envelope_row["dataset_id"],
                envelope_row["domain"],
                envelope_row["entity_id"],
                cursor,
            ),
            connection=connection,
        )

    def _require_claimed_conflict_source(
        self,
        conflict_id: str,
        *,
        dataset_id: str,
        resolved_by_device_id: str | None,
        resolution_action: str,
        resolution_notes: str | None,
        connection: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        conflict_row = _first(
            self.execute(
                "SELECT * FROM sync_conflicts WHERE conflict_id = ?",
                (conflict_id,),
                connection=connection,
            )
        )
        if (
            conflict_row is None
            or conflict_row.get("dataset_id") != dataset_id
            or conflict_row.get("server_sequence") is None
            or conflict_row.get("local_envelope_id") is None
            or not _conflict_row_matches_resolution_claim(
                conflict_row,
                resolved_by_device_id=resolved_by_device_id,
                resolution_action=resolution_action,
                resolution_notes=resolution_notes,
            )
        ):
            raise SyncStoreError("Sync conflict resolution claim does not match its source")
        envelope_row = _first(
            self.execute(
                """
                SELECT * FROM sync_envelopes
                 WHERE dataset_id = ?
                   AND client_envelope_id = ?
                   AND server_sequence = ?
                """,
                (
                    dataset_id,
                    conflict_row["local_envelope_id"],
                    conflict_row["server_sequence"],
                ),
                connection=connection,
            )
        )
        if (
            envelope_row is None
            or envelope_row.get("domain") != conflict_row.get("domain")
            or envelope_row.get("entity_id") != conflict_row.get("entity_id")
            or (
                envelope_row.get("status") == "accepted"
                and envelope_row.get("apply_status") != "conflict"
            )
        ):
            raise SyncStoreError("Sync conflict source envelope is not unresolved")
        return conflict_row, envelope_row

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

    def _get_dataset_row_for_update(
        self,
        dataset_id: str,
        *,
        connection: Any,
    ) -> dict[str, Any] | None:
        suffix = " FOR UPDATE" if self.backend_type == BackendType.POSTGRESQL else ""
        return _first(
            self.execute(
                "SELECT * FROM sync_datasets WHERE dataset_id = ?" + suffix,  # nosec B608
                (dataset_id,),
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

    def _require_attachment_binding_dataset_owner(
        self,
        dataset_id: str,
        owner_user_id: str,
        *,
        connection: Any,
    ) -> dict[str, Any]:
        row = _first(
            self.execute(
                """
                SELECT * FROM sync_datasets
                 WHERE dataset_id = ? AND owner_user_id = ?
                """,
                (dataset_id, owner_user_id),
                connection=connection,
            )
        )
        if row is None:
            raise SyncDatasetNotFoundError(f"Sync dataset not found: {dataset_id}")
        return row

    def _require_dataset_owner_for_update(
        self,
        dataset_id: str,
        owner_user_id: str,
        *,
        connection: Any,
    ) -> dict[str, Any]:
        suffix = " FOR UPDATE" if self.backend_type == BackendType.POSTGRESQL else ""
        row = _first(
            self.execute(
                """
                SELECT * FROM sync_datasets
                 WHERE dataset_id = ? AND owner_user_id = ?
                """
                + suffix,  # nosec B608 - backend-controlled row lock suffix.
                (dataset_id, owner_user_id),
                connection=connection,
            )
        )
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

    def _require_dataset_domain_for_update(
        self,
        dataset_id: str,
        domain: SyncDomain,
        *,
        connection: Any,
    ) -> dict[str, Any]:
        row = self._get_dataset_row_for_update(dataset_id, connection=connection)
        if row is None:
            raise SyncDatasetNotFoundError(f"Sync dataset not found: {dataset_id}")
        if domain not in _dataset_domains_from_row(row):
            raise SyncInvalidDomainError(
                f"Sync domain is not enrolled for dataset {dataset_id}: {domain}"
            )
        return row

    def _require_notes_organization_write_ready(
        self,
        row: Mapping[str, Any],
        domain: SyncDomain,
        *,
        trusted_bootstrap_id: str | None = None,
    ) -> None:
        if domain == "notes.link":
            metadata = decode_json(row.get("metadata_json"), default={}).get(
                "notes_link_v1"
            )
            if not isinstance(metadata, Mapping):
                raise SyncStoreError("notes_link_sync_not_ready")
            if metadata.get("state") == "ready":
                return
            if (
                trusted_bootstrap_id is not None
                and metadata.get("state") == "initializing"
                and metadata.get("bootstrap_id") == trusted_bootstrap_id
            ):
                return
            raise SyncStoreError("notes_link_sync_not_ready")
        if domain not in NOTES_ORGANIZATION_DOMAINS:
            return
        if set(NOTES_ORGANIZATION_DOMAINS).difference(_dataset_domains_from_row(dict(row))):
            raise SyncStoreError("notes_organization_sync_domains_incomplete")
        metadata = decode_json(row.get("metadata_json"), default={}).get(
            "notes_organization_v1"
        )
        if not isinstance(metadata, Mapping):
            raise SyncStoreError("notes_organization_sync_not_ready")
        if metadata.get("state") == "ready":
            return
        if (
            trusted_bootstrap_id is not None
            and metadata.get("state") == "initializing"
            and metadata.get("bootstrap_id") == trusted_bootstrap_id
        ):
            return
        raise SyncStoreError("notes_organization_sync_not_ready")

    @staticmethod
    def _require_notes_task_bootstrap_write_ready(
        row: Mapping[str, Any],
        *,
        envelope: SyncEnvelope | SyncEnvelopeCreate,
        bootstrap_id: str,
    ) -> None:
        """Authorize only the private source-verified dormant task bootstrap."""

        metadata = decode_json(row.get("metadata_json"), default={})
        readiness_key = {
            "notes.task": "notes_task_v1",
            "notes.task_activity": "notes_task_activity_v1",
        }.get(envelope.domain)
        expected_source = {
            "notes.task": "notes-task-bootstrap",
            "notes.task_activity": "notes-task-activity-bootstrap",
        }.get(envelope.domain)
        readiness = metadata.get(readiness_key) if readiness_key is not None else None
        routing = envelope.routing_metadata
        if (
            not isinstance(readiness, Mapping)
            or readiness.get("state") != "bootstrapping"
            or metadata.get("task_activity_capture_enabled") is not True
            or routing.get("bootstrap_capture") is not True
            or routing.get("bootstrap_id") != bootstrap_id
            or routing.get("source") != expected_source
        ):
            raise SyncStoreError("notes_task_sync_not_ready")

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
            allowed_domains = set(M1_SYNC_DOMAINS).union(
                SOURCE_CACHE_SYNC_DOMAINS,
                MEDIA_SYNC_DOMAINS,
                NOTES_ORGANIZATION_DOMAINS,
                NOTES_LINK_DOMAINS,
                NOTES_TASK_SYNC_DOMAINS,
                PERSONAL_CONTEXT_SYNC_DOMAINS,
            )
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
        organization_domains = set(dataset.domains).intersection(NOTES_ORGANIZATION_DOMAINS)
        if organization_domains and organization_domains != set(NOTES_ORGANIZATION_DOMAINS):
            raise SyncInvalidDomainError("notes_organization_sync_domains_incomplete")
        if organization_domains:
            organization_metadata = dataset.metadata.get("notes_organization_v1")
            state = (
                organization_metadata.get("state")
                if isinstance(organization_metadata, Mapping)
                else None
            )
            if state not in {"initializing", "ready", "failed"}:
                raise SyncStoreError("notes_organization_sync_not_ready")
        personal_context_domains = set(dataset.domains).intersection(
            PERSONAL_CONTEXT_SYNC_DOMAINS
        )
        if personal_context_domains and personal_context_domains != set(
            PERSONAL_CONTEXT_SYNC_DOMAINS
        ):
            raise SyncInvalidDomainError("personal_context_sync_domains_incomplete")
        task_domains = set(dataset.domains).intersection(NOTES_TASK_SYNC_DOMAINS)
        if task_domains and task_domains != set(NOTES_TASK_SYNC_DOMAINS):
            raise SyncInvalidDomainError("notes_task_sync_domains_incomplete")
        if task_domains and not notes_task_sync_is_ready(
            domains=dataset.domains,
            metadata=dataset.metadata,
        ):
            raise SyncStoreError("notes_task_sync_not_ready")

    def _validate_envelope_contract(self, envelope: SyncEnvelopeCreate) -> None:
        if envelope.domain not in SYNC_V2_INTERNAL_OPERATIONS:
            raise SyncInvalidDomainError(f"Sync v2 M1 domain is not supported: {envelope.domain}")
        if envelope.operation not in SYNC_V2_INTERNAL_OPERATIONS[envelope.domain]:
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
        has_prebootstrap_product_base = bool(
            envelope.domain == "notes.task"
            and envelope.base_server_cursor is None
            and envelope.base_object_revision is not None
            and envelope.base_object_hash is not None
            and envelope.routing_metadata.get("product_transition_base") is True
        )
        if has_any_base and not has_all_base and not has_prebootstrap_product_base:
            raise SyncStoreError(
                "Sync v2 M1 base metadata must be supplied as a complete set"
            )
        if envelope.domain in _WHOLE_OBJECT_DOMAINS:
            if (
                envelope.operation == "tombstone"
                and not has_all_base
                and not has_prebootstrap_product_base
            ):
                raise SyncStoreError(
                    f"Sync v2 M1 {envelope.domain} tombstones require base metadata"
                )
            if (
                envelope.operation == "upsert"
                and envelope.object_revision is not None
                and envelope.object_revision > 1
                and not has_all_base
                and not has_prebootstrap_product_base
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
            required = (
                {"attachment_id", "blob_hash", "size_bytes"}
                if envelope.adapter_version == 2
                else _ATTACHMENT_REF_REQUIRED_PAYLOAD_KEYS
            )
            missing = required.difference(envelope.payload)
            if missing:
                raise SyncStoreError(
                    "Sync v2 M1 attachment.ref envelopes require payload metadata fields: "
                    + ", ".join(sorted(missing))
                )
            if envelope.adapter_version == 2 and (
                envelope.schema_version != 2
                or envelope.object_revision is None
                or envelope.object_revision < 1
            ):
                raise SyncStoreError(
                    "attachment.ref v2 envelopes require schema version 2 and a positive revision"
                )

    def upsert_device(
        self,
        device: SyncDeviceUpsert,
        *,
        capabilities_resolver: Callable[
            [SyncDevice | None], dict[str, object]
        ]
        | None = None,
    ) -> SyncDevice:
        now = utcnow_iso()
        with self.backend.transaction() as conn:
            lock_suffix = (
                " FOR UPDATE"
                if self.backend_type == BackendType.POSTGRESQL
                else ""
            )
            existing = _first(
                self.execute(
                    "SELECT * FROM sync_devices WHERE device_id = ?"
                    + lock_suffix,  # nosec B608
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
                capabilities = (
                    capabilities_resolver(_device_from_row(existing))
                    if capabilities_resolver is not None
                    else device.capabilities
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
                        encode_json(capabilities, default={}),
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
                capabilities = (
                    capabilities_resolver(None)
                    if capabilities_resolver is not None
                    else device.capabilities
                )
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
                        encode_json(capabilities, default={}),
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

    def enroll_dataset(
        self,
        dataset: SyncDatasetCreate,
        *,
        preserve_personal_context: bool = True,
    ) -> SyncDataset:
        self._validate_dataset_contract(dataset)
        now = utcnow_iso()
        with self.backend.transaction() as conn:
            lock_suffix = (
                " FOR UPDATE" if self.backend_type == BackendType.POSTGRESQL else ""
            )
            existing = _first(
                self.execute(
                    "SELECT * FROM sync_datasets WHERE dataset_id = ?"
                    + lock_suffix,  # nosec B608 - backend-controlled row lock suffix.
                    (dataset.dataset_id,),
                    connection=conn,
                )
            )
            if existing:
                if existing.get("owner_user_id") != dataset.owner_user_id:
                    raise SyncStoreError(
                        f"Sync dataset already belongs to another user: {dataset.dataset_id}"
                    )
                raw_existing_metadata = existing.get("metadata_json")
                if not isinstance(raw_existing_metadata, str):
                    raise SyncStoreError("sync_dataset_metadata_invalid")
                try:
                    existing_metadata = json.loads(raw_existing_metadata)
                except ValueError as exc:
                    raise SyncStoreError("sync_dataset_metadata_invalid") from exc
                if not isinstance(existing_metadata, dict):
                    raise SyncStoreError("sync_dataset_metadata_invalid")
                server_metadata_keys = (
                    NOTES_TASK_SERVER_METADATA_KEYS
                    | NOTES_MOODBOARD_STUDIO_SERVER_METADATA_KEYS
                )
                if preserve_personal_context and "personal_context" in existing_metadata:
                    server_metadata_keys = server_metadata_keys | {"personal_context"}
                metadata = {
                    key: value
                    for key, value in dataset.metadata.items()
                    if key not in server_metadata_keys
                }
                metadata.update(
                    {
                        key: existing_metadata[key]
                        for key in server_metadata_keys
                        if key in existing_metadata
                    }
                )
                protected_personal_context_domains = (
                    _dataset_domains_from_row(existing)
                    & set(PERSONAL_CONTEXT_SYNC_DOMAINS)
                    if preserve_personal_context
                    else set()
                )
                effective_domains = list(
                    dict.fromkeys(
                        [*dataset.domains, *sorted(protected_personal_context_domains)]
                    )
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
                        encode_json(effective_domains, default=[]),
                        encode_json(metadata, default={}),
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
                        encode_json(dataset.domains, default=[]),
                        encode_json(dataset.metadata, default={}),
                        now,
                        now,
                        dataset.archived_at,
                    ),
                    connection=conn,
                )
            for domain in effective_domains if existing else dataset.domains:
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

    def complete_personal_context_link_receipt(
        self,
        *,
        user_id: str,
        dataset_id: str,
        device_id: str,
        profile_id: str,
        integrity_key_id: str,
        purge_generation: int,
        bootstrap_cursor: str,
    ) -> None:
        """Lock and compare the opaque Personal Context binding before receipt CAS."""

        with self.backend.transaction() as connection:
            row = _first(
                self.execute(
                    """
                    SELECT * FROM sync_datasets
                     WHERE dataset_id = ? AND owner_user_id = ?
                    """
                    + (
                        " FOR UPDATE"
                        if self.backend_type == BackendType.POSTGRESQL
                        else ""
                    ),  # nosec B608 - backend-controlled lock suffix.
                    (dataset_id, user_id),
                    connection=connection,
                )
            )
            metadata = decode_json(
                row.get("metadata_json") if row is not None else None,
                default={},
            )
            binding = metadata.get("personal_context") if isinstance(metadata, dict) else None
            if not isinstance(binding, dict) or (
                binding.get("profile_id") != profile_id
                or binding.get("integrity_key_id") != integrity_key_id
                or binding.get("purge_generation") != purge_generation
            ):
                raise SyncStoreError("personal_context_link_binding_stale")
            self.execute(
                """INSERT INTO sync_personal_context_link_receipts
                   (user_id, dataset_id, device_id, profile_id, integrity_key_id, purge_generation, bootstrap_cursor)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(user_id, dataset_id, device_id) DO UPDATE SET
                     profile_id = excluded.profile_id,
                     integrity_key_id = excluded.integrity_key_id,
                     purge_generation = excluded.purge_generation,
                     bootstrap_cursor = excluded.bootstrap_cursor""",
                (
                    user_id,
                    dataset_id,
                    device_id,
                    profile_id,
                    integrity_key_id,
                    purge_generation,
                    bootstrap_cursor,
                ),
                connection=connection,
            )

    def has_personal_context_link_receipt(
        self,
        *,
        user_id: str,
        dataset_id: str,
        device_id: str,
        profile_id: str,
        integrity_key_id: str,
        purge_generation: int,
    ) -> bool:
        """Return whether a device has the exact current Personal Context receipt."""

        result = self.execute(
            """SELECT 1 FROM sync_personal_context_link_receipts
               WHERE user_id = ? AND dataset_id = ? AND device_id = ? AND profile_id = ?
                 AND integrity_key_id = ? AND purge_generation = ?""",
            (user_id, dataset_id, device_id, profile_id, integrity_key_id, purge_generation),
        )
        return bool(result.rows)

    def bind_personal_context_dataset(
        self,
        *,
        dataset_id: str,
        user_id: str,
        expected_profile_id: str | None,
        expected_authority_id: str | None,
        profile_id: str,
        authority_id: str,
        integrity_key_id: str,
        purge_generation: int,
    ) -> SyncDataset:
        """Merge a canonical Personal Context binding into the locked dataset row."""

        if (
            not profile_id
            or not authority_id
            or not integrity_key_id
            or purge_generation < 0
        ):
            raise SyncStoreError("personal_context_authority_mismatch")
        with self.backend.transaction() as connection:
            row = self._require_dataset_owner_for_update(
                dataset_id,
                user_id,
                connection=connection,
            )
            metadata = decode_json(row.get("metadata_json"), default={})
            if not isinstance(metadata, dict):
                raise SyncStoreError("personal_context_authority_mismatch")
            current_binding = metadata.get("personal_context")
            if current_binding is None:
                if expected_profile_id is not None or expected_authority_id is not None:
                    raise SyncStoreError("personal_context_authority_mismatch")
                link_state = "bootstrap_pending"
            elif not isinstance(current_binding, Mapping) or (
                current_binding.get("profile_id") != expected_profile_id
                or current_binding.get("authority_id") != expected_authority_id
            ):
                raise SyncStoreError("personal_context_authority_mismatch")
            else:
                link_state = current_binding.get("link_state")
                if link_state not in {"bootstrap_pending", "complete"}:
                    raise SyncStoreError("personal_context_authority_mismatch")
            merged_metadata = dict(metadata)
            merged_metadata["personal_context"] = {
                "profile_id": profile_id,
                "authority_id": authority_id,
                "integrity_key_id": integrity_key_id,
                "purge_generation": purge_generation,
                "link_state": link_state,
            }
            raw_domains = decode_json(row.get("domain_set_json"), default=[])
            if not isinstance(raw_domains, list):
                raise SyncStoreError("personal_context_authority_mismatch")
            domains = list(dict.fromkeys([*raw_domains, *PERSONAL_CONTEXT_SYNC_DOMAINS]))
            now = utcnow_iso()
            self.execute(
                """
                UPDATE sync_datasets
                   SET domain_set_json = ?, metadata_json = ?, updated_at = ?
                 WHERE dataset_id = ? AND owner_user_id = ?
                """,
                (
                    encode_json(domains, default=[]),
                    encode_json(merged_metadata, default={}),
                    now,
                    dataset_id,
                    user_id,
                ),
                connection=connection,
            )
            for domain in PERSONAL_CONTEXT_SYNC_DOMAINS:
                self._ensure_domain_state(
                    dataset_id=dataset_id,
                    domain=domain,
                    adapter_version=1,
                    server_sequence=0,
                    connection=connection,
                )
            updated = self._get_dataset_row(
                dataset_id,
                owner_user_id=user_id,
                connection=connection,
            )
            if updated is None:
                raise SyncDatasetNotFoundError(f"Sync dataset not found: {dataset_id}")
        return _dataset_from_row(updated)

    def get_dataset(
        self,
        dataset_id: str,
        *,
        owner_user_id: str | None = None,
        connection: Any | None = None,
    ) -> SyncDataset | None:
        row = self._get_dataset_row(
            dataset_id,
            owner_user_id=owner_user_id,
            connection=connection,
        )
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
        connection: Any | None = None,
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
        result = self.execute(sql, tuple(params), connection=connection)
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
        *,
        connection: Any | None = None,
    ) -> SyncDeviceDomainAck:
        """Record the highest accepted sequence a device has applied for a domain."""

        now = utcnow_iso()
        with self.backend.transaction(connection) as conn:
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
            cursor = _first(
                self.execute(
                    """
                    SELECT * FROM sync_device_adapter_cursors
                     WHERE dataset_id = ? AND device_id = ? AND domain = ?
                       AND adapter_version = ?
                    """,
                    (
                        acknowledgment.dataset_id,
                        acknowledgment.device_id,
                        acknowledgment.domain,
                        acknowledgment.adapter_version,
                    ),
                    connection=conn,
                )
            )
            delivered = int((cursor or {}).get("max_delivered_sequence") or 0)
            if acknowledgment.adapter_version == 1:
                legacy_cursor = _first(
                    self.execute(
                        """
                        SELECT * FROM sync_device_cursors
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
                legacy_delivered = int(
                    (legacy_cursor or {}).get("last_pulled_sequence") or 0
                )
                if cursor is None or legacy_delivered > int(
                    cursor.get("last_pulled_sequence") or 0
                ):
                    delivered = max(delivered, legacy_delivered)
            if acknowledgment.through_server_sequence > delivered:
                raise SyncStoreError(
                    "Sync domain acknowledgment exceeds the delivered watermark"
                )
            existing = _first(
                self.execute(
                    """
                    SELECT * FROM sync_device_adapter_domain_acks
                     WHERE dataset_id = ? AND device_id = ? AND domain = ?
                       AND adapter_version = ?
                    """,
                    (
                        acknowledgment.dataset_id,
                        acknowledgment.device_id,
                        acknowledgment.domain,
                        acknowledgment.adapter_version,
                    ),
                    connection=conn,
                )
            )
            sequence = max(
                acknowledgment.through_server_sequence,
                int((existing or {}).get("through_server_sequence") or 0),
            )
            applied_at = acknowledgment.applied_at
            idempotency_key = acknowledgment.idempotency_key
            if existing is not None and acknowledgment.through_server_sequence < int(
                existing["through_server_sequence"]
            ):
                applied_at = existing["applied_at"]
                idempotency_key = existing.get("idempotency_key")
            self.execute(
                """
                INSERT INTO sync_device_adapter_domain_acks (
                    dataset_id, device_id, domain, adapter_version,
                    through_server_sequence, applied_at, updated_at, idempotency_key
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (dataset_id, device_id, domain, adapter_version)
                DO UPDATE SET through_server_sequence = CASE
                                  WHEN excluded.through_server_sequence >
                                       sync_device_adapter_domain_acks.through_server_sequence
                                  THEN excluded.through_server_sequence
                                  ELSE sync_device_adapter_domain_acks.through_server_sequence END,
                              applied_at = CASE
                                  WHEN excluded.through_server_sequence >=
                                       sync_device_adapter_domain_acks.through_server_sequence
                                  THEN excluded.applied_at
                                  ELSE sync_device_adapter_domain_acks.applied_at END,
                              updated_at = excluded.updated_at,
                              idempotency_key = CASE
                                  WHEN excluded.through_server_sequence >=
                                       sync_device_adapter_domain_acks.through_server_sequence
                                  THEN excluded.idempotency_key
                                  ELSE sync_device_adapter_domain_acks.idempotency_key END
                """,
                (
                    acknowledgment.dataset_id,
                    acknowledgment.device_id,
                    acknowledgment.domain,
                    acknowledgment.adapter_version,
                    sequence,
                    applied_at,
                    now,
                    idempotency_key,
                ),
                connection=conn,
            )
            if acknowledgment.adapter_version == 1:
                legacy = _first(
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
                legacy_sequence = int(
                    (legacy or {}).get("through_server_sequence") or 0
                )
                if legacy is not None and legacy_sequence > sequence:
                    sequence = legacy_sequence
                    applied_at = legacy["applied_at"]
                    idempotency_key = legacy.get("idempotency_key")
                self.execute(
                    """
                    INSERT INTO sync_device_domain_acks (
                        dataset_id, device_id, domain, through_server_sequence,
                        applied_at, updated_at, idempotency_key
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (dataset_id, device_id, domain)
                    DO UPDATE SET through_server_sequence = CASE
                                      WHEN excluded.through_server_sequence >
                                           sync_device_domain_acks.through_server_sequence
                                      THEN excluded.through_server_sequence
                                      ELSE sync_device_domain_acks.through_server_sequence END,
                                  applied_at = CASE
                                      WHEN excluded.through_server_sequence >=
                                           sync_device_domain_acks.through_server_sequence
                                      THEN excluded.applied_at
                                      ELSE sync_device_domain_acks.applied_at END,
                                  updated_at = excluded.updated_at,
                                  idempotency_key = CASE
                                      WHEN excluded.through_server_sequence >=
                                           sync_device_domain_acks.through_server_sequence
                                      THEN excluded.idempotency_key
                                      ELSE sync_device_domain_acks.idempotency_key END
                    """,
                    (
                        acknowledgment.dataset_id,
                        acknowledgment.device_id,
                        acknowledgment.domain,
                        sequence,
                        applied_at,
                        now,
                        idempotency_key,
                    ),
                    connection=conn,
                )
                self.execute(
                    """
                    UPDATE sync_device_adapter_domain_acks
                       SET through_server_sequence = CASE
                               WHEN through_server_sequence < ?
                               THEN (? + 0) ELSE through_server_sequence END,
                           updated_at = ?
                     WHERE dataset_id = ? AND device_id = ? AND domain = ?
                       AND adapter_version = 1
                    """,
                    (
                        sequence,
                        sequence,
                        now,
                        acknowledgment.dataset_id,
                        acknowledgment.device_id,
                        acknowledgment.domain,
                    ),
                    connection=conn,
                )
            row = _first(
                self.execute(
                    """
                    SELECT * FROM sync_device_adapter_domain_acks
                     WHERE dataset_id = ? AND device_id = ? AND domain = ?
                       AND adapter_version = ?
                    """,
                    (
                        acknowledgment.dataset_id,
                        acknowledgment.device_id,
                        acknowledgment.domain,
                        acknowledgment.adapter_version,
                    ),
                    connection=conn,
                )
            )
        return _device_domain_ack_from_row(row)

    def get_device_domain_ack(
        self,
        dataset_id: str,
        device_id: str,
        domain: SyncDomain,
        *,
        adapter_version: int = 1,
        connection: Any | None = None,
    ) -> SyncDeviceDomainAck | None:
        with self.backend.transaction(connection) as conn:
            self._require_dataset_domain(dataset_id, domain, connection=conn)
            self._require_device_for_dataset(dataset_id, device_id, connection=conn)
            row = _first(
                self.execute(
                    """
                    SELECT * FROM sync_device_adapter_domain_acks
                     WHERE dataset_id = ? AND device_id = ? AND domain = ?
                       AND adapter_version = ?
                    """,
                    (dataset_id, device_id, domain, adapter_version),
                    connection=conn,
                )
            )
            if adapter_version == 1:
                legacy = _first(
                    self.execute(
                        """
                        SELECT * FROM sync_device_domain_acks
                         WHERE dataset_id = ? AND device_id = ? AND domain = ?
                        """,
                        (dataset_id, device_id, domain),
                        connection=conn,
                    )
                )
                if legacy is not None and (
                    row is None
                    or int(legacy["through_server_sequence"])
                    > int(row["through_server_sequence"])
                ):
                    row = {**legacy, "adapter_version": 1}
        return None if row is None else _device_domain_ack_from_row(row)

    def upsert_device_blob_ack(
        self,
        acknowledgment: SyncDeviceBlobAckCreate,
        *,
        connection: Any | None = None,
    ) -> SyncDeviceBlobAck:
        """Record a device-level blob verification acknowledgment."""

        now = utcnow_iso()
        with self.backend.transaction(connection) as conn:
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

    def upsert_device_blob_id_ack(
        self,
        acknowledgment: SyncDeviceBlobIdAckCreate,
        *,
        connection: Any | None = None,
    ) -> SyncDeviceBlobIdAck:
        """Record immutable, authorized v2 blob-ID verification evidence."""

        now = utcnow_iso()
        with self.backend.transaction(connection) as conn:
            self._require_device_for_dataset(
                acknowledgment.dataset_id,
                acknowledgment.device_id,
                connection=conn,
            )
            blob = _first(
                self.execute(
                    """
                    SELECT blob.blob_id, blob.payload_hash
                      FROM sync_blob_objects AS blob
                      JOIN sync_datasets AS dataset
                        ON dataset.dataset_id = blob.dataset_id
                     WHERE blob.dataset_id = ? AND blob.blob_id = ?
                       AND blob.status = 'available'
                       AND (
                            dataset.scope_type = 'workspace'
                            OR blob.owner_user_id = dataset.owner_user_id
                       )
                    """,
                    (acknowledgment.dataset_id, acknowledgment.blob_id),
                    connection=conn,
                )
            )
            if blob is None:
                raise SyncStoreError("sync_blob_id_ack_not_authorized")
            if blob["payload_hash"] != acknowledgment.payload_hash:
                raise SyncStoreError("sync_blob_id_ack_digest_mismatch")
            existing = _first(
                self.execute(
                    """
                    SELECT * FROM sync_device_blob_id_acks
                     WHERE dataset_id = ? AND device_id = ? AND blob_id = ?
                    """,
                    (
                        acknowledgment.dataset_id,
                        acknowledgment.device_id,
                        acknowledgment.blob_id,
                    ),
                    connection=conn,
                )
            )
            if existing is not None and existing["payload_hash"] != acknowledgment.payload_hash:
                raise SyncStoreError("sync_blob_id_ack_digest_immutable")
            self.execute(
                """
                INSERT INTO sync_device_blob_id_acks (
                    dataset_id, device_id, blob_id, payload_hash,
                    verified_at, updated_at, idempotency_key
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (dataset_id, device_id, blob_id)
                DO UPDATE SET verified_at = excluded.verified_at,
                              updated_at = excluded.updated_at,
                              idempotency_key = excluded.idempotency_key
                """,
                (
                    acknowledgment.dataset_id,
                    acknowledgment.device_id,
                    acknowledgment.blob_id,
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
                    SELECT * FROM sync_device_blob_id_acks
                     WHERE dataset_id = ? AND device_id = ? AND blob_id = ?
                    """,
                    (
                        acknowledgment.dataset_id,
                        acknowledgment.device_id,
                        acknowledgment.blob_id,
                    ),
                    connection=conn,
                )
            )
        return _device_blob_id_ack_from_row(row)

    def list_device_acknowledgments(
        self,
        dataset_id: str,
        device_id: str,
        *,
        connection: Any | None = None,
    ) -> SyncDeviceAcknowledgmentSummary:
        """Return all domain and blob acknowledgments for one device in a dataset."""

        with self.backend.transaction(connection) as conn:
            self._require_device_for_dataset(dataset_id, device_id, connection=conn)
            version_rows = self.execute(
                """
                SELECT * FROM sync_device_adapter_domain_acks
                 WHERE dataset_id = ? AND device_id = ?
                 ORDER BY domain ASC, adapter_version ASC
                """,
                (dataset_id, device_id),
                connection=conn,
            ).rows
            legacy_rows = self.execute(
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
            blob_id_rows = self.execute(
                """
                SELECT * FROM sync_device_blob_id_acks
                 WHERE dataset_id = ? AND device_id = ?
                 ORDER BY updated_at ASC, blob_id ASC
                """,
                (dataset_id, device_id),
                connection=conn,
            ).rows
        version_ack_by_key = {
            (ack.domain, ack.adapter_version): ack
            for ack in (_device_domain_ack_from_row(row) for row in version_rows)
        }
        for row in legacy_rows:
            legacy_ack = _device_domain_ack_from_row(row)
            key = (legacy_ack.domain, 1)
            current = version_ack_by_key.get(key)
            if (
                current is None
                or legacy_ack.through_server_sequence
                > current.through_server_sequence
            ):
                version_ack_by_key[key] = legacy_ack
        version_acks = [
            version_ack_by_key[key] for key in sorted(version_ack_by_key)
        ]
        domain_acks = {
            ack.domain: ack for ack in version_acks if ack.adapter_version == 1
        }
        return SyncDeviceAcknowledgmentSummary(
            dataset_id=dataset_id,
            device_id=device_id,
            domain_acks=domain_acks,
            blob_acks=[_device_blob_ack_from_row(row) for row in blob_rows],
            version_acks=version_acks,
            blob_id_acks=[_device_blob_id_ack_from_row(row) for row in blob_id_rows],
        )

    def acknowledge_device_state_atomic(
        self,
        dataset_id: str,
        device_id: str,
        *,
        domain_acks: Sequence[SyncDeviceDomainAckCreate] = (),
        blob_acks: Sequence[SyncDeviceBlobAckCreate] = (),
        blob_id_acks: Sequence[SyncDeviceBlobIdAckCreate] = (),
    ) -> SyncDeviceAcknowledgmentSummary:
        """Validate and persist one acknowledgment request as a single unit."""

        with self.backend.transaction() as conn:
            device_row = self._require_device_for_dataset(
                dataset_id,
                device_id,
                connection=conn,
            )
            if (
                device_row.get("revoked_at") is not None
                or str(device_row.get("status") or "active") != "active"
            ):
                raise SyncStoreError("Sync device was not found or is not accessible")
            for acknowledgment in domain_acks:
                self._require_ack_request_identity(
                    dataset_id,
                    device_id,
                    acknowledgment.dataset_id,
                    acknowledgment.device_id,
                )
                self.upsert_device_domain_ack(acknowledgment, connection=conn)
            for acknowledgment in blob_acks:
                self._require_ack_request_identity(
                    dataset_id,
                    device_id,
                    acknowledgment.dataset_id,
                    acknowledgment.device_id,
                )
                self.upsert_device_blob_ack(acknowledgment, connection=conn)
            for acknowledgment in blob_id_acks:
                self._require_ack_request_identity(
                    dataset_id,
                    device_id,
                    acknowledgment.dataset_id,
                    acknowledgment.device_id,
                )
                capabilities = decode_json(
                    device_row.get("capabilities_json"),
                    default={},
                )
                version_map = (
                    capabilities.get("supported_adapter_versions")
                    if isinstance(capabilities, Mapping)
                    else None
                )
                versions = (
                    version_map.get("attachment.ref")
                    if isinstance(version_map, Mapping)
                    else None
                )
                if not isinstance(versions, Sequence) or isinstance(
                    versions,
                    (str, bytes, bytearray),
                ) or not any(
                    isinstance(version, int)
                    and not isinstance(version, bool)
                    and version == 2
                    for version in versions
                ):
                    raise SyncStoreError("sync_blob_id_ack_adapter_v2_required")
                self.upsert_device_blob_id_ack(acknowledgment, connection=conn)
            return self.list_device_acknowledgments(
                dataset_id,
                device_id,
                connection=conn,
            )

    @staticmethod
    def _require_ack_request_identity(
        dataset_id: str,
        device_id: str,
        acknowledgment_dataset_id: str,
        acknowledgment_device_id: str,
    ) -> None:
        if (
            acknowledgment_dataset_id != dataset_id
            or acknowledgment_device_id != device_id
        ):
            raise SyncStoreError(
                "Sync acknowledgment device or dataset does not match request"
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
                           AND apply_status IN ('pending', 'failed')
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

    @staticmethod
    def _notes_task_readiness_record(
        metadata: Mapping[str, Any],
        readiness_key: str,
    ) -> NotesTaskReadinessRecord:
        if readiness_key not in metadata:
            return default_notes_task_readiness_record()
        result = parse_notes_task_readiness_record(
            metadata[readiness_key],
            readiness_key=readiness_key,
        )
        if result.record is None:
            raise SyncStoreError("notes_task_readiness_state_invalid")
        return result.record

    def transition_notes_task_domain_readiness(
        self,
        dataset_id: str,
        *,
        owner_user_id: str,
        readiness_key: str,
        expected_state: str,
        state: str,
        source_dataset_id: str,
        source_cursor: str | None,
        source_count: int,
        source_fingerprint: str | None,
        reason_code: str | None = None,
        task_activity_capture_enabled: bool | None = None,
        captured_source_rebase: bool = False,
    ) -> SyncDataset:
        """Atomically persist one bounded dormant task-domain readiness transition."""

        if readiness_key not in NOTES_TASK_READINESS_REASON_CODES_BY_KEY:
            raise SyncStoreError("notes_task_readiness_domain_invalid")
        if (
            expected_state not in NOTES_TASK_READINESS_STATES
            or state not in NOTES_TASK_READINESS_STATES
        ):
            raise SyncStoreError("notes_task_readiness_transition_invalid")
        if (
            not isinstance(source_dataset_id, str)
            or source_dataset_id != dataset_id
            or source_dataset_id == _NOTES_TASK_LOCAL_UNBOUND_DATASET_ID
        ):
            raise SyncStoreError("notes_task_readiness_source_scope_invalid")
        if task_activity_capture_enabled is not None and not isinstance(
            task_activity_capture_enabled, bool
        ):
            raise SyncStoreError("notes_task_readiness_capture_invalid")
        if not isinstance(captured_source_rebase, bool):
            raise SyncStoreError("notes_task_readiness_source_changed")
        with self.backend.transaction() as conn:
            row = self._require_dataset_owner_for_update(
                dataset_id,
                owner_user_id,
                connection=conn,
            )
            if row.get("scope_type") != "personal":
                raise SyncStoreError("Sync dataset was not found or is not accessible")
            raw_metadata = row.get("metadata_json")
            if not isinstance(raw_metadata, str) or not raw_metadata:
                raise SyncStoreError("notes_task_readiness_state_invalid")
            try:
                metadata = json.loads(raw_metadata)
            except ValueError as exc:
                raise SyncStoreError("notes_task_readiness_state_invalid") from exc
            if not isinstance(metadata, dict):
                raise SyncStoreError("notes_task_readiness_state_invalid")
            current = self._notes_task_readiness_record(metadata, readiness_key)
            if current.state != expected_state:
                raise SyncStoreError("notes_task_readiness_compare_and_set_failed")
            if state not in _NOTES_TASK_READINESS_TRANSITIONS[expected_state]:
                raise SyncStoreError("notes_task_readiness_transition_invalid")
            if (
                isinstance(source_count, int)
                and not isinstance(source_count, bool)
                and source_count < current.source_count
            ):
                raise SyncStoreError("notes_task_readiness_progress_regressed")

            resume_phase: str | None = None
            if state == "blocked":
                if expected_state == "blocked":
                    resume_phase = current.resume_phase
                elif expected_state == "verifying":
                    resume_phase = "verifying"
                else:
                    resume_phase = "bootstrapping"
            parsed = parse_notes_task_readiness_record(
                {
                    "state": state,
                    "source_cursor": source_cursor,
                    "source_count": source_count,
                    "source_fingerprint": source_fingerprint,
                    "reason_code": reason_code,
                    "resume_phase": resume_phase,
                },
                readiness_key=readiness_key,
            )
            if parsed.record is None:
                raise SyncStoreError(
                    parsed.error_code or "notes_task_readiness_state_invalid"
                )
            requested = parsed.record

            current_count = current.source_count
            current_cursor = current.source_cursor
            current_cursor_key = current.source_cursor_key
            current_fingerprint = current.source_fingerprint
            capture_active = notes_task_capture_is_active(metadata)
            permitted_source_rebase = (
                captured_source_rebase
                and capture_active
                and expected_state == "bootstrapping"
                and state in {"bootstrapping", "verifying"}
                and source_cursor == current_cursor
                and source_count == current_count
            )
            resetting_empty = state == "not_enrolled" and current_count == 0
            if state == "blocked" and any(
                (
                    source_cursor != current_cursor,
                    source_count != current_count,
                    source_fingerprint != current_fingerprint,
                )
            ):
                raise SyncStoreError("notes_task_readiness_source_changed")
            if (
                expected_state == "blocked"
                and state not in {"blocked", "not_enrolled"}
            ):
                if state != current.resume_phase:
                    raise SyncStoreError("notes_task_readiness_transition_invalid")
                if any(
                    (
                        source_cursor != current_cursor,
                        source_count != current_count,
                        source_fingerprint != current_fingerprint,
                    )
                ):
                    raise SyncStoreError("notes_task_readiness_source_changed")
            if (
                expected_state == "ready"
                and requested.as_metadata() != current.as_metadata()
            ):
                raise SyncStoreError("notes_task_readiness_source_changed")
            if not resetting_empty:
                if source_count < current_count:
                    raise SyncStoreError("notes_task_readiness_progress_regressed")
                requested_cursor_key = requested.source_cursor_key
                if current_cursor_key is not None:
                    if requested_cursor_key is None:
                        raise SyncStoreError(
                            "notes_task_readiness_progress_regressed"
                        )
                    if readiness_key == "notes_task_v1":
                        if not isinstance(current_cursor_key, UUID) or not isinstance(
                            requested_cursor_key, UUID
                        ):
                            raise SyncStoreError(
                                "notes_task_readiness_cursor_invalid"
                            )
                        cursor_regressed = (
                            requested_cursor_key.int < current_cursor_key.int
                        )
                    else:
                        if not isinstance(current_cursor_key, tuple) or not isinstance(
                            requested_cursor_key, tuple
                        ):
                            raise SyncStoreError(
                                "notes_task_readiness_cursor_invalid"
                            )
                        current_created_at, current_activity_id = current_cursor_key
                        requested_created_at, requested_activity_id = (
                            requested_cursor_key
                        )
                        cursor_regressed = (
                            requested_created_at < current_created_at
                            or (
                                requested_created_at == current_created_at
                                and requested_activity_id.int
                                < current_activity_id.int
                            )
                        )
                    if cursor_regressed:
                        raise SyncStoreError(
                            "notes_task_readiness_progress_regressed"
                        )
                cursor_advanced = requested.source_cursor_key != current_cursor_key
                count_advanced = source_count != current_count
                if cursor_advanced != count_advanced:
                    raise SyncStoreError("notes_task_readiness_progress_regressed")
                if cursor_advanced and source_fingerprint == current_fingerprint:
                    raise SyncStoreError("notes_task_readiness_source_changed")
                if current_fingerprint is not None and source_fingerprint is None:
                    raise SyncStoreError("notes_task_readiness_progress_regressed")
                if (
                    not cursor_advanced
                    and not count_advanced
                    and current_fingerprint is not None
                    and source_fingerprint != current_fingerprint
                    and not permitted_source_rebase
                ):
                    raise SyncStoreError("notes_task_readiness_source_changed")
            if state == "not_enrolled" and not resetting_empty:
                raise SyncStoreError("notes_task_readiness_disable_forbidden")

            raw_capture = metadata.get("task_activity_capture_enabled", False)
            if not isinstance(raw_capture, bool):
                raise SyncStoreError("notes_task_readiness_state_invalid")
            capture_enabled = (
                raw_capture
                if task_activity_capture_enabled is None
                else task_activity_capture_enabled
            )
            metadata[readiness_key] = requested.as_metadata()
            task = self._notes_task_readiness_record(metadata, "notes_task_v1")
            activity = self._notes_task_readiness_record(
                metadata,
                "notes_task_activity_v1",
            )
            readiness_records = (task, activity)
            if capture_enabled and any(
                item.state == "not_enrolled" for item in readiness_records
            ):
                raise SyncStoreError("notes_task_readiness_capture_incomplete")
            if not capture_enabled and any(
                item.state in {"bootstrapping", "verifying", "ready"}
                for item in readiness_records
            ):
                raise SyncStoreError("notes_task_readiness_capture_required")
            if raw_capture and not capture_enabled and any(
                item.source_count != 0 or item.state == "ready"
                for item in readiness_records
            ):
                raise SyncStoreError("notes_task_readiness_capture_disable_forbidden")
            metadata["task_activity_capture_enabled"] = capture_enabled

            self.execute(
                "UPDATE sync_datasets SET metadata_json = ?, updated_at = ? "
                "WHERE dataset_id = ? AND owner_user_id = ?",
                (
                    encode_json(metadata, default={}),
                    utcnow_iso(),
                    dataset_id,
                    owner_user_id,
                ),
                connection=conn,
            )
            updated = self._get_dataset_row(dataset_id, connection=conn)
            if updated is None:
                raise SyncStoreError("Sync dataset readiness transition was not persisted")
            return _dataset_from_row(updated)

    @staticmethod
    def _notes_moodboard_studio_readiness_record(
        metadata: Mapping[str, Any],
        readiness_key: str,
    ) -> NotesMoodboardStudioReadinessRecord:
        """Parse one readiness record, defaulting absent domains to unenrolled.

        Args:
            metadata: Dataset metadata containing readiness records.
            readiness_key: Moodboard or Studio readiness domain key.

        Returns:
            The validated readiness record.

        Raises:
            SyncStoreError: If an existing record is malformed.
        """
        if readiness_key not in metadata:
            return default_notes_moodboard_studio_readiness_record()
        result = parse_notes_moodboard_studio_readiness_record(
            metadata[readiness_key],
            readiness_key=readiness_key,
        )
        if result.record is None:
            raise SyncStoreError("notes_moodboard_studio_readiness_state_invalid")
        return result.record

    def _transition_notes_moodboard_studio_readiness_records(
        self,
        dataset_id: str,
        *,
        owner_user_id: str,
        source_dataset_id: str,
        records: Sequence[
            tuple[str, str, str, str | None, int, str | None, str | None]
        ],
        moodboard_capture_enabled: bool | None = None,
        studio_document_capture_enabled: bool | None = None,
    ) -> SyncDataset:
        """Apply one atomic compare-and-set transition across readiness records.

        Capture remains disabled throughout bootstrap readiness. The method
        locks the owned dataset row, validates its personal Chatbook scope, and
        commits all requested domain transitions together.

        Args:
            dataset_id: Target Sync dataset ID.
            owner_user_id: Authenticated dataset owner.
            source_dataset_id: Source scope, which must equal ``dataset_id``.
            records: Domain transition tuples containing expected and requested
                state plus source progress evidence.
            moodboard_capture_enabled: Optional capture guard; only ``False``
                or ``None`` is accepted.
            studio_document_capture_enabled: Optional capture guard; only
                ``False`` or ``None`` is accepted.

        Returns:
            The updated dataset after the transaction commits.

        Raises:
            SyncStoreError: If ownership, scope, metadata, capture state,
                compare-and-set state, or source progress is invalid.
        """
        if (
            not isinstance(source_dataset_id, str)
            or source_dataset_id != dataset_id
            or source_dataset_id == _NOTES_TASK_LOCAL_UNBOUND_DATASET_ID
        ):
            raise SyncStoreError(
                "notes_moodboard_studio_readiness_source_scope_invalid"
            )
        if moodboard_capture_enabled is True or studio_document_capture_enabled is True:
            raise SyncStoreError(
                "notes_moodboard_studio_readiness_capture_forbidden"
            )
        if moodboard_capture_enabled not in {None, False}:
            raise SyncStoreError("notes_moodboard_studio_readiness_capture_invalid")
        if studio_document_capture_enabled not in {None, False}:
            raise SyncStoreError("notes_moodboard_studio_readiness_capture_invalid")
        for readiness_key, expected_state, state, *_ in records:
            if readiness_key not in NOTES_MOODBOARD_STUDIO_READINESS_REASON_CODES_BY_KEY:
                raise SyncStoreError("notes_moodboard_studio_readiness_domain_invalid")
            if (
                expected_state not in NOTES_MOODBOARD_STUDIO_READINESS_STATES
                or state not in NOTES_MOODBOARD_STUDIO_READINESS_STATES
            ):
                raise SyncStoreError(
                    "notes_moodboard_studio_readiness_transition_invalid"
                )

        with self.backend.transaction() as conn:
            row = self._require_dataset_owner_for_update(
                dataset_id,
                owner_user_id,
                connection=conn,
            )
            if (
                row.get("scope_type") != "personal"
                or row.get("encryption_policy") != DEFAULT_M1_ENCRYPTION_POLICY
            ):
                raise SyncStoreError(
                    "notes_moodboard_studio_readiness_source_scope_invalid"
                )
            raw_metadata = row.get("metadata_json")
            if not isinstance(raw_metadata, str) or not raw_metadata:
                raise SyncStoreError("notes_moodboard_studio_readiness_state_invalid")
            try:
                metadata = json.loads(raw_metadata)
            except ValueError as exc:
                raise SyncStoreError(
                    "notes_moodboard_studio_readiness_state_invalid"
                ) from exc
            if not isinstance(metadata, dict):
                raise SyncStoreError("notes_moodboard_studio_readiness_state_invalid")
            if (
                metadata.get("default_personal") is not True
                or metadata.get("client_family") != "chatbook"
            ):
                raise SyncStoreError(
                    "notes_moodboard_studio_readiness_source_scope_invalid"
                )
            for capture_key in (
                "moodboard_capture_enabled",
                "studio_document_capture_enabled",
            ):
                raw_capture = metadata.get(capture_key, False)
                if not isinstance(raw_capture, bool):
                    raise SyncStoreError(
                        "notes_moodboard_studio_readiness_state_invalid"
                    )
                if raw_capture:
                    raise SyncStoreError(
                        "notes_moodboard_studio_readiness_capture_forbidden"
                    )

            for (
                readiness_key,
                expected_state,
                state,
                source_cursor,
                source_count,
                source_fingerprint,
                reason_code,
            ) in records:
                current = self._notes_moodboard_studio_readiness_record(
                    metadata,
                    readiness_key,
                )
                requested = self._build_notes_moodboard_studio_readiness_record(
                    readiness_key=readiness_key,
                    current=current,
                    expected_state=expected_state,
                    state=state,
                    source_cursor=source_cursor,
                    source_count=source_count,
                    source_fingerprint=source_fingerprint,
                    reason_code=reason_code,
                )
                metadata[readiness_key] = requested.as_metadata()

            metadata["moodboard_capture_enabled"] = False
            metadata["studio_document_capture_enabled"] = False
            self.execute(
                "UPDATE sync_datasets SET metadata_json = ?, updated_at = ? "
                "WHERE dataset_id = ? AND owner_user_id = ?",
                (
                    encode_json(metadata, default={}),
                    utcnow_iso(),
                    dataset_id,
                    owner_user_id,
                ),
                connection=conn,
            )
            updated = self._get_dataset_row(dataset_id, connection=conn)
            if updated is None:
                raise SyncStoreError("Sync dataset readiness transition was not persisted")
            return _dataset_from_row(updated)

    def _build_notes_moodboard_studio_readiness_record(
        self,
        *,
        readiness_key: str,
        current: NotesMoodboardStudioReadinessRecord,
        expected_state: str,
        state: str,
        source_cursor: str | None,
        source_count: int,
        source_fingerprint: str | None,
        reason_code: str | None,
    ) -> NotesMoodboardStudioReadinessRecord:
        """Build and validate one readiness compare-and-set transition.

        Args:
            readiness_key: Domain-specific readiness metadata key.
            current: Current validated readiness record.
            expected_state: State the caller expects to replace.
            state: Requested next state.
            source_cursor: Requested canonical bootstrap cursor.
            source_count: Requested processed-object count.
            source_fingerprint: Requested source snapshot fingerprint.
            reason_code: Optional domain-specific blocked reason.

        Returns:
            The validated requested readiness record.

        Raises:
            SyncStoreError: If the compare-and-set fails, the transition is
                illegal, or source progress changes or regresses.
        """
        if current.state != expected_state:
            raise SyncStoreError(
                "notes_moodboard_studio_readiness_compare_and_set_failed"
            )
        if state not in _NOTES_TASK_READINESS_TRANSITIONS[expected_state]:
            raise SyncStoreError(
                "notes_moodboard_studio_readiness_transition_invalid"
            )
        if (
            isinstance(source_count, int)
            and not isinstance(source_count, bool)
            and source_count < current.source_count
        ):
            raise SyncStoreError(
                "notes_moodboard_studio_readiness_progress_regressed"
            )

        resume_phase: str | None = None
        if state == "blocked":
            if expected_state == "blocked":
                resume_phase = current.resume_phase
            elif expected_state == "verifying":
                resume_phase = "verifying"
            else:
                resume_phase = "bootstrapping"
        parsed = parse_notes_moodboard_studio_readiness_record(
            {
                "state": state,
                "source_cursor": source_cursor,
                "source_count": source_count,
                "source_fingerprint": source_fingerprint,
                "reason_code": reason_code,
                "resume_phase": resume_phase,
            },
            readiness_key=readiness_key,
        )
        if parsed.record is None:
            raise SyncStoreError(
                parsed.error_code
                or "notes_moodboard_studio_readiness_state_invalid"
            )
        requested = parsed.record

        current_fingerprint = current.source_fingerprint
        resetting_empty = state == "not_enrolled" and current.source_count == 0
        if state == "blocked" and any(
            (
                source_cursor != current.source_cursor,
                source_count != current.source_count,
                source_fingerprint != current_fingerprint,
            )
        ):
            raise SyncStoreError("notes_moodboard_studio_readiness_source_changed")
        if expected_state == "blocked" and state not in {"blocked", "not_enrolled"}:
            if state != current.resume_phase:
                raise SyncStoreError(
                    "notes_moodboard_studio_readiness_transition_invalid"
                )
            if any(
                (
                    source_cursor != current.source_cursor,
                    source_count != current.source_count,
                    source_fingerprint != current_fingerprint,
                )
            ):
                raise SyncStoreError("notes_moodboard_studio_readiness_source_changed")
        if expected_state == "ready" and requested.as_metadata() != current.as_metadata():
            raise SyncStoreError("notes_moodboard_studio_readiness_source_changed")
        if not resetting_empty:
            self._validate_notes_moodboard_studio_progress(
                readiness_key=readiness_key,
                current=current,
                requested=requested,
            )
        if state == "not_enrolled" and not resetting_empty:
            raise SyncStoreError(
                "notes_moodboard_studio_readiness_disable_forbidden"
            )
        return requested

    @staticmethod
    def _validate_notes_moodboard_studio_progress(
        *,
        readiness_key: str,
        current: NotesMoodboardStudioReadinessRecord,
        requested: NotesMoodboardStudioReadinessRecord,
    ) -> None:
        """Validate monotonic cursor, count, and fingerprint progress.

        Args:
            readiness_key: Domain-specific readiness metadata key.
            current: Current validated readiness record.
            requested: Requested validated readiness record.

        Raises:
            SyncStoreError: If progress regresses or source identity changes
                without a corresponding cursor and count advance.
        """
        if requested.source_count < current.source_count:
            raise SyncStoreError(
                "notes_moodboard_studio_readiness_progress_regressed"
            )
        if current.source_cursor_key is not None:
            if requested.source_cursor_key is None:
                raise SyncStoreError(
                    "notes_moodboard_studio_readiness_progress_regressed"
                )
            if _moodboard_studio_cursor_regressed(
                readiness_key,
                current.source_cursor_key,
                requested.source_cursor_key,
            ):
                raise SyncStoreError(
                    "notes_moodboard_studio_readiness_progress_regressed"
                )
        cursor_advanced = requested.source_cursor_key != current.source_cursor_key
        count_advanced = requested.source_count != current.source_count
        if cursor_advanced != count_advanced:
            raise SyncStoreError(
                "notes_moodboard_studio_readiness_progress_regressed"
            )
        if (
            cursor_advanced
            and requested.source_fingerprint == current.source_fingerprint
        ):
            raise SyncStoreError("notes_moodboard_studio_readiness_source_changed")
        if current.source_fingerprint is not None and requested.source_fingerprint is None:
            raise SyncStoreError(
                "notes_moodboard_studio_readiness_progress_regressed"
            )
        if (
            not cursor_advanced
            and not count_advanced
            and current.source_fingerprint is not None
            and requested.source_fingerprint != current.source_fingerprint
        ):
            raise SyncStoreError("notes_moodboard_studio_readiness_source_changed")

    def transition_notes_moodboard_graph_readiness(
        self,
        dataset_id: str,
        *,
        owner_user_id: str,
        expected_state: str,
        state: str,
        source_dataset_id: str,
        moodboard_source_cursor: str | None,
        moodboard_source_count: int,
        moodboard_source_fingerprint: str | None,
        placement_source_cursor: str | None,
        placement_source_count: int,
        placement_source_fingerprint: str | None,
        moodboard_reason_code: str | None = None,
        placement_reason_code: str | None = None,
        moodboard_capture_enabled: bool | None = None,
    ) -> SyncDataset:
        """Persist the coupled dormant moodboard and placement readiness records."""

        return self._transition_notes_moodboard_studio_readiness_records(
            dataset_id,
            owner_user_id=owner_user_id,
            source_dataset_id=source_dataset_id,
            records=[
                (
                    "notes_moodboard_v1",
                    expected_state,
                    state,
                    moodboard_source_cursor,
                    moodboard_source_count,
                    moodboard_source_fingerprint,
                    moodboard_reason_code,
                ),
                (
                    "notes_moodboard_note_v1",
                    expected_state,
                    state,
                    placement_source_cursor,
                    placement_source_count,
                    placement_source_fingerprint,
                    placement_reason_code,
                ),
            ],
            moodboard_capture_enabled=moodboard_capture_enabled,
        )

    def transition_notes_studio_document_readiness(
        self,
        dataset_id: str,
        *,
        owner_user_id: str,
        expected_state: str,
        state: str,
        source_dataset_id: str,
        source_cursor: str | None,
        source_count: int,
        source_fingerprint: str | None,
        reason_code: str | None = None,
        studio_document_capture_enabled: bool | None = None,
    ) -> SyncDataset:
        """Persist the independent dormant Studio readiness record."""

        return self._transition_notes_moodboard_studio_readiness_records(
            dataset_id,
            owner_user_id=owner_user_id,
            source_dataset_id=source_dataset_id,
            records=[
                (
                    "notes_studio_document_v1",
                    expected_state,
                    state,
                    source_cursor,
                    source_count,
                    source_fingerprint,
                    reason_code,
                )
            ],
            studio_document_capture_enabled=studio_document_capture_enabled,
        )

    def begin_notes_task_activation(
        self,
        dataset_id: str,
        *,
        owner_user_id: str,
    ) -> SyncDataset:
        """Enable task and activity capture together before either source scan."""

        with self.backend.transaction() as conn:
            row = self._require_dataset_owner_for_update(
                dataset_id,
                owner_user_id,
                connection=conn,
            )
            if row.get("scope_type") != "personal":
                raise SyncStoreError("Sync dataset was not found or is not accessible")
            metadata = decode_json(row.get("metadata_json"), default=None)
            if not isinstance(metadata, dict):
                raise SyncStoreError("notes_task_readiness_state_invalid")
            task = self._notes_task_readiness_record(metadata, "notes_task_v1")
            activity = self._notes_task_readiness_record(
                metadata,
                "notes_task_activity_v1",
            )
            if task.state == activity.state == "not_enrolled":
                enrolling = {
                    "state": "enrolling",
                    "source_cursor": None,
                    "source_count": 0,
                    "source_fingerprint": None,
                    "reason_code": None,
                    "resume_phase": None,
                }
                metadata["notes_task_v1"] = dict(enrolling)
                metadata["notes_task_activity_v1"] = dict(enrolling)
                metadata["task_activity_capture_enabled"] = True
            elif (
                task.state == "enrolling"
                and activity.state == "not_enrolled"
                and metadata.get("task_activity_capture_enabled") is not True
            ):
                metadata["notes_task_activity_v1"] = {
                    "state": "enrolling",
                    "source_cursor": None,
                    "source_count": 0,
                    "source_fingerprint": None,
                    "reason_code": None,
                    "resume_phase": None,
                }
                metadata["task_activity_capture_enabled"] = True
            elif (
                task.state == "not_enrolled"
                or activity.state == "not_enrolled"
                or metadata.get("task_activity_capture_enabled") is not True
            ):
                raise SyncStoreError("notes_task_readiness_state_invalid")
            else:
                return _dataset_from_row(row)

            self.execute(
                "UPDATE sync_datasets SET metadata_json = ?, updated_at = ? "
                "WHERE dataset_id = ? AND owner_user_id = ?",
                (
                    encode_json(metadata, default={}),
                    utcnow_iso(),
                    dataset_id,
                    owner_user_id,
                ),
                connection=conn,
            )
            updated = self._get_dataset_row(dataset_id, connection=conn)
            if updated is None:
                raise SyncStoreError("notes_task_activation_not_persisted")
            return _dataset_from_row(updated)

    def activate_notes_task_domains(
        self,
        dataset_id: str,
        *,
        owner_user_id: str,
    ) -> SyncDataset:
        """Publish both task domains in one transaction after coupled readiness."""

        with self.backend.transaction() as conn:
            row = self._require_dataset_owner_for_update(
                dataset_id,
                owner_user_id,
                connection=conn,
            )
            if row.get("scope_type") != "personal":
                raise SyncStoreError("Sync dataset was not found or is not accessible")
            metadata = decode_json(row.get("metadata_json"), default=None)
            domains = list(decode_json(row.get("domain_set_json"), default=[]))
            if not isinstance(metadata, dict) or not notes_task_sync_is_ready(
                domains=[*domains, *NOTES_TASK_SYNC_DOMAINS],
                metadata=metadata,
            ):
                raise SyncStoreError("notes_task_sync_not_ready")
            for domain in NOTES_TASK_SYNC_DOMAINS:
                if domain not in domains:
                    domains.append(domain)
            self.execute(
                "UPDATE sync_datasets SET domain_set_json = ?, updated_at = ? "
                "WHERE dataset_id = ? AND owner_user_id = ?",
                (
                    encode_json(domains, default=[]),
                    utcnow_iso(),
                    dataset_id,
                    owner_user_id,
                ),
                connection=conn,
            )
            for domain in NOTES_TASK_SYNC_DOMAINS:
                self._ensure_domain_state(
                    dataset_id=dataset_id,
                    domain=domain,
                    adapter_version=1,
                    server_sequence=0,
                    connection=conn,
                )
            updated = self._get_dataset_row(dataset_id, connection=conn)
            if updated is None:
                raise SyncStoreError("notes_task_activation_not_persisted")
            return _dataset_from_row(updated)

    def begin_notes_organization_bootstrap(
        self,
        dataset_id: str,
        *,
        owner_user_id: str,
        bootstrap_id: str,
    ) -> SyncDataset:
        """Atomically enroll the complete organization group in initializing state."""

        if not bootstrap_id.strip():
            raise SyncStoreError("Notes organization bootstrap ID is required")
        with self.backend.transaction() as conn:
            row = self._get_dataset_row_for_update(dataset_id, connection=conn)
            if (
                row is None
                or row.get("owner_user_id") != owner_user_id
                or row.get("scope_type") != "personal"
            ):
                raise SyncStoreError("Sync dataset was not found or is not accessible")
            metadata = decode_json(row.get("metadata_json"), default={})
            current = metadata.get("notes_organization_v1")
            if isinstance(current, Mapping) and current.get("state") in {
                "initializing",
                "ready",
            }:
                return _dataset_from_row(row)

            enrolled = list(decode_json(row.get("domain_set_json"), default=[]))
            for domain in NOTES_ORGANIZATION_DOMAINS:
                if domain not in enrolled:
                    enrolled.append(domain)
            metadata["notes_organization_v1"] = {
                "bootstrap_id": bootstrap_id,
                "state": "initializing",
                "captured_count": 0,
                "expected_count": 0,
                "error_code": None,
            }
            now = utcnow_iso()
            self.execute(
                "UPDATE sync_datasets SET domain_set_json = ?, metadata_json = ?, "
                "updated_at = ? WHERE dataset_id = ?",
                (
                    encode_json(enrolled, default=[]),
                    encode_json(metadata, default={}),
                    now,
                    dataset_id,
                ),
                connection=conn,
            )
            for domain in NOTES_ORGANIZATION_DOMAINS:
                self._ensure_domain_state(
                    dataset_id=dataset_id,
                    domain=domain,
                    adapter_version=1,
                    server_sequence=0,
                    connection=conn,
                )
            updated = self._get_dataset_row(dataset_id, connection=conn)
            if updated is None:
                raise SyncStoreError("Sync dataset bootstrap update was not persisted")
            return _dataset_from_row(updated)

    def transition_notes_organization_bootstrap(
        self,
        dataset_id: str,
        *,
        bootstrap_id: str,
        expected_state: str,
        state: str,
        captured_count: int,
        expected_count: int,
        error_code: str | None = None,
        ready_verifier: Callable[[], bool] | None = None,
    ) -> SyncDataset:
        """Compare-and-set one durable Notes organization bootstrap transition."""

        if expected_state not in {"initializing", "ready", "failed"} or state not in {
            "initializing",
            "ready",
            "failed",
        }:
            raise SyncStoreError("Notes organization bootstrap state is invalid")
        if captured_count < 0 or expected_count < 0:
            raise SyncStoreError("Notes organization bootstrap counts are invalid")
        with self.backend.transaction() as conn:
            row = self._get_dataset_row_for_update(dataset_id, connection=conn)
            if row is None:
                raise SyncStoreError("Sync dataset was not found or is not accessible")
            metadata = decode_json(row.get("metadata_json"), default={})
            current = metadata.get("notes_organization_v1")
            if not isinstance(current, Mapping) or (
                current.get("bootstrap_id") != bootstrap_id
                or current.get("state") != expected_state
            ):
                raise SyncStoreError("notes_organization_bootstrap_compare_and_set_failed")
            domains = set(_dataset_domains_from_row(row))
            if set(NOTES_ORGANIZATION_DOMAINS).difference(domains):
                raise SyncStoreError("notes_organization_sync_domains_incomplete")
            if state == "ready":
                if captured_count != expected_count or ready_verifier is None or not ready_verifier():
                    raise SyncStoreError("notes_organization_bootstrap_verification_failed")
                placeholders = ", ".join("?" for _ in NOTES_ORGANIZATION_DOMAINS)
                undrained = _first(
                    self.execute(
                        "SELECT COUNT(*) AS count FROM sync_envelopes "
                        "WHERE dataset_id = ? "
                        f"AND domain IN ({placeholders}) "  # nosec B608
                        "AND status = 'accepted' "
                        "AND apply_status NOT IN ('applied', 'superseded')",
                        (dataset_id, *NOTES_ORGANIZATION_DOMAINS),
                        connection=conn,
                    )
                )
                if undrained is None or int(undrained.get("count") or 0) != 0:
                    raise SyncStoreError("notes_organization_bootstrap_verification_failed")
                error_code = None
            metadata["notes_organization_v1"] = {
                "bootstrap_id": bootstrap_id,
                "state": state,
                "captured_count": captured_count,
                "expected_count": expected_count,
                "error_code": error_code,
            }
            self.execute(
                "UPDATE sync_datasets SET metadata_json = ?, updated_at = ? WHERE dataset_id = ?",
                (encode_json(metadata, default={}), utcnow_iso(), dataset_id),
                connection=conn,
            )
            updated = self._get_dataset_row(dataset_id, connection=conn)
            if updated is None:
                raise SyncStoreError("Sync dataset bootstrap transition was not persisted")
            return _dataset_from_row(updated)

    def begin_notes_link_bootstrap(
        self,
        dataset_id: str,
        *,
        owner_user_id: str,
        bootstrap_id: str,
    ) -> SyncDataset:
        """Atomically enroll notes.link without changing organization readiness."""

        if not bootstrap_id.strip():
            raise SyncStoreError("Notes link bootstrap ID is required")
        with self.backend.transaction() as conn:
            row = self._get_dataset_row_for_update(dataset_id, connection=conn)
            if (
                row is None
                or row.get("owner_user_id") != owner_user_id
                or row.get("scope_type") != "personal"
            ):
                raise SyncStoreError("Sync dataset was not found or is not accessible")
            metadata = decode_json(row.get("metadata_json"), default={})
            current = metadata.get("notes_link_v1")
            if isinstance(current, Mapping) and current.get("state") in {
                "initializing",
                "ready",
            }:
                return _dataset_from_row(row)
            enrolled = list(decode_json(row.get("domain_set_json"), default=[]))
            if "notes.note" not in enrolled:
                raise SyncStoreError("notes_link_note_domain_missing")
            if "notes.link" not in enrolled:
                enrolled.append("notes.link")
            metadata["notes_link_v1"] = {
                "bootstrap_id": bootstrap_id,
                "state": "initializing",
                "captured_count": 0,
                "expected_count": 0,
                "source_hash": None,
                "error_code": None,
            }
            self.execute(
                "UPDATE sync_datasets SET domain_set_json = ?, metadata_json = ?, "
                "updated_at = ? WHERE dataset_id = ?",
                (
                    encode_json(enrolled, default=[]),
                    encode_json(metadata, default={}),
                    utcnow_iso(),
                    dataset_id,
                ),
                connection=conn,
            )
            self._ensure_domain_state(
                dataset_id=dataset_id,
                domain="notes.link",
                adapter_version=1,
                server_sequence=0,
                connection=conn,
            )
            updated = self._get_dataset_row(dataset_id, connection=conn)
            if updated is None:
                raise SyncStoreError("Sync dataset link bootstrap update was not persisted")
            return _dataset_from_row(updated)

    def transition_notes_link_bootstrap(
        self,
        dataset_id: str,
        *,
        bootstrap_id: str,
        expected_state: str,
        state: str,
        captured_count: int,
        expected_count: int,
        source_hash: str | None,
        error_code: str | None = None,
        ready_verifier: Callable[[], bool] | None = None,
    ) -> SyncDataset:
        """Compare-and-set one durable notes.link bootstrap transition."""

        valid_states = {"initializing", "ready", "failed"}
        if expected_state not in valid_states or state not in valid_states:
            raise SyncStoreError("Notes link bootstrap state is invalid")
        if captured_count < 0 or expected_count < 0 or captured_count > expected_count:
            raise SyncStoreError("Notes link bootstrap counts are invalid")
        if source_hash is not None and (
            len(source_hash) != 64
            or any(character not in "0123456789abcdef" for character in source_hash)
        ):
            raise SyncStoreError("Notes link bootstrap source hash is invalid")
        with self.backend.transaction() as conn:
            row = self._get_dataset_row_for_update(dataset_id, connection=conn)
            if row is None:
                raise SyncStoreError("Sync dataset was not found or is not accessible")
            metadata = decode_json(row.get("metadata_json"), default={})
            current = metadata.get("notes_link_v1")
            if not isinstance(current, Mapping) or (
                current.get("bootstrap_id") != bootstrap_id
                or current.get("state") != expected_state
            ):
                raise SyncStoreError("notes_link_bootstrap_compare_and_set_failed")
            if set(NOTES_LINK_DOMAINS).difference(_dataset_domains_from_row(row)):
                raise SyncStoreError("notes_link_sync_domain_incomplete")
            current_hash = current.get("source_hash")
            if current_hash not in {None, source_hash}:
                raise SyncStoreError("notes_link_bootstrap_source_changed")
            if state == "ready":
                if (
                    captured_count != expected_count
                    or ready_verifier is None
                    or not ready_verifier()
                ):
                    raise SyncStoreError("notes_link_bootstrap_verification_failed")
                undrained = _first(
                    self.execute(
                        "SELECT COUNT(*) AS count FROM sync_envelopes "
                        "WHERE dataset_id = ? AND domain = 'notes.link' "
                        "AND status = 'accepted' "
                        "AND apply_status NOT IN ('applied', 'superseded')",
                        (dataset_id,),
                        connection=conn,
                    )
                )
                if undrained is None or int(undrained.get("count") or 0) != 0:
                    raise SyncStoreError("notes_link_bootstrap_verification_failed")
                error_code = None
            metadata["notes_link_v1"] = {
                "bootstrap_id": bootstrap_id,
                "state": state,
                "captured_count": captured_count,
                "expected_count": expected_count,
                "source_hash": source_hash,
                "error_code": error_code,
            }
            self.execute(
                "UPDATE sync_datasets SET metadata_json = ?, updated_at = ? WHERE dataset_id = ?",
                (encode_json(metadata, default={}), utcnow_iso(), dataset_id),
                connection=conn,
            )
            updated = self._get_dataset_row(dataset_id, connection=conn)
            if updated is None:
                raise SyncStoreError("Sync dataset link bootstrap transition was not persisted")
            return _dataset_from_row(updated)

    @staticmethod
    def _validate_notes_attachment_bootstrap_id(bootstrap_id: str) -> None:
        if (
            not isinstance(bootstrap_id, str)
            or not bootstrap_id.strip()
            or bootstrap_id != bootstrap_id.strip()
            or len(bootstrap_id.encode("utf-8")) > 128
        ):
            raise SyncStoreError("Notes attachment bootstrap ID is invalid")

    @staticmethod
    def _notes_attachment_source_hash(source_key: str) -> str:
        if not isinstance(source_key, str) or not source_key:
            raise SyncStoreError("Notes attachment source key is invalid")
        if len(source_key.encode("utf-8")) > 4_096:
            raise SyncStoreError("Notes attachment source key is too large")
        path = PurePosixPath(source_key)
        if (
            path.is_absolute()
            or "\\" in source_key
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise SyncStoreError("Notes attachment source key is invalid")
        return f"sha256:{hashlib.sha256(source_key.encode('utf-8')).hexdigest()}"

    @staticmethod
    def _notes_attachment_bootstrap_metadata(
        row: Mapping[str, Any],
        bootstrap_id: str,
    ) -> tuple[dict[str, Any], Mapping[str, Any]]:
        metadata = decode_json(row.get("metadata_json"), default={})
        current = metadata.get("notes_attachment_v2")
        if not isinstance(current, Mapping) or current.get("bootstrap_id") != bootstrap_id:
            raise SyncStoreError("notes_attachment_bootstrap_compare_and_set_failed")
        return metadata, current

    def begin_notes_attachment_bootstrap(
        self,
        dataset_id: str,
        *,
        owner_user_id: str,
        bootstrap_id: str,
    ) -> SyncDataset:
        """Enroll attachment.ref v2 and establish one stable bootstrap identity."""

        self._validate_notes_attachment_bootstrap_id(bootstrap_id)
        with self.backend.transaction() as conn:
            row = self._require_dataset_owner_for_update(
                dataset_id,
                owner_user_id,
                connection=conn,
            )
            if row.get("scope_type") != "personal":
                raise SyncStoreError("Sync dataset was not found or is not accessible")
            metadata = decode_json(row.get("metadata_json"), default={})
            current = metadata.get("notes_attachment_v2")
            if isinstance(current, Mapping):
                if (
                    current.get("state") not in {"initializing", "ready", "failed"}
                    or current.get("target_adapter_version") != 2
                    or not isinstance(current.get("bootstrap_id"), str)
                ):
                    raise SyncStoreError("notes_attachment_bootstrap_state_invalid")
                return _dataset_from_row(row)
            enrolled = list(decode_json(row.get("domain_set_json"), default=[]))
            if "notes.note" not in enrolled:
                raise SyncStoreError("notes_attachment_note_domain_missing")
            if "attachment.ref" not in enrolled:
                enrolled.append("attachment.ref")
            metadata["notes_attachment_v2"] = {
                "bootstrap_id": bootstrap_id,
                "state": "initializing",
                "target_adapter_version": 2,
                "captured_count": 0,
                "expected_count": 0,
                "source_hash": None,
                "source_cursor": None,
                "error_code": None,
            }
            self.execute(
                "UPDATE sync_datasets SET domain_set_json = ?, metadata_json = ?, "
                "updated_at = ? WHERE dataset_id = ? AND owner_user_id = ?",
                (
                    encode_json(enrolled, default=[]),
                    encode_json(metadata, default={}),
                    utcnow_iso(),
                    dataset_id,
                    owner_user_id,
                ),
                connection=conn,
            )
            self._ensure_domain_state(
                dataset_id=dataset_id,
                domain="attachment.ref",
                adapter_version=2,
                server_sequence=0,
                connection=conn,
            )
            updated = self._get_dataset_row(dataset_id, connection=conn)
            if updated is None:
                raise SyncStoreError("Sync dataset attachment bootstrap was not persisted")
            return _dataset_from_row(updated)

    def transition_notes_attachment_bootstrap(
        self,
        dataset_id: str,
        *,
        owner_user_id: str,
        bootstrap_id: str,
        expected_state: str,
        state: str,
        captured_count: int,
        expected_count: int,
        source_hash: str | None,
        source_cursor: str | None,
        error_code: str | None = None,
        ready_verifier: Callable[[], bool] | None = None,
    ) -> SyncDataset:
        """Compare-and-set one durable attachment bootstrap transition."""

        self._validate_notes_attachment_bootstrap_id(bootstrap_id)
        valid_states = {"initializing", "ready", "failed"}
        if expected_state not in valid_states or state not in valid_states:
            raise SyncStoreError("Notes attachment bootstrap state is invalid")
        if (
            isinstance(captured_count, bool)
            or isinstance(expected_count, bool)
            or captured_count < 0
            or expected_count < 0
            or captured_count > expected_count
        ):
            raise SyncStoreError("Notes attachment bootstrap counts are invalid")
        if source_hash is not None and re.fullmatch(r"[0-9a-f]{64}", source_hash) is None:
            raise SyncStoreError("Notes attachment bootstrap source hash is invalid")
        if source_cursor is not None and (
            not isinstance(source_cursor, str)
            or len(source_cursor.encode("utf-8")) > 64 * 1_024
        ):
            raise SyncStoreError("Notes attachment bootstrap cursor is invalid")
        if state == "failed":
            if error_code is None or re.fullmatch(r"[a-z][a-z0-9_]{0,127}", error_code) is None:
                raise SyncStoreError("Notes attachment bootstrap failure code is invalid")
        elif error_code is not None:
            raise SyncStoreError("Notes attachment bootstrap failure code is invalid")
        with self.backend.transaction() as conn:
            row = self._require_dataset_owner_for_update(
                dataset_id,
                owner_user_id,
                connection=conn,
            )
            metadata, current = self._notes_attachment_bootstrap_metadata(
                row,
                bootstrap_id,
            )
            if current.get("state") != expected_state:
                raise SyncStoreError("notes_attachment_bootstrap_compare_and_set_failed")
            if (
                current.get("target_adapter_version") != 2
                or "attachment.ref" not in _dataset_domains_from_row(row)
            ):
                raise SyncStoreError("notes_attachment_bootstrap_state_invalid")
            current_captured = current.get("captured_count")
            current_expected = current.get("expected_count")
            if (
                not isinstance(current_captured, int)
                or isinstance(current_captured, bool)
                or captured_count < current_captured
                or not isinstance(current_expected, int)
                or isinstance(current_expected, bool)
                or expected_count < current_expected
            ):
                raise SyncStoreError("notes_attachment_bootstrap_progress_regressed")
            current_hash = current.get("source_hash")
            if current_hash not in {None, source_hash}:
                raise SyncStoreError("notes_attachment_bootstrap_source_changed")
            if state == "ready":
                if (
                    source_hash is None
                    or captured_count != expected_count
                    or ready_verifier is None
                    or not ready_verifier()
                ):
                    raise SyncStoreError("notes_attachment_bootstrap_verification_failed")
                undrained = _first(
                    self.execute(
                        "SELECT COUNT(*) AS count FROM sync_envelopes "
                        "WHERE dataset_id = ? AND domain = 'attachment.ref' "
                        "AND status = 'accepted' "
                        "AND apply_status NOT IN ('applied', 'superseded')",
                        (dataset_id,),
                        connection=conn,
                    )
                )
                if undrained is None or int(undrained.get("count") or 0) != 0:
                    raise SyncStoreError("notes_attachment_bootstrap_verification_failed")
            metadata["notes_attachment_v2"] = {
                "bootstrap_id": bootstrap_id,
                "state": state,
                "target_adapter_version": 2,
                "captured_count": captured_count,
                "expected_count": expected_count,
                "source_hash": source_hash,
                "source_cursor": source_cursor,
                "error_code": error_code,
            }
            self.execute(
                "UPDATE sync_datasets SET metadata_json = ?, updated_at = ? "
                "WHERE dataset_id = ? AND owner_user_id = ?",
                (
                    encode_json(metadata, default={}),
                    utcnow_iso(),
                    dataset_id,
                    owner_user_id,
                ),
                connection=conn,
            )
            updated = self._get_dataset_row(dataset_id, connection=conn)
            if updated is None:
                raise SyncStoreError("Sync dataset attachment bootstrap transition was not persisted")
            return _dataset_from_row(updated)

    def resolve_notes_attachment_source_map(
        self,
        dataset_id: str,
        *,
        owner_user_id: str,
        bootstrap_id: str,
        note_id: str,
        source_key: str,
    ) -> SyncNotesAttachmentSourceMap:
        """Allocate one immutable UUIDv4 for one hashed bootstrap source key."""

        self._validate_notes_attachment_bootstrap_id(bootstrap_id)
        if not isinstance(note_id, str) or not note_id.strip():
            raise SyncStoreError("Notes attachment source note ID is invalid")
        source_key_hash = self._notes_attachment_source_hash(source_key)
        with self.backend.transaction() as conn:
            row = self._require_dataset_owner_for_update(
                dataset_id,
                owner_user_id,
                connection=conn,
            )
            self._notes_attachment_bootstrap_metadata(row, bootstrap_id)
            existing = _first(
                self.execute(
                    "SELECT * FROM sync_notes_attachment_source_map "
                    "WHERE dataset_id = ? AND bootstrap_id = ? AND source_key_hash = ?",
                    (dataset_id, bootstrap_id, source_key_hash),
                    connection=conn,
                )
            )
            if existing is not None:
                if existing.get("note_id") != note_id:
                    raise SyncStoreError("notes_attachment_source_map_identity_conflict")
                return _notes_attachment_source_map_from_row(existing)
            self.execute(
                "INSERT INTO sync_notes_attachment_source_map "
                "(dataset_id, bootstrap_id, source_key_hash, note_id, attachment_id, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    dataset_id,
                    bootstrap_id,
                    source_key_hash,
                    note_id,
                    str(uuid4()),
                    utcnow_iso(),
                ),
                connection=conn,
            )
            created = _first(
                self.execute(
                    "SELECT * FROM sync_notes_attachment_source_map "
                    "WHERE dataset_id = ? AND bootstrap_id = ? AND source_key_hash = ?",
                    (dataset_id, bootstrap_id, source_key_hash),
                    connection=conn,
                )
            )
            if created is None:
                raise SyncStoreError("Notes attachment source map was not persisted")
            return _notes_attachment_source_map_from_row(created)

    def record_notes_attachment_cleanup_candidate(
        self,
        dataset_id: str,
        *,
        owner_user_id: str,
        bootstrap_id: str,
        source_key: str,
        source_relative_path: str,
        source_blob_hash: str,
        source_size_bytes: int,
        source_modified_ns: int,
    ) -> SyncNotesAttachmentCleanupCandidate:
        """Persist immutable, non-authoritative cleanup evidence for one source."""

        self._validate_notes_attachment_bootstrap_id(bootstrap_id)
        source_key_hash = self._notes_attachment_source_hash(source_key)
        source_path_hash = self._notes_attachment_source_hash(source_relative_path)
        if source_path_hash != source_key_hash:
            raise SyncStoreError("Notes attachment cleanup source path does not match")
        if re.fullmatch(r"sha256:[0-9a-f]{64}", source_blob_hash) is None:
            raise SyncStoreError("Notes attachment cleanup blob hash is invalid")
        if (
            isinstance(source_size_bytes, bool)
            or source_size_bytes < 1
            or isinstance(source_modified_ns, bool)
            or source_modified_ns < 0
        ):
            raise SyncStoreError("Notes attachment cleanup source stat is invalid")
        with self.backend.transaction() as conn:
            row = self._require_dataset_owner_for_update(
                dataset_id,
                owner_user_id,
                connection=conn,
            )
            self._notes_attachment_bootstrap_metadata(row, bootstrap_id)
            mapping = _first(
                self.execute(
                    "SELECT * FROM sync_notes_attachment_source_map "
                    "WHERE dataset_id = ? AND bootstrap_id = ? AND source_key_hash = ?",
                    (dataset_id, bootstrap_id, source_key_hash),
                    connection=conn,
                )
            )
            if mapping is None:
                raise SyncStoreError("notes_attachment_source_map_missing")
            existing = _first(
                self.execute(
                    "SELECT * FROM sync_notes_attachment_cleanup_candidates "
                    "WHERE dataset_id = ? AND bootstrap_id = ? AND source_key_hash = ?",
                    (dataset_id, bootstrap_id, source_key_hash),
                    connection=conn,
                )
            )
            expected = {
                "attachment_id": mapping["attachment_id"],
                "source_relative_path": source_relative_path,
                "source_path_hash": source_path_hash,
                "source_blob_hash": source_blob_hash,
                "source_size_bytes": source_size_bytes,
                "source_modified_ns": source_modified_ns,
            }
            if existing is not None:
                if any(existing.get(key) != value for key, value in expected.items()):
                    raise SyncStoreError("notes_attachment_cleanup_candidate_conflict")
                return _notes_attachment_cleanup_candidate_from_row(existing)
            self.execute(
                "INSERT INTO sync_notes_attachment_cleanup_candidates "
                "(dataset_id, bootstrap_id, source_key_hash, attachment_id, "
                "source_relative_path, source_path_hash, source_blob_hash, "
                "source_size_bytes, source_modified_ns, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    dataset_id,
                    bootstrap_id,
                    source_key_hash,
                    mapping["attachment_id"],
                    source_relative_path,
                    source_path_hash,
                    source_blob_hash,
                    source_size_bytes,
                    source_modified_ns,
                    utcnow_iso(),
                ),
                connection=conn,
            )
            created = _first(
                self.execute(
                    "SELECT * FROM sync_notes_attachment_cleanup_candidates "
                    "WHERE dataset_id = ? AND bootstrap_id = ? AND source_key_hash = ?",
                    (dataset_id, bootstrap_id, source_key_hash),
                    connection=conn,
                )
            )
            if created is None:
                raise SyncStoreError("Notes attachment cleanup candidate was not persisted")
            return _notes_attachment_cleanup_candidate_from_row(created)

    def get_notes_attachment_bootstrap_source_by_hash(
        self,
        dataset_id: str,
        *,
        owner_user_id: str,
        bootstrap_id: str,
        source_key_hash: str,
    ) -> tuple[
        SyncNotesAttachmentSourceMap,
        SyncNotesAttachmentCleanupCandidate,
    ] | None:
        """Resolve one internal bootstrap source without exposing its path publicly."""

        self._validate_notes_attachment_bootstrap_id(bootstrap_id)
        if re.fullmatch(r"sha256:[0-9a-f]{64}", source_key_hash) is None:
            raise ValueError("Notes attachment source cursor is invalid")
        with self.backend.transaction() as conn:
            row = self._require_attachment_binding_dataset_owner(
                dataset_id,
                owner_user_id,
                connection=conn,
            )
            self._notes_attachment_bootstrap_metadata(row, bootstrap_id)
            source = _first(
                self.execute(
                    "SELECT source_map.*, cleanup.source_relative_path, "
                    "cleanup.source_path_hash, cleanup.source_blob_hash, "
                    "cleanup.source_size_bytes, cleanup.source_modified_ns, "
                    "cleanup.created_at AS cleanup_created_at "
                    "FROM sync_notes_attachment_source_map AS source_map "
                    "JOIN sync_notes_attachment_cleanup_candidates AS cleanup "
                    "ON cleanup.dataset_id = source_map.dataset_id "
                    "AND cleanup.bootstrap_id = source_map.bootstrap_id "
                    "AND cleanup.source_key_hash = source_map.source_key_hash "
                    "WHERE source_map.dataset_id = ? "
                    "AND source_map.bootstrap_id = ? "
                    "AND source_map.source_key_hash = ?",
                    (dataset_id, bootstrap_id, source_key_hash),
                    connection=conn,
                )
            )
            if source is None:
                return None
            cleanup_row = {
                "dataset_id": source["dataset_id"],
                "bootstrap_id": source["bootstrap_id"],
                "source_key_hash": source["source_key_hash"],
                "attachment_id": source["attachment_id"],
                "source_relative_path": source["source_relative_path"],
                "source_path_hash": source["source_path_hash"],
                "source_blob_hash": source["source_blob_hash"],
                "source_size_bytes": source["source_size_bytes"],
                "source_modified_ns": source["source_modified_ns"],
                "created_at": source["cleanup_created_at"],
            }
            return (
                _notes_attachment_source_map_from_row(source),
                _notes_attachment_cleanup_candidate_from_row(cleanup_row),
            )

    def list_notes_attachment_cleanup_candidates(
        self,
        dataset_id: str,
        *,
        owner_user_id: str,
        bootstrap_id: str,
        after_source_key_hash: str | None = None,
        limit: int = 1_000,
    ) -> tuple[SyncNotesAttachmentCleanupCandidate, ...]:
        """Return one bounded owner-scoped cleanup-candidate keyset page."""

        self._validate_notes_attachment_bootstrap_id(bootstrap_id)
        if isinstance(limit, bool) or not 1 <= limit <= 1_000:
            raise ValueError("cleanup candidate page limit must be 1..1000")
        if after_source_key_hash is not None and re.fullmatch(
            r"sha256:[0-9a-f]{64}", after_source_key_hash
        ) is None:
            raise ValueError("cleanup candidate cursor is invalid")
        with self.backend.transaction() as conn:
            row = self._require_attachment_binding_dataset_owner(
                dataset_id,
                owner_user_id,
                connection=conn,
            )
            self._notes_attachment_bootstrap_metadata(row, bootstrap_id)
            cursor = after_source_key_hash or ""
            rows = self.execute(
                "SELECT * FROM sync_notes_attachment_cleanup_candidates "
                "WHERE dataset_id = ? AND bootstrap_id = ? AND source_key_hash > ? "
                "ORDER BY source_key_hash LIMIT ?",
                (dataset_id, bootstrap_id, cursor, limit),
                connection=conn,
            ).rows
            return tuple(
                _notes_attachment_cleanup_candidate_from_row(item) for item in rows
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
            dataset_row = self._require_dataset_domain(
                envelope.dataset_id,
                envelope.domain,
                connection=conn,
            )
            self._require_notes_organization_write_ready(dataset_row, envelope.domain)
            existing = self._find_existing_envelope_for_idempotency(
                envelope,
                connection=conn,
            )
        if existing is None:
            return None
        return _envelope_from_row(existing)

    def _require_no_unresolved_materialization_conflict(
        self,
        dataset_id: str,
        *,
        envelope_status: str,
        connection: Any,
    ) -> None:
        """Reject new accepted history while a projected conflict needs review."""

        if envelope_status != "accepted":
            return
        blocker = self.get_unresolved_materialization_conflict(
            dataset_id,
            connection=connection,
        )
        if blocker is not None:
            raise SyncMaterializationPredecessorError(
                apply_status="conflict",
                conflict_id=blocker.conflict_id,
                domain=blocker.domain,
                entity_id=blocker.entity_id,
                server_sequence=blocker.server_sequence,
            )

    def get_unresolved_materialization_conflict(
        self,
        dataset_id: str,
        *,
        connection: Any | None = None,
    ) -> SyncConflict | None:
        """Return the earliest accepted projection conflict for a dataset."""

        row = _first(
            self.execute(
                """
                SELECT conflict.*
                  FROM sync_conflicts AS conflict
                  JOIN sync_envelopes AS envelope
                    ON envelope.dataset_id = conflict.dataset_id
                   AND envelope.client_envelope_id = conflict.local_envelope_id
                   AND envelope.server_sequence = conflict.server_sequence
                 WHERE conflict.dataset_id = ?
                   AND conflict.status = 'unresolved'
                   AND envelope.status = 'accepted'
                   AND envelope.apply_status = 'conflict'
                 ORDER BY envelope.server_sequence ASC
                 LIMIT 1
                """,
                (dataset_id,),
                connection=connection,
            )
        )
        return None if row is None else _conflict_from_row(row)

    def insert_envelope(
        self,
        envelope: SyncEnvelopeCreate,
        *,
        connection: Any | None = None,
    ) -> SyncEnvelope:
        self._validate_envelope_contract(envelope)
        with self.backend.transaction(connection) as conn:
            dataset_row = self._require_dataset_domain_for_update(
                envelope.dataset_id,
                envelope.domain,
                connection=conn,
            )
            self._require_notes_organization_write_ready(dataset_row, envelope.domain)

            existing = self._find_existing_envelope_for_idempotency(
                envelope,
                connection=conn,
            )
            if existing is not None:
                return _envelope_from_row(existing)

            self._require_no_unresolved_materialization_conflict(
                envelope.dataset_id,
                envelope_status=envelope.status,
                connection=conn,
            )
            self._require_expected_current_head(envelope, connection=conn)
            return self._insert_envelope_in_transaction(envelope, connection=conn)

    def insert_claimed_conflict_resolution_envelope(
        self,
        envelope: SyncEnvelopeCreate,
        *,
        conflict_id: str,
        dataset_id: str,
        resolved_by_device_id: str | None,
        resolution_action: str,
        resolution_notes: str | None,
        connection: Any,
    ) -> SyncEnvelope:
        """Append a claimed resolution under the caller's dataset authority."""

        self._validate_envelope_contract(envelope)
        with self.backend.transaction(connection) as conn:
            dataset_row = self._require_dataset_domain(
                envelope.dataset_id,
                envelope.domain,
                connection=conn,
            )
            self._require_notes_organization_write_ready(dataset_row, envelope.domain)
            conflict_row, source_row = self._require_claimed_conflict_source(
                conflict_id,
                dataset_id=dataset_id,
                resolved_by_device_id=resolved_by_device_id,
                resolution_action=resolution_action,
                resolution_notes=resolution_notes,
                connection=conn,
            )
            existing = self._find_existing_envelope_for_idempotency(
                envelope,
                connection=conn,
            )
            if existing is not None:
                return _envelope_from_row(existing)

            source_is_rebase_required = _is_rebase_required_conflict_source(
                conflict_row,
                source_row,
            )
            if (
                envelope.dataset_id != source_row.get("dataset_id")
                or envelope.domain != source_row.get("domain")
                or (
                    resolution_action == "overwrite"
                    and envelope.object_id != source_row.get("entity_id")
                )
                or (
                    resolution_action == "duplicate_rename"
                    and envelope.object_id == source_row.get("entity_id")
                )
            ):
                raise SyncHeadConflictError()
            if resolution_action in {"overwrite", "duplicate_rename"}:
                snapshot_cursor = (
                    None
                    if source_row.get("status") != "accepted"
                    or source_is_rebase_required
                    else int(source_row["server_sequence"])
                )
                self._require_expected_applied_head(
                    envelope,
                    through_server_cursor=snapshot_cursor,
                    connection=conn,
                )
            else:
                self._require_expected_current_head(envelope, connection=conn)
            return self._insert_envelope_in_transaction(envelope, connection=conn)

    def list_latest_applied_heads(
        self,
        dataset_id: str,
        *,
        through_server_cursor: int | None = None,
        connection: Any,
    ) -> list[SyncEnvelope]:
        """Return one immutable latest-applied head per identity at a cursor."""

        if through_server_cursor is None:
            query = """
                SELECT envelope.*
                  FROM sync_envelopes AS envelope
                 WHERE envelope.dataset_id = ?
                   AND envelope.status = 'accepted'
                   AND envelope.apply_status = 'applied'
                   AND NOT EXISTS (
                        SELECT 1
                          FROM sync_envelopes AS newer
                         WHERE newer.dataset_id = envelope.dataset_id
                           AND newer.domain = envelope.domain
                           AND newer.entity_id = envelope.entity_id
                           AND newer.status = 'accepted'
                           AND newer.apply_status = 'applied'
                           AND newer.server_sequence > envelope.server_sequence
                   )
                 ORDER BY envelope.domain ASC, envelope.entity_id ASC
            """
            params = (dataset_id,)
        else:
            query = """
                SELECT envelope.*
                  FROM sync_envelopes AS envelope
                 WHERE envelope.dataset_id = ?
                   AND envelope.status = 'accepted'
                   AND envelope.apply_status = 'applied'
                   AND envelope.server_sequence <= ?
                   AND NOT EXISTS (
                        SELECT 1
                          FROM sync_envelopes AS newer
                         WHERE newer.dataset_id = envelope.dataset_id
                           AND newer.domain = envelope.domain
                           AND newer.entity_id = envelope.entity_id
                           AND newer.status = 'accepted'
                           AND newer.apply_status = 'applied'
                           AND newer.server_sequence > envelope.server_sequence
                           AND newer.server_sequence <= ?
                   )
                 ORDER BY envelope.domain ASC, envelope.entity_id ASC
            """
            params = (dataset_id, through_server_cursor, through_server_cursor)
        rows = self.execute(
            query,
            params,
            connection=connection,
        ).rows
        return [_envelope_from_row(row) for row in rows]

    def _require_expected_applied_head(
        self,
        envelope: SyncEnvelopeCreate,
        *,
        through_server_cursor: int | None,
        connection: Any,
    ) -> None:
        if through_server_cursor is None:
            query = """
                SELECT *
                  FROM sync_envelopes
                 WHERE dataset_id = ? AND domain = ? AND entity_id = ?
                   AND status = 'accepted' AND apply_status = 'applied'
                 ORDER BY server_sequence DESC
                 LIMIT 1
            """
            params = (envelope.dataset_id, envelope.domain, envelope.object_id)
        else:
            query = """
                SELECT *
                  FROM sync_envelopes
                 WHERE dataset_id = ? AND domain = ? AND entity_id = ?
                   AND status = 'accepted' AND apply_status = 'applied'
                   AND server_sequence <= ?
                 ORDER BY server_sequence DESC
                 LIMIT 1
            """
            params = (
                envelope.dataset_id,
                envelope.domain,
                envelope.object_id,
                through_server_cursor,
            )
        projected = _first(
            self.execute(
                query,
                params,
                connection=connection,
            )
        )
        expected = (
            None,
            None,
            None,
        ) if projected is None else (
            int(projected["server_sequence"]),
            projected.get("object_revision"),
            projected.get("payload_hash"),
        )
        if (
            envelope.base_server_cursor,
            envelope.base_object_revision,
            envelope.base_object_hash,
        ) != expected:
            raise SyncHeadConflictError()

    def get_latest_applied_predecessor(
        self,
        envelope: SyncEnvelope,
        *,
        connection: Any,
    ) -> SyncEnvelope | None:
        """Return the projected base immediately preceding one canonical envelope."""

        if envelope.server_cursor is None:
            raise SyncStoreError("Stored Sync envelope is missing its server cursor")
        row = _first(
            self.execute(
                """
                SELECT *
                  FROM sync_envelopes
                 WHERE dataset_id = ? AND domain = ? AND entity_id = ?
                   AND status = 'accepted' AND apply_status = 'applied'
                   AND server_sequence < ?
                 ORDER BY server_sequence DESC
                 LIMIT 1
                """,
                (
                    envelope.dataset_id,
                    envelope.domain,
                    envelope.object_id,
                    envelope.server_cursor,
                ),
                connection=connection,
            )
        )
        return None if row is None else _envelope_from_row(row)

    def _require_expected_current_head(
        self,
        envelope: SyncEnvelopeCreate,
        *,
        connection: Any,
        planned_head: SyncEnvelopeCreate | None = None,
        allow_unstored_product_base: bool = False,
    ) -> None:
        """CAS one accepted envelope against its preflighted object head."""

        if envelope.status != "accepted":
            return
        if planned_head is not None:
            expected_cursor = 0
            expected_revision = planned_head.object_revision
            expected_hash = planned_head.payload_hash
        else:
            row = _first(
                self.execute(
                    """
                    SELECT envelope.*,
                           projected.object_revision AS projected_object_revision,
                           projected.object_hash AS projected_object_hash
                      FROM sync_current_heads AS head
                      JOIN sync_envelopes AS envelope
                        ON envelope.server_sequence = head.latest_server_cursor
                      LEFT JOIN sync_object_state AS projected
                        ON projected.dataset_id = head.dataset_id
                       AND projected.domain = head.domain
                       AND projected.object_id = head.object_id
                       AND projected.latest_server_cursor = head.latest_server_cursor
                     WHERE head.dataset_id = ?
                       AND head.domain = ?
                       AND head.object_id = ?
                    """,
                    (envelope.dataset_id, envelope.domain, envelope.object_id),
                    connection=connection,
                )
            )
            if row is None:
                if (
                    allow_unstored_product_base
                    and envelope.domain == "notes.task"
                    and envelope.base_server_cursor is None
                    and envelope.base_object_revision is not None
                    and envelope.base_object_hash is not None
                    and envelope.routing_metadata.get("product_transition_base")
                    is True
                ):
                    return
                if any(
                    value is not None
                    for value in (
                        envelope.base_server_cursor,
                        envelope.base_object_revision,
                        envelope.base_object_hash,
                    )
                ):
                    raise SyncHeadConflictError()
                return
            current = _envelope_from_row(row)
            expected_cursor = current.server_cursor
            if current.object_revision is None and row.get("projected_object_revision") is not None:
                expected_revision = int(row["projected_object_revision"])
                expected_hash = row.get("projected_object_hash")
            else:
                expected_revision = current.object_revision
                expected_hash = current.payload_hash

        if (
            envelope.base_server_cursor != expected_cursor
            or envelope.base_object_revision != expected_revision
            or envelope.base_object_hash != expected_hash
        ):
            raise SyncHeadConflictError()

    def _insert_envelope_in_transaction(
        self,
        envelope: SyncEnvelopeCreate,
        *,
        connection: Any,
    ) -> SyncEnvelope:
        now = utcnow_iso()
        self.execute(
            """
            INSERT INTO sync_envelopes (
                dataset_id, domain, entity_id, stable_key, operation,
                client_envelope_id, device_id, client_profile_id, client_sequence,
                mutation_group_id, mutation_step, mutation_step_count,
                mutation_plan_hash, client_timestamp, server_timestamp,
                base_server_cursor, base_object_revision, base_object_hash,
                object_revision, parent_id, schema_version, base_version,
                entity_version, dependency_json, routing_metadata_json,
                payload_ciphertext, payload_json, payload_clear_json, payload_hash,
                payload_size_bytes, created_at_client, received_at_server, deleted,
                encryption_metadata_json, adapter_version, status, apply_status,
                apply_error_code, apply_error_message, applied_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                envelope.mutation_group_id,
                envelope.mutation_step,
                envelope.mutation_step_count,
                envelope.mutation_plan_hash,
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
            connection=connection,
        )
        sequence = self.backend.get_last_insert_id(connection=connection)
        if sequence is None:
            row = _first(
                self.execute(
                    """
                    SELECT * FROM sync_envelopes
                     WHERE dataset_id = ? AND client_envelope_id = ?
                    """,
                    (envelope.dataset_id, envelope.client_envelope_id),
                    connection=connection,
                )
            )
        else:
            row = _first(
                self.execute(
                    "SELECT * FROM sync_envelopes WHERE server_sequence = ?",
                    (sequence,),
                    connection=connection,
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
                        connection=connection,
                    )
                )
        if row is None:
            raise SyncStoreError("Sync envelope insert did not produce a retrievable record")
        if (
            _envelope_fingerprint_from_row(row)
            != _envelope_fingerprint_from_create(envelope)
        ):
            raise SyncIdempotencyConflictError(
                "Sync envelope idempotency key was reused with different content"
            )
        inserted = _envelope_from_row(row)
        if (
            inserted.status == "accepted"
            and inserted.domain == "attachment.ref"
            and inserted.adapter_version == 2
        ):
            self._create_attachment_binding_for_envelope(
                inserted,
                connection=connection,
            )
        if inserted.status == "accepted":
            self.execute(
                """
                INSERT INTO sync_current_heads (
                    dataset_id, domain, object_id, latest_server_cursor
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT (dataset_id, domain, object_id)
                DO UPDATE SET latest_server_cursor = excluded.latest_server_cursor
                 WHERE excluded.latest_server_cursor > sync_current_heads.latest_server_cursor
                """,
                (
                    inserted.dataset_id,
                    inserted.domain,
                    inserted.object_id,
                    inserted.server_cursor,
                ),
                connection=connection,
            )
        self._ensure_domain_state(
            dataset_id=inserted.dataset_id,
            domain=inserted.domain,
            adapter_version=inserted.adapter_version,
            server_sequence=inserted.server_sequence,
            connection=connection,
        )
        return inserted

    def insert_envelopes_atomic(
        self,
        envelopes: Sequence[SyncEnvelopeCreate],
        *,
        trusted_notes_organization_bootstrap_id: str | None = None,
        trusted_notes_task_bootstrap_id: str | None = None,
        trusted_notes_task_coordinator: bool = False,
    ) -> list[SyncEnvelope]:
        """Insert one complete validated group or return its exact stored replay."""

        submitted_plan = list(envelopes)
        try:
            plan = self._validate_mutation_group_plan(submitted_plan)
        except SyncStoreError as exc:
            if self._has_existing_mutation_group(submitted_plan):
                raise SyncIdempotencyConflictError(
                    "Sync mutation group idempotency key was reused with different content"
                ) from exc
            raise
        first = plan[0]
        mutation_group_id = first.mutation_group_id
        if mutation_group_id is None:
            raise SyncStoreError("Atomic Sync append requires mutation group metadata")

        try:
            with self.backend.transaction() as conn:
                for envelope in plan:
                    if (
                        envelope.domain in {"notes.task", "notes.task_activity"}
                        and trusted_notes_task_bootstrap_id is not None
                    ):
                        dataset_row = self._get_dataset_row_for_update(
                            envelope.dataset_id,
                            connection=conn,
                        )
                        if dataset_row is None:
                            raise SyncDatasetNotFoundError(
                                f"Sync dataset not found: {envelope.dataset_id}"
                            )
                        self._require_notes_task_bootstrap_write_ready(
                            dataset_row,
                            envelope=envelope,
                            bootstrap_id=trusted_notes_task_bootstrap_id,
                        )
                    elif (
                        envelope.domain in {"notes.task", "notes.task_activity"}
                        and trusted_notes_task_coordinator
                    ):
                        dataset_row = self._get_dataset_row_for_update(
                            envelope.dataset_id,
                            connection=conn,
                        )
                        if dataset_row is None:
                            raise SyncDatasetNotFoundError(
                                f"Sync dataset not found: {envelope.dataset_id}"
                            )
                        metadata = decode_json(
                            dataset_row.get("metadata_json"),
                            default={},
                        )
                        if not notes_task_capture_is_active(metadata):
                            raise SyncStoreError("notes_task_sync_not_ready")
                    else:
                        dataset_row = self._require_dataset_domain_for_update(
                            envelope.dataset_id,
                            envelope.domain,
                            connection=conn,
                        )
                    self._require_notes_organization_write_ready(
                        dataset_row,
                        envelope.domain,
                        trusted_bootstrap_id=trusted_notes_organization_bootstrap_id,
                    )

                existing_rows = self._list_mutation_group_rows(
                    first.dataset_id,
                    mutation_group_id,
                    connection=conn,
                )
                if existing_rows:
                    return self._matched_mutation_group_replay(plan, existing_rows)

                for envelope in plan:
                    existing = self._find_existing_envelope_for_idempotency(
                        envelope,
                        connection=conn,
                    )
                    if existing is not None:
                        raise SyncIdempotencyConflictError(
                            "Sync mutation group idempotency key was reused with different content"
                        )

                self._require_no_unresolved_materialization_conflict(
                    first.dataset_id,
                    envelope_status=first.status,
                    connection=conn,
                )
                planned_heads: dict[tuple[str, str], SyncEnvelopeCreate] = {}
                for envelope in plan:
                    key = (envelope.domain, envelope.object_id)
                    self._require_expected_current_head(
                        envelope,
                        connection=conn,
                        planned_head=planned_heads.get(key),
                        allow_unstored_product_base=trusted_notes_task_coordinator,
                    )
                    if envelope.status == "accepted":
                        planned_heads[key] = envelope

                return [
                    self._insert_envelope_in_transaction(envelope, connection=conn)
                    for envelope in plan
                ]
        except SyncHeadConflictError:
            with self.backend.transaction() as conn:
                existing_rows = self._list_mutation_group_rows(
                    first.dataset_id,
                    mutation_group_id,
                    connection=conn,
                )
            if not existing_rows:
                raise
            return self._matched_mutation_group_replay(plan, existing_rows)
        except BackendDatabaseError as exc:
            if not _is_mutation_group_step_unique_error(exc):
                raise
            with self.backend.transaction() as conn:
                existing_rows = self._list_mutation_group_rows(
                    first.dataset_id,
                    mutation_group_id,
                    connection=conn,
                )
            if not existing_rows:
                raise
            return self._matched_mutation_group_replay(plan, existing_rows)

    def _has_existing_mutation_group(
        self,
        plan: Sequence[SyncEnvelopeCreate],
    ) -> bool:
        identities = {
            (envelope.dataset_id, envelope.mutation_group_id)
            for envelope in plan
            if envelope.mutation_group_id is not None
        }
        if not identities:
            return False
        with self.backend.transaction() as conn:
            return any(
                self._list_mutation_group_rows(
                    dataset_id,
                    mutation_group_id,
                    connection=conn,
                )
                for dataset_id, mutation_group_id in identities
            )

    def _validate_mutation_group_plan(
        self,
        envelopes: Sequence[SyncEnvelopeCreate],
    ) -> list[SyncEnvelopeCreate]:
        plan = list(envelopes)
        if not plan:
            raise SyncStoreError("Sync mutation group must contain at least one envelope")
        if len(plan) > SYNC_MUTATION_GROUP_MAX_SIZE:
            raise SyncStoreError("sync_restore_group_limit_exceeded")
        for envelope in plan:
            self._validate_envelope_contract(envelope)

        first = plan[0]
        if (
            first.mutation_group_id is None
            or first.mutation_step_count is None
            or first.mutation_plan_hash is None
        ):
            raise SyncStoreError("Atomic Sync append requires mutation group metadata")
        expected_metadata = (
            first.dataset_id,
            first.mutation_group_id,
            first.mutation_step_count,
            first.mutation_plan_hash,
        )
        if any(
            (
                envelope.dataset_id,
                envelope.mutation_group_id,
                envelope.mutation_step_count,
                envelope.mutation_plan_hash,
            )
            != expected_metadata
            for envelope in plan
        ):
            raise SyncStoreError(
                "Sync mutation group envelopes must share dataset, group, count, and hash"
            )
        if [envelope.mutation_step for envelope in plan] != list(
            range(first.mutation_step_count)
        ):
            raise SyncStoreError(
                "Sync mutation group steps must exactly match the ordered complete plan"
            )
        client_envelope_ids = [envelope.client_envelope_id for envelope in plan]
        if len(set(client_envelope_ids)) != len(client_envelope_ids):
            raise SyncStoreError("Sync mutation group client envelope ids must be unique")
        client_sequence_keys = [
            (envelope.device_id, envelope.client_sequence)
            for envelope in plan
            if envelope.device_id is not None and envelope.client_sequence is not None
        ]
        if len(set(client_sequence_keys)) != len(client_sequence_keys):
            raise SyncStoreError("Sync mutation group client sequence keys must be unique")
        return plan

    def _list_mutation_group_rows(
        self,
        dataset_id: str,
        mutation_group_id: str,
        *,
        connection: Any | None = None,
    ) -> list[dict[str, Any]]:
        rows = self.execute(
            """
            SELECT * FROM sync_envelopes
             WHERE dataset_id = ? AND mutation_group_id = ?
             ORDER BY mutation_step ASC
             LIMIT ?
            """,
            (
                dataset_id,
                mutation_group_id,
                SYNC_MUTATION_GROUP_MAX_SIZE + 1,
            ),
            connection=connection,
        ).rows
        if len(rows) > SYNC_MUTATION_GROUP_MAX_SIZE:
            raise SyncStoreError("sync_restore_group_limit_exceeded")
        return rows

    def _matched_mutation_group_replay(
        self,
        plan: Sequence[SyncEnvelopeCreate],
        existing_rows: Sequence[dict[str, Any]],
    ) -> list[SyncEnvelope]:
        if len(plan) != len(existing_rows) or any(
            _envelope_fingerprint_from_create(envelope)
            != _envelope_fingerprint_from_row(row)
            for envelope, row in zip(plan, existing_rows)
        ):
            raise SyncIdempotencyConflictError(
                "Sync mutation group idempotency key was reused with different content"
            )
        return [_envelope_from_row(row) for row in existing_rows]

    def list_mutation_group(
        self,
        dataset_id: str,
        mutation_group_id: str,
        *,
        connection: Any | None = None,
    ) -> list[SyncEnvelope]:
        """Return a complete mutation group ordered by zero-based step."""

        if not mutation_group_id.strip():
            raise SyncStoreError("Sync mutation group id must be non-empty")
        return [
            _envelope_from_row(row)
            for row in self._list_mutation_group_rows(
                dataset_id,
                mutation_group_id,
                connection=connection,
            )
        ]

    def list_envelopes_after(
        self,
        dataset_id: str,
        since_sequence: int,
        *,
        limit: int = 100,
        domains: Sequence[SyncDomain] | None = None,
        adapter_versions: Sequence[int] | None = None,
        status: str | Sequence[str] | None = None,
        exclude_device_id: str | None = None,
        connection: Any | None = None,
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
        if adapter_versions is not None:
            versions = sorted(set(adapter_versions))
            if not versions:
                return []
            placeholders = ", ".join("?" for _ in versions)
            sql += f" AND adapter_version IN ({placeholders})"
            params.extend(versions)
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
        result = self.execute(sql, tuple(params), connection=connection)
        return [_envelope_from_row(row) for row in result.rows]

    def summarize_domain_envelopes(
        self,
        dataset_id: str,
        domain: SyncDomain,
    ) -> SyncDomainEnvelopeSummary:
        """Return aggregate envelope health for one dataset domain."""

        with self.backend.transaction() as conn:
            self._require_dataset_domain(dataset_id, domain, connection=conn)
            aggregate_row = _first(
                self.execute(
                    """
                    SELECT COUNT(*) AS envelope_count,
                           COALESCE(
                               SUM(CASE WHEN apply_status = 'pending' THEN 1 ELSE 0 END),
                               0
                           ) AS pending_apply_count,
                           COALESCE(
                               SUM(CASE WHEN apply_status = 'failed' THEN 1 ELSE 0 END),
                               0
                           ) AS failed_apply_count
                      FROM sync_envelopes
                     WHERE dataset_id = ?
                       AND domain = ?
                    """,
                    (dataset_id, domain),
                    connection=conn,
                )
            ) or {}
            last_row = _first(
                self.execute(
                    """
                    SELECT * FROM sync_envelopes
                     WHERE dataset_id = ?
                       AND domain = ?
                     ORDER BY server_sequence DESC
                     LIMIT 1
                    """,
                    (dataset_id, domain),
                    connection=conn,
                )
            )
            last_failed_row = _first(
                self.execute(
                    """
                    SELECT * FROM sync_envelopes
                     WHERE dataset_id = ?
                       AND domain = ?
                       AND apply_status = 'failed'
                     ORDER BY server_sequence DESC
                     LIMIT 1
                    """,
                    (dataset_id, domain),
                    connection=conn,
                )
            )
        return SyncDomainEnvelopeSummary(
            domain=domain,
            envelope_count=int(aggregate_row.get("envelope_count") or 0),
            pending_apply_count=int(aggregate_row.get("pending_apply_count") or 0),
            failed_apply_count=int(aggregate_row.get("failed_apply_count") or 0),
            last_envelope=_envelope_from_row(last_row) if last_row is not None else None,
            last_failed_envelope=(
                _envelope_from_row(last_failed_row)
                if last_failed_row is not None
                else None
            ),
        )

    def list_envelopes_for_entity(
        self,
        dataset_id: str,
        domain: SyncDomain,
        *,
        entity_id: str | None = None,
        stable_key: str | None = None,
        limit: int = 100,
        connection: Any | None = None,
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
        result = self.execute(sql, tuple(params), connection=connection)
        return [_envelope_from_row(row) for row in result.rows]

    def get_historical_task_envelope(
        self,
        *,
        owner_user_id: str,
        dataset_id: str,
        task_id: str,
        object_revision: int,
        object_hash: str,
        envelope_id: str | None = None,
        connection: Any | None = None,
    ) -> SyncEnvelope | None:
        """Resolve one applied immutable task envelope from every anchor claim."""

        def _lookup(conn: Any) -> SyncEnvelope | None:
            dataset = self._require_dataset_domain(
                dataset_id,
                "notes.task",
                connection=conn,
            )
            if dataset.get("owner_user_id") != owner_user_id:
                return None
            if envelope_id is None:
                rows = self.execute(
                    """
                    SELECT * FROM sync_envelopes
                     WHERE dataset_id = ? AND domain = 'notes.task'
                       AND entity_id = ? AND object_revision = ? AND payload_hash = ?
                       AND operation = 'upsert' AND status = 'accepted'
                       AND apply_status = 'applied'
                     ORDER BY server_sequence DESC
                     LIMIT 2
                    """,
                    (dataset_id, task_id, object_revision, object_hash),
                    connection=conn,
                ).rows
                return _envelope_from_row(rows[0]) if len(rows) == 1 else None
            row = _first(
                self.execute(
                    """
                    SELECT * FROM sync_envelopes
                     WHERE dataset_id = ? AND client_envelope_id = ?
                       AND domain = 'notes.task' AND entity_id = ?
                       AND object_revision = ? AND payload_hash = ?
                       AND operation = 'upsert' AND status = 'accepted'
                       AND apply_status = 'applied'
                     LIMIT 1
                    """,
                    (
                        dataset_id,
                        envelope_id,
                        task_id,
                        object_revision,
                        object_hash,
                    ),
                    connection=conn,
                )
            )
            return _envelope_from_row(row) if row is not None else None

        if connection is not None:
            return _lookup(connection)
        with self.backend.transaction() as transaction_conn:
            return _lookup(transaction_conn)

    def get_projection_note_envelope(
        self,
        *,
        owner_user_id: str,
        dataset_id: str,
        note_id: str,
        envelope_id: str,
        object_hash: str,
        connection: Any | None = None,
    ) -> SyncEnvelope | None:
        """Resolve the exact applied note envelope named by a projection anchor."""

        def _lookup(conn: Any) -> SyncEnvelope | None:
            dataset = self._require_dataset_domain(
                dataset_id,
                "notes.note",
                connection=conn,
            )
            if dataset.get("owner_user_id") != owner_user_id:
                return None
            row = _first(
                self.execute(
                    """
                    SELECT * FROM sync_envelopes
                     WHERE dataset_id = ? AND client_envelope_id = ?
                       AND domain = 'notes.note' AND entity_id = ?
                       AND payload_hash = ? AND operation = 'upsert'
                       AND status = 'accepted' AND apply_status = 'applied'
                     LIMIT 1
                    """,
                    (dataset_id, envelope_id, note_id, object_hash),
                    connection=conn,
                )
            )
            return _envelope_from_row(row) if row is not None else None

        if connection is not None:
            return _lookup(connection)
        with self.backend.transaction() as transaction_conn:
            return _lookup(transaction_conn)

    def get_envelope_for_entity_at_or_before(
        self,
        dataset_id: str,
        domain: SyncDomain,
        *,
        entity_id: str,
        server_sequence: int,
    ) -> SyncEnvelope | None:
        """Return the newest accepted entity envelope at one durable boundary."""

        if server_sequence < 1:
            return None
        row = _first(
            self.execute(
                """
                SELECT * FROM sync_envelopes
                 WHERE dataset_id = ?
                   AND domain = ?
                   AND status = 'accepted'
                   AND entity_id = ?
                   AND server_sequence <= ?
                 ORDER BY server_sequence DESC
                 LIMIT 1
                """,
                (dataset_id, domain, entity_id, server_sequence),
            )
        )
        return _envelope_from_row(row) if row is not None else None

    def get_envelope_by_server_cursor(
        self,
        server_cursor: int,
        *,
        connection: Any | None = None,
    ) -> SyncEnvelope | None:
        row = _first(
            self.execute(
                "SELECT * FROM sync_envelopes WHERE server_sequence = ?",
                (server_cursor,),
                connection=connection,
            )
        )
        return _envelope_from_row(row) if row is not None else None

    def get_object_state(
        self,
        dataset_id: str,
        domain: SyncDomain,
        object_id: str,
        *,
        connection: Any | None = None,
    ) -> SyncObjectState | None:
        row = _first(
            self.execute(
                """
                SELECT * FROM sync_object_state
                 WHERE dataset_id = ? AND domain = ? AND object_id = ?
                """,
                (dataset_id, domain, object_id),
                connection=connection,
            )
        )
        if row is None:
            return None
        return _object_state_from_row(row)

    def get_current_head(
        self,
        dataset_id: str,
        domain: SyncDomain,
        object_id: str,
        *,
        connection: Any | None = None,
    ) -> SyncEnvelope | None:
        """Return one canonical head through the maintained head projection."""

        row = _first(
            self.execute(
                """
                SELECT envelope.*
                  FROM sync_current_heads AS head
                  JOIN sync_envelopes AS envelope
                    ON envelope.server_sequence = head.latest_server_cursor
                 WHERE head.dataset_id = ?
                   AND head.domain = ?
                   AND head.object_id = ?
                """,
                (dataset_id, domain, object_id),
                connection=connection,
            )
        )
        return _envelope_from_row(row) if row is not None else None

    def list_current_heads(
        self,
        dataset_id: str,
        domain: SyncDomain,
        *,
        limit: int,
        offset: int,
        connection: Any | None = None,
    ) -> list[SyncEnvelope]:
        """Return one bounded owner-scoped page from the head projection."""

        if limit < 1 or limit > 1000:
            raise SyncStoreError("Sync current-head limit must be between 1 and 1000")
        if offset < 0:
            raise SyncStoreError("Sync current-head offset must be non-negative")
        rows = self.execute(
            """
            SELECT envelope.*
              FROM sync_current_heads AS head
              JOIN sync_envelopes AS envelope
                ON envelope.server_sequence = head.latest_server_cursor
             WHERE head.dataset_id = ? AND head.domain = ?
             ORDER BY head.object_id ASC
             LIMIT ? OFFSET ?
            """,
            (dataset_id, domain, limit, offset),
            connection=connection,
        ).rows
        return [_envelope_from_row(row) for row in rows]

    def upsert_object_state(
        self,
        state: SyncObjectState,
        *,
        connection: Any | None = None,
        trusted_notes_task_bootstrap_id: str | None = None,
        trusted_notes_task_coordinator: bool = False,
    ) -> SyncObjectState:
        now = utcnow_iso()
        with self.backend.transaction(connection) as conn:
            if (
                state.domain in {"notes.task", "notes.task_activity"}
                and trusted_notes_task_bootstrap_id is not None
            ):
                row = self._require_dataset(state.dataset_id, connection=conn)
                metadata = decode_json(row.get("metadata_json"), default={})
                readiness = metadata.get(
                    "notes_task_v1"
                    if state.domain == "notes.task"
                    else "notes_task_activity_v1"
                )
                if (
                    not isinstance(readiness, Mapping)
                    or readiness.get("state") != "bootstrapping"
                    or metadata.get("task_activity_capture_enabled") is not True
                ):
                    raise SyncStoreError("notes_task_sync_not_ready")
            elif (
                state.domain in {"notes.task", "notes.task_activity"}
                and trusted_notes_task_coordinator
            ):
                row = self._require_dataset(state.dataset_id, connection=conn)
                metadata = decode_json(row.get("metadata_json"), default={})
                if not notes_task_capture_is_active(metadata):
                    raise SyncStoreError("notes_task_sync_not_ready")
            else:
                self._require_dataset_domain(
                    state.dataset_id, state.domain, connection=conn
                )
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
                WHERE excluded.latest_server_cursor > sync_object_state.latest_server_cursor
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
        stored = _object_state_from_row(row)
        if (
            stored.dataset_id,
            stored.domain,
            stored.object_id,
            stored.object_revision,
            stored.object_hash,
            stored.latest_server_cursor,
            stored.deleted,
        ) != (
            state.dataset_id,
            state.domain,
            state.object_id,
            state.object_revision,
            state.object_hash,
            state.latest_server_cursor,
            state.deleted,
        ):
            raise SyncStoreError("sync_object_state_stale_write")
        return stored

    def mark_envelope_apply_status(
        self,
        server_cursor: int,
        *,
        apply_status: SyncApplyStatus,
        apply_error_code: str | None = None,
        apply_error_message: str | None = None,
        connection: Any | None = None,
    ) -> SyncEnvelope:
        if apply_status not in SYNC_APPLY_STATUSES:
            raise SyncStoreError(f"Invalid Sync envelope apply status: {apply_status}")
        now = utcnow_iso()
        applied_at = now if apply_status == "applied" else None
        with self.backend.transaction(connection) as conn:
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

    def mark_bootstrap_envelope_verified(
        self,
        server_cursor: int,
        *,
        bootstrap_id: str,
        notes_task_bootstrap: bool = False,
        connection: Any | None = None,
    ) -> SyncEnvelope:
        """Atomically record a source-verified bootstrap step without product replay."""

        if connection is None:
            source = _first(
                self.execute(
                    "SELECT dataset_id, domain, entity_id FROM sync_envelopes "
                    "WHERE server_sequence = ?",
                    (server_cursor,),
                )
            )
            if source is None:
                raise SyncStoreError(
                    f"Sync envelope not found for server cursor: {server_cursor}"
                )
            with self.materialization_transaction(
                [(str(source["dataset_id"]), source["domain"], str(source["entity_id"]))]
            ) as guarded_connection:
                return self.mark_bootstrap_envelope_verified(
                    server_cursor,
                    bootstrap_id=bootstrap_id,
                    notes_task_bootstrap=notes_task_bootstrap,
                    connection=guarded_connection,
                )

        with self.backend.transaction(connection) as conn:
            row = _first(
                self.execute(
                    "SELECT * FROM sync_envelopes WHERE server_sequence = ?",
                    (server_cursor,),
                    connection=conn,
                )
            )
            if row is None:
                raise SyncStoreError(f"Sync envelope not found for server cursor: {server_cursor}")
            if row["domain"] == "notes.task" and notes_task_bootstrap:
                dataset_row = self._require_dataset(
                    str(row["dataset_id"]), connection=conn
                )
                self._require_notes_task_bootstrap_write_ready(
                    dataset_row,
                    envelope=_envelope_from_row(row),
                    bootstrap_id=bootstrap_id,
                )
            else:
                dataset_row = self._require_dataset_domain(
                    str(row["dataset_id"]), row["domain"], connection=conn
                )
            self._require_notes_organization_write_ready(
                dataset_row,
                row["domain"],
                trusted_bootstrap_id=bootstrap_id,
            )
            now = utcnow_iso()
            self.upsert_object_state(
                SyncObjectState(
                    dataset_id=str(row["dataset_id"]),
                    domain=row["domain"],
                    object_id=str(row["entity_id"]),
                    object_revision=int(row.get("object_revision") or 1),
                    object_hash=str(row.get("payload_hash") or ""),
                    latest_server_cursor=server_cursor,
                    deleted=row.get("operation") == "tombstone",
                ),
                connection=conn,
                trusted_notes_task_bootstrap_id=(
                    bootstrap_id if notes_task_bootstrap else None
                ),
            )
            self.execute(
                "UPDATE sync_envelopes SET apply_status = 'applied', apply_error_code = NULL, "
                "apply_error_message = NULL, applied_at = ? WHERE server_sequence = ?",
                (now, server_cursor),
                connection=conn,
            )
            updated = _first(
                self.execute(
                    "SELECT * FROM sync_envelopes WHERE server_sequence = ?",
                    (server_cursor,),
                    connection=conn,
                )
            )
        return _envelope_from_row(updated)

    def reconcile_bootstrap_envelope_superseded(
        self,
        server_cursor: int,
        *,
        bootstrap_id: str,
        superseded_by_cursor: int,
        connection: Any | None = None,
    ) -> SyncEnvelope:
        """Audit-reconcile a stale step after its correction is current and applied."""

        if superseded_by_cursor <= server_cursor:
            raise SyncStoreError("Bootstrap correction must follow the stale step")
        if connection is None:
            source = _first(
                self.execute(
                    "SELECT dataset_id, domain, entity_id FROM sync_envelopes "
                    "WHERE server_sequence = ?",
                    (server_cursor,),
                )
            )
            if source is None:
                raise SyncStoreError("Bootstrap reconciliation envelope was not found")
            with self.materialization_transaction(
                [(str(source["dataset_id"]), source["domain"], str(source["entity_id"]))]
            ) as guarded_connection:
                return self.reconcile_bootstrap_envelope_superseded(
                    server_cursor,
                    bootstrap_id=bootstrap_id,
                    superseded_by_cursor=superseded_by_cursor,
                    connection=guarded_connection,
                )
        with self.backend.transaction(connection) as conn:
            row = _first(
                self.execute(
                    "SELECT * FROM sync_envelopes WHERE server_sequence = ?",
                    (server_cursor,),
                    connection=conn,
                )
            )
            correction = _first(
                self.execute(
                    "SELECT * FROM sync_envelopes WHERE server_sequence = ?",
                    (superseded_by_cursor,),
                    connection=conn,
                )
            )
            if row is None or correction is None:
                raise SyncStoreError("Bootstrap reconciliation envelope was not found")
            dataset_row = self._require_dataset_domain(
                str(row["dataset_id"]), row["domain"], connection=conn
            )
            self._require_notes_organization_write_ready(
                dataset_row,
                row["domain"],
                trusted_bootstrap_id=bootstrap_id,
            )
            correction_metadata = decode_json(
                correction.get("routing_metadata_json"), default={}
            )
            if (
                correction.get("dataset_id") != row.get("dataset_id")
                or correction.get("domain") != row.get("domain")
                or correction.get("entity_id") != row.get("entity_id")
                or correction.get("status") != "accepted"
                or correction.get("apply_status") != "applied"
                or correction_metadata.get("source")
                != "notes-organization-bootstrap"
            ):
                raise SyncStoreError("Bootstrap correction is not durably applied")
            head = _first(
                self.execute(
                    """
                    SELECT latest_server_cursor FROM sync_current_heads
                     WHERE dataset_id = ? AND domain = ? AND object_id = ?
                    """,
                    (row["dataset_id"], row["domain"], row["entity_id"]),
                    connection=conn,
                )
            )
            if head is None or int(head["latest_server_cursor"]) != superseded_by_cursor:
                raise SyncStoreError("Bootstrap correction is not the current head")
            now = utcnow_iso()
            self.execute(
                """
                UPDATE sync_envelopes
                   SET apply_status = 'applied',
                       apply_error_code = 'sync_bootstrap_superseded',
                       apply_error_message = NULL,
                       applied_at = ?
                 WHERE server_sequence = ?
                """,
                (now, server_cursor),
                connection=conn,
            )
            updated = _first(
                self.execute(
                    "SELECT * FROM sync_envelopes WHERE server_sequence = ?",
                    (server_cursor,),
                    connection=conn,
                )
            )
        return _envelope_from_row(updated)

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
            self._require_device_for_dataset(
                cursor.dataset_id, cursor.device_id, connection=conn
            )
            existing = _first(
                self.execute(
                    """
                    SELECT * FROM sync_device_adapter_cursors
                     WHERE dataset_id = ? AND device_id = ? AND domain = ?
                       AND adapter_version = ?
                    """,
                    (
                        cursor.dataset_id,
                        cursor.device_id,
                        cursor.domain,
                        cursor.adapter_version,
                    ),
                    connection=conn,
                )
            )
            last_pulled = max(
                cursor.last_pulled_sequence,
                int((existing or {}).get("last_pulled_sequence") or 0),
            )
            max_delivered = max(
                cursor.max_delivered_sequence,
                int((existing or {}).get("max_delivered_sequence") or 0),
            )
            self.execute(
                """
                INSERT INTO sync_device_adapter_cursors (
                    dataset_id, device_id, domain, adapter_version,
                    last_pulled_sequence, max_delivered_sequence, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (dataset_id, device_id, domain, adapter_version)
                DO UPDATE SET last_pulled_sequence = CASE
                                  WHEN excluded.last_pulled_sequence >
                                       sync_device_adapter_cursors.last_pulled_sequence
                                  THEN excluded.last_pulled_sequence
                                  ELSE sync_device_adapter_cursors.last_pulled_sequence END,
                              max_delivered_sequence = CASE
                                  WHEN excluded.max_delivered_sequence >
                                       sync_device_adapter_cursors.max_delivered_sequence
                                  THEN excluded.max_delivered_sequence
                                  ELSE sync_device_adapter_cursors.max_delivered_sequence END,
                              updated_at = excluded.updated_at
                """,
                (
                    cursor.dataset_id,
                    cursor.device_id,
                    cursor.domain,
                    cursor.adapter_version,
                    last_pulled,
                    max_delivered,
                    now,
                ),
                connection=conn,
            )
            if cursor.adapter_version == 1:
                legacy = _first(
                    self.execute(
                        """
                        SELECT * FROM sync_device_cursors
                         WHERE dataset_id = ? AND device_id = ? AND domain = ?
                        """,
                        (cursor.dataset_id, cursor.device_id, cursor.domain),
                        connection=conn,
                    )
                )
                legacy_last_pulled = int(
                    (legacy or {}).get("last_pulled_sequence") or 0
                )
                if legacy_last_pulled > int(
                    (existing or {}).get("last_pulled_sequence") or 0
                ):
                    max_delivered = max(max_delivered, legacy_last_pulled)
                last_pulled = max(
                    last_pulled,
                    legacy_last_pulled,
                )
                self.execute(
                    """
                    INSERT INTO sync_device_cursors (
                        dataset_id, device_id, domain, last_pulled_sequence, updated_at
                    ) VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT (dataset_id, device_id, domain)
                    DO UPDATE SET last_pulled_sequence = CASE
                                      WHEN excluded.last_pulled_sequence >
                                           sync_device_cursors.last_pulled_sequence
                                      THEN excluded.last_pulled_sequence
                                      ELSE sync_device_cursors.last_pulled_sequence END,
                                  updated_at = excluded.updated_at
                    """,
                    (
                        cursor.dataset_id,
                        cursor.device_id,
                        cursor.domain,
                        last_pulled,
                        now,
                    ),
                    connection=conn,
                )
                self.execute(
                    """
                    UPDATE sync_device_adapter_cursors
                       SET last_pulled_sequence = CASE
                               WHEN last_pulled_sequence < ?
                               THEN (? + 0) ELSE last_pulled_sequence END,
                           max_delivered_sequence = CASE
                               WHEN max_delivered_sequence > ?
                               THEN max_delivered_sequence ELSE (? + 0) END,
                           updated_at = ?
                     WHERE dataset_id = ? AND device_id = ? AND domain = ?
                       AND adapter_version = 1
                    """,
                    (
                        last_pulled,
                        last_pulled,
                        max_delivered,
                        max_delivered,
                        now,
                        cursor.dataset_id,
                        cursor.device_id,
                        cursor.domain,
                    ),
                    connection=conn,
                )
            row = _first(
                self.execute(
                    """
                    SELECT * FROM sync_device_adapter_cursors
                     WHERE dataset_id = ? AND device_id = ? AND domain = ?
                       AND adapter_version = ?
                    """,
                    (
                        cursor.dataset_id,
                        cursor.device_id,
                        cursor.domain,
                        cursor.adapter_version,
                    ),
                    connection=conn,
                )
            )
        return _cursor_from_row(row)

    def get_device_cursor(
        self,
        dataset_id: str,
        device_id: str,
        domain: SyncDomain,
        *,
        adapter_version: int = 1,
    ) -> SyncDeviceCursor | None:
        with self.backend.transaction() as conn:
            self._require_dataset_domain(dataset_id, domain, connection=conn)
            self._require_device_for_dataset(dataset_id, device_id, connection=conn)
            row = _first(
                self.execute(
                    """
                    SELECT * FROM sync_device_adapter_cursors
                     WHERE dataset_id = ? AND device_id = ? AND domain = ?
                       AND adapter_version = ?
                    """,
                    (dataset_id, device_id, domain, adapter_version),
                    connection=conn,
                )
            )
            if adapter_version == 1:
                legacy = _first(
                    self.execute(
                        """
                        SELECT * FROM sync_device_cursors
                         WHERE dataset_id = ? AND device_id = ? AND domain = ?
                        """,
                        (dataset_id, device_id, domain),
                        connection=conn,
                    )
                )
                if legacy is not None and (
                    row is None
                    or int(legacy["last_pulled_sequence"])
                    > int(row["last_pulled_sequence"])
                ):
                    row = {
                        **legacy,
                        "adapter_version": 1,
                        "max_delivered_sequence": legacy["last_pulled_sequence"],
                    }
        if row is None:
            return None
        return _cursor_from_row(row)

    def insert_conflict(
        self,
        conflict: SyncConflictCreate,
        *,
        connection: Any | None = None,
    ) -> SyncConflict:
        now = utcnow_iso()
        with self.backend.transaction(connection) as conn:
            self._require_dataset_domain(conflict.dataset_id, conflict.domain, connection=conn)
            existing = _first(
                self.execute(
                    "SELECT * FROM sync_conflicts WHERE conflict_id = ?",
                    (conflict.conflict_id,),
                    connection=conn,
                )
            )
            if existing:
                return self._require_matching_conflict(existing, conflict)
            if conflict.local_envelope_id is not None and conflict.server_sequence is not None:
                existing = _first(
                    self.execute(
                        """
                        SELECT * FROM sync_conflicts
                         WHERE dataset_id = ?
                           AND local_envelope_id = ?
                           AND server_sequence = ?
                        """,
                        (
                            conflict.dataset_id,
                            conflict.local_envelope_id,
                            conflict.server_sequence,
                        ),
                        connection=conn,
                    )
                )
                if existing is not None:
                    return self._require_matching_conflict(existing, conflict)
            self.execute(
                """
                INSERT INTO sync_conflicts (
                    conflict_id, dataset_id, domain, entity_id, conflict_type,
                    status, base_envelope_id, local_envelope_id, remote_envelope_id,
                    server_sequence, metadata_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT DO NOTHING
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
            if (
                row is None
                and conflict.local_envelope_id is not None
                and conflict.server_sequence is not None
            ):
                row = _first(
                    self.execute(
                        """
                        SELECT * FROM sync_conflicts
                         WHERE dataset_id = ?
                           AND local_envelope_id = ?
                           AND server_sequence = ?
                        """,
                        (
                            conflict.dataset_id,
                            conflict.local_envelope_id,
                            conflict.server_sequence,
                        ),
                        connection=conn,
                    )
                )
            if row is None:
                raise SyncStoreError("Sync conflict could not be stored")
            return self._require_matching_conflict(row, conflict)

    @staticmethod
    def _require_matching_conflict(
        row: dict[str, Any],
        conflict: SyncConflictCreate,
    ) -> SyncConflict:
        expected = (
            conflict.dataset_id,
            conflict.domain,
            conflict.entity_id,
            conflict.conflict_type,
            conflict.base_envelope_id,
            conflict.local_envelope_id,
            conflict.remote_envelope_id,
            conflict.server_sequence,
            dict(conflict.metadata),
        )
        actual = (
            row.get("dataset_id"),
            row.get("domain"),
            row.get("entity_id"),
            row.get("conflict_type"),
            row.get("base_envelope_id"),
            row.get("local_envelope_id"),
            row.get("remote_envelope_id"),
            row.get("server_sequence"),
            decode_json(row.get("metadata_json"), default={}),
        )
        if actual != expected:
            raise SyncIdempotencyConflictError(
                "Sync conflict identity was reused with different content"
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

    def get_conflict(
        self,
        conflict_id: str,
        *,
        connection: Any | None = None,
    ) -> SyncConflict | None:
        """Return a conflict by ID without scanning dataset conflict lists."""

        row = _first(
            self.execute(
                "SELECT * FROM sync_conflicts WHERE conflict_id = ?",
                (conflict_id,),
                connection=connection,
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
        connection: Any | None = None,
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
        row = _first(self.execute(sql, tuple(params), connection=connection))
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
        connection: Any | None = None,
    ) -> SyncConflict:
        with self.backend.transaction(connection) as conn:
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
        connection: Any | None = None,
    ) -> SyncConflict:
        with self.backend.transaction(connection) as conn:
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
        connection: Any | None = None,
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
        with self.backend.transaction(connection) as conn:
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

    def revoke_key_record(self, *, user_id: str, key_record_id: str) -> SyncKeyRecord:
        """Revoke one device-wrapped key record without revoking the device."""

        now = utcnow_iso()
        with self.backend.transaction() as conn:
            self.execute(
                """
                UPDATE sync_key_records
                   SET revoked_at = COALESCE(revoked_at, ?)
                 WHERE user_id = ? AND key_record_id = ?
                """,
                (now, user_id, key_record_id),
                connection=conn,
            )
            row = _first(
                self.execute(
                    "SELECT * FROM sync_key_records WHERE user_id = ? AND key_record_id = ?",
                    (user_id, key_record_id),
                    connection=conn,
                )
            )
            if row is None:
                raise SyncStoreError("Sync key record was not found or is not accessible")
        return _key_record_from_row(row)

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

    def _create_attachment_binding_for_envelope(
        self,
        envelope: SyncEnvelope,
        *,
        connection: Any,
    ) -> SyncAttachmentRevisionBinding:
        """Observe blob availability and bind one accepted v2 revision atomically."""

        payload = envelope.payload or envelope.payload_clear
        attachment_id = str(payload.get("attachment_id") or "")
        blob_hash = str(payload.get("blob_hash") or "")
        size_value = payload.get("size_bytes")
        if (
            attachment_id != envelope.object_id
            or isinstance(size_value, bool)
            or not isinstance(size_value, int)
            or envelope.object_revision is None
            or envelope.server_cursor is None
        ):
            raise SyncStoreError("attachment.ref v2 binding metadata is invalid")
        suffix = " FOR UPDATE" if self.backend_type == BackendType.POSTGRESQL else ""
        blob_row = _first(
            self.execute(
                """
                SELECT blob.blob_id, blob.status
                  FROM sync_blob_objects AS blob
                  JOIN sync_datasets AS dataset
                    ON dataset.dataset_id = blob.dataset_id
                 WHERE blob.dataset_id = ?
                   AND blob.payload_hash = ?
                   AND blob.size_bytes = ?
                   AND (
                        dataset.scope_type = 'workspace'
                        OR blob.owner_user_id = dataset.owner_user_id
                   )
                 ORDER BY blob.blob_id ASC
                 LIMIT 1
                """
                + suffix,  # nosec B608 - backend-controlled row lock suffix.
                (envelope.dataset_id, blob_hash, size_value),
                connection=connection,
            )
        )
        if blob_row is not None and blob_row.get("status") in {"deleting", "deleted"}:
            raise SyncStoreError(
                f"Sync attachment binding cannot target a {blob_row['status']} blob"
            )
        blob_available = blob_row is not None and blob_row.get("status") == "available"
        binding = SyncAttachmentRevisionBindingCreate(
            dataset_id=envelope.dataset_id,
            attachment_id=attachment_id,
            attachment_revision=envelope.object_revision,
            blob_hash=blob_hash,
            size_bytes=size_value,
            establishing_server_cursor=envelope.server_cursor,
            availability_at_acceptance=(
                "available" if blob_available else "metadata_only"
            ),
            resolved_blob_id=(None if not blob_available else str(blob_row["blob_id"])),
        )
        return self._create_attachment_revision_binding(
            binding,
            connection=connection,
        )

    @staticmethod
    def _binding_identity_matches(
        row: Mapping[str, Any],
        binding: SyncAttachmentRevisionBindingCreate,
    ) -> bool:
        return (
            row.get("dataset_id") == binding.dataset_id
            and row.get("attachment_id") == binding.attachment_id
            and int(row.get("attachment_revision") or 0)
            == binding.attachment_revision
            and row.get("blob_hash") == binding.blob_hash
            and int(row.get("size_bytes") or 0) == binding.size_bytes
            and int(row.get("establishing_server_cursor") or 0)
            == binding.establishing_server_cursor
            and row.get("availability_at_acceptance")
            == binding.availability_at_acceptance
        )

    def _create_attachment_revision_binding(
        self,
        binding: SyncAttachmentRevisionBindingCreate,
        *,
        connection: Any,
    ) -> SyncAttachmentRevisionBinding:
        if binding.resolved_blob_id is not None:
            self._require_exact_available_blob_for_binding(
                binding,
                binding.resolved_blob_id,
                connection=connection,
            )
        self.execute(
            """
            INSERT INTO sync_attachment_revision_bindings (
                dataset_id, attachment_id, attachment_revision, blob_hash,
                size_bytes, establishing_server_cursor,
                availability_at_acceptance, resolved_blob_id,
                retention_released_at, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?)
            ON CONFLICT (dataset_id, attachment_id, attachment_revision) DO NOTHING
            """,
            (
                binding.dataset_id,
                binding.attachment_id,
                binding.attachment_revision,
                binding.blob_hash,
                binding.size_bytes,
                binding.establishing_server_cursor,
                binding.availability_at_acceptance,
                binding.resolved_blob_id,
                utcnow_iso(),
            ),
            connection=connection,
        )
        row = _first(
            self.execute(
                """
                SELECT * FROM sync_attachment_revision_bindings
                 WHERE dataset_id = ? AND attachment_id = ?
                   AND attachment_revision = ?
                """,
                (
                    binding.dataset_id,
                    binding.attachment_id,
                    binding.attachment_revision,
                ),
                connection=connection,
            )
        )
        if row is None:
            raise SyncStoreError(
                "Sync attachment binding insert did not produce a retrievable record"
            )
        if not self._binding_identity_matches(row, binding):
            raise SyncIdempotencyConflictError(
                "Sync attachment revision identity was reused with different content"
            )
        if (
            binding.resolved_blob_id is not None
            and row.get("resolved_blob_id") != binding.resolved_blob_id
        ):
            raise SyncStoreError("Sync attachment binding cannot be rebound")
        return _attachment_revision_binding_from_row(row)

    def get_attachment_revision_binding(
        self,
        dataset_id: str,
        attachment_id: str,
        attachment_revision: int,
        *,
        owner_user_id: str,
        connection: Any | None = None,
    ) -> SyncAttachmentRevisionBinding | None:
        """Return one dataset-scoped immutable attachment revision binding."""

        with self.backend.transaction(connection) as conn:
            self._require_attachment_binding_dataset_owner(
                dataset_id,
                owner_user_id,
                connection=conn,
            )
            row = _first(
                self.execute(
                    """
                    SELECT * FROM sync_attachment_revision_bindings
                     WHERE dataset_id = ? AND attachment_id = ?
                       AND attachment_revision = ?
                    """,
                    (dataset_id, attachment_id, attachment_revision),
                    connection=conn,
                )
            )
        return None if row is None else _attachment_revision_binding_from_row(row)

    def get_attachment_revision_binding_for_blob(
        self,
        dataset_id: str,
        blob_id: str,
        *,
        owner_user_id: str,
        connection: Any | None = None,
    ) -> SyncAttachmentRevisionBinding | None:
        """Return the immutable revision binding that resolved to one blob ID."""

        with self.backend.transaction(connection) as conn:
            self._require_attachment_binding_dataset_owner(
                dataset_id,
                owner_user_id,
                connection=conn,
            )
            rows = self.execute(
                """
                SELECT * FROM sync_attachment_revision_bindings
                 WHERE dataset_id = ? AND resolved_blob_id = ?
                 ORDER BY establishing_server_cursor DESC
                 LIMIT 1
                """,
                (dataset_id, blob_id),
                connection=conn,
            ).rows
        return None if not rows else _attachment_revision_binding_from_row(rows[0])

    def list_attachment_revision_bindings_for_blob(
        self,
        dataset_id: str,
        blob_id: str,
        *,
        owner_user_id: str,
        after_establishing_server_cursor: int = 0,
        after_attachment_id: str = "",
        after_attachment_revision: int = 0,
        limit: int = 1000,
        connection: Any | None = None,
    ) -> list[SyncAttachmentRevisionBinding]:
        """Return one bounded keyset page of unreleased bindings for a blob."""

        if isinstance(limit, bool) or limit < 1:
            raise SyncStoreError("Sync attachment binding page limit must be positive")
        page_limit = min(limit, 1000)
        with self.backend.transaction(connection) as conn:
            self._require_attachment_binding_dataset_owner(
                dataset_id,
                owner_user_id,
                connection=conn,
            )
            rows = self.execute(
                """
                SELECT * FROM sync_attachment_revision_bindings
                 WHERE dataset_id = ? AND resolved_blob_id = ?
                   AND retention_released_at IS NULL
                   AND (establishing_server_cursor, attachment_id,
                        attachment_revision) > (?, ?, ?)
                 ORDER BY establishing_server_cursor, attachment_id,
                          attachment_revision
                 LIMIT ?
                """,
                (
                    dataset_id,
                    blob_id,
                    after_establishing_server_cursor,
                    after_attachment_id,
                    after_attachment_revision,
                    page_limit,
                ),
                connection=conn,
            ).rows
        return [_attachment_revision_binding_from_row(row) for row in rows]

    def list_unreleased_attachment_revision_bindings(
        self,
        dataset_id: str,
        *,
        owner_user_id: str,
        after_establishing_server_cursor: int = 0,
        after_attachment_id: str = "",
        after_attachment_revision: int = 0,
        limit: int = 1000,
        connection: Any | None = None,
    ) -> list[SyncAttachmentRevisionBinding]:
        """Return one bounded keyset page of unreleased dataset bindings."""

        if isinstance(limit, bool) or limit < 1:
            raise SyncStoreError("Sync attachment binding page limit must be positive")
        page_limit = min(limit, 1000)
        with self.backend.transaction(connection) as conn:
            self._require_attachment_binding_dataset_owner(
                dataset_id,
                owner_user_id,
                connection=conn,
            )
            rows = self.execute(
                """
                SELECT binding.* FROM sync_attachment_revision_bindings AS binding
                 WHERE binding.dataset_id = ?
                   AND binding.retention_released_at IS NULL
                   AND NOT EXISTS (
                        SELECT 1
                          FROM sync_current_heads AS head
                          JOIN sync_envelopes AS envelope
                            ON envelope.server_sequence = head.latest_server_cursor
                         WHERE head.dataset_id = binding.dataset_id
                           AND head.domain = 'attachment.ref'
                           AND head.object_id = binding.attachment_id
                           AND envelope.adapter_version = 2
                           AND envelope.object_revision = binding.attachment_revision
                           AND envelope.operation <> 'tombstone'
                   )
                   AND (binding.establishing_server_cursor, binding.attachment_id,
                        binding.attachment_revision) > (?, ?, ?)
                 ORDER BY binding.establishing_server_cursor, binding.attachment_id,
                          binding.attachment_revision
                 LIMIT ?
                """,
                (
                    dataset_id,
                    after_establishing_server_cursor,
                    after_attachment_id,
                    after_attachment_revision,
                    page_limit,
                ),
                connection=conn,
            ).rows
        return [_attachment_revision_binding_from_row(row) for row in rows]

    def has_attachment_ref_v2_history(
        self,
        dataset_id: str,
        attachment_id: str,
        *,
        owner_user_id: str,
        connection: Any | None = None,
    ) -> bool:
        """Return whether an attachment identity has accepted adapter-v2 history."""

        with self.backend.transaction(connection) as conn:
            self._require_attachment_binding_dataset_owner(
                dataset_id,
                owner_user_id,
                connection=conn,
            )
            row = _first(
                self.execute(
                    """
                    SELECT 1 FROM sync_envelopes
                     WHERE dataset_id = ?
                       AND domain = 'attachment.ref'
                       AND entity_id = ?
                       AND adapter_version = 2
                       AND status = 'accepted'
                     LIMIT 1
                    """,
                    (dataset_id, attachment_id),
                    connection=conn,
                )
            )
        return row is not None

    def list_unresolved_attachment_revision_bindings(
        self,
        dataset_id: str,
        *,
        owner_user_id: str,
        after_establishing_server_cursor: int = 0,
        limit: int = 1000,
    ) -> list[SyncAttachmentRevisionBinding]:
        """Return one bounded keyset page of unreleased unresolved bindings."""

        if isinstance(limit, bool) or limit < 1:
            raise SyncStoreError("Sync attachment binding page limit must be positive")
        page_limit = min(limit, 1000)
        with self.backend.transaction() as conn:
            self._require_attachment_binding_dataset_owner(
                dataset_id,
                owner_user_id,
                connection=conn,
            )
            rows = self.execute(
                """
                SELECT * FROM sync_attachment_revision_bindings
                 WHERE dataset_id = ?
                   AND resolved_blob_id IS NULL
                   AND retention_released_at IS NULL
                   AND establishing_server_cursor > ?
                 ORDER BY establishing_server_cursor, attachment_id,
                          attachment_revision
                 LIMIT ?
                """,
                (dataset_id, after_establishing_server_cursor, page_limit),
                connection=conn,
            ).rows
        return [_attachment_revision_binding_from_row(row) for row in rows]

    def _require_exact_available_blob_for_binding(
        self,
        binding: SyncAttachmentRevisionBindingCreate | SyncAttachmentRevisionBinding,
        blob_id: str,
        *,
        connection: Any,
    ) -> dict[str, Any]:
        suffix = " FOR UPDATE" if self.backend_type == BackendType.POSTGRESQL else ""
        row = _first(
            self.execute(
                """
                SELECT blob.*
                  FROM sync_blob_objects AS blob
                  JOIN sync_datasets AS dataset
                    ON dataset.dataset_id = blob.dataset_id
                 WHERE blob.dataset_id = ? AND blob.blob_id = ?
                   AND (
                        dataset.scope_type = 'workspace'
                        OR blob.owner_user_id = dataset.owner_user_id
                   )
                """
                + suffix,  # nosec B608 - backend-controlled row lock suffix.
                (binding.dataset_id, blob_id),
                connection=connection,
            )
        )
        if (
            row is None
            or row.get("status") != "available"
            or row.get("payload_hash") != binding.blob_hash
            or int(row.get("size_bytes") or 0) != binding.size_bytes
        ):
            raise SyncStoreError(
                "Sync attachment binding requires an exact available blob"
            )
        return row

    def resolve_attachment_revision_binding(
        self,
        dataset_id: str,
        attachment_id: str,
        attachment_revision: int,
        *,
        blob_id: str,
        owner_user_id: str,
    ) -> SyncAttachmentRevisionBinding:
        """CAS one pending binding to one exact verified blob ID."""

        suffix = " FOR UPDATE" if self.backend_type == BackendType.POSTGRESQL else ""
        with self.backend.transaction() as conn:
            self._require_dataset_owner_for_update(
                dataset_id,
                owner_user_id,
                connection=conn,
            )
            row = _first(
                self.execute(
                    """
                    SELECT * FROM sync_attachment_revision_bindings
                     WHERE dataset_id = ? AND attachment_id = ?
                       AND attachment_revision = ?
                    """
                    + suffix,  # nosec B608 - backend-controlled row lock suffix.
                    (dataset_id, attachment_id, attachment_revision),
                    connection=conn,
                )
            )
            if row is None:
                raise SyncStoreError("Sync attachment binding was not found")
            binding = _attachment_revision_binding_from_row(row)
            if binding.resolved_blob_id is not None:
                if binding.resolved_blob_id != blob_id:
                    raise SyncStoreError("Sync attachment binding cannot be rebound")
                return binding
            if binding.retention_released_at is not None:
                raise SyncStoreError("Sync attachment binding retention was released")
            self._require_exact_available_blob_for_binding(
                binding,
                blob_id,
                connection=conn,
            )
            updated = self.execute(
                """
                UPDATE sync_attachment_revision_bindings
                   SET resolved_blob_id = ?
                 WHERE dataset_id = ? AND attachment_id = ?
                   AND attachment_revision = ? AND resolved_blob_id IS NULL
                   AND retention_released_at IS NULL
                """,
                (
                    blob_id,
                    dataset_id,
                    attachment_id,
                    attachment_revision,
                ),
                connection=conn,
            )
            if updated.rowcount != 1:
                raise SyncStoreError("Sync attachment binding resolution CAS failed")
            resolved_row = _first(
                self.execute(
                    """
                    SELECT * FROM sync_attachment_revision_bindings
                     WHERE dataset_id = ? AND attachment_id = ?
                       AND attachment_revision = ?
                    """,
                    (dataset_id, attachment_id, attachment_revision),
                    connection=conn,
                )
            )
        if resolved_row is None:
            raise SyncStoreError("Sync attachment binding resolution was not durable")
        return _attachment_revision_binding_from_row(resolved_row)

    def release_attachment_revision_binding(
        self,
        dataset_id: str,
        attachment_id: str,
        attachment_revision: int,
        *,
        released_at: str,
        owner_user_id: str,
        connection: Any | None = None,
    ) -> SyncAttachmentRevisionBinding:
        """Set the retention-release marker once without erasing audit identity."""

        if not released_at.strip():
            raise SyncStoreError("Sync attachment binding release timestamp is required")
        suffix = " FOR UPDATE" if self.backend_type == BackendType.POSTGRESQL else ""
        with self.backend.transaction(connection) as conn:
            self._require_dataset_owner_for_update(
                dataset_id,
                owner_user_id,
                connection=conn,
            )
            row = _first(
                self.execute(
                    """
                    SELECT * FROM sync_attachment_revision_bindings
                     WHERE dataset_id = ? AND attachment_id = ?
                       AND attachment_revision = ?
                    """
                    + suffix,  # nosec B608 - backend-controlled row lock suffix.
                    (dataset_id, attachment_id, attachment_revision),
                    connection=conn,
                )
            )
            if row is None:
                raise SyncStoreError("Sync attachment binding was not found")
            if row.get("retention_released_at") is None:
                self.execute(
                    """
                    UPDATE sync_attachment_revision_bindings
                       SET retention_released_at = ?
                     WHERE dataset_id = ? AND attachment_id = ?
                       AND attachment_revision = ?
                       AND retention_released_at IS NULL
                    """,
                    (released_at, dataset_id, attachment_id, attachment_revision),
                    connection=conn,
                )
                row = _first(
                    self.execute(
                        """
                        SELECT * FROM sync_attachment_revision_bindings
                         WHERE dataset_id = ? AND attachment_id = ?
                           AND attachment_revision = ?
                        """,
                        (dataset_id, attachment_id, attachment_revision),
                        connection=conn,
                    )
                )
        if row is None:
            raise SyncStoreError("Sync attachment binding release was not durable")
        return _attachment_revision_binding_from_row(row)

    def get_or_create_storage_namespace(
        self,
        dataset_id: str,
        *,
        owner_user_id: str,
    ) -> SyncDatasetStorageNamespace:
        """Return one server-issued opaque namespace under dataset owner authority."""

        suffix = " FOR UPDATE" if self.backend_type == BackendType.POSTGRESQL else ""
        with self.backend.transaction() as conn:
            self._require_dataset_owner_for_update(
                dataset_id,
                owner_user_id,
                connection=conn,
            )
            row = _first(
                self.execute(
                    """
                    SELECT * FROM sync_dataset_storage_namespaces
                     WHERE dataset_id = ? AND owner_user_id = ?
                    """
                    + suffix,  # nosec B608 - backend-controlled row lock suffix.
                    (dataset_id, owner_user_id),
                    connection=conn,
                )
            )
            if row is None:
                self.execute(
                    """
                    INSERT INTO sync_dataset_storage_namespaces (
                        dataset_id, owner_user_id, storage_namespace_id, created_at
                    ) VALUES (?, ?, ?, ?)
                    ON CONFLICT (dataset_id) DO NOTHING
                    """,
                    (dataset_id, owner_user_id, uuid4().hex, utcnow_iso()),
                    connection=conn,
                )
                row = _first(
                    self.execute(
                        """
                        SELECT * FROM sync_dataset_storage_namespaces
                         WHERE dataset_id = ? AND owner_user_id = ?
                        """,
                        (dataset_id, owner_user_id),
                        connection=conn,
                    )
                )
        if row is None:
            raise SyncStoreError("Sync dataset storage namespace could not be resolved")
        return _storage_namespace_from_row(row)

    def get_storage_namespace(
        self,
        dataset_id: str,
        *,
        owner_user_id: str,
        connection: Any | None = None,
    ) -> SyncDatasetStorageNamespace | None:
        """Return the existing opaque namespace under exact owner authority."""

        with self.backend.transaction(connection) as conn:
            row = _first(
                self.execute(
                    """
                    SELECT namespace.*
                      FROM sync_dataset_storage_namespaces AS namespace
                      JOIN sync_datasets AS dataset
                        ON dataset.dataset_id = namespace.dataset_id
                     WHERE namespace.dataset_id = ?
                       AND namespace.owner_user_id = ?
                       AND dataset.owner_user_id = ?
                    """,
                    (dataset_id, owner_user_id, owner_user_id),
                    connection=conn,
                )
            )
        return None if row is None else _storage_namespace_from_row(row)

    def _resolve_pending_bindings_for_blob(
        self,
        blob_row: Mapping[str, Any],
        *,
        connection: Any,
    ) -> None:
        if blob_row.get("status") != "available":
            return
        suffix = " FOR UPDATE" if self.backend_type == BackendType.POSTGRESQL else ""
        authoritative_blob = _first(
            self.execute(
                """
                SELECT blob.*
                  FROM sync_blob_objects AS blob
                  JOIN sync_datasets AS dataset
                    ON dataset.dataset_id = blob.dataset_id
                 WHERE blob.dataset_id = ? AND blob.blob_id = ?
                   AND blob.status = 'available'
                   AND (
                        dataset.scope_type = 'workspace'
                        OR blob.owner_user_id = dataset.owner_user_id
                   )
                """
                + suffix,  # nosec B608 - backend-controlled row lock suffix.
                (blob_row["dataset_id"], blob_row["blob_id"]),
                connection=connection,
            )
        )
        if authoritative_blob is None:
            raise SyncStoreError("Sync blob is unavailable under dataset owner authority")
        blob_row = authoritative_blob
        match_params = (
            blob_row["dataset_id"],
            blob_row["payload_hash"],
            int(blob_row["size_bytes"]),
        )
        current_rows = self.execute(
            """
            SELECT dataset_id, attachment_id, attachment_revision
              FROM sync_attachment_revision_bindings
             WHERE dataset_id = ? AND blob_hash = ? AND size_bytes = ?
               AND resolved_blob_id IS NULL AND retention_released_at IS NULL
               AND EXISTS (
                    SELECT 1 FROM sync_current_heads AS head
                     WHERE head.dataset_id =
                               sync_attachment_revision_bindings.dataset_id
                       AND head.domain = 'attachment.ref'
                       AND head.object_id =
                               sync_attachment_revision_bindings.attachment_id
                       AND head.latest_server_cursor =
                               sync_attachment_revision_bindings.establishing_server_cursor
               )
             ORDER BY establishing_server_cursor, attachment_id, attachment_revision
             LIMIT 1000
            """,
            match_params,
            connection=connection,
        ).rows
        historical_rows = self.execute(
            """
            SELECT dataset_id, attachment_id, attachment_revision
              FROM sync_attachment_revision_bindings
             WHERE dataset_id = ? AND blob_hash = ? AND size_bytes = ?
               AND resolved_blob_id IS NULL AND retention_released_at IS NULL
               AND NOT EXISTS (
                    SELECT 1 FROM sync_current_heads AS head
                     WHERE head.dataset_id =
                               sync_attachment_revision_bindings.dataset_id
                       AND head.domain = 'attachment.ref'
                       AND head.object_id =
                               sync_attachment_revision_bindings.attachment_id
                       AND head.latest_server_cursor =
                               sync_attachment_revision_bindings.establishing_server_cursor
               )
             ORDER BY establishing_server_cursor, attachment_id, attachment_revision
             LIMIT 1000
            """,
            match_params,
            connection=connection,
        ).rows
        for row in (*current_rows, *historical_rows):
            self.execute(
                """
                UPDATE sync_attachment_revision_bindings
                   SET resolved_blob_id = ?
                 WHERE dataset_id = ? AND attachment_id = ?
                   AND attachment_revision = ? AND resolved_blob_id IS NULL
                   AND retention_released_at IS NULL
                   AND blob_hash = ? AND size_bytes = ?
                """,
                (
                    blob_row["blob_id"],
                    row["dataset_id"],
                    row["attachment_id"],
                    row["attachment_revision"],
                    blob_row["payload_hash"],
                    int(blob_row["size_bytes"]),
                ),
                connection=connection,
            )

    def relocate_legacy_blob(
        self,
        blob_store: Any,
        *,
        dataset_id: str,
        owner_user_id: str,
        blob_id: str,
    ) -> SyncBlobObject:
        """Verify and relocate one legacy global object under its dataset namespace."""

        suffix = " FOR UPDATE" if self.backend_type == BackendType.POSTGRESQL else ""
        with self.backend.transaction() as conn:
            self._require_dataset_owner_for_update(
                dataset_id,
                owner_user_id,
                connection=conn,
            )
            namespace_row = _first(
                self.execute(
                    """
                    SELECT * FROM sync_dataset_storage_namespaces
                     WHERE dataset_id = ? AND owner_user_id = ?
                    """
                    + suffix,  # nosec B608 - backend-controlled row lock suffix.
                    (dataset_id, owner_user_id),
                    connection=conn,
                )
            )
            if namespace_row is None:
                self.execute(
                    """
                    INSERT INTO sync_dataset_storage_namespaces (
                        dataset_id, owner_user_id, storage_namespace_id, created_at
                    ) VALUES (?, ?, ?, ?)
                    ON CONFLICT (dataset_id) DO NOTHING
                    """,
                    (dataset_id, owner_user_id, uuid4().hex, utcnow_iso()),
                    connection=conn,
                )
                namespace_row = _first(
                    self.execute(
                        """
                        SELECT * FROM sync_dataset_storage_namespaces
                         WHERE dataset_id = ? AND owner_user_id = ?
                        """,
                        (dataset_id, owner_user_id),
                        connection=conn,
                    )
                )
            if namespace_row is None:
                raise SyncStoreError("Sync dataset storage namespace could not be resolved")
            blob_row = _first(
                self.execute(
                    """
                    SELECT * FROM sync_blob_objects
                     WHERE dataset_id = ? AND owner_user_id = ? AND blob_id = ?
                    """
                    + suffix,  # nosec B608 - backend-controlled row lock suffix.
                    (dataset_id, owner_user_id, blob_id),
                    connection=conn,
                )
            )
            if (
                blob_row is None
                or blob_row.get("status") != "available"
                or blob_row.get("storage_backend") != "local_fs"
            ):
                raise SyncStoreError("Sync legacy blob relocation target is unavailable")
            expected_key = blob_store.namespace_storage_key(
                namespace_row["storage_namespace_id"],
                blob_row["payload_hash"],
            )
            if blob_row["storage_key"] == expected_key:
                blob_store.verify_blob(
                    expected_key,
                    payload_hash=blob_row["payload_hash"],
                    expected_size=int(blob_row["size_bytes"]),
                )
            else:
                legacy_key = blob_store.legacy_storage_key(blob_row["payload_hash"])
                if blob_row["storage_key"] != legacy_key:
                    raise SyncStoreError("Sync legacy blob storage key is not relocatable")
                relocated_key = blob_store.relocate_legacy_blob(
                    legacy_storage_key=legacy_key,
                    storage_namespace_id=namespace_row["storage_namespace_id"],
                    payload_hash=blob_row["payload_hash"],
                    expected_size=int(blob_row["size_bytes"]),
                )
                updated = self.execute(
                    """
                    UPDATE sync_blob_objects
                       SET storage_key = ?, updated_at = ?
                     WHERE dataset_id = ? AND blob_id = ? AND storage_key = ?
                       AND status = 'available'
                    """,
                    (
                        relocated_key,
                        utcnow_iso(),
                        dataset_id,
                        blob_id,
                        legacy_key,
                    ),
                    connection=conn,
                )
                if updated.rowcount != 1:
                    raise SyncStoreError("Sync legacy blob relocation CAS failed")
                blob_row = _first(
                    self.execute(
                        """
                        SELECT * FROM sync_blob_objects
                         WHERE dataset_id = ? AND blob_id = ?
                        """,
                        (dataset_id, blob_id),
                        connection=conn,
                    )
                )
                if blob_row is None or blob_row.get("storage_key") != relocated_key:
                    raise SyncStoreError("Sync legacy blob relocation was not durable")
            self._resolve_pending_bindings_for_blob(blob_row, connection=conn)
        return _blob_object_from_row(dict(blob_row))

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
                         WHERE dataset_id = ?
                           AND owner_user_id = ?
                           AND (
                                device_id = ?
                                OR (device_id IS NULL AND ? IS NULL)
                           )
                           AND idempotency_key = ?
                        """,
                        (
                            session.dataset_id,
                            session.owner_user_id,
                            session.device_id,
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

    def get_blob_chunk(
        self,
        upload_id: str,
        chunk_index: int,
        *,
        dataset_id: str | None = None,
    ) -> SyncBlobChunk | None:
        """Return a recorded blob chunk for duplicate-upload preflight checks."""

        if dataset_id is None:
            row = _first(
                self.execute(
                    """
                    SELECT * FROM sync_blob_chunks
                     WHERE upload_id = ? AND chunk_index = ?
                    """,
                    (upload_id, chunk_index),
                )
            )
        else:
            row = _first(
                self.execute(
                    """
                    SELECT * FROM sync_blob_chunks
                     WHERE upload_id = ? AND dataset_id = ? AND chunk_index = ?
                    """,
                    (upload_id, dataset_id, chunk_index),
                )
            )
        if row is None:
            return None
        return _blob_chunk_from_row(row)

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

    def require_blob_upload_completion_allowed(
        self,
        blob: SyncBlobObjectCreate,
        *,
        connection: Any,
    ) -> None:
        """Reject storage publication after a fence or metadata conflict."""

        row = _first(
            self.execute(
                """
                SELECT blob.*
                  FROM sync_blob_objects AS blob
                  JOIN sync_datasets AS dataset
                    ON dataset.dataset_id = blob.dataset_id
                 WHERE blob.dataset_id = ? AND blob.payload_hash = ?
                   AND (
                        dataset.scope_type = 'workspace'
                        OR blob.owner_user_id = ?
                   )
                """,
                (blob.dataset_id, blob.payload_hash, blob.owner_user_id),
                connection=connection,
            )
        )
        if row is None:
            return
        existing_status = str(row.get("status"))
        if existing_status == "deleting":
            raise SyncStoreError("Sync blob is deleting")
        if existing_status not in {"available", "deleted"}:
            raise SyncStoreError("Sync blob is not available for upload completion")
        existing_fingerprint = _blob_object_fingerprint_from_row(row)
        requested_fingerprint = _blob_object_fingerprint_from_create(blob)
        existing_fingerprint.pop("status")
        requested_fingerprint.pop("status")
        if existing_fingerprint != requested_fingerprint:
            raise SyncIdempotencyConflictError(
                "Sync blob payload hash was reused with different metadata"
            )

    def complete_blob_upload(
        self,
        blob: SyncBlobObjectCreate,
        *,
        connection: Any | None = None,
    ) -> SyncBlobObject:
        """Commit a verified blob and deduplicate by dataset plus payload hash."""

        if blob.status != "available":
            raise SyncStoreError("Sync blob upload completion must become available")
        now = utcnow_iso()
        suffix = " FOR UPDATE" if self.backend_type == BackendType.POSTGRESQL else ""
        with self.backend.transaction(connection) as conn:
            dataset_row = self._get_dataset_row_for_update(
                blob.dataset_id,
                connection=conn,
            )
            if dataset_row is None:
                raise SyncDatasetNotFoundError(
                    f"Sync dataset not found: {blob.dataset_id}"
                )
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
                    SELECT blob.*
                      FROM sync_blob_objects AS blob
                      JOIN sync_datasets AS dataset
                        ON dataset.dataset_id = blob.dataset_id
                     WHERE blob.dataset_id = ? AND blob.payload_hash = ?
                       AND (
                            dataset.scope_type = 'workspace'
                            OR blob.owner_user_id = dataset.owner_user_id
                       )
                    """
                    + suffix,  # nosec B608 - backend-controlled row lock suffix.
                    (blob.dataset_id, blob.payload_hash),
                    connection=conn,
                )
            )
            if row is None:
                raise SyncStoreError(
                    "Sync blob is unavailable under dataset owner authority"
                )
            existing_fingerprint = _blob_object_fingerprint_from_row(row)
            requested_fingerprint = _blob_object_fingerprint_from_create(blob)
            existing_status = str(existing_fingerprint.pop("status"))
            requested_fingerprint.pop("status")
            if existing_status == "deleting":
                raise SyncStoreError("Sync blob is deleting")
            if existing_fingerprint != requested_fingerprint:
                raise SyncIdempotencyConflictError(
                    "Sync blob payload hash was reused with different metadata"
                )
            if existing_status == "deleted":
                self.execute(
                    """
                    UPDATE sync_blob_objects
                       SET status = 'available', deleted_at = NULL, updated_at = ?
                     WHERE dataset_id = ? AND blob_id = ? AND status = 'deleted'
                    """,
                    (now, blob.dataset_id, row["blob_id"]),
                    connection=conn,
                )
                row = _first(
                    self.execute(
                        """
                        SELECT * FROM sync_blob_objects
                         WHERE dataset_id = ? AND blob_id = ?
                        """,
                        (blob.dataset_id, row["blob_id"]),
                        connection=conn,
                    )
                )
                if row is None or row["status"] != "available":
                    raise SyncStoreError("Sync blob repair did not become available")
            elif existing_status != "available":
                raise SyncStoreError("Sync blob is not available for upload completion")
            self._resolve_pending_bindings_for_blob(row, connection=conn)
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
        include_unavailable: bool = False,
        connection: Any | None = None,
        for_update: bool = False,
    ) -> SyncBlobObject | None:
        """Return an available blob object scoped by dataset and optional identity filters."""

        suffix = (
            " FOR UPDATE"
            if for_update and self.backend_type == BackendType.POSTGRESQL
            else ""
        )
        with self.backend.transaction(connection) as conn:
            self._require_dataset(dataset_id, connection=conn)
            row = _first(
                self.execute(
                    """
                    SELECT *
                      FROM sync_blob_objects
                     WHERE dataset_id = ?
                       AND (? = 1 OR status = 'available')
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
                    """
                    + suffix,  # nosec B608 - backend-controlled row lock suffix.
                    (
                        dataset_id,
                        1 if include_unavailable else 0,
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
                    connection=conn,
                )
            )
        if row is None:
            return None
        return _blob_object_from_row(row)

    def list_blob_availability_by_hashes(
        self,
        dataset_id: str,
        payload_hashes: Sequence[str],
        *,
        owner_user_id: str,
        connection: Any | None = None,
    ) -> dict[str, SyncBlobAvailabilityStatus]:
        """Return one bounded owner-scoped availability map without exposing storage."""

        unique_hashes = list(dict.fromkeys(payload_hashes))
        if len(unique_hashes) > 200:
            raise SyncStoreError("Sync blob availability query exceeds its boundary")
        if not unique_hashes:
            return {}
        for payload_hash in unique_hashes:
            digest = payload_hash.removeprefix("sha256:")
            if (
                not payload_hash.startswith("sha256:")
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise SyncStoreError("payload_hash must be a canonical SHA-256 digest")
        placeholders = ",".join("?" for _ in unique_hashes)
        with self.backend.transaction(connection) as conn:
            self._require_dataset(dataset_id, connection=conn)
            rows = self.execute(
                "SELECT payload_hash, status FROM sync_blob_objects "
                "WHERE dataset_id = ? AND owner_user_id = ? "
                f"AND payload_hash IN ({placeholders})",  # nosec B608 - placeholders only.
                (dataset_id, owner_user_id, *unique_hashes),
                connection=conn,
            )
        return {
            str(row["payload_hash"]): typing.cast(
                SyncBlobAvailabilityStatus,
                row["status"],
            )
            for row in rows
        }

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

    def list_blob_objects_for_dataset_page(
        self,
        dataset_id: str,
        *,
        status: str = "available",
        after_updated_at: str | None = None,
        after_blob_id: str | None = None,
        limit: int = 1000,
        connection: Any | None = None,
    ) -> list[SyncBlobObject]:
        """Return one capped keyset page of blob metadata."""

        if isinstance(limit, bool) or limit < 1:
            raise SyncStoreError("Sync blob page limit must be positive")
        if (after_updated_at is None) != (after_blob_id is None):
            raise SyncStoreError("Sync blob page cursor is incomplete")
        page_limit = min(limit, 1000)
        params: list[Any] = [dataset_id, status]
        if after_updated_at is not None and after_blob_id is not None:
            query = """
                SELECT * FROM sync_blob_objects
                 WHERE dataset_id = ? AND status = ?
                   AND (updated_at, blob_id) > (?, ?)
                 ORDER BY updated_at, blob_id LIMIT ?
            """
            params.extend((after_updated_at, after_blob_id))
        else:
            query = """
                SELECT * FROM sync_blob_objects
                 WHERE dataset_id = ? AND status = ?
                 ORDER BY updated_at, blob_id LIMIT ?
            """
        params.append(page_limit)
        with self.backend.transaction(connection) as conn:
            self._require_dataset(dataset_id, connection=conn)
            rows = self.execute(
                query,
                tuple(params),
                connection=conn,
            ).rows
        return [_blob_object_from_row(row) for row in rows]

    def fence_blob_object_deleting(
        self,
        dataset_id: str,
        blob_id: str,
        *,
        connection: Any,
    ) -> SyncBlobObject | None:
        """Durably fence one available blob before physical deletion."""

        now = utcnow_iso()
        self.execute(
            """
            UPDATE sync_blob_objects
               SET status = 'deleting', updated_at = ?
             WHERE dataset_id = ? AND blob_id = ?
               AND status = 'available' AND deleted_at IS NULL
            """,
            (now, dataset_id, blob_id),
            connection=connection,
        )
        row = _first(
            self.execute(
                """
                SELECT * FROM sync_blob_objects
                 WHERE dataset_id = ? AND blob_id = ?
                """,
                (dataset_id, blob_id),
                connection=connection,
            )
        )
        return None if row is None else _blob_object_from_row(row)

    def finalize_blob_object_deleted(
        self,
        dataset_id: str,
        blob_id: str,
        *,
        connection: Any,
    ) -> SyncBlobObject | None:
        """Finalize a physically absent fenced blob as deleted."""

        now = utcnow_iso()
        self.execute(
            """
            UPDATE sync_blob_objects
               SET status = 'deleted', deleted_at = ?, updated_at = ?
             WHERE dataset_id = ? AND blob_id = ? AND status = 'deleting'
            """,
            (now, now, dataset_id, blob_id),
            connection=connection,
        )
        row = _first(
            self.execute(
                """
                SELECT * FROM sync_blob_objects
                 WHERE dataset_id = ? AND blob_id = ?
                """,
                (dataset_id, blob_id),
                connection=connection,
            )
        )
        return None if row is None else _blob_object_from_row(row)

    def get_domain_compaction_sequence(
        self,
        dataset_id: str,
        domain: SyncDomain,
    ) -> int:
        """Return the last retained compaction checkpoint sequence for a domain."""

        row = _first(
            self.execute(
                """
                SELECT last_compacted_sequence
                  FROM sync_domain_state
                 WHERE dataset_id = ? AND domain = ?
                """,
                (dataset_id, domain),
            )
        )
        return int((row or {}).get("last_compacted_sequence") or 0)

    def record_domain_compaction(
        self,
        dataset_id: str,
        domain: SyncDomain,
        *,
        through_server_sequence: int,
        state: Mapping[str, Any],
        adapter_version: int = 1,
        connection: Any | None = None,
    ) -> int:
        """Record a non-destructive compaction checkpoint for a domain."""

        now = utcnow_iso()
        with self.backend.transaction(connection) as conn:
            self._require_dataset_domain(dataset_id, domain, connection=conn)
            existing = _first(
                self.execute(
                    """
                    SELECT last_compacted_sequence
                      FROM sync_domain_state
                     WHERE dataset_id = ? AND domain = ?
                    """,
                    (dataset_id, domain),
                    connection=conn,
                )
            )
            last_compacted_sequence = max(
                int((existing or {}).get("last_compacted_sequence") or 0),
                through_server_sequence,
            )
            if existing is None:
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
                        0,
                        last_compacted_sequence,
                        encode_json(state, default={}),
                        now,
                    ),
                    connection=conn,
                )
            else:
                self.execute(
                    """
                    UPDATE sync_domain_state
                       SET adapter_version = ?,
                           last_compacted_sequence = ?,
                           state_json = ?,
                           updated_at = ?
                     WHERE dataset_id = ? AND domain = ?
                    """,
                    (
                        adapter_version,
                        last_compacted_sequence,
                        encode_json(state, default={}),
                        now,
                        dataset_id,
                        domain,
                    ),
                    connection=conn,
                )
        return last_compacted_sequence

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

    def _migrate_versioned_device_state(self, *, connection: Any) -> None:
        """Serialize, verify, and complete the additive adapter-state migration."""

        if self.backend_type == BackendType.POSTGRESQL:
            self.execute(
                "SELECT pg_advisory_xact_lock(?)",
                (SYNC_VERSIONED_DEVICE_STATE_MIGRATION_LOCK_KEY,),
                connection=connection,
            )
        completed_type = (
            "TIMESTAMPTZ" if self.backend_type == BackendType.POSTGRESQL else "TEXT"
        )
        self.execute(
            f"""
            CREATE TABLE IF NOT EXISTS sync_schema_migrations (
                migration_id TEXT PRIMARY KEY NOT NULL,
                completed_at {completed_type}
            )
            """,  # nosec B608 - backend-selected fixed timestamp type.
            connection=connection,
        )
        self.execute(
            """
            INSERT INTO sync_schema_migrations (migration_id, completed_at)
            VALUES (?, NULL)
            ON CONFLICT (migration_id) DO NOTHING
            """,
            (SYNC_VERSIONED_DEVICE_STATE_MIGRATION_ID,),
            connection=connection,
        )
        lock_suffix = " FOR UPDATE" if self.backend_type == BackendType.POSTGRESQL else ""
        marker = _first(
            self.execute(
                "SELECT completed_at FROM sync_schema_migrations WHERE migration_id = ?"
                + lock_suffix,  # nosec B608 - backend-controlled row lock suffix.
                (SYNC_VERSIONED_DEVICE_STATE_MIGRATION_ID,),
                connection=connection,
            )
        )
        if marker is None:
            raise SyncStoreError("Sync adapter-state migration authority is unavailable")

        if marker.get("completed_at") is None:
            schema = (
                SYNC_VERSIONED_DEVICE_STATE_POSTGRES_SCHEMA
                if self.backend_type == BackendType.POSTGRESQL
                else SYNC_VERSIONED_DEVICE_STATE_SQLITE_SCHEMA
            )
            for statement in schema.split(";"):
                if statement.strip():
                    self.execute(statement, connection=connection)

        self._verify_versioned_device_state_catalog(connection=connection)
        self._reconcile_versioned_device_state(connection=connection)
        if marker.get("completed_at") is None:
            self.execute(
                """
                UPDATE sync_schema_migrations
                   SET completed_at = ?
                 WHERE migration_id = ? AND completed_at IS NULL
                """,
                (utcnow_iso(), SYNC_VERSIONED_DEVICE_STATE_MIGRATION_ID),
                connection=connection,
            )

    def _verify_versioned_device_state_catalog(self, *, connection: Any) -> None:
        """Fail closed when the three additive side tables differ from the contract."""

        authority_info = self.backend.get_table_info(
            "sync_schema_migrations",
            connection=connection,
        )
        if [str(column.get("name")) for column in authority_info] != [
            "migration_id",
            "completed_at",
        ] or [str(column.get("type") or "").lower() for column in authority_info] != [
            "text",
            (
                "timestamp with time zone"
                if self.backend_type == BackendType.POSTGRESQL
                else "text"
            ),
        ] or [bool(column.get("nullable")) for column in authority_info] != [False, True]:
            raise SyncStoreError("Sync adapter-state migration catalog is incompatible")
        if self.backend_type == BackendType.POSTGRESQL:
            authority_pk = _first(
                self.execute(
                    """
                    SELECT string_agg(attribute.attname, ',' ORDER BY keys.ordinality)
                               AS primary_key_columns
                      FROM pg_constraint AS constraint_row
                      JOIN pg_class AS table_row
                        ON table_row.oid = constraint_row.conrelid
                      JOIN pg_namespace AS namespace
                        ON namespace.oid = table_row.relnamespace
                      JOIN unnest(constraint_row.conkey) WITH ORDINALITY
                           AS keys(attnum, ordinality) ON TRUE
                      JOIN pg_attribute AS attribute
                        ON attribute.attrelid = table_row.oid
                       AND attribute.attnum = keys.attnum
                     WHERE namespace.nspname = current_schema()
                       AND table_row.relname = 'sync_schema_migrations'
                       AND constraint_row.contype = 'p'
                    """,
                    connection=connection,
                )
            ) or {}
            authority_primary_key = str(
                authority_pk.get("primary_key_columns") or ""
            ).split(",")
        else:
            authority_primary_key = [
                str(row["name"])
                for row in self.execute(
                    "PRAGMA table_info(sync_schema_migrations)",
                    connection=connection,
                ).rows
                if int(row["pk"])
            ]
        if authority_primary_key != ["migration_id"]:
            raise SyncStoreError("Sync adapter-state migration catalog is incompatible")
        expected_columns = {
            "sync_device_adapter_cursors": [
                "dataset_id",
                "device_id",
                "domain",
                "adapter_version",
                "last_pulled_sequence",
                "max_delivered_sequence",
                "updated_at",
            ],
            "sync_device_adapter_domain_acks": [
                "dataset_id",
                "device_id",
                "domain",
                "adapter_version",
                "through_server_sequence",
                "applied_at",
                "updated_at",
                "idempotency_key",
            ],
            "sync_device_blob_id_acks": [
                "dataset_id",
                "device_id",
                "blob_id",
                "payload_hash",
                "verified_at",
                "updated_at",
                "idempotency_key",
            ],
        }
        expected_primary_keys = {
            "sync_device_adapter_cursors": [
                "dataset_id",
                "device_id",
                "domain",
                "adapter_version",
            ],
            "sync_device_adapter_domain_acks": [
                "dataset_id",
                "device_id",
                "domain",
                "adapter_version",
            ],
            "sync_device_blob_id_acks": ["dataset_id", "device_id", "blob_id"],
        }
        expected_indexes = {
            "sync_device_adapter_cursors": "idx_sync_device_adapter_cursors_device",
            "sync_device_adapter_domain_acks": (
                "idx_sync_device_adapter_domain_acks_device"
            ),
            "sync_device_blob_id_acks": "idx_sync_device_blob_id_acks_device",
        }
        expected_sqlite_table_sql: dict[str, str] = {}
        if self.backend_type != BackendType.POSTGRESQL:
            for statement in SYNC_VERSIONED_DEVICE_STATE_SQLITE_SCHEMA.split(";"):
                compact_statement = self._compact_catalog_sql(statement)
                for table_name in expected_columns:
                    if f"createtableifnotexists{table_name}(" in compact_statement:
                        expected_sqlite_table_sql[table_name] = compact_statement.replace(
                            "createtableifnotexists",
                            "createtable",
                            1,
                        )
        expected_postgres_checks = {
            "sync_device_adapter_cursors": {
                "checkadapter_version>0",
                "checklast_pulled_sequence>=0",
                "checkmax_delivered_sequence>=0",
                "checkmax_delivered_sequence<=last_pulled_sequence",
            },
            "sync_device_adapter_domain_acks": {
                "checkadapter_version>0",
                "checkthrough_server_sequence>=0",
            },
            "sync_device_blob_id_acks": set(),
        }
        timestamp_type = (
            "timestamp with time zone"
            if self.backend_type == BackendType.POSTGRESQL
            else "text"
        )
        sequence_type = (
            "bigint" if self.backend_type == BackendType.POSTGRESQL else "integer"
        )
        expected_types = {
            "sync_device_adapter_cursors": [
                "text",
                "text",
                "text",
                "integer",
                sequence_type,
                sequence_type,
                timestamp_type,
            ],
            "sync_device_adapter_domain_acks": [
                "text",
                "text",
                "text",
                "integer",
                sequence_type,
                timestamp_type,
                timestamp_type,
                "text",
            ],
            "sync_device_blob_id_acks": [
                "text",
                "text",
                "text",
                "text",
                timestamp_type,
                timestamp_type,
                "text",
            ],
        }
        for table_name, columns in expected_columns.items():
            info = self.backend.get_table_info(table_name, connection=connection)
            if (
                [str(column.get("name")) for column in info] != columns
                or [str(column.get("type") or "").lower() for column in info]
                != expected_types[table_name]
                or [bool(column.get("nullable")) for column in info]
                != [False] * (len(columns) - 1) + [table_name != "sync_device_adapter_cursors"]
            ):
                raise SyncStoreError("Sync adapter-state migration catalog is incompatible")
            if self.backend_type == BackendType.POSTGRESQL:
                pk_row = _first(
                    self.execute(
                        """
                        SELECT string_agg(attribute.attname, ',' ORDER BY keys.ordinality)
                                   AS primary_key_columns
                          FROM pg_constraint AS constraint_row
                          JOIN pg_class AS table_row
                            ON table_row.oid = constraint_row.conrelid
                          JOIN pg_namespace AS namespace
                            ON namespace.oid = table_row.relnamespace
                          JOIN unnest(constraint_row.conkey) WITH ORDINALITY
                               AS keys(attnum, ordinality) ON TRUE
                          JOIN pg_attribute AS attribute
                            ON attribute.attrelid = table_row.oid
                           AND attribute.attnum = keys.attnum
                         WHERE namespace.nspname = current_schema()
                           AND table_row.relname = ?
                           AND constraint_row.contype = 'p'
                        """,
                        (table_name,),
                        connection=connection,
                    )
                ) or {}
                primary_key = str(pk_row.get("primary_key_columns") or "").split(",")
                check_rows = self.execute(
                    """
                    SELECT pg_catalog.pg_get_constraintdef(constraint_row.oid, true)
                               AS definition
                      FROM pg_catalog.pg_constraint AS constraint_row
                      JOIN pg_catalog.pg_class AS table_row
                        ON table_row.oid = constraint_row.conrelid
                      JOIN pg_catalog.pg_namespace AS namespace
                        ON namespace.oid = table_row.relnamespace
                     WHERE namespace.nspname = current_schema()
                       AND table_row.relname = ?
                       AND constraint_row.contype = 'c'
                    """,
                    (table_name,),
                    connection=connection,
                ).rows
                checks_valid = {
                    self._compact_postgres_catalog_sql(row["definition"])
                    for row in check_rows
                } == expected_postgres_checks[table_name]
                index_rows = self.execute(
                    """
                    SELECT indexname, indexdef FROM pg_indexes
                     WHERE schemaname = current_schema() AND tablename = ?
                    """,
                    (table_name,),
                    connection=connection,
                ).rows
                index = next(
                    (
                        row
                        for row in index_rows
                        if row.get("indexname") == expected_indexes[table_name]
                    ),
                    None,
                )
                index_columns_valid = index is not None and "(device_id, dataset_id)" in str(
                    index.get("indexdef")
                )
            else:
                raw_info = self.execute(
                    f"PRAGMA table_info({table_name})",  # nosec B608 - fixed catalog names.
                    connection=connection,
                ).rows
                primary_key = [
                    str(row["name"])
                    for row in sorted(raw_info, key=lambda row: int(row["pk"]))
                    if int(row["pk"])
                ]
                table_row = _first(
                    self.execute(
                        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
                        (table_name,),
                        connection=connection,
                    )
                )
                checks_valid = self._compact_catalog_sql(
                    None if table_row is None else table_row.get("sql")
                ) == expected_sqlite_table_sql.get(table_name)
                index_rows = self.execute(
                    f"PRAGMA index_list({table_name})",  # nosec B608 - fixed catalog names.
                    connection=connection,
                ).rows
                index = next(
                    (
                        row
                        for row in index_rows
                        if row.get("name") == expected_indexes[table_name]
                    ),
                    None,
                )
                index_columns = (
                    []
                    if index is None
                    else [
                        row["name"]
                        for row in self.execute(
                            f"PRAGMA index_info({expected_indexes[table_name]})",  # nosec B608
                            connection=connection,
                        ).rows
                    ]
                )
                index_columns_valid = index_columns == ["device_id", "dataset_id"]
            if (
                primary_key != expected_primary_keys[table_name]
                or not checks_valid
                or not index_columns_valid
            ):
                raise SyncStoreError("Sync adapter-state migration catalog is incompatible")

    def _reconcile_versioned_device_state(self, *, connection: Any) -> None:
        """Seed and reconcile rollback-compatible adapter-v1 cursor/ack state."""

        self.execute(
            """
            INSERT INTO sync_device_adapter_cursors (
                dataset_id, device_id, domain, adapter_version,
                last_pulled_sequence, max_delivered_sequence, updated_at
            )
            SELECT dataset_id, device_id, domain, 1,
                   last_pulled_sequence, last_pulled_sequence, updated_at
              FROM sync_device_cursors
             WHERE 1 = 1
            ON CONFLICT (dataset_id, device_id, domain, adapter_version)
            DO UPDATE SET
                last_pulled_sequence = CASE
                    WHEN excluded.last_pulled_sequence
                         > sync_device_adapter_cursors.last_pulled_sequence
                    THEN excluded.last_pulled_sequence
                    ELSE sync_device_adapter_cursors.last_pulled_sequence END,
                max_delivered_sequence = CASE
                    WHEN excluded.last_pulled_sequence
                         > sync_device_adapter_cursors.last_pulled_sequence
                    THEN excluded.max_delivered_sequence
                    ELSE sync_device_adapter_cursors.max_delivered_sequence END,
                updated_at = CASE
                    WHEN excluded.last_pulled_sequence
                         > sync_device_adapter_cursors.last_pulled_sequence
                    THEN excluded.updated_at
                    ELSE sync_device_adapter_cursors.updated_at END
            """,
            connection=connection,
        )
        self.execute(
            """
            INSERT INTO sync_device_cursors (
                dataset_id, device_id, domain, last_pulled_sequence, updated_at
            )
            SELECT dataset_id, device_id, domain, last_pulled_sequence, updated_at
              FROM sync_device_adapter_cursors
             WHERE adapter_version = 1
            ON CONFLICT (dataset_id, device_id, domain)
            DO UPDATE SET
                last_pulled_sequence = CASE
                    WHEN excluded.last_pulled_sequence
                         > sync_device_cursors.last_pulled_sequence
                    THEN excluded.last_pulled_sequence
                    ELSE sync_device_cursors.last_pulled_sequence END,
                updated_at = CASE
                    WHEN excluded.last_pulled_sequence
                         > sync_device_cursors.last_pulled_sequence
                    THEN excluded.updated_at ELSE sync_device_cursors.updated_at END
            """,
            connection=connection,
        )
        self.execute(
            """
            UPDATE sync_device_cursors
               SET last_pulled_sequence = (
                       SELECT versioned.last_pulled_sequence
                         FROM sync_device_adapter_cursors AS versioned
                        WHERE versioned.dataset_id = sync_device_cursors.dataset_id
                          AND versioned.device_id = sync_device_cursors.device_id
                          AND versioned.domain = sync_device_cursors.domain
                          AND versioned.adapter_version = 1
                   ),
                   updated_at = (
                       SELECT versioned.updated_at
                         FROM sync_device_adapter_cursors AS versioned
                        WHERE versioned.dataset_id = sync_device_cursors.dataset_id
                          AND versioned.device_id = sync_device_cursors.device_id
                          AND versioned.domain = sync_device_cursors.domain
                          AND versioned.adapter_version = 1
                   )
             WHERE EXISTS (
                       SELECT 1 FROM sync_device_adapter_cursors AS versioned
                        WHERE versioned.dataset_id = sync_device_cursors.dataset_id
                          AND versioned.device_id = sync_device_cursors.device_id
                          AND versioned.domain = sync_device_cursors.domain
                          AND versioned.adapter_version = 1
                          AND versioned.last_pulled_sequence > sync_device_cursors.last_pulled_sequence
                   )
            """,
            connection=connection,
        )
        self.execute(
            """
            INSERT INTO sync_device_adapter_domain_acks (
                dataset_id, device_id, domain, adapter_version,
                through_server_sequence, applied_at, updated_at, idempotency_key
            )
            SELECT dataset_id, device_id, domain, 1, through_server_sequence,
                   applied_at, updated_at, idempotency_key
              FROM sync_device_domain_acks
             WHERE 1 = 1
            ON CONFLICT (dataset_id, device_id, domain, adapter_version)
            DO UPDATE SET
                through_server_sequence = CASE
                    WHEN excluded.through_server_sequence
                         > sync_device_adapter_domain_acks.through_server_sequence
                    THEN excluded.through_server_sequence
                    ELSE sync_device_adapter_domain_acks.through_server_sequence END,
                applied_at = CASE
                    WHEN excluded.through_server_sequence
                         > sync_device_adapter_domain_acks.through_server_sequence
                    THEN excluded.applied_at
                    ELSE sync_device_adapter_domain_acks.applied_at END,
                updated_at = CASE
                    WHEN excluded.through_server_sequence
                         > sync_device_adapter_domain_acks.through_server_sequence
                    THEN excluded.updated_at
                    ELSE sync_device_adapter_domain_acks.updated_at END,
                idempotency_key = CASE
                    WHEN excluded.through_server_sequence
                         > sync_device_adapter_domain_acks.through_server_sequence
                    THEN excluded.idempotency_key
                    ELSE sync_device_adapter_domain_acks.idempotency_key END
            """,
            connection=connection,
        )
        self.execute(
            """
            INSERT INTO sync_device_domain_acks (
                dataset_id, device_id, domain, through_server_sequence,
                applied_at, updated_at, idempotency_key
            )
            SELECT dataset_id, device_id, domain, through_server_sequence,
                   applied_at, updated_at, idempotency_key
              FROM sync_device_adapter_domain_acks
             WHERE adapter_version = 1
            ON CONFLICT (dataset_id, device_id, domain)
            DO UPDATE SET
                through_server_sequence = CASE
                    WHEN excluded.through_server_sequence
                         > sync_device_domain_acks.through_server_sequence
                    THEN excluded.through_server_sequence
                    ELSE sync_device_domain_acks.through_server_sequence END,
                applied_at = CASE
                    WHEN excluded.through_server_sequence
                         > sync_device_domain_acks.through_server_sequence
                    THEN excluded.applied_at ELSE sync_device_domain_acks.applied_at END,
                updated_at = CASE
                    WHEN excluded.through_server_sequence
                         > sync_device_domain_acks.through_server_sequence
                    THEN excluded.updated_at ELSE sync_device_domain_acks.updated_at END,
                idempotency_key = CASE
                    WHEN excluded.through_server_sequence
                         > sync_device_domain_acks.through_server_sequence
                    THEN excluded.idempotency_key
                    ELSE sync_device_domain_acks.idempotency_key END
            """,
            connection=connection,
        )
        self.execute(
            """
            UPDATE sync_device_domain_acks
               SET through_server_sequence = (
                       SELECT versioned.through_server_sequence
                         FROM sync_device_adapter_domain_acks AS versioned
                        WHERE versioned.dataset_id = sync_device_domain_acks.dataset_id
                          AND versioned.device_id = sync_device_domain_acks.device_id
                          AND versioned.domain = sync_device_domain_acks.domain
                          AND versioned.adapter_version = 1
                   ),
                   applied_at = (
                       SELECT versioned.applied_at
                         FROM sync_device_adapter_domain_acks AS versioned
                        WHERE versioned.dataset_id = sync_device_domain_acks.dataset_id
                          AND versioned.device_id = sync_device_domain_acks.device_id
                          AND versioned.domain = sync_device_domain_acks.domain
                          AND versioned.adapter_version = 1
                   ),
                   updated_at = (
                       SELECT versioned.updated_at
                         FROM sync_device_adapter_domain_acks AS versioned
                        WHERE versioned.dataset_id = sync_device_domain_acks.dataset_id
                          AND versioned.device_id = sync_device_domain_acks.device_id
                          AND versioned.domain = sync_device_domain_acks.domain
                          AND versioned.adapter_version = 1
                   )
             WHERE EXISTS (
                       SELECT 1 FROM sync_device_adapter_domain_acks AS versioned
                        WHERE versioned.dataset_id = sync_device_domain_acks.dataset_id
                          AND versioned.device_id = sync_device_domain_acks.device_id
                          AND versioned.domain = sync_device_domain_acks.domain
                          AND versioned.adapter_version = 1
                          AND versioned.through_server_sequence
                              > sync_device_domain_acks.through_server_sequence
                   )
            """,
            connection=connection,
        )
        mismatch = _first(
            self.execute(
                """
                SELECT
                    (SELECT COUNT(*) FROM sync_device_cursors) AS legacy_cursor_count,
                    (SELECT COUNT(*) FROM sync_device_adapter_cursors
                      WHERE adapter_version = 1) AS version_cursor_count,
                    (SELECT COUNT(*) FROM sync_device_domain_acks) AS legacy_ack_count,
                    (SELECT COUNT(*) FROM sync_device_adapter_domain_acks
                      WHERE adapter_version = 1) AS version_ack_count,
                    (SELECT COUNT(*) FROM sync_device_cursors AS legacy
                      JOIN sync_device_adapter_cursors AS versioned
                        ON versioned.dataset_id = legacy.dataset_id
                       AND versioned.device_id = legacy.device_id
                       AND versioned.domain = legacy.domain
                       AND versioned.adapter_version = 1
                     WHERE legacy.last_pulled_sequence <> versioned.last_pulled_sequence)
                        AS cursor_mismatches,
                    (SELECT COUNT(*) FROM sync_device_domain_acks AS legacy
                      JOIN sync_device_adapter_domain_acks AS versioned
                        ON versioned.dataset_id = legacy.dataset_id
                       AND versioned.device_id = legacy.device_id
                       AND versioned.domain = legacy.domain
                       AND versioned.adapter_version = 1
                     WHERE legacy.through_server_sequence <> versioned.through_server_sequence)
                        AS ack_mismatches
                """,
                connection=connection,
            )
        ) or {}
        if (
            int(mismatch.get("legacy_cursor_count") or 0)
            != int(mismatch.get("version_cursor_count") or 0)
            or int(mismatch.get("legacy_ack_count") or 0)
            != int(mismatch.get("version_ack_count") or 0)
            or int(mismatch.get("cursor_mismatches") or 0)
            or int(mismatch.get("ack_mismatches") or 0)
        ):
            raise SyncStoreError("Sync adapter cursor/ack migration verification failed")
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
                "mutation_group_id": "TEXT",
                "mutation_step": "INTEGER",
                "mutation_step_count": "INTEGER",
                "mutation_plan_hash": "TEXT",
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
                "mutation_group_id": "TEXT",
                "mutation_step": "INTEGER",
                "mutation_step_count": "INTEGER",
                "mutation_plan_hash": "TEXT",
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

    def _ensure_sync_current_heads_table(
        self, *, connection: Any, projection_exists: bool
    ) -> None:
        cursor_type = "BIGINT" if self.backend_type == BackendType.POSTGRESQL else "INTEGER"
        self.backend.create_tables(
            f"""
            CREATE TABLE IF NOT EXISTS sync_current_heads (
                dataset_id TEXT NOT NULL,
                domain TEXT NOT NULL,
                object_id TEXT NOT NULL,
                latest_server_cursor {cursor_type} NOT NULL,
                PRIMARY KEY (dataset_id, domain, object_id)
            )
            """,
            connection=connection,
        )
        self.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_sync_current_heads_dataset_domain_cursor
                ON sync_current_heads(dataset_id, domain, latest_server_cursor, object_id)
            """,
            connection=connection,
        )
        invalid_heads = self.execute(
            """
            SELECT heads.dataset_id, heads.domain, heads.object_id
              FROM sync_current_heads AS heads
              LEFT JOIN sync_envelopes AS envelope
                ON envelope.server_sequence = heads.latest_server_cursor
               AND envelope.dataset_id = heads.dataset_id
               AND envelope.domain = heads.domain
               AND envelope.entity_id = heads.object_id
             WHERE envelope.server_sequence IS NULL
                OR envelope.status <> 'accepted'
                OR envelope.apply_status = 'superseded'
            """,
            connection=connection,
        ).rows
        for head in invalid_heads:
            latest = _first(
                self.execute(
                    """
                    SELECT server_sequence
                     FROM sync_envelopes
                     WHERE dataset_id = ? AND domain = ? AND entity_id = ?
                       AND status = 'accepted'
                       AND apply_status <> 'superseded'
                     ORDER BY server_sequence DESC
                     LIMIT 1
                    """,
                    (head["dataset_id"], head["domain"], head["object_id"]),
                    connection=connection,
                )
            )
            if latest is None:
                self.execute(
                    """
                    DELETE FROM sync_current_heads
                     WHERE dataset_id = ? AND domain = ? AND object_id = ?
                    """,
                    (head["dataset_id"], head["domain"], head["object_id"]),
                    connection=connection,
                )
                continue
            self.execute(
                """
                UPDATE sync_current_heads
                   SET latest_server_cursor = ?
                 WHERE dataset_id = ? AND domain = ? AND object_id = ?
                """,
                (
                    latest["server_sequence"],
                    head["dataset_id"],
                    head["domain"],
                    head["object_id"],
                ),
                connection=connection,
            )
        if not projection_exists:
            self.execute(
                """
                INSERT INTO sync_current_heads (
                    dataset_id, domain, object_id, latest_server_cursor
                )
                SELECT dataset_id, domain, entity_id, MAX(server_sequence)
                  FROM sync_envelopes
                 WHERE status = 'accepted'
                   AND apply_status <> 'superseded'
                 GROUP BY dataset_id, domain, entity_id
                ON CONFLICT (dataset_id, domain, object_id)
                DO UPDATE SET latest_server_cursor = excluded.latest_server_cursor
                 WHERE excluded.latest_server_cursor > sync_current_heads.latest_server_cursor
                """,
                connection=connection,
            )

    def _ensure_sync_materialization_locks_table(self, *, connection: Any) -> None:
        timestamp_type = (
            "TIMESTAMPTZ" if self.backend_type == BackendType.POSTGRESQL else "TEXT"
        )
        self.backend.create_tables(
            f"""
            CREATE TABLE IF NOT EXISTS sync_materialization_locks (
                dataset_id TEXT NOT NULL,
                domain TEXT NOT NULL,
                object_id TEXT NOT NULL,
                updated_at {timestamp_type} NOT NULL,
                PRIMARY KEY (dataset_id, domain, object_id)
            )
            """,
            connection=connection,
        )

    def _ensure_attachment_binding_tables(self, *, connection: Any) -> None:
        """Ensure additive attachment binding and opaque namespace authority."""

        if self.backend_type == BackendType.POSTGRESQL:
            schema = """
            CREATE TABLE IF NOT EXISTS sync_attachment_revision_bindings (
                dataset_id TEXT NOT NULL,
                attachment_id TEXT NOT NULL,
                attachment_revision BIGINT NOT NULL,
                blob_hash TEXT NOT NULL,
                size_bytes BIGINT NOT NULL,
                establishing_server_cursor BIGINT NOT NULL,
                availability_at_acceptance TEXT NOT NULL,
                resolved_blob_id TEXT,
                retention_released_at TIMESTAMPTZ,
                created_at TIMESTAMPTZ NOT NULL,
                PRIMARY KEY (dataset_id, attachment_id, attachment_revision),
                CHECK (length(dataset_id) > 0),
                CHECK (attachment_id ~ '^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$'),
                CHECK (attachment_revision > 0),
                CHECK (blob_hash ~ '^sha256:[0-9a-f]{64}$'),
                CHECK (size_bytes > 0),
                CHECK (establishing_server_cursor > 0),
                CHECK (availability_at_acceptance IN ('available', 'metadata_only')),
                CHECK (resolved_blob_id IS NULL OR length(resolved_blob_id) > 0)
            );
            CREATE TABLE IF NOT EXISTS sync_dataset_storage_namespaces (
                dataset_id TEXT NOT NULL PRIMARY KEY,
                owner_user_id TEXT NOT NULL,
                storage_namespace_id TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL,
                CHECK (length(dataset_id) > 0),
                CHECK (length(owner_user_id) > 0),
                CHECK (storage_namespace_id ~ '^[0-9a-f]{32}$')
            );
            """
        else:
            schema = """
            CREATE TABLE IF NOT EXISTS sync_attachment_revision_bindings (
                dataset_id TEXT NOT NULL,
                attachment_id TEXT NOT NULL,
                attachment_revision INTEGER NOT NULL,
                blob_hash TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                establishing_server_cursor INTEGER NOT NULL,
                availability_at_acceptance TEXT NOT NULL,
                resolved_blob_id TEXT,
                retention_released_at TEXT,
                created_at TEXT NOT NULL,
                PRIMARY KEY (dataset_id, attachment_id, attachment_revision),
                CHECK (length(dataset_id) > 0),
                CHECK (
                    length(attachment_id) = 36
                    AND lower(attachment_id) = attachment_id
                    AND substr(attachment_id, 9, 1) = '-'
                    AND substr(attachment_id, 14, 1) = '-'
                    AND substr(attachment_id, 15, 1) = '4'
                    AND substr(attachment_id, 19, 1) = '-'
                    AND substr(attachment_id, 20, 1) IN ('8', '9', 'a', 'b')
                    AND substr(attachment_id, 24, 1) = '-'
                    AND length(replace(attachment_id, '-', '')) = 32
                    AND replace(attachment_id, '-', '') NOT GLOB '*[^0-9a-f]*'
                ),
                CHECK (attachment_revision > 0),
                CHECK (length(blob_hash) = 71),
                CHECK (substr(blob_hash, 1, 7) = 'sha256:'),
                CHECK (substr(blob_hash, 8) NOT GLOB '*[^0-9a-f]*'),
                CHECK (size_bytes > 0),
                CHECK (establishing_server_cursor > 0),
                CHECK (availability_at_acceptance IN ('available', 'metadata_only')),
                CHECK (resolved_blob_id IS NULL OR length(resolved_blob_id) > 0)
            );
            CREATE TABLE IF NOT EXISTS sync_dataset_storage_namespaces (
                dataset_id TEXT NOT NULL PRIMARY KEY,
                owner_user_id TEXT NOT NULL,
                storage_namespace_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                CHECK (length(dataset_id) > 0),
                CHECK (length(owner_user_id) > 0),
                CHECK (length(storage_namespace_id) = 32),
                CHECK (storage_namespace_id NOT GLOB '*[^0-9a-f]*')
            );
            """
        self.backend.create_tables(schema, connection=connection)
        for statement in (
            """
            CREATE INDEX IF NOT EXISTS idx_sync_attachment_bindings_unresolved
                ON sync_attachment_revision_bindings(
                    dataset_id, establishing_server_cursor, attachment_id,
                    attachment_revision
                )
                WHERE resolved_blob_id IS NULL AND retention_released_at IS NULL
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_sync_attachment_bindings_blob
                ON sync_attachment_revision_bindings(dataset_id, resolved_blob_id)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_sync_attachment_bindings_blob_retention
                ON sync_attachment_revision_bindings(
                    dataset_id, resolved_blob_id, establishing_server_cursor,
                    attachment_id, attachment_revision
                )
                WHERE retention_released_at IS NULL
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_sync_attachment_bindings_retention_release
                ON sync_attachment_revision_bindings(dataset_id, establishing_server_cursor, attachment_id, attachment_revision)
                WHERE retention_released_at IS NULL
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_sync_attachment_bindings_pending_digest
                ON sync_attachment_revision_bindings(
                    dataset_id, blob_hash, size_bytes, establishing_server_cursor,
                    attachment_id, attachment_revision
                )
                WHERE resolved_blob_id IS NULL
                  AND retention_released_at IS NULL
            """,
            """
            CREATE UNIQUE INDEX IF NOT EXISTS uq_sync_dataset_storage_namespace_id
                ON sync_dataset_storage_namespaces(storage_namespace_id)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_sync_dataset_storage_namespaces_owner
                ON sync_dataset_storage_namespaces(owner_user_id, dataset_id)
            """,
        ):
            self.execute(statement, connection=connection)
        if self.backend_type == BackendType.POSTGRESQL:
            self._verify_attachment_binding_tables_postgres(connection=connection)
        else:
            self._verify_attachment_binding_tables_sqlite(
                connection=connection,
                canonical_schema=schema,
            )

    def _ensure_notes_attachment_bootstrap_tables(self, *, connection: Any) -> None:
        """Verify bounded legacy-source identity and cleanup evidence authority."""

        if self.backend_type == BackendType.POSTGRESQL:
            self._verify_notes_attachment_bootstrap_tables_postgres(
                connection=connection,
            )
        else:
            self._verify_notes_attachment_bootstrap_tables_sqlite(
                connection=connection,
                canonical_schema=SYNC_SQLITE_SCHEMA,
                verify_indexes=True,
            )

    def _preflight_notes_attachment_bootstrap_tables(self, *, connection: Any) -> None:
        """Reject partial or malformed pre-existing bootstrap authority."""

        table_names = (
            "sync_notes_attachment_source_map",
            "sync_notes_attachment_cleanup_candidates",
        )
        existing = [
            self.backend.table_exists(table_name, connection=connection)
            for table_name in table_names
        ]
        if not any(existing):
            return
        if not all(existing):
            raise SyncStoreError("Sync attachment bootstrap catalog is malformed")
        self._ensure_notes_attachment_bootstrap_tables(connection=connection)

    def _verify_notes_attachment_bootstrap_tables_sqlite(
        self,
        *,
        connection: Any,
        canonical_schema: str,
        verify_indexes: bool,
    ) -> None:
        expected_columns = {
            "sync_notes_attachment_source_map": [
                ("dataset_id", "TEXT", 1, 1),
                ("bootstrap_id", "TEXT", 1, 2),
                ("source_key_hash", "TEXT", 1, 3),
                ("note_id", "TEXT", 1, 0),
                ("attachment_id", "TEXT", 1, 0),
                ("created_at", "TEXT", 1, 0),
            ],
            "sync_notes_attachment_cleanup_candidates": [
                ("dataset_id", "TEXT", 1, 1),
                ("bootstrap_id", "TEXT", 1, 2),
                ("source_key_hash", "TEXT", 1, 3),
                ("attachment_id", "TEXT", 1, 0),
                ("source_relative_path", "TEXT", 1, 0),
                ("source_path_hash", "TEXT", 1, 0),
                ("source_blob_hash", "TEXT", 1, 0),
                ("source_size_bytes", "INTEGER", 1, 0),
                ("source_modified_ns", "INTEGER", 1, 0),
                ("created_at", "TEXT", 1, 0),
            ],
        }
        expected_table_sql: dict[str, str] = {}
        for statement in canonical_schema.split(";"):
            compact = self._compact_catalog_sql(statement)
            for table_name in expected_columns:
                if f"createtableifnotexists{table_name}(" in compact:
                    expected_table_sql[table_name] = compact.replace(
                        "createtableifnotexists",
                        "createtable",
                        1,
                    )
        for table_name, expected in expected_columns.items():
            actual = [
                (row["name"], row["type"], int(row["notnull"]), int(row["pk"]))
                for row in self.execute(
                    f"PRAGMA table_info({table_name})",  # nosec B608 - fixed names.
                    connection=connection,
                ).rows
            ]
            table_row = _first(
                self.execute(
                    "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
                    (table_name,),
                    connection=connection,
                )
            )
            table_sql = self._compact_catalog_sql(
                None if table_row is None else table_row.get("sql")
            )
            if actual != expected or table_sql != expected_table_sql.get(table_name):
                raise SyncStoreError("Sync attachment bootstrap catalog is malformed")
        if not verify_indexes:
            return
        rows = self.execute(
            "SELECT name, sql FROM sqlite_master WHERE type = 'index' "
            "AND name IN (?, ?) ORDER BY name",
            (
                "idx_sync_notes_attachment_cleanup_page",
                "uq_sync_notes_attachment_source_id",
            ),
            connection=connection,
        ).rows
        actual_indexes = {
            row["name"]: self._compact_catalog_sql(row["sql"]) for row in rows
        }
        expected_indexes = {
            "idx_sync_notes_attachment_cleanup_page": "createindexidx_sync_notes_attachment_cleanup_pageonsync_notes_attachment_cleanup_candidates(dataset_id,bootstrap_id,source_key_hash)",
            "uq_sync_notes_attachment_source_id": "createuniqueindexuq_sync_notes_attachment_source_idonsync_notes_attachment_source_map(dataset_id,attachment_id)",
        }
        if actual_indexes != expected_indexes:
            raise SyncStoreError("Sync attachment bootstrap catalog is malformed")

    def _verify_notes_attachment_bootstrap_tables_postgres(
        self,
        *,
        connection: Any,
    ) -> None:
        expected_columns = {
            "sync_notes_attachment_cleanup_candidates": [
                ("dataset_id", "text", True),
                ("bootstrap_id", "text", True),
                ("source_key_hash", "text", True),
                ("attachment_id", "text", True),
                ("source_relative_path", "text", True),
                ("source_path_hash", "text", True),
                ("source_blob_hash", "text", True),
                ("source_size_bytes", "bigint", True),
                ("source_modified_ns", "bigint", True),
                ("created_at", "timestamp with time zone", True),
            ],
            "sync_notes_attachment_source_map": [
                ("dataset_id", "text", True),
                ("bootstrap_id", "text", True),
                ("source_key_hash", "text", True),
                ("note_id", "text", True),
                ("attachment_id", "text", True),
                ("created_at", "timestamp with time zone", True),
            ],
        }
        rows = self.execute(
            """
            SELECT relation.relname AS table_name,
                   attribute.attname AS column_name,
                   pg_catalog.format_type(attribute.atttypid, attribute.atttypmod) AS data_type,
                   attribute.attnotnull AS is_not_null
              FROM pg_catalog.pg_class AS relation
              JOIN pg_catalog.pg_namespace AS namespace
                ON namespace.oid = relation.relnamespace
              JOIN pg_catalog.pg_attribute AS attribute
                ON attribute.attrelid = relation.oid
             WHERE namespace.nspname = current_schema()
               AND relation.relname IN (
                    'sync_notes_attachment_source_map',
                    'sync_notes_attachment_cleanup_candidates'
               )
               AND relation.relkind = 'r'
               AND attribute.attnum > 0
               AND NOT attribute.attisdropped
             ORDER BY relation.relname, attribute.attnum
            """,
            connection=connection,
        ).rows
        actual_columns = {name: [] for name in expected_columns}
        for row in rows:
            actual_columns[row["table_name"]].append(
                (row["column_name"], row["data_type"], bool(row["is_not_null"]))
            )
        if actual_columns != expected_columns:
            raise SyncStoreError("Sync attachment bootstrap catalog is malformed")
        constraints = self.execute(
            """
            SELECT relation.relname AS table_name,
                   constraint_record.contype AS kind,
                   constraint_record.convalidated AS is_validated,
                   pg_catalog.pg_get_constraintdef(
                       constraint_record.oid, true
                   ) AS definition
              FROM pg_catalog.pg_constraint AS constraint_record
              JOIN pg_catalog.pg_class AS relation
                ON relation.oid = constraint_record.conrelid
              JOIN pg_catalog.pg_namespace AS namespace
                ON namespace.oid = relation.relnamespace
             WHERE namespace.nspname = current_schema()
               AND relation.relname IN (
                    'sync_notes_attachment_source_map',
                    'sync_notes_attachment_cleanup_candidates'
               )
               AND constraint_record.contype IN ('p', 'c')
             ORDER BY relation.relname,
                      constraint_record.contype,
                      constraint_record.conname
            """,
            connection=connection,
        ).rows
        uuid_definition = (
            "checkattachment_id~"
            "'^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-"
            "[89ab][0-9a-f]{3}-[0-9a-f]{12}$'"
        )
        expected_constraints = {
            "sync_notes_attachment_source_map": {
                ("p", "primarykeydataset_id,bootstrap_id,source_key_hash"),
                ("c", "checklengthdataset_id>0"),
                (
                    "c",
                    "checklengthbootstrap_id>=1andlengthbootstrap_id<=128",
                ),
                ("c", "checksource_key_hash~'^sha256:[0-9a-f]{64}$'"),
                ("c", "checklengthnote_id>0"),
                ("c", uuid_definition),
            },
            "sync_notes_attachment_cleanup_candidates": {
                ("p", "primarykeydataset_id,bootstrap_id,source_key_hash"),
                ("c", "checklengthdataset_id>0"),
                (
                    "c",
                    "checklengthbootstrap_id>=1andlengthbootstrap_id<=128",
                ),
                ("c", "checksource_key_hash~'^sha256:[0-9a-f]{64}$'"),
                ("c", uuid_definition),
                (
                    "c",
                    "checklengthsource_relative_path>=1and"
                    "lengthsource_relative_path<=4096",
                ),
                ("c", "checksource_path_hash~'^sha256:[0-9a-f]{64}$'"),
                ("c", "checksource_path_hash=source_key_hash"),
                ("c", "checksource_blob_hash~'^sha256:[0-9a-f]{64}$'"),
                ("c", "checksource_size_bytes>0"),
                ("c", "checksource_modified_ns>=0"),
            },
        }
        actual_constraints = {
            table_name: {
                (
                    str(row["kind"]),
                    self._compact_postgres_catalog_sql(row["definition"]),
                )
                for row in constraints
                if row["table_name"] == table_name
                and bool(row["is_validated"])
            }
            for table_name in expected_constraints
        }
        if actual_constraints != expected_constraints or not all(
            bool(row["is_validated"]) for row in constraints
        ):
            raise SyncStoreError("Sync attachment bootstrap catalog is malformed")
        indexes = self.execute(
            """
            SELECT index_relation.relname AS index_name,
                   table_relation.relname AS table_name,
                   index_record.indisunique AS is_unique,
                   index_record.indisvalid AS is_valid,
                   index_record.indisready AS is_ready,
                   pg_catalog.pg_get_indexdef(index_record.indexrelid) AS definition
              FROM pg_catalog.pg_index AS index_record
              JOIN pg_catalog.pg_class AS index_relation
                ON index_relation.oid = index_record.indexrelid
              JOIN pg_catalog.pg_class AS table_relation
                ON table_relation.oid = index_record.indrelid
              JOIN pg_catalog.pg_namespace AS namespace
                ON namespace.oid = table_relation.relnamespace
             WHERE namespace.nspname = current_schema()
               AND index_relation.relname IN (
                    'idx_sync_notes_attachment_cleanup_page',
                    'uq_sync_notes_attachment_source_id'
               )
             ORDER BY index_relation.relname
            """,
            connection=connection,
        ).rows
        expected_indexes = {
            "idx_sync_notes_attachment_cleanup_page": (
                "sync_notes_attachment_cleanup_candidates",
                False,
                "dataset_id,bootstrap_id,source_key_hash",
            ),
            "uq_sync_notes_attachment_source_id": (
                "sync_notes_attachment_source_map",
                True,
                "dataset_id,attachment_id",
            ),
        }
        if {row["index_name"] for row in indexes} != set(expected_indexes):
            raise SyncStoreError("Sync attachment bootstrap catalog is malformed")
        for row in indexes:
            table_name, unique, columns = expected_indexes[row["index_name"]]
            definition = self._compact_postgres_catalog_sql(row["definition"])
            try:
                definition_tail = definition.split("usingbtree", 1)[1]
            except IndexError:
                definition_tail = ""
            if (
                row["table_name"] != table_name
                or bool(row["is_unique"]) != unique
                or not row["is_valid"]
                or not row["is_ready"]
                or definition_tail != columns
            ):
                raise SyncStoreError("Sync attachment bootstrap catalog is malformed")

    @staticmethod
    def _compact_catalog_sql(value: Any) -> str:
        return "".join(str(value or "").lower().replace('"', "").split())

    def _verify_attachment_binding_tables_sqlite(
        self,
        *,
        connection: Any,
        canonical_schema: str,
    ) -> None:
        expected_columns = {
            "sync_attachment_revision_bindings": [
                ("dataset_id", "TEXT", 1, 1),
                ("attachment_id", "TEXT", 1, 2),
                ("attachment_revision", "INTEGER", 1, 3),
                ("blob_hash", "TEXT", 1, 0),
                ("size_bytes", "INTEGER", 1, 0),
                ("establishing_server_cursor", "INTEGER", 1, 0),
                ("availability_at_acceptance", "TEXT", 1, 0),
                ("resolved_blob_id", "TEXT", 0, 0),
                ("retention_released_at", "TEXT", 0, 0),
                ("created_at", "TEXT", 1, 0),
            ],
            "sync_dataset_storage_namespaces": [
                ("dataset_id", "TEXT", 1, 1),
                ("owner_user_id", "TEXT", 1, 0),
                ("storage_namespace_id", "TEXT", 1, 0),
                ("created_at", "TEXT", 1, 0),
            ],
        }
        expected_table_sql = {}
        for statement in canonical_schema.split(";"):
            compact_statement = self._compact_catalog_sql(statement)
            for table_name in expected_columns:
                if f"createtableifnotexists{table_name}(" in compact_statement:
                    expected_table_sql[table_name] = compact_statement.replace(
                        "createtableifnotexists",
                        "createtable",
                        1,
                    )
        expected_indexes = {
            "idx_sync_attachment_bindings_unresolved": "createindexidx_sync_attachment_bindings_unresolvedonsync_attachment_revision_bindings(dataset_id,establishing_server_cursor,attachment_id,attachment_revision)whereresolved_blob_idisnullandretention_released_atisnull",
            "idx_sync_attachment_bindings_blob": "createindexidx_sync_attachment_bindings_blobonsync_attachment_revision_bindings(dataset_id,resolved_blob_id)",
            "idx_sync_attachment_bindings_blob_retention": "createindexidx_sync_attachment_bindings_blob_retentiononsync_attachment_revision_bindings(dataset_id,resolved_blob_id,establishing_server_cursor,attachment_id,attachment_revision)whereretention_released_atisnull",
            "idx_sync_attachment_bindings_retention_release": "createindexidx_sync_attachment_bindings_retention_releaseonsync_attachment_revision_bindings(dataset_id,establishing_server_cursor,attachment_id,attachment_revision)whereretention_released_atisnull",
            "idx_sync_attachment_bindings_pending_digest": "createindexidx_sync_attachment_bindings_pending_digestonsync_attachment_revision_bindings(dataset_id,blob_hash,size_bytes,establishing_server_cursor,attachment_id,attachment_revision)whereresolved_blob_idisnullandretention_released_atisnull",
            "uq_sync_dataset_storage_namespace_id": "createuniqueindexuq_sync_dataset_storage_namespace_idonsync_dataset_storage_namespaces(storage_namespace_id)",
            "idx_sync_dataset_storage_namespaces_owner": "createindexidx_sync_dataset_storage_namespaces_owneronsync_dataset_storage_namespaces(owner_user_id,dataset_id)",
        }
        for table_name, expected in expected_columns.items():
            rows = self.execute(
                f"PRAGMA table_info({table_name})",  # nosec B608 - fixed catalog names.
                connection=connection,
            ).rows
            actual = [
                (row["name"], row["type"], int(row["notnull"]), int(row["pk"]))
                for row in rows
            ]
            table_row = _first(
                self.execute(
                    "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
                    (table_name,),
                    connection=connection,
                )
            )
            compact = self._compact_catalog_sql(
                None if table_row is None else table_row.get("sql")
            )
            if actual != expected or compact != expected_table_sql.get(table_name):
                raise SyncStoreError("Sync attachment authority catalog is malformed")
        rows = self.execute(
            """
            SELECT name, sql FROM sqlite_master
             WHERE type = 'index'
               AND name IN (
                    'idx_sync_attachment_bindings_unresolved',
                    'idx_sync_attachment_bindings_blob',
                    'idx_sync_attachment_bindings_blob_retention',
                    'idx_sync_attachment_bindings_retention_release',
                    'idx_sync_attachment_bindings_pending_digest',
                    'uq_sync_dataset_storage_namespace_id',
                    'idx_sync_dataset_storage_namespaces_owner'
               )
             ORDER BY name
            """,
            connection=connection,
        ).rows
        actual_indexes = {
            row["name"]: self._compact_catalog_sql(row["sql"])
            for row in rows
        }
        if actual_indexes != expected_indexes:
            raise SyncStoreError("Sync attachment authority catalog is malformed")

    def _verify_attachment_binding_tables_postgres(self, *, connection: Any) -> None:
        expected_columns = {
            "sync_attachment_revision_bindings": [
                ("dataset_id", "text", True),
                ("attachment_id", "text", True),
                ("attachment_revision", "bigint", True),
                ("blob_hash", "text", True),
                ("size_bytes", "bigint", True),
                ("establishing_server_cursor", "bigint", True),
                ("availability_at_acceptance", "text", True),
                ("resolved_blob_id", "text", False),
                ("retention_released_at", "timestamp with time zone", False),
                ("created_at", "timestamp with time zone", True),
            ],
            "sync_dataset_storage_namespaces": [
                ("dataset_id", "text", True),
                ("owner_user_id", "text", True),
                ("storage_namespace_id", "text", True),
                ("created_at", "timestamp with time zone", True),
            ],
        }
        rows = self.execute(
            """
            SELECT relation.relname AS table_name,
                   attribute.attname AS column_name,
                   pg_catalog.format_type(attribute.atttypid, attribute.atttypmod) AS data_type,
                   attribute.attnotnull AS is_not_null
              FROM pg_catalog.pg_class AS relation
              JOIN pg_catalog.pg_namespace AS namespace ON namespace.oid = relation.relnamespace
              JOIN pg_catalog.pg_attribute AS attribute ON attribute.attrelid = relation.oid
             WHERE namespace.nspname = current_schema()
               AND relation.relname IN ('sync_attachment_revision_bindings', 'sync_dataset_storage_namespaces')
               AND relation.relkind = 'r' AND attribute.attnum > 0 AND NOT attribute.attisdropped
             ORDER BY relation.relname, attribute.attnum
            """,
            connection=connection,
        ).rows
        actual_columns = {name: [] for name in expected_columns}
        for row in rows:
            actual_columns[row["table_name"]].append(
                (row["column_name"], row["data_type"], bool(row["is_not_null"]))
            )
        if actual_columns != expected_columns:
            raise SyncStoreError("Sync attachment authority catalog is malformed")
        constraints = self.execute(
            """
            SELECT relation.relname AS table_name, constraint_record.contype AS kind,
                   pg_catalog.pg_get_constraintdef(constraint_record.oid, true) AS definition
              FROM pg_catalog.pg_constraint AS constraint_record
              JOIN pg_catalog.pg_class AS relation ON relation.oid = constraint_record.conrelid
              JOIN pg_catalog.pg_namespace AS namespace ON namespace.oid = relation.relnamespace
             WHERE namespace.nspname = current_schema()
               AND relation.relname IN ('sync_attachment_revision_bindings', 'sync_dataset_storage_namespaces')
               AND constraint_record.contype IN ('p', 'c')
             ORDER BY relation.relname, constraint_record.contype, constraint_record.conname
            """,
            connection=connection,
        ).rows
        expected_constraints = {
            "sync_attachment_revision_bindings": {
                ("p", "primarykeydataset_id,attachment_id,attachment_revision"),
                ("c", "checklengthdataset_id>0"),
                (
                    "c",
                    "checkattachment_id~'^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$'",
                ),
                ("c", "checkattachment_revision>0"),
                ("c", "checkblob_hash~'^sha256:[0-9a-f]{64}$'"),
                ("c", "checksize_bytes>0"),
                ("c", "checkestablishing_server_cursor>0"),
                (
                    "c",
                    "checkavailability_at_acceptance=anyarray['available','metadata_only']",
                ),
                ("c", "checkresolved_blob_idisnullorlengthresolved_blob_id>0"),
            },
            "sync_dataset_storage_namespaces": {
                ("p", "primarykeydataset_id"),
                ("c", "checklengthdataset_id>0"),
                ("c", "checklengthowner_user_id>0"),
                ("c", "checkstorage_namespace_id~'^[0-9a-f]{32}$'"),
            },
        }
        actual_constraints = {
            table_name: {
                (
                    str(row["kind"]),
                    self._compact_postgres_catalog_sql(row["definition"]),
                )
                for row in constraints
                if row["table_name"] == table_name
            }
            for table_name in expected_constraints
        }
        if actual_constraints != expected_constraints:
            raise SyncStoreError("Sync attachment authority catalog is malformed")
        indexes = self.execute(
            """
            SELECT index_relation.relname AS index_name,
                   table_relation.relname AS table_name,
                   index_record.indisunique AS is_unique,
                   index_record.indisvalid AS is_valid,
                   index_record.indisready AS is_ready,
                   pg_catalog.pg_get_indexdef(index_record.indexrelid) AS definition,
                   pg_catalog.pg_get_expr(index_record.indpred, index_record.indrelid) AS predicate
              FROM pg_catalog.pg_index AS index_record
              JOIN pg_catalog.pg_class AS index_relation ON index_relation.oid = index_record.indexrelid
              JOIN pg_catalog.pg_class AS table_relation ON table_relation.oid = index_record.indrelid
              JOIN pg_catalog.pg_namespace AS namespace ON namespace.oid = table_relation.relnamespace
             WHERE namespace.nspname = current_schema()
               AND index_relation.relname IN (
                    'idx_sync_attachment_bindings_unresolved',
                    'idx_sync_attachment_bindings_blob',
                    'idx_sync_attachment_bindings_blob_retention',
                    'idx_sync_attachment_bindings_retention_release',
                    'idx_sync_attachment_bindings_pending_digest',
                    'uq_sync_dataset_storage_namespace_id',
                    'idx_sync_dataset_storage_namespaces_owner'
               )
             ORDER BY index_relation.relname
            """,
            connection=connection,
        ).rows
        expected_indexes = {
            "idx_sync_attachment_bindings_unresolved": (
                "sync_attachment_revision_bindings",
                False,
                "dataset_id,establishing_server_cursor,attachment_id,attachment_revision",
                "resolved_blob_idisnullandretention_released_atisnull",
            ),
            "idx_sync_attachment_bindings_blob": (
                "sync_attachment_revision_bindings",
                False,
                "dataset_id,resolved_blob_id",
                "",
            ),
            "idx_sync_attachment_bindings_blob_retention": (
                "sync_attachment_revision_bindings",
                False,
                "dataset_id,resolved_blob_id,establishing_server_cursor,attachment_id,attachment_revision",
                "retention_released_atisnull",
            ),
            "idx_sync_attachment_bindings_retention_release": (
                "sync_attachment_revision_bindings",
                False,
                "dataset_id,establishing_server_cursor,attachment_id,attachment_revision",
                "retention_released_atisnull",
            ),
            "idx_sync_attachment_bindings_pending_digest": (
                "sync_attachment_revision_bindings",
                False,
                "dataset_id,blob_hash,size_bytes,establishing_server_cursor,attachment_id,attachment_revision",
                "resolved_blob_idisnullandretention_released_atisnull",
            ),
            "uq_sync_dataset_storage_namespace_id": (
                "sync_dataset_storage_namespaces",
                True,
                "storage_namespace_id",
                "",
            ),
            "idx_sync_dataset_storage_namespaces_owner": (
                "sync_dataset_storage_namespaces",
                False,
                "owner_user_id,dataset_id",
                "",
            ),
        }
        if {row["index_name"] for row in indexes} != set(expected_indexes):
            raise SyncStoreError("Sync attachment authority catalog is malformed")
        for row in indexes:
            table_name, unique, columns, predicate = expected_indexes[
                row["index_name"]
            ]
            actual_predicate = self._compact_postgres_catalog_sql(
                row.get("predicate")
            )
            definition = self._compact_postgres_catalog_sql(row["definition"])
            try:
                definition_tail = definition.split("usingbtree", 1)[1]
            except IndexError:
                definition_tail = ""
            expected_tail = columns + (f"where{predicate}" if predicate else "")
            if (
                row["table_name"] != table_name
                or bool(row["is_unique"]) != unique
                or not row["is_valid"] or not row["is_ready"]
                or definition_tail != expected_tail
                or actual_predicate != predicate
            ):
                raise SyncStoreError("Sync attachment authority catalog is malformed")

    @classmethod
    def _compact_postgres_catalog_sql(cls, value: Any) -> str:
        compact = cls._compact_catalog_sql(value)
        for cast in ("::text", "::bigint"):
            compact = compact.replace(cast, "")
        return compact.replace("(", "").replace(")", "")

    def _ensure_envelope_m1_indexes(self, *, connection: Any) -> None:
        statements = [
            """
            CREATE INDEX IF NOT EXISTS idx_sync_envelopes_dataset_domain_object
                ON sync_envelopes(dataset_id, domain, entity_id)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_sync_envelopes_dataset_domain_entity_status_sequence
                ON sync_envelopes(dataset_id, domain, entity_id, status, server_sequence)
            """,
            """
            CREATE UNIQUE INDEX IF NOT EXISTS uq_sync_envelopes_dataset_mutation_group_step
                ON sync_envelopes(dataset_id, mutation_group_id, mutation_step)
                WHERE mutation_group_id IS NOT NULL AND mutation_step IS NOT NULL
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_sync_envelopes_dataset_mutation_group_step
                ON sync_envelopes(dataset_id, mutation_group_id, mutation_step)
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
            """
            CREATE INDEX IF NOT EXISTS idx_sync_envelopes_outstanding_apply
                ON sync_envelopes(dataset_id, server_sequence)
                WHERE status = 'accepted'
                  AND apply_status NOT IN ('applied', 'superseded')
            """,
        ]
        for statement in statements:
            self.execute(statement, connection=connection)

    def _ensure_conflict_indexes(self, *, connection: Any) -> None:
        if self._conflict_identity_index_exists(connection=connection):
            return
        self._dedupe_legacy_conflict_identities(connection=connection)
        self.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS uq_sync_conflicts_dataset_envelope_cursor
                ON sync_conflicts(dataset_id, local_envelope_id, server_sequence)
                WHERE local_envelope_id IS NOT NULL AND server_sequence IS NOT NULL
            """,
            connection=connection,
        )

    def _conflict_identity_index_exists(self, *, connection: Any) -> bool:
        index_name = "uq_sync_conflicts_dataset_envelope_cursor"
        if self.backend_type == BackendType.POSTGRESQL:
            row = _first(
                self.execute(
                    """
                    SELECT indexname
                      FROM pg_indexes
                     WHERE schemaname = current_schema()
                       AND tablename = ?
                       AND indexname = ?
                    """,
                    ("sync_conflicts", index_name),
                    connection=connection,
                )
            )
            return row is not None
        rows = self.execute(
            "PRAGMA index_list(sync_conflicts)",
            connection=connection,
        ).rows
        return any(str(row.get("name")) == index_name for row in rows)

    def _dedupe_legacy_conflict_identities(self, *, connection: Any) -> None:
        rows = self.execute(
            """
            SELECT *
              FROM sync_conflicts
             WHERE local_envelope_id IS NOT NULL
               AND server_sequence IS NOT NULL
             ORDER BY dataset_id, local_envelope_id, server_sequence,
                      created_at, conflict_id
            """,
            connection=connection,
        ).rows
        grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
        for row in rows:
            key = (
                str(row["dataset_id"]),
                str(row["local_envelope_id"]),
                int(row["server_sequence"]),
            )
            grouped.setdefault(key, []).append(row)

        losers: list[str] = []
        for duplicates in grouped.values():
            if len(duplicates) < 2:
                continue
            winner = min(
                duplicates,
                key=lambda row: (str(row.get("created_at") or ""), str(row["conflict_id"])),
            )
            fingerprint = self._legacy_conflict_fingerprint(winner)
            if any(
                self._legacy_conflict_fingerprint(row) != fingerprint
                for row in duplicates
            ):
                raise SyncStoreError(
                    "Sync conflict index migration found incompatible legacy duplicates"
                )
            losers.extend(
                str(row["conflict_id"])
                for row in duplicates
                if row["conflict_id"] != winner["conflict_id"]
            )

        for conflict_id in sorted(losers):
            self.execute(
                "DELETE FROM sync_conflicts WHERE conflict_id = ?",
                (conflict_id,),
                connection=connection,
            )

    @staticmethod
    def _legacy_conflict_fingerprint(row: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            row.get("dataset_id"),
            row.get("domain"),
            row.get("entity_id"),
            row.get("conflict_type"),
            row.get("status"),
            row.get("base_envelope_id"),
            row.get("local_envelope_id"),
            row.get("remote_envelope_id"),
            row.get("server_sequence"),
            encode_json(decode_json(row.get("metadata_json"), default={}), default={}),
            row.get("resolved_by_envelope_id"),
            row.get("resolved_by_device_id"),
            row.get("resolution_action"),
            row.get("resolution_notes"),
            row.get("resolved_at"),
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
