from __future__ import annotations

"""Internal storage models for Sync v2 M1."""

from dataclasses import dataclass, field
from typing import Any, Literal

SyncDomain = Literal["notes.note", "chat.conversation", "chat.message", "attachment.ref"]
SyncOperation = Literal["upsert", "append", "tombstone"]
DatasetScopeType = Literal["personal", "workspace"]
EncryptionPolicy = Literal["server_trusted_v1"]
ConflictStatus = Literal["unresolved", "resolved", "dismissed"]
SyncApplyStatus = Literal["pending", "applied", "failed", "conflict"]
SyncBlobAvailabilityStatus = Literal[
    "metadata_only",
    "uploading",
    "available",
    "verify_failed",
    "quarantined",
    "deleted",
]
SyncBlobUploadStatus = Literal[
    "created",
    "uploading",
    "complete",
    "cancelled",
    "expired",
    "verify_failed",
]
SyncRestoreCompletenessStatus = Literal[
    "metadata_ready",
    "blocked_by_conflicts",
    "blob_incomplete",
    "content_complete",
    "verified_complete",
]

M1_SYNC_DOMAINS: list[SyncDomain] = [
    "notes.note",
    "chat.conversation",
    "chat.message",
    "attachment.ref",
]
M1_SYNC_OPERATIONS: dict[SyncDomain, list[SyncOperation]] = {
    "notes.note": ["upsert", "tombstone"],
    "chat.conversation": ["upsert", "tombstone"],
    "chat.message": ["append", "tombstone"],
    "attachment.ref": ["upsert", "tombstone"],
}
DEFAULT_M1_ENCRYPTION_POLICY: EncryptionPolicy = "server_trusted_v1"


def _coalesce_identity(primary: str | None, legacy: str | None, *, field_name: str) -> str:
    value = primary or legacy
    if not value:
        raise ValueError(f"{field_name} is required")
    return value


def _coalesce_payload(
    payload: dict[str, Any],
    payload_clear: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if payload:
        return dict(payload), dict(payload)
    if payload_clear:
        return {}, dict(payload_clear)
    return {}, {}


@dataclass(frozen=True, slots=True)
class SyncDeviceUpsert:
    """Device registration data accepted by the Sync v2 store."""

    device_id: str
    user_id: str
    display_name: str
    client_type: str
    client_version: str | None = None
    capabilities: dict[str, Any] = field(default_factory=dict)
    revoked_at: str | None = None


@dataclass(frozen=True, slots=True)
class SyncDevice:
    """Stored Sync v2 device metadata."""

    device_id: str
    user_id: str
    display_name: str
    client_type: str
    client_version: str | None
    capabilities: dict[str, Any]
    registered_at: str
    last_seen_at: str
    revoked_at: str | None = None


@dataclass(frozen=True, slots=True)
class SyncDatasetCreate:
    """Dataset enrollment data accepted by the Sync v2 store."""

    dataset_id: str
    owner_user_id: str
    scope_type: DatasetScopeType = "personal"
    encryption_policy: EncryptionPolicy = DEFAULT_M1_ENCRYPTION_POLICY
    domains: list[SyncDomain] = field(default_factory=lambda: list(M1_SYNC_DOMAINS))
    workspace_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    archived_at: str | None = None


@dataclass(frozen=True, slots=True)
class SyncDataset:
    """Stored Sync v2 dataset metadata."""

    dataset_id: str
    owner_user_id: str
    scope_type: DatasetScopeType
    encryption_policy: EncryptionPolicy
    domains: list[SyncDomain]
    workspace_id: str | None
    metadata: dict[str, Any]
    created_at: str
    updated_at: str
    archived_at: str | None = None


@dataclass(frozen=True, slots=True)
class SyncEnvelopeCreate:
    """Envelope data accepted by the Sync v2 store."""

    dataset_id: str
    client_envelope_id: str
    domain: SyncDomain
    operation: SyncOperation
    object_id: str | None = None
    entity_id: str | None = None
    device_id: str | None = None
    client_profile_id: str | None = None
    client_sequence: int | None = None
    server_cursor: int | None = None
    server_sequence: int | None = None
    base_server_cursor: int | None = None
    base_object_revision: int | None = None
    base_object_hash: str | None = None
    object_revision: int | None = None
    parent_id: str | None = None
    schema_version: int = 1
    payload: dict[str, Any] = field(default_factory=dict)
    payload_hash: str | None = None
    payload_size_bytes: int | None = None
    created_at_client: str | None = None
    received_at_server: str | None = None
    deleted: bool = False
    encryption_metadata: dict[str, Any] = field(
        default_factory=lambda: {"policy": DEFAULT_M1_ENCRYPTION_POLICY}
    )
    status: str = "accepted"
    apply_status: SyncApplyStatus = "pending"
    apply_error_code: str | None = None
    apply_error_message: str | None = None
    applied_at: str | None = None
    payload_ciphertext: str | None = None
    payload_clear: dict[str, Any] = field(default_factory=dict)
    stable_key: str | None = None
    dependencies: list[dict[str, Any]] = field(default_factory=list)
    routing_metadata: dict[str, Any] = field(default_factory=dict)
    adapter_version: int = 1
    base_version: str | int | None = None
    entity_version: str | int | None = None
    client_timestamp: str | None = None
    server_timestamp: str | None = None

    def __post_init__(self) -> None:
        object_id = _coalesce_identity(self.object_id, self.entity_id, field_name="object_id")
        payload, payload_clear = _coalesce_payload(self.payload, self.payload_clear)
        object.__setattr__(self, "object_id", object_id)
        object.__setattr__(self, "entity_id", object_id)
        server_cursor = self.server_cursor if self.server_cursor is not None else self.server_sequence
        object.__setattr__(self, "server_cursor", server_cursor)
        object.__setattr__(self, "server_sequence", server_cursor)
        object.__setattr__(self, "payload", payload)
        object.__setattr__(self, "payload_clear", payload_clear)
        object.__setattr__(self, "schema_version", self.schema_version or self.adapter_version)
        object.__setattr__(self, "adapter_version", self.adapter_version or self.schema_version)
        if self.created_at_client is None and self.client_timestamp is not None:
            object.__setattr__(self, "created_at_client", self.client_timestamp)
        if self.client_timestamp is None and self.created_at_client is not None:
            object.__setattr__(self, "client_timestamp", self.created_at_client)
        if self.received_at_server is None and self.server_timestamp is not None:
            object.__setattr__(self, "received_at_server", self.server_timestamp)
        if self.server_timestamp is None and self.received_at_server is not None:
            object.__setattr__(self, "server_timestamp", self.received_at_server)
        if self.base_object_revision is None and isinstance(self.base_version, int):
            object.__setattr__(self, "base_object_revision", self.base_version)
        if self.object_revision is None and isinstance(self.entity_version, int):
            object.__setattr__(self, "object_revision", self.entity_version)


@dataclass(frozen=True, slots=True)
class SyncEnvelope:
    """Stored Sync v2 envelope."""

    dataset_id: str
    client_envelope_id: str
    domain: SyncDomain
    operation: SyncOperation
    server_cursor: int | None = None
    object_id: str | None = None
    entity_id: str | None = None
    server_sequence: int | None = None
    envelope_id: str | None = None
    device_id: str | None = None
    client_profile_id: str | None = None
    client_sequence: int | None = None
    base_server_cursor: int | None = None
    base_object_revision: int | None = None
    base_object_hash: str | None = None
    object_revision: int | None = None
    parent_id: str | None = None
    schema_version: int = 1
    payload: dict[str, Any] = field(default_factory=dict)
    payload_hash: str | None = None
    payload_size_bytes: int | None = None
    created_at_client: str | None = None
    received_at_server: str | None = None
    deleted: bool = False
    encryption_metadata: dict[str, Any] = field(default_factory=dict)
    status: str = "accepted"
    apply_status: SyncApplyStatus = "pending"
    apply_error_code: str | None = None
    apply_error_message: str | None = None
    applied_at: str | None = None
    payload_ciphertext: str | None = None
    payload_clear: dict[str, Any] = field(default_factory=dict)
    stable_key: str | None = None
    dependencies: list[dict[str, Any]] = field(default_factory=list)
    routing_metadata: dict[str, Any] = field(default_factory=dict)
    adapter_version: int = 1
    base_version: str | int | None = None
    entity_version: str | int | None = None
    client_timestamp: str | None = None
    server_timestamp: str | None = None

    def __post_init__(self) -> None:
        object_id = _coalesce_identity(self.object_id, self.entity_id, field_name="object_id")
        server_cursor = self.server_cursor if self.server_cursor is not None else self.server_sequence
        if server_cursor is None:
            raise ValueError("server_cursor is required")
        payload, payload_clear = _coalesce_payload(self.payload, self.payload_clear)
        object.__setattr__(self, "server_cursor", server_cursor)
        object.__setattr__(self, "server_sequence", server_cursor)
        object.__setattr__(self, "object_id", object_id)
        object.__setattr__(self, "entity_id", object_id)
        object.__setattr__(self, "payload", payload)
        object.__setattr__(self, "payload_clear", payload_clear)
        object.__setattr__(self, "schema_version", self.schema_version or self.adapter_version)
        object.__setattr__(self, "adapter_version", self.adapter_version or self.schema_version)
        if self.created_at_client is None and self.client_timestamp is not None:
            object.__setattr__(self, "created_at_client", self.client_timestamp)
        if self.client_timestamp is None and self.created_at_client is not None:
            object.__setattr__(self, "client_timestamp", self.created_at_client)
        if self.received_at_server is None and self.server_timestamp is not None:
            object.__setattr__(self, "received_at_server", self.server_timestamp)
        if self.server_timestamp is None and self.received_at_server is not None:
            object.__setattr__(self, "server_timestamp", self.received_at_server)
        if self.base_object_revision is None and isinstance(self.base_version, int):
            object.__setattr__(self, "base_object_revision", self.base_version)
        if self.object_revision is None and isinstance(self.entity_version, int):
            object.__setattr__(self, "object_revision", self.entity_version)


@dataclass(frozen=True, slots=True)
class SyncObjectState:
    """Materialized latest object state tracked by Sync v2."""

    dataset_id: str
    domain: SyncDomain
    object_id: str
    object_revision: int
    object_hash: str
    latest_server_cursor: int
    deleted: bool = False
    updated_at: str | None = None


@dataclass(frozen=True, slots=True)
class SyncDeviceCursor:
    """Per-device pull cursor for one domain."""

    dataset_id: str
    device_id: str
    domain: SyncDomain
    last_pulled_sequence: int
    updated_at: str | None = None


@dataclass(frozen=True, slots=True)
class SyncConflictCreate:
    """Conflict metadata accepted by the Sync v2 store."""

    conflict_id: str
    dataset_id: str
    domain: SyncDomain
    conflict_type: str
    object_id: str | None = None
    entity_id: str | None = None
    base_envelope_id: str | None = None
    local_envelope_id: str | None = None
    remote_envelope_id: str | None = None
    server_cursor: int | None = None
    server_sequence: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object_id = _coalesce_identity(self.object_id, self.entity_id, field_name="object_id")
        server_cursor = self.server_cursor if self.server_cursor is not None else self.server_sequence
        object.__setattr__(self, "object_id", object_id)
        object.__setattr__(self, "entity_id", object_id)
        object.__setattr__(self, "server_cursor", server_cursor)
        object.__setattr__(self, "server_sequence", server_cursor)


@dataclass(frozen=True, slots=True)
class SyncConflict:
    """Stored Sync v2 conflict metadata."""

    conflict_id: str
    dataset_id: str
    domain: SyncDomain
    object_id: str | None
    conflict_type: str
    status: ConflictStatus
    base_envelope_id: str | None
    local_envelope_id: str | None
    remote_envelope_id: str | None
    server_cursor: int | None
    metadata: dict[str, Any]
    created_at: str
    entity_id: str | None = None
    server_sequence: int | None = None
    resolved_at: str | None = None
    resolved_by_envelope_id: str | None = None
    resolved_by_device_id: str | None = None
    resolution_action: str | None = None
    resolution_notes: str | None = None

    def __post_init__(self) -> None:
        object_id = _coalesce_identity(self.object_id, self.entity_id, field_name="object_id")
        server_cursor = self.server_cursor if self.server_cursor is not None else self.server_sequence
        object.__setattr__(self, "object_id", object_id)
        object.__setattr__(self, "entity_id", object_id)
        object.__setattr__(self, "server_cursor", server_cursor)
        object.__setattr__(self, "server_sequence", server_cursor)


@dataclass(frozen=True, slots=True)
class SyncKeyRecordCreate:
    """Encrypted key material accepted by the Sync v2 store."""

    key_record_id: str
    dataset_id: str
    user_id: str
    key_purpose: str
    wrapped_key_blob: str
    device_id: str | None = None
    kdf_metadata: dict[str, Any] = field(default_factory=dict)
    recovery_hint: str | None = None
    rotation_of_key_record_id: str | None = None
    revoked_at: str | None = None


@dataclass(frozen=True, slots=True)
class SyncKeyRecord:
    """Stored encrypted key material metadata."""

    key_record_id: str
    dataset_id: str
    user_id: str
    key_purpose: str
    wrapped_key_blob: str
    device_id: str | None
    kdf_metadata: dict[str, Any]
    recovery_hint: str | None
    rotation_of_key_record_id: str | None
    created_at: str
    revoked_at: str | None = None


@dataclass(frozen=True, slots=True)
class SyncAttachmentCreate:
    """Attachment payload accepted by the Sync v2 store."""

    attachment_id: str
    dataset_id: str
    domain: SyncDomain
    content_type: str
    size_bytes: int
    payload_ciphertext: str
    payload_hash: str
    object_id: str | None = None
    entity_id: str | None = None
    encryption_policy: EncryptionPolicy = DEFAULT_M1_ENCRYPTION_POLICY
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object_id = _coalesce_identity(self.object_id, self.entity_id, field_name="object_id")
        object.__setattr__(self, "object_id", object_id)
        object.__setattr__(self, "entity_id", object_id)


@dataclass(frozen=True, slots=True)
class SyncAttachment:
    """Stored attachment payload metadata."""

    attachment_id: str
    dataset_id: str
    domain: SyncDomain
    object_id: str | None
    content_type: str
    size_bytes: int
    payload_ciphertext: str
    payload_hash: str
    encryption_policy: EncryptionPolicy
    metadata: dict[str, Any]
    created_at: str
    entity_id: str | None = None
    stored: bool = True

    def __post_init__(self) -> None:
        object_id = _coalesce_identity(self.object_id, self.entity_id, field_name="object_id")
        object.__setattr__(self, "object_id", object_id)
        object.__setattr__(self, "entity_id", object_id)


@dataclass(frozen=True, slots=True)
class SyncBlobUploadSession:
    """Core metadata for a resumable Sync v2 M2 blob upload session."""

    upload_id: str
    dataset_id: str
    attachment_id: str
    status: SyncBlobUploadStatus
    chunk_size: int
    chunk_count: int
    size_bytes: int
    payload_hash: str
    content_type: str
    uploaded_chunks: list[int] = field(default_factory=list)
    missing_chunks: list[int] = field(default_factory=list)
    quota: dict[str, Any] = field(default_factory=dict)
    expires_at: str | None = None
    blob_id: str | None = None


@dataclass(frozen=True, slots=True)
class SyncBlobUploadSessionCreate:
    """Upload-session metadata accepted by the Sync v2 M2 store."""

    upload_id: str
    dataset_id: str
    owner_user_id: str
    device_id: str | None
    attachment_id: str
    domain: SyncDomain
    object_id: str
    content_type: str
    size_bytes: int
    payload_hash: str
    chunk_size: int
    chunk_count: int
    reserved_quota_bytes: int
    status: SyncBlobUploadStatus = "created"
    idempotency_key: str | None = None
    expires_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SyncBlobChunkCreate:
    """Chunk metadata accepted by the Sync v2 M2 store."""

    upload_id: str
    dataset_id: str
    chunk_index: int
    offset_bytes: int
    size_bytes: int
    chunk_hash: str
    storage_key: str


@dataclass(frozen=True, slots=True)
class SyncBlobChunk:
    """Stored chunk metadata for one upload session."""

    upload_id: str
    dataset_id: str
    chunk_index: int
    offset_bytes: int
    size_bytes: int
    chunk_hash: str
    storage_key: str
    received_at: str


@dataclass(frozen=True, slots=True)
class SyncBlobObjectCreate:
    """Committed blob metadata accepted by the Sync v2 M2 store."""

    blob_id: str
    dataset_id: str
    owner_user_id: str
    attachment_id: str
    payload_hash: str
    content_type: str
    size_bytes: int
    storage_backend: str
    storage_key: str
    encryption_policy: EncryptionPolicy = DEFAULT_M1_ENCRYPTION_POLICY
    status: SyncBlobAvailabilityStatus = "available"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SyncBlobObject:
    """Committed blob metadata stored by Sync v2 M2."""

    blob_id: str
    dataset_id: str
    owner_user_id: str
    attachment_id: str
    payload_hash: str
    content_type: str
    size_bytes: int
    encryption_policy: EncryptionPolicy
    storage_backend: str
    storage_key: str
    status: SyncBlobAvailabilityStatus
    ref_count: int
    metadata: dict[str, Any]
    created_at: str
    updated_at: str
    deleted_at: str | None = None


@dataclass(frozen=True, slots=True)
class SyncBlobQuotaUsage:
    """Quota counters for committed and pending Sync v2 M2 blobs."""

    owner_user_id: str
    dataset_id: str | None = None
    reserved_blob_bytes: int = 0
    used_blob_bytes: int = 0
    active_upload_count: int = 0


@dataclass(frozen=True, slots=True)
class SyncBlobDownloadChunk:
    """Core chunk entry used by a resumable Sync v2 M2 blob download manifest."""

    chunk_index: int
    offset_bytes: int
    size_bytes: int
    chunk_hash: str
    download_url: str | None = None


@dataclass(frozen=True, slots=True)
class SyncRestoreDomainCompleteness:
    """Per-domain restore completeness counters for Sync v2 M2."""

    domain: SyncDomain
    status: SyncRestoreCompletenessStatus
    selected_count: int = 0
    safe_apply_count: int = 0
    conflict_count: int = 0
    tombstone_count: int = 0
    required_blob_count: int = 0
    available_blob_count: int = 0
    missing_blob_count: int = 0
    verified_blob_count: int = 0


@dataclass(frozen=True, slots=True)
class SyncRestoreBlobCompleteness:
    """Per-blob restore completeness detail for Sync v2 M2."""

    attachment_id: str
    payload_hash: str
    size_bytes: int
    content_type: str
    parent_domain: SyncDomain
    parent_object_id: str
    server_availability: SyncBlobAvailabilityStatus
    download_status: str | None = None
    required_for_restore: bool = True


@dataclass(frozen=True, slots=True)
class SyncRestoreManifestStats:
    """Database-side aggregate statistics for one restore-manifest dataset."""

    approximate_counts: dict[str, int] = field(default_factory=dict)
    byte_estimates: dict[str, int] = field(default_factory=dict)
    last_updated_at: str | None = None
    unresolved_conflicts: int = 0
    attachment_availability: dict[str, int] = field(default_factory=dict)
    attachment_size_classes: dict[str, int] = field(default_factory=dict)
    key_recovery_available: bool = False


__all__ = [
    "ConflictStatus",
    "DEFAULT_M1_ENCRYPTION_POLICY",
    "DatasetScopeType",
    "EncryptionPolicy",
    "M1_SYNC_DOMAINS",
    "M1_SYNC_OPERATIONS",
    "SyncApplyStatus",
    "SyncAttachment",
    "SyncAttachmentCreate",
    "SyncBlobAvailabilityStatus",
    "SyncBlobChunk",
    "SyncBlobChunkCreate",
    "SyncBlobDownloadChunk",
    "SyncBlobObject",
    "SyncBlobObjectCreate",
    "SyncBlobQuotaUsage",
    "SyncBlobUploadSession",
    "SyncBlobUploadSessionCreate",
    "SyncBlobUploadStatus",
    "SyncConflict",
    "SyncConflictCreate",
    "SyncDataset",
    "SyncDatasetCreate",
    "SyncDevice",
    "SyncDeviceCursor",
    "SyncDeviceUpsert",
    "SyncDomain",
    "SyncEnvelope",
    "SyncEnvelopeCreate",
    "SyncKeyRecord",
    "SyncKeyRecordCreate",
    "SyncObjectState",
    "SyncOperation",
    "SyncRestoreBlobCompleteness",
    "SyncRestoreCompletenessStatus",
    "SyncRestoreDomainCompleteness",
    "SyncRestoreManifestStats",
]
