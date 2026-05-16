from __future__ import annotations

"""Internal storage models for Sync v2."""

from dataclasses import dataclass, field
from typing import Any, Literal


SyncDomain = Literal["notes", "chat", "workspaces", "source_cache", "media"]
SyncOperation = Literal["upsert", "delete", "link", "unlink", "resolve_conflict"]
DatasetScopeType = Literal["personal", "workspace"]
EncryptionPolicy = Literal["client_private_v1", "server_trusted", "shared_workspace_v1"]
ConflictStatus = Literal["unresolved", "resolved", "dismissed"]


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
    encryption_policy: EncryptionPolicy = "client_private_v1"
    domains: list[SyncDomain] = field(default_factory=list)
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
    entity_id: str
    operation: SyncOperation
    adapter_version: int
    device_id: str | None = None
    stable_key: str | None = None
    client_timestamp: str | None = None
    base_version: str | int | None = None
    entity_version: str | int | None = None
    dependencies: list[dict[str, Any]] = field(default_factory=list)
    routing_metadata: dict[str, Any] = field(default_factory=dict)
    payload_ciphertext: str | None = None
    payload_clear: dict[str, Any] = field(default_factory=dict)
    payload_hash: str | None = None
    payload_size_bytes: int | None = None
    status: str = "accepted"


@dataclass(frozen=True, slots=True)
class SyncEnvelope:
    """Stored Sync v2 envelope."""

    server_sequence: int
    dataset_id: str
    client_envelope_id: str
    domain: SyncDomain
    entity_id: str
    operation: SyncOperation
    adapter_version: int
    server_timestamp: str
    device_id: str | None = None
    stable_key: str | None = None
    client_timestamp: str | None = None
    base_version: str | int | None = None
    entity_version: str | int | None = None
    dependencies: list[dict[str, Any]] = field(default_factory=list)
    routing_metadata: dict[str, Any] = field(default_factory=dict)
    payload_ciphertext: str | None = None
    payload_clear: dict[str, Any] = field(default_factory=dict)
    payload_hash: str | None = None
    payload_size_bytes: int | None = None
    status: str = "accepted"


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
    entity_id: str
    conflict_type: str
    base_envelope_id: str | None = None
    local_envelope_id: str | None = None
    remote_envelope_id: str | None = None
    server_sequence: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SyncConflict:
    """Stored Sync v2 conflict metadata."""

    conflict_id: str
    dataset_id: str
    domain: SyncDomain
    entity_id: str
    conflict_type: str
    status: ConflictStatus
    base_envelope_id: str | None
    local_envelope_id: str | None
    remote_envelope_id: str | None
    server_sequence: int | None
    metadata: dict[str, Any]
    created_at: str
    resolved_at: str | None = None
    resolved_by_envelope_id: str | None = None
    resolved_by_device_id: str | None = None
    resolution_action: str | None = None
    resolution_notes: str | None = None


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
    """Encrypted attachment payload accepted by the Sync v2 store."""

    attachment_id: str
    dataset_id: str
    domain: SyncDomain
    entity_id: str
    content_type: str
    size_bytes: int
    payload_ciphertext: str
    payload_hash: str
    encryption_policy: EncryptionPolicy = "client_private_v1"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SyncAttachment:
    """Stored encrypted attachment payload metadata."""

    attachment_id: str
    dataset_id: str
    domain: SyncDomain
    entity_id: str
    content_type: str
    size_bytes: int
    payload_ciphertext: str
    payload_hash: str
    encryption_policy: EncryptionPolicy
    metadata: dict[str, Any]
    created_at: str
    stored: bool = True


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
    "DatasetScopeType",
    "EncryptionPolicy",
    "SyncAttachment",
    "SyncAttachmentCreate",
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
    "SyncOperation",
    "SyncRestoreManifestStats",
]
