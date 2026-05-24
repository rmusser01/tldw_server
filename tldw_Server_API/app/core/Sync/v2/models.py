from __future__ import annotations

"""Internal storage models for Sync v2 M1."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

SyncDomain = Literal[
    "notes.note",
    "chat.conversation",
    "chat.message",
    "attachment.ref",
    "workspaces.workspace",
    "workspaces.source_ref",
    "source_cache.entry",
    "media.item",
    "media.keyword",
    "media.keyword_link",
]
SyncOperation = Literal["upsert", "append", "tombstone"]
DatasetScopeType = Literal["personal", "workspace"]
EncryptionPolicy = Literal[
    "server_trusted_v1",
    "passphrase_wrapped_v1",
    "device_wrapped_v1",
    "client_private_v1",
]
SyncKeyWrappedFor = Literal["server", "passphrase", "device", "recovery"]
SyncKeyRewrapStatus = Literal["not_required", "pending", "complete", "failed", "blocked"]
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
SyncDeviceStatus = Literal["pending_authorization", "active", "paused", "revoked"]
SyncDeviceAuthorizationStatus = Literal["pending", "approved", "rejected"]
SyncBackgroundLeaseStatus = Literal["acquired", "refreshed", "held_by_other"]
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
WORKSPACE_SYNC_DOMAINS: list[SyncDomain] = [
    "workspaces.workspace",
    "workspaces.source_ref",
]
WORKSPACE_SYNC_OPERATIONS: dict[SyncDomain, list[SyncOperation]] = {
    "workspaces.workspace": ["upsert", "tombstone"],
    "workspaces.source_ref": ["upsert", "tombstone"],
}
SOURCE_CACHE_SYNC_DOMAINS: list[SyncDomain] = ["source_cache.entry"]
SOURCE_CACHE_SYNC_OPERATIONS: dict[SyncDomain, list[SyncOperation]] = {
    "source_cache.entry": ["upsert", "tombstone"],
}
MEDIA_SYNC_DOMAINS: list[SyncDomain] = [
    "media.item",
    "media.keyword",
    "media.keyword_link",
]
MEDIA_SYNC_OPERATIONS: dict[SyncDomain, list[SyncOperation]] = {
    "media.item": ["upsert", "tombstone"],
    "media.keyword": ["upsert", "tombstone"],
    "media.keyword_link": ["upsert", "tombstone"],
}
SYNC_V2_SUPPORTED_DOMAINS: list[SyncDomain] = (
    list(M1_SYNC_DOMAINS)
    + list(WORKSPACE_SYNC_DOMAINS)
    + list(SOURCE_CACHE_SYNC_DOMAINS)
    + list(MEDIA_SYNC_DOMAINS)
)
SYNC_V2_SUPPORTED_OPERATIONS: dict[SyncDomain, list[SyncOperation]] = {
    **M1_SYNC_OPERATIONS,
    **WORKSPACE_SYNC_OPERATIONS,
    **SOURCE_CACHE_SYNC_OPERATIONS,
    **MEDIA_SYNC_OPERATIONS,
}
DEFAULT_M1_ENCRYPTION_POLICY: EncryptionPolicy = "server_trusted_v1"
SYNC_V2_ENCRYPTION_POLICIES: list[EncryptionPolicy] = [
    "server_trusted_v1",
    "passphrase_wrapped_v1",
    "device_wrapped_v1",
    "client_private_v1",
]
STRICT_ENCRYPTION_POLICIES: list[EncryptionPolicy] = [
    "passphrase_wrapped_v1",
    "device_wrapped_v1",
    "client_private_v1",
]
SYNC_KEY_WRAPPED_FOR_VALUES: list[SyncKeyWrappedFor] = [
    "server",
    "passphrase",
    "device",
    "recovery",
]
SYNC_KEY_REWRAP_STATUSES: list[SyncKeyRewrapStatus] = [
    "not_required",
    "pending",
    "complete",
    "failed",
    "blocked",
]


@dataclass(frozen=True, slots=True)
class SyncEncryptionPolicyMetadata:
    """Validated public metadata for a Sync v2 dataset encryption policy."""

    policy: EncryptionPolicy = DEFAULT_M1_ENCRYPTION_POLICY
    key_epoch: int = 1
    attestation: dict[str, Any] = field(default_factory=dict)
    kdf_metadata: dict[str, Any] = field(default_factory=dict)
    recovery_key_record_id: str | None = None
    device_key_record_ids: list[str] = field(default_factory=list)
    server_materialization: str | None = None

    def __post_init__(self) -> None:
        attestation = dict(self.attestation)
        kdf_metadata = dict(self.kdf_metadata)
        device_key_record_ids = [
            str(record_id).strip()
            for record_id in self.device_key_record_ids
            if str(record_id).strip()
        ]
        _validate_encryption_policy_metadata(
            policy=self.policy,
            key_epoch=self.key_epoch,
            attestation=attestation,
            kdf_metadata=kdf_metadata,
            recovery_key_record_id=self.recovery_key_record_id,
            device_key_record_ids=device_key_record_ids,
            server_materialization=self.server_materialization,
        )
        object.__setattr__(self, "attestation", attestation)
        object.__setattr__(self, "kdf_metadata", kdf_metadata)
        object.__setattr__(self, "device_key_record_ids", device_key_record_ids)


def _validate_encryption_policy_metadata(
    *,
    policy: EncryptionPolicy,
    key_epoch: int,
    attestation: dict[str, Any],
    kdf_metadata: dict[str, Any],
    recovery_key_record_id: str | None,
    device_key_record_ids: list[str],
    server_materialization: str | None,
) -> None:
    if policy not in SYNC_V2_ENCRYPTION_POLICIES:
        raise ValueError(f"unsupported Sync v2 encryption policy: {policy}")
    if isinstance(key_epoch, bool) or key_epoch < 1:
        raise ValueError("encryption policy key_epoch must be greater than or equal to 1")
    if policy == "server_trusted_v1":
        _validate_server_trusted_policy_metadata(attestation)
        return
    if policy == "passphrase_wrapped_v1":
        _validate_passphrase_wrapped_policy_metadata(
            kdf_metadata=kdf_metadata,
            recovery_key_record_id=recovery_key_record_id,
        )
        return
    if policy == "device_wrapped_v1":
        if not device_key_record_ids:
            raise ValueError("device_wrapped_v1 requires at least one device key record")
        return
    if policy == "client_private_v1" and server_materialization != "metadata_only":
        raise ValueError("client_private_v1 requires metadata_only server materialization")


def _validate_server_trusted_policy_metadata(attestation: dict[str, Any]) -> None:
    if attestation.get("configured") is not True:
        raise ValueError("server_trusted_v1 requires configured attestation metadata")
    if not str(attestation.get("scope") or "").strip():
        raise ValueError("server_trusted_v1 requires attestation scope metadata")
    covers = attestation.get("covers")
    if not isinstance(covers, list) or not any(str(item).strip() for item in covers):
        raise ValueError("server_trusted_v1 requires covered storage metadata")


def _validate_passphrase_wrapped_policy_metadata(
    *,
    kdf_metadata: dict[str, Any],
    recovery_key_record_id: str | None,
) -> None:
    if not str(kdf_metadata.get("algorithm") or "").strip():
        raise ValueError("passphrase_wrapped_v1 requires KDF algorithm metadata")
    params_hash = str(kdf_metadata.get("params_hash") or "").strip()
    if not params_hash.startswith("sha256:") or params_hash == "sha256:":
        raise ValueError("passphrase_wrapped_v1 requires a sha256 KDF params hash")
    if not str(recovery_key_record_id or "").strip():
        raise ValueError("passphrase_wrapped_v1 requires a recovery key record reference")


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
    status: SyncDeviceStatus = "active"
    user_label: str | None = None
    authorized_at: str | None = None
    revoked_at: str | None = None
    revoked_reason: str | None = None


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
    status: SyncDeviceStatus = "active"
    user_label: str | None = None
    authorized_at: str | None = None
    revoked_at: str | None = None
    revoked_reason: str | None = None


@dataclass(frozen=True, slots=True)
class SyncDeviceAuthorizationCreate:
    """Device authorization request accepted by the Sync v2 store."""

    authorization_id: str
    dataset_id: str
    user_id: str
    device_id: str
    authorization_method: str
    idempotency_key: str | None = None


@dataclass(frozen=True, slots=True)
class SyncDeviceAuthorization:
    """Stored device authorization request."""

    authorization_id: str
    dataset_id: str
    user_id: str
    device_id: str
    authorization_method: str
    status: SyncDeviceAuthorizationStatus
    requested_at: str
    approved_at: str | None = None
    approving_device_id: str | None = None
    idempotency_key: str | None = None


@dataclass(frozen=True, slots=True)
class SyncDeviceDomainAckCreate:
    """Per-device domain acknowledgment accepted by the Sync v2 store."""

    dataset_id: str
    device_id: str
    domain: SyncDomain
    through_server_sequence: int
    applied_at: str
    idempotency_key: str | None = None


@dataclass(frozen=True, slots=True)
class SyncDeviceDomainAck:
    """Stored per-device domain acknowledgment."""

    dataset_id: str
    device_id: str
    domain: SyncDomain
    through_server_sequence: int
    applied_at: str
    updated_at: str
    idempotency_key: str | None = None


@dataclass(frozen=True, slots=True)
class SyncDeviceBlobAckCreate:
    """Per-device blob verification acknowledgment accepted by the Sync v2 store."""

    dataset_id: str
    device_id: str
    attachment_id: str
    payload_hash: str
    verified_at: str
    idempotency_key: str | None = None


@dataclass(frozen=True, slots=True)
class SyncDeviceBlobAck:
    """Stored per-device blob verification acknowledgment."""

    dataset_id: str
    device_id: str
    attachment_id: str
    payload_hash: str
    verified_at: str
    updated_at: str
    idempotency_key: str | None = None


@dataclass(frozen=True, slots=True)
class SyncDeviceAcknowledgmentSummary:
    """Aggregated device acknowledgments for one dataset/device."""

    dataset_id: str
    device_id: str
    domain_acks: dict[SyncDomain, SyncDeviceDomainAck] = field(default_factory=dict)
    blob_acks: list[SyncDeviceBlobAck] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class SyncBackgroundPolicyUpsert:
    """Background sync policy and user intent accepted by the Sync v2 store."""

    dataset_id: str
    device_id: str
    enabled: bool = True
    minimum_interval_seconds: int = 300
    backoff_floor_seconds: int = 60
    max_batch_size: int = 100
    max_blob_bytes_per_run: int | None = None
    respect_metered_networks: bool = True
    maintenance_window: dict[str, Any] | None = None
    paused_reason: str | None = None
    pending_local_changes: bool = False


@dataclass(frozen=True, slots=True)
class SyncBackgroundPolicy:
    """Stored background sync policy and user intent for one dataset/device."""

    dataset_id: str
    device_id: str
    enabled: bool
    minimum_interval_seconds: int
    backoff_floor_seconds: int
    max_batch_size: int
    max_blob_bytes_per_run: int | None
    respect_metered_networks: bool
    maintenance_window: dict[str, Any] | None
    paused_reason: str | None
    pending_local_changes: bool
    updated_at: str


@dataclass(frozen=True, slots=True)
class SyncBackgroundLeaseCreate:
    """Advisory background sync lease request accepted by the Sync v2 store."""

    dataset_id: str
    device_id: str
    lease_id: str
    ttl_seconds: int
    requested_at: str | None = None


@dataclass(frozen=True, slots=True)
class SyncBackgroundLease:
    """Stored advisory background sync lease."""

    dataset_id: str
    device_id: str
    lease_id: str
    status: SyncBackgroundLeaseStatus
    acquired: bool
    expires_at: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class SyncBackgroundDomainStatus:
    """Aggregated background sync status for one domain."""

    domain: SyncDomain
    last_server_sequence: int = 0
    last_pulled_sequence: int = 0
    cursor_lag_count: int = 0
    unresolved_conflicts: int = 0
    replayable_failures: int = 0
    last_successful_push_at: str | None = None
    last_successful_pull_at: str | None = None
    blob_completeness: dict[str, int] = field(default_factory=dict)


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
    rotation_source_key_record_ids: tuple[str, ...] = ()
    revoked_at: str | None = None
    encryption_policy: EncryptionPolicy = DEFAULT_M1_ENCRYPTION_POLICY
    key_epoch: int = 1
    active_from_server_sequence: int | None = None
    superseded_at: str | None = None
    wrapped_for: SyncKeyWrappedFor = "recovery"
    rewrap_status: SyncKeyRewrapStatus = "not_required"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "rotation_source_key_record_ids",
            _normalize_key_record_source_ids(self.rotation_source_key_record_ids),
        )
        _validate_key_record_rotation_metadata(
            encryption_policy=self.encryption_policy,
            key_epoch=self.key_epoch,
            active_from_server_sequence=self.active_from_server_sequence,
            wrapped_for=self.wrapped_for,
            rewrap_status=self.rewrap_status,
        )


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
    rotation_source_key_record_ids: tuple[str, ...] = ()
    revoked_at: str | None = None
    encryption_policy: EncryptionPolicy = DEFAULT_M1_ENCRYPTION_POLICY
    key_epoch: int = 1
    active_from_server_sequence: int | None = None
    superseded_at: str | None = None
    wrapped_for: SyncKeyWrappedFor = "recovery"
    rewrap_status: SyncKeyRewrapStatus = "not_required"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "rotation_source_key_record_ids",
            _normalize_key_record_source_ids(self.rotation_source_key_record_ids),
        )
        _validate_key_record_rotation_metadata(
            encryption_policy=self.encryption_policy,
            key_epoch=self.key_epoch,
            active_from_server_sequence=self.active_from_server_sequence,
            wrapped_for=self.wrapped_for,
            rewrap_status=self.rewrap_status,
        )


def _normalize_key_record_source_ids(source_ids: Sequence[str] | None) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                str(source_id).strip()
                for source_id in source_ids or ()
                if str(source_id).strip()
            }
        )
    )


def _validate_key_record_rotation_metadata(
    *,
    encryption_policy: EncryptionPolicy,
    key_epoch: int,
    active_from_server_sequence: int | None,
    wrapped_for: SyncKeyWrappedFor,
    rewrap_status: SyncKeyRewrapStatus,
) -> None:
    if encryption_policy not in SYNC_V2_ENCRYPTION_POLICIES:
        raise ValueError(f"unsupported Sync v2 encryption policy: {encryption_policy}")
    if isinstance(key_epoch, bool) or key_epoch < 1:
        raise ValueError("Sync key record key_epoch must be greater than or equal to 1")
    if (
        active_from_server_sequence is not None
        and (
            isinstance(active_from_server_sequence, bool)
            or active_from_server_sequence < 0
        )
    ):
        raise ValueError("Sync key record active_from_server_sequence must be non-negative")
    if wrapped_for not in SYNC_KEY_WRAPPED_FOR_VALUES:
        raise ValueError(f"unsupported Sync key wrapped_for value: {wrapped_for}")
    if rewrap_status not in SYNC_KEY_REWRAP_STATUSES:
        raise ValueError(f"unsupported Sync key rewrap_status value: {rewrap_status}")


@dataclass(frozen=True, slots=True)
class SyncKeyRotationKeyRecord:
    """Redacted key-record metadata returned by key rotation flows."""

    key_record_id: str
    key_epoch: int
    encryption_policy: EncryptionPolicy
    wrapped_for: SyncKeyWrappedFor
    rewrap_status: SyncKeyRewrapStatus
    device_id: str | None = None
    key_purpose: str = "dataset_recovery"
    active_from_server_sequence: int | None = None
    superseded_at: str | None = None
    revoked_at: str | None = None
    rotation_of_key_record_id: str | None = None


@dataclass(frozen=True, slots=True)
class SyncKeyRotationEnvelopeRange:
    """Accepted envelope range retained under old key material."""

    from_server_sequence: int | None = None
    through_server_sequence: int | None = None
    envelope_count: int = 0


@dataclass(frozen=True, slots=True)
class SyncKeyRotationResult:
    """Redacted key rotation preview or commit result."""

    dataset_id: str
    target_encryption_policy: EncryptionPolicy
    next_key_epoch: int
    active_from_server_sequence: int
    can_commit: bool
    committed: bool
    retained_envelope_range: SyncKeyRotationEnvelopeRange
    affected_key_records: list[SyncKeyRotationKeyRecord] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    device_ids: list[str] = field(default_factory=list)
    recovery_target_count: int = 0
    rotation_id: str | None = None
    new_key_record: SyncKeyRotationKeyRecord | None = None


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
    device_id: str | None = None
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
class SyncBlobDownloadManifest:
    """Core manifest describing resumable Sync v2 M2 blob download availability."""

    dataset_id: str
    attachment_id: str
    availability: SyncBlobAvailabilityStatus
    content_type: str
    size_bytes: int
    payload_hash: str
    chunks: list[SyncBlobDownloadChunk] = field(default_factory=list)
    blob_id: str | None = None
    expires_at: str | None = None


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
    "MEDIA_SYNC_DOMAINS",
    "MEDIA_SYNC_OPERATIONS",
    "STRICT_ENCRYPTION_POLICIES",
    "SOURCE_CACHE_SYNC_DOMAINS",
    "SOURCE_CACHE_SYNC_OPERATIONS",
    "SYNC_KEY_REWRAP_STATUSES",
    "SYNC_KEY_WRAPPED_FOR_VALUES",
    "SYNC_V2_ENCRYPTION_POLICIES",
    "SYNC_V2_SUPPORTED_DOMAINS",
    "SYNC_V2_SUPPORTED_OPERATIONS",
    "SyncApplyStatus",
    "SyncAttachment",
    "SyncAttachmentCreate",
    "SyncBackgroundDomainStatus",
    "SyncBackgroundLease",
    "SyncBackgroundLeaseCreate",
    "SyncBackgroundLeaseStatus",
    "SyncBackgroundPolicy",
    "SyncBackgroundPolicyUpsert",
    "SyncBlobAvailabilityStatus",
    "SyncBlobChunk",
    "SyncBlobChunkCreate",
    "SyncBlobDownloadChunk",
    "SyncBlobDownloadManifest",
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
    "SyncEncryptionPolicyMetadata",
    "SyncKeyRewrapStatus",
    "SyncKeyRecord",
    "SyncKeyRecordCreate",
    "SyncKeyRotationEnvelopeRange",
    "SyncKeyRotationKeyRecord",
    "SyncKeyRotationResult",
    "SyncKeyWrappedFor",
    "SyncObjectState",
    "SyncOperation",
    "SyncRestoreBlobCompleteness",
    "SyncRestoreCompletenessStatus",
    "SyncRestoreDomainCompleteness",
    "SyncRestoreManifestStats",
    "WORKSPACE_SYNC_DOMAINS",
    "WORKSPACE_SYNC_OPERATIONS",
]
