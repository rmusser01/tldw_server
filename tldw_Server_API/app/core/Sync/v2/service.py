from __future__ import annotations

"""Business service for Sync v2 protocol operations."""

import hashlib
import inspect
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from uuid import uuid4

from .adapters import (
    ATTACHMENT_REF_SERVER_AVAILABILITY,
    AdapterAccepted,
    AdapterConflict,
    AdapterDeferred,
    AdapterRejected,
    AttachmentRefValidationError,
    SyncAdapterContext,
    SyncAdapterRegistry,
    SyncDomainAdapter,
    extract_attachment_ref_metadata,
)
from .blob_store import LocalSyncBlobStore, SyncBlobStoreError
from .errors import (
    SyncIdempotencyConflictError,
    SyncInvalidDomainError,
    SyncStoreError,
)
from .materializers import MaterializationResult, SyncMaterializer
from .models import (
    DEFAULT_M1_ENCRYPTION_POLICY,
    M1_SYNC_DOMAINS,
    SYNC_V2_ENCRYPTION_POLICIES,
    SYNC_V2_SUPPORTED_DOMAINS,
    SYNC_V2_SUPPORTED_OPERATIONS,
    WORKSPACE_SYNC_DOMAINS,
    ConflictStatus,
    EncryptionPolicy,
    SyncAttachment,
    SyncAttachmentCreate,
    SyncBackgroundDomainStatus,
    SyncBackgroundLease,
    SyncBackgroundLeaseCreate,
    SyncBackgroundPolicy,
    SyncBackgroundPolicyUpsert,
    SyncBlobChunk,
    SyncBlobChunkCreate,
    SyncBlobDownloadChunk,
    SyncBlobDownloadManifest,
    SyncBlobObject,
    SyncBlobObjectCreate,
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
    SyncDeviceBlobAckCreate,
    SyncDeviceCursor,
    SyncDeviceDomainAckCreate,
    SyncDeviceUpsert,
    SyncDomain,
    SyncEnvelope,
    SyncEnvelopeCreate,
    SyncKeyRecord,
    SyncKeyRecordCreate,
    SyncKeyRewrapStatus,
    SyncKeyRotationKeyRecord,
    SyncKeyRotationResult,
    SyncKeyWrappedFor,
    SyncOperation,
    SyncRestoreBlobCompleteness,
    SyncRestoreCompletenessStatus,
    SyncRestoreDomainCompleteness,
)
from .profile import SyncProfileStatus, SyncV2ProfileManager
from .replay import SyncReplayRepairer, SyncReplayRepairResult
from .restore import (
    OBJECT_RESTORE_DOMAINS,
    WHOLE_OBJECT_RESTORE_DOMAINS,
    attachment_available_locally,
    attachment_restore_status,
    attachment_verified_locally,
    build_local_inventory_index,
    find_local_inventory_item,
    local_inventory_matches,
    restore_action_for_domain,
)
from .security import (
    PrivatePayloadValidationError,
    SyncV2ServerTrustedEncryptionStatus,
    server_trusted_encryption_status_from_env,
    validate_private_payload,
)
from .store import SyncV2Store

SYNC_DATASET_RECOVERY_KEY_PURPOSE = "dataset_recovery"
SYNC_KEY_RECOVERY_MAX_WRAPPED_KEY_BYTES = 64 * 1024


def _safe_projection_error_message(exc: Exception) -> str:
    return f"Projection failed: {type(exc).__name__}"


def _key_recovery_metadata_string(
    metadata: Mapping[str, object],
    *keys: str,
    nested_parent: str | None = None,
) -> str | None:
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    nested = metadata.get(nested_parent) if nested_parent else None
    if isinstance(nested, Mapping):
        for key in keys:
            value = nested.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class SyncV2Settings:
    """Server settings surfaced through Sync v2 capabilities."""

    protocol_version: str = "sync-v2-m1"
    min_supported_protocol_version: str = "sync-v2-m1"
    max_batch_size: int = 100
    max_pull_page_size: int = 100
    max_envelope_payload_bytes: int = 262_144
    max_attachment_bytes: int = 1_048_576
    supports_attachments: bool = False
    max_blob_bytes: int | None = None
    max_chunk_bytes: int = 4_194_304
    max_active_blob_uploads: int = 8
    user_blob_quota_bytes: int | None = None
    reserved_blob_bytes: int = 0
    used_blob_bytes: int = 0
    blob_storage_backend: str = "local_fs"
    blob_checksum_algorithm: str = "sha256"
    supports_resumable_upload: bool = True
    supports_resumable_download: bool = True
    supports_chunk_checksums: bool = True
    supported_domains: list[SyncDomain] = field(default_factory=lambda: list(SYNC_V2_SUPPORTED_DOMAINS))
    operations: dict[SyncDomain, list[SyncOperation]] = field(
        default_factory=lambda: {domain: list(operations) for domain, operations in SYNC_V2_SUPPORTED_OPERATIONS.items()}
    )
    encryption_policies: list[EncryptionPolicy] = field(default_factory=lambda: [DEFAULT_M1_ENCRYPTION_POLICY])
    server_trusted_encryption: SyncV2ServerTrustedEncryptionStatus = field(
        default_factory=server_trusted_encryption_status_from_env
    )
    restore_manifest_scan_limit: int = 10_000


@dataclass(frozen=True, slots=True)
class SyncV2Capabilities:
    protocol_version: str
    min_supported_protocol_version: str
    supported_domains: list[SyncDomain]
    operations: dict[SyncDomain, list[SyncOperation]]
    encryption: dict[str, object]
    blob_transfer: dict[str, object]
    encryption_policies: list[EncryptionPolicy]
    max_batch_size: int
    max_envelope_payload_bytes: int
    max_attachment_bytes: int
    quota: dict[str, object] = field(default_factory=dict)
    supports_restore_manifest: bool = True
    supports_conflicts: bool = True
    supports_attachments: bool = True
    server_time: str | None = None
    warnings: list[dict[str, str]] = field(default_factory=list)


WorkspaceAccessChecker = Callable[[str, str, str], bool]


@dataclass(frozen=True, slots=True)
class SyncDeviceRegistration:
    device: SyncDevice
    server_capabilities: SyncV2Capabilities
    required_actions: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class SyncDatasetEnrollment:
    dataset: SyncDataset
    cursors: dict[str, str] = field(default_factory=dict)
    key_setup_required: bool = False


@dataclass(frozen=True, slots=True)
class SyncBackgroundStatus:
    dataset_id: str
    device_id: str
    policy: SyncBackgroundPolicy
    lease: SyncBackgroundLease | None = None
    domains: list[SyncBackgroundDomainStatus] = field(default_factory=list)
    conflict_count: int = 0
    replayable_failure_count: int = 0
    quota_pressure: dict[str, object] = field(default_factory=dict)
    restore_completeness: SyncRestoreCompletenessStatus = "metadata_ready"
    server_time: str | None = None


@dataclass(frozen=True, slots=True)
class SyncPushAccepted:
    client_envelope_id: str
    server_sequence: int
    domain: SyncDomain
    entity_id: str
    object_revision: int | None = None
    apply_status: str | None = None
    apply_error_code: str | None = None
    apply_error_message: str | None = None


@dataclass(frozen=True, slots=True)
class SyncPushRejected:
    client_envelope_id: str
    error_code: str
    message: str
    retryable: bool = False


@dataclass(frozen=True, slots=True)
class SyncPushConflict:
    conflict_id: str
    client_envelope_id: str
    domain: SyncDomain
    entity_id: str
    server_sequence: int | None = None
    message: str | None = None


@dataclass(frozen=True, slots=True)
class SyncPushResult:
    dataset_id: str
    accepted: list[SyncPushAccepted] = field(default_factory=list)
    rejected: list[SyncPushRejected] = field(default_factory=list)
    conflicts: list[SyncPushConflict] = field(default_factory=list)
    next_cursor: str | None = None


@dataclass(frozen=True, slots=True)
class SyncPullResult:
    dataset_id: str
    encryption_policy: EncryptionPolicy = DEFAULT_M1_ENCRYPTION_POLICY
    envelopes: list[SyncEnvelope] = field(default_factory=list)
    next_cursor: str | None = None
    has_more: bool = False


@dataclass(frozen=True, slots=True)
class SyncRestoreManifestDevice:
    device_id: str
    display_name: str | None
    client_type: str | None
    client_version: str | None
    last_seen_at: str | None
    revoked_at: str | None


@dataclass(frozen=True, slots=True)
class SyncRestoreManifestDataset:
    dataset_id: str
    scope_type: str
    encryption_policy: EncryptionPolicy
    domains: list[SyncDomain]
    workspace_id: str | None
    approximate_counts: dict[str, int] = field(default_factory=dict)
    byte_estimates: dict[str, int] = field(default_factory=dict)
    last_updated_at: str | None = None
    unresolved_conflicts: int = 0
    attachment_availability: dict[str, int] = field(default_factory=dict)
    attachment_size_classes: dict[str, int] = field(default_factory=dict)
    key_recovery_available: bool = False
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SyncRestoreManifest:
    datasets: list[SyncRestoreManifestDataset] = field(default_factory=list)
    devices: list[SyncRestoreManifestDevice] = field(default_factory=list)
    generated_at: str | None = None
    filters_applied: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SyncRestorePreviewDataset:
    dataset_id: str
    domains: list[SyncDomain]
    approximate_counts: dict[str, int] = field(default_factory=dict)
    byte_estimates: dict[str, int] = field(default_factory=dict)
    latest_cursor: int | None = None
    latest_cursors: dict[str, int] = field(default_factory=dict)
    envelope_ranges: list[SyncRestorePreviewEnvelopeRange] = field(default_factory=list)
    total_count: int = 0
    encryption_policy: EncryptionPolicy = DEFAULT_M1_ENCRYPTION_POLICY
    key_recovery_available: bool = False


@dataclass(frozen=True, slots=True)
class SyncRestorePreviewEnvelopeRange:
    dataset_id: str
    domain: SyncDomain
    from_cursor: int
    to_cursor: int
    envelope_count: int


@dataclass(frozen=True, slots=True)
class SyncRestorePreviewObject:
    dataset_id: str
    domain: SyncDomain
    object_id: str
    action: str
    server_revision: int | None = None
    server_hash: str | None = None
    server_cursor: int | None = None
    local_revision: int | None = None
    local_hash: str | None = None
    local_deleted: bool | None = None
    deleted: bool = False
    parent_id: str | None = None


@dataclass(frozen=True, slots=True)
class SyncRestorePreviewObjectConflict:
    dataset_id: str
    domain: SyncDomain
    object_id: str
    conflict_type: str
    server_revision: int | None = None
    server_hash: str | None = None
    server_cursor: int | None = None
    server_deleted: bool = False
    local_revision: int | None = None
    local_hash: str | None = None
    local_deleted: bool = False
    message: str | None = None


@dataclass(frozen=True, slots=True)
class SyncRestorePreviewAttachmentRef:
    dataset_id: str
    attachment_id: str
    object_id: str
    parent_domain: SyncDomain
    parent_object_id: str
    content_type: str
    size_bytes: int
    payload_hash: str
    availability: str
    server_cursor: int


@dataclass(frozen=True, slots=True)
class SyncRestorePreviewWarning:
    code: str
    message: str
    dataset_id: str | None = None
    attachment_id: str | None = None
    object_id: str | None = None
    payload_hash: str | None = None


@dataclass(frozen=True, slots=True)
class SyncRestorePreview:
    datasets: list[SyncRestorePreviewDataset] = field(default_factory=list)
    safe_applies: list[SyncRestorePreviewObject] = field(default_factory=list)
    object_conflicts: list[SyncRestorePreviewObjectConflict] = field(default_factory=list)
    tombstones: list[SyncRestorePreviewObject] = field(default_factory=list)
    attachment_refs: list[SyncRestorePreviewAttachmentRef] = field(default_factory=list)
    missing_blobs: list[SyncRestorePreviewAttachmentRef] = field(default_factory=list)
    envelope_ranges: list[SyncRestorePreviewEnvelopeRange] = field(default_factory=list)
    total_counts: dict[str, int] = field(default_factory=dict)
    encryption: dict[str, object] = field(default_factory=dict)
    key_status: dict[str, dict[str, bool]] = field(default_factory=dict)
    warnings: list[SyncRestorePreviewWarning] = field(default_factory=list)
    generated_at: str | None = None
    filters_applied: dict[str, object] = field(default_factory=dict)
    restore_status: SyncRestoreCompletenessStatus | None = None
    domain_details: list[SyncRestoreDomainCompleteness] = field(default_factory=list)
    blob_details: list[SyncRestoreBlobCompleteness] = field(default_factory=list)
    metadata_only_allowed: bool = True


class SyncV2Service:
    """Core Sync v2 service with injected persistence and adapter dependencies."""

    def __init__(
        self,
        *,
        store: SyncV2Store,
        adapters: SyncAdapterRegistry,
        materializers: Mapping[SyncDomain, SyncMaterializer] | None = None,
        clock: Callable[[], str] | None = None,
        id_factory: Callable[[str], str] | None = None,
        blob_store: LocalSyncBlobStore | None = None,
        settings: SyncV2Settings | None = None,
        workspace_access_checker: WorkspaceAccessChecker | None = None,
    ) -> None:
        self.store = store
        self.adapters = adapters
        self.materializers = dict(materializers or {})
        self.clock = clock or (lambda: datetime.now(timezone.utc).isoformat())
        self.id_factory = id_factory or (lambda prefix: f"{prefix}-{uuid4().hex}")
        self.blob_store = blob_store
        self.settings = settings or SyncV2Settings()
        self.workspace_access_checker = workspace_access_checker

    def capabilities(self) -> SyncV2Capabilities:
        blob_transfer: dict[str, object] = {"supported": False}
        quota: dict[str, object] = {}
        if self.settings.supports_attachments:
            max_blob_bytes = self.settings.max_blob_bytes or self.settings.max_attachment_bytes
            blob_transfer = {
                "supported": True,
                "resumable_upload": self.settings.supports_resumable_upload,
                "resumable_download": self.settings.supports_resumable_download,
                "chunk_checksums": self.settings.supports_chunk_checksums,
                "full_checksum": self.settings.blob_checksum_algorithm,
                "storage_backend": self.settings.blob_storage_backend,
            }
            quota = {
                "max_blob_bytes": max_blob_bytes,
                "max_chunk_bytes": self.settings.max_chunk_bytes,
                "max_active_uploads": self.settings.max_active_blob_uploads,
                "user_blob_quota_bytes": self.settings.user_blob_quota_bytes,
                "reserved_blob_bytes": self.settings.reserved_blob_bytes,
                "used_blob_bytes": self.settings.used_blob_bytes,
            }
        return SyncV2Capabilities(
            protocol_version=self.settings.protocol_version,
            min_supported_protocol_version=self.settings.min_supported_protocol_version,
            supported_domains=list(self.settings.supported_domains),
            operations={domain: list(operations) for domain, operations in self.settings.operations.items()},
            encryption=self.settings.server_trusted_encryption.encryption,
            blob_transfer=blob_transfer,
            encryption_policies=list(self.settings.encryption_policies),
            max_batch_size=self.settings.max_batch_size,
            max_envelope_payload_bytes=self.settings.max_envelope_payload_bytes,
            max_attachment_bytes=self.settings.max_attachment_bytes,
            quota=quota,
            supports_attachments=self.settings.supports_attachments,
            server_time=self.clock() or None,
            warnings=self.settings.server_trusted_encryption.warnings,
        )

    def register_device(
        self,
        *,
        user_id: str,
        display_name: str,
        client_type: str,
        device_id: str | None = None,
        client_version: str | None = None,
        capabilities: dict[str, object] | None = None,
    ) -> SyncDeviceRegistration:
        device = self.store.upsert_device(
            SyncDeviceUpsert(
                device_id=device_id or self.id_factory("device"),
                user_id=user_id,
                display_name=display_name,
                client_type=client_type,
                client_version=client_version,
                capabilities=dict(capabilities or {}),
            )
        )
        return SyncDeviceRegistration(device=device, server_capabilities=self.capabilities())

    def list_devices(
        self,
        *,
        user_id: str,
        include_revoked: bool = False,
    ) -> list[SyncDevice]:
        """List registered devices for a user."""

        return self.store.list_devices_for_user(
            user_id,
            include_revoked=include_revoked,
        )

    def update_device(
        self,
        *,
        user_id: str,
        device_id: str,
        display_name: str | None = None,
        user_label: str | None = None,
        client_version: str | None = None,
        capabilities: dict[str, object] | None = None,
    ) -> SyncDevice:
        """Update mutable device metadata without changing lifecycle state."""

        existing = self.store.get_device(user_id, device_id)
        if existing is None:
            raise SyncStoreError("Sync device was not found or is not accessible")
        return self.store.upsert_device(
            SyncDeviceUpsert(
                device_id=existing.device_id,
                user_id=existing.user_id,
                display_name=display_name or existing.display_name,
                client_type=existing.client_type,
                client_version=(
                    client_version
                    if client_version is not None
                    else existing.client_version
                ),
                capabilities=(
                    dict(capabilities)
                    if capabilities is not None
                    else dict(existing.capabilities)
                ),
                status=existing.status,
                user_label=user_label if user_label is not None else existing.user_label,
                authorized_at=existing.authorized_at,
                revoked_at=existing.revoked_at,
                revoked_reason=existing.revoked_reason,
            )
        )

    def create_device_authorization(
        self,
        *,
        user_id: str,
        dataset_id: str,
        device_id: str,
        authorization_method: str,
        idempotency_key: str | None = None,
    ) -> SyncDeviceAuthorization:
        """Create a pending device authorization request."""

        self._require_dataset_access(user_id=user_id, dataset_id=dataset_id)
        return self.store.create_device_authorization(
            SyncDeviceAuthorizationCreate(
                authorization_id=self.id_factory("device-authorization"),
                dataset_id=dataset_id,
                user_id=user_id,
                device_id=device_id,
                authorization_method=authorization_method,
                idempotency_key=idempotency_key,
            )
        )

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

        self._require_dataset_access(user_id=user_id, dataset_id=dataset_id)
        if approving_device_id is not None:
            self._require_registered_device(user_id, approving_device_id)
        return self.store.approve_device_authorization(
            authorization_id,
            user_id=user_id,
            dataset_id=dataset_id,
            approving_device_id=approving_device_id,
            idempotency_key=idempotency_key,
        )

    def revoke_device(
        self,
        *,
        user_id: str,
        device_id: str,
        reason: str | None = None,
        revoke_key_records: bool = False,
    ) -> SyncDevice:
        """Revoke a device from future sync operations."""

        return self.store.revoke_device(
            user_id=user_id,
            device_id=device_id,
            reason=reason,
            revoke_key_records=revoke_key_records,
        )

    def pause_device(self, *, user_id: str, device_id: str) -> SyncDevice:
        """Pause a device so it cannot perform device-scoped sync calls."""

        existing = self.store.get_device(user_id, device_id)
        if existing is None or existing.status == "revoked":
            raise SyncStoreError("Sync device was not found or is not accessible")
        return self.store.upsert_device(
            SyncDeviceUpsert(
                device_id=existing.device_id,
                user_id=existing.user_id,
                display_name=existing.display_name,
                client_type=existing.client_type,
                client_version=existing.client_version,
                capabilities=dict(existing.capabilities),
                status="paused",
                user_label=existing.user_label,
                authorized_at=existing.authorized_at,
            )
        )

    def resume_device(self, *, user_id: str, device_id: str) -> SyncDevice:
        """Resume a paused device after user approval."""

        existing = self.store.get_device(user_id, device_id)
        if existing is None or existing.status in {"revoked", "pending_authorization"}:
            raise SyncStoreError("Sync device was not found or is not accessible")
        return self.store.upsert_device(
            SyncDeviceUpsert(
                device_id=existing.device_id,
                user_id=existing.user_id,
                display_name=existing.display_name,
                client_type=existing.client_type,
                client_version=existing.client_version,
                capabilities=dict(existing.capabilities),
                status="active",
                user_label=existing.user_label,
                authorized_at=existing.authorized_at or self.clock(),
            )
        )

    def acknowledge_device_state(
        self,
        *,
        user_id: str,
        dataset_id: str,
        device_id: str,
        domain_acks: Sequence[SyncDeviceDomainAckCreate] = (),
        blob_acks: Sequence[SyncDeviceBlobAckCreate] = (),
    ) -> SyncDeviceAcknowledgmentSummary:
        """Record a device's durable application/verification acknowledgments."""

        self._require_registered_device(user_id, device_id)
        self._require_dataset_access(user_id=user_id, dataset_id=dataset_id)
        for acknowledgment in domain_acks:
            if acknowledgment.dataset_id != dataset_id or acknowledgment.device_id != device_id:
                raise SyncStoreError("Sync acknowledgment device or dataset does not match request")
            self.store.upsert_device_domain_ack(acknowledgment)
        for acknowledgment in blob_acks:
            if acknowledgment.dataset_id != dataset_id or acknowledgment.device_id != device_id:
                raise SyncStoreError("Sync acknowledgment device or dataset does not match request")
            self.store.upsert_device_blob_ack(acknowledgment)
        return self.store.list_device_acknowledgments(dataset_id, device_id)

    def get_background_policy(
        self,
        *,
        user_id: str,
        dataset_id: str,
        device_id: str,
    ) -> SyncBackgroundPolicy:
        """Return stored or default background sync policy for a dataset/device."""

        self._require_registered_device(user_id, device_id)
        self._require_dataset_access(user_id=user_id, dataset_id=dataset_id)
        stored = self.store.get_background_policy(dataset_id, device_id)
        if stored is not None:
            return stored
        return self._default_background_policy(dataset_id, device_id)

    def update_background_policy(
        self,
        *,
        user_id: str,
        dataset_id: str,
        device_id: str,
        enabled: bool | None = None,
        minimum_interval_seconds: int | None = None,
        backoff_floor_seconds: int | None = None,
        max_batch_size: int | None = None,
        max_blob_bytes_per_run: int | None = None,
        respect_metered_networks: bool | None = None,
        maintenance_window: dict[str, object] | None = None,
        paused_reason: str | None = None,
        pending_local_changes: bool | None = None,
    ) -> SyncBackgroundPolicy:
        """Persist background sync policy hints and user pause/resume intent."""

        current = self.get_background_policy(
            user_id=user_id,
            dataset_id=dataset_id,
            device_id=device_id,
        )
        resolved_enabled = current.enabled if enabled is None else enabled
        return self.store.upsert_background_policy(
            SyncBackgroundPolicyUpsert(
                dataset_id=dataset_id,
                device_id=device_id,
                enabled=resolved_enabled,
                minimum_interval_seconds=(
                    current.minimum_interval_seconds
                    if minimum_interval_seconds is None
                    else minimum_interval_seconds
                ),
                backoff_floor_seconds=(
                    current.backoff_floor_seconds
                    if backoff_floor_seconds is None
                    else backoff_floor_seconds
                ),
                max_batch_size=current.max_batch_size if max_batch_size is None else max_batch_size,
                max_blob_bytes_per_run=(
                    current.max_blob_bytes_per_run
                    if max_blob_bytes_per_run is None
                    else max_blob_bytes_per_run
                ),
                respect_metered_networks=(
                    current.respect_metered_networks
                    if respect_metered_networks is None
                    else respect_metered_networks
                ),
                maintenance_window=(
                    current.maintenance_window
                    if maintenance_window is None
                    else dict(maintenance_window)
                ),
                paused_reason=(
                    paused_reason
                    if paused_reason is not None
                    else (None if resolved_enabled else current.paused_reason)
                ),
                pending_local_changes=(
                    current.pending_local_changes
                    if pending_local_changes is None
                    else pending_local_changes
                ),
            )
        )

    def acquire_background_lease(
        self,
        *,
        user_id: str,
        dataset_id: str,
        device_id: str,
        lease_id: str | None = None,
        ttl_seconds: int = 120,
    ) -> SyncBackgroundLease:
        """Acquire or refresh a short-lived advisory background sync lease."""

        self._require_registered_device(user_id, device_id)
        self._require_dataset_access(user_id=user_id, dataset_id=dataset_id)
        return self.store.acquire_background_lease(
            SyncBackgroundLeaseCreate(
                dataset_id=dataset_id,
                device_id=device_id,
                lease_id=lease_id or self.id_factory("background-lease"),
                ttl_seconds=ttl_seconds,
                requested_at=self.clock(),
            )
        )

    def background_status(
        self,
        *,
        user_id: str,
        dataset_id: str,
        device_id: str,
    ) -> SyncBackgroundStatus:
        """Return profile-level and per-domain background sync status."""

        self._require_registered_device(user_id, device_id)
        dataset = self._require_dataset_access(user_id=user_id, dataset_id=dataset_id)
        policy = self.get_background_policy(
            user_id=user_id,
            dataset_id=dataset_id,
            device_id=device_id,
        )
        domains = self.store.summarize_background_domains(
            dataset_id,
            device_id,
            domains=dataset.domains,
        )
        lease = self.store.get_background_lease(dataset_id, device_id)
        quota = self.store.summarize_blob_quota(user_id, dataset_id=dataset_id)
        quota_limit = self.settings.user_blob_quota_bytes
        quota_used = quota.used_blob_bytes + quota.reserved_blob_bytes
        quota_pressure = {
            "reserved_blob_bytes": quota.reserved_blob_bytes,
            "used_blob_bytes": quota.used_blob_bytes,
            "active_upload_count": quota.active_upload_count,
            "limit_bytes": quota_limit,
            "pressure_ratio": (
                round(quota_used / quota_limit, 4)
                if quota_limit and quota_limit > 0
                else 0.0
            ),
        }
        conflict_count = sum(domain.unresolved_conflicts for domain in domains)
        replayable_failure_count = sum(domain.replayable_failures for domain in domains)
        missing_blob_count = sum(
            domain.blob_completeness.get("missing_blob_count", 0)
            for domain in domains
        )
        if conflict_count:
            restore_completeness: SyncRestoreCompletenessStatus = "blocked_by_conflicts"
        elif missing_blob_count:
            restore_completeness = "blob_incomplete"
        elif any(domain.last_server_sequence for domain in domains):
            restore_completeness = "content_complete"
        else:
            restore_completeness = "metadata_ready"
        return SyncBackgroundStatus(
            dataset_id=dataset_id,
            device_id=device_id,
            policy=policy,
            lease=lease,
            domains=domains,
            conflict_count=conflict_count,
            replayable_failure_count=replayable_failure_count,
            quota_pressure=quota_pressure,
            restore_completeness=restore_completeness,
            server_time=self.clock(),
        )

    def enroll_dataset(
        self,
        *,
        user_id: str,
        dataset_id: str | None = None,
        scope_type: str = "personal",
        domains: Sequence[SyncDomain] | None = None,
        encryption_policy: EncryptionPolicy = DEFAULT_M1_ENCRYPTION_POLICY,
        workspace_id: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> SyncDatasetEnrollment:
        self._require_server_trusted_encryption_ready()
        if scope_type == "workspace":
            self._require_workspace_sync_access(user_id=user_id, workspace_id=workspace_id)
            enrolled_domains = list(domains or WORKSPACE_SYNC_DOMAINS)
        else:
            enrolled_domains = list(domains or M1_SYNC_DOMAINS)
        dataset = self.store.enroll_dataset(
            SyncDatasetCreate(
                dataset_id=dataset_id or self.id_factory("dataset"),
                owner_user_id=user_id,
                scope_type=scope_type,
                encryption_policy=encryption_policy,
                domains=enrolled_domains,
                workspace_id=workspace_id,
                metadata=dict(metadata or {}),
            )
        )
        return SyncDatasetEnrollment(
            dataset=dataset,
            cursors=dict.fromkeys(dataset.domains, "0"),
            key_setup_required=False,
        )

    def profile(
        self,
        *,
        user_id: str,
        device_id: str | None = None,
    ) -> SyncProfileStatus:
        """Return current profile state without creating sync records."""

        return self._profile_manager().profile(user_id=user_id, device_id=device_id)

    def bootstrap_profile(
        self,
        *,
        user_id: str,
        mode: str,
        device_id: str | None = None,
        device_name: str | None = None,
        client_profile_id: str | None = None,
        client_family: str = "chatbook",
        client_version: str | None = None,
        client_instance: dict[str, object] | None = None,
        requested_domains: Sequence[SyncDomain] | None = None,
    ) -> SyncProfileStatus:
        """Idempotently bootstrap the user's default Sync v2 M1 profile."""

        return self._profile_manager().bootstrap_profile(
            user_id=user_id,
            mode=mode,
            device_id=device_id,
            device_name=device_name,
            client_profile_id=client_profile_id,
            client_family=client_family,
            client_version=client_version,
            client_instance=dict(client_instance or {}),
            requested_domains=requested_domains,
        )

    def profile_status(
        self,
        *,
        user_id: str,
        dataset_id: str,
        device_id: str | None = None,
    ) -> SyncProfileStatus:
        """Return profile-level and per-domain status for an existing dataset."""

        return self._profile_manager().profile_status(
            user_id=user_id,
            dataset_id=dataset_id,
            device_id=device_id,
        )

    def push(
        self,
        *,
        user_id: str,
        dataset_id: str,
        device_id: str,
        envelopes: Sequence[SyncEnvelopeCreate],
    ) -> SyncPushResult:
        self._require_registered_device(user_id, device_id)
        try:
            dataset = self._require_dataset_access(user_id=user_id, dataset_id=dataset_id)
        except SyncStoreError:
            return SyncPushResult(
                dataset_id=dataset_id,
                rejected=[
                    SyncPushRejected(
                        client_envelope_id=envelope.client_envelope_id,
                        error_code="dataset_not_found_or_forbidden",
                        message="Sync dataset was not found or is not accessible",
                    )
                    for envelope in envelopes
                ],
            )

        accepted: list[SyncPushAccepted] = []
        rejected: list[SyncPushRejected] = []
        conflicts: list[SyncPushConflict] = []

        for index, envelope in enumerate(envelopes):
            if index >= self.settings.max_batch_size:
                rejected.append(
                    SyncPushRejected(
                        client_envelope_id=envelope.client_envelope_id,
                        error_code="batch_limit_exceeded",
                        message="Sync push batch exceeded the server envelope limit",
                    )
                )
                continue
            if envelope.dataset_id != dataset_id:
                rejected.append(
                    SyncPushRejected(
                        client_envelope_id=envelope.client_envelope_id,
                        error_code="dataset_mismatch",
                        message="Envelope dataset_id must match the push dataset_id",
                    )
                )
                continue
            if envelope.device_id is not None and envelope.device_id != device_id:
                rejected.append(
                    SyncPushRejected(
                        client_envelope_id=envelope.client_envelope_id,
                        error_code="device_mismatch",
                        message="Envelope device_id must match the authenticated push device_id",
                    )
                )
                continue
            if envelope.domain not in dataset.domains:
                rejected.append(
                    SyncPushRejected(
                        client_envelope_id=envelope.client_envelope_id,
                        error_code="domain_not_enrolled",
                        message=f"Sync domain is not enrolled for this dataset: {envelope.domain}",
                    )
                )
                continue
            if self._payload_exceeds_size_limit(envelope):
                rejected.append(
                    SyncPushRejected(
                        client_envelope_id=envelope.client_envelope_id,
                        error_code="payload_too_large",
                        message="Sync envelope payload exceeds the server size limit",
                    )
                )
                continue
            envelope = replace(envelope, device_id=envelope.device_id or device_id)
            try:
                outcome = self._evaluate_envelope(dataset, envelope)
            except KeyError:
                rejected.append(
                    SyncPushRejected(
                        client_envelope_id=envelope.client_envelope_id,
                        error_code="unknown_domain",
                        message=f"Sync domain is not registered: {envelope.domain}",
                    )
                )
                continue
            except PrivatePayloadValidationError as exc:
                rejected.append(
                    SyncPushRejected(
                        client_envelope_id=envelope.client_envelope_id,
                        error_code="private_payload_validation_failed",
                        message=str(exc),
                    )
                )
                continue

            if isinstance(outcome, AdapterRejected):
                rejected.append(
                    SyncPushRejected(
                        client_envelope_id=outcome.client_envelope_id,
                        error_code=outcome.error_code,
                        message=outcome.message,
                        retryable=outcome.retryable,
                    )
                )
                continue
            if isinstance(outcome, AdapterDeferred):
                rejected.append(
                    SyncPushRejected(
                        client_envelope_id=outcome.client_envelope_id,
                        error_code="adapter_deferred",
                        message=outcome.message,
                        retryable=True,
                    )
                )
                continue
            if isinstance(outcome, AdapterConflict):
                try:
                    conflicts.append(self._store_conflict(dataset, envelope, outcome))
                except SyncIdempotencyConflictError:
                    rejected.append(
                        SyncPushRejected(
                            client_envelope_id=envelope.client_envelope_id,
                            error_code="idempotency_conflict",
                            message="Sync envelope ID was reused with different content",
                        )
                    )
                continue

            try:
                inserted = self.store.insert_envelope(replace(envelope, status="accepted"))
            except SyncInvalidDomainError:
                rejected.append(
                    SyncPushRejected(
                        client_envelope_id=envelope.client_envelope_id,
                        error_code="domain_not_enrolled",
                        message=f"Sync domain is not enrolled for this dataset: {envelope.domain}",
                    )
                )
                continue
            except SyncIdempotencyConflictError:
                rejected.append(
                    SyncPushRejected(
                        client_envelope_id=envelope.client_envelope_id,
                        error_code="idempotency_conflict",
                        message="Sync envelope ID was reused with different content",
                    )
                )
                continue
            if inserted.apply_status != "applied":
                materialization = self._materialize_envelope(inserted)
                inserted = self._envelope_snapshot(inserted)
                if materialization.status == "conflict":
                    conflicts.append(
                        self._store_materialization_conflict(
                            dataset,
                            inserted,
                            materialization,
                        )
                    )
                    continue
            accepted.append(self._push_accepted_from_envelope(inserted))

        sequences = [item.server_sequence for item in accepted]
        sequences.extend(item.server_sequence for item in conflicts if item.server_sequence is not None)
        next_sequence = max(sequences, default=None)
        return SyncPushResult(
            dataset_id=dataset_id,
            accepted=accepted,
            rejected=rejected,
            conflicts=conflicts,
            next_cursor=str(next_sequence) if next_sequence is not None else None,
        )

    def pull(
        self,
        *,
        user_id: str,
        dataset_id: str,
        device_id: str,
        cursor: str | int | None = None,
        domains: Sequence[SyncDomain] | None = None,
        page_size: int | None = None,
        include_own_changes: bool = False,
    ) -> SyncPullResult:
        self._require_registered_device(user_id, device_id)
        if page_size is not None and page_size < 1:
            raise SyncStoreError("Sync pull page_size must be greater than zero")
        dataset = self._require_dataset_access(user_id=user_id, dataset_id=dataset_id)

        selected_domains = self._selected_domains(dataset, domains)
        since_sequence = self._resolve_cursor(dataset_id, device_id, cursor, selected_domains)
        page_limit = min(page_size or self.settings.max_pull_page_size, self.settings.max_pull_page_size)
        raw_envelopes, visible = self._scan_pull_page(
            dataset_id=dataset_id,
            device_id=device_id,
            since_sequence=since_sequence,
            domains=selected_domains,
            page_limit=page_limit,
            include_own_changes=include_own_changes,
        )

        page = visible[:page_limit]
        has_visible_lookahead = len(visible) > page_limit
        has_more = has_visible_lookahead or len(raw_envelopes) > page_limit
        if has_visible_lookahead and page:
            next_sequence = page[-1].server_sequence
        else:
            next_sequence = max(
                (envelope.server_sequence for envelope in raw_envelopes),
                default=since_sequence,
            )
        if cursor is None and raw_envelopes:
            self._update_cursors(dataset_id, device_id, selected_domains, next_sequence)
        return SyncPullResult(
            dataset_id=dataset_id,
            encryption_policy=dataset.encryption_policy,
            envelopes=page,
            next_cursor=str(next_sequence),
            has_more=has_more,
        )

    def restore_manifest(
        self,
        *,
        user_id: str,
        device_id: str | None = None,
        dataset_ids: Sequence[str] | None = None,
        domains: Sequence[SyncDomain] | None = None,
    ) -> SyncRestoreManifest:
        if device_id is not None:
            self._require_registered_device(user_id, device_id)
        selected_domains = set(domains or [])
        datasets = self._accessible_datasets(user_id=user_id, dataset_ids=dataset_ids)
        devices = [
            SyncRestoreManifestDevice(
                device_id=device.device_id,
                display_name=device.display_name,
                client_type=device.client_type,
                client_version=device.client_version,
                last_seen_at=device.last_seen_at,
                revoked_at=device.revoked_at,
            )
            for device in self.store.list_devices_for_user(user_id)
        ]

        manifest_datasets = [
            self._manifest_dataset(dataset, user_id=user_id, domains=selected_domains) for dataset in datasets
        ]
        return SyncRestoreManifest(
            datasets=manifest_datasets,
            devices=devices,
            generated_at=self.clock() or None,
            filters_applied={
                "dataset_ids": list(dataset_ids or []),
                "domains": list(domains or []),
            },
        )

    def restore_preview(
        self,
        *,
        user_id: str,
        device_id: str | None = None,
        dataset_ids: Sequence[str] | None = None,
        domains: Sequence[SyncDomain] | None = None,
        selected_object_ids: Sequence[str] | None = None,
        selected_attachment_ids: Sequence[str] | None = None,
        metadata_only: bool = False,
        local_inventory: Sequence[Mapping[str, object]] | None = None,
        attachment_availability: Mapping[str, str] | None = None,
    ) -> SyncRestorePreview:
        """Return metadata needed to preview a restore plan for Sync v2 M1."""

        if device_id is not None:
            self._require_registered_device(user_id, device_id)
        selected_domains = set(domains or [])
        selected_object_id_set = _normalize_selection_set(selected_object_ids)
        selected_attachment_id_set = _normalize_selection_set(selected_attachment_ids)
        local_index = build_local_inventory_index(local_inventory)
        datasets = self._accessible_datasets(user_id=user_id, dataset_ids=dataset_ids)

        preview_datasets: list[SyncRestorePreviewDataset] = []
        safe_applies: list[SyncRestorePreviewObject] = []
        object_conflicts: list[SyncRestorePreviewObjectConflict] = []
        tombstones: list[SyncRestorePreviewObject] = []
        attachment_refs: list[SyncRestorePreviewAttachmentRef] = []
        missing_blobs: list[SyncRestorePreviewAttachmentRef] = []
        blob_details: list[SyncRestoreBlobCompleteness] = []
        envelope_ranges: list[SyncRestorePreviewEnvelopeRange] = []
        total_counts: dict[str, int] = {}
        key_status: dict[str, dict[str, bool]] = {}
        warnings: list[SyncRestorePreviewWarning] = []

        for dataset in datasets:
            dataset_domains = [
                domain for domain in dataset.domains if not selected_domains or domain in selected_domains
            ]
            stats = self.store.summarize_restore_manifest_dataset(
                dataset.dataset_id,
                user_id=user_id,
                domains=dataset_domains,
            )
            key_status[dataset.dataset_id] = {
                "key_recovery_available": stats.key_recovery_available,
            }
            if not stats.key_recovery_available:
                warnings.append(
                    SyncRestorePreviewWarning(
                        code="sync_key_recovery_missing",
                        message=(
                            "No active Sync v2 key recovery bundle is available "
                            "for this dataset."
                        ),
                        dataset_id=dataset.dataset_id,
                    )
                )
            for domain, count in stats.approximate_counts.items():
                total_counts[domain] = total_counts.get(domain, 0) + count
            domain_envelopes = {
                domain: self.store.list_envelopes_after(
                    dataset.dataset_id,
                    0,
                    limit=self.settings.restore_manifest_scan_limit,
                    domains=[domain],
                    status="accepted",
                )
                for domain in dataset_domains
            }
            latest_cursors: dict[str, int] = {}
            dataset_ranges: list[SyncRestorePreviewEnvelopeRange] = []
            for domain, envelopes in domain_envelopes.items():
                cursors = [envelope.server_cursor or 0 for envelope in envelopes if envelope.server_cursor]
                if not cursors:
                    continue
                latest_cursors[domain] = max(cursors)
                dataset_range = SyncRestorePreviewEnvelopeRange(
                    dataset_id=dataset.dataset_id,
                    domain=domain,
                    from_cursor=min(cursors),
                    to_cursor=max(cursors),
                    envelope_count=len(cursors),
                )
                dataset_ranges.append(dataset_range)
                envelope_ranges.append(dataset_range)
            latest_cursor = max(latest_cursors.values(), default=None)
            preview_datasets.append(
                SyncRestorePreviewDataset(
                    dataset_id=dataset.dataset_id,
                    domains=dataset_domains,
                    approximate_counts=stats.approximate_counts,
                    byte_estimates=stats.byte_estimates,
                    latest_cursor=latest_cursor,
                    latest_cursors=latest_cursors,
                    envelope_ranges=dataset_ranges,
                    total_count=sum(stats.approximate_counts.values()),
                    encryption_policy=dataset.encryption_policy,
                    key_recovery_available=stats.key_recovery_available,
                )
            )

            latest_object_envelopes: dict[tuple[SyncDomain, str], SyncEnvelope] = {}
            for domain in dataset_domains:
                if domain not in OBJECT_RESTORE_DOMAINS:
                    continue
                for envelope in domain_envelopes.get(domain, []):
                    if envelope.apply_status == "conflict":
                        continue
                    if selected_object_id_set and envelope.object_id not in selected_object_id_set:
                        continue
                    latest_object_envelopes[(domain, envelope.object_id)] = envelope
            for (domain, object_id), envelope in sorted(
                latest_object_envelopes.items(),
                key=lambda item: item[1].server_cursor or 0,
            ):
                object_state = self.store.get_object_state(dataset.dataset_id, domain, object_id)
                server_revision = (
                    object_state.object_revision
                    if object_state is not None and object_state.latest_server_cursor == envelope.server_cursor
                    else envelope.object_revision
                )
                if server_revision is None and domain in {"notes.note", "chat.conversation"}:
                    server_revision = 1
                server_hash = (
                    object_state.object_hash
                    if object_state is not None and object_state.latest_server_cursor == envelope.server_cursor
                    else envelope.payload_hash
                )
                deleted = (
                    object_state.deleted
                    if object_state is not None and object_state.latest_server_cursor == envelope.server_cursor
                    else envelope.operation == "tombstone" or envelope.deleted
                )
                local_item = find_local_inventory_item(
                    local_index,
                    dataset_id=dataset.dataset_id,
                    domain=domain,
                    object_id=object_id,
                )
                local_matches = (
                    local_item is not None
                    and local_inventory_matches(
                        local_item,
                        object_revision=server_revision,
                        object_hash=server_hash,
                        deleted=deleted,
                    )
                )
                if deleted:
                    tombstones.append(
                        SyncRestorePreviewObject(
                            dataset_id=dataset.dataset_id,
                            domain=domain,
                            object_id=object_id,
                            action=restore_action_for_domain(
                                domain,
                                deleted=True,
                                local_present=local_item is not None,
                            ),
                            server_revision=server_revision,
                            server_hash=server_hash,
                            server_cursor=envelope.server_cursor,
                            local_revision=local_item.object_revision if local_item is not None else None,
                            local_hash=local_item.object_hash if local_item is not None else None,
                            local_deleted=local_item.deleted if local_item is not None else None,
                            deleted=True,
                            parent_id=envelope.parent_id,
                        )
                    )
                    continue
                if local_item is None or local_matches:
                    safe_applies.append(
                        SyncRestorePreviewObject(
                            dataset_id=dataset.dataset_id,
                            domain=domain,
                            object_id=object_id,
                            action=restore_action_for_domain(
                                domain,
                                deleted=False,
                                local_present=local_item is not None,
                            ),
                            server_revision=server_revision,
                            server_hash=server_hash,
                            server_cursor=envelope.server_cursor,
                            local_revision=local_item.object_revision if local_item is not None else None,
                            local_hash=local_item.object_hash if local_item is not None else None,
                            local_deleted=local_item.deleted if local_item is not None else None,
                            deleted=False,
                            parent_id=envelope.parent_id,
                        )
                    )
                    continue
                conflict_type = (
                    "whole_object_conflict"
                    if domain in WHOLE_OBJECT_RESTORE_DOMAINS
                    else "stable_id_conflict"
                )
                object_conflicts.append(
                    SyncRestorePreviewObjectConflict(
                        dataset_id=dataset.dataset_id,
                        domain=domain,
                        object_id=object_id,
                        conflict_type=conflict_type,
                        server_revision=server_revision,
                        server_hash=server_hash,
                        server_cursor=envelope.server_cursor,
                        server_deleted=False,
                        local_revision=local_item.object_revision,
                        local_hash=local_item.object_hash,
                        local_deleted=local_item.deleted,
                        message="Local object differs from the server restore candidate.",
                    )
                )

            seen_refs: set[tuple[str, str]] = set()
            latest_attachment_envelopes: dict[str, SyncEnvelope] = {}
            for envelope in domain_envelopes.get("attachment.ref", []):
                latest_attachment_envelopes[envelope.object_id] = envelope
            for envelope in sorted(latest_attachment_envelopes.values(), key=lambda item: item.server_cursor or 0):
                if envelope.operation == "tombstone" or envelope.deleted:
                    continue
                try:
                    metadata = extract_attachment_ref_metadata(envelope)
                except AttachmentRefValidationError:
                    continue
                if selected_attachment_id_set and (
                    metadata.attachment_id not in selected_attachment_id_set
                    and metadata.payload_hash not in selected_attachment_id_set
                ):
                    continue
                if (
                    not selected_attachment_id_set
                    and selected_object_id_set
                    and metadata.parent_object_id not in selected_object_id_set
                ):
                    continue
                ref_key = (metadata.attachment_id, metadata.payload_hash)
                if ref_key in seen_refs:
                    continue
                seen_refs.add(ref_key)
                server_blob = None
                if self.settings.supports_attachments:
                    server_blob = self.store.get_blob_object(
                        dataset.dataset_id,
                        attachment_id=metadata.attachment_id,
                        payload_hash=metadata.payload_hash,
                        owner_user_id=user_id,
                    )
                metadata_claims_server_blob = (
                    not self.settings.supports_attachments
                    and _attachment_ref_has_server_blob(metadata.availability)
                )
                server_availability = "available" if server_blob is not None or metadata_claims_server_blob else "metadata_only"
                download_status = attachment_restore_status(
                    attachment_availability,
                    attachment_id=metadata.attachment_id,
                    payload_hash=metadata.payload_hash,
                )
                verified_locally = attachment_verified_locally(
                    attachment_availability,
                    attachment_id=metadata.attachment_id,
                    payload_hash=metadata.payload_hash,
                )
                required_for_restore = not metadata_only
                blob_details.append(
                    SyncRestoreBlobCompleteness(
                        attachment_id=metadata.attachment_id,
                        payload_hash=metadata.payload_hash,
                        size_bytes=metadata.size_bytes,
                        content_type=metadata.content_type,
                        parent_domain=metadata.parent_domain,
                        parent_object_id=metadata.parent_object_id,
                        server_availability=server_availability,
                        download_status=download_status,
                        required_for_restore=required_for_restore,
                    )
                )
                summary = SyncRestorePreviewAttachmentRef(
                    dataset_id=dataset.dataset_id,
                    attachment_id=metadata.attachment_id,
                    object_id=envelope.object_id,
                    parent_domain=metadata.parent_domain,
                    parent_object_id=metadata.parent_object_id,
                    content_type=metadata.content_type,
                    size_bytes=metadata.size_bytes,
                    payload_hash=metadata.payload_hash,
                    availability=server_availability,
                    server_cursor=envelope.server_cursor or 0,
                )
                attachment_refs.append(summary)
                if required_for_restore and server_availability != "available" and not attachment_available_locally(
                    attachment_availability,
                    attachment_id=metadata.attachment_id,
                    payload_hash=metadata.payload_hash,
                ):
                    missing_blobs.append(summary)
                    warnings.append(
                        SyncRestorePreviewWarning(
                            code="sync_attachment_blob_missing",
                            message=("Attachment blob is not available from the Sync v2 " "M1 server."),
                            dataset_id=dataset.dataset_id,
                            attachment_id=metadata.attachment_id,
                            object_id=envelope.object_id,
                            payload_hash=metadata.payload_hash,
                        )
                    )
                if verified_locally and server_availability != "available":
                    warnings.append(
                        SyncRestorePreviewWarning(
                            code="sync_attachment_blob_verified_without_server_copy",
                            message=(
                                "Attachment blob is verified locally but is not available "
                                "from the Sync v2 server."
                            ),
                            dataset_id=dataset.dataset_id,
                            attachment_id=metadata.attachment_id,
                            object_id=envelope.object_id,
                            payload_hash=metadata.payload_hash,
                        )
                    )

        domain_details = _build_restore_domain_details(
            safe_applies=safe_applies,
            tombstones=tombstones,
            object_conflicts=object_conflicts,
            blob_details=blob_details,
            metadata_only=metadata_only,
        )
        restore_status = _restore_status_from_domain_details(
            domain_details,
            has_blob_selection=bool(blob_details),
            metadata_only=metadata_only,
        )

        return SyncRestorePreview(
            datasets=preview_datasets,
            safe_applies=safe_applies,
            object_conflicts=object_conflicts,
            tombstones=tombstones,
            attachment_refs=attachment_refs,
            missing_blobs=missing_blobs,
            envelope_ranges=envelope_ranges,
            total_counts=dict(sorted(total_counts.items())),
            encryption=self.settings.server_trusted_encryption.encryption,
            key_status=key_status,
            warnings=warnings,
            generated_at=self.clock() or None,
            filters_applied={
                "dataset_ids": list(dataset_ids or []),
                "domains": list(domains or []),
                "selected_object_ids": sorted(selected_object_id_set),
                "selected_attachment_ids": sorted(selected_attachment_id_set),
                "metadata_only": metadata_only,
            },
            restore_status=restore_status,
            domain_details=domain_details,
            blob_details=blob_details,
            metadata_only_allowed=True,
        )

    def repair(
        self,
        *,
        user_id: str,
        dataset_id: str,
        device_id: str | None = None,
        domains: Sequence[SyncDomain] | None = None,
        since_cursor: int = 0,
        failed_only: bool = False,
        limit: int | None = None,
    ) -> SyncReplayRepairResult:
        """Replay accepted envelopes to repair server-side materialized projections."""

        if since_cursor < 0:
            raise SyncStoreError("Invalid sync cursor: repair since_cursor must be non-negative")
        if limit is not None and limit < 1:
            raise SyncStoreError("Sync repair limit must be greater than zero")
        if device_id is not None:
            self._require_registered_device(user_id, device_id)
        dataset = self._require_dataset_access(user_id=user_id, dataset_id=dataset_id)
        selected_domains = self._selected_domains(dataset, domains)

        def _repair_materialize(envelope: SyncEnvelope) -> MaterializationResult:
            materialization = self._materialize_envelope(envelope)
            if materialization.status == "conflict":
                snapshot = self._envelope_snapshot(envelope)
                self._store_materialization_conflict(dataset, snapshot, materialization)
            return materialization

        return SyncReplayRepairer(
            store=self.store,
            materializers=self.materializers,
            materialize=_repair_materialize,
            snapshot=self._envelope_snapshot,
            scan_limit=self.settings.restore_manifest_scan_limit,
        ).run(
            dataset_id=dataset.dataset_id,
            domains=selected_domains,
            since_cursor=since_cursor,
            failed_only=failed_only,
            limit=limit,
        )

    def store_attachment(
        self,
        *,
        user_id: str,
        dataset_id: str,
        domain: SyncDomain,
        entity_id: str,
        attachment_id: str,
        content_type: str,
        size_bytes: int,
        payload_ciphertext: str,
        payload_hash: str,
        encryption_policy: EncryptionPolicy = DEFAULT_M1_ENCRYPTION_POLICY,
        metadata: dict[str, object] | None = None,
    ) -> SyncAttachment:
        """Persist a small encrypted attachment payload for later restore."""

        blob_store = self._require_blob_transfer()
        if encryption_policy != DEFAULT_M1_ENCRYPTION_POLICY:
            raise SyncStoreError("Sync attachment persistence requires server_trusted_v1 encryption")
        payload = payload_ciphertext.encode("utf-8")
        if (
            size_bytes > self.settings.max_attachment_bytes
            or _ciphertext_exceeds_attachment_limit(
                payload_ciphertext,
                self.settings.max_attachment_bytes,
            )
            or len(payload) != size_bytes
        ):
            raise SyncStoreError("Sync attachment payload exceeds the server size limit")
        if not payload_ciphertext:
            raise SyncStoreError("Sync attachment payload_ciphertext is required")
        self._validate_sha256_hash(payload_hash, field_name="payload_hash")
        self._require_blob_dataset(user_id=user_id, dataset_id=dataset_id, domain=domain)
        blob_id = self.id_factory("blob")
        upload_id = self.id_factory("blob-upload")
        storage_key = self._write_single_chunk_blob(
            blob_store=blob_store,
            upload_id=upload_id,
            payload=payload,
            payload_hash=payload_hash,
        )
        blob = self.store.complete_blob_upload(
            SyncBlobObjectCreate(
                blob_id=blob_id,
                dataset_id=dataset_id,
                owner_user_id=user_id,
                attachment_id=attachment_id,
                payload_hash=payload_hash,
                content_type=content_type,
                size_bytes=size_bytes,
                storage_backend=self.settings.blob_storage_backend,
                storage_key=storage_key,
                encryption_policy=encryption_policy,
                metadata=dict(metadata or {}),
            )
        )
        attachment_metadata = dict(metadata or {})
        attachment_metadata.update(
            {
                "blob_id": blob.blob_id,
                "storage_backend": blob.storage_backend,
                "storage_key": blob.storage_key,
            }
        )
        return self.store.store_attachment(
            SyncAttachmentCreate(
                attachment_id=attachment_id,
                dataset_id=dataset_id,
                domain=domain,
                entity_id=entity_id,
                content_type=content_type,
                size_bytes=size_bytes,
                payload_ciphertext=payload_ciphertext,
                payload_hash=payload_hash,
                encryption_policy=encryption_policy,
                metadata=attachment_metadata,
            )
        )

    def create_blob_upload_session(
        self,
        *,
        user_id: str,
        dataset_id: str,
        device_id: str | None,
        domain: SyncDomain,
        entity_id: str,
        attachment_id: str,
        content_type: str,
        size_bytes: int,
        payload_hash: str,
        chunk_size: int,
        chunk_count: int,
        idempotency_key: str | None = None,
        encryption_policy: EncryptionPolicy = DEFAULT_M1_ENCRYPTION_POLICY,
        metadata: dict[str, object] | None = None,
    ) -> SyncBlobUploadSession:
        """Create or resume a quota-checked M2 blob upload session."""

        self._require_blob_transfer()
        if encryption_policy != DEFAULT_M1_ENCRYPTION_POLICY:
            raise SyncStoreError("Sync blob upload requires server_trusted_v1 encryption")
        if device_id is not None:
            self._require_registered_device(user_id, device_id)
        self._require_blob_dataset(user_id=user_id, dataset_id=dataset_id, domain=domain)
        self._validate_blob_limits(
            user_id=user_id,
            dataset_id=dataset_id,
            size_bytes=size_bytes,
            chunk_size=chunk_size,
            chunk_count=chunk_count,
        )
        self._validate_sha256_hash(payload_hash, field_name="payload_hash")
        return self.store.create_blob_upload_session(
            SyncBlobUploadSessionCreate(
                upload_id=self.id_factory("blob-upload"),
                dataset_id=dataset_id,
                owner_user_id=user_id,
                device_id=device_id,
                attachment_id=attachment_id,
                domain=domain,
                object_id=entity_id,
                content_type=content_type,
                size_bytes=size_bytes,
                payload_hash=payload_hash,
                chunk_size=chunk_size,
                chunk_count=chunk_count,
                reserved_quota_bytes=size_bytes,
                idempotency_key=idempotency_key,
                metadata=dict(metadata or {}),
            )
        )

    def get_blob_upload_session(
        self,
        *,
        user_id: str,
        dataset_id: str,
        upload_id: str,
    ) -> SyncBlobUploadSession:
        """Return an upload session after checking dataset ownership."""

        self._require_blob_dataset(user_id=user_id, dataset_id=dataset_id)
        session = self.store.get_blob_upload_session(upload_id, dataset_id=dataset_id)
        if session is None:
            raise SyncStoreError(f"Sync blob upload session not found: {upload_id}")
        if session.device_id is not None:
            self._require_registered_device(user_id, session.device_id)
        return session

    def upload_blob_chunk(
        self,
        *,
        user_id: str,
        dataset_id: str,
        upload_id: str,
        chunk_index: int,
        offset_bytes: int,
        chunk_payload: bytes,
        chunk_hash: str,
    ) -> SyncBlobChunk:
        """Verify, store, and record one resumable upload chunk."""

        blob_store = self._require_blob_transfer()
        session = self.get_blob_upload_session(
            user_id=user_id,
            dataset_id=dataset_id,
            upload_id=upload_id,
        )
        self._validate_sha256_hash(chunk_hash, field_name="chunk_hash")
        if len(chunk_payload) > self.settings.max_chunk_bytes:
            raise SyncStoreError("Sync blob chunk exceeds the server size limit")
        if offset_bytes < 0:
            raise SyncStoreError("Sync blob chunk offset is invalid")
        expected_size = min(session.chunk_size, max(session.size_bytes - offset_bytes, 0))
        if len(chunk_payload) != expected_size:
            raise SyncStoreError("Sync blob chunk size does not match the upload session")
        try:
            storage_key = blob_store.write_upload_chunk(
                upload_id=upload_id,
                chunk_index=chunk_index,
                payload=chunk_payload,
                expected_hash=chunk_hash,
            )
        except SyncBlobStoreError as exc:
            raise SyncStoreError(str(exc)) from exc
        return self.store.record_blob_chunk(
            SyncBlobChunkCreate(
                upload_id=upload_id,
                dataset_id=dataset_id,
                chunk_index=chunk_index,
                offset_bytes=offset_bytes,
                size_bytes=len(chunk_payload),
                chunk_hash=chunk_hash,
                storage_key=storage_key,
            )
        )

    def complete_blob_upload(
        self,
        *,
        user_id: str,
        dataset_id: str,
        upload_id: str,
    ) -> SyncBlobObject:
        """Verify all chunks and mark a blob object available."""

        blob_store = self._require_blob_transfer()
        session = self.get_blob_upload_session(
            user_id=user_id,
            dataset_id=dataset_id,
            upload_id=upload_id,
        )
        if session.missing_chunks:
            raise SyncStoreError("Sync blob upload session is missing chunks")
        try:
            storage_key = blob_store.commit_upload(
                upload_id=upload_id,
                payload_hash=session.payload_hash,
                chunk_indexes=list(range(session.chunk_count)),
            )
        except SyncBlobStoreError as exc:
            raise SyncStoreError(str(exc)) from exc
        return self.store.complete_blob_upload(
            SyncBlobObjectCreate(
                blob_id=self.id_factory("blob"),
                dataset_id=dataset_id,
                owner_user_id=user_id,
                attachment_id=session.attachment_id,
                payload_hash=session.payload_hash,
                content_type=session.content_type,
                size_bytes=session.size_bytes,
                storage_backend=self.settings.blob_storage_backend,
                storage_key=storage_key,
                encryption_policy=DEFAULT_M1_ENCRYPTION_POLICY,
                metadata={},
            )
        )

    def cancel_blob_upload(
        self,
        *,
        user_id: str,
        dataset_id: str,
        upload_id: str,
    ) -> SyncBlobUploadSession:
        """Cancel an upload session and remove staged chunks."""

        blob_store = self._require_blob_transfer()
        session = self.get_blob_upload_session(
            user_id=user_id,
            dataset_id=dataset_id,
            upload_id=upload_id,
        )
        session = self.store.cancel_blob_upload_session(session.upload_id, dataset_id=dataset_id)
        blob_store.discard_upload(upload_id)
        return session

    def blob_download_manifest(
        self,
        *,
        user_id: str,
        dataset_id: str,
        attachment_id: str,
        chunk_size: int | None = None,
    ) -> SyncBlobDownloadManifest:
        """Return resumable download metadata for an available blob or metadata-only ref."""

        blob_store = self._require_blob_transfer()
        self._require_blob_dataset(user_id=user_id, dataset_id=dataset_id)
        blob = self.store.get_blob_object(
            dataset_id,
            attachment_id=attachment_id,
            owner_user_id=user_id,
        )
        if blob is None:
            return self._metadata_only_blob_manifest(
                dataset_id=dataset_id,
                attachment_id=attachment_id,
            )
        payload = self._read_blob_payload(blob_store=blob_store, blob=blob)
        normalized_chunk_size = self._normalize_download_chunk_size(chunk_size)
        chunks = [
            SyncBlobDownloadChunk(
                chunk_index=index,
                offset_bytes=offset,
                size_bytes=len(payload[offset : offset + normalized_chunk_size]),
                chunk_hash=_sha256_bytes(payload[offset : offset + normalized_chunk_size]),
                download_url=(
                    f"/api/v1/sync/attachments/{attachment_id}"
                    f"?dataset_id={dataset_id}&offset={offset}"
                    f"&size={len(payload[offset : offset + normalized_chunk_size])}"
                ),
            )
            for index, offset in enumerate(range(0, len(payload), normalized_chunk_size))
        ]
        return SyncBlobDownloadManifest(
            dataset_id=dataset_id,
            attachment_id=attachment_id,
            blob_id=blob.blob_id,
            availability="available",
            content_type=blob.content_type,
            size_bytes=blob.size_bytes,
            payload_hash=blob.payload_hash,
            chunks=chunks,
        )

    def read_blob_bytes(
        self,
        *,
        user_id: str,
        dataset_id: str,
        attachment_id: str,
        offset: int = 0,
        size: int | None = None,
    ) -> bytes:
        """Read a byte range from an available Sync v2 M2 blob."""

        if offset < 0:
            raise SyncStoreError("Sync blob download offset is invalid")
        if size is not None and size < 0:
            raise SyncStoreError("Sync blob download size is invalid")
        blob_store = self._require_blob_transfer()
        self._require_blob_dataset(user_id=user_id, dataset_id=dataset_id)
        blob = self.store.get_blob_object(
            dataset_id,
            attachment_id=attachment_id,
            owner_user_id=user_id,
        )
        if blob is None:
            raise SyncStoreError("Sync blob was not found or is not accessible")
        payload = self._read_blob_payload(blob_store=blob_store, blob=blob)
        stop = None if size is None else offset + size
        return payload[offset:stop]

    def list_conflicts(
        self,
        *,
        user_id: str,
        dataset_id: str,
        status: ConflictStatus | None = None,
    ) -> list[SyncConflict]:
        self._require_dataset_access(user_id=user_id, dataset_id=dataset_id)
        return self.store.list_conflicts(dataset_id, status=status)

    def resolve_conflict(
        self,
        *,
        user_id: str,
        conflict_id: str,
        dataset_id: str | None = None,
        action: str,
        resolution_envelope: SyncEnvelopeCreate | None = None,
        resolved_by_envelope_id: str | None = None,
        resolved_by_device_id: str | None = None,
        notes: str | None = None,
    ) -> SyncConflict:
        conflict = self.store.get_conflict(conflict_id)
        if conflict is None:
            raise SyncStoreError("Sync conflict was not found or is not accessible")
        if dataset_id is not None and conflict.dataset_id != dataset_id:
            raise SyncStoreError("Sync conflict was not found or is not accessible")
        try:
            dataset = self._require_dataset_access(user_id=user_id, dataset_id=conflict.dataset_id)
        except SyncStoreError as exc:
            raise SyncStoreError("Sync conflict was not found or is not accessible") from exc
        if action not in {"overwrite", "duplicate_rename", "skip"}:
            raise SyncStoreError(f"Sync conflict resolution action is not supported: {action}")
        if conflict.status != "unresolved":
            if self._is_conflict_resolution_replay(
                conflict,
                action=action,
                resolution_envelope=resolution_envelope,
                resolved_by_envelope_id=resolved_by_envelope_id,
                resolved_by_device_id=resolved_by_device_id,
                notes=notes,
            ):
                return conflict
            raise SyncStoreError("Sync conflict is already resolved")
        if resolved_by_device_id is not None:
            self._require_registered_device(user_id, resolved_by_device_id)
        resolution_server_cursor: int | None = None
        if action in {"overwrite", "duplicate_rename"} and resolution_envelope is None:
            raise SyncStoreError(f"Sync {action} requires a resolution envelope")
        if action == "skip" and resolution_envelope is not None:
            raise SyncStoreError(f"Sync {action} must not include a resolution envelope")
        resolution_claim: tuple[str | None, str, str | None] | None = None
        if resolution_envelope is not None:
            resolution_device_id = resolved_by_device_id or resolution_envelope.device_id
            self._require_registered_device(user_id, resolution_device_id or "")
            if resolution_envelope.dataset_id != dataset.dataset_id:
                raise SyncStoreError("Sync resolution envelope dataset_id must match the conflict dataset")
            if resolution_envelope.domain != conflict.domain:
                raise SyncStoreError("Sync resolution envelope must target the conflict domain")
            if action == "duplicate_rename":
                if resolution_envelope.entity_id == conflict.entity_id:
                    raise SyncStoreError("Sync duplicate_rename resolution envelope must use a distinct object_id")
            elif resolution_envelope.entity_id != conflict.entity_id:
                raise SyncStoreError("Sync resolution envelope must target the conflict entity")
            if (
                resolved_by_device_id is not None
                and resolution_envelope.device_id is not None
                and resolution_envelope.device_id != resolved_by_device_id
            ):
                raise SyncStoreError("Sync resolution envelope device_id must match resolved_by_device_id")
            if self._payload_exceeds_size_limit(resolution_envelope):
                raise SyncStoreError("Sync resolution envelope payload exceeds the server size limit")
            try:
                outcome = self._evaluate_envelope(dataset, resolution_envelope)
            except PrivatePayloadValidationError as exc:
                raise SyncStoreError("Sync resolution envelope private payload validation failed") from exc
            if not isinstance(outcome, AdapterAccepted):
                raise SyncStoreError("Sync resolution envelope was not accepted")
            self.store.claim_conflict_resolution(
                conflict_id,
                dataset_id=dataset.dataset_id,
                resolved_by_device_id=resolution_device_id,
                resolution_action=action,
                resolution_notes=notes,
            )
            resolution_claim = (resolution_device_id, action, notes)
            try:
                inserted = self.store.insert_envelope(
                    replace(
                        resolution_envelope,
                        device_id=resolution_device_id,
                        status="accepted",
                    )
                )
                if inserted.apply_status != "applied":
                    materialization = self._materialize_envelope(inserted)
                    inserted = self._envelope_snapshot(inserted)
                    if materialization.status == "conflict":
                        self._store_materialization_conflict(dataset, inserted, materialization)
                    if materialization.status in {"failed", "conflict"} or inserted.apply_status in {
                        "failed",
                        "conflict",
                    }:
                        raise SyncStoreError("Sync resolution envelope was not applied")
            except Exception:
                self.store.release_conflict_resolution_claim(
                    conflict_id,
                    dataset_id=dataset.dataset_id,
                    resolved_by_device_id=resolution_device_id,
                    resolution_action=action,
                    resolution_notes=notes,
                )
                raise
            resolved_by_envelope_id = inserted.envelope_id
            resolution_server_cursor = inserted.server_cursor
            resolved_by_device_id = resolution_device_id
        resolved_status: ConflictStatus = "dismissed" if action == "skip" else "resolved"
        try:
            return self.store.resolve_conflict(
                conflict_id,
                dataset_id=dataset.dataset_id,
                server_cursor=resolution_server_cursor,
                status=resolved_status,
                resolved_by_envelope_id=resolved_by_envelope_id,
                resolved_by_device_id=resolved_by_device_id,
                resolution_action=action,
                resolution_notes=notes,
            )
        except Exception:
            if resolution_claim is not None:
                claim_device_id, claim_action, claim_notes = resolution_claim
                self.store.release_conflict_resolution_claim(
                    conflict_id,
                    dataset_id=dataset.dataset_id,
                    resolved_by_device_id=claim_device_id,
                    resolution_action=claim_action,
                    resolution_notes=claim_notes,
                )
            raise

    def _is_conflict_resolution_replay(
        self,
        conflict: SyncConflict,
        *,
        action: str,
        resolution_envelope: SyncEnvelopeCreate | None,
        resolved_by_envelope_id: str | None,
        resolved_by_device_id: str | None,
        notes: str | None,
    ) -> bool:
        if action != conflict.resolution_action or notes != conflict.resolution_notes:
            return False
        effective_device_id = resolved_by_device_id
        if effective_device_id is None and resolution_envelope is not None:
            effective_device_id = resolution_envelope.device_id
        if effective_device_id != conflict.resolved_by_device_id:
            return False
        effective_envelope_id = resolved_by_envelope_id
        if effective_envelope_id is None and resolution_envelope is not None:
            existing = self._find_existing_resolution_envelope(
                conflict.dataset_id,
                resolution_envelope,
                effective_device_id=effective_device_id,
            )
            if existing is None:
                return False
            effective_envelope_id = existing.envelope_id
        return effective_envelope_id == conflict.resolved_by_envelope_id

    def _find_existing_resolution_envelope(
        self,
        dataset_id: str,
        resolution_envelope: SyncEnvelopeCreate,
        *,
        effective_device_id: str | None,
    ) -> SyncEnvelope | None:
        if resolution_envelope.dataset_id != dataset_id:
            return None
        try:
            return self.store.get_existing_envelope_for_idempotency(
                replace(
                    resolution_envelope,
                    device_id=effective_device_id,
                    status="accepted",
                )
            )
        except SyncIdempotencyConflictError:
            return None

    def store_key_recovery_bundle(
        self,
        *,
        user_id: str,
        dataset_id: str,
        device_id: str | None,
        key_purpose: str,
        wrapped_key_blob: str,
        kdf_metadata: dict[str, object] | None = None,
        recovery_hint: str | None = None,
        rotation_of_key_record_id: str | None = None,
        encryption_policy: EncryptionPolicy = DEFAULT_M1_ENCRYPTION_POLICY,
        key_epoch: int = 1,
        active_from_server_sequence: int | None = None,
        superseded_at: str | None = None,
        wrapped_for: SyncKeyWrappedFor = "recovery",
        rewrap_status: SyncKeyRewrapStatus = "not_required",
    ) -> SyncKeyRecord:
        dataset = self._require_dataset_access(user_id=user_id, dataset_id=dataset_id)
        if device_id is not None:
            self._require_registered_device(user_id, device_id)
        self._validate_key_recovery_bundle(
            user_id=user_id,
            dataset_id=dataset.dataset_id,
            key_purpose=key_purpose,
            wrapped_key_blob=wrapped_key_blob,
            kdf_metadata=kdf_metadata,
            rotation_of_key_record_id=rotation_of_key_record_id,
        )
        return self.store.store_key_record(
            SyncKeyRecordCreate(
                key_record_id=self.id_factory("key"),
                dataset_id=dataset.dataset_id,
                user_id=user_id,
                device_id=device_id,
                key_purpose=key_purpose,
                wrapped_key_blob=wrapped_key_blob,
                kdf_metadata=dict(kdf_metadata or {}),
                recovery_hint=recovery_hint,
                rotation_of_key_record_id=rotation_of_key_record_id,
                encryption_policy=encryption_policy,
                key_epoch=key_epoch,
                active_from_server_sequence=active_from_server_sequence,
                superseded_at=superseded_at,
                wrapped_for=wrapped_for,
                rewrap_status=rewrap_status,
            )
        )

    def list_key_recovery_bundles(
        self,
        *,
        user_id: str,
        dataset_id: str,
        device_id: str | None = None,
        key_purpose: str | None = "dataset_recovery",
    ) -> list[SyncKeyRecord]:
        dataset = self._require_dataset_access(user_id=user_id, dataset_id=dataset_id)
        return [
            record
            for record in self.store.list_key_records(
                dataset.dataset_id,
                user_id=user_id,
                device_id=device_id,
                key_purpose=key_purpose,
            )
            if record.revoked_at is None
        ]

    def preview_key_rotation(
        self,
        *,
        user_id: str,
        dataset_id: str,
        target_encryption_policy: EncryptionPolicy,
        source_key_record_ids: Sequence[str] | None = None,
    ) -> SyncKeyRotationResult:
        dataset = self._require_dataset_access(user_id=user_id, dataset_id=dataset_id)
        if target_encryption_policy not in SYNC_V2_ENCRYPTION_POLICIES:
            self._raise_invalid_key_rotation()

        all_records = self.store.list_key_records(
            dataset.dataset_id,
            user_id=user_id,
            key_purpose=SYNC_DATASET_RECOVERY_KEY_PURPOSE,
        )
        active_records = [
            record
            for record in all_records
            if record.revoked_at is None and record.superseded_at is None
        ]
        selected_records, blockers = self._select_key_rotation_sources(
            active_records=active_records,
            all_records=all_records,
            source_key_record_ids=source_key_record_ids,
        )
        if not selected_records:
            blockers.append("sync_key_rotation_no_active_source_keys")

        highest_epoch = max((record.key_epoch for record in all_records), default=0)
        retained_range = self.store.get_dataset_envelope_range(dataset.dataset_id)
        active_from = (retained_range.through_server_sequence or 0) + 1
        summaries = [self._key_rotation_record_summary(record) for record in selected_records]
        return SyncKeyRotationResult(
            dataset_id=dataset.dataset_id,
            target_encryption_policy=target_encryption_policy,
            next_key_epoch=max(highest_epoch + 1, 1),
            active_from_server_sequence=active_from,
            can_commit=not blockers,
            committed=False,
            retained_envelope_range=retained_range,
            affected_key_records=summaries,
            blockers=list(dict.fromkeys(blockers)),
            device_ids=sorted(
                {
                    record.device_id
                    for record in selected_records
                    if record.device_id is not None
                }
            ),
            recovery_target_count=len(selected_records),
        )

    def commit_key_rotation(
        self,
        *,
        user_id: str,
        dataset_id: str,
        rotation_id: str,
        target_encryption_policy: EncryptionPolicy,
        wrapped_key_blob: str,
        kdf_metadata: Mapping[str, object] | None = None,
        source_key_record_ids: Sequence[str] | None = None,
        wrapped_for: SyncKeyWrappedFor = "recovery",
        rewrap_status: SyncKeyRewrapStatus = "complete",
        recovery_hint: str | None = None,
    ) -> SyncKeyRotationResult:
        clean_rotation_id = str(rotation_id or "").strip()
        if not clean_rotation_id:
            self._raise_invalid_key_rotation()

        dataset = self._require_dataset_access(user_id=user_id, dataset_id=dataset_id)
        self._validate_key_recovery_bundle(
            user_id=user_id,
            dataset_id=dataset.dataset_id,
            key_purpose=SYNC_DATASET_RECOVERY_KEY_PURPOSE,
            wrapped_key_blob=wrapped_key_blob,
            kdf_metadata=kdf_metadata,
            rotation_of_key_record_id=None,
        )
        new_key_record_id = self._key_rotation_record_id(
            user_id=user_id,
            dataset_id=dataset.dataset_id,
            rotation_id=clean_rotation_id,
        )
        new_record, superseded_records, retained_range = self.store.commit_key_rotation(
            SyncKeyRecordCreate(
                key_record_id=new_key_record_id,
                dataset_id=dataset.dataset_id,
                user_id=user_id,
                key_purpose=SYNC_DATASET_RECOVERY_KEY_PURPOSE,
                wrapped_key_blob=wrapped_key_blob,
                kdf_metadata=dict(kdf_metadata or {}),
                recovery_hint=recovery_hint,
                encryption_policy=target_encryption_policy,
                wrapped_for=wrapped_for,
                rewrap_status=rewrap_status,
            ),
            source_key_record_ids=source_key_record_ids or [],
            superseded_at=self.clock(),
        )
        return self._key_rotation_result_from_records(
            dataset_id=dataset.dataset_id,
            target_encryption_policy=target_encryption_policy,
            next_key_epoch=new_record.key_epoch,
            active_from_server_sequence=new_record.active_from_server_sequence or 1,
            retained_range=retained_range,
            source_records=superseded_records,
            new_record=new_record,
            rotation_id=clean_rotation_id,
            committed=True,
        )

    def _validate_key_recovery_bundle(
        self,
        *,
        user_id: str,
        dataset_id: str,
        key_purpose: str,
        wrapped_key_blob: str,
        kdf_metadata: Mapping[str, object] | None,
        rotation_of_key_record_id: str | None,
    ) -> None:
        if key_purpose != SYNC_DATASET_RECOVERY_KEY_PURPOSE:
            self._raise_invalid_key_recovery_bundle()
        if not isinstance(wrapped_key_blob, str) or not wrapped_key_blob.strip():
            self._raise_invalid_key_recovery_bundle()
        if len(wrapped_key_blob.encode("utf-8")) > SYNC_KEY_RECOVERY_MAX_WRAPPED_KEY_BYTES:
            self._raise_invalid_key_recovery_bundle()

        metadata = dict(kdf_metadata or {})
        algorithm = _key_recovery_metadata_string(
            metadata,
            "algorithm",
            "wrapping_algorithm",
            nested_parent="wrapping",
        )
        salt = _key_recovery_metadata_string(metadata, "salt", nested_parent="kdf")
        if algorithm is None or salt is None:
            self._raise_invalid_key_recovery_bundle()

        if rotation_of_key_record_id is None:
            return
        active_rotation_target = any(
            record.key_record_id == rotation_of_key_record_id and record.revoked_at is None
            for record in self.store.list_key_records(
                dataset_id,
                user_id=user_id,
                key_purpose=SYNC_DATASET_RECOVERY_KEY_PURPOSE,
            )
        )
        if not active_rotation_target:
            self._raise_invalid_key_recovery_bundle()

    @staticmethod
    def _raise_invalid_key_recovery_bundle() -> None:
        raise SyncStoreError("Sync key recovery bundle is invalid")

    @staticmethod
    def _raise_invalid_key_rotation() -> None:
        raise SyncStoreError("Sync key rotation is invalid")

    @staticmethod
    def _key_rotation_record_id(*, user_id: str, dataset_id: str, rotation_id: str) -> str:
        namespace = f"{user_id}\0{dataset_id}\0{rotation_id}".encode()
        return f"key-rotation-{hashlib.sha256(namespace).hexdigest()[:32]}"

    @staticmethod
    def _key_rotation_record_summary(record: SyncKeyRecord) -> SyncKeyRotationKeyRecord:
        return SyncKeyRotationKeyRecord(
            key_record_id=record.key_record_id,
            key_epoch=record.key_epoch,
            encryption_policy=record.encryption_policy,
            wrapped_for=record.wrapped_for,
            rewrap_status=record.rewrap_status,
            device_id=record.device_id,
            key_purpose=record.key_purpose,
            active_from_server_sequence=record.active_from_server_sequence,
            superseded_at=record.superseded_at,
            revoked_at=record.revoked_at,
            rotation_of_key_record_id=record.rotation_of_key_record_id,
        )

    def _select_key_rotation_sources(
        self,
        *,
        active_records: Sequence[SyncKeyRecord],
        all_records: Sequence[SyncKeyRecord],
        source_key_record_ids: Sequence[str] | None,
    ) -> tuple[list[SyncKeyRecord], list[str]]:
        requested_ids = [
            str(record_id).strip()
            for record_id in source_key_record_ids or []
            if str(record_id).strip()
        ]
        requested_ids = list(dict.fromkeys(requested_ids))
        if not requested_ids:
            return list(active_records), []

        active_by_id = {record.key_record_id: record for record in active_records}
        all_by_id = {record.key_record_id: record for record in all_records}
        selected: list[SyncKeyRecord] = []
        blockers: list[str] = []
        for record_id in requested_ids:
            active_record = active_by_id.get(record_id)
            if active_record is not None:
                selected.append(active_record)
                continue
            if record_id in all_by_id:
                blockers.append("sync_key_rotation_source_inactive")
            else:
                blockers.append("sync_key_rotation_source_missing")
        return selected, blockers

    def _existing_key_rotation_sources(
        self,
        *,
        existing_records: Sequence[SyncKeyRecord],
        existing_new_record: SyncKeyRecord,
        source_key_record_ids: Sequence[str] | None,
    ) -> list[SyncKeyRecord]:
        requested_ids = [
            str(record_id).strip()
            for record_id in source_key_record_ids or []
            if str(record_id).strip()
        ]
        if not requested_ids and existing_new_record.rotation_of_key_record_id:
            requested_ids = [existing_new_record.rotation_of_key_record_id]
        record_by_id = {record.key_record_id: record for record in existing_records}
        return [
            record_by_id[record_id]
            for record_id in dict.fromkeys(requested_ids)
            if record_id in record_by_id
        ]

    @staticmethod
    def _validate_existing_key_rotation_record(
        record: SyncKeyRecord,
        *,
        target_encryption_policy: EncryptionPolicy,
        wrapped_key_blob: str,
        kdf_metadata: Mapping[str, object] | None,
        wrapped_for: SyncKeyWrappedFor,
        rewrap_status: SyncKeyRewrapStatus,
    ) -> None:
        if (
            record.encryption_policy != target_encryption_policy
            or record.wrapped_key_blob != wrapped_key_blob
            or record.kdf_metadata != dict(kdf_metadata or {})
            or record.wrapped_for != wrapped_for
            or record.rewrap_status != rewrap_status
        ):
            raise SyncIdempotencyConflictError(
                "Sync key rotation ID was reused with different key material"
            )

    def _key_rotation_result_from_records(
        self,
        *,
        dataset_id: str,
        target_encryption_policy: EncryptionPolicy,
        next_key_epoch: int,
        active_from_server_sequence: int,
        retained_range,
        source_records: Sequence[SyncKeyRecord],
        new_record: SyncKeyRecord,
        rotation_id: str,
        committed: bool,
    ) -> SyncKeyRotationResult:
        affected = [self._key_rotation_record_summary(record) for record in source_records]
        return SyncKeyRotationResult(
            dataset_id=dataset_id,
            target_encryption_policy=target_encryption_policy,
            next_key_epoch=next_key_epoch,
            active_from_server_sequence=active_from_server_sequence,
            can_commit=True,
            committed=committed,
            retained_envelope_range=retained_range,
            affected_key_records=affected,
            blockers=[],
            device_ids=sorted(
                {
                    record.device_id
                    for record in source_records
                    if record.device_id is not None
                }
            ),
            recovery_target_count=len(source_records),
            rotation_id=rotation_id,
            new_key_record=self._key_rotation_record_summary(new_record),
        )

    def _evaluate_envelope(
        self,
        dataset: SyncDataset,
        envelope: SyncEnvelopeCreate,
    ) -> AdapterAccepted | AdapterRejected | AdapterConflict | AdapterDeferred:
        if envelope.adapter_version not in self.adapters.get(envelope.domain).supported_adapter_versions:
            return AdapterRejected(
                client_envelope_id=envelope.client_envelope_id,
                error_code="unsupported_adapter_version",
                message=f"Adapter version is not supported for {envelope.domain}",
            )
        if dataset.encryption_policy == "client_private_v1":
            validate_private_payload(
                payload_ciphertext=envelope.payload_ciphertext,
                payload_clear=envelope.payload_clear,
            )
        context = SyncAdapterContext(
            prior_envelopes=self.store.list_envelopes_for_entity(
                dataset.dataset_id,
                envelope.domain,
                entity_id=envelope.entity_id,
                stable_key=envelope.stable_key,
                limit=100,
            )
        )
        adapter = self.adapters.get(envelope.domain)
        return _call_adapter_evaluate(adapter, envelope, dataset=dataset, context=context)

    def _require_registered_device(self, user_id: str, device_id: str) -> SyncDevice:
        if not device_id:
            raise SyncStoreError("Sync device was not found or is not accessible")
        device = self.store.get_device(user_id, device_id)
        if (
            device is not None
            and device.revoked_at is None
            and device.status == "active"
        ):
            return device
        raise SyncStoreError("Sync device was not found or is not accessible")

    def _require_dataset_access(self, *, user_id: str, dataset_id: str) -> SyncDataset:
        dataset = self.store.get_dataset(dataset_id)
        if dataset is None or dataset.archived_at is not None:
            raise SyncStoreError("Sync dataset was not found or is not accessible")
        if dataset.scope_type == "personal":
            if dataset.owner_user_id != user_id:
                raise SyncStoreError("Sync dataset was not found or is not accessible")
            return dataset
        if dataset.scope_type == "workspace":
            self._require_workspace_sync_access(
                user_id=user_id,
                workspace_id=dataset.workspace_id,
            )
            return dataset
        raise SyncStoreError("Sync dataset was not found or is not accessible")

    def _accessible_datasets(
        self,
        *,
        user_id: str,
        dataset_ids: Sequence[str] | None = None,
    ) -> list[SyncDataset]:
        if dataset_ids:
            return [
                self._require_dataset_access(user_id=user_id, dataset_id=dataset_id)
                for dataset_id in dataset_ids
            ]
        datasets: list[SyncDataset] = []
        for dataset in self.store.list_datasets_for_user(user_id):
            try:
                self._require_dataset_access(
                    user_id=user_id,
                    dataset_id=dataset.dataset_id,
                )
            except SyncStoreError:
                continue
            datasets.append(dataset)
        return datasets

    def _require_workspace_sync_access(
        self,
        *,
        user_id: str,
        workspace_id: str | None,
    ) -> None:
        if not workspace_id or not workspace_id.strip():
            raise SyncStoreError("Sync workspace was not found or is not accessible")
        if self.workspace_access_checker is None:
            raise SyncStoreError("Sync workspace was not found or is not accessible")
        try:
            granted = bool(self.workspace_access_checker(user_id, workspace_id, "sync"))
        except Exception as exc:
            raise SyncStoreError("Sync workspace was not found or is not accessible") from exc
        if not granted:
            raise SyncStoreError("Sync workspace was not found or is not accessible")

    def _default_background_policy(
        self,
        dataset_id: str,
        device_id: str,
    ) -> SyncBackgroundPolicy:
        return SyncBackgroundPolicy(
            dataset_id=dataset_id,
            device_id=device_id,
            enabled=True,
            minimum_interval_seconds=300,
            backoff_floor_seconds=60,
            max_batch_size=self.settings.max_batch_size,
            max_blob_bytes_per_run=(
                self.settings.max_blob_bytes
                if self.settings.max_blob_bytes is not None
                else self.settings.max_attachment_bytes
            ),
            respect_metered_networks=True,
            maintenance_window=None,
            paused_reason=None,
            pending_local_changes=False,
            updated_at=self.clock(),
        )

    def _require_server_trusted_encryption_ready(self) -> None:
        if not self.settings.server_trusted_encryption.encryption.get("ready", False):
            raise SyncStoreError(
                "sync_encryption_attestation_required: Sync v2 M1 requires "
                "server_trusted_v1 at-rest encryption readiness before dataset enrollment"
            )

    def _payload_exceeds_size_limit(self, envelope: SyncEnvelopeCreate) -> bool:
        max_bytes = self.settings.max_envelope_payload_bytes
        if envelope.payload_size_bytes is not None and envelope.payload_size_bytes > max_bytes:
            return True
        actual_size = 0
        if envelope.payload_ciphertext is not None:
            actual_size += len(envelope.payload_ciphertext.encode("utf-8"))
        actual_size += _compact_json_size(envelope.payload_clear)
        actual_size += _compact_json_size(envelope.routing_metadata)
        actual_size += _compact_json_size(envelope.dependencies)
        return actual_size > max_bytes

    def _store_conflict(
        self,
        dataset: SyncDataset,
        envelope: SyncEnvelopeCreate,
        outcome: AdapterConflict,
    ) -> SyncPushConflict:
        inserted = self.store.insert_envelope(replace(envelope, status="conflict"))
        existing = self.store.get_unresolved_conflict_for_envelope(
            dataset.dataset_id,
            local_envelope_id=envelope.client_envelope_id,
            server_sequence=inserted.server_sequence,
        )
        if existing is not None:
            return SyncPushConflict(
                conflict_id=existing.conflict_id,
                client_envelope_id=outcome.client_envelope_id,
                domain=existing.domain,
                entity_id=existing.entity_id,
                server_sequence=existing.server_sequence,
                message=outcome.message,
            )
        conflict = self.store.insert_conflict(
            SyncConflictCreate(
                conflict_id=self.id_factory("conflict"),
                dataset_id=dataset.dataset_id,
                domain=outcome.domain,
                entity_id=outcome.entity_id,
                conflict_type=outcome.conflict_type,
                local_envelope_id=envelope.client_envelope_id,
                server_sequence=inserted.server_sequence,
                metadata=dict(outcome.metadata),
            )
        )
        return SyncPushConflict(
            conflict_id=conflict.conflict_id,
            client_envelope_id=outcome.client_envelope_id,
            domain=outcome.domain,
            entity_id=outcome.entity_id,
            server_sequence=inserted.server_sequence,
            message=outcome.message,
        )

    def _materialize_envelope(self, envelope: SyncEnvelope) -> MaterializationResult:
        materializer = self.materializers.get(envelope.domain)
        if materializer is None:
            return MaterializationResult(status="skipped")
        try:
            return materializer.apply(envelope, store=self.store)
        except Exception as exc:  # noqa: BLE001 - materializer failures are captured as replayable sync state.
            error_code = "sync_projection_failed"
            error_message = _safe_projection_error_message(exc)
            if envelope.server_cursor is not None:
                self.store.mark_envelope_apply_status(
                    envelope.server_cursor,
                    apply_status="failed",
                    apply_error_code=error_code,
                    apply_error_message=error_message,
                )
            return MaterializationResult(
                status="failed",
                error_code=error_code,
                message=error_message,
            )

    def _envelope_snapshot(self, envelope: SyncEnvelope) -> SyncEnvelope:
        """Reload an envelope after projection updates apply status fields."""

        if envelope.server_cursor is None:
            return envelope
        candidates = self.store.list_envelopes_after(
            envelope.dataset_id,
            max(envelope.server_cursor - 1, 0),
            limit=1,
            domains=[envelope.domain],
            status=None,
        )
        for candidate in candidates:
            if candidate.server_cursor == envelope.server_cursor:
                return candidate
        return envelope

    def _push_accepted_from_envelope(self, envelope: SyncEnvelope) -> SyncPushAccepted:
        object_revision = envelope.object_revision
        state = self.store.get_object_state(
            envelope.dataset_id,
            envelope.domain,
            envelope.object_id,
        )
        if state is not None and state.latest_server_cursor == envelope.server_cursor:
            object_revision = state.object_revision
        return SyncPushAccepted(
            client_envelope_id=envelope.client_envelope_id,
            server_sequence=envelope.server_sequence,
            domain=envelope.domain,
            entity_id=envelope.entity_id,
            object_revision=object_revision,
            apply_status=envelope.apply_status,
            apply_error_code=envelope.apply_error_code,
            apply_error_message=envelope.apply_error_message,
        )

    def _store_materialization_conflict(
        self,
        dataset: SyncDataset,
        envelope: SyncEnvelope,
        result: MaterializationResult,
    ) -> SyncPushConflict:
        existing = self.store.get_unresolved_conflict_for_envelope(
            dataset.dataset_id,
            local_envelope_id=envelope.client_envelope_id,
            server_sequence=envelope.server_sequence,
        )
        if existing is not None:
            return SyncPushConflict(
                conflict_id=existing.conflict_id,
                client_envelope_id=envelope.client_envelope_id,
                domain=existing.domain,
                entity_id=existing.entity_id,
                server_sequence=existing.server_sequence,
                message=result.message,
            )
        conflict = self.store.insert_conflict(
            SyncConflictCreate(
                conflict_id=self.id_factory("conflict"),
                dataset_id=dataset.dataset_id,
                domain=envelope.domain,
                entity_id=envelope.entity_id,
                conflict_type=result.conflict_type or "materialization_conflict",
                local_envelope_id=envelope.client_envelope_id,
                server_sequence=envelope.server_sequence,
                metadata=dict(result.metadata),
            )
        )
        return SyncPushConflict(
            conflict_id=conflict.conflict_id,
            client_envelope_id=envelope.client_envelope_id,
            domain=envelope.domain,
            entity_id=envelope.entity_id,
            server_sequence=envelope.server_sequence,
            message=result.message,
        )

    def _resolve_cursor(
        self,
        dataset_id: str,
        device_id: str,
        cursor: str | int | None,
        domains: Sequence[SyncDomain] | None,
    ) -> int:
        if cursor is not None:
            return self._parse_cursor(cursor)
        cursor_domains = list(domains or self.settings.supported_domains)
        cursors: list[int] = []
        for domain in cursor_domains:
            stored = self.store.get_device_cursor(dataset_id, device_id, domain)
            cursors.append(stored.last_pulled_sequence if stored is not None else 0)
        return min(cursors, default=0)

    def _parse_cursor(self, cursor: str | int | None) -> int:
        if cursor is None:
            return 0
        try:
            parsed = int(cursor)
        except (TypeError, ValueError) as exc:
            raise SyncStoreError(f"Invalid sync cursor: {cursor!r}") from exc
        if parsed < 0:
            raise SyncStoreError(f"Invalid sync cursor: {cursor!r}")
        return parsed

    def _selected_domains(
        self,
        dataset: SyncDataset,
        domains: Sequence[SyncDomain] | None,
    ) -> list[SyncDomain]:
        allowed = set(dataset.domains)
        requested = list(domains or dataset.domains)
        return [domain for domain in requested if domain in allowed and self.adapters.has_domain(domain)]

    def _profile_manager(self) -> SyncV2ProfileManager:
        return SyncV2ProfileManager(
            store=self.store,
            capabilities_factory=self.capabilities,
            id_factory=self.id_factory,
            scan_limit=self.settings.restore_manifest_scan_limit,
        )

    def _update_cursors(
        self,
        dataset_id: str,
        device_id: str,
        domains: Sequence[SyncDomain],
        sequence: int,
    ) -> None:
        for domain in domains:
            self.store.update_device_cursor(
                SyncDeviceCursor(
                    dataset_id=dataset_id,
                    device_id=device_id,
                    domain=domain,
                    last_pulled_sequence=sequence,
                )
            )

    def _scan_pull_page(
        self,
        *,
        dataset_id: str,
        device_id: str,
        since_sequence: int,
        domains: Sequence[SyncDomain],
        page_limit: int,
        include_own_changes: bool,
    ) -> tuple[list[SyncEnvelope], list[SyncEnvelope]]:
        raw = self.store.list_envelopes_after(
            dataset_id,
            since_sequence,
            limit=page_limit + 1,
            domains=domains,
            status="accepted",
            exclude_device_id=None if include_own_changes else device_id,
        )
        visible = [envelope for envelope in raw if envelope.apply_status != "conflict"]
        return raw, visible

    def _manifest_dataset(
        self,
        dataset: SyncDataset,
        *,
        user_id: str,
        domains: set[SyncDomain],
    ) -> SyncRestoreManifestDataset:
        selected_domains = [domain for domain in dataset.domains if not domains or domain in domains]
        stats = self.store.summarize_restore_manifest_dataset(
            dataset.dataset_id,
            user_id=user_id,
            domains=selected_domains,
        )
        last_updated_at = stats.last_updated_at or dataset.updated_at
        if last_updated_at < dataset.updated_at:
            last_updated_at = dataset.updated_at
        metadata: dict[str, object] = {} if dataset.encryption_policy == "client_private_v1" else dict(dataset.metadata)
        return SyncRestoreManifestDataset(
            dataset_id=dataset.dataset_id,
            scope_type=dataset.scope_type,
            encryption_policy=dataset.encryption_policy,
            domains=selected_domains,
            workspace_id=dataset.workspace_id,
            approximate_counts=stats.approximate_counts,
            byte_estimates=stats.byte_estimates,
            last_updated_at=last_updated_at,
            unresolved_conflicts=stats.unresolved_conflicts,
            attachment_availability=stats.attachment_availability,
            attachment_size_classes=stats.attachment_size_classes,
            key_recovery_available=stats.key_recovery_available,
            metadata=metadata,
        )

    def _require_blob_transfer(self) -> LocalSyncBlobStore:
        if not self.settings.supports_attachments or self.blob_store is None:
            raise SyncStoreError(
                "sync_blob_transfer_not_supported: Sync v2 M1 does not support binary blob transfer"
            )
        return self.blob_store

    def _metadata_only_blob_manifest(
        self,
        *,
        dataset_id: str,
        attachment_id: str,
    ) -> SyncBlobDownloadManifest:
        envelopes = self.store.list_envelopes_for_entity(
            dataset_id,
            "attachment.ref",
            entity_id=attachment_id,
            limit=1,
        )
        if not envelopes or envelopes[0].operation == "tombstone":
            raise SyncStoreError("Sync blob was not found or is not accessible")
        try:
            metadata = extract_attachment_ref_metadata(envelopes[0])
        except AttachmentRefValidationError as exc:
            raise SyncStoreError("Sync blob was not found or is not accessible") from exc
        if metadata.attachment_id != attachment_id:
            raise SyncStoreError("Sync blob was not found or is not accessible")
        availability = "deleted" if metadata.availability == "deleted" else "metadata_only"
        return SyncBlobDownloadManifest(
            dataset_id=dataset_id,
            attachment_id=metadata.attachment_id,
            availability=availability,
            content_type=metadata.content_type,
            size_bytes=metadata.size_bytes,
            payload_hash=metadata.payload_hash,
            chunks=[],
        )

    def _normalize_download_chunk_size(self, chunk_size: int | None) -> int:
        normalized = self.settings.max_chunk_bytes if chunk_size is None else chunk_size
        if normalized <= 0:
            raise SyncStoreError("Sync blob download chunk size is invalid")
        if normalized > self.settings.max_chunk_bytes:
            raise SyncStoreError("Sync blob chunk exceeds the server size limit")
        return normalized

    def _read_blob_payload(
        self,
        *,
        blob_store: LocalSyncBlobStore,
        blob: SyncBlobObject,
    ) -> bytes:
        try:
            payload = blob_store.read_blob(blob.storage_key)
        except (OSError, SyncBlobStoreError) as exc:
            raise SyncStoreError("Sync blob was not found or is not accessible") from exc
        if len(payload) != blob.size_bytes or _sha256_bytes(payload) != blob.payload_hash:
            raise SyncStoreError("Sync blob storage integrity check failed")
        return payload

    def _require_blob_dataset(
        self,
        *,
        user_id: str,
        dataset_id: str,
        domain: SyncDomain | None = None,
    ) -> SyncDataset:
        dataset = self._require_dataset_access(user_id=user_id, dataset_id=dataset_id)
        if domain is not None and domain not in dataset.domains:
            raise SyncInvalidDomainError(f"Sync domain is not enrolled for this dataset: {domain}")
        return dataset

    def _validate_blob_limits(
        self,
        *,
        user_id: str,
        dataset_id: str,
        size_bytes: int,
        chunk_size: int,
        chunk_count: int,
    ) -> None:
        max_blob_bytes = self.settings.max_blob_bytes or self.settings.max_attachment_bytes
        if size_bytes <= 0 or size_bytes > max_blob_bytes:
            raise SyncStoreError("Sync attachment payload exceeds the server size limit")
        if chunk_size <= 0 or chunk_size > self.settings.max_chunk_bytes:
            raise SyncStoreError("Sync blob chunk exceeds the server size limit")
        if chunk_count <= 0 or chunk_size * chunk_count < size_bytes:
            raise SyncStoreError("Sync blob chunk shape is invalid")
        quota = self.store.summarize_blob_quota(user_id, dataset_id=dataset_id)
        if (
            self.settings.user_blob_quota_bytes is not None
            and quota.used_blob_bytes + quota.reserved_blob_bytes + size_bytes
            > self.settings.user_blob_quota_bytes
        ):
            raise SyncStoreError("Sync blob quota exceeded")
        if quota.active_upload_count >= self.settings.max_active_blob_uploads:
            raise SyncStoreError("Sync blob active upload limit exceeded")

    def _validate_sha256_hash(self, value: str, *, field_name: str) -> None:
        prefix = "sha256:"
        if not value.startswith(prefix) or len(value) != len(prefix) + 64:
            raise SyncStoreError(f"Sync blob {field_name} must be sha256:<64 hex chars>")
        try:
            bytes.fromhex(value[len(prefix) :])
        except ValueError as exc:
            raise SyncStoreError(f"Sync blob {field_name} digest must be hex") from exc

    def _write_single_chunk_blob(
        self,
        *,
        blob_store: LocalSyncBlobStore,
        upload_id: str,
        payload: bytes,
        payload_hash: str,
    ) -> str:
        try:
            blob_store.write_upload_chunk(
                upload_id=upload_id,
                chunk_index=0,
                payload=payload,
                expected_hash=payload_hash,
            )
            return blob_store.commit_upload(
                upload_id=upload_id,
                payload_hash=payload_hash,
                chunk_indexes=[0],
            )
        except SyncBlobStoreError as exc:
            raise SyncStoreError(str(exc)) from exc

    def _latest_cursor_for_domains(
        self,
        dataset_id: str,
        domains: Sequence[SyncDomain],
    ) -> int | None:
        if not domains:
            return None
        envelopes = self.store.list_envelopes_after(
            dataset_id,
            0,
            limit=self.settings.restore_manifest_scan_limit,
            domains=domains,
        )
        return max((envelope.server_cursor or 0 for envelope in envelopes), default=None)


def _compact_json_size(value: object) -> int:
    return len(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))


def _ciphertext_exceeds_attachment_limit(
    payload_ciphertext: str,
    max_attachment_bytes: int,
) -> bool:
    """Return whether persisted ciphertext text exceeds the attachment cap."""

    return len(payload_ciphertext.encode("utf-8")) > max_attachment_bytes


def _attachment_ref_has_server_blob(availability: str) -> bool:
    return availability.strip().lower() in ATTACHMENT_REF_SERVER_AVAILABILITY


def _normalize_selection_set(values: Sequence[str] | None) -> set[str]:
    selected: set[str] = set()
    for value in values or []:
        text = str(value).strip()
        if text:
            selected.add(text)
    return selected


def _build_restore_domain_details(
    *,
    safe_applies: Sequence[SyncRestorePreviewObject],
    tombstones: Sequence[SyncRestorePreviewObject],
    object_conflicts: Sequence[SyncRestorePreviewObjectConflict],
    blob_details: Sequence[SyncRestoreBlobCompleteness],
    metadata_only: bool,
) -> list[SyncRestoreDomainCompleteness]:
    counts: dict[SyncDomain, dict[str, int]] = {}

    def domain_counts(domain: SyncDomain) -> dict[str, int]:
        return counts.setdefault(
            domain,
            {
                "selected_count": 0,
                "safe_apply_count": 0,
                "conflict_count": 0,
                "tombstone_count": 0,
                "required_blob_count": 0,
                "available_blob_count": 0,
                "missing_blob_count": 0,
                "verified_blob_count": 0,
            },
        )

    for item in safe_applies:
        bucket = domain_counts(item.domain)
        bucket["selected_count"] += 1
        bucket["safe_apply_count"] += 1
    for item in tombstones:
        bucket = domain_counts(item.domain)
        bucket["selected_count"] += 1
        bucket["tombstone_count"] += 1
    for item in object_conflicts:
        bucket = domain_counts(item.domain)
        bucket["selected_count"] += 1
        bucket["conflict_count"] += 1
    for item in blob_details:
        bucket = domain_counts("attachment.ref")
        bucket["selected_count"] += 1
        if item.required_for_restore:
            bucket["required_blob_count"] += 1
            if item.server_availability == "available":
                bucket["available_blob_count"] += 1
            elif item.download_status not in {
                "available",
                "present",
                "stored",
                "server",
                "verified",
                "verified_complete",
            }:
                bucket["missing_blob_count"] += 1
            if item.download_status in {"verified", "verified_complete"}:
                bucket["verified_blob_count"] += 1

    domain_order = {domain: index for index, domain in enumerate(SYNC_V2_SUPPORTED_DOMAINS)}
    details: list[SyncRestoreDomainCompleteness] = []
    for domain, values in sorted(
        counts.items(),
        key=lambda item: domain_order.get(item[0], len(domain_order)),
    ):
        details.append(
            SyncRestoreDomainCompleteness(
                domain=domain,
                status=_restore_status_for_counts(values, metadata_only=metadata_only),
                selected_count=values["selected_count"],
                safe_apply_count=values["safe_apply_count"],
                conflict_count=values["conflict_count"],
                tombstone_count=values["tombstone_count"],
                required_blob_count=values["required_blob_count"],
                available_blob_count=values["available_blob_count"],
                missing_blob_count=values["missing_blob_count"],
                verified_blob_count=values["verified_blob_count"],
            )
        )
    return details


def _restore_status_for_counts(
    values: Mapping[str, int],
    *,
    metadata_only: bool,
) -> SyncRestoreCompletenessStatus:
    if values["conflict_count"] > 0:
        return "blocked_by_conflicts"
    required_blob_count = values["required_blob_count"]
    if required_blob_count == 0:
        if metadata_only and values["selected_count"] > 0:
            return "metadata_ready"
        return "content_complete"
    if values["missing_blob_count"] > 0:
        return "blob_incomplete"
    if values["verified_blob_count"] >= required_blob_count:
        return "verified_complete"
    return "content_complete"


def _restore_status_from_domain_details(
    domain_details: Sequence[SyncRestoreDomainCompleteness],
    *,
    has_blob_selection: bool,
    metadata_only: bool,
) -> SyncRestoreCompletenessStatus:
    if any(item.status == "blocked_by_conflicts" for item in domain_details):
        return "blocked_by_conflicts"
    if any(item.status == "blob_incomplete" for item in domain_details):
        return "blob_incomplete"
    if metadata_only and has_blob_selection:
        return "metadata_ready"
    required_blob_count = sum(item.required_blob_count for item in domain_details)
    verified_blob_count = sum(item.verified_blob_count for item in domain_details)
    if required_blob_count > 0 and verified_blob_count >= required_blob_count:
        return "verified_complete"
    return "content_complete"


def _call_adapter_evaluate(
    adapter: SyncDomainAdapter,
    envelope: SyncEnvelopeCreate,
    *,
    dataset: SyncDataset,
    context: SyncAdapterContext,
) -> AdapterAccepted | AdapterRejected | AdapterConflict | AdapterDeferred:
    evaluate = adapter.evaluate_envelope
    if _evaluate_accepts_context(evaluate):
        return evaluate(envelope, dataset=dataset, context=context)
    return evaluate(envelope, dataset=dataset)


def _evaluate_accepts_context(evaluate: Callable[..., object]) -> bool:
    parameters = inspect.signature(evaluate).parameters.values()
    return any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        or (parameter.name == "context" and parameter.kind != inspect.Parameter.POSITIONAL_ONLY)
        for parameter in parameters
    )


__all__ = [
    "SyncDatasetEnrollment",
    "SyncDeviceRegistration",
    "SyncPullResult",
    "SyncPushAccepted",
    "SyncPushConflict",
    "SyncPushRejected",
    "SyncPushResult",
    "SyncRestoreManifest",
    "SyncRestoreManifestDataset",
    "SyncRestoreManifestDevice",
    "SyncRestorePreview",
    "SyncRestorePreviewAttachmentRef",
    "SyncRestorePreviewDataset",
    "SyncRestorePreviewWarning",
    "SyncV2Capabilities",
    "SyncV2Service",
    "SyncV2Settings",
]
