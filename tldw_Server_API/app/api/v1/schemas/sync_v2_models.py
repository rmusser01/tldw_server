from __future__ import annotations

"""Pydantic schemas for the Sync v2 M1 protocol API."""

from typing import Any, Literal

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    field_validator,
    model_validator,
)

from tldw_Server_API.app.core.exceptions import NoteAttachmentPolicyError
from tldw_Server_API.app.core.Notes.attachment_policy import (
    canonicalize_note_attachment_file_name,
)
from tldw_Server_API.app.core.Sync.v2.models import (
    NOTES_LINK_DOMAINS,
    NOTES_LINK_SYNC_OPERATIONS,
    NOTES_ORGANIZATION_DOMAINS,
    NOTES_ORGANIZATION_SYNC_OPERATIONS,
    normalize_supported_adapter_versions,
    normalize_sync_v2_requested_domains,
    sync_v2_dataset_writable_adapter_versions,
    sync_v2_domain_schemas,
    sync_v2_server_supported_adapter_versions,
    validate_notes_note_upsert_payload,
)

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
    "notes.keyword",
    "notes.keyword_link",
    "notes.keyword_collection",
    "notes.keyword_collection_link",
    "notes.folder",
    "notes.folder_link",
    "notes.link",
    "notes.task",
    "notes.task_activity",
    "personal_context.manifest",
    "personal_context.scope",
    "personal_context.record",
    "personal_context.proposal",
    "personal_context.purge",
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
ConflictResolutionAction = Literal["overwrite", "duplicate_rename", "skip"]
SyncApplyStatus = Literal["pending", "applied", "failed", "conflict", "superseded"]
SyncProfileBootstrapMode = Literal["server_frontend", "offline_sync"]
SyncRestorePreviewAction = Literal["apply", "append", "delete", "hide", "noop"]
SyncRestoreOrderedActionKind = Literal["apply", "tombstone", "noop", "conflict"]
SyncDeviceStatus = Literal["pending_authorization", "active", "paused", "revoked"]
SyncDeviceAuthorizationStatus = Literal["pending", "approved", "rejected"]
SyncBackgroundLeaseStatus = Literal["acquired", "refreshed", "held_by_other"]
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
PERSONAL_CONTEXT_SYNC_DOMAINS: tuple[SyncDomain, ...] = (
    "personal_context.manifest",
    "personal_context.scope",
    "personal_context.record",
    "personal_context.proposal",
    "personal_context.purge",
)
PERSONAL_CONTEXT_SYNC_OPERATIONS: dict[SyncDomain, list[SyncOperation]] = {
    domain: ["upsert", "tombstone"] for domain in PERSONAL_CONTEXT_SYNC_DOMAINS
}
SYNC_V2_SUPPORTED_DOMAINS: list[SyncDomain] = (
    list(M1_SYNC_DOMAINS)
    + list(WORKSPACE_SYNC_DOMAINS)
    + list(SOURCE_CACHE_SYNC_DOMAINS)
    + list(MEDIA_SYNC_DOMAINS)
    + list(NOTES_ORGANIZATION_DOMAINS)
    + list(NOTES_LINK_DOMAINS)
    + list(PERSONAL_CONTEXT_SYNC_DOMAINS)
)
SYNC_V2_SUPPORTED_OPERATIONS: dict[SyncDomain, list[SyncOperation]] = {
    **M1_SYNC_OPERATIONS,
    **WORKSPACE_SYNC_OPERATIONS,
    **SOURCE_CACHE_SYNC_OPERATIONS,
    **MEDIA_SYNC_OPERATIONS,
    **NOTES_ORGANIZATION_SYNC_OPERATIONS,
    **NOTES_LINK_SYNC_OPERATIONS,
    **PERSONAL_CONTEXT_SYNC_OPERATIONS,
}
SYNC_V2_KNOWN_DOMAINS: tuple[SyncDomain, ...] = (
    *SYNC_V2_SUPPORTED_DOMAINS,
    "notes.task",
    "notes.task_activity",
)
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
SYNC_V2_MAX_PUSH_ENVELOPES = 100

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


def _default_encryption() -> dict[str, Any]:
    return {
        "policy": DEFAULT_M1_ENCRYPTION_POLICY,
        "ready": True,
        "attestation": {
            "scope": "user_database_directory",
            "covers": ["Sync_v2.db", "ChaChaNotes.db"],
            "configured": True,
        },
    }


def _default_blob_transfer() -> dict[str, bool]:
    return {"supported": False}


def _validate_sha256_hash(value: Any) -> str:
    text = str(value or "").strip()
    if not text.startswith("sha256:") or text == "sha256:":
        raise ValueError("hash values must use sha256:<digest>")
    return text


def _normalize_object_map(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    raise ValueError("value must be an object")


def _validate_policy_hash(value: Any, *, field_name: str) -> str:
    text = str(value or "").strip()
    if not text.startswith("sha256:") or text == "sha256:":
        raise ValueError(f"{field_name} must use sha256:<digest>")
    return text


class SyncEncryptionPolicyMetadata(BaseModel):
    """Public, non-secret metadata for a Sync v2 dataset encryption policy."""

    policy: EncryptionPolicy = DEFAULT_M1_ENCRYPTION_POLICY
    key_epoch: int = Field(1, ge=1)
    attestation: dict[str, Any] = Field(default_factory=dict)
    kdf_metadata: dict[str, Any] = Field(default_factory=dict)
    recovery_key_record_id: str | None = None
    device_key_record_ids: list[str] = Field(default_factory=list)
    server_materialization: str | None = None

    @field_validator("attestation", "kdf_metadata", mode="before")
    @classmethod
    def _default_metadata_maps(cls, value: Any) -> dict[str, Any]:
        return _normalize_object_map(value)

    @field_validator("device_key_record_ids", mode="before")
    @classmethod
    def _default_device_key_records(cls, value: Any) -> list[str]:
        if value is None:
            return []
        return value

    @field_validator("device_key_record_ids", mode="after")
    @classmethod
    def _normalize_device_key_records(cls, value: list[str]) -> list[str]:
        return [record_id.strip() for record_id in value if record_id.strip()]

    @model_validator(mode="after")
    def _validate_policy_metadata(self) -> SyncEncryptionPolicyMetadata:
        if self.policy == "server_trusted_v1":
            _validate_server_trusted_policy_metadata(self.attestation)
        elif self.policy == "passphrase_wrapped_v1":
            _validate_passphrase_wrapped_policy_metadata(
                kdf_metadata=self.kdf_metadata,
                recovery_key_record_id=self.recovery_key_record_id,
            )
        elif self.policy == "device_wrapped_v1":
            if not self.device_key_record_ids:
                raise ValueError("device_wrapped_v1 requires at least one device key record")
        elif self.policy == "client_private_v1" and self.server_materialization != "metadata_only":
            raise ValueError("client_private_v1 requires metadata_only server materialization")
        return self

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)


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
    _validate_policy_hash(kdf_metadata.get("params_hash"), field_name="params_hash")
    if not str(recovery_key_record_id or "").strip():
        raise ValueError("passphrase_wrapped_v1 requires a recovery key record reference")


def _with_transition_aliases(data: Any) -> Any:
    if not isinstance(data, dict):
        return data
    normalized = dict(data)
    if not normalized.get("object_id") and normalized.get("entity_id"):
        normalized["object_id"] = normalized["entity_id"]
    if normalized.get("server_cursor") is None and normalized.get("server_sequence") is not None:
        normalized["server_cursor"] = normalized["server_sequence"]
    if normalized.get("payload") is None and normalized.get("payload_clear") is not None:
        normalized["payload"] = normalized["payload_clear"]
    if normalized.get("created_at_client") is None and normalized.get("client_timestamp") is not None:
        normalized["created_at_client"] = normalized["client_timestamp"]
    if normalized.get("received_at_server") is None and normalized.get("server_timestamp") is not None:
        normalized["received_at_server"] = normalized["server_timestamp"]
    return normalized


class PersonalContextSyncCapabilitiesResponse(BaseModel):
    """Typed readiness and bounded transport contract for Personal Context."""

    available: bool = False
    blockers: list[str] = Field(
        default_factory=lambda: ["personal_context_profile_key_unavailable"]
    )
    authorization_policy: Literal["server_trusted_v1"] = "server_trusted_v1"
    min_schema_version: Literal[1] = 1
    max_schema_version: Literal[1] = 1
    integrity_algorithm: Literal["hmac-sha256-v1"] = "hmac-sha256-v1"
    integrity_key_distribution: Literal["wrapped-bootstrap-v1"] = "wrapped-bootstrap-v1"
    privacy_cleanup_ack: Literal["personal-context-cleanup-v1"] = "personal-context-cleanup-v1"
    purge_generation: Literal["personal-context-purge-v1"] = "personal-context-purge-v1"
    max_record_bytes: int = Field(16_384, ge=16_384)
    max_search_results: int = Field(20, ge=20)
    max_proposals_per_turn: int = Field(5, ge=5)
    max_proposals_per_session: int = Field(25, ge=25)
    max_unresolved_proposals: int = Field(200, ge=200)

    model_config = ConfigDict(extra="forbid")


class SyncCapabilitiesResponse(BaseModel):
    """Server-supported Sync v2 M1 protocol capabilities."""

    protocol_version: str = "sync-v2-m1"
    min_supported_protocol_version: str = "sync-v2-m1"
    domains: list[SyncDomain] = Field(
        default_factory=lambda: list(SYNC_V2_SUPPORTED_DOMAINS),
        validation_alias=AliasChoices("domains", "supported_domains"),
    )
    operations: dict[SyncDomain, list[SyncOperation]] = Field(
        default_factory=lambda: {domain: list(operations) for domain, operations in SYNC_V2_SUPPORTED_OPERATIONS.items()}
    )
    domain_schemas: dict[SyncDomain, dict[str, Any]] = Field(
        default_factory=sync_v2_domain_schemas
    )
    supported_adapter_versions: dict[SyncDomain, list[int]] = Field(
        default_factory=sync_v2_server_supported_adapter_versions
    )
    writable_adapter_versions: dict[SyncDomain, list[int]] = Field(
        default_factory=sync_v2_dataset_writable_adapter_versions
    )
    encryption: dict[str, Any] = Field(default_factory=_default_encryption)
    encryption_policies: list[EncryptionPolicy] = Field(default_factory=lambda: [DEFAULT_M1_ENCRYPTION_POLICY])
    personal_context: PersonalContextSyncCapabilitiesResponse = Field(
        default_factory=PersonalContextSyncCapabilitiesResponse
    )
    blob_transfer: dict[str, Any] = Field(default_factory=_default_blob_transfer)
    quota: dict[str, Any] = Field(default_factory=dict)
    max_batch_size: int = Field(100, ge=1)
    max_envelope_payload_bytes: int = Field(262_144, ge=1)
    max_attachment_bytes: int = Field(1_048_576, ge=1)
    supports_restore_manifest: bool = True
    supports_conflicts: bool = True
    supports_attachments: bool = False
    compatibility_flags: dict[str, bool] = Field(default_factory=dict)
    server_time: str | None = None
    warnings: list[dict[str, str]] = Field(default_factory=list)

    @field_validator("protocol_version", "min_supported_protocol_version", mode="before")
    @classmethod
    def _normalize_protocol_version(cls, value: Any) -> str:
        if value in (None, 2, "2"):
            return "sync-v2-m1"
        return str(value)

    @field_validator("domains", mode="before")
    @classmethod
    def _default_m1_domains(cls, value: Any) -> list[SyncDomain]:
        if value in (None, []):
            return list(SYNC_V2_SUPPORTED_DOMAINS)
        if isinstance(value, list) and all(domain in SYNC_V2_KNOWN_DOMAINS for domain in value):
            return value
        return list(SYNC_V2_SUPPORTED_DOMAINS)

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class SyncDeviceRegisterRequest(BaseModel):
    """Request to register or refresh a Sync v2 device."""

    device_id: str | None = Field(None, description="Existing device ID when refreshing registration.")
    display_name: str = Field(..., description="Human-readable device name.")
    client_type: str = Field("chatbook", description="Client family, such as chatbook or webui.")
    client_version: str | None = None
    supported_domains: list[SyncDomain] = Field(default_factory=lambda: list(M1_SYNC_DOMAINS))
    supported_adapter_versions: dict[str, list[StrictInt]] | None = None
    capabilities: dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("capabilities", "client_capabilities"),
    )

    @model_validator(mode="after")
    def _normalize_supported_adapter_versions(self) -> SyncDeviceRegisterRequest:
        requested_domains = normalize_sync_v2_requested_domains(
            self.supported_domains
        )
        legacy_requested = self.capabilities.get("requested_domains")
        if legacy_requested is not None:
            normalized_legacy_requested = normalize_sync_v2_requested_domains(
                legacy_requested
            )
            if (
                "supported_domains" in self.model_fields_set
                and normalized_legacy_requested != requested_domains
            ):
                raise ValueError(
                    "capabilities.requested_domains must match supported_domains"
                )
            if "supported_domains" not in self.model_fields_set:
                requested_domains = normalized_legacy_requested
        self.supported_domains = requested_domains
        supplied = self.supported_adapter_versions
        if supplied is None:
            supplied = self.capabilities.get("supported_adapter_versions")
        normalized = normalize_supported_adapter_versions(
            supplied,
            requested_domains=requested_domains,
        )
        self.supported_adapter_versions = normalized
        self.capabilities = {
            **self.capabilities,
            "requested_domains": requested_domains,
            "supported_adapter_versions": normalized,
        }
        return self


class SyncDeviceRegisterResponse(BaseModel):
    """Response after a device registration or refresh."""

    device_id: str
    server_capabilities: SyncCapabilitiesResponse = Field(
        default_factory=SyncCapabilitiesResponse,
        validation_alias=AliasChoices("server_capabilities", "capabilities"),
    )
    required_actions: list[str] = Field(default_factory=list)
    registered_at: str | None = None
    last_seen_at: str | None = None


class SyncDeviceResponse(BaseModel):
    """Device lifecycle metadata returned by Sync v2 M3 endpoints."""

    device_id: str
    user_id: str
    display_name: str
    client_type: str
    client_version: str | None = None
    capabilities: dict[str, Any] = Field(default_factory=dict)
    registered_at: str
    last_seen_at: str
    status: SyncDeviceStatus = "active"
    user_label: str | None = None
    authorized_at: str | None = None
    revoked_at: str | None = None
    revoked_reason: str | None = None


class SyncDeviceUpdateRequest(BaseModel):
    """Mutable user-facing metadata for an existing Sync v2 device."""

    display_name: str | None = None
    user_label: str | None = None
    client_version: str | None = None
    capabilities: dict[str, Any] | None = None


class SyncDeviceRevokeRequest(BaseModel):
    """Request to revoke a Sync v2 device."""

    reason: str | None = None
    revoke_key_records: bool = False


class SyncDeviceAuthorizationCreateRequest(BaseModel):
    """Request to create a pending Sync v2 device authorization."""

    dataset_id: str = Field(..., min_length=1)
    device_id: str = Field(..., min_length=1)
    authorization_method: str = Field(..., min_length=1)
    idempotency_key: str | None = None


class SyncDeviceAuthorizationApproveRequest(BaseModel):
    """Request to approve a pending Sync v2 device authorization."""

    dataset_id: str = Field(..., min_length=1)
    approving_device_id: str | None = None
    idempotency_key: str | None = None


class SyncDeviceAuthorizationResponse(BaseModel):
    """Stored Sync v2 device authorization metadata."""

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


class SyncDeviceDomainAckRequest(BaseModel):
    """Per-domain device acknowledgment submitted by a Sync v2 client."""

    domain: SyncDomain
    through_server_sequence: int = Field(..., ge=0)
    applied_at: str
    adapter_version: int = Field(1, ge=1)
    idempotency_key: str | None = None


class SyncDeviceBlobAckRequest(BaseModel):
    """Per-blob device verification acknowledgment submitted by a Sync v2 client."""

    attachment_id: str = Field(..., min_length=1)
    payload_hash: str
    verified_at: str
    idempotency_key: str | None = None

    @field_validator("payload_hash")
    @classmethod
    def _validate_payload_hash(cls, value: Any) -> str:
        return _validate_sha256_hash(value)


class SyncDeviceBlobIdAckRequest(BaseModel):
    """Immutable blob-ID verification evidence for adapter-v2 flows."""

    blob_id: str = Field(..., min_length=1)
    payload_hash: str
    verified_at: str
    idempotency_key: str | None = None

    @field_validator("payload_hash")
    @classmethod
    def _validate_payload_hash(cls, value: Any) -> str:
        return _validate_sha256_hash(value)


class SyncDeviceAcknowledgmentsRequest(BaseModel):
    """Batch of domain/blob acknowledgments for one dataset/device."""

    dataset_id: str = Field(..., min_length=1)
    device_id: str = Field(..., min_length=1)
    domain_acks: list[SyncDeviceDomainAckRequest] = Field(default_factory=list)
    blob_acks: list[SyncDeviceBlobAckRequest] = Field(default_factory=list)
    blob_id_acks: list[SyncDeviceBlobIdAckRequest] = Field(
        default_factory=list,
        max_length=800,
    )


class SyncDeviceDomainAckResponse(BaseModel):
    """Stored per-domain device acknowledgment."""

    dataset_id: str
    device_id: str
    domain: SyncDomain
    through_server_sequence: int = Field(..., ge=0)
    applied_at: str
    updated_at: str
    adapter_version: int = Field(1, ge=1)
    idempotency_key: str | None = None


class SyncDeviceBlobAckResponse(BaseModel):
    """Stored per-blob device verification acknowledgment."""

    dataset_id: str
    device_id: str
    attachment_id: str
    payload_hash: str
    verified_at: str
    updated_at: str
    idempotency_key: str | None = None


class SyncDeviceBlobIdAckResponse(BaseModel):
    """Stored immutable blob-ID verification evidence."""

    dataset_id: str
    device_id: str
    blob_id: str
    payload_hash: str
    verified_at: str
    updated_at: str
    idempotency_key: str | None = None


class SyncDeviceAcknowledgmentsResponse(BaseModel):
    """Stored acknowledgment summary for one dataset/device pair."""

    dataset_id: str
    device_id: str
    domain_acks: dict[SyncDomain, SyncDeviceDomainAckResponse] = Field(default_factory=dict)
    blob_acks: list[SyncDeviceBlobAckResponse] = Field(default_factory=list)
    version_acks: list[SyncDeviceDomainAckResponse] = Field(default_factory=list)
    blob_id_acks: list[SyncDeviceBlobIdAckResponse] = Field(default_factory=list)


class SyncBackgroundPolicyPatchRequest(BaseModel):
    """Background sync policy/user-intent update for one dataset/device."""

    dataset_id: str = Field(..., min_length=1)
    device_id: str = Field(..., min_length=1)
    enabled: bool | None = None
    minimum_interval_seconds: int | None = Field(None, ge=1)
    backoff_floor_seconds: int | None = Field(None, ge=1)
    max_batch_size: int | None = Field(None, ge=1)
    max_blob_bytes_per_run: int | None = Field(None, ge=0)
    respect_metered_networks: bool | None = None
    maintenance_window: dict[str, Any] | None = None
    paused_reason: str | None = None
    pending_local_changes: bool | None = None


class SyncBackgroundPolicyResponse(BaseModel):
    """Background sync policy hints and stored user intent."""

    dataset_id: str
    device_id: str
    enabled: bool = True
    minimum_interval_seconds: int = Field(300, ge=1)
    backoff_floor_seconds: int = Field(60, ge=1)
    max_batch_size: int = Field(100, ge=1)
    max_blob_bytes_per_run: int | None = Field(None, ge=0)
    respect_metered_networks: bool = True
    maintenance_window: dict[str, Any] | None = None
    paused_reason: str | None = None
    pending_local_changes: bool = False
    updated_at: str | None = None


class SyncBackgroundLeaseRequest(BaseModel):
    """Request to acquire or refresh an advisory background sync lease."""

    dataset_id: str = Field(..., min_length=1)
    device_id: str = Field(..., min_length=1)
    lease_id: str | None = None
    ttl_seconds: int = Field(120, ge=1, le=3600)


class SyncBackgroundLeaseResponse(BaseModel):
    """Current advisory background sync lease state."""

    dataset_id: str
    device_id: str
    lease_id: str
    status: SyncBackgroundLeaseStatus
    acquired: bool
    expires_at: str
    updated_at: str


class SyncBackgroundDomainStatusResponse(BaseModel):
    """Per-domain background sync status counters."""

    domain: SyncDomain
    last_server_sequence: int = Field(0, ge=0)
    last_pulled_sequence: int = Field(0, ge=0)
    cursor_lag_count: int = Field(0, ge=0)
    unresolved_conflicts: int = Field(0, ge=0)
    replayable_failures: int = Field(0, ge=0)
    last_successful_push_at: str | None = None
    last_successful_pull_at: str | None = None
    blob_completeness: dict[str, int] = Field(default_factory=dict)


class SyncBackgroundStatusResponse(BaseModel):
    """Profile-level and per-domain background sync status."""

    dataset_id: str
    device_id: str
    policy: SyncBackgroundPolicyResponse
    lease: SyncBackgroundLeaseResponse | None = None
    domains: list[SyncBackgroundDomainStatusResponse] = Field(default_factory=list)
    conflict_count: int = Field(0, ge=0)
    replayable_failure_count: int = Field(0, ge=0)
    quota_pressure: dict[str, Any] = Field(default_factory=dict)
    restore_completeness: SyncRestoreCompletenessStatus = "metadata_ready"
    server_time: str | None = None


SyncRetentionCandidateType = Literal[
    "envelope_compaction",
    "tombstone_prune",
    "binding_release",
    "blob_gc",
]


class SyncRetentionDryRunRequest(BaseModel):
    """Request a read-only retention/GC candidate scan."""

    dataset_id: str = Field(..., min_length=1)
    device_id: str | None = Field(None, min_length=1)
    domains: list[SyncDomain] | None = None
    audit_mode: bool = True
    minimum_envelope_age_seconds: int = Field(0, ge=0)
    minimum_tombstone_age_seconds: int = Field(0, ge=0)
    offline_restore_window_seconds: int = Field(0, ge=0)
    limit: int | None = Field(None, ge=1)


class SyncRetentionCandidateResponse(BaseModel):
    """One redacted dry-run retention, compaction, or blob-GC candidate."""

    candidate_type: SyncRetentionCandidateType
    dataset_id: str
    domain: SyncDomain | None = None
    object_id: str | None = None
    server_sequence: int | None = Field(None, ge=1)
    blob_id: str | None = None
    attachment_id: str | None = None
    attachment_revision: int | None = Field(None, ge=1)
    payload_hash: str | None = None
    size_bytes: int | None = Field(None, ge=0)
    blockers: list[str] = Field(default_factory=list)
    required_device_ids: list[str] = Field(default_factory=list)
    unacknowledged_device_ids: list[str] = Field(default_factory=list)
    reason: str | None = None


class SyncRetentionDryRunResponse(BaseModel):
    """Read-only retention/GC dry-run response."""

    dataset_id: str
    dry_run: bool = True
    mutation_performed: bool = False
    evaluated_at: str | None = None
    audit_mode: bool = True
    minimum_envelope_age_seconds: int = Field(0, ge=0)
    minimum_tombstone_age_seconds: int = Field(0, ge=0)
    offline_restore_window_seconds: int = Field(0, ge=0)
    candidate_count: int = Field(0, ge=0)
    blocked_count: int = Field(0, ge=0)
    blocker_counts: dict[str, int] = Field(default_factory=dict)
    candidates: list[SyncRetentionCandidateResponse] = Field(default_factory=list)


class SyncRetentionCompactRequest(BaseModel):
    """Request guarded Sync v2 retention compaction/GC."""

    dataset_id: str = Field(..., min_length=1)
    device_id: str | None = Field(None, min_length=1)
    domains: list[SyncDomain] | None = None
    confirm: bool = False
    apply_envelope_compaction: bool = True
    apply_tombstone_prune: bool = True
    apply_binding_release: bool = True
    apply_blob_gc: bool = True
    minimum_envelope_age_seconds: int = Field(0, ge=0)
    minimum_tombstone_age_seconds: int = Field(0, ge=0)
    offline_restore_window_seconds: int = Field(0, ge=0)
    limit: int | None = Field(None, ge=1)


class SyncRetentionCompactResponse(BaseModel):
    """Guarded retention compaction/GC apply response."""

    dataset_id: str
    dry_run: bool = True
    mutation_performed: bool = False
    confirmation_required: bool = False
    evaluated_at: str | None = None
    candidate_count: int = Field(0, ge=0)
    applied_count: int = Field(0, ge=0)
    blocked_count: int = Field(0, ge=0)
    skipped_count: int = Field(0, ge=0)
    blockers: list[str] = Field(default_factory=list)
    blocker_counts: dict[str, int] = Field(default_factory=dict)
    domain_compactions: list[dict[str, Any]] = Field(default_factory=list)
    binding_releases: list[dict[str, Any]] = Field(default_factory=list)
    blob_gc: list[dict[str, Any]] = Field(default_factory=list)


class SyncDiagnosticsDomainResponse(BaseModel):
    """Redacted diagnostics for one Sync v2 domain."""

    domain: SyncDomain
    envelope_count: int = Field(0, ge=0)
    object_count: int = Field(0, ge=0)
    latest_server_sequence: int = Field(0, ge=0)
    failed_apply_count: int = Field(0, ge=0)
    unresolved_conflict_count: int = Field(0, ge=0)


class SyncDiagnosticsDeviceDomainLagResponse(BaseModel):
    """Redacted cursor lag for one device/domain pair."""

    domain: SyncDomain
    last_pulled_sequence: int = Field(0, ge=0)
    latest_server_sequence: int = Field(0, ge=0)
    lag_count: int = Field(0, ge=0)


class SyncDiagnosticsDeviceResponse(BaseModel):
    """Redacted diagnostics for one device."""

    device_id: str
    status: str
    last_seen_at: str | None = None
    domain_lag: list[SyncDiagnosticsDeviceDomainLagResponse] = Field(default_factory=list)


class SyncDiagnosticsBlobHealthResponse(BaseModel):
    """Redacted blob and upload diagnostics."""

    blob_object_count: int = Field(0, ge=0)
    available_blob_bytes: int = Field(0, ge=0)
    active_upload_count: int = Field(0, ge=0)
    reserved_blob_bytes: int = Field(0, ge=0)
    quota_limit_bytes: int | None = Field(None, ge=0)


class SyncDiagnosticsKeySummaryResponse(BaseModel):
    """Redacted key diagnostics without wrapped key material."""

    key_record_count: int = Field(0, ge=0)
    active_key_record_count: int = Field(0, ge=0)
    revoked_key_record_count: int = Field(0, ge=0)
    superseded_key_record_count: int = Field(0, ge=0)
    rewrap_pending_count: int = Field(0, ge=0)
    recovery_available: bool = False


class SyncDiagnosticsRetentionSummaryResponse(BaseModel):
    """Redacted retention dry-run diagnostics."""

    dry_run: bool = True
    mutation_performed: bool = False
    candidate_count: int = Field(0, ge=0)
    blocked_count: int = Field(0, ge=0)
    blocker_counts: dict[str, int] = Field(default_factory=dict)


class SyncRecoveryActionDescriptorResponse(BaseModel):
    """One explicit recovery hint; returning it never invokes the action."""

    action: Literal[
        "resume_upload",
        "retry_upload",
        "retry_verify",
        "repair_projection",
        "resolve_conflict",
        "restore_attachment",
        "restore_note",
        "release_quarantine",
        "bootstrap_resume",
        "gc_retry",
        "wait_for_retention",
    ]
    reason_code: str
    target_type: Literal[
        "dataset", "attachment", "blob", "upload", "conflict", "envelope"
    ] = "dataset"
    target_id: str | None = None
    retryable: bool = True
    requires_confirmation: bool = False

    model_config = ConfigDict(extra="forbid")


class SyncAttachmentDiagnosticSampleResponse(BaseModel):
    """One bounded owner-authorized attachment lifecycle sample."""

    category: str
    code: str
    attachment_id: str | None = None
    blob_id: str | None = None
    server_cursor: int | None = Field(None, ge=1)
    recovery_actions: list[SyncRecoveryActionDescriptorResponse] = Field(
        default_factory=list,
        max_length=4,
    )

    model_config = ConfigDict(extra="forbid")


class SyncAttachmentDiagnosticsResponse(BaseModel):
    """Bounded read-only Notes attachment lifecycle diagnostics."""

    counts: dict[str, int] = Field(default_factory=dict)
    samples: list[SyncAttachmentDiagnosticSampleResponse] = Field(
        default_factory=list,
        max_length=500,
    )
    recovery_actions: list[SyncRecoveryActionDescriptorResponse] = Field(
        default_factory=list,
        max_length=32,
    )

    model_config = ConfigDict(extra="forbid")


class SyncDiagnosticsResponse(BaseModel):
    """Redacted Sync v2 diagnostics response."""

    dataset_id: str
    generated_at: str | None = None
    domains: list[SyncDiagnosticsDomainResponse] = Field(default_factory=list)
    devices: list[SyncDiagnosticsDeviceResponse] = Field(default_factory=list)
    blob_health: SyncDiagnosticsBlobHealthResponse = Field(
        default_factory=SyncDiagnosticsBlobHealthResponse
    )
    key_summary: SyncDiagnosticsKeySummaryResponse = Field(
        default_factory=SyncDiagnosticsKeySummaryResponse
    )
    retention: SyncDiagnosticsRetentionSummaryResponse = Field(
        default_factory=SyncDiagnosticsRetentionSummaryResponse
    )
    attachment_lifecycle: SyncAttachmentDiagnosticsResponse = Field(
        default_factory=SyncAttachmentDiagnosticsResponse
    )


class SyncDatasetEnrollRequest(BaseModel):
    """Request to create or join a sync dataset."""

    dataset_id: str | None = Field(None, description="Existing dataset ID when joining a dataset.")
    device_id: str | None = None
    scope_type: DatasetScopeType = "personal"
    workspace_id: str | None = None
    domains: list[SyncDomain] = Field(default_factory=lambda: list(M1_SYNC_DOMAINS))
    encryption_policy: EncryptionPolicy = DEFAULT_M1_ENCRYPTION_POLICY
    metadata: dict[str, Any] = Field(default_factory=dict)


class SyncDatasetEnrollResponse(BaseModel):
    """Dataset metadata returned after enrollment."""

    dataset_id: str
    scope_type: DatasetScopeType
    encryption_policy: EncryptionPolicy
    domains: list[SyncDomain] = Field(default_factory=list)
    workspace_id: str | None = None
    cursors: dict[str, str | int] = Field(default_factory=dict)
    key_setup_required: bool = False
    created_at: str | None = None
    updated_at: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SyncProfileDeviceStatusResponse(BaseModel):
    """Device registration status in profile responses."""

    device_id: str | None = None
    registered: bool = False
    client_profile_id: str | None = None
    last_seen_at: str | None = None
    mode: SyncProfileBootstrapMode | None = None
    client_type: str | None = None
    client_version: str | None = None


class SyncNotesOrganizationStatusResponse(BaseModel):
    """Safe Notes organization bootstrap progress exposed to clients."""

    state: Literal["initializing", "ready", "failed"]
    captured_count: int = Field(0, ge=0)
    expected_count: int = Field(0, ge=0)
    error_code: str | None = None

    model_config = ConfigDict(extra="forbid")


class SyncNotesAttachmentCleanupSampleResponse(BaseModel):
    """One bounded public-safe legacy cleanup candidate."""

    source_key_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    attachment_id: str
    state: Literal["captured"] = "captured"
    blocker_code: str | None = None

    model_config = ConfigDict(extra="forbid")


class SyncNotesAttachmentBootstrapDiagnosticsResponse(BaseModel):
    """Read-only bounded legacy attachment bootstrap diagnostics."""

    state: Literal["not_started", "initializing", "ready", "failed"]
    captured_count: int = Field(0, ge=0)
    expected_count: int = Field(0, ge=0)
    cursor: str | None = Field(None, pattern=r"^sha256:[0-9a-f]{64}$")
    error_code: str | None = None
    dry_run: bool = False
    source_candidate_count: int | None = Field(None, ge=0, le=1_000)
    source_candidate_count_is_lower_bound: bool = False
    cleanup_candidates: list[SyncNotesAttachmentCleanupSampleResponse] = Field(
        default_factory=list,
        max_length=100,
    )
    recovery_actions: list[SyncRecoveryActionDescriptorResponse] = Field(
        default_factory=list,
        max_length=4,
    )

    model_config = ConfigDict(extra="forbid")


class SyncProfileDatasetStatusResponse(BaseModel):
    """Default personal dataset metadata in profile responses."""

    dataset_id: str
    scope: DatasetScopeType
    default_personal: bool
    client_family: str | None = None
    domains: list[SyncDomain] = Field(default_factory=list)
    created_at: str | None = None
    updated_at: str | None = None
    encryption_policy: EncryptionPolicy = DEFAULT_M1_ENCRYPTION_POLICY
    server_frontend_mutation_enabled: bool = True
    server_frontend_mutation_blockers: list[str] = Field(default_factory=list)
    notes_organization: SyncNotesOrganizationStatusResponse | None = None
    notes_link: SyncNotesOrganizationStatusResponse | None = None
    notes_attachment: SyncNotesOrganizationStatusResponse | None = None


class SyncProfileDomainStatusResponse(BaseModel):
    """Per-domain Sync v2 M1 status summary."""

    domain: SyncDomain
    last_server_cursor: int = Field(0, ge=0)
    envelope_count: int = Field(0, ge=0)
    pending_apply_count: int = Field(0, ge=0)
    pending_apply: int = Field(0, ge=0)
    failed_apply_count: int = Field(0, ge=0)
    unresolved_conflicts: int = Field(0, ge=0)
    last_apply_status: str | None = None
    last_apply_result: dict[str, Any] = Field(default_factory=dict)
    repair_status: dict[str, Any] = Field(default_factory=dict)
    server_frontend_mutation_enabled: bool = True
    server_frontend_mutation_blockers: list[str] = Field(default_factory=list)


class SyncProfileResponse(BaseModel):
    """Read-only Sync v2 M1 profile/status response."""

    protocol_version: str = "sync-v2-m1"
    min_supported_protocol_version: str = "sync-v2-m1"
    profile_bootstrapped: bool
    user_id: str
    active_dataset_id: str | None = None
    device: SyncProfileDeviceStatusResponse | None = None
    dataset: SyncProfileDatasetStatusResponse | None = None
    server_cursor: int = Field(0, ge=0)
    capabilities: SyncCapabilitiesResponse = Field(default_factory=SyncCapabilitiesResponse)
    domain_status: list[SyncProfileDomainStatusResponse] = Field(default_factory=list)
    warnings: list[dict[str, str]] = Field(default_factory=list)


class SyncProfileBootstrapRequest(BaseModel):
    """Request to bootstrap a server-connected Chatbook profile."""

    client_family: str = "chatbook"
    mode: SyncProfileBootstrapMode
    device_id: str | None = None
    device_name: str | None = None
    client_profile_id: str | None = None
    client_instance: dict[str, Any] = Field(default_factory=dict)
    requested_domains: list[SyncDomain] = Field(default_factory=lambda: list(M1_SYNC_DOMAINS))
    supported_adapter_versions: dict[str, list[StrictInt]] | None = None

    @model_validator(mode="after")
    def _normalize_supported_adapter_versions(self) -> SyncProfileBootstrapRequest:
        requested_domains = normalize_sync_v2_requested_domains(
            self.requested_domains
        )
        self.requested_domains = requested_domains
        supplied = self.supported_adapter_versions
        if supplied is None:
            supplied = self.client_instance.get("supported_adapter_versions")
        normalized = normalize_supported_adapter_versions(
            supplied,
            requested_domains=requested_domains,
        )
        self.supported_adapter_versions = normalized
        self.client_instance = {
            **self.client_instance,
            "supported_adapter_versions": normalized,
        }
        return self


class SyncProfileBootstrapResponse(SyncProfileResponse):
    """Response from explicit profile bootstrap."""

    created: bool = False


class SyncPersonalContextBootstrapRequest(BaseModel):
    """Authenticated registered-device request for canonical Personal Context."""

    device_id: str
    required_schema_version: int | None = Field(None, ge=1)
    required_quotas: dict[str, int] = Field(default_factory=dict)
    expected_purge_generation: int | None = Field(None, ge=0)

    model_config = ConfigDict(extra="forbid")


class SyncPersonalContextSchemaAttention(BaseModel):
    """Exact content-free schema bounds blocking bootstrap review."""

    kind: Literal["schema_incompatible"]
    required_schema_version: int = Field(..., ge=1)
    server_min_schema_version: int = Field(..., ge=1)
    server_max_schema_version: int = Field(..., ge=1)

    model_config = ConfigDict(extra="forbid")


class SyncPersonalContextQuotaAttention(BaseModel):
    """Exact content-free quota deficits blocking bootstrap review."""

    kind: Literal["quota_incompatible"]
    required_quotas: dict[str, int]
    available_quotas: dict[str, int]
    insufficient_quotas: list[str]

    model_config = ConfigDict(extra="forbid")


class SyncPersonalContextPurgeAttention(BaseModel):
    """Exact content-free purge generations blocking bootstrap review."""

    kind: Literal["purge_generation_mismatch"]
    expected_purge_generation: int = Field(..., ge=0)
    current_purge_generation: int = Field(..., ge=0)

    model_config = ConfigDict(extra="forbid")


class SyncPersonalContextBootstrapErrorDetail(BaseModel):
    """Stable bootstrap failure with actionable content-free review facts."""

    error_code: str
    message: str
    attention: (
        SyncPersonalContextSchemaAttention
        | SyncPersonalContextQuotaAttention
        | SyncPersonalContextPurgeAttention
        | None
    ) = Field(None, discriminator="kind")

    model_config = ConfigDict(extra="forbid")


class SyncPersonalContextBootstrapErrorResponse(BaseModel):
    """FastAPI error envelope for Personal Context bootstrap."""

    detail: SyncPersonalContextBootstrapErrorDetail

    model_config = ConfigDict(extra="forbid")


class SyncPersonalContextBootstrapResponse(BaseModel):
    """Canonical bootstrap snapshot with device-wrapped integrity key material."""

    dataset_id: str
    authority_id: str
    manifest: dict[str, Any]
    scopes: list[dict[str, Any]] = Field(default_factory=list)
    records: list[dict[str, Any]] = Field(default_factory=list)
    proposals: list[dict[str, Any]] = Field(default_factory=list)
    purge_generation: int
    schema_version: int
    quotas: dict[str, int]
    cursor: str
    integrity_key_id: str
    key_record_id: str
    wrapped_key_blob: str


class SyncPersonalContextLinkCompleteRequest(BaseModel):
    """Cursor-bound acknowledgement that one device completed reconciliation."""

    device_id: str
    dataset_id: str
    bootstrap_cursor: str

    model_config = ConfigDict(extra="forbid")


class SyncRestoreManifestDataset(BaseModel):
    """Metadata-only dataset entry in a restore manifest."""

    dataset_id: str
    scope_type: DatasetScopeType
    encryption_policy: EncryptionPolicy
    domains: list[SyncDomain] = Field(default_factory=list)
    workspace_id: str | None = None
    approximate_counts: dict[str, int] = Field(default_factory=dict)
    byte_estimates: dict[str, int] = Field(default_factory=dict)
    last_updated_at: str | None = None
    unresolved_conflicts: int = Field(0, ge=0)
    attachment_availability: dict[str, int] = Field(default_factory=dict)
    attachment_size_classes: dict[str, int] = Field(default_factory=dict)
    key_recovery_available: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class SyncRestoreManifestDevice(BaseModel):
    """Device metadata included in restore inventory."""

    device_id: str
    display_name: str | None = None
    client_type: str | None = None
    client_version: str | None = None
    last_seen_at: str | None = None
    revoked_at: str | None = None


class SyncRestoreManifestResponse(BaseModel):
    """Metadata-only restore inventory for the authenticated user."""

    datasets: list[SyncRestoreManifestDataset] = Field(default_factory=list)
    devices: list[SyncRestoreManifestDevice] = Field(default_factory=list)
    generated_at: str | None = None
    filters_applied: dict[str, Any] = Field(default_factory=dict)


class SyncRestorePreviewLocalInventoryItem(BaseModel):
    """Local object fingerprint supplied for restore conflict preview."""

    dataset_id: str | None = None
    domain: SyncDomain
    object_id: str = Field(..., min_length=1, validation_alias=AliasChoices("object_id", "entity_id"))
    adapter_version: int = Field(1, ge=1)
    object_revision: int | None = Field(None, ge=0, validation_alias=AliasChoices("object_revision", "entity_version"))
    object_hash: str | None = Field(None, validation_alias=AliasChoices("object_hash", "payload_hash"))
    deleted: bool = False
    attachment_availability: dict[str, str] = Field(default_factory=dict)

    @property
    def entity_id(self) -> str:
        return self.object_id

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class SyncRestorePreviewRequest(BaseModel):
    """Client inventory request for a Sync v2 M1 restore preview."""

    device_id: str | None = None
    dataset_ids: list[str] = Field(default_factory=list, max_length=100)
    domains: list[SyncDomain] = Field(default_factory=list, max_length=100)
    selected_object_ids: list[str] = Field(default_factory=list, max_length=10_000)
    selected_attachment_ids: list[str] = Field(default_factory=list, max_length=10_000)
    metadata_only: bool = False
    local_inventory: list[SyncRestorePreviewLocalInventoryItem] = Field(
        default_factory=list,
        max_length=10_000,
    )
    attachment_availability: dict[str, str] = Field(default_factory=dict)


class SyncRestorePreviewEnvelopeRange(BaseModel):
    """Cursor range needed to replay one domain into a restoring client."""

    dataset_id: str
    domain: SyncDomain
    from_cursor: int = Field(..., ge=0)
    to_cursor: int = Field(..., ge=0)
    envelope_count: int = Field(..., ge=0)


class SyncRestorePreviewDataset(BaseModel):
    """Dataset-level restore preview summary."""

    dataset_id: str
    domains: list[SyncDomain] = Field(default_factory=list)
    approximate_counts: dict[str, int] = Field(default_factory=dict)
    byte_estimates: dict[str, int] = Field(default_factory=dict)
    latest_cursor: int | None = Field(None, ge=0)
    latest_cursors: dict[str, int] = Field(default_factory=dict)
    envelope_ranges: list[SyncRestorePreviewEnvelopeRange] = Field(default_factory=list)
    total_count: int = Field(0, ge=0)
    encryption_policy: EncryptionPolicy = DEFAULT_M1_ENCRYPTION_POLICY
    key_recovery_available: bool = False


class SyncRestorePreviewObject(BaseModel):
    """One safe restore candidate or tombstone action."""

    dataset_id: str
    domain: SyncDomain
    object_id: str
    action: SyncRestorePreviewAction
    server_revision: int | None = Field(None, ge=0)
    server_hash: str | None = None
    server_cursor: int | None = Field(None, ge=0)
    local_revision: int | None = Field(None, ge=0)
    local_hash: str | None = None
    local_deleted: bool | None = None
    deleted: bool = False
    parent_id: str | None = None


class SyncRestorePreviewObjectConflict(BaseModel):
    """Whole-object or stable-ID restore conflict surfaced before local apply."""

    dataset_id: str
    domain: SyncDomain
    object_id: str
    conflict_type: str
    server_revision: int | None = Field(None, ge=0)
    server_hash: str | None = None
    server_cursor: int | None = Field(None, ge=0)
    server_deleted: bool = False
    local_revision: int | None = Field(None, ge=0)
    local_hash: str | None = None
    local_deleted: bool = False
    message: str | None = None


class SyncRestoreOrderedAction(BaseModel):
    """One content-free action in canonical restore execution order."""

    plan_index: int = Field(..., ge=0)
    action: SyncRestoreOrderedActionKind
    dataset_id: str
    domain: SyncDomain
    object_id: str
    operation: SyncOperation
    server_cursor: int = Field(..., ge=0)
    adapter_version: int = Field(..., ge=1)
    mutation_group_id: str | None = None
    mutation_step: int | None = Field(None, ge=0)
    mutation_step_count: int | None = Field(None, ge=1)
    code: str | None = None

    model_config = ConfigDict(extra="forbid", frozen=True)


class SyncRestorePreviewAttachmentRef(BaseModel):
    """Attachment metadata surfaced in a restore preview."""

    dataset_id: str
    attachment_id: str
    object_id: str
    parent_domain: SyncDomain
    parent_object_id: str
    content_type: str
    size_bytes: int = Field(..., ge=0)
    payload_hash: str
    availability: str
    server_cursor: int = Field(..., ge=0)
    adapter_version: int = Field(..., ge=1)


class SyncRestorePreviewWarning(BaseModel):
    """Restore preview warning with stable client-actionable code."""

    code: str
    message: str
    dataset_id: str | None = None
    attachment_id: str | None = None
    object_id: str | None = None
    payload_hash: str | None = None


class SyncRestorePreviewResponse(BaseModel):
    """Non-mutating Sync v2 M1 restore preview response."""

    datasets: list[SyncRestorePreviewDataset] = Field(default_factory=list)
    ordered_actions: list[SyncRestoreOrderedAction] = Field(default_factory=list)
    safe_applies: list[SyncRestorePreviewObject] = Field(default_factory=list)
    object_conflicts: list[SyncRestorePreviewObjectConflict] = Field(default_factory=list)
    tombstones: list[SyncRestorePreviewObject] = Field(default_factory=list)
    attachment_refs: list[SyncRestorePreviewAttachmentRef] = Field(default_factory=list)
    missing_blobs: list[SyncRestorePreviewAttachmentRef] = Field(default_factory=list)
    envelope_ranges: list[SyncRestorePreviewEnvelopeRange] = Field(default_factory=list)
    total_counts: dict[str, int] = Field(default_factory=dict)
    encryption: dict[str, Any] = Field(default_factory=dict)
    key_status: dict[str, dict[str, bool]] = Field(default_factory=dict)
    warnings: list[SyncRestorePreviewWarning] = Field(default_factory=list)
    generated_at: str | None = None
    filters_applied: dict[str, Any] = Field(default_factory=dict)
    restore_status: SyncRestoreCompletenessStatus | None = None
    domain_details: list[SyncRestoreDomainCompleteness] = Field(default_factory=list)
    blob_details: list[SyncRestoreBlobCompleteness] = Field(default_factory=list)
    metadata_only_allowed: bool = True


class SyncBlobUploadCreateRequest(BaseModel):
    """Request to create or resume a Sync v2 M2 blob upload session."""

    dataset_id: str = Field(..., min_length=1)
    device_id: str | None = None
    domain: SyncDomain
    object_id: str = Field(..., min_length=1, validation_alias=AliasChoices("object_id", "entity_id"))
    attachment_id: str = Field(..., min_length=1)
    content_type: str = Field(..., min_length=1)
    size_bytes: int = Field(..., ge=1)
    payload_hash: str
    chunk_size: int = Field(..., ge=1)
    chunk_count: int = Field(..., ge=1)
    idempotency_key: str | None = None
    encryption_policy: EncryptionPolicy = DEFAULT_M1_ENCRYPTION_POLICY
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("payload_hash")
    @classmethod
    def _validate_payload_hash(cls, value: Any) -> str:
        return _validate_sha256_hash(value)

    @model_validator(mode="after")
    def _validate_chunk_shape(self) -> SyncBlobUploadCreateRequest:
        if self.chunk_size * self.chunk_count < self.size_bytes:
            raise ValueError("chunk_count and chunk_size must cover size_bytes")
        raw_intent = self.metadata.get("notes_attachment_intent")
        if self.domain != "attachment.ref":
            if raw_intent is not None:
                raise ValueError(
                    "notes_attachment_intent is reserved for attachment.ref uploads"
                )
            return self
        if not isinstance(raw_intent, dict):
            raise ValueError("attachment.ref uploads require notes_attachment_intent")
        intent_type = raw_intent.get("intent")
        intent_model = (
            SyncNotesAttachmentCreateIntent
            if intent_type == "create"
            else SyncNotesAttachmentReplaceIntent
            if intent_type == "replace"
            else None
        )
        if intent_model is None:
            raise ValueError("notes_attachment_intent has an unsupported intent")
        intent = intent_model.model_validate(raw_intent)
        if (
            intent.attachment_id != self.attachment_id
            or intent.attachment_id != self.object_id
        ):
            raise ValueError(
                "notes_attachment_intent attachment_id must match the upload identity"
            )
        self.metadata = {
            **self.metadata,
            "notes_attachment_intent": intent.model_dump(mode="json"),
        }
        return self

    @property
    def entity_id(self) -> str:
        return self.object_id

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class _SyncNotesAttachmentIntent(BaseModel):
    """Shared immutable identity for a Notes attachment upload intent."""

    note_id: str
    attachment_id: str

    @field_validator("note_id", "attachment_id")
    @classmethod
    def _validate_uuid4(cls, value: Any, info: Any) -> str:
        """Validate one canonical Notes attachment UUIDv4 field."""

        from uuid import RFC_4122, UUID

        if not isinstance(value, str):
            raise ValueError(f"{info.field_name} must be a canonical lowercase UUIDv4")
        try:
            parsed = UUID(value)
        except ValueError as exc:
            raise ValueError(
                f"{info.field_name} must be a canonical lowercase UUIDv4"
            ) from exc
        if parsed.version != 4 or parsed.variant != RFC_4122 or str(parsed) != value:
            raise ValueError(f"{info.field_name} must be a canonical lowercase UUIDv4")
        return value

    model_config = ConfigDict(extra="forbid")


class SyncNotesAttachmentCreateIntent(_SyncNotesAttachmentIntent):
    """Immutable resumable-upload intent for a new Notes attachment."""

    intent: Literal["create"] = "create"
    file_name: str

    @field_validator("file_name")
    @classmethod
    def _canonicalize_file_name(cls, value: Any) -> str:
        """Canonicalize the requested create-intent filename."""

        try:
            return canonicalize_note_attachment_file_name(value)[0]
        except NoteAttachmentPolicyError as exc:
            raise ValueError(str(exc)) from exc


class SyncNotesAttachmentReplaceIntent(_SyncNotesAttachmentIntent):
    """Immutable resumable-upload intent for attachment content replacement."""

    intent: Literal["replace"] = "replace"
    base_server_cursor: StrictInt = Field(..., ge=1)
    base_object_revision: StrictInt = Field(..., ge=1)
    base_object_hash: str

    @field_validator("base_object_hash")
    @classmethod
    def _validate_object_hash(cls, value: Any) -> str:
        """Validate the optimistic replacement base digest."""

        import re

        if (
            not isinstance(value, str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None
        ):
            raise ValueError("base_object_hash must be a lowercase SHA-256 digest")
        return value


class SyncBlobUploadSessionResponse(BaseModel):
    """Current state for a resumable Sync v2 M2 blob upload session."""

    upload_id: str
    dataset_id: str
    attachment_id: str
    status: SyncBlobUploadStatus
    chunk_size: int = Field(..., ge=1)
    chunk_count: int = Field(..., ge=1)
    uploaded_chunks: list[int] = Field(default_factory=list)
    missing_chunks: list[int] = Field(default_factory=list)
    expires_at: str | None = None
    blob_id: str | None = None
    quota: dict[str, Any] = Field(default_factory=dict)


class SyncBlobChunkUploadResponse(BaseModel):
    """Result after accepting one Sync v2 M2 blob chunk."""

    upload_id: str
    chunk_index: int = Field(..., ge=0)
    accepted: bool = True
    size_bytes: int = Field(..., ge=0)
    chunk_hash: str
    missing_chunks: list[int] = Field(default_factory=list)

    @field_validator("chunk_hash")
    @classmethod
    def _validate_chunk_hash(cls, value: Any) -> str:
        return _validate_sha256_hash(value)


class SyncBlobUploadCompleteResponse(BaseModel):
    """Result after verifying and committing a Sync v2 M2 blob upload."""

    upload_id: str
    dataset_id: str
    attachment_id: str
    blob_id: str
    status: SyncBlobAvailabilityStatus
    stored: bool
    deduplicated: bool = False
    size_bytes: int = Field(..., ge=0)
    payload_hash: str
    download_url: str | None = None
    quota: dict[str, Any] = Field(default_factory=dict)

    @field_validator("payload_hash")
    @classmethod
    def _validate_payload_hash(cls, value: Any) -> str:
        return _validate_sha256_hash(value)


class SyncBlobDownloadChunk(BaseModel):
    """One downloadable chunk entry in a Sync v2 M2 blob manifest."""

    chunk_index: int = Field(..., ge=0)
    offset_bytes: int = Field(..., ge=0)
    size_bytes: int = Field(..., ge=0)
    chunk_hash: str
    download_url: str | None = None

    @field_validator("chunk_hash")
    @classmethod
    def _validate_chunk_hash(cls, value: Any) -> str:
        return _validate_sha256_hash(value)


class SyncBlobDownloadManifestResponse(BaseModel):
    """Manifest for resumable Sync v2 M2 blob download."""

    dataset_id: str
    attachment_id: str
    blob_id: str | None = None
    availability: SyncBlobAvailabilityStatus
    content_type: str
    size_bytes: int = Field(..., ge=0)
    payload_hash: str
    chunks: list[SyncBlobDownloadChunk] = Field(default_factory=list)
    expires_at: str | None = None

    @field_validator("payload_hash")
    @classmethod
    def _validate_payload_hash(cls, value: Any) -> str:
        return _validate_sha256_hash(value)


class SyncRestoreDomainCompleteness(BaseModel):
    """Per-domain restore completeness counters for Sync v2 M2."""

    domain: SyncDomain
    status: SyncRestoreCompletenessStatus
    selected_count: int = Field(0, ge=0)
    safe_apply_count: int = Field(0, ge=0)
    conflict_count: int = Field(0, ge=0)
    tombstone_count: int = Field(0, ge=0)
    required_blob_count: int = Field(0, ge=0)
    available_blob_count: int = Field(0, ge=0)
    missing_blob_count: int = Field(0, ge=0)
    verified_blob_count: int = Field(0, ge=0)
    warnings: list[SyncRestorePreviewWarning] = Field(default_factory=list)


class SyncRestoreBlobCompleteness(BaseModel):
    """Per-blob restore completeness detail for Sync v2 M2."""

    attachment_id: str
    payload_hash: str
    size_bytes: int = Field(..., ge=0)
    content_type: str
    parent_domain: SyncDomain
    parent_object_id: str
    server_availability: SyncBlobAvailabilityStatus
    download_status: str | None = None
    required_for_restore: bool = True
    warnings: list[SyncRestorePreviewWarning] = Field(default_factory=list)

    @field_validator("payload_hash")
    @classmethod
    def _validate_payload_hash(cls, value: Any) -> str:
        return _validate_sha256_hash(value)


class SyncRestoreCompletenessResponse(BaseModel):
    """Profile-level restore completeness summary for Sync v2 M2."""

    restore_status: SyncRestoreCompletenessStatus
    domain_details: list[SyncRestoreDomainCompleteness] = Field(default_factory=list)
    blob_details: list[SyncRestoreBlobCompleteness] = Field(default_factory=list)
    metadata_only_allowed: bool = True
    generated_at: str | None = None


class SyncV2Envelope(BaseModel):
    """Protocol unit exchanged by Sync v2 clients and the server."""

    envelope_id: str | None = None
    client_envelope_id: str
    dataset_id: str
    device_id: str | None = None
    client_profile_id: str | None = None
    client_sequence: int | None = Field(None, ge=0)
    base_server_cursor: int | None = Field(None, ge=0)
    base_object_revision: int | None = Field(None, ge=0)
    base_object_hash: str | None = None
    server_cursor: int | None = Field(None, ge=0)
    domain: SyncDomain
    operation: SyncOperation
    object_id: str = Field(..., min_length=1)
    parent_id: str | None = None
    schema_version: int = Field(1, ge=1)
    adapter_version: int = Field(1, ge=1)
    payload: dict[str, Any] = Field(default_factory=dict)
    payload_hash: str = Field(..., min_length=1)
    object_revision: int | None = Field(None, ge=0)
    created_at_client: str | None = None
    received_at_server: str | None = None
    deleted: bool = False
    encryption_metadata: dict[str, Any] = Field(default_factory=lambda: {"policy": DEFAULT_M1_ENCRYPTION_POLICY})
    payload_size_bytes: int | None = Field(None, ge=0)
    payload_ciphertext: str | None = None
    dependencies: list[dict[str, Any]] = Field(default_factory=list)
    routing_metadata: dict[str, Any] = Field(default_factory=dict)
    stable_key: str | None = None
    status: str | None = None
    apply_status: SyncApplyStatus | None = None
    apply_error_code: str | None = None
    apply_error_message: str | None = None
    applied_at: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_aliases(cls, data: Any) -> Any:
        return _with_transition_aliases(data)

    @field_validator("payload", "routing_metadata", "encryption_metadata", mode="before")
    @classmethod
    def _default_object_maps(cls, value: Any) -> dict[str, Any]:
        return _normalize_object_map(value)

    @field_validator("dependencies", mode="before")
    @classmethod
    def _default_dependencies(cls, value: Any) -> list[dict[str, Any]]:
        if value is None:
            return []
        return value

    @model_validator(mode="after")
    def _validate_m1_contract(self) -> SyncV2Envelope:
        allowed_operations = SYNC_V2_SUPPORTED_OPERATIONS[self.domain]
        if self.operation not in allowed_operations:
            raise ValueError(f"{self.operation} is not supported for {self.domain}")

        policy = self.encryption_metadata.get("policy", DEFAULT_M1_ENCRYPTION_POLICY)
        if policy != DEFAULT_M1_ENCRYPTION_POLICY:
            raise ValueError(f"{policy} is not accepted by the Sync v2 M1 envelope contract")

        base_values = (
            self.base_server_cursor,
            self.base_object_revision,
            self.base_object_hash,
        )
        has_any_base = any(value is not None for value in base_values)
        has_all_base = all(value is not None for value in base_values)
        if not self.payload_hash.strip():
            raise ValueError("payload_hash is required by the Sync v2 M1 envelope contract")
        if has_any_base and not has_all_base:
            raise ValueError("base metadata must be supplied as a complete set")
        if self.domain in _WHOLE_OBJECT_DOMAINS:
            if self.operation == "tombstone" and not has_all_base:
                raise ValueError(f"{self.domain} tombstone envelopes require base metadata")
            if (
                self.operation == "upsert"
                and self.object_revision is not None
                and self.object_revision > 1
                and not has_all_base
            ):
                raise ValueError(f"{self.domain} update envelopes require base metadata")

        if self.domain == "chat.message" and (
            self.operation == "append" and (not self.object_id.strip() or not self.payload_hash.strip())
        ):
            raise ValueError("chat.message append envelopes require object_id and payload_hash")

        if self.domain == "notes.note" and self.operation == "upsert":
            validate_notes_note_upsert_payload(self.payload)

        if self.domain == "attachment.ref":
            if self.adapter_version == 2:
                from tldw_Server_API.app.core.Sync.v2.attachment_refs_v2 import (
                    attachment_ref_v2_object_hash,
                    parse_attachment_ref_v2_payload,
                    validate_attachment_ref_v2_object_id,
                    validate_attachment_ref_v2_routing_metadata,
                )

                if self.schema_version != 2:
                    raise ValueError(
                        "attachment.ref adapter version 2 requires schema version 2"
                    )
                attachment = parse_attachment_ref_v2_payload(
                    self.operation,
                    self.payload,
                )
                validate_attachment_ref_v2_object_id(self.object_id)
                validate_attachment_ref_v2_routing_metadata(
                    self.operation,
                    self.routing_metadata,
                )
                if self.object_id != str(attachment.attachment_id):
                    raise ValueError(
                        "attachment.ref object_id must match payload attachment_id"
                    )
                if self.payload_hash != attachment_ref_v2_object_hash(
                    self.operation,
                    attachment,
                    object_revision=self.object_revision,
                ):
                    raise ValueError(
                        "attachment.ref v2 payload_hash must match the canonical object hash"
                    )
            else:
                missing = _ATTACHMENT_REF_REQUIRED_PAYLOAD_KEYS.difference(self.payload)
                if missing:
                    raise ValueError(
                        "attachment.ref envelopes require payload metadata fields: "
                        + ", ".join(sorted(missing))
                    )
                attachment_id = self.payload.get("attachment_id")
                if (
                    isinstance(attachment_id, str)
                    and self.object_id != attachment_id.strip()
                ):
                    raise ValueError(
                        "attachment.ref object_id must match payload attachment_id"
                    )
        return self

    @property
    def server_sequence(self) -> int | None:
        return self.server_cursor

    @property
    def entity_id(self) -> str:
        return self.object_id

    @property
    def payload_clear(self) -> dict[str, Any]:
        return self.payload

    @property
    def client_timestamp(self) -> str | None:
        return self.created_at_client

    @property
    def server_timestamp(self) -> str | None:
        return self.received_at_server

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class SyncV2EnvelopeResponse(SyncV2Envelope):
    """Server envelope fields returned by pull but never accepted from push."""

    mutation_group_id: str | None = None
    mutation_step: int | None = Field(None, ge=0)
    mutation_step_count: int | None = Field(None, ge=1)
    mutation_plan_hash: str | None = None


class SyncPushOptions(BaseModel):
    """Client push behavior flags from the locked Sync v2 M1 contract."""

    stop_on_conflict: bool = False

    model_config = ConfigDict(extra="ignore")


class SyncPushRequest(BaseModel):
    """Batch of client-originated envelopes pushed to the server."""

    dataset_id: str
    device_id: str = Field(..., min_length=1)
    client_profile_id: str | None = None
    base_server_cursor: int | None = Field(None, ge=0)
    envelopes: list[SyncV2Envelope] = Field(
        default_factory=list,
        max_length=SYNC_V2_MAX_PUSH_ENVELOPES,
    )
    idempotency_key: str | None = None
    last_known_cursor: str | None = None
    options: SyncPushOptions = Field(default_factory=SyncPushOptions)


class SyncPushAcceptedEnvelope(BaseModel):
    """Accepted push outcome for one client envelope."""

    client_envelope_id: str
    server_cursor: int = Field(..., ge=0, validation_alias=AliasChoices("server_cursor", "server_sequence"))
    domain: SyncDomain | None = None
    object_id: str | None = Field(None, validation_alias=AliasChoices("object_id", "entity_id"))
    object_revision: int | None = Field(None, ge=0)
    apply_status: SyncApplyStatus | None = None
    apply_error_code: str | None = None
    apply_error_message: str | None = None

    @property
    def server_sequence(self) -> int:
        return self.server_cursor

    @property
    def entity_id(self) -> str | None:
        return self.object_id

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class SyncPushRejectedEnvelope(BaseModel):
    """Rejected push outcome for one client envelope."""

    client_envelope_id: str
    error_code: str
    message: str
    retryable: bool = False


class SyncPushConflictEnvelope(BaseModel):
    """Conflict push outcome for one client envelope."""

    conflict_id: str
    client_envelope_id: str
    domain: SyncDomain
    object_id: str = Field(..., validation_alias=AliasChoices("object_id", "entity_id"))
    server_cursor: int | None = Field(
        None,
        ge=0,
        validation_alias=AliasChoices("server_cursor", "server_sequence"),
    )
    message: str | None = None

    @property
    def entity_id(self) -> str:
        return self.object_id

    @property
    def server_sequence(self) -> int | None:
        return self.server_cursor

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class SyncPushResponse(BaseModel):
    """Per-envelope outcomes returned after a push batch."""

    dataset_id: str
    accepted: list[SyncPushAcceptedEnvelope] = Field(default_factory=list)
    rejected: list[SyncPushRejectedEnvelope] = Field(default_factory=list)
    conflicts: list[SyncPushConflictEnvelope] = Field(default_factory=list)
    next_cursor: str | None = None


class SyncPullResponse(BaseModel):
    """Stable cursor-ordered envelopes returned by pull."""

    dataset_id: str
    envelopes: list[SyncV2EnvelopeResponse] = Field(default_factory=list)
    next_cursor: str | None = None
    has_more: bool = False


class SyncRepairRequest(BaseModel):
    """Request to replay accepted envelopes into server-side projections."""

    dataset_id: str = Field(..., min_length=1)
    device_id: str | None = None
    domains: list[SyncDomain] = Field(default_factory=list)
    since_cursor: int = Field(0, ge=0)
    failed_only: bool = False
    limit: int | None = Field(None, ge=1)


class SyncRepairEnvelopeError(BaseModel):
    """One failed envelope from a replay/repair operation."""

    server_cursor: int | None = Field(None, ge=0)
    client_envelope_id: str
    domain: SyncDomain
    object_id: str
    error_code: str | None = None
    message: str | None = None


class SyncRepairDomainResult(BaseModel):
    """Per-domain replay/repair counters."""

    domain: SyncDomain
    scanned_count: int = Field(0, ge=0)
    attempted_count: int = Field(0, ge=0)
    applied_count: int = Field(0, ge=0)
    failed_count: int = Field(0, ge=0)
    conflict_count: int = Field(0, ge=0)
    skipped_count: int = Field(0, ge=0)
    last_cursor: int = Field(0, ge=0)
    errors: list[SyncRepairEnvelopeError] = Field(default_factory=list)


class SyncRepairResponse(BaseModel):
    """Replay/repair result for a dataset projection repair run."""

    dataset_id: str
    domains: list[SyncDomain] = Field(default_factory=list)
    from_cursor: int = Field(0, ge=0)
    to_cursor: int = Field(0, ge=0)
    scanned_count: int = Field(0, ge=0)
    attempted_count: int = Field(0, ge=0)
    applied_count: int = Field(0, ge=0)
    failed_count: int = Field(0, ge=0)
    conflict_count: int = Field(0, ge=0)
    skipped_count: int = Field(0, ge=0)
    domain_results: list[SyncRepairDomainResult] = Field(default_factory=list)
    repair_status: dict[str, Any] = Field(default_factory=dict)


class SyncAttachmentUploadRequest(BaseModel):
    """Request metadata for uploading a future encrypted sync attachment."""

    dataset_id: str
    domain: SyncDomain
    object_id: str = Field(..., validation_alias=AliasChoices("object_id", "entity_id"))
    attachment_id: str
    content_type: str
    size_bytes: int = Field(..., ge=0)
    payload_ciphertext: str
    payload_hash: str
    encryption_policy: EncryptionPolicy = DEFAULT_M1_ENCRYPTION_POLICY
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def entity_id(self) -> str:
        return self.object_id

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class SyncAttachmentUploadResponse(BaseModel):
    """Response after storing or deduplicating a sync attachment."""

    attachment_id: str
    dataset_id: str
    stored: bool
    size_bytes: int = Field(..., ge=0)
    payload_hash: str
    download_url: str | None = None
    expires_at: str | None = None


class SyncConflictRecord(BaseModel):
    """Durable conflict metadata visible to sync clients."""

    conflict_id: str
    dataset_id: str
    domain: SyncDomain
    object_id: str = Field(..., validation_alias=AliasChoices("object_id", "entity_id"))
    conflict_type: str
    status: ConflictStatus = "unresolved"
    base_envelope_id: str | None = None
    local_envelope_id: str | None = None
    remote_envelope_id: str | None = None
    server_cursor: int | None = Field(
        None,
        ge=0,
        validation_alias=AliasChoices("server_cursor", "server_sequence"),
    )
    metadata: dict[str, Any] = Field(default_factory=dict)
    resolved_by_envelope_id: str | None = None
    created_at: str | None = None
    resolved_at: str | None = None

    @property
    def entity_id(self) -> str:
        return self.object_id

    @property
    def server_sequence(self) -> int | None:
        return self.server_cursor

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class SyncConflictResolution(BaseModel):
    """One explicit M1 conflict-resolution decision."""

    conflict_id: str = Field(..., min_length=1)
    action: ConflictResolutionAction
    resolution_envelope: SyncV2Envelope | None = None

    @model_validator(mode="after")
    def _validate_resolution_envelope(self) -> SyncConflictResolution:
        if self.action == "duplicate_rename" and self.resolution_envelope is None:
            raise ValueError("duplicate_rename requires a resolution_envelope")
        if self.action == "skip" and self.resolution_envelope is not None:
            raise ValueError("skip must not include a resolution_envelope")
        return self

    model_config = ConfigDict(extra="forbid")


class SyncConflictResolveRequest(BaseModel):
    """Request to resolve one or more M1 conflicts."""

    dataset_id: str = Field(..., min_length=1)
    device_id: str = Field(..., min_length=1)
    resolutions: list[SyncConflictResolution] = Field(default_factory=list, min_length=1)

    model_config = ConfigDict(extra="forbid")


class SyncConflictResolveResolvedItem(BaseModel):
    """Successful M1 conflict-resolution outcome for one conflict."""

    conflict_id: str
    action: ConflictResolutionAction
    status: ConflictStatus
    envelope_id: str | None = Field(
        None,
        validation_alias=AliasChoices("envelope_id", "resolved_by_envelope_id"),
    )
    server_cursor: int | None = Field(
        None,
        ge=0,
        validation_alias=AliasChoices("server_cursor", "server_sequence"),
    )

    @property
    def server_sequence(self) -> int | None:
        return self.server_cursor

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class SyncConflictResolveRejectedItem(BaseModel):
    """Rejected M1 conflict-resolution outcome for one conflict."""

    conflict_id: str
    action: ConflictResolutionAction
    error_code: str
    message: str
    retryable: bool = False


class SyncConflictResolveResponse(BaseModel):
    """Batch M1 conflict-resolution response."""

    dataset_id: str
    server_cursor: int | None = Field(None, ge=0)
    resolved: list[SyncConflictResolveResolvedItem] = Field(default_factory=list)
    rejected: list[SyncConflictResolveRejectedItem] = Field(default_factory=list)


class SyncKeyRecoveryBundleRequest(BaseModel):
    """Client-generated encrypted key recovery material for a dataset."""

    dataset_id: str
    device_id: str | None = None
    key_purpose: str = "dataset_recovery"
    wrapped_key_blob: str
    kdf_metadata: dict[str, Any] = Field(default_factory=dict)
    recovery_hint: str | None = None
    rotation_of_key_record_id: str | None = None
    encryption_policy: EncryptionPolicy = DEFAULT_M1_ENCRYPTION_POLICY
    key_epoch: int = Field(1, ge=1)
    active_from_server_sequence: int | None = Field(None, ge=0)
    superseded_at: str | None = None
    wrapped_for: SyncKeyWrappedFor = "recovery"
    rewrap_status: SyncKeyRewrapStatus = "not_required"


class SyncKeyRecoveryBundleRecord(BaseModel):
    """Stored encrypted recovery material returned to an authenticated client."""

    key_record_id: str
    dataset_id: str
    device_id: str | None = None
    key_purpose: str
    wrapped_key_blob: str
    kdf_metadata: dict[str, Any] = Field(default_factory=dict)
    recovery_hint: str | None = None
    rotation_of_key_record_id: str | None = None
    created_at: str | None = None
    revoked_at: str | None = None
    encryption_policy: EncryptionPolicy = DEFAULT_M1_ENCRYPTION_POLICY
    key_epoch: int = Field(1, ge=1)
    active_from_server_sequence: int | None = Field(None, ge=0)
    superseded_at: str | None = None
    wrapped_for: SyncKeyWrappedFor = "recovery"
    rewrap_status: SyncKeyRewrapStatus = "not_required"


class SyncKeyRotationPreviewRequest(BaseModel):
    """Request a redacted key rotation preview without mutating key state."""

    dataset_id: str
    target_encryption_policy: EncryptionPolicy = DEFAULT_M1_ENCRYPTION_POLICY
    source_key_record_ids: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")

    @field_validator("source_key_record_ids", mode="after")
    @classmethod
    def _normalize_source_key_record_ids(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(str(record_id).strip() for record_id in value if str(record_id).strip()))


class SyncKeyRotationCommitRequest(SyncKeyRotationPreviewRequest):
    """Commit a new key epoch with client-supplied wrapped key material."""

    rotation_id: str
    wrapped_key_blob: str
    kdf_metadata: dict[str, Any] = Field(default_factory=dict)
    recovery_hint: str | None = None
    wrapped_for: SyncKeyWrappedFor = "recovery"
    rewrap_status: SyncKeyRewrapStatus = "complete"

    @field_validator("rotation_id")
    @classmethod
    def _rotation_id_required(cls, value: str) -> str:
        cleaned = str(value or "").strip()
        if not cleaned:
            raise ValueError("rotation_id is required")
        return cleaned


class SyncKeyRotationKeyRecord(BaseModel):
    """Redacted key-record metadata returned by key rotation flows."""

    key_record_id: str
    key_epoch: int = Field(..., ge=1)
    encryption_policy: EncryptionPolicy
    wrapped_for: SyncKeyWrappedFor
    rewrap_status: SyncKeyRewrapStatus
    device_id: str | None = None
    key_purpose: str = "dataset_recovery"
    active_from_server_sequence: int | None = Field(None, ge=0)
    superseded_at: str | None = None
    revoked_at: str | None = None
    rotation_of_key_record_id: str | None = None


class SyncKeyRotationEnvelopeRange(BaseModel):
    """Accepted envelope range retained under old key material."""

    from_server_sequence: int | None = Field(None, ge=1)
    through_server_sequence: int | None = Field(None, ge=0)
    envelope_count: int = Field(0, ge=0)


class SyncKeyRotationResponse(BaseModel):
    """Redacted key rotation preview or commit response."""

    dataset_id: str
    target_encryption_policy: EncryptionPolicy
    next_key_epoch: int = Field(..., ge=1)
    active_from_server_sequence: int = Field(..., ge=1)
    can_commit: bool
    committed: bool = False
    retained_envelope_range: SyncKeyRotationEnvelopeRange
    affected_key_records: list[SyncKeyRotationKeyRecord] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    device_ids: list[str] = Field(default_factory=list)
    recovery_target_count: int = Field(0, ge=0)
    rotation_id: str | None = None
    new_key_record: SyncKeyRotationKeyRecord | None = None


class SyncKeyRecoveryBundleListResponse(BaseModel):
    """Recovery bundle records available for a dataset."""

    dataset_id: str
    key_records: list[SyncKeyRecoveryBundleRecord] = Field(default_factory=list)


__all__ = [
    "ConflictResolutionAction",
    "ConflictStatus",
    "DEFAULT_M1_ENCRYPTION_POLICY",
    "DatasetScopeType",
    "EncryptionPolicy",
    "M1_SYNC_DOMAINS",
    "M1_SYNC_OPERATIONS",
    "MEDIA_SYNC_DOMAINS",
    "MEDIA_SYNC_OPERATIONS",
    "PERSONAL_CONTEXT_SYNC_DOMAINS",
    "PERSONAL_CONTEXT_SYNC_OPERATIONS",
    "PersonalContextSyncCapabilitiesResponse",
    "STRICT_ENCRYPTION_POLICIES",
    "SOURCE_CACHE_SYNC_DOMAINS",
    "SOURCE_CACHE_SYNC_OPERATIONS",
    "SYNC_V2_ENCRYPTION_POLICIES",
    "SYNC_V2_SUPPORTED_DOMAINS",
    "SYNC_V2_SUPPORTED_OPERATIONS",
    "SYNC_V2_MAX_PUSH_ENVELOPES",
    "SyncAttachmentUploadRequest",
    "SyncAttachmentUploadResponse",
    "SyncBlobAvailabilityStatus",
    "SyncBlobChunkUploadResponse",
    "SyncBlobDownloadChunk",
    "SyncBlobDownloadManifestResponse",
    "SyncBlobUploadCompleteResponse",
    "SyncBlobUploadCreateRequest",
    "SyncBlobUploadSessionResponse",
    "SyncBlobUploadStatus",
    "SyncNotesAttachmentCreateIntent",
    "SyncNotesAttachmentReplaceIntent",
    "SyncCapabilitiesResponse",
    "SyncConflictRecord",
    "SyncConflictResolution",
    "SyncConflictResolveRequest",
    "SyncConflictResolveResolvedItem",
    "SyncConflictResolveRejectedItem",
    "SyncConflictResolveResponse",
    "SyncDatasetEnrollRequest",
    "SyncDatasetEnrollResponse",
    "SyncDiagnosticsBlobHealthResponse",
    "SyncDiagnosticsDeviceDomainLagResponse",
    "SyncDiagnosticsDeviceResponse",
    "SyncDiagnosticsDomainResponse",
    "SyncDiagnosticsKeySummaryResponse",
    "SyncDiagnosticsResponse",
    "SyncDiagnosticsRetentionSummaryResponse",
    "SyncDeviceRegisterRequest",
    "SyncDeviceRegisterResponse",
    "SyncDomain",
    "SyncEncryptionPolicyMetadata",
    "SyncKeyRewrapStatus",
    "SyncKeyRecoveryBundleListResponse",
    "SyncKeyRecoveryBundleRequest",
    "SyncKeyRecoveryBundleRecord",
    "SyncKeyRotationCommitRequest",
    "SyncKeyRotationEnvelopeRange",
    "SyncKeyRotationKeyRecord",
    "SyncKeyRotationPreviewRequest",
    "SyncKeyRotationResponse",
    "SyncKeyWrappedFor",
    "SyncOperation",
    "SyncApplyStatus",
    "SyncProfileBootstrapMode",
    "SyncProfileBootstrapRequest",
    "SyncProfileBootstrapResponse",
    "SyncPersonalContextBootstrapRequest",
    "SyncPersonalContextBootstrapResponse",
    "SyncPersonalContextBootstrapErrorResponse",
    "SyncPersonalContextBootstrapErrorDetail",
    "SyncPersonalContextSchemaAttention",
    "SyncPersonalContextQuotaAttention",
    "SyncPersonalContextPurgeAttention",
    "SyncPersonalContextLinkCompleteRequest",
    "SyncNotesAttachmentBootstrapDiagnosticsResponse",
    "SyncNotesAttachmentCleanupSampleResponse",
    "SyncNotesOrganizationStatusResponse",
    "SyncProfileDatasetStatusResponse",
    "SyncProfileDeviceStatusResponse",
    "SyncProfileDomainStatusResponse",
    "SyncProfileResponse",
    "SyncPullResponse",
    "SyncPushAcceptedEnvelope",
    "SyncPushConflictEnvelope",
    "SyncPushOptions",
    "SyncPushRejectedEnvelope",
    "SyncPushRequest",
    "SyncPushResponse",
    "SyncRepairDomainResult",
    "SyncRepairEnvelopeError",
    "SyncRepairRequest",
    "SyncRepairResponse",
    "SyncRetentionCandidateResponse",
    "SyncRetentionCandidateType",
    "SyncRetentionCompactRequest",
    "SyncRetentionCompactResponse",
    "SyncRetentionDryRunRequest",
    "SyncRetentionDryRunResponse",
    "SyncRestoreManifestDevice",
    "SyncRestoreManifestDataset",
    "SyncRestoreManifestResponse",
    "SyncRestoreOrderedAction",
    "SyncRestorePreviewAttachmentRef",
    "SyncRestorePreviewDataset",
    "SyncRestorePreviewRequest",
    "SyncRestorePreviewResponse",
    "SyncRestorePreviewWarning",
    "SyncRestoreBlobCompleteness",
    "SyncRestoreCompletenessResponse",
    "SyncRestoreCompletenessStatus",
    "SyncRestoreDomainCompleteness",
    "SyncV2Envelope",
    "SyncV2EnvelopeResponse",
    "WORKSPACE_SYNC_DOMAINS",
    "WORKSPACE_SYNC_OPERATIONS",
]
