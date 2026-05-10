from __future__ import annotations

"""Pydantic schemas for the Sync v2 protocol API."""

from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator


SyncDomain = Literal["notes", "chat", "workspaces", "source_cache", "media"]
SyncOperation = Literal["upsert", "delete", "link", "unlink", "resolve_conflict"]
DatasetScopeType = Literal["personal", "workspace"]
EncryptionPolicy = Literal["client_private_v1", "server_trusted", "shared_workspace_v1"]
ConflictStatus = Literal["unresolved", "resolved", "dismissed"]
ConflictResolutionAction = Literal["accept_local", "accept_remote", "merge", "dismiss"]

V1_SYNC_DOMAINS: list[SyncDomain] = ["notes", "chat", "workspaces", "source_cache", "media"]
V1_SYNC_OPERATIONS: list[SyncOperation] = ["upsert", "delete", "link", "unlink", "resolve_conflict"]
V1_ENCRYPTION_POLICIES: list[EncryptionPolicy] = [
    "client_private_v1",
    "server_trusted",
    "shared_workspace_v1",
]

_PRIVATE_CLEAR_PAYLOAD_KEYS = {
    "attachment",
    "attachments",
    "body",
    "content",
    "summary",
    "text",
    "title",
    "transcript",
}


def _normalize_object_map(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    return value


def _find_private_clear_payload_key(value: Any, path: str = "payload_clear") -> str | None:
    if isinstance(value, dict):
        for key, nested in value.items():
            normalized_key = str(key).strip().lower().replace("-", "_")
            nested_path = f"{path}.{key}"
            if normalized_key in _PRIVATE_CLEAR_PAYLOAD_KEYS or normalized_key.startswith("attachment_"):
                return nested_path
            private_nested_key = _find_private_clear_payload_key(nested, nested_path)
            if private_nested_key:
                return private_nested_key
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            private_nested_key = _find_private_clear_payload_key(nested, f"{path}[{index}]")
            if private_nested_key:
                return private_nested_key
    return None


class SyncCapabilitiesResponse(BaseModel):
    """Server-supported Sync v2 protocol capabilities."""

    protocol_version: int = Field(2, ge=2, description="Current Sync protocol version.")
    min_supported_protocol_version: int = Field(2, ge=2, description="Oldest accepted Sync protocol version.")
    supported_domains: list[SyncDomain] = Field(default_factory=lambda: list(V1_SYNC_DOMAINS))
    supported_operations: list[SyncOperation] = Field(default_factory=lambda: list(V1_SYNC_OPERATIONS))
    encryption_policies: list[EncryptionPolicy] = Field(default_factory=lambda: list(V1_ENCRYPTION_POLICIES))
    max_batch_size: int = Field(100, ge=1, description="Maximum envelopes accepted by one push.")
    max_envelope_payload_bytes: int = Field(262_144, ge=1)
    max_attachment_bytes: int = Field(1_048_576, ge=1)
    supports_restore_manifest: bool = True
    supports_conflicts: bool = True
    supports_attachments: bool = True
    compatibility_flags: dict[str, bool] = Field(default_factory=dict)
    server_time: str | None = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "protocol_version": 2,
                "min_supported_protocol_version": 2,
                "supported_domains": ["notes", "chat", "workspaces", "source_cache", "media"],
                "encryption_policies": ["client_private_v1", "server_trusted", "shared_workspace_v1"],
                "max_batch_size": 100,
                "supports_restore_manifest": True,
            }
        }
    )


class SyncDeviceRegisterRequest(BaseModel):
    """Request to register or refresh a Sync v2 device."""

    device_id: str | None = Field(None, description="Existing device ID when refreshing registration.")
    display_name: str = Field(..., description="Human-readable device name.")
    client_type: str = Field("chatbook", description="Client family, such as chatbook or webui.")
    client_version: str | None = None
    supported_domains: list[SyncDomain] = Field(default_factory=list)
    capabilities: dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("capabilities", "client_capabilities"),
    )


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


class SyncDatasetEnrollRequest(BaseModel):
    """Request to create or join a sync dataset."""

    dataset_id: str | None = Field(None, description="Existing dataset ID when joining a dataset.")
    device_id: str | None = None
    scope_type: DatasetScopeType = "personal"
    workspace_id: str | None = None
    domains: list[SyncDomain] = Field(default_factory=lambda: list(V1_SYNC_DOMAINS))
    encryption_policy: EncryptionPolicy = "client_private_v1"
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


class SyncV2Envelope(BaseModel):
    """Protocol unit exchanged by Sync v2 clients and the server."""

    client_envelope_id: str
    dataset_id: str
    domain: SyncDomain
    entity_id: str
    operation: SyncOperation
    adapter_version: int = Field(..., ge=1)
    device_id: str | None = None
    stable_key: str | None = None
    client_timestamp: str | None = None
    server_timestamp: str | None = None
    server_sequence: int | None = Field(None, ge=0)
    base_version: str | int | None = None
    entity_version: str | int | None = None
    dependencies: list[dict[str, Any]] = Field(default_factory=list)
    routing_metadata: dict[str, Any] = Field(default_factory=dict)
    payload_ciphertext: str | None = None
    payload_clear: dict[str, Any] = Field(default_factory=dict)
    payload_hash: str
    payload_size_bytes: int | None = Field(None, ge=0)
    encryption_policy: EncryptionPolicy = "client_private_v1"
    status: str | None = None

    @field_validator("routing_metadata", "payload_clear", mode="before")
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
    def _reject_clear_private_payload(self) -> "SyncV2Envelope":
        if self.encryption_policy != "client_private_v1":
            return self

        private_key_path = _find_private_clear_payload_key(self.payload_clear)
        if private_key_path:
            raise ValueError(
                f"{private_key_path} must be encrypted for client_private_v1 sync envelopes"
            )
        return self

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "client_envelope_id": "env-1",
                "dataset_id": "dataset-1",
                "domain": "notes",
                "entity_id": "note-1",
                "operation": "upsert",
                "adapter_version": 1,
                "routing_metadata": {"entity_kind": "note"},
                "payload_ciphertext": "base64-or-jwe-opaque-payload",
                "payload_clear": {"status": "active"},
                "payload_hash": "sha256:...",
            }
        }
    )


class SyncPushRequest(BaseModel):
    """Batch of client-originated envelopes pushed to the server."""

    dataset_id: str
    device_id: str | None = None
    envelopes: list[SyncV2Envelope] = Field(default_factory=list)
    idempotency_key: str | None = None
    last_known_cursor: str | None = None


class SyncPushAcceptedEnvelope(BaseModel):
    """Accepted push outcome for one client envelope."""

    client_envelope_id: str
    server_sequence: int = Field(..., ge=0)
    domain: SyncDomain | None = None
    entity_id: str | None = None


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
    entity_id: str
    server_sequence: int | None = Field(None, ge=0)
    message: str | None = None


class SyncPushResponse(BaseModel):
    """Per-envelope outcomes returned after a push batch."""

    dataset_id: str
    accepted: list[SyncPushAcceptedEnvelope] = Field(default_factory=list)
    rejected: list[SyncPushRejectedEnvelope] = Field(default_factory=list)
    conflicts: list[SyncPushConflictEnvelope] = Field(default_factory=list)
    next_cursor: str | None = None


class SyncPullResponse(BaseModel):
    """Stable sequence-ordered envelopes returned by pull."""

    dataset_id: str
    envelopes: list[SyncV2Envelope] = Field(default_factory=list)
    next_cursor: str | None = None
    has_more: bool = False


class SyncAttachmentUploadRequest(BaseModel):
    """Request metadata for uploading a small encrypted sync attachment."""

    dataset_id: str
    domain: SyncDomain
    entity_id: str
    attachment_id: str
    content_type: str
    size_bytes: int = Field(..., ge=0)
    payload_ciphertext: str
    payload_hash: str
    encryption_policy: EncryptionPolicy = "client_private_v1"
    metadata: dict[str, Any] = Field(default_factory=dict)


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
    entity_id: str
    conflict_type: str
    status: ConflictStatus = "unresolved"
    base_envelope_id: str | None = None
    local_envelope_id: str | None = None
    remote_envelope_id: str | None = None
    server_sequence: int | None = Field(None, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    resolved_by_envelope_id: str | None = None
    created_at: str | None = None
    resolved_at: str | None = None


class SyncConflictResolveRequest(BaseModel):
    """Request to resolve a conflict by action or replacement envelope."""

    conflict_id: str | None = None
    action: ConflictResolutionAction
    resolution_envelope: SyncV2Envelope | None = None
    resolved_by_device_id: str | None = None
    notes: str | None = None


class SyncKeyRecoveryBundleRequest(BaseModel):
    """Client-generated encrypted key recovery material for a dataset."""

    dataset_id: str
    device_id: str | None = None
    key_purpose: str = "dataset_recovery"
    wrapped_key_blob: str
    kdf_metadata: dict[str, Any] = Field(default_factory=dict)
    recovery_hint: str | None = None
    rotation_of_key_record_id: str | None = None


__all__ = [
    "ConflictResolutionAction",
    "ConflictStatus",
    "DatasetScopeType",
    "EncryptionPolicy",
    "SyncAttachmentUploadRequest",
    "SyncAttachmentUploadResponse",
    "SyncCapabilitiesResponse",
    "SyncConflictRecord",
    "SyncConflictResolveRequest",
    "SyncDatasetEnrollRequest",
    "SyncDatasetEnrollResponse",
    "SyncDeviceRegisterRequest",
    "SyncDeviceRegisterResponse",
    "SyncDomain",
    "SyncKeyRecoveryBundleRequest",
    "SyncOperation",
    "SyncPullResponse",
    "SyncPushAcceptedEnvelope",
    "SyncPushConflictEnvelope",
    "SyncPushRejectedEnvelope",
    "SyncPushRequest",
    "SyncPushResponse",
    "SyncRestoreManifestDevice",
    "SyncRestoreManifestDataset",
    "SyncRestoreManifestResponse",
    "SyncV2Envelope",
]
