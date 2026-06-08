"""Pydantic schemas for workspace CRUD."""
from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, ValidationInfo, field_validator, model_validator


WORKSPACE_MIGRATION_MAX_MANIFEST_BYTES = 256 * 1024
WORKSPACE_MIGRATION_MAX_DIAGNOSTICS_BYTES = 64 * 1024
WORKSPACE_MIGRATION_MAX_CHUNK_METADATA_BYTES = 64 * 1024
WORKSPACE_MIGRATION_MAX_CHUNK_BYTES = 2 * 1024 * 1024
WORKSPACE_MIGRATION_MAX_DECLARED_CHUNKS = 512
WORKSPACE_MEMBERSHIP_MAX_PROVENANCE_BYTES = 16 * 1024
WORKSPACE_MEMBERSHIP_MAX_METADATA_BYTES = 16 * 1024
_SHA256_RE = re.compile(r"^[a-fA-F0-9]{64}$")

WorkspaceProfile = Literal["research", "project"]
WorkspaceKind = Literal["research_workspace", "project_workspace"]
WorkspaceAssistantKind = Literal["persona"]
WorkspacePersonaMemoryMode = Literal["read_only", "read_write"]
WorkspaceEffectiveAssistantDefaultStatus = Literal["available", "unavailable", "none"]
WorkspaceEffectiveAssistantDefaultSource = Literal["workspace", "none"]
WorkspaceAssistantDefaultDegradedReason = Literal[
    "persona_deleted",
    "persona_unavailable",
    "persona_feature_disabled",
    "permission_denied",
    "invalid_default",
    "unsupported_assistant_kind",
]
WorkspaceResolutionStatus = Literal["complete", "partial", "failed"]
WorkspaceAttentionState = Literal[
    "ready",
    "setup_pending",
    "working",
    "needs_attention",
    "blocked",
    "archived",
]
WorkspaceProjectRootBackend = Literal["host_local", "sandbox_volume"]
WorkspaceProjectRootState = Literal[
    "not_configured",
    "provisioning",
    "attached",
    "unavailable",
    "missing",
    "detached",
    "failed",
    "cleanup_pending",
    "archived",
]
WorkspaceMembershipResourceType = Literal[
    "workspace_note",
    "media",
    "workspace_source",
    "workspace_artifact",
    "chat",
]
WorkspaceMembershipRole = Literal[
    "member",
    "source",
    "artifact",
    "conversation",
    "runtime",
    "reference",
]
WorkspaceMembershipTransferPolicy = Literal["link", "copy", "promote", "import"]


def _json_size_bytes(value: Any) -> int:
    return len(
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )


def _validate_sha256(value: str) -> str:
    if not _SHA256_RE.fullmatch(value):
        raise ValueError("must be a 64-character SHA-256 hex digest")
    return value.lower()


def _validate_json_size(value: Any, *, field_name: str, max_bytes: int) -> Any:
    try:
        size = _json_size_bytes(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be JSON serializable") from exc
    if size > max_bytes:
        raise ValueError(f"{field_name} exceeds {max_bytes} bytes")
    return value


class WorkspaceUpsertRequest(BaseModel):
    name: str
    archived: bool = False
    study_materials_policy: Literal["general", "workspace"] = "general"
    workspace_profile: WorkspaceProfile = "research"


class WorkspaceAssistantDefaults(BaseModel):
    """Persisted default assistant selection for new Workspace-scoped chats."""

    assistant_kind: WorkspaceAssistantKind
    assistant_id: str = Field(..., min_length=1, max_length=128)
    persona_memory_mode: WorkspacePersonaMemoryMode = "read_only"
    voice: None = None
    style: None = None
    tool_policy_profile_id: None = None

    @field_validator("voice", "style", "tool_policy_profile_id", mode="before")
    @classmethod
    def _deferred_fields_must_be_null(cls, value: Any, info: ValidationInfo) -> None:
        if value is not None:
            raise ValueError(f"{info.field_name} must be null")
        return None


class WorkspaceEffectiveAssistantDefault(BaseModel):
    """Resolved Workspace assistant default exposed to clients with degradation state."""

    status: WorkspaceEffectiveAssistantDefaultStatus
    source: WorkspaceEffectiveAssistantDefaultSource
    assistant_kind: WorkspaceAssistantKind | None = None
    assistant_id: str | None = None
    label: str | None = None
    persona_memory_mode: WorkspacePersonaMemoryMode | None = None
    degraded_reason: WorkspaceAssistantDefaultDegradedReason | None = None

    @model_validator(mode="after")
    def _validate_status_relations(self) -> "WorkspaceEffectiveAssistantDefault":
        if self.status == "available":
            if self.assistant_kind is None or self.assistant_id is None:
                raise ValueError("available assistant defaults require assistant_kind and assistant_id")
            if self.degraded_reason is not None:
                raise ValueError("available assistant defaults must not include degraded_reason")
            return self
        if self.status == "unavailable":
            if self.degraded_reason is None:
                raise ValueError("unavailable assistant defaults require degraded_reason")
            return self
        populated_none_fields = [
            field_name
            for field_name in (
                "assistant_kind",
                "assistant_id",
                "label",
                "persona_memory_mode",
                "degraded_reason",
            )
            if getattr(self, field_name) is not None
        ]
        if populated_none_fields:
            raise ValueError("none assistant defaults must not include assistant or degradation fields")
        return self


class WorkspacePatchRequest(BaseModel):
    name: str | None = None
    archived: bool | None = None
    study_materials_policy: Literal["general", "workspace"] | None = None
    workspace_profile: WorkspaceProfile | None = None
    banner_title: str | None = None
    banner_subtitle: str | None = None
    banner_color: str | None = None
    audio_provider: str | None = None
    audio_model: str | None = None
    audio_voice: str | None = None
    audio_speed: float | None = None
    assistant_defaults: WorkspaceAssistantDefaults | None = None
    confirm_read_write_assistant_default: StrictBool | None = None
    version: int = Field(..., description="Current version for optimistic locking")

    @model_validator(mode="after")
    def _validate_assistant_default_confirmation(self) -> "WorkspacePatchRequest":
        if self.assistant_defaults is None:
            if self.confirm_read_write_assistant_default is not None:
                raise ValueError("confirm_read_write_assistant_default only applies to assistant_defaults")
            return self
        if self.assistant_defaults.persona_memory_mode == "read_write":
            if self.confirm_read_write_assistant_default is not True:
                raise ValueError("read_write assistant defaults require confirm_read_write_assistant_default=true")
            return self
        if self.confirm_read_write_assistant_default is not None:
            raise ValueError("confirm_read_write_assistant_default only applies to read_write assistant defaults")
        return self


class WorkspaceResponse(BaseModel):
    id: str
    name: str | None = None
    archived: bool = False
    study_materials_policy: Literal["general", "workspace"] = "general"
    workspace_profile: WorkspaceProfile = "research"
    deleted: bool = False
    banner_title: str | None = None
    banner_subtitle: str | None = None
    banner_color: str | None = None
    audio_provider: str | None = None
    audio_model: str | None = None
    audio_voice: str | None = None
    audio_speed: float | None = None
    assistant_defaults: WorkspaceAssistantDefaults | None = None
    effective_assistant_default: WorkspaceEffectiveAssistantDefault | None = None
    created_at: str
    last_modified: str
    version: int


class WorkspaceListResponse(BaseModel):
    items: list[WorkspaceResponse]
    total: int


# --- Membership schemas ---

class WorkspaceMembershipCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    resource_type: str = Field(..., min_length=1)
    resource_id: str = Field(..., min_length=1)
    role: WorkspaceMembershipRole = "member"
    label: str | None = Field(default=None, max_length=512)
    transfer_policy: WorkspaceMembershipTransferPolicy = "link"
    provenance: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("provenance")
    @classmethod
    def _validate_provenance_size(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_json_size(
            value,
            field_name="provenance",
            max_bytes=WORKSPACE_MEMBERSHIP_MAX_PROVENANCE_BYTES,
        )

    @field_validator("metadata")
    @classmethod
    def _validate_membership_metadata_size(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_json_size(
            value,
            field_name="metadata",
            max_bytes=WORKSPACE_MEMBERSHIP_MAX_METADATA_BYTES,
        )


class WorkspaceMembershipSummaryResponse(BaseModel):
    title: str | None = None
    subtitle: str | None = None
    href: str | None = None
    updated_at: str | None = None
    state: str = "available"
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkspaceMembershipResponse(BaseModel):
    workspace_id: str
    resource_type: WorkspaceMembershipResourceType
    resource_id: str
    role: WorkspaceMembershipRole
    label: str | None = None
    transfer_policy: WorkspaceMembershipTransferPolicy = "link"
    provenance: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    summary: WorkspaceMembershipSummaryResponse | None = None
    created_at: str
    updated_at: str
    version: int
    deleted: bool = False


class WorkspaceMembershipListSummary(BaseModel):
    total: int = 0
    by_resource_type: dict[str, int] = Field(default_factory=dict)
    by_role: dict[str, int] = Field(default_factory=dict)


class WorkspaceMembershipListResponse(BaseModel):
    workspace_id: str
    items: list[WorkspaceMembershipResponse]
    total: int
    next_cursor: str | None = None
    summary: WorkspaceMembershipListSummary = Field(default_factory=WorkspaceMembershipListSummary)


class WorkspaceResourceMembershipListResponse(BaseModel):
    resource_type: WorkspaceMembershipResourceType
    resource_id: str
    items: list[WorkspaceMembershipResponse]
    total: int
    next_cursor: str | None = None
    summary: WorkspaceMembershipListSummary = Field(default_factory=WorkspaceMembershipListSummary)


class WorkspaceContextMembershipSummary(BaseModel):
    total: int = 0
    by_resource_type: dict[str, int] = Field(default_factory=dict)
    by_role: dict[str, int] = Field(default_factory=dict)


# --- Source schemas ---

class WorkspaceSourceCreateRequest(BaseModel):
    id: str
    media_id: int
    title: str
    source_type: str
    url: str | None = None
    position: int = 0
    selected: bool = True


class WorkspaceSourceUpdateRequest(BaseModel):
    title: str | None = None
    source_type: str | None = None
    url: str | None = None
    position: int | None = None
    selected: bool | None = None
    version: int = Field(..., description="Current version for optimistic locking")


class WorkspaceSourceResponse(BaseModel):
    id: str
    workspace_id: str
    media_id: int
    title: str
    source_type: str
    url: str | None = None
    position: int = 0
    selected: bool = True
    added_at: str
    version: int


class WorkspaceSourceSelectionRequest(BaseModel):
    selected_ids: list[str]


class WorkspaceSourceReorderRequest(BaseModel):
    ordered_ids: list[str]


WorkspaceSourceLifecycleState = Literal[
    "queued",
    "ingesting",
    "extracting",
    "chunking",
    "indexing",
    "queryable",
    "partially_queryable",
    "failed",
    "retrying",
    "missing_media",
    "blocked_by_permissions",
]

WorkspaceCapabilityServiceState = Literal[
    "available",
    "private",
    "not_configured",
    "needs_approval",
    "unknown",
    "blocked",
    "degraded",
]


class WorkspaceSourceReadiness(BaseModel):
    metadata_ready: bool = False
    text_extracted: bool = False
    fts_ready: bool = False
    vector_ready: bool = False
    citation_ready: bool = False
    summary_ready: bool = False
    tool_accessible: bool = False


class WorkspaceSourceJobStatus(BaseModel):
    id: int | None = None
    uuid: str | None = None
    status: str | None = None
    job_type: str | None = None
    progress_percent: float | None = None
    progress_message: str | None = None
    error_message: str | None = None


class WorkspaceSourceStatusResponse(BaseModel):
    id: str
    workspace_id: str
    media_id: int | None = None
    title: str
    source_type: str
    url: str | None = None
    selected: bool = True
    state: WorkspaceSourceLifecycleState
    status_reason: str
    readiness: WorkspaceSourceReadiness
    progress_percent: float | None = None
    progress_message: str | None = None
    job: WorkspaceSourceJobStatus | None = None
    updated_at: str = ""


class WorkspaceSourceStatusSummary(BaseModel):
    total: int = 0
    selected: int = 0
    queryable: int = 0
    partially_queryable: int = 0
    processing: int = 0
    failed: int = 0
    missing: int = 0


class WorkspaceSourceStatusListResponse(BaseModel):
    workspace_id: str
    sources: list[WorkspaceSourceStatusResponse]
    summary: WorkspaceSourceStatusSummary


class WorkspaceCapabilityService(BaseModel):
    state: WorkspaceCapabilityServiceState
    reason_code: str | None = None
    management_surface: str | None = None


class WorkspaceAllowedAction(BaseModel):
    allowed: bool
    reason_code: str | None = None


class WorkspaceContextPartialError(BaseModel):
    scope: str = "workspace"
    code: str = "dependency_resolution_partial"
    message: str = ""


class WorkspaceResolution(BaseModel):
    status: WorkspaceResolutionStatus = "complete"
    partial_errors: list[WorkspaceContextPartialError] = Field(default_factory=list)


class WorkspaceFileInventory(BaseModel):
    state: str | None = None
    indexed_file_count: int | None = None
    total_file_count: int | None = None
    updated_at: str | None = None
    available: bool = False


class WorkspaceProjectRoot(BaseModel):
    state: WorkspaceProjectRootState = "not_configured"
    root_id: str | None = None
    backend: WorkspaceProjectRootBackend | None = None
    display_name: str | None = None
    path_hint: str | None = None
    git_state: str | None = None
    file_inventory_state: str | None = None
    file_inventory: WorkspaceFileInventory = Field(default_factory=WorkspaceFileInventory)
    indexing_state: str | None = None
    sandbox_mount_state: str | None = None
    mcp_trust_state: str | None = None


class WorkspacePrimaryRootAttachRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: WorkspaceProjectRootBackend
    root_id: str | None = None
    absolute_root: str | None = None
    sandbox_volume_id: str | None = None
    display_name: str | None = Field(default=None, max_length=120)
    replace_existing: StrictBool = False
    expected_workspace_version: int | None = Field(default=None, ge=1)
    strict_sandbox_validation: StrictBool = False


class WorkspaceSandboxRootProvisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    display_name: str | None = Field(default=None, max_length=120)
    requested_runtime: str | None = Field(default=None, max_length=80)
    root_id: str | None = Field(default=None, max_length=128)
    replace_existing: StrictBool = False
    expected_workspace_version: int | None = Field(default=None, ge=1)


WorkspaceFileInventoryState = Literal[
    "not_started",
    "queued",
    "scanning",
    "current",
    "partial",
    "stale",
    "failed",
    "disabled",
]
WorkspaceFileInventoryEntryKind = Literal["file", "directory", "symlink", "other"]


class WorkspaceFileInventoryScanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    force: StrictBool = False
    expected_root_version: int | None = Field(default=None, ge=1)


class WorkspaceFileInventoryJobStatus(BaseModel):
    id: int | None = None
    uuid: str | None = None
    status: str | None = None
    job_type: str | None = None
    progress_percent: float | None = None
    progress_message: str | None = None
    error_message: str | None = None


class WorkspaceFileInventoryCounts(BaseModel):
    files: int = 0
    directories: int = 0
    symlinks: int = 0
    ignored: int = 0
    indexing_candidates: int = 0
    diagnostics: int = 0
    total_entries: int = 0


class WorkspaceFileInventoryDiagnostic(BaseModel):
    code: str
    message: str
    path_hint: str | None = None


class WorkspaceFileInventoryStatusResponse(BaseModel):
    workspace_id: str
    root_id: str | None = None
    state: WorkspaceFileInventoryState
    durable_state: str | None = None
    stale: bool = False
    last_scan_id: str | None = None
    last_scan_started_at: str | None = None
    last_scan_completed_at: str | None = None
    root_version: int | None = None
    scan_root_version: int | None = None
    ignore_policy_fingerprint: str | None = None
    root_snapshot_token: str | None = None
    counts: WorkspaceFileInventoryCounts = Field(default_factory=WorkspaceFileInventoryCounts)
    diagnostics: list[WorkspaceFileInventoryDiagnostic] = Field(default_factory=list)
    job: WorkspaceFileInventoryJobStatus | None = None
    updated_at: str | None = None


class WorkspaceFileInventoryItemResponse(BaseModel):
    relative_path: str
    entry_kind: str
    size_bytes: int | None = None
    mtime_ns: int | None = None
    mode_bits: int | None = None
    extension: str | None = None
    mime_hint: str | None = None
    language_hint: str | None = None
    ignored: bool = False
    ignore_reason: str | None = None
    indexing_candidate: bool = False


class WorkspaceFileInventoryItemsResponse(BaseModel):
    workspace_id: str
    root_id: str | None = None
    items: list[WorkspaceFileInventoryItemResponse]
    next_cursor: str | None = None
    limit: int


class WorkspaceRuntimeBinding(BaseModel):
    service: str
    state: WorkspaceCapabilityServiceState
    reason_code: str | None = None
    management_surface: str | None = None


WorkspaceOperationStatus = Literal[
    "queued",
    "running",
    "succeeded",
    "failed",
    "conflicted",
    "expired",
]


class WorkspaceOperationResponse(BaseModel):
    operation_id: str
    workspace_id: str
    command: str
    status: WorkspaceOperationStatus
    started_at: str
    updated_at: str
    retryable: bool = False
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    poll_href: str


class WorkspaceRootResponse(WorkspaceProjectRoot):
    workspace_id: str | None = None
    is_primary: bool = True
    version: int | None = None
    updated_at: str | None = None


class WorkspaceRootsResponse(BaseModel):
    workspace_id: str
    workspace_profile: WorkspaceProfile = "research"
    primary_root: WorkspaceRootResponse | None = None
    roots: list[WorkspaceRootResponse] = Field(default_factory=list)


class WorkspaceSandboxRootProvisionResponse(BaseModel):
    workspace_id: str
    workspace_profile: WorkspaceProfile = "project"
    operation: WorkspaceOperationResponse
    primary_root: WorkspaceRootResponse | None = None


class WorkspaceCapabilitiesResponse(BaseModel):
    workspace_id: str
    workspace_profile: WorkspaceProfile = "research"
    workspace_kind: WorkspaceKind = "research_workspace"
    access_level: Literal["owner", "editor", "viewer"] = "owner"
    resolution: WorkspaceResolution = Field(default_factory=WorkspaceResolution)
    project_root: WorkspaceProjectRoot = Field(default_factory=WorkspaceProjectRoot)
    source_summary: WorkspaceSourceStatusSummary
    workspace_services: dict[str, WorkspaceCapabilityService]
    allowed_actions: dict[str, WorkspaceAllowedAction]


class WorkspaceSourcePreviewSummary(BaseModel):
    available: bool = False
    detail_href: str | None = None
    snippet_count: int | None = None
    total_chars: int | None = None
    unavailable_reason: str | None = None


class WorkspaceContextSource(BaseModel):
    id: str
    workspace_id: str
    media_id: int | None = None
    title: str
    source_type: str
    url: str | None = None
    position: int = 0
    selected: bool = True
    added_at: str
    version: int
    state: WorkspaceSourceLifecycleState
    status_reason: str
    readiness: WorkspaceSourceReadiness
    progress_percent: float | None = None
    progress_message: str | None = None
    job: WorkspaceSourceJobStatus | None = None
    updated_at: str = ""
    preview: WorkspaceSourcePreviewSummary


class WorkspaceContextSources(BaseModel):
    items: list[WorkspaceContextSource]
    summary: WorkspaceSourceStatusSummary


class WorkspaceContextResponse(BaseModel):
    workspace_id: str
    workspace_profile: WorkspaceProfile = "research"
    workspace_kind: WorkspaceKind = "research_workspace"
    schema_version: int = 2
    generated_at: str
    workspace: WorkspaceResponse
    attention_state: WorkspaceAttentionState = "needs_attention"
    resolution: WorkspaceResolution = Field(default_factory=WorkspaceResolution)
    project_root: WorkspaceProjectRoot = Field(default_factory=WorkspaceProjectRoot)
    sources: WorkspaceContextSources
    capabilities: WorkspaceCapabilitiesResponse
    services: dict[str, WorkspaceCapabilityService]
    allowed_actions: dict[str, WorkspaceAllowedAction]
    active_jobs: list[WorkspaceSourceJobStatus] = Field(default_factory=list)
    active_operations: list[WorkspaceOperationResponse] = Field(default_factory=list)
    memberships: WorkspaceContextMembershipSummary = Field(default_factory=WorkspaceContextMembershipSummary)
    partial_errors: list[WorkspaceContextPartialError] = Field(default_factory=list)


WorkspaceSourcePreviewMode = Literal[
    "available",
    "pending",
    "failed",
    "missing_media",
    "empty",
]

WorkspaceSourcePreviewSnippetKind = Literal["content_excerpt", "chunk"]


class WorkspaceSourcePreviewSnippet(BaseModel):
    id: str
    source_id: str
    media_id: int | None = None
    kind: WorkspaceSourcePreviewSnippetKind
    text: str
    start_char: int | None = None
    end_char: int | None = None
    chunk_index: int | None = None
    chunk_uuid: str | None = None
    chunk_type: str | None = None


class WorkspaceSourcePreviewResponse(BaseModel):
    workspace_id: str
    source_id: str
    media_id: int | None = None
    title: str
    source_type: str
    url: str | None = None
    state: WorkspaceSourceLifecycleState
    status_reason: str
    readiness: WorkspaceSourceReadiness
    content_available: bool
    preview_mode: WorkspaceSourcePreviewMode
    unavailable_reason: str | None = None
    text_preview: str | None = None
    text_total_chars: int | None = None
    text_truncated: bool = False
    snippets: list[WorkspaceSourcePreviewSnippet] = Field(default_factory=list)
    generated_at: str


class StatusResponse(BaseModel):
    ok: bool = True


# --- Migration schemas ---

WorkspaceMigrationStatus = Literal["created", "finalized", "failed"]


class WorkspaceMigrationChunkDeclaration(BaseModel):
    id: str = Field(..., min_length=1, max_length=128)
    sha256: str
    byte_count: int = Field(..., ge=0, le=WORKSPACE_MIGRATION_MAX_CHUNK_BYTES)
    chunk_kind: str = Field(default="workspace_bundle", min_length=1, max_length=64)

    @field_validator("sha256")
    @classmethod
    def _validate_chunk_sha256(cls, value: str) -> str:
        return _validate_sha256(value)


class WorkspaceMigrationCreateRequest(BaseModel):
    id: str = Field(..., min_length=1, max_length=128)
    idempotency_key: str = Field(..., min_length=1, max_length=160)
    target_workspace_id: str = Field(..., min_length=1, max_length=128)
    target_workspace_name: str = Field(..., min_length=1, max_length=256)
    source_product: str = Field(default="research-workspace-webui", min_length=1, max_length=128)
    manifest_hash: str
    declared_chunks: list[WorkspaceMigrationChunkDeclaration] = Field(
        default_factory=list,
        max_length=WORKSPACE_MIGRATION_MAX_DECLARED_CHUNKS,
    )
    manifest: dict[str, Any] = Field(default_factory=dict)
    diagnostics: dict[str, Any] = Field(default_factory=dict)

    @field_validator("manifest_hash")
    @classmethod
    def _validate_manifest_hash(cls, value: str) -> str:
        return _validate_sha256(value)

    @field_validator("manifest")
    @classmethod
    def _validate_manifest_size(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_json_size(
            value,
            field_name="manifest",
            max_bytes=WORKSPACE_MIGRATION_MAX_MANIFEST_BYTES,
        )

    @field_validator("diagnostics")
    @classmethod
    def _validate_diagnostics_size(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_json_size(
            value,
            field_name="diagnostics",
            max_bytes=WORKSPACE_MIGRATION_MAX_DIAGNOSTICS_BYTES,
        )

    @model_validator(mode="after")
    def _validate_declared_chunk_ids(self) -> "WorkspaceMigrationCreateRequest":
        ids = [chunk.id for chunk in self.declared_chunks]
        if len(ids) != len(set(ids)):
            raise ValueError("declared_chunks must use unique chunk ids")
        return self


class WorkspaceMigrationChunkUploadRequest(BaseModel):
    sha256: str
    byte_count: int = Field(..., ge=0, le=WORKSPACE_MIGRATION_MAX_CHUNK_BYTES)
    chunk_kind: str = Field(default="workspace_bundle", min_length=1, max_length=64)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("sha256")
    @classmethod
    def _validate_upload_sha256(cls, value: str) -> str:
        return _validate_sha256(value)

    @field_validator("metadata")
    @classmethod
    def _validate_metadata_size(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_json_size(
            value,
            field_name="metadata",
            max_bytes=WORKSPACE_MIGRATION_MAX_CHUNK_METADATA_BYTES,
        )


class WorkspaceMigrationFinalizeRequest(BaseModel):
    manifest_hash: str

    @field_validator("manifest_hash")
    @classmethod
    def _validate_finalize_manifest_hash(cls, value: str) -> str:
        return _validate_sha256(value)


class WorkspaceMigrationClientDeleteAckRequest(BaseModel):
    acknowledged_manifest_hash: str

    @field_validator("acknowledged_manifest_hash")
    @classmethod
    def _validate_ack_manifest_hash(cls, value: str) -> str:
        return _validate_sha256(value)


class WorkspaceMigrationChunkReceiptResponse(BaseModel):
    id: str
    migration_id: str
    sha256: str
    byte_count: int
    chunk_kind: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    status: Literal["accepted"] = "accepted"
    accepted_at: str


class WorkspaceMigrationResponse(BaseModel):
    id: str
    idempotency_key: str
    target_workspace_id: str
    target_workspace_name: str
    source_product: str
    manifest_hash: str
    status: WorkspaceMigrationStatus
    declared_chunk_count: int
    accepted_chunk_count: int
    missing_chunk_ids: list[str]
    client_delete_eligible: bool = False
    created_at: str
    updated_at: str
    finalized_at: str | None = None
    recovery_manifest: dict[str, Any] = Field(default_factory=dict)
    chunks: list[WorkspaceMigrationChunkReceiptResponse] = Field(default_factory=list)


# --- Artifact schemas ---

WorkspaceArtifactReviewState = Literal[
    "draft",
    "reviewing",
    "accepted",
    "needs_revision",
    "rejected",
    "exported",
    "assigned",
    "archived",
]

WorkspaceArtifactExportFormat = Literal["md", "html", "json"]


class WorkspaceArtifactRedaction(BaseModel):
    model_config = ConfigDict(extra="allow")

    support_safe: StrictBool = True
    redacted: StrictBool = False
    retention_class: str | None = None
    redacted_fields: list[str] = Field(default_factory=list)


class WorkspaceArtifactCreateRequest(BaseModel):
    id: str
    artifact_type: str
    title: str
    status: str = "pending"
    content: str | None = None
    content_type: str = "text/markdown"
    preview_text: str | None = None
    summary: str | None = None
    review_state: WorkspaceArtifactReviewState = "draft"
    owner_scope: str = "user"
    owner_id: str | None = None
    project_id: str | None = None
    task_id: str | None = None
    source_collection_id: str | None = None
    producer_metadata: dict[str, Any] = Field(default_factory=dict)
    source_lineage: dict[str, Any] = Field(default_factory=dict)
    review_metadata: dict[str, Any] = Field(default_factory=dict)
    version_metadata: dict[str, Any] = Field(default_factory=dict)
    export_refs: list[dict[str, Any]] = Field(default_factory=list)
    redaction: WorkspaceArtifactRedaction = Field(default_factory=WorkspaceArtifactRedaction)
    schema_version: int = 1


class WorkspaceArtifactUpdateRequest(BaseModel):
    title: str | None = None
    status: str | None = None
    content: str | None = None
    content_type: str | None = None
    preview_text: str | None = None
    summary: str | None = None
    review_state: WorkspaceArtifactReviewState | None = None
    owner_scope: str | None = None
    owner_id: str | None = None
    project_id: str | None = None
    task_id: str | None = None
    source_collection_id: str | None = None
    producer_metadata: dict[str, Any] | None = None
    source_lineage: dict[str, Any] | None = None
    review_metadata: dict[str, Any] | None = None
    version_metadata: dict[str, Any] | None = None
    export_refs: list[dict[str, Any]] | None = None
    redaction: WorkspaceArtifactRedaction | None = None
    schema_version: int | None = None
    total_tokens: int | None = None
    total_cost_usd: float | None = None
    completed_at: str | None = None
    version: int = Field(..., description="Current version for optimistic locking")


class WorkspaceArtifactResponse(BaseModel):
    id: str
    workspace_id: str
    artifact_type: str
    title: str
    status: str = "pending"
    content: str | None = None
    content_type: str = "text/markdown"
    preview_text: str | None = None
    summary: str | None = None
    review_state: WorkspaceArtifactReviewState = "draft"
    owner_scope: str = "user"
    owner_id: str | None = None
    project_id: str | None = None
    task_id: str | None = None
    source_collection_id: str | None = None
    root_artifact_id: str
    artifact_version_id: str
    previous_version_id: str | None = None
    producer_metadata: dict[str, Any] = Field(default_factory=dict)
    source_lineage: dict[str, Any] = Field(default_factory=dict)
    review_metadata: dict[str, Any] = Field(default_factory=dict)
    version_metadata: dict[str, Any] = Field(default_factory=dict)
    export_refs: list[dict[str, Any]] = Field(default_factory=list)
    redaction: WorkspaceArtifactRedaction = Field(default_factory=WorkspaceArtifactRedaction)
    schema_version: int = 1
    total_tokens: int | None = None
    total_cost_usd: float | None = None
    created_at: str
    completed_at: str | None = None
    version: int


class WorkspaceArtifactExportRequest(BaseModel):
    format: WorkspaceArtifactExportFormat
    artifact_version_id: str | None = Field(
        default=None,
        description="Optional artifact version id to export. Defaults to the current artifact version.",
    )


class WorkspaceArtifactExportResponse(BaseModel):
    workspace_id: str
    artifact_id: str
    artifact_version_id: str
    review_state: WorkspaceArtifactReviewState
    format: WorkspaceArtifactExportFormat
    content_type: str
    content: str
    bytes: int
    metadata: dict[str, Any] = Field(default_factory=dict)
    export_ref: dict[str, Any] = Field(default_factory=dict)
    generated_at: str


# --- Note schemas ---

class WorkspaceNoteCreateRequest(BaseModel):
    title: str = ""
    content: str = ""
    keywords: list[str] = Field(default_factory=list)


class WorkspaceNoteUpdateRequest(BaseModel):
    title: str | None = None
    content: str | None = None
    keywords_json: str | None = None
    version: int = Field(..., description="Current version for optimistic locking")


class WorkspaceNoteResponse(BaseModel):
    id: int
    workspace_id: str
    title: str
    content: str
    keywords_json: str = "[]"
    created_at: str
    last_modified: str
    version: int
