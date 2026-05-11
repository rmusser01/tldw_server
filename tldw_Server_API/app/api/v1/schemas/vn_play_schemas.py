"""Pydantic schemas for VN Play runtime APIs."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr, model_validator

from tldw_Server_API.app.api.v1.schemas.pagination import OffsetPaginationMeta, validate_offset_pagination_aliases
from tldw_Server_API.app.core.VN_Play.constants import (
    LINKED_CHAT_MODE_READ_ONLY_CONTEXT,
    SESSION_STATUS_ACTIVE,
    TRUST_LEVEL_LOCAL,
)

VNPlayMode = Literal["freeform", "story", "scripted_story"]
VNPlaySessionStatus = Literal["active", "paused", "completed", "archived", "failed"]
VNPlayTrustLevel = Literal["local", "trusted_restore", "untrusted_import", "mixed"]
VNPlayLinkedChatMode = Literal["read_only_context"]
VNPlaySetupTrustLevel = Literal["local", "trusted_restore", "untrusted_import", "unknown"]
VNPlaySetupTrustSource = Literal["local_pack", "latest_import_journal", "unknown"]
VNPlaySetupWarningSeverity = Literal["info", "warning", "high_risk"]
VNPlaySetupCompatibilityStatus = Literal["compatible", "different_character", "unknown"]
VNPlaySetupEmptyStateScope = Literal["global", "filter", "page"]
VNPlayBranchRestoreTarget = Literal["branch_latest", "choice_point"]
VNPlayBranchWarningSeverity = Literal["info", "warning", "high_risk"]
VNPlayGenerationRawDebugState = Literal["absent", "available", "redacted", "revealed"]
VNPlayTurnStatus = Literal[
    "pending",
    "model_calling",
    "model_failed",
    "parse_failed",
    "completed",
    "abandoned",
    "cancelled",
]
VNPlayEventType = Literal[
    "session_started",
    "turn_started",
    "turn_completed",
    "turn_failed",
    "user_turn",
    "model_turn",
    "choice_presented",
    "choice_selected",
    "scene_state_changed",
    "visual_directive_requested",
    "visual_directive_applied",
    "visual_directive_rejected",
    "model_turn_parse_failed",
    "safety_gate_triggered",
    "runtime_gate_failed",
    "session_settings_changed",
    "session_checkpoint_created",
    "session_restored",
    "script_generation_canceled",
    "script_generation_revision_activated",
]
VNPlayEventSource = Literal["user", "model", "runtime", "system"]


class VNPlaySessionCreate(BaseModel):
    """Request body for creating a VN Play session."""

    model_config = ConfigDict(extra="forbid")

    mode: VNPlayMode = Field(..., description="Runtime mode: freeform or story.")
    title: StrictStr = Field(..., min_length=1, max_length=500)
    primary_character_id: StrictInt = Field(..., ge=1)
    vn_asset_pack_id: StrictInt = Field(..., ge=1)
    additional_character_ids: list[StrictInt] = Field(default_factory=list)
    linked_chat_id: StrictStr | None = Field(default=None, min_length=1)
    asset_manifest_version: StrictStr | None = Field(default=None, min_length=1)
    source_world_book_ids: list[StrictInt] = Field(default_factory=list)
    content_rating: StrictStr = Field(default="general", min_length=1, max_length=100)
    trust_level: VNPlayTrustLevel = TRUST_LEVEL_LOCAL
    linked_chat_mode: VNPlayLinkedChatMode = LINKED_CHAT_MODE_READ_ONLY_CONTEXT
    seed: StrictStr | None = Field(default=None, min_length=1)
    settings: dict[str, Any] = Field(default_factory=dict)
    script_id: StrictInt | None = Field(default=None, ge=1)
    script_version_id: StrictInt | None = Field(default=None, ge=1)
    acknowledgements: list[StrictStr] = Field(default_factory=list)


class VNPlaySessionUpdate(BaseModel):
    """Request body for patching a VN Play session."""

    model_config = ConfigDict(extra="forbid")

    title: StrictStr | None = Field(default=None, min_length=1, max_length=500)
    status: VNPlaySessionStatus | None = None
    linked_chat_id: StrictStr | None = Field(default=None, min_length=1)
    linked_chat_mode: VNPlayLinkedChatMode | None = None
    settings: dict[str, Any] | None = None
    deleted: StrictBool | None = None


class VNPlayEventResponse(BaseModel):
    """Serialized VN Play event."""

    model_config = ConfigDict(from_attributes=True)

    id: StrictInt
    session_id: StrictInt
    owner_user_id: StrictInt
    sequence_number: StrictInt
    event_type: VNPlayEventType
    event_payload: dict[str, Any] = Field(default_factory=dict)
    source: VNPlayEventSource = "runtime"
    model_provider: str | None = None
    model_name: str | None = None
    branch_node_id: int | None = None
    created_at: str | None = None


class VNPlaySceneStateResponse(BaseModel):
    """Serialized derived VN Play scene state."""

    model_config = ConfigDict(from_attributes=True)

    session_id: StrictInt
    owner_user_id: StrictInt
    last_event_id: int | None = None
    background: dict[str, Any] | None = None
    depth: dict[str, Any] | None = None
    active_sprites: list[dict[str, Any]] = Field(default_factory=list)
    current_background_item_id: int | None = None
    current_depth_item_id: int | None = None
    active_sprite_items: list[dict[str, Any]] = Field(default_factory=list)
    location_key: str | None = None
    mood: str | None = None
    time_of_day: str | None = None
    weather: str | None = None
    active_branch_node_id: int | None = None
    visible_choices: list[dict[str, Any]] = Field(default_factory=list)
    transcript_cursor: int | None = None
    scene_version: StrictInt = Field(default=0, ge=0)
    warnings: list[Any] = Field(default_factory=list)
    updated_at: str | None = None


class VNPlaySessionResponse(BaseModel):
    """Serialized VN Play session metadata."""

    model_config = ConfigDict(from_attributes=True)

    id: StrictInt
    owner_user_id: StrictInt
    mode: VNPlayMode
    title: str
    status: VNPlaySessionStatus = SESSION_STATUS_ACTIVE
    primary_character_id: StrictInt
    additional_character_ids: list[int] = Field(default_factory=list)
    linked_chat_id: str | None = None
    vn_asset_pack_id: StrictInt
    asset_manifest_version: str | None = None
    source_world_book_ids: list[int] = Field(default_factory=list)
    content_rating: str = "general"
    trust_level: VNPlayTrustLevel = TRUST_LEVEL_LOCAL
    linked_chat_mode: VNPlayLinkedChatMode = LINKED_CHAT_MODE_READ_ONLY_CONTEXT
    seed: str | None = None
    settings: dict[str, Any] = Field(default_factory=dict)
    script_id: int | None = None
    script_version_id: int | None = None
    script_manifest_snapshot_id: int | None = None
    script_policy_snapshot_id: int | None = None
    script_generation_profile_snapshot_id: int | None = None
    script_position: dict[str, Any] = Field(default_factory=dict)
    scene_version: StrictInt = Field(default=0, ge=0)
    active_turn_request_id: int | None = None
    current_scene: VNPlaySceneStateResponse | None = None
    scene_state: VNPlaySceneStateResponse | None = None
    created_at: str | None = None
    updated_at: str | None = None
    deleted: bool = False


class VNPlaySetupCharacterOption(BaseModel):
    """Selector-safe character metadata for VN Play setup."""

    id: StrictInt
    name: StrictStr
    description_preview: str | None = None
    tags: list[str] = Field(default_factory=list)
    favorite: StrictBool = False
    deleted: StrictBool = False
    has_image: StrictBool = False


class VNPlaySetupCompatibility(BaseModel):
    """Compatibility summary between a setup character and asset pack."""

    status: VNPlaySetupCompatibilityStatus
    reason_codes: list[str] = Field(default_factory=list)


class VNPlaySetupWarning(BaseModel):
    """Stable setup warning for frontend acknowledgement flows."""

    code: StrictStr
    severity: VNPlaySetupWarningSeverity
    message: StrictStr
    requires_acknowledgement: StrictBool = False


class VNPlaySetupWarningSummary(BaseModel):
    """Aggregate warning metadata for one setup option."""

    highest_severity: VNPlaySetupWarningSeverity = "info"
    requires_acknowledgement: StrictBool = False
    warnings: list[VNPlaySetupWarning] = Field(default_factory=list)


class VNPlaySetupAssetPackOption(BaseModel):
    """Asset pack setup option with backend-computed readiness and warnings."""

    id: StrictInt
    title: StrictStr
    primary_character_id: StrictInt
    content_rating: StrictStr
    status: StrictStr
    trust_level: VNPlaySetupTrustLevel
    trust_source: VNPlaySetupTrustSource
    ready: StrictBool
    readiness_status: StrictStr
    readiness_warnings: list[str] = Field(default_factory=list)
    readiness_errors: list[str] = Field(default_factory=list)
    compatibility: VNPlaySetupCompatibility
    warning_summary: VNPlaySetupWarningSummary
    recommended: StrictBool = False


class VNPlaySetupScriptVersionOption(BaseModel):
    """Published script version option with backend-computed readiness."""

    id: StrictInt
    script_id: StrictInt
    title: StrictStr
    version_number: StrictInt
    label: str | None = None
    asset_pack_id: StrictInt
    manifest_snapshot_id: StrictInt
    policy_snapshot_id: StrictInt
    generation_profile_snapshot_id: StrictInt
    policy_profile_id: StrictStr
    generation_profile_id: StrictStr
    content_rating: StrictStr
    ready: StrictBool
    warning_summary: VNPlaySetupWarningSummary
    recommended: StrictBool = False


class VNPlaySetupDefaults(BaseModel):
    """Optional unambiguous defaults for creating a VN Play session."""

    mode: VNPlayMode | None = None
    character_id: int | None = None
    asset_pack_id: int | None = None
    script_id: int | None = None
    script_version_id: int | None = None
    policy_profile_id: str | None = None
    generation_profile_id: str | None = None
    content_rating: str | None = None


class VNPlaySetupPagination(BaseModel):
    """Pagination metadata for setup option selectors."""

    limit: StrictInt = Field(..., ge=1)
    offset: StrictInt = Field(..., ge=0)
    has_more: StrictBool
    total: int | None = Field(default=None, ge=0)


class VNPlaySetupEmptyState(BaseModel):
    """Scoped empty state hint for setup clients."""

    code: StrictStr
    scope: VNPlaySetupEmptyStateScope
    message: StrictStr


class VNPlaySetupPaginationSet(BaseModel):
    """Pagination metadata for both setup selectors."""

    characters: VNPlaySetupPagination
    asset_packs: VNPlaySetupPagination


class VNPlaySetupOptionsResponse(BaseModel):
    """Aggregated VN Play setup options for API and custom frontend clients."""

    characters: list[VNPlaySetupCharacterOption] = Field(default_factory=list)
    selected_character: VNPlaySetupCharacterOption | None = None
    asset_packs: list[VNPlaySetupAssetPackOption] = Field(default_factory=list)
    script_versions: list[VNPlaySetupScriptVersionOption] = Field(default_factory=list)
    defaults: VNPlaySetupDefaults = Field(default_factory=VNPlaySetupDefaults)
    pagination: VNPlaySetupPaginationSet
    empty_states: list[VNPlaySetupEmptyState] = Field(default_factory=list)
    generated_at: StrictStr


class VNPlayTurnRequest(BaseModel):
    """Request body for advancing a VN Play session by one turn."""

    model_config = ConfigDict(extra="forbid")

    input_text: StrictStr | None = Field(default=None, min_length=1)
    choice_id: StrictStr | None = Field(default=None, min_length=1)
    custom_action: dict[str, Any] | None = None
    client_scene_version: StrictInt = Field(..., ge=0)
    idempotency_key: StrictStr = Field(..., min_length=1, max_length=200)
    provider: StrictStr | None = Field(default=None, min_length=1)
    model: StrictStr | None = Field(default=None, min_length=1)
    options: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _require_exactly_one_input(self) -> VNPlayTurnRequest:
        provided = [
            self.input_text is not None,
            self.choice_id is not None,
            self.custom_action is not None,
        ]
        if sum(provided) != 1:
            raise ValueError("exactly_one_turn_input_required")
        return self


class VNPlayTurnResponse(BaseModel):
    """Response for a submitted VN Play turn."""

    model_config = ConfigDict(from_attributes=True)

    turn_request_id: StrictInt
    status: VNPlayTurnStatus
    scene_version: StrictInt = Field(..., ge=0)
    replayed: bool = False
    session: VNPlaySessionResponse | None = None
    current_scene: VNPlaySceneStateResponse | None = None
    events: list[VNPlayEventResponse] = Field(default_factory=list)
    warnings: list[Any] = Field(default_factory=list)
    error_code: str | None = None
    error_message: str | None = None


class VNPlayCheckpointCreate(BaseModel):
    """Request body for creating a named VN Play checkpoint."""

    model_config = ConfigDict(extra="forbid")

    label: StrictStr = Field(..., min_length=1, max_length=300)
    event_id: StrictInt | None = Field(default=None, ge=1)
    scene_version: StrictInt | None = Field(default=None, ge=0)


class VNPlayCheckpointResponse(BaseModel):
    """Serialized VN Play checkpoint."""

    model_config = ConfigDict(from_attributes=True)

    id: StrictInt
    session_id: StrictInt
    owner_user_id: StrictInt
    label: str
    event_id: int | None = None
    scene_version: StrictInt
    scene_state_snapshot: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None


class VNPlaySaveSlotCreate(BaseModel):
    """Request body for creating or overwriting a VN Play save slot."""

    model_config = ConfigDict(extra="forbid")

    slot_key: StrictStr = Field(..., min_length=1, max_length=120)
    title: StrictStr = Field(..., min_length=1, max_length=300)
    metadata: dict[str, Any] = Field(default_factory=dict)
    idempotency_key: StrictStr = Field(..., min_length=1, max_length=200)


class VNPlaySaveSlotUpdate(BaseModel):
    """Request body for updating save slot metadata."""

    model_config = ConfigDict(extra="forbid")

    title: StrictStr | None = Field(default=None, min_length=1, max_length=300)
    metadata: dict[str, Any] | None = None


class VNPlaySaveSlotRestoreRequest(BaseModel):
    """Request body for restoring a VN Play save slot."""

    model_config = ConfigDict(extra="forbid")

    client_scene_version: StrictInt = Field(..., ge=0)
    idempotency_key: StrictStr = Field(..., min_length=1, max_length=200)


class VNPlaySaveSlotResponse(BaseModel):
    """Serialized mutable save slot pointer."""

    model_config = ConfigDict(from_attributes=True)

    id: StrictInt
    session_id: StrictInt
    owner_user_id: StrictInt
    slot_key: StrictStr
    title: StrictStr
    checkpoint_id: StrictInt
    metadata: dict[str, Any] = Field(default_factory=dict)
    deleted: StrictBool = False
    replayed: StrictBool = False
    created_at: str | None = None
    updated_at: str | None = None


class VNPlayScriptActionRequest(BaseModel):
    """Request body for idempotent scripted-story runtime actions."""

    model_config = ConfigDict(extra="forbid")

    client_scene_version: StrictInt = Field(..., ge=0)
    idempotency_key: StrictStr = Field(..., min_length=1, max_length=200)


class VNPlayScriptStateResponse(BaseModel):
    """Spoiler-safe scripted-story runtime state."""

    session_id: StrictInt
    scene_version: StrictInt = Field(..., ge=0)
    position: dict[str, Any] = Field(default_factory=dict)
    variables: dict[str, Any] = Field(default_factory=dict)
    waiting_choice: dict[str, Any] | None = None
    waiting_generation_confirmation: dict[str, Any] | None = None
    active_generation: dict[str, Any] | None = None
    ended: StrictBool = False


class VNPlayScriptDebugStateResponse(BaseModel):
    """Owner-visible scripted-story debug state with pinned script metadata."""

    session_id: StrictInt
    scene_version: StrictInt = Field(..., ge=0)
    position: dict[str, Any] = Field(default_factory=dict)
    variables: dict[str, Any] = Field(default_factory=dict)
    waiting_choice: dict[str, Any] | None = None
    ended: StrictBool = False
    script_id: StrictInt | None = None
    script_version_id: StrictInt | None = None
    script_manifest_snapshot_id: StrictInt | None = None
    script_policy_snapshot_id: StrictInt | None = None
    script_generation_profile_snapshot_id: StrictInt | None = None
    version_number: StrictInt | None = None
    version_label: StrictStr | None = None
    program: dict[str, Any] = Field(default_factory=dict)
    script_defaults: dict[str, Any] = Field(default_factory=dict)
    validation: Any | None = None


class VNPlayScriptActionResponse(BaseModel):
    """Response for scripted-story runtime actions."""

    status: StrictStr
    replayed: StrictBool = False
    scene_version: StrictInt = Field(..., ge=0)
    session: VNPlaySessionResponse
    current_scene: VNPlaySceneStateResponse | None = None
    script_state: VNPlayScriptStateResponse
    events: list[VNPlayEventResponse] = Field(default_factory=list)
    warnings: list[Any] = Field(default_factory=list)


class VNPlayGenerationActionRequest(BaseModel):
    """Request body for idempotent scripted generation commands."""

    model_config = ConfigDict(extra="forbid")

    client_scene_version: StrictInt = Field(..., ge=0)
    idempotency_key: StrictStr = Field(..., min_length=1, max_length=200)


class VNPlayGenerationProfileSummary(BaseModel):
    """Owner-safe generation profile lineage summary."""

    profile_key: StrictStr
    snapshot_id: StrictInt
    provider_class: StrictStr | None = None
    moderation_required: StrictBool | None = None
    estimated_cost_class: StrictStr | None = None


class VNPlayGenerationHistoryItem(BaseModel):
    """Owner-safe generation revision history item."""

    id: StrictInt
    generation_id: StrictInt
    generation_point_key: StrictStr
    revision_number: StrictInt
    status: StrictStr
    active: StrictBool = False
    output_schema: StrictStr
    public_output: dict[str, Any] = Field(default_factory=dict)
    applied_visuals: list[dict[str, Any]] = Field(default_factory=list)
    rejected_visuals: list[dict[str, Any]] = Field(default_factory=list)
    public_error_code: StrictStr | None = None
    source: StrictStr = "model"
    profile: VNPlayGenerationProfileSummary
    created_at: StrictStr | None = None


class VNPlayGenerationHistoryResponse(BaseModel):
    """Offset-paginated owner-safe generation revision history."""

    items: list[VNPlayGenerationHistoryItem] = Field(default_factory=list)
    pagination: OffsetPaginationMeta
    total: StrictInt | None = Field(default=None, ge=0)
    limit: StrictInt | None = Field(default=None, ge=1)
    offset: StrictInt | None = Field(default=None, ge=0)
    has_more: StrictBool | None = None
    next_offset: StrictInt | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _validate_aliases(self) -> VNPlayGenerationHistoryResponse:
        validate_offset_pagination_aliases(self)
        return self


class VNPlayGenerationRevisionListResponse(BaseModel):
    """Offset-paginated owner-safe revision list for one generation point."""

    items: list[VNPlayGenerationHistoryItem] = Field(default_factory=list)
    pagination: OffsetPaginationMeta
    total: StrictInt | None = Field(default=None, ge=0)
    limit: StrictInt | None = Field(default=None, ge=1)
    offset: StrictInt | None = Field(default=None, ge=0)
    has_more: StrictBool | None = None
    next_offset: StrictInt | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _validate_aliases(self) -> VNPlayGenerationRevisionListResponse:
        validate_offset_pagination_aliases(self)
        return self


class VNPlayGenerationRevisionDebugResponse(BaseModel):
    """Owner/admin-only generation revision diagnostics."""

    id: StrictInt
    generation_id: StrictInt
    generation_request_id: StrictInt
    generation_point_key: StrictStr
    revision_number: StrictInt
    status: StrictStr
    output_schema: StrictStr
    public_output: dict[str, Any] = Field(default_factory=dict)
    raw_output_debug_state: VNPlayGenerationRawDebugState = "absent"
    raw_output_debug: dict[str, Any] | None = None
    parser_diagnostics: dict[str, Any] = Field(default_factory=dict)
    moderation_diagnostics: dict[str, Any] = Field(default_factory=dict)
    model_metadata: dict[str, Any] = Field(default_factory=dict)
    usage_metadata: dict[str, Any] = Field(default_factory=dict)
    request: dict[str, Any] = Field(default_factory=dict)
    profile: VNPlayGenerationProfileSummary
    created_at: StrictStr | None = None


class VNPlayRestoreRequest(BaseModel):
    """Request body for restoring a VN Play checkpoint."""

    model_config = ConfigDict(extra="forbid")

    checkpoint_id: StrictInt = Field(..., ge=1)
    client_scene_version: StrictInt = Field(..., ge=0)
    idempotency_key: StrictStr = Field(..., min_length=1, max_length=200)


class VNPlayBranchWarning(BaseModel):
    """Stable branch-navigation warning payload."""

    code: StrictStr
    severity: VNPlayBranchWarningSeverity = "warning"
    recoverable: StrictBool = True
    message: StrictStr | None = None
    branch_id: StrictInt | None = None
    event_id: StrictInt | None = None


class VNPlayBranchEventRange(BaseModel):
    """Event interval covered by a branch node."""

    start_event_id: StrictInt | None = None
    start_sequence_number: StrictInt | None = None
    latest_event_id: StrictInt | None = None
    latest_sequence_number: StrictInt | None = None


class VNPlayBranchRestoreCapability(BaseModel):
    """Restore targets available for a branch node."""

    supported: StrictBool
    default_target: VNPlayBranchRestoreTarget | None = None
    target_names: list[VNPlayBranchRestoreTarget] = Field(default_factory=list)
    targets: dict[str, dict[str, StrictInt | None] | None] = Field(default_factory=dict)


class VNPlayBranchPathStep(BaseModel):
    """One step in a VN Play branch path."""

    branch_id: StrictInt
    branch_label: StrictStr | None = None
    choice_id: StrictStr | None = None
    choice_text: StrictStr | None = None
    depth: StrictInt = Field(..., ge=0)


class VNPlayBranchNavigationNode(BaseModel):
    """Derived navigation data for one branch node."""

    branch_id: StrictInt
    parent_branch_id: StrictInt | None = None
    parent_event_id: StrictInt | None = None
    choice_selected_event_id: StrictInt | None = None
    branch_label: StrictStr | None = None
    choice_id: StrictStr | None = None
    choice_text: StrictStr | None = None
    branch_path: list[dict[str, Any]] = Field(default_factory=list)
    depth: StrictInt = Field(..., ge=0)
    status: StrictStr = "active"
    is_active: StrictBool = False
    is_on_active_path: StrictBool = False
    event_range: VNPlayBranchEventRange = Field(default_factory=VNPlayBranchEventRange)
    subtree_event_range: VNPlayBranchEventRange = Field(default_factory=VNPlayBranchEventRange)
    restore: VNPlayBranchRestoreCapability
    warnings: list[VNPlayBranchWarning] = Field(default_factory=list)


class VNPlayBranchNavigationResponse(BaseModel):
    """Derived branch navigation read model for a VN Play session."""

    session_id: StrictInt
    mode: VNPlayMode
    scene_version: StrictInt = Field(..., ge=0)
    last_event_id: StrictInt | None = None
    active_branch_node_id: StrictInt | None = None
    active_path: list[VNPlayBranchPathStep] = Field(default_factory=list)
    branches: list[VNPlayBranchNavigationNode] = Field(default_factory=list)
    warnings: list[VNPlayBranchWarning] = Field(default_factory=list)


class VNPlayBranchRestoreRequest(BaseModel):
    """Request body for restoring a VN Play branch."""

    model_config = ConfigDict(extra="forbid")

    client_scene_version: StrictInt = Field(..., ge=0)
    idempotency_key: StrictStr = Field(..., min_length=1, max_length=200)
    target: VNPlayBranchRestoreTarget = "branch_latest"


class VNPlayBranchRestoreResponse(BaseModel):
    """Response for branch restore operations."""

    status: StrictStr
    replayed: StrictBool = False
    restore_event_id: StrictInt
    target_event_id: StrictInt | None = None
    scene_version: StrictInt = Field(..., ge=0)
    session: VNPlaySessionResponse
    current_scene: VNPlaySceneStateResponse
    branch_navigation: VNPlayBranchNavigationResponse
    branch_id: StrictInt | None = None
    checkpoint_id: StrictInt | None = None
    save_slot_id: StrictInt | None = None
    target: VNPlayBranchRestoreTarget | None = None


class VNPlayRetryTurnRequest(BaseModel):
    """Request body for retrying the last VN Play user turn."""

    model_config = ConfigDict(extra="forbid")

    client_scene_version: StrictInt = Field(..., ge=0)
    idempotency_key: StrictStr = Field(..., min_length=1, max_length=200)


class VNPlayBranchResponse(BaseModel):
    """Serialized VN Play branch node."""

    model_config = ConfigDict(from_attributes=True)

    id: StrictInt
    session_id: StrictInt
    owner_user_id: StrictInt
    parent_event_id: int | None = None
    branch_label: str | None = None
    branch_path: list[Any] = Field(default_factory=list)
    status: str = "active"
    created_at: str | None = None
    updated_at: str | None = None


__all__ = [
    "VNPlayBranchEventRange",
    "VNPlayBranchNavigationNode",
    "VNPlayBranchNavigationResponse",
    "VNPlayBranchPathStep",
    "VNPlayBranchRestoreCapability",
    "VNPlayBranchRestoreRequest",
    "VNPlayBranchRestoreResponse",
    "VNPlayBranchRestoreTarget",
    "VNPlayBranchResponse",
    "VNPlayBranchWarning",
    "VNPlayBranchWarningSeverity",
    "VNPlayCheckpointCreate",
    "VNPlayCheckpointResponse",
    "VNPlayEventResponse",
    "VNPlayRestoreRequest",
    "VNPlayRetryTurnRequest",
    "VNPlaySaveSlotCreate",
    "VNPlaySaveSlotResponse",
    "VNPlaySaveSlotRestoreRequest",
    "VNPlaySaveSlotUpdate",
    "VNPlaySceneStateResponse",
    "VNPlayScriptActionRequest",
    "VNPlayScriptActionResponse",
    "VNPlayGenerationActionRequest",
    "VNPlayGenerationHistoryItem",
    "VNPlayGenerationHistoryResponse",
    "VNPlayGenerationProfileSummary",
    "VNPlayGenerationRawDebugState",
    "VNPlayGenerationRevisionListResponse",
    "VNPlayGenerationRevisionDebugResponse",
    "VNPlayScriptStateResponse",
    "VNPlaySessionCreate",
    "VNPlaySessionResponse",
    "VNPlaySessionUpdate",
    "VNPlaySetupAssetPackOption",
    "VNPlaySetupCharacterOption",
    "VNPlaySetupCompatibility",
    "VNPlaySetupDefaults",
    "VNPlaySetupEmptyState",
    "VNPlaySetupEmptyStateScope",
    "VNPlaySetupOptionsResponse",
    "VNPlaySetupPagination",
    "VNPlaySetupPaginationSet",
    "VNPlaySetupScriptVersionOption",
    "VNPlaySetupTrustLevel",
    "VNPlaySetupTrustSource",
    "VNPlaySetupWarning",
    "VNPlaySetupWarningSeverity",
    "VNPlaySetupWarningSummary",
    "VNPlayTurnRequest",
    "VNPlayTurnResponse",
    "VNPlayEventType",
    "VNPlayLinkedChatMode",
    "VNPlayMode",
    "VNPlaySessionStatus",
    "VNPlayTrustLevel",
    "VNPlayTurnStatus",
]
