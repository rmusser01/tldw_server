from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ScopeType = Literal["global", "org", "team", "user"]
CapabilityAdapterScopeType = Literal["global", "org", "team"]
ProfileMode = Literal["preset", "custom"]
AssignmentTargetType = Literal["default", "group", "persona"]
PolicyProvenanceSourceKind = Literal[
    "profile",
    "profile_path_scope_object",
    "assignment_path_scope_object",
    "assignment_inline",
    "assignment_override",
    "capability_mapping",
    "runtime_constraint",
]
PolicyProvenanceEffect = Literal["merged", "replaced", "narrowed", "blocked"]
ApprovalMode = Literal[
    "allow_silently",
    "ask_every_time",
    "ask_outside_profile",
    "ask_on_sensitive_actions",
    "temporary_elevation_allowed",
]
ApprovalDecision = Literal["approved", "denied"]
ApprovalDuration = Literal["once", "session", "conversation"]
ToolRiskClass = Literal["low", "medium", "high", "unclassified"]
ToolMetadataSource = Literal["explicit", "heuristic", "fallback"]
PathScopeMode = Literal["none", "workspace_root", "cwd_descendants"]
PathScopeEnforcement = Literal["approval_required_when_unenforceable"]
WorkspaceSourceMode = Literal["inline", "named"]
WorkspaceTrustSource = Literal["user_local", "shared_registry"]
ExternalAuthTemplateTargetType = Literal["header", "env"]
CredentialSlotPrivilegeClass = Literal["read", "write", "admin"]
TrustSignerBindingStatus = Literal["active", "inactive", "revoked"]


class ACPProfileCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=512)
    owner_scope_type: ScopeType = Field(default="global")
    owner_scope_id: int | None = None
    profile: dict[str, Any] = Field(default_factory=dict)
    is_active: bool = True


class ACPProfileUpdateRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=512)
    owner_scope_type: ScopeType | None = None
    owner_scope_id: int | None = None
    profile: dict[str, Any] | None = None
    is_active: bool | None = None


class ACPProfileResponse(BaseModel):
    id: int
    name: str
    description: str | None = None
    owner_scope_type: ScopeType
    owner_scope_id: int | None = None
    profile: dict[str, Any] = Field(default_factory=dict)
    is_active: bool
    created_by: int | None = None
    updated_by: int | None = None
    created_at: datetime | str | None = None
    updated_at: datetime | str | None = None


class PermissionProfileCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=512)
    owner_scope_type: ScopeType = Field(default="global")
    owner_scope_id: int | None = None
    mode: ProfileMode = "custom"
    path_scope_object_id: int | None = None
    policy_document: dict[str, Any] = Field(default_factory=dict)
    is_active: bool = True


class PermissionProfileUpdateRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=512)
    owner_scope_type: ScopeType | None = None
    owner_scope_id: int | None = None
    mode: ProfileMode | None = None
    path_scope_object_id: int | None = None
    policy_document: dict[str, Any] | None = None
    is_active: bool | None = None


class PermissionProfileResponse(BaseModel):
    id: int
    name: str
    description: str | None = None
    owner_scope_type: ScopeType
    owner_scope_id: int | None = None
    mode: ProfileMode
    path_scope_object_id: int | None = None
    policy_document: dict[str, Any] = Field(default_factory=dict)
    is_active: bool
    is_immutable: bool = False
    created_by: int | None = None
    updated_by: int | None = None
    created_at: datetime | str | None = None
    updated_at: datetime | str | None = None


class PathScopeObjectCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=512)
    owner_scope_type: ScopeType = Field(default="global")
    owner_scope_id: int | None = None
    path_scope_document: dict[str, Any] = Field(default_factory=dict)
    is_active: bool = True


class PathScopeObjectUpdateRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=512)
    owner_scope_type: ScopeType | None = None
    owner_scope_id: int | None = None
    path_scope_document: dict[str, Any] | None = None
    is_active: bool | None = None


class PathScopeObjectResponse(BaseModel):
    id: int
    name: str
    description: str | None = None
    owner_scope_type: ScopeType
    owner_scope_id: int | None = None
    path_scope_document: dict[str, Any] = Field(default_factory=dict)
    is_active: bool
    created_by: int | None = None
    updated_by: int | None = None
    created_at: datetime | str | None = None
    updated_at: datetime | str | None = None


class PolicyAssignmentCreateRequest(BaseModel):
    target_type: AssignmentTargetType
    target_id: str | None = Field(default=None, max_length=200)
    owner_scope_type: ScopeType = Field(default="global")
    owner_scope_id: int | None = None
    profile_id: int | None = None
    path_scope_object_id: int | None = None
    workspace_source_mode: WorkspaceSourceMode | None = None
    workspace_set_object_id: int | None = None
    inline_policy_document: dict[str, Any] = Field(default_factory=dict)
    approval_policy_id: int | None = None
    is_active: bool = True


class PolicyAssignmentUpdateRequest(BaseModel):
    target_type: AssignmentTargetType | None = None
    target_id: str | None = Field(default=None, max_length=200)
    owner_scope_type: ScopeType | None = None
    owner_scope_id: int | None = None
    profile_id: int | None = None
    path_scope_object_id: int | None = None
    workspace_source_mode: WorkspaceSourceMode | None = None
    workspace_set_object_id: int | None = None
    inline_policy_document: dict[str, Any] | None = None
    approval_policy_id: int | None = None
    is_active: bool | None = None


class PolicyAssignmentResponse(BaseModel):
    id: int
    target_type: AssignmentTargetType
    target_id: str | None = None
    owner_scope_type: ScopeType
    owner_scope_id: int | None = None
    profile_id: int | None = None
    path_scope_object_id: int | None = None
    workspace_source_mode: WorkspaceSourceMode | None = None
    workspace_set_object_id: int | None = None
    inline_policy_document: dict[str, Any] = Field(default_factory=dict)
    approval_policy_id: int | None = None
    is_active: bool
    is_immutable: bool = False
    has_override: bool = False
    override_id: int | None = None
    override_active: bool = False
    override_updated_at: datetime | str | None = None
    created_by: int | None = None
    updated_by: int | None = None
    created_at: datetime | str | None = None
    updated_at: datetime | str | None = None


class PolicyAssignmentWorkspaceCreateRequest(BaseModel):
    workspace_id: str = Field(..., min_length=1, max_length=255)


class PolicyAssignmentWorkspaceResponse(BaseModel):
    assignment_id: int
    workspace_id: str
    created_by: int | None = None
    created_at: datetime | str | None = None


class WorkspaceSetObjectCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=512)
    owner_scope_type: ScopeType = Field(default="user")
    owner_scope_id: int | None = None
    is_active: bool = True


class WorkspaceSetObjectUpdateRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=512)
    owner_scope_type: ScopeType | None = None
    owner_scope_id: int | None = None
    is_active: bool | None = None


class WorkspaceSetObjectResponse(BaseModel):
    id: int
    name: str
    description: str | None = None
    owner_scope_type: ScopeType
    owner_scope_id: int | None = None
    is_active: bool
    readiness_summary: WorkspaceSourceReadinessSummaryResponse | None = None
    created_by: int | None = None
    updated_by: int | None = None
    created_at: datetime | str | None = None
    updated_at: datetime | str | None = None


class WorkspaceSetObjectMemberCreateRequest(BaseModel):
    workspace_id: str = Field(..., min_length=1, max_length=255)


class WorkspaceSetObjectMemberResponse(BaseModel):
    workspace_set_object_id: int
    workspace_id: str
    created_by: int | None = None
    created_at: datetime | str | None = None


class SharedWorkspaceCreateRequest(BaseModel):
    workspace_id: str = Field(..., min_length=1, max_length=255)
    display_name: str = Field(..., min_length=1, max_length=200)
    absolute_root: str = Field(..., min_length=1, max_length=1024)
    owner_scope_type: ScopeType = Field(default="team")
    owner_scope_id: int | None = None
    is_active: bool = True


class SharedWorkspaceUpdateRequest(BaseModel):
    workspace_id: str | None = Field(default=None, min_length=1, max_length=255)
    display_name: str | None = Field(default=None, min_length=1, max_length=200)
    absolute_root: str | None = Field(default=None, min_length=1, max_length=1024)
    owner_scope_type: ScopeType | None = None
    owner_scope_id: int | None = None
    is_active: bool | None = None


class SharedWorkspaceResponse(BaseModel):
    id: int
    workspace_id: str
    display_name: str
    absolute_root: str
    owner_scope_type: Literal["global", "org", "team"]
    owner_scope_id: int | None = None
    is_active: bool
    readiness_summary: WorkspaceSourceReadinessSummaryResponse | None = None
    created_by: int | None = None
    updated_by: int | None = None
    created_at: datetime | str | None = None
    updated_at: datetime | str | None = None


class WorkspaceSourceReadinessSummaryResponse(BaseModel):
    is_multi_root_ready: bool = True
    warning_codes: list[str] = Field(default_factory=list)
    warning_message: str | None = None
    conflicting_workspace_ids: list[str] = Field(default_factory=list)
    conflicting_workspace_roots: list[str] = Field(default_factory=list)
    unresolved_workspace_ids: list[str] = Field(default_factory=list)


class GovernanceAuditNavigateTargetResponse(BaseModel):
    tab: str
    object_kind: str
    object_id: str


class GovernanceAuditFindingResponse(BaseModel):
    finding_type: str
    severity: str
    scope_type: ScopeType
    scope_id: int | None = None
    object_kind: str
    object_id: str
    object_label: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
    navigate_to: GovernanceAuditNavigateTargetResponse
    related_object_kind: str | None = None
    related_object_id: str | None = None
    related_object_label: str | None = None


class GovernanceAuditCountsResponse(BaseModel):
    error: int = 0
    warning: int = 0


class GovernanceAuditFindingListResponse(BaseModel):
    items: list[GovernanceAuditFindingResponse] = Field(default_factory=list)
    total: int = 0
    counts: GovernanceAuditCountsResponse = Field(default_factory=GovernanceAuditCountsResponse)


class PolicyOverrideUpsertRequest(BaseModel):
    override_policy_document: dict[str, Any] = Field(default_factory=dict)
    is_active: bool = True


class PolicyOverrideResponse(BaseModel):
    id: int
    assignment_id: int
    override_policy_document: dict[str, Any] = Field(default_factory=dict)
    is_active: bool
    created_by: int | None = None
    updated_by: int | None = None
    created_at: datetime | str | None = None
    updated_at: datetime | str | None = None


class ApprovalPolicyCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=512)
    owner_scope_type: ScopeType = Field(default="global")
    owner_scope_id: int | None = None
    mode: ApprovalMode
    rules: dict[str, Any] = Field(default_factory=dict)
    is_active: bool = True


class ApprovalPolicyUpdateRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=512)
    owner_scope_type: ScopeType | None = None
    owner_scope_id: int | None = None
    mode: ApprovalMode | None = None
    rules: dict[str, Any] | None = None
    is_active: bool | None = None


class ApprovalPolicyResponse(BaseModel):
    id: int
    name: str
    description: str | None = None
    owner_scope_type: ScopeType
    owner_scope_id: int | None = None
    mode: ApprovalMode
    rules: dict[str, Any] = Field(default_factory=dict)
    is_active: bool
    is_immutable: bool = False
    created_by: int | None = None
    updated_by: int | None = None
    created_at: datetime | str | None = None
    updated_at: datetime | str | None = None


class ApprovalDecisionCreateRequest(BaseModel):
    approval_policy_id: int | None = None
    context_key: str = Field(..., min_length=1, max_length=255)
    conversation_id: str | None = Field(default=None, max_length=255)
    tool_name: str = Field(..., min_length=1, max_length=255)
    scope_key: str = Field(..., min_length=1, max_length=255)
    decision: ApprovalDecision
    duration: ApprovalDuration = "once"


class ApprovalDecisionResponse(BaseModel):
    id: int
    approval_policy_id: int | None = None
    context_key: str
    conversation_id: str | None = None
    tool_name: str
    scope_key: str
    decision: ApprovalDecision
    consume_on_match: bool = False
    expires_at: datetime | str | None = None
    consumed_at: datetime | str | None = None
    created_by: int | None = None
    created_at: datetime | str | None = None


class CapabilityAdapterMappingCreateRequest(BaseModel):
    mapping_id: str = Field(..., min_length=1, max_length=200)
    title: str | None = Field(default=None, max_length=200)
    description: str | None = Field(default=None, max_length=512)
    owner_scope_type: CapabilityAdapterScopeType = Field(default="global")
    owner_scope_id: int | None = None
    capability_name: str = Field(..., min_length=1, max_length=200)
    adapter_contract_version: int = Field(default=1, ge=1)
    resolved_policy_document: dict[str, Any] = Field(default_factory=dict)
    supported_environment_requirements: list[str] = Field(default_factory=list)
    is_active: bool = True


class CapabilityAdapterMappingUpdateRequest(BaseModel):
    mapping_id: str | None = Field(default=None, min_length=1, max_length=200)
    title: str | None = Field(default=None, max_length=200)
    description: str | None = Field(default=None, max_length=512)
    owner_scope_type: CapabilityAdapterScopeType | None = None
    owner_scope_id: int | None = None
    capability_name: str | None = Field(default=None, min_length=1, max_length=200)
    adapter_contract_version: int | None = Field(default=None, ge=1)
    resolved_policy_document: dict[str, Any] | None = None
    supported_environment_requirements: list[str] | None = None
    is_active: bool | None = None


class CapabilityAdapterMappingResponse(BaseModel):
    id: int
    mapping_id: str
    title: str
    description: str | None = None
    owner_scope_type: CapabilityAdapterScopeType
    owner_scope_id: int | None = None
    capability_name: str
    adapter_contract_version: int
    resolved_policy_document: dict[str, Any] = Field(default_factory=dict)
    supported_environment_requirements: list[str] = Field(default_factory=list)
    is_active: bool
    created_by: int | None = None
    updated_by: int | None = None
    created_at: datetime | str | None = None
    updated_at: datetime | str | None = None


class CapabilityAdapterMappingNormalizedResponse(BaseModel):
    mapping_id: str
    title: str
    description: str | None = None
    owner_scope_type: CapabilityAdapterScopeType
    owner_scope_id: int | None = None
    capability_name: str
    adapter_contract_version: int
    resolved_policy_document: dict[str, Any] = Field(default_factory=dict)
    supported_environment_requirements: list[str] = Field(default_factory=list)
    is_active: bool


class CapabilityAdapterMappingScopeSummaryResponse(BaseModel):
    owner_scope_type: CapabilityAdapterScopeType
    owner_scope_id: int | None = None
    display_scope: str


class CapabilityAdapterMappingPreviewResponse(BaseModel):
    normalized_mapping: CapabilityAdapterMappingNormalizedResponse
    warnings: list[str] = Field(default_factory=list)
    affected_scope_summary: CapabilityAdapterMappingScopeSummaryResponse


class EffectivePolicySourceResponse(BaseModel):
    assignment_id: int
    target_type: AssignmentTargetType
    target_id: str | None = None
    owner_scope_type: ScopeType
    owner_scope_id: int | None = None
    profile_id: int | None = None
    path_scope_object_id: int | None = None


class EffectivePolicyProvenanceResponse(BaseModel):
    field: str
    value: Any
    source_kind: PolicyProvenanceSourceKind
    assignment_id: int | None = None
    profile_id: int | None = None
    override_id: int | None = None
    capability_name: str | None = None
    mapping_id: str | None = None
    mapping_scope_type: CapabilityAdapterScopeType | None = None
    mapping_scope_id: int | None = None
    resolved_effects: dict[str, Any] = Field(default_factory=dict)
    resolution_intent: Literal["allow", "deny"] | None = None
    effect: PolicyProvenanceEffect


class EffectivePolicyCapabilityMappingResponse(BaseModel):
    capability_name: str
    resolution_intent: Literal["allow", "deny"] | None = None
    mapping_id: str | None = None
    mapping_scope_type: CapabilityAdapterScopeType | None = None
    mapping_scope_id: int | None = None
    resolved_effects: dict[str, Any] = Field(default_factory=dict)
    supported_environment_requirements: list[str] = Field(default_factory=list)
    unsupported_environment_requirements: list[str] = Field(default_factory=list)


class EffectivePolicyResponse(BaseModel):
    enabled: bool
    allowed_tools: list[str] = Field(default_factory=list)
    denied_tools: list[str] = Field(default_factory=list)
    capabilities: list[str] = Field(default_factory=list)
    approval_policy_id: int | None = None
    approval_mode: ApprovalMode | None = None
    policy_document: dict[str, Any] = Field(default_factory=dict)
    authored_policy_document: dict[str, Any] = Field(default_factory=dict)
    resolved_policy_document: dict[str, Any] = Field(default_factory=dict)
    resolved_capabilities: list[str] = Field(default_factory=list)
    unresolved_capabilities: list[str] = Field(default_factory=list)
    capability_mapping_summary: list[EffectivePolicyCapabilityMappingResponse] = Field(default_factory=list)
    capability_warnings: list[str] = Field(default_factory=list)
    supported_environment_requirements: list[str] = Field(default_factory=list)
    unsupported_environment_requirements: list[str] = Field(default_factory=list)
    selected_assignment_id: int | None = None
    selected_workspace_source_mode: WorkspaceSourceMode | None = None
    selected_workspace_set_object_id: int | None = None
    selected_workspace_set_object_name: str | None = None
    selected_workspace_trust_source: WorkspaceTrustSource | None = None
    selected_workspace_scope_type: ScopeType | None = None
    selected_workspace_scope_id: int | None = None
    selected_assignment_workspace_ids: list[str] = Field(default_factory=list)
    sources: list[EffectivePolicySourceResponse] = Field(default_factory=list)
    provenance: list[EffectivePolicyProvenanceResponse] = Field(default_factory=list)


class ToolRegistryEntryResponse(BaseModel):
    tool_name: str
    display_name: str
    description: str | None = None
    module: str
    module_display_name: str | None = None
    category: str
    risk_class: ToolRiskClass
    capabilities: list[str] = Field(default_factory=list)
    mutates_state: bool = False
    uses_filesystem: bool = False
    uses_processes: bool = False
    uses_network: bool = False
    uses_credentials: bool = False
    supports_arguments_preview: bool = False
    path_boundable: bool = False
    path_argument_hints: list[str] = Field(default_factory=list)
    metadata_source: ToolMetadataSource
    metadata_warnings: list[str] = Field(default_factory=list)


class ToolRegistryModuleResponse(BaseModel):
    module: str
    display_name: str
    tool_count: int
    risk_summary: dict[str, int] = Field(default_factory=dict)
    metadata_warnings: list[str] = Field(default_factory=list)


class ToolRegistrySummaryResponse(BaseModel):
    entries: list[ToolRegistryEntryResponse] = Field(default_factory=list)
    modules: list[ToolRegistryModuleResponse] = Field(default_factory=list)


class ExternalServerCreateRequest(BaseModel):
    server_id: str = Field(..., min_length=1, max_length=128)
    name: str = Field(..., min_length=1, max_length=200)
    transport: str = Field(..., min_length=1, max_length=64)
    config: dict[str, Any] = Field(default_factory=dict)
    owner_scope_type: ScopeType = Field(default="global")
    owner_scope_id: int | None = None
    enabled: bool = True


class ExternalServerUpdateRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    transport: str | None = Field(default=None, min_length=1, max_length=64)
    config: dict[str, Any] | None = None
    owner_scope_type: ScopeType | None = None
    owner_scope_id: int | None = None
    enabled: bool | None = None


class ExternalServerDiscoveryRefreshRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    server_id: str | None = Field(default=None, max_length=128)

    @field_validator("server_id")
    @classmethod
    def normalize_server_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("server_id must be a non-empty string")
        return normalized


class ExternalServerDiscoveryRefreshResponse(BaseModel):
    ok: bool
    server_id: str | None = None
    reconciled_servers: int = 0
    refreshed_servers: int = 0
    total_servers: int = 0
    virtual_tools: int = 0
    errors: dict[str, str] = Field(default_factory=dict)
    requires_restart: bool = False
    message: str | None = None


class ExternalServerCredentialSlotCreateRequest(BaseModel):
    slot_name: str = Field(..., min_length=1, max_length=128)
    display_name: str = Field(..., min_length=1, max_length=200)
    secret_kind: str = Field(..., min_length=1, max_length=64)
    privilege_class: CredentialSlotPrivilegeClass
    is_required: bool = True


class ExternalServerCredentialSlotUpdateRequest(BaseModel):
    display_name: str | None = Field(default=None, min_length=1, max_length=200)
    secret_kind: str | None = Field(default=None, min_length=1, max_length=64)
    privilege_class: CredentialSlotPrivilegeClass | None = None
    is_required: bool | None = None


class ExternalServerCredentialSlotResponse(BaseModel):
    server_id: str
    slot_name: str
    display_name: str
    secret_kind: str
    privilege_class: CredentialSlotPrivilegeClass
    is_required: bool
    secret_configured: bool = False
    created_by: int | None = None
    updated_by: int | None = None
    created_at: datetime | str | None = None
    updated_at: datetime | str | None = None


class ExternalServerAuthTemplateMappingRequest(BaseModel):
    slot_name: str = Field(..., min_length=1, max_length=128)
    target_type: ExternalAuthTemplateTargetType
    target_name: str = Field(..., min_length=1, max_length=256)
    prefix: str = Field(default="", max_length=512)
    suffix: str = Field(default="", max_length=512)
    required: bool = True


class ExternalServerAuthTemplateMappingResponse(BaseModel):
    slot_name: str
    target_type: ExternalAuthTemplateTargetType
    target_name: str
    prefix: str = ""
    suffix: str = ""
    required: bool = True


class ExternalServerAuthTemplateUpdateRequest(BaseModel):
    mode: Literal["template"] = "template"
    mappings: list[ExternalServerAuthTemplateMappingRequest] = Field(default_factory=list)


class ExternalServerAuthTemplateResponse(BaseModel):
    mode: Literal["template"] = "template"
    mappings: list[ExternalServerAuthTemplateMappingResponse] = Field(default_factory=list)


class ExternalServerResponse(BaseModel):
    id: str
    name: str
    enabled: bool
    owner_scope_type: ScopeType
    owner_scope_id: int | None = None
    transport: str
    config: dict[str, Any] = Field(default_factory=dict)
    secret_configured: bool
    key_hint: str | None = None
    server_source: str = "managed"
    legacy_source_ref: str | None = None
    superseded_by_server_id: str | None = None
    binding_count: int = 0
    runtime_executable: bool = True
    auth_template_present: bool = False
    auth_template_valid: bool = False
    auth_template_blocked_reason: str | None = None
    credential_slots: list[ExternalServerCredentialSlotResponse] = Field(default_factory=list)
    created_by: int | None = None
    updated_by: int | None = None
    created_at: datetime | str | None = None
    updated_at: datetime | str | None = None


class ExternalSecretSetRequest(BaseModel):
    secret: str = Field(..., min_length=1, max_length=8192)


class ExternalSecretSetResponse(BaseModel):
    server_id: str
    secret_configured: bool
    key_hint: str | None = None
    updated_at: datetime | str | None = None


class ExternalServerSlotSecretSetResponse(BaseModel):
    server_id: str
    slot_name: str
    secret_configured: bool
    key_hint: str | None = None
    updated_at: datetime | str | None = None


class CredentialBindingResponse(BaseModel):
    id: int
    binding_target_type: str
    binding_target_id: str
    external_server_id: str
    slot_name: str | None = None
    credential_ref: str
    managed_secret_ref_id: int | None = None
    binding_mode: str
    usage_rules: dict[str, Any] = Field(default_factory=dict)
    created_by: int | None = None
    updated_by: int | None = None
    created_at: datetime | str | None = None
    updated_at: datetime | str | None = None


class ProfileCredentialBindingUpsertRequest(BaseModel):
    managed_secret_ref_id: int | None = Field(default=None, ge=1)


class AssignmentCredentialBindingUpsertRequest(BaseModel):
    binding_mode: str = Field(default="grant", pattern="^(grant|disable)$")
    managed_secret_ref_id: int | None = Field(default=None, ge=1)


class McpCredentialSlotStatusResponse(BaseModel):
    server_id: str
    slot_name: str
    binding_target_type: Literal["profile", "assignment"]
    binding_target_id: str
    credential_ref: str
    managed_secret_ref_id: int | None = None
    state: Literal[
        "ready",
        "missing",
        "expired",
        "reauth_required",
        "approval_required",
        "backend_unavailable",
    ]
    blocked_reason: str | None = None
    backend_name: str | None = None
    expires_at: datetime | str | None = None


class EffectiveExternalAccessSlotResponse(BaseModel):
    slot_name: str
    display_name: str | None = None
    granted_by: str | None = None
    disabled_by_assignment: bool = False
    secret_available: bool = False
    runtime_usable: bool = False
    blocked_reason: str | None = None


class EffectiveExternalAccessEntryResponse(BaseModel):
    server_id: str
    server_name: str | None = None
    granted_by: str | None = None
    disabled_by_assignment: bool = False
    server_source: str = "managed"
    superseded_by_server_id: str | None = None
    secret_available: bool = False
    runtime_executable: bool = False
    blocked_reason: str | None = None
    requested_slots: list[str] = Field(default_factory=list)
    bound_slots: list[str] = Field(default_factory=list)
    missing_bound_slots: list[str] = Field(default_factory=list)
    missing_secret_slots: list[str] = Field(default_factory=list)
    slots: list[EffectiveExternalAccessSlotResponse] = Field(default_factory=list)


class EffectiveExternalAccessResponse(BaseModel):
    servers: list[EffectiveExternalAccessEntryResponse] = Field(default_factory=list)


class MCPHubDeleteResponse(BaseModel):
    ok: bool


class GovernancePackDocumentRequest(BaseModel):
    manifest: dict[str, Any] = Field(default_factory=dict)
    profiles: list[dict[str, Any]] = Field(default_factory=list)
    approvals: list[dict[str, Any]] = Field(default_factory=list)
    personas: list[dict[str, Any]] = Field(default_factory=list)
    assignments: list[dict[str, Any]] = Field(default_factory=list)


class GovernancePackDryRunRequest(BaseModel):
    owner_scope_type: ScopeType = Field(default="user")
    owner_scope_id: int | None = None
    pack: GovernancePackDocumentRequest


class GovernancePackImportRequest(BaseModel):
    owner_scope_type: ScopeType = Field(default="user")
    owner_scope_id: int | None = None
    pack: GovernancePackDocumentRequest


class GovernancePackSourceRequest(BaseModel):
    source_type: Literal["local_path", "git"]
    local_path: str | None = None
    repo_url: str | None = None
    ref: str | None = None
    ref_kind: Literal["branch", "tag", "commit"] | None = None
    subpath: str | None = None


class GovernancePackSourcePrepareRequest(BaseModel):
    source: GovernancePackSourceRequest


class GovernancePackSourceCandidateResponse(BaseModel):
    id: int
    source_type: str
    source_location: str
    source_ref_requested: str | None = None
    source_ref_kind: Literal["branch", "tag", "commit"] | None = None
    source_subpath: str | None = None
    source_commit_resolved: str | None = None
    pack_content_digest: str
    source_verified: bool | None = None
    source_verification_mode: str | None = None
    signer_fingerprint: str | None = None
    signer_identity: str | None = None
    verified_object_type: str | None = None
    verification_result_code: str | None = None
    verification_warning_code: str | None = None
    source_fetched_at: datetime | str | None = None
    fetched_by: int | None = None


class GovernancePackSourcePrepareResponse(BaseModel):
    candidate: GovernancePackSourceCandidateResponse
    manifest: GovernancePackReportManifestResponse


class GovernancePackSourceDryRunRequest(BaseModel):
    owner_scope_type: ScopeType = Field(default="user")
    owner_scope_id: int | None = None
    candidate_id: int = Field(..., ge=1)


class GovernancePackSourceImportRequest(BaseModel):
    owner_scope_type: ScopeType = Field(default="user")
    owner_scope_id: int | None = None
    candidate_id: int = Field(..., ge=1)


GovernancePackSourceUpdateStatus = Literal[
    "newer_version_available",
    "no_update",
    "source_drift_same_version",
]


class GovernancePackSourceUpdateCheckResponse(BaseModel):
    governance_pack_id: int
    status: GovernancePackSourceUpdateStatus
    installed_manifest: GovernancePackReportManifestResponse
    candidate_manifest: GovernancePackReportManifestResponse | None = None
    source_commit_resolved: str | None = None
    pack_content_digest: str | None = None
    signer_fingerprint: str | None = None
    signer_identity: str | None = None
    verified_object_type: str | None = None
    verification_result_code: str | None = None
    verification_warning_code: str | None = None


class GovernancePackSourceUpgradePrepareResponse(BaseModel):
    status: GovernancePackSourceUpdateStatus
    installed_manifest: GovernancePackReportManifestResponse
    candidate_manifest: GovernancePackReportManifestResponse | None = None
    candidate: GovernancePackSourceCandidateResponse
    manifest: GovernancePackReportManifestResponse
    signer_fingerprint: str | None = None
    signer_identity: str | None = None
    verified_object_type: str | None = None
    verification_result_code: str | None = None
    verification_warning_code: str | None = None


class GovernancePackSourceUpgradeDryRunRequest(BaseModel):
    source_governance_pack_id: int = Field(..., ge=1)
    owner_scope_type: ScopeType = Field(default="user")
    owner_scope_id: int | None = None
    candidate_id: int = Field(..., ge=1)


class GovernancePackSourceUpgradeExecuteRequest(BaseModel):
    source_governance_pack_id: int = Field(..., ge=1)
    owner_scope_type: ScopeType = Field(default="user")
    owner_scope_id: int | None = None
    candidate_id: int = Field(..., ge=1)
    planner_inputs_fingerprint: str = Field(..., min_length=1)
    adapter_state_fingerprint: str = Field(..., min_length=1)


class GovernancePackUpgradeDryRunRequest(BaseModel):
    """Request body for governance-pack upgrade planning."""

    source_governance_pack_id: int = Field(..., ge=1)
    owner_scope_type: ScopeType = Field(default="user")
    owner_scope_id: int | None = None
    pack: GovernancePackDocumentRequest


class GovernancePackReportManifestResponse(BaseModel):
    """Manifest summary returned in governance-pack dry-run reports."""

    pack_id: str
    pack_version: str
    title: str
    description: str | None = None


class GovernancePackDryRunReportResponse(BaseModel):
    """Portable governance-pack dry-run validation report."""

    manifest: GovernancePackReportManifestResponse
    digest: str
    resolved_capabilities: list[str] = Field(default_factory=list)
    unresolved_capabilities: list[str] = Field(default_factory=list)
    capability_mapping_summary: list[EffectivePolicyCapabilityMappingResponse] = Field(default_factory=list)
    supported_environment_requirements: list[str] = Field(default_factory=list)
    unsupported_environment_requirements: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    blocked_objects: list[str] = Field(default_factory=list)
    verdict: Literal["importable", "blocked"]


class GovernancePackDryRunResponse(BaseModel):
    """Envelope for governance-pack import dry-run responses."""

    report: GovernancePackDryRunReportResponse


class GovernancePackUpgradeObjectDiffResponse(BaseModel):
    """Object-level diff entry produced by governance-pack upgrade planning."""

    object_type: str
    source_object_id: str
    change_type: str
    previous_digest: str | None = None
    next_digest: str | None = None


class GovernancePackUpgradeDependencyImpactResponse(BaseModel):
    """Dependency impact entry produced by governance-pack upgrade planning."""

    object_type: str
    source_object_id: str
    change_type: str
    impact: str
    dependent_type: str
    dependent_id: int
    reference_field: str
    target_type: str | None = None
    target_id: str | None = None


class GovernancePackUpgradePlanResponse(BaseModel):
    """Planner output for a governance-pack upgrade dry run."""

    source_governance_pack_id: int
    source_manifest: dict[str, Any] = Field(default_factory=dict)
    target_manifest: dict[str, Any] = Field(default_factory=dict)
    object_diff: list[GovernancePackUpgradeObjectDiffResponse] = Field(default_factory=list)
    dependency_impact: list[GovernancePackUpgradeDependencyImpactResponse] = Field(default_factory=list)
    structural_conflicts: list[str] = Field(default_factory=list)
    behavioral_conflicts: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    planner_inputs_fingerprint: str
    adapter_state_fingerprint: str
    upgradeable: bool


class GovernancePackUpgradeDryRunResponse(BaseModel):
    """Envelope for governance-pack upgrade dry-run responses."""

    plan: GovernancePackUpgradePlanResponse


class GovernancePackUpgradeExecuteRequest(BaseModel):
    """Request body for transactional governance-pack upgrade execution."""

    source_governance_pack_id: int = Field(..., ge=1)
    owner_scope_type: ScopeType = Field(default="user")
    owner_scope_id: int | None = None
    planner_inputs_fingerprint: str = Field(..., min_length=1)
    adapter_state_fingerprint: str = Field(..., min_length=1)
    pack: GovernancePackDocumentRequest


class GovernancePackUpgradeExecutionResponse(BaseModel):
    """Execution result for a completed governance-pack upgrade."""

    upgrade_id: int
    source_governance_pack_id: int
    target_governance_pack_id: int
    from_pack_version: str
    to_pack_version: str
    planner_inputs_fingerprint: str
    adapter_state_fingerprint: str
    imported_object_ids: dict[str, list[int]] = Field(default_factory=dict)
    imported_object_counts: dict[str, int] = Field(default_factory=dict)


class GovernancePackUpgradeHistoryEntryResponse(BaseModel):
    """Lineage entry describing a past governance-pack upgrade."""

    id: int
    pack_id: str
    owner_scope_type: ScopeType
    owner_scope_id: int | None = None
    from_governance_pack_id: int
    to_governance_pack_id: int
    from_pack_version: str
    to_pack_version: str
    status: str
    planned_by: int | None = None
    executed_by: int | None = None
    planner_inputs_fingerprint: str | None = None
    adapter_state_fingerprint: str | None = None
    plan_summary: dict[str, Any] = Field(default_factory=dict)
    accepted_resolutions: dict[str, Any] = Field(default_factory=dict)
    failure_summary: str | None = None
    planned_at: datetime | str | None = None
    executed_at: datetime | str | None = None


class GovernancePackObjectProvenanceResponse(BaseModel):
    object_type: str
    object_id: str
    source_object_id: str


class GovernancePackSummaryResponse(BaseModel):
    id: int
    pack_id: str
    pack_version: str
    title: str
    description: str | None = None
    owner_scope_type: ScopeType
    owner_scope_id: int | None = None
    bundle_digest: str
    source_type: str | None = None
    source_location: str | None = None
    source_ref_requested: str | None = None
    source_ref_kind: Literal["branch", "tag", "commit"] | None = None
    source_subpath: str | None = None
    source_commit_resolved: str | None = None
    pack_content_digest: str | None = None
    source_verified: bool | None = None
    source_verification_mode: str | None = None
    signer_fingerprint: str | None = None
    signer_identity: str | None = None
    verified_object_type: str | None = None
    verification_result_code: str | None = None
    verification_warning_code: str | None = None
    source_fetched_at: datetime | str | None = None
    fetched_by: int | None = None
    manifest: dict[str, Any] = Field(default_factory=dict)
    is_active_install: bool = True
    superseded_by_governance_pack_id: int | None = None
    installed_from_upgrade_id: int | None = None
    created_by: int | None = None
    updated_by: int | None = None
    created_at: datetime | str | None = None
    updated_at: datetime | str | None = None


class GovernancePackDetailResponse(GovernancePackSummaryResponse):
    normalized_ir: dict[str, Any] = Field(default_factory=dict)
    imported_objects: list[GovernancePackObjectProvenanceResponse] = Field(default_factory=list)


class GovernancePackImportResponse(BaseModel):
    governance_pack_id: int
    imported_object_counts: dict[str, int] = Field(default_factory=dict)
    blocked_objects: list[str] = Field(default_factory=list)
    report: GovernancePackDryRunReportResponse


class GovernancePackTrustedSignerBinding(BaseModel):
    fingerprint: str = Field(..., min_length=1, max_length=256)
    display_name: str | None = Field(default=None, max_length=200)
    repo_bindings: list[str] = Field(default_factory=list)
    status: TrustSignerBindingStatus = "active"

    @field_validator("fingerprint", mode="before")
    @classmethod
    def _validate_fingerprint(cls, value: Any) -> str:
        cleaned = str(value or "").strip()
        if not cleaned:
            raise ValueError("fingerprint is required")
        return cleaned

    @field_validator("repo_bindings", mode="before")
    @classmethod
    def _validate_repo_bindings(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            values = [value]
        elif isinstance(value, (list, tuple, set)):
            values = list(value)
        else:
            raise ValueError("repo_bindings must be a list of strings")
        cleaned_values: list[str] = []
        for item in values:
            cleaned = str(item or "").strip()
            if not cleaned:
                raise ValueError("repo binding is required")
            cleaned_values.append(cleaned)
        return cleaned_values

    @model_validator(mode="after")
    def _require_repo_bindings(self) -> GovernancePackTrustedSignerBinding:
        if not self.repo_bindings:
            raise ValueError("trusted signer repo_bindings must not be empty")
        return self


def _validate_non_blank_string_list(value: Any, *, field_name: str) -> list[str]:
    """Normalize a string collection while rejecting whitespace-only entries."""
    if value is None:
        return []
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple, set)):
        values = list(value)
    else:
        raise ValueError(f"{field_name} must be a list of strings")
    cleaned_values: list[str] = []
    for item in values:
        cleaned = str(item or "").strip()
        if not cleaned:
            raise ValueError(f"{field_name} entries cannot be blank")
        cleaned_values.append(cleaned)
    return cleaned_values


class GovernancePackTrustPolicyRequest(BaseModel):
    """Request payload for updating the deployment-wide governance-pack trust policy."""
    allow_local_path_sources: bool = False
    allowed_local_roots: list[str] = Field(default_factory=list)
    allow_git_sources: bool = False
    allowed_git_hosts: list[str] = Field(default_factory=list)
    allowed_git_repositories: list[str] = Field(default_factory=list)
    allowed_git_ref_kinds: list[str] = Field(default_factory=list)
    require_git_signature_verification: bool = False
    trusted_signers: list[GovernancePackTrustedSignerBinding] = Field(default_factory=list)
    trusted_git_key_fingerprints: list[str] = Field(default_factory=list)
    policy_fingerprint: str = Field(..., min_length=1)

    @field_validator("trusted_git_key_fingerprints", mode="before")
    @classmethod
    def _validate_trusted_git_key_fingerprints(cls, value: Any) -> list[str]:
        return _validate_non_blank_string_list(value, field_name="fingerprint")


class GovernancePackTrustPolicyResponse(BaseModel):
    """Deployment-wide governance-pack trust policy response payload."""
    allow_local_path_sources: bool = False
    allowed_local_roots: list[str] = Field(default_factory=list)
    allow_git_sources: bool = False
    allowed_git_hosts: list[str] = Field(default_factory=list)
    allowed_git_repositories: list[str] = Field(default_factory=list)
    allowed_git_ref_kinds: list[str] = Field(default_factory=list)
    require_git_signature_verification: bool = False
    trusted_signers: list[GovernancePackTrustedSignerBinding] = Field(default_factory=list)
    policy_fingerprint: str | None = None
