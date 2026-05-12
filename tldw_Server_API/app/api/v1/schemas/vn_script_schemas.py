"""Pydantic schemas for the VN scripts API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tldw_Server_API.app.api.v1.schemas.pagination import OffsetPaginationMeta

ScriptStatus = Literal["draft", "ready", "archived"]
ContentRating = Literal["general", "teen", "suggestive", "mature"]


class VNScriptCreate(BaseModel):
    """Create request for a VN script shell."""

    title: str = Field(..., min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=1000)
    primary_asset_pack_id: int = Field(..., ge=1)
    policy_profile_id: str = Field(default="local_default", min_length=1, max_length=80)
    generation_profile_id: str = Field(default="story_default", min_length=1, max_length=80)
    generation_profiles: dict[str, str] = Field(default_factory=dict)
    generation_profile_ids: dict[str, str] | None = None
    content_rating: ContentRating = "general"

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _merge_generation_profile_ids(self) -> "VNScriptCreate":
        """Let legacy generation_profile_ids override generation_profiles when supplied."""
        if self.generation_profile_ids is not None:
            self.generation_profiles = dict(self.generation_profile_ids)
        return self


class VNScriptPatch(BaseModel):
    """Patch request for VN script metadata."""

    title: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=1000)
    status: ScriptStatus | None = None
    primary_asset_pack_id: int | None = Field(default=None, ge=1)
    policy_profile_id: str | None = Field(default=None, min_length=1, max_length=80)
    generation_profile_id: str | None = Field(default=None, min_length=1, max_length=80)
    generation_profiles: dict[str, str] | None = None
    generation_profile_ids: dict[str, str] | None = None
    content_rating: ContentRating | None = None

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _merge_generation_profile_ids(self) -> "VNScriptPatch":
        """Map legacy generation_profile_ids onto generation_profiles for patch compatibility."""
        if self.generation_profile_ids is not None:
            self.generation_profiles = dict(self.generation_profile_ids)
        return self


class VNScriptResponse(BaseModel):
    """VN script metadata response."""

    id: int
    title: str
    description: str | None = None
    status: str
    primary_asset_pack_id: int
    policy_profile_id: str
    generation_profile_id: str
    generation_profiles: dict[str, str] = Field(default_factory=dict)
    content_rating: str


class VNScriptListResponse(BaseModel):
    """Offset-paginated VN script list response."""

    items: list[VNScriptResponse]
    limit: int = Field(..., ge=1)
    offset: int = Field(..., ge=0)
    total: int = Field(..., ge=0)
    has_more: bool
    next_offset: int | None = Field(default=None, ge=0)
    pagination: OffsetPaginationMeta


class VNScriptTemplateSummary(BaseModel):
    """Preview-safe starter template catalog entry."""

    id: str
    label: str
    description: str
    category: str
    recommended_content_rating: ContentRating
    required_capabilities: list[str] = Field(default_factory=list)
    preview: dict[str, Any]
    default_title: str
    default_description: str | None = None


class VNScriptTemplateListResponse(BaseModel):
    """Starter template catalog response."""

    items: list[VNScriptTemplateSummary]


class VNScriptAuthoringOperation(BaseModel):
    """Preview-safe VN script operation metadata."""

    op: str
    label: str
    category: str
    capability_tokens: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class VNScriptAuthoringSnippet(BaseModel):
    """Preview-safe VN script snippet metadata."""

    id: str
    schema_version: Literal["vn_script_program.v1"]
    label: str
    operation_sequence: list[str] = Field(default_factory=list)
    required_capability_tokens: list[str] = Field(default_factory=list)
    parameters_schema: dict[str, Any]
    default_parameters: dict[str, Any] = Field(default_factory=dict)
    preview: list[dict[str, Any]]

    model_config = ConfigDict(extra="forbid")


class VNScriptAuthoringCatalogResponse(BaseModel):
    """VN script authoring operation and snippet catalog response."""

    schema_version: Literal["vn_script_authoring_catalog.v1"]
    program_schema_version: Literal["vn_script_program.v1"]
    capability_tokens: list[str] = Field(default_factory=list)
    generation_output_schemas: list[str] = Field(default_factory=list)
    operation_categories: dict[str, list[str]]
    operations: list[VNScriptAuthoringOperation]
    snippets: list[VNScriptAuthoringSnippet]
    limits: dict[str, int]

    model_config = ConfigDict(extra="forbid")


class VNScriptSnippetAnchor(BaseModel):
    """Snippet insertion anchor."""

    label: str = Field(..., min_length=1)
    mode: str = Field(default="append", min_length=1)
    op_index: int | None = Field(default=None, ge=0)

    model_config = ConfigDict(extra="forbid")


class VNScriptSnippetPreviewRequest(BaseModel):
    """Preview a snippet patch against the stored or supplied draft."""

    snippet_id: str = Field(..., min_length=1, max_length=80)
    anchor: VNScriptSnippetAnchor
    parameters: dict[str, Any] = Field(default_factory=dict)
    draft: dict[str, Any] | None = None
    draft_revision: int | None = Field(default=None, ge=0)

    model_config = ConfigDict(extra="forbid")


class VNScriptSnippetApplyRequest(BaseModel):
    """Apply a snippet patch to the stored draft."""

    if_revision: int = Field(..., ge=0)
    snippet_id: str = Field(..., min_length=1, max_length=80)
    anchor: VNScriptSnippetAnchor
    parameters: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class VNScriptSnippetPatchSummary(BaseModel):
    """Summary of a snippet patch."""

    inserted_ops: int = Field(..., ge=0)
    created_labels: list[str] = Field(default_factory=list)
    changed_paths: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class VNScriptSnippetPreviewResponse(BaseModel):
    """Snippet preview response."""

    script_id: int
    base_revision: int
    snippet_id: str
    draft: dict[str, Any]
    diagnostics: dict[str, Any]
    patch_summary: VNScriptSnippetPatchSummary
    warnings: list[dict[str, Any]] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class VNScriptSnippetApplyResponse(BaseModel):
    """Snippet apply response."""

    script_id: int
    revision: int
    snippet_id: str
    draft: dict[str, Any]
    diagnostics: dict[str, Any]
    patch_summary: VNScriptSnippetPatchSummary

    model_config = ConfigDict(extra="forbid")


class VNScriptCreateFromTemplateRequest(BaseModel):
    """Create request for a VN script starter template."""

    title: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=1000)
    primary_asset_pack_id: int = Field(..., ge=1)
    policy_profile_id: str = Field(default="local_default", min_length=1, max_length=80)
    generation_profile_id: str = Field(default="story_default", min_length=1, max_length=80)
    generation_profiles: dict[str, str] = Field(default_factory=dict)
    generation_profile_ids: dict[str, str] | None = None
    content_rating: ContentRating = "general"

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _merge_generation_profile_ids(self) -> "VNScriptCreateFromTemplateRequest":
        """Let legacy generation_profile_ids override generation_profiles for template creates."""
        if self.generation_profile_ids is not None:
            self.generation_profiles = dict(self.generation_profile_ids)
        return self


class VNScriptDraftResponse(BaseModel):
    """Mutable script draft response."""

    script_id: int
    revision: int
    draft: dict[str, Any]
    diagnostics: dict[str, Any]


class VNScriptCreateFromTemplateResponse(BaseModel):
    """Create-from-template response with the stored draft."""

    script: VNScriptResponse
    draft: VNScriptDraftResponse


class VNScriptDraftPutRequest(BaseModel):
    """Whole-draft replacement request."""

    if_revision: int = Field(..., ge=0)
    draft: dict[str, Any]

    model_config = ConfigDict(extra="forbid")


class VNScriptValidateRequest(BaseModel):
    """Validate a supplied draft, defaulting to current draft when omitted."""

    draft: dict[str, Any] | None = None

    model_config = ConfigDict(extra="forbid")


class VNScriptValidationResponse(BaseModel):
    """Script validation response."""

    valid: bool
    errors: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[dict[str, Any]] = Field(default_factory=list)


class VNScriptDiagnosticsResponse(BaseModel):
    """Current draft diagnostics response."""

    script_id: int
    revision: int
    diagnostics: dict[str, Any]


class VNScriptPublishRequest(BaseModel):
    """Publish immutable script version request."""

    draft_revision: int = Field(..., ge=0)
    label: str | None = Field(default=None, max_length=120)
    idempotency_key: str = Field(..., min_length=1, max_length=200)
    acknowledgements: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class VNScriptPublishResponse(BaseModel):
    """Publish response for immutable script versions."""

    script_id: int
    version_id: int
    version_number: int
    status: str
    asset_pack_id: int
    manifest_snapshot_id: int
    policy_snapshot_id: int
    generation_profile_snapshot_id: int
    generation_profile_snapshots: dict[str, int] = Field(default_factory=dict)
    validation: dict[str, Any]
    created_at: str


class VNScriptVersionResponse(BaseModel):
    """Immutable script version response."""

    id: int
    script_id: int
    version_number: int
    label: str | None = None
    draft_revision: int
    program: dict[str, Any]
    asset_pack_id: int
    manifest_snapshot_id: int
    policy_snapshot_id: int
    generation_profile_snapshot_id: int
    generation_profile_snapshots: dict[str, int] = Field(default_factory=dict)
    script_defaults: dict[str, Any]
    validation: dict[str, Any]
    created_at: str


class VNScriptVersionListResponse(BaseModel):
    """Offset-paginated script version list response."""

    items: list[VNScriptVersionResponse]
    limit: int = Field(..., ge=1)
    offset: int = Field(..., ge=0)
    total: int = Field(..., ge=0)
    has_more: bool
    next_offset: int | None = Field(default=None, ge=0)
    pagination: OffsetPaginationMeta


class VNScriptManifestSnapshotResponse(BaseModel):
    """Pinned manifest snapshot response."""

    id: int
    script_id: int
    version_id: int | None = None
    asset_pack_id: int
    manifest: dict[str, Any]
    manifest_hash: str
    created_at: str


class VNScriptVersionPolicyEvaluateRequest(BaseModel):
    """Evaluate policy for a published script version."""

    context: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class VNScriptVersionPolicyEvaluateResponse(BaseModel):
    """Policy evaluation response for a published script version."""

    decision: str
    profile_id: str
    reasons: list[dict[str, Any]] = Field(default_factory=list)
    blocked: bool
    requires_acknowledgement: bool
    remediation: list[str] = Field(default_factory=list)
