"""Pydantic schemas for the VN scripts API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

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
    content_rating: ContentRating = "general"

    model_config = ConfigDict(extra="forbid")


class VNScriptPatch(BaseModel):
    """Patch request for VN script metadata."""

    title: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=1000)
    status: ScriptStatus | None = None
    primary_asset_pack_id: int | None = Field(default=None, ge=1)
    policy_profile_id: str | None = Field(default=None, min_length=1, max_length=80)
    generation_profile_id: str | None = Field(default=None, min_length=1, max_length=80)
    content_rating: ContentRating | None = None

    model_config = ConfigDict(extra="forbid")


class VNScriptResponse(BaseModel):
    """VN script metadata response."""

    id: int
    title: str
    description: str | None = None
    status: str
    primary_asset_pack_id: int
    policy_profile_id: str
    generation_profile_id: str
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


class VNScriptDraftResponse(BaseModel):
    """Mutable script draft response."""

    script_id: int
    revision: int
    draft: dict[str, Any]
    diagnostics: dict[str, Any]


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
