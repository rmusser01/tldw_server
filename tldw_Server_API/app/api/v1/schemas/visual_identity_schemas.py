"""Pydantic schemas for visual identity expression pack APIs."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ActorKind = Literal["character", "persona"]


class VisualIdentityCapabilitiesResponse(BaseModel):
    """Runtime limits and supported formats for visual identity assets."""

    upload_max_bytes: int = Field(..., ge=1)
    archive_max_bytes: int = Field(..., ge=1)
    max_dimension: int = Field(..., ge=1)
    max_frame_count: int = Field(..., ge=1)
    supported_mime_types: list[str] = Field(default_factory=list)
    avif_enabled: bool


class VisualIdentityExpressionSlotResponse(BaseModel):
    """One supported expression slot."""

    key: str
    label: str
    canonical: bool = True
    aliases: list[str] = Field(default_factory=list)


class VisualIdentityPackCreate(BaseModel):
    """Request body for creating a visual identity pack shell."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(..., min_length=1, max_length=500)
    description: str = ""
    default_expression_key: str = Field(default="neutral", min_length=1, max_length=100)
    source_kind: str = Field(default="manual", min_length=1, max_length=100)
    source_context: dict[str, Any] = Field(default_factory=dict)


class VisualIdentityPackUpdate(BaseModel):
    """Request body for patching visual identity pack metadata."""

    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, min_length=1, max_length=500)
    description: str | None = None
    status: Literal["active", "archived"] | None = None
    default_expression_key: str | None = Field(default=None, min_length=1, max_length=100)
    source_kind: str | None = Field(default=None, min_length=1, max_length=100)
    source_context: dict[str, Any] | None = None


class VisualIdentityPackResponse(BaseModel):
    """Serialized visual identity pack metadata."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    owner_user_id: int
    title: str
    description: str = ""
    status: str
    active_version_id: int | None = None
    default_expression_key: str
    source_kind: str
    source_context: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None
    version: int


class VisualIdentityAssetResponse(BaseModel):
    """Serialized visual identity asset metadata."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    owner_user_id: int
    pack_id: int | None = None
    draft_id: int | None = None
    pack_version_id: int | None = None
    expression_key: str
    original_expression_key: str = ""
    display_label: str = ""
    source_filename: str
    content_type: str
    bytes: int
    sha256: str
    width: int
    height: int
    is_animated: bool = False
    frame_count: int | None = None
    duration_ms: int | None = None
    preview_relpath: str | None = None
    source_context: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None


class VisualIdentityDraftResponse(BaseModel):
    """Serialized visual identity draft metadata."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    owner_user_id: int
    pack_id: int | None = None
    title: str
    status: str
    source_kind: str
    source_filename: str = ""
    import_job_id: str | None = None
    validation_summary: dict[str, Any] = Field(default_factory=dict)
    slot_map: dict[str, Any] = Field(default_factory=dict)
    default_expression_key: str = "neutral"
    error: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None
    version: int
    assets: list[VisualIdentityAssetResponse] = Field(default_factory=list)
    pack_version_id: int | None = None
    asset_ids: list[int] = Field(default_factory=list)
    binding_id: int | None = None


class VisualIdentityDraftSlotUpdate(BaseModel):
    """Request body for updating one draft expression slot mapping."""

    model_config = ConfigDict(extra="forbid")

    asset_id: int | None = Field(default=None, ge=1)
    expression_key: str | None = Field(default=None, min_length=1, max_length=100)
    display_label: str | None = Field(default=None, max_length=200)
    metadata: dict[str, Any] = Field(default_factory=dict)


class VisualIdentityDraftActivateRequest(BaseModel):
    """Request body for activating a draft and optionally binding it."""

    model_config = ConfigDict(extra="forbid")

    actor_kind: ActorKind | None = None
    actor_id: int | str | None = None


class VisualIdentityBindingRequest(BaseModel):
    """Request body for binding an actor to a visual identity pack version."""

    model_config = ConfigDict(extra="forbid")

    actor_kind: ActorKind
    actor_id: int | str
    pack_id: int = Field(..., ge=1)
    active_version_id: int | None = Field(default=None, ge=1)


class VisualIdentityBindingResponse(BaseModel):
    """Serialized active visual identity binding metadata."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    owner_user_id: int
    actor_kind: ActorKind
    actor_id: str
    pack_id: int
    active_version_id: int
    status: str
    created_at: str | None = None
    updated_at: str | None = None
    version: int


class VisualIdentityResolveResponse(BaseModel):
    """Resolved visual identity asset for an actor expression request."""

    actor_kind: ActorKind
    actor_id: int | str
    pack_id: int | None = None
    pack_version_id: int | None = None
    expression_key: str | None = None
    requested_expression_key: str | None = None
    asset_id: int | None = None
    storage_relpath: str | None = None
    fallback_reason: str | None = None
    is_animated: bool = False
    content_type: str | None = None
    asset_url: str | None = None
    role_id: str | None = None
    role_label: str | None = None
    resolution_source: str | None = None


class VisualIdentityImportZipStartResponse(BaseModel):
    """Response returned after a ZIP import draft/job is started."""

    draft_id: int
    job_id: str | int | None = None
    status: str
    source_filename: str
    import_job_id: str | None = None


class VisualIdentityGeneratedFileAssetRequest(BaseModel):
    """Request body for attaching a generated image file to a draft asset."""

    model_config = ConfigDict(extra="forbid")

    generated_file_id: int = Field(..., ge=1)
    expression_key: str = Field(..., min_length=1, max_length=100)
    draft_id: int | None = Field(default=None, ge=1)
    source_feature: str = Field(default="image_gen", min_length=1, max_length=100)
    source_context: dict[str, Any] = Field(default_factory=dict)
    idempotency_key: str = Field(..., min_length=1, max_length=160)
