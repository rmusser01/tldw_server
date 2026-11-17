"""Core-owned VN Play setup models, re-exported by the API schemas package (TASK-13950.15)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, StrictBool, StrictInt, StrictStr

VNPlayMode = Literal["freeform", "story", "scripted_story"]
VNPlaySetupTrustLevel = Literal["local", "trusted_restore", "untrusted_import", "unknown"]
VNPlaySetupTrustSource = Literal["local_pack", "latest_import_journal", "unknown"]
VNPlaySetupWarningSeverity = Literal["info", "warning", "high_risk"]
VNPlaySetupCompatibilityStatus = Literal["compatible", "different_character", "unknown"]
VNPlaySetupEmptyStateScope = Literal["global", "filter", "page"]


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
    generation_profile_snapshot_id: StrictInt | None = None
    policy_profile_id: StrictStr
    generation_profile_id: StrictStr
    generation_profile_key: StrictStr = "default"
    generation_profile_snapshot_immutable: StrictBool = True
    provider_class: StrictStr | None = None
    max_automatic_generation_batch_count: StrictInt | None = Field(default=None, ge=0)
    moderation_required: StrictBool | None = None
    estimated_cost_class: StrictStr | None = None
    supported_output_schemas: list[StrictStr] = Field(default_factory=list)
    dynamic_choice_support: StrictBool = False
    scene_update_support: StrictBool = False
    confirmation_required: StrictBool = False
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
