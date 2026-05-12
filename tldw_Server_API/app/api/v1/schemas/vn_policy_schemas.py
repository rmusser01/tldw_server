"""Pydantic schemas for the VN policy API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tldw_Server_API.app.api.v1.schemas.pagination import OffsetPaginationMeta

ContentRating = Literal["general", "teen", "suggestive", "mature"]
ModelExpansionScope = Literal["none", "turn", "scene", "session"]
AuditMode = Literal["none", "metadata", "full"]
PolicyDecision = Literal["allow", "warn", "block"]


class VNPolicyProfileCreate(BaseModel):
    """Create request for a VN policy profile."""

    profile_id: str = Field(..., min_length=1, max_length=80, pattern=r"^[a-zA-Z0-9_.:-]+$")
    display_name: str = Field(..., min_length=1, max_length=160)
    description: str | None = Field(default=None, max_length=1000)
    definition: dict[str, Any] = Field(..., min_length=1)

    model_config = ConfigDict(extra="forbid")


class VNPolicyProfilePatch(BaseModel):
    """Patch request for a VN policy profile."""

    display_name: str | None = Field(default=None, min_length=1, max_length=160)
    description: str | None = Field(default=None, max_length=1000)
    definition: dict[str, Any] | None = None

    model_config = ConfigDict(extra="forbid")


class VNGenerationProfileCreate(BaseModel):
    """Create request for a VN generation profile."""

    profile_id: str = Field(..., min_length=1, max_length=80, pattern=r"^[a-zA-Z0-9_.:-]+$")
    display_name: str = Field(..., min_length=1, max_length=160)
    description: str | None = Field(default=None, max_length=1000)
    provider: str = Field(..., min_length=1, max_length=120)
    model: str = Field(..., min_length=1, max_length=160)
    supports_structured_output: bool
    temperature_default: float = Field(..., ge=0)
    temperature_min: float = Field(..., ge=0)
    temperature_max: float = Field(..., ge=0)
    max_output_tokens: int = Field(..., ge=1, le=262144)
    allowed_content_ratings: list[ContentRating] = Field(..., min_length=1)
    max_choices: int = Field(..., ge=1, le=32)
    max_branch_depth: int = Field(..., ge=1, le=1000)
    max_model_expansion_scope: ModelExpansionScope
    tts_allowed: bool
    output_persistence_max_days: int = Field(..., ge=0, le=3650)
    audit_mode: AuditMode

    model_config = ConfigDict(extra="forbid")

    @field_validator("allowed_content_ratings")
    @classmethod
    def _dedupe_ratings(cls, value: list[ContentRating]) -> list[ContentRating]:
        deduped = list(dict.fromkeys(value))
        if not deduped:
            raise ValueError("At least one content rating is required")
        return deduped

    @model_validator(mode="after")
    def _validate_temperature_bounds(self) -> "VNGenerationProfileCreate":
        if self.temperature_max < self.temperature_min:
            raise ValueError("temperature_max must be greater than or equal to temperature_min")
        if not self.temperature_min <= self.temperature_default <= self.temperature_max:
            raise ValueError("temperature_default must be within min/max bounds")
        return self

    def definition_payload(self) -> dict[str, Any]:
        """Return the fields stored as the profile definition."""
        return {
            "provider": self.provider,
            "model": self.model,
            "supports_structured_output": self.supports_structured_output,
            "temperature_default": self.temperature_default,
            "temperature_min": self.temperature_min,
            "temperature_max": self.temperature_max,
            "max_output_tokens": self.max_output_tokens,
            "allowed_content_ratings": list(self.allowed_content_ratings),
            "max_choices": self.max_choices,
            "max_branch_depth": self.max_branch_depth,
            "max_model_expansion_scope": self.max_model_expansion_scope,
            "tts_allowed": self.tts_allowed,
            "output_persistence_max_days": self.output_persistence_max_days,
            "audit_mode": self.audit_mode,
        }


class VNGenerationProfilePatch(BaseModel):
    """Patch request for a VN generation profile."""

    display_name: str | None = Field(default=None, min_length=1, max_length=160)
    description: str | None = Field(default=None, max_length=1000)
    provider: str | None = Field(default=None, min_length=1, max_length=120)
    model: str | None = Field(default=None, min_length=1, max_length=160)
    supports_structured_output: bool | None = None
    temperature_default: float | None = Field(default=None, ge=0)
    temperature_min: float | None = Field(default=None, ge=0)
    temperature_max: float | None = Field(default=None, ge=0)
    max_output_tokens: int | None = Field(default=None, ge=1, le=262144)
    allowed_content_ratings: list[ContentRating] | None = Field(default=None, min_length=1)
    max_choices: int | None = Field(default=None, ge=1, le=32)
    max_branch_depth: int | None = Field(default=None, ge=1, le=1000)
    max_model_expansion_scope: ModelExpansionScope | None = None
    tts_allowed: bool | None = None
    output_persistence_max_days: int | None = Field(default=None, ge=0, le=3650)
    audit_mode: AuditMode | None = None

    model_config = ConfigDict(extra="forbid")


class VNPolicyProfileResponse(BaseModel):
    """Policy profile response."""

    profile_id: str
    display_name: str
    description: str | None = None
    definition: dict[str, Any]
    version: int
    builtin: bool
    disabled: bool


class VNGenerationProfileResponse(BaseModel):
    """Generation profile response."""

    profile_id: str
    display_name: str
    description: str | None = None
    definition: dict[str, Any]
    version: int
    builtin: bool
    disabled: bool


class VNPolicyProfileListResponse(BaseModel):
    """Offset-paginated policy profile list response."""

    items: list[VNPolicyProfileResponse]
    limit: int = Field(..., ge=1)
    offset: int = Field(..., ge=0)
    total: int = Field(..., ge=0)
    has_more: bool
    next_offset: int | None = Field(default=None, ge=0)
    pagination: OffsetPaginationMeta


class VNGenerationProfileListResponse(BaseModel):
    """Offset-paginated generation profile list response."""

    items: list[VNGenerationProfileResponse]
    limit: int = Field(..., ge=1)
    offset: int = Field(..., ge=0)
    total: int = Field(..., ge=0)
    has_more: bool
    next_offset: int | None = Field(default=None, ge=0)
    pagination: OffsetPaginationMeta


class VNPolicyReason(BaseModel):
    """Reason entry in a VN policy evaluation response."""

    code: str
    severity: Literal["info", "warning", "error"]
    message: str
    requires_acknowledgement: bool = False


class VNPolicyEvaluateRequest(BaseModel):
    """Policy preflight request."""

    target_type: Literal[
        "asset_pack",
        "script_draft",
        "script_version",
        "session_setup",
        "runtime_action",
        "tts_request",
    ]
    target_id: int | None = Field(default=None, ge=1)
    policy_profile_id: str = Field(default="local_default", min_length=1, max_length=80)
    context: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class VNPolicyEvaluateResponse(BaseModel):
    """Policy preflight response."""

    decision: PolicyDecision
    profile_id: str
    reasons: list[VNPolicyReason] = Field(default_factory=list)
    blocked: bool
    requires_acknowledgement: bool
    remediation: list[str] = Field(default_factory=list)
    target_type: str | None = None
    target_id: int | None = None
