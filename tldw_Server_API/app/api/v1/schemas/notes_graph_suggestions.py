"""Bounded public schemas for nested Notes graph suggestion routes."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

_FINGERPRINT_PATTERN = r"^sha256:[0-9a-f]{64}$"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SuggestionRunCreateRequest(_StrictModel):
    provider: str | None = Field(default=None, min_length=1, max_length=128)
    model: str | None = Field(default=None, min_length=1, max_length=256)

    @field_validator("provider", "model", mode="before")
    @classmethod
    def _trim_optional(cls, value: object) -> object:
        if value is None or not isinstance(value, str):
            return value
        return value.strip()


class SuggestionRunCancelRequest(_StrictModel):
    expected_revision: int = Field(ge=1)


class SuggestionDecisionRequest(_StrictModel):
    expected_revision: int = Field(ge=1)
    expected_source_fingerprint: str = Field(pattern=_FINGERPRINT_PATTERN)
    expected_target_fingerprint: str | None = Field(default=None, pattern=_FINGERPRINT_PATTERN)


class SuggestionResetRequest(_StrictModel):
    expected_rejection_revision: int = Field(ge=0)
    source_fingerprint: str = Field(pattern=_FINGERPRINT_PATTERN)
    confirm: Literal[True]


class SuggestionCapabilityLimitsResponse(_StrictModel):
    max_candidates: int = Field(ge=1, le=30)
    max_relationships: int = Field(ge=1, le=5)
    max_tags: int = Field(ge=1, le=5)
    max_new_tags: int = Field(ge=1, le=2)
    max_tag_catalog: int = Field(ge=1, le=100)
    max_estimated_input_tokens: int = Field(ge=1, le=24_000)
    max_output_tokens: int = Field(ge=1, le=2_000)
    provider_timeout_seconds: int = Field(ge=1, le=120)
    response_candidates: Literal[1]


class SuggestionCapabilitiesResponse(_StrictModel):
    provider: str
    model: str
    endpoint_origin_revision: str = Field(pattern=_FINGERPRINT_PATTERN)
    data_boundary: Literal["local", "remote", "unknown"]
    disclosure_external: bool
    outbound_data_categories: tuple[str, ...]
    generation_available: bool
    unavailable_reason: str | None
    limits: SuggestionCapabilityLimitsResponse
    allowed_actions: tuple[str, ...]
    revision: str = Field(pattern=_FINGERPRINT_PATTERN)


class SuggestionRunResponse(_StrictModel):
    id: str
    provider: str
    model: str
    state: str
    revision: int = Field(ge=1)
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    suggestion_count: int = Field(ge=0)
    related_note_count: int = Field(ge=0)
    tag_count: int = Field(ge=0)
    invalid_item_count: int = Field(ge=0)
    cancellation_available: bool
    error_code: str | None
    guidance_key: str | None


class SuggestionRunListResponse(_StrictModel):
    items: tuple[SuggestionRunResponse, ...]
    next_cursor: str | None


class SuggestionEvidenceResponse(_StrictModel):
    side: Literal["source", "target"]
    note_id: str
    field: Literal["title", "content"]
    start_offset: int = Field(ge=0)
    end_offset: int = Field(ge=1)
    text: str = Field(max_length=480)


class SuggestionItemResponse(_StrictModel):
    id: str
    run_id: str
    kind: Literal["related_note", "tag"]
    state: str
    revision: int = Field(ge=1)
    source_note_id: str
    source_fingerprint: str = Field(pattern=_FINGERPRINT_PATTERN)
    target_note_id: str | None
    target_fingerprint: str | None = Field(default=None, pattern=_FINGERPRINT_PATTERN)
    normalized_tag: str | None
    display_tag: str | None
    existing_tag: bool
    match_strength: str | None
    rationale: str | None = Field(default=None, max_length=240)
    evidence: tuple[SuggestionEvidenceResponse, ...]
    updated_at: datetime


class SuggestionListResponse(_StrictModel):
    items: tuple[SuggestionItemResponse, ...]
    next_cursor: str | None
    current_source_fingerprint: str = Field(pattern=_FINGERPRINT_PATTERN)
    rejection_set_revision: int = Field(ge=0)
    rejection_count: int = Field(ge=0)


class SuggestionMutationResponse(_StrictModel):
    resource_id: str
    state: str
    revision: int = Field(ge=0)
    cleared_count: int | None = Field(default=None, ge=0)


class SuggestionErrorResponse(_StrictModel):
    error_code: str
    message: str


class SuggestionHTTPErrorResponse(_StrictModel):
    detail: SuggestionErrorResponse


__all__ = [
    "SuggestionCapabilitiesResponse",
    "SuggestionDecisionRequest",
    "SuggestionErrorResponse",
    "SuggestionHTTPErrorResponse",
    "SuggestionItemResponse",
    "SuggestionListResponse",
    "SuggestionMutationResponse",
    "SuggestionResetRequest",
    "SuggestionRunCancelRequest",
    "SuggestionRunCreateRequest",
    "SuggestionRunListResponse",
    "SuggestionRunResponse",
]
