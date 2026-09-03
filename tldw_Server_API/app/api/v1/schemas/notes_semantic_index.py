"""Schemas for nested Notes semantic-index operations."""

from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from tldw_Server_API.app.core.Notes_Graph.semantic_endpoint import (
    canonical_semantic_endpoint_origin,
)

_SEMANTIC_OUTBOUND_CATEGORIES = {"note_content_chunks", "note_title"}


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SemanticCapabilitiesResponse(_StrictModel):
    active_note_count: int = Field(ge=0)
    estimated_chunk_count: int = Field(ge=0)
    estimated_run_count: int = Field(ge=0)
    provider_label: str = Field(min_length=1, max_length=128)
    model: str = Field(min_length=1, max_length=256)
    endpoint_display: str | None = Field(min_length=1, max_length=512)
    execution_boundary: Literal["external", "local"]
    storage_boundary: Literal["external", "local", "unavailable"]
    storage_label: str = Field(min_length=1, max_length=128)
    outbound_data_categories: tuple[
        Literal["note_content_chunks", "note_title"], ...
    ]
    capability_revision: str = Field(min_length=1, max_length=128)
    indexing_available: bool
    unavailable_reason: str | None
    metric: Literal["cosine"] = "cosine"
    resolved_dimensions: int | None = Field(default=None, ge=1)
    dimension_probe_required: bool
    renewal_requires_delete: bool
    manage_authorized: bool = False

    @field_validator("provider_label", "model", "storage_label")
    @classmethod
    def validate_consent_label(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("consent identity labels must not be blank")
        return normalized

    @field_validator("endpoint_display")
    @classmethod
    def validate_endpoint_display(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if canonical_semantic_endpoint_origin(value) != value:
            raise ValueError("endpoint_display must be a sanitized origin")
        return value

    @model_validator(mode="after")
    def validate_disclosure_identity(self) -> SemanticCapabilitiesResponse:
        if (
            len(self.outbound_data_categories) != len(_SEMANTIC_OUTBOUND_CATEGORIES)
            or set(self.outbound_data_categories) != _SEMANTIC_OUTBOUND_CATEGORIES
        ):
            raise ValueError("the exact semantic outbound disclosure is required")
        if self.dimension_probe_required and (
            self.resolved_dimensions is not None
            or not self.indexing_available
            or self.unavailable_reason is not None
        ):
            raise ValueError("dimension disclosure is contradictory")
        if (
            self.indexing_available
            and self.resolved_dimensions is None
            and not self.dimension_probe_required
        ):
            raise ValueError("dimension disclosure is contradictory")
        if self.indexing_available and (
            self.storage_boundary == "unavailable"
            or self.unavailable_reason is not None
            or self.endpoint_display is None
        ):
            raise ValueError("available capability disclosure is contradictory")
        if not self.indexing_available and self.unavailable_reason is None:
            raise ValueError("unavailable capability disclosure requires a reason")
        if self.endpoint_display is None and self.unavailable_reason is None:
            raise ValueError("missing endpoint disclosure requires an unavailable reason")
        return self


class SemanticRunResponse(_StrictModel):
    run_id: str
    mode: str
    status: str
    revision: int = Field(ge=0)
    indexed_notes: int = Field(ge=0)
    excluded_notes: int = Field(ge=0)
    failed_notes: int = Field(ge=0)
    pending_notes: int = Field(ge=0)
    published_chunks: int = Field(ge=0)
    cleanup_complete: bool
    error_code: str | None
    link: str


class SemanticIndexStatusResponse(_StrictModel):
    state: Literal["off", "preparing", "ready", "updating", "needs_attention"]
    detail_reason: str | None
    desired_state: Literal["enabled", "disabled"]
    configuration_revision: int = Field(ge=0)
    semantic_index_revision: int = Field(ge=0)
    active_generation_id: str | None
    active_generation_usable: bool
    indexed_notes: int = Field(ge=0)
    excluded_notes: int = Field(ge=0)
    failed_notes: int = Field(ge=0)
    pending_notes: int = Field(ge=0)
    published_chunks: int = Field(ge=0)
    cleanup_pending: bool
    active_run: SemanticRunResponse | None


class SemanticIndexMutationResponse(_StrictModel):
    resource: SemanticIndexStatusResponse
    run: SemanticRunResponse


class SemanticEnableRequest(_StrictModel):
    expected_revision: int = Field(ge=0)
    capability_revision: str = Field(min_length=1, max_length=128)


class SemanticDisableRequest(_StrictModel):
    expected_revision: int = Field(ge=0)


class SemanticRunCreateRequest(_StrictModel):
    mode: Literal["rebuild", "retry_failed"]
    expected_revision: int = Field(ge=0)


class SemanticRunCancelRequest(_StrictModel):
    expected_revision: int = Field(ge=0)


class SemanticHTTPErrorDetail(_StrictModel):
    error_code: str
    message: str


class SemanticHTTPErrorResponse(_StrictModel):
    detail: SemanticHTTPErrorDetail


__all__ = [
    "SemanticCapabilitiesResponse",
    "SemanticDisableRequest",
    "SemanticEnableRequest",
    "SemanticHTTPErrorResponse",
    "SemanticIndexMutationResponse",
    "SemanticIndexStatusResponse",
    "SemanticRunCancelRequest",
    "SemanticRunCreateRequest",
    "SemanticRunResponse",
]
