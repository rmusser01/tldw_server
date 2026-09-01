"""Schemas for nested Notes semantic-index operations."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SemanticCapabilitiesResponse(_StrictModel):
    active_note_count: int = Field(ge=0)
    estimated_chunk_count: int = Field(ge=0)
    estimated_run_count: int = Field(ge=0)
    provider_label: str
    model: str
    execution_boundary: Literal["external", "local"]
    storage_boundary: Literal["external", "local", "unavailable"]
    storage_label: str
    outbound_data_categories: tuple[str, ...]
    capability_revision: str
    indexing_available: bool
    unavailable_reason: str | None
    metric: Literal["cosine"] = "cosine"
    resolved_dimensions: int | None = Field(default=None, ge=1)
    manage_authorized: bool = False


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
