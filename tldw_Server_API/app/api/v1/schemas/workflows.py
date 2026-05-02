from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tldw_Server_API.app.api.v1.schemas.pagination import CursorPaginationMeta, OffsetPaginationMeta


class WorkflowInputSchema(BaseModel):
    # Free-form for v0.1 stub
    params: dict[str, Any] = Field(default_factory=dict)


class StepConfig(BaseModel):
    id: str = Field(..., description="Step identifier")
    name: str | None = None
    type: str = Field(..., description="Step type name")
    config: dict[str, Any] = Field(default_factory=dict)
    retry: int | None = 0
    timeout_seconds: float | None = 300
    on_success: str | None = None
    on_failure: str | None = None
    on_timeout: str | None = None


class WorkflowRagSearchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1)
    sources: list[str] | None = None
    search_mode: Literal["fts", "vector", "hybrid"] | None = None
    hybrid_alpha: float | None = Field(default=None, ge=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=1, le=100)
    min_score: float | None = Field(default=None, ge=0.0, le=1.0)
    expand_query: bool | None = None
    expansion_strategies: list[Literal["acronym", "synonym", "domain", "entity"]] | None = None
    spell_check: bool | None = None
    enable_cache: bool | None = None
    cache_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    adaptive_cache: bool | None = None
    cache_ttl: int | None = Field(default=None, ge=0)
    enable_table_processing: bool | None = None
    table_method: Literal["markdown", "html", "hybrid"] | None = None
    include_sibling_chunks: bool | None = None
    sibling_window: int | None = Field(default=None, ge=0, le=20)
    enable_parent_expansion: bool | None = None
    include_parent_document: bool | None = None
    parent_max_tokens: int | None = Field(default=None, ge=1, le=8192)
    enable_reranking: bool | None = None
    reranking_strategy: Literal["flashrank", "cross_encoder", "hybrid", "none"] | None = None
    rerank_top_k: int | None = Field(default=None, ge=1, le=100)
    enable_citations: bool | None = None
    citation_style: Literal["apa", "mla", "chicago", "harvard", "ieee"] | None = None
    include_page_numbers: bool | None = None
    enable_chunk_citations: bool | None = None
    enable_generation: bool | None = None
    generation_model: str | None = Field(default=None, max_length=128)
    generation_prompt: str | None = Field(default=None, max_length=8192)
    max_generation_tokens: int | None = Field(default=None, ge=1, le=8192)
    enable_security_filter: bool | None = None
    detect_pii: bool | None = None
    redact_pii: bool | None = None
    sensitivity_level: Literal["public", "internal", "confidential", "restricted"] | None = None
    content_filter: bool | None = None
    timeout_seconds: int | None = Field(default=None, ge=1, le=3600)
    highlight_results: bool | None = None
    highlight_query_terms: bool | None = None
    track_cost: bool | None = None

    @field_validator("query", mode="before")
    @classmethod
    def _normalize_query(cls, value):
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                raise ValueError("query must not be empty")
            return stripped
        return value

    @field_validator(
        "search_mode",
        "table_method",
        "reranking_strategy",
        "citation_style",
        "sensitivity_level",
        mode="before",
    )
    @classmethod
    def _normalize_lower(cls, value):
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @field_validator("sources", mode="before")
    @classmethod
    def _normalize_sources(cls, value):
        if value is None:
            return value
        if isinstance(value, str):
            raw = value.strip()
            if not raw:
                return []
            if "," in raw:
                return [s.strip() for s in raw.split(",") if s.strip()]
            return [raw]
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        return value

    @field_validator("expansion_strategies", mode="before")
    @classmethod
    def _normalize_expansion_strategies(cls, value):
        if value is None:
            return value
        if isinstance(value, str):
            raw = value.strip()
            if not raw:
                return []
            return [s.strip().lower() for s in raw.split(",") if s.strip()]
        if isinstance(value, list):
            return [str(v).strip().lower() for v in value if str(v).strip()]
        return value


class WorkflowDefinitionCreate(BaseModel):
    name: str
    version: int = 1
    description: str | None = None
    tags: list[str] = Field(default_factory=list)
    inputs: dict[str, Any] = Field(default_factory=dict)
    # Optional completion webhook configuration. Accepts either a URL string or
    # an object with fields like {"url": str, "include_outputs": bool}.
    on_completion_webhook: dict[str, Any] | str | None = None
    steps: list[StepConfig]
    metadata: dict[str, Any] = Field(default_factory=dict)
    visibility: Literal["private"] = "private"


class WorkflowDefinitionResponse(BaseModel):
    id: int
    name: str
    version: int
    description: str | None
    tags: list[str]
    is_active: bool


class RunRequest(BaseModel):
    inputs: dict[str, Any] = Field(default_factory=dict)
    idempotency_key: str | None = None
    session_id: str | None = None
    # Scoped secrets are injected into the execution context but never persisted.
    # Keys are provider/module specific (e.g., api keys), values are strings.
    secrets: dict[str, str] | None = Field(default_factory=dict)
    # Per-run validation behavior for safety checks (e.g., artifact scope)
    # 'block' rejects on validation failure; 'non-block' logs/warns and proceeds.
    validation_mode: Literal["block", "non-block"] = "block"


class AdhocRunRequest(BaseModel):
    definition: WorkflowDefinitionCreate
    inputs: dict[str, Any] = Field(default_factory=dict)
    idempotency_key: str | None = None
    session_id: str | None = None
    secrets: dict[str, str] | None = Field(default_factory=dict)
    validation_mode: Literal["block", "non-block"] = "block"


class WorkflowRunResponse(BaseModel):
    id: str | None = None
    run_id: str
    workflow_id: int | None = None
    user_id: str | None = None
    status: str
    status_reason: str | None = None
    inputs: dict[str, Any] = Field(default_factory=dict)
    outputs: dict[str, Any] | None = None
    error: str | None = None
    definition_version: int | None = None
    validation_mode: str | None = None


class EventResponse(BaseModel):
    event_seq: int
    event_type: str
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: str


# Listing schemas (lighter than full run response)
class WorkflowRunListItem(BaseModel):
    run_id: str
    workflow_id: int | None = None
    user_id: str | None = None
    status: str
    status_reason: str | None = None
    definition_version: int | None = None
    created_at: str | None = None
    started_at: str | None = None
    ended_at: str | None = None


class WorkflowRunListResponse(BaseModel):
    runs: list[WorkflowRunListItem]
    limit: int
    offset: int | None = None
    has_more: bool | None = Field(default=None, description="Alias for pagination.has_more")
    total: int | None = Field(
        default=None,
        ge=0,
        description="Alias for pagination.total when the total is known",
    )
    pagination: OffsetPaginationMeta | CursorPaginationMeta
    next_offset: int | None = None
    next_cursor: str | None = None

    @model_validator(mode="after")
    def _default_pagination_aliases(self) -> "WorkflowRunListResponse":
        if self.has_more is None:
            self.has_more = self.pagination.has_more
        if self.total is None and hasattr(self.pagination, "total"):
            self.total = self.pagination.total
        if self.next_offset is None and isinstance(self.pagination, OffsetPaginationMeta):
            self.next_offset = self.pagination.next_offset
        if self.next_cursor is None and isinstance(self.pagination, CursorPaginationMeta):
            self.next_cursor = self.pagination.next_cursor
        return self

class WorkflowFailureSummary(BaseModel):
    reason_code_core: str | None = None
    reason_code_detail: str | None = None
    category: str | None = None
    blame_scope: str | None = None
    retryable: bool | None = None
    retry_recommendation: str | None = None
    error_summary: str | None = None
    internal_detail: dict[str, Any] | None = None


class WorkflowStepAttemptResponse(BaseModel):
    attempt_id: str
    step_run_id: str
    step_id: str
    attempt_number: int
    status: str
    started_at: str
    ended_at: str | None = None
    duration_ms: int | None = None
    reason_code_core: str | None = None
    reason_code_detail: str | None = None
    retryable: bool | None = None
    error_summary: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkflowRunStepResponse(BaseModel):
    step_run_id: str
    step_id: str
    name: str | None = None
    type: str | None = None
    status: str
    attempt_count: int = 0
    started_at: str | None = None
    ended_at: str | None = None
    error: str | None = None
    latest_failure: WorkflowFailureSummary | None = None


class WorkflowStepAttemptsResponse(BaseModel):
    run_id: str
    step_id: str
    attempts: list[WorkflowStepAttemptResponse] = Field(default_factory=list)


class WorkflowRunStepsResponse(BaseModel):
    run_id: str
    steps: list[WorkflowRunStepResponse] = Field(default_factory=list)


class WorkflowRunInvestigationResponse(BaseModel):
    run_id: str
    status: str
    schema_version: int
    derived_from_event_seq: int = 0
    failed_step: WorkflowRunStepResponse | None = None
    primary_failure: WorkflowFailureSummary | None = None
    attempts: list[WorkflowStepAttemptResponse] = Field(default_factory=list)
    evidence: dict[str, Any] = Field(default_factory=dict)
    recommended_actions: list[str] = Field(default_factory=list)


class WorkflowValidationIssue(BaseModel):
    code: str
    message: str
    step_id: str | None = None
    step_type: str | None = None


class WorkflowPreflightRequest(BaseModel):
    definition: Any
    validation_mode: Literal["block", "non-block"] = "block"


class WorkflowPreflightResponse(BaseModel):
    valid: bool
    errors: list[WorkflowValidationIssue] = Field(default_factory=list)
    warnings: list[WorkflowValidationIssue] = Field(default_factory=list)
