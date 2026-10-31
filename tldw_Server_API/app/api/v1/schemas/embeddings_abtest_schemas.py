from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from tldw_Server_API.app.api.v1.schemas.pagination import PagePaginationMeta

# Core-owned since TASK-13950.15; re-exported for the API layer.
from tldw_Server_API.app.core.Evaluations.embeddings_abtest_schemas import (  # noqa: F401 - re-exported (TASK-13950.15)
    ABTestArm,
    ABTestChunking,
    ABTestCleanupPolicy,
    ABTestLimits,
    ABTestQuery,
    ABTestReRanker,
    ABTestRetrieval,
    EmbeddingsABTestConfig,
)
from tldw_Server_API.app.core.Evaluations.run_state import normalize_run_status


class EmbeddingsABTestCreateRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    name: str
    config: EmbeddingsABTestConfig
    run_immediately: bool | None = Field(default=False)


class ArmSummary(BaseModel):
    arm_id: str
    provider: str
    model: str
    dimensions: int | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    latency_ms: dict[str, float] = Field(default_factory=dict)
    doc_counts: dict[str, int] = Field(default_factory=dict)


class EmbeddingsABTestResultSummary(BaseModel):
    test_id: str
    status: Literal['pending', 'running', 'completed', 'failed', 'cancelled']
    arms: list[ArmSummary] = Field(default_factory=list)
    per_query: list[dict[str, Any]] | None = None

    @field_validator("status", mode="before")
    @classmethod
    def _normalize_status(cls, value: Any) -> str:
        return normalize_run_status(value)


class EmbeddingsABTestResultRow(BaseModel):
    model_config = ConfigDict(extra='forbid')
    result_id: str
    test_id: str
    arm_id: str
    query_id: str
    ranked_ids: list[str] = Field(default_factory=list)
    scores: list[float] | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    latency_ms: float | None = None
    ranked_distances: list[float] | None = None
    ranked_metadatas: list[dict[str, Any]] | None = None
    ranked_documents: list[str] | None = None
    rerank_scores: list[float] | None = None
    created_at: str | None = None


class EmbeddingsABTestCreateResponse(BaseModel):
    test_id: str
    status: Literal['pending', 'created'] = 'created'


class EmbeddingsABTestStatusResponse(BaseModel):
    test_id: str
    status: Literal['pending', 'running', 'completed', 'failed', 'cancelled']
    progress: dict[str, float] | None = None

    @field_validator("status", mode="before")
    @classmethod
    def _normalize_status(cls, value: Any) -> str:
        return normalize_run_status(value)


class EmbeddingsABTestResultsResponse(BaseModel):
    summary: EmbeddingsABTestResultSummary
    results: list[EmbeddingsABTestResultRow] = Field(default_factory=list)
    page: int = 1
    page_size: int = 50
    total: int = 0
    pagination: PagePaginationMeta


class EmbeddingsABTestRunRequest(BaseModel):
    """Run request wrapper to enforce validation at the boundary.

    Keep strict extra='forbid' and carry the full AB test config.
    """
    model_config = ConfigDict(extra='forbid')
    config: EmbeddingsABTestConfig
