"""Core-owned embeddings A/B test models, re-exported by the API schemas package (TASK-13950.15)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ABTestArm(BaseModel):
    model_config = ConfigDict(extra='forbid')
    provider: str = Field(description="Embedding provider id, e.g., 'openai' or 'huggingface'")
    model: str = Field(description="Embedding model id")
    dimensions: int | None = Field(default=None, description="Requested output dimensions if supported")


class ABTestChunking(BaseModel):
    model_config = ConfigDict(extra='forbid')
    method: str = Field(description="Chunking method, e.g., 'words' or 'sentences'")
    size: int = Field(ge=1, description="Max chunk size")
    overlap: int = Field(ge=0, description="Chunk overlap")
    language: str | None = Field(default=None, description="Language hint")


class ABTestReRanker(BaseModel):
    model_config = ConfigDict(extra='forbid')
    provider: str = Field(description="Reranker provider id")
    model: str = Field(description="Reranker model id")


class ABTestRetrieval(BaseModel):
    model_config = ConfigDict(extra='forbid')
    k: int = Field(ge=1, le=1000, description="Top-k results to return")
    search_mode: Literal['fts', 'vector', 'hybrid'] | None = Field(default='vector', description="Retrieval mode")
    hybrid_alpha: float | None = Field(default=None, description="0=FTS only, 1=Vector only (hybrid blend)")
    re_ranker: ABTestReRanker | None = Field(default=None, description="Optional reranker config")
    index_params: dict[str, str] | None = Field(default=None, description="Index parameters for vector store")
    apply_reranker: bool | None = Field(default=False, description="Apply reranker in vector-only mode when re_ranker is set")


class ABTestQuery(BaseModel):
    model_config = ConfigDict(extra='forbid')
    text: str = Field(description="Query text")
    expected_ids: list[int] | None = Field(default=None, description="Optional ground truth media ids")
    metadata: dict[str, str] | None = Field(default=None, description="Optional query metadata")


class ABTestLimits(BaseModel):
    model_config = ConfigDict(extra='forbid')
    max_docs: int | None = Field(default=None, ge=1, description="Max docs to include from corpus")
    timeout_s: int | None = Field(default=None, ge=1, description="Global timeout in seconds")


class ABTestCleanupPolicy(BaseModel):
    model_config = ConfigDict(extra='forbid')
    on_complete: bool = Field(default=False, description="Delete collections on completion")
    ttl_hours: int | None = Field(default=None, ge=1, description="TTL for collections if not deleted immediately")


class EmbeddingsABTestConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    # Allow single arm for toggle-style tests; still supports classic A/B with 2+ arms
    arms: list[ABTestArm] = Field(min_length=1, description="Embedding models to compare")
    # Permit empty corpus in test-mode where synthetic or pre-existing collections may be used
    media_ids: list[int] = Field(default_factory=list, min_length=0, description="Media IDs to build the corpus from")
    chunking: ABTestChunking | None = None
    retrieval: ABTestRetrieval
    queries: list[ABTestQuery] = Field(min_length=1, description="Queries to evaluate")
    metric_level: Literal['media', 'chunk'] | None = Field(default='media', description="Metric granularity")
    limits: ABTestLimits | None = None
    reuse_existing: bool | None = Field(default=True, description="Reuse matching collections if available")
    cleanup_policy: ABTestCleanupPolicy | None = None
