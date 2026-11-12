"""
Unified RAG Schema - Single comprehensive schema for the unified pipeline

This schema maps directly to the unified_rag_pipeline parameters,
providing a clean API interface with all features accessible.
"""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict

try:
    # Pydantic v2
    from pydantic import model_validator
except Exception:
    model_validator = None  # type: ignore

# Core-owned since TASK-13950.15; re-exported for the API layer.
from tldw_Server_API.app.core.RAG.rag_schemas_unified import (  # noqa: F401 - re-exported (TASK-13950.15)
    KnowledgeSourceEmbeddingStatus,
    KnowledgeSourceHealthEntry,
    KnowledgeSourceHealthResponse,
    KnowledgeSourceIndexStatus,
    UnifiedBatchRequest,
    UnifiedRAGRequest,
    UnifiedRAGResponse,
)

from ._compat import Field


class UnifiedBatchResponse(BaseModel):
    """Response for batch processing."""

    results: list[UnifiedRAGResponse] = Field(
        description="Results for each query"
    )

    total_queries: int = Field(
        description="Total number of queries processed"
    )

    successful: int = Field(
        description="Number of successful queries"
    )

    failed: int = Field(
        description="Number of failed queries"
    )

    total_time: float = Field(
        description="Total batch processing time"
    )

    checkpoint_id: Optional[str] = Field(
        default=None,
        description="Checkpoint ID when checkpointing is enabled.",
    )

    model_config = ConfigDict(frozen=True, json_schema_extra={
        "example": {
            "results": [],  # List of UnifiedRAGResponse objects
            "total_queries": 2,
            "successful": 2,
            "failed": 0,
            "total_time": 0.75,
            "checkpoint_id": "rag_batch_ab12cd34",
        }
    })


class ImplicitFeedbackEvent(BaseModel):
    """Schema for implicit feedback signals from WebUI (click/expand/copy/dwell/citation)."""
    event_type: Literal["click", "expand", "copy", "dwell_time", "citation_used"] = Field(
        description="Type of implicit event"
    )
    query: Optional[str] = Field(default=None, description="Original query text")
    feedback_id: Optional[str] = Field(default=None, description="Optional feedback correlation id")
    doc_id: Optional[str] = Field(default=None, description="Document/chunk id involved")
    chunk_ids: Optional[list[str]] = Field(default=None, description="Optional chunk ids involved")
    rank: Optional[int] = Field(default=None, description="Rank position of the doc in the displayed list")
    impression_list: Optional[list[str]] = Field(
        default=None,
        description="Ordered doc ids visible when the event happened"
    )
    corpus: Optional[str] = Field(default=None, description="Corpus/namespace if set in the request")
    user_id: Optional[str] = Field(default=None, description="User id if available")
    session_id: Optional[str] = Field(default=None, description="Browser session id if available")
    conversation_id: Optional[str] = Field(default=None, description="Conversation id if available")
    message_id: Optional[str] = Field(default=None, description="Message id if available")
    dwell_ms: Optional[int] = Field(default=None, ge=0, description="Dwell time in milliseconds")

    if model_validator is not None:
        @model_validator(mode="before")
        def _validate_dwell_ms(cls, values):  # type: ignore
            if isinstance(values, dict) and values.get("event_type") == "dwell_time":
                dwell_ms = values.get("dwell_ms")
                if dwell_ms is None:
                    raise ValueError("dwell_ms is required when event_type='dwell_time'")
            return values

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "event_type": "dwell_time",
            "query": "how to reset auth",
            "doc_id": "doc_1",
            "chunk_ids": ["chunk_9"],
            "rank": 2,
            "impression_list": ["doc_1", "doc_2", "doc_3"],
            "corpus": "media_db",
            "session_id": "sess_abc123",
            "conversation_id": "C_...",
            "message_id": "M_...",
            "dwell_ms": 3000,
        }
    })
