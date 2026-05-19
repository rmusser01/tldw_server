"""Core result contract for RAG internal mapping."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(slots=True)
class RAGResult:
    """Canonical internal RAG result shape used by response mappers."""

    documents: list[Any]
    query: str
    expanded_queries: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    citations: list[dict[str, Any]] = field(default_factory=list)
    academic_citations: list[str] = field(default_factory=list)
    chunk_citations: list[dict[str, Any]] = field(default_factory=list)
    feedback_id: Optional[str] = None
    generated_answer: Optional[str | dict[str, Any]] = None
    cache_hit: bool = False
    errors: list[str] = field(default_factory=list)
    security_report: Optional[dict[str, Any]] = None
    total_time: float = 0.0
    claims: Optional[list[dict[str, Any]]] = None
    factuality: Optional[dict[str, Any]] = None
    verification_report: Optional[dict[str, Any]] = None
    retrieval_metrics: Optional[dict[str, Any]] = None
    faithfulness: Optional[dict[str, Any]] = None
    query_classification: Optional[dict[str, Any]] = None
    reformulated_query: Optional[str] = None
    research_summary: Optional[dict[str, Any]] = None
    suggestions: Optional[list[str]] = None
    images: Optional[list[dict[str, Any]]] = None
    videos: Optional[list[dict[str, Any]]] = None
