"""
Unified RAG Pipeline - Single Function with All Features

This module provides a single, unified RAG pipeline function where ALL features
are accessible via explicit parameters. No configuration files, no presets,
just direct parameter control.

Design Philosophy:
- One function to rule them all
- Every feature is an optional parameter
- No hidden configuration
- Transparent execution flow
- Mix and match any features
"""

import asyncio
import calendar
import hashlib
import inspect
import os
import re
import sqlite3
import time
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any, Callable, Literal, Optional, cast

from loguru import logger

from tldw_Server_API.app.core.DB_Management.media_db.api import (
    managed_media_database,
)
from tldw_Server_API.app.core.LLM_Calls.structured_output import (
    StructuredOutputOptions,
    StructuredOutputParseError,
    parse_structured_output,
)
from tldw_Server_API.app.core.testing import (
    is_test_mode as _shared_is_test_mode,
)
from tldw_Server_API.app.core.testing import (
    is_truthy as _shared_is_truthy,
)

from .retrieval_executor import execute_retrieval_phase
from .generation_executor import execute_generation_phase
from .retrieval_plan import RetrievalPlan

_RERANK_DEBUG_DOCUMENT_CONTENT_MAX_CHARS = 500


def _serialize_result_document(doc: Any) -> dict[str, Any]:
    """Serialize a pipeline document into the response-compatible document shape."""
    if isinstance(doc, dict):
        metadata = dict(doc.get("metadata") or {})
        source = doc.get("source")
        if source is not None:
            metadata.setdefault(
                "source",
                source.value if hasattr(source, "value") else str(source),
            )
        for field_name in ("media_id", "note_id", "chunk_id", "record_id", "start", "end"):
            value = doc.get(field_name)
            if value is not None:
                metadata.setdefault(field_name, value)
        doc_id = doc.get("id")
        if doc_id is not None:
            metadata.setdefault("chunk_id", str(doc_id))
        return {
            "id": doc_id,
            "content": doc.get("content") or doc.get("text") or doc.get("body"),
            "score": doc.get("score", 0.0),
            "metadata": metadata,
        }

    metadata = dict(getattr(doc, "metadata", {}) or {})
    try:
        source = getattr(doc, "source", None)
        if source is not None:
            metadata.setdefault(
                "source",
                source.value if hasattr(source, "value") else str(source),
            )
    except (AttributeError, TypeError, ValueError):
        pass

    doc_id = getattr(doc, "id", None)
    if doc_id is not None:
        metadata.setdefault("chunk_id", str(doc_id))

    return {
        "id": doc_id,
        "content": getattr(doc, "content", None),
        "score": getattr(doc, "score", 0.0),
        "metadata": metadata,
    }


def _resolve_include_rerank_debug_documents(explicit_flag: Any) -> bool:
    """Resolve whether rerank debug snapshots are enabled."""
    if explicit_flag is None:
        return _shared_is_truthy(os.getenv("RAG_INCLUDE_RERANK_DEBUG_DOCUMENTS", "false"))
    if isinstance(explicit_flag, bool):
        return explicit_flag
    return _shared_is_truthy(explicit_flag)


def _truncate_rerank_debug_content(content: Any, *, max_chars: int = _RERANK_DEBUG_DOCUMENT_CONTENT_MAX_CHARS) -> str | None:
    if content is None:
        return None
    text = str(content)
    if max_chars > 0 and len(text) > max_chars:
        return f"{text[:max_chars].rstrip()}..."
    return text


def _serialize_rerank_debug_document(
    doc: Any,
    *,
    include_content: bool,
    max_content_chars: int = _RERANK_DEBUG_DOCUMENT_CONTENT_MAX_CHARS,
) -> dict[str, Any]:
    """Serialize rerank debug snapshots without duplicating the full document body by default."""
    serialized = _serialize_result_document(doc)
    metadata = dict(serialized.get("metadata", {}) or {})

    def _pick(field_name: str) -> Any:
        value = serialized.get(field_name)
        if value is None:
            value = metadata.get(field_name)
        return value

    debug_doc: dict[str, Any] = {}
    doc_id = _pick("id") or _pick("chunk_id") or _pick("record_id")
    if doc_id is not None:
        debug_doc["id"] = doc_id
    score = _pick("score")
    if score is not None:
        debug_doc["score"] = score
    source = _pick("source")
    if source is not None:
        debug_doc["source"] = source

    identity_fields = (
        "chunk_id",
        "media_id",
        "note_id",
        "record_id",
        "sql_target_id",
        "start",
        "end",
        "chunk_start",
        "chunk_end",
        "page_number",
        "section_title",
        "title",
        "path",
    )
    debug_metadata: dict[str, Any] = {}
    for field_name in identity_fields:
        value = _pick(field_name)
        if value is None:
            continue
        if field_name in {"chunk_id", "media_id", "note_id", "record_id", "start", "end"}:
            debug_doc[field_name] = value
        debug_metadata[field_name] = value
    if debug_metadata:
        debug_doc["metadata"] = debug_metadata

    if include_content:
        content = _truncate_rerank_debug_content(
            serialized.get("content"),
            max_chars=max_content_chars,
        )
        if content:
            debug_doc["content"] = content

    return debug_doc


def _serialize_rerank_debug_documents(
    documents: list[Any] | None,
    *,
    include_content: bool,
    limit: int,
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    return [
        _serialize_rerank_debug_document(doc, include_content=include_content)
        for doc in list(documents or [])[:limit]
    ]


# Optional dependency placeholders (typed as Any to keep mypy tolerant for missing deps).
_get_telemetry_manager: Any = None
_spell_check_query: Any = None
_highlight_results: Any = None
_track_llm_cost: Any = None
_QueryAnalyzer: Any = None
_QueryIntent: Any = None
_QueryRouter: Any = None
_QueryRewriter: Any = None
_generate_hypothetical_answer: Any = None
_hyde_embed_text: Any = None
_expand_acronyms: Any = None
_expand_synonyms: Any = None
_entity_recognition_expansion: Any = None
_domain_specific_expansion: Any = None
_multi_strategy_expansion: Any = None
_SemanticCache: Any = None
_AdaptiveCache: Any = None
_get_shared_cache: Any = None
_MultiDatabaseRetriever: Any = None
_RetrievalConfig: Any = None
_SQLRetriever: Any = None
_SecurityFilters: Any = None
_SensitivityLevel: Any = None
_TableProcessor: Any = None
_enhanced_chunk_documents: Any = None
_filter_chunks_by_type: Any = None
_expand_with_parent_context: Any = None
_prioritize_by_chunk_type: Any = None
_create_reranker: Any = None
_RerankingStrategy: Any = None
_RerankingConfig: Any = None
_apply_multi_vector_passages: Any = None
_MultiVectorConfig: Any = None
_PRFConfig: Any = None
_apply_prf: Any = None
_PrecomputedSpanConfig: Any = None
_apply_precomputed_spans: Any = None
_RewriteCache: Any = None
_load_prompt: Any = None
_Chunker: Any = None
_ChunkerConfig: Any = None
_CitationGenerator: Any = None
_CitationStyle: Any = None
_GranularityRouter: Any = None
_GranularityDecision: Any = None
_route_query_granularity: Any = None
_QueryType: Any = None
_Granularity: Any = None
_EvidenceAccumulator: Any = None
_AccumulationResult: Any = None
_accumulate_evidence: Any = None
_EvidenceChainBuilder: Any = None
_ChainBuildResult: Any = None
_build_evidence_chains: Any = None
_DocumentGrader: Any = None
_GradingConfig: Any = None
_grade_and_filter_documents: Any = None
_WebFallbackConfig: Any = None
_web_search_fallback: Any = None
_merge_web_results: Any = None
_fallback_to_web_search: Any = None
_KnowledgeStripsProcessor: Any = None
_KnowledgeStripsResult: Any = None
_process_knowledge_strips: Any = None
_FastGroundednessGrader: Any = None
_FastGroundednessResult: Any = None
_UtilityGrader: Any = None
_UtilityResult: Any = None
_check_fast_groundedness: Any = None
_grade_utility: Any = None
_AnswerGenerator: Any = None
_PostGenerationVerifier: Any = None
_rag_low_conf: Any = None
_rag_req_hc: Any = None
_downweight_injection_docs: Any = None
_detect_injection_score: Any = None
_check_numeric_fidelity: Any = None
_build_hard_citations: Any = None
_build_quote_citations: Any = None
_sanitize_html_allowlist: Any = None
_apply_content_policy: Any = None
_gate_docs_by_ocr_confidence: Any = None
_UnifiedFeedbackSystem: Any = None
_UserPersonalizationStore: Any = None
_Tracer: Any = None
_get_coordinator: Any = None
_CircuitBreakerConfig: Any = None
_RetryConfig: Any = None
_RetryPolicy: Any = None
_PerformanceMonitor: Any = None
_ClaimsEngine: Any = None
_ClaimsJobContext: Any = None
_resolve_claims_job_budget: Any = None
try:
    # OpenTelemetry telemetry manager (metrics + tracing)
    from tldw_Server_API.app.core.Metrics.telemetry import (
        OTEL_AVAILABLE as _OTEL_AVAILABLE,
    )
    from tldw_Server_API.app.core.Metrics.telemetry import (
        get_telemetry_manager as _get_telemetry_manager,
    )
except ImportError:  # pragma: no cover - optional dependency
    _get_telemetry_manager = None
    _OTEL_AVAILABLE = False

get_telemetry_manager = _get_telemetry_manager
OTEL_AVAILABLE = _OTEL_AVAILABLE

class _NoopSpan:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


def otel_span(name: str, *args, **kwargs):
    if get_telemetry_manager is not None and OTEL_AVAILABLE:
        try:
            tm = get_telemetry_manager()
            tracer = tm.get_tracer("tldw.rag")
            return tracer.start_as_current_span(name, *args, **kwargs)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass
    return _NoopSpan()

# Core types
import contextlib

from .metrics_collector import MetricsCollector, QueryMetrics
from .evidence_models import RetrievedEvidence
from .post_retrieval_coordinator import coordinate_standard_result_evidence
from .request_resolution import (
    ResolvedRAGRequest,
    resolve_legacy_standard_pipeline_request,
)
from .result_model import RAGResult
from .retrieval_plan import build_retrieval_plan
from .types import DataSource, Document

try:
    from tldw_Server_API.app.core.Text2SQL.source_registry import (
        normalize_sources_internal as _normalize_sources_internal,
    )
except ImportError:
    _normalize_sources_internal = None

normalize_sources_internal = _normalize_sources_internal

# Import all modules at module level to avoid 500ms overhead
try:
    from .quick_wins import (
        highlight_results as _highlight_results,
    )
    from .quick_wins import (
        spell_check_query as _spell_check_query,
    )
    from .quick_wins import (
        track_llm_cost as _track_llm_cost,
    )
except ImportError:
    _spell_check_query = None
    _highlight_results = None
    _track_llm_cost = None

spell_check_query = _spell_check_query
highlight_func = _highlight_results
track_llm_cost = _track_llm_cost

# Query intent analysis / routing
try:
    from .query_features import (
        QueryAnalyzer as _QueryAnalyzer,
    )
    from .query_features import (
        QueryIntent as _QueryIntent,
    )
    from .query_features import (
        QueryRewriter as _QueryRewriter,
    )
    from .query_features import (
        QueryRouter as _QueryRouter,
    )
except ImportError:
    _QueryAnalyzer = None
    _QueryIntent = None
    _QueryRouter = None
    _QueryRewriter = None

QueryAnalyzer = _QueryAnalyzer
QueryIntent = _QueryIntent
QueryRouter = _QueryRouter
QueryRewriter = _QueryRewriter

# Query classifier (agentic search routing)
_classify_and_reformulate: Any = None
_QueryClassification: Any = None
_research_loop: Any = None
_create_default_registry: Any = None
_ClarificationDecision: Any = None
_assess_query_for_clarification: Any = None
try:
    from .query_classifier import (
        QueryClassification as _QueryClassification,
    )
    from .query_classifier import (
        classify_and_reformulate as _classify_and_reformulate,
    )
    from .research_agent import (
        create_default_registry as _create_default_registry,
    )
    from .research_agent import (
        research_loop as _research_loop,
    )
except ImportError:
    _classify_and_reformulate = None
    _QueryClassification = None
    _research_loop = None
    _create_default_registry = None

try:
    from .clarification_gate import (
        ClarificationDecision as _ClarificationDecision,
    )
    from .clarification_gate import (
        assess_query_for_clarification as _assess_query_for_clarification,
    )
except ImportError:
    _ClarificationDecision = None
    _assess_query_for_clarification = None

classify_and_reformulate = _classify_and_reformulate
QueryClassification = _QueryClassification
research_loop = _research_loop
create_default_registry = _create_default_registry
ClarificationDecision = _ClarificationDecision
assess_query_for_clarification = _assess_query_for_clarification

# Suggestion generator
_generate_suggestions: Any = None
try:
    from .suggestion_generator import generate_suggestions as _generate_suggestions
except ImportError:
    _generate_suggestions = None
generate_suggestions = _generate_suggestions

# Structured response writer
_format_context_xml: Any = None
_build_writer_system_prompt: Any = None
_build_writer_user_prompt: Any = None
_get_writer_depth_policy: Any = None
try:
    from .response_writer import (
        build_writer_system_prompt as _build_writer_system_prompt,
    )
    from .response_writer import (
        build_writer_user_prompt as _build_writer_user_prompt,
    )
    from .response_writer import (
        format_context_xml as _format_context_xml,
    )
    from .response_writer import (
        get_writer_depth_policy as _get_writer_depth_policy,
    )
except ImportError:
    _format_context_xml = None
    _build_writer_system_prompt = None
    _build_writer_user_prompt = None
    _get_writer_depth_policy = None
format_context_xml = _format_context_xml
build_writer_system_prompt = _build_writer_system_prompt
build_writer_user_prompt = _build_writer_user_prompt
get_writer_depth_policy = _get_writer_depth_policy

# HyDE utilities
try:
    from .hyde import (
        embed_text as _hyde_embed_text,
    )
    from .hyde import (
        generate_hypothetical_answer as _generate_hypothetical_answer,
    )
except ImportError:
    _generate_hypothetical_answer = None
    _hyde_embed_text = None

generate_hypothetical_answer = _generate_hypothetical_answer
hyde_embed_text = _hyde_embed_text

try:
    from .query_expansion import (
        domain_specific_expansion as _domain_specific_expansion,
    )
    from .query_expansion import (
        entity_recognition_expansion as _entity_recognition_expansion,
    )
    from .query_expansion import (
        expand_acronyms as _expand_acronyms,
    )
    from .query_expansion import (
        expand_synonyms as _expand_synonyms,
    )
    from .query_expansion import (
        multi_strategy_expansion as _multi_strategy_expansion,
    )
except ImportError:
    _expand_acronyms = None
    _expand_synonyms = None
    _entity_recognition_expansion = None
    _domain_specific_expansion = None
    _multi_strategy_expansion = None

expand_acronyms = _expand_acronyms
expand_synonyms = _expand_synonyms
entity_recognition_expansion = _entity_recognition_expansion
domain_specific_expansion = _domain_specific_expansion
multi_strategy_expansion = _multi_strategy_expansion

try:
    from .semantic_cache import (
        AdaptiveCache as _AdaptiveCache,
    )
    from .semantic_cache import (
        SemanticCache as _SemanticCache,
    )
    from .semantic_cache import (
        get_shared_cache as _get_shared_cache,
    )
except ImportError:
    _SemanticCache = None
    _AdaptiveCache = None
    _get_shared_cache = None

SemanticCache = _SemanticCache
AdaptiveCache = _AdaptiveCache
get_shared_cache = _get_shared_cache

try:
    from .database_retrievers import (
        MultiDatabaseRetriever as _MultiDatabaseRetriever,
    )
    from .database_retrievers import (
        RetrievalConfig as _RetrievalConfig,
    )
    from .database_retrievers import (
        SQLRetriever as _SQLRetriever,
    )
except ImportError:
    _MultiDatabaseRetriever = None
    _RetrievalConfig = None
    _SQLRetriever = None

MultiDatabaseRetriever = _MultiDatabaseRetriever
RetrievalConfig = _RetrievalConfig
SQLRetriever = _SQLRetriever

try:
    from .security_filters import SecurityFilters as _SecurityFilters
    from .security_filters import SensitivityLevel as _SensitivityLevel
except ImportError:
    _SecurityFilters = None
    _SensitivityLevel = None

SecurityFilter = _SecurityFilters
SensitivityLevel = _SensitivityLevel

try:
    from .table_serialization import TableProcessor as _TableProcessor
except ImportError:
    _TableProcessor = None

TableProcessor = _TableProcessor

try:
    from .enhanced_chunking_integration import (
        enhanced_chunk_documents as _enhanced_chunk_documents,
    )
    from .enhanced_chunking_integration import (
        expand_with_parent_context as _expand_with_parent_context,
    )
    from .enhanced_chunking_integration import (
        filter_chunks_by_type as _filter_chunks_by_type,
    )
    from .enhanced_chunking_integration import (
        prioritize_by_chunk_type as _prioritize_by_chunk_type,
    )
except ImportError:
    _enhanced_chunk_documents = None
    _filter_chunks_by_type = None
    _expand_with_parent_context = None
    _prioritize_by_chunk_type = None

enhanced_chunk_documents = _enhanced_chunk_documents
filter_chunks_by_type = _filter_chunks_by_type
expand_with_parent_context = _expand_with_parent_context
prioritize_by_chunk_type = _prioritize_by_chunk_type

try:
    from .advanced_reranking import (
        RerankingConfig as _RerankingConfig,
    )
    from .advanced_reranking import (
        RerankingStrategy as _RerankingStrategy,
    )
    from .advanced_reranking import (
        create_reranker as _create_reranker,
    )
except ImportError:
    _create_reranker = None
    _RerankingStrategy = None
    _RerankingConfig = None

create_reranker = _create_reranker
RerankingStrategy = _RerankingStrategy
RerankingConfig = _RerankingConfig

# Advanced retrieval (multi-vector passages)
try:
    from .advanced_retrieval import (
        MultiVectorConfig as _MultiVectorConfig,
    )
    from .advanced_retrieval import (
        apply_multi_vector_passages as _apply_multi_vector_passages,
    )
except ImportError:
    _apply_multi_vector_passages = None
    _MultiVectorConfig = None

apply_multi_vector_passages = _apply_multi_vector_passages
MultiVectorConfig = _MultiVectorConfig

# Pseudo-relevance feedback (PRF)
try:
    from .prf import PRFConfig as _PRFConfig
    from .prf import apply_prf as _apply_prf
except ImportError:
    _PRFConfig = None
    _apply_prf = None

PRFConfig = _PRFConfig
apply_prf = _apply_prf

# Precomputed span index (multi-vector helper)
try:
    from .precomputed_spans import (
        PrecomputedSpanConfig as _PrecomputedSpanConfig,
    )
    from .precomputed_spans import (
        apply_precomputed_spans as _apply_precomputed_spans,
    )
except ImportError:
    _PrecomputedSpanConfig = None
    _apply_precomputed_spans = None

PrecomputedSpanConfig = _PrecomputedSpanConfig
apply_precomputed_spans = _apply_precomputed_spans

try:
    from .rewrite_cache import RewriteCache as _RewriteCache
except ImportError:
    _RewriteCache = None

RewriteCache = _RewriteCache

try:
    from tldw_Server_API.app.core.Utils.prompt_loader import load_prompt as _load_prompt
except ImportError:
    _load_prompt = None

def load_prompt(*args, **kwargs):
    if _load_prompt is None:
        return None
    return _load_prompt(*args, **kwargs)

# Chunking support
try:
    from tldw_Server_API.app.core.Chunking import Chunker as _Chunker
    from tldw_Server_API.app.core.Chunking import ChunkerConfig as _ChunkerConfig
except ImportError:
    _Chunker = None
    _ChunkerConfig = None

Chunker = _Chunker
ChunkerConfig = _ChunkerConfig
try:
    from .citations import CitationGenerator as _CitationGenerator
    from .citations import CitationStyle as _CitationStyle
except ImportError:
    _CitationGenerator = None
    _CitationStyle = None

CitationGenerator = _CitationGenerator
CitationStyle = _CitationStyle

# Dynamic granularity routing
try:
    from .granularity_router import (
        GranularityDecision as _GranularityDecision,
    )
    from .granularity_router import (
        GranularityRouter as _GranularityRouter,
    )
    from .granularity_router import (
        route_query_granularity as _route_query_granularity,
    )
    from .types import Granularity as _Granularity
    from .types import QueryType as _QueryType
except ImportError:
    _GranularityRouter = None
    _GranularityDecision = None
    _route_query_granularity = None
    _QueryType = None
    _Granularity = None

GranularityRouter = _GranularityRouter
GranularityDecision = _GranularityDecision
route_query_granularity = _route_query_granularity
QueryType = _QueryType
Granularity = _Granularity

# Progressive evidence accumulation
try:
    from .evidence_accumulator import (
        AccumulationResult as _AccumulationResult,
    )
    from .evidence_accumulator import (
        EvidenceAccumulator as _EvidenceAccumulator,
    )
    from .evidence_accumulator import (
        accumulate_evidence as _accumulate_evidence,
    )
except ImportError:
    _EvidenceAccumulator = None
    _AccumulationResult = None
    _accumulate_evidence = None

EvidenceAccumulator = _EvidenceAccumulator
AccumulationResult = _AccumulationResult
accumulate_evidence = _accumulate_evidence

# Multi-hop evidence chains
try:
    from .evidence_chains import (
        ChainBuildResult as _ChainBuildResult,
    )
    from .evidence_chains import (
        EvidenceChainBuilder as _EvidenceChainBuilder,
    )
    from .evidence_chains import (
        build_evidence_chains as _build_evidence_chains,
    )
except ImportError:
    _EvidenceChainBuilder = None
    _ChainBuildResult = None
    _build_evidence_chains = None

EvidenceChainBuilder = _EvidenceChainBuilder
ChainBuildResult = _ChainBuildResult
build_evidence_chains = _build_evidence_chains

# Self-Correcting RAG: Document Grading (Stage 1)
try:
    from .document_grader import (
        DocumentGrader as _DocumentGrader,
    )
    from .document_grader import (
        GradingConfig as _GradingConfig,
    )
    from .document_grader import (
        grade_and_filter_documents as _grade_and_filter_documents,
    )
except ImportError:
    _DocumentGrader = None
    _GradingConfig = None
    _grade_and_filter_documents = None

DocumentGrader = _DocumentGrader
GradingConfig = _GradingConfig
grade_and_filter_documents = _grade_and_filter_documents

# Self-Correcting RAG: Web Fallback (Stage 3)
try:
    from .web_fallback import (
        WebFallbackConfig as _WebFallbackConfig,
    )
    from .web_fallback import (
        fallback_to_web_search as _fallback_to_web_search,
    )
    from .web_fallback import (
        merge_web_results as _merge_web_results,
    )
    from .web_fallback import (
        web_search_fallback as _web_search_fallback,
    )
except ImportError:
    _WebFallbackConfig = None
    _web_search_fallback = None
    _merge_web_results = None
    _fallback_to_web_search = None

WebFallbackConfig = _WebFallbackConfig
web_search_fallback = _web_search_fallback
merge_web_results = _merge_web_results
fallback_to_web_search = _fallback_to_web_search

# Self-Correcting RAG: Knowledge Strips (Stage 4)
try:
    from .knowledge_strips import (
        KnowledgeStripsProcessor as _KnowledgeStripsProcessor,
    )
    from .knowledge_strips import (
        KnowledgeStripsResult as _KnowledgeStripsResult,
    )
    from .knowledge_strips import (
        process_knowledge_strips as _process_knowledge_strips,
    )
except ImportError:
    _KnowledgeStripsProcessor = None
    _KnowledgeStripsResult = None
    _process_knowledge_strips = None

KnowledgeStripsProcessor = _KnowledgeStripsProcessor
KnowledgeStripsResult = _KnowledgeStripsResult
process_knowledge_strips = _process_knowledge_strips

# Self-Correcting RAG: Quality Graders (Stages 5-6)
try:
    from .quality_graders import (
        FastGroundednessGrader as _FastGroundednessGrader,
    )
    from .quality_graders import (
        FastGroundednessResult as _FastGroundednessResult,
    )
    from .quality_graders import (
        UtilityGrader as _UtilityGrader,
    )
    from .quality_graders import (
        UtilityResult as _UtilityResult,
    )
    from .quality_graders import (
        check_fast_groundedness as _check_fast_groundedness,
    )
    from .quality_graders import (
        grade_utility as _grade_utility,
    )
except ImportError:
    _FastGroundednessGrader = None
    _FastGroundednessResult = None
    _UtilityGrader = None
    _UtilityResult = None
    _check_fast_groundedness = None
    _grade_utility = None

FastGroundednessGrader = _FastGroundednessGrader
FastGroundednessResult = _FastGroundednessResult
UtilityGrader = _UtilityGrader
UtilityResult = _UtilityResult
check_fast_groundedness = _check_fast_groundedness
grade_utility = _grade_utility

try:
    from .generation import AnswerGenerator as _AnswerGenerator
except ImportError:
    _AnswerGenerator = None

AnswerGenerator = _AnswerGenerator

try:
    from .post_generation_verifier import PostGenerationVerifier as _PostGenerationVerifier
except ImportError:
    _PostGenerationVerifier = None

PostGenerationVerifier = _PostGenerationVerifier

# RAG config helpers for consistent toggles/defaults
try:
    from tldw_Server_API.app.core.config import (
        rag_low_confidence_behavior as _rag_low_conf,
    )
    from tldw_Server_API.app.core.config import (
        rag_require_hard_citations as _rag_req_hc,
    )
except ImportError:
    _rag_low_conf = None
    _rag_req_hc = None

try:
    # Guardrails utilities: injection filtering, numeric fidelity, hard citations
    from .guardrails import (
        apply_content_policy as _apply_content_policy,
    )
    from .guardrails import (
        build_hard_citations as _build_hard_citations,
    )
    from .guardrails import (
        build_quote_citations as _build_quote_citations,
    )
    from .guardrails import (
        check_numeric_fidelity as _check_numeric_fidelity,
    )
    from .guardrails import (
        detect_injection_score as _detect_injection_score,
    )
    from .guardrails import (
        downweight_injection_docs as _downweight_injection_docs,
    )
    from .guardrails import (
        gate_docs_by_ocr_confidence as _gate_docs_by_ocr_confidence,
    )
    from .guardrails import (
        sanitize_html_allowlist as _sanitize_html_allowlist,
    )
except ImportError:
    _downweight_injection_docs = None
    _detect_injection_score = None
    _check_numeric_fidelity = None
    _build_hard_citations = None
    _build_quote_citations = None
    _sanitize_html_allowlist = None
    _apply_content_policy = None
    _gate_docs_by_ocr_confidence = None

downweight_injection_docs = _downweight_injection_docs
detect_injection_score = _detect_injection_score
check_numeric_fidelity = _check_numeric_fidelity
build_hard_citations = _build_hard_citations
build_quote_citations = _build_quote_citations
sanitize_html_allowlist = _sanitize_html_allowlist
apply_content_policy = _apply_content_policy
gate_docs_by_ocr_confidence = _gate_docs_by_ocr_confidence

try:
    from .analytics_system import UnifiedFeedbackSystem as _UnifiedFeedbackSystem
except ImportError:
    _UnifiedFeedbackSystem = None

UnifiedFeedbackSystem = _UnifiedFeedbackSystem


def _normalize_chunk_type_value(value: Any) -> Optional[str]:
    try:
        if Chunker is not None:
            normalized = Chunker.normalize_chunk_type(value)
            if isinstance(normalized, str) and normalized:
                return normalized
    except (AttributeError, TypeError, ValueError):
        pass
    if value is None:
        return None
    try:
        raw = str(value).strip().lower()
    except (AttributeError, TypeError, ValueError):
        return None
    if not raw:
        return None
    aliases = {
        "header": "heading",
        "header_atx": "heading",
        "header_line": "heading",
        "hr": "heading",
        "paragraph": "text",
        "list_unordered": "list",
        "list_ordered": "list",
        "code_fence": "code",
        "table_md": "table",
    }
    if raw in aliases:
        return aliases[raw]
    if raw in {"image", "video", "audio", "file", "media"}:
        return "media"
    return raw


def _coerce_security_filter_sequence(value: Any, *, description: str) -> list[Any]:
    """Normalize security-filter outputs and fail closed on invalid shapes."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, (str, bytes, bytearray, dict)):
        logger.warning(
            "Security filter returned unsupported {} type {}; dropping all documents",
            description,
            type(value).__name__,
        )
        return []
    try:
        return list(value)
    except TypeError:
        logger.warning(
            "Security filter returned non-iterable {} type {}; dropping all documents",
            description,
            type(value).__name__,
        )
        return []

try:
    from .user_personalization_store import UserPersonalizationStore as _UserPersonalizationStore
except ImportError:
    _UserPersonalizationStore = None

UserPersonalizationStore = _UserPersonalizationStore

try:
    from .observability import Tracer as _Tracer
except ImportError:
    _Tracer = None

Tracer = _Tracer

# Resilience helpers
try:
    from .resilience import (
        CircuitBreakerConfig as _CircuitBreakerConfig,
    )
    from .resilience import (
        RetryConfig as _RetryConfig,
    )
    from .resilience import (
        RetryPolicy as _RetryPolicy,
    )
    from .resilience import (
        get_coordinator as _get_coordinator,
    )
except ImportError:
    _get_coordinator = None
    _CircuitBreakerConfig = None
    _RetryConfig = None
    _RetryPolicy = None

get_coordinator = _get_coordinator
CircuitBreakerConfig = _CircuitBreakerConfig
RetryConfig = _RetryConfig
RetryPolicy = _RetryPolicy

try:
    from .performance_monitor import PerformanceMonitor as _PerformanceMonitor
except ImportError:
    _PerformanceMonitor = None

PerformanceMonitor = _PerformanceMonitor

# Claims extraction/verification
try:
    from tldw_Server_API.app.core.Claims_Extraction.budget_guard import (
        ClaimsJobContext as _ClaimsJobContext,
    )
    from tldw_Server_API.app.core.Claims_Extraction.budget_guard import (
        resolve_claims_job_budget as _resolve_claims_job_budget,
    )

    from .claims import ClaimsEngine as _ClaimsEngine
except ImportError:
    _ClaimsEngine = None
    _ClaimsJobContext = None
    _resolve_claims_job_budget = None

ClaimsEngine = _ClaimsEngine
ClaimsJobContext = _ClaimsJobContext
resolve_claims_job_budget = _resolve_claims_job_budget


@dataclass
class UnifiedSearchResult:
    """Unified result structure for all RAG queries."""
    documents: list[Document]
    query: str
    expanded_queries: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    citations: list[dict[str, Any]] = field(default_factory=list)
    feedback_id: Optional[str] = None
    generated_answer: Optional[str] = None
    cache_hit: bool = False
    errors: list[str] = field(default_factory=list)
    security_report: Optional[dict[str, Any]] = None
    total_time: float = 0.0


# Unified pipeline return type (schema response or internal result).
# This pipeline returns mixed shapes (schema/model dicts or dataclasses),
# so keep it permissive for now.
UnifiedPipelineResult = Any


def build_retrieval_only_result(
    *,
    resolved_request: ResolvedRAGRequest,
    retrieval_plan: RetrievalPlan,
    retrieval_result: RetrievedEvidence,
) -> RAGResult:
    metadata = dict(retrieval_result.metadata or {})
    metadata.setdefault(
        "retrieval_plan",
        {
            "query": retrieval_plan.query,
            "sources": list(retrieval_plan.sources),
            "search_mode": retrieval_plan.search_mode,
            "top_k": retrieval_plan.top_k,
            "index_namespace": retrieval_plan.index_namespace,
        },
    )
    return RAGResult(
        query=resolved_request.query,
        documents=list(retrieval_result.documents),
        metadata=metadata,
    )


_CANONICAL_SOURCE_TO_DATASOURCE: dict[str, DataSource] = {
    "media_db": DataSource.MEDIA_DB,
    "notes": DataSource.NOTES,
    "characters": DataSource.CHARACTER_CARDS,
    "chats": DataSource.CHAT_HISTORY,
    "kanban": DataSource.KANBAN,
    "sql": DataSource.SQL,
    "prompts": DataSource.PROMPTS,
    "world_books": DataSource.WORLD_BOOKS,
    "dictionaries": DataSource.DICTIONARIES,
    "claims": DataSource.CLAIMS,
}

_DATASOURCE_TO_CANONICAL_SOURCE: dict[DataSource, str] = {
    DataSource.MEDIA_DB: "media_db",
    DataSource.NOTES: "notes",
    DataSource.CHARACTER_CARDS: "characters",
    DataSource.CHAT_HISTORY: "chats",
    DataSource.KANBAN: "kanban",
    DataSource.SQL: "sql",
    DataSource.PROMPTS: "prompts",
    DataSource.WORLD_BOOKS: "world_books",
    DataSource.DICTIONARIES: "dictionaries",
    DataSource.CLAIMS: "claims",
}


def _normalize_pipeline_sources(sources: Optional[list[Any]]) -> list[str]:
    """Normalize source tokens using the canonical source registry."""
    incoming: list[str] = []
    if sources is None:
        incoming = ["media_db"]
    else:
        for source in sources:
            if isinstance(source, DataSource):
                canonical = _DATASOURCE_TO_CANONICAL_SOURCE.get(source)
                if canonical is None:
                    raise ValueError(f"Invalid source '{source}'")
                incoming.append(canonical)
            else:
                incoming.append(str(source))

    if normalize_sources_internal is None:
        normalized = [str(value).strip().lower() for value in incoming]
    else:
        normalized = normalize_sources_internal(incoming)

    for normalized_source in normalized:
        if normalized_source not in _CANONICAL_SOURCE_TO_DATASOURCE:
            raise ValueError(f"Invalid source '{normalized_source}'")
    return normalized


def _sources_to_data_sources(sources: list[str]) -> list[DataSource]:
    """Map canonical source ids into DataSource enum values."""
    data_sources: list[DataSource] = []
    for source in sources:
        mapped = _CANONICAL_SOURCE_TO_DATASOURCE.get(source)
        if mapped is None:
            raise ValueError(f"Invalid source '{source}'")
        data_sources.append(mapped)
    return data_sources


def _scope_includes_data_source(
    resolved_sources: Optional[list[DataSource]],
    source: DataSource,
) -> bool:
    """Return whether an already-normalized retrieval scope includes a source."""
    return source in (resolved_sources or [])


def _has_explicit_include_values(values: Any) -> bool:
    """Return whether a source include filter carries at least one concrete id."""
    if values is None:
        return False
    if isinstance(values, (list, tuple, set)):
        return any(str(value).strip() for value in values if value is not None)
    return bool(str(values).strip())


def _explicit_include_data_sources(
    *,
    include_media_ids: Any,
    include_note_ids: Any,
) -> list[DataSource]:
    """Return source types that have explicit include allowlists."""
    explicit_sources: list[DataSource] = []
    if _has_explicit_include_values(include_media_ids):
        explicit_sources.append(DataSource.MEDIA_DB)
    if _has_explicit_include_values(include_note_ids):
        explicit_sources.append(DataSource.NOTES)
    return explicit_sources


def _metadata_truthy(value: Any) -> bool:
    """Interpret common metadata flag encodings as booleans."""
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y", "on"}
    return bool(value)


def _document_metadata(doc: Any) -> dict[str, Any]:
    """Return a defensive copy of a retrieved document's metadata mapping."""
    if isinstance(doc, dict):
        metadata = doc.get("metadata")
        return dict(metadata) if isinstance(metadata, dict) else {}
    metadata = getattr(doc, "metadata", None)
    return dict(metadata) if isinstance(metadata, dict) else {}


def _document_canonical_source(doc: Any) -> str:
    """Resolve a document's public source id from its source field or metadata."""
    source_value: Any = None
    if isinstance(doc, dict):
        source_value = doc.get("source")
        metadata = doc.get("metadata")
        if source_value is None and isinstance(metadata, dict):
            source_value = metadata.get("source")
    else:
        source_value = getattr(doc, "source", None)
        if source_value is None:
            metadata = getattr(doc, "metadata", None)
            if isinstance(metadata, dict):
                source_value = metadata.get("source")

    if isinstance(source_value, DataSource):
        return _DATASOURCE_TO_CANONICAL_SOURCE.get(source_value, source_value.value)
    source_text = str(source_value or "").strip().lower()
    if not source_text:
        return "media_db"
    mapped = _CANONICAL_SOURCE_TO_DATASOURCE.get(source_text)
    if mapped is not None:
        return _DATASOURCE_TO_CANONICAL_SOURCE.get(mapped, source_text)
    aliases = {
        "media": "media_db",
        "notes_db": "notes",
        "chat_history": "chats",
        "character_cards": "characters",
        "character_cards_db": "characters",
        "kanban_db": "kanban",
        "task_boards": "kanban",
        "prompts_db": "prompts",
        "worldbooks": "world_books",
        "world_book": "world_books",
        "world_books_db": "world_books",
        "dictionary": "dictionaries",
        "chat_dictionary": "dictionaries",
        "chat_dictionaries": "dictionaries",
        "chat_dictionaries_db": "dictionaries",
    }
    return aliases.get(source_text, source_text)


def _filter_workspace_artifacts(
    documents: list[Any],
    *,
    workspace_id: Optional[str],
) -> tuple[list[Any], dict[str, int]]:
    """Exclude workspace/test/generated artifacts outside the requested workspace scope."""
    filtered_documents: list[Any] = []
    filtered_counts: dict[str, int] = {}
    requested_workspace = str(workspace_id).strip() if workspace_id is not None else ""

    for doc in documents:
        metadata = _document_metadata(doc)
        doc_workspace = str(metadata.get("workspace_id") or "").strip()
        origin = str(metadata.get("origin") or "").strip().lower()
        visibility = str(metadata.get("visibility") or "").strip().lower()
        is_workspace_artifact = (
            bool(doc_workspace)
            or _metadata_truthy(metadata.get("is_generated"))
            or _metadata_truthy(metadata.get("is_test_artifact"))
            or origin in {"workspace", "generated", "test", "test_artifact", "workspace_artifact"}
            or visibility == "workspace"
        )

        should_filter = False
        if is_workspace_artifact and not requested_workspace:
            should_filter = True
        elif requested_workspace and doc_workspace and doc_workspace != requested_workspace:
            should_filter = True

        if should_filter:
            source = _document_canonical_source(doc)
            filtered_counts[source] = filtered_counts.get(source, 0) + 1
            continue
        filtered_documents.append(doc)

    return filtered_documents, filtered_counts


def _build_source_status(
    requested_sources: list[str],
    *,
    retriever: Any,
    documents: list[Any],
    filtered_counts: dict[str, int],
) -> dict[str, dict[str, Any]]:
    """Build per-source availability and result-count diagnostics for UI recovery."""
    retriever_map = getattr(retriever, "retrievers", {}) or {}
    available_sources = set(retriever_map.keys()) if isinstance(retriever_map, dict) else set()
    counts: dict[str, int] = {}
    for doc in documents:
        source = _document_canonical_source(doc)
        counts[source] = counts.get(source, 0) + 1

    status: dict[str, dict[str, Any]] = {}
    for source in requested_sources:
        data_source = _CANONICAL_SOURCE_TO_DATASOURCE.get(source)
        count = counts.get(source, 0)
        if data_source not in available_sources:
            entry: dict[str, Any] = {
                "status": "unavailable",
                "count": 0,
                "reason": "no_retriever_configured",
            }
        elif count == 0:
            entry = {
                "status": "empty",
                "count": 0,
                "reason": "no_matching_entries",
            }
        else:
            entry = {
                "status": "searched",
                "count": count,
            }
        filtered_count = filtered_counts.get(source, 0)
        if filtered_count:
            entry["filtered_artifact_count"] = filtered_count
        status[source] = entry
    return status


def normalize_documents_for_generation(docs: list[Any]) -> list[Document]:
    """Normalize retrieval outputs into generation-compatible documents."""
    normalized: list[Document] = []
    for doc in docs or []:
        if isinstance(doc, Document):
            normalized.append(doc)
            continue
        if isinstance(doc, dict):
            metadata = doc.get("metadata") or {}
            source_val = metadata.get("source") if isinstance(metadata, dict) else None
            source = DataSource.MEDIA_DB
            if source_val is not None:
                try:
                    source = DataSource(str(source_val))
                except (ValueError, TypeError):
                    source = DataSource.MEDIA_DB
            normalized.append(
                Document(
                    id=str(doc.get("id")),
                    content=str(doc.get("content") or ""),
                    metadata=metadata if isinstance(metadata, dict) else {},
                    source=source,
                    score=float(doc.get("score") or 0.0),
                )
            )
    return normalized


def _resolve_sqlite_rag_db_path(
    explicit_path: Optional[str],
    *,
    db: Any = None,
) -> Optional[str]:
    """Return a usable SQLite path, suppressing Postgres placeholder values."""
    raw_path = explicit_path
    if raw_path is None and db is not None:
        raw_path = getattr(db, "db_path_str", None)
        if raw_path is None:
            raw_path = getattr(db, "db_path", None)
    if raw_path is None:
        return None

    normalized = str(raw_path).strip()
    if not normalized:
        return None

    backend_type = getattr(db, "backend_type", None)
    if backend_type is None:
        backend = getattr(db, "backend", None)
        backend_type = getattr(backend, "backend_type", None)
    backend_name = str(getattr(backend_type, "name", backend_type or "")).upper()

    if backend_name == "POSTGRESQL" and normalized in {":memory:", "/:memory:"}:
        return None
    return normalized


async def unified_rag_pipeline(
    # ========== REQUIRED PARAMETERS ==========
    query: str,

    # ========== DATA SOURCES ==========
    sources: Optional[list[str]] = None,  # ["media_db", "notes", "characters", "chats"]
    media_db_path: Optional[str] = None,
    notes_db_path: Optional[str] = None,
    character_db_path: Optional[str] = None,
    kanban_db_path: Optional[str] = None,
    sql_target_id: str = "media_db",
    sql_retriever: Any = None,

    # ========== SEARCH CONFIGURATION ==========
    search_mode: Literal["fts", "vector", "hybrid"] = "hybrid",
    fts_level: Literal["media", "chunk"] = "media",
    enable_text_late_chunking: bool = False,
    chunk_method: Optional[str] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    chunk_language: Optional[str] = None,
    hybrid_alpha: float = 0.7,  # 0=FTS only, 1=Vector only
    adaptive_hybrid_weights: bool = False,
    enable_intent_routing: bool = False,
    auto_temporal_filters: bool = False,
    top_k: int = 10,
    min_score: float = 0.0,
    # ========== QUERY EXPANSION ==========
    expand_query: bool = False,
    expansion_strategies: Optional[list[str]] = None,  # ["acronym", "synonym", "domain", "entity"]
    spell_check: bool = False,
    max_query_variations: int = 3,

    # ========== PSEUDO-RELEVANCE FEEDBACK (PRF) ==========
    enable_prf: bool = False,
    prf_terms: int = 10,
    prf_sources: Optional[list[str]] = None,  # ["keywords", "entities", "numbers"]
    prf_alpha: float = 0.3,
    prf_top_n: int = 8,

    # ========== HYDE ==========
    enable_hyde: bool = False,
    hyde_provider: Optional[str] = None,
    hyde_model: Optional[str] = None,

    # ========== GAP ANALYSIS / FOLLOW-UPS ==========
    enable_gap_analysis: bool = False,
    max_followup_searches: int = 2,

    # ========== CACHING ==========
    enable_cache: bool = True,
    cache_threshold: float = 0.85,
    adaptive_cache: bool = True,

    # ========== FILTERING ==========
    keyword_filter: Optional[list[str]] = None,  # Filter by these keywords
    include_media_ids: Optional[list[int]] = None,
    include_note_ids: Optional[list[str]] = None,

    # ========== SECURITY & PRIVACY ==========
    enable_security_filter: bool = False,
    detect_pii: bool = False,
    redact_pii: bool = False,
    sensitivity_level: Literal["public", "internal", "confidential", "restricted"] = "public",
    content_filter: bool = False,

    # ========== DOCUMENT PROCESSING ==========
    enable_table_processing: bool = False,
    table_method: Literal["markdown", "html", "hybrid"] = "markdown",

    # ========== VLM LATE CHUNKING ==========
    enable_vlm_late_chunking: bool = False,
    vlm_backend: Optional[str] = None,
    vlm_detect_tables_only: bool = True,
    vlm_max_pages: Optional[int] = None,
    vlm_late_chunk_top_k_docs: int = 3,

    # ========== CHUNKING & CONTEXT ==========
    enable_enhanced_chunking: bool = False,
    chunk_type_filter: Optional[list[str]] = None,  # ["text", "code", "table", "list"]
    enable_parent_expansion: bool = False,
    parent_context_size: int = 500,
    include_sibling_chunks: bool = False,
    sibling_window: int = 1,
    include_parent_document: bool = False,
    parent_max_tokens: Optional[int] = 1200,

    # ========== ADVANCED RETRIEVAL ==========
    enable_multi_vector_passages: bool = False,
    mv_span_chars: int = 300,
    mv_stride: int = 150,
    mv_max_spans: int = 8,
    mv_flatten_to_spans: bool = False,
    enable_precomputed_spans: bool = False,
    enable_numeric_table_boost: bool = False,

    # ========== RERANKING ==========
    enable_reranking: bool = True,
    reranking_strategy: Literal["flashrank", "cross_encoder", "hybrid", "llama_cpp", "llm_scoring", "two_tier", "none"] = "flashrank",
    rerank_top_k: Optional[int] = None,  # Defaults to top_k if not specified
    reranking_model: Optional[str] = None,  # Optional model id/path for rerankers (GGUF path or HF model id)
    # Two-tier specific: request-level gating overrides (optional)
    rerank_min_relevance_prob: Optional[float] = None,
    rerank_sentinel_margin: Optional[float] = None,

    # ========== LEARNED FUSION & CALIBRATION ==========
    enable_learned_fusion: bool = False,
    calibrator_version: Optional[str] = None,
    abstention_policy: Literal["continue", "ask", "decline"] = "continue",

    # ========== CITATIONS ==========
    enable_citations: bool = False,
    citation_style: Literal["apa", "mla", "chicago", "harvard", "ieee"] = "apa",
    include_page_numbers: bool = False,
    enable_chunk_citations: bool = True,

    # ========== ANSWER GENERATION ==========
    enable_generation: bool = True,
    strict_extractive: bool = False,
    generation_model: Optional[str] = None,
    generation_provider: Optional[str] = None,
    generation_prompt: Optional[str] = None,
    enable_pre_retrieval_clarification: Optional[bool] = None,
    clarification_timeout_sec: Optional[float] = None,
    max_generation_tokens: int = 500,
    # Abstention & multi-turn synthesis
    enable_abstention: bool = False,
    abstention_behavior: Literal["continue", "ask", "decline"] = "continue",
    enable_multi_turn_synthesis: bool = False,
    synthesis_time_budget_sec: Optional[float] = None,
    synthesis_draft_tokens: Optional[int] = None,
    synthesis_refine_tokens: Optional[int] = None,

    # ========== POST-VERIFICATION (ADAPTIVE) ==========
    enable_post_verification: bool = False,
    adaptive_max_retries: int = 1,
    adaptive_unsupported_threshold: float = 0.15,
    adaptive_max_claims: int = 20,
    adaptive_time_budget_sec: Optional[float] = None,
    low_confidence_behavior: Literal["continue", "ask", "decline"] = "continue",
    adaptive_advanced_rewrites: Optional[bool] = None,
    # Optional: perform a bounded full pipeline re-run on low confidence
    adaptive_rerun_on_low_confidence: bool = False,
    adaptive_rerun_include_generation: bool = True,
    adaptive_rerun_bypass_cache: bool = False,
    adaptive_rerun_time_budget_sec: Optional[float] = None,
    adaptive_rerun_doc_budget: Optional[int] = None,
    # ========== QUERY DECOMPOSITION & MULTI-HOP ==========
    enable_query_decomposition: bool = False,
    max_subqueries: int = 4,
    subquery_time_budget_sec: Optional[float] = None,
    subquery_doc_budget: Optional[int] = None,
    subquery_max_concurrency: int = 3,
    # ========== GRAPH-AUGMENTED RETRIEVAL ==========
    enable_graph_retrieval: bool = False,
    graph_version: Optional[str] = None,
    graph_neighbors_k: int = 16,
    graph_alpha: float = 0.4,
    # Internal guard to prevent nested rerun loops
    _adaptive_rerun: bool = False,

    # ========== FEEDBACK ==========
    collect_feedback: bool = False,
    feedback_user_id: Optional[str] = None,
    apply_feedback_boost: bool = False,

    # ========== MONITORING & OBSERVABILITY ==========
    enable_monitoring: bool = False,
    enable_observability: bool = False,
    trace_id: Optional[str] = None,

    # ========== PERFORMANCE ==========
    enable_performance_analysis: bool = False,
    timeout_seconds: Optional[float] = None,

    # ========== STREAMING ==========
    enable_streaming: bool = False,

    # ========== INDEXING / NAMESPACE ==========
    index_namespace: Optional[str] = None,
    retrieval_plan: Optional[RetrievalPlan] = None,
    resolved_request: Optional[ResolvedRAGRequest] = None,

    # ========== QUICK WINS ==========
    highlight_results: bool = False,
    highlight_query_terms: bool = False,
    track_cost: bool = False,
    debug_mode: bool = False,
    include_rerank_debug_documents: Optional[bool] = None,

    # ========== GENERATION GUARDRAILS ==========
    # Pre-generation: instruction-injection filtering and down-weighting
    enable_injection_filter: bool = True,
    injection_filter_strength: float = 0.5,
    # Content policy: lightweight PII/PHI filtering and sanitation
    enable_content_policy_filter: bool = False,
    content_policy_types: Optional[list[str]] = None,  # ["pii", "phi"]
    content_policy_mode: Literal["redact", "drop", "annotate"] = "redact",
    enable_html_sanitizer: bool = False,
    html_allowed_tags: Optional[list[str]] = None,
    html_allowed_attrs: Optional[list[str]] = None,
    ocr_confidence_threshold: Optional[float] = None,
    # Post-generation: hard citations per sentence and numeric fidelity checks
    require_hard_citations: bool = False,
    enable_numeric_fidelity: bool = False,
    numeric_fidelity_behavior: Literal["continue", "ask", "decline", "retry"] = "continue",

    # ========== CLAIMS & FACTUALITY ==========
    enable_claims: bool = False,
    claim_extractor: Literal["aps", "claimify", "auto"] = "auto",
    claim_verifier: Literal["nli", "llm", "hybrid"] = "hybrid",
    claims_top_k: int = 5,
    claims_conf_threshold: float = 0.7,
    claims_max: int = 25,
    nli_model: Optional[str] = None,
    claims_concurrency: int = 8,
    numeric_precision_mode: Literal["standard", "strict", "academic"] = "standard",
    doc_only_verification: bool = False,
    generate_verification_report: bool = False,

    # ========== DOC-RESEARCHER FEATURES ==========
    # Dynamic granularity selection
    enable_dynamic_granularity: bool = False,
    # Progressive evidence accumulation
    enable_evidence_accumulation: bool = False,
    accumulation_max_rounds: int = 3,
    accumulation_time_budget_sec: Optional[float] = None,
    # Multi-hop evidence chains
    enable_evidence_chains: bool = False,

    # ========== SELF-CORRECTING RAG ==========
    # Stage 1: Document Grading - filter documents by LLM-assessed relevance
    enable_document_grading: bool = False,
    grading_threshold: float = 0.5,
    grading_model: Optional[str] = None,
    grading_provider: Optional[str] = None,
    grading_batch_size: int = 5,
    grading_timeout_sec: float = 30.0,
    grading_fallback_to_score: bool = True,
    grading_fallback_min_score: float = 0.3,
    # Stage 2: Query Rewriting Loop - rewrite query when grading shows low relevance
    enable_query_rewriting_loop: bool = False,
    max_rewrite_attempts: int = 2,
    rewrite_relevance_threshold: float = 0.3,
    # Stage 3: Web Search Fallback - fall back to web search when local retrieval fails
    enable_web_fallback: bool = False,
    web_fallback_threshold: float = 0.25,
    web_search_engine: str = "duckduckgo",
    web_fallback_result_count: int = 5,
    web_fallback_merge_strategy: Literal["prepend", "append", "interleave"] = "prepend",
    # Stage 4: Knowledge Strips - partition documents into semantic units
    enable_knowledge_strips: bool = False,
    strip_size_tokens: int = 100,
    strip_min_relevance: float = 0.3,
    max_strips: int = 20,
    # Stage 5: Fast Hallucination Check - lightweight groundedness check
    enable_fast_hallucination_check: bool = False,
    fast_hallucination_timeout_sec: float = 5.0,
    fast_hallucination_provider: Optional[str] = None,
    fast_hallucination_model: Optional[str] = None,
    # Stage 6: Utility Grading - rate response usefulness
    enable_utility_grading: bool = False,
    utility_grading_timeout_sec: float = 5.0,
    utility_grading_provider: Optional[str] = None,
    utility_grading_model: Optional[str] = None,

    # ========== BATCH PROCESSING ==========
    enable_batch: bool = False,
    batch_queries: Optional[list[str]] = None,
    batch_concurrent: int = 5,

    # ========== RESILIENCE ==========
    enable_resilience: bool = False,
    retry_attempts: int = 3,
    circuit_breaker: bool = False,

    # ========== CACHING EXTRAS ==========
    cache_ttl: int = 3600,

    # ========== FILTERING EXTRAS ==========
    enable_date_filter: bool = False,
    date_range: Optional[dict[str, str]] = None,
    filter_media_types: Optional[list[str]] = None,

    # ========== ALT INPUTS ==========
    media_db: Any = None,
    chacha_db: Any = None,

    # ========== ERROR HANDLING ==========
    fallback_on_error: bool = False,

    # ========== USER CONTEXT ==========
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,

    # ========== RETRIEVAL QUALITY METRICS ==========
    ground_truth_doc_ids: Optional[list[str]] = None,
    metrics_k: int = 10,

    # ========== FAITHFULNESS EVALUATION ==========
    enable_faithfulness_eval: bool = False,

    # ========== SEARCH AGENT / RESEARCH ==========
    # Multi-mode search depth (speed/balanced/quality) - sets parameter presets
    search_depth_mode: Optional[Literal["speed", "balanced", "quality"]] = None,
    # LLM-based query classification (routes queries intelligently)
    enable_query_classification: bool = False,
    # Chat history for conversational follow-up reformulation
    chat_history: Optional[list[dict[str, str]]] = None,
    # Standalone query reformulation for chat context
    enable_query_reformulation: bool = True,
    # Iterative agentic research loop
    enable_research_loop: bool = False,
    # Deduplicate repeated equivalent research actions and reuse results
    enable_research_action_dedup: bool = True,
    # Discussion/forum search
    enable_discussion_search: bool = False,
    discussion_platforms: Optional[list[str]] = None,  # ["reddit", "stackoverflow", "hackernews"]
    # URL scraping action availability in research loop
    search_url_scraping: bool = True,
    # Research progress streaming
    enable_research_progress: bool = False,
    research_progress_callback: Optional[Callable] = None,
    # Research loop iteration overrides
    research_max_iterations: Optional[int] = None,
    research_max_iterations_speed: int = 0,
    research_max_iterations_balanced: int = 0,
    research_max_iterations_quality: int = 0,
    # Classifier LLM settings (uses default if empty)
    classifier_provider: Optional[str] = None,
    classifier_model: Optional[str] = None,

    # ========== FOLLOW-UP SUGGESTIONS ==========
    enable_suggestions: bool = False,
    num_suggestions: int = 5,

    # ========== STRUCTURED RESPONSE WRITER ==========
    enable_structured_response: bool = False,

    # ========== MEDIA SEARCH ==========
    enable_image_search: bool = False,
    enable_video_search: bool = False,
    rag_profile: Optional[Literal["fast", "balanced", "accuracy"]] = None,

    # ========== ADDITIONAL PARAMETERS ==========
    **kwargs: Any
) -> UnifiedPipelineResult:
    """
    Unified RAG Pipeline - All features accessible via parameters.

    This is the ONE function for all RAG operations. Every feature is controlled
    by explicit parameters. No configuration files, no presets, just parameters.

    Args:
        query: The search query (required)
        sources: List of databases to search
        ... (see parameters above for all options)

    Returns:
        UnifiedSearchResult with all requested data

    Example:
        result = await unified_rag_pipeline(
            query="What is machine learning?",
            expand_query=True,
            expansion_strategies=["synonym", "acronym"],
            enable_citations=True,
            enable_reranking=True,
            reranking_strategy="hybrid"
        )
    """

    request_metadata: dict[str, Any] = {}
    inbound_metadata = kwargs.get("metadata")
    if isinstance(inbound_metadata, dict):
        request_metadata.update(inbound_metadata)
    if generation_prompt is not None:
        request_metadata["generation_prompt"] = generation_prompt
    request_metadata["max_generation_tokens"] = max_generation_tokens
    workspace_id = kwargs.get("workspace_id")
    if workspace_id is None:
        workspace_id = request_metadata.get("workspace_id")
    if workspace_id is not None:
        workspace_id = str(workspace_id).strip() or None
        if workspace_id is not None:
            request_metadata["workspace_id"] = workspace_id
    prompts_db_path = kwargs.get("prompts_db_path")
    world_books_db_path = kwargs.get("world_books_db_path")
    chat_dictionaries_db_path = kwargs.get("chat_dictionaries_db_path")

    if resolved_request is None:
        resolved_request = resolve_legacy_standard_pipeline_request(
            query=query,
            search_mode=search_mode,
            top_k=top_k,
            sources=sources,
            min_score=min_score,
            index_namespace=index_namespace,
            rag_profile=rag_profile,
            user_id=user_id,
            feedback_user_id=feedback_user_id,
            enable_generation=enable_generation,
            include_sources=bool(kwargs.get("include_sources", True)),
            include_metadata=bool(kwargs.get("include_metadata", True)),
            metadata=request_metadata,
        )

    if retrieval_plan is None:
        retrieval_plan = build_retrieval_plan(resolved_request)

    def _resolve_effective_enable_generation() -> bool:
        return bool((resolved_request.payload or {}).get("enable_generation", enable_generation))

    effective_enable_generation = _resolve_effective_enable_generation()

    retrieval_query = query
    retrieval_sources = sources
    retrieval_search_mode = search_mode
    retrieval_top_k = top_k
    retrieval_min_score = min_score
    retrieval_index_namespace = index_namespace

    def _planned_index_namespace(plan: RetrievalPlan) -> Optional[str]:
        if plan.index_namespace is not None:
            return plan.index_namespace
        normalized_sources = {
            getattr(source, "value", str(source)).strip()
            for source in plan.sources
        }
        if "media_db" in normalized_sources:
            return plan.collection_names.get("media_db")
        return None

    def _refresh_retrieval_snapshot_from_current_knobs(
        *,
        prefer_current_scalars: bool = False,
    ) -> None:
        nonlocal retrieval_query, retrieval_sources, retrieval_search_mode
        nonlocal retrieval_top_k, retrieval_min_score, retrieval_index_namespace, retrieval_plan

        snapshot_query = query
        snapshot_sources = sources
        snapshot_search_mode = search_mode
        snapshot_top_k = top_k
        snapshot_min_score = min_score
        snapshot_index_namespace = index_namespace

        if retrieval_plan is not None:
            planned_query = str(retrieval_plan.query or "").strip()
            if planned_query:
                snapshot_query = planned_query
            planned_sources = [
                str(getattr(source, "value", source)).strip()
                for source in retrieval_plan.sources
                if str(getattr(source, "value", source)).strip()
            ]
            if planned_sources:
                snapshot_sources = planned_sources
            planned_search_mode = str(retrieval_plan.search_mode or "").strip().lower()
            if planned_search_mode in {"fts", "vector", "hybrid"}:
                snapshot_search_mode = cast(
                    Literal["fts", "vector", "hybrid"],
                    planned_search_mode,
                )
            if not prefer_current_scalars:
                with contextlib.suppress(TypeError, ValueError):
                    snapshot_top_k = max(1, int(retrieval_plan.top_k))
                with contextlib.suppress(TypeError, ValueError):
                    snapshot_min_score = float(retrieval_plan.min_score)
            snapshot_index_namespace = _planned_index_namespace(retrieval_plan)

        retrieval_query = snapshot_query
        retrieval_sources = snapshot_sources
        retrieval_search_mode = snapshot_search_mode
        with contextlib.suppress(TypeError, ValueError):
            retrieval_top_k = max(1, int(snapshot_top_k))
        with contextlib.suppress(TypeError, ValueError):
            retrieval_min_score = float(snapshot_min_score)
        retrieval_index_namespace = snapshot_index_namespace

        if retrieval_plan is not None:
            retrieval_plan = replace(
                retrieval_plan,
                query=str(retrieval_query),
                sources=tuple(str(source) for source in (retrieval_sources or [])),
                search_mode=str(retrieval_search_mode),
                top_k=int(retrieval_top_k),
                min_score=float(retrieval_min_score),
                index_namespace=retrieval_index_namespace,
            )

    if retrieval_plan is not None:
        planned_query = str(retrieval_plan.query or "").strip()
        if planned_query:
            retrieval_query = planned_query
        planned_sources = [
            str(getattr(source, "value", source)).strip()
            for source in retrieval_plan.sources
            if str(getattr(source, "value", source)).strip()
        ]
        if planned_sources:
            retrieval_sources = planned_sources
        planned_search_mode = str(retrieval_plan.search_mode or "").strip().lower()
        if planned_search_mode in {"fts", "vector", "hybrid"}:
            retrieval_search_mode = cast(
                Literal["fts", "vector", "hybrid"],
                planned_search_mode,
            )
        with contextlib.suppress(TypeError, ValueError):
            retrieval_top_k = max(1, int(retrieval_plan.top_k))
        with contextlib.suppress(TypeError, ValueError):
            retrieval_min_score = float(retrieval_plan.min_score)
        retrieval_index_namespace = _planned_index_namespace(retrieval_plan)

    query_expansion_corpus = (
        retrieval_plan.index_namespace
        if retrieval_plan is not None and retrieval_plan.index_namespace is not None
        else index_namespace
    )

    def _retrieval_plan_for(query_text: str) -> Optional[RetrievalPlan]:
        if retrieval_plan is None:
            return None
        if query_text == retrieval_plan.query:
            return retrieval_plan
        try:
            return replace(retrieval_plan, query=query_text)
        except TypeError:
            return retrieval_plan

    def _sync_effective_query(new_query: str) -> None:
        nonlocal query, retrieval_query, retrieval_plan
        query = str(new_query)
        retrieval_query = query
        result.query = query
        result_payload = getattr(result, "payload", None)
        if isinstance(result_payload, dict):
            result_payload["query"] = query
        resolved_request.query = query
        resolved_request.payload["query"] = query
        retrieval_plan = replace(retrieval_plan, query=query)

    # Basic input validation (short-circuit before heavier setup)
    if not isinstance(query, str) or not query.strip():
        msg = "Invalid query"
        metadata = {"original_query": query}
        try:
            inbound_meta = kwargs.get("metadata")
            if isinstance(inbound_meta, dict):
                metadata.update(inbound_meta)
        except TypeError:
            pass
        timings = {"total": 0.0}
        # Consistent contract: return UnifiedRAGResponse for all outcomes
        try:
            from tldw_Server_API.app.api.v1.schemas.rag_schemas_unified import UnifiedRAGResponse
            return UnifiedRAGResponse(
                documents=[],
                query=(query if isinstance(query, str) else ""),
                expanded_queries=[],
                metadata=metadata,
                timings=timings,
                citations=[],
                academic_citations=[],
                chunk_citations=[],
                generated_answer=msg,
                cache_hit=False,
                errors=[msg],
                security_report=None,
                total_time=0.0,
                claims=None,
                factuality=None,
            )
        except (ImportError, TypeError, ValueError):
            # Fallback to dataclass if schema import fails (non-API contexts)
            return UnifiedSearchResult(
                documents=[],
                query=query if isinstance(query, str) else "",
                expanded_queries=[],
                metadata=metadata,
                timings=timings,
                citations=[],
                feedback_id=None,
                generated_answer=msg,
                cache_hit=False,
                errors=[msg],
                security_report=None,
                total_time=0.0,
            )

    # Normalize common alias/compat args
    expand_query = expand_query or kwargs.get("enable_expansion", False)
    media_db_path = _resolve_sqlite_rag_db_path(media_db_path, db=media_db)

    # ========== SEARCH DEPTH MODE PRESETS ==========
    # Apply parameter presets based on search_depth_mode. Individual parameter
    # overrides (if explicitly non-default) take precedence over mode presets.
    retrieval_scalar_overrides_applied = False
    if search_depth_mode is not None:
        _mode_presets = {
            "speed": {
                "top_k": 3,
                "enable_reranking": False,
                "reranking_strategy": "none",
                "expand_query": False,
                "enable_hyde": False,
                "enable_web_fallback": False,
                "enable_multi_vector_passages": False,
            },
            "balanced": {
                "top_k": 10,
                "enable_reranking": True,
                "reranking_strategy": "flashrank",
                "expand_query": True,
                "enable_hyde": False,
                "enable_web_fallback": False,
                "enable_multi_vector_passages": False,
            },
            "quality": {
                "top_k": 25,
                "enable_reranking": True,
                "reranking_strategy": "two_tier",
                "expand_query": True,
                "enable_hyde": True,
                "enable_web_fallback": True,
                "enable_multi_vector_passages": True,
            },
        }
        presets = _mode_presets.get(search_depth_mode, {})
        # Only apply preset if the caller didn't explicitly set the param
        # (we check against the function defaults)
        if presets:
            if top_k == 10:
                preset_top_k = presets.get("top_k", top_k)
                if preset_top_k != top_k:
                    retrieval_scalar_overrides_applied = True
                top_k = preset_top_k
            if enable_reranking is True and "enable_reranking" in presets:
                enable_reranking = presets["enable_reranking"]
            if reranking_strategy == "flashrank" and "reranking_strategy" in presets:
                reranking_strategy = presets["reranking_strategy"]
            if not expand_query and presets.get("expand_query"):
                expand_query = True
            if not enable_hyde and presets.get("enable_hyde"):
                enable_hyde = True
            if not enable_web_fallback and presets.get("enable_web_fallback"):
                enable_web_fallback = True
            if not enable_multi_vector_passages and presets.get("enable_multi_vector_passages"):
                enable_multi_vector_passages = True

    _refresh_retrieval_snapshot_from_current_knobs(
        prefer_current_scalars=retrieval_scalar_overrides_applied,
    )

    # Initialize result and timing
    start_time = time.time()
    result = UnifiedSearchResult(
        documents=[],
        query=query,
        metadata={"original_query": query}
    )
    standard_evidence_coordinated = False
    claims_payload = None
    factuality_payload = None
    # Merge inbound metadata if provided (API pattern)
    try:
        inbound_meta = kwargs.get("metadata")
        if isinstance(inbound_meta, dict):
            result.metadata.update(inbound_meta)
    except TypeError:
        pass

    def _ensure_profile_resolution_metadata() -> dict[str, Any]:
        profile_resolution = result.metadata.get("profile_resolution")
        if not isinstance(profile_resolution, dict):
            profile_resolution = {}
        profile_resolution.setdefault("requested_profile", rag_profile)
        profile_resolution.setdefault("applied_profile", rag_profile)
        profile_resolution.setdefault("effective_overrides_count", 0)
        if not isinstance(profile_resolution.get("degraded_features"), list):
            profile_resolution["degraded_features"] = []
        result.metadata["profile_resolution"] = profile_resolution
        return profile_resolution

    def _record_profile_degradation(
        *,
        component: str,
        from_value: str,
        to_value: str,
        reason: str,
    ) -> None:
        profile_resolution = _ensure_profile_resolution_metadata()
        degraded_features = profile_resolution.get("degraded_features")
        if not isinstance(degraded_features, list):
            degraded_features = []
            profile_resolution["degraded_features"] = degraded_features
        degraded_features.append(
            {
                "component": component,
                "from": from_value,
                "to": to_value,
                "reason": reason,
            }
        )

    _ensure_profile_resolution_metadata()

    cache_instance = None
    cache_max_size = 1000
    try:
        from tldw_Server_API.app.core.config import RAG_SERVICE_CONFIG
        cfg = cast(dict[str, Any], RAG_SERVICE_CONFIG) if isinstance(RAG_SERVICE_CONFIG, dict) else {}
        cache_max_size = int((cfg.get("cache") or {}).get("max_cache_size", cache_max_size))
    except (ImportError, TypeError, ValueError):
        pass
    cache_namespace = retrieval_index_namespace or (user_id or None)
    if cache_namespace is None:
        try:
            parts = [media_db_path, notes_db_path, character_db_path, kanban_db_path, prompts_db_path]
            if any(parts):
                joined = "|".join([str(p or "") for p in parts])
                cache_namespace = f"db:{hashlib.sha256(joined.encode('utf-8')).hexdigest()[:12]}"
        except (TypeError, ValueError):
            cache_namespace = None
    workspace_cache_component = workspace_id or "global"
    if cache_namespace is not None:
        cache_namespace = f"{cache_namespace}|workspace:{workspace_cache_component}"
    elif workspace_id is not None:
        cache_namespace = f"workspace:{workspace_cache_component}"

    def _get_cache_instance():
        nonlocal cache_instance
        if cache_instance is not None:
            return cache_instance
        cache_cls = None
        if adaptive_cache and AdaptiveCache:
            cache_cls = AdaptiveCache
        elif SemanticCache:
            cache_cls = SemanticCache

        if cache_cls:
            try:
                if get_shared_cache:
                    cache_instance = get_shared_cache(
                        cache_cls=cache_cls,
                        similarity_threshold=cache_threshold,
                        ttl=cache_ttl,
                        max_size=cache_max_size,
                        namespace=cache_namespace,
                    )
                else:
                    cache_instance = cache_cls(
                        similarity_threshold=cache_threshold,
                        ttl=cache_ttl,
                        namespace=cache_namespace,
                    )
            except TypeError:
                cache_instance = cache_cls(similarity_threshold=cache_threshold)
        # Register with the RAGCache facade so health endpoints see real stats
        if cache_instance is not None:
            try:
                from .advanced_cache import register_semantic_cache
                register_semantic_cache(cache_instance)
            except (ImportError, TypeError):
                pass
        return cache_instance

    # --- Internal helpers (defined early so downstream phases can use them) ---
    async def _with_timeout(coro, timeout: Optional[float]):
        if timeout and timeout > 0:
            return await asyncio.wait_for(coro, timeout=timeout)
        return await coro

    async def _resilient_call(component: str, func, *args, **kwargs):
        """Apply circuit breaker, retries, and timeout around async operations when enabled."""
        breaker = None
        if enable_resilience and circuit_breaker and get_coordinator and CircuitBreakerConfig:
            try:
                coord = get_coordinator()
                if component not in coord.circuit_breakers:
                    coord.register_circuit_breaker(component, CircuitBreakerConfig())
                breaker = coord.circuit_breakers[component]
            except (AttributeError, KeyError, TypeError):
                breaker = None

        async def _attempt():
            if breaker is not None:
                return await breaker.call(func, *args, **kwargs)
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            return func(*args, **kwargs)

        if enable_resilience and (retry_attempts or 0) > 1 and RetryPolicy and RetryConfig:
            policy = RetryPolicy(RetryConfig(max_attempts=int(retry_attempts or 1)))
            call_coro = policy.execute(_attempt)
        else:
            call_coro = _attempt()

        return await _with_timeout(call_coro, timeout_seconds)

    # Initialize monitoring if requested
    metrics = None
    if enable_monitoring:
        metrics = QueryMetrics(
            query_id=str(uuid.uuid4()),
            query=query,
            timestamp=start_time,
            total_duration=0.0,
        )

    def _apply_generation_gate(reason: str, *, coverage: Optional[float] = None, unsupported_ratio: Optional[float] = None, threshold: Optional[float] = None) -> None:
        """Record a gating event in metadata for downstream observability."""
        gate = result.metadata.setdefault("generation_gate", {})
        gate.update({
            "reason": reason,
            "at": time.time(),
        })
        if coverage is not None:
            gate["coverage"] = coverage
        if unsupported_ratio is not None:
            gate["unsupported_ratio"] = unsupported_ratio
        if threshold is not None:
            gate["threshold"] = threshold

    class _EarlyReturn(Exception):
        """Internal control-flow sentinel for intentional short-circuit paths."""

    resolved_data_sources: list[DataSource] = []

    try:
        try:
            retrieval_sources = _normalize_pipeline_sources(retrieval_sources)
            resolved_data_sources = _sources_to_data_sources(retrieval_sources)
            explicit_include_sources = _explicit_include_data_sources(
                include_media_ids=include_media_ids,
                include_note_ids=include_note_ids,
            )
            if explicit_include_sources:
                original_retrieval_sources = list(retrieval_sources)
                requested_sources = set(resolved_data_sources)
                resolved_data_sources = [
                    source
                    for source in explicit_include_sources
                    if source in requested_sources
                ]
                retrieval_sources = [
                    _DATASOURCE_TO_CANONICAL_SOURCE[source]
                    for source in resolved_data_sources
                ]
                resolved_request.payload["sources"] = list(retrieval_sources)
                if retrieval_plan is not None:
                    scoped_retrieval_plan = replace(
                        retrieval_plan,
                        sources=tuple(retrieval_sources),
                        index_namespace=None,
                    )
                    retrieval_plan = replace(
                        scoped_retrieval_plan,
                        index_namespace=_planned_index_namespace(scoped_retrieval_plan),
                    )
                    retrieval_index_namespace = _planned_index_namespace(retrieval_plan)
                result.metadata["explicit_source_selection"] = {
                    "enabled": True,
                    "requested_sources": original_retrieval_sources,
                    "resolved_sources": list(retrieval_sources),
                    "include_media_ids_count": len(include_media_ids or []),
                    "include_note_ids_count": len(include_note_ids or []),
                }
            result.metadata["sources_requested"] = list(retrieval_sources)
        except ValueError as exc:
            result.errors.append(f"invalid_source: {exc}")
            result.metadata["source_validation_error"] = str(exc)
            raise _EarlyReturn()

        if (
            sql_retriever is None
            and SQLRetriever is not None
            and any(src == DataSource.SQL for src in resolved_data_sources)
        ):
            sql_db_path = _resolve_sqlite_rag_db_path(media_db_path, db=media_db)
            if sql_db_path:
                try:
                    sql_retriever = SQLRetriever(
                        db_path=str(sql_db_path),
                        target_id=str(sql_target_id or "media_db"),
                    )
                except (
                    AttributeError,
                    ConnectionError,
                    OSError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                ) as exc:
                    result.errors.append(f"SQL retriever init failed: {exc}")

        def _build_multi_retriever(db_paths: dict[str, str]):
            base_kwargs: dict[str, Any] = {"user_id": user_id or "0", "media_db": media_db}
            if chacha_db is not None:
                base_kwargs["chacha_db"] = chacha_db
            prompts_db = kwargs.get("prompts_db")
            if prompts_db is not None:
                base_kwargs["prompts_db"] = prompts_db
            if sql_retriever is not None:
                base_kwargs["sql_retriever"] = sql_retriever

            optional_keys = ("chacha_db", "prompts_db", "sql_retriever")
            variants = [dict(base_kwargs)]
            for key in optional_keys:
                if key in base_kwargs:
                    variants.append({k: v for k, v in base_kwargs.items() if k != key})
            variants.append({
                k: v
                for k, v in base_kwargs.items()
                if k not in set(optional_keys)
            })
            seen: set[tuple[str, ...]] = set()
            for variant in variants:
                key = tuple(sorted(variant.keys()))
                if key in seen:
                    continue
                seen.add(key)
                try:
                    return MultiDatabaseRetriever(db_paths, **variant)
                except TypeError:
                    continue
            try:
                return MultiDatabaseRetriever(db_paths, user_id=user_id or "0")
            except TypeError:
                return MultiDatabaseRetriever(db_paths)

        def _build_pipeline_db_paths() -> dict[str, str]:
            """Return the full source-to-database map for every retrieval pass."""
            db_paths: dict[str, str] = {}
            if media_db_path:
                db_paths["media_db"] = str(media_db_path)
            if notes_db_path:
                db_paths["notes_db"] = str(notes_db_path)
            if prompts_db_path:
                db_paths["prompts_db"] = str(prompts_db_path)
            if character_db_path:
                db_paths["character_cards_db"] = str(character_db_path)
                db_paths["world_books_db"] = str(world_books_db_path or character_db_path)
                db_paths["chat_dictionaries_db"] = str(chat_dictionaries_db_path or character_db_path)
            else:
                if world_books_db_path:
                    db_paths["world_books_db"] = str(world_books_db_path)
                if chat_dictionaries_db_path:
                    db_paths["chat_dictionaries_db"] = str(chat_dictionaries_db_path)
            if kanban_db_path:
                db_paths["kanban_db"] = str(kanban_db_path)
            return db_paths

        source_status_retriever: Any = None
        cumulative_filtered_artifact_counts: dict[str, int] = {}

        def _apply_workspace_filtering_to_result() -> None:
            """Filter the current result documents and refresh source diagnostics."""
            nonlocal result
            filtered_documents, filtered_artifact_counts = _filter_workspace_artifacts(
                list(result.documents or []),
                workspace_id=workspace_id,
            )
            for source, count in filtered_artifact_counts.items():
                cumulative_filtered_artifact_counts[source] = (
                    cumulative_filtered_artifact_counts.get(source, 0) + count
                )
            result.documents = filtered_documents
            if source_status_retriever is not None:
                result.metadata["source_status"] = _build_source_status(
                    _normalize_pipeline_sources(list(retrieval_sources)),
                    retriever=source_status_retriever,
                    documents=filtered_documents,
                    filtered_counts=dict(cumulative_filtered_artifact_counts),
                )
            if workspace_id is not None:
                result.metadata["workspace_id"] = workspace_id
            result.metadata["documents_retrieved"] = len(filtered_documents)

        # ========== LEARNED FUSION / CALIBRATION HELPERS ==========
        def _decorate_calibration_metadata() -> None:
            """
            Attach learned-fusion specific fields (if present) to reranking calibration.

            This keeps Two-Tier calibration behavior intact while making the new
            enable_learned_fusion / calibrator_version flags observable.
            """
            if not isinstance(result.metadata, dict):
                return
            cal = result.metadata.get("reranking_calibration")
            if not isinstance(cal, dict):
                return
            # Fused score is the calibrated probability of the top document
            if "fused_score" not in cal and "top_doc_prob" in cal:
                with contextlib.suppress(TypeError, ValueError):
                    cal["fused_score"] = float(cal.get("top_doc_prob") or 0.0)
            # Mark whether learned fusion was explicitly requested
            if enable_learned_fusion:
                cal["enabled"] = True
            # Version tag is purely informational for now
            if calibrator_version:
                cal.setdefault("version", calibrator_version)
            result.metadata["reranking_calibration"] = cal

        # ========== PRE-RETRIEVAL CLARIFICATION ==========
        effective_pre_clarify = (
            enable_pre_retrieval_clarification
            if enable_pre_retrieval_clarification is not None
            else effective_enable_generation
        )
        if effective_pre_clarify and assess_query_for_clarification is not None:
            clarification_start = time.time()
            decision = await assess_query_for_clarification(
                query=query,
                chat_history=chat_history,
                timeout_sec=float(clarification_timeout_sec or 1.5),
            )
            result.timings["pre_retrieval_clarification"] = time.time() - clarification_start
            if decision.required:
                result.generated_answer = decision.question or "Could you clarify your request?"
                result.documents = []
                result.timings["retrieval"] = 0.0
                result.metadata.setdefault("clarification", {})
                result.metadata["clarification"].update({
                    "required": True,
                    "stage": "pre_retrieval",
                    "reason": decision.reason,
                    "confidence": decision.confidence,
                    "detector": decision.detector,
                })
                result.metadata.setdefault("retrieval_bypassed", {})
                result.metadata["retrieval_bypassed"]["reason"] = "pre_retrieval_clarification"
                try:
                    from tldw_Server_API.app.core.Metrics.metrics_manager import increment_counter
                    increment_counter(
                        "rag_clarification_triggered_total",
                        1,
                        labels={"stage": "pre_retrieval", "detector": decision.detector},
                    )
                except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
                    pass
                raise _EarlyReturn()

        # ========== SPELL CHECK ==========
        if spell_check:
            if spell_check_query:
                spell_start = time.time()
                corrected = await spell_check_query(query)
                if corrected != query:
                    result.metadata["original_query"] = query
                    result.metadata["corrected_query"] = corrected
                    _sync_effective_query(corrected)
                result.timings["spell_check"] = time.time() - spell_start
            else:
                result.errors.append("Spell check module not available")
                logger.warning("Spell check requested but module not available")
        # ========== QUERY CLASSIFICATION & REFORMULATION ==========
        _classification = None
        _skip_retrieval_stack = False
        _skip_retrieval_reason: Optional[str] = None
        _skip_local_retrieval = False

        async def _prefetch_routed_external_docs(
            *,
            query_text: str,
            use_web: bool,
            use_academic: bool,
            use_discussions: bool,
        ) -> list["Document"]:
            """Best-effort external prefetch when classification disables local DB search."""
            prefetched_docs: list[Document] = []
            max_results = max(1, min(int(retrieval_top_k or 10), 25))
            seen_ids: set[str] = set()

            def _append_doc(doc: Document) -> None:
                if doc.id in seen_ids:
                    return
                seen_ids.add(doc.id)
                prefetched_docs.append(doc)

            # Discussion-prefetch first for opinion/experience-focused questions.
            if use_discussions and enable_discussion_search:
                try:
                    from tldw_Server_API.app.core.Web_Scraping.WebSearch_APIs import search_discussions

                    discussion_results = await search_discussions(
                        query=query_text,
                        platforms=discussion_platforms,
                        max_results=max_results,
                    )
                    for item in discussion_results or []:
                        doc_id = str(item.get("url") or item.get("id") or f"discussion:{len(prefetched_docs)}")
                        _append_doc(Document(
                            id=doc_id,
                            content=str(item.get("content", "")),
                            metadata={
                                "title": item.get("title", ""),
                                "url": item.get("url", ""),
                                "source_type": "discussion",
                                "platform": item.get("platform", ""),
                            },
                            source=DataSource.WEB_CONTENT,
                            score=float(item.get("score", 0.5) or 0.5),
                        ))
                except Exception as exc:  # noqa: BLE001 - best effort
                    result.errors.append(f"Discussion prefetch failed: {exc}")

            # Academic/web prefetch via existing web search processor.
            if use_academic or use_web:
                try:
                    from tldw_Server_API.app.core.Web_Scraping.WebSearch_APIs import (
                        perform_websearch,
                        process_web_search_results,
                    )

                    web_query = query_text
                    if use_academic:
                        web_query = (
                            f"{query_text} "
                            "site:arxiv.org OR site:scholar.google.com OR site:semanticscholar.org"
                        )

                    raw_results = await asyncio.to_thread(
                        perform_websearch,
                        search_engine="duckduckgo",
                        search_query=web_query,
                        content_country="US",
                        search_lang="en",
                        output_lang="en",
                        result_count=max_results,
                    )
                    processed = process_web_search_results(raw_results, "duckduckgo")
                    for item in (processed.get("results", []) or [])[:max_results]:
                        url = str(item.get("url", ""))
                        if not url:
                            continue
                        source_type = "academic" if use_academic else "web"
                        _append_doc(Document(
                            id=url,
                            content=str(item.get("content", item.get("snippet", ""))),
                            metadata={
                                "title": item.get("title", ""),
                                "url": url,
                                "source_type": source_type,
                            },
                            source=DataSource.WEB_CONTENT,
                            score=float(item.get("score", 0.5) or 0.5),
                        ))
                except Exception as exc:  # noqa: BLE001 - best effort
                    result.errors.append(f"Web/academic prefetch failed: {exc}")

            return prefetched_docs[:max_results]

        if enable_query_classification and classify_and_reformulate is not None:
            try:
                _cls_start = time.time()
                _cls_provider = classifier_provider or generation_provider or "openai"
                _cls_model = classifier_model or generation_model
                _classification = await classify_and_reformulate(
                    query=query,
                    chat_history=chat_history,
                    llm_provider=_cls_provider,
                    llm_model=_cls_model,
                )
                result.timings["query_classification"] = time.time() - _cls_start
                result.metadata["query_classification"] = {
                    "skip_search": _classification.skip_search,
                    "search_local_db": _classification.search_local_db,
                    "search_web": _classification.search_web,
                    "search_academic": _classification.search_academic,
                    "search_discussions": _classification.search_discussions,
                    "detected_intent": _classification.detected_intent,
                    "confidence": _classification.confidence,
                    "reasoning": _classification.reasoning,
                }

                # Apply classification decisions
                if _classification.skip_search:
                    # Skip retrieval entirely, return early with generation only
                    logger.info(f"Query classification: skip_search=True for '{query[:80]}'")
                    result.metadata["classification_skip_search"] = True
                    _skip_retrieval_stack = True
                    _skip_retrieval_reason = "classification_skip_search"
                    # If generation is enabled, we still want to generate a response
                    # but without search context — this is handled downstream

                # Apply reformulated query
                if _classification.standalone_query and _classification.standalone_query != query:
                    result.metadata["reformulated_query"] = _classification.standalone_query
                    _sync_effective_query(_classification.standalone_query)
                    logger.debug(f"Query reformulated to: {query[:100]}")

                # Route web fallback based on classification
                if _classification.search_web and not enable_web_fallback:
                    enable_web_fallback = True
                if _classification.search_discussions and not enable_discussion_search:
                    enable_discussion_search = True
                if not _classification.search_local_db:
                    _skip_local_retrieval = True
                    result.metadata["classification_local_retrieval"] = "disabled"

                    # If local retrieval is disabled but external routes are requested,
                    # prefetch external docs directly in non-research-loop mode.
                    if (
                        not enable_research_loop
                        and (_classification.search_web or _classification.search_academic or _classification.search_discussions)
                    ):
                        external_docs = await _prefetch_routed_external_docs(
                            query_text=query,
                            use_web=bool(_classification.search_web),
                            use_academic=bool(_classification.search_academic),
                            use_discussions=bool(_classification.search_discussions),
                        )
                        if external_docs:
                            result.documents = external_docs
                            _skip_retrieval_stack = True
                            _skip_retrieval_reason = "classification_external_prefetch"
                            result.metadata["classification_external_prefetch"] = {
                                "enabled": True,
                                "document_count": len(external_docs),
                                "search_web": bool(_classification.search_web),
                                "search_academic": bool(_classification.search_academic),
                                "search_discussions": bool(_classification.search_discussions),
                            }
                        else:
                            _skip_retrieval_stack = True
                            _skip_retrieval_reason = "classification_local_disabled"
                            result.metadata["classification_external_prefetch"] = {
                                "enabled": True,
                                "document_count": 0,
                                "search_web": bool(_classification.search_web),
                                "search_academic": bool(_classification.search_academic),
                                "search_discussions": bool(_classification.search_discussions),
                            }

            except Exception as _cls_exc:
                logger.warning(f"Query classification failed: {_cls_exc!r}")
                result.errors.append(f"Query classification failed: {_cls_exc}")
        elif chat_history and enable_query_reformulation and classify_and_reformulate is not None:
            # Even without full classification, do reformulation if chat history present
            try:
                from .query_classifier import reformulate_query as _reformulate_q
                _ref_start = time.time()
                _ref_provider = classifier_provider or generation_provider or "openai"
                _ref_model = classifier_model or generation_model
                reformulated = await _reformulate_q(
                    query=query,
                    chat_history=chat_history,
                    llm_provider=_ref_provider,
                    llm_model=_ref_model,
                )
                if reformulated and reformulated != query:
                    result.metadata["reformulated_query"] = reformulated
                    _sync_effective_query(reformulated)
                result.timings["query_reformulation"] = time.time() - _ref_start
            except Exception as _ref_exc:
                logger.warning(f"Query reformulation failed: {_ref_exc!r}")

        # ========== AGENTIC RESEARCH LOOP ==========
        # If research loop is enabled and we're in balanced/quality mode,
        # delegate to the research agent instead of the standard pipeline
        if (
            enable_research_loop
            and research_loop is not None
            and _classification is not None
            and not _classification.skip_search
        ):
            try:
                _research_start = time.time()
                _research_mode = search_depth_mode or "balanced"
                _db_ctx = {
                    "media_db_path": media_db_path,
                    "notes_db_path": notes_db_path,
                    "character_db_path": character_db_path,
                    "kanban_db_path": kanban_db_path,
                }
                if media_db is not None:
                    _db_ctx["media_db"] = media_db
                if chacha_db is not None:
                    _db_ctx["chacha_db"] = chacha_db

                def _positive_int_or_none(value: Any) -> Optional[int]:
                    try:
                        ivalue = int(value)
                    except (TypeError, ValueError):
                        return None
                    return ivalue if ivalue > 0 else None

                _mode_specific_overrides = {
                    "speed": _positive_int_or_none(research_max_iterations_speed),
                    "balanced": _positive_int_or_none(research_max_iterations_balanced),
                    "quality": _positive_int_or_none(research_max_iterations_quality),
                }
                _effective_max_iterations = _positive_int_or_none(research_max_iterations)
                if _effective_max_iterations is None:
                    _effective_max_iterations = _mode_specific_overrides.get(_research_mode)
                if _effective_max_iterations is None:
                    with contextlib.suppress(TypeError, ValueError):
                        import os as _os
                        _env_key = f"SEARCH_MAX_ITERATIONS_{_research_mode.upper()}"
                        _effective_max_iterations = _positive_int_or_none(_os.getenv(_env_key))

                _progress_cb = research_progress_callback if enable_research_progress else None
                _registry = None
                if create_default_registry is not None:
                    with contextlib.suppress(TypeError, ValueError):
                        _registry = create_default_registry(
                            discussion_platforms=discussion_platforms,
                            enable_url_scraping=bool(search_url_scraping),
                            enable_image_search=bool(enable_image_search),
                            enable_video_search=bool(enable_video_search),
                            on_progress=_progress_cb,
                        )

                _research_output = await research_loop(
                    query=query,
                    classification=_classification,
                    mode=_research_mode,
                    llm_provider=classifier_provider or generation_provider or "openai",
                    llm_model=classifier_model or generation_model,
                    max_iterations=_effective_max_iterations,
                    on_progress=_progress_cb,
                    registry=_registry,
                    db_context=_db_ctx,
                    discussion_platforms=discussion_platforms,
                    enable_url_scraping=bool(search_url_scraping),
                    enable_action_dedup=bool(enable_research_action_dedup),
                    enable_image_search=bool(enable_image_search),
                    enable_video_search=bool(enable_video_search),
                )

                # Convert research results to Document objects
                from .types import DataSource as _DS
                from .types import Document as _Doc
                _research_docs = []
                for r_item in _research_output.all_results:
                    _research_docs.append(_Doc(
                        id=str(r_item.get("id", r_item.get("url", ""))),
                        content=str(r_item.get("content", "")),
                        metadata={
                            "title": r_item.get("title", ""),
                            "url": r_item.get("url", ""),
                            "source_type": r_item.get("source", "research"),
                            "platform": r_item.get("platform", ""),
                        },
                        source=_DS.WEB_CONTENT if r_item.get("source") in ("web", "discussion", "academic", "scraped_url") else _DS.MEDIA_DB,
                        score=float(r_item.get("score", 0.5)),
                    ))

                result.documents = _research_docs
                result.timings["research_loop"] = time.time() - _research_start
                result.metadata["research"] = {
                    "total_iterations": _research_output.total_iterations,
                    "total_results": _research_output.total_results,
                    "completed": _research_output.completed,
                    "final_reasoning": _research_output.final_reasoning,
                    "max_iterations_requested": _effective_max_iterations,
                    "url_scraping_enabled": bool(search_url_scraping),
                    "action_dedup": _research_output.metadata.get("action_dedup", {}),
                    "discussion_platforms": discussion_platforms or ["reddit", "stackoverflow", "hackernews"],
                    "url_dedup": _research_output.metadata.get("url_dedup", {}),
                    "steps": [
                        {
                            "iteration": s.iteration,
                            "action": s.action_name,
                            "reasoning": s.reasoning,
                            "result_count": s.output.result_count if s.output else 0,
                            "duration_sec": s.duration_sec,
                        }
                        for s in _research_output.steps
                    ],
                }

                # Collect media search results from research steps
                if enable_image_search or enable_video_search:
                    _images = []
                    _videos = []
                    for _step in _research_output.steps:
                        if _step.output and _step.output.success and _step.output.results:
                            _step_type = (_step.output.metadata or {}).get("type", "")
                            if _step_type == "images":
                                _images.extend(_step.output.results)
                            elif _step_type == "videos":
                                _videos.extend(_step.output.results)
                    if _images:
                        result.metadata["images"] = _images
                    if _videos:
                        result.metadata["videos"] = _videos

                logger.info(
                    f"Research loop completed: {_research_output.total_iterations} iterations, "
                    f"{_research_output.total_results} results in "
                    f"{_research_output.total_duration_sec:.2f}s"
                )

                # Skip cache/retrieval after a successful research loop so the
                # assembled research evidence remains the source of truth.
                _skip_retrieval_stack = True
                _skip_retrieval_reason = "research_loop"
                result.metadata["research_retrieval_bypassed"] = True

            except Exception as _res_exc:
                logger.warning(f"Research loop failed, falling back to standard pipeline: {_res_exc!r}")
                result.errors.append(f"Research loop failed: {_res_exc}")
                # Fall through to standard pipeline

        # ========== PRODUCTION DEFAULTS (env-based) ==========
        # If running in production, enable stricter guardrails by default
        try:
            import os as _os
            _prod_env = _shared_is_truthy(_os.getenv("tldw_production", "false"))
            _strict_env = _shared_is_truthy(_os.getenv("RAG_GUARDRAILS_STRICT", "false"))
            if _prod_env or _strict_env:
                if not enable_numeric_fidelity:
                    enable_numeric_fidelity = True
                if not require_hard_citations:
                    require_hard_citations = True
                # Behavior default can be tuned via env when it's left as "continue"
                if (numeric_fidelity_behavior == "continue"):
                    _beh = _os.getenv("RAG_NUMERIC_FIDELITY_BEHAVIOR", "ask").strip().lower()
                    if _beh in {"continue", "ask", "decline", "retry"}:
                        numeric_fidelity_behavior = _beh  # type: ignore
        except (TypeError, ValueError):
            pass

        # Apply config-driven defaults for confidence/citation gates when not explicitly set
        try:
            if _rag_low_conf:
                cfg_lcb = _rag_low_conf()
                if (low_confidence_behavior or "continue") == "continue" and cfg_lcb != "continue":
                    low_confidence_behavior = cfg_lcb
            if _rag_req_hc and not bool(require_hard_citations) and bool(_rag_req_hc(default=False)):
                require_hard_citations = True
        except (TypeError, ValueError):
            pass

        # Apply config-driven default for strict extractive generation
        _rag_strict: Any = None
        with contextlib.suppress(ImportError):
            from tldw_Server_API.app.core.config import rag_strict_extractive as _rag_strict
        try:
            if _rag_strict is not None and not bool(strict_extractive) and bool(_rag_strict(default=False)):
                strict_extractive = True
        except (TypeError, ValueError):
            pass

        # Precompute query analysis once for reuse across phases
        analysis = None
        analysis_intent_val = None
        analysis_complexity_val = None
        analysis_domain = None
        if QueryAnalyzer and retrieval_query:
            try:
                qa = QueryAnalyzer()
                analysis = qa.analyze_query(retrieval_query)
                analysis_intent = getattr(analysis, "intent", None)
                analysis_complexity = getattr(analysis, "complexity", None)
                analysis_intent_val = getattr(analysis_intent, "value", str(analysis_intent)) if analysis_intent is not None else None
                analysis_complexity_val = getattr(analysis_complexity, "value", str(analysis_complexity)) if analysis_complexity is not None else None
                analysis_domain = getattr(analysis, "domain", None)
            except (AttributeError, TypeError, ValueError, RuntimeError):
                analysis = None
                analysis_intent_val = None
                analysis_complexity_val = None
                analysis_domain = None

        # ========== QUERY EXPANSION ==========
        expanded_queries = [retrieval_query]
        if expand_query:
            expansion_start = time.time()
            try:
                # Try rewrite cache first
                cached_rewrites: list[str] = []
                intent_label = analysis_intent_val
                if RewriteCache and user_id:
                    try:
                        rc = RewriteCache(user_id=user_id)
                        cached = rc.get(
                            retrieval_query,
                            intent=intent_label,
                            corpus=query_expansion_corpus,
                        )
                        if cached:
                            cached_rewrites = [c for c in cached if isinstance(c, str) and c.strip()]
                    except ValueError as exc:
                        logger.debug(f"Rewrite cache disabled for user_id={user_id}: {exc}")
                    except (AttributeError, OSError, RuntimeError, TypeError, sqlite3.Error):
                        pass
                strategies = (expansion_strategies or ["acronym", "synonym"]).copy()
                expanded_variants: list[str] = []
                if multi_strategy_expansion:
                    if query_expansion_corpus:
                        expanded = await multi_strategy_expansion(
                            retrieval_query,
                            strategies=strategies,
                            corpus=query_expansion_corpus,
                        )
                    else:
                        # Avoid passing None to preserve expected call signature in tests
                        expanded = await multi_strategy_expansion(
                            retrieval_query,
                            strategies=strategies,
                        )
                    if isinstance(expanded, list):
                        expanded_variants.extend([q for q in expanded if isinstance(q, str)])
                    elif isinstance(expanded, str) and expanded.strip():
                        expanded_variants.append(expanded)
                rewriter_variants: list[str] = []
                if QueryRewriter and query:
                    try:
                        rewriter = QueryRewriter()
                        rw = rewriter.rewrite_query(
                            query,
                            strategies=["decompose", "generalize", "specify", "clarify"],
                        )
                        rewriter_variants = [
                            r.rewritten_query for r in rw
                            if getattr(r, "rewritten_query", None)
                        ]
                    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                        rewriter_variants = []

                # Merge and dedupe all candidates while preserving order
                candidate_queries: list[str] = [query]
                candidate_queries.extend(cached_rewrites)
                candidate_queries.extend(expanded_variants)
                candidate_queries.extend(rewriter_variants)
                expanded_queries = []
                seen = set()
                for q in candidate_queries:
                    if not isinstance(q, str):
                        continue
                    s = q.strip()
                    if not s or s in seen:
                        continue
                    seen.add(s)
                    expanded_queries.append(s)

                # Bound the number of variations to avoid excessive retrieval fan-out
                try:
                    max_variations = max(0, int(max_query_variations))
                except (TypeError, ValueError):
                    max_variations = 3
                limit = 1 + max_variations
                if len(expanded_queries) > limit:
                    expanded_queries = expanded_queries[:limit]
                # Persist effective rewrites for future reuse (best-effort)
                try:
                    if RewriteCache and user_id and len(expanded_queries) > 1:
                        rew = [q for q in expanded_queries if q != query][:5]
                        if rew:
                            rc = RewriteCache(user_id=user_id)
                        rc.put(
                            retrieval_query,
                            rewrites=rew,
                            intent=intent_label,
                            corpus=retrieval_index_namespace,
                        )
                except ValueError as exc:
                    logger.debug(f"Rewrite cache write disabled for user_id={user_id}: {exc}")
                except (AttributeError, TypeError):
                    pass
                result.expanded_queries = [q for q in expanded_queries if q != query]
                if result.expanded_queries:
                    result.metadata.setdefault("query_expansion", {})
                    result.metadata["query_expansion"]["variations"] = len(result.expanded_queries)
                result.timings["query_expansion"] = time.time() - expansion_start
                if metrics:
                    metrics.expansion_time = result.timings["query_expansion"]
            except (
                AttributeError,
                ConnectionError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
                asyncio.TimeoutError,
            ) as e:
                result.errors.append(f"Query expansion failed: {str(e)}")
                logger.warning(f"Query expansion error: {e}")

        # ========== INTENT ROUTING (optional) ==========
        if enable_intent_routing and QueryRouter:
            try:
                router = QueryRouter()
                routing = router.route_query(query)
                # Map routing decisions to current pipeline knobs conservatively
                # Keep search_mode hybrid by default; adjust hybrid_alpha and top_k
                strat = str(routing.get("retrieval_strategy", "")).lower()
                if strat == "precise":
                    # Favor lexical; shift hybrid_alpha toward FTS
                    with contextlib.suppress(TypeError, ValueError):
                        hybrid_alpha = min(max(0.0, float(hybrid_alpha)), 1.0)
                    hybrid_alpha = min(hybrid_alpha, 0.5)
                elif strat == "broad":
                    # Favor semantic
                    with contextlib.suppress(TypeError, ValueError):
                        hybrid_alpha = min(max(0.0, float(hybrid_alpha)), 1.0)
                    hybrid_alpha = max(hybrid_alpha, 0.7)
                # Respect suggested top_k when present
                try:
                    tk = int(routing.get("top_k", top_k))
                    if 1 <= tk <= 100:
                        if tk != top_k:
                            retrieval_scalar_overrides_applied = True
                        top_k = tk
                except (TypeError, ValueError):
                    pass
                result.metadata["intent_routing"] = {
                    "strategy": strat,
                    "hybrid_alpha": hybrid_alpha,
                    "top_k": top_k,
                }
            except (AttributeError, RuntimeError, TypeError, ValueError) as e:
                result.errors.append(f"Intent routing failed: {e}")

        # ========== DYNAMIC GRANULARITY ROUTING ==========
        granularity_decision = None
        if enable_dynamic_granularity and GranularityRouter and route_query_granularity:
            try:
                granularity_start = time.time()
                granularity_decision = route_query_granularity(query)

                # Apply granularity-specific retrieval parameters
                if granularity_decision:
                    params = granularity_decision.retrieval_params
                    # Override top_k if not explicitly set by user
                    if "top_k" in params:
                        if params["top_k"] != top_k:
                            retrieval_scalar_overrides_applied = True
                        top_k = params["top_k"]
                    # Override fts_level
                    if "fts_level" in params:
                        fts_level = params["fts_level"]
                    # Override parent expansion settings
                    if params.get("enable_parent_expansion"):
                        enable_parent_expansion = True
                        if "parent_context_size" in params:
                            parent_context_size = params["parent_context_size"]
                    if params.get("include_parent_document"):
                        include_parent_document = True
                        if "parent_max_tokens" in params:
                            parent_max_tokens = params["parent_max_tokens"]
                    # Override multi-vector passages
                    if params.get("enable_multi_vector_passages"):
                        enable_multi_vector_passages = True
                        if "mv_span_chars" in params:
                            mv_span_chars = params["mv_span_chars"]
                        if "mv_stride" in params:
                            mv_stride = params["mv_stride"]
                        if "mv_max_spans" in params:
                            mv_max_spans = params["mv_max_spans"]

                    result.metadata["granularity_routing"] = {
                        "query_type": granularity_decision.query_type.value,
                        "granularity": granularity_decision.granularity.value,
                        "confidence": granularity_decision.confidence,
                        "reasoning": granularity_decision.reasoning,
                    }

                result.timings["granularity_routing"] = time.time() - granularity_start
            except (AttributeError, RuntimeError, TypeError, ValueError) as e:
                result.errors.append(f"Granularity routing failed: {e}")
                logger.warning(f"Granularity routing error: {e}")

        _refresh_retrieval_snapshot_from_current_knobs(
            prefer_current_scalars=retrieval_scalar_overrides_applied,
        )

        # ========== CACHE CHECK ==========
        cached_documents = None
        if enable_cache and not _skip_retrieval_stack and not _skip_local_retrieval:
            cache_start = time.time()
            cache = _get_cache_instance()

            if cache:
                # First try direct get on the main query (support sync or async)
                try:
                    get_fn = cache.get
                    if asyncio.iscoroutinefunction(get_fn):
                        direct = await get_fn(query)
                    else:
                        direct = get_fn(query)
                except (AttributeError, OSError, RuntimeError, TypeError):
                    direct = None
                if direct:
                    cached_documents = direct
                    result.cache_hit = True
                else:
                    # Check cache for all query variations
                    for q in expanded_queries:
                        try:
                            find_fn = getattr(cache, 'find_similar', None)
                            if find_fn is None:
                                break
                            if asyncio.iscoroutinefunction(find_fn):
                                cached_result = await find_fn(q)
                            else:
                                cached_result = find_fn(q)
                        except (AttributeError, OSError, RuntimeError, TypeError):
                            cached_result = None
                        if cached_result:
                            # find_similar returns (key, query, sim) or (query, sim)
                            if len(cached_result) == 3:
                                _, cached_query, similarity = cached_result
                            else:
                                cached_query, similarity = cached_result
                            try:
                                if asyncio.iscoroutinefunction(get_fn):
                                    cached_documents = await get_fn(cached_query)
                                else:
                                    cached_documents = get_fn(cached_query)
                            except (AttributeError, OSError, RuntimeError, TypeError):
                                cached_documents = None
                            if cached_documents:
                                result.cache_hit = True
                                result.metadata["cache_similarity"] = similarity
                                result.metadata["cached_query"] = cached_query
                                break

                if result.cache_hit:
                    empty_cached_docs = False
                    if isinstance(cached_documents, dict):
                        docs = cached_documents.get("documents")
                        if isinstance(docs, list) and not docs:
                            empty_cached_docs = True
                    elif isinstance(cached_documents, list) and not cached_documents:
                        empty_cached_docs = True
                    if empty_cached_docs:
                        # Treat empty cached results as a miss to avoid stale false negatives.
                        result.cache_hit = False
                        cached_documents = None

                if result.cache_hit:
                    if isinstance(cached_documents, dict):
                        ans = cached_documents.get("answer")
                        if ans is not None:
                            result.generated_answer = ans
                        docs = cached_documents.get("documents")
                        if isinstance(docs, list):
                            result.documents = docs
                        if cached_documents.get("cached") is True:
                            result.metadata["cached_flag"] = True
                    elif isinstance(cached_documents, list):
                        # Backward compatibility: older cache entries stored document lists directly
                        result.documents = cached_documents
                    result.metadata.setdefault("cached_flag", True)

            result.timings["cache_check"] = time.time() - cache_start
            if metrics:
                metrics.cache_lookup_time = result.timings["cache_check"]
        elif _skip_retrieval_stack or _skip_local_retrieval:
            result.metadata["cache_bypassed"] = {
                "reason": (
                    _skip_retrieval_reason
                    if _skip_retrieval_stack
                    else "classification_local_disabled"
                )
            }

        # ========== INTENT-BASED WEIGHTING (optional) ==========
        if adaptive_hybrid_weights and search_mode == "hybrid":
            try:
                local_analysis = analysis
                if local_analysis is None and QueryAnalyzer and query:
                    try:
                        qa = QueryAnalyzer()
                        local_analysis = qa.analyze_query(query)
                    except (AttributeError, TypeError, ValueError, RuntimeError):
                        local_analysis = None
                # Conceptual queries favor semantic; specific factual favor keyword
                if local_analysis and getattr(local_analysis, "intent", None) is not None:
                    if local_analysis.intent in {
                        getattr(QueryIntent, "EXPLORATORY", None),
                        getattr(QueryIntent, "DEFINITIONAL", None),
                        getattr(QueryIntent, "ANALYTICAL", None),
                        getattr(QueryIntent, "PROCEDURAL", None),
                    }:
                        hybrid_alpha = 0.7
                    elif local_analysis.intent in {
                        getattr(QueryIntent, "FACTUAL", None),
                        getattr(QueryIntent, "COMPARATIVE", None),
                        getattr(QueryIntent, "TEMPORAL", None),
                    }:
                        hybrid_alpha = 0.4
                    result.metadata["query_intent"] = getattr(local_analysis.intent, "value", str(local_analysis.intent))
                result.metadata["adaptive_hybrid_alpha"] = hybrid_alpha
            except (AttributeError, TypeError, ValueError):
                pass

        # ========== HyDE PREP (optional) ==========
        hyde_vector = None
        if enable_hyde and generate_hypothetical_answer and hyde_embed_text:
            try:
                hyde_start = time.time()
                # Read defaults if present
                try:
                    from tldw_Server_API.app.core.config import load_and_log_configs
                    cfg = load_and_log_configs()
                    if not isinstance(cfg, dict):
                        cfg = {}
                    raw_provider = cfg.get("RAG_HYDE_PROVIDER")
                    raw_model = cfg.get("RAG_HYDE_MODEL")
                    hyde_provider = hyde_provider or (str(raw_provider).strip() if raw_provider else None)
                    hyde_model = hyde_model or (str(raw_model).strip() if raw_model else None)
                except (ImportError, AttributeError, OSError, TypeError, ValueError):
                    pass
                hypo = generate_hypothetical_answer(query, hyde_provider, hyde_model)
                vec = await hyde_embed_text(hypo)
                if vec:
                    hyde_vector = vec
                    result.metadata["hyde_applied"] = True
                result.timings["hyde_prep"] = time.time() - hyde_start
            except (
                AttributeError,
                ConnectionError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
                asyncio.TimeoutError,
            ) as e:
                result.errors.append(f"HyDE prep failed: {e}")

        # ========== AUTO TEMPORAL FILTERS (optional) ==========
        if auto_temporal_filters:
            try:
                qlower = query.lower()
                start_dt = None
                end_dt = None

                now = datetime.utcnow()
                # Relative expressions
                if "yesterday" in qlower:
                    start_dt = now - timedelta(days=1)
                    end_dt = now
                elif "last week" in qlower or "past week" in qlower:
                    start_dt = now - timedelta(days=7)
                    end_dt = now
                elif "last month" in qlower:
                    # Compute previous calendar month
                    y = now.year
                    m = now.month - 1 if now.month > 1 else 12
                    y = y if now.month > 1 else y - 1
                    start_dt = datetime(y, m, 1)
                    _, last_day = calendar.monthrange(y, m)
                    end_dt = datetime(y, m, last_day, 23, 59, 59)
                elif "past month" in qlower:
                    start_dt = now - timedelta(days=30)
                    end_dt = now

                # Quarters like Q1 2024
                m_quarter = re.search(r"\bq([1-4])\s*(20\d{2}|19\d{2})\b", qlower)
                if m_quarter:
                    qn = int(m_quarter.group(1))
                    y = int(m_quarter.group(2))
                    qm = {1: 1, 2: 4, 3: 7, 4: 10}[qn]
                    start_dt = datetime(y, qm, 1)
                    end_month = qm + 2
                    _, last_day = calendar.monthrange(y, end_month)
                    end_dt = datetime(y, end_month, last_day, 23, 59, 59)

                # Month name + year, e.g., January 2023
                month_names = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
                m_month_year = re.search(r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+(20\d{2}|19\d{2})\b", qlower)
                if m_month_year:
                    mon = month_names.get(m_month_year.group(1))
                    y = int(m_month_year.group(2))
                    if mon:
                        start_dt = datetime(y, mon, 1)
                        _, last_day = calendar.monthrange(y, mon)
                        end_dt = datetime(y, mon, last_day, 23, 59, 59)

                # Year-only reference (prefer exact year range)
                m_year = re.search(r"\b(20\d{2}|19\d{2})\b", qlower)
                if m_year and start_dt is None and end_dt is None:
                    y = int(m_year.group(1))
                    start_dt = datetime(y, 1, 1)
                    end_dt = datetime(y, 12, 31, 23, 59, 59)

                if start_dt is None and end_dt is None:
                    # Conservative default: last 7 days window when auto filtering is enabled
                    start_dt = now - timedelta(days=7)
                    end_dt = now

                if start_dt and end_dt:
                    enable_date_filter = True
                    date_range = {"start": start_dt.isoformat(), "end": end_dt.isoformat()}
                    result.metadata["temporal_filter"] = {
                        "start": date_range["start"],
                        "end": date_range["end"],
                        "source": "auto",
                    }
            except (AttributeError, TypeError, ValueError, RuntimeError):
                pass

        # ========== DOCUMENT RETRIEVAL ==========
        if not result.cache_hit and not _skip_retrieval_stack and not _skip_local_retrieval:
            retrieval_start = time.time()
            try:
                # --- OTEL: retrieval span ---
                _otel_cm = None
                _otel_span = None
                if enable_observability and get_telemetry_manager:
                    try:
                        _tm = get_telemetry_manager()
                        _tr = _tm.get_tracer("tldw.rag")
                        _attrs = {
                            "rag.phase": "retrieval",
                            "rag.search_mode": str(retrieval_search_mode),
                            "rag.top_k": int(retrieval_top_k or 0),
                            "rag.index_namespace": str(retrieval_index_namespace or "")
                        }
                        _otel_cm = _tr.start_as_current_span("rag.retrieval")
                        _otel_span = _otel_cm.__enter__()
                        for _k, _v in _attrs.items():
                            with contextlib.suppress(AttributeError, RuntimeError, TypeError, ValueError):
                                _otel_span.set_attribute(_k, _v)
                    except (AttributeError, RuntimeError, TypeError, ValueError):
                        _otel_cm = None
                        _otel_span = None
                if MultiDatabaseRetriever and RetrievalConfig:

                    # Set up database paths
                    db_paths = _build_pipeline_db_paths()

                    # Initialize retriever (tests may patch this constructor).
                    retriever = _build_multi_retriever(db_paths)
                    source_status_retriever = retriever

                    # Configure retrieval
                    config = RetrievalConfig(
                        max_results=retrieval_top_k,
                        min_score=retrieval_min_score,
                        use_fts=(retrieval_search_mode in ["fts", "hybrid"]),
                        use_vector=(retrieval_search_mode in ["vector", "hybrid"]),
                        include_metadata=True,
                        fts_level=fts_level,
                        enable_text_late_chunking=enable_text_late_chunking,
                        chunk_method=chunk_method,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        chunk_language=chunk_language,
                    )
                    # Optional date filter
                    if enable_date_filter and date_range and isinstance(date_range, dict):
                        try:
                            start = datetime.fromisoformat(date_range.get("start", "")) if date_range.get("start") else None
                            end = datetime.fromisoformat(date_range.get("end", "")) if date_range.get("end") else None
                            if start and end:
                                config.date_filter = (start, end)
                        except (TypeError, ValueError):
                            pass
                    # Fallback: use metadata-written temporal filter (auto)
                    if getattr(config, 'date_filter', None) is None:
                        tf = result.metadata.get("temporal_filter") if isinstance(result.metadata, dict) else None
                        if isinstance(tf, dict):
                            try:
                                start_val = tf.get("start")
                                end_val = tf.get("end")
                                if start_val and end_val:
                                    config.date_filter = (datetime.fromisoformat(start_val), datetime.fromisoformat(end_val))
                            except (TypeError, ValueError):
                                pass

                    def _execution_collection_names() -> dict[str, str]:
                        user_key = str(user_id or feedback_user_id or "0").strip() or "0"
                        source_names = {
                            str(getattr(source, "value", source)).strip().lower()
                            for source in retrieval_sources
                            if str(getattr(source, "value", source)).strip()
                        }
                        collection_names: dict[str, str] = {
                            "media_db": f"user_{user_key}_media_embeddings",
                            "notes": f"user_{user_key}_notes_embeddings",
                        }
                        if {"character_cards", "characters"} & source_names:
                            collection_names["character_cards"] = f"user_{user_key}_character_embeddings"
                        if {"kanban", "kanban_db"} & source_names:
                            collection_names["kanban"] = f"user_{user_key}_kanban_embeddings"
                        return collection_names

                    def _effective_retrieval_plan_for_query(
                        query_text: str,
                        *,
                        top_k_override: Optional[int] = None,
                    ) -> RetrievalPlan:
                        plan = _retrieval_plan_for(query_text)
                        if plan is None:
                            plan = RetrievalPlan(
                                query=query_text,
                                sources=tuple(str(source) for source in retrieval_sources),
                                search_mode=str(retrieval_search_mode),
                                top_k=int(retrieval_top_k),
                                min_score=float(retrieval_min_score),
                                index_namespace=retrieval_index_namespace,
                                collection_names=_execution_collection_names(),
                            )
                        if top_k_override is not None:
                            try:
                                return replace(plan, top_k=max(1, int(top_k_override)))
                            except (TypeError, ValueError):
                                return plan
                        return plan

                    async def _execute_retrieval_variant(
                        component: str,
                        query_text: str,
                        *,
                        retrieval_cfg: Optional[Any] = None,
                        top_k_override: Optional[int] = None,
                    ) -> list[Document]:
                        evidence = await _resilient_call(
                            component,
                            execute_retrieval_phase,
                            resolved_request=resolved_request,
                            retrieval_plan=_effective_retrieval_plan_for_query(
                                query_text,
                                top_k_override=top_k_override,
                            ),
                            retriever=retriever,
                            retrieval_config=retrieval_cfg or config,
                            allowed_media_ids=include_media_ids,
                            allowed_note_ids=include_note_ids,
                        )
                        evidence_docs = getattr(evidence, "documents", None)
                        return list(evidence_docs or [])

                    effective_retrieval_plan = _effective_retrieval_plan_for_query(retrieval_query)

                    retrieved_evidence = await _resilient_call(
                        "retrieval",
                        execute_retrieval_phase,
                        resolved_request=resolved_request,
                        retrieval_plan=effective_retrieval_plan,
                        retriever=retriever,
                        retrieval_config=config,
                        allowed_media_ids=include_media_ids,
                        allowed_note_ids=include_note_ids,
                    )
                    documents = list(retrieved_evidence.documents)

                    # Fallback: if no documents were retrieved via MultiDatabaseRetriever,
                    # perform a direct Media DB FTS-only search. This guards against
                    # configuration or adapter issues that can cause hybrid retrieval
                    # to silently return an empty set even when media is present.
                    if (
                        (not documents)
                        and _scope_includes_data_source(resolved_data_sources, DataSource.MEDIA_DB)
                        and (media_db_path or media_db is not None)
                        and retrieval_search_mode in ("fts", "hybrid")
                    ):
                        try:
                            from .database_retrievers import MediaDBRetriever as _MDBR
                            from .database_retrievers import RetrievalConfig as _RCfg
                            fb_cfg = _RCfg(
                                max_results=retrieval_top_k,
                                min_score=retrieval_min_score,
                                use_fts=True,
                                use_vector=False,
                                include_metadata=True,
                                fts_level=fts_level,
                            )
                            fb_retriever = _MDBR(
                                db_path=media_db_path,
                                config=fb_cfg,
                                user_id=str(user_id or "0"),
                                media_db=media_db,
                            )
                            fallback_docs = await fb_retriever.retrieve(
                                query=retrieval_query,
                                media_type=None,
                                allowed_media_ids=include_media_ids,
                            )
                            if fallback_docs:
                                documents = fallback_docs
                                if isinstance(result.metadata, dict):
                                    result.metadata.setdefault("fallbacks", {})
                                    result.metadata["fallbacks"]["media_db_fts"] = True
                        except (
                            AttributeError,
                            ConnectionError,
                            OSError,
                            RuntimeError,
                            TypeError,
                            ValueError,
                            asyncio.TimeoutError,
                            sqlite3.Error,
                        ) as _fb_err:
                            result.errors.append(f"Media DB fallback retrieval failed: {str(_fb_err)}")

                    # Optionally run HyDE-enhanced media retrieval and merge
                    if (
                        enable_hyde
                        and hyde_vector
                        and retrieval_search_mode == "hybrid"
                        and _scope_includes_data_source(resolved_data_sources, DataSource.MEDIA_DB)
                    ):
                        try:
                            media_retr = retriever.retrievers.get(DataSource.MEDIA_DB)
                            if media_retr and hasattr(media_retr, "retrieve_hybrid"):
                                hyde_docs = await media_retr.retrieve_hybrid(
                                    query=retrieval_query,
                                    alpha=hybrid_alpha,
                                    index_namespace=retrieval_index_namespace,
                                    query_vector=hyde_vector,
                                )
                                by_id: dict[str, Document] = {d.id: d for d in documents}
                                for d in hyde_docs:
                                    cur = by_id.get(d.id)
                                    if cur is None or float(getattr(d, "score", 0.0)) > float(getattr(cur, "score", 0.0)):
                                        by_id[d.id] = d
                                documents = sorted(by_id.values(), key=lambda x: getattr(x, "score", 0.0), reverse=True)
                                result.metadata["hyde_merged_count"] = len(hyde_docs)
                        except (
                            AttributeError,
                            ConnectionError,
                            OSError,
                            RuntimeError,
                            TypeError,
                            ValueError,
                            asyncio.TimeoutError,
                        ) as e:
                            result.errors.append(f"HyDE retrieval merge failed: {e}")

                    # Optional: expand retrieval across query variants
                    if expand_query and expanded_queries and len(expanded_queries) > 1:
                        exp_start = time.time()
                        try:
                            extra_queries = [q for q in expanded_queries if q != retrieval_query]
                            exp_docs: list[Document] = []
                            for eq in extra_queries:
                                try:
                                    try:
                                        expansion_top_k = max(1, int(retrieval_top_k or 1))
                                        exp_cfg = replace(config, max_results=expansion_top_k)
                                    except (TypeError, ValueError):
                                        expansion_top_k = None
                                        exp_cfg = config
                                    eq_docs = await _execute_retrieval_variant(
                                        "retrieval_expansion",
                                        eq,
                                        retrieval_cfg=exp_cfg,
                                        top_k_override=expansion_top_k,
                                    )
                                    if eq_docs:
                                        exp_docs.extend(eq_docs)
                                except (
                                    AttributeError,
                                    ConnectionError,
                                    OSError,
                                    RuntimeError,
                                    TypeError,
                                    ValueError,
                                    asyncio.TimeoutError,
                                ) as _eq_err:
                                    result.errors.append(f"Query expansion retrieval failed: {eq}: {str(_eq_err)}")
                                    continue

                            if exp_docs:
                                by_id = {d.id: d for d in documents}
                                added = 0
                                for d in exp_docs:
                                    cur = by_id.get(d.id)
                                    if cur is None or float(getattr(d, "score", 0.0)) > float(getattr(cur, "score", 0.0)):
                                        if cur is None:
                                            added += 1
                                        by_id[d.id] = d
                                documents = sorted(
                                    by_id.values(),
                                    key=lambda x: getattr(x, "score", 0.0),
                                    reverse=True,
                                )
                                result.metadata.setdefault("query_expansion", {})
                                result.metadata["query_expansion"]["retrieval_queries"] = len(extra_queries)
                                result.metadata["query_expansion"]["retrieval_added"] = int(added)
                            result.timings["query_expansion_retrieval"] = time.time() - exp_start
                        except (
                            AttributeError,
                            ConnectionError,
                            OSError,
                            RuntimeError,
                            TypeError,
                            ValueError,
                            asyncio.TimeoutError,
                        ) as _exp_err:
                            result.errors.append(f"Query expansion retrieval failed: {str(_exp_err)}")

                    result.documents = documents
                    _apply_workspace_filtering_to_result()
                    # Optional PRF second-pass retrieval to fill remaining slots
                    if (
                        enable_prf
                        and apply_prf
                        and PRFConfig
                        and result.documents
                        and len(result.documents) < retrieval_top_k
                    ):
                        try:
                            prf_cfg = PRFConfig(
                                max_terms=int(prf_terms or 0),
                                sources=prf_sources or ["keywords", "entities", "numbers"],
                                alpha=float(prf_alpha or 0.0),
                                top_n=int(prf_top_n or 0),
                            )
                            prf_query, prf_meta = await apply_prf(
                                retrieval_query,
                                result.documents,
                                prf_cfg,
                            )
                            result.metadata.setdefault("prf", {})
                            result.metadata["prf"].update(prf_meta)

                            # Only perform a second pass when PRF is enabled and query changed
                            if prf_meta.get("enabled") and prf_query and prf_query != retrieval_query:
                                remaining_slots = max(0, retrieval_top_k - len(result.documents))
                                if remaining_slots > 0:
                                    prf_docs = await _execute_retrieval_variant(
                                        "retrieval_prf",
                                        prf_query,
                                        top_k_override=remaining_slots,
                                    )
                                    prf_docs = prf_docs or []
                                    existing_ids = {d.id for d in result.documents}
                                    added = 0
                                    for d in prf_docs:
                                        if d.id not in existing_ids:
                                            result.documents.append(d)
                                            existing_ids.add(d.id)
                                            added += 1
                                            if len(result.documents) >= retrieval_top_k:
                                                break
                                    result.metadata["prf"]["second_pass_performed"] = True
                                    result.metadata["prf"]["second_pass_added"] = int(added)
                                else:
                                    result.metadata["prf"]["second_pass_performed"] = False
                                    result.metadata["prf"]["second_pass_added"] = 0
                        except (
                            AttributeError,
                            ConnectionError,
                            OSError,
                            RuntimeError,
                            TypeError,
                            ValueError,
                            asyncio.TimeoutError,
                        ) as _prf_err:
                            result.errors.append(f"PRF second-pass retrieval failed: {str(_prf_err)}")

                    # Optional: guided query decomposition to broaden recall for multi-part queries
                    if enable_query_decomposition:
                        decomp_start = time.time()
                        try:
                            # Prefer agentic planner-style decomposition when available and
                            # the query appears complex/compound; otherwise fall back to a
                            # lightweight heuristic split.
                            q_norm = (query or "").strip()
                            subqueries: list[str] = []
                            used_agentic = False

                            # Use QueryAnalyzer (if available) to detect complex queries that
                            # benefit from decomposition (comparative/causal/temporal/analytical).
                            intent_val = analysis_intent_val
                            comp_val = analysis_complexity_val
                            # Try delegating to agentic-style decomposition when the query
                            # is complex and of a multi-part intent.
                            multi_intents = {"comparative", "causal", "analytical", "temporal"}
                            if intent_val in multi_intents and comp_val == "complex":
                                try:
                                    from .agentic_chunker import AgenticConfig as _ACfg
                                    from .agentic_chunker import _decompose_query as _agentic_decompose
                                    subgoal_max = max_subqueries if max_subqueries is not None else 3
                                    acfg = _ACfg(enable_query_decomposition=True, subgoal_max=int(subgoal_max))
                                    subqueries = _agentic_decompose(q_norm, acfg) or []
                                    used_agentic = True
                                except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
                                    used_agentic = False

                            if not used_agentic:
                                # Lightweight heuristic decomposition: split on common coordinators/punctuation.
                                if q_norm:
                                    parts = re.split(r"\b(?:and then|then|and|,|;|\?)\b", q_norm, flags=re.IGNORECASE)
                                    subqueries = [p.strip() for p in parts if p and len(p.strip()) >= 3]
                                if not subqueries:
                                    subqueries = [q_norm] if q_norm else []

                            # Ensure primary query is the first entry
                            if q_norm:
                                seen_norm = set()
                                ordered = [q_norm]
                                seen_norm.add(q_norm.lower())
                                for sq in subqueries:
                                    sq_norm = (sq or "").strip()
                                    if not sq_norm:
                                        continue
                                    key = sq_norm.lower()
                                    if key in seen_norm:
                                        continue
                                    seen_norm.add(key)
                                    ordered.append(sq_norm)
                                subqueries = ordered

                            # Apply max_subqueries cap if provided (includes primary query implicitly)
                            try:
                                max_sub = int(max_subqueries or 0)
                            except (TypeError, ValueError):
                                max_sub = 0
                            if max_sub and len(subqueries) > max_sub:
                                subqueries = subqueries[: max_sub]

                            meta_decomp: dict[str, Any] = {
                                "enabled": True,
                                "subqueries": [],
                            }
                            if intent_val is not None:
                                meta_decomp["intent"] = intent_val
                            if comp_val is not None:
                                meta_decomp["complexity"] = comp_val
                            time_budget = float(subquery_time_budget_sec) if subquery_time_budget_sec else None
                            try:
                                doc_budget = int(subquery_doc_budget) if subquery_doc_budget is not None else None
                            except (TypeError, ValueError):
                                doc_budget = None

                            base_ids = {d.id for d in result.documents}
                            total_added = 0

                            # Only run additional retrievals for secondary subqueries
                            if len(subqueries) > 1:
                                try:
                                    subquery_max_results = max(1, int(retrieval_top_k or 1))
                                    if doc_budget is not None:
                                        subquery_max_results = max(1, min(subquery_max_results, int(doc_budget)))
                                except (TypeError, ValueError):
                                    subquery_max_results = max(1, int(retrieval_top_k or 1))

                                async def _fetch_subquery(sq: str) -> list[Document]:
                                    try:
                                        sq_cfg = replace(config, max_results=subquery_max_results)
                                    except TypeError:
                                        sq_cfg = config
                                    res = await _execute_retrieval_variant(
                                        "retrieval_decomposition",
                                        sq,
                                        retrieval_cfg=sq_cfg,
                                        top_k_override=subquery_max_results,
                                    )
                                    return res if isinstance(res, list) else []

                                subquery_results: dict[str, Any] = {}
                                subqueries_to_run = list(subqueries[1:])
                                try:
                                    max_workers = max(1, int(subquery_max_concurrency or 1))
                                except (TypeError, ValueError):
                                    max_workers = 3
                                max_workers = min(max_workers, len(subqueries_to_run)) or 1
                                sem = asyncio.Semaphore(max_workers)

                                async def _run_subquery(sq: str) -> None:
                                    async with sem:
                                        subquery_results[sq] = await _fetch_subquery(sq)

                                tasks = [asyncio.create_task(_run_subquery(sq)) for sq in subqueries_to_run]
                                if tasks:
                                    if time_budget is not None:
                                        remaining = max(0.0, time_budget - (time.time() - decomp_start))
                                        done, pending = await asyncio.wait(tasks, timeout=remaining)
                                        for p in pending:
                                            p.cancel()
                                    else:
                                        done, _ = await asyncio.wait(tasks)
                                    for task in done:
                                        try:
                                            task.result()
                                        except (
                                            AttributeError,
                                            ConnectionError,
                                            OSError,
                                            RuntimeError,
                                            TypeError,
                                            ValueError,
                                            asyncio.TimeoutError,
                                        ) as _sq_err:
                                            result.errors.append(
                                                f"Decomposition subquery retrieval failed: {_sq_err}"
                                            )

                                for sq in subqueries_to_run:
                                    if time_budget is not None and (time.time() - decomp_start) >= time_budget:
                                        break
                                    if doc_budget is not None and total_added >= doc_budget:
                                        break
                                    sq_docs = subquery_results.get(sq) or []
                                    added_ids: list[str] = []
                                    for d in sq_docs:
                                        if d.id not in base_ids:
                                            result.documents.append(d)
                                            base_ids.add(d.id)
                                            added_ids.append(d.id)
                                            total_added += 1
                                            if doc_budget is not None and total_added >= doc_budget:
                                                break
                                    meta_decomp["subqueries"].append({
                                        "query": sq,
                                        "added_doc_ids": added_ids,
                                    })
                                    if doc_budget is not None and total_added >= doc_budget:
                                        break

                                # Re-sort docs by score and cap to top_k
                                try:
                                    result.documents = sorted(
                                        result.documents,
                                        key=lambda d: getattr(d, "score", 0.0),
                                        reverse=True,
                                    )[: retrieval_top_k]
                                except (TypeError, ValueError):
                                    # Fallback: leave documents in current order
                                    pass

                            meta_decomp["total_added"] = int(total_added)
                            meta_decomp["elapsed_sec"] = round(time.time() - decomp_start, 6)
                            meta_decomp["time_budget_sec"] = float(time_budget) if time_budget is not None else None
                            meta_decomp["doc_budget"] = int(doc_budget) if doc_budget is not None else None
                            result.metadata["decomposition"] = meta_decomp
                        except (
                            AttributeError,
                            ConnectionError,
                            OSError,
                            RuntimeError,
                            TypeError,
                            ValueError,
                            asyncio.TimeoutError,
                        ) as _dec_err:
                            result.errors.append(f"Query decomposition failed: {str(_dec_err)}")

                    _apply_workspace_filtering_to_result()

                    # Attach retrieval guidance prompt in metadata for downstream awareness/debugging
                    try:
                        _rg = load_prompt("rag", "retrieval_guidance")
                        if _rg:
                            result.metadata["retrieval_guidance"] = _rg
                    except (ImportError, OSError, RuntimeError, TypeError, ValueError):
                        pass
                    result.metadata["sources_searched"] = list(retrieval_sources)
                    result.metadata["documents_retrieved"] = len(documents)

                    result.timings["retrieval"] = time.time() - retrieval_start
                    # Record phase duration with difficulty label
                    try:
                        from tldw_Server_API.app.core.Metrics.metrics_manager import observe_histogram
                        def _difficulty(docs: list) -> str:
                            try:
                                if not docs:
                                    difficulty = "hard"
                                else:
                                    high = sum(
                                        1
                                        for d in docs
                                        if float(getattr(d, "score", 0.0)) >= max(min_score, 0.3)
                                    )
                                    if high >= max(3, int(0.3 * len(docs))):
                                        difficulty = "easy"
                                    elif high >= 1:
                                        difficulty = "medium"
                                    else:
                                        difficulty = "hard"
                            except (AttributeError, RuntimeError, TypeError, ValueError):
                                return "unknown"
                            return difficulty
                        observe_histogram("rag_phase_duration_seconds", result.timings["retrieval"], labels={"phase": "retrieval", "difficulty": _difficulty(result.documents or [])})
                        # Also attach difficulty as OTEL attribute if span is active
                        if _otel_span is not None:
                            try:
                                _otel_span.set_attribute("rag.query_difficulty", _difficulty(result.documents or []))
                                _otel_span.set_attribute("rag.doc_count", int(len(result.documents or [])))
                            except (AttributeError, RuntimeError, TypeError, ValueError):
                                pass
                    except (ImportError, RuntimeError, TypeError, ValueError):
                        pass
                    if metrics:
                        metrics.retrieval_time = result.timings["retrieval"]

            except (
                AttributeError,
                ConnectionError,
                OSError,
                RuntimeError,
                Exception,
                TypeError,
                ValueError,
                asyncio.TimeoutError,
                sqlite3.Error,
            ) as e:
                result.errors.append(f"Document retrieval failed: {str(e)}")
                logger.error(f"Retrieval error: {e}")
                # Sample payload exemplar on retrieval failure
                try:
                    from .payload_exemplars import maybe_record_exemplar
                    maybe_record_exemplar(
                        query=query,
                        documents=result.documents or [],
                        answer=result.generated_answer or "",
                        reason="retrieval_error",
                        user_id=user_id,
                        namespace=index_namespace,
                    )
                except (ImportError, OSError, RuntimeError, TypeError, ValueError):
                    pass

                # On retrieval failure, attempt a best-effort Media DB FTS fallback.
                # This is especially important in local/test environments where
                # vector stores or adapters may be misconfigured but the Media DB
                # itself contains the uploaded content.
                if (
                    (not result.documents)
                    and _scope_includes_data_source(resolved_data_sources, DataSource.MEDIA_DB)
                    and (media_db_path or media_db is not None)
                    and search_mode in ("fts", "hybrid")
                ):
                    try:
                        from .database_retrievers import MediaDBRetriever as _MDBR
                        from .database_retrievers import RetrievalConfig as _RCfg
                        fb_cfg = _RCfg(
                            max_results=top_k,
                            min_score=min_score,
                            use_fts=True,
                            use_vector=False,
                            include_metadata=True,
                            fts_level=fts_level,
                        )
                        fb_retriever = _MDBR(
                            db_path=media_db_path,
                            config=fb_cfg,
                            user_id=str(user_id or "0"),
                            media_db=media_db,
                        )
                        fallback_docs = await fb_retriever.retrieve(
                            query=query,
                            media_type=None,
                            allowed_media_ids=include_media_ids,
                        )
                        if fallback_docs:
                            result.documents = fallback_docs
                            if isinstance(result.metadata, dict):
                                result.metadata.setdefault("fallbacks", {})
                                result.metadata["fallbacks"]["media_db_fts_on_error"] = True
                    except (
                        AttributeError,
                        ConnectionError,
                        OSError,
                        RuntimeError,
                        TypeError,
                        ValueError,
                        asyncio.TimeoutError,
                        sqlite3.Error,
                    ) as _fb_err:
                        result.errors.append(f"Media DB fallback retrieval on error failed: {str(_fb_err)}")
            finally:
                # Ensure OTEL span is closed
                if _otel_cm is not None:
                    with contextlib.suppress(ImportError, RuntimeError, TypeError, ValueError):
                        _otel_cm.__exit__(None, None, None)
        elif _skip_retrieval_stack:
            result.timings["retrieval"] = 0.0
            result.metadata["retrieval_bypassed"] = {
                "reason": _skip_retrieval_reason or "retrieval_bypassed",
                "documents_preserved": int(len(result.documents or [])),
            }
        elif _skip_local_retrieval:
            result.timings["retrieval"] = 0.0
            result.metadata["retrieval_bypassed"] = {
                "reason": "classification_local_disabled",
                "documents_preserved": int(len(result.documents or [])),
            }

        # ========== MULTI-VECTOR PASSAGES (optional, pre-rerank) ==========
        if enable_multi_vector_passages and result.documents:
            mv_start = time.time()
            try:
                used_precomputed = False
                if enable_precomputed_spans and apply_precomputed_spans and PrecomputedSpanConfig:
                    try:
                        pcfg = PrecomputedSpanConfig()
                        pre_docs = await apply_precomputed_spans(
                            query=query,
                            documents=result.documents,
                            config=pcfg,
                            user_id=user_id,
                        )
                        result.metadata.setdefault("multi_vector", {})
                        # When implementation is available, pre_docs can override documents
                        if pre_docs:
                            result.documents = pre_docs[: top_k]
                            result.metadata["multi_vector"]["precomputed_spans"] = True
                            used_precomputed = True
                        else:
                            result.metadata["multi_vector"]["precomputed_spans"] = False
                    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                        # If precomputed path fails, fall back silently to on-the-fly spans
                        result.metadata.setdefault("multi_vector", {})
                        result.metadata["multi_vector"]["precomputed_spans"] = False

                if apply_multi_vector_passages and MultiVectorConfig:
                    cfg = MultiVectorConfig(
                        span_chars=int(mv_span_chars or 300),
                        stride=int(mv_stride or 150),
                        max_spans_per_doc=int(mv_max_spans or 8),
                        flatten_to_spans=bool(mv_flatten_to_spans or False),
                    )
                    mv_docs = await apply_multi_vector_passages(
                        query=query,
                        documents=result.documents,
                        config=cfg,
                        user_id=user_id,
                    )
                    if mv_docs:
                        result.documents = mv_docs[: top_k]
                        result.metadata.setdefault("multi_vector", {})
                        result.metadata["multi_vector"].update({
                            "enabled": True,
                            "span_chars": cfg.span_chars,
                            "stride": cfg.stride,
                            "max_spans_per_doc": cfg.max_spans_per_doc,
                            "flattened": cfg.flatten_to_spans,
                        })
                else:
                    result.errors.append("Multi-vector module not available")
            except (
                AttributeError,
                ConnectionError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
                asyncio.TimeoutError,
            ) as e:
                result.errors.append(f"Multi-vector passages failed: {e}")
            finally:
                result.timings["multi_vector"] = time.time() - mv_start
                try:
                    from tldw_Server_API.app.core.Metrics.metrics_manager import observe_histogram
                    observe_histogram("rag_phase_duration_seconds", result.timings["multi_vector"], labels={"phase": "multi_vector", "difficulty": str(result.metadata.get("query_intent", "na"))})
                except (ImportError, RuntimeError, TypeError, ValueError):
                    pass

        # ========== NUMERIC/TABLE-AWARE BOOST (optional, pre-rerank) ==========
        # Record the metadata block even when no documents are retrieved so tests can
        # confirm the feature path was considered for numeric queries.
        if enable_numeric_table_boost:
            try:
                import re as _re
                q_has_num = bool(_re.search(r"\d", query)) or bool(_re.search(r"\b(percent|percentage|million|billion|thousand|\$|usd|eur|kg|g|lb|%|k|m|b)\b", query, _re.I))
            except (TypeError, ValueError):
                q_has_num = False
            if q_has_num:
                affected = 0
                if result.documents:
                    for d in result.documents:
                        try:
                            md = getattr(d, "metadata", None) or {}
                            chunk_type = str(md.get("chunk_type", "")).lower()
                            text = getattr(d, "content", "") or ""
                            numbers = sum(1 for _ in _re.finditer(r"\d", text))
                            looks_table = (chunk_type == "table") or (text.count("|") >= 3) or ("\t" in text)
                            if looks_table or numbers >= 6:
                                score_val = float(getattr(d, "score", 0.0) or 0.0)
                                # modest boost within [0,1]
                                d.score = min(1.0, score_val * 1.1 + 0.02)
                                md["numeric_table_boost"] = True
                                d.metadata = md
                                affected += 1
                        except (AttributeError, TypeError, ValueError):
                            continue
                # Always emit the metadata block when numeric intent is detected
                result.metadata["numeric_table_boost"] = {"enabled": True, "affected": int(affected)}

        # ========== GAP ANALYSIS / FOLLOW-UPS (optional) ==========
        if enable_gap_analysis and result.documents:
            try:
                ga_start = time.time()
                followups: list[str] = []
                # Try a lightweight LLM to propose follow-ups
                try:
                    from tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib import analyze as llm_analyze
                    # Determine default provider/model from config if available
                    try:
                        from tldw_Server_API.app.core.config import load_and_log_configs
                        _cfg = load_and_log_configs() or {}
                        _prov = (_cfg.get("RAG_DEFAULT_LLM_PROVIDER") or "openai").strip()
                        _model = (_cfg.get("RAG_DEFAULT_LLM_MODEL") or "gpt-4o-mini").strip()
                    except (ImportError, AttributeError, OSError, TypeError, ValueError):
                        _prov, _model = "openai", "gpt-4o-mini"
                    prompt = (
                        "You help a search system identify missing information.\n"
                        "Given the user query and several retrieved snippets, propose up to 2 concise follow-up search queries "
                        "that would likely fill important gaps. Return ONLY a JSON array of strings.\n\n"
                        f"Query: {query}\n\nSnippets:\n"
                    )
                    for d in result.documents[:5]:
                        snippet = (d.content or "")[:300].replace("\n", " ")
                        prompt += f"- {snippet}\n"
                    prompt += "\nJSON:"
                    llm_out = llm_analyze(api_name=_prov, input_data="", custom_prompt_arg=prompt, model_override=_model)
                    if isinstance(llm_out, str):
                        try:
                            parsed = parse_structured_output(
                                llm_out,
                                options=StructuredOutputOptions(parse_mode="lenient", strip_think_tags=True),
                            )
                            if isinstance(parsed, list):
                                followups = [q for q in parsed if isinstance(q, str) and q.strip()]
                            elif isinstance(parsed, dict):
                                wrapped_followups = parsed.get("followups") or parsed.get("suggestions")
                                if isinstance(wrapped_followups, list):
                                    followups = [q for q in wrapped_followups if isinstance(q, str) and q.strip()]
                        except (StructuredOutputParseError, TypeError, ValueError):
                            followups = [s.strip("- ") for s in llm_out.splitlines() if s.strip()]
                except (
                    AttributeError,
                    ConnectionError,
                    OSError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                    asyncio.TimeoutError,
                ):
                    # Fallback
                    followups = [f"detailed {query}", f"examples {query}"]
                followups = [q for q in followups if isinstance(q, str) and q.strip()][:max_followup_searches]
                if followups:
                    # Run in parallel
                    tasks = [
                        _execute_retrieval_variant("retrieval_followup", fq)
                        for fq in followups
                    ]
                    try:
                        follow_results = await asyncio.gather(*tasks)
                    except (
                        ConnectionError,
                        OSError,
                        RuntimeError,
                        TypeError,
                        ValueError,
                        asyncio.TimeoutError,
                    ):
                        follow_results = []
                    # Merge by id, keep higher score
                    merged = {d.id: d for d in result.documents}
                    for lst in follow_results:
                        for d in (lst or []):
                            prev = merged.get(d.id)
                            if prev is None or float(getattr(d, "score", 0.0)) > float(getattr(prev, "score", 0.0)):
                                merged[d.id] = d
                    result.documents = sorted(merged.values(), key=lambda x: getattr(x, "score", 0.0), reverse=True)[:top_k]
                    result.metadata["followups"] = followups
                result.timings["gap_analysis"] = time.time() - ga_start
            except (
                AttributeError,
                ConnectionError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
                asyncio.TimeoutError,
            ) as e:
                result.errors.append(f"Gap analysis failed: {e}")

        # ========== KEYWORD FILTERING ==========
        if keyword_filter and result.documents:
            filter_start = time.time()
            filtered_docs = []
            for doc in result.documents:
                content_lower = doc.content.lower()
                if any(keyword.lower() in content_lower for keyword in keyword_filter):
                    filtered_docs.append(doc)

            result.metadata["pre_filter_count"] = len(result.documents)
            result.documents = filtered_docs
            result.metadata["post_filter_count"] = len(filtered_docs)
            result.timings["keyword_filter"] = time.time() - filter_start

        # ========== INSTRUCTION-INJECTION FILTERING (pre-reranking) ==========
        if enable_injection_filter and result.documents:
            inj_start = time.time()
            try:
                if downweight_injection_docs:
                    summary = downweight_injection_docs(result.documents, strength=float(injection_filter_strength or 0.5))
                    result.metadata.setdefault("injection_filter", {})
                    result.metadata["injection_filter"].update({
                        "affected": int(summary.get("affected", 0)),
                        "total": int(summary.get("total", len(result.documents))),
                        "strength": float(injection_filter_strength or 0.5),
                    })
                    # Optional metric
                    try:
                        from tldw_Server_API.app.core.Metrics.metrics_manager import increment_counter
                        if int(summary.get("affected", 0)) > 0:
                            increment_counter("rag_injection_chunks_downweighted_total", int(summary.get("affected", 0)))
                    except (ImportError, RuntimeError, TypeError, ValueError):
                        pass
                else:
                    result.errors.append("Injection filter module not available")
            except (
                AttributeError,
                ConnectionError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
                asyncio.TimeoutError,
            ) as e:
                result.errors.append(f"Injection filtering failed: {str(e)}")
            finally:
                result.timings["injection_filter"] = time.time() - inj_start

        # ========== OPTIONAL CHUNK TYPE FILTER (metadata-based) ==========
        if chunk_type_filter and result.documents:
            try:
                allowed: set[str] = set()
                for t in chunk_type_filter:
                    norm = _normalize_chunk_type_value(t)
                    if norm:
                        allowed.add(norm)
                before_count = len(result.documents)
                filtered_docs = []
                for d in result.documents:
                    doc_type = _normalize_chunk_type_value((d.metadata or {}).get("chunk_type"))
                    if doc_type and doc_type in allowed:
                        filtered_docs.append(d)
                result.documents = filtered_docs
                result.metadata["chunk_type_filter_before"] = before_count
                result.metadata["chunk_type_filter_after"] = len(result.documents)
            except (AttributeError, TypeError, ValueError):
                pass

        # ========== CONTENT POLICY FILTERS & SANITATION ==========
        if result.documents:
            try:
                # OCR gating
                if ocr_confidence_threshold is not None:
                    try:
                        from tldw_Server_API.app.core.Metrics.metrics_manager import increment_counter
                        dropped = gate_docs_by_ocr_confidence(result.documents, float(ocr_confidence_threshold))
                        if dropped > 0:
                            increment_counter("rag_ocr_dropped_docs_total", dropped)
                    except (ImportError, RuntimeError, TypeError, ValueError):
                        pass
                # HTML sanitation
                if enable_html_sanitizer:
                    sanitized = 0
                    for d in (result.documents or []):
                        try:
                            before = d.content or ""
                            after = sanitize_html_allowlist(before, html_allowed_tags, html_allowed_attrs)
                            if after != before:
                                d.content = after
                                sanitized += 1
                        except (AttributeError, TypeError, ValueError):
                            continue
                    try:
                        if sanitized > 0:
                            from tldw_Server_API.app.core.Metrics.metrics_manager import increment_counter
                            increment_counter("rag_sanitized_docs_total", sanitized)
                    except (ImportError, RuntimeError, TypeError, ValueError):
                        pass
                # Content policy (PII/PHI)
                if enable_content_policy_filter:
                    summary = apply_content_policy(result.documents, policy_types=(content_policy_types or ["pii"]), mode=str(content_policy_mode or "redact"))
                    result.metadata.setdefault("content_policy", {})
                    result.metadata["content_policy"].update({
                        "enabled": True,
                        "types": content_policy_types or ["pii"],
                        "mode": content_policy_mode,
                        "affected": int(summary.get("affected", 0)),
                        "dropped": int(summary.get("dropped", 0)),
                    })
                    try:
                        from tldw_Server_API.app.core.Metrics.metrics_manager import increment_counter
                        if int(summary.get("affected", 0)) > 0:
                            increment_counter("rag_policy_filtered_chunks_total", int(summary.get("affected", 0)), labels={"mode": str(content_policy_mode or "redact")})
                    except (ImportError, RuntimeError, TypeError, ValueError):
                        pass
            except (AttributeError, RuntimeError, TypeError, ValueError):
                # Non-fatal: continue
                pass

        # ========== SECURITY FILTERING ==========
        if enable_security_filter and result.documents:
            security_start = time.time()
            try:
                if SecurityFilter and SensitivityLevel:
                    security_filter = SecurityFilter()
                    sensitivity_map = {
                        "public": SensitivityLevel.PUBLIC,
                        "internal": SensitivityLevel.INTERNAL,
                        "confidential": SensitivityLevel.CONFIDENTIAL,
                        "restricted": SensitivityLevel.RESTRICTED,
                    }
                    sensitivity_value = sensitivity_map.get(sensitivity_level)
                    if sensitivity_value is None:
                        valid_levels = ", ".join(sensitivity_map.keys())
                        raise ValueError(
                            f"Invalid sensitivity_level '{sensitivity_level}'. Expected one of: {valid_levels}"
                        )

                    # Detect PII if requested
                    if detect_pii:
                        detect_pii_batch = getattr(security_filter, "detect_pii_batch", None)
                        if callable(detect_pii_batch):
                            pii_report = detect_pii_batch([doc.content for doc in result.documents])
                            if inspect.isawaitable(pii_report):
                                pii_report = await pii_report
                        else:
                            detector = getattr(security_filter, "pii_detector", None)
                            detect_single = getattr(detector, "detect_pii", None) if detector is not None else None
                            if callable(detect_single):
                                pii_report = []
                                for doc in result.documents:
                                    matches = detect_single(doc.content)
                                    if inspect.isawaitable(matches):
                                        matches = await matches
                                    pii_report.append(
                                        [
                                            match.to_dict() if hasattr(match, "to_dict") else match
                                            for match in (matches or [])
                                        ]
                                    )
                            else:
                                pii_report = []
                        result.security_report = {"pii_detected": pii_report}

                    filtered_docs = None
                    filter_by_sensitivity = getattr(security_filter, "filter_by_sensitivity", None)
                    if callable(filter_by_sensitivity):
                        filtered_docs = filter_by_sensitivity(
                            result.documents,
                            max_level=sensitivity_value,
                        )
                        if inspect.isawaitable(filtered_docs):
                            filtered_docs = await filtered_docs
                        filtered_docs = _coerce_security_filter_sequence(
                            filtered_docs,
                            description="document sequence",
                        )

                        if redact_pii:
                            redact_pii_fn = getattr(security_filter, "redact_pii", None)
                            if callable(redact_pii_fn):
                                for doc in filtered_docs:
                                    redacted = redact_pii_fn(doc.content)
                                    doc.content = await redacted if inspect.isawaitable(redacted) else redacted
                    else:
                        filter_documents = getattr(security_filter, "filter_documents", None)
                        if callable(filter_documents):
                            serialized_docs = [
                                {
                                    "id": doc.id,
                                    "content": doc.content,
                                    "metadata": dict(doc.metadata or {}),
                                    "_document_ref": doc,
                                }
                                for doc in result.documents
                            ]
                            filtered_payloads = filter_documents(
                                serialized_docs,
                                user_id="anonymous",
                                max_sensitivity=sensitivity_value,
                                mask_pii=bool(redact_pii),
                            )
                            if inspect.isawaitable(filtered_payloads):
                                filtered_payloads = await filtered_payloads

                            filtered_docs = []
                            for payload in _coerce_security_filter_sequence(
                                filtered_payloads,
                                description="payload sequence",
                            ):
                                if not isinstance(payload, dict):
                                    logger.warning(
                                        "Security filter payload entry had unsupported type {}; skipping entry",
                                        type(payload).__name__,
                                    )
                                    continue
                                doc_ref = payload.get("_document_ref")
                                if doc_ref is None:
                                    continue
                                doc_ref.content = payload.get("content", doc_ref.content)
                                merged_metadata = dict(doc_ref.metadata or {})
                                payload_metadata = payload.get("metadata")
                                if isinstance(payload_metadata, dict):
                                    merged_metadata.update(payload_metadata)
                                if "pii_masked" in payload:
                                    merged_metadata["pii_masked"] = payload["pii_masked"]
                                if "pii_types" in payload:
                                    merged_metadata["pii_types"] = payload["pii_types"]
                                doc_ref.metadata = merged_metadata
                                filtered_docs.append(doc_ref)

                    if filtered_docs is not None:
                        result.documents = filtered_docs
                    result.timings["security_filter"] = time.time() - security_start

            except ImportError:
                result.errors.append("Security filter module not available")
                logger.warning("Security filter requested but module not available")
            except (
                AttributeError,
                ConnectionError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
                asyncio.TimeoutError,
            ) as e:
                result.errors.append(f"Security filter failed: {str(e)}")
                logger.error(f"Security filter error: {e}")

        # ========== TABLE PROCESSING ==========
        if enable_table_processing and result.documents:
            table_start = time.time()
            try:
                if TableProcessor:
                    processor = TableProcessor()
                    processed_docs = []

                    for doc in result.documents:
                        process_document = getattr(processor, "process_document", None)
                        process_document_tables = getattr(processor, "process_document_tables", None)

                        if callable(process_document):
                            processed = process_document(doc.content, method=table_method)
                            processed = await processed if inspect.isawaitable(processed) else processed
                            doc.content = processed
                        elif callable(process_document_tables):
                            processed = process_document_tables(doc.content, serialize_method=table_method)
                            processed = await processed if inspect.isawaitable(processed) else processed
                            if isinstance(processed, tuple):
                                processed_text, table_metadata = processed
                                doc.content = processed_text
                                if table_metadata:
                                    merged_metadata = dict(doc.metadata or {})
                                    merged_metadata["table_metadata"] = table_metadata
                                    doc.metadata = merged_metadata
                            else:
                                doc.content = processed
                        processed_docs.append(doc)

                    result.documents = processed_docs
                    result.timings["table_processing"] = time.time() - table_start

            except ImportError:
                result.errors.append("Table processing module not available")
                logger.warning("Table processing requested but module not available")

        # ========== VLM LATE CHUNKING (Optional) ==========
        if enable_vlm_late_chunking and result.documents:
            vlm_start = time.time()
            try:
                try:
                    from tldw_Server_API.app.core.Ingestion_Media_Processing.VLM.registry import (
                        get_backend as _get_vlm_backend,
                    )
                except ImportError:
                    def _get_vlm_backend(name=None):
                        return None

                # Pick backend
                backend = _get_vlm_backend(vlm_backend if vlm_backend not in (None, "auto") else None)
                if backend is None:
                    result.errors.append("VLM requested but no backend available")
                else:
                    # Operate on top-k documents to bound cost
                    # Allow media_db and notes_db sources when a local PDF path is present
                    allowed_sources = {"media_db", "notes_db"}
                    selected_docs = [
                        d for d in result.documents
                        if (d.metadata or {}).get("source") in allowed_sources and (d.metadata or {}).get("url")
                    ]
                    selected_docs = selected_docs[: max(1, int(vlm_late_chunk_top_k_docs or 1))]

                    vlm_added: list[Document] = []
                    for doc in selected_docs:
                        url = (doc.metadata or {}).get("url")
                        page_limit = vlm_max_pages
                        if not url:
                            continue
                        # Resolve PDF path: strictly require local file path (no remote URLs)
                        pdf_path = None
                        cleanup_tmp = False
                        try:
                            from pathlib import Path
                            pdf_path_obj = Path(str(url))
                            if pdf_path_obj.exists() and pdf_path_obj.suffix.lower() == ".pdf":
                                pdf_path = str(pdf_path_obj)
                            else:
                                # Unsupported: not a local PDF path
                                continue
                        except (OSError, TypeError, ValueError):
                            continue

                        # Use document-level VLM when available
                        try:
                            detections = []
                            if hasattr(backend, "process_pdf"):
                                res = backend.process_pdf(pdf_path, max_pages=page_limit)
                                by_page: list[dict[str, Any]] = []
                                if isinstance(getattr(res, "extra", None), dict):
                                    by_page = res.extra.get("by_page") or []
                                if by_page:
                                    for entry in by_page:
                                        page_no = entry.get("page")
                                        for d in (entry.get("detections") or []):
                                            label = str(d.get("label"))
                                            if vlm_detect_tables_only and label.lower() != "table":
                                                continue
                                            detections.append({
                                                "label": label,
                                                "score": float(d.get("score", 0.0)),
                                                "bbox": d.get("bbox") or [0.0, 0.0, 0.0, 0.0],
                                                "page": page_no,
                                            })
                            else:
                                # Per-page image mode
                                try:
                                    import pymupdf
                                    with pymupdf.open(pdf_path) as _doc:
                                        total_pages = len(_doc)
                                        max_pages = min(page_limit or total_pages, total_pages)
                                        for i, page in enumerate(_doc, start=1):
                                            if i > max_pages:
                                                break
                                            pix = page.get_pixmap(matrix=pymupdf.Matrix(2.0, 2.0), alpha=False)
                                            img_bytes = pix.tobytes("png")
                                            res = backend.process_image(img_bytes, context={"page": i, "pdf_path": pdf_path})
                                            for det in (getattr(res, "detections", []) or []):
                                                label = str(getattr(det, "label", ""))
                                                if vlm_detect_tables_only and label.lower() != "table":
                                                    continue
                                                detections.append({
                                                    "label": label,
                                                    "score": float(getattr(det, "score", 0.0)),
                                                    "bbox": list(getattr(det, "bbox", [0.0, 0.0, 0.0, 0.0])),
                                                    "page": i,
                                                })
                                except (ImportError, OSError, RuntimeError, TypeError, ValueError):
                                    continue

                            # Convert detections into lightweight Documents for reranking/search
                            for idx, d in enumerate(detections[:100]):  # bound new docs per source
                                label = str(d.get("label", "vlm"))
                                score = d.get("score", 0.0)
                                bbox = d.get("bbox")
                                page_no = d.get("page")
                                chunk_text = f"Detected {label} ({score:.2f}) on page {page_no} at {bbox}"
                                vlm_added.append(
                                    Document(
                                        id=f"vlm:{doc.id}:{idx}",
                                        content=chunk_text,
                                        source=doc.source,
                                        metadata={
                                            **(doc.metadata or {}),
                                            "chunk_type": ("table" if str(label).lower() == "table" else "vlm"),
                                            "page": page_no,
                                            "bbox": bbox,
                                            "derived_from": doc.id,
                                        },
                                        score=float(getattr(doc, "score", 0.0)),
                                    )
                                )
                        finally:
                            # No temp cleanup needed; remote URLs are not supported
                            pass
                    if vlm_added:
                        # Extend document list for downstream processing/reranking
                        result.documents.extend(vlm_added)
                result.timings["vlm_late_chunking"] = time.time() - vlm_start
            except (
                AttributeError,
                ConnectionError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
                asyncio.TimeoutError,
            ) as e:
                result.errors.append(f"VLM late-chunking failed: {e}")
                logger.warning(f"VLM late-chunking failed: {e}")

        # ========== PROGRESSIVE EVIDENCE ACCUMULATION ==========
        if enable_evidence_accumulation and result.documents and EvidenceAccumulator:
            accumulation_start = time.time()
            try:
                accumulator = EvidenceAccumulator(
                    max_rounds=accumulation_max_rounds,
                    enable_gap_assessment=True,
                )

                # Create retrieval function for additional rounds
                async def _additional_retrieval(gap_query: str, exclude_ids: set):
                    if not (MultiDatabaseRetriever and RetrievalConfig):
                        return []
                    try:
                        # Reuse the same retriever setup
                        db_paths = _build_pipeline_db_paths()

                        retriever = _build_multi_retriever(db_paths)
                        config = RetrievalConfig(
                            max_results=retrieval_top_k,
                            min_score=retrieval_min_score,
                            use_fts=(retrieval_search_mode in ["fts", "hybrid"]),
                            use_vector=(retrieval_search_mode in ["vector", "hybrid"]),
                            include_metadata=True,
                            fts_level=fts_level,
                            enable_text_late_chunking=enable_text_late_chunking,
                            chunk_method=chunk_method,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                            chunk_language=chunk_language,
                        )
                        data_sources = list(resolved_data_sources)
                        if not data_sources:
                            return []
                        new_docs = await retriever.retrieve(
                            query=gap_query,
                            sources=data_sources,
                            config=config,
                            index_namespace=retrieval_index_namespace,
                            retrieval_plan=_retrieval_plan_for(gap_query),
                            allowed_media_ids=include_media_ids,
                            allowed_note_ids=include_note_ids,
                        )
                        # Filter out already-seen documents
                        return [d for d in new_docs if d.id not in exclude_ids]
                    except (
                        AttributeError,
                        ConnectionError,
                        OSError,
                        RuntimeError,
                        TypeError,
                        ValueError,
                        asyncio.TimeoutError,
                    ) as e:
                        logger.warning(f"Additional retrieval failed: {e}")
                        return []

                accumulation_result = await accumulator.accumulate(
                    query=query,
                    initial_results=result.documents,
                    retrieval_fn=_additional_retrieval,
                    time_budget_sec=accumulation_time_budget_sec,
                )

                # Update documents with accumulated results
                result.documents = accumulation_result.documents
                result.metadata["evidence_accumulation"] = {
                    "total_rounds": accumulation_result.total_rounds,
                    "is_sufficient": accumulation_result.is_sufficient,
                    "sufficiency_reason": accumulation_result.sufficiency_reason,
                    "initial_docs": accumulation_result.metadata.get("initial_docs", 0),
                    "final_docs": len(accumulation_result.documents),
                    "docs_added": accumulation_result.metadata.get("docs_added", 0),
                }
                result.timings["evidence_accumulation"] = time.time() - accumulation_start

            except (
                AttributeError,
                ConnectionError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
                asyncio.TimeoutError,
            ) as e:
                result.errors.append(f"Evidence accumulation failed: {e}")
                logger.warning(f"Evidence accumulation error: {e}")

        # Apply personalization priors (pre-rerank) if requested
        try:
            if apply_feedback_boost and result.documents and UserPersonalizationStore:
                store = UserPersonalizationStore(feedback_user_id or user_id)
                result.documents = store.boost_documents(result.documents, corpus=index_namespace)
                result.metadata.setdefault("personalization", {})["boost_applied_pre_rerank"] = True
        except ValueError as exc:
            logger.debug(f"Personalization boost disabled for user_id={feedback_user_id or user_id}: {exc}")
        except (AttributeError, OSError, RuntimeError, TypeError, sqlite3.Error):
            pass

        # ========== SELF-CORRECTING RAG: DOCUMENT GRADING (Stage 1) ==========
        # Grade documents for relevance BEFORE expensive reranking/generation
        if enable_document_grading and result.documents and DocumentGrader:
            grading_start = time.time()
            try:
                grading_config = GradingConfig(
                    provider=grading_provider or "openai",
                    model=grading_model,
                    batch_size=grading_batch_size,
                    timeout_seconds=grading_timeout_sec,
                    fallback_to_score=grading_fallback_to_score,
                    fallback_min_score=grading_fallback_min_score,
                )
                grader = DocumentGrader(config=grading_config)

                filtered_docs, grading_metadata = await grader.filter_relevant(
                    query=query,
                    documents=result.documents,
                    threshold=grading_threshold,
                    provider=grading_provider,
                    model=grading_model,
                )

                # Store grading metadata
                result.metadata["document_grading"] = {
                    "enabled": True,
                    "threshold": grading_threshold,
                    "total_graded": grading_metadata.get("total_graded", 0),
                    "relevant_count": grading_metadata.get("relevant_count", 0),
                    "filtered_count": grading_metadata.get("filtered_count", 0),
                    "removed_count": grading_metadata.get("removed_count", 0),
                    "avg_relevance": grading_metadata.get("avg_relevance", 0.0),
                    "grading_latency_ms": grading_metadata.get("total_latency_ms", 0),
                }

                # Check if we should trigger query rewriting loop (Stage 2)
                avg_relevance = grading_metadata.get("avg_relevance", 0.0)
                if enable_query_rewriting_loop and avg_relevance < rewrite_relevance_threshold:
                    result.metadata["document_grading"]["low_relevance_detected"] = True
                    result.metadata["document_grading"]["will_rewrite"] = True

                # Update documents with filtered set
                if filtered_docs:
                    result.documents = filtered_docs
                else:
                    # If no documents passed grading, keep original for fallback
                    result.metadata["document_grading"]["fallback_to_original"] = True
                    logger.warning(f"No documents passed grading threshold {grading_threshold}, keeping originals")

                result.timings["document_grading"] = time.time() - grading_start

                # --- OTEL span for document grading ---
                if enable_observability and get_telemetry_manager:
                    try:
                        _tm = get_telemetry_manager()
                        _tr = _tm.get_tracer("tldw.rag")
                        with _tr.start_as_current_span("rag.document_grading") as _span:
                            _span.set_attribute("rag.phase", "document_grading")
                            _span.set_attribute("rag.grading_threshold", grading_threshold)
                            _span.set_attribute("rag.docs_graded", grading_metadata.get("total_graded", 0))
                            _span.set_attribute("rag.docs_passed", grading_metadata.get("filtered_count", 0))
                            _span.set_attribute("rag.avg_relevance", avg_relevance)
                    except (AttributeError, RuntimeError, TypeError, ValueError):
                        pass

            except (
                AttributeError,
                ConnectionError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
                asyncio.TimeoutError,
            ) as e:
                result.errors.append(f"Document grading failed: {e}")
                logger.warning(f"Document grading error: {e}")
                result.metadata["document_grading"] = {"enabled": True, "error": str(e)}

        # ========== SELF-CORRECTING RAG: QUERY REWRITING LOOP (Stage 2) ==========
        # When document grading shows low relevance, rewrite query and retry retrieval
        if (
            enable_query_rewriting_loop
            and QueryRewriter
            and result.metadata.get("document_grading", {}).get("low_relevance_detected")
        ):
            rewrite_loop_start = time.time()
            rewrite_attempts = []
            best_relevance = result.metadata.get("document_grading", {}).get("avg_relevance", 0.0)
            best_docs = result.documents
            current_query = query

            try:
                rewriter = QueryRewriter()

                for attempt in range(max_rewrite_attempts):
                    # Rewrite the query using improve_for_retrieval strategy
                    rewrites = rewriter.rewrite_query(
                        current_query,
                        strategies=["improve_for_retrieval"],
                        failed_docs=result.documents,
                        failure_reason=f"avg_relevance_{best_relevance:.2f}",
                    )

                    if not rewrites:
                        logger.debug(f"No rewrites generated on attempt {attempt + 1}")
                        break

                    # Use the best rewrite (highest confidence)
                    best_rewrite = max(rewrites, key=lambda r: r.confidence)
                    rewritten_query = best_rewrite.rewritten_query

                    rewrite_attempts.append({
                        "attempt": attempt + 1,
                        "original_query": current_query,
                        "rewritten_query": rewritten_query,
                        "rewrite_type": best_rewrite.rewrite_type,
                        "confidence": best_rewrite.confidence,
                        "explanation": best_rewrite.explanation,
                    })

                    logger.debug(f"Query rewrite attempt {attempt + 1}: '{rewritten_query}'")

                    # Re-run retrieval with rewritten query
                    if MultiDatabaseRetriever and RetrievalConfig:
                        try:
                            db_paths = _build_pipeline_db_paths()

                            retriever = _build_multi_retriever(db_paths)
                            retrieval_config = RetrievalConfig(
                                max_results=retrieval_top_k,
                                min_score=retrieval_min_score,
                                use_fts=(retrieval_search_mode in ["fts", "hybrid"]),
                                use_vector=(retrieval_search_mode in ["vector", "hybrid"]),
                                include_metadata=True,
                                fts_level=fts_level,
                                enable_text_late_chunking=enable_text_late_chunking,
                                chunk_method=chunk_method,
                                chunk_size=chunk_size,
                                chunk_overlap=chunk_overlap,
                                chunk_language=chunk_language,
                            )
                            data_sources = list(resolved_data_sources)

                            new_docs = await retriever.retrieve(
                                query=rewritten_query,
                                sources=data_sources,
                                config=retrieval_config,
                                index_namespace=retrieval_index_namespace,
                                retrieval_plan=_retrieval_plan_for(rewritten_query),
                            )

                            if new_docs:
                                # Re-grade the new documents
                                if enable_document_grading and DocumentGrader:
                                    grading_config = GradingConfig(
                                        provider=grading_provider or "openai",
                                        model=grading_model,
                                        batch_size=grading_batch_size,
                                        timeout_seconds=grading_timeout_sec,
                                        fallback_to_score=grading_fallback_to_score,
                                        fallback_min_score=grading_fallback_min_score,
                                    )
                                    grader = DocumentGrader(config=grading_config)
                                    _, new_grading_metadata = await grader.filter_relevant(
                                        query=rewritten_query,
                                        documents=new_docs,
                                        threshold=grading_threshold,
                                    )

                                    new_avg_relevance = new_grading_metadata.get("avg_relevance", 0.0)
                                    rewrite_attempts[-1]["new_avg_relevance"] = new_avg_relevance

                                    # Check if we've improved
                                    if new_avg_relevance > best_relevance:
                                        best_relevance = new_avg_relevance
                                        best_docs = new_docs
                                        current_query = rewritten_query

                                    # Check if we've exceeded threshold
                                    if new_avg_relevance >= rewrite_relevance_threshold:
                                        logger.info(f"Query rewrite succeeded after {attempt + 1} attempts, relevance: {new_avg_relevance:.2f}")
                                        rewrite_attempts[-1]["success"] = True
                                        break
                                else:
                                    # No grading, just use new docs
                                    best_docs = new_docs
                                    current_query = rewritten_query
                                    break

                        except (
                            AttributeError,
                            ConnectionError,
                            OSError,
                            RuntimeError,
                            TypeError,
                            ValueError,
                            asyncio.TimeoutError,
                        ) as ret_err:
                            logger.warning(f"Retrieval with rewritten query failed: {ret_err}")
                            rewrite_attempts[-1]["error"] = str(ret_err)

                # Update result with best documents found
                if best_docs:
                    result.documents = best_docs

                # Store query rewriting loop metadata
                result.metadata["query_rewrite_loop"] = {
                    "enabled": True,
                    "attempts": len(rewrite_attempts),
                    "max_attempts": max_rewrite_attempts,
                    "initial_relevance": result.metadata.get("document_grading", {}).get("avg_relevance", 0.0),
                    "final_relevance": best_relevance,
                    "threshold": rewrite_relevance_threshold,
                    "improved": best_relevance > result.metadata.get("document_grading", {}).get("avg_relevance", 0.0),
                    "rewrite_attempts": rewrite_attempts,
                }

                result.timings["query_rewrite_loop"] = time.time() - rewrite_loop_start

            except (
                AttributeError,
                ConnectionError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
                asyncio.TimeoutError,
            ) as e:
                result.errors.append(f"Query rewriting loop failed: {e}")
                logger.warning(f"Query rewriting loop error: {e}")
                result.metadata["query_rewrite_loop"] = {"enabled": True, "error": str(e)}

        # ========== RERANKING ==========
        if enable_reranking and result.documents and reranking_strategy != "none":
            rerank_start = time.time()
            if (
                reranking_strategy == "two_tier"
                and not (create_reranker and RerankingStrategy and RerankingConfig)
            ):
                _record_profile_degradation(
                    component="reranking_strategy",
                    from_value="two_tier",
                    to_value="hybrid",
                    reason="unavailable_dependency",
                )
                reranking_strategy = "hybrid"
            try:
                # --- OTEL: reranking span ---
                _otel_cm_rk = None
                _otel_span_rk = None
                if enable_observability and get_telemetry_manager:
                    try:
                        _tm = get_telemetry_manager()
                        _tr = _tm.get_tracer("tldw.rag")
                        _attrs = {
                            "rag.phase": "rerank",
                            "rag.strategy": str(reranking_strategy),
                            "rag.top_k": int((rerank_top_k or top_k) or 0),
                        }
                        _otel_cm_rk = _tr.start_as_current_span("rag.rerank")
                        _otel_span_rk = _otel_cm_rk.__enter__()
                        for _k, _v in _attrs.items():
                            with contextlib.suppress(AttributeError, RuntimeError, TypeError, ValueError):
                                _otel_span_rk.set_attribute(_k, _v)
                    except (AttributeError, RuntimeError, TypeError, ValueError):
                        _otel_cm_rk = None
                        _otel_span_rk = None
                if create_reranker and RerankingStrategy and RerankingConfig:
                    strategy_map = {
                        "flashrank": RerankingStrategy.FLASHRANK,
                        "cross_encoder": RerankingStrategy.CROSS_ENCODER,
                        "hybrid": RerankingStrategy.HYBRID,
                        "llama_cpp": RerankingStrategy.LLAMA_CPP,
                        "diversity": RerankingStrategy.DIVERSITY,
                        "llm_scoring": RerankingStrategy.LLM_SCORING,
                        "two_tier": RerankingStrategy.TWO_TIER,
                    }

                    # Determine LLM reranker provider/model from config when requested
                    selected_strategy = strategy_map[reranking_strategy]
                    llm_client = None
                    if selected_strategy == RerankingStrategy.LLM_SCORING:
                        try:
                            import tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib as sgl
                            from tldw_Server_API.app.core.config import load_and_log_configs
                            cfg = load_and_log_configs()
                            if not isinstance(cfg, dict):
                                cfg = {}
                            prov = (cfg.get('RAG_LLM_RERANKER_PROVIDER') or '').strip()
                            model = (cfg.get('RAG_LLM_RERANKER_MODEL') or '').strip()
                            if not model:
                                # No model set -> fallback to FlashRank
                                selected_strategy = RerankingStrategy.FLASHRANK
                            else:
                                class _LLMClient:
                                    def __init__(self, provider: str, model_name: str):
                                        self.provider = provider or 'openai'
                                        self.model_name = model_name
                                    def analyze(self, prompt_text: str):
                                        # Use analyze with prompt as custom_prompt_arg
                                        return sgl.analyze(
                                            api_name=self.provider,
                                            input_data="",
                                            custom_prompt_arg=prompt_text,
                                            api_key=None,
                                            system_message=None,
                                            temp=None,
                                            model_override=self.model_name,
                                        )
                                llm_client = _LLMClient(prov, model)
                        except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
                            selected_strategy = RerankingStrategy.FLASHRANK

                    # Determine model for reranker when applicable
                    model_name_for_reranker = None
                    if selected_strategy == RerankingStrategy.LLAMA_CPP:
                        try:
                            from tldw_Server_API.app.core.config import load_and_log_configs
                            cfg = load_and_log_configs()
                            if not isinstance(cfg, dict):
                                cfg = {}
                        except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
                            cfg = {}
                        # Precedence: explicit param -> env/config
                        model_name_for_reranker = reranking_model or cfg.get("RAG_LLAMA_RERANKER_MODEL")
                    elif selected_strategy == RerankingStrategy.CROSS_ENCODER:
                        try:
                            from tldw_Server_API.app.core.config import load_and_log_configs
                            cfg = load_and_log_configs()
                            if not isinstance(cfg, dict):
                                cfg = {}
                        except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
                            cfg = {}
                        model_name_for_reranker = reranking_model or cfg.get("RAG_TRANSFORMERS_RERANKER_MODEL")

                    rerank_config = RerankingConfig(
                        strategy=selected_strategy,
                        top_k=rerank_top_k or top_k,
                        model_name=model_name_for_reranker,
                        # Request-level gating overrides (TwoTier)
                        min_relevance_prob=rerank_min_relevance_prob,
                        sentinel_margin=rerank_sentinel_margin,
                    )
                    include_rerank_snapshots = debug_mode and _resolve_include_rerank_debug_documents(
                        include_rerank_debug_documents
                    )
                    rerank_debug_limit = max(1, int(rerank_top_k or top_k or 1))
                    try:
                        if include_rerank_snapshots and isinstance(result.metadata, dict):
                            result.metadata["pre_rerank_documents"] = _serialize_rerank_debug_documents(
                                result.documents,
                                include_content=True,
                                limit=rerank_debug_limit,
                            )
                        reranker = create_reranker(selected_strategy, rerank_config, llm_client=llm_client)
                        reranked = await _resilient_call("reranking", reranker.rerank, query, result.documents)
                    except Exception as rerank_exc:
                        if reranking_strategy != "two_tier":
                            raise
                        logger.warning(
                            "Two-tier reranker failed at runtime; degrading to hybrid strategy: {}",
                            rerank_exc,
                        )
                        _record_profile_degradation(
                            component="reranking_strategy",
                            from_value="two_tier",
                            to_value="hybrid",
                            reason="runtime_failure",
                        )
                        reranking_strategy = "hybrid"
                        selected_strategy = RerankingStrategy.HYBRID
                        rerank_config = RerankingConfig(
                            strategy=selected_strategy,
                            top_k=rerank_top_k or top_k,
                            model_name=None,
                            min_relevance_prob=rerank_min_relevance_prob,
                            sentinel_margin=rerank_sentinel_margin,
                        )
                        reranker = create_reranker(selected_strategy, rerank_config, llm_client=llm_client)
                        reranked = await _resilient_call("reranking", reranker.rerank, query, result.documents)
                    if reranked and hasattr(reranked[0], 'document'):
                        result.documents = [sd.document for sd in reranked[:(rerank_top_k or top_k)]]
                    else:
                        result.documents = reranked[:(rerank_top_k or top_k)]
                    if include_rerank_snapshots and isinstance(result.metadata, dict):
                        result.metadata["reranked_documents"] = _serialize_rerank_debug_documents(
                            result.documents,
                            include_content=True,
                            limit=rerank_debug_limit,
                        )

                    result.timings["reranking"] = time.time() - rerank_start
                    try:
                        from tldw_Server_API.app.core.Metrics.metrics_manager import observe_histogram
                        observe_histogram("rag_reranking_duration_seconds", result.timings["reranking"], labels={"strategy": reranking_strategy})
                        # Also record as a generic phase without difficulty
                        observe_histogram("rag_phase_duration_seconds", result.timings["reranking"], labels={"phase": "reranking", "difficulty": "na"})
                        if _otel_span_rk is not None:
                            with contextlib.suppress(AttributeError, RuntimeError, TypeError, ValueError):
                                _otel_span_rk.set_attribute("rag.doc_count", int(len(result.documents or [])))
                    except (ImportError, RuntimeError, TypeError, ValueError):
                        pass
                    if metrics:
                        metrics.reranking_time = result.timings["reranking"]

                    # If reranker exposes calibration metadata (e.g., TwoTier), record it
                    try:
                        if hasattr(reranker, 'last_metadata') and isinstance(reranker.last_metadata, dict):
                            result.metadata.setdefault("reranking_calibration", {})
                            result.metadata["reranking_calibration"].update(reranker.last_metadata)
                            # Attach learned-fusion specific decoration when applicable
                            _decorate_calibration_metadata()
                    except (AttributeError, RuntimeError, TypeError, ValueError):
                        pass

                    # For non Two-Tier strategies, if learned fusion is requested but no
                    # calibrator metadata exists, compute a simple fused probability from
                    # the top document score so that downstream gating can still use a
                    # calibrated signal.
                    if enable_learned_fusion:
                        try:
                            if isinstance(result.metadata, dict) and "reranking_calibration" not in result.metadata:
                                top_doc = result.documents[0] if result.documents else None
                                if top_doc is not None:
                                    import math as _math_lf
                                    import os as _os_lf
                                    # Use shared env weights to stay consistent with Two-Tier,
                                    # but only CE-style weight is applied since we only have
                                    # a single rerank score available here.
                                    try:
                                        bias = float(_os_lf.getenv("RAG_RERANK_CALIB_BIAS", "-1.5"))
                                    except (TypeError, ValueError):
                                        bias = -1.5
                                    try:
                                        w_ce = float(_os_lf.getenv("RAG_RERANK_CALIB_W_CE", "2.5"))
                                    except (TypeError, ValueError):
                                        w_ce = 2.5
                                    try:
                                        raw = float(getattr(top_doc, "score", 0.0) or 0.0)
                                    except (TypeError, ValueError):
                                        raw = 0.0
                                    logit = bias + (w_ce * raw)
                                    try:
                                        fused_prob = 1.0 / (1.0 + _math_lf.exp(-logit))
                                    except (TypeError, ValueError):
                                        fused_prob = 0.5
                                    # Threshold from env (same as Two-Tier gating default)
                                    try:
                                        thr = float(_os_lf.getenv("RAG_MIN_RELEVANCE_PROB", "0.35"))
                                    except (TypeError, ValueError):
                                        thr = 0.35
                                    gated_flag = fused_prob < thr
                                    result.metadata["reranking_calibration"] = {
                                        "strategy": str(reranking_strategy),
                                        "top_doc_score": raw,
                                        "fused_score": fused_prob,
                                        "threshold": thr,
                                        "gated": gated_flag,
                                    }
                                    # Ensure learned-fusion metadata decoration is applied
                                    _decorate_calibration_metadata()
                        except (AttributeError, TypeError, ValueError):
                            pass

                else:
                    result.errors.append("Reranking module not available")
                    logger.warning("Reranking requested but module not available")
            except (
                AttributeError,
                ConnectionError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
                asyncio.TimeoutError,
            ) as e:
                result.errors.append(f"Reranking failed: {str(e)}")
                logger.error(f"Reranking error: {e}")
                # Sample payload exemplar on reranking failure
                try:
                    from .payload_exemplars import maybe_record_exemplar
                    maybe_record_exemplar(
                        query=query,
                        documents=result.documents or [],
                        answer=result.generated_answer or "",
                        reason="reranking_error",
                        user_id=user_id,
                        namespace=index_namespace,
                    )
                except (ImportError, RuntimeError, TypeError, ValueError):
                    pass
            finally:
                if _otel_cm_rk is not None:
                    with contextlib.suppress(AttributeError, RuntimeError, TypeError, ValueError):
                        _otel_cm_rk.__exit__(None, None, None)

        # ========== SELF-CORRECTING RAG: WEB SEARCH FALLBACK (Stage 3) ==========
        # When local retrieval has low relevance, fall back to web search
        if enable_web_fallback and fallback_to_web_search:
            web_fallback_start = time.time()
            try:
                # Compute relevance signal for web fallback decision
                # Use reranking calibration if available, otherwise use avg document score
                relevance_signal = 0.5  # Default
                if isinstance(result.metadata, dict):
                    cal = result.metadata.get("reranking_calibration", {})
                    if isinstance(cal, dict) and "fused_score" in cal:
                        relevance_signal = float(cal.get("fused_score", 0.5))
                    elif result.documents:
                        # Fall back to average document score
                        scores = [
                            float(getattr(d, "score", 0.0) or 0.0)
                            for d in result.documents
                        ]
                        relevance_signal = sum(scores) / len(scores) if scores else 0.0

                # Only trigger if below threshold
                if relevance_signal < web_fallback_threshold:
                    logger.debug(
                        f"Web fallback triggered: relevance {relevance_signal:.2f} < threshold {web_fallback_threshold}"
                    )

                    merged_docs, web_metadata = await fallback_to_web_search(
                        query=query,
                        local_docs=result.documents or [],
                        relevance_signal=relevance_signal,
                        threshold=web_fallback_threshold,
                        engine=web_search_engine,
                        result_count=web_fallback_result_count,
                        merge_strategy=web_fallback_merge_strategy,
                        max_total=top_k * 2,  # Allow some extra docs from web
                    )

                    if web_metadata.get("triggered"):
                        result.documents = merged_docs
                        result.metadata["web_fallback"] = {
                            "enabled": True,
                            "triggered": True,
                            "relevance_signal": relevance_signal,
                            "threshold": web_fallback_threshold,
                            "web_results_count": web_metadata.get("web_results_count", 0),
                            "merged_count": web_metadata.get("merged_count", 0),
                            "engine_used": web_metadata.get("engine_used"),
                            "merge_strategy": web_fallback_merge_strategy,
                            "search_time_ms": web_metadata.get("search_time_ms", 0),
                        }

                        # --- OTEL span for web fallback ---
                        if enable_observability and get_telemetry_manager:
                            try:
                                _tm = get_telemetry_manager()
                                _tr = _tm.get_tracer("tldw.rag")
                                with _tr.start_as_current_span("rag.web_fallback") as _span:
                                    _span.set_attribute("rag.phase", "web_fallback")
                                    _span.set_attribute("rag.web_results", web_metadata.get("web_results_count", 0))
                                    _span.set_attribute("rag.engine", web_search_engine)
                            except (AttributeError, RuntimeError, TypeError, ValueError):
                                pass
                else:
                    result.metadata["web_fallback"] = {
                        "enabled": True,
                        "triggered": False,
                        "relevance_signal": relevance_signal,
                        "threshold": web_fallback_threshold,
                        "reason": "relevance_above_threshold",
                    }

                result.timings["web_fallback"] = time.time() - web_fallback_start

            except (
                AttributeError,
                ConnectionError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
                asyncio.TimeoutError,
            ) as e:
                result.errors.append(f"Web fallback failed: {e}")
                logger.warning(f"Web fallback error: {e}")
                result.metadata["web_fallback"] = {"enabled": True, "error": str(e)}

        # ========== WHY THESE SOURCES (metadata) ==========
        try:
            docs = result.documents or []
            if docs:
                import urllib.parse
                def _host(u: Optional[str]) -> Optional[str]:
                    try:
                        if not u:
                            return None
                        return urllib.parse.urlparse(str(u)).hostname
                    except (AttributeError, TypeError, ValueError):
                        return None
                hosts = []
                sources_ = []
                ages = []
                scores = []
                now_ts = time.time()
                for d in docs:
                    md = getattr(d, 'metadata', None) or (d.get('metadata') if isinstance(d, dict) else {}) or {}
                    url = md.get('url')
                    h = _host(url)
                    if h:
                        hosts.append(h)
                    src = md.get('source') or str(getattr(d, 'source', '') or '')
                    if src:
                        sources_.append(str(src))
                    created = md.get('last_modified') or md.get('created_at')
                    ts = None
                    try:
                        if isinstance(created, (int, float)):
                            ts = float(created)
                        elif isinstance(created, str) and created:
                            ts = datetime.fromisoformat(created.replace('Z','+00:00')).timestamp()
                    except (TypeError, ValueError):
                        ts = None
                    if ts is not None:
                        ages.append(max(0.0, (now_ts - ts) / 86400.0))
                    try:
                        scores.append(float(getattr(d, 'score', d.get('score', 0.0) if isinstance(d, dict) else 0.0)))
                    except (TypeError, ValueError):
                        scores.append(0.0)
                n = max(1, len(docs))
                uniq_hosts = len(set(hosts)) if hosts else 0
                uniq_sources = len(set(sources_)) if sources_ else 0
                diversity = min(1.0, max(uniq_hosts, uniq_sources) / float(n))
                fresh_portion = 0.5
                if ages:
                    fresh = sum(1 for a in ages if a <= 90.0)
                    fresh_portion = fresh / float(len(ages))
                if scores:
                    smin, smax = min(scores), max(scores)
                    if smax > smin:
                        topicality = sum((s - smin) / (smax - smin) for s in scores) / float(len(scores))
                    else:
                        topicality = 1.0
                else:
                    topicality = 0.0
                def _title(md):
                    try:
                        return (md.get('title') or '') if isinstance(md, dict) else ''
                    except (AttributeError, TypeError, ValueError):
                        return ''
                top_contexts = []
                for d in docs[: min(10, n)]:
                    md = getattr(d, 'metadata', None) or (d.get('metadata') if isinstance(d, dict) else {}) or {}
                    top_contexts.append({
                        "id": getattr(d, 'id', d.get('id') if isinstance(d, dict) else None),
                        "title": _title(md),
                        "score": float(getattr(d, 'score', md.get('score', 0.0) if isinstance(md, dict) else 0.0) or 0.0),
                        "url": md.get('url'),
                        "source": md.get('source') or str(getattr(d, 'source', '') or ''),
                    })
                result.metadata["why_these_sources"] = {
                    "diversity": round(float(diversity), 4),
                    "freshness": round(float(fresh_portion), 4),
                    "topicality": round(float(topicality), 4),
                    "top_contexts": top_contexts,
                }
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            pass

        # ========== SIBLING INCLUSION ==========
        if include_sibling_chunks and result.documents and sibling_window and sibling_window > 0:
            siblings_start = time.time()
            try:
                # Index docs by parent and index
                parents: dict[str, dict[int, Document]] = {}
                for d in result.documents:
                    pid = str(d.metadata.get("parent_id", ""))
                    cidx_md = d.metadata.get("chunk_index", -1)
                    cidx = int(cidx_md) if isinstance(cidx_md, int) or (isinstance(cidx_md, str) and cidx_md.isdigit()) else -1
                    if pid and cidx >= 0:
                        parents.setdefault(pid, {})[cidx] = d

                sibling_added: list[Document] = []
                seen_ids = {getattr(d, 'id', None) for d in result.documents}

                for d in list(result.documents):
                    pid = str(d.metadata.get("parent_id", ""))
                    cidx_md = d.metadata.get("chunk_index", -1)
                    cidx = int(cidx_md) if isinstance(cidx_md, int) or (isinstance(cidx_md, str) and cidx_md.isdigit()) else -1
                    if not pid or cidx < 0:
                        continue
                    siblings = parents.get(pid, {})
                    # expand symmetrically up to window size
                    for w in range(1, int(sibling_window) + 1):
                        for adj in (cidx - w, cidx + w):
                            sdoc = siblings.get(adj)
                            if sdoc is not None and getattr(sdoc, 'id', None) not in seen_ids:
                                sibling_added.append(sdoc)
                                seen_ids.add(getattr(sdoc, 'id', None))

                if sibling_added:
                    result.documents.extend(sibling_added)
                result.metadata["siblings_added_count"] = len(sibling_added)
                result.timings["sibling_inclusion"] = time.time() - siblings_start
            except (AttributeError, TypeError, ValueError) as e:
                result.errors.append(f"Sibling inclusion failed: {str(e)}")

        # Cap documents to top_k when reranking is disabled
        if (not enable_reranking or reranking_strategy == "none") and result.documents:
            try:
                max_docs = int(top_k or 0)
            except (TypeError, ValueError):
                max_docs = 0
            if max_docs > 0 and len(result.documents) > max_docs:
                try:
                    result.documents = sorted(
                        result.documents,
                        key=lambda d: getattr(d, "score", 0.0),
                        reverse=True,
                    )[:max_docs]
                except (AttributeError, TypeError, ValueError):
                    result.documents = list(result.documents)[:max_docs]

        evidence_chain_result = None

        # ========== CITATION GENERATION ==========
        if enable_citations and result.documents:
            citation_start = time.time()
            try:
                if CitationGenerator:
                    generator = CitationGenerator()
                    # Map style string to enum if available
                    style_map = {
                        "apa": getattr(CitationStyle, "APA", None),
                        "mla": getattr(CitationStyle, "MLA", None),
                        "chicago": getattr(CitationStyle, "CHICAGO", None),
                        "harvard": getattr(CitationStyle, "HARVARD", None),
                        "ieee": getattr(CitationStyle, "IEEE", None),
                    }
                    style_enum = style_map.get(citation_style) or next(iter([v for v in style_map.values() if v is not None]), None)

                    if enable_evidence_chains and hasattr(generator, "generate_citations_with_chains"):
                        dual, chain_result = await generator.generate_citations_with_chains(
                            documents=result.documents,
                            query=query,
                            generated_answer=result.generated_answer,
                            style=style_enum if style_enum is not None else CitationStyle.MLA if CitationStyle else None,
                            include_chunks=bool(enable_chunk_citations),
                            max_citations=min(len(result.documents), (rerank_top_k or top_k or 10)),
                        )
                        if chain_result:
                            evidence_chain_result = chain_result
                    else:
                        dual = await generator.generate_citations(
                            documents=result.documents,
                            query=query,
                            style=style_enum if style_enum is not None else CitationStyle.MLA if CitationStyle else None,
                            include_chunks=bool(enable_chunk_citations),
                            max_citations=min(len(result.documents), (rerank_top_k or top_k or 10))
                        )

                    # Combined citations list for backward compatibility
                    result.citations = (
                        [{"type": "academic", "formatted": s} for s in (dual.academic_citations or [])] +
                        ([{"type": "chunk", **c.to_dict()} for c in (dual.chunk_citations or [])])
                    )
                    # Expose detailed structures via metadata
                    result.metadata["academic_citations"] = dual.academic_citations or []
                    result.metadata["chunk_citations"] = [c.to_dict() for c in (dual.chunk_citations or [])]
                    result.metadata["inline_citations"] = dual.inline_markers or {}
                    result.metadata["citation_map"] = dual.citation_map or {}

                    result.timings["citation_generation"] = time.time() - citation_start
                else:
                    result.errors.append("Citation module not available")
                    logger.warning("Citations requested but module not available")
            except (
                AttributeError,
                ConnectionError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
                asyncio.TimeoutError,
            ) as e:
                result.errors.append(f"Citation generation failed: {str(e)}")
                logger.error(f"Citation error: {e}")

        # ========== SELF-CORRECTING RAG: KNOWLEDGE STRIPS (Stage 4) ==========
        # Partition documents into semantic strips and filter by relevance before generation
        if enable_knowledge_strips and result.documents and process_knowledge_strips:
            strips_start = time.time()
            try:
                filtered_docs, strips_metadata = await process_knowledge_strips(
                    query=query,
                    documents=result.documents,
                    strip_size_tokens=strip_size_tokens,
                    min_relevance=strip_min_relevance,
                    max_strips=max_strips,
                    use_llm_grading=False,  # Use heuristic for speed by default
                )

                if filtered_docs:
                    result.documents = filtered_docs

                result.metadata["knowledge_strips"] = {
                    "enabled": True,
                    "total_strips": strips_metadata.get("total_strips", 0),
                    "relevant_strips": strips_metadata.get("relevant_strips", 0),
                    "filtered_strips": strips_metadata.get("filtered_strips", 0),
                    "avg_relevance": strips_metadata.get("avg_relevance", 0.0),
                    "strip_size_tokens": strip_size_tokens,
                    "min_relevance": strip_min_relevance,
                    "resulting_docs": len(filtered_docs) if filtered_docs else 0,
                }

                result.timings["knowledge_strips"] = time.time() - strips_start

                # --- OTEL span for knowledge strips ---
                if enable_observability and get_telemetry_manager:
                    try:
                        _tm = get_telemetry_manager()
                        _tr = _tm.get_tracer("tldw.rag")
                        with _tr.start_as_current_span("rag.knowledge_strips") as _span:
                            _span.set_attribute("rag.phase", "knowledge_strips")
                            _span.set_attribute("rag.total_strips", strips_metadata.get("total_strips", 0))
                            _span.set_attribute("rag.filtered_strips", strips_metadata.get("filtered_strips", 0))
                    except (AttributeError, RuntimeError, TypeError, ValueError):
                        pass

            except (
                AttributeError,
                ConnectionError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
                asyncio.TimeoutError,
            ) as e:
                result.errors.append(f"Knowledge strips processing failed: {e}")
                logger.warning(f"Knowledge strips error: {e}")
                result.metadata["knowledge_strips"] = {"enabled": True, "error": str(e)}

        # ========== EVIDENCE CHAINS (before generation for chain-aware citations) ==========
        if enable_evidence_chains and result.documents and EvidenceChainBuilder:
            chain_start = time.time()
            try:
                if evidence_chain_result is None:
                    chain_builder = EvidenceChainBuilder(
                        enable_llm_extraction=True,
                    )

                    # Build chains - note: we don't have the answer yet, so chains are built from docs
                    # This will be re-run after generation if needed for claim extraction
                    evidence_chain_result = await chain_builder.build_chains(
                        query=query,
                        documents=result.documents,
                        generated_answer=None,  # Will be updated post-generation
                    )

                if evidence_chain_result:
                    result.metadata["evidence_chains"] = {
                        "total_chains": len(evidence_chain_result.chains),
                        "overall_confidence": evidence_chain_result.overall_confidence,
                        "multi_hop_detected": evidence_chain_result.multi_hop_detected,
                        "total_nodes": evidence_chain_result.metadata.get("total_nodes", 0),
                    }

                result.timings["evidence_chains"] = time.time() - chain_start

            except (
                AttributeError,
                ConnectionError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
                asyncio.TimeoutError,
            ) as e:
                result.errors.append(f"Evidence chain building failed: {e}")
                logger.warning(f"Evidence chains error: {e}")

        # ========== ANSWER GENERATION ==========
        # Honor reranking calibration gating if present (e.g., TwoTier strategy)
        try:
            _cal = result.metadata.get("reranking_calibration") if isinstance(result.metadata, dict) else None
            gated_generation = bool(_cal.get("gated")) if isinstance(_cal, dict) else False
            # When calibration metadata is present, ensure fused_score/version/decision
            # are wired for observability.
            if isinstance(_cal, dict):
                if "fused_score" not in _cal and "top_doc_prob" in _cal:
                    with contextlib.suppress(TypeError, ValueError):
                        _cal["fused_score"] = float(_cal.get("top_doc_prob") or 0.0)
                if enable_learned_fusion:
                    _cal["enabled"] = True
                if calibrator_version:
                    _cal.setdefault("version", calibrator_version)
                # Decision will be finalized in the gated branch below
                result.metadata["reranking_calibration"] = _cal
        except (AttributeError, TypeError, ValueError):
            gated_generation = False

        effective_enable_generation = _resolve_effective_enable_generation()

        if effective_enable_generation and not result.documents:
            result.generated_answer = None
            result.metadata["answer_generation_skipped"] = "no_documents"
            effective_enable_generation = False

        if not effective_enable_generation:
            retrieval_only_result = build_retrieval_only_result(
                resolved_request=resolved_request,
                retrieval_plan=retrieval_plan,
                retrieval_result=RetrievedEvidence(
                    documents=list(result.documents or []),
                    metadata=dict(result.metadata or {}),
                ),
            )
            retrieval_only_result = coordinate_standard_result_evidence(
                retrieval_only_result,
                resolved_request,
                retrieval_plan=retrieval_plan,
            )
            standard_evidence_coordinated = True
            result.documents = list(retrieval_only_result.documents)
            result.metadata.update(dict(retrieval_only_result.metadata or {}))

        if effective_enable_generation and not gated_generation and not result.cache_hit:
            generation_start = time.time()
            try:
                # --- OTEL: generation span ---
                _otel_cm_gen = None
                _otel_span_gen = None
                if enable_observability and get_telemetry_manager:
                    try:
                        _tm = get_telemetry_manager()
                        _tr = _tm.get_tracer("tldw.rag")
                        _attrs = {
                            "rag.phase": "generation",
                            "rag.model": str(generation_model or ""),
                            "rag.provider": str(generation_provider or ""),
                            "rag.multi_turn": bool(enable_multi_turn_synthesis),
                        }
                        _otel_cm_gen = _tr.start_as_current_span("rag.generation")
                        _otel_span_gen = _otel_cm_gen.__enter__()
                        for _k, _v in _attrs.items():
                            with contextlib.suppress(AttributeError, RuntimeError, TypeError, ValueError):
                                _otel_span_gen.set_attribute(_k, _v)
                    except (AttributeError, RuntimeError, TypeError, ValueError):
                        _otel_cm_gen = None
                        _otel_span_gen = None
                # Strict extractive path: assemble answer from retrieved spans only
                if bool(strict_extractive):
                    try:
                        # Simple extractive assembly: pick top sentences from top documents
                        max_sents = 6
                        chosen: list[str] = []
                        import re as _re
                        q_terms = [t.lower() for t in _re.findall(r"[A-Za-z0-9_-]{3,}", query or "")][:10]
                        for doc in (result.documents or [])[: min(5, len(result.documents or []))]:
                            text = (getattr(doc, 'content', '') or '').strip()
                            if not text:
                                continue
                            sents = [s.strip() for s in _re.split(r"(?<=[\.!?])\s+", text) if s.strip()]
                            # prefer a sentence containing a query term
                            hit = None
                            for s in sents:
                                low = s.lower()
                                if any(t in low for t in q_terms):
                                    hit = s
                                    break
                            if not hit and sents:
                                hit = sents[0]
                            if hit and hit not in chosen:
                                chosen.append(hit)
                                if len(chosen) >= max_sents:
                                    break
                        result.generated_answer = " " .join(chosen).strip()
                    except (AttributeError, TypeError, ValueError) as _se:
                        result.errors.append(f"Strict extractive assembly failed: {_se}")
                        result.generated_answer = None
                elif AnswerGenerator:
                    generator = AnswerGenerator(
                        model=generation_model,
                        provider=generation_provider,
                    )

                    # Prepare base context from top documents
                    context_docs = (result.documents[:5] if result.documents else [])

                    # Structured response writer: XML-tagged context with citation rules
                    if enable_structured_response and format_context_xml is not None and build_writer_system_prompt is not None:
                        _writer_chunks = []
                        for _wd in context_docs:
                            _wd_meta = getattr(_wd, 'metadata', {}) or {}
                            _writer_chunks.append({
                                "content": getattr(_wd, 'content', str(_wd)),
                                "title": _wd_meta.get("title", ""),
                                "url": _wd_meta.get("url", "") or _wd_meta.get("source", ""),
                                "metadata": _wd_meta,
                            })
                        context = format_context_xml(_writer_chunks)
                        _writer_mode = search_depth_mode or "balanced"
                        _writer_policy = None
                        if get_writer_depth_policy is not None:
                            with contextlib.suppress(TypeError, ValueError):
                                _writer_policy = get_writer_depth_policy(
                                    mode=_writer_mode,
                                    max_generation_tokens=max_generation_tokens,
                                )
                        _writer_sys = build_writer_system_prompt(
                            mode=_writer_mode,
                            max_generation_tokens=max_generation_tokens,
                        )
                        # Override generation_prompt with structured writer prompt
                        if not generation_prompt:
                            generation_prompt = _writer_sys
                        else:
                            generation_prompt = f"{generation_prompt}\n\n{_writer_sys}"
                        if build_writer_user_prompt is not None:
                            context = build_writer_user_prompt(query=query, context_xml=context)
                        result.metadata.setdefault("structured_writer", {})
                        result.metadata["structured_writer"].update({
                            "enabled": True,
                            "mode": _writer_mode,
                            "context_results": len(_writer_chunks),
                            "max_generation_tokens": int(max_generation_tokens),
                        })
                        if isinstance(_writer_policy, dict):
                            result.metadata["structured_writer"]["depth_policy"] = _writer_policy
                    else:
                        context = "\n\n".join([getattr(doc, 'content', str(doc)) for doc in context_docs])

                    if enable_multi_turn_synthesis:
                        # Strict budget control
                        t0 = time.time()
                        budget = float(synthesis_time_budget_sec) if synthesis_time_budget_sec else None
                        aborted = False

                        # Draft
                        draft_tokens = int(synthesis_draft_tokens or min(max_generation_tokens, 400))
                        d_start = time.time()
                        draft_out = await generator.generate(
                            query=query,
                            context=context,
                            prompt_template=generation_prompt,
                            max_tokens=draft_tokens,
                        )
                        d_ans = draft_out.get("answer") if isinstance(draft_out, dict) else draft_out
                        if isinstance(d_ans, str):
                            d_ans_text = d_ans
                        elif d_ans is None:
                            d_ans_text = ""
                        else:
                            try:
                                d_ans_text = "".join(str(x) for x in d_ans)
                            except (TypeError, ValueError):
                                d_ans_text = str(d_ans)
                        d_dt = time.time() - d_start

                        # Critique
                        c_text = None
                        c_dt = 0.0
                        try:
                            import tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib as sgl
                            # Construct a compact critique prompt using small snippets
                            snippets = []
                            for d in context_docs[:3]:
                                s = (getattr(d, 'content', '') or '')[:250].replace('\n', ' ')
                                if s:
                                    snippets.append(f"- {s}")
                            crit_prompt = (
                                "You are a careful reviewer.\n"
                                "Given the user query, retrieved snippets, and the draft answer, list the top 3 issues (missing facts or unsupported claims).\n"
                                f"Query: {query}\nSnippets:\n" + "\n".join(snippets) + f"\n\nDraft:\n{d_ans_text}\n\nIssues:"
                            )
                            c_start = time.time()
                            c_text = sgl.analyze(api_name="openai", input_data="", custom_prompt_arg=crit_prompt, model_override=None)
                            c_dt = time.time() - c_start
                        except (ImportError, ConnectionError, OSError, RuntimeError, TypeError, ValueError, asyncio.TimeoutError):
                            c_text = "- Ensure claims are supported by provided snippets.\n- Add missing specifics.\n- Clarify ambiguous statements."
                        if isinstance(c_text, str):
                            c_text_val = c_text
                        elif c_text is None:
                            c_text_val = ""
                        else:
                            try:
                                c_text_val = "".join(str(x) for x in c_text)
                            except (TypeError, ValueError):
                                c_text_val = str(c_text)

                        # Check budget
                        if budget is not None and (time.time() - t0) >= budget:
                            aborted = True
                            result.generated_answer = d_ans_text
                            result.metadata.setdefault("synthesis", {})
                            result.metadata["synthesis"].update({"enabled": True, "aborted": True, "durations": {"draft": d_dt, "critique": c_dt, "refine": 0.0}})
                        else:
                            # Refine
                            refine_tokens = int(synthesis_refine_tokens or max_generation_tokens)
                            r_ctx = context + "\n\nCRITIQUE:\n" + c_text_val
                            r_start = time.time()
                            r_out = await generator.generate(
                                query=query,
                                context=r_ctx,
                                prompt_template=generation_prompt,
                                max_tokens=refine_tokens,
                            )
                            r_ans = r_out.get("answer") if isinstance(r_out, dict) else r_out
                            r_dt = time.time() - r_start
                            result.generated_answer = r_ans
                            result.metadata.setdefault("synthesis", {})
                            result.metadata["synthesis"].update({"enabled": True, "aborted": False, "durations": {"draft": d_dt, "critique": c_dt, "refine": r_dt}})
                    else:
                        async def _generate_standard_answer(
                            *,
                            query: str,
                            context: str,
                            generation_prompt: Optional[str] = None,
                            max_generation_tokens: Optional[int] = None,
                        ) -> Any:
                            return await _resilient_call(
                                "generation",
                                generator.generate,
                                query=query,
                                context=context,
                                prompt_template=generation_prompt,
                                max_tokens=max_generation_tokens,
                            )

                        resolved_request.payload["generation_prompt"] = generation_prompt
                        resolved_request.payload["max_generation_tokens"] = max_generation_tokens
                        generation_phase_kwargs: dict[str, Any] = {
                            "resolved_request": resolved_request,
                            "retrieval_plan": retrieval_plan,
                            "derived_evidence": SimpleNamespace(
                                documents=list(context_docs),
                                metadata=dict(result.metadata),
                                citations=list((result.metadata or {}).get("chunk_citations", []) or []),
                                verification_report=(result.metadata or {}).get("verification_report"),
                            ),
                            "generate_answer_fn": _generate_standard_answer,
                            "generation_context": context,
                        }
                        generation_result = await execute_generation_phase(**generation_phase_kwargs)
                        if isinstance(generation_result, dict):
                            generation_result = coordinate_standard_result_evidence(
                                generation_result,
                                resolved_request,
                                retrieval_plan=retrieval_plan,
                            )
                        else:
                            generation_result = coordinate_standard_result_evidence(
                                generation_result,
                                resolved_request,
                                retrieval_plan=retrieval_plan,
                            )
                        standard_evidence_coordinated = True
                        if isinstance(generation_result, dict):
                            result.generated_answer = generation_result.get(
                                "generated_answer",
                                generation_result.get("answer"),
                            )
                            generation_documents = (
                                generation_result.get("documents")
                                or generation_result.get("sources")
                            )
                            if generation_documents is not None:
                                result.documents = list(generation_documents)
                            result.metadata.update(dict(generation_result.get("metadata") or {}))
                            if generation_result.get("chunk_citations") is not None:
                                result.metadata["chunk_citations"] = list(
                                    generation_result.get("chunk_citations") or []
                                )
                            if generation_result.get("verification_report") is not None:
                                result.metadata["verification_report"] = generation_result.get("verification_report")
                        else:
                            result.generated_answer = generation_result.generated_answer
                            result.metadata.update(dict(generation_result.metadata or {}))
                            result.metadata["chunk_citations"] = list(generation_result.chunk_citations or [])
                            if generation_result.verification_report is not None:
                                result.metadata["verification_report"] = generation_result.verification_report
                    result.timings["answer_generation"] = time.time() - generation_start
                    try:
                        from tldw_Server_API.app.core.Metrics.metrics_manager import observe_histogram
                        observe_histogram("rag_phase_duration_seconds", result.timings["answer_generation"], labels={"phase": "generation", "difficulty": str(result.metadata.get("query_intent", "na"))})
                        if enable_multi_turn_synthesis:
                            observe_histogram("rag_phase_duration_seconds", result.timings["answer_generation"], labels={"phase": "synthesis", "difficulty": str(result.metadata.get("query_intent", "na"))})
                        if _otel_span_gen is not None:
                            try:
                                _ans_len = len(result.generated_answer or "")
                                _otel_span_gen.set_attribute("rag.answer_length", int(_ans_len))
                            except (AttributeError, RuntimeError, TypeError, ValueError):
                                pass
                    except (ImportError, RuntimeError, TypeError, ValueError):
                        pass
                    if metrics:
                        metrics.generation_time = result.timings["answer_generation"]

                    # Follow-up suggestions generation (bounded-latency, best effort)
                    if enable_suggestions and generate_suggestions is not None and result.generated_answer:
                        try:
                            _suggestions_start = time.time()
                            _suggestions = await generate_suggestions(
                                query=query,
                                response_text=str(result.generated_answer),
                                chat_history=chat_history,
                                llm_provider=classifier_provider or generation_provider or "openai",
                                llm_model=classifier_model or generation_model,
                                num_suggestions=num_suggestions,
                                llm_timeout_sec=3.0,
                            )
                            result.metadata["suggestions"] = _suggestions
                            result.timings["suggestions"] = time.time() - _suggestions_start
                        except Exception as _sug_exc:
                            logger.debug(f"Suggestion generation failed: {_sug_exc!r}")

            except ImportError:
                result.errors.append("Generation module not available")
                logger.warning("Answer generation requested but module not available")
            except (
                AttributeError,
                ConnectionError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
                asyncio.TimeoutError,
            ) as e:
                result.errors.append(f"Answer generation failed: {str(e)}")
                logger.error(f"Generation error: {e}")
                try:
                    from .payload_exemplars import maybe_record_exemplar
                    maybe_record_exemplar(
                        query=query,
                        documents=result.documents or [],
                        answer=result.generated_answer or "",
                        reason="generation_error",
                        user_id=user_id,
                        namespace=index_namespace,
                    )
                except (ImportError, RuntimeError, TypeError, ValueError):
                    pass
            finally:
                if _otel_cm_gen is not None:
                    with contextlib.suppress(AttributeError, RuntimeError, TypeError, ValueError):
                        _otel_cm_gen.__exit__(None, None, None)
        elif effective_enable_generation and gated_generation:
            # Record a metadata entry and bump a metric for observability
            result.metadata.setdefault("generation_gate", {})
            result.metadata["generation_gate"].update({
                "reason": "low_relevance_probability",
                "at": time.time(),
            })
            try:
                from tldw_Server_API.app.core.Metrics.metrics_manager import increment_counter
                increment_counter("rag_generation_gated_total", 1, labels={"strategy": "two_tier"})
            except (ImportError, RuntimeError, TypeError, ValueError):
                pass
            # Sample payload exemplar when generation is gated
            try:
                from .payload_exemplars import maybe_record_exemplar
                maybe_record_exemplar(
                    query=query,
                    documents=result.documents or [],
                    answer=result.generated_answer or "",
                    reason="generation_gated",
                    user_id=user_id,
                    namespace=index_namespace,
                )
            except (ImportError, RuntimeError, TypeError, ValueError):
                pass

            # Decide abstention policy for calibration-based gating.
            # Priority:
            #   1) enable_learned_fusion + abstention_policy
            #   2) enable_abstention + abstention_behavior
            #   3) default "continue" (no replacement answer)
            effective_policy: str = "continue"
            if enable_learned_fusion:
                effective_policy = abstention_policy or "continue"
            elif enable_abstention:
                effective_policy = abstention_behavior or "continue"

            # Record the decision in reranking calibration metadata when present
            try:
                if isinstance(result.metadata, dict):
                    cal = result.metadata.get("reranking_calibration")
                    if isinstance(cal, dict):
                        cal["decision"] = effective_policy
                        result.metadata["reranking_calibration"] = cal
            except (AttributeError, TypeError, ValueError):
                pass

            # Abstention / clarifying question path based on effective policy
            if effective_policy in {"ask", "decline"}:
                try:
                    if effective_policy == "ask":
                        clar_q = None
                        # Form a concise clarifying question using query analysis if available
                        if QueryAnalyzer:
                            try:
                                domain = analysis_domain
                                clar_q = f"Please clarify: what specific aspect of '{query}' should I focus on{f' in {domain}' if domain else ''}?"
                            except (AttributeError, TypeError, ValueError):
                                clar_q = None
                        if not clar_q:
                            clar_q = f"Could you clarify which specific details about '{query}' you need?"
                        result.generated_answer = clar_q
                    elif effective_policy == "decline":
                        result.generated_answer = "I don’t have sufficient grounded evidence to answer confidently. Please clarify your question or provide more context."
                except (AttributeError, TypeError, ValueError):
                    pass

        # ========== HARD CITATIONS (per-sentence) ==========
        # Build per-sentence citation map using claims (if available) or heuristic fallback
        try:
            if result.generated_answer:
                hc = None
                # Prefer claims payload if present
                claims_payload = result.metadata.get("claims") if isinstance(result.metadata, dict) else None
                if build_hard_citations:
                    hc = build_hard_citations(result.generated_answer, result.documents or [], claims_payload=claims_payload)
                if isinstance(hc, dict):
                    result.metadata["hard_citations"] = hc
                    # If hard-citation coverage is incomplete and strict mode is requested, apply behavior
                    if bool(require_hard_citations):
                        cov = float(hc.get("coverage") or 0.0)
                        if cov < 1.0:
                            _apply_generation_gate("missing_hard_citations", coverage=cov)
                            try:
                                from tldw_Server_API.app.core.Metrics.metrics_manager import increment_counter
                                increment_counter("rag_missing_hard_citations_total", 1)
                            except (ImportError, RuntimeError, TypeError, ValueError):
                                pass
                            # Honor low_confidence_behavior
                            if low_confidence_behavior == "ask":
                                note = "\n\n[Note] Some statements lack supporting citations. Please clarify or provide sources."
                                result.generated_answer = (result.generated_answer or "") + note
                            elif low_confidence_behavior == "decline":
                                result.generated_answer = "Insufficient evidence: missing citations for some statements."
                        # Gauge for coverage (report once per answer)
                        try:
                            from tldw_Server_API.app.core.Metrics.metrics_manager import set_gauge
                            set_gauge("rag_hard_citation_coverage", cov, labels={"strategy": "standard"})
                        except (ImportError, RuntimeError, TypeError, ValueError):
                            pass
        except (AttributeError, RuntimeError, TypeError, ValueError) as e:
            result.errors.append(f"Hard citations mapping failed: {str(e)}")

        # ========== QUOTE-LEVEL CITATIONS ==========
        try:
            if result.generated_answer and build_quote_citations:
                qc = build_quote_citations(result.generated_answer, result.documents or [])
                if isinstance(qc, dict):
                    result.metadata["quote_citations"] = qc
        except (AttributeError, RuntimeError, TypeError, ValueError) as e:
            result.errors.append(f"Quote citations mapping failed: {str(e)}")

        # ========== FAST GROUNDEDNESS CHECK (Self-Correcting RAG Stage 5) ==========
        # Lightweight check before expensive claims extraction
        fast_grounded = True  # default assumption
        fast_groundedness_confidence = 0.0
        try:
            if enable_fast_hallucination_check and result.generated_answer and check_fast_groundedness:
                with otel_span("rag.fast_groundedness_check"):
                    fg_result, fg_meta = await check_fast_groundedness(
                        query=query,
                        answer=result.generated_answer,
                        documents=result.documents or [],
                        provider=fast_hallucination_provider,
                        model=fast_hallucination_model,
                        timeout_sec=fast_hallucination_timeout_sec,
                    )
                    fast_grounded = fg_result.is_grounded
                    fast_groundedness_confidence = fg_result.confidence
                    result.metadata["fast_groundedness"] = fg_meta

                    # If highly confident grounded, can optionally skip full claims
                    if fast_grounded and fast_groundedness_confidence >= 0.9:
                        result.metadata["fast_groundedness"]["skip_full_claims"] = True
        except (
            AttributeError,
            ConnectionError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            asyncio.TimeoutError,
        ) as e:
            result.errors.append(f"Fast groundedness check failed: {str(e)}")

        # ========== CLAIMS & FACTUALITY ==========
        # Optionally skip if fast groundedness check passed with high confidence
        skip_claims_due_to_groundedness = (
            enable_fast_hallucination_check
            and fast_grounded
            and fast_groundedness_confidence >= 0.9
            and result.metadata.get("fast_groundedness", {}).get("skip_full_claims", False)
        )
        if enable_claims and result.generated_answer and not skip_claims_due_to_groundedness:
            try:
                # Import shared analyze function for LLM calls
                import tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib as sgl

                def _analyze(api_name: str, input_data: Any, custom_prompt_arg: Optional[str] = None,
                             api_key: Optional[str] = None, system_message: Optional[str] = None,
                             temp: Optional[float] = None, **kwargs):
                    return sgl.analyze(api_name, input_data, custom_prompt_arg, api_key, system_message, temp, **kwargs)

                if ClaimsEngine:
                    engine = ClaimsEngine(_analyze)
                    # Default NLI model from environment if not provided
                    if not nli_model:
                        import os
                        nli_model = os.environ.get("RAG_NLI_MODEL") or os.environ.get("RAG_NLI_MODEL_PATH")
                    job_context = None
                    if ClaimsJobContext is not None:
                        user_id_val = None
                        try:
                            user_id_val = int(user_id) if user_id is not None else None
                        except (TypeError, ValueError):
                            user_id_val = None
                        job_context = ClaimsJobContext(
                            user_id=user_id_val,
                            request_id=trace_id,
                            endpoint="rag_claims",
                        )
                    job_budget = None
                    if resolve_claims_job_budget is not None:
                        settings_obj = None
                        try:
                            from tldw_Server_API.app.core.config import settings as _settings
                            if isinstance(_settings, dict):
                                settings_obj = _settings
                            elif hasattr(_settings, "dict"):
                                settings_obj = _settings.dict()
                        except (ImportError, AttributeError, TypeError):
                            settings_obj = None
                        budget_usd = kwargs.get("claims_budget_usd")
                        budget_tokens = kwargs.get("claims_budget_tokens")
                        budget_strict = kwargs.get("claims_budget_strict")
                        try:
                            budget_usd = float(budget_usd) if budget_usd is not None else None
                        except (TypeError, ValueError):
                            budget_usd = None
                        try:
                            budget_tokens = int(budget_tokens) if budget_tokens is not None else None
                        except (TypeError, ValueError):
                            budget_tokens = None
                        if isinstance(budget_strict, str):
                            budget_strict = _shared_is_truthy(budget_strict)
                        job_budget = resolve_claims_job_budget(
                            settings=settings_obj,
                            max_cost_usd=budget_usd,
                            max_tokens=budget_tokens,
                            strict=budget_strict if isinstance(budget_strict, bool) else None,
                        )
                    # Build a per-claim retrieval that uses MultiDatabaseRetriever and hybrid search when available
                    async def _retrieve_for_claim(c_text: str, top_k: int = 5):
                        try:
                            if MultiDatabaseRetriever and RetrievalConfig:
                                db_paths = _build_pipeline_db_paths()
                                # Initialize multi retriever scoped to user's databases
                                mdr = _build_multi_retriever(db_paths)
                                ds = list(resolved_data_sources)

                                # For media_db, attempt hybrid; for others, simple retrieve
                                docs: list[Any] = []
                                # Media hybrid
                                med = mdr.retrievers.get(DataSource.MEDIA_DB)
                                if (
                                    med is not None
                                    and _scope_includes_data_source(resolved_data_sources, DataSource.MEDIA_DB)
                                ):
                                    rh = getattr(med, 'retrieve_hybrid', None)
                                    if rh is not None and asyncio.iscoroutinefunction(rh) and search_mode == "hybrid":
                                        media_docs = await rh(query=c_text, alpha=hybrid_alpha)
                                    else:
                                        media_docs = await med.retrieve(query=c_text)
                                    docs.extend(media_docs)
                                # Other sources
                                for src in ds:
                                    if src == DataSource.MEDIA_DB:
                                        continue
                                    retr = mdr.retrievers.get(src)
                                    if retr is not None:
                                        try:
                                            more = await retr.retrieve(query=c_text)
                                            docs.extend(more)
                                        except (
                                            AttributeError,
                                            ConnectionError,
                                            OSError,
                                            RuntimeError,
                                            TypeError,
                                            ValueError,
                                            asyncio.TimeoutError,
                                        ):
                                            pass
                                # Sort and cap
                                docs = sorted(docs, key=lambda d: getattr(d, 'score', 0.0), reverse=True)
                                return docs[:top_k]
                        except (
                            AttributeError,
                            ConnectionError,
                            OSError,
                            RuntimeError,
                            TypeError,
                            ValueError,
                            asyncio.TimeoutError,
                        ) as e:
                            logger.debug(f"Per-claim retrieval fallback to base docs due to error: {e}")
                        return result.documents[:top_k] if result.documents else []

                    # Prefer pre-extracted claims if available for current documents
                    claims_out = None
                    try:
                        pre_claims: list[str] = []
                        if media_db_path and (result.documents or []):
                            from tldw_Server_API.app.core.config import settings as _settings
                            with managed_media_database(
                                client_id=str(_settings.get("SERVER_CLIENT_ID", "SERVER_API_V1")),
                                db_path=media_db_path,
                                initialize=False,
                                suppress_close_exceptions=(
                                    AttributeError,
                                    RuntimeError,
                                    TypeError,
                                    ValueError,
                                    sqlite3.Error,
                                ),
                            ) as db:
                                # Collect media IDs present in documents
                                media_ids: list[int] = []
                                for d in result.documents:
                                    try:
                                        mid = d.metadata.get("media_id") if isinstance(d.metadata, dict) else None
                                        if mid is not None:
                                            media_ids.append(int(mid))
                                    except (TypeError, ValueError):
                                        continue
                                media_ids = list(dict.fromkeys(media_ids))[:5]
                                if media_ids:
                                    # Fetch a small number of claims per media
                                    for mid in media_ids:
                                        rows = db.execute_query(
                                            "SELECT claim_text FROM Claims WHERE media_id = ? AND deleted = 0 LIMIT ?",
                                            (int(mid), int(claims_max)),
                                        ).fetchall()
                                        pre_claims.extend([r[0] for r in rows])
                        if pre_claims:
                            # Verify these claims directly, skipping extraction
                            from tldw_Server_API.app.core.Claims_Extraction.claims_engine import Claim as _Claim
                            verifications = []
                            for i, ctext in enumerate(pre_claims[:claims_max]):
                                cv = await engine.verifier.verify(
                                    claim=_Claim(id=f"pc{i+1}", text=ctext),
                                    query=query,
                                    base_documents=result.documents or [],
                                    retrieve_fn=_retrieve_for_claim,
                                    top_k=claims_top_k,
                                    conf_threshold=claims_conf_threshold,
                                    mode=(claim_verifier or "hybrid").strip().lower(),
                                    budget=job_budget,
                                    job_context=job_context,
                                    doc_only_mode=doc_only_verification,
                                    numeric_precision_mode=numeric_precision_mode,
                                )
                                verifications.append(cv)
                            supported = sum(1 for v in verifications if v.label == "supported")
                            refuted = sum(1 for v in verifications if v.label == "refuted")
                            nei = sum(1 for v in verifications if v.label == "nei")
                            total = max(1, len(verifications))
                            precision = supported / total
                            coverage = (supported + refuted) / total
                            # Enhanced claim output with new fields
                            claims_payload = [
                                {
                                    "id": v.claim.id,
                                    "text": v.claim.text,
                                    "span": list(v.claim.span) if v.claim.span else None,
                                    "claim_type": v.claim.claim_type.value if hasattr(v.claim, "claim_type") else "general",
                                    "status": v.status.value if hasattr(v, "status") else v.label,
                                    "label": v.label,
                                    "confidence": v.confidence,
                                    "match_level": v.match_level.value if hasattr(v, "match_level") else "interpretation",
                                    "source_authority": v.source_authority.value if hasattr(v, "source_authority") else 1,
                                    "requires_external_knowledge": getattr(v, "requires_external_knowledge", False),
                                    "evidence": [
                                        {
                                            "doc_id": e.doc_id,
                                            "snippet": e.snippet,
                                            "score": e.score,
                                            "authority": e.authority.value if hasattr(e, "authority") else 1,
                                        }
                                        for e in v.evidence
                                    ],
                                    "citations": v.citations,
                                    "rationale": v.rationale,
                                }
                                for v in verifications
                            ]
                            factuality_payload = {
                                "supported": supported,
                                "refuted": refuted,
                                "nei": nei,
                                "precision": precision,
                                "coverage": coverage,
                            }
                        else:
                            # Fall back to on-the-fly extraction from the generated answer
                            claims_run = await engine.run(
                                answer=result.generated_answer,
                                query=query,
                                documents=result.documents or [],
                                claim_extractor=claim_extractor,
                                claim_verifier=claim_verifier,
                                claims_top_k=claims_top_k,
                                claims_conf_threshold=claims_conf_threshold,
                                claims_max=claims_max,
                                retrieve_fn=_retrieve_for_claim,
                                nli_model=nli_model,
                                claims_concurrency=claims_concurrency,
                                job_budget=job_budget,
                                job_context=job_context,
                            )
                            claims_payload = claims_run.get("claims")
                            factuality_payload = claims_run.get("summary")
                            verifications = claims_run.get("verifications", [])
                    except (
                        AttributeError,
                        ConnectionError,
                        OSError,
                        RuntimeError,
                        TypeError,
                        ValueError,
                        asyncio.TimeoutError,
                    ) as _eclaims:
                        logger.debug(f"Pre-extracted claims path failed: {_eclaims}")
                        claims_run = await engine.run(
                            answer=result.generated_answer,
                            query=query,
                            documents=result.documents or [],
                            claim_extractor=claim_extractor,
                            claim_verifier=claim_verifier,
                            claims_top_k=claims_top_k,
                            claims_conf_threshold=claims_conf_threshold,
                            claims_max=claims_max,
                            retrieve_fn=_retrieve_for_claim,
                            nli_model=nli_model,
                            claims_concurrency=claims_concurrency,
                            job_budget=job_budget,
                            job_context=job_context,
                        )
                        claims_payload = claims_run.get("claims")
                        factuality_payload = claims_run.get("summary")
                        verifications = claims_run.get("verifications", [])
                    # Also store in metadata for debugging/analytics
                    result.metadata["claims"] = claims_payload
                    result.metadata["factuality"] = factuality_payload

                    # Generate verification report if requested
                    if generate_verification_report and verifications:
                        try:
                            from tldw_Server_API.app.core.Claims_Extraction.verification_report import (
                                generate_verification_report as gen_report,
                            )
                            # Use the full VerificationReport class with raw verification objects
                            report = gen_report(
                                verifications=verifications,
                                query=query,
                                answer_text=result.generated_answer,
                                metadata={"source": "unified_pipeline"},
                            )
                            result.metadata["verification_report"] = report.to_dict()
                        except (ImportError, RuntimeError, TypeError, ValueError) as _ereport:
                            logger.debug(f"Verification report generation failed: {_ereport}")
            except (
                AttributeError,
                ConnectionError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
                asyncio.TimeoutError,
            ) as e:
                result.errors.append(f"Claims analysis failed: {str(e)}")
                logger.error(f"Claims analysis error: {e}")

        # ========== NUMERIC FIDELITY (verify numeric tokens) ==========
        try:
            if result.generated_answer and check_numeric_fidelity:
                nf = check_numeric_fidelity(result.generated_answer, result.documents or [])
                if nf:
                    result.metadata.setdefault("numeric_fidelity", {})
                    result.metadata["numeric_fidelity"].update({
                        "present": sorted(nf.present),
                        "missing": sorted(nf.missing),
                        "source_numbers": sorted(nf.union_source_numbers)[:100],
                    })
                    if nf.missing:
                        try:
                            from tldw_Server_API.app.core.Metrics.metrics_manager import increment_counter
                            increment_counter("rag_numeric_mismatches_total", len(nf.missing))
                        except (ImportError, RuntimeError, TypeError, ValueError):
                            pass
                        # Optional corrective action
                        if enable_numeric_fidelity and numeric_fidelity_behavior in {"retry", "ask", "decline"}:
                            if numeric_fidelity_behavior == "retry":
                                # Best-effort: targeted retrieval on missing numbers (bounded)
                                try:
                                    if (
                                        MultiDatabaseRetriever
                                        and RetrievalConfig
                                        and media_db_path
                                        and _scope_includes_data_source(resolved_data_sources, DataSource.MEDIA_DB)
                                    ):
                                        mdr = _build_multi_retriever({"media_db": media_db_path})
                                        conf = RetrievalConfig(max_results=min(10, top_k), min_score=min_score, use_fts=True, use_vector=True, include_metadata=True, fts_level=fts_level, enable_text_late_chunking=enable_text_late_chunking, chunk_method=chunk_method, chunk_size=chunk_size, chunk_overlap=chunk_overlap, chunk_language=chunk_language)
                                        numeric_added: list[Document] = []
                                        for tok in list(nf.missing)[:3]:
                                            try:
                                                numeric_added.extend(await mdr.retrieve(query=f"{query} {tok}", sources=[DataSource.MEDIA_DB], config=conf, index_namespace=index_namespace))
                                            except (
                                                AttributeError,
                                                ConnectionError,
                                                OSError,
                                                RuntimeError,
                                                TypeError,
                                                ValueError,
                                                asyncio.TimeoutError,
                                            ):
                                                continue
                                        if numeric_added:
                                            # Merge with existing docs and optionally re-rerank in place
                                            by_id_numeric: dict[str, Document] = {getattr(d, 'id', ''): d for d in (result.documents or [])}
                                            for d in numeric_added:
                                                cur = by_id_numeric.get(getattr(d, 'id', ''))
                                                if cur is None or float(getattr(d, 'score', 0.0)) > float(getattr(cur, 'score', 0.0)):
                                                    by_id_numeric[getattr(d, 'id', '')] = d
                                            result.documents = sorted(by_id_numeric.values(), key=lambda x: getattr(x, 'score', 0.0), reverse=True)[: max(top_k, 10)]
                                            result.metadata.setdefault("numeric_fidelity", {})
                                            result.metadata["numeric_fidelity"]["retry_docs_added"] = len(numeric_added)
                                            # Attempt quick regeneration if generator is available
                                            if AnswerGenerator:
                                                try:
                                                    generator = AnswerGenerator(
                                                        model=generation_model,
                                                        provider=generation_provider,
                                                    )
                                                    context = "\n\n".join([getattr(d, 'content', str(d)) for d in (result.documents[:5] if result.documents else [])])
                                                    regen = await generator.generate(query=query, context=context, prompt_template=generation_prompt, max_tokens=max_generation_tokens)
                                                    if isinstance(regen, dict) and regen.get("answer"):
                                                        result.generated_answer = regen.get("answer")
                                                    elif isinstance(regen, str):
                                                        result.generated_answer = regen
                                                except (
                                                    AttributeError,
                                                    ConnectionError,
                                                    OSError,
                                                    RuntimeError,
                                                    TypeError,
                                                    ValueError,
                                                    asyncio.TimeoutError,
                                                ):
                                                    pass
                                except (
                                    AttributeError,
                                    ConnectionError,
                                    OSError,
                                    RuntimeError,
                                    TypeError,
                                    ValueError,
                                    asyncio.TimeoutError,
                                ):
                                    pass
                            elif numeric_fidelity_behavior == "ask":
                                note = "\n\n[Note] Some numeric values could not be verified against sources. Please clarify or provide references."
                                result.generated_answer = (result.generated_answer or "") + note
                            elif numeric_fidelity_behavior == "decline":
                                result.generated_answer = "Insufficient evidence to verify numeric claims in the current context."
        except (AttributeError, RuntimeError, TypeError, ValueError) as e:
            result.errors.append(f"Numeric fidelity check failed: {str(e)}")

        # ========== POST-GENERATION VERIFICATION (ADAPTIVE) ==========
        # May run even if enable_claims was False; uses existing results if available.
        try:
            # Allow env defaults if parameters not explicitly set
            if adaptive_time_budget_sec is None:
                try:
                    import os
                    adaptive_time_budget_sec = float(os.getenv("RAG_ADAPTIVE_TIME_BUDGET_SEC", "0")) or None
                except (TypeError, ValueError):
                    adaptive_time_budget_sec = None
            if enable_post_verification and result.generated_answer and PostGenerationVerifier:
                verifier = PostGenerationVerifier(
                    max_retries=adaptive_max_retries,
                    unsupported_threshold=adaptive_unsupported_threshold,
                    max_claims=adaptive_max_claims,
                    time_budget_sec=adaptive_time_budget_sec,
                    use_advanced_rewrites=adaptive_advanced_rewrites,
                )
                vres = await verifier.verify_and_maybe_fix(
                    query=query,
                    answer=result.generated_answer,
                    base_documents=result.documents or [],
                    media_db_path=media_db_path,
                    notes_db_path=notes_db_path,
                    character_db_path=character_db_path,
                    user_id=user_id,
                    generation_model=generation_model,
                    generation_provider=generation_provider,
                    existing_claims=claims_payload,
                    existing_summary=factuality_payload,
                    search_mode=search_mode,
                    hybrid_alpha=hybrid_alpha,
                    top_k=top_k,
                )
                # Attach verification metadata
                result.metadata.setdefault("post_verification", {})
                result.metadata["post_verification"].update({
                    "unsupported_ratio": vres.unsupported_ratio,
                    "total_claims": vres.total_claims,
                    "unsupported_count": vres.unsupported_count,
                    "fixed": vres.fixed,
                    "reason": vres.reason,
                })
                # Gauge for NLI unsupported ratio
                try:
                    from tldw_Server_API.app.core.Metrics.metrics_manager import set_gauge
                    set_gauge("rag_nli_unsupported_ratio", float(vres.unsupported_ratio or 0.0), labels={"strategy": "standard"})
                except (ImportError, RuntimeError, TypeError, ValueError):
                    pass
                # Optionally override final answer on successful repair
                if vres.fixed and vres.new_answer:
                    result.generated_answer = vres.new_answer
                # Behavior toggles on low confidence and not fixed
                low_confidence = (vres.unsupported_ratio > adaptive_unsupported_threshold) and (not vres.fixed)
                if low_confidence:
                    _apply_generation_gate(
                        "nli_low_confidence",
                        unsupported_ratio=vres.unsupported_ratio,
                        threshold=adaptive_unsupported_threshold,
                    )
                    try:
                        from tldw_Server_API.app.core.Metrics.metrics_manager import increment_counter
                        increment_counter("rag_nli_low_confidence_total", 1)
                    except (ImportError, RuntimeError, TypeError, ValueError):
                        pass
                    if low_confidence_behavior == "ask":
                        note = "\n\n[Note] Evidence is insufficient; please clarify or provide more context."
                        result.generated_answer = (result.generated_answer or "") + note
                    elif low_confidence_behavior == "decline":
                        result.generated_answer = "Insufficient evidence found to answer confidently."
                # Sample payload exemplar on failure for debugging (redacted)
                try:
                    if low_confidence:
                        from .payload_exemplars import maybe_record_exemplar
                        maybe_record_exemplar(
                            query=query,
                            documents=result.documents or [],
                            answer=result.generated_answer or "",
                            reason="post_verification_low_confidence",
                            user_id=user_id,
                            namespace=index_namespace,
                        )
                except (ImportError, RuntimeError, TypeError, ValueError):
                    pass

                # Optional: bounded full pipeline rerun to seek a better answer
                try:
                    if low_confidence and adaptive_rerun_on_low_confidence and not _adaptive_rerun:
                        rerun_start = time.time()
                        # Prefer to broaden recall on rerun
                        rerun_expand = expand_query if expand_query else True
                        # Build rerun with post-verification off and a guard to prevent nesting
                        new_result = await unified_rag_pipeline(
                            query=query,
                            sources=sources,
                            media_db_path=media_db_path,
                            notes_db_path=notes_db_path,
                            character_db_path=character_db_path,
                            kanban_db_path=kanban_db_path,
                            # Prefer adapters to avoid raw SQL in prod
                            media_db=media_db,
                            chacha_db=chacha_db,
                            # Use same retrieval/reranking settings but broaden expansion
                            search_mode=search_mode,
                            fts_level=fts_level,
                            hybrid_alpha=hybrid_alpha,
                            top_k=top_k,
                            min_score=min_score,
                            expand_query=rerun_expand,
                            expansion_strategies=expansion_strategies,
                            spell_check=spell_check,
                            enable_cache=False if adaptive_rerun_bypass_cache else enable_cache,
                            cache_threshold=cache_threshold,
                            adaptive_cache=adaptive_cache,
                            keyword_filter=keyword_filter,
                            include_media_ids=include_media_ids,
                            include_note_ids=include_note_ids,
                            enable_security_filter=enable_security_filter,
                            detect_pii=detect_pii,
                            redact_pii=redact_pii,
                            sensitivity_level=sensitivity_level,
                            content_filter=content_filter,
                            enable_table_processing=enable_table_processing,
                            table_method=table_method,
                            enable_vlm_late_chunking=enable_vlm_late_chunking,
                            vlm_backend=vlm_backend,
                            vlm_detect_tables_only=vlm_detect_tables_only,
                            vlm_max_pages=vlm_max_pages,
                            vlm_late_chunk_top_k_docs=vlm_late_chunk_top_k_docs,
                            enable_enhanced_chunking=enable_enhanced_chunking,
                            chunk_type_filter=chunk_type_filter,
                            enable_parent_expansion=enable_parent_expansion,
                            parent_context_size=parent_context_size,
                            include_sibling_chunks=include_sibling_chunks,
                            sibling_window=sibling_window,
                            include_parent_document=include_parent_document,
                            parent_max_tokens=parent_max_tokens,
                            enable_reranking=enable_reranking,
                            reranking_strategy=reranking_strategy,
                            rerank_top_k=rerank_top_k,
                            reranking_model=reranking_model,
                            rerank_min_relevance_prob=rerank_min_relevance_prob,
                            rerank_sentinel_margin=rerank_sentinel_margin,
                            enable_citations=enable_citations,
                            citation_style=citation_style,
                            include_page_numbers=include_page_numbers,
                            enable_chunk_citations=enable_chunk_citations,
                            enable_generation=bool(adaptive_rerun_include_generation),
                            generation_model=generation_model,
                            generation_provider=generation_provider,
                            generation_prompt=generation_prompt,
                            max_generation_tokens=max_generation_tokens,
                            # Disable post-verification in rerun to avoid loops
                            enable_post_verification=False,
                            # Guard: mark this as an adaptive rerun
                            _adaptive_rerun=True,
                            # Preserve guardrails & claims defaults
                            enable_injection_filter=enable_injection_filter,
                            injection_filter_strength=injection_filter_strength,
                            require_hard_citations=require_hard_citations,
                            enable_numeric_fidelity=enable_numeric_fidelity,
                            numeric_fidelity_behavior=numeric_fidelity_behavior,
                            enable_claims=False,  # skip claims during rerun to save time
                            index_namespace=index_namespace,
                            user_id=user_id,
                            session_id=session_id,
                            enable_monitoring=enable_monitoring,
                            enable_observability=enable_observability,
                            trace_id=trace_id,
                            enable_performance_analysis=enable_performance_analysis,
                            timeout_seconds=timeout_seconds,
                            highlight_results=highlight_results,
                            highlight_query_terms=highlight_query_terms,
                            track_cost=track_cost,
                            debug_mode=debug_mode,
                            include_rerank_debug_documents=include_rerank_debug_documents,
                        )
                        # Quick verify the new answer without repairs to compare factuality
                        new_ratio = None
                        if PostGenerationVerifier and (new_result.generated_answer or "").strip():
                            v2 = await PostGenerationVerifier(max_retries=0, max_claims=min(10, adaptive_max_claims)).verify_and_maybe_fix(
                                query=query,
                                answer=new_result.generated_answer,
                                base_documents=(new_result.documents[:int(adaptive_rerun_doc_budget)] if (adaptive_rerun_doc_budget and isinstance(adaptive_rerun_doc_budget, int)) else (new_result.documents or [])),
                                media_db_path=media_db_path,
                                notes_db_path=notes_db_path,
                                character_db_path=character_db_path,
                                user_id=user_id,
                                generation_model=generation_model,
                                generation_provider=generation_provider,
                                search_mode=search_mode,
                                hybrid_alpha=hybrid_alpha,
                                top_k=top_k,
                            )
                            new_ratio = v2.unsupported_ratio
                        # Adoption decision with guardrails regression checks
                        adopt = (new_ratio is not None and new_ratio < vres.unsupported_ratio)
                        try:
                            # Numeric fidelity regression check
                            old_nf_missing = None
                            new_nf_missing = None
                            if check_numeric_fidelity and (result.generated_answer or "").strip():
                                old_nf = check_numeric_fidelity(result.generated_answer, result.documents or [])
                                if old_nf:
                                    old_nf_missing = len(old_nf.missing)
                            else:
                                # fallback to existing metadata if available
                                try:
                                    old_nf_missing = len((result.metadata.get("numeric_fidelity") or {}).get("missing", [])) if isinstance(result.metadata, dict) else None
                                except (AttributeError, TypeError, ValueError):
                                    old_nf_missing = None
                            if check_numeric_fidelity and (new_result.generated_answer or "").strip():
                                new_nf = check_numeric_fidelity(new_result.generated_answer, new_result.documents or [])
                                if new_nf:
                                    new_nf_missing = len(new_nf.missing)
                            # Hard citation coverage regression check
                            old_cov = None
                            new_cov = None
                            try:
                                cov_raw = (result.metadata.get("hard_citations") or {}).get("coverage") if isinstance(result.metadata, dict) else None
                                old_cov = float(cov_raw) if cov_raw is not None else None
                            except (TypeError, ValueError):
                                old_cov = None
                            if build_hard_citations and (new_result.generated_answer or "").strip():
                                hc2 = build_hard_citations(new_result.generated_answer, new_result.documents or [], claims_payload=None)
                                if isinstance(hc2, dict):
                                    new_cov = float(hc2.get("coverage") or 0.0)

                            # If both NF counts present, block adoption on regression
                            if adopt and (old_nf_missing is not None and new_nf_missing is not None):
                                if new_nf_missing > old_nf_missing:
                                    adopt = False
                            # If both coverage present, block adoption on regression
                            if adopt and (old_cov is not None and new_cov is not None):
                                if new_cov < old_cov:
                                    adopt = False
                        except (AttributeError, TypeError, ValueError):
                            # On checker failure, keep original adoption decision
                            pass
                        dur = time.time() - rerun_start
                        result.metadata.setdefault("adaptive_rerun", {})
                        result.metadata["adaptive_rerun"].update({
                            "performed": True,
                            "duration": round(dur, 6),
                            "old_ratio": vres.unsupported_ratio,
                            "new_ratio": new_ratio,
                            "adopted": bool(adopt),
                            "bypass_cache": bool(adaptive_rerun_bypass_cache),
                            "old_nf_missing": old_nf_missing if 'old_nf_missing' in locals() else None,
                            "new_nf_missing": new_nf_missing if 'new_nf_missing' in locals() else None,
                            "old_hard_citation_coverage": old_cov if 'old_cov' in locals() else None,
                            "new_hard_citation_coverage": new_cov if 'new_cov' in locals() else None,
                        })
                        # Metrics for rerun
                        try:
                            from tldw_Server_API.app.core.Metrics.metrics_manager import (
                                increment_counter,
                                observe_histogram,
                            )
                            increment_counter("rag_adaptive_rerun_performed_total", 1)
                            if adopt:
                                increment_counter("rag_adaptive_rerun_adopted_total", 1)
                            observe_histogram("rag_adaptive_rerun_duration_seconds", dur, labels={"adopted": "true" if adopt else "false"})
                        except (AttributeError, RuntimeError, TypeError, ValueError):
                            pass
                        # Budget check and metric
                        try:
                            if adaptive_rerun_time_budget_sec is not None and dur > float(adaptive_rerun_time_budget_sec):
                                from tldw_Server_API.app.core.Metrics.metrics_manager import increment_counter
                                increment_counter("rag_phase_budget_exhausted_total", 1, labels={"phase": "adaptive_rerun"})
                                result.metadata["adaptive_rerun"]["budget_exhausted"] = True
                        except (ImportError, RuntimeError, TypeError, ValueError):
                            pass
                        if adopt:
                            # Replace documents, citations and answer with rerun outputs
                            result.documents = new_result.documents
                            result.citations = new_result.citations
                            result.metadata.update(dict((new_result.metadata or {}).items()))
                            result.generated_answer = new_result.generated_answer
                except (
                    AttributeError,
                    ConnectionError,
                    OSError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                    asyncio.TimeoutError,
                ) as _er:
                    result.errors.append(f"Adaptive rerun failed: {str(_er)}")
                    logger.debug(f"Adaptive rerun error: {_er}")
        except (
            AttributeError,
            ConnectionError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            asyncio.TimeoutError,
        ) as e:
            # Non-fatal: log and continue
            result.errors.append(f"Post-verification failed: {str(e)}")
            logger.warning(f"Post-verification error: {e}")

        # ========== POST-GENERATION EVIDENCE CHAINS UPDATE ==========
        # Now that we have the generated answer, rebuild evidence chains with claim extraction
        if enable_evidence_chains and result.generated_answer and result.documents and EvidenceChainBuilder:
            try:
                chain_builder = EvidenceChainBuilder(enable_llm_extraction=True)

                # Rebuild chains with the generated answer for claim extraction
                evidence_chain_result = await chain_builder.build_chains(
                    query=query,
                    documents=result.documents,
                    generated_answer=result.generated_answer,
                )

                if evidence_chain_result:
                    # Update metadata with full chain information
                    chains_data = []
                    for chain in evidence_chain_result.chains:
                        chains_data.append({
                            "hop_count": chain.hop_count,
                            "chain_confidence": chain.chain_confidence,
                            "source_documents": chain.get_source_documents(),
                            "root_claims": chain.root_claims,
                            "nodes_count": len(chain.nodes),
                        })

                    result.metadata["evidence_chains"] = {
                        "total_chains": len(evidence_chain_result.chains),
                        "overall_confidence": evidence_chain_result.overall_confidence,
                        "multi_hop_detected": evidence_chain_result.multi_hop_detected,
                        "total_claims": evidence_chain_result.metadata.get("total_claims", 0),
                        "supported_claims": evidence_chain_result.metadata.get("supported_claims", 0),
                        "chains": chains_data[:5],  # Include top 5 chains
                    }

                    # Optionally include full chain data for debugging
                    if debug_mode:
                        result.metadata["evidence_chains_full"] = evidence_chain_result.to_dict()

            except (
                AttributeError,
                ConnectionError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
                asyncio.TimeoutError,
            ) as e:
                result.errors.append(f"Post-generation evidence chain update failed: {e}")
                logger.debug(f"Evidence chain update error: {e}")

        # ========== UTILITY GRADING (Self-Correcting RAG Stage 6) ==========
        # Rate response usefulness independent of factual grounding
        try:
            if enable_utility_grading and result.generated_answer and grade_utility:
                with otel_span("rag.utility_grading"):
                    ug_result, ug_meta = await grade_utility(
                        query=query,
                        answer=result.generated_answer,
                        provider=utility_grading_provider,
                        model=utility_grading_model,
                        timeout_sec=utility_grading_timeout_sec,
                    )
                    result.metadata["utility_grade"] = ug_meta
        except (
            AttributeError,
            ConnectionError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            asyncio.TimeoutError,
        ) as e:
            result.errors.append(f"Utility grading failed: {str(e)}")

        # ========== USER FEEDBACK ==========
        if collect_feedback:
            feedback_start = time.time()
            try:
                if UnifiedFeedbackSystem:
                    collector = UnifiedFeedbackSystem()
                    result.feedback_id = str(uuid.uuid4())
                    result.metadata["feedback_enabled"] = True

                    # Apply feedback boost if requested
                    if apply_feedback_boost and result.documents:
                        try:
                            if UserPersonalizationStore:
                                store = UserPersonalizationStore(feedback_user_id or user_id)
                                result.documents = store.boost_documents(result.documents, corpus=index_namespace)
                        except ValueError as exc:
                            logger.debug(f"Personalization boost disabled for user_id={feedback_user_id or user_id}: {exc}")
                        except (AttributeError, RuntimeError, TypeError):
                            pass
                    # Record anonymized search analytics
                    with contextlib.suppress(AttributeError, ConnectionError, OSError, RuntimeError, TypeError, ValueError, asyncio.TimeoutError):
                        await collector.record_search(
                            query=query,
                            results_count=len(result.documents or []),
                            cache_hit=bool(result.cache_hit),
                            latency_ms=(time.time() - start_time) * 1000.0,
                        )

                    result.timings["feedback"] = time.time() - feedback_start

            except ImportError:
                result.errors.append("Feedback module not available")
                logger.warning("Feedback requested but module not available")
            except (
                AttributeError,
                ConnectionError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
                asyncio.TimeoutError,
            ) as e:
                result.errors.append(f"Feedback system failed: {str(e)}")
                logger.error(f"Feedback error: {e}")

        # ========== RESULT HIGHLIGHTING ==========
        if highlight_results and result.documents:
            highlight_start = time.time()
            try:
                if highlight_func:
                    highlight_context = SimpleNamespace(
                        config={"highlighting": {"enabled": True}},
                        query=query if highlight_query_terms else "",
                        documents=result.documents,
                    )
                    highlighted = highlight_func(highlight_context)
                    highlighted_context = await highlighted if inspect.isawaitable(highlighted) else highlighted
                    if getattr(highlighted_context, "documents", None) is not None:
                        result.documents = highlighted_context.documents

                    result.timings["highlighting"] = time.time() - highlight_start

            except ImportError:
                result.errors.append("Highlighting module not available")
                logger.warning("Highlighting requested but module not available")

        # ========== COST TRACKING ==========
        if track_cost:
            try:
                if track_llm_cost:
                    # Calculate estimated cost
                    total_tokens = sum(len(doc.content.split()) for doc in result.documents)
                    cost = await track_llm_cost(
                        model=generation_model or "gpt-3.5-turbo",
                        input_tokens=total_tokens,
                        output_tokens=len(result.generated_answer.split()) if result.generated_answer else 0
                    )

                    result.metadata["estimated_cost"] = cost

            except ImportError:
                result.errors.append("Cost tracking module not available")

        _apply_workspace_filtering_to_result()

        # ========== CACHE STORAGE ==========
        if enable_cache and not result.cache_hit and result.documents:
            try:
                # Store in cache for future use
                cache = _get_cache_instance()

                if cache:
                    # Support both async/sync and set/add method names
                    set_fn = getattr(cache, 'set', None) or getattr(cache, 'add', None)
                    if set_fn:
                        cache_payload = {
                            "documents": list(result.documents),
                            "answer": result.generated_answer,
                            "cached": True,
                        }
                        cache_queries = [query]
                        cache_queries.extend(
                            [q for q in (result.expanded_queries or []) if isinstance(q, str)]
                        )
                        seen = set()
                        for cq in cache_queries:
                            if not isinstance(cq, str):
                                continue
                            cq = cq.strip()
                            if not cq or cq in seen:
                                continue
                            seen.add(cq)
                            if asyncio.iscoroutinefunction(set_fn):
                                await set_fn(cq, cache_payload, ttl=cache_ttl)
                            else:
                                set_fn(cq, cache_payload, ttl=cache_ttl)
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as e:
                logger.error(f"Cache storage error: {e}")

        # ========== OBSERVABILITY ==========
        if enable_observability:
            try:
                if Tracer:
                    tracer = Tracer()
                    await tracer.trace(
                        trace_id=trace_id or str(uuid.uuid4()),
                        operation="unified_rag_pipeline",
                        query=query,
                        timings=result.timings,
                        metadata=result.metadata
                    )

            except ImportError:
                result.errors.append("Observability module not available")

        # ========== PERFORMANCE ANALYSIS ==========
        if enable_performance_analysis:
            try:
                if PerformanceMonitor:
                    monitor = PerformanceMonitor()
                    analysis = await monitor.analyze(
                        timings=result.timings,
                        document_count=len(result.documents),
                        cache_hit=result.cache_hit
                    )

                    result.metadata["performance_analysis"] = analysis

            except ImportError:
                result.errors.append("Performance monitor not available")

    except _EarlyReturn:
        pass
    except (
        AttributeError,
        ConnectionError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        asyncio.TimeoutError,
    ) as e:
        result.errors.append(f"Pipeline error: {str(e)}")
        logger.exception("Unified pipeline error: {}", e)
        if fallback_on_error:
            return {
                "query": query,
                "documents": [],
                "answer": "",
                "cached": False,
                "error": str(e),
                "metadata": result.metadata,
                "timings": result.timings,
            }

    finally:
        # Calculate total time
        result.total_time = time.time() - start_time
        result.timings["total"] = result.total_time

        # Finalize metrics if monitoring
        if metrics:
            metrics.total_duration = result.total_time
            metrics.cache_hit = result.cache_hit
            metrics.documents_retrieved = len(result.documents)

            if enable_monitoring:
                try:
                    collector = MetricsCollector()
                    collector.end_query(metrics)
                except (RuntimeError, TypeError, ValueError) as e:
                    logger.error(f"Metrics recording error: {e}")

        # ---- Retrieval quality metrics (optional) ----
        if ground_truth_doc_ids and result.documents:
            try:
                from .retrieval_metrics import evaluate_retrieval as _eval_retrieval
                _retrieved_ids = [
                    str(getattr(d, "id", ""))
                    for d in result.documents
                ]
                _ret_metrics = _eval_retrieval(
                    _retrieved_ids,
                    [str(gid) for gid in ground_truth_doc_ids],
                    k=metrics_k,
                )
                result.metadata["retrieval_metrics"] = _ret_metrics.to_dict()
                # Emit to Prometheus
                try:
                    from tldw_Server_API.app.core.Metrics.metrics_manager import observe_histogram
                    observe_histogram("rag_retrieval_precision", _ret_metrics.precision)
                    observe_histogram("rag_retrieval_recall", _ret_metrics.recall)
                    observe_histogram("rag_retrieval_mrr", _ret_metrics.mrr)
                    observe_histogram("rag_retrieval_ndcg", _ret_metrics.ndcg)
                    if hasattr(_ret_metrics, "f1"):
                        observe_histogram("rag_retrieval_f1", _ret_metrics.f1)
                except (ImportError, AttributeError, TypeError):
                    pass
            except Exception as _rm_err:
                logger.warning(f"Retrieval metrics computation failed: {_rm_err}")
                result.errors.append(f"Retrieval metrics failed: {_rm_err}")

        # ---- Faithfulness evaluation (optional) ----
        if enable_faithfulness_eval and result.generated_answer:
            try:
                from .faithfulness import FaithfulnessEvaluator as _FaithEval
                # Build context from retrieved documents
                _ctx_parts = [
                    getattr(d, "content", "") for d in (result.documents or [])
                ]
                _ctx_text = "\n\n".join(p for p in _ctx_parts if p)
                if _ctx_text:
                    # Attempt to find an LLM callable from kwargs or auto-construct one
                    _llm_obj = kwargs.get("faithfulness_llm")
                    if _llm_obj is None:
                        # Auto-construct an LLM adapter from the pipeline's config
                        try:
                            from tldw_Server_API.app.core.config import load_and_log_configs as _load_cfg
                            from tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib import (
                                analyze as _sgl_analyze,
                            )

                            _f_cfg = _load_cfg() or {}
                            _f_prov = (
                                generation_provider
                                or _f_cfg.get("RAG_DEFAULT_LLM_PROVIDER")
                                or _f_cfg.get("default_api")
                                or "openai"
                            ).strip()
                            _f_model = generation_model or _f_cfg.get("RAG_DEFAULT_LLM_MODEL")

                            class _FaithfulnessLLMAdapter:
                                """Wraps analyze() to satisfy the LLMCallable protocol."""

                                async def generate(self, prompt: str) -> str:
                                    import asyncio as _aio
                                    result_text = await _aio.get_running_loop().run_in_executor(
                                        None,
                                        lambda: _sgl_analyze(
                                            api_name=_f_prov,
                                            input_data="",
                                            custom_prompt_arg=prompt,
                                            model_override=_f_model,
                                        ),
                                    )
                                    return str(result_text) if result_text else ""

                            _llm_obj = _FaithfulnessLLMAdapter()
                        except (ImportError, AttributeError, TypeError) as _auto_err:
                            logger.debug(
                                f"Could not auto-construct faithfulness LLM: {_auto_err}"
                            )

                    if _llm_obj is not None:
                        _faith_eval = _FaithEval(_llm_obj)
                        _faith_result = await _faith_eval.evaluate_detailed(
                            result.generated_answer, _ctx_text
                        )
                        result.metadata["faithfulness"] = _faith_result.to_dict()
                        # Emit faithfulness score to Prometheus
                        try:
                            from tldw_Server_API.app.core.Metrics.metrics_manager import set_gauge
                            _f_score = _faith_result.to_dict().get("faithfulness_score")
                            if _f_score is not None:
                                set_gauge("rag_eval_faithfulness_score", float(_f_score), labels={"dataset": "online"})
                        except (ImportError, AttributeError, TypeError, ValueError):
                            pass
                    else:
                        logger.debug(
                            "Faithfulness eval requested but no LLM available"
                        )
            except Exception as _fe_err:
                logger.warning(f"Faithfulness evaluation failed: {_fe_err}")
                result.errors.append(f"Faithfulness eval failed: {_fe_err}")

        # Debug output if requested
        if debug_mode:
            try:
                _qh = hashlib.md5((query or "").encode("utf-8"), usedforsecurity=False).hexdigest()[:8]
                logger.debug(f"Query hash={_qh} len={len(query or '')}")
            except (AttributeError, TypeError, ValueError):
                logger.debug("Received query (hash unavailable)")
            logger.debug(f"Documents found: {len(result.documents)}")
            logger.debug(f"Cache hit: {result.cache_hit}")
            logger.debug(f"Timings: {result.timings}")
            logger.debug(f"Errors: {result.errors}")

    if not standard_evidence_coordinated:
        result = coordinate_standard_result_evidence(
            result,
            resolved_request,
            retrieval_plan=retrieval_plan,
        )

    # Convert to Pydantic response
    try:
        from tldw_Server_API.app.api.v1.schemas.rag_schemas_unified import UnifiedRAGResponse
        doc_dicts = [_serialize_result_document(d) for d in (result.documents or [])]
        return UnifiedRAGResponse(
            documents=doc_dicts,
            query=result.query,
            expanded_queries=result.expanded_queries,
            metadata=result.metadata,
            timings=result.timings,
            citations=result.citations,
            academic_citations=(result.metadata or {}).get("academic_citations", []),
            chunk_citations=(result.metadata or {}).get("chunk_citations", []),
            generated_answer=result.generated_answer,
            cache_hit=result.cache_hit,
            errors=result.errors,
            security_report=result.security_report,
            total_time=result.total_time,
            claims=claims_payload,
            factuality=factuality_payload,
        )
    except (ImportError, TypeError, ValueError):
        # Fallback: return a minimal dict if Pydantic is not available
        return {
            "documents": [_serialize_result_document(d) for d in (result.documents or [])],
            "query": result.query,
            "expanded_queries": result.expanded_queries,
            "metadata": result.metadata,
            "timings": result.timings,
            "citations": result.citations,
            "generated_answer": result.generated_answer,
            "cache_hit": result.cache_hit,
            "errors": result.errors,
            "security_report": result.security_report,
            "total_time": result.total_time,
            "claims": claims_payload,
            "factuality": factuality_payload,
        }




# ========== BATCH PROCESSING WRAPPER ==========
async def unified_batch_pipeline(
    queries: list[str],
    max_concurrent: int = 5,
    on_progress: Optional[Callable[[int, int], Any]] = None,
    on_query_done: Optional[Callable[[int, str, Optional[UnifiedPipelineResult], Optional[BaseException]], Any]] = None,
    query_indices: Optional[list[int]] = None,
    **kwargs
) -> list[UnifiedPipelineResult]:
    """
    Process multiple queries concurrently using the unified pipeline.

    Args:
        queries: List of queries to process
        max_concurrent: Maximum concurrent executions
        on_progress: Optional callback(completed, total) called after each query completes.
            Can be used to save checkpoint progress incrementally.
        on_query_done: Optional callback(index, query, result, error) fired when each query
            completes. ``index`` refers to the original query index if ``query_indices`` is provided,
            otherwise it is the local index in ``queries``.
        query_indices: Optional list of original query indices, parallel to ``queries``.
        **kwargs: All parameters supported by unified_rag_pipeline_core

    Returns:
        List of results in the same order as queries
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    # Lightweight normalizer to dedupe/cluster identical queries
    def _normalize(q: str) -> str:
        try:
            q = (q or "").strip().lower()
            out = []
            prev_space = False
            for ch in q:
                if ch.isalnum():
                    out.append(ch)
                    prev_space = False
                else:
                    if not prev_space:
                        out.append(" ")
                        prev_space = True
            return "".join(out).strip()
        except (AttributeError, TypeError):
            return q or ""

    # Group indices by normalized query (identicals)
    normalized_map: dict[str, list[int]] = {}
    for idx, q in enumerate(queries or []):
        normalized_map.setdefault(_normalize(q), []).append(idx)

    # Deduped representatives (first occurrence of each normalized key)
    unique_keys = list(normalized_map.keys())
    rep_texts = [queries[normalized_map[k][0]] for k in unique_keys]

    # Near-duplicate clustering via cosine similarity of embeddings (best-effort)
    clusters: dict[int, list[int]] = {}
    import os as _os
    _disable_cluster = _shared_is_truthy(_os.getenv("RAG_BATCH_DISABLE_CLUSTERING", ""))
    _test_mode = _shared_is_test_mode()
    if _disable_cluster or _test_mode:
        clusters = {i: [i] for i in range(len(unique_keys))}
    else:
        try:
            from tldw_Server_API.app.core.Embeddings.Embeddings_Server.Embeddings_Create import (
                create_embeddings_batch,
                get_embedding_config,
            )
            # Get embeddings for representative texts
            cfg = get_embedding_config()
            vectors = await asyncio.get_running_loop().run_in_executor(
                None,
                create_embeddings_batch,
                rep_texts,
                cfg,
                None,
            )
            # Normalize vectors to unit length for cosine
            def _norm(v):
                try:
                    import math
                    if hasattr(v, 'tolist'):
                        v = v.tolist()
                    s = math.sqrt(sum((float(x) or 0.0) ** 2 for x in v))
                    if s > 0:
                        return [float(x) / s for x in v]
                except (TypeError, ValueError):
                    pass
                return v
            vecs = [_norm(v) for v in (vectors or [])]
            # Cosine similarity
            def _cos(a, b):
                try:
                    return float(sum((ai * bi) for ai, bi in zip(a, b)))
                except (TypeError, ValueError):
                    return 0.0
            # Threshold from env or default 0.9
            try:
                thr = float(_os.getenv('RAG_BATCH_NEAR_DUP_THRESHOLD', '0.9'))
            except (TypeError, ValueError):
                thr = 0.9
            used = set()
            for i, vi in enumerate(vecs):
                if i in used:
                    continue
                clusters[i] = [i]
                used.add(i)
                for j in range(i + 1, len(vecs)):
                    if j in used:
                        continue
                    vj = vecs[j]
                    if not isinstance(vi, list) or not isinstance(vj, list):
                        continue
                    if _cos(vi, vj) >= thr:
                        clusters[i].append(j)
                        used.add(j)
        except Exception as exc:  # noqa: BLE001 - best-effort clustering; never fail batch
            logger.warning(f"Batch query clustering disabled due to error: {exc}")
            # Fallback: each unique becomes its own cluster
            clusters = {i: [i] for i in range(len(unique_keys))}

    # Map cluster head index -> representative query text
    heads = list(clusters.keys())
    head_queries = [rep_texts[h] for h in heads]
    base_retrieval_plan = kwargs.get("retrieval_plan")
    base_resolved_request = kwargs.get("resolved_request")

    def _pipeline_kwargs_for_query(query_text: str) -> dict[str, Any]:
        effective_kwargs = dict(kwargs)
        if isinstance(base_retrieval_plan, RetrievalPlan):
            effective_kwargs["retrieval_plan"] = replace(
                base_retrieval_plan,
                query=query_text,
            )
        if isinstance(base_resolved_request, ResolvedRAGRequest):
            payload = dict(base_resolved_request.payload or {})
            payload["query"] = query_text
            effective_kwargs["resolved_request"] = replace(
                base_resolved_request,
                query=query_text,
                payload=payload,
            )
        return effective_kwargs

    # Run head queries via batch_utils for concurrency control and fail-fast
    head_results: list[Any] = []
    if on_query_done is not None:
        if query_indices is not None and len(query_indices) != len(queries):
            logger.warning(
                "unified_batch_pipeline: query_indices length mismatch ({} != {}); ignoring mapping",
                len(query_indices),
                len(queries),
            )
            query_indices = None

        async def _process_head(index: int, query: str) -> UnifiedPipelineResult:
            async with semaphore:
                return await unified_rag_pipeline(
                    query=query,
                    **_pipeline_kwargs_for_query(query),
                )

        head_results = [RuntimeError("Missing result")] * len(head_queries)
        tasks: dict[asyncio.Task[UnifiedPipelineResult], int] = {}

        progress_total = len(queries)
        progress_count = 0
        progress_lock = asyncio.Lock()

        async def _notify_progress(delta: int = 1) -> None:
            nonlocal progress_count
            if not on_progress:
                return
            async with progress_lock:
                progress_count += delta
                current = progress_count
            try:
                callback_result = on_progress(current, progress_total)
                if asyncio.iscoroutine(callback_result):
                    await callback_result
            except Exception as cb_err:  # noqa: BLE001 - callbacks must not fail pipeline
                logger.warning(f"Batch on_progress callback failed: {cb_err}")

        def _resolve_index(local_idx: int) -> int:
            if query_indices is not None and local_idx < len(query_indices):
                return query_indices[local_idx]
            return local_idx

        for idx, q in enumerate(head_queries):
            task = asyncio.create_task(_process_head(idx, q))
            tasks[task] = idx

        pending = set(tasks.keys())
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                head_idx = tasks.get(task)
                if head_idx is None:
                    continue
                result: Optional[UnifiedPipelineResult] = None
                err: Optional[BaseException] = None
                try:
                    result = await task
                    head_results[head_idx] = result
                except BaseException as exc:  # noqa: BLE001 - surface as error
                    err = exc
                    head_results[head_idx] = exc

                head_key = heads[head_idx]
                members = clusters.get(head_key, [])
                for i_uq in members:
                    orig_indices = normalized_map.get(unique_keys[i_uq], [])
                    for local_idx in orig_indices:
                        global_idx = _resolve_index(local_idx)
                        query_text = queries[local_idx] if local_idx < len(queries) else ""
                        if on_query_done:
                            try:
                                callback_result = on_query_done(global_idx, query_text, result, err)
                                if asyncio.iscoroutine(callback_result):
                                    await callback_result
                            except Exception as cb_err:  # noqa: BLE001 - callbacks must not fail pipeline
                                logger.warning(f"Batch on_query_done callback failed: {cb_err}")
                        await _notify_progress(1)
    else:
        try:
            from .batch_utils import run_batch as _run_batch

            async def _process_head(query: str) -> UnifiedPipelineResult:
                return await unified_rag_pipeline(
                    query=query,
                    **_pipeline_kwargs_for_query(query),
                )

            _batch_result = await _run_batch(
                items=head_queries,
                func=_process_head,
                max_concurrency=max_concurrent,
                on_progress=on_progress,
            )
            # Reconstruct list in original order with errors slotted in
            head_results = _batch_result.ordered_results_with_errors(
                default=RuntimeError("Missing result"),
            )
        except ImportError:
            # Fallback to inline semaphore if batch_utils unavailable
            async def process_with_semaphore(query: str) -> UnifiedPipelineResult:
                async with semaphore:
                    return await unified_rag_pipeline(
                        query=query,
                        **_pipeline_kwargs_for_query(query),
                    )

            tasks = [process_with_semaphore(q) for q in head_queries]
            head_results = await asyncio.gather(*tasks, return_exceptions=True)

    # Build final results in original order, reusing unique results
    final_results: list[Optional[UnifiedPipelineResult]] = [None] * len(queries)
    reuse_count = 0
    # Build mapping from unique key index -> head result
    # unique_keys[i] corresponds to rep_texts[i]
    # Find which head each i belongs to
    head_for: dict[int, int] = {}
    for h, members in clusters.items():
        for m in members:
            head_for[m] = h
    # Stitch results
    for i_uq, key in enumerate(unique_keys):
        # Find the head index for this unique
        h = head_for.get(i_uq, i_uq)
        ures = head_results[heads.index(h)] if h in heads else head_results[0]
        indices = normalized_map.get(key, [])
        for pos, i in enumerate(indices):
            if isinstance(ures, BaseException):
                final_results[i] = UnifiedSearchResult(documents=[], query=queries[i], errors=[str(ures)])
            else:
                reuse_count += 1 if pos > 0 else 0
                ures_any = cast(Any, ures)
                # Copy minimal fields for non-heads to preserve original query text
                final_results[i] = (
                    ures_any if pos == 0 and queries[i] == rep_texts[i_uq]
                    else UnifiedSearchResult(
                        documents=ures_any.documents,
                        query=queries[i],
                        expanded_queries=ures_any.expanded_queries,
                        metadata=ures_any.metadata,
                        timings=ures_any.timings,
                        citations=ures_any.citations,
                        feedback_id=ures_any.feedback_id,
                        generated_answer=ures_any.generated_answer,
                        cache_hit=ures_any.cache_hit,
                        errors=ures_any.errors,
                        security_report=ures_any.security_report,
                        total_time=ures_any.total_time,
                    )
                )

    # Metrics: record reuse count
    try:
        if reuse_count > 0:
            from tldw_Server_API.app.core.Metrics.metrics_manager import increment_counter
            increment_counter("rag_batch_query_reuse_total", reuse_count)
    except (ImportError, RuntimeError, TypeError, ValueError):
        pass

    for i, res in enumerate(final_results):
        if res is None:
            final_results[i] = UnifiedSearchResult(
                documents=[],
                query=queries[i] if i < len(queries) else "",
                errors=["Missing batch result"],
            )
    return cast(list[UnifiedPipelineResult], final_results)


# ========== SIMPLE CONVENIENCE WRAPPERS ==========

async def simple_search(
    query: str,
    top_k: int = 10,
    *,
    sources: Optional[list[str]] = None,
    media_db: Any = None,
    chacha_db: Any = None,
    media_db_path: Optional[str] = None,
    notes_db_path: Optional[str] = None,
    character_db_path: Optional[str] = None,
    kanban_db_path: Optional[str] = None,
    user_id: Optional[str] = None,
) -> list[Document]:
    """
    Simple search wrapper for basic use cases.

    Args:
        query: Search query
        top_k: Number of results

    Returns:
        List of documents
    """
    result = await unified_rag_pipeline(
        query=query,
        top_k=top_k,
        expand_query=False,
        enable_cache=True,
        enable_reranking=True,
        sources=sources,
        media_db=media_db,
        chacha_db=chacha_db,
        media_db_path=media_db_path,
        notes_db_path=notes_db_path,
        character_db_path=character_db_path,
        kanban_db_path=kanban_db_path,
        user_id=user_id,
    )
    if isinstance(result, UnifiedSearchResult):
        return result.documents
    docs = getattr(result, "documents", None)
    if isinstance(docs, list) and all(isinstance(d, Document) for d in docs):
        return docs
    return []


async def advanced_search(
    query: str,
    with_citations: bool = True,
    with_answer: bool = True,
    **kwargs
) -> UnifiedPipelineResult:
    """
    Advanced search with commonly used features enabled.

    Args:
        query: Search query
        with_citations: Enable citation generation
        with_answer: Enable answer generation
        **kwargs: Additional parameters

    Returns:
        Full search result
    """
    return await unified_rag_pipeline(
        query=query,
        expand_query=True,
        expansion_strategies=["acronym", "synonym", "domain"],
        enable_cache=True,
        enable_reranking=True,
        reranking_strategy="hybrid",
        enable_citations=with_citations,
        enable_generation=with_answer,
        enable_table_processing=True,
        enable_performance_analysis=True,
        **kwargs
    )
def compute_temporal_range_from_query(query: str) -> Optional[dict[str, str]]:
    """Compute an approximate temporal range from a natural language query.

    Returns dict with ISO start/end if a range can be inferred; otherwise None.
    Conservative default: last 7 days if common patterns not found.
    """
    try:
        qlower = (query or "").lower()
        start_dt = None
        end_dt = None
        now = datetime.utcnow()
        if "yesterday" in qlower:
            start_dt = now - timedelta(days=1)
            end_dt = now
        elif "last week" in qlower or "past week" in qlower:
            start_dt = now - timedelta(days=7)
            end_dt = now
        elif "last month" in qlower:
            y = now.year
            m = now.month - 1 if now.month > 1 else 12
            y = y if now.month > 1 else y - 1
            start_dt = datetime(y, m, 1)
            _, last_day = calendar.monthrange(y, m)
            end_dt = datetime(y, m, last_day, 23, 59, 59)
        elif "past month" in qlower:
            start_dt = now - timedelta(days=30)
            end_dt = now
        m_quarter = re.search(r"\bq([1-4])\s*(20\d{2}|19\d{2})\b", qlower)
        if m_quarter:
            qn = int(m_quarter.group(1))
            y = int(m_quarter.group(2))
            qm = {1:1,2:4,3:7,4:10}[qn]
            start_dt = datetime(y, qm, 1)
            end_month = qm + 2
            _, last_day = calendar.monthrange(y, end_month)
            end_dt = datetime(y, end_month, last_day, 23, 59, 59)
        month_names = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
        m_month_year = re.search(r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+(20\d{2}|19\d{2})\b", qlower)
        if m_month_year:
            mon = month_names.get(m_month_year.group(1))
            y = int(m_month_year.group(2))
            if mon:
                start_dt = datetime(y, mon, 1)
                _, last_day = calendar.monthrange(y, mon)
                end_dt = datetime(y, mon, last_day, 23, 59, 59)
        m_year = re.search(r"\b(20\d{2}|19\d{2})\b", qlower)
        if m_year and start_dt is None and end_dt is None:
            y = int(m_year.group(1))
            start_dt = datetime(y,1,1)
            end_dt = datetime(y,12,31,23,59,59)
        if start_dt is None and end_dt is None:
            start_dt = now - timedelta(days=7)
            end_dt = now
        if start_dt is None or end_dt is None:
            return None
        return {"start": start_dt.isoformat(), "end": end_dt.isoformat()}
    except (AttributeError, TypeError, ValueError):
        return None
