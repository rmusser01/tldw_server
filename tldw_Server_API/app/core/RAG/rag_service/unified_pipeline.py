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
import copy
import hashlib
import inspect
import json
import math
import os
import re
import sqlite3
import time
import uuid
from collections import Counter  # noqa: F401 - Counter now read by pipeline/scope.py via its own import
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from functools import partial
from numbers import Real
from types import SimpleNamespace  # noqa: F401  # no longer read here; kept so rag_split verify sees the same top-level imports as its base
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional, Protocol, TypedDict, cast, overload

from loguru import logger

from tldw_Server_API.app.core.Chat.bounded_daemon import (
    SYNC_ADAPTER_CALL_POOL,  # noqa: F401 - patched here; pipeline/generation.py reads up.SYNC_ADAPTER_CALL_POOL
    await_bounded_sync_call,  # noqa: F401 - patched here; pipeline/generation.py reads up.await_bounded_sync_call
    await_owned_worker,  # noqa: F401 - patched here; pipeline/claims.py reads up.await_owned_worker
)
from tldw_Server_API.app.core.Chat.Chat_Deps import ChatAPIError
from tldw_Server_API.app.core.DB_Management.media_db.api import (
    managed_media_database,  # noqa: F401 - patched here; pipeline/claims.py reads up.managed_media_database
)
from tldw_Server_API.app.core.DB_Management.scope_context import (
    ScopeContext,
    content_authorization_cache_scope,
    get_scope,
)
from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError  # noqa: F401 - patched here; pipeline/augment.py reads up.LLMQuotaExceededError
from tldw_Server_API.app.core.LLM_Calls.structured_output import (
    StructuredOutputOptions,  # noqa: F401 - patched here; pipeline/augment.py reads up.StructuredOutputOptions
    StructuredOutputParseError,  # noqa: F401 - patched here; pipeline/augment.py reads up.StructuredOutputParseError
    parse_structured_output,  # noqa: F401 - patched here; pipeline/augment.py reads up.parse_structured_output
)
from tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib import (
    SummaryProviderError,
)
from tldw_Server_API.app.core.Provider_Credentials.byok_runtime import ByokResolutionError
from tldw_Server_API.app.core.RAG.exceptions import RAGConfigurationError
from tldw_Server_API.app.core.testing import (
    is_production_like_env as _shared_is_production_like_env,  # noqa: F401 - read as up._shared_is_production_like_env by pipeline/query_prep.py
    is_test_mode as _shared_is_test_mode,
    is_truthy as _shared_is_truthy,
)

from .generation_executor import execute_generation_phase  # noqa: F401  # read as up.execute_generation_phase
from .hyde import (
    _embedding_model_from_config,
    _embedding_provider_from_config,
    _mark_runtime_used_for_embeddings,
    _resolve_runtime_embedding_call,
    _run_sync_embedding_call,
    _runtime_local_embedding_call_kwargs,
)
from .retrieval_executor import execute_retrieval_phase  # noqa: F401 - patched here; pipeline/retrieval.py reads up.execute_retrieval_phase
from .retrieval_plan import RetrievalPlan

_RERANK_DEBUG_DOCUMENT_CONTENT_MAX_CHARS = 500
_RAG_PROVIDER_FAILURES = (ByokResolutionError, ChatAPIError)


class _ProviderFailureBypass(BaseException):
    """Carry a typed provider failure across retry/circuit accounting unchanged."""

    __slots__ = ("failure",)

    def __init__(self, failure: BaseException) -> None:
        self.failure = failure
        super().__init__()


def _bounded_provider_failure_code(exc: BaseException) -> str:
    """Map a typed provider failure to an allowlisted metadata code."""
    allowed = {
        "invalid_provider_credentials",
        "missing_provider_credentials",
        "provider_configuration_invalid",
        "credential_store_unavailable",
        "credential_scope_revoked",
    }
    code = str(getattr(exc, "code", "") or getattr(exc, "error_code", "") or "")
    if code in allowed:
        return code
    if getattr(exc, "status_code", None) in {401, 403}:
        return "invalid_provider_credentials"
    return "provider_unavailable"


def _serialize_result_document(doc: Any) -> dict[str, Any]:
    """Serialize a pipeline document into the response-compatible document shape."""
    if isinstance(doc, dict):
        metadata = dict(doc.get("metadata") or {})
        source = doc.get("source")
        serialized_source = None
        if source is not None:
            serialized_source = source.value if hasattr(source, "value") else str(source)
            metadata["source"] = serialized_source
        for field_name in (
            "media_id",
            "note_id",
            "chunk_id",
            "record_id",
            "chunk_index",
            "start",
            "end",
            "start_char",
            "end_char",
        ):
            value = doc.get(field_name)
            if value is not None:
                metadata.setdefault(field_name, value)
        doc_id = doc.get("id")
        if doc_id is not None:
            metadata.setdefault("chunk_id", str(doc_id))
        serialized = {
            "id": doc_id,
            "content": doc.get("content") or doc.get("text") or doc.get("body"),
            "score": doc.get("score", 0.0),
            "metadata": metadata,
        }
        if serialized_source is not None:
            serialized["source"] = serialized_source
        return serialized

    metadata = dict(getattr(doc, "metadata", {}) or {})
    serialized_source = None
    try:
        source = getattr(doc, "source", None)
        if source is not None:
            serialized_source = source.value if hasattr(source, "value") else str(source)
            metadata["source"] = serialized_source
    except (AttributeError, TypeError, ValueError):
        pass

    doc_id = getattr(doc, "id", None)
    if doc_id is not None:
        metadata.setdefault("chunk_id", str(doc_id))
    for field_name in ("chunk_index", "start_char", "end_char"):
        value = getattr(doc, field_name, None)
        if value is not None:
            metadata.setdefault(field_name, value)

    serialized = {
        "id": doc_id,
        "content": getattr(doc, "content", None),
        "score": getattr(doc, "score", 0.0),
        "metadata": metadata,
    }
    if serialized_source is not None:
        serialized["source"] = serialized_source
    return serialized


@overload
def _clone_cached_document(doc: "Document") -> "Document": ...


@overload
def _clone_cached_document(doc: Any) -> Any: ...


def _clone_cached_document(doc: Any) -> Any:
    """Return an isolated cache document without duplicating embedding buffers."""
    if isinstance(doc, Document):
        return replace(
            doc,
            metadata=copy.deepcopy(doc.metadata),
            citations=copy.deepcopy(doc.citations),
            source_document_metadata=copy.deepcopy(doc.source_document_metadata),
            children_ids=list(doc.children_ids),
            embedding=doc.embedding,
        )

    if isinstance(doc, dict):
        cloned = dict(doc)
        for field_name in ("metadata", "citations", "source_document_metadata"):
            if field_name in cloned:
                cloned[field_name] = copy.deepcopy(cloned[field_name])
        if isinstance(cloned.get("children_ids"), list):
            cloned["children_ids"] = list(cloned["children_ids"])
        return cloned

    return copy.deepcopy(doc)


def _clone_cached_documents(documents: list[Any]) -> list[Any]:
    """Return independent document instances for cache store/load boundaries."""
    return [_clone_cached_document(doc) for doc in documents]


def _restore_cached_document(document: Any) -> Optional["Document"]:
    """Restore one supported cache wire document to the pipeline document type."""
    if isinstance(document, Document):
        return _clone_cached_document(document)
    if not isinstance(document, dict):
        return None
    content = document.get("content")
    document_id = document.get("id")
    if not isinstance(content, str) or not isinstance(document_id, (str, int)):
        return None
    metadata = copy.deepcopy(document.get("metadata") or {})
    if not isinstance(metadata, dict):
        metadata = {}
    source_value = document.get("source", metadata.get("source"))
    if isinstance(source_value, DataSource):
        source = source_value
    else:
        try:
            source = DataSource(str(source_value))
        except (TypeError, ValueError):
            source = DataSource.MEDIA_DB
    try:
        score = float(document.get("score", 0.0) or 0.0)
    except (TypeError, ValueError):
        score = 0.0
    return Document(
        id=str(document_id),
        content=content,
        metadata=metadata,
        source=source,
        score=score,
    )


def _extract_cached_documents(value: Any) -> list["Document"]:
    """Extract only non-empty retrieval documents from a cache payload."""
    if callable(_sanitize_cache_payload):
        value = _sanitize_cache_payload(value)
        if value is None:
            return []
    if isinstance(value, dict):
        raw_documents = value.get("documents")
    elif isinstance(value, list):
        raw_documents = value
    else:
        return []
    if not isinstance(raw_documents, list) or not raw_documents:
        return []
    return [
        restored
        for document in raw_documents
        if (restored := _restore_cached_document(document)) is not None
    ]


_MAX_CACHED_QUERY_CHARS = 4096


def _validate_semantic_cache_match(value: Any) -> Optional[tuple[str, float]]:
    """Validate and bound the untrusted result returned by ``find_similar``."""
    if not isinstance(value, tuple) or len(value) not in {2, 3}:
        return None
    cached_query, similarity = value[-2:]
    if not isinstance(cached_query, str):
        return None
    cached_query = cached_query.strip()
    if not cached_query or len(cached_query) > _MAX_CACHED_QUERY_CHARS:
        return None
    if isinstance(similarity, bool) or not isinstance(similarity, Real):
        return None
    similarity_value = float(similarity)
    if not math.isfinite(similarity_value) or not 0.0 <= similarity_value <= 1.0:
        return None
    return cached_query, similarity_value


async def _invoke_cache_callable(callable_obj: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Invoke a cache callable once and await wrapped awaitable results."""
    value = callable_obj(*args, **kwargs)
    if inspect.isawaitable(value):
        return await value
    return value


def _cache_identity_digest(value: dict[str, Any]) -> str:
    """Return a deterministic short digest for cache identity components."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def _resolve_security_user_id(user_id: Any, feedback_user_id: Any) -> str:
    """Resolve the best available user identifier for security filtering."""
    for candidate in (user_id, feedback_user_id):
        if candidate is None:
            continue
        value = str(candidate).strip()
        if value:
            return value
    return "anonymous"


def _resolve_include_rerank_debug_documents(explicit_flag: Any) -> bool:
    """Resolve whether rerank debug snapshots are enabled."""
    if explicit_flag is None:
        return _shared_is_truthy(os.getenv("RAG_INCLUDE_RERANK_DEBUG_DOCUMENTS", "false"))
    if isinstance(explicit_flag, bool):
        return explicit_flag
    return _shared_is_truthy(explicit_flag)


def _truncate_rerank_debug_content(
    content: Any, *, max_chars: int = _RERANK_DEBUG_DOCUMENT_CONTENT_MAX_CHARS
) -> str | None:
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
        _serialize_rerank_debug_document(doc, include_content=include_content) for doc in list(documents or [])[:limit]
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
_generate_hypothetical_answer_async: Any = None
_hyde_embed_text: Any = None
_expand_acronyms: Any = None
_expand_synonyms: Any = None
_entity_recognition_expansion: Any = None
_domain_specific_expansion: Any = None
_multi_strategy_expansion: Any = None
_SemanticCache: Any = None
_AdaptiveCache: Any = None
_get_shared_cache: Any = None
_MediaDBRetriever: Any = None
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
_create_preinstalled_local_reranker: Any = None
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

from .evidence_models import RetrievedEvidence
from .metrics_collector import MetricsCollector, QueryMetrics  # noqa: F401 - patched here; pipeline/post_processing.py reads up.MetricsCollector
from .post_retrieval_coordinator import coordinate_standard_result_evidence
from .profiles import (
    SLIDES_SOURCE_PROFILE,
    SLIDES_SOURCE_RERANKING_STRATEGIES,
)
from .request_resolution import (
    ResolvedRAGRequest,
    resolve_legacy_standard_pipeline_request,
)
from .result_model import RAGResult
from .retrieval_plan import build_retrieval_plan
from .types import DataSource, Document
from .pipeline.run_config import RagRunConfig
from .pipeline import run_state as _run_state
from .pipeline import post_processing as _post_processing
from .pipeline import verification as _verification
from .pipeline import claims as _claims
from .pipeline import generation as _generation
from .pipeline import rerank as _rerank
from .pipeline import post_retrieval as _post_retrieval
from .pipeline import processing as _processing
from .pipeline import self_correct as _self_correct
from .pipeline import augment as _augment
from .pipeline import filters as _filters
from .pipeline import retrieval as _retrieval
from .pipeline import scope as _scope
from .pipeline import query_prep as _query_prep
from .pipeline import research as _research

if TYPE_CHECKING:
    from .database_retrievers import (
        ChatHistoryRetriever as ChatHistoryRetrieverType,
        MediaDBRetriever as MediaDBRetrieverType,
        NotesDBRetriever as NotesDBRetrieverType,
    )


class _SlidesSourceProjectionAllocator(Protocol):
    """Describe the optional allocator's keyword-only projection budgets."""

    def __call__(
        self,
        documents: list[Document],
        *,
        total_chars: int,
        separator_chars: int,
    ) -> tuple[list[Document], list[int]]: ...


class _SearchDepthPreset(TypedDict):
    top_k: int
    enable_reranking: bool
    reranking_strategy: Literal["none", "flashrank", "two_tier"]
    expand_query: bool
    enable_hyde: bool
    enable_web_fallback: bool
    enable_multi_vector_passages: bool


_SEARCH_DEPTH_PRESETS: dict[str, _SearchDepthPreset] = {
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


def _consume_bound_sgl_response(
    response: Any,
    provider: str,
    on_content: Callable[[], None] | None = None,
    fail_closed: bool = False,
) -> Any:
    """Consume a bound SGL response and reject its legacy error-string form."""
    if isinstance(response, Iterator):
        from .generation import _classify_stream_content, _extract_stream_text

        chunks: list[str] = []
        for chunk in response:
            chunks.append(str(chunk))
            has_content, has_error = _classify_stream_content(
                _extract_stream_text(chunk)
            )
            if on_content is not None and has_content:
                on_content()
            if fail_closed and has_error:
                raise SummaryProviderError(code="provider_failure", provider=provider)
        response = "".join(chunks)
    if fail_closed and isinstance(response, str) and response.startswith("Error:"):
        raise SummaryProviderError(code="provider_failure", provider=provider)
    return response


_normalize_sources_internal: Callable[[list[str] | None], list[str]] | None
try:
    from tldw_Server_API.app.core.Text2SQL.source_registry import (
        normalize_sources_internal as _imported_normalize_sources_internal,
    )
except ImportError:
    _normalize_sources_internal = None
else:
    _normalize_sources_internal = _imported_normalize_sources_internal

normalize_sources_internal = _normalize_sources_internal

# Import all modules at module level to avoid 500ms overhead
try:
    from .quick_wins import (
        highlight_results as _highlight_results,
        spell_check_query as _spell_check_query,
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
        QueryIntent as _QueryIntent,
        QueryRewriter as _QueryRewriter,
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
        classify_and_reformulate as _classify_and_reformulate,
    )
    from .research_agent import (
        create_default_registry as _create_default_registry,
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
        build_writer_user_prompt as _build_writer_user_prompt,
        format_context_xml as _format_context_xml,
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
        generate_hypothetical_answer as _generate_hypothetical_answer,
        generate_hypothetical_answer_async as _generate_hypothetical_answer_async,
    )
except ImportError:
    _generate_hypothetical_answer = None
    _generate_hypothetical_answer_async = None
    _hyde_embed_text = None

generate_hypothetical_answer = _generate_hypothetical_answer
generate_hypothetical_answer_async = _generate_hypothetical_answer_async
hyde_embed_text = _hyde_embed_text

try:
    from .query_expansion import (
        domain_specific_expansion as _domain_specific_expansion,
        entity_recognition_expansion as _entity_recognition_expansion,
        expand_acronyms as _expand_acronyms,
        expand_synonyms as _expand_synonyms,
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

_sanitize_cache_payload: Callable[[Any], Any | None] | None
try:
    from .semantic_cache import (
        AdaptiveCache as _AdaptiveCache,
        SemanticCache as _SemanticCache,
        _sanitize_retrieval_payload as _imported_sanitize_cache_payload,
        get_shared_cache as _get_shared_cache,
    )
except ImportError:
    _SemanticCache = None
    _AdaptiveCache = None
    _get_shared_cache = None
    _sanitize_cache_payload = None
else:
    _sanitize_cache_payload = _imported_sanitize_cache_payload

SemanticCache = _SemanticCache
AdaptiveCache = _AdaptiveCache
get_shared_cache = _get_shared_cache

_allocate_slides_source_projection_caps_impl: _SlidesSourceProjectionAllocator | None
_ChatHistoryRetriever: type["ChatHistoryRetrieverType"] | None
_NotesDBRetriever: type["NotesDBRetrieverType"] | None
try:
    from .database_retrievers import (
        ChatHistoryRetriever as _ImportedChatHistoryRetriever,
        MediaDBRetriever as _MediaDBRetriever,
        MultiDatabaseRetriever as _MultiDatabaseRetriever,
        NotesDBRetriever as _ImportedNotesDBRetriever,
        RetrievalConfig as _RetrievalConfig,
        SQLRetriever as _SQLRetriever,
        _allocate_slides_source_projection_caps as _imported_slides_source_projection_caps,
    )
except ImportError:
    _allocate_slides_source_projection_caps_impl = None
    _ChatHistoryRetriever = None
    _MediaDBRetriever = None
    _MultiDatabaseRetriever = None
    _NotesDBRetriever = None
    _RetrievalConfig = None
    _SQLRetriever = None
else:
    _allocate_slides_source_projection_caps_impl = _imported_slides_source_projection_caps
    _ChatHistoryRetriever = _ImportedChatHistoryRetriever
    _NotesDBRetriever = _ImportedNotesDBRetriever

ChatHistoryRetriever = _ChatHistoryRetriever
MediaDBRetriever = _MediaDBRetriever
MultiDatabaseRetriever = _MultiDatabaseRetriever
NotesDBRetriever = _NotesDBRetriever
RetrievalConfig = _RetrievalConfig
SQLRetriever = _SQLRetriever
allocate_slides_source_projection_caps = _allocate_slides_source_projection_caps_impl

try:
    from .security_filters import SecurityFilters as _SecurityFilters, SensitivityLevel as _SensitivityLevel
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
        expand_with_parent_context as _expand_with_parent_context,
        filter_chunks_by_type as _filter_chunks_by_type,
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
        RerankingStrategy as _RerankingStrategy,
        create_preinstalled_local_reranker as _create_preinstalled_local_reranker,
        create_reranker as _create_reranker,
    )
except ImportError:
    _create_preinstalled_local_reranker = None
    _create_reranker = None
    _RerankingStrategy = None
    _RerankingConfig = None

create_preinstalled_local_reranker = _create_preinstalled_local_reranker
create_reranker = _create_reranker
RerankingStrategy = _RerankingStrategy
RerankingConfig = _RerankingConfig

# Advanced retrieval (multi-vector passages)
try:
    from .advanced_retrieval import (
        MultiVectorConfig as _MultiVectorConfig,
        apply_multi_vector_passages as _apply_multi_vector_passages,
    )
except ImportError:
    _apply_multi_vector_passages = None
    _MultiVectorConfig = None

apply_multi_vector_passages = _apply_multi_vector_passages
MultiVectorConfig = _MultiVectorConfig

# Pseudo-relevance feedback (PRF)
try:
    from .prf import PRFConfig as _PRFConfig, apply_prf as _apply_prf
except ImportError:
    _PRFConfig = None
    _apply_prf = None

PRFConfig = _PRFConfig
apply_prf = _apply_prf

# Precomputed span index (multi-vector helper)
try:
    from .precomputed_spans import (
        PrecomputedSpanConfig as _PrecomputedSpanConfig,
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
    from tldw_Server_API.app.core.Context_Integrity.prompt_loader import load_prompt as _load_prompt
except ImportError:
    _load_prompt = None


def load_prompt(*args, **kwargs):
    if _load_prompt is None:
        return None
    return _load_prompt(*args, **kwargs)


# Chunking support
try:
    from tldw_Server_API.app.core.Chunking import Chunker as _Chunker, ChunkerConfig as _ChunkerConfig
except ImportError:
    _Chunker = None
    _ChunkerConfig = None

Chunker = _Chunker
ChunkerConfig = _ChunkerConfig
try:
    from .citations import CitationGenerator as _CitationGenerator, CitationStyle as _CitationStyle
except ImportError:
    _CitationGenerator = None
    _CitationStyle = None

CitationGenerator = _CitationGenerator
CitationStyle = _CitationStyle

# Dynamic granularity routing
try:
    from .granularity_router import (
        GranularityDecision as _GranularityDecision,
        GranularityRouter as _GranularityRouter,
        route_query_granularity as _route_query_granularity,
    )
    from .types import Granularity as _Granularity, QueryType as _QueryType
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
        EvidenceAccumulator as _EvidenceAccumulator,
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
        EvidenceChainBuilder as _EvidenceChainBuilder,
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
        GradingConfig as _GradingConfig,
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
        fallback_to_web_search as _fallback_to_web_search,
        merge_web_results as _merge_web_results,
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
        KnowledgeStripsResult as _KnowledgeStripsResult,
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
        FastGroundednessResult as _FastGroundednessResult,
        UtilityGrader as _UtilityGrader,
        UtilityResult as _UtilityResult,
        check_fast_groundedness as _check_fast_groundedness,
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
        rag_require_hard_citations as _rag_req_hc,
    )
except ImportError:
    _rag_low_conf = None
    _rag_req_hc = None

try:
    # Guardrails utilities: injection filtering, numeric fidelity, hard citations
    from .guardrails import (
        apply_content_policy as _apply_content_policy,
        build_hard_citations as _build_hard_citations,
        build_quote_citations as _build_quote_citations,
        check_numeric_fidelity as _check_numeric_fidelity,
        detect_injection_score as _detect_injection_score,
        downweight_injection_docs as _downweight_injection_docs,
        gate_docs_by_ocr_confidence as _gate_docs_by_ocr_confidence,
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
        RetryConfig as _RetryConfig,
        RetryPolicy as _RetryPolicy,
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
        resolve_claims_job_budget as _resolve_claims_job_budget,
    )
except ImportError:
    _ClaimsJobContext = None
    _resolve_claims_job_budget = None

try:
    from .claims import ClaimsEngine as _ClaimsEngine
except ImportError:
    _ClaimsEngine = None

ClaimsEngine = _ClaimsEngine
ClaimsJobContext = _ClaimsJobContext
resolve_claims_job_budget = _resolve_claims_job_budget


def _resolve_claims_engine() -> Any:
    """Recover the real claims engine after an import-order cycle completes."""
    global ClaimsEngine
    if ClaimsEngine is not None:
        return ClaimsEngine
    try:
        from tldw_Server_API.app.core.Claims_Extraction.claims_engine import (
            ClaimsEngine as _RealClaimsEngine,
        )

        from .claims import ClaimsEngine as _CandidateClaimsEngine
    except ImportError:
        return None
    if _CandidateClaimsEngine is _RealClaimsEngine:
        ClaimsEngine = _RealClaimsEngine
    return _CandidateClaimsEngine


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


def _fuse_slides_source_documents(
    source_results: Sequence[Sequence[Document]],
    *,
    source_limits: tuple[int, int, int],
    top_k: int,
) -> list[Document]:
    """Round-robin source-ranked candidates so unlike score scales stay fair."""
    ranked_sources: list[list[Document]] = []
    for documents, limit in zip(source_results, source_limits):
        ranked_sources.append(
            sorted(
                documents,
                key=lambda document: (
                    -float(getattr(document, "score", 0.0)),
                    str(getattr(document, "id", "")),
                ),
            )[:limit]
        )
    fused: list[Document] = []
    rank = 0
    while len(fused) < top_k:
        added = False
        for documents in ranked_sources:
            if rank < len(documents):
                fused.append(documents[rank])
                added = True
                if len(fused) >= top_k:
                    break
        if not added:
            break
        rank += 1
    return fused


def _bound_slides_source_documents(
    documents: list[Any],
    *,
    max_source_chars: int,
    top_k: int,
) -> list[Any]:
    """Copy documents within the exact formatted aggregate max-plus-one budget."""
    marker = "_standalone_source_projection_truncated"
    accumulated = 0
    bounded: list[Any] = []
    for document in documents[:top_k]:
        if accumulated > max_source_chars:
            break
        original_metadata = (
            document.get("metadata") if isinstance(document, dict) else getattr(document, "metadata", None)
        )
        metadata = dict(original_metadata) if isinstance(original_metadata, dict) else {}
        title = metadata.get("title") or metadata.get("source_title")
        if not isinstance(title, str) or not title.strip():
            raw_id = document.get("id") if isinstance(document, dict) else getattr(document, "id", None)
            title = str(raw_id).strip() if raw_id is not None else "source"
        title = title.strip() or "source"
        if isinstance(document, dict):
            original_content = str(document.get("content") or "")
        else:
            original_content = str(getattr(document, "content", "") or "")

        separator = "\n\n" if bounded else ""
        available = max_source_chars + 1 - accumulated
        truncated = bool(metadata.get(marker))
        preformatted = bool(metadata.get("_standalone_source_preformatted"))
        if preformatted:
            fixed_prefix = separator
            fixed_suffix = ""
            projected_title = ""
        else:
            fixed_prefix = f"{separator}# "
            fixed_suffix = "\n\n"
            max_title_chars = max(0, available - len(fixed_prefix) - len(fixed_suffix))
            projected_title = title[:max_title_chars]
            if projected_title != title:
                truncated = True
        available_content = max(
            0,
            available - len(fixed_prefix) - len(projected_title) - len(fixed_suffix),
        )
        projected_content = original_content[:available_content]
        if projected_content != original_content:
            truncated = True

        formatted_size = len(fixed_prefix) + len(projected_title) + len(fixed_suffix) + len(projected_content)
        accumulated += formatted_size
        if accumulated > max_source_chars:
            truncated = True
        if not preformatted:
            metadata["title"] = projected_title
        if truncated:
            metadata[marker] = True

        if isinstance(document, dict):
            bounded.append(
                {
                    **document,
                    "content": projected_content,
                    "metadata": metadata,
                }
            )
        elif isinstance(document, Document):
            bounded.append(
                replace(
                    document,
                    content=projected_content,
                    metadata=metadata,
                )
            )
        else:
            copied_document = copy.copy(document)
            copied_document.content = projected_content
            copied_document.metadata = metadata
            bounded.append(copied_document)
        if accumulated >= max_source_chars + 1:
            break
    return bounded


async def retrieve_slides_source_documents_v1(
    *,
    query: str,
    owner_user_id: str,
    top_k: int,
    media_db: Any,
    max_source_chars: int = 200_000,
    chacha_db: Any | None = None,
    reranking_strategy: Literal["none", "flashrank", "cross_encoder"] = "none",
) -> RAGResult:
    """Run the closed, owner-local retrieval path for HTML slide sources."""
    if not isinstance(query, str) or not query.strip() or len(query) > 20_000:
        raise ValueError("query must contain between 1 and 20,000 characters")
    if not isinstance(owner_user_id, str) or not owner_user_id.strip():
        raise ValueError("owner_user_id must be a non-empty string")
    if len(owner_user_id.encode("utf-8")) > 256:
        raise ValueError("owner_user_id must not exceed 256 UTF-8 bytes")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 100:
        raise ValueError("top_k must be an integer between 1 and 100")
    if (
        isinstance(max_source_chars, bool)
        or not isinstance(max_source_chars, int)
        or not 1 <= max_source_chars <= 200_000
    ):
        raise ValueError("max_source_chars must be an integer between 1 and 200,000")
    if reranking_strategy not in SLIDES_SOURCE_RERANKING_STRATEGIES:
        raise ValueError(f"Unsupported local reranking strategy: {reranking_strategy!r}")
    if media_db is None:
        raise ValueError("media_db is required")
    if chacha_db is None:
        raise ValueError("chacha_db is required")
    local_reranker = None
    if reranking_strategy != "none":
        if create_preinstalled_local_reranker is None:
            raise RAGConfigurationError("Local reranking support is unavailable.")
        local_reranker = create_preinstalled_local_reranker(
            reranking_strategy,
            top_k=top_k,
        )

    payload = {
        "query": query,
        "strategy": "standard",
        "sources": ["media_db", "notes", "chats"],
        "search_mode": "fts",
        "fts_level": "chunk",
        "top_k": top_k,
        "min_score": 0.0,
        "enable_generation": False,
        "enable_text_late_chunking": False,
        "enable_reranking": reranking_strategy != "none",
        "reranking_strategy": reranking_strategy,
        "user_id": owner_user_id,
        "feedback_user_id": owner_user_id,
        "rag_profile": SLIDES_SOURCE_PROFILE,
    }
    resolved_request = ResolvedRAGRequest(
        query=query,
        strategy="standard",
        payload=payload,
        index_namespace=None,
        rag_profile=SLIDES_SOURCE_PROFILE,
        user_id=owner_user_id,
        feedback_user_id=owner_user_id,
    )
    retrieval_plan = RetrievalPlan(
        query=query,
        sources=("media_db", "notes", "chats"),
        search_mode="fts",
        top_k=top_k,
        min_score=0.0,
        index_namespace=None,
        collection_names={},
    )
    if (
        MediaDBRetriever is None
        or NotesDBRetriever is None
        or ChatHistoryRetriever is None
        or RetrievalConfig is None
        or allocate_slides_source_projection_caps is None
    ):
        raise RAGConfigurationError("Owner-local database retrieval is unavailable.")
    retrieval_config = RetrievalConfig(
        max_results=top_k,
        use_fts=True,
        use_vector=False,
        fts_level="chunk",
        enable_text_late_chunking=False,
    )
    media_retriever = MediaDBRetriever(
        None,
        config=retrieval_config,
        user_id=owner_user_id,
        media_db=media_db,
    )
    notes_retriever = NotesDBRetriever(
        None,
        config=retrieval_config,
        chacha_db=chacha_db,
    )
    chats_retriever = ChatHistoryRetriever(
        None,
        config=retrieval_config,
        chacha_db=chacha_db,
    )

    async def retrieve_source_candidates(
        retriever: "MediaDBRetrieverType | NotesDBRetrieverType | ChatHistoryRetrieverType",
    ) -> list[Document]:
        return await retriever.retrieve_slides_source_candidates_v1(
            query=query,
            owner_user_id=owner_user_id,
            top_k=top_k,
        )

    try:
        source_results = await asyncio.gather(
            retrieve_source_candidates(media_retriever),
            retrieve_source_candidates(notes_retriever),
            retrieve_source_candidates(chats_retriever),
        )
        candidates = _fuse_slides_source_documents(
            source_results,
            source_limits=(top_k, top_k, top_k),
            top_k=top_k,
        )
        selected_candidates, projection_caps = allocate_slides_source_projection_caps(
            candidates,
            total_chars=max_source_chars + 1,
            separator_chars=2,
        )
        retrievers_by_source: dict[
            DataSource,
            MediaDBRetrieverType | NotesDBRetrieverType | ChatHistoryRetrieverType,
        ] = {
            DataSource.MEDIA_DB: media_retriever,
            DataSource.NOTES: notes_retriever,
            DataSource.CHAT_HISTORY: chats_retriever,
        }
        projection_requests: dict[DataSource, list[tuple[Document, int]]] = {
            source: [] for source in retrievers_by_source
        }
        for candidate, char_cap in zip(selected_candidates, projection_caps):
            if candidate.source not in projection_requests:
                raise RAGConfigurationError("Unexpected local source candidate.")
            projection_requests[candidate.source].append((candidate, char_cap))

        async def project_source(source: DataSource) -> list[Document]:
            return await retrievers_by_source[source].project_slides_source_documents_v1(
                projections=projection_requests[source],
                owner_user_id=owner_user_id,
            )

        projected_source_results = await asyncio.gather(
            project_source(DataSource.MEDIA_DB),
            project_source(DataSource.NOTES),
            project_source(DataSource.CHAT_HISTORY),
        )
        projected_by_key = {
            (document.source, document.id): document
            for source_documents in projected_source_results
            for document in source_documents
        }
        documents = [
            projected_by_key[(candidate.source, candidate.id)]
            for candidate in selected_candidates
            if (candidate.source, candidate.id) in projected_by_key
        ]
        marker = "_standalone_source_projection_truncated"
        if len(selected_candidates) < len(candidates) and documents:
            sentinel = documents[-1]
            sentinel_metadata = dict(sentinel.metadata)
            sentinel_metadata[marker] = True
            documents[-1] = replace(sentinel, metadata=sentinel_metadata)
        documents = _bound_slides_source_documents(
            documents,
            max_source_chars=max_source_chars,
            top_k=top_k,
        )
        has_truncated_projection = any(
            isinstance(getattr(document, "metadata", None), dict) and bool(document.metadata.get(marker))
            for document in documents
        )
        if local_reranker is not None and documents and not has_truncated_projection:
            scored_documents = await local_reranker.rerank(query, documents)
            documents = _bound_slides_source_documents(
                [scored.document for scored in scored_documents],
                max_source_chars=max_source_chars,
                top_k=top_k,
            )
        retrieval_result = RetrievedEvidence(
            documents=documents,
            metadata={
                "resolved_request": {
                    "query": resolved_request.query,
                    "user_id": resolved_request.user_id,
                },
                "retrieval_plan": {
                    "query": retrieval_plan.query,
                    "sources": list(retrieval_plan.sources),
                    "search_mode": retrieval_plan.search_mode,
                    "top_k": retrieval_plan.top_k,
                },
            },
        )
        return build_retrieval_only_result(
            resolved_request=resolved_request,
            retrieval_plan=retrieval_plan,
            retrieval_result=retrieval_result,
        )
    finally:
        close = getattr(media_retriever, "close", None)
        if callable(close):
            close()


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

        should_filter = (
            is_workspace_artifact and not requested_workspace
        ) or (
            bool(requested_workspace)
            and bool(doc_workspace)
            and doc_workspace != requested_workspace
        )

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
    source_failures: set[DataSource] | None = None,
    security_excluded_counts: dict[str, int] | None = None,
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
        elif source_failures and data_source in source_failures:
            entry = {
                "status": "error",
                "count": count,
                "reason": "retrieval_failed",
            }
        elif count == 0:
            entry = {
                "status": "empty",
                "count": 0,
                "reason": (
                    "security_filtered"
                    if security_excluded_counts and security_excluded_counts.get(source, 0) > 0
                    else "no_matching_entries"
                ),
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


def _should_restore_classification_local_retrieval(
    *,
    skip_retrieval_stack: bool,
    skip_local_retrieval: bool,
    classification_local_retrieval: Any,
) -> bool:
    """Return whether research failure may re-enable classification-disabled local retrieval."""
    return (
        not skip_retrieval_stack
        and skip_local_retrieval
        and classification_local_retrieval == "disabled"
    )


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
    reranking_strategy: Literal[
        "flashrank", "cross_encoder", "hybrid", "llama_cpp", "llm_scoring", "two_tier", "none"
    ] = "flashrank",
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
    include_retrieval_diagnostics: bool = True,

    # ========== STREAMING ==========
    enable_streaming: bool = False,
    # ========== INDEXING / NAMESPACE ==========
    index_namespace: Optional[str] = None,
    retrieval_plan: Optional[RetrievalPlan] = None,
    resolved_request: Optional[ResolvedRAGRequest] = None,
    credential_runtime: Any = None,

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
    **kwargs: Any,
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

    rcfg = RagRunConfig.from_call_locals(locals())
    state = _run_state.PipelineState(rcfg)

    ambient_scope = get_scope()
    state.authorization_cache_scope: dict[str, object] | None = None
    state.authorization_cache_error: str | None = None
    if ambient_scope is not None:
        if isinstance(ambient_scope, ScopeContext):
            try:
                state.authorization_cache_scope = content_authorization_cache_scope(
                    ambient_scope
                )
            except (TypeError, ValueError):
                state.authorization_cache_error = "malformed"
        else:
            state.authorization_cache_error = "malformed"

    state.runtime_user_id = (
        getattr(rcfg.runtime.credential_runtime, "_user_id", None)
        if rcfg.runtime.credential_runtime is not None
        else None
    )
    if state.runtime_user_id is not None:
        if ambient_scope is None:
            state.authorization_cache_error = "missing"
        elif state.authorization_cache_scope is None:
            state.authorization_cache_error = "malformed"
        elif (
            type(state.authorization_cache_scope["user_id"]) is not type(state.runtime_user_id)
            or state.authorization_cache_scope["user_id"] != state.runtime_user_id
        ):
            state.authorization_cache_error = "user_mismatch"

    request_metadata: dict[str, Any] = {}
    inbound_metadata = rcfg.request.metadata
    if isinstance(inbound_metadata, dict):
        request_metadata.update(inbound_metadata)
    if state.generation_prompt is not None:
        request_metadata["generation_prompt"] = state.generation_prompt
    request_metadata["max_generation_tokens"] = rcfg.generation.max_generation_tokens
    state.workspace_id = rcfg.scope.workspace_id
    if state.workspace_id is None:
        state.workspace_id = request_metadata.get("workspace_id")
    if state.workspace_id is not None:
        state.workspace_id = str(state.workspace_id).strip() or None
        if state.workspace_id is not None:
            request_metadata["workspace_id"] = state.workspace_id
    state.prompts_db_path = rcfg.runtime.prompts_db_path
    state.world_books_db_path = rcfg.runtime.world_books_db_path
    state.chat_dictionaries_db_path = rcfg.runtime.chat_dictionaries_db_path

    if state.resolved_request is None:
        state.resolved_request = resolve_legacy_standard_pipeline_request(
            query=state.query,
            search_mode=rcfg.retrieval.search_mode,
            top_k=state.top_k,
            sources=rcfg.scope.sources,
            min_score=rcfg.retrieval.min_score,
            index_namespace=rcfg.retrieval.index_namespace,
            rag_profile=rcfg.request.rag_profile,
            user_id=rcfg.scope.user_id,
            feedback_user_id=rcfg.scope.feedback_user_id,
            enable_generation=rcfg.generation.enable_generation,
            include_sources=bool(rcfg.request.include_sources),
            include_metadata=bool(rcfg.request.include_metadata),
            metadata=request_metadata,
        )

    if state.retrieval_plan is None:
        state.retrieval_plan = build_retrieval_plan(state.resolved_request)

    state.effective_enable_generation = state.resolve_effective_enable_generation()

    state.retrieval_query = state.query
    state.retrieval_sources = rcfg.scope.sources
    state.retrieval_search_mode = rcfg.retrieval.search_mode
    state.retrieval_top_k = state.top_k
    state.retrieval_min_score = rcfg.retrieval.min_score
    state.retrieval_index_namespace = rcfg.retrieval.index_namespace

    if state.retrieval_plan is not None:
        planned_query = str(state.retrieval_plan.query or "").strip()
        if planned_query:
            state.retrieval_query = planned_query
        planned_sources = [
            str(getattr(source, "value", source)).strip()
            for source in state.retrieval_plan.sources
            if str(getattr(source, "value", source)).strip()
        ]
        if planned_sources:
            state.retrieval_sources = planned_sources
        planned_search_mode = str(state.retrieval_plan.search_mode or "").strip().lower()
        if planned_search_mode in {"fts", "vector", "hybrid"}:
            state.retrieval_search_mode = cast(
                Literal["fts", "vector", "hybrid"],
                planned_search_mode,
            )
        with contextlib.suppress(TypeError, ValueError):
            state.retrieval_top_k = max(1, int(state.retrieval_plan.top_k))
        with contextlib.suppress(TypeError, ValueError):
            state.retrieval_min_score = float(state.retrieval_plan.min_score)
        state.retrieval_index_namespace = state.planned_index_namespace(state.retrieval_plan)

    state.query_expansion_corpus = (
        state.retrieval_plan.index_namespace
        if state.retrieval_plan is not None and state.retrieval_plan.index_namespace is not None
        else rcfg.retrieval.index_namespace
    )

    # Basic input validation (short-circuit before heavier setup)
    if not isinstance(state.query, str) or not state.query.strip():
        msg = "Invalid query"
        metadata = {"original_query": state.query}
        try:
            inbound_meta = rcfg.request.metadata
            if isinstance(inbound_meta, dict):
                metadata.update(inbound_meta)
        except TypeError:
            pass
        timings = {"total": 0.0}
        # Consistent contract: return UnifiedRAGResponse for all outcomes
        try:
            from tldw_Server_API.app.core.RAG.rag_schemas_unified import UnifiedRAGResponse

            return UnifiedRAGResponse(
                documents=[],
                query=(state.query if isinstance(state.query, str) else ""),
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
                query=state.query if isinstance(state.query, str) else "",
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
    explicit_fields = state.resolved_request.explicit_fields
    explicit_payload = state.resolved_request.payload
    if "expand_query" in explicit_fields:
        state.expand_query = explicit_payload["expand_query"]
    else:
        state.expand_query = state.expand_query or rcfg.query_prep.enable_expansion
    state.media_db_path = _resolve_sqlite_rag_db_path(state.media_db_path, db=rcfg.runtime.media_db)

    # ========== SEARCH DEPTH MODE PRESETS ==========
    # Apply parameter presets based on search_depth_mode. Individual parameter
    # overrides take precedence; legacy calls retain their default-value heuristic.
    state.retrieval_scalar_overrides_applied = False
    if rcfg.request.search_depth_mode is not None:
        presets = _SEARCH_DEPTH_PRESETS.get(rcfg.request.search_depth_mode)
        # Canonical requests distinguish an omitted value from an explicit default.
        if presets:
            if "top_k" in explicit_fields:
                state.top_k = explicit_payload["top_k"]
                state.retrieval_plan = replace(state.retrieval_plan, top_k=state.top_k)
            elif state.top_k == 10:
                preset_top_k = presets.get("top_k", state.top_k)
                if preset_top_k != state.top_k:
                    state.retrieval_scalar_overrides_applied = True
                state.top_k = preset_top_k
            if "enable_reranking" in explicit_fields:
                state.enable_reranking = explicit_payload["enable_reranking"]
            elif state.enable_reranking is True and "enable_reranking" in presets:
                state.enable_reranking = presets["enable_reranking"]
            if "reranking_strategy" in explicit_fields:
                state.reranking_strategy = explicit_payload["reranking_strategy"]
            elif state.reranking_strategy == "flashrank" and "reranking_strategy" in presets:
                state.reranking_strategy = presets["reranking_strategy"]
            if "expand_query" in explicit_fields:
                state.expand_query = explicit_payload["expand_query"]
            elif not state.expand_query and presets.get("expand_query"):
                state.expand_query = True
            if "enable_hyde" in explicit_fields:
                state.enable_hyde = explicit_payload["enable_hyde"]
            elif not state.enable_hyde and presets.get("enable_hyde"):
                state.enable_hyde = True
            if "enable_web_fallback" in explicit_fields:
                state.enable_web_fallback = explicit_payload["enable_web_fallback"]
            elif not state.enable_web_fallback and presets.get("enable_web_fallback"):
                state.enable_web_fallback = True
            if "enable_multi_vector_passages" in explicit_fields:
                state.enable_multi_vector_passages = explicit_payload["enable_multi_vector_passages"]
            elif not state.enable_multi_vector_passages and presets.get("enable_multi_vector_passages"):
                state.enable_multi_vector_passages = True

    state.refresh_retrieval_snapshot_from_current_knobs(
        prefer_current_scalars=state.retrieval_scalar_overrides_applied,
    )

    # Initialize result and timing
    state.start_time = time.time()
    state.result = UnifiedSearchResult(
        documents=[],
        query=state.query,
        metadata={"original_query": state.query}
    )
    state.optional_provider_failure_code: str | None = None

    state.standard_evidence_coordinated = False
    state.claims_payload = None
    state.factuality_payload = None
    # Merge inbound metadata if provided (API pattern)
    try:
        inbound_meta = rcfg.request.metadata
        if isinstance(inbound_meta, dict):
            state.result.metadata.update(inbound_meta)
    except TypeError:
        pass
    state.result.metadata["retrieval_cache_hit"] = False
    state.result.metadata["generation_executed"] = False
    cache_bypass_modes = [
        mode
        for enabled, mode in (
            (state.expand_query, "query_expansion"),
            (state.enable_hyde, "hyde"),
            (rcfg.query_prep.enable_prf, "prf"),
            (rcfg.retrieval.enable_query_decomposition, "query_decomposition"),
            (rcfg.retrieval.enable_gap_analysis, "gap_analysis"),
        )
        if enabled
    ]
    state.retrieval_cache_eligible = bool(
        state.enable_cache
        and not cache_bypass_modes
        and not rcfg.query_prep.auto_temporal_filters
    )
    if state.enable_cache and cache_bypass_modes:
        state.result.metadata["cache_bypassed"] = {
            "reason": "secondary_retrieval_mode",
            "modes": cache_bypass_modes,
        }
    elif state.enable_cache and rcfg.query_prep.auto_temporal_filters:
        state.result.metadata["cache_bypassed"] = {
            "reason": "auto_temporal_window",
        }
    if state.enable_cache and state.authorization_cache_error is not None:
        state.retrieval_cache_eligible = False
        state.result.metadata["cache_bypassed"] = {
            "reason": "content_authorization_scope_unavailable",
            "code": state.authorization_cache_error,
        }

    if rcfg.observability.include_retrieval_diagnostics:
        state.ensure_profile_resolution_metadata()

    state.cache_instance = None
    state.cache_setup_attempted = False
    state.cache_retrieval_snapshot: Optional[list[Any]] = None
    state.base_retrieval_executed = False
    state.cache_max_size = 1000
    try:
        from tldw_Server_API.app.core.config import RAG_SERVICE_CONFIG

        cfg = cast(dict[str, Any], RAG_SERVICE_CONFIG) if isinstance(RAG_SERVICE_CONFIG, dict) else {}
        state.cache_max_size = int((cfg.get("cache") or {}).get("max_cache_size", state.cache_max_size))
    except (ImportError, TypeError, ValueError):
        pass
    # --- Internal helpers (defined early so downstream phases can use them) ---
    # Initialize monitoring if requested
    state.metrics = None
    if rcfg.observability.enable_monitoring:
        state.metrics = QueryMetrics(
            query_id=str(uuid.uuid4()),
            query=state.query,
            timestamp=state.start_time,
            total_duration=0.0,
        )

    state.resolved_data_sources: list[DataSource] = []

    try:
        _scope.resolve_sources(rcfg, state)

        # ========== LEARNED FUSION / CALIBRATION HELPERS ==========
        # ========== PRE-RETRIEVAL CLARIFICATION ==========
        await _query_prep.clarify_and_spellcheck(rcfg, state)
        # ========== QUERY CLASSIFICATION & REFORMULATION ==========
        await _query_prep.run_query_classification(rcfg, state)

        # ========== AGENTIC RESEARCH LOOP ==========
        # If research loop is enabled and we're in balanced/quality mode,
        # delegate to the research agent instead of the standard pipeline
        await _research.run_research_loop(rcfg, state)
                # Fall through to standard pipeline

        # ========== PRODUCTION DEFAULTS (env-based) ==========
        # If running in production, enable stricter guardrails by default
        _query_prep.apply_guardrail_defaults(rcfg, state)

        # ========== QUERY EXPANSION ==========
        await _query_prep.run_query_expansion(rcfg, state)

        # ========== INTENT ROUTING (optional) ==========
        _query_prep.route_and_weight(rcfg, state)

        # ========== HyDE PREP (optional) ==========
        await _query_prep.prepare_hyde_and_temporal_filters(rcfg, state)

        # ========== CACHE CHECK ==========
        # Resolve auto-temporal scope before deriving the retrieval cache identity.
        await _retrieval.lookup_retrieval_cache(rcfg, state)

        # ========== DOCUMENT RETRIEVAL ==========
        await _retrieval.run_retrieval(rcfg, state)

        if (
            state.retrieval_cache_eligible
            and state.base_retrieval_executed
            and not state.result.cache_hit
            and not state._skip_retrieval_stack
            and not state._skip_local_retrieval
            and state.result.documents
        ):
            state.apply_workspace_filtering_to_result()
            state.cache_retrieval_snapshot = _clone_cached_documents(
                list(state.result.documents)
            )

        # ========== MULTI-VECTOR PASSAGES (optional, pre-rerank) ==========
        await _augment.augment_passages(rcfg, state)

        # ========== GAP ANALYSIS / FOLLOW-UPS (optional) ==========
        await _augment.run_gap_analysis(rcfg, state)

        # ========== INSTRUCTION-INJECTION FILTERING (pre-reranking) ==========
        _filters.apply_content_filters(rcfg, state)

        # ========== SECURITY FILTERING ==========
        await _filters.apply_security_filter(rcfg, state)

        # ========== TABLE PROCESSING ==========
        await _processing.process_tables_and_vlm(rcfg, state)

        # ========== PROGRESSIVE EVIDENCE ACCUMULATION ==========
        await _processing.run_evidence_accumulation(rcfg, state)

        # Apply personalization priors (pre-rerank) if requested
        try:
            if rcfg.observability.apply_feedback_boost and state.result.documents and UserPersonalizationStore:
                store = UserPersonalizationStore(rcfg.scope.feedback_user_id or rcfg.scope.user_id)
                state.result.documents = store.boost_documents(state.result.documents, corpus=rcfg.retrieval.index_namespace)
                state.result.metadata.setdefault("personalization", {})["boost_applied_pre_rerank"] = True
        except ValueError as exc:
            logger.debug(f"Personalization boost disabled for user_id={rcfg.scope.feedback_user_id or rcfg.scope.user_id}: {exc}")
        except (AttributeError, OSError, RuntimeError, TypeError, sqlite3.Error):
            pass

        # ========== SELF-CORRECTING RAG: DOCUMENT GRADING (Stage 1) ==========
        # Grade documents for relevance BEFORE expensive reranking/generation
        await _self_correct.grade_documents(rcfg, state)

        # ========== SELF-CORRECTING RAG: QUERY REWRITING LOOP (Stage 2) ==========
        # When document grading shows low relevance, rewrite query and retry retrieval
        await _self_correct.run_rewrite_loop(rcfg, state)

        # ========== RERANKING ==========
        await _rerank.run_rerank(rcfg, state)

        # ========== SELF-CORRECTING RAG: WEB SEARCH FALLBACK (Stage 3) ==========
        # When local retrieval has low relevance, fall back to web search
        await _post_retrieval.run_web_fallback(rcfg, state)

        # ========== WHY THESE SOURCES (metadata) ==========
        _post_retrieval.explain_and_expand_sources(rcfg, state)

        await _post_retrieval.build_citations_and_chains(rcfg, state)

        _generation.prepare_generation(rcfg, state)

        await _generation.run_generation(rcfg, state)

        await _verification.check_citations_and_groundedness(rcfg, state)
        await _claims.run_claims(rcfg, state)

        await _verification.run_numeric_fidelity(rcfg, state)

        await _verification.run_post_verification(rcfg, state)

        await _post_processing.finish_answer(rcfg, state)

        await _post_processing.store_and_observe(rcfg, state)

    except _run_state.EarlyReturn:
        pass
    except _RAG_PROVIDER_FAILURES:
        raise
    except (
        AttributeError,
        ConnectionError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        asyncio.TimeoutError,
    ):
        state.result.errors.append("pipeline_failed")
        logger.error("Unified pipeline failed")
        if rcfg.resilience.fallback_on_error:
            return {
                "query": state.query,
                "documents": [],
                "answer": "",
                "cached": False,
                "error": "pipeline_failed",
                "metadata": state.result.metadata,
                "timings": state.result.timings,
            }

    finally:
        await _post_processing.finalize_run(rcfg, state)

    if not state.standard_evidence_coordinated:
        state.result = coordinate_standard_result_evidence(
            state.result,
            state.resolved_request,
            retrieval_plan=state.retrieval_plan,
        )

    # Convert to Pydantic response
    try:
        from tldw_Server_API.app.core.RAG.rag_schemas_unified import UnifiedRAGResponse

        doc_dicts = [_serialize_result_document(d) for d in (state.result.documents or [])]
        return UnifiedRAGResponse(
            documents=doc_dicts,
            query=state.result.query,
            expanded_queries=state.result.expanded_queries,
            metadata=state.result.metadata,
            timings=state.result.timings,
            citations=state.result.citations,
            academic_citations=(state.result.metadata or {}).get("academic_citations", []),
            chunk_citations=(state.result.metadata or {}).get("chunk_citations", []),
            generated_answer=state.result.generated_answer,
            cache_hit=state.result.cache_hit,
            errors=state.result.errors,
            security_report=state.result.security_report,
            total_time=state.result.total_time,
            claims=state.claims_payload,
            factuality=state.factuality_payload,
        )
    except (ImportError, TypeError, ValueError):
        # Fallback: return a minimal dict if Pydantic is not available
        return {
            "documents": [_serialize_result_document(d) for d in (state.result.documents or [])],
            "query": state.result.query,
            "expanded_queries": state.result.expanded_queries,
            "metadata": state.result.metadata,
            "timings": state.result.timings,
            "citations": state.result.citations,
            "generated_answer": state.result.generated_answer,
            "cache_hit": state.result.cache_hit,
            "errors": state.result.errors,
            "security_report": state.result.security_report,
            "total_time": state.result.total_time,
            "claims": state.claims_payload,
            "factuality": state.factuality_payload,
        }


# ========== BATCH PROCESSING WRAPPER ==========
async def unified_batch_pipeline(
    queries: list[str],
    max_concurrent: int = 5,
    on_progress: Optional[Callable[[int, int], Any]] = None,
    on_query_done: Optional[Callable[[int, str, Optional[UnifiedPipelineResult], Optional[BaseException]], Any]] = None,
    query_indices: Optional[list[int]] = None,
    credential_runtime: Any = None,
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
    if credential_runtime is not None:
        kwargs["credential_runtime"] = credential_runtime
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
            provider = _embedding_provider_from_config(cfg)
            effective_model = _embedding_model_from_config(cfg)
            handle = None
            call_kwargs: dict[str, Any] = {}
            if credential_runtime is not None:
                if provider == "openai":
                    handle, call_kwargs = await _resolve_runtime_embedding_call(
                        credential_runtime,
                        provider,
                        effective_model,
                    )
                else:
                    call_kwargs = _runtime_local_embedding_call_kwargs(cfg, provider)
            vectors = await _run_sync_embedding_call(
                partial(
                    create_embeddings_batch,
                    rep_texts,
                    cfg,
                    None,
                    **call_kwargs,
                ),
                on_success=(
                    partial(
                        _mark_runtime_used_for_embeddings,
                        credential_runtime=credential_runtime,
                        handle=handle,
                    )
                    if handle is not None
                    else None
                ),
            )

            # Normalize vectors to unit length for cosine
            def _norm(v):
                try:
                    import math

                    if hasattr(v, "tolist"):
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
                thr = float(_os.getenv("RAG_BATCH_NEAR_DUP_THRESHOLD", "0.9"))
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
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - best-effort clustering; never fail batch
            logger.warning("Batch query clustering disabled due to embedding failure")
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
        try:
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
                        if isinstance(exc, _RAG_PROVIDER_FAILURES):
                            raise
                        err = exc
                        head_results[head_idx] = exc

                    head_key = heads[head_idx]
                    members = clusters.get(head_key, [])
                    for i_uq in members:
                        orig_indices = normalized_map.get(unique_keys[i_uq], [])
                        for local_idx in orig_indices:
                            global_idx = _resolve_index(local_idx)
                            query_text = queries[local_idx] if local_idx < len(queries) else ""
                            if bool(on_query_done):
                                try:
                                    callback_result = on_query_done(global_idx, query_text, result, err)
                                    if asyncio.iscoroutine(callback_result):
                                        await callback_result
                                except Exception as cb_err:  # noqa: BLE001 - callbacks must not fail pipeline
                                    logger.warning(f"Batch on_query_done callback failed: {cb_err}")
                            await _notify_progress(1)
        except BaseException:
            for child_task in tasks:
                if not child_task.done():
                    child_task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
    else:
        try:
            from .batch_utils import run_batch as _run_batch

            async def _process_batch_query(query: str) -> UnifiedPipelineResult:
                return await unified_rag_pipeline(
                    query=query,
                    **_pipeline_kwargs_for_query(query),
                )

            _batch_result = await _run_batch(
                items=head_queries,
                func=_process_batch_query,
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

            fallback_coroutines = [process_with_semaphore(q) for q in head_queries]
            head_results = await asyncio.gather(*fallback_coroutines, return_exceptions=True)

    for head_result in head_results:
        if isinstance(head_result, _RAG_PROVIDER_FAILURES):
            raise head_result

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
                    ures_any
                    if pos == 0 and queries[i] == rep_texts[i_uq]
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
    credential_runtime: Any = None,
) -> list[Document]:
    """
    Simple search wrapper for basic use cases.

    Args:
        query: Search query
        top_k: Number of results

    Returns:
        List of documents
    """
    pipeline_kwargs = {
        "query": query,
        "top_k": top_k,
        "expand_query": False,
        "enable_cache": True,
        "enable_reranking": True,
        # Simple search returns documents only; generating an answer would spend LLM tokens for nothing.
        "enable_generation": False,
        "sources": sources,
        "media_db": media_db,
        "chacha_db": chacha_db,
        "media_db_path": media_db_path,
        "notes_db_path": notes_db_path,
        "character_db_path": character_db_path,
        "kanban_db_path": kanban_db_path,
        "user_id": user_id,
    }
    if credential_runtime is not None:
        pipeline_kwargs["credential_runtime"] = credential_runtime
    result = await unified_rag_pipeline(
        **pipeline_kwargs,
    )
    if isinstance(result, UnifiedSearchResult):
        return result.documents
    docs = getattr(result, "documents", None)
    if isinstance(docs, list) and all(isinstance(d, Document) for d in docs):
        return docs
    return []


async def advanced_search(
    query: str, with_citations: bool = True, with_answer: bool = True, **kwargs
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
        **kwargs,
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
            qm = {1: 1, 2: 4, 3: 7, 4: 10}[qn]
            start_dt = datetime(y, qm, 1)
            end_month = qm + 2
            _, last_day = calendar.monthrange(y, end_month)
            end_dt = datetime(y, end_month, last_day, 23, 59, 59)
        month_names = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
        m_month_year = re.search(
            r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+(20\d{2}|19\d{2})\b",
            qlower,
        )
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
            start_dt = datetime(y, 1, 1)
            end_dt = datetime(y, 12, 31, 23, 59, 59)
        if start_dt is None and end_dt is None:
            start_dt = now - timedelta(days=7)
            end_dt = now
        if start_dt is None or end_dt is None:
            return None
        return {"start": start_dt.isoformat(), "end": end_dt.isoformat()}
    except (AttributeError, TypeError, ValueError):
        return None
