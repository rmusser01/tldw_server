"""Frozen source authorization and fail-closed retrieval for shared chat."""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from tldw_Server_API.app.core.DB_Management.media_db import api as media_db_api
from tldw_Server_API.app.core.RAG.rag_service.unified_pipeline import (
    unified_rag_pipeline,
)
from tldw_Server_API.app.core.Workspaces.status_projection import (
    build_source_status_projection,
)

SourceMode = Literal["all", "include"]
RAGPipeline = Callable[..., Awaitable[Any]]

_MAX_SOURCES = 500
_MAX_SOURCE_ID_CHARS = 512
_MAX_EVIDENCE = 20
_MAX_EVIDENCE_TEXT_CHARS = 4_000
_MAX_EVIDENCE_TEXT_TOTAL_CHARS = 48_000
_MAX_SOURCE_TITLE_CHARS = 512
_MAX_CHUNK_ID_CHARS = 512
_MAX_LOCATOR = 2_147_483_647


class SharedWorkspaceChatServiceError(RuntimeError):
    """Base class with a stable code and disclosure-safe message."""

    code = "shared_workspace_unavailable"
    retryable = True


class SharedWorkspaceSourceScopeInvalid(SharedWorkspaceChatServiceError):
    """Raised when a requested source scope is invalid."""

    code = "invalid_shared_chat_request"
    retryable = False

    def __init__(self) -> None:
        super().__init__("The shared chat request is invalid.")


class SharedWorkspaceSourceSubsetRequired(SharedWorkspaceChatServiceError):
    """Raised when all queryable sources exceed the shared-chat cap."""

    code = "source_subset_required"
    retryable = False

    def __init__(self) -> None:
        super().__init__("Select a smaller set of shared sources.")


class SharedWorkspaceSourceChanged(SharedWorkspaceChatServiceError):
    """Raised when a frozen source authorization snapshot no longer matches."""

    code = "shared_source_changed"
    retryable = False

    def __init__(self) -> None:
        super().__init__("The selected shared sources changed.")


class SharedWorkspaceRetrievalUnavailable(SharedWorkspaceChatServiceError):
    """Raised when retrieval cannot produce a fully verified result."""

    code = "retrieval_unavailable"
    retryable = True

    def __init__(self) -> None:
        super().__init__("Shared workspace retrieval is temporarily unavailable.")


class SharedWorkspaceNoRelevantEvidence(SharedWorkspaceChatServiceError):
    """Raised when retrieval returns no usable verified evidence."""

    code = "no_relevant_evidence"
    retryable = False

    def __init__(self) -> None:
        super().__init__("No relevant shared evidence was found.")


class _SharedWorkspaceDataUnavailable(SharedWorkspaceChatServiceError):
    def __init__(self) -> None:
        super().__init__("Shared workspace data is temporarily unavailable.")


class _NonQueryableSource(ValueError):
    pass


@dataclass(frozen=True)
class SharedSourceSnapshotItem:
    """Authorization and content identity for one canonical workspace source."""

    source_id: str
    media_id: int
    media_uuid: str
    content_hash: str
    readiness_class: str


@dataclass(frozen=True)
class SharedSourceSnapshot:
    """Immutable, canonical source scope used by one shared-chat request."""

    mode: SourceMode
    items: tuple[SharedSourceSnapshotItem, ...]
    snapshot_hash: str

    @property
    def source_ids(self) -> tuple[str, ...]:
        return tuple(item.source_id for item in self.items)

    @property
    def media_ids(self) -> tuple[int, ...]:
        return tuple(sorted({item.media_id for item in self.items}))


@dataclass(frozen=True)
class VerifiedSharedEvidence:
    """Bounded retrieved evidence with canonical shared-source provenance."""

    label: str
    source_id: str
    source_title: str
    content: str
    score: float
    chunk_index: int | None
    start_char: int | None
    end_char: int | None


@dataclass(frozen=True)
class SharedRetrievalPolicy:
    """Immutable allowlist of parameters for media-only shared retrieval."""

    pinned_parameters: tuple[tuple[str, Any], ...]
    dynamic_parameter_names: frozenset[str]
    reviewed_inert_parameter_names: frozenset[str]
    reviewed_absent_kwarg_names: frozenset[str]

    @property
    def pinned_parameter_names(self) -> frozenset[str]:
        return frozenset(name for name, _value in self.pinned_parameters).union(
            self.dynamic_parameter_names
        )

    def build_call(
        self,
        *,
        media_ids: tuple[int, ...],
        media_db_path: str,
        media_db: Any,
        owner_user_id: int,
    ) -> dict[str, Any]:
        call = dict(self.pinned_parameters)
        call["sources"] = list(call["sources"])
        call.update(
            media_db_path=media_db_path,
            media_db=media_db,
            include_media_ids=list(media_ids),
            index_namespace=f"user_{owner_user_id}_media_embeddings",
            user_id=str(owner_user_id),
        )
        return call


_PINNED_RETRIEVAL_PARAMETERS: tuple[tuple[str, Any], ...] = (
    ("sources", ("media_db",)),
    ("notes_db_path", None),
    ("character_db_path", None),
    ("kanban_db_path", None),
    ("sql_target_id", "media_db"),
    ("sql_retriever", None),
    ("search_mode", "fts"),
    ("fts_level", "chunk"),
    ("enable_text_late_chunking", False),
    ("adaptive_hybrid_weights", False),
    ("enable_intent_routing", False),
    ("auto_temporal_filters", False),
    ("top_k", _MAX_EVIDENCE),
    ("min_score", 0.0),
    ("expand_query", False),
    ("spell_check", False),
    ("enable_prf", False),
    ("enable_hyde", False),
    ("hyde_provider", None),
    ("hyde_model", None),
    ("enable_gap_analysis", False),
    ("enable_cache", False),
    ("adaptive_cache", False),
    ("keyword_filter", None),
    ("include_note_ids", None),
    ("enable_security_filter", False),
    ("detect_pii", False),
    ("redact_pii", False),
    ("content_filter", False),
    ("enable_table_processing", False),
    ("enable_vlm_late_chunking", False),
    ("enable_enhanced_chunking", False),
    ("chunk_type_filter", None),
    ("enable_parent_expansion", False),
    ("include_sibling_chunks", False),
    ("include_parent_document", False),
    ("enable_multi_vector_passages", False),
    ("mv_flatten_to_spans", False),
    ("enable_precomputed_spans", False),
    ("enable_numeric_table_boost", False),
    ("enable_reranking", False),
    ("reranking_strategy", "none"),
    ("reranking_model", None),
    ("enable_learned_fusion", False),
    ("enable_citations", False),
    ("include_page_numbers", False),
    ("enable_chunk_citations", False),
    ("enable_generation", False),
    ("strict_extractive", False),
    ("generation_model", None),
    ("generation_provider", None),
    ("generation_prompt", None),
    ("enable_pre_retrieval_clarification", False),
    ("enable_abstention", False),
    ("enable_multi_turn_synthesis", False),
    ("enable_post_verification", False),
    ("adaptive_max_retries", 0),
    ("adaptive_unsupported_threshold", 0.0),
    ("adaptive_max_claims", 0),
    ("adaptive_time_budget_sec", None),
    ("adaptive_advanced_rewrites", False),
    ("adaptive_rerun_on_low_confidence", False),
    ("adaptive_rerun_include_generation", False),
    ("adaptive_rerun_bypass_cache", False),
    ("adaptive_rerun_time_budget_sec", None),
    ("adaptive_rerun_doc_budget", None),
    ("enable_query_decomposition", False),
    ("enable_graph_retrieval", False),
    ("_adaptive_rerun", False),
    ("collect_feedback", False),
    ("feedback_user_id", None),
    ("apply_feedback_boost", False),
    ("enable_monitoring", False),
    ("enable_observability", False),
    ("trace_id", None),
    ("enable_performance_analysis", False),
    ("timeout_seconds", None),
    ("include_retrieval_diagnostics", False),
    ("enable_streaming", False),
    ("retrieval_plan", None),
    ("resolved_request", None),
    ("credential_runtime", None),
    ("highlight_results", False),
    ("highlight_query_terms", False),
    ("track_cost", False),
    ("debug_mode", False),
    ("include_rerank_debug_documents", False),
    ("enable_injection_filter", False),
    ("enable_content_policy_filter", False),
    ("enable_html_sanitizer", False),
    ("ocr_confidence_threshold", None),
    ("require_hard_citations", False),
    ("enable_numeric_fidelity", False),
    ("enable_claims", False),
    ("doc_only_verification", False),
    ("generate_verification_report", False),
    ("enable_dynamic_granularity", False),
    ("enable_evidence_accumulation", False),
    ("enable_evidence_chains", False),
    ("enable_document_grading", False),
    ("grading_model", None),
    ("grading_provider", None),
    ("grading_fallback_to_score", False),
    ("enable_query_rewriting_loop", False),
    ("enable_web_fallback", False),
    ("enable_knowledge_strips", False),
    ("enable_fast_hallucination_check", False),
    ("fast_hallucination_provider", None),
    ("fast_hallucination_model", None),
    ("enable_utility_grading", False),
    ("utility_grading_provider", None),
    ("utility_grading_model", None),
    ("enable_batch", False),
    ("batch_queries", None),
    ("enable_resilience", False),
    ("circuit_breaker", False),
    ("enable_date_filter", False),
    ("date_range", None),
    ("filter_media_types", None),
    ("chacha_db", None),
    ("fallback_on_error", False),
    ("session_id", None),
    ("ground_truth_doc_ids", None),
    ("enable_faithfulness_eval", False),
    ("search_depth_mode", None),
    ("enable_query_classification", False),
    ("chat_history", None),
    ("enable_query_reformulation", False),
    ("enable_research_loop", False),
    ("enable_research_action_dedup", False),
    ("enable_discussion_search", False),
    ("discussion_platforms", None),
    ("search_url_scraping", False),
    ("enable_research_progress", False),
    ("research_progress_callback", None),
    ("classifier_provider", None),
    ("classifier_model", None),
    ("enable_suggestions", False),
    ("enable_structured_response", False),
    ("enable_image_search", False),
    ("enable_video_search", False),
    ("rag_profile", None),
)

_DYNAMIC_RETRIEVAL_PARAMETERS = frozenset(
    {"media_db_path", "media_db", "include_media_ids", "index_namespace", "user_id"}
)

_REVIEWED_INERT_RETRIEVAL_PARAMETERS = frozenset(
    {
        "chunk_method",
        "chunk_size",
        "chunk_overlap",
        "chunk_language",
        "hybrid_alpha",
        "expansion_strategies",
        "max_query_variations",
        "prf_terms",
        "prf_sources",
        "prf_alpha",
        "prf_top_n",
        "max_followup_searches",
        "cache_threshold",
        "sensitivity_level",
        "table_method",
        "vlm_backend",
        "vlm_detect_tables_only",
        "vlm_max_pages",
        "vlm_late_chunk_top_k_docs",
        "parent_context_size",
        "sibling_window",
        "parent_max_tokens",
        "mv_span_chars",
        "mv_stride",
        "mv_max_spans",
        "rerank_top_k",
        "rerank_min_relevance_prob",
        "rerank_sentinel_margin",
        "calibrator_version",
        "abstention_policy",
        "citation_style",
        "clarification_timeout_sec",
        "max_generation_tokens",
        "abstention_behavior",
        "synthesis_time_budget_sec",
        "synthesis_draft_tokens",
        "synthesis_refine_tokens",
        "low_confidence_behavior",
        "max_subqueries",
        "subquery_time_budget_sec",
        "subquery_doc_budget",
        "subquery_max_concurrency",
        "graph_version",
        "graph_neighbors_k",
        "graph_alpha",
        "injection_filter_strength",
        "content_policy_types",
        "content_policy_mode",
        "html_allowed_tags",
        "html_allowed_attrs",
        "numeric_fidelity_behavior",
        "claim_extractor",
        "claim_verifier",
        "claims_top_k",
        "claims_conf_threshold",
        "claims_max",
        "nli_model",
        "claims_concurrency",
        "numeric_precision_mode",
        "accumulation_max_rounds",
        "accumulation_time_budget_sec",
        "grading_threshold",
        "grading_batch_size",
        "grading_timeout_sec",
        "grading_fallback_min_score",
        "max_rewrite_attempts",
        "rewrite_relevance_threshold",
        "web_fallback_threshold",
        "web_search_engine",
        "web_fallback_result_count",
        "web_fallback_merge_strategy",
        "strip_size_tokens",
        "strip_min_relevance",
        "max_strips",
        "fast_hallucination_timeout_sec",
        "utility_grading_timeout_sec",
        "batch_concurrent",
        "retry_attempts",
        "cache_ttl",
        "metrics_k",
        "research_max_iterations",
        "research_max_iterations_speed",
        "research_max_iterations_balanced",
        "research_max_iterations_quality",
        "num_suggestions",
    }
)

_REVIEWED_ABSENT_PIPELINE_KWARGS = frozenset(
    {
        "metadata",
        "workspace_id",
        "prompts_db_path",
        "world_books_db_path",
        "chat_dictionaries_db_path",
        "include_sources",
        "include_metadata",
        "prompts_db",
        "enable_expansion",
        "claims_budget_usd",
        "claims_budget_tokens",
        "claims_budget_strict",
        "faithfulness_llm",
    }
)

SHARED_RETRIEVAL_POLICY = SharedRetrievalPolicy(
    pinned_parameters=_PINNED_RETRIEVAL_PARAMETERS,
    dynamic_parameter_names=_DYNAMIC_RETRIEVAL_PARAMETERS,
    reviewed_inert_parameter_names=_REVIEWED_INERT_RETRIEVAL_PARAMETERS,
    reviewed_absent_kwarg_names=_REVIEWED_ABSENT_PIPELINE_KWARGS,
)


class _FrozenMediaStatusReader:
    """Serve readiness projection from media rows already read with lifecycle flags."""

    def __init__(self, rows: dict[int, dict[str, Any]]) -> None:
        self._rows = rows

    def get_media_status_by_id(
        self,
        media_id: int,
        *,
        include_deleted: bool = False,
        include_trash: bool = False,
    ) -> dict[str, Any] | None:
        del include_deleted, include_trash
        row = self._rows.get(media_id)
        return dict(row) if row is not None else None


@dataclass(frozen=True)
class SharedWorkspaceChatService:
    """Resolve frozen owner source scope and return only verified media evidence."""

    owner_chacha_db: Any = field(repr=False, compare=False)
    owner_media_db: Any = field(repr=False, compare=False)
    owner_media_db_path: str = field(repr=False)
    owner_user_id: int
    workspace_id: str
    rag_pipeline: RAGPipeline = field(
        default=unified_rag_pipeline,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if (
            not isinstance(self.owner_user_id, int)
            or isinstance(self.owner_user_id, bool)
            or self.owner_user_id <= 0
            or not isinstance(self.workspace_id, str)
            or not self.workspace_id.strip()
            or not isinstance(self.owner_media_db_path, str)
            or not self.owner_media_db_path.strip()
            or self.owner_chacha_db is None
            or self.owner_media_db is None
            or not callable(self.rag_pipeline)
        ):
            raise SharedWorkspaceSourceScopeInvalid()

    def resolve_source_snapshot(
        self,
        *,
        mode: SourceMode,
        source_ids: Sequence[str] = (),
        frozen_source_ids: Sequence[str] | None = None,
    ) -> SharedSourceSnapshot:
        """Resolve current authoritative source rows into a canonical snapshot."""
        if mode not in {"all", "include"}:
            raise SharedWorkspaceSourceScopeInvalid()
        requested = _normalize_source_ids(source_ids)
        if len(requested) > _MAX_SOURCES:
            raise SharedWorkspaceSourceScopeInvalid()
        if mode == "include" and (not requested or frozen_source_ids is not None):
            raise SharedWorkspaceSourceScopeInvalid()
        if mode == "all" and requested:
            raise SharedWorkspaceSourceScopeInvalid()

        if frozen_source_ids is not None:
            frozen = _normalize_source_ids(frozen_source_ids)
            if mode != "all" or not frozen or len(frozen) > _MAX_SOURCES:
                raise SharedWorkspaceSourceChanged()
            return self._snapshot_for_exact_ids(
                mode="all",
                source_ids=frozen,
                changed_on_mismatch=True,
            )

        rows = self._load_source_rows()
        if mode == "include":
            return self._snapshot_for_exact_ids(
                mode=mode,
                source_ids=requested,
                rows=rows,
                changed_on_mismatch=False,
            )

        try:
            queryable_items = self._queryable_items(rows)
        except ValueError:
            raise _SharedWorkspaceDataUnavailable() from None
        if len(queryable_items) > _MAX_SOURCES:
            raise SharedWorkspaceSourceSubsetRequired()
        if not queryable_items:
            raise SharedWorkspaceSourceScopeInvalid()
        return _build_snapshot(mode="all", items=queryable_items)

    def revalidate_source_snapshot(
        self,
        *,
        snapshot: SharedSourceSnapshot,
    ) -> SharedSourceSnapshot:
        """Rebuild exactly the frozen IDs and reject any authorization/content drift."""
        if not _snapshot_is_internally_valid(snapshot):
            raise SharedWorkspaceSourceChanged()
        current = self._snapshot_for_exact_ids(
            mode=snapshot.mode,
            source_ids=snapshot.source_ids,
            changed_on_mismatch=True,
        )
        if current.items != snapshot.items or current.snapshot_hash != snapshot.snapshot_hash:
            raise SharedWorkspaceSourceChanged()
        return current

    async def retrieve_verified_evidence(
        self,
        *,
        query: str,
        snapshot: SharedSourceSnapshot,
    ) -> tuple[VerifiedSharedEvidence, ...]:
        """Run locked media retrieval and reject the full result on provenance drift."""
        if (
            not isinstance(query, str)
            or not query.strip()
            or len(query) > 10_000
            or not _snapshot_is_internally_valid(snapshot)
        ):
            raise SharedWorkspaceRetrievalUnavailable()
        media_ids = snapshot.media_ids
        if not media_ids or len(snapshot.items) > _MAX_SOURCES:
            raise SharedWorkspaceRetrievalUnavailable()
        canonical_source_by_media = self._canonical_source_by_media(snapshot)

        call = SHARED_RETRIEVAL_POLICY.build_call(
            media_ids=media_ids,
            media_db_path=self.owner_media_db_path,
            media_db=self.owner_media_db,
            owner_user_id=self.owner_user_id,
        )
        try:
            result = await self.rag_pipeline(query=query, **call)
        except Exception:  # noqa: BLE001 - sanitize every pipeline failure at this boundary.
            raise SharedWorkspaceRetrievalUnavailable() from None

        documents = _result_value(result, "documents", [])
        errors = _result_value(result, "errors", [])
        generated_answer = _result_value(result, "generated_answer", None)
        metadata = _result_value(result, "metadata", {})
        cache_hit = _result_value(result, "cache_hit", False)
        if (
            not isinstance(documents, list)
            or _result_value(result, "query", None) != query
            or _result_value(result, "expanded_queries", None) != []
            or errors != []
            or generated_answer is not None
            or cache_hit is not False
            or _result_has_derived_outputs(result)
            or not _metadata_is_locked_retrieval(
                metadata,
                query=query,
                media_count=len(media_ids),
                document_count=len(documents),
                index_namespace=f"user_{self.owner_user_id}_media_embeddings",
            )
        ):
            raise SharedWorkspaceRetrievalUnavailable()

        validated: list[dict[str, Any]] = []
        record_by_identity: dict[tuple[int, str], dict[str, Any]] = {}
        for document in documents:
            identity, normalized = _validate_document(document, canonical_source_by_media)
            previous = record_by_identity.get(identity)
            if previous is not None:
                if previous != normalized:
                    raise SharedWorkspaceRetrievalUnavailable()
                continue
            record_by_identity[identity] = normalized
            validated.append(normalized)

        retained: list[dict[str, Any]] = []
        remaining_chars = _MAX_EVIDENCE_TEXT_TOTAL_CHARS
        for item in validated:
            if len(retained) >= _MAX_EVIDENCE or remaining_chars <= 0:
                continue
            content = item["content"].strip()[
                : min(_MAX_EVIDENCE_TEXT_CHARS, remaining_chars)
            ]
            if not content:
                continue
            remaining_chars -= len(content)
            retained.append({**item, "content": content})

        if not retained:
            raise SharedWorkspaceNoRelevantEvidence()
        return tuple(
            VerifiedSharedEvidence(label=f"E{index}", **item)
            for index, item in enumerate(retained, start=1)
        )

    def _canonical_source_by_media(
        self,
        snapshot: SharedSourceSnapshot,
    ) -> dict[int, tuple[str, str]]:
        try:
            rows = _index_source_rows(self._load_source_rows())
            canonical: dict[int, tuple[str, str]] = {}
            for item in snapshot.items:
                row = rows[item.source_id]
                if _positive_int(row.get("media_id")) != item.media_id:
                    raise ValueError("source remapped")
                current = canonical.get(item.media_id)
                candidate = (item.source_id, _safe_title(row.get("title")))
                if current is None or candidate[0] < current[0]:
                    canonical[item.media_id] = candidate
            return canonical
        except (KeyError, ValueError):
            raise SharedWorkspaceRetrievalUnavailable() from None

    def _load_source_rows(self) -> list[dict[str, Any]]:
        try:
            rows = self.owner_chacha_db.list_workspace_sources(self.workspace_id)
        except Exception:  # noqa: BLE001 - owner storage failures must not disclose details.
            raise _SharedWorkspaceDataUnavailable() from None
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise _SharedWorkspaceDataUnavailable()
        return [dict(row) for row in rows]

    def _snapshot_for_exact_ids(
        self,
        *,
        mode: SourceMode,
        source_ids: tuple[str, ...],
        changed_on_mismatch: bool,
        rows: list[dict[str, Any]] | None = None,
    ) -> SharedSourceSnapshot:
        source_rows = rows if rows is not None else self._load_source_rows()
        try:
            row_index = _index_source_rows(source_rows)
            selected = [row_index[source_id] for source_id in source_ids]
            items = self._queryable_items(selected, reject_nonqueryable=True)
        except (KeyError, _NonQueryableSource, ValueError):
            if changed_on_mismatch:
                raise SharedWorkspaceSourceChanged() from None
            raise SharedWorkspaceSourceScopeInvalid() from None
        if len(items) != len(source_ids):
            if changed_on_mismatch:
                raise SharedWorkspaceSourceChanged()
            raise SharedWorkspaceSourceScopeInvalid()
        return _build_snapshot(mode=mode, items=items)

    def _queryable_items(
        self,
        rows: list[dict[str, Any]],
        *,
        reject_nonqueryable: bool = False,
    ) -> tuple[SharedSourceSnapshotItem, ...]:
        indexed = _index_source_rows(rows)
        sorted_rows = [indexed[source_id] for source_id in sorted(indexed)]
        media_rows: dict[int, dict[str, Any]] = {}
        row_media_ids: dict[str, int] = {}
        invalid_sources: set[str] = set()

        for row in sorted_rows:
            source_id = _source_row_id(row)
            media_id = _positive_int(row.get("media_id"))
            row_media_ids[source_id] = media_id
            if media_id in media_rows:
                continue
            try:
                media = media_db_api.get_media_by_id(
                    self.owner_media_db,
                    media_id,
                    include_deleted=True,
                    include_trash=True,
                )
            except Exception:  # noqa: BLE001 - owner storage failures must not disclose details.
                raise _SharedWorkspaceDataUnavailable() from None
            if not isinstance(media, dict):
                invalid_sources.add(source_id)
                continue
            media_rows[media_id] = dict(media)

        live_rows: list[dict[str, Any]] = []
        for row in sorted_rows:
            source_id = _source_row_id(row)
            media_id = row_media_ids.get(source_id)
            media = media_rows.get(media_id) if media_id is not None else None
            if (
                media is None
                or _truthy(media.get("deleted"))
                or _truthy(media.get("is_trash"))
            ):
                invalid_sources.add(source_id)
                continue
            if not _bounded_exact_text(
                media.get("uuid"), 512
            ) or not _bounded_exact_text(media.get("content_hash"), 512):
                raise ValueError("invalid media identity")
            live_rows.append(row)

        try:
            projection = build_source_status_projection(
                workspace_id=self.workspace_id,
                sources=live_rows,
                media_db=_FrozenMediaStatusReader(media_rows),
                jobs=[],
            )
        except Exception:  # noqa: BLE001 - projection failures must not disclose owner data.
            raise _SharedWorkspaceDataUnavailable() from None
        status_index = {
            str(status.get("id")): status
            for status in projection.get("sources", [])
            if isinstance(status, dict)
        }

        items: list[SharedSourceSnapshotItem] = []
        for row in sorted_rows:
            source_id = _source_row_id(row)
            media_id = row_media_ids.get(source_id)
            media = media_rows.get(media_id) if media_id is not None else None
            readiness_class = _readiness_class(status_index.get(source_id))
            if source_id in invalid_sources or media is None or readiness_class is None:
                if reject_nonqueryable:
                    raise _NonQueryableSource()
                continue
            items.append(
                SharedSourceSnapshotItem(
                    source_id=source_id,
                    media_id=media_id,
                    media_uuid=_bounded_exact_text(media.get("uuid"), 512),
                    content_hash=_bounded_exact_text(media.get("content_hash"), 512),
                    readiness_class=readiness_class,
                )
            )
        return tuple(items)


def _normalize_source_ids(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, str) or not isinstance(values, Sequence):
        raise SharedWorkspaceSourceScopeInvalid()
    normalized: list[str] = []
    for value in values:
        if not isinstance(value, str):
            raise SharedWorkspaceSourceScopeInvalid()
        source_id = _bounded_exact_text(value, _MAX_SOURCE_ID_CHARS)
        if not source_id:
            raise SharedWorkspaceSourceScopeInvalid()
        normalized.append(source_id)
    if len(normalized) != len(set(normalized)):
        raise SharedWorkspaceSourceScopeInvalid()
    return tuple(normalized)


def _source_row_id(row: dict[str, Any]) -> str:
    source_id = _bounded_exact_text(row.get("id"), _MAX_SOURCE_ID_CHARS)
    if not source_id:
        raise ValueError("invalid source row")
    return source_id


def _index_source_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        source_id = _source_row_id(row)
        if source_id in indexed:
            raise ValueError("duplicate source row")
        indexed[source_id] = row
    return indexed


def _positive_int(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("invalid integer")
    if isinstance(value, int):
        result = value
    elif isinstance(value, str) and value.strip().isdigit():
        result = int(value.strip())
    else:
        raise ValueError("invalid integer")
    if result <= 0:
        raise ValueError("invalid integer")
    return result


def _bounded_exact_text(value: Any, limit: int) -> str:
    if not isinstance(value, str):
        return ""
    if (
        not value
        or value != value.strip()
        or len(value) > limit
        or not value.isprintable()
    ):
        return ""
    return value


def _readiness_class(status: dict[str, Any] | None) -> str | None:
    if not isinstance(status, dict):
        return None
    state = str(status.get("state") or "").strip().lower()
    readiness = status.get("readiness") or {}
    citation_ready = bool(readiness.get("citation_ready"))
    tool_accessible = bool(readiness.get("tool_accessible"))
    if state == "queryable" and citation_ready and tool_accessible:
        return "queryable"
    if (
        state == "partially_queryable"
        and bool(readiness.get("text_extracted"))
        and bool(readiness.get("fts_ready"))
        and citation_ready
        and tool_accessible
    ):
        return "partially_queryable"
    return None


def _build_snapshot(
    *,
    mode: SourceMode,
    items: Sequence[SharedSourceSnapshotItem],
) -> SharedSourceSnapshot:
    sorted_items = tuple(sorted(items, key=lambda item: item.source_id))
    payload = [
        {
            "source_id": item.source_id,
            "media_id": item.media_id,
            "media_uuid": item.media_uuid,
            "content_hash": item.content_hash,
            "readiness_class": item.readiness_class,
        }
        for item in sorted_items
    ]
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return SharedSourceSnapshot(
        mode=mode,
        items=sorted_items,
        snapshot_hash=hashlib.sha256(encoded).hexdigest(),
    )


def _snapshot_is_internally_valid(snapshot: Any) -> bool:
    if not isinstance(snapshot, SharedSourceSnapshot) or snapshot.mode not in {"all", "include"}:
        return False
    if not snapshot.items or len(snapshot.items) > _MAX_SOURCES:
        return False
    source_ids: set[str] = set()
    for item in snapshot.items:
        if (
            not isinstance(item, SharedSourceSnapshotItem)
            or not _bounded_exact_text(item.source_id, _MAX_SOURCE_ID_CHARS)
            or item.source_id in source_ids
            or not isinstance(item.media_id, int)
            or isinstance(item.media_id, bool)
            or item.media_id <= 0
            or not _bounded_exact_text(item.media_uuid, 512)
            or not _bounded_exact_text(item.content_hash, 512)
            or item.readiness_class not in {"queryable", "partially_queryable"}
        ):
            return False
        source_ids.add(item.source_id)
    rebuilt = _build_snapshot(mode=snapshot.mode, items=snapshot.items)
    return rebuilt.items == snapshot.items and rebuilt.snapshot_hash == snapshot.snapshot_hash


def _result_value(result: Any, name: str, default: Any) -> Any:
    if isinstance(result, dict):
        return result.get(name, default)
    return getattr(result, name, default)


def _is_nonempty(value: Any) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, str | bytes | list | tuple | set | dict):
        return bool(value)
    return bool(value)


_ALLOWED_RETRIEVAL_METADATA = frozenset(
    {
        "original_query",
        "retrieval_cache_hit",
        "generation_executed",
        "explicit_source_selection",
        "sources_requested",
        "sources_searched",
        "documents_retrieved",
        "retrieval_guidance",
        "retrieval_plan",
        "answer_generation_skipped",
    }
)

_DERIVED_RESULT_FIELDS = (
    "citations",
    "academic_citations",
    "chunk_citations",
    "feedback_id",
    "security_report",
    "claims",
    "factuality",
    "verification_report",
    "retrieval_metrics",
    "faithfulness",
    "query_classification",
    "reformulated_query",
    "research_summary",
    "suggestions",
    "images",
    "videos",
)


def _result_has_derived_outputs(result: Any) -> bool:
    return any(
        _is_nonempty(_result_value(result, field_name, None))
        for field_name in _DERIVED_RESULT_FIELDS
    )


def _metadata_is_locked_retrieval(
    metadata: Any,
    *,
    query: str,
    media_count: int,
    document_count: int,
    index_namespace: str,
) -> bool:
    if not isinstance(metadata, dict):
        return False
    if set(metadata) - _ALLOWED_RETRIEVAL_METADATA:
        return False
    if "original_query" in metadata and metadata["original_query"] != query:
        return False
    for key in ("generation_executed", "retrieval_cache_hit"):
        if key in metadata and metadata[key] is not False:
            return False
    for key in ("sources_requested", "sources_searched"):
        reported = metadata.get(key)
        if reported is not None and reported != ["media_db"]:
            return False
    if "documents_retrieved" in metadata and (
        not isinstance(metadata["documents_retrieved"], int)
        or isinstance(metadata["documents_retrieved"], bool)
        or metadata["documents_retrieved"] != document_count
    ):
        return False
    if "retrieval_guidance" in metadata and not isinstance(
        metadata["retrieval_guidance"], str
    ):
        return False
    if "answer_generation_skipped" in metadata and (
        document_count != 0 or metadata["answer_generation_skipped"] != "no_documents"
    ):
        return False

    explicit_scope = metadata.get("explicit_source_selection")
    if explicit_scope is not None and explicit_scope != {
        "enabled": True,
        "requested_sources": ["media_db"],
        "resolved_sources": ["media_db"],
        "include_media_ids_count": media_count,
        "include_note_ids_count": 0,
        "scope_intersection_empty": False,
        "cache_disabled": False,
    }:
        return False
    retrieval_plan = metadata.get("retrieval_plan")
    return retrieval_plan is None or retrieval_plan == {
        "query": query,
        "sources": ["media_db"],
        "search_mode": "fts",
        "top_k": _MAX_EVIDENCE,
        "index_namespace": index_namespace,
    }


def _validate_document(
    document: Any,
    canonical_source_by_media: dict[int, tuple[str, str]],
) -> tuple[tuple[int, str], dict[str, Any]]:
    metadata = _document_value(document, "metadata", {})
    if not isinstance(metadata, dict):
        raise SharedWorkspaceRetrievalUnavailable()
    source = _document_value(document, "source", None)
    if hasattr(source, "value"):
        source = source.value
    if source != "media_db":
        raise SharedWorkspaceRetrievalUnavailable()
    metadata_source = metadata.get("source")
    if hasattr(metadata_source, "value"):
        metadata_source = metadata_source.value
    if metadata_source is not None and metadata_source != source:
        raise SharedWorkspaceRetrievalUnavailable()
    try:
        media_id = _positive_int(metadata.get("media_id"))
    except ValueError:
        raise SharedWorkspaceRetrievalUnavailable() from None
    canonical_source = canonical_source_by_media.get(media_id)
    if not canonical_source:
        raise SharedWorkspaceRetrievalUnavailable()
    source_id, source_title = canonical_source

    top_identity = _document_value(document, "id", None)
    metadata_identity = metadata.get("chunk_id")
    if top_identity is None and metadata_identity is None:
        raise SharedWorkspaceRetrievalUnavailable()
    canonical_top_identity = (
        _canonical_chunk_identity(top_identity) if top_identity is not None else None
    )
    canonical_metadata_identity = (
        _canonical_chunk_identity(metadata_identity)
        if metadata_identity is not None
        else None
    )
    if (
        canonical_top_identity is not None
        and canonical_metadata_identity is not None
        and canonical_top_identity != canonical_metadata_identity
    ):
        raise SharedWorkspaceRetrievalUnavailable()
    chunk_identity = canonical_top_identity or canonical_metadata_identity
    if chunk_identity is None:
        raise SharedWorkspaceRetrievalUnavailable()

    raw_content = _document_value(document, "content", None)
    if not isinstance(raw_content, str):
        raise SharedWorkspaceRetrievalUnavailable()
    score = _finite_float(_document_value(document, "score", 0.0))
    chunk_index = _strict_document_locator(
        document,
        metadata,
        ("chunk_index",),
    )
    start_char = _strict_document_locator(
        document,
        metadata,
        ("start_char", "start"),
    )
    end_char = _strict_document_locator(
        document,
        metadata,
        ("end_char", "end"),
    )
    if start_char is not None and end_char is not None and end_char < start_char:
        raise SharedWorkspaceRetrievalUnavailable()
    return (media_id, chunk_identity), {
        "source_id": source_id,
        "source_title": source_title,
        "content": raw_content,
        "score": score,
        "chunk_index": chunk_index,
        "start_char": start_char,
        "end_char": end_char,
    }


def _document_value(document: Any, name: str, default: Any) -> Any:
    if isinstance(document, dict):
        return document.get(name, default)
    return getattr(document, name, default)


def _finite_float(value: Any) -> float:
    if isinstance(value, bool):
        raise SharedWorkspaceRetrievalUnavailable()
    try:
        score = float(value)
    except (TypeError, ValueError):
        raise SharedWorkspaceRetrievalUnavailable() from None
    if not math.isfinite(score):
        raise SharedWorkspaceRetrievalUnavailable()
    return score


def _canonical_chunk_identity(value: Any) -> str:
    if isinstance(value, bool):
        raise SharedWorkspaceRetrievalUnavailable()
    if isinstance(value, int):
        if value < 0:
            raise SharedWorkspaceRetrievalUnavailable()
        identity = str(value)
    elif isinstance(value, str):
        identity = _bounded_exact_text(value, _MAX_CHUNK_ID_CHARS)
    else:
        raise SharedWorkspaceRetrievalUnavailable()
    if not identity or len(identity) > _MAX_CHUNK_ID_CHARS:
        raise SharedWorkspaceRetrievalUnavailable()
    return identity


def _strict_document_locator(
    document: Any,
    metadata: dict[str, Any],
    names: tuple[str, ...],
) -> int | None:
    values: list[int] = []
    try:
        for name in names:
            if metadata.get(name) is not None:
                values.append(_positive_or_zero_int(metadata[name]))
            document_value = _document_value(document, name, None)
            if document_value is not None:
                values.append(_positive_or_zero_int(document_value))
    except ValueError:
        raise SharedWorkspaceRetrievalUnavailable() from None
    if not values:
        return None
    if any(value != values[0] for value in values[1:]):
        raise SharedWorkspaceRetrievalUnavailable()
    return values[0]


def _positive_or_zero_int(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("invalid locator")
    if isinstance(value, int):
        result = value
    elif isinstance(value, str) and value.isdigit():
        result = int(value)
        if str(result) != value:
            raise ValueError("invalid locator")
    else:
        raise ValueError("invalid locator")
    if result < 0 or result > _MAX_LOCATOR:
        raise ValueError("invalid locator")
    return result


def _safe_title(value: Any) -> str:
    raw = str(value or "")
    printable = "".join(character if character.isprintable() else " " for character in raw)
    normalized = " ".join(printable.split()).strip()
    return (normalized or "Shared source")[:_MAX_SOURCE_TITLE_CHARS]


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, int | float):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on", "enabled"}


__all__ = [
    "SHARED_RETRIEVAL_POLICY",
    "SharedRetrievalPolicy",
    "SharedSourceSnapshot",
    "SharedSourceSnapshotItem",
    "SharedWorkspaceChatService",
    "SharedWorkspaceChatServiceError",
    "SharedWorkspaceNoRelevantEvidence",
    "SharedWorkspaceRetrievalUnavailable",
    "SharedWorkspaceSourceChanged",
    "SharedWorkspaceSourceScopeInvalid",
    "SharedWorkspaceSourceSubsetRequired",
    "VerifiedSharedEvidence",
]
