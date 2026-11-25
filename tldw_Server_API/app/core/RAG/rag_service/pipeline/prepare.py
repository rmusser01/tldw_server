"""prepare code moved from unified_pipeline.py (TASK-13950.24)."""

import contextlib
import time
import uuid
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Literal, Optional, cast

from tldw_Server_API.app.core.RAG.rag_service import unified_pipeline as up

from .run_config import RagRunConfig

if TYPE_CHECKING:
    from .run_state import PipelineState


def prepare_request(rcfg: RagRunConfig, state: "PipelineState") -> None:
    """Scope and credential checks, request resolution and the retrieval plan snapshot."""
    ambient_scope = up.get_scope()
    state.authorization_cache_scope: dict[str, object] | None = None
    state.authorization_cache_error: str | None = None
    if ambient_scope is not None:
        if isinstance(ambient_scope, up.ScopeContext):
            try:
                state.authorization_cache_scope = up.content_authorization_cache_scope(
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
        state.resolved_request = up.resolve_legacy_standard_pipeline_request(
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
        state.retrieval_plan = up.build_retrieval_plan(state.resolved_request)

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


def apply_search_depth_presets(rcfg: RagRunConfig, state: "PipelineState") -> None:
    """Alias normalization and the search-depth mode presets."""
    # Normalize common alias/compat args
    explicit_fields = state.resolved_request.explicit_fields
    explicit_payload = state.resolved_request.payload
    if "expand_query" in explicit_fields:
        state.expand_query = explicit_payload["expand_query"]
    else:
        state.expand_query = state.expand_query or rcfg.query_prep.enable_expansion
    state.media_db_path = up._resolve_sqlite_rag_db_path(state.media_db_path, db=rcfg.runtime.media_db)

    # ========== SEARCH DEPTH MODE PRESETS ==========
    # Apply parameter presets based on search_depth_mode. Individual parameter
    # overrides take precedence; legacy calls retain their default-value heuristic.
    state.retrieval_scalar_overrides_applied = False
    if rcfg.request.search_depth_mode is not None:
        presets = up._SEARCH_DEPTH_PRESETS.get(rcfg.request.search_depth_mode)
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


def init_run(rcfg: RagRunConfig, state: "PipelineState") -> None:
    """Result, timing, retrieval-cache eligibility and monitoring setup."""
    # Initialize result and timing
    state.start_time = time.time()
    state.result = up.UnifiedSearchResult(
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
        state.metrics = up.QueryMetrics(
            query_id=str(uuid.uuid4()),
            query=state.query,
            timestamp=state.start_time,
            total_duration=0.0,
        )

    state.resolved_data_sources: list[up.DataSource] = []
