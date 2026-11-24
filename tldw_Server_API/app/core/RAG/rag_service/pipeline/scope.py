"""scope code moved from unified_pipeline.py (TASK-13950.24)."""

from collections import Counter
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from tldw_Server_API.app.core.RAG.rag_service import unified_pipeline as up

from ..pipeline import run_state as _run_state
from .run_config import RagRunConfig

if TYPE_CHECKING:
    from .run_state import PipelineState


def resolve_sources(rcfg: RagRunConfig, state: "PipelineState") -> None:
    """Resolve data sources and explicit source selection; raises EarlyReturn on an invalid source."""
    try:
        state.retrieval_sources = up._normalize_pipeline_sources(state.retrieval_sources)
        state.resolved_data_sources = up._sources_to_data_sources(state.retrieval_sources)
        explicit_include_sources = up._explicit_include_data_sources(
            include_media_ids=rcfg.scope.include_media_ids,
            include_note_ids=rcfg.scope.include_note_ids,
        )
        if explicit_include_sources:
            original_retrieval_sources = list(state.retrieval_sources)
            requested_sources = set(state.resolved_data_sources)
            scoped_data_sources = [source for source in explicit_include_sources if source in requested_sources]
            scope_intersection_empty = not scoped_data_sources
            state.resolved_data_sources = (
                list(explicit_include_sources) if scope_intersection_empty else scoped_data_sources
            )
            state.retrieval_sources = [up._DATASOURCE_TO_CANONICAL_SOURCE[source] for source in state.resolved_data_sources]
            state.resolved_request.payload["sources"] = list(state.retrieval_sources)
            cache_disabled_for_scope = bool(state.enable_cache)
            if cache_disabled_for_scope:
                state.enable_cache = False
                state.retrieval_cache_eligible = False
                state.result.metadata["cache_bypassed"] = {
                    "reason": "explicit_source_selection",
                }
            if state.retrieval_plan is not None:
                scoped_retrieval_plan = replace(
                    state.retrieval_plan,
                    sources=tuple(state.retrieval_sources),
                    index_namespace=None,
                )
                state.retrieval_plan = replace(
                    scoped_retrieval_plan,
                    index_namespace=state.planned_index_namespace(scoped_retrieval_plan),
                )
                state.retrieval_index_namespace = state.planned_index_namespace(state.retrieval_plan)
            state.result.metadata["explicit_source_selection"] = {
                "enabled": True,
                "requested_sources": original_retrieval_sources,
                "resolved_sources": list(state.retrieval_sources),
                "include_media_ids_count": len(rcfg.scope.include_media_ids or []),
                "include_note_ids_count": len(rcfg.scope.include_note_ids or []),
                "scope_intersection_empty": scope_intersection_empty,
                "cache_disabled": cache_disabled_for_scope,
            }
        state.result.metadata["sources_requested"] = list(state.retrieval_sources)
    except ValueError as exc:
        state.result.errors.append(f"invalid_source: {exc}")
        state.result.metadata["source_validation_error"] = str(exc)
        raise _run_state.EarlyReturn() from exc

    if (
        state.sql_retriever is None
        and up.SQLRetriever is not None
        and any(src == up.DataSource.SQL for src in state.resolved_data_sources)
    ):
        sql_db_path = up._resolve_sqlite_rag_db_path(state.media_db_path, db=rcfg.runtime.media_db)
        if sql_db_path:
            try:
                state.sql_retriever = up.SQLRetriever(
                    db_path=str(sql_db_path),
                    target_id=str(rcfg.scope.sql_target_id or "media_db"),
                )
            except (
                AttributeError,
                ConnectionError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as exc:
                state.result.errors.append(f"SQL retriever init failed: {exc}")

    state.source_status_retriever: Any = None
    state.source_failures: set[up.DataSource] = set()
    state.cumulative_filtered_artifact_counts: dict[str, int] = {}
    state.security_excluded_counts: Counter[str] = Counter()
