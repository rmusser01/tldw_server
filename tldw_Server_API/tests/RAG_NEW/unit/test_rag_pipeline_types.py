"""RagRunConfig: one typed field per ``unified_rag_pipeline`` parameter and ``**kwargs`` key (TASK-13950.24)."""

from __future__ import annotations

import dataclasses
import inspect
from collections import Counter

import pytest

from tldw_Server_API.app.api.v1.schemas.rag_schemas_unified import UnifiedRAGRequest
from tldw_Server_API.app.core.RAG.rag_service.pipeline import run_config
from tldw_Server_API.app.core.RAG.rag_service.pipeline.run_config import (
    FIELD_GROUP,
    RagRunConfig,
    hidden_kwarg_values,
)
from tldw_Server_API.app.core.RAG.rag_service.pipeline.run_state import MUTATED_PARAMS, PipelineState
from tldw_Server_API.app.core.RAG.rag_service.unified_pipeline import unified_rag_pipeline

pytestmark = pytest.mark.unit

# R12: parameters per group, and the 13 ``**kwargs`` keys that became typed fields.
PARAMS_PER_GROUP = {
    "request": 3, "scope": 10, "runtime": 11, "retrieval": 24, "processing": 20, "query_prep": 20,
    "cache": 4, "safety": 14, "rerank": 8, "generation": 18, "citations": 6, "verification": 27,
    "self_correct": 28, "research": 12, "observability": 14, "resilience": 5, "inert": 9,
}  # fmt: skip
HIDDEN_KEYS = {
    "request": {"metadata", "include_sources", "include_metadata"},
    "query_prep": {"enable_expansion"},
    "scope": {"workspace_id"},
    "runtime": {"prompts_db_path", "world_books_db_path", "chat_dictionaries_db_path", "prompts_db", "faithfulness_llm"},
    "verification": {"claims_budget_usd", "claims_budget_tokens", "claims_budget_strict"},
}
HIDDEN = {k for ks in HIDDEN_KEYS.values() for k in ks}
PARAMS = {n: p for n, p in inspect.signature(unified_rag_pipeline).parameters.items() if n != "kwargs"}
FIELDS = {f.name: f for _, cls in run_config._GROUPS for f in dataclasses.fields(cls)}

# The request schema and the config overlap on 194 names; these are the rest, frozen so a new field on either
# side is a conscious decision.
REQUEST_ONLY = frozenset(
    {"collection_id", "corpus", "enable_analytics", "explain_only", "strategy", "use_connection_pool", "use_embedding_cache"}
    | {f"agentic_{s}" for s in (
        "adaptive_budgets", "cache_ttl_sec", "coverage_target", "debug_trace", "enable_metrics",
        "enable_query_decomposition", "enable_section_index", "enable_semantic_within", "enable_table_support",
        "enable_tools", "enable_vlm_late_chunking", "extractive_only", "max_redundancy", "max_tokens_read",
        "max_tool_calls", "min_corroborating_docs", "prefer_structural_anchors", "provider_embedding_model_id",
        "quote_spans", "subgoal_max", "time_budget_sec", "top_k_docs", "use_llm_planner",
        "use_provider_embeddings_within", "vlm_backend", "vlm_detect_tables_only", "vlm_late_chunk_top_k_docs",
        "vlm_max_pages", "window_chars",
    )}
)  # fmt: skip
CONFIG_ONLY = frozenset(
    {
        "_adaptive_rerun", "abstention_policy", "adaptive_hybrid_weights", "auto_temporal_filters", "cache_ttl",
        "calibrator_version", "chacha_db", "character_db_path", "credential_runtime", "date_range",
        "enable_date_filter", "enable_enhanced_chunking", "enable_gap_analysis", "enable_graph_retrieval",
        "enable_hyde", "enable_learned_fusion", "enable_precomputed_spans", "enable_prf", "enable_streaming",
        "fallback_on_error", "filter_media_types", "graph_alpha", "graph_neighbors_k", "graph_version",
        "hyde_model", "hyde_provider", "include_retrieval_diagnostics", "kanban_db_path",
        "max_followup_searches", "media_db", "media_db_path", "notes_db_path", "prf_alpha", "prf_sources",
        "prf_terms", "prf_top_n", "research_progress_callback", "resolved_request", "retrieval_plan",
        "sql_retriever",
    }
    | HIDDEN - {"workspace_id"}
)  # fmt: skip


def test_every_parameter_is_exactly_one_config_field() -> None:
    assert len(PARAMS) == 233
    assert set(FIELD_GROUP) == set(PARAMS) | HIDDEN
    assert len(FIELD_GROUP) == 246 == len(FIELDS)
    sizes = Counter(g for k, g in FIELD_GROUP.items() if k in PARAMS)
    assert dict(sizes) == PARAMS_PER_GROUP
    hidden = {g: {k for k, kg in FIELD_GROUP.items() if kg == g and k in HIDDEN} for g in HIDDEN_KEYS}
    assert hidden == HIDDEN_KEYS


def test_config_defaults_equal_the_signature_defaults() -> None:
    for name, param in PARAMS.items():
        field = FIELDS[name]
        if param.default is inspect.Parameter.empty:
            assert field.default is dataclasses.MISSING, name
        elif param.default is None or isinstance(param.default, bool):
            assert field.default is param.default, name
        else:
            assert field.default == param.default, name
        assert str(field.type) == str(param.annotation) or field.type == param.annotation, name


def test_hidden_kwarg_values_reads_the_thirteen_keys() -> None:
    want = dict.fromkeys(HIDDEN) | {"include_sources": True, "include_metadata": True, "enable_expansion": False}
    assert hidden_kwarg_values() == want
    assert len(want) == 13


def _sentinel_call_locals() -> tuple[dict[str, object], dict[str, object]]:
    values = {k: object() for k in FIELD_GROUP}
    named = {k: v for k, v in values.items() if k not in HIDDEN}
    return values, {**named, "kwargs": {k: values[k] for k in HIDDEN}}


def test_from_call_locals_keeps_every_value_by_identity() -> None:
    values, call_locals = _sentinel_call_locals()
    out = RagRunConfig.from_call_locals(call_locals).as_kwargs()
    assert set(out) == set(values) and len(out) == 246
    assert all(out[k] is v for k, v in values.items())


def test_unknown_keys_are_ignored() -> None:
    cfg = RagRunConfig.from_kwargs(query="q", corpus="x")
    assert "corpus" not in cfg.as_kwargs()
    assert not any(getattr(g, "corpus", None) for g in (getattr(cfg, n) for n in FIELD_GROUP.values()))
    assert cfg.request.query == "q"


def test_from_kwargs_honours_exactly_the_config_fields() -> None:
    marker = object()
    for key in HIDDEN:
        got = RagRunConfig.from_kwargs(query="q", **{key: marker}).as_kwargs()[key]
        assert got is hidden_kwarg_values(**{key: marker})[key] is marker
    for key in FIELD_GROUP:
        if key != "query":
            assert RagRunConfig.from_kwargs(query="q", **{key: marker}).as_kwargs()[key] is marker
    assert RagRunConfig.from_kwargs(query="q", not_a_field=marker).as_kwargs() == RagRunConfig.from_kwargs(query="q").as_kwargs()


def test_an_explicit_none_beats_the_default() -> None:
    cfg = RagRunConfig.from_kwargs(query="q", include_sources=None, top_k=None)
    assert cfg.request.include_sources is None and cfg.retrieval.top_k is None


def test_config_is_frozen() -> None:
    cfg = RagRunConfig.from_kwargs(query="q")
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.retrieval = cfg.retrieval  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.retrieval.top_k = 1  # type: ignore[misc]


def test_config_tracks_the_request_schema() -> None:
    model = UnifiedRAGRequest.model_fields
    shared = set(FIELD_GROUP) & set(model)
    assert len(shared) == 194
    assert set(model) - set(FIELD_GROUP) == REQUEST_ONLY and len(REQUEST_ONLY) == 36
    assert set(FIELD_GROUP) - set(model) == CONFIG_ONLY and len(CONFIG_ONLY) == 52

    def same(a: object, b: object) -> bool:
        return a is b if a is None or b is None or isinstance(a, bool) or isinstance(b, bool) else a == b

    differing = {
        n
        for n in shared
        if model[n].is_required() != (FIELDS[n].default is dataclasses.MISSING)
        or (not model[n].is_required() and not same(model[n].default, FIELDS[n].default))
    }
    assert differing == {"sources", "enable_generation"}


FROZEN_MUTATED = frozenset(
    {
        "adaptive_time_budget_sec", "date_range", "enable_cache", "enable_date_filter", "enable_discussion_search",
        "enable_hyde", "enable_multi_vector_passages", "enable_numeric_fidelity", "enable_parent_expansion",
        "enable_reranking", "enable_web_fallback", "expand_query", "fts_level", "generation_prompt", "hybrid_alpha",
        "hyde_model", "hyde_provider", "include_parent_document", "low_confidence_behavior", "media_db_path",
        "mv_max_spans", "mv_span_chars", "mv_stride", "nli_model", "numeric_fidelity_behavior",
        "parent_context_size", "parent_max_tokens", "query", "require_hard_citations", "reranking_strategy",
        "resolved_request", "retrieval_plan", "sql_retriever", "strict_extractive", "top_k",
    }
)  # fmt: skip


def test_mutated_parameters_are_frozen() -> None:
    assert len(MUTATED_PARAMS) == 35 and {n for _, n in MUTATED_PARAMS} == FROZEN_MUTATED
    assert all(FIELD_GROUP[n] == g for g, n in MUTATED_PARAMS)


def test_unset_state_slot_reads_like_an_unbound_local() -> None:
    cfg = RagRunConfig.from_kwargs(query="q", top_k=7)
    state = PipelineState(cfg)
    with pytest.raises(UnboundLocalError, match="local variable 'result' where it is not associated with a value"):
        _ = state.result
    with pytest.raises(AttributeError):
        _ = state.nope
    assert state.top_k == cfg.retrieval.top_k == 7 and state.query == "q" and state.rcfg is cfg
    state.result = 1
    assert state.result == 1
    del state.result
    with pytest.raises(UnboundLocalError):
        _ = state.result
