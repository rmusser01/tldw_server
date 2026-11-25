"""TASK-13950.24 PR D3: every local that crosses a planned stage boundary lives on ``PipelineState``.

``cells`` is Appendix B's partition at this PR's base (bcf2e8bf21); ``rewrite`` and ``publish`` come from
``rag_split shared`` over those cells.
"""

MANIFEST = {
    "base": "bcf2e8bf21",
    "cells": [
        # prepare (L)
        [2355, 2462], [2514, 2565], [2568, 2644],
        # scope, query_prep, research (K)
        [2649, 2724], [2728, 2781], [2783, 3010], [3015, 3235], [3240, 3303], [3306, 3415], [3418, 3535], [3538, 3663],
        # retrieval (J): lookup, run_retrieval and its five leaves
        [3667, 3727], [3730, 4457], [3756, 3895], [3901, 4048], [4050, 4105], [4108, 4308], [4310, 4360],
        # augment, filters (I2)
        [4459, 4594], [4597, 4775], [4778, 4904], [4907, 5045],
        # processing, self_correct (I1)
        [5048, 5228], [5231, 5347], [5351, 5438], [5442, 5619],
        # rerank (H) and its three leaves; post_retrieval (H)
        [5622, 5996], [5654, 5792], [5793, 5874], [5877, 5963], [6000, 6079], [6082, 6230], [6232, 6435],
        # generation (G) and its four leaves
        [6439, 6492], [6494, 7020], [6551, 6609], [6611, 6856], [6857, 6918], [6952, 7020],
        # verification (F): citations, claims and its two leaves, numeric fidelity, post verification and its two leaves
        [7024, 7114], [7115, 7543], [7198, 7336], [7337, 7515], [7546, 7695], [7699, 8061], [7708, 7810], [7814, 8036],
        # post_processing (E): finish, store, finalize (the `finally` body)
        [8065, 8249], [8252, 8332], [8362, 8564],
    ],
    "publish": ['_execute_retrieval_variant', '_analyze', '_mark_claims_used', '_run_claims_engine', '_retrieve_for_claim'],
    "rewrite": {
        "_analyze": "state._analyze",
        "_classification": "state._classification",
        "_execute_retrieval_variant": "state._execute_retrieval_variant",
        "_mark_claims_used": "state._mark_claims_used",
        "_otel_span": "state._otel_span",
        "_otel_span_gen": "state._otel_span_gen",
        "_otel_span_rk": "state._otel_span_rk",
        "_retrieve_for_claim": "state._retrieve_for_claim",
        "_run_claims_engine": "state._run_claims_engine",
        "_skip_local_retrieval": "state._skip_local_retrieval",
        "_skip_retrieval_reason": "state._skip_retrieval_reason",
        "_skip_retrieval_stack": "state._skip_retrieval_stack",
        "analysis": "state.analysis",
        "analysis_complexity_val": "state.analysis_complexity_val",
        "analysis_domain": "state.analysis_domain",
        "analysis_intent_val": "state.analysis_intent_val",
        "authorization_cache_error": "state.authorization_cache_error",
        "base_retrieval_executed": "state.base_retrieval_executed",
        "cache_retrieval_snapshot": "state.cache_retrieval_snapshot",
        "claims_engine_type": "state.claims_engine_type",
        "claims_handle": "state.claims_handle",
        "claims_payload": "state.claims_payload",
        "claims_state": "state.claims_state",
        "config": "state.config",
        "context": "state.context",
        "context_docs": "state.context_docs",
        "documents": "state.documents",
        "effective_enable_generation": "state.effective_enable_generation",
        "engine": "state.engine",
        "expanded_queries": "state.expanded_queries",
        "factuality_payload": "state.factuality_payload",
        "gated_generation": "state.gated_generation",
        "generation_start": "state.generation_start",
        "generator": "state.generator",
        "hyde_vector": "state.hyde_vector",
        "include_rerank_snapshots": "state.include_rerank_snapshots",
        "job_budget": "state.job_budget",
        "job_context": "state.job_context",
        "llm_client": "state.llm_client",
        "low_confidence": "state.low_confidence",
        "metrics": "state.metrics",
        "query_expansion_corpus": "state.query_expansion_corpus",
        "rerank_config": "state.rerank_config",
        "rerank_debug_limit": "state.rerank_debug_limit",
        "rerank_start": "state.rerank_start",
        "reranker": "state.reranker",
        "reranker_credential_handle": "state.reranker_credential_handle",
        "resolved_data_sources": "state.resolved_data_sources",
        "retrieval_cache_eligible": "state.retrieval_cache_eligible",
        "retrieval_scalar_overrides_applied": "state.retrieval_scalar_overrides_applied",
        "retrieval_start": "state.retrieval_start",
        "retriever": "state.retriever",
        "selected_strategy": "state.selected_strategy",
        "skip_claims_due_to_groundedness": "state.skip_claims_due_to_groundedness",
        "standard_evidence_coordinated": "state.standard_evidence_coordinated",
        "start_time": "state.start_time",
        "vres": "state.vres",
    },
}
