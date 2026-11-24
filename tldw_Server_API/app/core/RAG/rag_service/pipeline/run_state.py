"""``PipelineState``: the shared mutable state of one ``unified_rag_pipeline`` call (TASK-13950.24).

Each slot is one local of the old function: the 35 parameters the body reassigns (copied from the
frozen ``RagRunConfig``) and ``result``. A slot that is not set reads like an unbound local, so a
read that fails today fails the same way: ``UnboundLocalError``, which the pipeline's ``except``
ladder does not catch, rather than an ``AttributeError``, which it would.
"""

from typing import Any

from .run_config import RagRunConfig

# (group, name) of every parameter the pipeline body assigns; frozen by test_rag_pipeline_types.py.
MUTATED_PARAMS: tuple[tuple[str, str], ...] = (
    ("verification", "adaptive_time_budget_sec"),
    ("scope", "date_range"),
    ("cache", "enable_cache"),
    ("scope", "enable_date_filter"),
    ("research", "enable_discussion_search"),
    ("query_prep", "enable_hyde"),
    ("retrieval", "enable_multi_vector_passages"),
    ("verification", "enable_numeric_fidelity"),
    ("processing", "enable_parent_expansion"),
    ("rerank", "enable_reranking"),
    ("self_correct", "enable_web_fallback"),
    ("query_prep", "expand_query"),
    ("retrieval", "fts_level"),
    ("generation", "generation_prompt"),
    ("retrieval", "hybrid_alpha"),
    ("query_prep", "hyde_model"),
    ("query_prep", "hyde_provider"),
    ("processing", "include_parent_document"),
    ("verification", "low_confidence_behavior"),
    ("runtime", "media_db_path"),
    ("retrieval", "mv_max_spans"),
    ("retrieval", "mv_span_chars"),
    ("retrieval", "mv_stride"),
    ("verification", "nli_model"),
    ("verification", "numeric_fidelity_behavior"),
    ("processing", "parent_context_size"),
    ("processing", "parent_max_tokens"),
    ("request", "query"),
    ("citations", "require_hard_citations"),
    ("rerank", "reranking_strategy"),
    ("runtime", "resolved_request"),
    ("runtime", "retrieval_plan"),
    ("runtime", "sql_retriever"),
    ("generation", "strict_extractive"),
    ("retrieval", "top_k"),
)


class PipelineState:
    """One call's mutable state; ``rcfg`` is the frozen configuration it started from."""

    __slots__: tuple[str, ...] = (
        "rcfg",
        "adaptive_time_budget_sec",
        "date_range",
        "enable_cache",
        "enable_date_filter",
        "enable_discussion_search",
        "enable_hyde",
        "enable_multi_vector_passages",
        "enable_numeric_fidelity",
        "enable_parent_expansion",
        "enable_reranking",
        "enable_web_fallback",
        "expand_query",
        "fts_level",
        "generation_prompt",
        "hybrid_alpha",
        "hyde_model",
        "hyde_provider",
        "include_parent_document",
        "low_confidence_behavior",
        "media_db_path",
        "mv_max_spans",
        "mv_span_chars",
        "mv_stride",
        "nli_model",
        "numeric_fidelity_behavior",
        "parent_context_size",
        "parent_max_tokens",
        "query",
        "require_hard_citations",
        "reranking_strategy",
        "resolved_request",
        "retrieval_plan",
        "sql_retriever",
        "strict_extractive",
        "top_k",
        "result",
    )

    def __init__(self, rcfg: RagRunConfig) -> None:
        self.rcfg = rcfg
        for group, name in MUTATED_PARAMS:
            setattr(self, name, getattr(getattr(rcfg, group), name))

    def __getattr__(self, name: str) -> Any:
        """Only reached for an unset slot or an unknown name."""
        if name in PipelineState.__slots__:
            raise UnboundLocalError(f"cannot access local variable '{name}' where it is not associated with a value")
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
