"""Typed, frozen run configuration for ``unified_rag_pipeline`` (TASK-13950.24).

``RagRunConfig`` holds every pipeline parameter, grouped into 17 frozen dataclasses, so that
stage functions can read ``rcfg.<group>.<name>`` instead of closing over a 300-line signature.
Each field carries the signature's annotation and default; the 13 keys the pipeline reads from
``**kwargs`` are typed fields too. Unknown keys are ignored, exactly as the pipeline ignores them,
and values are stored by identity (nothing is copied or coerced).

This module imports nothing from ``unified_pipeline``, so any module may import its names.
"""

from collections.abc import Mapping
from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import Any, Callable, Literal, Optional

from ..request_resolution import ResolvedRAGRequest
from ..retrieval_plan import RetrievalPlan


@dataclass(frozen=True, slots=True)
class RequestOptions:
    """The ``request`` group of pipeline parameters."""

    query: str
    search_depth_mode: Optional[Literal['speed', 'balanced', 'quality']] = None
    rag_profile: Optional[Literal['fast', 'balanced', 'accuracy']] = None
    metadata: Any = None  # read from **kwargs
    include_sources: Any = True  # read from **kwargs
    include_metadata: Any = True  # read from **kwargs


@dataclass(frozen=True, slots=True)
class ScopeOptions:
    """The ``scope`` group of pipeline parameters."""

    sources: Optional[list[str]] = None
    sql_target_id: str = 'media_db'
    keyword_filter: Optional[list[str]] = None
    include_media_ids: Optional[list[int]] = None
    include_note_ids: Optional[list[str]] = None
    feedback_user_id: Optional[str] = None
    enable_date_filter: bool = False
    date_range: Optional[dict[str, str]] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    workspace_id: Any = None  # read from **kwargs


@dataclass(frozen=True, slots=True)
class RuntimeHandles:
    """The ``runtime`` group of pipeline parameters."""

    media_db_path: Optional[str] = None
    notes_db_path: Optional[str] = None
    character_db_path: Optional[str] = None
    kanban_db_path: Optional[str] = None
    sql_retriever: Any = None
    retrieval_plan: Optional[RetrievalPlan] = None
    resolved_request: Optional[ResolvedRAGRequest] = None
    credential_runtime: Any = None
    media_db: Any = None
    chacha_db: Any = None
    research_progress_callback: Optional[Callable] = None
    prompts_db_path: Any = None  # read from **kwargs
    world_books_db_path: Any = None  # read from **kwargs
    chat_dictionaries_db_path: Any = None  # read from **kwargs
    prompts_db: Any = None  # read from **kwargs
    faithfulness_llm: Any = None  # read from **kwargs


@dataclass(frozen=True, slots=True)
class RetrievalOptions:
    """The ``retrieval`` group of pipeline parameters."""

    search_mode: Literal['fts', 'vector', 'hybrid'] = 'hybrid'
    fts_level: Literal['media', 'chunk'] = 'media'
    hybrid_alpha: float = 0.7
    adaptive_hybrid_weights: bool = False
    top_k: int = 10
    min_score: float = 0.0
    enable_gap_analysis: bool = False
    max_followup_searches: int = 2
    enable_multi_vector_passages: bool = False
    mv_span_chars: int = 300
    mv_stride: int = 150
    mv_max_spans: int = 8
    mv_flatten_to_spans: bool = False
    enable_precomputed_spans: bool = False
    enable_numeric_table_boost: bool = False
    enable_query_decomposition: bool = False
    max_subqueries: int = 4
    subquery_time_budget_sec: Optional[float] = None
    subquery_doc_budget: Optional[int] = None
    subquery_max_concurrency: int = 3
    index_namespace: Optional[str] = None
    enable_evidence_accumulation: bool = False
    accumulation_max_rounds: int = 3
    accumulation_time_budget_sec: Optional[float] = None


@dataclass(frozen=True, slots=True)
class ProcessingOptions:
    """The ``processing`` group of pipeline parameters."""

    enable_text_late_chunking: bool = False
    chunk_method: Optional[str] = None
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    chunk_language: Optional[str] = None
    enable_table_processing: bool = False
    table_method: Literal['markdown', 'html', 'hybrid'] = 'markdown'
    enable_vlm_late_chunking: bool = False
    vlm_backend: Optional[str] = None
    vlm_detect_tables_only: bool = True
    vlm_max_pages: Optional[int] = None
    vlm_late_chunk_top_k_docs: int = 3
    enable_enhanced_chunking: bool = False
    chunk_type_filter: Optional[list[str]] = None
    enable_parent_expansion: bool = False
    parent_context_size: int = 500
    include_sibling_chunks: bool = False
    sibling_window: int = 1
    include_parent_document: bool = False
    parent_max_tokens: Optional[int] = 1200


@dataclass(frozen=True, slots=True)
class QueryPrepOptions:
    """The ``query_prep`` group of pipeline parameters."""

    enable_intent_routing: bool = False
    auto_temporal_filters: bool = False
    expand_query: bool = False
    expansion_strategies: Optional[list[str]] = None
    spell_check: bool = False
    max_query_variations: int = 3
    enable_prf: bool = False
    prf_terms: int = 10
    prf_sources: Optional[list[str]] = None
    prf_alpha: float = 0.3
    prf_top_n: int = 8
    enable_hyde: bool = False
    hyde_provider: Optional[str] = None
    hyde_model: Optional[str] = None
    enable_pre_retrieval_clarification: Optional[bool] = None
    clarification_timeout_sec: Optional[float] = None
    enable_dynamic_granularity: bool = False
    enable_query_classification: bool = False
    chat_history: Optional[list[dict[str, str]]] = None
    enable_query_reformulation: bool = True
    enable_expansion: Any = False  # read from **kwargs


@dataclass(frozen=True, slots=True)
class CacheOptions:
    """The ``cache`` group of pipeline parameters."""

    enable_cache: bool = True
    cache_threshold: float = 0.85
    adaptive_cache: bool = True
    cache_ttl: int = 3600


@dataclass(frozen=True, slots=True)
class SafetyOptions:
    """The ``safety`` group of pipeline parameters."""

    enable_security_filter: bool = False
    detect_pii: bool = False
    redact_pii: bool = False
    sensitivity_level: Literal['public', 'internal', 'confidential', 'restricted'] = 'public'
    content_filter: bool = False
    enable_injection_filter: bool = True
    injection_filter_strength: float = 0.5
    enable_content_policy_filter: bool = False
    content_policy_types: Optional[list[str]] = None
    content_policy_mode: Literal['redact', 'drop', 'annotate'] = 'redact'
    enable_html_sanitizer: bool = False
    html_allowed_tags: Optional[list[str]] = None
    html_allowed_attrs: Optional[list[str]] = None
    ocr_confidence_threshold: Optional[float] = None


@dataclass(frozen=True, slots=True)
class RerankOptions:
    """The ``rerank`` group of pipeline parameters."""

    enable_reranking: bool = True
    reranking_strategy: Literal['flashrank', 'cross_encoder', 'hybrid', 'llama_cpp', 'llm_scoring', 'two_tier', 'none'] = 'flashrank'
    rerank_top_k: Optional[int] = None
    reranking_model: Optional[str] = None
    rerank_min_relevance_prob: Optional[float] = None
    rerank_sentinel_margin: Optional[float] = None
    enable_learned_fusion: bool = False
    calibrator_version: Optional[str] = None


@dataclass(frozen=True, slots=True)
class GenerationOptions:
    """The ``generation`` group of pipeline parameters."""

    abstention_policy: Literal['continue', 'ask', 'decline'] = 'continue'
    enable_generation: bool = True
    strict_extractive: bool = False
    generation_model: Optional[str] = None
    generation_provider: Optional[str] = None
    generation_prompt: Optional[str] = None
    max_generation_tokens: int = 500
    enable_abstention: bool = False
    abstention_behavior: Literal['continue', 'ask', 'decline'] = 'continue'
    enable_multi_turn_synthesis: bool = False
    synthesis_time_budget_sec: Optional[float] = None
    synthesis_draft_tokens: Optional[int] = None
    synthesis_refine_tokens: Optional[int] = None
    enable_suggestions: bool = False
    num_suggestions: int = 5
    enable_structured_response: bool = False
    enable_image_search: bool = False
    enable_video_search: bool = False


@dataclass(frozen=True, slots=True)
class CitationOptions:
    """The ``citations`` group of pipeline parameters."""

    enable_citations: bool = False
    citation_style: Literal['apa', 'mla', 'chicago', 'harvard', 'ieee'] = 'apa'
    include_page_numbers: bool = False
    enable_chunk_citations: bool = True
    require_hard_citations: bool = False
    enable_evidence_chains: bool = False


@dataclass(frozen=True, slots=True)
class VerificationOptions:
    """The ``verification`` group of pipeline parameters."""

    enable_post_verification: bool = False
    adaptive_max_retries: int = 1
    adaptive_unsupported_threshold: float = 0.15
    adaptive_max_claims: int = 20
    adaptive_time_budget_sec: Optional[float] = None
    low_confidence_behavior: Literal['continue', 'ask', 'decline'] = 'continue'
    adaptive_advanced_rewrites: Optional[bool] = None
    adaptive_rerun_on_low_confidence: bool = False
    adaptive_rerun_include_generation: bool = True
    adaptive_rerun_bypass_cache: bool = False
    adaptive_rerun_time_budget_sec: Optional[float] = None
    adaptive_rerun_doc_budget: Optional[int] = None
    _adaptive_rerun: bool = False
    enable_numeric_fidelity: bool = False
    numeric_fidelity_behavior: Literal['continue', 'ask', 'decline', 'retry'] = 'continue'
    enable_claims: bool = False
    claim_extractor: Literal['aps', 'claimify', 'auto'] = 'auto'
    claim_verifier: Literal['nli', 'llm', 'hybrid'] = 'hybrid'
    claims_top_k: int = 5
    claims_conf_threshold: float = 0.7
    claims_max: int = 25
    nli_model: Optional[str] = None
    claims_concurrency: int = 8
    numeric_precision_mode: Literal['standard', 'strict', 'academic'] = 'standard'
    doc_only_verification: bool = False
    generate_verification_report: bool = False
    enable_faithfulness_eval: bool = False
    claims_budget_usd: Any = None  # read from **kwargs
    claims_budget_tokens: Any = None  # read from **kwargs
    claims_budget_strict: Any = None  # read from **kwargs


@dataclass(frozen=True, slots=True)
class SelfCorrectOptions:
    """The ``self_correct`` group of pipeline parameters."""

    enable_document_grading: bool = False
    grading_threshold: float = 0.5
    grading_model: Optional[str] = None
    grading_provider: Optional[str] = None
    grading_batch_size: int = 5
    grading_timeout_sec: float = 30.0
    grading_fallback_to_score: bool = True
    grading_fallback_min_score: float = 0.3
    enable_query_rewriting_loop: bool = False
    max_rewrite_attempts: int = 2
    rewrite_relevance_threshold: float = 0.3
    enable_web_fallback: bool = False
    web_fallback_threshold: float = 0.25
    web_search_engine: str = 'duckduckgo'
    web_fallback_result_count: int = 5
    web_fallback_merge_strategy: Literal['prepend', 'append', 'interleave'] = 'prepend'
    enable_knowledge_strips: bool = False
    strip_size_tokens: int = 100
    strip_min_relevance: float = 0.3
    max_strips: int = 20
    enable_fast_hallucination_check: bool = False
    fast_hallucination_timeout_sec: float = 5.0
    fast_hallucination_provider: Optional[str] = None
    fast_hallucination_model: Optional[str] = None
    enable_utility_grading: bool = False
    utility_grading_timeout_sec: float = 5.0
    utility_grading_provider: Optional[str] = None
    utility_grading_model: Optional[str] = None


@dataclass(frozen=True, slots=True)
class ResearchOptions:
    """The ``research`` group of pipeline parameters."""

    enable_research_loop: bool = False
    enable_research_action_dedup: bool = True
    enable_discussion_search: bool = False
    discussion_platforms: Optional[list[str]] = None
    search_url_scraping: bool = True
    enable_research_progress: bool = False
    research_max_iterations: Optional[int] = None
    research_max_iterations_speed: int = 0
    research_max_iterations_balanced: int = 0
    research_max_iterations_quality: int = 0
    classifier_provider: Optional[str] = None
    classifier_model: Optional[str] = None


@dataclass(frozen=True, slots=True)
class ObservabilityOptions:
    """The ``observability`` group of pipeline parameters."""

    collect_feedback: bool = False
    apply_feedback_boost: bool = False
    enable_monitoring: bool = False
    enable_observability: bool = False
    trace_id: Optional[str] = None
    enable_performance_analysis: bool = False
    include_retrieval_diagnostics: bool = True
    highlight_results: bool = False
    highlight_query_terms: bool = False
    track_cost: bool = False
    debug_mode: bool = False
    include_rerank_debug_documents: Optional[bool] = None
    ground_truth_doc_ids: Optional[list[str]] = None
    metrics_k: int = 10


@dataclass(frozen=True, slots=True)
class ResilienceOptions:
    """The ``resilience`` group of pipeline parameters."""

    timeout_seconds: Optional[float] = None
    enable_resilience: bool = False
    retry_attempts: int = 3
    circuit_breaker: bool = False
    fallback_on_error: bool = False


@dataclass(frozen=True, slots=True)
class InertOptions:
    """The ``inert`` group of pipeline parameters."""

    enable_graph_retrieval: bool = False
    graph_version: Optional[str] = None
    graph_neighbors_k: int = 16
    graph_alpha: float = 0.4
    enable_streaming: bool = False
    enable_batch: bool = False
    batch_queries: Optional[list[str]] = None
    batch_concurrent: int = 5
    filter_media_types: Optional[list[str]] = None


@dataclass(frozen=True, slots=True)
class RagRunConfig:
    """Every pipeline parameter, in 17 groups."""

    request: RequestOptions
    scope: ScopeOptions
    runtime: RuntimeHandles
    retrieval: RetrievalOptions
    processing: ProcessingOptions
    query_prep: QueryPrepOptions
    cache: CacheOptions
    safety: SafetyOptions
    rerank: RerankOptions
    generation: GenerationOptions
    citations: CitationOptions
    verification: VerificationOptions
    self_correct: SelfCorrectOptions
    research: ResearchOptions
    observability: ObservabilityOptions
    resilience: ResilienceOptions
    inert: InertOptions

    @classmethod
    def from_kwargs(cls, /, **kwargs: Any) -> "RagRunConfig":
        """Build from the dict a caller passes to ``unified_rag_pipeline`` (unknown keys are ignored)."""
        values = {k: kwargs[k] for k in FIELD_GROUP if k in kwargs}
        values.update(hidden_kwarg_values(**kwargs))
        return cls(**{g: gcls(**{f.name: values[f.name] for f in fields(gcls) if f.name in values}) for g, gcls in _GROUPS})

    @classmethod
    def from_call_locals(cls, values: Mapping[str, Any], /) -> "RagRunConfig":
        """Build from ``locals()`` taken first thing in ``unified_rag_pipeline``."""
        return cls.from_kwargs(**{k: v for k, v in values.items() if k != "kwargs"}, **values["kwargs"])

    def as_kwargs(self) -> dict[str, Any]:
        """Every field as a flat ``{name: value}`` dict (shallow: values are the stored objects)."""
        return {f.name: getattr(getattr(self, g), f.name) for g, gcls in _GROUPS for f in fields(gcls)}


def hidden_kwarg_values(**kwargs: Any) -> dict[str, Any]:
    """The 13 keys the pipeline reads from ``**kwargs``, each with the default it has always had."""
    return {
        "chat_dictionaries_db_path": kwargs.get("chat_dictionaries_db_path"),
        "claims_budget_strict": kwargs.get("claims_budget_strict"),
        "claims_budget_tokens": kwargs.get("claims_budget_tokens"),
        "claims_budget_usd": kwargs.get("claims_budget_usd"),
        "enable_expansion": kwargs.get("enable_expansion", False),
        "faithfulness_llm": kwargs.get("faithfulness_llm"),
        "include_metadata": kwargs.get("include_metadata", True),
        "include_sources": kwargs.get("include_sources", True),
        "metadata": kwargs.get("metadata"),
        "prompts_db": kwargs.get("prompts_db"),
        "prompts_db_path": kwargs.get("prompts_db_path"),
        "workspace_id": kwargs.get("workspace_id"),
        "world_books_db_path": kwargs.get("world_books_db_path"),
    }


_GROUPS: tuple[tuple[str, type], ...] = (
    ("request", RequestOptions),
    ("scope", ScopeOptions),
    ("runtime", RuntimeHandles),
    ("retrieval", RetrievalOptions),
    ("processing", ProcessingOptions),
    ("query_prep", QueryPrepOptions),
    ("cache", CacheOptions),
    ("safety", SafetyOptions),
    ("rerank", RerankOptions),
    ("generation", GenerationOptions),
    ("citations", CitationOptions),
    ("verification", VerificationOptions),
    ("self_correct", SelfCorrectOptions),
    ("research", ResearchOptions),
    ("observability", ObservabilityOptions),
    ("resilience", ResilienceOptions),
    ("inert", InertOptions),
)

# field name -> group name, for every field of every group
FIELD_GROUP: Mapping[str, str] = MappingProxyType({f.name: g for g, gcls in _GROUPS for f in fields(gcls)})
