"""
Blessed configuration profiles for the unified RAG pipeline.

These profiles provide small, opinionated presets for `unified_rag_pipeline`
so callers do not need to reason about dozens of individual flags for
common scenarios.

Current built-in profiles:
    - "production"  : Safe, predictable defaults suitable for latency- and
                      safety-conscious deployments.
    - "research"    : High-quality, feature-rich configuration for analysis
                      and model/retrieval experiments.
    - "cheap"       : Cost- and latency-optimized configuration with most
                      expensive extras disabled.
    - "fast"        : Latency-first profile for short, direct answers.
    - "balanced"    : Practical quality/latency tradeoff for most usage.
    - "accuracy"    : Quality-first profile with stronger retrieval/synthesis.

Profiles are intentionally conservative: they override only a subset of
pipeline parameters and rely on function-level defaults for everything else.
Callers can always override any individual flag on top of a profile.
"""

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, Optional

ProfileName = Literal[
    "production",
    "research",
    "cheap",
    "fast",
    "balanced",
    "accuracy",
    "slides_source_retrieval_v1",
]

SLIDES_SOURCE_PROFILE = "slides_source_retrieval_v1"
SLIDES_SOURCE_RERANKING_STRATEGIES = frozenset({"none", "flashrank", "cross_encoder"})

_SLIDES_SOURCE_DISABLED_STAGES = (
    "enable_text_late_chunking",
    "adaptive_hybrid_weights",
    "enable_intent_routing",
    "auto_temporal_filters",
    "expand_query",
    "spell_check",
    "enable_prf",
    "enable_hyde",
    "enable_gap_analysis",
    "enable_cache",
    "adaptive_cache",
    "enable_table_processing",
    "enable_vlm_late_chunking",
    "enable_enhanced_chunking",
    "enable_parent_expansion",
    "include_sibling_chunks",
    "include_parent_document",
    "enable_multi_vector_passages",
    "enable_precomputed_spans",
    "enable_numeric_table_boost",
    "enable_learned_fusion",
    "enable_citations",
    "enable_chunk_citations",
    "enable_generation",
    "strict_extractive",
    "enable_pre_retrieval_clarification",
    "enable_abstention",
    "enable_multi_turn_synthesis",
    "enable_post_verification",
    "adaptive_rerun_on_low_confidence",
    "adaptive_rerun_include_generation",
    "enable_query_decomposition",
    "enable_graph_retrieval",
    "collect_feedback",
    "apply_feedback_boost",
    "enable_monitoring",
    "enable_observability",
    "enable_performance_analysis",
    "enable_streaming",
    "highlight_results",
    "track_cost",
    "debug_mode",
    "include_rerank_debug_documents",
    "enable_injection_filter",
    "enable_content_policy_filter",
    "enable_html_sanitizer",
    "require_hard_citations",
    "enable_numeric_fidelity",
    "enable_claims",
    "doc_only_verification",
    "generate_verification_report",
    "enable_dynamic_granularity",
    "enable_evidence_accumulation",
    "enable_evidence_chains",
    "enable_document_grading",
    "enable_query_rewriting_loop",
    "enable_web_fallback",
    "enable_knowledge_strips",
    "enable_fast_hallucination_check",
    "enable_utility_grading",
    "enable_batch",
    "enable_resilience",
    "enable_date_filter",
    "fallback_on_error",
    "enable_faithfulness_eval",
    "enable_query_classification",
    "enable_query_reformulation",
    "enable_research_loop",
    "enable_discussion_search",
    "search_url_scraping",
    "enable_research_progress",
    "enable_suggestions",
    "enable_structured_response",
    "enable_image_search",
    "enable_video_search",
)

_SLIDES_SOURCE_DEFAULTS = MappingProxyType(
    {
        "sources": ("media_db", "notes", "chats"),
        "search_mode": "fts",
        "fts_level": "chunk",
        "enable_reranking": False,
        "reranking_strategy": "none",
        "reranking_model": None,
        "adaptive_max_retries": 0,
        "generation_provider": None,
        "generation_model": None,
        "generation_prompt": None,
        "search_depth_mode": None,
        "chat_history": None,
        "discussion_platforms": None,
        "rag_profile": None,
        **dict.fromkeys(_SLIDES_SOURCE_DISABLED_STAGES, False),
    }
)


@dataclass(frozen=True)
class RAGProfile:
    """Container for a named RAG profile."""

    name: ProfileName
    description: str
    defaults: Mapping[str, Any]


_PROFILES: dict[ProfileName, RAGProfile] = {
    "production": RAGProfile(
        name="production",
        description=(
            "Safe production profile: hybrid search with reranking, "
            "semantic cache, and stricter guardrails (numeric fidelity, "
            "citations, basic content filtering). Expensive research "
            "features remain disabled by default."
        ),
        defaults={
            # Retrieval
            "search_mode": "hybrid",
            "top_k": 12,
            "enable_cache": True,
            "adaptive_cache": True,
            # Query processing (lightweight)
            "expand_query": True,
            "expansion_strategies": ["acronym", "synonym"],
            "enable_query_decomposition": False,
            "enable_gap_analysis": False,
            "enable_hyde": False,
            "enable_prf": False,
            # Guardrails / safety
            "enable_security_filter": True,
            "content_filter": True,
            "enable_injection_filter": True,
            "enable_content_policy_filter": True,
            "content_policy_types": ["pii"],
            "content_policy_mode": "redact",
            "require_hard_citations": True,
            "enable_numeric_fidelity": True,
            "numeric_fidelity_behavior": "ask",
            "enable_post_verification": True,
            "adaptive_max_retries": 1,
            "low_confidence_behavior": "ask",
            # Generation
            "enable_generation": True,
            "strict_extractive": False,
            "max_generation_tokens": 512,
            # Reranking
            "enable_reranking": True,
            "reranking_strategy": "flashrank",
            # Monitoring / observability
            "enable_monitoring": True,
            "enable_observability": False,
        },
    ),
    "research": RAGProfile(
        name="research",
        description=(
            "Research profile: enables most advanced retrieval (expansion, "
            "PRF, HyDE, decomposition, multi-vector) and verification "
            "features for quality analysis. Higher latency and cost."
        ),
        defaults={
            # Retrieval
            "search_mode": "hybrid",
            "top_k": 20,
            "enable_cache": False,
            "adaptive_cache": False,
            "enable_multi_vector_passages": True,
            "enable_precomputed_spans": True,
            # Query processing
            "expand_query": True,
            "expansion_strategies": ["acronym", "synonym", "domain", "entity"],
            "spell_check": True,
            "enable_prf": True,
            "prf_terms": 12,
            "prf_top_n": 10,
            "enable_hyde": True,
            "enable_gap_analysis": True,
            "enable_query_decomposition": True,
            "max_subqueries": 4,
            # Guardrails / verification
            "enable_security_filter": True,
            "enable_injection_filter": True,
            "enable_content_policy_filter": True,
            "content_policy_types": ["pii"],
            "content_policy_mode": "redact",
            "require_hard_citations": True,
            "enable_numeric_fidelity": True,
            "numeric_fidelity_behavior": "ask",
            "enable_claims": True,
            "claims_top_k": 5,
            "claims_max": 20,
            "enable_post_verification": True,
            "adaptive_max_retries": 2,
            "adaptive_time_budget_sec": 20.0,
            "adaptive_rerun_on_low_confidence": True,
            "adaptive_rerun_time_budget_sec": 15.0,
            # Generation
            "enable_generation": True,
            "enable_multi_turn_synthesis": True,
            "synthesis_time_budget_sec": 30.0,
            "max_generation_tokens": 1024,
            # Reranking
            "enable_reranking": True,
            "reranking_strategy": "hybrid",
            "enable_learned_fusion": True,
            # Monitoring / observability
            "enable_monitoring": True,
            "enable_observability": True,
            "enable_performance_analysis": True,
            "track_cost": True,
            "debug_mode": False,
        },
    ),
    "cheap": RAGProfile(
        name="cheap",
        description=(
            "Cheap/fast profile: favors lower latency and cost by disabling "
            "most expensive extras (HyDE, PRF, claims, adaptive reruns) and "
            "using simpler retrieval/reranking. Guardrails remain on but "
            "post-verification and numeric checks are relaxed."
        ),
        defaults={
            # Retrieval
            "search_mode": "fts",
            "top_k": 8,
            "enable_cache": True,
            "adaptive_cache": False,
            # Query processing
            "expand_query": False,
            "enable_prf": False,
            "enable_hyde": False,
            "enable_gap_analysis": False,
            "enable_query_decomposition": False,
            # Guardrails (minimal but present)
            "enable_security_filter": True,
            "enable_injection_filter": True,
            "enable_content_policy_filter": False,
            "require_hard_citations": False,
            "enable_numeric_fidelity": False,
            "enable_claims": False,
            "enable_post_verification": False,
            # Generation
            "enable_generation": True,
            "strict_extractive": False,
            "max_generation_tokens": 384,
            # Reranking
            "enable_reranking": True,
            "reranking_strategy": "flashrank",
            # Monitoring / observability
            "enable_monitoring": False,
            "enable_observability": False,
            "enable_performance_analysis": False,
            "track_cost": False,
        },
    ),
    "fast": RAGProfile(
        name="fast",
        description=(
            "Latency-first profile using compact instruction-style prompting "
            "with lightweight retrieval and no decomposition."
        ),
        defaults={
            "generation_prompt": "instruction_tuned",
            "enable_query_decomposition": False,
            "enable_reranking": True,
            "reranking_strategy": "flashrank",
            "top_k": 6,
            "max_generation_tokens": 440,
            "enable_structured_response": False,
            "enable_multi_turn_synthesis": False,
            "require_hard_citations": False,
            "enable_claims": False,
        },
    ),
    "balanced": RAGProfile(
        name="balanced",
        description=(
            "Balanced quality/latency profile with compact multi-hop guidance "
            "and hybrid reranking."
        ),
        defaults={
            "generation_prompt": "multi_hop_compact",
            "enable_query_decomposition": True,
            "max_subqueries": 3,
            "subquery_time_budget_sec": 2.5,
            "enable_reranking": True,
            "reranking_strategy": "hybrid",
            "top_k": 10,
            "max_generation_tokens": 1000,
            "enable_structured_response": True,
            "require_hard_citations": False,
            "enable_claims": False,
        },
    ),
    "accuracy": RAGProfile(
        name="accuracy",
        description=(
            "Quality-first profile emphasizing decomposition depth, stronger "
            "reranking, and expert synthesis prompting."
        ),
        defaults={
            "generation_prompt": "expert_synthesis",
            "enable_query_decomposition": True,
            "max_subqueries": 5,
            "subquery_time_budget_sec": 6.0,
            "enable_reranking": True,
            "reranking_strategy": "two_tier",
            "top_k": 16,
            "rerank_top_k": 16,
            "max_generation_tokens": 2200,
            "enable_structured_response": True,
            "require_hard_citations": True,
            "enable_numeric_fidelity": True,
            "enable_claims": True,
        },
    ),
    "slides_source_retrieval_v1": RAGProfile(
        name="slides_source_retrieval_v1",
        description=(
            "Closed, owner-local FTS retrieval used only to snapshot source "
            "material for standalone HTML presentations."
        ),
        defaults=_SLIDES_SOURCE_DEFAULTS,
    ),
}


def list_profiles() -> dict[ProfileName, RAGProfile]:
    """Return a copy of all registered profiles keyed by name."""
    return dict(_PROFILES)


def get_profile(name: ProfileName) -> RAGProfile:
    """Fetch a profile by name."""
    if name not in _PROFILES:
        # Defensive: keep error message explicit for callers
        raise ValueError(f"Unknown RAG profile: {name!r}")
    return _PROFILES[name]


def get_profile_kwargs(
    name: ProfileName,
    overrides: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """
    Build keyword arguments for `unified_rag_pipeline` from a profile.

    The returned dict can be passed as `**kwargs` to the pipeline. Any
    provided overrides take precedence over profile defaults.
    """
    profile = get_profile(name)
    if name == SLIDES_SOURCE_PROFILE and overrides:
        raise ValueError(f"{SLIDES_SOURCE_PROFILE!r} does not accept overrides")
    kwargs: dict[str, Any] = dict(profile.defaults)
    if overrides:
        # Copy into a mutable dict to avoid mutating caller mappings
        for key, value in overrides.items():
            kwargs[key] = value
    return kwargs


def apply_profile_to_kwargs(
    name: ProfileName,
    existing: Optional[MutableMapping[str, Any]] = None,
) -> dict[str, Any]:
    """
    Merge a profile into an existing kwargs-style mapping.

    This is useful when a caller already has a dict of parameters and wants
    to layer a profile underneath as a set of defaults.
    """
    if name == SLIDES_SOURCE_PROFILE and existing:
        raise ValueError(f"{SLIDES_SOURCE_PROFILE!r} does not accept overrides")
    base = dict(get_profile(name).defaults)
    if existing:
        base.update(existing)
    return base


def get_multi_tenant_safe_kwargs(
    namespace: str,
    overrides: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """
    Build kwargs for a multi-tenant-safe production configuration.

    This helper layers stricter defaults for multi-tenant deployments on top
    of the "production" profile:

    - Requires a non-empty `namespace` and sets it as `index_namespace`.
    - Disables OTEL-style observability (`enable_observability=False`) while
      keeping lightweight metrics via `enable_monitoring=True`.

    Callers should still configure global settings such as
    `RAG_PAYLOAD_EXEMPLAR_SAMPLING=0` if they want to fully disable payload
    exemplars in shared storage.
    """
    if not namespace or not str(namespace).strip():
        raise ValueError("Multi-tenant safe profile requires a non-empty namespace.")

    # Start from production defaults so we inherit guardrails and safety knobs.
    base = get_profile_kwargs("production", overrides=overrides)

    # Enforce per-tenant namespace and disable OTEL observability by default.
    base["index_namespace"] = str(namespace).strip()
    base["enable_observability"] = False
    base.setdefault("enable_monitoring", True)

    return base
