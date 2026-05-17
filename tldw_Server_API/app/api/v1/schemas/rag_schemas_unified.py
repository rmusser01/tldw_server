"""
Unified RAG Schema - Single comprehensive schema for the unified pipeline

This schema maps directly to the unified_rag_pipeline parameters,
providing a clean API interface with all features accessible.
"""

from typing import Any, Literal, Optional

from pydantic import BaseModel

try:
    # Pydantic v2
    from pydantic import field_validator, model_validator  # type: ignore
except Exception:
    model_validator = None  # type: ignore
    from pydantic import validator as field_validator  # type: ignore
from pydantic import ConfigDict

from ._compat import Field
from tldw_Server_API.app.core.Text2SQL.source_registry import (
    normalize_source,
    normalize_sources_public,
)

# Load contextual retrieval defaults from settings (config.txt/env)
try:
    from tldw_Server_API.app.core.config import settings as _settings
    _ctx_defaults = _settings.get("RAG_CONTEXTUAL_DEFAULTS", {}) if isinstance(_settings, dict) else {}
    _DEF_SIB_WIN = int(_ctx_defaults.get("sibling_window", 1))
    _DEF_INC_PARENT = bool(_ctx_defaults.get("include_parent_document", False))
    _DEF_PARENT_MAX_TOK = int(_ctx_defaults.get("parent_max_tokens", 1200))
    _DEF_INC_SIB = bool(_ctx_defaults.get("include_sibling_chunks", False))
    _DEF_ENH_CHUNK = bool(_ctx_defaults.get("enable_enhanced_chunking", False))
    # Default FTS level for RAG: 'media' or 'chunk'
    _DEF_FTS_LVL = str(_settings.get("RAG_DEFAULT_FTS_LEVEL", "media")).lower()
    if _DEF_FTS_LVL not in ("media", "chunk"):
        _DEF_FTS_LVL = "media"
except Exception:
    _DEF_SIB_WIN = 1
    _DEF_INC_PARENT = False
    _DEF_PARENT_MAX_TOK = 1200
    _DEF_INC_SIB = False
    _DEF_ENH_CHUNK = False
    _DEF_FTS_LVL = "media"


class UnifiedRAGRequest(BaseModel):
    """
    Unified RAG request with ALL features as optional parameters.
    Every feature in the RAG system is accessible through this single schema.
    """

    # ========== REQUIRED ==========
    query: str = Field(
        ...,
        description="The search query",
        min_length=1,
        max_length=20000,
        example="What is machine learning?"
    )

    # ========== DATA SOURCES ==========
    sources: Optional[list[str]] = Field(
        default=["media_db"],
        description=(
            "Databases to search: media_db, notes, chats, characters, kanban, "
            "prompts, world_books, dictionaries, sql"
        ),
        example=[
            "media_db",
            "notes",
            "chats",
            "characters",
            "kanban",
            "prompts",
            "world_books",
            "dictionaries",
        ],
    )
    sql_target_id: str = Field(
        default="media_db",
        description="SQL target identifier used when sources includes 'sql'",
        example="media_db",
    )

    # ========== STRATEGY SELECTION ==========
    strategy: Literal["standard", "agentic"] = Field(
        default="standard",
        description="Pipeline strategy: standard (pre-chunked) or agentic (query-time synthetic chunk)",
        example="agentic",
    )
    rag_profile: Optional[Literal["fast", "balanced", "accuracy"]] = Field(
        default=None,
        description="Switchable profile defaults applied at request handling time",
        example="balanced",
    )

    if model_validator is not None:
        @model_validator(mode="before")
        def _map_legacy_min_relevance(cls, values):  # type: ignore
            if isinstance(values, dict) and "min_relevance_score" in values and "min_score" not in values:
                values["min_score"] = values.get("min_relevance_score")
            return values

    # Optional corpus/namespace for per-corpus indexing and synonyms
    corpus: Optional[str] = Field(
        default=None,
        description="Alias for index_namespace used to select corpus-specific synonyms",
        example="my_corpus"
    )
    index_namespace: Optional[str] = Field(
        default=None,
        description="Corpus/namespace identifier (enables per-corpus synonyms & indexing)",
        example="my_corpus"
    )
    workspace_id: Optional[str] = Field(
        default=None,
        description=(
            "Optional workspace scope. Generated, test, and workspace-scoped artifacts "
            "are excluded unless this is provided."
        ),
        example="workspace_123",
    )

    @field_validator("sources", mode="before")
    @classmethod
    def _validate_sources(cls, v):
        if v is None:
            return normalize_sources_public(None)
        if not isinstance(v, list):
            raise ValueError("sources must be a list of strings")
        if any(not isinstance(source, str) for source in v):
            raise ValueError("sources entries must be strings")
        return normalize_sources_public(v)

    @field_validator("sql_target_id", mode="before")
    @classmethod
    def _validate_sql_target_id(cls, v):
        if v is None:
            return "media_db"
        if not isinstance(v, str):
            raise ValueError("sql_target_id must be a string")
        return normalize_source(v)

    # Map corpus -> index_namespace at model-level (before validation)
    if model_validator is not None:
        @model_validator(mode="before")
        def _alias_corpus_namespace(cls, values):  # type: ignore
            if isinstance(values, dict):
                v = values.get("index_namespace")
                corpus = values.get("corpus")
                if (v is None or (isinstance(v, str) and not v.strip())) and isinstance(corpus, str):
                    values["index_namespace"] = corpus.strip() or v
            return values

    # ========== SEARCH CONFIGURATION ==========
    search_mode: Literal["fts", "vector", "hybrid"] = Field(
        default="hybrid",
        description="Search mode: fts (full-text), vector (semantic), or hybrid",
        example="hybrid"
    )

    # FTS granularity: media-level (Media.title/content) or chunk-level (UnvectorizedMediaChunks)
    fts_level: Literal["media", "chunk"] = Field(
        default=_DEF_FTS_LVL,
        description="FTS granularity: 'media' searches Media FTS; 'chunk' searches plaintext chunks",
        example="chunk"
    )
    enable_text_late_chunking: bool = Field(
        default=False,
        description="When enabled with chunk-level search, bypass persisted text chunks and late-chunk matched media in memory for this query",
        example=False,
    )
    chunk_method: Optional[Literal["semantic", "tokens", "paragraphs", "sentences", "words", "ebook_chapters", "json", "propositions"]] = Field(
        default=None,
        description="Transient text late chunking method for this query",
        example="sentences",
    )
    chunk_size: Optional[int] = Field(
        default=None,
        gt=0,
        description="Transient text late chunk size for this query",
        example=500,
    )
    chunk_overlap: Optional[int] = Field(
        default=None,
        ge=0,
        description="Transient text late chunk overlap for this query",
        example=50,
    )
    chunk_language: Optional[str] = Field(
        default=None,
        description="Optional language override for transient text late chunking",
        example="en",
    )

    hybrid_alpha: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Hybrid search weight (0=FTS only, 1=Vector only)",
        example=0.7
    )

    enable_intent_routing: bool = Field(
        default=False,
        description="Analyze query intent and adjust retrieval knobs (top_k, hybrid weighting) before retrieval",
        example=False,
    )

    top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of results to return",
        example=10
    )

    min_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum relevance score",
        example=0.0
    )

    @field_validator("min_score", mode="before")
    @classmethod
    def _alias_min_relevance_score(cls, v):
        # Field-level aliasing handled by model-level pre-validators.
        return v

    @field_validator("chunk_overlap")
    @classmethod
    def _validate_chunk_overlap(cls, v, info):
        if v is None:
            return v
        chunk_size = info.data.get("chunk_size")
        if chunk_size is not None and v >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return v

    # ========== QUERY EXPANSION ==========
    expand_query: bool = Field(
        default=False,
        description="Enable query expansion",
        example=True
    )

    expansion_strategies: Optional[list[str]] = Field(
        default=None,
        description="Expansion strategies: acronym, synonym, domain, entity",
        example=["acronym", "synonym"]
    )

    spell_check: bool = Field(
        default=False,
        description="Enable spell checking",
        example=False
    )
    max_query_variations: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum number of query expansion variations to retrieve",
        example=3,
    )

    # ========== CACHING ==========
    enable_cache: bool = Field(
        default=True,
        description="Enable semantic caching",
        example=True
    )

    cache_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Cache similarity threshold",
        example=0.85
    )

    adaptive_cache: bool = Field(
        default=True,
        description="Use adaptive cache thresholds",
        example=True
    )

    # ========== FILTERING ==========
    keyword_filter: Optional[list[str]] = Field(
        default=None,
        description="Filter results by keywords",
        example=["python", "api"]
    )

    # Explicit selection of items per source
    include_media_ids: Optional[list[int]] = Field(
        default=None,
        description="Restrict search to these Media DB item IDs",
        example=[1, 2, 3]
    )
    collection_id: Optional[int] = Field(
        default=None,
        ge=1,
        description="Restrict search to ready media items in this durable media collection",
        example=7,
    )
    include_note_ids: Optional[list[str]] = Field(
        default=None,
        description="Restrict search to these Note IDs (ChaChaNotes UUIDs)",
        example=["a1b2c3-uuid", "d4e5f6-uuid"]
    )

    # ========== SECURITY & PRIVACY ==========
    enable_security_filter: bool = Field(
        default=False,
        description="Enable security filtering",
        example=False
    )

    detect_pii: bool = Field(
        default=False,
        description="Detect personally identifiable information",
        example=False
    )

    redact_pii: bool = Field(
        default=False,
        description="Redact detected PII",
        example=False
    )

    sensitivity_level: Literal["public", "internal", "confidential", "restricted"] = Field(
        default="public",
        description="Maximum sensitivity level for results",
        example="internal"
    )

    content_filter: bool = Field(
        default=False,
        description="Enable content filtering",
        example=False
    )

    # ========== DOCUMENT PROCESSING ==========
    enable_table_processing: bool = Field(
        default=False,
        description="Enable table extraction and processing",
        example=False
    )

    table_method: Literal["markdown", "html", "hybrid"] = Field(
        default="markdown",
        description="Table serialization method",
        example="markdown"
    )

    # ========== VLM LATE CHUNKING ==========
    enable_vlm_late_chunking: bool = Field(
        default=False,
        description="Enable late VLM chunking on retrieved documents (media_db PDFs)",
        example=False
    )
    vlm_backend: Optional[str] = Field(
        default=None,
        description="VLM backend name (e.g., 'hf_table_transformer', 'docling')",
        example="hf_table_transformer"
    )
    vlm_detect_tables_only: bool = Field(
        default=True,
        description="When true, keep only table detections; otherwise include images/figures as 'vlm'",
        example=True
    )
    vlm_max_pages: Optional[int] = Field(
        default=None,
        ge=1,
        le=1000,
        description="Max pages per document to process with VLM",
        example=3
    )
    vlm_late_chunk_top_k_docs: int = Field(
        default=3,
        ge=1,
        le=50,
        description="Number of top retrieved documents to apply VLM late-chunking to",
        example=3
    )

    # ========== CHUNKING & CONTEXT ==========

    chunk_type_filter: Optional[list[str]] = Field(
        default=None,
        description="Filter chunks by type: text, code, table, list",
        example=["text", "code"]
    )

    enable_parent_expansion: bool = Field(
        default=False,
        description="Expand chunks with parent document context",
        example=False
    )

    parent_context_size: int = Field(
        default=500,
        ge=100,
        le=2000,
        description="Size of parent context in characters",
        example=500
    )

    include_sibling_chunks: bool = Field(
        default=_DEF_INC_SIB,
        description="Include adjacent chunks from same document",
        example=False
    )

    sibling_window: int = Field(
        default=_DEF_SIB_WIN,
        ge=0,
        le=50,
        description="Number of sibling chunks to include on each side when include_sibling_chunks is true",
        example=2
    )

    include_parent_document: bool = Field(
        default=_DEF_INC_PARENT,
        description="Include the full parent document for each selected chunk",
        example=False
    )

    parent_max_tokens: Optional[int] = Field(
        default=_DEF_PARENT_MAX_TOK,
        ge=1,
        description="Maximum tokens allowed to include a parent document; if parent exceeds, it is omitted",
        example=1200
    )

    # ========== AGENTIC CHUNKING (OPTIONAL) ==========
    agentic_top_k_docs: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Number of coarse documents to consider when building the synthetic chunk",
        example=3,
    )
    agentic_window_chars: int = Field(
        default=1200,
        ge=200,
        le=20000,
        description="Window size around hits for span extraction (characters)",
        example=1200,
    )
    agentic_max_tokens_read: int = Field(
        default=6000,
        ge=500,
        le=20000,
        description="Approximate token budget for assembling the synthetic chunk",
        example=6000,
    )
    agentic_max_tool_calls: int = Field(
        default=8,
        ge=1,
        le=64,
        description="Reserved for future LLM-tool orchestration; ignored by baseline",
        example=8,
    )
    agentic_extractive_only: bool = Field(
        default=True,
        description="If true, assembled chunk is purely extractive from sources",
        example=True,
    )
    agentic_quote_spans: bool = Field(
        default=True,
        description="Include span quotes and offsets in provenance metadata",
        example=True,
    )
    agentic_debug_trace: bool = Field(
        default=False,
        description="Emit extra debug logs for agentic execution",
        example=False,
    )
    agentic_enable_tools: bool = Field(
        default=False,
        description="Enable bounded tool loop (search_within/open_section/expand_window/quote_spans)",
        example=False,
    )
    agentic_use_llm_planner: bool = Field(
        default=False,
        description="Use an LLM to plan tool use (ReAct style); falls back to heuristics when unavailable",
        example=False,
    )
    agentic_time_budget_sec: Optional[float] = Field(
        default=None,
        ge=0.1,
        le=30.0,
        description="Wall-clock time budget for the agentic tool loop",
        example=5.0,
    )
    agentic_cache_ttl_sec: int = Field(
        default=600,
        ge=1,
        le=86400,
        description="TTL for ephemeral chunk cache (seconds)",
        example=600,
    )
    # Query decomposition (multi-hop)
    agentic_enable_query_decomposition: bool = Field(
        default=False,
        description="Enable heuristic multi-hop decomposition; run tool loop for each sub-goal and merge spans",
        example=False,
    )
    agentic_subgoal_max: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum number of sub-goals to consider when decomposing queries",
        example=3,
    )
    # Intra-doc semantic search
    agentic_enable_semantic_within: bool = Field(
        default=True,
        description="Use paragraph-level semantic search within documents (hashed embeddings)",
        example=True,
    )
    # Section index and anchors
    agentic_enable_section_index: bool = Field(
        default=True,
        description="Build simple heading→offset index and prefer structural anchors for open_section",
        example=True,
    )
    agentic_prefer_structural_anchors: bool = Field(
        default=True,
        description="Prefer structural anchors when selecting spans near section boundaries",
        example=True,
    )
    # Table/figure support (heuristic)
    agentic_enable_table_support: bool = Field(
        default=True,
        description="Prefer table-like spans when queries mention tables/figures; integrates with VLM late chunks when available",
        example=True,
    )
    # VLM late chunking (agentic path)
    agentic_enable_vlm_late_chunking: bool = Field(
        default=False,
        description="Use VLM late chunking for top-k PDFs to add table/figure hints inside agentic pipeline",
        example=False,
    )
    agentic_vlm_backend: Optional[str] = Field(
        default=None,
        description="VLM backend name (e.g., 'hf_table_transformer', 'docling'); auto-select when None",
        example="hf_table_transformer",
    )
    agentic_vlm_detect_tables_only: bool = Field(
        default=True,
        description="Keep only table detections; set false to include other VLM hints",
        example=True,
    )
    agentic_vlm_max_pages: Optional[int] = Field(
        default=None,
        ge=1,
        le=1000,
        description="Max pages per document to scan with VLM during agentic execution",
        example=3,
    )
    agentic_vlm_late_chunk_top_k_docs: int = Field(
        default=2,
        ge=1,
        le=25,
        description="Top-k PDFs from coarse retrieval to pass to VLM late chunking (agentic)",
        example=2,
    )
    # Provider embeddings for intra-doc vectors
    agentic_use_provider_embeddings_within: bool = Field(
        default=False,
        description="Use configured embeddings provider to embed paragraphs for intra-doc semantic search; falls back to hashed",
        example=False,
    )
    agentic_provider_embedding_model_id: Optional[str] = Field(
        default=None,
        description="Optional embedding model ID to override default for intra-doc paragraph embeddings",
        example="text-embedding-3-small",
    )

    # Agentic adaptive budgets & observability
    agentic_adaptive_budgets: bool = Field(
        default=True,
        description="Adapt max tool calls and read budget based on early coverage/corroboration",
        example=True,
    )
    agentic_coverage_target: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Target fraction of query term coverage before early stop",
        example=0.8,
    )
    agentic_min_corroborating_docs: int = Field(
        default=2,
        ge=1,
        le=10,
        description="Minimum distinct documents contributing spans before early stop",
        example=2,
    )
    agentic_max_redundancy: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Max allowable redundancy (1 - merged/raw span length) before discouraging more spans",
        example=0.9,
    )
    agentic_enable_metrics: bool = Field(
        default=True,
        description="Emit agentic metrics (tool calls, timings, span histograms)",
        example=True,
    )

    # ========== ADVANCED RETRIEVAL ==========
    enable_multi_vector_passages: bool = Field(
        default=False,
        description="Enable multi-vector passage selection (ColBERT-style approximation) before reranking",
        example=False,
    )
    mv_span_chars: int = Field(
        default=300,
        ge=100,
        le=2000,
        description="Span window size in characters for multi-vector scoring",
        example=300,
    )
    mv_stride: int = Field(
        default=150,
        ge=50,
        le=1000,
        description="Stride between spans in characters",
        example=150,
    )
    mv_max_spans: int = Field(
        default=8,
        ge=1,
        le=64,
        description="Maximum spans per document to consider",
        example=8,
    )
    mv_flatten_to_spans: bool = Field(
        default=False,
        description="Return best span per document as pseudo-documents instead of reordering parent docs",
        example=False,
    )
    enable_numeric_table_boost: bool = Field(
        default=False,
        description="When query includes numbers/units, modestly boost table-like and number-dense chunks before reranking",
        example=False,
    )

    # ========== CLAIMS & FACTUALITY ==========
    enable_claims: bool = Field(
        default=False,
        description="Extract and verify factual claims from the generated answer",
        example=False,
    )
    claim_extractor: Literal["aps", "claimify", "ner", "auto"] = Field(
        default="auto",
        description="Claim extraction strategy",
        example="auto",
    )
    claim_verifier: Literal["nli", "llm", "hybrid"] = Field(
        default="hybrid",
        description="Claim verification strategy",
        example="hybrid",
    )
    claims_top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Top-K evidence items per claim",
        example=5,
    )
    claims_conf_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for supported/refuted labels",
        example=0.7,
    )
    claims_max: int = Field(
        default=25,
        ge=1,
        le=100,
        description="Maximum number of claims to extract",
        example=25,
    )
    claims_concurrency: int = Field(
        default=8,
        ge=1,
        le=32,
        description="Maximum parallel claim verifications",
        example=8,
    )
    nli_model: Optional[str] = Field(
        default=None,
        description="Local HuggingFace model id or path for MNLI (e.g., roberta-large-mnli or /models/mnli)",
        example="roberta-large-mnli",
    )
    numeric_precision_mode: Literal["standard", "strict", "academic"] = Field(
        default="standard",
        description="Numeric precision mode for claim verification: standard (5% tolerance), strict (1%), academic (exact match)",
        example="standard",
    )
    doc_only_verification: bool = Field(
        default=False,
        description="Only accept evidence from provided documents; flag claims requiring external knowledge",
        example=False,
    )
    generate_verification_report: bool = Field(
        default=False,
        description="Generate a structured JSON verification report in the response",
        example=False,
    )

    # ========== DOC-RESEARCHER FEATURES ==========
    enable_dynamic_granularity: bool = Field(
        default=False,
        description="Auto-select retrieval granularity (document/chunk/passage) based on query type",
        example=False,
    )
    enable_evidence_accumulation: bool = Field(
        default=False,
        description="Iteratively retrieve additional evidence until sufficient or budget reached",
        example=False,
    )
    accumulation_max_rounds: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Maximum evidence accumulation rounds",
        example=3,
    )
    accumulation_time_budget_sec: Optional[float] = Field(
        default=None,
        description="Optional time budget for evidence accumulation (seconds)",
        example=3.0,
    )
    enable_evidence_chains: bool = Field(
        default=False,
        description="Build multi-hop evidence chains linking facts to claims",
        example=False,
    )

    # ========== SELF-CORRECTING RAG ==========
    # Stage 1: Document Grading - filter documents by LLM-assessed relevance
    enable_document_grading: bool = Field(
        default=False,
        description="Pre-filter retrieved documents using LLM-based relevance grading before reranking",
        example=False,
    )
    grading_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum relevance score to keep a document during grading",
        example=0.5,
    )
    grading_model: Optional[str] = Field(
        default=None,
        description="Optional model override for document grading",
        example="gpt-4o-mini",
    )
    grading_provider: Optional[str] = Field(
        default=None,
        description="Optional LLM provider for document grading",
        example="openai",
    )
    grading_batch_size: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Batch size for document grading requests",
        example=5,
    )
    grading_timeout_sec: float = Field(
        default=30.0,
        ge=1.0,
        le=120.0,
        description="Timeout for document grading in seconds",
        example=30.0,
    )
    grading_fallback_to_score: bool = Field(
        default=True,
        description="Fall back to document score when LLM grading fails",
        example=True,
    )
    grading_fallback_min_score: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum score threshold for fallback scoring when LLM grading unavailable",
        example=0.3,
    )

    # Stage 2: Query Rewriting Loop - rewrite query when grading shows low relevance
    enable_query_rewriting_loop: bool = Field(
        default=False,
        description="Enable query rewriting loop when document grading shows low relevance",
        example=False,
    )
    max_rewrite_attempts: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Maximum query rewrite attempts in the loop",
        example=2,
    )
    rewrite_relevance_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Average relevance below this triggers query rewriting",
        example=0.3,
    )

    # Stage 3: Web Search Fallback - fall back to web search when local retrieval fails
    enable_web_fallback: bool = Field(
        default=False,
        description="Enable web search fallback when local retrieval yields low relevance",
        example=False,
    )
    web_fallback_threshold: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Relevance threshold below which to trigger web fallback",
        example=0.25,
    )
    web_search_engine: str = Field(
        default="duckduckgo",
        description="Web search engine to use for fallback",
        example="duckduckgo",
    )
    web_fallback_result_count: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of web search results to fetch",
        example=5,
    )
    web_fallback_merge_strategy: Literal["prepend", "append", "interleave"] = Field(
        default="prepend",
        description="How to merge web results with local results",
        example="prepend",
    )

    # Stage 4: Knowledge Strips - partition documents into semantic units
    enable_knowledge_strips: bool = Field(
        default=False,
        description="Partition documents into semantic strips and filter by relevance",
        example=False,
    )
    strip_size_tokens: int = Field(
        default=100,
        ge=20,
        le=500,
        description="Target size for each knowledge strip in tokens",
        example=100,
    )
    strip_min_relevance: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum relevance score to keep a strip",
        example=0.3,
    )
    max_strips: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum number of strips to pass to generation",
        example=20,
    )

    # Stage 5: Fast Hallucination Check - lightweight groundedness check
    enable_fast_hallucination_check: bool = Field(
        default=False,
        description="Run fast groundedness check after generation (before full claims)",
        example=False,
    )
    fast_hallucination_timeout_sec: float = Field(
        default=5.0,
        ge=1.0,
        le=30.0,
        description="Timeout for fast hallucination check in seconds",
        example=5.0,
    )
    fast_hallucination_provider: Optional[str] = Field(
        default=None,
        description="Optional LLM provider for groundedness check",
        example="openai",
    )
    fast_hallucination_model: Optional[str] = Field(
        default=None,
        description="Optional model override for groundedness check",
        example="gpt-4o-mini",
    )

    # Stage 6: Utility Grading - rate response usefulness
    enable_utility_grading: bool = Field(
        default=False,
        description="Rate response usefulness on a 1-5 scale after claims verification",
        example=False,
    )
    utility_grading_timeout_sec: float = Field(
        default=5.0,
        ge=1.0,
        le=30.0,
        description="Timeout for utility grading in seconds",
        example=5.0,
    )
    utility_grading_provider: Optional[str] = Field(
        default=None,
        description="Optional LLM provider for utility grading",
        example="openai",
    )
    utility_grading_model: Optional[str] = Field(
        default=None,
        description="Optional model override for utility grading",
        example="gpt-4o-mini",
    )

    # ========== RERANKING ==========
    enable_reranking: bool = Field(
        default=True,
        description="Enable document reranking",
        example=True
    )

    reranking_strategy: Literal["flashrank", "cross_encoder", "hybrid", "llama_cpp", "llm_scoring", "two_tier", "none"] = Field(
        default="flashrank",
        description="Reranking strategy",
        example="hybrid"
    )

    rerank_top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=100,
        description="Number of documents to rerank (defaults to top_k)",
        example=20
    )
    # Optional reranker model identifier/path (e.g., GGUF path for llama.cpp)
    reranking_model: Optional[str] = Field(
        default=None,
        description="Optional reranker model identifier (e.g., GGUF path for llama.cpp)",
        example="./models/Qwen3-Embedding-0.6B_f16.gguf"
    )
    # Two-tier strategy: request-level gating overrides (optional)
    rerank_min_relevance_prob: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Override minimum calibrated probability to allow generation (Two-Tier)",
        example=0.5,
    )
    rerank_sentinel_margin: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Override required margin between top probability and sentinel (Two-Tier)",
        example=0.15,
    )

    # ========== CITATIONS ==========
    enable_citations: bool = Field(
        default=False,
        description="Generate citations from results",
        example=True
    )

    citation_style: Literal["apa", "mla", "chicago", "harvard", "ieee"] = Field(
        default="apa",
        description="Academic citation format style",
        example="apa"
    )

    include_page_numbers: bool = Field(
        default=False,
        description="Include page numbers in citations",
        example=False
    )

    enable_chunk_citations: bool = Field(
        default=True,
        description="Generate chunk-level citations for answer verification",
        example=True
    )

    # ========== ANSWER GENERATION ==========
    enable_generation: bool = Field(
        default=False,
        description="Generate an answer from retrieved context",
        example=True
    )
    strict_extractive: bool = Field(
        default=False,
        description="Strict extractive mode: assemble the answer only from retrieved spans (no free-form generation). Intended for sentence-level grounding with hard citations.",
        example=False,
    )

    generation_model: Optional[str] = Field(
        default=None,
        description="LLM model for answer generation",
        example="gpt-3.5-turbo"
    )
    generation_provider: Optional[str] = Field(
        default=None,
        description="LLM provider for answer generation",
        example="openai",
    )

    generation_prompt: Optional[str] = Field(
        default=None,
        description="Custom prompt template for generation",
        example=None
    )
    enable_pre_retrieval_clarification: Optional[bool] = Field(
        default=None,
        description="When None, defaults to true if enable_generation=true. Enables pre-retrieval clarification gating.",
        example=None,
    )
    clarification_timeout_sec: Optional[float] = Field(
        default=None,
        ge=0.05,
        le=10.0,
        description="Timeout budget for LLM clarification decision on borderline queries.",
        example=1.5,
    )

    max_generation_tokens: int = Field(
        default=500,
        ge=50,
        le=4000,
        description="Maximum tokens for generated answer",
        example=500
    )
    # Abstention & multi-turn synthesis
    enable_abstention: bool = Field(
        default=False,
        description="Allow abstaining (or asking for clarification) when evidence is thin based on calibrated gating",
        example=False,
    )
    abstention_behavior: Literal["continue", "ask", "decline"] = Field(
        default="continue",
        description="When abstaining: continue (no answer), ask a clarifying question, or decline",
        example="ask",
    )
    enable_multi_turn_synthesis: bool = Field(
        default=False,
        description="Use draft → critique → refine generation under a strict time/token budget",
        example=False,
    )
    synthesis_time_budget_sec: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Total time budget in seconds for multi-turn synthesis",
        example=5.0,
    )
    synthesis_draft_tokens: Optional[int] = Field(
        default=None,
        ge=32,
        le=4000,
        description="Max tokens for the draft stage (defaults to min(max_generation_tokens, 400))",
        example=300,
    )
    synthesis_refine_tokens: Optional[int] = Field(
        default=None,
        ge=32,
        le=4000,
        description="Max tokens for the refine stage (defaults to max_generation_tokens)",
        example=500,
    )

    # ========== GENERATION GUARDRAILS ==========
    enable_content_policy_filter: bool = Field(
        default=False,
        description="Apply PII/PHI content policy on retrieved chunks before generation",
        example=False,
    )
    content_policy_types: Optional[list[str]] = Field(
        default=None,
        description="Policy types to enforce: ['pii', 'phi']",
        example=["pii", "phi"],
    )
    content_policy_mode: Literal["redact", "drop", "annotate"] = Field(
        default="redact",
        description="Policy action: redact matches, drop offending chunks, or annotate only",
        example="redact",
    )
    enable_html_sanitizer: bool = Field(
        default=False,
        description="Sanitize HTML with an allow-list of tags/attributes",
        example=False,
    )
    html_allowed_tags: Optional[list[str]] = Field(
        default=None,
        description="Allowed HTML tags for sanitizer",
        example=["p", "b", "i", "ul", "li", "code"],
    )
    html_allowed_attrs: Optional[list[str]] = Field(
        default=None,
        description="Allowed HTML attributes for sanitizer",
        example=["href", "title"],
    )
    ocr_confidence_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Drop OCR-derived chunks whose metadata.ocr_confidence is below this threshold",
        example=0.6,
    )

    # ========== POST-VERIFICATION (ADAPTIVE) ==========
    enable_post_verification: bool = Field(
        default=False,
        description="Verify generated answer against evidence; optionally attempt a bounded repair",
        example=False,
    )
    adaptive_max_retries: int = Field(
        default=1,
        ge=0,
        le=3,
        description="Maximum adaptive repair attempts when evidence is insufficient",
        example=1,
    )
    adaptive_unsupported_threshold: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="If (refuted + NEI)/total_claims exceeds this, trigger repair",
        example=0.15,
    )
    adaptive_max_claims: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum claims to analyze during post-verification",
        example=20,
    )
    adaptive_time_budget_sec: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Optional hard cap on post-verification wall time",
        example=10.0,
    )
    low_confidence_behavior: Literal["continue", "ask", "decline"] = Field(
        default="continue",
        description="Behavior when evidence remains insufficient after retries",
        example="ask",
    )
    adaptive_advanced_rewrites: Optional[bool] = Field(
        default=None,
        description="Override env to enable/disable HyDE + multi-strategy rewrites in the adaptive pass",
        example=True,
    )
    adaptive_rerun_on_low_confidence: bool = Field(
        default=False,
        description="If true, run a bounded full pipeline rerun on low confidence to seek a better answer",
        example=True,
    )
    adaptive_rerun_include_generation: bool = Field(
        default=True,
        description="If true, the adaptive rerun may include answer generation; otherwise stops after retrieval/rerank",
        example=True,
    )
    adaptive_rerun_bypass_cache: bool = Field(
        default=False,
        description="If true, the adaptive rerun forces enable_cache=false to avoid stale cache hits",
        example=True,
    )
    adaptive_rerun_time_budget_sec: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Optional hard cap on adaptive rerun wall time; emits rag_phase_budget_exhausted_total on breach",
        example=5.0,
    )
    adaptive_rerun_doc_budget: Optional[int] = Field(
        default=None,
        ge=1,
        description="Optional cap on documents passed to quick-verify during rerun adoption checks",
        example=10,
    )

    # ========== QUERY DECOMPOSITION & MULTI-HOP ==========
    enable_query_decomposition: bool = Field(
        default=False,
        description="Enable guided decomposition into sub-queries for complex questions",
        example=False,
    )
    max_subqueries: int = Field(
        default=4,
        ge=1,
        le=10,
        description="Maximum number of sub-queries to generate (including the primary query)",
        example=4,
    )
    subquery_time_budget_sec: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Soft time budget for decomposition and per-subquery retrieval",
        example=4.0,
    )
    subquery_doc_budget: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum docs per sub-query sent to synthesis",
        example=8,
    )
    subquery_max_concurrency: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum concurrent subquery retrievals",
        example=3,
    )

    # ========== FEEDBACK ==========
    collect_feedback: bool = Field(
        default=False,
        description="Enable feedback collection",
        example=False
    )

    feedback_user_id: Optional[str] = Field(
        default=None,
        description="User ID for feedback tracking",
        example="user123"
    )

    apply_feedback_boost: bool = Field(
        default=False,
        description="Apply feedback-based result boosting",
        example=False
    )

    # ========== MONITORING & ANALYTICS ==========
    enable_monitoring: bool = Field(
        default=False,
        description="Enable performance monitoring",
        example=True
    )

    enable_analytics: bool = Field(
        default=True,
        description="Enable analytics collection (server QA)",
        example=True
    )

    # ========== PERFORMANCE ==========
    use_connection_pool: bool = Field(
        default=True,
        description="Use database connection pooling",
        example=True
    )

    use_embedding_cache: bool = Field(
        default=True,
        description="Use LRU cache for embeddings",
        example=True
    )

    enable_observability: bool = Field(
        default=False,
        description="Enable observability tracing",
        example=False
    )

    trace_id: Optional[str] = Field(
        default=None,
        description="Trace ID for observability",
        example=None
    )

    # ========== PERFORMANCE ==========
    enable_performance_analysis: bool = Field(
        default=False,
        description="Enable detailed performance analysis",
        example=False
    )

    timeout_seconds: Optional[float] = Field(
        default=None,
        ge=1.0,
        le=60.0,
        description="Request timeout in seconds",
        example=10.0
    )

    # ========== QUICK WINS ==========
    highlight_results: bool = Field(
        default=False,
        description="Highlight matching terms in results",
        example=False
    )

    highlight_query_terms: bool = Field(
        default=False,
        description="Highlight query terms specifically",
        example=False
    )

    track_cost: bool = Field(
        default=False,
        description="Track estimated API costs",
        example=False
    )

    debug_mode: bool = Field(
        default=False,
        description="Enable debug logging",
        example=False
    )

    include_rerank_debug_documents: Optional[bool] = Field(
        default=None,
        description="Override whether pre_rerank_documents and reranked_documents debug snapshots are attached when debug_mode is enabled",
        example=False,
    )

    # ========== EXPLAIN / DRY-RUN ==========
    explain_only: bool = Field(
        default=False,
        description="When strategy=agentic and enable_generation=false, return only plan + spans/provenance",
        example=False,
    )

    # ========== SEARCH AGENT / RESEARCH ==========
    search_depth_mode: Optional[Literal["speed", "balanced", "quality"]] = Field(
        default=None,
        description="Search depth mode that sets parameter presets. "
                    "speed: fast with minimal processing, balanced: multi-step with moderate depth, "
                    "quality: deep research with exhaustive coverage. "
                    "Individual parameters override mode presets when explicitly set.",
        example="balanced",
    )
    enable_query_classification: bool = Field(
        default=False,
        description="Enable LLM-based query classification to route queries intelligently. "
                    "Determines whether search is needed, which sources to activate, "
                    "and reformulates conversational follow-ups.",
        example=False,
    )
    chat_history: Optional[list[dict[str, str]]] = Field(
        default=None,
        description="Conversation history for contextual query reformulation. "
                    "Each entry: {\"role\": \"user\"|\"assistant\", \"content\": \"...\"}",
        example=[
            {"role": "user", "content": "Tell me about Python"},
            {"role": "assistant", "content": "Python is a programming language..."},
            {"role": "user", "content": "What are its main libraries?"},
        ],
    )
    enable_query_reformulation: bool = Field(
        default=True,
        description="Enable standalone query reformulation when chat_history is provided. "
                    "Rewrites conversational follow-ups into self-contained queries.",
        example=True,
    )
    enable_research_loop: bool = Field(
        default=False,
        description="Enable iterative agentic research loop. The system iteratively searches, "
                    "analyzes results, and refines queries until sufficient information is found. "
                    "Requires enable_query_classification=true.",
        example=False,
    )
    enable_research_action_dedup: bool = Field(
        default=True,
        description="Skip duplicate research loop actions by reusing prior results for equivalent action signatures.",
        example=True,
    )
    enable_discussion_search: bool = Field(
        default=False,
        description="Enable discussion/forum search (Reddit, StackOverflow, HackerNews) "
                    "as a search source for community knowledge and opinions.",
        example=False,
    )
    discussion_platforms: Optional[list[str]] = Field(
        default=None,
        description="Discussion platforms to search: reddit, stackoverflow, hackernews, quora",
        example=["reddit", "stackoverflow"],
    )
    search_url_scraping: bool = Field(
        default=True,
        description="Enable URL scraping action during iterative research loops. "
                    "When false, the scrape_url action is disabled.",
        example=True,
    )
    enable_research_progress: bool = Field(
        default=False,
        description="Enable research progress events in streaming responses. "
                    "Emits reasoning steps, search actions, and result counts.",
        example=False,
    )
    research_max_iterations: Optional[int] = Field(
        default=None,
        ge=1,
        le=200,
        description="Explicit max iteration override for research loop (takes precedence over mode-specific overrides).",
        example=8,
    )
    research_max_iterations_speed: int = Field(
        default=0,
        ge=0,
        le=200,
        description="Mode-specific max iterations for speed mode (0 = use defaults/config).",
        example=2,
    )
    research_max_iterations_balanced: int = Field(
        default=0,
        ge=0,
        le=200,
        description="Mode-specific max iterations for balanced mode (0 = use defaults/config).",
        example=6,
    )
    research_max_iterations_quality: int = Field(
        default=0,
        ge=0,
        le=300,
        description="Mode-specific max iterations for quality mode (0 = use defaults/config).",
        example=25,
    )
    classifier_provider: Optional[str] = Field(
        default=None,
        description="LLM provider for query classification (uses generation_provider if empty)",
        example="openai",
    )
    classifier_model: Optional[str] = Field(
        default=None,
        description="LLM model for query classification (uses generation_model if empty)",
        example="gpt-4o-mini",
    )

    # ========== FOLLOW-UP SUGGESTIONS ==========
    enable_suggestions: bool = Field(
        default=False,
        description="Generate follow-up question suggestions after the main response. "
                    "Returns 4-5 suggested next questions in the response metadata.",
        example=False,
    )
    num_suggestions: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Number of follow-up suggestions to generate (default 5).",
        example=5,
    )

    # ========== STRUCTURED RESPONSE WRITER ==========
    enable_structured_response: bool = Field(
        default=False,
        description="Use structured response writer with XML-tagged context and citation enforcement. "
                    "Formats retrieval results as indexed XML blocks and requires per-sentence [N] citations. "
                    "In quality mode, enforces a comprehensive research report style (2000+ words).",
        example=False,
    )

    # ========== MEDIA SEARCH ==========
    enable_image_search: bool = Field(
        default=False,
        description="Enable image search as a research action. "
                    "The LLM reformulates queries for optimal image search results.",
        example=False,
    )
    enable_video_search: bool = Field(
        default=False,
        description="Enable video search (primarily YouTube) as a research action. "
                    "The LLM reformulates queries for optimal video search results.",
        example=False,
    )

    # ========== GENERATION GUARDRAILS ==========
    enable_injection_filter: bool = Field(
        default=True,
        description="Detect and down-weight chunks with instruction-injection risk before generation",
        example=True,
    )
    injection_filter_strength: float = Field(
        default=0.5,
        ge=0.05,
        le=1.0,
        description="Multiplicative down-weight factor applied to risky chunks",
        example=0.5,
    )
    require_hard_citations: bool = Field(
        default=False,
        description="Require a supporting span (doc_id + offsets) for each sentence in the answer; attaches hard_citations metadata and may apply low_confidence_behavior",
        example=False,
    )
    enable_numeric_fidelity: bool = Field(
        default=False,
        description="Verify numeric values in the answer appear in sources; optionally retry retrieval or ask/decline",
        example=False,
    )
    numeric_fidelity_behavior: Literal["continue", "ask", "decline", "retry"] = Field(
        default="continue",
        description="Behavior when numeric values are not found in sources",
        example="ask",
    )

    # ========== BATCH PROCESSING ==========
    enable_batch: bool = Field(
        default=False,
        description="Enable batch query processing",
        example=False
    )

    batch_queries: Optional[list[str]] = Field(
        default=None,
        description="Additional queries for batch processing",
        example=["query1", "query2"]
    )

    batch_concurrent: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum concurrent batch queries",
        example=5
    )

    # ========== RESILIENCE ==========
    enable_resilience: bool = Field(
        default=False,
        description="Enable resilience features",
        example=False
    )

    retry_attempts: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Number of retry attempts",
        example=3
    )

    circuit_breaker: bool = Field(
        default=False,
        description="Enable circuit breaker",
        example=False
    )

    # ========== USER CONTEXT ==========
    user_id: Optional[str] = Field(
        default=None,
        description="User ID for personalization",
        example="user123"
    )

    session_id: Optional[str] = Field(
        default=None,
        description="Session ID for tracking",
        example="session456"
    )

    # ========== RETRIEVAL QUALITY METRICS ==========
    ground_truth_doc_ids: Optional[list[str]] = Field(
        default=None,
        description="Ground truth relevant document IDs for computing retrieval metrics "
                    "(precision@K, recall@K, MRR, NDCG@K). When provided, metrics are "
                    "returned in the response metadata.",
        example=["doc_1", "doc_3", "doc_7"]
    )
    metrics_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="K value for @K retrieval metrics (precision@K, recall@K, etc.)"
    )

    # ========== FAITHFULNESS EVALUATION ==========
    enable_faithfulness_eval: bool = Field(
        default=False,
        description="Enable claim-level faithfulness evaluation of generated answers. "
                    "Requires a faithfulness_llm to be passed via kwargs."
    )

    model_config = ConfigDict(json_schema_extra={
        "examples": [
            {
                "query": "What has this project captured about onboarding?",
                "sources": [
                    "media_db",
                    "notes",
                    "chats",
                    "characters",
                    "kanban",
                    "prompts",
                    "world_books",
                    "dictionaries",
                ],
                "expand_query": True,
                "expansion_strategies": ["synonym", "acronym"],
                "enable_citations": True,
                "enable_generation": True,
                "enable_reranking": True,
                "reranking_strategy": "hybrid",
            },
            {
                "query": "Find the reusable prompt guidance for release notes.",
                "sources": ["notes", "prompts"],
                "search_mode": "hybrid",
                "top_k": 8,
            },
        ]
    })

    @field_validator('sources')
    @classmethod
    def validate_sources(cls, v):
        if v is None:
            return normalize_sources_public(None)
        if not isinstance(v, list):
            raise ValueError("sources must be a list of strings")
        if any(not isinstance(source, str) for source in v):
            raise ValueError("sources entries must be strings")
        return normalize_sources_public(v)

    @field_validator('expansion_strategies')
    @classmethod
    def validate_expansion_strategies(cls, v):
        if v:
            valid_strategies = {"acronym", "synonym", "semantic", "domain", "entity"}
            invalid = set(v) - valid_strategies
            if invalid:
                raise ValueError(f"Invalid strategies: {invalid}. Valid options: {valid_strategies}")
        return v

    @field_validator('chunk_type_filter')
    @classmethod
    def validate_chunk_types(cls, v):
        if v:
            valid_types = {"text", "code", "table", "list"}
            invalid = set(v) - valid_types
            if invalid:
                raise ValueError(f"Invalid chunk types: {invalid}. Valid options: {valid_types}")
        return v

    if model_validator is not None:
        @model_validator(mode="after")
        def _validate_research_loop_contract(self):  # type: ignore
            if bool(getattr(self, "enable_research_loop", False)) and not bool(
                getattr(self, "enable_query_classification", False)
            ):
                raise ValueError(
                    "enable_research_loop=true requires enable_query_classification=true"
                )
            return self


KnowledgeSourceIndexStatus = Literal[
    "ready",
    "indexing",
    "stale",
    "empty",
    "unavailable",
    "error",
    "unknown",
]
KnowledgeSourceEmbeddingStatus = Literal[
    "ready",
    "indexing",
    "missing",
    "unavailable",
    "not_applicable",
    "error",
    "unknown",
]


class KnowledgeSourceHealthEntry(BaseModel):
    """Safe pre-query readiness details for a canonical Knowledge QA source."""

    source_id: str
    label: str
    available: bool
    searchable: bool
    item_count: Optional[int] = None
    indexed_count: Optional[int] = None
    last_updated: Optional[str] = None
    last_indexed: Optional[str] = None
    index_status: KnowledgeSourceIndexStatus = "unknown"
    embedding_status: KnowledgeSourceEmbeddingStatus = "unknown"
    disabled_reason: Optional[str] = None
    workspace_scoped: bool = False
    hidden_by_default: bool = False
    privacy_note: Optional[str] = None


class KnowledgeSourceHealthResponse(BaseModel):
    """Read-only source health response for Knowledge QA clients."""

    sources: list[KnowledgeSourceHealthEntry]


class UnifiedRAGResponse(BaseModel):
    """Unified response structure for RAG queries."""

    documents: list[dict[str, Any]] = Field(
        description="Retrieved documents"
    )

    query: str = Field(
        description="Original query"
    )

    expanded_queries: list[str] = Field(
        default_factory=list,
        description="Expanded query variations"
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )

    timings: dict[str, float] = Field(
        default_factory=dict,
        description="Performance timings"
    )

    citations: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Generated citations (academic and chunk-level)"
    )

    academic_citations: list[str] = Field(
        default_factory=list,
        description="Formatted academic citations (MLA/APA/etc)"
    )

    chunk_citations: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Chunk-level citations for answer verification"
    )

    feedback_id: Optional[str] = Field(
        default=None,
        description="Feedback tracking ID"
    )

    generated_answer: Optional[str | dict[str, Any]] = Field(
        default=None,
        description="Generated answer from context"
    )

    cache_hit: bool = Field(
        default=False,
        description="Whether result was from cache"
    )

    errors: list[str] = Field(
        default_factory=list,
        description="Any errors encountered"
    )

    security_report: Optional[dict[str, Any]] = Field(
        default=None,
        description="Security analysis report"
    )

    total_time: float = Field(
        default=0.0,
        description="Total execution time"
    )

    # ========== CLAIMS & FACTUALITY ==========
    claims: Optional[list[dict[str, Any]]] = Field(
        default=None,
        description="Per-claim verification results with evidence",
    )
    factuality: Optional[dict[str, Any]] = Field(
        default=None,
        description="Summary of factuality (supported/refuted/nei, precision, coverage)",
    )
    verification_report: Optional[dict[str, Any]] = Field(
        default=None,
        description="Structured verification report for machine-readable audit (when generate_verification_report=true)",
    )

    # ========== RETRIEVAL QUALITY METRICS ==========
    retrieval_metrics: Optional[dict[str, Any]] = Field(
        default=None,
        description="Retrieval quality metrics (precision@K, recall@K, MRR, NDCG@K, F1@K) "
                    "when ground_truth_doc_ids are provided",
    )

    # ========== FAITHFULNESS EVALUATION ==========
    faithfulness: Optional[dict[str, Any]] = Field(
        default=None,
        description="Faithfulness evaluation results (claim-level analysis, faithfulness score) "
                    "when enable_faithfulness_eval is true",
    )

    # ========== SEARCH AGENT / RESEARCH ==========
    query_classification: Optional[dict[str, Any]] = Field(
        default=None,
        description="Query classification results (skip_search, search routing, detected intent) "
                    "when enable_query_classification is true",
    )
    reformulated_query: Optional[str] = Field(
        default=None,
        description="Standalone reformulated query when chat_history was provided",
    )
    research_summary: Optional[dict[str, Any]] = Field(
        default=None,
        description="Research loop summary (iterations, steps, results) "
                    "when enable_research_loop is true",
    )

    # ========== FOLLOW-UP SUGGESTIONS ==========
    suggestions: Optional[list[str]] = Field(
        default=None,
        description="Follow-up question suggestions generated after the response "
                    "when enable_suggestions is true",
    )

    # ========== MEDIA SEARCH ==========
    images: Optional[list[dict[str, Any]]] = Field(
        default=None,
        description="Image search results when enable_image_search is true. "
                    "Each entry has: title, url, thumbnail_url, source, description.",
    )
    videos: Optional[list[dict[str, Any]]] = Field(
        default=None,
        description="Video search results when enable_video_search is true. "
                    "Each entry has: title, url, thumbnail_url, source, description.",
    )

    model_config = ConfigDict(frozen=True, json_schema_extra={
        "example": {
            "documents": [
                {
                    "id": "doc1",
                    "content": "Machine learning is a subset of artificial intelligence...",
                    "score": 0.95,
                    "metadata": {"source": "media_db", "title": "ML Introduction"}
                }
            ],
            "query": "What is machine learning?",
            "expanded_queries": ["machine learning definition", "ML explanation"],
            "metadata": {
                "sources_searched": ["media_db", "notes"],
                "documents_retrieved": 10,
                "cache_hit": False
            },
            "timings": {
                "query_expansion": 0.05,
                "retrieval": 0.2,
                "reranking": 0.1,
                "total": 0.35
            },
            "citations": [
                {
                    "text": "Machine learning is a subset of artificial intelligence",
                    "source": "ML Introduction",
                    "confidence": 0.95,
                    "type": "exact"
                }
            ],
            "generated_answer": "Machine learning is a branch of AI that enables systems to learn from data...",
            "cache_hit": False,
            "errors": [],
            "total_time": 0.35
        }
    })


class UnifiedBatchRequest(BaseModel):
    """Request for batch processing multiple queries."""

    queries: list[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of queries to process",
        example=["What is AI?", "Explain neural networks", "Define machine learning"]
    )

    max_concurrent: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum concurrent processing",
        example=5
    )

    enable_checkpoint: bool = Field(
        default=False,
        description="Enable incremental checkpointing for batch runs (supports resume).",
    )

    # Include all optional parameters from UnifiedRAGRequest that will be applied to all queries
    # Data Sources
    sources: Optional[list[str]] = Field(
        default=["media_db", "notes", "characters"],
        description=(
            "Databases to search: media_db, notes, chats, characters, kanban, "
            "prompts, world_books, dictionaries, sql"
        ),
    )
    # Indexing / Namespace
    corpus: Optional[str] = Field(default=None, description="Alias for index_namespace")
    index_namespace: Optional[str] = Field(default=None, description="Corpus/namespace identifier")

    @field_validator("sources", mode="before")
    @classmethod
    def _validate_batch_sources(cls, v: Any) -> list[str]:
        if v is None:
            return normalize_sources_public(None)
        if not isinstance(v, list):
            raise ValueError("sources must be a list of strings")
        if any(not isinstance(source, str) for source in v):
            raise ValueError("sources entries must be strings")
        return normalize_sources_public(v)

    # Search Configuration
    search_mode: Literal["fts", "vector", "hybrid"] = Field(default="hybrid")
    hybrid_alpha: float = Field(default=0.7, ge=0.0, le=1.0)
    enable_intent_routing: bool = Field(default=False)
    top_k: int = Field(default=10, ge=1, le=100)
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)
    adaptive_advanced_rewrites: Optional[bool] = Field(default=None)

    # Query Expansion
    expand_query: bool = Field(default=False)
    expansion_strategies: Optional[list[str]] = Field(default=None)
    spell_check: bool = Field(default=False)
    max_query_variations: int = Field(default=3, ge=0, le=10)

    # Caching
    enable_cache: bool = Field(default=True)
    cache_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    adaptive_cache: bool = Field(default=True)
    enable_text_late_chunking: bool = Field(default=False)
    chunk_method: Optional[Literal["semantic", "tokens", "paragraphs", "sentences", "words", "ebook_chapters", "json", "propositions"]] = Field(
        default=None,
        description="Transient text late chunking method for this batch query set",
    )
    chunk_size: Optional[int] = Field(default=None, gt=0)
    chunk_overlap: Optional[int] = Field(default=None, ge=0)
    chunk_language: Optional[str] = Field(default=None)

    # Filtering
    keyword_filter: Optional[list[str]] = Field(default=None)

    # Security & Privacy
    enable_security_filter: bool = Field(default=False)
    detect_pii: bool = Field(default=False)
    redact_pii: bool = Field(default=False)
    sensitivity_level: Literal["public", "internal", "confidential", "restricted"] = Field(default="public")
    content_filter: bool = Field(default=False)

    # Document Processing
    enable_table_processing: bool = Field(default=False)
    table_method: Literal["markdown", "html", "hybrid"] = Field(default="markdown")
    # VLM late chunking
    enable_vlm_late_chunking: bool = Field(default=False)
    vlm_backend: Optional[str] = Field(default=None)
    vlm_detect_tables_only: bool = Field(default=True)
    vlm_max_pages: Optional[int] = Field(default=None, ge=1, le=1000)
    vlm_late_chunk_top_k_docs: int = Field(default=3, ge=1, le=50)

    # Chunking & Context
    enable_enhanced_chunking: bool = Field(default=_DEF_ENH_CHUNK)
    chunk_type_filter: Optional[list[str]] = Field(default=None)
    enable_parent_expansion: bool = Field(default=False)
    parent_context_size: int = Field(default=500, ge=100, le=2000)
    include_sibling_chunks: bool = Field(default=False)
    sibling_window: int = Field(default=_DEF_SIB_WIN, ge=0, le=50)
    include_parent_document: bool = Field(default=_DEF_INC_PARENT)
    parent_max_tokens: Optional[int] = Field(default=_DEF_PARENT_MAX_TOK, ge=1)

    @field_validator("chunk_overlap")
    @classmethod
    def _validate_batch_chunk_overlap(cls, v, info):
        if v is None:
            return v
        chunk_size = info.data.get("chunk_size")
        if chunk_size is not None and v >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return v

    # Advanced retrieval
    enable_multi_vector_passages: bool = Field(default=False)
    mv_span_chars: int = Field(default=300, ge=100, le=2000)
    mv_stride: int = Field(default=150, ge=50, le=1000)
    mv_max_spans: int = Field(default=8, ge=1, le=64)
    mv_flatten_to_spans: bool = Field(default=False)
    enable_numeric_table_boost: bool = Field(default=False)

    # Reranking
    enable_reranking: bool = Field(default=True)
    reranking_strategy: Literal["flashrank", "cross_encoder", "hybrid", "llama_cpp", "llm_scoring", "two_tier", "none"] = Field(default="flashrank")
    rerank_top_k: Optional[int] = Field(default=None, ge=1, le=100)
    reranking_model: Optional[str] = Field(default=None)
    rerank_min_relevance_prob: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    rerank_sentinel_margin: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    # Citations
    enable_citations: bool = Field(default=False)
    citation_style: Literal["apa", "mla", "chicago", "harvard"] = Field(default="apa")
    include_page_numbers: bool = Field(default=False)

    # Answer Generation
    enable_generation: bool = Field(default=False)
    strict_extractive: bool = Field(default=False)
    generation_model: Optional[str] = Field(default=None)
    generation_provider: Optional[str] = Field(default=None)
    generation_prompt: Optional[str] = Field(default=None)
    max_generation_tokens: int = Field(default=500, ge=50, le=4000)
    # Search Agent / Round 2 enhancements
    enable_suggestions: bool = Field(default=False)
    num_suggestions: int = Field(default=5, ge=1, le=10)
    enable_structured_response: bool = Field(default=False)
    enable_image_search: bool = Field(default=False)
    enable_video_search: bool = Field(default=False)
    enable_abstention: bool = Field(default=False)
    abstention_behavior: Literal["continue", "ask", "decline"] = Field(default="continue")
    enable_multi_turn_synthesis: bool = Field(default=False)
    synthesis_time_budget_sec: Optional[float] = Field(default=None, ge=0.0)
    synthesis_draft_tokens: Optional[int] = Field(default=None, ge=32, le=4000)
    synthesis_refine_tokens: Optional[int] = Field(default=None, ge=32, le=4000)
    # Guardrails
    enable_content_policy_filter: bool = Field(default=False)
    content_policy_types: Optional[list[str]] = Field(default=None)
    content_policy_mode: Literal["redact", "drop", "annotate"] = Field(default="redact")
    enable_html_sanitizer: bool = Field(default=False)
    html_allowed_tags: Optional[list[str]] = Field(default=None)
    html_allowed_attrs: Optional[list[str]] = Field(default=None)
    ocr_confidence_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    # Post-Verification (Adaptive)
    enable_post_verification: bool = Field(default=False)
    adaptive_max_retries: int = Field(default=1, ge=0, le=3)
    adaptive_unsupported_threshold: float = Field(default=0.15, ge=0.0, le=1.0)
    adaptive_max_claims: int = Field(default=20, ge=1, le=100)
    adaptive_time_budget_sec: Optional[float] = Field(default=None, ge=0.0)
    low_confidence_behavior: Literal["continue", "ask", "decline"] = Field(default="continue")
    adaptive_advanced_rewrites: Optional[bool] = Field(default=None)
    adaptive_rerun_on_low_confidence: bool = Field(default=False)
    adaptive_rerun_include_generation: bool = Field(default=True)
    adaptive_rerun_bypass_cache: bool = Field(default=False)
    adaptive_rerun_time_budget_sec: Optional[float] = Field(default=None, ge=0.0)
    adaptive_rerun_doc_budget: Optional[int] = Field(default=None, ge=1)

    # Query Decomposition & Multi-hop
    enable_query_decomposition: bool = Field(default=False)
    max_subqueries: int = Field(default=4, ge=1, le=10)
    subquery_time_budget_sec: Optional[float] = Field(default=None, ge=0.0)
    subquery_doc_budget: Optional[int] = Field(default=None, ge=1)
    subquery_max_concurrency: int = Field(default=3, ge=1, le=10)

    # Feedback
    collect_feedback: bool = Field(default=False)
    feedback_user_id: Optional[str] = Field(default=None)
    apply_feedback_boost: bool = Field(default=False)

    # Monitoring
    enable_monitoring: bool = Field(default=False)
    enable_observability: bool = Field(default=False)
    trace_id: Optional[str] = Field(default=None)

    # Performance
    enable_performance_analysis: bool = Field(default=False)
    timeout_seconds: Optional[float] = Field(default=None, ge=1.0, le=60.0)

    # Quick Wins
    highlight_results: bool = Field(default=False)
    highlight_query_terms: bool = Field(default=False)
    track_cost: bool = Field(default=False)
    debug_mode: bool = Field(default=False)

    # Generation Guardrails
    enable_injection_filter: bool = Field(default=True)
    injection_filter_strength: float = Field(default=0.5, ge=0.05, le=1.0)
    require_hard_citations: bool = Field(default=False)
    enable_numeric_fidelity: bool = Field(default=False)
    numeric_fidelity_behavior: Literal["continue", "ask", "decline", "retry"] = Field(default="continue")

    # Batch Processing (excluding batch fields as this IS a batch request)
    # enable_batch, batch_queries, batch_concurrent are not included

    # Resilience
    enable_resilience: bool = Field(default=False)
    retry_attempts: int = Field(default=3, ge=1, le=5)
    circuit_breaker: bool = Field(default=False)

    # User Context
    user_id: Optional[str] = Field(default=None)
    session_id: Optional[str] = Field(default=None)

    # ========== RETRIEVAL QUALITY METRICS ==========
    ground_truth_doc_ids: Optional[list[str]] = Field(
        default=None,
        description="Ground truth relevant document IDs for computing retrieval metrics "
                    "(precision@K, recall@K, MRR, NDCG@K). Applied to each query in the batch.",
    )
    metrics_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="K value for @K retrieval metrics (precision@K, recall@K, etc.)",
    )

    # ========== FAITHFULNESS EVALUATION ==========
    enable_faithfulness_eval: bool = Field(
        default=False,
        description="Enable claim-level faithfulness evaluation of generated answers.",
    )

    if model_validator is not None:
        @model_validator(mode="before")
        def _map_legacy_min_relevance_batch(cls, values):  # type: ignore
            if isinstance(values, dict) and "min_relevance_score" in values and "min_score" not in values:
                values["min_score"] = values.get("min_relevance_score")
            return values
    # Batch: Map corpus -> index_namespace at model-level
    if model_validator is not None:
        @model_validator(mode="before")
        def _alias_corpus_namespace_batch(cls, values):  # type: ignore
            if isinstance(values, dict):
                v = values.get("index_namespace")
                corpus = values.get("corpus")
                if (v is None or (isinstance(v, str) and not v.strip())) and isinstance(corpus, str):
                    values["index_namespace"] = corpus.strip() or v
            return values

    model_config = ConfigDict(json_schema_extra={
        "examples": [
            {
                "queries": ["What is AI?", "Explain neural networks"],
                "sources": [
                    "media_db",
                    "notes",
                    "chats",
                    "characters",
                    "kanban",
                    "prompts",
                    "world_books",
                    "dictionaries",
                ],
                "max_concurrent": 5,
                "expand_query": True,
                "enable_citations": True,
                "enable_reranking": True,
            },
            {
                "queries": ["Summarize release-note prompt guidance"],
                "sources": ["notes", "prompts"],
                "max_concurrent": 2,
            },
        ]
    })


class UnifiedBatchResponse(BaseModel):
    """Response for batch processing."""

    results: list[UnifiedRAGResponse] = Field(
        description="Results for each query"
    )

    total_queries: int = Field(
        description="Total number of queries processed"
    )

    successful: int = Field(
        description="Number of successful queries"
    )

    failed: int = Field(
        description="Number of failed queries"
    )

    total_time: float = Field(
        description="Total batch processing time"
    )

    checkpoint_id: Optional[str] = Field(
        default=None,
        description="Checkpoint ID when checkpointing is enabled.",
    )

    model_config = ConfigDict(frozen=True, json_schema_extra={
        "example": {
            "results": [],  # List of UnifiedRAGResponse objects
            "total_queries": 2,
            "successful": 2,
            "failed": 0,
            "total_time": 0.75,
            "checkpoint_id": "rag_batch_ab12cd34",
        }
    })


class ImplicitFeedbackEvent(BaseModel):
    """Schema for implicit feedback signals from WebUI (click/expand/copy/dwell/citation)."""
    event_type: Literal["click", "expand", "copy", "dwell_time", "citation_used"] = Field(
        description="Type of implicit event"
    )
    query: Optional[str] = Field(default=None, description="Original query text")
    feedback_id: Optional[str] = Field(default=None, description="Optional feedback correlation id")
    doc_id: Optional[str] = Field(default=None, description="Document/chunk id involved")
    chunk_ids: Optional[list[str]] = Field(default=None, description="Optional chunk ids involved")
    rank: Optional[int] = Field(default=None, description="Rank position of the doc in the displayed list")
    impression_list: Optional[list[str]] = Field(
        default=None,
        description="Ordered doc ids visible when the event happened"
    )
    corpus: Optional[str] = Field(default=None, description="Corpus/namespace if set in the request")
    user_id: Optional[str] = Field(default=None, description="User id if available")
    session_id: Optional[str] = Field(default=None, description="Browser session id if available")
    conversation_id: Optional[str] = Field(default=None, description="Conversation id if available")
    message_id: Optional[str] = Field(default=None, description="Message id if available")
    dwell_ms: Optional[int] = Field(default=None, ge=0, description="Dwell time in milliseconds")

    if model_validator is not None:
        @model_validator(mode="before")
        def _validate_dwell_ms(cls, values):  # type: ignore
            if isinstance(values, dict) and values.get("event_type") == "dwell_time":
                dwell_ms = values.get("dwell_ms")
                if dwell_ms is None:
                    raise ValueError("dwell_ms is required when event_type='dwell_time'")
            return values

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "event_type": "dwell_time",
            "query": "how to reset auth",
            "doc_id": "doc_1",
            "chunk_ids": ["chunk_9"],
            "rank": 2,
            "impression_list": ["doc_1", "doc_2", "doc_3"],
            "corpus": "media_db",
            "session_id": "sess_abc123",
            "conversation_id": "C_...",
            "message_id": "M_...",
            "dwell_ms": 3000,
        }
    })
