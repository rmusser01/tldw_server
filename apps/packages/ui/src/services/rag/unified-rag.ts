export type RagSource =
  | "media_db"
  | "notes"
  | "chats"
  | "characters"
  | "kanban"
  | "prompts"
  | "world_books"
  | "dictionaries"
export type RagStrategy = "standard" | "agentic"
export type RagSearchMode = "fts" | "vector" | "hybrid"
export type RagFtsLevel = "media" | "chunk"
export type RagTextChunkMethod =
  | "semantic"
  | "tokens"
  | "paragraphs"
  | "sentences"
  | "words"
  | "ebook_chapters"
  | "json"
  | "propositions"
export type RagExpansionStrategy =
  | "acronym"
  | "synonym"
  | "semantic"
  | "domain"
  | "entity"
export type RagSensitivityLevel =
  | "public"
  | "internal"
  | "confidential"
  | "restricted"
export type RagTableMethod = "markdown" | "html" | "hybrid"
export type RagChunkType = "text" | "code" | "table" | "list"
export type RagClaimExtractor = "aps" | "claimify" | "ner" | "auto"
export type RagClaimVerifier = "nli" | "llm" | "hybrid"
export type RagRerankStrategy =
  | "flashrank"
  | "cross_encoder"
  | "hybrid"
  | "llama_cpp"
  | "llm_scoring"
  | "two_tier"
  | "none"
export type RagCitationStyle =
  | "apa"
  | "mla"
  | "chicago"
  | "harvard"
  | "ieee"
export type RagAbstentionBehavior = "continue" | "ask" | "decline"
export type RagContentPolicyType = "pii" | "phi"
export type RagContentPolicyMode = "redact" | "drop" | "annotate"
export type RagLowConfidenceBehavior = "continue" | "ask" | "decline"
export type RagNumericFidelityBehavior = "continue" | "ask" | "decline" | "retry"
export type RagNumericPrecisionMode = "standard" | "strict" | "academic"
export type RagPresetName = "fast" | "balanced" | "thorough" | "custom"
export type RagProfile = "fast" | "balanced" | "accuracy" | "none"

export type RagSettings = {
  query: string
  sources: RagSource[]
  strategy: RagStrategy
  rag_profile: RagProfile
  corpus: string
  index_namespace: string
  search_mode: RagSearchMode
  fts_level: RagFtsLevel
  enable_text_late_chunking: boolean
  chunk_method: RagTextChunkMethod
  chunk_size: number
  chunk_overlap: number
  chunk_language: string
  hybrid_alpha: number
  enable_intent_routing: boolean
  top_k: number
  min_score: number
  expand_query: boolean
  expansion_strategies: RagExpansionStrategy[]
  spell_check: boolean
  max_query_variations: number
  enable_cache: boolean
  cache_threshold: number
  adaptive_cache: boolean
  keyword_filter: string
  include_media_ids: number[]
  include_note_ids: string[]
  enable_security_filter: boolean
  detect_pii: boolean
  redact_pii: boolean
  sensitivity_level: RagSensitivityLevel
  content_filter: boolean
  enable_table_processing: boolean
  table_method: RagTableMethod
  enable_vlm_late_chunking: boolean
  vlm_backend: string | null
  vlm_detect_tables_only: boolean
  vlm_max_pages: number
  vlm_late_chunk_top_k_docs: number
  chunk_type_filter: RagChunkType[]
  enable_parent_expansion: boolean
  parent_context_size: number
  include_sibling_chunks: boolean
  sibling_window: number
  include_parent_document: boolean
  parent_max_tokens: number
  agentic_top_k_docs: number
  agentic_window_chars: number
  agentic_max_tokens_read: number
  agentic_max_tool_calls: number
  agentic_extractive_only: boolean
  agentic_quote_spans: boolean
  agentic_debug_trace: boolean
  agentic_enable_tools: boolean
  agentic_use_llm_planner: boolean
  agentic_time_budget_sec: number
  agentic_cache_ttl_sec: number
  agentic_enable_query_decomposition: boolean
  agentic_subgoal_max: number
  agentic_enable_semantic_within: boolean
  agentic_enable_section_index: boolean
  agentic_prefer_structural_anchors: boolean
  agentic_enable_table_support: boolean
  agentic_enable_vlm_late_chunking: boolean
  agentic_vlm_backend: string | null
  agentic_vlm_detect_tables_only: boolean
  agentic_vlm_max_pages: number
  agentic_vlm_late_chunk_top_k_docs: number
  agentic_use_provider_embeddings_within: boolean
  agentic_provider_embedding_model_id: string
  agentic_adaptive_budgets: boolean
  agentic_coverage_target: number
  agentic_min_corroborating_docs: number
  agentic_max_redundancy: number
  agentic_enable_metrics: boolean
  enable_multi_vector_passages: boolean
  mv_span_chars: number
  mv_stride: number
  mv_max_spans: number
  mv_flatten_to_spans: boolean
  enable_numeric_table_boost: boolean
  enable_claims: boolean
  claim_extractor: RagClaimExtractor
  claim_verifier: RagClaimVerifier
  claims_top_k: number
  claims_conf_threshold: number
  claims_max: number
  claims_concurrency: number
  nli_model: string
  numeric_precision_mode: RagNumericPrecisionMode
  doc_only_verification: boolean
  generate_verification_report: boolean
  enable_dynamic_granularity: boolean
  enable_evidence_accumulation: boolean
  accumulation_max_rounds: number
  accumulation_time_budget_sec: number | null
  enable_evidence_chains: boolean
  enable_document_grading: boolean
  grading_threshold: number
  grading_model: string | null
  grading_provider: string | null
  grading_batch_size: number
  grading_timeout_sec: number
  grading_fallback_to_score: boolean
  grading_fallback_min_score: number
  enable_query_rewriting_loop: boolean
  max_rewrite_attempts: number
  rewrite_relevance_threshold: number
  enable_reranking: boolean
  reranking_strategy: RagRerankStrategy
  rerank_top_k: number
  reranking_model: string
  rerank_min_relevance_prob: number
  rerank_sentinel_margin: number
  enable_citations: boolean
  citation_style: RagCitationStyle
  include_page_numbers: boolean
  enable_chunk_citations: boolean
  enable_generation: boolean
  strict_extractive: boolean
  generation_model: string | null
  generation_provider: string | null
  generation_prompt: string | null
  max_generation_tokens: number
  enable_abstention: boolean
  abstention_behavior: RagAbstentionBehavior
  enable_multi_turn_synthesis: boolean
  synthesis_time_budget_sec: number
  synthesis_draft_tokens: number
  synthesis_refine_tokens: number
  enable_content_policy_filter: boolean
  content_policy_types: RagContentPolicyType[]
  content_policy_mode: RagContentPolicyMode
  enable_html_sanitizer: boolean
  html_allowed_tags: string[]
  html_allowed_attrs: string[]
  ocr_confidence_threshold: number
  enable_post_verification: boolean
  enable_web_fallback: boolean
  web_fallback_threshold: number
  web_search_engine: string
  web_fallback_result_count: number
  web_fallback_merge_strategy: "prepend" | "append" | "interleave"
  enable_knowledge_strips: boolean
  strip_size_tokens: number
  strip_min_relevance: number
  max_strips: number
  enable_fast_hallucination_check: boolean
  fast_hallucination_timeout_sec: number
  fast_hallucination_provider: string | null
  fast_hallucination_model: string | null
  enable_utility_grading: boolean
  utility_grading_timeout_sec: number
  utility_grading_provider: string | null
  utility_grading_model: string | null
  adaptive_max_retries: number
  adaptive_unsupported_threshold: number
  adaptive_max_claims: number
  adaptive_time_budget_sec: number
  low_confidence_behavior: RagLowConfidenceBehavior
  adaptive_advanced_rewrites: boolean
  adaptive_rerun_on_low_confidence: boolean
  adaptive_rerun_include_generation: boolean
  adaptive_rerun_bypass_cache: boolean
  adaptive_rerun_time_budget_sec: number
  adaptive_rerun_doc_budget: number
  enable_query_decomposition: boolean
  max_subqueries: number
  subquery_time_budget_sec: number | null
  subquery_doc_budget: number | null
  subquery_max_concurrency: number
  collect_feedback: boolean
  feedback_user_id: string
  apply_feedback_boost: boolean
  enable_monitoring: boolean
  enable_analytics: boolean
  enable_observability: boolean
  trace_id: string
  use_connection_pool: boolean
  use_embedding_cache: boolean
  enable_performance_analysis: boolean
  timeout_seconds: number
  highlight_results: boolean
  highlight_query_terms: boolean
  track_cost: boolean
  debug_mode: boolean
  explain_only: boolean
  enable_injection_filter: boolean
  injection_filter_strength: number
  require_hard_citations: boolean
  enable_numeric_fidelity: boolean
  numeric_fidelity_behavior: RagNumericFidelityBehavior
  enable_batch: boolean
  batch_queries: string[]
  batch_concurrent: number
  enable_resilience: boolean
  retry_attempts: number
  circuit_breaker: boolean
  ground_truth_doc_ids: string[]
  metrics_k: number
  enable_faithfulness_eval: boolean
  user_id: string | null
  session_id: string | null
}

export const DEFAULT_RAG_SETTINGS: RagSettings = {
  query: "",
  sources: ["media_db", "notes", "characters", "chats"],
  strategy: "standard",
  rag_profile: "none",
  corpus: "",
  index_namespace: "",
  search_mode: "hybrid",
  fts_level: "chunk",
  enable_text_late_chunking: false,
  chunk_method: "sentences",
  chunk_size: 500,
  chunk_overlap: 50,
  chunk_language: "",
  hybrid_alpha: 0.5,
  enable_intent_routing: true,
  top_k: 8,
  min_score: 0.2,
  expand_query: false,
  expansion_strategies: ["acronym", "synonym", "semantic", "domain", "entity"],
  spell_check: true,
  max_query_variations: 3,
  enable_cache: true,
  cache_threshold: 0.8,
  adaptive_cache: true,
  keyword_filter: "",
  include_media_ids: [],
  include_note_ids: [],
  enable_security_filter: true,
  detect_pii: true,
  redact_pii: false,
  sensitivity_level: "internal",
  content_filter: true,
  enable_table_processing: true,
  table_method: "hybrid",
  enable_vlm_late_chunking: false,
  vlm_backend: "auto",
  vlm_detect_tables_only: false,
  vlm_max_pages: 20,
  vlm_late_chunk_top_k_docs: 5,
  chunk_type_filter: ["text", "code", "table", "list"],
  enable_parent_expansion: true,
  parent_context_size: 500,
  include_sibling_chunks: true,
  sibling_window: 1,
  include_parent_document: false,
  parent_max_tokens: 800,
  agentic_top_k_docs: 12,
  agentic_window_chars: 5000,
  agentic_max_tokens_read: 5000,
  agentic_max_tool_calls: 6,
  agentic_extractive_only: false,
  agentic_quote_spans: true,
  agentic_debug_trace: false,
  agentic_enable_tools: true,
  agentic_use_llm_planner: true,
  agentic_time_budget_sec: 30,
  agentic_cache_ttl_sec: 600,
  agentic_enable_query_decomposition: true,
  agentic_subgoal_max: 5,
  agentic_enable_semantic_within: true,
  agentic_enable_section_index: true,
  agentic_prefer_structural_anchors: true,
  agentic_enable_table_support: true,
  agentic_enable_vlm_late_chunking: false,
  agentic_vlm_backend: "auto",
  agentic_vlm_detect_tables_only: false,
  agentic_vlm_max_pages: 20,
  agentic_vlm_late_chunk_top_k_docs: 5,
  agentic_use_provider_embeddings_within: false,
  agentic_provider_embedding_model_id: "",
  agentic_adaptive_budgets: true,
  agentic_coverage_target: 0.8,
  agentic_min_corroborating_docs: 2,
  agentic_max_redundancy: 0.9,
  agentic_enable_metrics: false,
  enable_multi_vector_passages: false,
  mv_span_chars: 1200,
  mv_stride: 400,
  mv_max_spans: 5,
  mv_flatten_to_spans: true,
  enable_numeric_table_boost: false,
  enable_claims: false,
  claim_extractor: "auto",
  claim_verifier: "hybrid",
  claims_top_k: 8,
  claims_conf_threshold: 0.5,
  claims_max: 20,
  claims_concurrency: 4,
  nli_model: "default",
  numeric_precision_mode: "standard",
  doc_only_verification: false,
  generate_verification_report: false,
  enable_dynamic_granularity: false,
  enable_evidence_accumulation: false,
  accumulation_max_rounds: 3,
  accumulation_time_budget_sec: null,
  enable_evidence_chains: false,
  enable_document_grading: false,
  grading_threshold: 0.5,
  grading_model: null,
  grading_provider: null,
  grading_batch_size: 5,
  grading_timeout_sec: 30,
  grading_fallback_to_score: true,
  grading_fallback_min_score: 0.3,
  enable_query_rewriting_loop: false,
  max_rewrite_attempts: 2,
  rewrite_relevance_threshold: 0.3,
  enable_reranking: true,
  reranking_strategy: "flashrank",
  rerank_top_k: 20,
  reranking_model: "default",
  rerank_min_relevance_prob: 0.2,
  rerank_sentinel_margin: 0.05,
  enable_citations: true,
  citation_style: "apa",
  include_page_numbers: true,
  enable_chunk_citations: true,
  enable_generation: true,
  strict_extractive: false,
  generation_model: null,
  generation_provider: null,
  generation_prompt: null,
  max_generation_tokens: 800,
  enable_abstention: true,
  abstention_behavior: "ask",
  enable_multi_turn_synthesis: false,
  synthesis_time_budget_sec: 45,
  synthesis_draft_tokens: 400,
  synthesis_refine_tokens: 400,
  enable_content_policy_filter: true,
  content_policy_types: ["pii", "phi"],
  content_policy_mode: "redact",
  enable_html_sanitizer: true,
  html_allowed_tags: [
    "b",
    "i",
    "em",
    "strong",
    "a",
    "code",
    "pre",
    "ul",
    "ol",
    "li",
    "p",
    "br",
    "blockquote"
  ],
  html_allowed_attrs: ["href", "title", "target", "rel"],
  ocr_confidence_threshold: 0.6,
  enable_post_verification: false,
  enable_web_fallback: false,
  web_fallback_threshold: 0.25,
  web_search_engine: "duckduckgo",
  web_fallback_result_count: 5,
  web_fallback_merge_strategy: "prepend",
  enable_knowledge_strips: false,
  strip_size_tokens: 100,
  strip_min_relevance: 0.3,
  max_strips: 20,
  enable_fast_hallucination_check: false,
  fast_hallucination_timeout_sec: 5,
  fast_hallucination_provider: null,
  fast_hallucination_model: null,
  enable_utility_grading: false,
  utility_grading_timeout_sec: 5,
  utility_grading_provider: null,
  utility_grading_model: null,
  adaptive_max_retries: 1,
  adaptive_unsupported_threshold: 0.4,
  adaptive_max_claims: 20,
  adaptive_time_budget_sec: 30,
  low_confidence_behavior: "ask",
  adaptive_advanced_rewrites: false,
  adaptive_rerun_on_low_confidence: true,
  adaptive_rerun_include_generation: true,
  adaptive_rerun_bypass_cache: false,
  adaptive_rerun_time_budget_sec: 30,
  adaptive_rerun_doc_budget: 12,
  enable_query_decomposition: false,
  max_subqueries: 4,
  subquery_time_budget_sec: null,
  subquery_doc_budget: null,
  subquery_max_concurrency: 3,
  collect_feedback: false,
  feedback_user_id: "",
  apply_feedback_boost: false,
  enable_monitoring: false,
  enable_analytics: false,
  enable_observability: false,
  trace_id: "",
  use_connection_pool: true,
  use_embedding_cache: true,
  enable_performance_analysis: false,
  timeout_seconds: 45,
  highlight_results: true,
  highlight_query_terms: true,
  track_cost: false,
  debug_mode: false,
  explain_only: false,
  enable_injection_filter: true,
  injection_filter_strength: 0.7,
  require_hard_citations: false,
  enable_numeric_fidelity: false,
  numeric_fidelity_behavior: "ask",
  enable_batch: false,
  batch_queries: [],
  batch_concurrent: 2,
  enable_resilience: true,
  retry_attempts: 2,
  circuit_breaker: true,
  ground_truth_doc_ids: [],
  metrics_k: 10,
  enable_faithfulness_eval: false,
  user_id: null,
  session_id: null
}

export const RAG_PRESET_OVERRIDES: Record<
  Exclude<RagPresetName, "custom">,
  Partial<RagSettings>
> = {
  fast: {
    search_mode: "fts",
    top_k: 5,
    enable_reranking: false,
    enable_citations: false,
    max_generation_tokens: 300,
    timeout_seconds: 20
  },
  balanced: {},
  thorough: {
    top_k: 20,
    min_score: 0.1,
    enable_reranking: true,
    rerank_top_k: 50,
    enable_citations: true,
    enable_claims: true,
    enable_post_verification: true,
    enable_multi_turn_synthesis: true,
    synthesis_time_budget_sec: 90,
    max_generation_tokens: 1200
  }
}

export const applyRagPreset = (
  preset: Exclude<RagPresetName, "custom">,
  base: RagSettings = DEFAULT_RAG_SETTINGS
): RagSettings => ({
  ...base,
  ...RAG_PRESET_OVERRIDES[preset]
})

export const buildRagSearchRequest = (settings: RagSettings) => {
  const clampNumber = (
    value: number,
    min: number,
    max: number,
    fallback: number
  ): number => {
    if (!Number.isFinite(value)) return fallback
    if (value < min) return min
    if (value > max) return max
    return value
  }

  const normalizedSettings: RagSettings = {
    ...settings,
    // Guard against stale persisted values from earlier builds that violate
    // current backend validation constraints.
    parent_context_size: clampNumber(settings.parent_context_size, 100, 2000, 100),
    agentic_time_budget_sec: clampNumber(settings.agentic_time_budget_sec, 0.1, 30, 30),
    agentic_max_redundancy: clampNumber(settings.agentic_max_redundancy, 0, 1, 1),
    chunk_size: Math.max(1, Math.round(settings.chunk_size)),
    chunk_overlap: Math.max(
      0,
      Math.min(Math.round(settings.chunk_overlap), Math.max(0, Math.round(settings.chunk_size) - 1))
    ),
  }

  const {
    query,
    timeout_seconds,
    generation_model,
    generation_provider,
    generation_prompt,
    user_id,
    session_id,
    rag_profile,
    ...rest
  } = normalizedSettings
  const options: Record<string, unknown> = {}
  for (const [key, value] of Object.entries(rest)) {
    if (value === undefined || value === null) continue
    if (typeof value === "string" && value.trim() === "") continue
    if (Array.isArray(value) && value.length === 0) continue
    if (
      !normalizedSettings.enable_text_late_chunking &&
      (key === "chunk_method" ||
        key === "chunk_size" ||
        key === "chunk_overlap" ||
        key === "chunk_language")
    ) {
      continue
    }
    options[key] = value
  }
  if (generation_model) options.generation_model = generation_model
  if (generation_provider) options.generation_provider = generation_provider
  if (generation_prompt) options.generation_prompt = generation_prompt
  if (rag_profile && rag_profile !== "none") options.rag_profile = rag_profile
  if (user_id) options.user_id = user_id
  if (session_id) options.session_id = session_id
  const timeoutMs =
    Number.isFinite(timeout_seconds) && timeout_seconds > 0
      ? Math.round(timeout_seconds * 1000)
      : undefined
  return {
    query: query.trim(),
    options,
    timeoutMs
  }
}

export const toRagAdvancedOptions = (settings: RagSettings) => {
  const { query, search_mode, top_k, enable_generation, enable_citations, sources, rag_profile, ...rest } = settings
  const options: Record<string, unknown> = {}
  for (const [key, value] of Object.entries(rest)) {
    if (value === undefined || value === null) continue
    if (typeof value === "string" && value.trim() === "") continue
    if (Array.isArray(value) && value.length === 0) continue
    if (
      !settings.enable_text_late_chunking &&
      (key === "chunk_method" ||
        key === "chunk_size" ||
        key === "chunk_overlap" ||
        key === "chunk_language")
    ) {
      continue
    }
    options[key] = value
  }
  if (rag_profile && rag_profile !== "none") options.rag_profile = rag_profile
  return options
}
