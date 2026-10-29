# RAG retrieval

This document describes the unified retrieval pipeline: request profiles, query transformation, hybrid FTS+vector retrieval with weighted RRF fusion, cross-source ordering, reranking, generation with hard citations/abstention, and the deep-search webresearch endpoints the chatbook research engine reuses.

## Authoritative files

All paths under `tldw_Server_API/` unless noted.

| Concern | File | Key symbols |
| --- | --- | --- |
| Routes | `app/api/v1/endpoints/rag_unified.py` | all `/api/v1/rag/*`; the unified search endpoint, ablation, capabilities |
| Schemas | `app/core/RAG/rag_schemas_unified.py`, re-exported by `app/api/v1/schemas/rag_schemas_unified.py` | `UnifiedRAGRequest/Response`, batch variants |
| Pipeline | `app/core/RAG/rag_service/unified_pipeline.py` | `unified_rag_pipeline`, `unified_batch_pipeline`, `simple_search` / `advanced_search`, `UnifiedSearchResult` |
| Retrieval | `rag_service/database_retrievers.py` | FTS and vector legs, `retrieve_hybrid`, weighted reciprocal-rank fusion |
| Reranking | `rag_service/advanced_reranking.py` | strategies: flashrank (default), cross_encoder, hybrid, llama_cpp, llm_scoring, two_tier, none |
| Profiles | `rag_service/profiles.py` | public `fast` / `balanced` / `accuracy`; internal profiles not request-accepted |
| Vector stores | `rag_service/vector_stores/` | Chroma + PGVector behind the factory |
| Quality | `rag_service/{citations,claims,faithfulness,guardrails,post_generation_verifier,clarification_gate}.py` | hard citations, claims extraction/verification, numeric fidelity, abstention/clarification |
| Deep search | `app/api/v1/endpoints/research.py`, `app/core/Web_Scraping/WebSearch_APIs.py` | `POST /research/websearch`; `generate_and_search` / `analyze_and_aggregate` |

## Unified search (dataflow)

1. Request → profile/default resolution (explicit > profile default > helper default > schema default) → per-user DB adapter/path injection → pipeline kwargs.
2. Phases (OTel-spanned): semantic cache check → optional query expansion / HyDE / pseudo-relevance feedback / decomposition → **retrieval** with per-source collections (`user_{id}_{source}_embeddings`).
3. Media retrieval: FTS and vector legs run concurrently; one side failing degrades to the survivor, both failing raises; fusion is weighted RRF with `hybrid_alpha = 0.7` weighting **vector** (0.3 FTS) and k=60.
4. FTS: `media_fts` bm25 with sanitized queries; chunk-level via `unvectorized_chunks_fts` (auto-ensured/rebuilt).
5. Vector: query embedding (precomputed vectors accepted for HyDE), namespace wildcards/lists, `allowed_media_ids` Chroma `$in` filters; falls back to FTS when the store is missing.
6. Cross-source fusion (ADR-053): multi-source results ordered by RRF over per-source ranks rescaled to (0,1] — never sort multi-source results by raw score (incompatible scales).
7. Optional document grading, then **reranking** (default flashrank; two_tier = fast pass + LLM scoring).
8. Post-retrieval: parent/sibling expansion, security filters, workspace-artifact filtering (generated/test artifacts excluded unless `workspace_id`).
9. Optional **generation** with guardrails: hard per-sentence citations, numeric fidelity, abstention (`continue|ask|decline`), clarification gate. Streaming requires generation.
10. `metadata.source_status` reports each source `searched|empty|unavailable` honestly.

Sources normalize to `media_db, notes, chats, characters, kanban, prompts, world_books, dictionaries, sql`. Strategy: `standard` (explicit kwargs) vs `agentic` (query-time synthetic chunks). Claims (`enable_claims`) uses extractor `aps|claimify|auto` and verifier `nli|llm|hybrid`; ingestion-time claims live in the adjacent `core/Claims_Extraction/` module.

## Deep-search request (dataflow)

1. `POST /api/v1/research/websearch` — `aggregate=true` requires a relevance or final-answer LLM.
2. Phase 1 (`generate_and_search`, dedicated thread pool, LLM state captured): optional sub-query generation → jittered fan-out → per-provider websearch (Google CSE, Brave, DuckDuckGo, Kagi, Searx, Tavily…) → normalized results. Fatal phase-1 errors map to HTTP codes.
3. Phase 2 (`analyze_and_aggregate`, cancel-event aware): per-result relevance evaluation with article scraping, review/selection, LLM aggregation, optional discussion search.
4. Providers are guarded by the outbound egress policy and a websearch circuit breaker; client disconnects set the cancel event without crashing the endpoint.

## Failure behaviors (verified)

| Case | Behavior |
| --- | --- |
| One hybrid leg fails | Survivor's documents retained; both empty → `RAGDatabaseError` |
| Vector store unavailable | Falls back to FTS |
| Streaming without generation | Rejected |
| Weak evidence | Citations/numeric-fidelity/clarification features abstain or ask |
| Batch interrupted | Checkpoint/resume via `/rag/batch/resume/{checkpoint_id}` |

## Config keys (selection)

Request-level dominates: `search_mode` (fts|vector|hybrid), `fts_level`, `hybrid_alpha` (0.7), `top_k`, `min_score`, `reranking_strategy` + rerank keys, `enable_hyde` / `enable_prf` (+ params), `enable_query_decomposition`, `rag_profile`, `index_namespace`/`corpus`, `require_hard_citations`, `enable_numeric_fidelity`, `enable_claims` (+ extractor/verifier/top-k/confidence), `enable_post_verification` (+ adaptive family). File config `[RAG]`: `flashrank_model_name`, `flashrank_cache_dir`, `vector_store_type` (chromadb default; pgvector/postgres accepted), `pgvector.*`.

## Verified gotchas

1. `hybrid_alpha = 0.7` weights **vector** 0.7 — not FTS (the classic misreading).
2. `characters` (cards) and `chats` (history) are separate sources.
3. Internal profiles (`production/research/cheap`) are rejected at the request schema.
4. Generated/test workspace artifacts are filtered from global retrieval unless a `workspace_id` is given.
5. `Docs/Code_Documentation/RAG-Developer-Guide.md` and `Docs/API-related/RAG_API_Documentation.md` are current; `Docs/RAG/` holds older research notes — prefer the former two.

## Governing decisions

ADR-053 (`Docs/ADR/053-rag-cross-source-fusion.md`), ADR-022 (embeddings pipeline), ADR-026 (egress policy). Current references: the RAG module README, `RAG-Developer-Guide.md`.

## Related docs

- [chunking-embeddings.md](./chunking-embeddings.md) — the vectors being searched
- [research.md](./research.md) — the jobs engine that drives deep research
