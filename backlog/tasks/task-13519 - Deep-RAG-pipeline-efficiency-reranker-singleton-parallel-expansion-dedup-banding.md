---
id: TASK-13519
title: Deep RAG pipeline efficiency (reranker singleton, parallel expansion, dedup
  banding)
status: To Do
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Batch 7. Plan: Docs/Plans/2026-10-06-perf-batch-7-rag-pipeline-implementation-plan.md. Reranker registry singleton (advanced_reranking.py:1796, unified_pipeline.py:6293); asyncio.gather for query-expansion variants (unified_pipeline.py:4491); query-embedding LRU (database_retrievers.py:2189); MMR precomputed word sets (advanced_reranking.py:1228); LLM reranker multi-passage batching (:1646); semantic cache matrix similarity (semantic_cache.py:325); late-chunking limits + cached token sets (database_retrievers.py:1430 and :373); SimHash-banded ingest dedup (ChromaDB_Library.py:1698); rerank candidates capped at top-k.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
TASK-13950.24 (2026-11-24): the unified_rag_pipeline decomposition moved the code this task's line references point at. Line numbers in the description are stale. The reranker singleton, rerank candidate cap and calibration now live in tldw_Server_API/app/core/RAG/rag_service/pipeline/rerank.py (run_rerank: select_reranker, apply_rerank, calibrate_rerank). Query expansion, HyDE and variant merging (the asyncio.gather change) live in pipeline/retrieval.py (run_retrieval, merge_hyde_and_variants, run_decomposition) and pipeline/query_prep.py (run_query_expansion). The orchestrator is run_rag_pipeline in unified_pipeline.py; unified_rag_pipeline is now a kwargs shim.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
