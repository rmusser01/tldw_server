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
