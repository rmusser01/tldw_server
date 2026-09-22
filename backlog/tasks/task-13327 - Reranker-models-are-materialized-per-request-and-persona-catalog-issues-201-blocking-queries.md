---
id: TASK-13327
title: >-
  Reranker models are materialized per request and persona catalog issues 201
  blocking queries
status: To Do
assignee: []
created_date: '2026-09-22 04:57'
labels:
  - efficiency
  - rag
  - api
dependencies: []
references:
  - 'tldw_Server_API/app/core/RAG/rag_service/advanced_reranking.py:1782'
  - 'tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py:6265'
  - 'tldw_Server_API/app/api/v1/endpoints/persona.py:7566'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Two independent per-request costs with the same shape.

(1) RAG: create_reranker constructs a transformer model in __init__ with NO memoization anywhere (grep for lru_cache/_CACHE returns one unrelated env line). On the DEFAULT path enable_reranking=True with flashrank; the balanced profile uses hybrid which constructs a FlashRankReranker, and the research profile uses two_tier which constructs a cross-encoder. The two-tier degradation path constructs a SECOND reranker in the same request. Cost driver: one model materialization per request; scales with request rate; the constant is weight deserialization (hundreds of MB for bge-class) and resident memory multiplies by in-flight requests. Destination: rag_service/reranker_registry.py keyed on (strategy, model_name, device, revision, local_files_only, trust_remote_code).

(2) persona_catalog: db.list_persona_profiles(limit=200) then db.list_persona_policy_rules per profile inside the loop, neither awaited nor offloaded, while the file own _run_persona_db_call to_thread wrapper is used 106 times elsewhere and the BATCHED loader _load_persona_buddy_rows_for_projection is called one line above. Up to 201 sequential blocking SQLite calls on the loop thread, stalling every concurrent request including persona_stream websockets.

Source: synthesis F26
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Rerankers are cached for process lifetime keyed on their construction inputs
- [ ] #2 persona_catalog offloads DB work and batch-loads policy rules
- [ ] #3 A test asserts query count for the catalog endpoint
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
