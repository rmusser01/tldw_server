---
id: TASK-13327
title: >-
  Reranker models are materialized per request and persona catalog issues 201
  blocking queries
status: Done
assignee: []
created_date: '2026-09-22 04:57'
updated_date: '2026-09-24 00:53'
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
- [x] #1 Rerankers are cached for process lifetime keyed on their construction inputs
- [x] #2 persona_catalog offloads DB work and batch-loads policy rules
- [x] #3 A test asserts query count for the catalog endpoint
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Commit f17b537c96. AC1 in advanced_reranking.py: four loaders, each functools.lru_cache(maxsize=4) and keyed on their construction inputs. _load_flashrank_ranker takes (model_name, cache_dir). _load_preinstalled_flashrank_ranker takes (model_name, model_dir, required_files tuple). _load_cross_encoder_model takes (model_id, device, revision, local_files_only, trust_remote_code); it returns the sentence-transformers CE or the raw transformers tokenizer/model/torch. _load_qwen3_reranker_model takes (model_id, revision, device). The reranker objects stay per-request and cheap, so per-request knobs (top_k, batch_size, thresholds) are unaffected. Hybrid and two-tier reuse the same cached models because they construct the same inner rerankers. clear_reranker_model_cache() is used by an autouse fixture in tests/conftest.py. It lives in advanced_reranking.py rather than the suggested reranker_registry.py because the loaders are private to that module; a registry file would only re-export them. Known ceiling (ponytail comment): the fixed LRU bound has no memory accounting, and two first requests racing on the same key may both load; lru_cache is not single-flight. AC2: persona_catalog does all DB work in one _run_persona_db_call (to_thread) via _build_persona_catalog. Policy rules come from the new PersonaStateStore.list_persona_policy_rules_for_personas: one IN query, scoped by user_id, deleted=0, delegated on CharactersRAGDB and added to the delegation contract list in test_chacha_persona_state_store.py. AC3: tests/Persona/test_persona_catalog.py::test_persona_catalog_query_count_is_constant_and_off_loop counts execute_query calls and checks each runs off the event loop. On old code it FAILS (2 profiles = 6 queries, 12 profiles = 26, all on-loop). On new code it passes (3 queries for both, none on-loop), and it checks that soft-deleted rules stay hidden. tests/RAG_NEW/unit/test_reranker_model_cache.py has 3 tests: flashrank loads once across 3 creates plus a hybrid and again for a new model; the cross-encoder loads once across top_k changes and again for a new revision; a failed load is retried and not cached. All 3 FAIL on old code and pass on new. Suite before/after on RAG_NEW + RAG + Persona + persona ChaChaNotesDB files + the Evaluations reranker toggle + Workflows content adapters, run with -n 4 and TLDW_TEST_NO_DOCKER=1. Before: 8 failed / 3214 passed / 6 skipped. After: 8 failed / 3218 passed / 6 skipped. The FAILED sets are identical and all pre-existing: persona *_routes_include_rate_limit_dependency x4, generation_prompt_loader, pgvector sanitizers, rag_profiles flashrank local bundle (verified failing on 4a84d02b55 too), knowledge_qa_live_regressions. Known skip: tests/RAG/test_restricted_postgres_media_retrieval.py was --ignore'd because it hangs in Postgres C code on this machine regardless of my change; another session has the same file hung for 11h, and a pytest-timeout signal cannot interrupt it. Bandit -ll on the three touched source files: no findings (only pre-existing nosec notices). Docs: none needed, since no API contract changed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Reranker models are now materialized once per process per construction key (bounded LRU of 4) instead of on every RAG request, and failed loads are retried. The persona catalog now does its DB work off the event loop in a fixed 3 queries, down from up to 401 blocking calls, using a new batched policy-rule loader. Tests pin the load count and the query count; both fail on the old code.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
