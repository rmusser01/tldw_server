---
id: TASK-15427
title: 'PR B (TASK-13956) deferred minors: RAG LLM-stage drift test, reformulation
  provider canonicalization, a stream test''s outbound Groq call'
status: Done
labels:
- quotas
- backend
- tests
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up of TASK-13956 PR B (#3426). (1) Test hygiene: RAG_NEW/integration/test_rag_stream_parity.py::test_rag_streaming_generation_provider_override makes a real outbound call to api.groq.com:443 (groq_adapter.chat via usage_accounting.chat on a bounded_daemon thread; the test stubs only generate_streaming_response); it predates PR B. (2) rag_unified._RAG_OTHER_LLM_STAGES lists the RAG request switches that make an LLM call besides the answer, so an over-limit user on their own key with one of them on keeps the 402; a new switch is not forced to declare itself, so add a drift test against the RAG request schema's enable_* fields with a declared no-LLM list (a missed stage is still refused at its own call, so this guards the up-front answer, not spend). (3) _rag_llm_providers does not canonicalize the reformulation's provider before the dedup (one extra users_own_key_serves lookup, decision unaffected); and test_rag_search_still_refuses_an_own_key_answer_with_another_llm_stage[history-on-a-server-key] does not assert which providers were asked.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 test_rag_streaming_generation_provider_override opens no connection to a non-loopback address (stub the provider call), shown by a socket probe
- [x] #2 A test fails when a RAG request schema switch is neither in _RAG_OTHER_LLM_STAGES nor in a declared no-LLM list
- [x] #3 _rag_llm_providers canonicalizes the reformulation provider before the dedup, and [history-on-a-server-key] asserts the providers asked, with tests
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
AC #1's socket probe: the repo has one, tldw_Server_API/tests/helpers/offline_network.block_live_network() (it blocks and records getaddrinfo, gethostbyname, connect and connect_ex); TASK-13956's runs used a local equivalent outside the repo.
Plan: Docs/superpowers/plans/2026-11-07-rag-own-key-minors.md. Follow-ups: the RG under-commit is TASK-15421 (re-verified, Redis governor included; not re-filed); TASK-15481 (hosted plan and failover); TASK-15482 (the stream's discarded prefetch answer). Owner answers: Decision 1 (Change summary) waived, 2026-11-07 ("Yes, waive for all"); Decision 2 (TASK-15482) file it and do it next after TASK-15427, 2026-11-07.
RED/GREEN (as measured): Task 1 1 failed (assert [('dns', 'api.groq.com')] == []) -> 1 passed; Task 2 2 failed -> 2 passed (mutations: a new bool field on UnifiedRAGRequest, enable_post_verification dropped from the list: 1 failed each); Task 3 1 failed, 11 passed -> 12 passed (assert ['openai', 'OAI'] == ['openai']; mutation: the reformulation appends None -> history-on-a-server-key 'assert [] == ['openai', 'groq']'). Focused set 40 -> 43 passed (1 skipped), 0 outbound attempts under a socket probe. Wider set (57 files, shared PostgreSQL at 127.0.0.1:5432 down locally, so PostgreSQL cases skip): base 3 failed, 996 passed, 99 skipped -> 3 failed, 999 passed, 99 skipped (the same 3 live-server e2e rbac failures; +3 passed are this plan's new tests); probe lines 3 -> 2 (the groq line is gone). tests/Docs + lint + CI 1497 passed, 6 skipped; backlog-py 164 passed; shard coverage, test-quality ratchet, route-auth ratchet and OpenAPI unchanged.
Decisions: all 98 boolean switches of UnifiedRAGRequest and UnifiedBatchRequest classified with file:line (plan, 'AC #2: every switch, classified'); 12 make another LLM call and stay in _RAG_OTHER_LLM_STAGES; enable_abstention, enable_knowledge_strips, enable_web_fallback, enable_query_rewriting_loop, adaptive_rerun_on_low_confidence and adaptive_advanced_rewrites make none of their own (or only inside a listed stage) and leave it, so an over-limit user on their own key with them on is admitted (every call is still checked at its own call where the call-site gate runs); the other 86 are declared in test_rag_agentic_api._NO_OTHER_LLM_CALL. The drift test covers boolean switches only; string modes (strategy, search_depth_mode, reranking_strategy) stay checked by value in _rag_llm_providers. The stream test stubs perform_chat_api_call_async and asserts no network, not the stub's calls (they are TASK-15482's discarded prefetch answer). Follow-ups: TASK-15421 (RG under-commit, re-verified incl. the Redis governor), TASK-15481, TASK-15482.
Docs: none (Usage_Quotas.md names the list by name; ADR-083 unchanged). Bandit: No issues identified (Medium 0, High 0).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
test_rag_streaming_generation_provider_override stubs the standard pipeline's answer call and asserts, through block_live_network(), that it opens no connection; the call it made to api.groq.com came from the stream's retrieval step, which runs and discards a full answer (TASK-15482). Every boolean switch of the RAG request schemas is classified: _RAG_OTHER_LLM_STAGES keeps the 12 that make an LLM call besides the answer and the reformulation and drops six that make none of their own; a drift test fails on a switch in neither list, in both, or in neither schema. _rag_llm_providers canonicalizes the reformulation's provider as the answer's is canonicalized, so an alias is looked up once, and the refusal test asserts the providers asked. Follow-ups filed: TASK-15481, TASK-15482; TASK-15421 annotated.
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
