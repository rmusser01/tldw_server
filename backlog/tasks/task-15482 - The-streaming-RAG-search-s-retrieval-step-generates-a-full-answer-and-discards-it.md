---
id: TASK-15482
title: The streaming RAG search's retrieval step generates a full answer and discards
  it
status: Done
labels:
- rag
- backend
- bug
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
User-facing: every Knowledge QA question (the WebUI streams) pays for two answers, and waits for the first before its stream starts, since 2026-04-24; recommended next after TASK-15427. Found by TASK-15427 (the outbound Groq call in test_rag_streaming_generation_provider_override). /api/v1/rag/search/stream retrieves through the standard pipeline before it streams the answer. _retrieve_standard_documents sets kwargs['enable_generation'] = False (RAG/rag_service/streaming_executor.py:381 on cb70f0c5d1) but passes the endpoint's resolved_request (:379; rag_unified.py:2340), whose payload says enable_generation True (the stream route requires it, rag_unified.py:2308-2309), and unified_rag_pipeline prefers the payload (_resolve_effective_enable_generation, unified_pipeline.py:2390-2393, re-read at :6900; since cd184c9fb9, 2026-04-24). So the retrieval step runs the full non-streaming answer (execute_generation_phase, unified_pipeline.py:7255, to the provider call at generation.py:531) and any post-answer stage the request turns on (claims, post-verification, suggestions), keeps only the documents (streaming_executor.py:391-431), and _stream_generation_events (:685-722) then generates the answer again. Every streaming RAG request pays for two answers and waits for the first before its stream starts; on the user's own key their provider bills both. Measured by a socket probe: the prefetch's answer reached api.groq.com from a provider-sync-adapter thread, and on dev four cases of RAG/test_restricted_postgres_media_retrieval.py's two stream tests reach api.openai.com the same way (stack through streaming_executor.py:391 and unified_pipeline.py:7255). Fix sketch: give the prefetch a request with enable_generation False. Caveat: a plain copy (dataclasses.replace) is wrong. The prefetch's pipeline writes the effective query (spell correction, classification, chat-history reformulation) back into the shared resolved request (_sync_effective_query, unified_pipeline.py:2507-2518: resolved_request.query and payload['query']) and the streamed answer reads resolved_request.query (_stream_generation_events, streaming_executor.py:696; wired at :841-842), so with a copy the stream would answer the raw follow-up (what about it?). The copy must carry the effective query back, or the override must not replace the object. Knowledge QA takes this route (chat-rag.ts:592, knowledgeQaClient.ts:51-52). With suggestions on, the prefetch also makes a suggestions call whose result it discards. Pre-retrieval clarification defaults to the effective enable_generation (unified_pipeline.py:3163-3167), so set enable_pre_retrieval_clarification explicitly when the request leaves it None, to keep the stream's clarification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A streaming RAG search makes one answer call, the streamed one: its retrieval step runs no generation and no post-answer stage, with a test that counts the provider calls
- [x] #2 The streamed answer still uses the effective query (chat-history reformulation, spell correction), with a test
- [x] #3 The stream still runs pre-retrieval clarification when neither the request nor its profile sets enable_pre_retrieval_clarification, as today, with a test
- [x] #4 test_rag_streaming_generation_provider_override passes its socket probe without its provider stub (the stub is removed), and test_restricted_postgres_media_retrieval.py's stream tests reach no provider
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Plan: Docs/superpowers/plans/2026-11-07-rag-stream-one-answer.md. Fix: unified_rag_pipeline honours an explicit enable_generation=False over a resolved payload that says True (_resolve_effective_enable_generation, unified_pipeline.py:2390-2391), so the stream's retrieval step (streaming_executor.py:381, which already passes False) runs retrieval only on the shared request, which keeps the effective query; the step sets enable_pre_retrieval_clarification from the request's enable_generation when it is unset. Owner answer, Change summary: waived by the owner on 2026-11-07 ('Yes, both': the TASK-15482 PR and the benchmark judge PR); recorded as a full Change summary section. Owner approved the plan on 2026-11-07, accepting the failure-path event order change (Review Focus 6).
RED/GREEN (record the real output): base 4 failed, 4 passed (provider-override [('dns', 'api.groq.com')] == []; one-call [False, False, True] == [True]; clarification[turned-off] [False, True] == [True]; pipeline[explicit-off-beats-the-payload] 'generation phase should not run'); pipeline line only 1 failed, 7 passed (clarification[unset-follows-generation] streamed an answer); with the retrieval step's explicit clarification 8 passed. Mutations: the task's plain-copy sketch passes AC #1 and fails both effective-query cases (the prompt asks the raw query); forcing clarification on fails [turned-off] (assert [] == [True]). Focused set (not postgresql) 142 passed -> 148 passed, probe log 2 lines (the two sqlite stream cases of test_restricted_postgres_media_retrieval.py, api.openai.com) -> 0. Wider set 5 failed, 1780 passed, 2 skipped -> 5 failed, 1786 passed, 2 skipped (the same 5 environment/order failures: test_rag_profiles flashrank and four test_restricted_postgres_media_retrieval sqlite cases), probe 0. tests/Docs + lint + CI 1586 passed, 6 skipped, 2 failed + 1 error (environment only: the shared venv lacks coverage and setuptools: test_jobs_postgres_exports_recorded_coverage x2, test_runtime_package_discovery.py); backlog-py 179 passed; shard coverage, test-quality ratchet, route-auth ratchet, OpenAPI and the scope-predicate baseline unchanged. PostgreSQL cases: the postgresql-only focused set (37 cases, incl. the two postgresql stream cases) 37 passed with a probe log of 0 lines (base: 37 passed, probe 2 lines, api.openai.com).
Decisions: unified_rag_pipeline honours an explicit enable_generation=False over a payload that says True (no other caller passes that pair; the 20 call sites in the plan), so no copy of the shared request is made and _sync_effective_query keeps writing the effective query where the stream reads it. Every post-answer stage is gated on generated_answer and is skipped; the stream's own claims overlay is heuristic (no LLM call). With post-verification on, the stream's contexts are its own retrieval, no longer an adopted rerun of the discarded answer. One answer call means one usage row and one call-site quota check fewer per stream request; the boundary check is unchanged; on /search/stream it still refuses an over-limit own-key user for post-answer stages that no longer run there (fail-closed, not filed).
Docs: none (no doc describes the retrieval step's answer; the event contract is unchanged). Bandit: No issues identified (Medium 0, High 0).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
/api/v1/rag/search/stream makes one answer call, the streamed one. Its retrieval step already asked for retrieval only, but unified_rag_pipeline preferred the resolved payload's enable_generation True (since cd184c9fb9), so it ran and discarded a full answer and every post-answer stage the request turned on. The pipeline now honours an explicit enable_generation=False; the retrieval step keeps the shared request, so the stream still answers the effective query (spell correction, chat-history reformulation), and it sets pre-retrieval clarification from the request's enable_generation when unset, so the stream still clarifies by default. test_rag_streaming_generation_provider_override passes its socket probe without a provider stub, and test_restricted_postgres_media_retrieval.py's stream tests reach no provider.
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
