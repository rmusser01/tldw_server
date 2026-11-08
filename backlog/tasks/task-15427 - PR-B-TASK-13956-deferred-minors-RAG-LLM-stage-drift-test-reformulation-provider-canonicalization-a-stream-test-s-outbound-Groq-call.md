---
id: TASK-15427
title: 'PR B (TASK-13956) deferred minors: RAG LLM-stage drift test, reformulation
  provider canonicalization, a stream test''s outbound Groq call'
status: In Progress
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
- [ ] #1 test_rag_streaming_generation_provider_override opens no connection to a non-loopback address (stub the provider call), shown by a socket probe
- [ ] #2 A test fails when a RAG request schema switch is neither in _RAG_OTHER_LLM_STAGES nor in a declared no-LLM list
- [ ] #3 _rag_llm_providers canonicalizes the reformulation provider before the dedup, and [history-on-a-server-key] asserts the providers asked, with tests
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
AC #1's socket probe: the repo has one, tldw_Server_API/tests/helpers/offline_network.block_live_network() (it blocks and records getaddrinfo, gethostbyname, connect and connect_ex); TASK-13956's runs used a local equivalent outside the repo.
Plan: Docs/superpowers/plans/2026-11-07-rag-own-key-minors.md. Follow-ups: the RG under-commit is TASK-15421 (re-verified, Redis governor included; not re-filed); TASK-15481 (hosted plan and failover); TASK-15482 (the stream's discarded prefetch answer). Owner answers: Decision 1 (Change summary) waived, 2026-11-07 ("Yes, waive for all"); Decision 2 (TASK-15482) file it and do it next after TASK-15427, 2026-11-07.
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
