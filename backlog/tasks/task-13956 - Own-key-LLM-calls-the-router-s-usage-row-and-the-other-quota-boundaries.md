---
id: TASK-13956
title: 'Own-key LLM calls: the router''s usage row and the other quota boundaries'
status: In Progress
labels:
- quotas
- backend
- byok
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up of TASK-13955, which made a call on the user's own provider key (on a provider that bills per key) exempt from the per-user LLM limits. Two gaps remain. (1) Boundaries: only /chat/completions and the Messages API pass their provider to enforce_llm_quota. These still refuse an over-limit user before the key is known, even when every call would run on the user's own key: character chat (character_chat_sessions.py, the completion check and the roleplay-resume check; the completion's provider is resolved only after the auto-router, so the check moves below it), RAG generation (rag_unified.py, six sites), evaluations and benchmarks (evaluations_unified.py, evaluations_crud.py, evaluations_benchmarks.py, benchmark_api.py), flashcards and quizzes (route dependencies, to become inline calls), chat document generation (chat_documents.py), watchlists (watchlists.py), the media endpoints with an api_name (media/add.py, ingest_web_content.py and the seven process_* modules), and a media ingest job's start (media_ingest_jobs_worker._admit_or_refuse_analysis). Pass the provider where the endpoint knows it before any LLM work. Character chat first: it is a primary UI surface. Evaluation token reservations keep counting own-key calls (owner decision 2026-10-29). (2) Router calls: routing/accounting.log_model_router_usage records the auto-router's call without a credential_source, so it counts (at most 64 output tokens a call). Its callers are chat.py's _log_router_usage (the router handle is router_credential_handle), chat_route_resolver.py's log_router_usage (its execute_router_call resolves with resolve_byok_credentials and passes a raw key, so its llm_call_context(records_itself=True) should also carry credential_source=billed_credential_source(provider, byok_resolution.source)) and character_chat_sessions.py's _log_router_usage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each LLM quota boundary that knows its provider before any LLM work passes it to enforce_llm_quota, with a test per endpoint family, character chat first
- [ ] #2 A media ingest job's start check admits analysis for an over-limit user whose analysis provider resolves to their own key, without marking the job admitted, with a test
- [ ] #3 The auto-router's usage row on /chat/completions, character chat and chat_route_resolver records the router call's credential_source, with a test
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Plan, PR A of 3: Docs/superpowers/plans/2026-11-05-own-key-character-chat-router-rows.md (character chat, the router's row, the no-route 402). PR B: evaluations, chat documents, RAG. PR C: the boundaries whose calls never use a stored key, ADR-083 amendment, close-out.
PR A (#3423): character chat names the reply's provider after the auto-router (_enforce_character_reply_quota; a saved reply in a chat with memory extraction on still counts: extraction runs on the server's key) and the resume its saved provider; an admitted reply refused at the call answers the 402, not 502 Chat provider error (_execute_character_provider_call lets LLMQuotaExceededError through); the router's row records the router call's credential_source on /chat/completions, character chat and chat_route_resolver (whose raw-key router call also marks its context with billed_credential_source(provider, source, app_config, server snapshot)); a router call refused at the call that leaves no route answers the 402 on /chat/completions and character chat, not 503 auto_routing_failed (enforce_llm_quota_if_the_router_was_refused; any other lost route keeps its 503 or 400). RED/GREEN: focused 13 failed, 15 passed on base code (per task here: Task 1 9 failed, 4 passed; Task 2 4 failed, 11 passed with Task 1 in) -> 28 passed. Wider SQLite 5071 passed, 53 skipped -> 5089 passed, 53 skipped; PostgreSQL 4 passed, 2 skipped -> 4 passed, 2 skipped. Mutations: no-route guard dropped -> auto_routing_failed; extraction ignored -> 200 (no request leaves the machine); guard unconditional -> 402 for a router outage. Bandit: no issues identified (Medium 0, High 0).
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
