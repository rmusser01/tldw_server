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
- [ ] #1 Each LLM quota boundary whose LLM calls run on the user's stored key names every call's provider to enforce_llm_quota before any LLM work, so an over-limit user whose calls all run on their own key is let through unmarked; a request with a call the boundary cannot place on the user's key still counts; a boundary whose calls run on the server's key keeps refusing; a test per endpoint family, character chat first
- [ ] #2 A media ingest job's start check keeps refusing an over-limit user who stored a key for the analysis provider, because job analysis runs on the server's key, and keeps the job admission marker; making job analysis use the stored key is TASK-15426; with a test
- [ ] #3 The auto-router's usage row on /chat/completions, character chat and chat_route_resolver records the router call's credential_source, with a test
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Plan, PR A of 3: Docs/superpowers/plans/2026-11-05-own-key-character-chat-router-rows.md (character chat, the router's row, the no-route 402). PR B: evaluations, chat documents, RAG. PR C: the boundaries whose calls never use a stored key, ADR-083 amendment, close-out.
PR A (#3423): character chat names the reply's provider after the auto-router (_enforce_character_reply_quota; a saved reply in a chat with memory extraction on still counts: extraction runs on the server's key) and the resume its saved provider; an admitted reply refused at the call answers the 402, not 502 Chat provider error (_execute_character_provider_call lets LLMQuotaExceededError through); the router's row records the router call's credential_source on /chat/completions, character chat and chat_route_resolver (whose raw-key router call also marks its context with billed_credential_source(provider, source, app_config, server snapshot)); a router call refused at the call that leaves no route answers the 402 on /chat/completions and character chat, not 503 auto_routing_failed (enforce_llm_quota_if_the_router_was_refused; any other lost route keeps its 503 or 400). RED/GREEN: focused 13 failed, 15 passed on base code (per task here: Task 1 9 failed, 4 passed; Task 2 4 failed, 11 passed with Task 1 in) -> 28 passed. Wider SQLite 5071 passed, 53 skipped -> 5089 passed, 53 skipped; PostgreSQL 4 passed, 2 skipped -> 4 passed, 2 skipped. Mutations: no-route guard dropped -> auto_routing_failed; extraction ignored -> 200 (no request leaves the machine); guard unconditional -> 402 for a router outage. Bandit: no issues identified (Medium 0, High 0).
Plan, PR B of 3: Docs/superpowers/plans/2026-11-05-own-key-evaluations-documents-rag.md.
PR B (#3426): enforce_llm_quota takes several providers (admitted past a limit only when every one runs on the user's own key; a missing name refuses). Named: the judge-model evaluations (each batch item's), chat document generation, RAG search/stream/batch/resume/advanced (the answer's provider and, with a chat history, the reformulation's, read from the resolved request so a rag_profile or a server Search-Agent default counts; any other LLM stage in _RAG_OTHER_LLM_STAGES, the quality depth, a non-standard strategy or an LLM reranker counts; ablation unchanged). own_key= answers for a single provider only. Evaluation token reservations unchanged (owner, 2026-10-29). RED/GREEN: focused 15 failed, 15 passed -> 30 passed (netblock 0). Wider SQLite 4 failed, 1481 passed, 161 skipped, 3 errors (load) -> 5 failed, 1506 passed, 161 skipped, 0 errors (the failures are the 3 e2e rbac ids, the abtest id and test_rag_search_natural_language_media_fallback, all load or environment dependent, the last two passing alone, 8 passed); PostgreSQL 70 passed -> 70 passed. Mutations: claims dropped from the list -> [claims] not 402; raw request at stream and batch -> [accuracy] and [suggestions-by-default] admitted; own_key taken for a list -> admitted. Bandit: No issues identified (Medium 0, High 0).
PR B final review (fable, clean session): 0 Critical, 1 Important, 4 Minor; no path where an admitted-unmarked user's call on another key is neither refused nor counted. Important fixed (3a402df4e6): _RAG_OTHER_LLM_STAGES listed six switches that make no LLM call (structured response, image and video search, query decomposition, intent routing, dynamic granularity), so the balanced profile and the SEARCH_STRUCTURED_RESPONSE / SEARCH_IMAGE_SEARCH / SEARCH_VIDEO_SEARCH defaults refused an own-key user; dropped, with evidence accumulation and evidence chains kept (they call an LLM). RED 3 failed, 13 passed -> 16 passed (netblock 0); scoped re-review: addressed.
ACs #1 and #2 amended (owner answer 2026-11-05: Amend + follow-up). Old #1: Each LLM quota boundary that knows its provider before any LLM work passes it to enforce_llm_quota, with a test per endpoint family, character chat first. Old #2: A media ingest job's start check admits analysis for an over-limit user whose analysis provider resolves to their own key, without marking the job admitted, with a test. Why: media analysis (and the ingest job), the study assistants, watchlist summaries, evaluation runs and benchmarks never resolve a stored user key, so their calls are the operator's and count; naming their provider would move the 402 into the middle of the work. Follow-up TASK-15426. AC #3 unchanged. PR B's deferred minors and a pre-existing outbound test: TASK-15427.
Plan, PR C of 3: Docs/superpowers/plans/2026-11-05-own-key-server-key-boundaries-close-out.md.
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
