---
id: TASK-13955
title: BYOK calls do not count against the user's own LLM limits
status: In Progress
labels:
- quotas
- backend
- byok
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Owner decision 2026-10-26, verbatim: 'BYOK users own keys shuold not count against their own costs'. Raised by PR #3346's final review (TASK-13936): a call made with the user's own stored provider key (tldw-level BYOK) still counts against that user's per-user LLM limits, although the operator pays nothing for it when the vendor bills the user's key. Credential resolution knows the source (ResolvedByokCredentials.source: user, team, org, server_default, none) but drops it before the usage recorder and the quota checks. Scope: record the call with its real tokens and cost and its credential_source; a user key counts as the user's own only on a provider that bills per key, sent to its vendor's own host (a key stored for a custom OpenAI-compatible slot, a local server or an unknown provider, or for a billed provider whose base URL the operator points at its own model server or a gateway, is user_unmetered and counts); leave credential_source 'user' out of limits.llm_tokens_per_month, limits.llm_cost_per_day_usd, limits.llm_cost_per_month_usd and the hosted billing plan's llm_tokens_month, and make that plan sum count on PostgreSQL at all (it bound a timezone-aware month start against a TIMESTAMP column and returned 0); do not check such a call at the call; at /chat/completions and the Messages API (/messages, /v1/messages), let an over-limit user through when the provider resolves to their own key, and skip the hosted plan's pre-check for such a request. Out of scope, still refusing an over-limit own-key user before the key is known (follow-up TASK-13956): character chat, RAG generation, evaluations and benchmarks, flashcards and quizzes, chat document generation, watchlists, the media endpoints and a media ingest job's start. Org and team shared keys still count (owner decision 2026-10-29). The Resource Governor token caps, virtual-key budgets and evaluation token limits still count own-key calls (owner decision 2026-10-29). Out of scope: OpenRouter's is_byok, which #3346 prices. Plan: Docs/superpowers/plans/2026-10-26-byok-quota-exemption.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A call made with the user's own stored key for a provider that bills per key is recorded in llm_usage_log with its real tokens and cost and credential_source 'user', on SQLite and on PostgreSQL, with a test on each; a key stored for a custom OpenAI-compatible slot, a local server or an unknown provider, or for a billed provider whose configured base URL is not on its vendor's host, is recorded as user_unmetered, with tests
- [ ] #2 limits.llm_tokens_per_month, limits.llm_cost_per_day_usd, limits.llm_cost_per_month_usd and the hosted billing plan's llm_tokens_month do not count rows with credential_source 'user', while rows with user_unmetered, team, org, server_default or no source still count, with tests on SQLite and on PostgreSQL; on PostgreSQL the hosted plan's sum counts at all (it returned 0 before), with a test
- [ ] #3 For a user over an LLM limit, a call on their own key is not refused at the call (the four adapter methods, perform_chat_api_call and perform_chat_api_call_async, a caller that marks the call context), and a call on a server, team, org or operator-endpoint key still is, with tests; a refused failover call does not mark the fallback provider unhealthy, with a test
- [ ] #4 Only /chat/completions and the Messages API (/messages, /v1/messages) admit an over-limit user whose provider resolves to their own key, without marking the request admitted, and still refuse one without their own key for that provider; /chat/completions also skips the hosted billing plan's pre-check for such a request; with tests. Every other LLM quota boundary is unchanged and still refuses (TASK-13956)
- [ ] #5 A database without the new column still records the row, without its source, and the limits still count every row and refuse, with a test; a PostgreSQL usage-table statement that fails with one of the bootstrap's handled errors (a timeout, a lost connection) is logged as a warning, with a test; a database error such as a missing privilege still stops startup
- [ ] #6 Usage_Quotas.md, the BYOK user guide and a new ADR amending ADR-064 say what counts, and the Docs/Published mirror is refreshed
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Plan: Docs/superpowers/plans/2026-10-26-byok-quota-exemption.md (owner-approved 2026-10-29, as recommended). Owner answers 2026-10-29: Decision 3 (scope) this PR's two endpoints plus follow-up TASK-13956; Decision 4 (shared keys) keep counting org and team keys, so the F2 question task is not filed; Decision 5 (other spend limits) exempt only the hosted plan's llm_tokens_month, keep counting virtual-key budgets, Resource Governor caps and evaluation tokens. Follow-up filed: TASK-13956.
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
