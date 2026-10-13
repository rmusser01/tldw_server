---
id: TASK-13437
title: Gate LLM tokens/month at every LLM entry point
status: In Progress
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deferred from spec 2. limits.llm_tokens_per_month is checked only on /chat/completions; character chat, RAG generation, workflows and persona LLM calls are counted in llm_usage_log but not gated. Parent: TASK-13434.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every user-initiated LLM call checks limits.llm_tokens_per_month before dispatch
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Verified 2026-10-09: most LLM paths were not recorded at all (only chat completions, the router, claims and embeddings wrote llm_usage_log). Spec: Docs/Design/2026-10-09-llm-usage-accounting-design.md. PR 1a records every chat provider call for a known user in ChatProvider.__init_subclass__, keeps the existing recorders (marked records_itself), carries the call state into pool threads and the chat queue, and excludes embeddings from the token counter. No new refusals. Next: PR 1b (WorkerSDK, workflow engine, WebSockets, executor sites), PR 2 (the check and the cost keys).
PR 1b: background roots get a user through llm_call_root (WorkerSDK run and run_prepared, the workflow engine's start_run and continue_run, the agent-task and audio jobs workers, watchlist output enrichment), audio and realtime WebSockets bind theirs in _audio_ws_authenticate, and the loop's default executor carries the LLM call state so run_in_executor(None, ...) keeps the user. Still unrecorded: MCP tool calls, the in-process claims rebuild threads, system jobs with no user. Next: PR 2 (the check and the cost keys).
PR 2a: one LLM quota decision (llm_quota_refusal) and one 402 error (LLMQuotaExceededError, an HTTPException); limits.llm_cost_per_day_usd and limits.llm_cost_per_month_usd; boundary checks at /chat/completions, character chat, RAG with generation, document generation, the study assistants, the in-request evaluations and media analysis; local providers priced at zero in the catalog; startup warnings for LLM_USAGE_ENABLED off and retention under 31 days; the profile's quotas.llm block; the Messages API is checked at the boundary. RAG batch now returns spec 2's 402 instead of a 500. Next: PR 2b, the check at the call.
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
