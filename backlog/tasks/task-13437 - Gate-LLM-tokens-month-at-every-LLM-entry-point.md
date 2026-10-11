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
