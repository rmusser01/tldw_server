---
id: TASK-13437
title: Gate LLM tokens/month at every LLM entry point
status: To Do
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
