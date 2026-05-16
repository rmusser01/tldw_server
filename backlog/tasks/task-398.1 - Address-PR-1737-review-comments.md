---
id: TASK-398.1
title: Address PR 1737 review comments
status: In Progress
assignee: []
created_date: '2026-05-16 02:36'
updated_date: '2026-05-16 02:38'
labels:
  - chat
  - cost-control
  - llm-cache
  - planning
  - pr-review
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1737'
documentation:
  - Docs/superpowers/plans/2026-05-16-chat-cache-cost-v2-implementation-plan.md
parent_task_id: TASK-398
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address unresolved Gemini review threads on PR #1737 for the cache-cost v2 implementation plan. Keep this docs-only and focused on retention enforcement and latency-field schema consistency.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Debug prompt-envelope retention cleanup is explicit in the implementation plan.
- [x] #2 Local prefill latency milliseconds use integer schema guidance and tests call out integer behavior.
- [ ] #3 Review threads are resolved after verification.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented docs review fixes in Docs/superpowers/plans/2026-05-16-chat-cache-cost-v2-implementation-plan.md: added explicit debug prompt-envelope retention pruning service/test coverage and switched local_prefill_latency_ms schema guidance to INTEGER with integer-storage test expectations.

Verification: git diff --check passed; targeted rg confirmed retention pruning and local_prefill_latency_ms INTEGER content; ASCII scan returned no matches. Bandit is not applicable because this change only edits docs and Backlog metadata.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
