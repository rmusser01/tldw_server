---
id: TASK-12960
title: Migrate user-visible RAG generation service prompts
status: To Do
assignee: []
created_date: '2026-07-14 00:18'
labels:
  - prompts
  - service-prompts
  - domain-migration
  - backend
dependencies:
  - TASK-12959
references:
  - TASK-12956
  - TASK-12958
documentation:
  - Docs/superpowers/plans/2026-07-13-service-prompts-domain-rag-generation.md
  - Docs/Design/service-prompt-inventory.md
---

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Both approved RAG generation IDs are migrated with exact contracts and no retrieval, reranking, judging, or deferred Scheduler scope expansion.
- [ ] #2 Focused tests and every implementation gate pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Execute Docs/superpowers/plans/2026-07-13-service-prompts-domain-rag-generation.md after TASK-12959.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Third rollout domain; implementation remains To Do.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
