---
id: TASK-12962
title: Migrate eligible extraction and chunking service prompts
status: To Do
assignee: []
created_date: '2026-07-14 00:20'
labels:
  - prompts
  - service-prompts
  - domain-migration
  - backend
dependencies:
  - TASK-12961
references:
  - TASK-12956
  - TASK-12958
documentation:
  - >-
    Docs/superpowers/plans/2026-07-13-service-prompts-domain-extraction-chunking.md
  - Docs/Design/service-prompt-inventory.md
---

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Both approved free-text extraction/chunking IDs are migrated without expanding into OCR, structured extraction, recursive workflows, or deferred Scheduler definitions.
- [ ] #2 Focused tests and every implementation gate pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Execute Docs/superpowers/plans/2026-07-13-service-prompts-domain-extraction-chunking.md after TASK-12961.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fifth rollout domain; implementation remains To Do.
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
