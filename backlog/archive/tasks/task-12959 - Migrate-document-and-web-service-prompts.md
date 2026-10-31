---
id: TASK-12959
title: Migrate document and web service prompts
status: To Do
assignee: []
created_date: '2026-07-14 00:16'
updated_date: '2026-07-14 00:18'
labels:
  - prompts
  - service-prompts
  - domain-migration
  - backend
dependencies:
  - TASK-12957
references:
  - TASK-12956
  - TASK-12958
documentation:
  - >-
    Docs/superpowers/specs/2026-07-12-user-customizable-service-prompts-design.md
  - Docs/Design/service-prompt-inventory.md
  - Docs/superpowers/plans/2026-07-13-service-prompts-domain-documents-web.md
---

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 All 32 approved documents/web IDs are registered and every named consumer resolves one immutable atomic bundle.
- [ ] #2 Tests prove byte-equivalent defaults, precedence, browser transport, authenticated ownership, protected Jobs pinning, and exact size limits.
- [ ] #3 No deferred core-Scheduler or excluded definition is migrated, and every implementation gate passes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Execute Docs/superpowers/plans/2026-07-13-service-prompts-domain-documents-web.md after TASK-12957.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Second rollout domain; implementation remains To Do. It consumes the registry-owned browser message policies established by TASK-12957.
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
