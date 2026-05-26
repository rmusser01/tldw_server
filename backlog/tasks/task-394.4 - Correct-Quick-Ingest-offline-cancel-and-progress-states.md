---
id: TASK-394.4
title: Correct Quick Ingest offline cancel and progress states
status: To Do
assignee: []
created_date: '2026-05-16 00:43'
labels:
  - quick-ingest
  - ux
  - task-4
dependencies: []
documentation:
  - >-
    Docs/superpowers/plans/2026-05-16-quick-ingest-ux-remediation-implementation-plan.md
parent_task_id: TASK-394
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute implementation plan Task 4: align offline checks, cancel/close behavior, in-flight processing, progress copy, and background status with real system state.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Offline and network failure states surface before or during submit with actionable recovery
- [ ] #2 Cancel/close behavior distinguishes draft dismissal from in-flight processing
- [ ] #3 Progress/background status copy does not imply unsupported background jobs or hidden completion tracking
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
