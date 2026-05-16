---
id: TASK-394.6
title: Refresh current Quick Ingest wizard verification coverage
status: To Do
assignee: []
created_date: '2026-05-16 00:44'
labels:
  - quick-ingest
  - tests
  - task-6
dependencies: []
documentation:
  - >-
    Docs/superpowers/plans/2026-05-16-quick-ingest-ux-remediation-implementation-plan.md
parent_task_id: TASK-394
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute implementation plan Task 6: update or replace stale selectors/tests and add focused coverage for the active wizard states touched by this remediation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Current active wizard tests cover default, URL/text/file, validation, success/failure, and close/cancel paths where feasible
- [ ] #2 Legacy selector tests are removed, renamed, or clearly quarantined if stale
- [ ] #3 Verification commands for the touched UI slice pass or have documented environmental blockers
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
