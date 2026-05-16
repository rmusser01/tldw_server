---
id: TASK-394.4
title: Correct Quick Ingest offline cancel and progress states
status: Done
assignee: []
created_date: '2026-05-16 00:43'
updated_date: '2026-05-16 03:18'
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
- [x] #1 Offline and network failure states surface before or during submit with actionable recovery
- [x] #2 Cancel/close behavior distinguishes draft dismissal from in-flight processing
- [x] #3 Progress/background status copy does not imply unsupported background jobs or hidden completion tracking
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Started Task 4 in quick-ingest UX remediation branch. Scope: offline/disconnected processing guard, cancel/close distinction, neutral progress copy, and minimized widget terminal-state accuracy.

Implemented offline processing guards in Add and Review steps using the shared connection store, added retry recovery affordances, neutralized global processing copy, and split minimized widget terminal states into Done, Failed, Cancelled, and Interrupted. Functional commit: 9958abdc8.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Task 4 complete. Offline/disconnected users can still add and configure items, but processing actions are blocked with explicit recovery copy and Retry connection. Review prevents final processing while disconnected. Progress copy now uses neutral processing/indexing language. Minimized progress widget distinguishes completed, failed, cancelled, and interrupted sessions with non-success terminal states. Verification: focused Vitest suite passed (48 tests), focused Playwright dismiss/resume flow passed (1 test, 46.2s), and git diff --check passed. Bandit skipped because this task touched frontend TypeScript/TSX only and no Python code.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
