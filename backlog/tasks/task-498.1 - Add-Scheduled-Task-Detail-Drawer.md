---
id: TASK-498.1
title: Add Scheduled Task Detail Drawer
status: Done
parent_task_id: TASK-498
references:
- TASK-498
- Docs/superpowers/plans/2026-06-01-scheduled-tasks-automation-workbench-phase1-implementation-plan.md
modified_files:
- apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTaskDetailDrawer.tsx
- apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTaskDetailDrawer.test.tsx
- apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTasksPage.tsx
- apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTaskTable.tsx
- apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx
- backlog/tasks/task-498.1 - Add-Scheduled-Task-Detail-Drawer.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Task 4 for Scheduled Tasks Automation Workbench Phase 1. Add a functional Inspect detail drawer, TDD coverage for reminder and Watchlists task details, page click-through coverage, and task-specific aria-labels for repeated row actions without moving Watchlists configuration into /scheduled-tasks.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added the Scheduled Task detail drawer for reminder and Watchlists tasks, wired the existing Inspect state from the page into the drawer, kept Watchlists links as click-throughs into the Watchlists workspace, and added task-specific accessible names for repeated table row actions. Verification: initial drawer spec red failed because ScheduledTaskDetailDrawer did not exist; focused drawer spec passed 2 tests; drawer plus page specs passed 18 tests; scheduled-task status helper spec passed 10 tests; git diff --check passed. Bandit not applicable because touched implementation is TypeScript/TSX UI code only.
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
