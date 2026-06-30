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
- apps/packages/ui/src/components/Option/ScheduledTasks/scheduled-task-status.ts
- backlog/tasks/task-498.1 - Add-Scheduled-Task-Detail-Drawer.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Task 4 for Scheduled Tasks Automation Workbench Phase 1. Add a functional Inspect detail drawer, TDD coverage for reminder and Watchlists task details, page click-through coverage, and task-specific aria-labels for repeated row actions without moving Watchlists configuration into /scheduled-tasks.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Inspect opens a Scheduled Task detail drawer for reminder and Watchlists rows.
- [x] Drawer content shows product status, task type, management owner, schedule, timezone, run timing, and source reference details.
- [x] Reminder drawer actions support Edit reminder and Delete reminder from the page without overlapping stale drawer UI.
- [x] Watchlists drawer actions remain read-only deep links into the Watchlists workspace and do not move Watchlists configuration into `/scheduled-tasks`.
- [x] Detail drawer selection reconciles with latest scheduled task query data and closes when the selected task id is no longer present.
- [x] Unbounded primitive source values are truncated in the drawer while retaining the full value for inspection.
- [x] Focused tests cover drawer details, Watchlists links, table Inspect click-through, drawer edit/delete actions, stale refetch reconciliation, and status helpers.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed Task 4 review follow-up. The page now stores selectedTaskId, derives the drawer task from the latest scheduled task query data, and clears stale selection when a refetch removes the selected id. Page tests now cover drawer Edit reminder and Delete reminder actions, including editor/drawer non-overlap and stale drawer cleanup after deletion. The drawer truncates long primitive source values while preserving the full value via title. Shared display helpers for native reminder detection, status tag color, and timestamp formatting were moved into scheduled-task-status.ts and reused by the table and drawer. TDD red evidence: focused suite failed on long link_url rendering in full and stale drawer remaining after refetch removal. Green evidence: final focused Vitest command passed 3 files / 32 tests; git diff --check passed. Bandit not applicable because touched implementation is TypeScript/TSX UI code only.
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
