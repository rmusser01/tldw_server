---
id: TASK-2337
title: Implement Scheduled Tasks Phase 3 overview and task cross-links
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-09 07:20'
labels:
  - scheduled-tasks
  - webui
  - ux
  - implementation
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Stage 5 of the Scheduled Tasks Phase 3 plan: make projected results discoverable from the Scheduled Tasks overview, task table, and task detail drawer while preserving Watchlists as the owner for external monitor setup and management.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Overview surfaces newest projected result signals and provides a clear link to the Results tab without durable review-count claims in projected mode.
- [x] #2 Tasks table exposes a Results action only for tasks that have projected result signals, while native reminder edit/delete and Watchlists owner controls remain unchanged.
- [x] #3 Task detail drawer links to the latest result/run when known and preserves Watchlists owner copy.
- [x] #4 No Watchlists-owned edit controls are introduced in Scheduled Tasks.
- [x] #5 Focused overview/table/detail/page tests and full Scheduled Tasks tests pass; verification and Bandit/frontend-only rationale are recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-09-scheduled-tasks-phase3-results-inbox-home-surfacing-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- ScheduledTaskOverview now renders the newest projected result signal with a direct action into the Results tab, without review-count copy in projected mode.
- ScheduledTaskTable now computes task result presence from projected result items and only shows a Results row action when at least one projected signal exists for that task.
- ScheduledTaskDetailDrawer now accepts the latest result signal and exposes an exact latest-result link before the existing native reminder or Watchlists-owned actions.
- Watchlists rows still show Watchlists settings/activity/reports links and remain read-only from Scheduled Tasks; no Watchlists-owned edit controls were added.
- Verification: ./node_modules/.bin/vitest run src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx src/components/Option/ScheduledTasks/__tests__/ScheduledTaskDetailDrawer.test.tsx passed 36 tests.
- Verification: ./node_modules/.bin/vitest run src/components/Option/ScheduledTasks/__tests__ passed 136 tests.
- Verification: git diff --check passed before task finalization.
- Bandit: skipped because this slice only changes frontend TypeScript/React and Backlog/plan text, not Python executable code.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented Scheduled Tasks Stage 5 overview and task cross-links. Projected result signals are now discoverable from Overview, task rows expose a gated Results action, and task details link to the latest known result while preserving Watchlists ownership boundaries.
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
