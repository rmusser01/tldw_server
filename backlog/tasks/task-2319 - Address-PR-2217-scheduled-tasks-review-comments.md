---
id: TASK-2319
title: Address PR 2217 scheduled-tasks review comments
status: Done
assignee: []
created_date: '2026-06-08 06:44'
updated_date: 2026-06-08 00:04
labels:
  - scheduled-tasks
  - code-review
  - webui
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/2217'
modified_files:
  - apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTaskDetailDrawer.tsx
  - apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTaskTable.tsx
  - apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTasksPage.tsx
  - apps/packages/ui/src/components/Option/ScheduledTasks/WatchlistJobReadOnlyPanel.tsx
  - apps/packages/ui/src/components/Option/ScheduledTasks/WatchlistTaskActionLinks.tsx
  - apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx
  - apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/reminder-schedule-utils.test.ts
  - apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/scheduled-task-status.test.ts
  - apps/packages/ui/src/components/Option/ScheduledTasks/reminder-schedule-utils.ts
  - apps/packages/ui/src/components/Option/ScheduledTasks/scheduled-task-status.ts
  - backlog/tasks/task-494 - Create-scheduled-tasks-Automation-Workbench-UX-PRD.md
  - backlog/tasks/task-2319 - Address-PR-2217-scheduled-tasks-review-comments.md
priority: high
---

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Rebase PR branch onto latest origin/dev
- [x] #2 Resolve valid review comments for scheduled task UI and safety issues
- [x] #3 Document or reply to invalid/stale review comments
- [x] #4 Run focused frontend/backend verification and push updates
<!-- AC:END -->

## Implementation Notes
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Rebased PR branch onto latest `origin/dev`.
- Sanitized Watchlists `manage_url` rendering so only path-only `/watchlists` URLs are accepted; unsafe or cross-origin values fall back to `/watchlists?tab=jobs`.
- Reused one `WatchlistTaskActionLinks` renderer from both the task table and detail drawer to remove duplicated Watchlists deep-link rendering.
- Replaced native task filters with AntD `Select` controls and assigned stable IDs to avoid colliding with Drawer title IDs in the accessibility tree.
- Added an abortable `/openapi.json` scheduled-task support probe with an 8-second timeout.
- Added missing standard Backlog frontmatter fields to TASK-494.
- Logged browser timezone detection failures before falling back to UTC.
- Rejected `#` nth-weekday cron syntax because APScheduler `CronTrigger.from_crontab` accepts `mon#2` but normalizes it to plain `mon`, so allowing it would misrepresent the saved schedule.
- Verified and kept AntD `Alert.title` and `Space.orientation` because the installed AntD 6.2.0 runtime warns that `message` and `direction` are deprecated.
- Verified and kept `?` rejected in cron expressions because APScheduler rejects `CronTrigger.from_crontab("0 9 ? * mon")`.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary
<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed PR 2217 scheduled-task review feedback after rebasing on latest dev. Frontend verification passed for the expanded scheduled-task suite: 65 tests. Backend scheduled-task control-plane tests passed: 4 tests. TypeScript package check completed with unrelated baseline errors in `OpenUIRenderer.tsx` only. Bandit is not applicable because no Python source files were changed.
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
