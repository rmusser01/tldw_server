---
id: TASK-2336
title: Implement Scheduled Tasks Phase 3 notification links and dedupe
status: Done
labels:
- scheduled-tasks
- webui
- ux
- implementation
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Stage 4 of the Scheduled Tasks Phase 3 plan: normalize scheduled-task notification targets, add deterministic dedupe helpers for task/run/result signals, preserve existing notification behavior, and add focused tests for notification-derived result links and dedupe behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Notification payloads with link_url, source_task_id, source_task_run_id, source_job_id, source metadata, or task/run/result ids resolve to scheduled-task Results deep links when possible.
- [x] #2 Exact result ids win over run ids, and run ids win over task-scoped fallback links.
- [x] #3 Dedupe keys keep separate runs separate and keep failure/result signals separate when no exact result id exists.
- [x] #4 Existing notification stream, mark-read, dismiss, and snooze behavior is unchanged; helpers remain pure unless needed by Home in a later stage.
- [x] #5 Focused scheduled-task result/link and notification service tests pass; verification and Bandit/frontend-only rationale are recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-09-scheduled-tasks-phase3-results-inbox-home-surfacing-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Extended `scheduled-task-result-links.ts` to normalize result, run, and task notification targets from direct fields, `link_url`, `link_type`/`link_id`, source task/run ids, and Watchlists job ids.
- Added signal-kind inference for notification-derived targets so failures, running signals, results, and completed/no-result notifications do not collapse into the same run-level dedupe key.
- Added `notificationIds`, `signalKind`, `createdAt`, and `severity` to normalized scheduled-task notification targets for later Home merging.
- Added `mergeScheduledTaskNotificationTargets` to collapse duplicate notification targets by dedupe key while preserving all notification ids.
- Kept notification service behavior unchanged; Stage 4 helper tests import the pure scheduled-task link helper, and existing notification service tests still cover stream, list, read, dismiss, and snooze behavior.
- Left the plan's non-blocking Home notification load item unchecked because it belongs with the Home integration in Stage 6.
- Verification:
  - `./node_modules/.bin/vitest run src/components/Option/ScheduledTasks/__tests__/scheduled-task-results.test.ts` passed 12 tests.
  - `./node_modules/.bin/vitest run src/components/Option/ScheduledTasks/__tests__` passed 136 tests.
  - `./node_modules/.bin/vitest run src/services/__tests__/notifications.test.ts` passed 9 tests.
- Bandit skipped because this task touched only frontend TypeScript tests/helpers, Backlog metadata, and the implementation plan; no Python/backend files changed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented scheduled-task notification target normalization and dedupe helpers. Result ids now take priority over run ids, run ids take priority over task-scoped fallbacks, and duplicate notification-derived targets retain their source notification ids without changing existing notification service behavior.
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
