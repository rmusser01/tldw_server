---
id: TASK-2335
title: Implement Scheduled Tasks Phase 3 results panel and detail drawer
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
Implement Stage 3 of the Scheduled Tasks Phase 3 plan: replace the temporary Results tab shell with a source-agnostic results panel, filters, capability-aware action visibility, and a result detail drawer that explains provenance, owner, run/result state, and next links without showing unsupported review or retry controls.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Results panel renders success, failure, running, and completed/no-results projected signals with visible state and owner text.
- [x] #2 Projected-mode filters support result state, task type, and owner while hiding durable Review state filters.
- [x] #3 Empty states distinguish no tasks, no results, and no filter matches.
- [x] #4 Detail drawer shows result summary, provenance labels, owner, task/run/result ids where available, and Watchlists deep links for Watchlist-owned results.
- [x] #5 Retry/review controls are hidden in projected mode and a concise capability note explains when they appear.
- [x] #6 ScheduledTasksPage opens/closes the result drawer from result rows and result/run/task deep links while preserving task detail drawer behavior.
- [x] #7 Focused panel, drawer, page, and Scheduled Tasks tests pass; verification and Bandit/frontend-only rationale are recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-09-scheduled-tasks-phase3-results-inbox-home-surfacing-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added `ScheduledTaskResultsPanel.tsx` with source-agnostic projected result scanning, state/type/owner filters, normalized-mode review filtering, and separate no-task/no-result/no-filter-match empty states.
- Added `ScheduledTaskResultDetailDrawer.tsx` with provenance, owner, task/run/result ids, result summary, Watchlists deep links, and capability-aware review/retry controls.
- Wired `ScheduledTasksPage.tsx` Results rows and result/run/task deep links to the detail drawer while preserving the existing Tasks drawer route behavior.
- Gave the result drawer an explicit `aria-labelledby` target because Ant Design's test-mode generated drawer id can collide with other generated ids, producing an empty accessible dialog name.
- Rendered page-level task/result detail drawers only when active to avoid inactive overlay DOM competing in the accessibility tree.
- Updated `ScheduledTasksPage.test.tsx` to assert direct result drawer opening from `/scheduled-tasks/results?result_id=...`.
- Verification:
  - `./node_modules/.bin/vitest run src/components/Option/ScheduledTasks/__tests__/ScheduledTaskResultsPanel.test.tsx src/components/Option/ScheduledTasks/__tests__/ScheduledTaskResultDetailDrawer.test.tsx src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx` passed 41 tests.
  - `./node_modules/.bin/vitest run src/components/Option/ScheduledTasks/__tests__` passed 133 tests.
  - `git diff --check` passed.
  - `./node_modules/.bin/tsc --noEmit` failed with Node heap exhaustion; `node --max-old-space-size=8192 ./node_modules/typescript/bin/tsc --noEmit` completed and failed only on existing unrelated Notes, background, and voice-cloning type errors. Touched Scheduled Tasks type errors found by the first high-heap run were fixed before final verification.
- Bandit skipped because this task touched only frontend TypeScript/TSX files, tests, Backlog metadata, and the implementation plan; no Python/backend files changed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the Scheduled Tasks Results panel and detail drawer for projected result signals. The Results tab now has filterable source-agnostic result rows, accurate empty states, deep-link-driven result inspection, Watchlists continuation links, and hidden retry/review actions unless future result capabilities support them.
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
