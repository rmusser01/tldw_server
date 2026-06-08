---
id: TASK-2323
title: Address Scheduled Tasks Phase 2A PR review feedback
status: Done
labels:
- scheduled-tasks
- webui
- review-fix
- pr-2317
priority: high
references:
- https://github.com/rmusser01/tldw_server/pull/2317
- TASK-2322
modified_files:
- apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTaskCreatePanel.tsx
- apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTasksPage.tsx
- apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTaskCreatePanel.test.tsx
- apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx
- backlog/tasks/task-2323 - Address-Scheduled-Tasks-Phase-2A-PR-review-feedback.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rebase PR #2317 on latest dev and address actionable PR review comments/check failures for the Scheduled Tasks Phase 2A create framework. Scope is review-fix work only: Ant Design prop/API corrections, any verified CI failures caused by this branch, tests, and PR comment resolution. Preserve Phase 2A scope: no backend contracts, no Watchlists UX/functionality changes, no Home/RAG/ACP/Jobs/Scheduler implementation unless required by an existing failing check directly caused by this PR.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 PR branch is rebased on latest `origin/dev`.
- [x] #2 Actionable PR review feedback is verified against the local codebase before applying changes.
- [x] #3 Verified code changes are tested and committed.
- [x] #4 Non-applicable review suggestions are documented with evidence.
- [x] #5 PR branch is force-pushed safely after rebase.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Rebased `codex/scheduled-tasks-phase2-create-spec` on latest `origin/dev` (`785245fc4f`) with no conflicts.
- Reviewed PR #2317 comments. Gemini's high-priority `Space direction` and `Alert message` suggestions are not applicable to this repo's installed Ant Design v6.2 API: local Vitest emitted deprecation warnings saying to use `orientation` for `Space` and `title` for `Alert`.
- Applied the verified medium-priority UI consistency suggestion by simplifying template state labels so repeated states render as consistent labels such as `Handoff only`, rather than adding template names inside some status tags.
- Updated `ScheduledTaskCreatePanel` coverage to assert repeated state labels with `getAllByText`.
- Verification: `cd apps/packages/ui && bunx vitest run src/components/Option/ScheduledTasks/__tests__/ScheduledTaskCreatePanel.test.tsx src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx --maxWorkers=1 --no-file-parallelism` passed with 2 files and 36 tests.
- Verification: `cd apps/packages/ui && bunx vitest run src/components/Option/ScheduledTasks/__tests__/scheduled-task-route-state.test.ts src/components/Option/ScheduledTasks/__tests__/scheduled-task-templates.test.ts src/components/Option/ScheduledTasks/__tests__/ScheduledTaskCreatePanel.test.tsx src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx src/components/Option/ScheduledTasks/__tests__/ScheduledTaskDetailDrawer.test.tsx src/components/Option/ScheduledTasks/__tests__/scheduled-task-status.test.ts src/components/Option/ScheduledTasks/__tests__/reminder-schedule-utils.test.ts src/routes/__tests__/scheduled-tasks-route.test.tsx --maxWorkers=1 --no-file-parallelism` passed with 8 files and 95 tests.
- Verification: `cd apps/tldw-frontend && bun run lint` passed with 0 errors and existing warnings outside the touched Scheduled Tasks files.
- Verification: `git diff --check` passed.
- Bandit skip: review fix touched frontend TypeScript/TSX tests and Backlog tracking only; no backend Python files changed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Rebased PR #2317 onto latest `origin/dev` and addressed verified review feedback. Kept the Ant Design `Space orientation` and `Alert title` props because local AntD v6.2 warns that `direction` and `message` are deprecated in this repo. Simplified duplicate template state labels and updated the Create panel test accordingly. No backend or Watchlists files changed.
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
