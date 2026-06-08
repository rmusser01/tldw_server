---
id: TASK-2322
title: Implement Scheduled Tasks Automation Workbench Phase 2A create framework
status: Done
labels:
- scheduled-tasks
- webui
- ux
- phase-2a
- implementation
priority: high
references:
- TASK-2321
- TASK-2320
- Docs/superpowers/plans/2026-06-08-scheduled-tasks-automation-workbench-phase2a-create-framework-implementation-plan.md
- Docs/superpowers/specs/2026-06-08-scheduled-tasks-automation-workbench-phase2-creation-design.md
modified_files:
- apps/packages/ui/src/components/Option/ScheduledTasks/scheduled-task-route-state.ts
- apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/scheduled-task-route-state.test.ts
- apps/packages/ui/src/components/Option/ScheduledTasks/scheduled-task-templates.ts
- apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/scheduled-task-templates.test.ts
- apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTaskCreatePanel.tsx
- apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTaskCreatePanel.test.tsx
- apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTasksPage.tsx
- apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the frontend-only Phase 2A /scheduled-tasks Create framework from the approved spec and implementation plan. Scope: URL-addressable Overview/Tasks/Create tabs, task detail deep links, static template registry, deterministic template finder, Create panel, Reminder as the only fully available creation template, handoff-only Watch/Ingest/Advanced panels, planned RAG/Agent states, conservative reminder success copy, URL privacy safeguards, extension-sized behavior, and focused tests. Do not add backend contracts or change Watchlists deep-workspace behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 URL state helpers support Overview, Tasks, Create, selected template, task detail, invalid tab, invalid template, and invalid task states.
- [x] #2 Template registry and matcher keep Reminder as the only fully available template and keep Watch/Ingest/Advanced handoff-only while RAG/Agent remain planned.
- [x] #3 Create panel renders templates by intent, not source vendor, and uses handoff panels without claiming a task was created.
- [x] #4 ScheduledTasksPage integrates tabs, create flow, detail deep links, invalid route states, and created reminder detail navigation while preserving Phase 1 overview/table/detail behavior.
- [x] #5 Focused ScheduledTasks and route tests pass; extension route smoke is updated or skip rationale is recorded.
- [x] #6 No backend files are changed; Bandit is run only if backend Python changes unexpectedly, otherwise skip rationale is recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-08-scheduled-tasks-automation-workbench-phase2a-create-framework-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Task 1: Added pure scheduled task route-state helpers for Phase 2A tabs, template IDs, task IDs, invalid tabs, and URL search serialization.
- Task 1 verification: `cd apps/packages/ui && bunx vitest run src/components/Option/ScheduledTasks/__tests__/scheduled-task-route-state.test.ts --maxWorkers=1 --no-file-parallelism` passed with 6 tests.
- Bandit skip: Task 1 changed frontend TypeScript and Backlog tracking only; no backend Python files were touched.
- Task 1 quality follow-up: Normalized caller-provided `templateId` and `taskId` before URL serialization so whitespace-only IDs are omitted instead of encoded.
- Task 1 quality verification: `cd apps/packages/ui && bunx vitest run src/components/Option/ScheduledTasks/__tests__/scheduled-task-route-state.test.ts --maxWorkers=1 --no-file-parallelism` passed with 8 tests.
- Task 2: Added the static Scheduled Tasks template registry and deterministic matcher. Reminder is the only available Phase 2A creation template; Watch, Ingest, and Advanced are handoff-only; Recurring Question and Agent Task remain planned.
- Task 2 red verification: `cd apps/packages/ui && bunx vitest run src/components/Option/ScheduledTasks/__tests__/scheduled-task-templates.test.ts --maxWorkers=1 --no-file-parallelism` failed because `../scheduled-task-templates` did not exist.
- Task 2 verification: `cd apps/packages/ui && bunx vitest run src/components/Option/ScheduledTasks/__tests__/scheduled-task-templates.test.ts --maxWorkers=1 --no-file-parallelism` passed with 10 tests.
- Bandit skip: Task 2 changed frontend TypeScript tests/helpers and Backlog tracking only; no backend Python files were touched.
- Task 2 spec-review follow-up: Removed the extra Ingest matcher keyword `channel`; the focused template test still matches `keep this channel searchable` through the required `searchable` keyword.
- Task 2 follow-up verification: `cd apps/packages/ui && bunx vitest run src/components/Option/ScheduledTasks/__tests__/scheduled-task-templates.test.ts --maxWorkers=1 --no-file-parallelism` passed with 10 tests.
- Task 2 code-quality follow-up: Hardened template matching to use word/phrase boundaries instead of raw substrings, and treated bare-domain source text with fragments as unsafe handoff text.
- Task 2 code-quality red verification: `cd apps/packages/ui && bunx vitest run src/components/Option/ScheduledTasks/__tests__/scheduled-task-templates.test.ts --maxWorkers=1 --no-file-parallelism` failed on `renew credentials` matching Watch through `new` and `example.com/feed#private` being accepted.
- Task 2 code-quality verification: `cd apps/packages/ui && bunx vitest run src/components/Option/ScheduledTasks/__tests__/scheduled-task-templates.test.ts --maxWorkers=1 --no-file-parallelism` passed with 13 tests.
- Task 2 privacy follow-up: Broadened sensitive handoff URL parameter detection to reject compound keys such as `access_token`, `refresh_token`, `id_token`, and `client_secret`.
- Task 2 privacy red verification: `cd apps/packages/ui && bunx vitest run src/components/Option/ScheduledTasks/__tests__/scheduled-task-templates.test.ts --maxWorkers=1 --no-file-parallelism` failed because `https://example.com/feed?access_token=secret` was accepted.
- Task 2 privacy verification: `cd apps/packages/ui && bunx vitest run src/components/Option/ScheduledTasks/__tests__/scheduled-task-templates.test.ts --maxWorkers=1 --no-file-parallelism` passed with 14 tests.
- Task 3: Added the standalone `ScheduledTaskCreatePanel` with template finder/filter controls, intent-oriented template cards, selected-state Reminder editor handoff, Watch/Ingest/Advanced handoff panels, planned capability panels, and unsafe setup-note summary suppression.
- Task 3 red verification: `cd apps/packages/ui && bunx vitest run src/components/Option/ScheduledTasks/__tests__/ScheduledTaskCreatePanel.test.tsx --maxWorkers=1 --no-file-parallelism` failed because `../ScheduledTaskCreatePanel` did not exist.
- Task 3 verification: `cd apps/packages/ui && bunx vitest run src/components/Option/ScheduledTasks/__tests__/ScheduledTaskCreatePanel.test.tsx --maxWorkers=1 --no-file-parallelism` passed with 6 tests.
- Task 3 template regression verification: `cd apps/packages/ui && bunx vitest run src/components/Option/ScheduledTasks/__tests__/scheduled-task-templates.test.ts --maxWorkers=1 --no-file-parallelism` passed with 14 tests.
- Task 3 additional check: `cd apps/packages/ui && bunx tsc --noEmit --pretty false` was attempted but exited with Node heap out-of-memory before type diagnostics were produced.
- Bandit skip: Task 3 changed frontend TypeScript/TSX tests/components and Backlog tracking only; no backend Python files were touched.
- Task 3 code-quality follow-up: Removed raw free-form handoff notes from visible/copyable setup summaries and added prose-secret warning coverage for values like `api key: sk-test-secret`.
- Task 3 code-quality red verification: `cd apps/packages/ui && bunx vitest run src/components/Option/ScheduledTasks/__tests__/ScheduledTaskCreatePanel.test.tsx --maxWorkers=1 --no-file-parallelism` failed because prose secret text rendered in the setup summary and no warning appeared.
- Task 3 code-quality verification: `cd apps/packages/ui && bunx vitest run src/components/Option/ScheduledTasks/__tests__/ScheduledTaskCreatePanel.test.tsx --maxWorkers=1 --no-file-parallelism` passed with 7 tests.
- Task 3 code-quality template regression verification: `cd apps/packages/ui && bunx vitest run src/components/Option/ScheduledTasks/__tests__/scheduled-task-templates.test.ts --maxWorkers=1 --no-file-parallelism` passed with 14 tests.
- Task 3 code-quality diff check: `git diff --check` passed.
- Task 4: Integrated URL-backed Overview, Tasks, and Create tabs into `ScheduledTasksPage` using the Phase 2A route-state helpers. Overview now owns summary/callout content, Tasks owns the empty/table/detail drawer surface, and Create renders `ScheduledTaskCreatePanel` with URL-selected templates.
- Task 4: Added task detail deep-link synchronization for `?tab=tasks&task_id=...`, row inspection URL updates, close/removed-task route clearing, invalid-tab fallback copy, and invalid task non-blocking copy.
- Task 4 TDD red verification: `cd apps/packages/ui && bunx vitest run src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx --maxWorkers=1 --no-file-parallelism` failed with the expected missing tab/deep-link/Create editor behaviors before production changes.
- Task 4 verification: `cd apps/packages/ui && bunx vitest run src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx --maxWorkers=1 --no-file-parallelism` passed with 28 tests.
- Task 4 CreatePanel regression verification: `cd apps/packages/ui && bunx vitest run src/components/Option/ScheduledTasks/__tests__/ScheduledTaskCreatePanel.test.tsx --maxWorkers=1 --no-file-parallelism` passed with 7 tests.
- Task 4 diff check: `git diff --check` passed.
- Bandit skip: Task 4 changed frontend TypeScript/TSX tests/components and Backlog tracking only; no backend Python files were touched.
- Task 4 quality fix: After successful reminder creation from the Create tab, `ScheduledTasksPage` now clears the selected `template` route param while staying on `?tab=create`, so the create form unmounts and the template list is visible without implementing Task 5 created-task detail navigation.
- Task 4 quality red verification: `cd apps/packages/ui && bunx vitest run src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx --maxWorkers=1 --no-file-parallelism` failed because the Title input remained mounted after successful reminder creation.
- Task 4 quality verification: `cd apps/packages/ui && bunx vitest run src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx --maxWorkers=1 --no-file-parallelism` passed with 28 tests.
- Task 4 quality CreatePanel regression verification: `cd apps/packages/ui && bunx vitest run src/components/Option/ScheduledTasks/__tests__/ScheduledTaskCreatePanel.test.tsx --maxWorkers=1 --no-file-parallelism` passed with 7 tests.
- Task 4 quality diff check: `git diff --check` passed.
- Task 5: Wired successful Reminder creation from the Create panel to the created task detail. The page now stores the returned created task as a temporary detail fallback, routes to `?tab=tasks&task_id=<created.id>`, prefers refreshed list data when it arrives, and clears the fallback when the list catches up, selection changes, detail closes, or the selected reminder is deleted.
- Task 5 red verification: `cd apps/packages/ui && bunx vitest run src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx --maxWorkers=1 --no-file-parallelism` failed with 2 expected failures because successful Create-panel reminder creation still left the Create tab selected and did not open a detail dialog.
- Task 5 verification: `cd apps/packages/ui && bunx vitest run src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx --maxWorkers=1 --no-file-parallelism` passed with 29 tests.
- Task 5 CreatePanel regression verification: `cd apps/packages/ui && bunx vitest run src/components/Option/ScheduledTasks/__tests__/ScheduledTaskCreatePanel.test.tsx --maxWorkers=1 --no-file-parallelism` passed with 7 tests.
- Task 5 diff check: `git diff --check` passed.
- Bandit skip: Task 5 changed frontend TypeScript/TSX tests/components and Backlog tracking only; no backend Python files were touched.
- Task 6: Inspected the extension E2E smoke test and scheduled-tasks route parity test. The current Scheduled Tasks page still renders the `Track reminders, Watchlist monitors...` description asserted by the extension smoke, and both web and extension route shells still import the shared `ScheduledTasksPage`, so no route or extension test edits were needed.
- Task 6 touched files: Backlog tracking only. `apps/extension/tests/e2e/integrations-and-scheduled-tasks.spec.ts` and `apps/packages/ui/src/routes/__tests__/scheduled-tasks-route.test.tsx` were intentionally left unchanged.
- Task 6 route verification: `cd apps/packages/ui && bunx vitest run src/routes/__tests__/scheduled-tasks-route.test.tsx --maxWorkers=1 --no-file-parallelism` passed with 1 test file and 3 tests.
- Task 6 focused verification: `cd apps/packages/ui && bunx vitest run src/components/Option/ScheduledTasks/__tests__/scheduled-task-route-state.test.ts src/components/Option/ScheduledTasks/__tests__/scheduled-task-templates.test.ts src/components/Option/ScheduledTasks/__tests__/ScheduledTaskCreatePanel.test.tsx src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx src/components/Option/ScheduledTasks/__tests__/scheduled-task-status.test.ts src/components/Option/ScheduledTasks/__tests__/reminder-schedule-utils.test.ts src/routes/__tests__/scheduled-tasks-route.test.tsx --maxWorkers=1 --no-file-parallelism` passed with 7 test files and 92 tests.
- Task 6 extension E2E execution skip: not run for this parity task because the required instructions only requested inspection plus Vitest route/focused groups; the smoke assertion did not require a copy update.
- Task 6 Bandit skip: no backend Python files were changed; Task 6 touched Backlog tracking only.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the frontend-only Scheduled Tasks Phase 2A create framework. Added URL route-state helpers, static intent template registry/matcher, the Create panel, URL-backed Overview/Tasks/Create tabs, task detail deep links, invalid-route handling, privacy-safe handoff summaries, and created-reminder detail navigation with temporary fallback while refreshed task data catches up. Reminder remains the only fully createable template; Watch/Ingest/Advanced remain handoff-only; Recurring Question and Agent Task remain planned. Watchlists and backend code were not changed.

Final verification on 2026-06-08:
- `cd apps/packages/ui && bunx vitest run src/components/Option/ScheduledTasks/__tests__/scheduled-task-route-state.test.ts src/components/Option/ScheduledTasks/__tests__/scheduled-task-templates.test.ts src/components/Option/ScheduledTasks/__tests__/ScheduledTaskCreatePanel.test.tsx src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx src/components/Option/ScheduledTasks/__tests__/ScheduledTaskDetailDrawer.test.tsx src/components/Option/ScheduledTasks/__tests__/scheduled-task-status.test.ts src/components/Option/ScheduledTasks/__tests__/reminder-schedule-utils.test.ts src/routes/__tests__/scheduled-tasks-route.test.tsx --maxWorkers=1 --no-file-parallelism` passed with 8 files and 95 tests.
- `git diff --check` passed.
- Changed-scope check found only docs/spec/plan, Backlog records, and frontend Scheduled Tasks TypeScript/TSX/test files.
- Bandit skipped because no backend Python files were changed.
- Extension E2E was not run; Task 6 verified the extension smoke copy still matches and route shells still import the shared ScheduledTasksPage, with focused route Vitest coverage passing.
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
