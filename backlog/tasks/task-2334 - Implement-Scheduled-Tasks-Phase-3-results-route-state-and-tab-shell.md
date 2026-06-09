---
id: TASK-2334
title: Implement Scheduled Tasks Phase 3 results route state and tab shell
status: Done
labels:
- scheduled-tasks
- webui
- routing
- implementation
priority: high
modified_files:
- Docs/superpowers/plans/2026-06-09-scheduled-tasks-phase3-results-inbox-home-surfacing-implementation-plan.md
- apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTasksPage.tsx
- apps/packages/ui/src/components/Option/ScheduledTasks/scheduled-task-route-state.ts
- apps/packages/ui/src/components/Option/ScheduledTasks/scheduled-task-results.ts
- apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx
- apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/scheduled-task-route-state.test.ts
- apps/packages/ui/src/routes/route-registry.tsx
- apps/packages/ui/src/routes/__tests__/scheduled-tasks-route.test.tsx
- apps/tldw-frontend/extension/routes/route-registry.tsx
- apps/tldw-frontend/pages/scheduled-tasks/results.tsx
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Stage 2 of the Scheduled Tasks Phase 3 plan: make Results a first-class `/scheduled-tasks` tab, parse and build result/run/task deep links, add `/scheduled-tasks/results` aliases for extension and hosted WebUI, add a minimal Results tab shell wired to projected result items, and preserve existing Tasks tab detail behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `?tab=results` is accepted as a first-class Scheduled Tasks tab and invalid tabs still fall back to Overview.
- [x] #2 `result_id`, `run_id`, and `task_id` are parsed and serialized safely for the Results tab without breaking Tasks tab task-detail behavior.
- [x] #3 `/scheduled-tasks/results` aliases to the same Scheduled Tasks route in the extension and hosted WebUI.
- [x] #4 ScheduledTasksPage renders a Results tab shell from projected result items and can deep-link to result/run/task-scoped results without a dead-end.
- [x] #5 Focused route-state and ScheduledTasksPage tests cover the new tab, deep-link builders, alias route behavior where practical, and missing-result handling.
- [x] #6 Verification and Bandit/frontend-only rationale are recorded in Backlog.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-09-scheduled-tasks-phase3-results-inbox-home-surfacing-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Added Results to the Scheduled Tasks tab IA and extended route parsing/building with safe `result_id`, `run_id`, and task-scoped result route state. Newline-bearing route ids are rejected before serialization. Invalid tabs continue to fall back to Overview.

ScheduledTasksPage now defaults `/scheduled-tasks/results` aliases into the Results tab, projects current task-list data into latest automation signals, renders a minimal source-agnostic Results shell, supports exact result/run/task selection, and shows a non-blocking missing-result warning for stale deep links. Existing Tasks tab task-detail state remains separate.

Added `/scheduled-tasks/results` route entries to shared and extension registries and added the hosted Next.js alias page. Verification: `./node_modules/.bin/vitest run src/components/Option/ScheduledTasks/__tests__/scheduled-task-route-state.test.ts src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx src/routes/__tests__/scheduled-tasks-route.test.tsx` passed (3 files, 47 tests). `./node_modules/.bin/vitest run src/components/Option/ScheduledTasks/__tests__` passed (9 files, 124 tests). Bandit skipped because this task touched only frontend TypeScript/TSX, route-page, Markdown plan, and Backlog files; no Python code was changed.

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented Scheduled Tasks Phase 3 Stage 2 route and tab shell. Results is now a first-class Scheduled Tasks tab with safe `result_id`, `run_id`, and task-scoped route state, `/scheduled-tasks/results` aliases exist for shared, extension, and hosted WebUI surfaces, and ScheduledTasksPage renders a projected latest-signals shell with exact deep-link selection and missing-result handling while preserving the existing Tasks tab drawer behavior. Focused Stage 2 tests and the full ScheduledTasks component test folder passed. Bandit was not applicable because no Python/backend files were touched.
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
