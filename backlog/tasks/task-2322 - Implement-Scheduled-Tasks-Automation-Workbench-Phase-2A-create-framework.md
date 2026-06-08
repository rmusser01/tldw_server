---
id: TASK-2322
title: Implement Scheduled Tasks Automation Workbench Phase 2A create framework
status: In Progress
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
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the frontend-only Phase 2A /scheduled-tasks Create framework from the approved spec and implementation plan. Scope: URL-addressable Overview/Tasks/Create tabs, task detail deep links, static template registry, deterministic template finder, Create panel, Reminder as the only fully available creation template, handoff-only Watch/Ingest/Advanced panels, planned RAG/Agent states, conservative reminder success copy, URL privacy safeguards, extension-sized behavior, and focused tests. Do not add backend contracts or change Watchlists deep-workspace behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 URL state helpers support Overview, Tasks, Create, selected template, task detail, invalid tab, invalid template, and invalid task states.
- [ ] #2 Template registry and matcher keep Reminder as the only fully available template and keep Watch/Ingest/Advanced handoff-only while RAG/Agent remain planned.
- [ ] #3 Create panel renders templates by intent, not source vendor, and uses handoff panels without claiming a task was created.
- [ ] #4 ScheduledTasksPage integrates tabs, create flow, detail deep links, invalid route states, and created reminder detail navigation while preserving Phase 1 overview/table/detail behavior.
- [ ] #5 Focused ScheduledTasks and route tests pass; extension route smoke is updated or skip rationale is recorded.
- [ ] #6 No backend files are changed; Bandit is run only if backend Python changes unexpectedly, otherwise skip rationale is recorded.
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
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

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
