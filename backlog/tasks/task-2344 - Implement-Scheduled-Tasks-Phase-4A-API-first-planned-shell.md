---
id: TASK-2344
title: Implement Scheduled Tasks Phase 4A API-first planned shell
status: Done
labels:
- scheduled-tasks
- ux
- frontend
priority: High
documentation:
- Docs/superpowers/specs/2026-06-09-scheduled-tasks-phase4-recurring-question-agent-task-api-contract-design.md
modified_files:
- apps/packages/ui/src/components/Option/ScheduledTasks/scheduled-task-planned-template-copy.ts
- apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/scheduled-task-planned-template-copy.test.ts
- apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTaskCreatePanel.tsx
- apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTaskCreatePanel.test.tsx
- apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/scheduled-task-template-capabilities.test.ts
- apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTaskResultsPanel.tsx
- apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTaskResultsPanel.test.tsx
- apps/packages/ui/src/components/Option/CompanionHome/cards/AutomationInboxCard.tsx
- apps/packages/ui/src/components/Option/CompanionHome/__tests__/AutomationInboxCard.test.tsx
- apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx
- backlog/tasks/task-2344 - Implement-Scheduled-Tasks-Phase-4A-API-first-planned-shell.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute the approved Phase 4A plan for API-first planned scheduled task shells: planned template copy/model, planned create panels, capability guards, results/home copy, route-level deep-link coverage, focused verification, and branch hygiene.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-09-scheduled-tasks-phase4a-api-first-planned-shell-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Task 1 planned-family helper added for `recurring_question` and `agent_task` with `createEnabled: false`, requirements text, planned result destinations, safety copy, and navigation links. Verification: RED `bunx vitest run src/components/Option/ScheduledTasks/__tests__/scheduled-task-planned-template-copy.test.ts` failed on missing `../scheduled-task-planned-template-copy`; GREEN same command passed with `1 passed`, `4 passed`. Bandit skipped: touched scope is TypeScript/frontend only.
- Task 1 review follow-up: added planned-family test assertions for each model's `availabilityReason`, including the `not executable in this client yet` contract language. Verification: `bunx vitest run src/components/Option/ScheduledTasks/__tests__/scheduled-task-planned-template-copy.test.ts` passed with `1 passed`, `4 passed`.
- Task 2 planned create panels: expanded ScheduledTaskCreatePanel planned tests for `recurring_question` and `agent_task`, then replaced the minimal planned panel with the Task 1 planned-template model rendering for status, planned availability, non-creatable copy, requirements, result destinations, safety lines, and related destination links. RED verification: `bunx vitest run src/components/Option/ScheduledTasks/__tests__/ScheduledTaskCreatePanel.test.tsx` initially failed before dependency relinking because `antd` was unresolved. After `bun install` restored the UI package dependency graph, the same command failed as intended with 2 failed planned-panel tests missing `Planned automation type` and rich model copy. GREEN verification: same focused command passed with `1 passed` file and `13 passed` tests. Bandit skipped: touched scope is TypeScript/frontend only.
- Task 3 capability guardrail tests: added assertions that `recurring_question` and `agent_task` return `null` from `resolveTemplateCapabilityState` even with Watch-style gates and `creationAdapterSupported: true`, plus coverage that applying those capabilities preserves the base `planned` state. Verification: `bunx vitest run src/components/Option/ScheduledTasks/__tests__/scheduled-task-templates.test.ts src/components/Option/ScheduledTasks/__tests__/scheduled-task-template-capabilities.test.ts` passed with `2 passed` files and `30 passed` tests. No RED occurred because the helper already had the required non-Watch/Ingest guard. Bandit skipped: touched implementation scope is TypeScript/frontend test-only.
- Task 4 results/home routing copy: added exact-copy regression assertions for the projected Results alert and Home Automation Inbox empty description, then updated only the relevant UI copy so future scheduled questions and agent outputs are described as policy/API-routed future destinations. No filtering, routes, loading, item rendering, or backend calls changed. RED verification: `cd apps/packages/ui && bunx vitest run src/components/Option/ScheduledTasks/__tests__/ScheduledTaskResultsPanel.test.tsx src/components/Option/CompanionHome/__tests__/AutomationInboxCard.test.tsx` failed with 2 expected missing-copy assertions and 7 passing tests. GREEN verification: same focused command passed with 2 files and 9 tests. `git diff --check` passed. Bandit gate: initial worktree-local `.venv` source failed because `.venv` is absent in this worktree; reran with shared repo virtualenv at `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv`, wrote `/tmp/bandit_task_2344.json`, and report contained `errors: []`, `results: []`, `loc: 0` for the frontend TypeScript/TSX scope.
- Task 4 follow-up review fix: updated the stale `ScheduledTasksPage.test.tsx` projected Results copy assertion to the same future-routing policy/API string used by the Results panel. Verification: `cd apps/packages/ui && bunx vitest run src/components/Option/ScheduledTasks/__tests__/ScheduledTaskResultsPanel.test.tsx src/components/Option/CompanionHome/__tests__/AutomationInboxCard.test.tsx` passed with 2 files and 9 tests; `cd apps/packages/ui && bunx vitest run src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx` passed with 1 file and 34 tests. `git diff --check` passed. Bandit: explicit `.tsx` file scan is not applicable and reported a Python AST parse error with no findings; reran the containing frontend test directory and `/tmp/bandit_task_2344_page_copy_dir.json` reported `errors: []`, `results: []`, `loc: 0`.
- Task 5 route-level deep-link coverage: added `ScheduledTasksPage.test.tsx` coverage for `/scheduled-tasks?tab=create&template=recurring_question` and `/scheduled-tasks?tab=create&template=agent_task`. Verification: `cd apps/packages/ui && bunx vitest run src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx` passed with 1 file and 36 tests. No production code changed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the Phase 4A API-first planned shell for Scheduled Tasks. Added a typed planned-template helper/model for Recurring Question and Agent Task, rendered non-executable planned panels with requirements, safety/result-routing copy, and related links, added guardrail coverage preventing Watch/Ingest capability gates from activating planned families, clarified future result routing in Results and Home copy, and added route-level deep-link coverage for both planned templates.

Verification recorded on 2026-06-09:
- `cd apps/packages/ui && bunx vitest run src/components/Option/ScheduledTasks/__tests__/scheduled-task-planned-template-copy.test.ts src/components/Option/ScheduledTasks/__tests__/scheduled-task-template-capabilities.test.ts src/components/Option/ScheduledTasks/__tests__/ScheduledTaskCreatePanel.test.tsx src/components/Option/ScheduledTasks/__tests__/ScheduledTaskResultsPanel.test.tsx src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx src/components/Option/CompanionHome/__tests__/AutomationInboxCard.test.tsx` passed with 6 files and 75 tests.
- `git diff --check` passed.
- Browser verification used Next dev server on `localhost:18002` plus a tiny local mock API on `127.0.0.1:8000` for health/openapi/scheduled-tasks. Verified Recurring Question and Agent Task planned shells render their planned copy, requirements/safety, links, and no create buttons. Verified `/scheduled-tasks/results` renders the future result-routing copy.
- Bandit: frontend TypeScript/TSX implementation does not produce Python code findings; previous focused Bandit scans of touched frontend paths reported `errors: []`, `results: []`, `loc: 0` where applicable. No backend code changed.

Known skip/blocker: hosted Home visual verification reached the Automation Inbox but displayed the existing partial-error state because the ad hoc mock API did not implement Companion Home snapshot endpoints. Home empty-state copy is covered by focused component tests.
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
