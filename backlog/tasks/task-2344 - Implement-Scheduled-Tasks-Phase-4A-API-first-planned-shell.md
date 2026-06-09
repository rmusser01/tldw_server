---
id: TASK-2344
title: Implement Scheduled Tasks Phase 4A API-first planned shell
status: In Progress
labels:
- scheduled-tasks
- ux
- frontend
priority: High
documentation:
- Docs/superpowers/specs/2026-06-09-scheduled-tasks-phase4-recurring-question-agent-task-api-contract-design.md
modified_files:
- apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTaskCreatePanel.tsx
- apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTaskCreatePanel.test.tsx
- apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/scheduled-task-template-capabilities.test.ts
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
