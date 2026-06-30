---
id: TASK-45.44.3.14
title: Close remaining Jobs Scheduler Watchlists product-state exceptions
status: Done
labels:
- design-system
- webui
- product-state
priority: medium
parent_task_id: TASK-45.44.3
references:
- https://github.com/rmusser01/tldw_server/issues/1660
- https://github.com/rmusser01/tldw_server/pull/2044
- apps/packages/ui/src/components/Common/Workflow/steps/AnalyzeBookWorkflow.tsx
- apps/packages/ui/src/components/Option/AgentTasks/index.tsx
- apps/packages/ui/src/components/Common/Workflow/__tests__/AnalyzeBookWorkflow.product-state.test.tsx
- apps/packages/ui/src/components/Option/AgentTasks/__tests__/AgentTasksPage.connection.test.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
modified_files:
- apps/packages/ui/src/components/Common/Workflow/steps/AnalyzeBookWorkflow.tsx
- apps/packages/ui/src/components/Option/AgentTasks/index.tsx
- apps/packages/ui/src/components/Common/Workflow/__tests__/AnalyzeBookWorkflow.product-state.test.tsx
- apps/packages/ui/src/components/Option/AgentTasks/__tests__/AgentTasksPage.connection.test.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address PR 2044 review feedback by migrating the remaining Jobs/Scheduler/Watchlists path-map product-state exceptions in Common Workflow and AgentTasks to design-system primitives, then removing the remaining baseline rows so the area reaches zero exceptions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 AnalyzeBookWorkflow product-state alerts render through the design-system Alert primitive.
- [x] #2 AgentTasks setup/error alerts render through the design-system Alert primitive and running/triage task state labels render through the design-system Badge primitive.
- [x] #3 The remaining AnalyzeBookWorkflow and AgentTasks product-state baseline rows are removed.
- [x] #4 Product-state guard verification reports zero Jobs/Scheduler/Watchlists area exceptions.
- [x] #5 Focused regression coverage records the migrated design-system primitive usage.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Replaced the three AnalyzeBookWorkflow AntD Alert product-state callouts with `@/components/ui/primitives/Alert`.
- Replaced AgentTasks unsupported/setup/workspace/error AntD Alert callouts with the design-system Alert primitive.
- Replaced the AgentTasks running and triage task-state AntD Tags with design-system Badge.
- Removed the three `AnalyzeBookWorkflow.tsx` and six `Option/AgentTasks/index.tsx` baseline rows.
- Verification:
  - PASS: `bunx vitest run src/components/Option/Watchlists/SettingsTab/__tests__/SettingsTab.help.test.tsx src/components/Option/AgentTasks/__tests__/AgentTasksPage.connection.test.tsx src/components/Common/Workflow/__tests__/AnalyzeBookWorkflow.product-state.test.tsx --reporter=dot` -> 3 files passed, 19 tests passed.
  - PASS: `bun run verify:design-system-state` -> 233 total baseline exceptions and no Jobs/Scheduler/Watchlists product-area entry remains.
  - PASS: `git diff --check`.
  - BLOCKED by existing unrelated TypeScript debt: `NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false` exits 2. Captured output in `/tmp/pr2044-tsc.log`; `rg -n "AnalyzeBookWorkflow|AgentTasks|SettingsTab|design-system-product-state-baseline|45\\.44\\.3\\.14" /tmp/pr2044-tsc.log` returns no diagnostics.
  - SKIPPED: Bandit, because this slice touches frontend TypeScript/TSX, JSON baseline metadata, and Backlog markdown only.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Closed the remaining Jobs/Scheduler/Watchlists product-state exceptions by migrating AnalyzeBookWorkflow and AgentTasks product-state UI to design-system primitives, adding focused regression coverage, and reducing the owned product-area baseline count from 9 to 0. The design-system guard now reports 233 total legacy exceptions with no Jobs/Scheduler/Watchlists area remaining.
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
