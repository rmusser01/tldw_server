---
id: TASK-45.44.3.13
title: Migrate Watchlists SettingsTab alerts to design-system Alert
status: Done
labels:
- design-system
- webui
- product-state
priority: medium
parent_task_id: TASK-45.44.3
references:
- https://github.com/rmusser01/tldw_server/issues/1660
- apps/packages/ui/src/components/Option/Watchlists/SettingsTab/SettingsTab.tsx
- apps/packages/ui/src/components/Option/Watchlists/SettingsTab/__tests__/SettingsTab.help.test.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
- https://github.com/rmusser01/tldw_server/pull/2044
modified_files:
- apps/packages/ui/src/components/Option/Watchlists/SettingsTab/SettingsTab.tsx
- apps/packages/ui/src/components/Option/Watchlists/SettingsTab/__tests__/SettingsTab.help.test.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue TASK-45.44.3 by replacing Watchlists SettingsTab AntD Alert product-state callouts with the shared design-system Alert primitive, preserving settings, diagnostics, cluster subscription, and unavailable copy while removing migrated baseline exceptions and recording focused verification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 SettingsTab no longer imports or renders AntD Alert for product-state callouts.
- [x] #2 SettingsTab guidance, diagnostics, cluster subscription, cluster error, and unavailable callouts render through the design-system Alert primitive.
- [x] #3 The SettingsTab product-state baseline exceptions are removed and guard verification records the updated counts.
- [x] #4 Focused SettingsTab regression coverage verifies design-system Alert usage.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Replaced the five SettingsTab AntD Alert callouts with `@/components/ui/primitives/Alert` while preserving the existing TTL note, internal diagnostics note, cluster monitor selection prompt, cluster load warning, and settings unavailable copy.
- Added focused SettingsTab coverage that asserts the default guidance callouts render with `data-ds-component="Alert"`.
- Added a file-local `localStorage` shim for this test file because the current Vitest runtime did not provide a global `localStorage`; this also stabilizes the existing onboarding-path assertions in the file.
- Review follow-up: clear the singleton localStorage shim in `beforeEach` so one test cannot leak stored onboarding state into another test.
- Removed the five `src/components/Option/Watchlists/SettingsTab/SettingsTab.tsx` entries from `design-system-product-state-baseline.json`.
- Verification:
  - PASS: `bunx vitest run src/components/Option/Watchlists/SettingsTab/__tests__/SettingsTab.help.test.tsx --reporter=dot` -> 6 passed.
  - PASS after review follow-up: `bunx vitest run src/components/Option/Watchlists/SettingsTab/__tests__/SettingsTab.help.test.tsx src/components/Option/AgentTasks/__tests__/AgentTasksPage.connection.test.tsx src/components/Common/Workflow/__tests__/AnalyzeBookWorkflow.product-state.test.tsx --reporter=dot` -> 3 files passed, 19 tests passed.
  - PASS: `bun run verify:design-system-state` -> 242 total baseline exceptions; Jobs/Scheduler/Watchlists area down to 9.
  - PASS after TASK-45.44.3.14 follow-up: `bun run verify:design-system-state` -> 233 total baseline exceptions; Jobs/Scheduler/Watchlists area down to 0.
  - BLOCKED by existing unrelated debt: `NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false` exits 2 on existing diagnostics; no diagnostics mention `SettingsTab`, `SettingsTab.help.test.tsx`, the baseline, or `TASK-45.44.3.13`.
  - SKIPPED: Bandit, because this slice touches frontend TypeScript/JSON and Backlog metadata only.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated Watchlists SettingsTab product-state alerts to the shared design-system Alert primitive, added focused regression coverage for the default design-system guidance callouts, stabilized and cleared the test file's localStorage dependency, and removed the five migrated SettingsTab baseline exceptions. TASK-45.44.3.14 subsequently closed the remaining area exceptions, so the product-state guard now reports 233 total baseline exceptions and 0 remaining Jobs/Scheduler/Watchlists exceptions.
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
