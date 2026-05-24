---
id: TASK-45.44.3.12
title: Migrate WatchlistSetupWizard alerts to design-system Alert
status: Done
assignee: []
created_date: '2026-05-24 02:39'
updated_date: '2026-05-24 02:39'
labels:
  - design-system
  - webui
  - watchlists
  - product-state
dependencies: []
parent_task_id: TASK-45.44.3
priority: medium
references:
  - https://github.com/rmusser01/tldw_server/pull/2039
modified_files:
  - apps/packages/ui/src/components/Option/Watchlists/SetupWizard/WatchlistSetupWizard.tsx
  - apps/packages/ui/src/components/Option/Watchlists/SetupWizard/__tests__/WatchlistSetupWizard.test.tsx
  - apps/packages/ui/scripts/design-system-product-state-baseline.json
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue TASK-45.44.3 by replacing Watchlists SetupWizard AntD Alert callouts with the shared design-system Alert primitive, preserving copy and wizard behavior, removing migrated baseline exceptions, and recording focused verification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 WatchlistSetupWizard info and error callouts render via design-system Alert.
- [x] #2 The WatchlistSetupWizard Alert baseline exceptions are removed from design-system-product-state-baseline.json.
- [x] #3 Focused WatchlistSetupWizard coverage asserts the design-system Alert marker for the migrated callouts.
- [x] #4 Design-system product-state verification passes or records existing unrelated blockers.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect WatchlistSetupWizard alert callouts and existing tests, then add focused assertions requiring `[data-ds-component="Alert"]` for both migrated callout paths.
2. Migrate WatchlistSetupWizard Alert usage from AntD props to the design-system Alert primitive while preserving copy and wizard behavior.
3. Remove WatchlistSetupWizard Alert exceptions from the product-state baseline and run focused Vitest plus design-system verification.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- RED: focused WatchlistSetupWizard test failed on both new `[data-ds-component="Alert"]` assertions because the collection guidance and validation error still rendered through the AntD Alert mock.
- Migrated WatchlistSetupWizard collection-scope guidance and validation error callouts from AntD Alert props to the shared design-system Alert primitive while preserving copy and wizard behavior.
- Removed the two WatchlistSetupWizard Alert entries from the product-state baseline.
- Verification: `bunx vitest run src/components/Option/Watchlists/SetupWizard/__tests__/WatchlistSetupWizard.test.tsx --reporter=dot` passed 5 tests.
- Verification: `bunx vitest run src/components/Option/Watchlists/SetupWizard/__tests__/WatchlistSetupWizard.test.tsx src/components/Option/Watchlists/SetupWizard/__tests__/watchlist-setup-model.test.ts --reporter=dot` passed 10 tests across 2 files.
- Verification: `bunx vitest run src/design-system/__tests__/product-state-guard.test.ts --reporter=dot` passed 54 tests.
- Verification: `bun run verify:design-system-state` passed with 247 total exceptions and 14 Jobs/Scheduler/Watchlists exceptions.
- Verification: WatchlistSetupWizard baseline rows are 0 and total baseline rows are 247.
- Verification: `git diff --check` passed.
- TypeScript: `NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false` exits 2 with 347 existing diagnostics; no diagnostics mention WatchlistSetupWizard, its tests, the product-state baseline, or this task.
- Bandit skipped: UI-only TypeScript/JSON/backlog changes; no Python touched.
- PR: https://github.com/rmusser01/tldw_server/pull/2039
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated the WatchlistSetupWizard collection-scope guidance and validation error callouts to the shared design-system Alert primitive, added focused coverage for both migrated paths, and removed the two obsolete WatchlistSetupWizard product-state baseline exceptions. Focused SetupWizard and design-system guard verification passed; the full product-state verifier now reports 247 total baseline exceptions and 14 Jobs/Scheduler/Watchlists exceptions. TypeScript remains blocked by existing unrelated repo-wide diagnostics, with none in this slice.
<!-- SECTION:FINAL_SUMMARY:END -->

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
