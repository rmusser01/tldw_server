---
id: TASK-45.44.3.11
title: Migrate TemplateEditor warning alerts to design-system Alert
status: Done
assignee: []
created_date: ''
updated_date: 2026-05-24 02:15
labels:
- design-system
- webui
- watchlists
- product-state
dependencies: []
parent_task_id: TASK-45.44.3
priority: medium
references:
- https://github.com/rmusser01/tldw_server/pull/2037
modified_files:
- apps/packages/ui/src/components/Option/Watchlists/TemplatesTab/TemplateEditor.tsx
- apps/packages/ui/src/components/Option/Watchlists/TemplatesTab/__tests__/TemplateEditor.mode-contract.test.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue TASK-45.44.3 by replacing Watchlists TemplateEditor AntD Alert warning callouts with the shared design-system Alert primitive, preserving warning copy and layout, removing migrated baseline exceptions, and recording focused verification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 TemplateEditor warning callouts render via design-system Alert.
- [x] #2 The TemplateEditor Alert baseline exceptions are removed from design-system-product-state-baseline.json.
- [x] #3 Focused TemplateEditor coverage asserts the design-system Alert marker for the migrated warning paths.
- [x] #4 Design-system product-state verification passes or records existing unrelated blockers.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect TemplateEditor warning callouts and existing tests, then add focused assertions requiring `[data-ds-component="Alert"]` for both warning paths.
2. Migrate TemplateEditor Alert usage from AntD props to the design-system Alert primitive while preserving copy and layout.
3. Remove TemplateEditor Alert exceptions from the product-state baseline and run focused Vitest plus design-system verification.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- RED: focused TemplateEditor mode-contract test failed because the drift/repair warning content did not have a `[data-ds-component="Alert"]` ancestor.
- Migrated the TemplateEditor version-drift and visual-repair warnings from AntD Alert props to the shared design-system Alert primitive while preserving warning copy and the repair action.
- Removed the two TemplateEditor Alert entries from the product-state baseline.
- Verification: `bunx vitest run src/components/Option/Watchlists/TemplatesTab/__tests__/TemplateEditor.mode-contract.test.tsx --reporter=dot` passed 13 tests.
- Verification: `bunx vitest run src/design-system/__tests__/product-state-guard.test.ts --reporter=dot` passed 54 tests.
- Verification: `bun run verify:design-system-state` passed with 249 total exceptions and 16 Jobs/Scheduler/Watchlists exceptions.
- Verification: TemplateEditor baseline rows are 0 and total baseline rows are 249.
- Verification: `git diff --check` passed.
- TypeScript: `NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false` exits 2 with 347 existing diagnostics; no diagnostics mention TemplateEditor, its mode-contract test, the product-state baseline, or this task.
- Bandit skipped: UI-only TypeScript/JSON/backlog changes; no Python touched.
- PR: https://github.com/rmusser01/tldw_server/pull/2037
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated the remaining TemplateEditor warning Alert callouts to the shared design-system Alert primitive, added focused coverage for the migrated warning paths, and removed the two obsolete TemplateEditor product-state baseline exceptions. Focused UI and design-system guard verification passed; the full product-state verifier now reports 249 total baseline exceptions and 16 Jobs/Scheduler/Watchlists exceptions. TypeScript remains blocked by existing unrelated repo-wide diagnostics, with none in this slice.
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
