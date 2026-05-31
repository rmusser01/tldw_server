---
id: TASK-45.44.2.5
title: Migrate TablePreview alerts to design-system Alert
status: Done
labels:
- design-system
- webui
- extension
- product-state
priority: medium
parent_task_id: TASK-45.44.2
references:
- apps/packages/ui/src/components/Option/DataTables/TablePreview.tsx
- apps/packages/ui/src/components/Option/DataTables/__tests__/TablePreview.product-state.test.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
- https://github.com/rmusser01/tldw_server/issues/1659
- https://github.com/rmusser01/tldw_server/pull/1786
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue the Ingestion/Library/media product-state migration by replacing TablePreview generation error and warning AntD Alerts with the shared design-system Alert primitive, then remove the matching baseline exceptions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 TablePreview generation error alert renders through the shared design-system Alert primitive while preserving copy and retry behavior.
- [x] #2 TablePreview generation warning alert renders through the shared design-system Alert primitive while preserving the warning list content.
- [x] #3 The two TablePreview AntD Alert baseline entries are removed without introducing new guard findings.
- [x] #4 Focused Vitest, design-system guard verification, TypeScript touched-scope check, and diff whitespace checks are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Scope intentionally stayed limited to TablePreview generation error/warning alerts, their focused DOM regression test, and the matching guard baseline entries.
- PR: https://github.com/rmusser01/tldw_server/pull/1786
- The initial focused Vitest run failed as expected because the current AntD Alert markup did not provide `data-ds-component="Alert"`.
- Verification:
  - `bunx vitest run src/components/Option/DataTables/__tests__/TablePreview.product-state.test.tsx --maxWorkers=1 --no-file-parallelism` passed, 2 tests.
  - `bunx vitest run src/design-system/__tests__/product-state-guard.test.ts --maxWorkers=1 --no-file-parallelism` passed, 52 tests.
  - `bun run verify:design-system-state` passed and reported 394 baseline exceptions.
  - `node -e "const data=require('./apps/packages/ui/scripts/design-system-product-state-baseline.json'); console.log(data.length); console.log(JSON.stringify(data.filter(e=>e.path==='src/components/Option/DataTables/TablePreview.tsx'), null, 2));"` reported `394` and `[]`.
  - `git diff --check` passed.
  - `bunx tsc --noEmit --pretty false` still exits 2 on existing unrelated repo TypeScript errors; filtered output for `TablePreview`, the baseline file, and this task file had no matches.
  - Bandit skipped because the touched implementation files are UI TypeScript/JSON/Backlog task files only.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated TablePreview generation error and warning product-state alerts from direct AntD Alert usage to the shared design-system Alert primitive, added focused DOM coverage for both branches, and removed the two matching TablePreview product-state baseline exceptions. Baseline count moved from 396 to 394 total exceptions.
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
