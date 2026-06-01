---
id: TASK-45.44.2.7
title: Migrate remaining DataTables alerts to design-system Alert
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-31 16:46'
labels:
  - design-system
  - webui
  - extension
  - product-state
dependencies: []
references:
  - apps/packages/ui/src/components/Option/DataTables/CreateTableWizard.tsx
  - apps/packages/ui/src/components/Option/DataTables/SaveTablePanel.tsx
  - apps/packages/ui/src/components/Option/DataTables/TableDetailModal.tsx
  - apps/packages/ui/scripts/design-system-product-state-baseline.json
  - 'https://github.com/rmusser01/tldw_server/issues/1659'
parent_task_id: TASK-45.44.2
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue the Ingestion/Library/media product-state migration by replacing the remaining DataTables AntD Alert product-state callouts in CreateTableWizard, SaveTablePanel, and TableDetailModal with the shared design-system Alert primitive, then remove the matching baseline exceptions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 CreateTableWizard, SaveTablePanel, and TableDetailModal product-state callouts render through the shared design-system Alert primitive while preserving user-facing copy/actions.
- [x] #2 The remaining DataTables AntD Alert product-state baseline entries for the touched files are removed without introducing new unbaselined findings for those paths.
- [x] #3 Focused regression coverage verifies the design-system Alert marker for the migrated branches.
- [x] #4 Focused tests, scoped product-state guard verification, TypeScript/touched-scope check, diff whitespace check, and Bandit skip rationale are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added focused design-system Alert regression tests for the CreateTableWizard source-selection tip, SaveTablePanel missing-table warning, and TableDetailModal load-error state. The initial focused Vitest run failed as expected because the existing AntD Alert markup did not expose data-ds-component="Alert".
- Replaced the three remaining DataTables AntD Alert product-state callouts with the shared design-system Alert primitive while preserving existing titles, body copy, and state severity.
- Removed the three matching DataTables baseline exceptions; baseline count moved from 82 to 79 and the touched paths now have zero baseline entries.
- Verification: focused DataTables Alert tests passed 3/3; product-state guard unit passed 54/54; bun run verify:design-system-state exited 0 with 79 baseline exceptions; node --max-old-space-size=8192 ./node_modules/typescript/bin/tsc --noEmit --pretty false exited 0; git diff --check exited 0.
- Bandit skipped because this slice touched frontend TSX/test/JSON/Backlog markdown only, with no Python code.

- PR review follow-up: extracted the repeated DataTables design-system Alert ancestor assertion into src/test-utils/designSystemAlert.ts with sync and async helpers, then updated the three new DataTables alert tests to import the shared helper without changing assertion semantics.
- Review-fix verification: focused DataTables Alert tests passed 3/3; node --max-old-space-size=8192 ./node_modules/typescript/bin/tsc --noEmit --pretty false exited 0; git diff --check exited 0.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated the remaining DataTables product-state alerts in CreateTableWizard, SaveTablePanel, and TableDetailModal from direct AntD Alert usage to the shared design-system Alert primitive. Added focused DOM coverage for all three migrated branches and removed the obsolete DataTables baseline exceptions, reducing the product-state baseline from 82 to 79 entries.
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
