---
id: TASK-454
title: Migrate QuickSaveWorkflow capture alert to design-system Alert
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-20 06:43'
labels:
  - design-system
  - product-state
  - ui
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1889'
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Migrate the QuickSaveWorkflow captured-content success notice from AntD Alert to the canonical design-system Alert while preserving title, captured page metadata, and preview behavior. Remove the matching baseline exception and verify the guard.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 QuickSaveWorkflow renders the capture success notice through the canonical design-system Alert primitive.
- [x] #2 The captured selection/page title, URL, and success title content remain visible after migration.
- [x] #3 The QuickSaveWorkflow Alert baseline exception is removed without introducing new blocked product-state findings.
- [x] #4 Focused tests and design-system product-state verification pass, with known TypeScript/Bandit skips recorded if applicable.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Implemented test-first: the QuickSaveWorkflow regression drives the mocked Chrome capture path and failed on the missing canonical Alert marker before replacing the AntD Alert.

Verification recorded for this slice:
- RED: bunx vitest run src/components/Common/Workflow/__tests__/QuickSaveWorkflow.product-state.test.tsx --reporter=dot failed on zero data-ds-component="Alert" markers before the implementation.
- GREEN: bunx vitest run src/components/Common/Workflow/__tests__/QuickSaveWorkflow.product-state.test.tsx --reporter=dot passed: 1 test.
- bunx vitest run src/components/Common/Workflow/__tests__/WizardShell.test.tsx src/components/Common/Workflow/__tests__/QuickSaveWorkflow.product-state.test.tsx --reporter=dot passed: 2 files, 2 tests.
- bunx vitest run src/design-system/__tests__/product-state-guard.test.ts --reporter=dot passed: 52 tests.
- bun run verify:design-system-state passed; baseline exceptions are now 327.
- git diff --check passed.
- Full bunx tsc --noEmit --pretty false still exits 2 from inherited baseline debt; filtered touched-file diagnostics for QuickSaveWorkflow/task-454/baseline matched 0 lines.
- Bandit skipped because this slice changes TypeScript UI/test, JSON baseline, and task metadata only; no Python code touched.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated QuickSaveWorkflow captured-content success notice from AntD Alert to the canonical design-system Alert. Added focused coverage for the mocked Chrome capture path and removed the QuickSaveWorkflow baseline entry, reducing product-state baseline exceptions from 328 to 327.

Verification:
- RED: bunx vitest run src/components/Common/Workflow/__tests__/QuickSaveWorkflow.product-state.test.tsx --reporter=dot failed on zero data-ds-component="Alert" markers.
- GREEN: bunx vitest run src/components/Common/Workflow/__tests__/QuickSaveWorkflow.product-state.test.tsx --reporter=dot passed: 1 test.
- bunx vitest run src/components/Common/Workflow/__tests__/WizardShell.test.tsx src/components/Common/Workflow/__tests__/QuickSaveWorkflow.product-state.test.tsx --reporter=dot passed: 2 files, 2 tests.
- bunx vitest run src/design-system/__tests__/product-state-guard.test.ts --reporter=dot passed: 52 tests.
- bun run verify:design-system-state passed; baseline exceptions are now 327.
- git diff --check passed.
- Full bunx tsc --noEmit --pretty false still exits 2 from inherited baseline debt; filtered touched-file diagnostics for QuickSaveWorkflow/task-454/baseline matched 0 lines.
- Bandit skipped because this slice changes TypeScript UI/test, JSON baseline, and task metadata only; no Python code touched.
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
