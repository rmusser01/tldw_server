---
id: TASK-452
title: Migrate PagesTab load error to design-system Alert
status: Done
assignee: []
created_date: ''
updated_date: 2026-05-20 03:48
labels:
- design-system
- product-state
- ui
dependencies: []
references:
- https://github.com/rmusser01/tldw_server/pull/1884
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Migrate the DocumentWorkspace PagesTab PDF load-error banner from AntD Alert to the canonical design-system Alert while preserving translated title and load-error message. Remove the matching baseline exception and verify the guard.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 PagesTab renders PDF load errors through the canonical design-system Alert primitive.
- [x] #2 The existing translated error title and load-error message fallback behavior are preserved.
- [x] #3 The PagesTab Alert baseline exception is removed without introducing new blocked product-state findings.
- [x] #4 Focused tests and design-system product-state verification pass, with known TypeScript/Bandit skips recorded if applicable.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented test-first: the PagesTab regression drives the mocked react-pdf load-error branch and failed on the missing canonical Alert marker before the AntD Alert was replaced with the design-system Alert.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated PagesTab's PDF load-error banner from AntD Alert to the canonical design-system Alert and added focused coverage for the DS wrapper plus translated title/message. Removed the PagesTab baseline entry, reducing product-state baseline exceptions from 331 to 330.

Verification:
- RED: bunx vitest run src/components/DocumentWorkspace/LeftSidebar/__tests__/PagesTab.test.tsx --reporter=dot failed on the missing data-ds-component="Alert" wrapper.
- GREEN: bunx vitest run src/components/DocumentWorkspace/LeftSidebar/__tests__/PagesTab.test.tsx --reporter=dot passed.
- bunx vitest run src/design-system/__tests__/product-state-guard.test.ts --reporter=dot passed: 52 tests.
- bun run verify:design-system-state passed; baseline exceptions are now 330.
- git diff --check passed.
- Full bunx tsc --noEmit --pretty false still exits 2 from inherited baseline debt; filtered touched-file diagnostics for PagesTab/task-452/baseline matched 0 lines.
- Bandit skipped because this slice changes TypeScript UI/test JSON task metadata only, with no Python code touched.
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
