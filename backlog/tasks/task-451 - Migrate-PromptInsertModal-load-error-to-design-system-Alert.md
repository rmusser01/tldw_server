---
id: TASK-451
title: Migrate PromptInsertModal load error to design-system Alert
status: Done
assignee: []
created_date: ''
updated_date: 2026-05-20 03:15
labels:
- design-system
- product-state
- ui
dependencies: []
references:
- https://github.com/rmusser01/tldw_server/pull/1883
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Migrate the PromptInsertModal prompt-load error banner from AntD Alert to the canonical design-system Alert while preserving translated title and server/error fallback description. Remove the matching baseline exception and verify the guard.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 PromptInsertModal renders prompt-load errors through the canonical design-system Alert primitive.
- [x] #2 The existing translated error title and fallback/error message behavior are preserved.
- [x] #3 The PromptInsertModal Alert baseline exception is removed without introducing new blocked product-state findings.
- [x] #4 Focused tests and design-system product-state verification pass, with known TypeScript/Bandit skips recorded if applicable.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented test-first: the new PromptInsertModal regression failed on the missing canonical Alert marker, then passed after replacing the AntD prompt-load error banner with the design-system Alert and preserving the error title/description path.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated PromptInsertModal's prompt-load error banner from AntD Alert to the canonical design-system Alert, added focused coverage for the DS wrapper plus translated title, and addressed PR #1883 review feedback by adding a Retry action that calls React Query refetch. Removed the PromptInsertModal baseline entry, reducing product-state baseline exceptions from 332 to 331.

Verification:
- RED: bunx vitest run src/components/Common/__tests__/PromptInsertModal.test.tsx --reporter=dot failed on the missing data-ds-component="Alert" wrapper before the migration.
- RED review fix: bunx vitest run src/components/Common/__tests__/PromptInsertModal.test.tsx --reporter=dot failed on the missing Retry button before wiring refetch.
- GREEN: bunx vitest run src/components/Common/__tests__/PromptInsertModal.test.tsx --reporter=dot passed: 2 tests.
- bunx vitest run src/design-system/__tests__/product-state-guard.test.ts --reporter=dot passed: 52 tests.
- bun run verify:design-system-state passed; baseline exceptions are now 331.
- git diff --check passed.
- Full bunx tsc --noEmit --pretty false still exits 2 from inherited baseline debt; filtered touched-file diagnostics for PromptInsertModal/task-451/baseline matched 0 lines.
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
