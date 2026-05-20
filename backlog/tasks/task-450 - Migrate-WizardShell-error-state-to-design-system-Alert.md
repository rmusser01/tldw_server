---
id: TASK-450
title: Migrate WizardShell error state to design-system Alert
status: Done
assignee: []
created_date: ''
updated_date: 2026-05-20 02:22
labels:
- design-system
- product-state
- ui
dependencies: []
references:
- https://github.com/rmusser01/tldw_server/pull/1880
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Migrate the WizardShell workflow error banner from AntD Alert to the canonical design-system Alert while preserving translated title, error text, closable dismissal, and workflow navigation behavior. Remove the matching baseline exception and verify the guard.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 WizardShell renders workflow errors through the canonical design-system Alert primitive.
- [x] #2 The existing translated error title, error message, and close-to-clear behavior are preserved.
- [x] #3 The WizardShell Alert baseline exception is removed without introducing new blocked product-state findings.
- [x] #4 Focused tests and design-system product-state verification pass, with known TypeScript/Bandit skips recorded if applicable.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented test-first: the new WizardShell regression failed on the missing canonical Alert marker, then passed after replacing the AntD error banner with the design-system Alert and preserving setError(null) dismissal.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated WizardShell's workflow error banner from AntD Alert to the canonical design-system Alert and added focused coverage for the DS wrapper plus dismiss-to-clear behavior. Removed the WizardShell baseline entry, reducing product-state baseline exceptions from 333 to 332.

Verification:
- RED: bunx vitest run src/components/Common/Workflow/__tests__/WizardShell.test.tsx --reporter=dot failed on the missing data-ds-component="Alert" wrapper.
- GREEN: bunx vitest run src/components/Common/Workflow/__tests__/WizardShell.test.tsx --reporter=dot passed.
- bunx vitest run src/components/Common/Workflow/__tests__ --reporter=dot passed: 4 files, 9 tests; jsdom emitted existing CSS parse warnings.
- bunx vitest run src/design-system/__tests__/product-state-guard.test.ts --reporter=dot passed: 52 tests.
- bun run verify:design-system-state passed; baseline exceptions are now 332.
- git diff --check passed.
- Full bunx tsc --noEmit --pretty false still exits 2 from inherited baseline debt; filtered touched-file diagnostics for WizardShell/task-450/baseline matched 0 lines.
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
