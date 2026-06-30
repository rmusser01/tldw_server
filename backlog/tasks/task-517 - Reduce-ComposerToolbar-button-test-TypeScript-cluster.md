---
id: TASK-517
title: Reduce ComposerToolbar button test TypeScript cluster
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-02 19:23'
labels: []
dependencies: []
references:
  - TASK-516
  - >-
    apps/packages/ui/src/components/Option/Playground/__tests__/ComposerToolbar.test.tsx
  - apps/packages/ui/tsconfig.json
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue reducing the shared UI package-wide TypeScript compiler baseline by fixing the contained diagnostics in `src/components/Option/Playground/__tests__/ComposerToolbar.test.tsx`. Current package `tsc` output reports two test-only errors where generic `HTMLElement` test queries are passed to `HTMLButtonElement` array lookup from `querySelectorAll("button")`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Current ComposerToolbar compiler diagnostics are captured.
- [x] #2 Root cause is documented and tied to test element narrowing rather than behavior changes.
- [x] #3 The `ComposerToolbar.test.tsx` compiler cluster is removed from package `tsc` output.
- [x] #4 Focused ComposerToolbar test is run or an explicit blocker is recorded.
- [x] #5 Remaining package-wide `tsc` baseline count is recorded.
- [x] #6 Bandit decision is recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Captured red evidence from `/tmp/task516-tsc-final.txt`: package `tsc` reported two diagnostics in `src/components/Option/Playground/__tests__/ComposerToolbar.test.tsx` where `HTMLElement` results from `screen.getByTestId` were passed to `Array<HTMLButtonElement>.indexOf`.
- Root cause was test element narrowing only. The surrounding `contextStrip.querySelectorAll("button")` result is a `NodeListOf<HTMLButtonElement>`, while Testing Library returns generic `HTMLElement` for test-id queries.
- Narrowed the saved and advanced test-id elements to `HTMLButtonElement` before comparing their positions in the button list.
- Focused verification: `bunx vitest run src/components/Option/Playground/__tests__/ComposerToolbar.test.tsx` passed: 22 tests.
- Package verification: `bunx tsc --noEmit --pretty false > /tmp/task517-tsc-final.txt 2>&1` still exits nonzero from the known baseline, but diagnostics dropped from 65 in `/tmp/task516-tsc-final.txt` to 63 in `/tmp/task517-tsc-final.txt`; `rg -n 'ComposerToolbar\.test\.tsx' /tmp/task517-tsc-final.txt` returns no matches.
- Bandit skipped: this is a TypeScript test-only WebUI change with no Python touched.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Removed the `ComposerToolbar.test.tsx` TypeScript cluster by narrowing the two test-id button elements to `HTMLButtonElement` before comparing them with the `querySelectorAll("button")` result. Focused Vitest passed with 22 tests, and package `tsc` baseline dropped from 65 to 63 with no remaining ComposerToolbar diagnostics.
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
