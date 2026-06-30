---
id: TASK-520
title: Reduce ResearchWorkspace stage13 translation test TypeScript cluster
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-02 19:38'
labels: []
dependencies: []
references:
  - TASK-519
  - >-
    apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/ResearchWorkspace.stage13.source-transfer.test.tsx
  - apps/packages/ui/tsconfig.json
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue reducing the shared UI package-wide TypeScript compiler baseline by fixing the contained diagnostics in `src/components/Option/ResearchWorkspace/__tests__/ResearchWorkspace.stage13.source-transfer.test.tsx`. Current package `tsc` output reports two test translation mock errors where `count` and `workspaceName` are read from a parameter that can be a string fallback.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Current ResearchWorkspace stage13 compiler diagnostics are captured.
- [x] #2 Root cause is documented and tied to translation mock option narrowing rather than behavior changes.
- [x] #3 The `ResearchWorkspace.stage13.source-transfer.test.tsx` compiler cluster is removed from package `tsc` output.
- [x] #4 Focused ResearchWorkspace stage13 source-transfer test is run or an explicit blocker is recorded.
- [x] #5 Remaining package-wide `tsc` baseline count is recorded.
- [x] #6 Bandit decision is recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Captured red evidence from `/tmp/task519-tsc-final.txt`: package `tsc` reported two diagnostics in `src/components/Option/ResearchWorkspace/__tests__/ResearchWorkspace.stage13.source-transfer.test.tsx` because the translation mock read `count` and `workspaceName` from a parameter typed as string-or-options.
- Root cause was translation mock narrowing only. The mock supports both string fallback and object options, but it needed a local narrowed `options` value before reading option-only interpolation fields.
- Added an `options` local that is undefined for string fallbacks and used it for `defaultValue`, `count`, and `workspaceName` lookups.
- Focused verification: `bunx vitest run src/components/Option/ResearchWorkspace/__tests__/ResearchWorkspace.stage13.source-transfer.test.tsx` passed: 9 tests.
- Package verification: `bunx tsc --noEmit --pretty false > /tmp/task520-tsc-final.txt 2>&1` still exits nonzero from the known baseline, but diagnostics dropped from 59 in `/tmp/task519-tsc-final.txt` to 57 in `/tmp/task520-tsc-final.txt`; `rg -n 'ResearchWorkspace\.stage13\.source-transfer\.test\.tsx' /tmp/task520-tsc-final.txt` returns no matches.
- Bandit skipped: this is a TypeScript test-only WebUI change with no Python touched.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Removed the `ResearchWorkspace.stage13.source-transfer.test.tsx` TypeScript cluster by narrowing the translation mock's fallback/options parameter before reading `count` and `workspaceName`. Focused Vitest passed with 9 tests, and package `tsc` baseline dropped from 59 to 57 with no remaining stage13 source-transfer diagnostics.
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
