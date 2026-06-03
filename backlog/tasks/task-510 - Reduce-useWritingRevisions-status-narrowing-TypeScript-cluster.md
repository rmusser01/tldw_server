---
id: TASK-510
title: Reduce useWritingRevisions status narrowing TypeScript cluster
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-02 15:49'
labels: []
dependencies: []
references:
  - TASK-509
  - >-
    apps/packages/ui/src/components/Option/WritingPlayground/hooks/useWritingRevisions.ts
  - >-
    apps/packages/ui/src/components/Option/WritingPlayground/writing-revision-types.ts
  - >-
    apps/packages/ui/src/components/Option/WritingPlayground/__tests__/useWritingRevisions.test.tsx
  - apps/packages/ui/tsconfig.json
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue reducing the shared UI package-wide TypeScript compiler baseline by fixing the contained useWritingRevisions type narrowing cluster. Current package `tsc` output reports three errors in `src/components/Option/WritingPlayground/hooks/useWritingRevisions.ts` around failed apply result narrowing and widened revision `status` literals in regenerated proposal updates.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Current useWritingRevisions compiler diagnostics are captured.
- [x] #2 Root cause is documented and tied to TypeScript narrowing/literal widening rather than behavior changes.
- [x] #3 The `useWritingRevisions.ts` compiler cluster is removed from package `tsc` output.
- [x] #4 Focused useWritingRevisions test is run or an explicit blocker is recorded.
- [x] #5 Remaining package-wide `tsc` baseline count is recorded.
- [x] #6 Bandit decision is recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Captured red evidence from `/tmp/task509-tsc-final.txt`: package `tsc` reported three diagnostics in `src/components/Option/WritingPlayground/hooks/useWritingRevisions.ts` around `ApplyEditorTextResult.reason` narrowing and widened `status` literals in regenerated revision updates.
- Root cause was TypeScript type narrowing/literal widening only. The apply-result branch now checks `result.applied === false` before reading `reason`, preserving the same applied/conflict behavior while making the discriminant explicit for the compiler.
- Regenerated revision updates now keep map output typed as `WritingRevisionProposal` and validate the replacement object with `satisfies WritingRevisionProposal` so `status: "rejected"` and `status: "pending"` remain literal proposal statuses.
- Focused verification attempted with `bunx vitest run src/components/Option/WritingPlayground/__tests__/useWritingRevisions.test.tsx`; it failed before assertions with `SecurityError: localStorage is not available for opaque origins`, and all 11 tests were skipped. This appears to be a pre-existing test-environment issue from the suite's opaque-origin JSDOM setup, not this hook change.
- Package verification: `bunx tsc --noEmit --pretty false > /tmp/task510-tsc-final.txt 2>&1` still exits nonzero from the known baseline, but diagnostics dropped from 82 in `/tmp/task509-tsc-final.txt` to 79 in `/tmp/task510-tsc-final.txt`; `rg -n 'useWritingRevisions\.ts' /tmp/task510-tsc-final.txt` returns no matches.
- Bandit skipped: this is a TypeScript-only WebUI change with no Python touched.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Removed the `useWritingRevisions.ts` TypeScript cluster by making apply-result false-branch narrowing explicit and preserving regenerated revision status literals as `WritingRevisionProposal` values. Package `tsc` baseline is now 79 errors, down from 82 after TASK-509, with no remaining `useWritingRevisions.ts` diagnostics. Focused Vitest remains blocked by the suite's opaque-origin `localStorage` setup failure.
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
