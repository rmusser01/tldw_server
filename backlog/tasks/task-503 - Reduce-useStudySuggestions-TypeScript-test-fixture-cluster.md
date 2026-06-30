---
id: TASK-503
title: Reduce useStudySuggestions TypeScript test fixture cluster
status: Done
references:
- TASK-502
- apps/packages/ui/src/components/StudySuggestions/hooks/__tests__/useStudySuggestions.test.tsx
- apps/packages/ui/src/services/studySuggestions.ts
- apps/packages/ui/tsconfig.json
modified_files:
- apps/packages/ui/src/components/StudySuggestions/hooks/__tests__/useStudySuggestions.test.tsx
- backlog/tasks/task-503 - Reduce-useStudySuggestions-TypeScript-test-fixture-cluster.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue reducing the shared UI package-wide TypeScript compiler baseline by fixing the contained useStudySuggestions test fixture typing cluster. Current package `tsc` output reports four errors in `src/components/StudySuggestions/hooks/__tests__/useStudySuggestions.test.tsx`, all around uncontextualized snapshot response fixtures and a mock implementation signature.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Current useStudySuggestions compiler diagnostics are captured.
- [x] #2 Root cause is documented and tied to test fixture typing rather than production behavior.
- [x] #3 The `useStudySuggestions.test.tsx` compiler cluster is removed from package `tsc` output.
- [x] #4 Focused behavior test is run or an explicit blocker is recorded.
- [x] #5 Remaining package-wide `tsc` baseline count is recorded.
- [x] #6 Bandit decision is recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Red compiler evidence came from `/tmp/task502-tsc-final.txt`, which contained four `useStudySuggestions.test.tsx` diagnostics around snapshot fixture assignability and delayed replacement snapshot resolver typing.
- Root cause was test-only fixture typing: `buildSnapshot` and `buildSnapshotV2` returned uncontextualized object literals, so fields such as `anchor_type` widened to `string` instead of the service response unions. Production hook behavior was not changed.
- Annotated the snapshot fixture builders, the delayed replacement resolver, and the delayed `Promise` with `StudySuggestionSnapshotResponse`.
- Focused test: `bunx vitest run src/components/StudySuggestions/hooks/__tests__/useStudySuggestions.test.tsx` from `apps/packages/ui` passed 4/4.
- Package compiler capture: `bunx tsc --noEmit --pretty false > /tmp/task503-tsc-final.txt 2>&1` from `apps/packages/ui` still exits 2 for the known baseline, but `error TS` lines reduced from 104 to 100 and `rg -n 'StudySuggestions/hooks/__tests__/useStudySuggestions' /tmp/task503-tsc-final.txt` returned no matches.
- Bandit skipped: this is a TypeScript test-only change and Bandit is a Python security scanner; no Python touched scope exists for this task.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Removed the four-error `useStudySuggestions.test.tsx` package `tsc` cluster by contextually typing local snapshot response fixtures. The shared UI baseline is now 100 `error TS` lines after this slice.
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
