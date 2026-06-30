---
id: TASK-507
title: Reduce WorkspaceStatusBar mock state TypeScript cluster
status: Done
references:
- TASK-506
- apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/WorkspaceStatusBar.test.tsx
- apps/packages/ui/src/types/connection.ts
- apps/packages/ui/tsconfig.json
modified_files:
- apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/WorkspaceStatusBar.test.tsx
- backlog/tasks/task-507 - Reduce-WorkspaceStatusBar-mock-state-TypeScript-cluster.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue reducing the shared UI package-wide TypeScript compiler baseline by fixing the contained WorkspaceStatusBar test mock state typing cluster. Current package `tsc` output reports three errors in `src/components/Option/ResearchWorkspace/__tests__/WorkspaceStatusBar.test.tsx` because the mock connection state is inferred with literal-only `errorKind` and `configStep` values.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Current WorkspaceStatusBar compiler diagnostics are captured.
- [x] #2 Root cause is documented and tied to test mock state typing rather than production behavior.
- [x] #3 The `WorkspaceStatusBar.test.tsx` compiler cluster is removed from package `tsc` output.
- [x] #4 Focused behavior test is run or an explicit blocker is recorded.
- [x] #5 Remaining package-wide `tsc` baseline count is recorded.
- [x] #6 Bandit decision is recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Red compiler evidence came from `/tmp/task506-tsc-final.txt`, which contained three `WorkspaceStatusBar.test.tsx` diagnostics because the mock connection state inferred `errorKind` and `configStep` from their initial `"none"` values.
- Root cause was test-only mock state typing. Production WorkspaceStatusBar behavior was not changed.
- Typed the mock store state as `ConnectionState`, and corrected the auth-error fixture from the derived UX label `"error_auth"` to the underlying `ConnectionState["errorKind"]` value `"auth"`.
- Focused test: `bunx vitest run src/components/Option/ResearchWorkspace/__tests__/WorkspaceStatusBar.test.tsx` from `apps/packages/ui` passed 3/3.
- Package compiler capture: `bunx tsc --noEmit --pretty false > /tmp/task507-tsc-final.txt 2>&1` from `apps/packages/ui` still exits 2 for the known baseline, but `error TS` lines reduced from 90 to 87 and `rg -n 'WorkspaceStatusBar\.test' /tmp/task507-tsc-final.txt` returned no matches.
- Bandit skipped: this is a TypeScript test-only change and Bandit is a Python security scanner; no Python touched scope exists for this task.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Removed the three-error `WorkspaceStatusBar.test.tsx` package `tsc` cluster by typing the mock connection store state and aligning its auth-error fixture with `ConnectionState`. The shared UI baseline is now 87 `error TS` lines after this slice.
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
