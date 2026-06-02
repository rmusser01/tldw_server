---
id: TASK-504
title: Reduce Characters manager updateCharacter mock call TypeScript cluster
status: Done
references:
- TASK-503
- apps/packages/ui/src/components/Option/Characters/__tests__/Manager.first-use.test.tsx
- apps/packages/ui/tsconfig.json
modified_files:
- apps/packages/ui/src/components/Option/Characters/__tests__/Manager.first-use.test.tsx
- backlog/tasks/task-504 - Reduce-Characters-manager-updateCharacter-mock-call-TypeScript-cluster.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue reducing the shared UI package-wide TypeScript compiler baseline by fixing the contained Characters manager first-use test mock call tuple cluster. Current package `tsc` output reports four errors in `src/components/Option/Characters/__tests__/Manager.first-use.test.tsx` where `updateCharacter.mock.calls.at(-1)` is inferred as an empty tuple.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Current Characters manager compiler diagnostics are captured.
- [x] #2 Root cause is documented and tied to test mock function typing rather than production behavior.
- [x] #3 The `Manager.first-use.test.tsx` updateCharacter mock call tuple cluster is removed from package `tsc` output.
- [x] #4 Focused behavior test is run or an explicit blocker is recorded.
- [x] #5 Remaining package-wide `tsc` baseline count is recorded.
- [x] #6 Bandit decision is recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Red compiler evidence came from `/tmp/task503-tsc-final.txt`, which contained four `Manager.first-use.test.tsx` diagnostics where `latestCall?.[0]` and `latestCall?.[1]` indexed into an empty tuple inferred from `updateCharacter: vi.fn(async () => ({}))`.
- Root cause was test-only mock function typing. The production Characters manager behavior was not changed.
- Added `updateCharacter` mock parameters for the character id and optional payload so Vitest records the mock call tuple as `[string | number, unknown?]`.
- Focused test: `bunx vitest run src/components/Option/Characters/__tests__/Manager.first-use.test.tsx -t "replaces existing folder token when reassigning folder in edit mode|submits edit flow through the shared form component"` from `apps/packages/ui` passed 2/2 selected tests.
- Package compiler capture: `bunx tsc --noEmit --pretty false > /tmp/task504-tsc-final.txt 2>&1` from `apps/packages/ui` still exits 2 for the known baseline, but `error TS` lines reduced from 100 to 96 and `rg -n 'Manager\.first-use' /tmp/task504-tsc-final.txt` returned no matches.
- Bandit skipped: this is a TypeScript test-only change and Bandit is a Python security scanner; no Python touched scope exists for this task.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Removed the four-error `Manager.first-use.test.tsx` package `tsc` cluster by parameterizing the local `updateCharacter` mock. The shared UI baseline is now 96 `error TS` lines after this slice.
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
