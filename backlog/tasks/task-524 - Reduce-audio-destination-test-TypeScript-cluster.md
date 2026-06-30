---
id: TASK-524
title: Reduce audio destination test TypeScript cluster
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-02 19:50'
labels: []
dependencies: []
references:
  - TASK-523
  - >-
    apps/packages/ui/src/hooks/__tests__/audioCaptureCoordinator.low-level.test.tsx
  - apps/packages/ui/src/hooks/__tests__/useMicStream.test.tsx
  - apps/packages/ui/tsconfig.json
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue reducing the shared UI package-wide TypeScript compiler baseline by fixing the two test-only diagnostics in `src/hooks/__tests__/audioCaptureCoordinator.low-level.test.tsx` and `src/hooks/__tests__/useMicStream.test.tsx`. Current package `tsc` output reports direct casts from `{ kind: string }` mock destination objects to `AudioDestinationNode`, which require an explicit `unknown` bridge for partial DOM mocks.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Current audio destination mock compiler diagnostics are captured.
- [x] #2 Root cause is documented and tied to partial DOM test mocks rather than behavior changes.
- [x] #3 The two `AudioDestinationNode` test diagnostics are removed from package `tsc` output.
- [x] #4 Focused audio hook tests are run or an explicit blocker is recorded.
- [x] #5 Remaining package-wide `tsc` baseline count is recorded.
- [x] #6 Bandit decision is recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Captured red evidence from `/tmp/task523-tsc-final.txt`: package `tsc` reported two diagnostics where `{ kind: "destination" }` partial audio mocks were cast directly to `AudioDestinationNode`.
- Root cause was partial DOM test mocks only. The mock only needs to stand in for the `destination` target used by the hook wiring, not implement the full Web Audio destination node surface.
- Updated both mock `AudioContext.destination` properties to cast through `unknown` before `AudioDestinationNode`, matching the existing partial mock style used elsewhere in the same files.
- Focused verification: `bunx vitest run src/hooks/__tests__/audioCaptureCoordinator.low-level.test.tsx src/hooks/__tests__/useMicStream.test.tsx` passed: 4 tests.
- Package verification: `bunx tsc --noEmit --pretty false > /tmp/task524-tsc-final.txt 2>&1` still exits nonzero from the known baseline, but diagnostics dropped from 51 in `/tmp/task523-tsc-final.txt` to 49 in `/tmp/task524-tsc-final.txt`; searching for the two audio test paths and `AudioDestinationNode` returns no matches.
- Bandit skipped: this is a TypeScript test-only WebUI change with no Python touched.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Removed the two audio test `AudioDestinationNode` TypeScript diagnostics by casting the intentionally partial mock destination objects through `unknown` before the DOM type. Focused Vitest passed with 4 tests, and package `tsc` baseline dropped from 51 to 49 with no remaining audio destination diagnostics.
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
