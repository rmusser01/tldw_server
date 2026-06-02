---
id: TASK-527
title: Reduce audio capture plan test TypeScript diagnostic
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-02 20:00'
labels: []
dependencies: []
references:
  - TASK-526
  - apps/packages/ui/src/audio/__tests__/resolve-audio-capture-plan.test.ts
  - apps/packages/ui/src/audio/source-types.ts
  - apps/packages/ui/src/audio/resolve-audio-capture-plan.ts
  - apps/packages/ui/tsconfig.json
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue reducing the shared UI package-wide TypeScript compiler baseline by fixing the contained diagnostic in `src/audio/__tests__/resolve-audio-capture-plan.test.ts`. Current package `tsc` output reports a test fixture inferred with `sourceKind: string` instead of the `AudioSourceKind` literal union.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Current audio capture plan compiler diagnostic is captured.
- [x] #2 Root cause is documented and tied to test fixture literal widening rather than behavior changes.
- [x] #3 The `resolve-audio-capture-plan.test.ts` diagnostic is removed from package `tsc` output.
- [x] #4 Focused audio capture plan test is run or an explicit blocker is recorded.
- [x] #5 Remaining package-wide `tsc` baseline count is recorded.
- [x] #6 Bandit decision is recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Captured red evidence from `/tmp/task526-tsc-final.txt`: package `tsc` reported one diagnostic in `src/audio/__tests__/resolve-audio-capture-plan.test.ts` where the shared `requestedSource` fixture inferred `sourceKind` as `string`.
- Root cause was test fixture literal widening only. The resolver expects `AudioCaptureRequestedSource`, and the fixture value already uses a valid `AudioSourceKind`.
- Imported `AudioCaptureRequestedSource` from `@/audio` and annotated the `requestedSource` fixture, preserving the object used by assertions.
- Focused verification: `bunx vitest run src/audio/__tests__/resolve-audio-capture-plan.test.ts` passed: 1 test.
- Package verification: `bunx tsc --noEmit --pretty false > /tmp/task527-tsc-final.txt 2>&1` still exits nonzero from the known baseline, but diagnostics dropped from 42 in `/tmp/task526-tsc-final.txt` to 41 in `/tmp/task527-tsc-final.txt`; searching for `resolve-audio-capture-plan.test.ts` returns no matches.
- Bandit skipped: this is a TypeScript test-only WebUI change with no Python touched.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Removed the `resolve-audio-capture-plan.test.ts` TypeScript diagnostic by typing the shared requested-source fixture as `AudioCaptureRequestedSource`, preventing `sourceKind` from widening to `string`. Focused Vitest passed with 1 test, and package `tsc` baseline dropped from 42 to 41 with no remaining audio capture plan diagnostic.
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
