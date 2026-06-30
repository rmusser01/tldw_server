---
id: TASK-501
title: Reduce useComposerQueue TypeScript test mock cluster
status: Done
references:
- TASK-500
- apps/packages/ui/src/components/Chat/composer/__tests__/useComposerQueue.test.tsx
- apps/packages/ui/src/components/Chat/composer/hooks/useComposerQueue.ts
- apps/packages/ui/tsconfig.json
modified_files:
- apps/packages/ui/src/components/Chat/composer/__tests__/useComposerQueue.test.tsx
- backlog/tasks/task-501 - Reduce-useComposerQueue-TypeScript-test-mock-cluster.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue reducing the shared UI package-wide TypeScript compiler baseline by fixing the contained useComposerQueue test mock typing cluster. Current package `tsc` output reports six errors in `src/components/Chat/composer/__tests__/useComposerQueue.test.tsx`, all from untyped Vitest mocks being passed into strongly typed hook options.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Current useComposerQueue compiler diagnostics are captured.
- [x] #2 Root cause is documented and tied to test mock typing rather than production behavior.
- [x] #3 The `useComposerQueue.test.tsx` compiler cluster is removed from package `tsc` output.
- [x] #4 Focused behavior test is run or an explicit blocker is recorded.
- [x] #5 Remaining package-wide `tsc` baseline count is recorded.
- [x] #6 Bandit decision is recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect the useComposerQueue hook option types and current test diagnostics.
2. Use current package `tsc` output as red evidence for the six-error mock typing cluster.
3. Add minimal test-only typed mock helpers or option aliases so mock functions satisfy the hook contract.
4. Run focused useComposerQueue tests, then package `bunx tsc --noEmit --pretty false` and record remaining baseline counts.
5. Record Bandit decision and final evidence.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Red evidence: package `bunx tsc --noEmit --pretty false` reported six `TS2322` errors in `src/components/Chat/composer/__tests__/useComposerQueue.test.tsx`.
- Root cause: the test helper typed hook callbacks as generic `ReturnType<typeof vi.fn>`, so Vitest mocks were not assignable to the strongly typed `UseComposerQueueOptions` callback signatures.
- Fixed the test-only typing by importing `UseComposerQueueOptions`, using those exact option callback types in `BaseProps`, and typing default/test mocks for `sendQueuedRequest`, `stopStreamingRequest`, `onEnqueueBlocked`, and `onEnqueueSuccess`.
- Focused behavior verification passed with `bunx vitest run src/components/Chat/composer/__tests__/useComposerQueue.test.tsx` (10 tests).
- Green evidence: follow-up package `bunx tsc --noEmit --pretty false` no longer reports `useComposerQueue.test.tsx`; remaining baseline dropped from 116 to 110 `error TS` lines.
- Bandit was not run because this task touched only TypeScript/Backlog files and no Python code.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Removed the contained useComposerQueue test mock cluster from the shared UI package-wide TypeScript baseline. The package compiler still fails on unrelated remaining clusters, but this slice reduced the baseline by six `error TS` lines.
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
