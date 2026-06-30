---
id: TASK-500
title: Reduce WorkspaceChatPanel TypeScript test baseline cluster
status: Done
references:
- TASK-499
- apps/packages/ui/src/components/Option/ChatWorkspace/__tests__/WorkspaceChatPanel.test.tsx
- apps/packages/ui/tsconfig.json
modified_files:
- apps/packages/ui/src/components/Option/ChatWorkspace/__tests__/WorkspaceChatPanel.test.tsx
- backlog/tasks/task-500 - Reduce-WorkspaceChatPanel-TypeScript-test-baseline-cluster.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue reducing the shared UI package-wide TypeScript compiler baseline by fixing the contained WorkspaceChatPanel test cluster. The current package `tsc` output reports 12 errors in `src/components/Option/ChatWorkspace/__tests__/WorkspaceChatPanel.test.tsx`, mostly around untyped mock call tuples.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Current WorkspaceChatPanel compiler diagnostics are captured.
- [x] #2 Root cause is documented and tied to test mock typing rather than production behavior.
- [x] #3 The `WorkspaceChatPanel.test.tsx` compiler cluster is removed from package `tsc` output.
- [x] #4 Focused behavior test is run or an explicit blocker is recorded.
- [x] #5 Remaining package-wide `tsc` baseline count is recorded.
- [x] #6 Bandit decision is recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect the WorkspaceChatPanel test and the exact compiler diagnostics to identify the mock typing root cause.
2. Use the current package `tsc` output as red evidence for the 12-error cluster.
3. Make the smallest test-only typing changes needed to preserve behavior and remove the cluster.
4. Run the focused WorkspaceChatPanel test if practical, then rerun package `bunx tsc --noEmit --pretty false` and record remaining baseline counts.
5. Record Bandit decision and final evidence.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Red evidence: package `bunx tsc --noEmit --pretty false` reported 12 errors in `src/components/Option/ChatWorkspace/__tests__/WorkspaceChatPanel.test.tsx`.
- Root cause: Vitest inferred `chatHookState.useMessageOption` and `chatHookState.onSubmit` as zero-argument mocks, so `mock.calls[0][0]` appeared to index into empty tuples; one message fixture also missed required `Message` fields.
- Fixed the test-only typing by deriving mock signatures from the real `useMessageOption` hook, adding a checked `getSubmitPayload()` helper, and completing the one `Message` fixture with required fields.
- Focused behavior verification passed with `bunx vitest run src/components/Option/ChatWorkspace/__tests__/WorkspaceChatPanel.test.tsx` (15 tests).
- Green evidence: follow-up package `bunx tsc --noEmit --pretty false` no longer reports `WorkspaceChatPanel.test.tsx`; remaining baseline dropped from 128 to 116 `error TS` lines.
- Bandit was not run because this task touched only TypeScript/Backlog files and no Python code.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Removed the contained WorkspaceChatPanel test cluster from the shared UI package-wide TypeScript baseline. The package compiler still fails on unrelated remaining clusters, but this slice reduced the baseline by 12 `error TS` lines.
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
