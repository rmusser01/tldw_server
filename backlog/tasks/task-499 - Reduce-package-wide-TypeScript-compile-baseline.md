---
id: TASK-499
title: Reduce package-wide TypeScript compile baseline
status: Done
references:
- TASK-495
- TASK-498
- apps/package.json
- apps/packages/ui/tsconfig.json
- apps/extension/tsconfig.compile.json
- apps/tldw-frontend/tsconfig.json
modified_files:
- apps/packages/ui/src/services/__tests__/voice-conversation.test.ts
- backlog/tasks/task-499 - Reduce-package-wide-TypeScript-compile-baseline.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Start reducing the package-wide TypeScript compiler failure baseline for the shared WebUI/extension workspace. Capture the current `bunx tsc --noEmit --pretty false` failure clusters, choose a small high-confidence slice, fix the underlying type issues, and record verification evidence.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Current package-wide TypeScript compiler failure clusters are captured with counts.
- [x] #2 One coherent cluster is selected and fixed without broad refactors or unrelated formatting churn.
- [x] #3 The selected cluster has red/green compiler evidence.
- [x] #4 Focused verification and package-wide `bunx tsc --noEmit --pretty false` are recorded, including any remaining baseline failures.
- [x] #5 Bandit decision is recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Run the package-wide TypeScript command and summarize current failure clusters.
2. Select one small, coherent cluster that can be fixed without broad refactors.
3. Use the failing compiler output as the red test, then make the minimal code changes to remove that cluster.
4. Verify the focused files and package-wide compiler command, documenting any remaining baseline errors.
5. Record Bandit decision and final task evidence.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Captured the shared UI package baseline with `bunx tsc --noEmit --pretty false` from `apps/packages/ui`.
- Initial compiler baseline: 140 `error TS` lines / 270 output lines. Largest file clusters included `src/services/__tests__/voice-conversation.test.ts` (12), `src/components/Option/ChatWorkspace/__tests__/WorkspaceChatPanel.test.tsx` (12), `src/components/Chat/composer/__tests__/useComposerQueue.test.tsx` (6), and `src/components/Option/Onboarding/__tests__/OnboardingConnectForm.design-system.test.tsx` (6).
- Selected the `voice-conversation.test.ts` cluster because all errors came from one discriminated-union narrowing issue in a test-only file.
- Red evidence: baseline `tsc` reported 12 `TS2339` errors in `voice-conversation.test.ts` for accessing `value` or `reason` after Vitest expectations that do not narrow union types.
- Fixed the test by adding local assertion helpers that use explicit boolean-literal comparisons before returning the success config or failure reason.
- Green evidence: `voice-conversation.test.ts` no longer appears in the follow-up `tsc` output; remaining baseline is 128 `error TS` lines / 246 output lines.
- Focused behavior verification passed with `bunx vitest run src/services/__tests__/voice-conversation.test.ts` (12 tests).
- Bandit was not run because this task touched only TypeScript/Backlog files and no Python code.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Started reducing the shared UI package-wide TypeScript baseline by eliminating the contained `voice-conversation.test.ts` discriminated-union narrowing cluster. The compiler baseline dropped from 140 to 128 `error TS` lines; the package-wide command still fails on unrelated remaining baseline clusters.
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
