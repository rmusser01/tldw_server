---
id: TASK-525
title: Reduce chat settings deep research test TypeScript cluster
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-02 19:54'
labels: []
dependencies: []
references:
  - TASK-524
  - apps/packages/ui/src/services/__tests__/chat-settings.deep-research.test.ts
  - >-
    apps/packages/ui/src/services/__tests__/chat-settings.deep-research-pinned.test.ts
  - >-
    apps/packages/ui/src/services/__tests__/chat-settings.deep-research-history.test.ts
  - apps/packages/ui/src/services/chat-settings.ts
  - apps/packages/ui/tsconfig.json
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue reducing the shared UI package-wide TypeScript compiler baseline by fixing the three test-only diagnostics in the chat-settings deep research tests. Current package `tsc` output reports spread-argument errors where the mocked `tldwClient` forwards `unknown[]` rest args into zero-argument `vi.fn()` storage mocks.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Current chat-settings deep research compiler diagnostics are captured.
- [x] #2 Root cause is documented and tied to test mock rest-spread forwarding rather than behavior changes.
- [x] #3 The three chat-settings deep research test diagnostics are removed from package `tsc` output.
- [x] #4 Focused chat-settings deep research tests are run or an explicit blocker is recorded.
- [x] #5 Remaining package-wide `tsc` baseline count is recorded.
- [x] #6 Bandit decision is recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Captured red evidence from `/tmp/task524-tsc-final.txt`: package `tsc` reported three spread-argument diagnostics in the chat-settings deep research test files.
- Root cause was test mock forwarding only. The mocked `tldwClient` methods forwarded `unknown[]` rest args into local `vi.fn()` mocks inferred as zero-argument functions, and these tests do not assert remote-client arguments.
- Replaced the rest-spread wrappers with direct calls to `storageState.initialize`, `storageState.getChatSettings`, and `storageState.updateChatSettings` in the three test files.
- Focused verification: `bunx vitest run src/services/__tests__/chat-settings.deep-research.test.ts src/services/__tests__/chat-settings.deep-research-pinned.test.ts src/services/__tests__/chat-settings.deep-research-history.test.ts` passed: 13 tests.
- Package verification: `bunx tsc --noEmit --pretty false > /tmp/task525-tsc-final.txt 2>&1` still exits nonzero from the known baseline, but diagnostics dropped from 49 in `/tmp/task524-tsc-final.txt` to 46 in `/tmp/task525-tsc-final.txt`; searching for `chat-settings.deep-research` returns no matches.
- Bandit skipped: this is a TypeScript test-only WebUI change with no Python touched.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Removed the three chat-settings deep-research test TypeScript diagnostics by replacing `unknown[]` rest-spread forwarding with direct calls to the local storage mocks. Focused Vitest passed with 13 tests, and package `tsc` baseline dropped from 49 to 46 with no remaining `chat-settings.deep-research` diagnostics.
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
