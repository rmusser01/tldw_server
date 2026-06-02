---
id: TASK-521
title: Reduce Playground search integration test TypeScript cluster
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-02 19:41'
labels: []
dependencies: []
references:
  - TASK-520
  - >-
    apps/packages/ui/src/components/Option/Playground/__tests__/Playground.search.integration.test.tsx
  - apps/packages/ui/src/services/chat-settings.ts
  - apps/packages/ui/tsconfig.json
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue reducing the shared UI package-wide TypeScript compiler baseline by fixing the contained diagnostics in `src/components/Option/Playground/__tests__/Playground.search.integration.test.tsx`. Current package `tsc` output reports two test-only spread diagnostics where chat-settings mocks are called through `unknown[]` forwarding instead of their single params object shape.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Current Playground search integration compiler diagnostics are captured.
- [x] #2 Root cause is documented and tied to test chat-settings mock forwarding rather than behavior changes.
- [x] #3 The `Playground.search.integration.test.tsx` compiler cluster is removed from package `tsc` output.
- [x] #4 Focused Playground search integration test is run or an explicit blocker is recorded.
- [x] #5 Remaining package-wide `tsc` baseline count is recorded.
- [x] #6 Bandit decision is recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Captured red evidence from `/tmp/task520-tsc-final.txt`: package `tsc` reported two diagnostics in `src/components/Option/Playground/__tests__/Playground.search.integration.test.tsx` where `unknown[]` args were spread into chat-settings mocks.
- Root cause was test mock forwarding only. The real chat-settings functions take one params object, while the test forwarded generic unknown rest args to mocks inferred as zero-argument functions.
- Added `ChatSettingsSyncParams` and `ChatSettingsPatchParams` test-only types, gave the mocks explicit one-parameter async signatures with `unknown` returns, and forwarded mocked service calls by named `params`.
- Focused verification: `bunx vitest run src/components/Option/Playground/__tests__/Playground.search.integration.test.tsx` passed: 10 tests.
- Package verification: `bunx tsc --noEmit --pretty false > /tmp/task521-tsc-final.txt 2>&1` still exits nonzero from the known baseline, but diagnostics dropped from 57 in `/tmp/task520-tsc-final.txt` to 55 in `/tmp/task521-tsc-final.txt`; `rg -n 'Playground\.search\.integration\.test\.tsx' /tmp/task521-tsc-final.txt` returns no matches.
- Bandit skipped: this is a TypeScript test-only WebUI change with no Python touched.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Removed the `Playground.search.integration.test.tsx` TypeScript cluster by typing the chat-settings mock parameter shapes and forwarding calls by params object instead of spreading `unknown[]`. Focused Vitest passed with 10 tests, and package `tsc` baseline dropped from 57 to 55 with no remaining Playground search diagnostics.
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
