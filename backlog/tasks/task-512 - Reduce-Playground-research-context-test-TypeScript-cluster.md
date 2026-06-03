---
id: TASK-512
title: Reduce Playground research-context test TypeScript cluster
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-02 19:05'
labels: []
dependencies: []
references:
  - TASK-511
  - >-
    apps/packages/ui/src/components/Option/Playground/__tests__/Playground.research-context.integration.test.tsx
  - apps/packages/ui/src/services/chat-settings.ts
  - apps/packages/ui/src/types/chat-session-settings.ts
  - apps/packages/ui/tsconfig.json
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue reducing the shared UI package-wide TypeScript compiler baseline by fixing the contained diagnostics in `src/components/Option/Playground/__tests__/Playground.research-context.integration.test.tsx`. Current package `tsc` output reports three test typing errors around chat-settings mock forwarding and a mock implementation whose parameter does not match the inferred zero-argument mock signature.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Current Playground research-context compiler diagnostics are captured.
- [x] #2 Root cause is documented and tied to test mock typing rather than behavior changes.
- [x] #3 The `Playground.research-context.integration.test.tsx` compiler cluster is removed from package `tsc` output.
- [x] #4 Focused Playground research-context test is run or an explicit blocker is recorded.
- [x] #5 Remaining package-wide `tsc` baseline count is recorded.
- [x] #6 Bandit decision is recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Captured red evidence from `/tmp/task511-tsc-final.txt`: package `tsc` reported three diagnostics in `src/components/Option/Playground/__tests__/Playground.research-context.integration.test.tsx` around spreading `unknown[]` into chat-settings mocks and a `mockImplementation` callback whose parameter did not match the inferred zero-argument mock signature.
- Root cause was test mock typing only. The chat-settings test state inferred zero-argument mocks from `vi.fn(async () => null)`, while the real service accepts a single params object.
- Added `ChatSettingsSyncParams` and `ChatSettingsPatchParams` test-only types, gave the mocks explicit one-parameter async signatures with loose `unknown` returns, and forwarded mocked service calls by named `params` rather than spreading `unknown[]`.
- Added the missing `useDesktop` export to the same test file's `useMediaQuery` mock so the focused suite reflects the current component imports.
- Focused verification attempted with `bunx vitest run src/components/Option/Playground/__tests__/Playground.research-context.integration.test.tsx`. Initial run failed because the mock lacked `useDesktop`; after adding it, the suite reached current runtime assertions but still failed 13/17 with stale test expectations, including `setSelectedQuickPrompt is not a function` from the mocked `useMessageOption` path and missing persisted attachment surfaces. This is recorded as a focused-suite blocker outside the compiler-only slice.
- Package verification: `bunx tsc --noEmit --pretty false > /tmp/task512-tsc-final.txt 2>&1` still exits nonzero from the known baseline, but diagnostics dropped from 76 in `/tmp/task511-tsc-final.txt` to 73 in `/tmp/task512-tsc-final.txt`; `rg -n 'Playground\.research-context\.integration\.test\.tsx' /tmp/task512-tsc-final.txt` returns no matches.
- Bandit skipped: this is a TypeScript test-only WebUI change with no Python touched.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Removed the `Playground.research-context.integration.test.tsx` TypeScript cluster by typing the chat-settings mock parameter shapes and forwarding calls without spreading `unknown[]`. Also added the missing `useDesktop` export to the same test's media-query mock so the focused suite reaches current runtime assertions. Package `tsc` baseline dropped from 76 to 73 with no remaining Playground research-context diagnostics; the focused suite still has stale runtime expectations and is recorded as blocked.
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
