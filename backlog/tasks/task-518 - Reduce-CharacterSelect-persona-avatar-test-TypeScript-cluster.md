---
id: TASK-518
title: Reduce CharacterSelect persona avatar test TypeScript cluster
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-02 19:29'
labels: []
dependencies: []
references:
  - TASK-517
  - >-
    apps/packages/ui/src/components/Sidepanel/Chat/__tests__/CharacterSelect.persona-avatar.test.tsx
  - apps/packages/ui/src/components/Sidepanel/Chat/CharacterSelect.tsx
  - apps/packages/ui/tsconfig.json
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue reducing the shared UI package-wide TypeScript compiler baseline by fixing the contained diagnostics in `src/components/Sidepanel/Chat/__tests__/CharacterSelect.persona-avatar.test.tsx`. Current package `tsc` output reports two test-only spread diagnostics where zero-argument mocked tldw client methods are called through `unknown[]` forwarding.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Current CharacterSelect persona avatar compiler diagnostics are captured.
- [x] #2 Root cause is documented and tied to test mock forwarding rather than behavior changes.
- [x] #3 The `CharacterSelect.persona-avatar.test.tsx` compiler cluster is removed from package `tsc` output.
- [x] #4 Focused CharacterSelect persona avatar test is run or an explicit blocker is recorded.
- [x] #5 Remaining package-wide `tsc` baseline count is recorded.
- [x] #6 Bandit decision is recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Captured red evidence from `/tmp/task517-tsc-final.txt`: package `tsc` reported two diagnostics in `src/components/Sidepanel/Chat/__tests__/CharacterSelect.persona-avatar.test.tsx` where `unknown[]` args were spread into zero-argument mock functions.
- Root cause was test mock forwarding only. The hoisted `initialize` and `listPersonaProfiles` mocks are declared as zero-argument functions, so forwarding generic unknown args does not match their tuple type.
- Replaced `(...args: unknown[]) => mocks.initialize(...args)` and `mocks.listPersonaProfiles(...args)` with direct zero-argument calls.
- Focused verification: `bunx vitest run src/components/Sidepanel/Chat/__tests__/CharacterSelect.persona-avatar.test.tsx` passed: 2 tests.
- Package verification: `bunx tsc --noEmit --pretty false > /tmp/task518-tsc-final.txt 2>&1` still exits nonzero from the known baseline, but diagnostics dropped from 63 in `/tmp/task517-tsc-final.txt` to 61 in `/tmp/task518-tsc-final.txt`; `rg -n 'CharacterSelect\.persona-avatar\.test\.tsx' /tmp/task518-tsc-final.txt` returns no matches.
- Bandit skipped: this is a TypeScript test-only WebUI change with no Python touched.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Removed the `CharacterSelect.persona-avatar.test.tsx` TypeScript cluster by replacing `unknown[]` spread forwarding with direct calls to the zero-argument mocked tldw client methods. Focused Vitest passed with 2 tests, and package `tsc` baseline dropped from 63 to 61 with no remaining persona-avatar diagnostics.
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
