---
id: TASK-523
title: Reduce research chat context test TypeScript cluster
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-02 19:47'
labels: []
dependencies: []
references:
  - TASK-522
  - >-
    apps/packages/ui/src/components/Option/Playground/__tests__/research-chat-context.test.ts
  - apps/packages/ui/src/components/Option/Playground/research-chat-context.ts
  - apps/packages/ui/src/services/tldw/TldwApiClient.ts
  - apps/packages/ui/tsconfig.json
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue reducing the shared UI package-wide TypeScript compiler baseline by fixing the contained diagnostics in `src/components/Option/Playground/__tests__/research-chat-context.test.ts`. Current package `tsc` output reports two test-only fixture diagnostics where `as const` makes nested context arrays readonly and incompatible with the mutable `AttachedResearchContext` shape.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Current research chat context compiler diagnostics are captured.
- [x] #2 Root cause is documented and tied to test fixture readonly inference rather than behavior changes.
- [x] #3 The `research-chat-context.test.ts` compiler cluster is removed from package `tsc` output.
- [x] #4 Focused research chat context test is run or an explicit blocker is recorded.
- [x] #5 Remaining package-wide `tsc` baseline count is recorded.
- [x] #6 Bandit decision is recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Captured red evidence from `/tmp/task522-tsc-final.txt`: package `tsc` reported two diagnostics in `src/components/Option/Playground/__tests__/research-chat-context.test.ts` where `as const` made nested `outline` arrays readonly and incompatible with `AttachedResearchContext`.
- Root cause was test fixture inference only. The helpers expect mutable `AttachedResearchContext` arrays from the API client type, while two inline fixtures were frozen as readonly literals.
- Typed the affected inline `active` and `baseline` fixtures as `AttachedResearchContext` and removed `as const`, preserving fixture values and assertions.
- Focused verification: `bunx vitest run src/components/Option/Playground/__tests__/research-chat-context.test.ts` passed: 16 tests.
- Package verification: `bunx tsc --noEmit --pretty false > /tmp/task523-tsc-final.txt 2>&1` still exits nonzero from the known baseline, but diagnostics dropped from 53 in `/tmp/task522-tsc-final.txt` to 51 in `/tmp/task523-tsc-final.txt`; searching for `research-chat-context.test.ts` in `/tmp/task523-tsc-final.txt` returns no matches.
- Bandit skipped: this is a TypeScript test-only WebUI change with no Python touched.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Removed the `research-chat-context.test.ts` TypeScript cluster by typing the two inline fixtures as `AttachedResearchContext` instead of using `as const`, avoiding readonly nested arrays while keeping the same fixture values and assertions. Focused Vitest passed with 16 tests, and package `tsc` baseline dropped from 53 to 51 with no remaining research-chat-context diagnostics.
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
