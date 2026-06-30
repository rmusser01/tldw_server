---
id: TASK-519
title: Reduce system prompt utils test TypeScript cluster
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-02 19:33'
labels: []
dependencies: []
references:
  - TASK-518
  - apps/packages/ui/src/components/Common/__tests__/system-prompt-utils.test.ts
  - apps/packages/ui/src/components/Common/system-prompt-utils.ts
  - apps/packages/ui/src/db/dexie/types.ts
  - apps/packages/ui/tsconfig.json
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue reducing the shared UI package-wide TypeScript compiler baseline by fixing the contained diagnostics in `src/components/Common/__tests__/system-prompt-utils.test.ts`. Current package `tsc` output reports two test fixture errors where mocked `getPromptByIdFn` results omit required Dexie `Prompt` fields.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Current system prompt utils compiler diagnostics are captured.
- [x] #2 Root cause is documented and tied to incomplete test prompt fixtures rather than behavior changes.
- [x] #3 The `system-prompt-utils.test.ts` compiler cluster is removed from package `tsc` output.
- [x] #4 Focused system prompt utils test is run or an explicit blocker is recorded.
- [x] #5 Remaining package-wide `tsc` baseline count is recorded.
- [x] #6 Bandit decision is recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Captured red evidence from `/tmp/task518-tsc-final.txt`: package `tsc` reported two diagnostics in `src/components/Common/__tests__/system-prompt-utils.test.ts` because mocked `getPromptByIdFn` results returned partial prompt objects missing required Dexie `Prompt` fields.
- Root cause was incomplete test fixtures only. `GetPromptByIdFn` returns `Promise<Prompt | undefined>`, and `Prompt` requires `title`, `is_system`, and `createdAt` in addition to `id` and `content`.
- Added the required prompt fields to the two mocked prompt returns while preserving the tested content behavior.
- Focused verification: `bunx vitest run src/components/Common/__tests__/system-prompt-utils.test.ts` passed: 4 tests.
- Package verification: `bunx tsc --noEmit --pretty false > /tmp/task519-tsc-final.txt 2>&1` still exits nonzero from the known baseline, but diagnostics dropped from 61 in `/tmp/task518-tsc-final.txt` to 59 in `/tmp/task519-tsc-final.txt`; `rg -n 'system-prompt-utils\.test\.ts' /tmp/task519-tsc-final.txt` returns no matches.
- Bandit skipped: this is a TypeScript test-only WebUI change with no Python touched.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Removed the `system-prompt-utils.test.ts` TypeScript cluster by completing the mocked Dexie `Prompt` fixtures with required `title`, `is_system`, and `createdAt` fields. Focused Vitest passed with 4 tests, and package `tsc` baseline dropped from 61 to 59 with no remaining system-prompt-utils diagnostics.
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
