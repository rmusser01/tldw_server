---
id: TASK-511
title: Reduce PromptBody search pagination test TypeScript cluster
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-02 19:00'
labels: []
dependencies: []
references:
  - TASK-510
  - >-
    apps/packages/ui/src/components/Option/Prompt/__tests__/PromptBody.search-pagination.test.tsx
  - apps/packages/ui/src/services/prompt-sync.ts
  - apps/packages/ui/tsconfig.json
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue reducing the shared UI package-wide TypeScript compiler baseline by fixing the contained diagnostics in `src/components/Option/Prompt/__tests__/PromptBody.search-pagination.test.tsx`. Current package `tsc` output reports three test typing errors around prompt sync mock result shapes and tuple-safe clipboard mock call access.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Current PromptBody search-pagination compiler diagnostics are captured.
- [x] #2 Root cause is documented and tied to test mock typing rather than behavior changes.
- [x] #3 The `PromptBody.search-pagination.test.tsx` compiler cluster is removed from package `tsc` output.
- [x] #4 Focused PromptBody search-pagination test is run or an explicit blocker is recorded.
- [x] #5 Remaining package-wide `tsc` baseline count is recorded.
- [x] #6 Bandit decision is recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Captured red evidence from `/tmp/task510-tsc-final.txt`: package `tsc` reported three diagnostics in `src/components/Option/Prompt/__tests__/PromptBody.search-pagination.test.tsx` at the prompt sync mock result overrides and clipboard mock call access.
- Root cause was test mock typing only. `pushToStudio` was inferred from its initial `{ success: true }` implementation, so later test cases adding `syncStatus` were rejected even though the mocked service path returns sync metadata.
- Added a small `PromptSyncMockResult` test type and used it for the `pushToStudio` mock return promise so later mock resolutions can include `syncStatus`, `localId`, or `error` without changing behavior.
- Typed the local clipboard `writeText` mock with a `_text: string` parameter so `writeText.mock.calls[0]?.[0]` has a valid tuple element.
- Focused verification: `bunx vitest run src/components/Option/Prompt/__tests__/PromptBody.search-pagination.test.tsx` passed: 58 tests.
- Package verification: `bunx tsc --noEmit --pretty false > /tmp/task511-tsc-final.txt 2>&1` still exits nonzero from the known baseline, but diagnostics dropped from 79 in `/tmp/task510-tsc-final.txt` to 76 in `/tmp/task511-tsc-final.txt`; `rg -n 'PromptBody\.search-pagination\.test\.tsx' /tmp/task511-tsc-final.txt` returns no matches.
- Bandit skipped: this is a TypeScript test-only WebUI change with no Python touched.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Removed the `PromptBody.search-pagination.test.tsx` TypeScript cluster by widening the prompt sync mock result shape used by the test and typing the clipboard write mock as string-accepting. Focused Vitest passed with 58 tests, and package `tsc` baseline dropped from 79 to 76 with no remaining PromptBody search-pagination diagnostics.
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
