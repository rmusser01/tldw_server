---
id: TASK-515
title: Reduce background proxy test TypeScript cluster
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-02 19:15'
labels: []
dependencies: []
references:
  - TASK-514
  - apps/packages/ui/src/services/__tests__/background-proxy.test.ts
  - apps/packages/ui/tsconfig.json
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue reducing the shared UI package-wide TypeScript compiler baseline by fixing the contained diagnostics in `src/services/__tests__/background-proxy.test.ts`. Current package `tsc` output reports two test-only TS2352 cast diagnostics around a partial streaming `Response` mock and tuple access to recorded fetch calls.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Current background-proxy compiler diagnostics are captured.
- [x] #2 Root cause is documented and tied to intentional partial test mocks rather than behavior changes.
- [x] #3 The `background-proxy.test.ts` compiler cluster is removed from package `tsc` output.
- [x] #4 Focused background-proxy test is run or an explicit blocker is recorded.
- [x] #5 Remaining package-wide `tsc` baseline count is recorded.
- [x] #6 Bandit decision is recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Captured red evidence from `/tmp/task514-tsc-final.txt`: package `tsc` reported two TS2352 diagnostics in `src/services/__tests__/background-proxy.test.ts` around a partial streaming `Response` mock and tuple access to `fetchSpy.mock.calls[0]`.
- Root cause was intentional partial test mocks. The streaming test only needs `ok`, `status`, and `body.getReader`, not a complete DOM `Response`; the fetch call assertion knows the mock call tuple shape even though TypeScript sees the raw calls array as broader.
- Added explicit `unknown` bridge casts at the two reported sites so the intent is visible to the compiler without changing runtime behavior.
- Focused verification: `bunx vitest run src/services/__tests__/background-proxy.test.ts` passed: 26 tests.
- Package verification: `bunx tsc --noEmit --pretty false > /tmp/task515-tsc-final.txt 2>&1` still exits nonzero from the known baseline, but diagnostics dropped from 69 in `/tmp/task514-tsc-final.txt` to 67 in `/tmp/task515-tsc-final.txt`; `rg -n 'background-proxy\.test\.ts' /tmp/task515-tsc-final.txt` returns no matches.
- Bandit skipped: this is a TypeScript test-only WebUI change with no Python touched.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Removed the `background-proxy.test.ts` TypeScript cluster by making intentional partial test mock casts explicit through `unknown`. Focused Vitest passed with 26 tests, and package `tsc` baseline dropped from 69 to 67 with no remaining background-proxy diagnostics.
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
