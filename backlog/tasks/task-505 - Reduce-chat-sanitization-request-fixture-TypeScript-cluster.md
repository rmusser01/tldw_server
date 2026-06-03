---
id: TASK-505
title: Reduce chat sanitization request fixture TypeScript cluster
status: Done
references:
- TASK-504
- apps/packages/ui/src/services/__tests__/tldw-api-client.chat-sanitization-regression.test.ts
- apps/packages/ui/src/services/tldw/TldwApiClient.ts
- apps/packages/ui/tsconfig.json
modified_files:
- apps/packages/ui/src/services/__tests__/tldw-api-client.chat-sanitization-regression.test.ts
- backlog/tasks/task-505 - Reduce-chat-sanitization-request-fixture-TypeScript-cluster.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue reducing the shared UI package-wide TypeScript compiler baseline by fixing the contained chat sanitization regression test request fixture cluster. Current package `tsc` output reports three errors in `src/services/__tests__/tldw-api-client.chat-sanitization-regression.test.ts` because a readonly `as const` request fixture is passed where `ChatCompletionRequest` expects mutable messages.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Current chat sanitization compiler diagnostics are captured.
- [x] #2 Root cause is documented and tied to test fixture typing rather than production behavior.
- [x] #3 The `tldw-api-client.chat-sanitization-regression.test.ts` compiler cluster is removed from package `tsc` output.
- [x] #4 Focused behavior test is run or an explicit blocker is recorded.
- [x] #5 Remaining package-wide `tsc` baseline count is recorded.
- [x] #6 Bandit decision is recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Red compiler evidence came from `/tmp/task504-tsc-final.txt`, which contained three `tldw-api-client.chat-sanitization-regression.test.ts` diagnostics where a readonly `as const` request fixture could not satisfy `ChatCompletionRequest`.
- Root cause was test-only fixture typing: `messages` was inferred as a readonly tuple, while the API request type expects mutable `ChatMessage[]`. Production client behavior was not changed.
- Annotated the shared request fixture as `ChatCompletionRequest` and removed the `as const` assertion.
- Focused test: `bunx vitest run src/services/__tests__/tldw-api-client.chat-sanitization-regression.test.ts` from `apps/packages/ui` passed 5/5.
- Package compiler capture: `bunx tsc --noEmit --pretty false > /tmp/task505-tsc-final.txt 2>&1` from `apps/packages/ui` still exits 2 for the known baseline, but `error TS` lines reduced from 96 to 93 and `rg -n 'tldw-api-client\.chat-sanitization-regression' /tmp/task505-tsc-final.txt` returned no matches.
- Bandit skipped: this is a TypeScript test-only change and Bandit is a Python security scanner; no Python touched scope exists for this task.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Removed the three-error `tldw-api-client.chat-sanitization-regression.test.ts` package `tsc` cluster by typing the local request fixture as `ChatCompletionRequest`. The shared UI baseline is now 93 `error TS` lines after this slice.
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
