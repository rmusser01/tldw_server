---
id: TASK-508
title: Reduce TldwApiClient share-link scope TypeScript cluster
status: Done
references:
- TASK-507
- apps/packages/ui/src/services/__tests__/tldw-api-client.share-links.test.ts
- apps/packages/ui/src/services/tldw/TldwApiClient.ts
- apps/packages/ui/src/services/tldw/domains/chat-rag.ts
- apps/packages/ui/tsconfig.json
modified_files:
- apps/packages/ui/src/services/tldw/TldwApiClient.ts
- backlog/tasks/task-508 - Reduce-TldwApiClient-share-link-scope-TypeScript-cluster.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue reducing the shared UI package-wide TypeScript compiler baseline by fixing the contained TldwApiClient conversation share-link scope cluster. Current package `tsc` output reports two errors in `src/services/__tests__/tldw-api-client.share-links.test.ts` because the class `listConversationShareLinks` and `revokeConversationShareLink` methods do not accept the workspace scope options that the domain implementation and regression test expect.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Current share-link compiler diagnostics are captured.
- [x] #2 Focused share-link test is run before the fix and the failure/root cause is recorded.
- [x] #3 The TldwApiClient class share-link scope behavior matches the existing domain implementation.
- [x] #4 The `tldw-api-client.share-links.test.ts` compiler cluster is removed from package `tsc` output.
- [x] #5 Focused behavior test passes after the fix.
- [x] #6 Remaining package-wide `tsc` baseline count and Bandit decision are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Red compiler evidence came from `/tmp/task507-tsc-final.txt`, which contained two `tldw-api-client.share-links.test.ts` diagnostics because `TldwApiClientBase.listConversationShareLinks` and `revokeConversationShareLink` did not accept the scoped options already supported by the chat-rag domain implementation.
- Pre-fix focused test: `bunx vitest run src/services/__tests__/tldw-api-client.share-links.test.ts` from `apps/packages/ui` passed 4/4 at runtime, indicating the domain wiring already covered behavior, while the class method surface still drifted from the expected type contract.
- Aligned the class methods with the domain implementation by accepting `{ scope?: ChatScope }`, converting it through `toChatScopeParams`, and appending the resulting query with `appendPathQuery`.
- Post-fix focused test: `bunx vitest run src/services/__tests__/tldw-api-client.share-links.test.ts` from `apps/packages/ui` passed 4/4.
- Package compiler capture: `bunx tsc --noEmit --pretty false > /tmp/task508-tsc-final.txt 2>&1` from `apps/packages/ui` still exits 2 for the known baseline, but `error TS` lines reduced from 87 to 85 and `rg -n 'tldw-api-client\.share-links' /tmp/task508-tsc-final.txt` returned no matches.
- Bandit skipped: this is a TypeScript-only change and Bandit is a Python security scanner; no Python touched scope exists for this task.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Removed the two-error `tldw-api-client.share-links.test.ts` package `tsc` cluster by aligning the class share-link list/revoke methods with the existing scoped domain implementation. The shared UI baseline is now 85 `error TS` lines after this slice.
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
