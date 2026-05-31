---
id: TASK-498
title: Address PR 2043 temp blob commit collision review
status: Done
labels:
- sync-v2
- code-review
priority: high
references:
- 'PR #2043'
- Qodo review thread https://github.com/rmusser01/tldw_server/pull/2043#discussion_r3293879128
modified_files:
- tldw_Server_API/app/core/Sync/v2/blob_store.py
- tldw_Server_API/tests/Sync/test_sync_v2_blob_store.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix the PR #2043 review finding where LocalSyncBlobStore.commit_upload uses a deterministic temp file path for content-addressed blobs, allowing concurrent commits of the same payload hash to collide in the shared blob root.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- Same-payload commits from distinct upload IDs use distinct temporary paths under the final blob directory.
- The final blob key remains content-addressed and identical for identical payloads, and committed bytes read back unchanged.
- Unique temporary files are atomically replaced into the final location and cleanup keeps the existing error handling behavior.
- The focused regression test, blob-store test file, full Sync suite, `git diff --check`, and Bandit verification are recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a regression test that demonstrates deterministic temp-path collision during concurrent/same-hash blob commits.
2. Change LocalSyncBlobStore.commit_upload to use a per-upload unique temporary file in the final blob directory before atomic replace.
3. Run focused blob-store tests, Sync tests, git diff checks, and Bandit on touched production code.
4. Push fix and reply to the PR review thread.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Resolved the PR #2043 Qodo reliability finding by changing LocalSyncBlobStore.commit_upload to use a unique temporary file per upload attempt in the final blob directory before atomic replace. Added a regression test proving same-payload uploads use distinct commit temp paths. Verification: targeted regression first failed on the old deterministic path, then passed after the fix; `python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_blob_store.py` => 5 passed; `python -m pytest tldw_Server_API/tests/Sync` => 425 passed, 6 warnings; `git diff --check` => clean; Bandit on `tldw_Server_API/app/core/Sync/v2/blob_store.py` => 0 findings.
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
