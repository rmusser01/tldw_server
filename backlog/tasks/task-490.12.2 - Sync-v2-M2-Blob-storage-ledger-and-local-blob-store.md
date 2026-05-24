---
id: TASK-490.12.2
title: 'Sync v2 M2: Blob storage ledger and local blob store'
status: Done
labels:
- sync
- sync-v2
- m2
- attachments
- storage
priority: medium
parent_task_id: TASK-490.12
documentation:
- Docs/Design/Sync_V2_M2_Restore_Completeness_and_Blobs.md
- Docs/superpowers/plans/2026-05-23-sync-v2-m2-restore-completeness-blobs-implementation-plan.md
assignee:
- '@Codex'
modified_files:
- tldw_Server_API/app/core/DB_Management/Sync_DB.py
- tldw_Server_API/app/core/Sync/v2/store.py
- tldw_Server_API/app/core/Sync/v2/service.py
- tldw_Server_API/app/core/Sync/v2/blob_store.py
- tldw_Server_API/tests/Sync/test_sync_v2_store.py
- tldw_Server_API/tests/Sync/test_sync_v2_blob_store.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add Sync v2 blob metadata tables, upload-session/chunk state, quota accounting, and a local filesystem blob-store adapter rooted under the user's encrypted storage scope.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Sync DB migrations add blob object, upload session, and chunk tables for SQLite and Postgres schema definitions.
- [x] #2 Store tests cover idempotent sessions, quota reserve/release, checksum validation, dedupe by payload hash, and abandoned upload cleanup.
- [x] #3 Local blob-store adapter writes only under the configured per-user sync_blobs root and uses atomic verified commit for completed blobs.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-23-sync-v2-m2-restore-completeness-blobs-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Implemented Sync v2 M2 blob ledger tables for SQLite/Postgres, core blob upload/session/chunk/object dataclasses, SyncStore facade methods, and LocalSyncBlobStore atomic chunk/commit handling under a confined root.
- Verified TDD red path: new blob ledger/blob-store tests initially failed on missing SyncBlobChunkCreate import before implementation.
- Verification: targeted blob tests passed (5 passed); store/blob suite passed (46 passed); full Sync v2 glob passed (297 passed, 6 warnings); Ruff passed on touched files; Bandit on touched production files returned 0 findings; git diff --check passed.
- Post-cleanup verification repeated: blob-focused tests passed (4 passed); full Sync v2 glob passed again (297 passed, 6 warnings); Ruff passed; Bandit JSON totals remained 0 findings; git diff --check passed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the Sync v2 M2 blob storage foundation: database ledger tables for blob objects, upload sessions, and chunks; idempotent/quota-aware store operations; and a local filesystem blob store that verifies SHA-256 chunks, confines paths to the configured root, and atomically commits completed blobs.
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
