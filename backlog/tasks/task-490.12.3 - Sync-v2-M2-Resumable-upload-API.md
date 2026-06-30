---
id: TASK-490.12.3
title: 'Sync v2 M2: Resumable upload API'
status: Done
labels:
- sync
- sync-v2
- m2
- attachments
- api
priority: medium
parent_task_id: TASK-490.12
documentation:
- Docs/Design/Sync_V2_M2_Restore_Completeness_and_Blobs.md
- Docs/superpowers/plans/2026-05-23-sync-v2-m2-restore-completeness-blobs-implementation-plan.md
modified_files:
- tldw_Server_API/app/api/v1/endpoints/sync.py
- tldw_Server_API/app/core/Sync/v2/service.py
- tldw_Server_API/app/core/Sync/v2/factory.py
- tldw_Server_API/app/core/Sync/v2/blob_store.py
- tldw_Server_API/app/core/Sync/v2/models.py
- tldw_Server_API/app/core/DB_Management/Sync_DB.py
- tldw_Server_API/tests/Sync/test_sync_v2_service.py
- tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Expose authenticated resumable blob upload session, chunk upload, status, complete, cancel, and small-blob wrapper endpoints under /api/v1/sync.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Upload APIs enforce dataset ownership, enrolled domain checks, quota limits, chunk limits, idempotency, and safe validation errors.
- [x] #2 Chunk upload and complete APIs verify per-chunk and full payload hashes before marking a blob available.
- [x] #3 The existing POST /api/v1/sync/attachments path uses the same validation and commit path for small blobs instead of a parallel storage implementation.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-23-sync-v2-m2-restore-completeness-blobs-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Started implementation after TASK-490.12.2 landed in commit 49333c2c2.
- Implemented M2 resumable blob upload service methods and HTTP routes for create/status/chunk/complete/cancel under /api/v1/sync/blob-uploads.
- Routed small POST /api/v1/sync/attachments through the same local blob-store commit and blob ledger path before storing attachment metadata.
- Added quota accounting for active upload count plus service/API tests for ownership/domain/quota/hash validation and raw chunk completion.
- TDD red: the new service/API tests first failed because SyncV2Service did not accept blob_store and no upload methods/routes existed.
- Verification: targeted new tests passed (6 passed); service+endpoint Sync v2 suites passed (79 passed, 6 warnings); full Sync v2 glob passed (303 passed, 6 warnings); Ruff passed; Bandit on touched production files returned 0 findings; git diff --check passed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the Sync v2 M2 resumable upload API: authenticated blob upload sessions, raw chunk upload, completion, cancellation, quota/hash/domain validation, per-user local blob-store wiring, and small attachment uploads sharing the same commit path.
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
