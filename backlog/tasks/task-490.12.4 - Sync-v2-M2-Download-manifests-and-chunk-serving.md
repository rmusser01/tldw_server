---
id: TASK-490.12.4
title: 'Sync v2 M2: Download manifests and chunk serving'
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
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Expose authenticated blob download manifests and resumable chunk or whole-blob serving for committed personal-dataset attachment blobs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Download manifest reports size, content type, chunk map, per-chunk hashes, full payload hash, and availability for the selected attachment/blob.
- [x] #2 Download APIs enforce authenticated dataset ownership and never expose blob bytes across users or datasets.
- [x] #3 Download tests cover missing, metadata-only, available, and checksum manifest cases.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-23-sync-v2-m2-restore-completeness-blobs-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Started implementation after TASK-490.12.3 landed in commit c74d83f63.
- Added dataset-scoped SyncBlobObject lookup through SyncDatabase/SyncV2Store, including deduplicated attachment lookup through sync_attachments payload hashes.
- Added core SyncBlobDownloadManifest generation and read_blob_bytes support with checksum/size verification before serving committed local blobs.
- Added authenticated GET /api/v1/sync/attachments/{attachment_id}/manifest and GET /api/v1/sync/attachments/{attachment_id} byte-serving endpoints.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented M2 download manifests and blob byte serving for committed Sync v2 attachment blobs. The service now reports metadata-only refs, available blob chunk maps with per-chunk hashes, and rejects missing or cross-user/cross-dataset blob access. Verification: targeted download tests passed, full Sync suite passed (309 passed, 6 warnings), Ruff passed on touched files, Bandit passed on touched Sync scope with JSON report at /tmp/bandit_sync_v2_m2_download.json, and git diff --check passed.
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
