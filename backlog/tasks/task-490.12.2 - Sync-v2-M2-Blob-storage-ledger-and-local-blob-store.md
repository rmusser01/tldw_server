---
id: TASK-490.12.2
title: 'Sync v2 M2: Blob storage ledger and local blob store'
status: To Do
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
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add Sync v2 blob metadata tables, upload-session/chunk state, quota accounting, and a local filesystem blob-store adapter rooted under the user's encrypted storage scope.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Sync DB migrations add blob object, upload session, and chunk tables for SQLite and Postgres schema definitions.
- [ ] #2 Store tests cover idempotent sessions, quota reserve/release, checksum validation, dedupe by payload hash, and abandoned upload cleanup.
- [ ] #3 Local blob-store adapter writes only under the configured per-user sync_blobs root and uses atomic verified commit for completed blobs.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-23-sync-v2-m2-restore-completeness-blobs-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
