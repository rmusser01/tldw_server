---
id: TASK-490.12.5
title: 'Sync v2 M2: Restore completeness and selective restore status'
status: Done
labels:
- sync
- sync-v2
- m2
- restore
priority: medium
parent_task_id: TASK-490.12
documentation:
- Docs/Design/Sync_V2_M2_Restore_Completeness_and_Blobs.md
- Docs/superpowers/plans/2026-05-23-sync-v2-m2-restore-completeness-blobs-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Extend restore manifest/preview with profile-level restore status, per-domain detail, per-blob detail, explicit metadata-only mode, and verified-complete reporting.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Restore preview reports metadata_ready, blocked_by_conflicts, blob_incomplete, content_complete, and verified_complete states as applicable.
- [x] #2 Restore into an existing profile remains blocked until Note and Chat conversation metadata conflicts are explicitly handled.
- [x] #3 Restore APIs support selected domains/objects/blobs and explicit metadata-only restore without hiding missing required blobs.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-23-sync-v2-m2-restore-completeness-blobs-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Started after TASK-490.12.4 landed in commit 67237a2b9.
- Added selected_object_ids, selected_attachment_ids, and metadata_only restore preview request handling.
- Added profile-level restore_status plus per-domain and per-blob completeness details to core restore preview output.
- M2 restore preview now uses committed blob objects for server availability when blob transfer is enabled, while M1 metadata-only mode preserves attachment.ref availability semantics.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented restore completeness reporting for Sync v2 M2. Restore preview now accepts selected object IDs, selected attachment IDs, and explicit metadata-only mode; reports profile-level restore_status; includes per-domain counts/statuses and per-blob details; uses the M2 blob ledger for server availability when blob transfer is enabled while preserving M1 attachment.ref availability semantics when it is not. Verification: targeted red/green tests passed, full Sync suite passed (310 passed, 6 warnings), restore e2e passed (4 passed), Ruff passed on touched files, Bandit passed on touched Sync scope with JSON report at /tmp/bandit_sync_v2_m2_restore_completeness.json, and git diff --check passed.
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
