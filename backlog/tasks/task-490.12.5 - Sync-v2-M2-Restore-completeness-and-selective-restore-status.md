---
id: TASK-490.12.5
title: 'Sync v2 M2: Restore completeness and selective restore status'
status: To Do
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
- [ ] #1 Restore preview reports metadata_ready, blocked_by_conflicts, blob_incomplete, content_complete, and verified_complete states as applicable.
- [ ] #2 Restore into an existing profile remains blocked until Note and Chat conversation metadata conflicts are explicitly handled.
- [ ] #3 Restore APIs support selected domains/objects/blobs and explicit metadata-only restore without hiding missing required blobs.
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
