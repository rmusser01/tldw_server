---
id: TASK-490.13.7
title: 'Sync v2 M3: Retention GC and observability'
status: In Progress
labels:
- sync
- sync-v2
- m3
- retention
- observability
priority: medium
parent_task_id: TASK-490.13
documentation:
- Docs/Design/Sync_V2_M3_Polished_Multi_Device.md
- Docs/API/Sync_V2_M3.md
- Docs/superpowers/plans/2026-05-23-sync-v2-m3-polished-multi-device-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add conservative retention, compaction, blob garbage-collection, and diagnostics surfaces after per-device acknowledgments make safety provable.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Retention dry-run reports envelope, tombstone, compaction, and blob GC candidates without mutation and explains blockers.
- [ ] #2 Compaction and deletion remain policy-gated and require device acknowledgments, retention windows, and active-reference checks.
- [ ] #3 User/admin diagnostics report sync health, lag, failures, quota, blob health, key blockers, and retention status with strict redaction.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Started Stage 7 with a dry-run-only retention candidate slice. This first implementation unit will not delete, compact, or GC data; it should only calculate candidates and blockers.
Completed child TASK-490.13.7.1: dry-run-only retention candidate calculation now reports envelope_compaction, tombstone_prune, and blob_gc candidates with stable blockers and no mutation, including restore-window blockers and latest-tombstone object-chain handling. Destructive compaction/GC and diagnostics remain open for later Stage 7 slices.
Completed child TASK-490.13.7.2: GET /api/v1/sync/diagnostics now reports redacted dataset/domain counts, device lag, blob/upload pressure, key summary, and retention dry-run summary with payload/ciphertext/key material omitted. Destructive compaction/GC remains open for the next Stage 7 slice.
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
