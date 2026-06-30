---
id: TASK-490.13.7
title: 'Sync v2 M3: Retention GC and observability'
status: Done
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
- [x] #1 Retention dry-run reports envelope, tombstone, compaction, and blob GC candidates without mutation and explains blockers.
- [x] #2 Compaction and deletion remain policy-gated and require device acknowledgments, retention windows, and active-reference checks.
- [x] #3 User/admin diagnostics report sync health, lag, failures, quota, blob health, key blockers, and retention status with strict redaction.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Completed Stage 7 through three child slices: TASK-490.13.7.1 retention dry-run candidates, TASK-490.13.7.2 redacted diagnostics, and TASK-490.13.7.3 guarded compaction/GC foundation. The apply path remains conservative: no envelope audit-log deletion and no physical blob byte deletion in this slice; only domain compaction checkpoints and eligible blob metadata soft-delete are applied after blockers clear.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed Sync v2 M3 Stage 7 retention, compaction/GC, and observability foundation. Added dry-run retention candidates, redacted diagnostics, and a guarded compact endpoint with explicit confirmation, all-or-nothing blocker refusal for selected candidates, non-destructive domain checkpoints, and soft-deleted eligible blob metadata. Verification for the final guarded slice: retention tests passed with 11 passed; neighboring tests passed with 178 passed; full Sync suite passed with 412 passed and 6 warnings; Ruff passed on touched files; Bandit report /tmp/bandit_sync_v2_m3_retention_compact.json has 0 results. Known deferrals: physical blob byte deletion, envelope audit-log deletion, and audit-event search remain later work.
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
