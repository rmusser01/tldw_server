---
id: TASK-490.13.7
title: 'Sync v2 M3: Retention GC and observability'
status: To Do
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
