---
id: TASK-490.13.7.1
title: 'Sync v2 M3: Retention dry-run candidate foundation'
status: Done
assignee: []
created_date: ''
updated_date: 2026-05-24 00:44
labels:
- sync
- sync-v2
- m3
- retention
dependencies: []
documentation:
- Docs/Design/Sync_V2_M3_Polished_Multi_Device.md
- Docs/API/Sync_V2_M3.md
- Docs/superpowers/plans/2026-05-23-sync-v2-m3-polished-multi-device-implementation-plan.md
parent_task_id: TASK-490.13.7
priority: medium
modified_files:
- Docs/API/Sync_V2_M3.md
- Docs/Design/Sync_V2_M3_Polished_Multi_Device.md
- Docs/superpowers/plans/2026-05-23-sync-v2-m3-polished-multi-device-implementation-plan.md
- tldw_Server_API/app/api/v1/endpoints/sync.py
- tldw_Server_API/app/api/v1/schemas/sync_v2_models.py
- tldw_Server_API/app/core/DB_Management/Sync_DB.py
- tldw_Server_API/app/core/Sync/v2/service.py
- tldw_Server_API/app/core/Sync/v2/store.py
- tldw_Server_API/tests/Sync/test_sync_v2_retention.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the first Stage 7 slice: a dry-run-only retention candidate calculator for Sync v2 that reports eligible envelope/tombstone/compaction/blob-reference candidates and blockers without mutating envelope, materialized, key, device, or blob state.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Dry-run API/service reports candidate counts and object references without deleting, compacting, rekeying, or garbage-collecting data.
- [x] #2 Candidate calculation is blocked by unacknowledged active devices, retention windows, tombstone hold windows, active blob references, or audit mode and returns stable blocker codes.
- [x] #3 Focused model/store/service/endpoint tests prove no mutation and cover acknowledged vs unacknowledged device behavior.
- [x] #4 Implementation plan Stage 7 Step 1/2 and Backlog notes record verification and known deferrals.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the Stage 7 dry-run-only retention candidate foundation. Added red/green coverage for unacknowledged active devices, acknowledged eligible superseded envelopes, restore-window blockers, tombstone hold-window blockers, audit-mode blockers, active blob references, redacted HTTP responses, and no-mutation guarantees. The implementation reports envelope_compaction, tombstone_prune, and blob_gc candidates with stable blocker codes, treats latest tombstones as owning the object retention chain, and never mutates envelopes, materialized state, keys, devices, or blobs.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed the first Sync v2 M3 retention/GC slice as a read-only dry run. Added service/API models, a POST /api/v1/sync/retention/dry-run endpoint, store/DB blob metadata listing, and tests covering device acknowledgment blockers, restore-window blocking, tombstone-window blocking, audit-mode blocking, active blob reference blocking, redacted responses, and no mutation. Latest tombstones now suppress older object compaction candidates until tombstone retention clears. Destructive compaction and blob deletion remain deferred to later Stage 7 work. Verification: retention tests passed with 6 passed; neighboring service/endpoint/store tests passed with 171 passed; full Sync suite passed with 405 passed and 6 warnings; Ruff passed on touched files; Bandit report /tmp/bandit_sync_v2_m3_retention_dry_run.json has 0 results.
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
