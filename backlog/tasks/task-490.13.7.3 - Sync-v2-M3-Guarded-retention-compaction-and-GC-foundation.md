---
id: TASK-490.13.7.3
title: 'Sync v2 M3: Guarded retention compaction and GC foundation'
status: Done
labels:
- sync
- sync-v2
- m3
- retention
- gc
priority: medium
parent_task_id: TASK-490.13.7
documentation:
- Docs/Design/Sync_V2_M3_Polished_Multi_Device.md
- Docs/API/Sync_V2_M3.md
- Docs/superpowers/plans/2026-05-23-sync-v2-m3-polished-multi-device-implementation-plan.md
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
Implement a conservative apply path for Sync v2 retention candidates that refuses blocked candidates, records domain compaction checkpoints without deleting envelope audit logs, and soft-deletes eligible blob metadata only after dry-run blockers clear.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 POST /api/v1/sync/retention/compact requires explicit confirmation and returns a dry-run/error result without mutation otherwise.
- [x] #2 Compaction applies only unblocked envelope/tombstone candidates and records per-domain compaction checkpoints while preserving the append-only envelope log.
- [x] #3 Blob GC applies only unblocked blob candidates and soft-deletes available blob metadata so it no longer appears as restore-available.
- [x] #4 Focused tests prove blocked candidates do not mutate state, eligible candidates update compaction/blob metadata, and API responses are redacted.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented the conservative guarded retention apply foundation. POST /api/v1/sync/retention/compact re-runs dry-run blockers with audit mode off, requires explicit confirmation, refuses selected blocked candidates without mutation, records per-domain compaction checkpoints without deleting envelope audit logs, and soft-deletes eligible blob metadata without physically deleting blob bytes.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed the guarded retention compaction/GC foundation. Added store/DB methods for domain compaction checkpoints and blob metadata soft-delete, service/apply response models, POST /api/v1/sync/retention/compact, API/design docs, and focused tests for confirmation-required no-op, blocked-candidate no-op, non-destructive domain checkpointing, eligible blob metadata soft-delete, and redacted endpoint responses. Physical blob byte deletion and audit-event search remain later work. Verification: retention tests passed with 11 passed; neighboring retention/endpoint/service/store/diagnostics tests passed with 178 passed; full Sync suite passed with 412 passed and 6 warnings; Ruff passed on touched files; Bandit report /tmp/bandit_sync_v2_m3_retention_compact.json has 0 results.
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
