---
id: TASK-490.13.14
title: 'Sync v2 M3: Add key epoch storage state'
status: Done
labels:
- sync
- sync-v2
- m3
- encryption
- storage
- tdd
priority: medium
parent_task_id: TASK-490.13
documentation:
- Docs/Design/Sync_V2_M3_Polished_Multi_Device.md
- Docs/API/Sync_V2_M3.md
- Docs/superpowers/plans/2026-05-23-sync-v2-m3-polished-multi-device-implementation-plan.md
modified_files:
- Docs/superpowers/plans/2026-05-23-sync-v2-m3-polished-multi-device-implementation-plan.md
- tldw_Server_API/app/api/v1/endpoints/sync.py
- tldw_Server_API/app/api/v1/schemas/sync_v2_models.py
- tldw_Server_API/app/core/DB_Management/Sync_DB.py
- tldw_Server_API/app/core/Sync/v2/models.py
- tldw_Server_API/app/core/Sync/v2/service.py
- tldw_Server_API/tests/Sync/test_sync_v2_models.py
- tldw_Server_API/tests/Sync/test_sync_v2_service.py
- tldw_Server_API/tests/Sync/test_sync_v2_store.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Extend Sync v2 key recovery records with policy, epoch, active/superseded sequence metadata, wrapped-for state, and rewrap status needed before key-rotation APIs are added.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Failing tests are written first for storing/listing key epoch and rotation-state metadata.
- [x] #2 Sync key record create/read models include encryption_policy, key_epoch, active_from_server_sequence, superseded_at, wrapped_for, and rewrap_status with safe defaults for existing records.
- [x] #3 SQLite schema migration preserves existing key records and initializes safe default epoch/policy metadata.
- [x] #4 Store/service key recovery paths persist and return the new fields while preserving existing validation and redaction behavior.
- [x] #5 Roadmap Stage 6 Step 2 is checked off and Backlog records verification evidence.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
TDD red evidence: the first focused run failed because `SyncKeyRecoveryBundleRequest` lacked `encryption_policy`, invalid metadata was ignored, `SyncKeyRecordCreate` rejected `encryption_policy`, legacy key-record columns were absent, and `SyncV2Service.store_key_recovery_bundle` did not accept `encryption_policy`.

Migration ordering was adjusted after the green pass initially exposed schema replay creating indexes against legacy `sync_key_records` before adding the new columns.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented Sync v2 M3 key epoch storage state. Key recovery records now carry encryption_policy, key_epoch, active_from_server_sequence, superseded_at, wrapped_for, and rewrap_status through API schemas, core models, service calls, endpoint exports, store persistence, idempotency fingerprints, and SQLite/PostgreSQL schema definitions. Legacy SQLite key-record tables are migrated before schema replay so existing rows receive safe server_trusted_v1 epoch-1 recovery defaults. Verification: red tests first failed for missing fields/migration/service support; targeted red set passed with 9 passed; focused Sync model/store/service/security suite passed with 194 passed; full Sync suite passed with 378 passed and 6 warnings; Ruff passed on touched files; Bandit touched production scope wrote /tmp/bandit_sync_v2_m3_key_epoch_storage.json with 0 results; git diff --check passed.
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
