---
id: TASK-490.13.15
title: 'Sync v2 M3: Add key rotation preview and commit APIs'
status: Done
labels:
- sync
- sync-v2
- m3
- encryption
- key-rotation
- api
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
- tldw_Server_API/app/core/Sync/v2/store.py
- tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py
- tldw_Server_API/tests/Sync/test_sync_v2_models.py
- tldw_Server_API/tests/Sync/test_sync_v2_service.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement idempotent Sync v2 key rotation preview and commit flows with safe response redaction, active/superseded epoch state updates, and endpoint tests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Failing tests are written first for key rotation preview, commit idempotency, superseded source state, and safe redaction.
- [x] #2 API schemas expose preview/commit request and response shapes without returning wrapped key blobs or KDF secret metadata.
- [x] #3 Service preview reports affected active key records, next epoch, device/recovery targets, retained envelope range, and blockers for inaccessible or missing-key datasets.
- [x] #4 Commit creates an idempotent new key epoch, supersedes active prior records, persists rewrap status, rejects revoked/missing rotation sources, and does not expose key material in responses or errors.
- [x] #5 Endpoints POST /api/v1/sync/key-rotation/preview and /api/v1/sync/key-rotation/commit map errors safely and are covered by tests.
- [x] #6 Roadmap Stage 6 Step 3 is checked off and Backlog records verification evidence.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
TDD red evidence: the first focused run failed because `SyncKeyRotationPreviewRequest`/`SyncKeyRotationCommitRequest`/`SyncKeyRotationResponse` did not exist, `SyncV2Service` had no `preview_key_rotation`/`commit_key_rotation` methods, and the `/key-rotation/preview` and `/key-rotation/commit` endpoints returned 404.

Bandit initially flagged dynamic SQL `IN` construction in the DB rotation helper; the implementation was changed to fixed parameterized queries plus Python filtering/per-row updates, and Bandit then reported zero findings.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented Sync v2 M3 key rotation preview and commit APIs. Added redacted rotation request/response schemas, core redacted rotation result models, service preview/commit flows, fixed-SQL DB/store helpers for retained envelope range and atomic new-key/supersede commits, and POST /api/v1/sync/key-rotation/preview plus /api/v1/sync/key-rotation/commit. Commit uses client rotation_id for idempotency, creates the next key epoch, supersedes active source records, and keeps wrapped key blobs/KDF metadata out of preview/commit responses and safe error logs. Verification: red tests first failed for missing schemas/service/routes; new target set passed with 11 passed; focused Sync model/store/service/endpoint/security suite passed with 225 passed; full Sync suite passed with 389 passed and 6 warnings; Ruff passed on touched files; Bandit touched production scope wrote /tmp/bandit_sync_v2_m3_key_rotation_api.json with 0 results; git diff --check passed.
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
