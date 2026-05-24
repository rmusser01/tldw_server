---
id: TASK-490.13.7.2
title: 'Sync v2 M3: Diagnostics endpoint foundation'
status: Done
labels:
- sync
- sync-v2
- m3
- diagnostics
- observability
priority: medium
parent_task_id: TASK-490.13.7
documentation:
- Docs/Design/Sync_V2_M3_Polished_Multi_Device.md
- Docs/API/Sync_V2_M3.md
- Docs/superpowers/plans/2026-05-23-sync-v2-m3-polished-multi-device-implementation-plan.md
modified_files:
- Docs/API/Sync_V2_M3.md
- Docs/superpowers/plans/2026-05-23-sync-v2-m3-polished-multi-device-implementation-plan.md
- tldw_Server_API/app/api/v1/endpoints/sync.py
- tldw_Server_API/app/api/v1/schemas/sync_v2_models.py
- tldw_Server_API/app/core/Sync/v2/service.py
- tldw_Server_API/tests/Sync/test_sync_v2_diagnostics.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the first redacted Sync v2 diagnostics endpoint for Stage 7 observability, focused on dataset/domain counts, conflict counts, device lag, blob/upload pressure, key summary, and a retention dry-run summary without exposing payloads or secrets.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 GET /api/v1/sync/diagnostics returns redacted dataset health for the authenticated user's dataset.
- [x] #2 Diagnostics include domain envelope counts, object counts, conflict count, active upload count, blob object count/bytes, per-device domain lag, key status summary, and retention dry-run summary.
- [x] #3 Diagnostics enforce dataset/device access checks and do not include payloads, ciphertext, wrapped keys, KDF salts, or recovery secrets.
- [x] #4 Focused service/endpoint tests and docs cover the implemented diagnostic contract and redaction behavior.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented the first redacted Sync v2 M3 diagnostics endpoint. The GET /api/v1/sync/diagnostics route now returns dataset/domain envelope and object counts, failed apply counts, unresolved conflict counts, profile-level device lag, blob/upload pressure, key status summary, and a retention dry-run summary. The response intentionally omits payloads, ciphertext, blob storage keys, wrapped keys, KDF salts, recovery hints, and conflict metadata.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed the Stage 7 diagnostics endpoint foundation. Added service dataclasses/aggregation, Pydantic response schemas, GET /api/v1/sync/diagnostics, API documentation, and focused endpoint tests for redacted dataset health and cross-user access denial. Verification: diagnostics tests passed with 2 passed; neighboring diagnostics/retention/endpoint/service tests passed with 120 passed; full Sync suite passed with 407 passed and 6 warnings; Ruff passed on touched diagnostics files; Bandit report /tmp/bandit_sync_v2_m3_diagnostics.json has 0 results.
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
