---
id: TASK-490.13.2
title: 'Sync v2 M3: Device lifecycle and acknowledgments'
status: Done
labels:
- sync
- sync-v2
- m3
- devices
priority: medium
parent_task_id: TASK-490.13
documentation:
- Docs/Design/Sync_V2_M3_Polished_Multi_Device.md
- Docs/API/Sync_V2_M3.md
- Docs/superpowers/plans/2026-05-23-sync-v2-m3-polished-multi-device-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make Sync v2 devices user-manageable and add authorization plus per-device acknowledgment primitives required for background sync, retention, and safe revocation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Users can list, update, pause, authorize, and revoke registered devices with profile/domain status details.
- [x] #2 Revoked devices fail closed across push, pull, restore, blob, conflict, repair, and key recovery APIs while historical envelopes remain auditable.
- [x] #3 Per-device domain and blob acknowledgments are persisted idempotently and exposed for later retention/GC decisions.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added Sync v2 device lifecycle persistence: device status/user labels/authorization metadata/revocation reasons, authorization records, and idempotent per-device domain/blob acknowledgment tables.
- Added store and service methods for listing/updating devices, authorization request/approval, pause/resume/revoke, and acknowledgment submission.
- Enforced active-device checks for device-scoped push/pull/conflict/blob/restore/repair flows while preserving no-device restore/front-end mode.
- Added `/api/v1/sync/devices`, `/api/v1/sync/device-authorizations`, and `/api/v1/sync/device-acknowledgments` API wiring plus optional device scoping for restore preview/repair.
- Added store, service, and endpoint regression coverage for pending authorization, revocation, hidden-by-default revoked devices, audit visibility, and acknowledgment idempotency.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed M3 device lifecycle and acknowledgment primitives for Sync v2.

Verification:
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_store.py tldw_Server_API/tests/Sync/test_sync_v2_service.py tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py tldw_Server_API/tests/Sync/test_sync_v2_models.py -v` -> 165 passed, 6 warnings.
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m ruff check tldw_Server_API/app/core/DB_Management/Sync_DB.py tldw_Server_API/app/core/Sync/v2/models.py tldw_Server_API/app/core/Sync/v2/store.py tldw_Server_API/app/core/Sync/v2/service.py tldw_Server_API/app/api/v1/schemas/sync_v2_models.py tldw_Server_API/app/api/v1/endpoints/sync.py tldw_Server_API/tests/Sync/test_sync_v2_store.py tldw_Server_API/tests/Sync/test_sync_v2_service.py tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py` -> All checks passed.
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r tldw_Server_API/app/core/DB_Management/Sync_DB.py tldw_Server_API/app/core/Sync/v2/models.py tldw_Server_API/app/core/Sync/v2/store.py tldw_Server_API/app/core/Sync/v2/service.py tldw_Server_API/app/api/v1/schemas/sync_v2_models.py tldw_Server_API/app/api/v1/endpoints/sync.py -f json -o /tmp/bandit_sync_v2_m3_devices.json` -> 0 findings.
- `git diff --check` -> clean.

Known skips/blockers: none.
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
