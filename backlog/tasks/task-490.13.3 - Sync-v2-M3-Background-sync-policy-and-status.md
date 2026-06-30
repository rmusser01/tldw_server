---
id: TASK-490.13.3
title: 'Sync v2 M3: Background sync policy and status'
status: Done
labels:
- sync
- sync-v2
- m3
- background-sync
priority: medium
parent_task_id: TASK-490.13
documentation:
- Docs/Design/Sync_V2_M3_Polished_Multi_Device.md
- Docs/API/Sync_V2_M3.md
- Docs/superpowers/plans/2026-05-23-sync-v2-m3-polished-multi-device-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add server-side policy, status, pause/resume intent, and advisory lease primitives for Chatbook-run background sync without replacing Sync v2 idempotent push/pull/blob APIs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Clients can fetch background sync policy hints and store pause/resume intent per dataset/device.
- [x] #2 Advisory per-device sync leases prevent overlapping local workers without weakening idempotency guarantees.
- [x] #3 Profile and per-domain background status reports last success, lag, conflicts, replayable failures, blob completeness, and quota pressure.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added background policy persistence keyed by dataset/device, including pause intent, metered-network guidance, batch/blob limits, maintenance-window metadata, and pending-local-change signal.
- Added advisory background leases keyed by dataset/device. Same-lease refreshes extend the lease; competing active leases return the existing lease as `held_by_other` with `acquired: false`.
- Added profile/domain background status aggregation for cursor lag, last push/pull timestamps, unresolved conflicts, replayable failures, blob completeness counters, restore completeness, and quota pressure.
- Added API wiring for `GET/PATCH /api/v1/sync/background-policy`, `POST /api/v1/sync/background-leases`, and `GET /api/v1/sync/background-status`.
- Re-enabling background sync clears the stored `paused_reason` so resumed devices do not keep stale pause context.
- Revoked or otherwise inactive devices fail closed through the same active-device gate used by push/pull/blob/restore flows.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed M3 background sync policy/status and advisory lease primitives for Sync v2.

Verification:
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_models.py tldw_Server_API/tests/Sync/test_sync_v2_store.py tldw_Server_API/tests/Sync/test_sync_v2_service.py tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py -v` -> 170 passed, 6 warnings.
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m ruff check tldw_Server_API/app/core/DB_Management/Sync_DB.py tldw_Server_API/app/core/Sync/v2/models.py tldw_Server_API/app/core/Sync/v2/store.py tldw_Server_API/app/core/Sync/v2/service.py tldw_Server_API/app/api/v1/schemas/sync_v2_models.py tldw_Server_API/app/api/v1/endpoints/sync.py tldw_Server_API/tests/Sync/test_sync_v2_models.py tldw_Server_API/tests/Sync/test_sync_v2_store.py tldw_Server_API/tests/Sync/test_sync_v2_service.py tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py` -> All checks passed.
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r tldw_Server_API/app/core/DB_Management/Sync_DB.py tldw_Server_API/app/core/Sync/v2/models.py tldw_Server_API/app/core/Sync/v2/store.py tldw_Server_API/app/core/Sync/v2/service.py tldw_Server_API/app/api/v1/schemas/sync_v2_models.py tldw_Server_API/app/api/v1/endpoints/sync.py -f json -o /tmp/bandit_sync_v2_m3_background.json` -> 0 findings.
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
