---
id: TASK-490.3
title: 'Sync v2 M1: Add profile bootstrap and status'
status: Done
assignee:
- '@Codex'
labels:
- sync
- sync-v2
- m1
- api
- backend
priority: high
parent_task_id: TASK-490
documentation:
- Docs/superpowers/specs/2026-05-23-chatbook-sync-v2-roadmap-prd-design.md
- Docs/superpowers/plans/2026-05-23-chatbook-sync-v2-m1-implementation-plan.md
modified_files:
- tldw_Server_API/app/core/Sync/v2/profile.py
- tldw_Server_API/app/core/Sync/v2/service.py
- tldw_Server_API/app/core/Sync/v2/factory.py
- tldw_Server_API/app/core/Sync/v2/security.py
- tldw_Server_API/app/api/v1/endpoints/sync.py
- tldw_Server_API/app/api/v1/schemas/sync_v2_models.py
- tldw_Server_API/tests/Sync/test_sync_v2_profile_bootstrap.py
- tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py
- tldw_Server_API/tests/Sync/test_sync_v2_server_trusted_encryption.py
- tldw_Server_API/tests/Sync/test_sync_v2_domain_adapters.py
- tldw_Server_API/tests/Sync/test_sync_v2_media_compat.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add read-only profile status and explicit profile bootstrap for Chatbook server-connected modes, including default personal dataset creation, device/profile registration, per-domain status, protocol version, and honest server_trusted_v1 at-rest coverage reporting.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 GET /api/v1/sync/profile is read-only and never creates durable sync state.
- [x] #2 POST /api/v1/sync/profile/bootstrap idempotently registers the device/profile and creates or returns the default personal dataset.
- [x] #3 Profile/status responses include protocol version, domains, cursors, encryption posture, device status, per-domain counts, conflicts, and apply health.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-23-chatbook-sync-v2-m1-implementation-plan.md#task-3-add-profile-bootstrap-and-status
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- TDD red:
  `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_profile_bootstrap.py tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py tldw_Server_API/tests/Sync/test_sync_v2_server_trusted_encryption.py -q`
  failed during collection because `server_trusted_encryption_status_from_config` did not exist yet.
- Added `SyncV2ServerTrustedEncryptionStatus` and deterministic readiness evaluation from `SYNC_V2_AT_REST_ENCRYPTION_MODE`, `SYNC_V2_SERVER_TRUSTED_ENABLED`, and `AUTH_MODE`. `encrypted_volume` and `managed_storage` can be ready when enabled; `development_unencrypted` is reported explicitly but not ready.
- Added `SyncV2ProfileManager` plus `SyncV2Service.profile`, `bootstrap_profile`, and `profile_status`. Bootstrap fails closed before durable writes when `server_trusted_v1` is not ready.
- Added `GET /api/v1/sync/profile` and `POST /api/v1/sync/profile/bootstrap`, and updated Task 3 endpoint tests to the locked M1 contract while preserving lower-level register/enroll route coverage.
- Review follow-up TDD red:
  focused regressions for bounded status aggregation, fresh read-only profile storage creation, `/datasets/enroll` readiness, omitted-device bootstrap idempotency, unknown device mode serialization, and M1 registry assumptions failed as expected before fixes: 8 failed, 5 warnings.
- Fixed profile/status aggregation to page all domain envelopes so current server cursor, pending/failed apply counts, and last apply status are not stale when the scan limit is smaller than the dataset.
- Added a read-only profile dependency path that returns an empty profile from deterministic capability settings without opening or creating Sync v2 storage for fresh users.
- Added the same server_trusted_v1 readiness gate to direct dataset enrollment, and kept omitted-device bootstrap spec-compliant: an optional `client_profile_id` reuses an active matching device, otherwise the server generates and returns a `device_id` that the client must persist for later idempotent operations.
- Normalized unknown persisted device modes to `None` for profile responses instead of allowing schema serialization failures.
- Updated touched registry/media compatibility tests to the M1 domain contract: default registry exposes only M1 domains, legacy media is excluded from default sync, chat service conflict coverage uses `chat.message`, and legacy media dataset enrollment is rejected by the M1 service.
- Focused review regressions passed after the first review fix: bounded status aggregation,
  fresh read-only profile storage behavior, `/datasets/enroll` readiness,
  omitted-device bootstrap reuse with `client_profile_id`, unknown device mode
  serialization, and M1 registry assumptions.
- Re-review TDD red:
  `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_profile_bootstrap.py::test_bootstrap_without_device_id_and_profile_id_generates_device tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py::test_profile_bootstrap_endpoint_without_device_or_profile_generates_device -q`
  failed as expected before the production fix: 2 failed, 5 warnings, because missing `client_profile_id` was still rejected.
- Re-review focused green:
  `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_profile_bootstrap.py::test_bootstrap_without_device_id_and_profile_id_generates_device tldw_Server_API/tests/Sync/test_sync_v2_profile_bootstrap.py::test_bootstrap_without_device_id_reuses_device_by_client_profile_id tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py::test_profile_bootstrap_endpoint_without_device_or_profile_generates_device tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py::test_profile_bootstrap_endpoint_reuses_omitted_device_by_client_profile_id -q`
  passed: 4 passed, 5 warnings.
- Final verification:
  `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_profile_bootstrap.py tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py tldw_Server_API/tests/Sync/test_sync_v2_server_trusted_encryption.py -q`
  passed: 23 passed, 5 warnings.
- Task 2 smoke:
  `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_models.py tldw_Server_API/tests/Sync/test_sync_v2_store.py tldw_Server_API/tests/Sync/test_sync_v2_object_state.py -q`
  passed: 69 passed, 5 warnings.
- Updated registry/media tests:
  `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_domain_adapters.py tldw_Server_API/tests/Sync/test_sync_v2_media_compat.py -q`
  passed: 47 passed, 5 warnings.
- Bandit:
  `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r tldw_Server_API/app/core/Sync/v2/profile.py tldw_Server_API/app/core/Sync/v2/service.py tldw_Server_API/app/core/Sync/v2/factory.py tldw_Server_API/app/core/Sync/v2/security.py tldw_Server_API/app/api/v1/endpoints/sync.py tldw_Server_API/app/api/v1/schemas/sync_v2_models.py -f json -o /tmp/bandit_sync_v2_task3.json`
  completed with `results: []`, `errors: []`, and 3 existing `nosec` skips.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
The profile endpoint is read-only for fresh users without creating durable Sync v2 storage, bootstrap fails closed when server_trusted_v1 at-rest readiness is not configured, direct dataset enrollment is now protected by the same readiness gate, and omitted-device bootstrap remains spec-compliant: optional `client_profile_id` reuses an active matching device, while omitting both IDs generates a registered `device_id` that the client must persist. Profile/status responses compute cursor and per-domain apply health beyond bounded scan limits, legacy/internal device modes serialize safely, and touched registry/media tests are aligned to the locked M1 domain contract. No remaining Task 3 blockers are known.
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
