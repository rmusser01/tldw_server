---
id: TASK-220
title: Expose Sync v2 API endpoints
status: Done
assignee: []
created_date: '2026-05-10 04:56'
updated_date: '2026-05-10 05:41'
labels:
  - sync
  - api
  - endpoints
dependencies:
  - TASK-217
references:
  - tldw_Server_API/app/api/v1/endpoints/sync.py
  - tldw_Server_API/app/core/Sync/v2/service.py
documentation:
  - >-
    Docs/superpowers/plans/2026-05-10-chatbook-sync-engine-implementation-plan.md
  - Docs/superpowers/specs/2026-05-10-chatbook-sync-engine-prd-design.md
  - tldw_Server_API/app/api/v1/schemas/sync_v2_models.py
priority: medium
---

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Sync v2 endpoints are exposed under /api/v1/sync as thin wrappers around SyncV2Service without duplicating service logic.
- [x] #2 Endpoint tests cover capabilities, device register, dataset enroll, restore manifest filters, push idempotency and unsupported adapter versions, pull filters/echo/paging, conflicts list/resolve, attachments feature-detect response, and key recovery bundle.
- [x] #3 Legacy /send and /get behavior is deliberately preserved or explicitly tested according to the chosen compatibility policy.
- [x] #4 Focused endpoint/error-mapping pytest passes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add endpoint-level tests in tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py using a small FastAPI app with dependency overrides for user and SyncV2Service.
2. Cover the required Sync v2 route behavior through HTTP requests, including sanitized error responses and legacy /send + /get compatibility policy.
3. Implement thin Sync v2 route wrappers in tldw_Server_API/app/api/v1/endpoints/sync.py, reusing the existing router mounted at /api/v1/sync.
4. Add only minimal SyncV2Service helper methods for conflicts and key recovery if endpoint wiring needs service-owned access to store operations.
5. Verify with focused endpoint/error tests, focused Sync v2 service/store/model/security tests if service changes are made, git diff --check, and Bandit on touched production files.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Starting Task 4 endpoint implementation from branch codex/sync-v2-schemas. MCP task_view did not see TASK-220, but backlog CLI resolves the task file in this worktree; using CLI fallback for task updates.

Implemented Sync v2 endpoint wrappers under the existing /api/v1/sync router. Added endpoint coverage for capabilities, register, enroll, restore filters, push idempotency/rejections, pull filters/echo/paging, conflicts list/resolve, attachment feature-detect response, key recovery metadata, safe error details, and legacy /send + /get compatibility. Added minimal SyncV2Service helpers for conflict listing/resolution and encrypted key recovery storage; attachment capability now reports unsupported until persistence exists.
Verification: python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py tldw_Server_API/tests/Sync/test_sync_error_mapping.py -v => 13 passed. python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_service.py tldw_Server_API/tests/Sync/test_sync_v2_security.py tldw_Server_API/tests/Sync/test_sync_v2_models.py tldw_Server_API/tests/Sync/test_sync_v2_store.py -q => 61 passed. git diff --check => clean. Bandit on touched production files wrote /tmp/bandit_sync_v2_endpoints.json with 0 results.

Controller spec-fix pass: added the Sync v2 service dependency to the attachments feature-detect stub so the route remains auth/service-gated. Extended endpoint coverage with loguru log capture proving safe Sync v2 error logging does not emit wrapped keys, encrypted payload values, or known plaintext. Focused endpoint test run: python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py -q -> 11 passed.

Code-quality review fix pass: persisted submitted conflict resolution envelopes before marking conflicts resolved; carried dataset encryption_policy through pull responses instead of hard-coding client_private_v1; made SyncPushRequest.device_id required at the API boundary; made /attachments feature detection ignore strict body shape while remaining service-gated; converted new Sync v2 endpoint handlers to sync functions so FastAPI runs blocking service/store work in its threadpool. Verification: python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py tldw_Server_API/tests/Sync/test_sync_error_mapping.py tldw_Server_API/tests/Sync/test_sync_v2_service.py tldw_Server_API/tests/Sync/test_sync_v2_security.py tldw_Server_API/tests/Sync/test_sync_v2_models.py tldw_Server_API/tests/Sync/test_sync_v2_store.py -q -> 80 passed.

Second quality re-review fix pass: SyncV2Service now requires registered, non-revoked user-owned devices for push, pull, conflict resolution attribution/resolution envelopes, and key recovery metadata. Conflict resolution now maps private resolution-envelope payload validation failures to SyncStoreError so endpoints return a safe client error instead of a generic 500. Added regression coverage for unregistered device rejection across device-scoped service paths and invalid private resolution envelope handling. Verification: python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py tldw_Server_API/tests/Sync/test_sync_error_mapping.py tldw_Server_API/tests/Sync/test_sync_v2_service.py tldw_Server_API/tests/Sync/test_sync_v2_security.py tldw_Server_API/tests/Sync/test_sync_v2_models.py tldw_Server_API/tests/Sync/test_sync_v2_store.py -q -> 83 passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Exposed the first Sync v2 API surface under /api/v1/sync while preserving legacy /send and /get. Endpoint handlers stay thin around SyncV2Service, with safe HTTP error mapping that avoids returning encrypted payloads, wrapped keys, or known plaintext. Attachment upload is intentionally feature-detectable and returns 501 until persistence is implemented.
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
