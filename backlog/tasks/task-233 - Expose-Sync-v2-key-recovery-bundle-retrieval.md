---
id: TASK-233
title: Expose Sync v2 key recovery bundle retrieval
status: Done
assignee: []
created_date: '2026-05-10 16:14'
updated_date: '2026-05-10 16:17'
labels:
  - sync
  - server
  - security
dependencies: []
priority: medium
---

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Authenticated users can retrieve active key recovery bundle records for a dataset they can access
- [x] #2 Retrieval supports optional device and key-purpose filters and returns wrapped key material plus KDF metadata only through the dedicated key endpoint
- [x] #3 Restore manifests remain metadata-only and do not include wrapped key blobs or KDF metadata
- [x] #4 Missing or inaccessible datasets return the existing safe sync error response without leaking key material
- [x] #5 Focused Sync v2 service and endpoint tests Bandit and diff checks pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing service and endpoint tests for listing key recovery bundles with safe filtering and no manifest leakage. 2. Add response schemas for recovery bundle records and list response. 3. Implement SyncV2Service list_key_recovery_bundles and a GET /api/v1/sync/keys/recovery-bundle endpoint. 4. Update Sync v2 API docs. 5. Run focused tests Bandit and diff checks, then update Backlog and commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented Sync v2 key recovery bundle retrieval with a service method that verifies dataset ownership, filters by optional device_id and key_purpose, and returns only non-revoked records. Added a GET /api/v1/sync/keys/recovery-bundle endpoint with response schemas that include wrapped_key_blob and kdf_metadata only on the dedicated key endpoint. Restore manifests remain metadata-only. Updated Docs/API/sync-v2.md. Verification: initial red run failed on missing service method and 405 GET route; focused retrieval tests passed; broader Sync v2 service/endpoint/restore tests passed with 54 tests; Bandit JSON had empty errors/results; git diff --check clean.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added authenticated Sync v2 key recovery bundle retrieval. The server now lists active wrapped key records for accessible datasets with optional device/key-purpose filters, keeps restore manifests free of wrapped key material, and documents the dedicated retrieval endpoint needed by new devices during recovery-based restore.
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
