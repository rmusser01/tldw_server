---
id: TASK-397
title: Add Sync v2 attachment persistence
status: Done
assignee: []
created_date: '2026-05-16 01:12'
labels:
  - sync
  - server
  - attachments
dependencies: []
references:
  - Docs/API/sync-v2.md
  - Docs/Design/Sync-Engine.md
  - tldw_Server_API/app/api/v1/endpoints/sync.py
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Persist small encrypted Sync v2 attachment payloads server-side behind /api/v1/sync/attachments so Chatbook can upload ciphertext attachments for restore hydration while keeping large binary replication out of scope.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 POST /api/v1/sync/attachments stores or deduplicates encrypted attachment payloads for an accessible dataset and returns stored metadata.
- [x] #2 Attachment persistence enforces dataset access and enrolled domain validation and rejects unsupported plaintext or oversize payloads without leaking ciphertext in errors or logs.
- [x] #3 Restore manifests include attachment availability and size-class summaries from persisted attachments.
- [x] #4 Focused Sync v2 store service endpoint tests plus Bandit and diff checks pass.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Notes

<!-- SECTION:NOTES:BEGIN -->
- Added `Docs/superpowers/plans/2026-05-16-sync-v2-attachment-persistence.md`.
- Verification: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_store.py tldw_Server_API/tests/Sync/test_sync_v2_service.py tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py -q` passed with 86 tests.
- Verification: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -q -r tldw_Server_API/app/api/v1/endpoints/sync.py tldw_Server_API/app/core/DB_Management/Sync_DB.py tldw_Server_API/app/core/Sync/v2/models.py tldw_Server_API/app/core/Sync/v2/service.py tldw_Server_API/app/core/Sync/v2/store.py` passed; Bandit printed existing `nosec encountered` warnings for annotated SQL lines but returned success.
- Verification: `git diff --check` passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented server-side Sync v2 attachment persistence for small `client_private_v1` ciphertext payloads. The endpoint now validates sanitized request bodies, stores or idempotently deduplicates attachments, enforces dataset ownership and enrolled-domain checks, rejects plaintext-policy and oversize uploads without echoing ciphertext, and restore manifests summarize persisted attachment availability and size classes. Updated Sync v2 API/design docs and focused store/service/endpoint coverage.
<!-- SECTION:FINAL_SUMMARY:END -->
