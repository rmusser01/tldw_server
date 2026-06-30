---
id: TASK-490.9
title: 'Sync v2 M1: Implement restore preview'
status: Done
assignee:
  - '@Codex'
created_date: ''
updated_date: '2026-05-23 14:50'
labels:
  - sync
  - sync-v2
  - m1
  - restore
  - backend
dependencies: []
documentation:
  - Docs/superpowers/specs/2026-05-23-chatbook-sync-v2-roadmap-prd-design.md
  - Docs/superpowers/plans/2026-05-23-chatbook-sync-v2-m1-implementation-plan.md
parent_task_id: TASK-490
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement restore preview for clean and non-empty Chatbook profiles, including local inventory comparison, safe applies, whole-object conflicts, tombstones, attachment ref missing-blob warnings, envelope ranges, counts, and cross-user isolation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Restore preview supports empty and non-empty inventories with safe applies and explicit conflicts.
- [x] #2 Preview includes tombstones, attachment refs, missing blob warnings, per-domain counts, latest cursors, and envelope ranges.
- [x] #3 Cross-user access is blocked for datasets, envelope ranges, object summaries, conflicts, and attachment refs.
- [x] #4 Restore preview endpoint and e2e tests pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-23-chatbook-sync-v2-m1-implementation-plan.md#task-9-implement-restore-preview-and-conflict-review-data
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added restore preview local-inventory normalization helpers in `tldw_Server_API/app/core/Sync/v2/restore.py`.
- Expanded `SyncV2Service.restore_preview` to classify safe apply/noop/append actions, whole-object Notes and conversation conflicts, tombstone delete/hide actions, per-domain cursor ranges, total/domain counts, encryption status, key recovery status, and attachment ref missing-blob warnings.
- Updated restore preview API schemas and endpoint conversion so Chatbook can submit typed local inventory fingerprints.
- Changed requested cross-user/inaccessible dataset preview requests to fail closed with the existing `sync_resource_not_found` HTTP mapping.
- Reworked the Chatbook restore e2e test onto M1 domains and server-trusted encryption, asserting restore manifest/preview metadata stays free of payload bodies and wrapped recovery key material.
- RED: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_restore_preview.py -q` initially failed 6 tests because preview fields, local inventory comparison, materializer-independent attachment refs, and cross-user blocking were missing.
- Verification: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_restore_preview.py tldw_Server_API/tests/e2e/test_chatbook_sync_v2_restore.py -q` passed 7 tests.
- Verification: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py tldw_Server_API/tests/Sync/test_sync_v2_service.py tldw_Server_API/tests/Sync/test_sync_v2_attachment_refs.py tldw_Server_API/tests/Sync/test_sync_v2_models.py -q` passed 115 tests.
- Lint note: new/reworked restore files pass targeted Ruff; a broader touched-file Ruff run still reports pre-existing baseline issues in `sync.py` and `service.py` outside this slice.
- Bandit: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r tldw_Server_API/app/core/Sync/v2/restore.py tldw_Server_API/app/core/Sync/v2/service.py tldw_Server_API/app/api/v1/endpoints/sync.py tldw_Server_API/app/api/v1/schemas/sync_v2_models.py -f json -o /tmp/bandit_task_490_9_restore_preview.json` completed with zero findings.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Restore preview now returns a non-mutating M1 plan for Chatbook restores: safe apply/noop/append actions, explicit whole-object conflicts for Notes and conversation metadata, tombstone actions, attachment ref summaries with missing-blob warnings, per-domain cursor ranges/counts/latest cursors, encryption/key status, and fail-closed cross-user dataset handling. The Chatbook restore e2e now exercises the current M1 server-trusted flow.
<!-- SECTION:FINAL_SUMMARY:END -->

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
