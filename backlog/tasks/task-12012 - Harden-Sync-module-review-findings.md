---
id: TASK-12012
title: Harden Sync module review findings
status: Done
created_date: 2026-06-24 04:26
labels:
- sync
- code-review
- bugfix
priority: high
references:
- tldw_Server_API/app/core/Sync
- Review findings from current thread
modified_files:
- IMPLEMENTATION_PLAN_sync_module_review_fixes_12012.md
- tldw_Server_API/app/core/DB_Management/Sync_DB.py
- tldw_Server_API/app/core/DB_Management/media_db/media_database_impl.py
- tldw_Server_API/app/core/DB_Management/media_db/runtime/fts_ops.py
- tldw_Server_API/app/core/Sync/Sync_Client.py
- tldw_Server_API/app/core/Sync/v2/blob_store.py
- tldw_Server_API/app/core/Sync/v2/domain_adapters/chat.py
- tldw_Server_API/app/core/Sync/v2/domain_adapters/notes.py
- tldw_Server_API/app/core/Sync/v2/domain_adapters/workspaces.py
- tldw_Server_API/app/core/Sync/v2/service.py
- tldw_Server_API/app/core/Sync/v2/store.py
- tldw_Server_API/tests/MediaDB2/test_sync_client.py
- tldw_Server_API/tests/Sync/test_sync_v2_blob_store.py
- tldw_Server_API/tests/Sync/test_sync_v2_domain_adapters.py
- tldw_Server_API/tests/Sync/test_sync_v2_service.py
updated_date: 2026-06-25 02:10
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix validated current-code review findings in `tldw_Server_API/app/core/Sync`: blob chunk idempotency, Sync v2 payload size enforcement, legacy media FTS lookup, tombstone adapter handling, and legacy outbound sync marker advancement.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Duplicate blob chunks with conflicting content are rejected before staged chunk data can be overwritten.
- [x] #2 Sync v2 envelope payload data contributes to the configured payload size limit.
- [x] #3 Legacy media updates that affect title/content do not crash when refreshing FTS data.
- [x] #4 Workspace, note, and chat adapters treat Sync v2 tombstone operations as deletes for conflict detection.
- [x] #5 Legacy outbound sync advances past log rows from other clients instead of retrying the same skipped rows forever.
- [x] #6 Focused regression tests cover each fixed behavior.
- [x] #7 Targeted tests and Bandit run on the touched scope.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add regression tests that capture the reviewed failures and verify they fail against current code.
2. Patch Sync v2 blob upload handling, payload size accounting, domain adapter tombstone predicates, and legacy sync client behavior with minimal changes.
3. Re-run the focused tests and Bandit, then update this task with touched files and verification results.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Created `IMPLEMENTATION_PLAN_sync_module_review_fixes_12012.md` with staged regression, implementation, verification, and finalization steps.
Implemented the Sync review fixes and verified them.

Red checks observed before implementation:
- Focused regression run failed for duplicate blob chunk overwrite, tombstone operation-only handling, other-client outbound cursor advancement, and missing legacy Media FTS helper.
- Adjusted payload regression then failed because a divergent `envelope.payload` was accepted despite exceeding the configured limit.

Green verification:
- `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_service.py::test_push_rejects_divergent_legacy_payload_over_actual_serialized_size_limit tldw_Server_API/tests/Sync/test_sync_v2_service.py::test_blob_upload_conflicting_duplicate_chunk_does_not_overwrite_existing_chunk tldw_Server_API/tests/Sync/test_sync_v2_domain_adapters.py::test_domain_adapters_conflict_tombstone_operation_without_deleted_flag tldw_Server_API/tests/MediaDB2/test_sync_client.py::TestClientSyncEnginePush::test_push_advances_past_other_clients_changes_without_network_call tldw_Server_API/tests/MediaDB2/test_sync_client.py::TestClientSyncEnginePullApply::test_pull_and_apply_media_title_update_refreshes_fts -q` -> 7 passed.
- `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_service.py tldw_Server_API/tests/Sync/test_sync_v2_domain_adapters.py tldw_Server_API/tests/MediaDB2/test_sync_client.py -q` -> 168 passed.
- `source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/Sync tldw_Server_API/app/core/DB_Management/Sync_DB.py -f json -o /tmp/bandit_sync_review_fixes_12012.json` -> 0 findings.
Reopened after PR review comments on #2502. Follow-up fixes: add annotations to new regression tests, move legacy Media FTS lookup behind DB_Management, document blob hash helper, and make staged blob chunk publication no-overwrite under concurrent writers.
PR review follow-up implemented and verified.

Follow-up changes:
- Added type annotations to the new regression tests flagged in review.
- Moved the legacy Media FTS title/content lookup into `media_db.runtime.fts_ops` and bound it on `MediaDatabase`.
- Added a docstring for `_sha256_file`.
- Reworked staged blob chunk writes to publish via a unique temp file plus no-overwrite `os.link`, verifying any pre-existing target before accepting idempotency.
- Added a focused blob-store race regression proving a losing concurrent writer cannot overwrite an already published chunk.

Follow-up verification:
- Focused regressions: `python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_blob_store.py::test_local_blob_store_does_not_overwrite_chunk_after_publish_race tldw_Server_API/tests/Sync/test_sync_v2_service.py::test_push_rejects_divergent_legacy_payload_over_actual_serialized_size_limit tldw_Server_API/tests/Sync/test_sync_v2_service.py::test_blob_upload_conflicting_duplicate_chunk_does_not_overwrite_existing_chunk tldw_Server_API/tests/Sync/test_sync_v2_domain_adapters.py::test_domain_adapters_conflict_tombstone_operation_without_deleted_flag tldw_Server_API/tests/MediaDB2/test_sync_client.py::TestClientSyncEnginePush::test_push_advances_past_other_clients_changes_without_network_call tldw_Server_API/tests/MediaDB2/test_sync_client.py::TestClientSyncEnginePullApply::test_pull_and_apply_media_title_update_refreshes_fts -q` -> 8 passed.
- Broader touched tests: `python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_blob_store.py tldw_Server_API/tests/Sync/test_sync_v2_service.py tldw_Server_API/tests/Sync/test_sync_v2_domain_adapters.py tldw_Server_API/tests/MediaDB2/test_sync_client.py -q` -> 174 passed.
- Bandit touched app scope: `python -m bandit -r tldw_Server_API/app/core/Sync tldw_Server_API/app/core/DB_Management/Sync_DB.py tldw_Server_API/app/core/DB_Management/media_db/runtime/fts_ops.py tldw_Server_API/app/core/DB_Management/media_db/media_database_impl.py -f json -o /tmp/bandit_sync_review_fixes_12012_pr_comments.json` -> 0 findings.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Hardened the Sync module review findings by preventing conflicting duplicate blob chunks from overwriting staged data, explicitly accounting for divergent inline envelope payload bytes, using `tombstone` as the Sync v2 delete operation in domain adapters, restoring legacy Media FTS lookup for title/content updates, and advancing legacy outbound sync past log rows created by other local clients. Added focused regression tests for the fixed behaviors and verified the touched test files plus Bandit on the touched Sync scope.
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
