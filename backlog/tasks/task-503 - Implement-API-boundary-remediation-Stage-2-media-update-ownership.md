---
id: TASK-503
title: Implement API boundary remediation Stage 2 media update ownership
status: Done
labels:
- api-boundary
- media-db
- stage-2
priority: High
documentation:
- Docs/superpowers/specs/2026-06-01-api-boundary-remediation-design.md
modified_files:
- tldw_Server_API/app/core/DB_Management/media_db/runtime/media_item_update_ops.py
- tldw_Server_API/app/core/DB_Management/media_db/media_database_impl.py
- tldw_Server_API/app/api/v1/endpoints/media/item.py
- tldw_Server_API/app/api/v1/utils/http_errors.py
- tldw_Server_API/tests/DB_Management/test_media_db_media_item_update_ops.py
- tldw_Server_API/tests/Media/test_media_item_endpoint_error_mapping.py
- tldw_Server_API/tests/Media_Ingestion_Modification/test_media_versions.py
- tldw_Server_API/tests/Utils/test_api_v1_utils.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Stage 2 of the accepted API boundary remediation plan: move user-facing media item update invariants and side effects from the API endpoint into a public MediaDatabase operation, then thin the endpoint to delegation/error mapping.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 MediaDatabase exposes a public apply_media_item_update operation that owns versioning, optimistic locking, content hash/chunking/vector side effects, sync logging, document-version creation, FTS refresh, and collection-stale hooks.
- [x] #2 update_media_item endpoint delegates non-empty updates to MediaDatabase.apply_media_item_update and retains only no-op response, error mapping, RAG invalidation, and detail response shaping.
- [x] #3 DB unit tests cover missing media, optimistic conflicts, metadata-only updates, changed content updates, and identical content versioning behavior.
- [x] #4 Endpoint tests prove delegation fields/prompt/analysis separation, private helper removal, and effect-driven RAG invalidation.
- [x] #5 Focused pytest and Bandit verification results are recorded in the task final summary.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-01-api-boundary-remediation-implementation-plan.md#stage-2-media-db-update-ownership
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented Stage 2 media update ownership. Added MediaDatabase.apply_media_item_update as the public DB-layer operation for user-facing media updates; it owns optimistic locking, version increments, content hashing, chunk/vector stale flags, document-version creation, FTS refresh, sync logging, and collection stale hooks. Thinned update_media_item so non-empty row updates delegate to the DB operation, while the endpoint retains no-op response handling, HTTP error translation, effect-driven RAG invalidation, and response shaping. Added a shared map_db_error_to_http not_found_detail option so promoted 404 InputError responses can keep sanitized endpoint details.

Verification:
- source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/DB_Management/test_media_db_media_item_update_ops.py tldw_Server_API/tests/Media/test_media_item_endpoint_error_mapping.py tldw_Server_API/tests/Media_Ingestion_Modification/test_media_versions.py tldw_Server_API/tests/Utils/test_api_v1_utils.py -q => 123 passed, 10 warnings.
- source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/DB_Management/media_db/runtime/media_item_update_ops.py tldw_Server_API/app/core/DB_Management/media_db/media_database_impl.py tldw_Server_API/app/api/v1/endpoints/media/item.py tldw_Server_API/app/api/v1/utils/http_errors.py -f json -o /tmp/bandit_api_boundary_stage2.json => exit 0, JSON results empty.

Reviews:
- Spec compliance re-review: PASS.
- Code-quality review: PASS.

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
