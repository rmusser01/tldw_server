---
id: TASK-490.13.4
title: 'Sync v2 M3: Workspace dataset foundation'
status: Done
labels:
- sync
- sync-v2
- m3
- workspace
priority: medium
parent_task_id: TASK-490.13
documentation:
- Docs/Design/Sync_V2_M3_Polished_Multi_Device.md
- Docs/API/Sync_V2_M3.md
- Docs/superpowers/plans/2026-05-23-sync-v2-m3-polished-multi-device-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Introduce workspace-scoped Sync v2 datasets with permission and key-policy boundaries before enabling broad collaborative content sync.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Dataset scope supports personal and workspace datasets with fail-closed workspace membership checks.
- [x] #2 All dataset-scoped sync, blob, restore, conflict, repair, and key APIs re-check workspace permission for workspace datasets.
- [x] #3 Initial workspace domains are limited to workspace metadata/source references until collaborative Notes/Chat semantics are separately designed.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Workspace auth boundary review before production edits:
- Existing workspace REST endpoints (`tldw_Server_API/app/api/v1/endpoints/workspaces.py`) gate on the current user's ChaChaNotes DB plus workspace existence (`_require_workspace`), but do not expose a reusable per-workspace membership or sync-permission helper.
- Existing sharing code (`SharedWorkspaceRepo` / `SharedWorkspaceDBResolver`) validates share records and access levels for shared workspace access, but it is share-token/scope oriented rather than a direct Sync v2 dataset membership API.
- Stage 4 will add a narrow Sync v2 workspace access checker boundary: personal datasets continue using `owner_user_id` checks; workspace datasets require `workspace_id` and fail closed unless the checker explicitly grants the requesting user sync permission.
- The first workspace dataset domains remain limited to `workspaces.workspace` and `workspaces.source_ref`.

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the Sync v2 M3 workspace dataset foundation.

- Added explicit `workspaces.workspace` and `workspaces.source_ref` domains, operations, capabilities, schema validation, registry wiring, and workspace domain adapter coverage.
- Added scope-aware dataset contract validation so personal datasets remain M1-only and workspace datasets require `workspace_id` plus the workspace metadata/source-ref domains.
- Added a fail-closed `workspace_access_checker` boundary in `SyncV2Service`; all dataset-scoped sync, background, device authorization, restore, repair, conflict, key, and blob entrypoints now use centralized dataset access checks.
- Wired the default service factory to grant sync access only when the user's local ChaChaNotes DB contains the workspace, leaving a narrow replacement boundary for a future durable workspace membership service.
- Relaxed owner-bound internal device/blob checks for workspace-scoped datasets after service-level workspace permission has been established, so authorized workspace members are not blocked by the dataset creator's `owner_user_id`.

Verification:
- RED: targeted tests failed first for missing workspace domains, unsupported workspace dataset contracts, missing registry support, and owner-bound workspace member access.
- PASS: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_models.py tldw_Server_API/tests/Sync/test_sync_v2_store.py tldw_Server_API/tests/Sync/test_sync_v2_service.py tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py tldw_Server_API/tests/Sync/test_sync_v2_domain_adapters.py tldw_Server_API/tests/Sync/test_sync_v2_media_compat.py tldw_Server_API/tests/Sync/test_sync_v2_attachment_refs.py tldw_Server_API/tests/Sync/test_sync_v2_restore_preview.py tldw_Server_API/tests/Sync/test_sync_v2_profile_bootstrap.py -q` (`257 passed, 6 warnings`).
- PASS: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m ruff check ...` on touched production and test files.
- PASS: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r ... -f json -o /tmp/bandit_sync_v2_m3_workspace.json` (`0 findings`).
- PASS: `git diff --check`.

Known skips/blockers:
- Collaborative Notes/Chat workspace semantics remain intentionally out of scope; workspace datasets are limited to workspace metadata and source references.

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
