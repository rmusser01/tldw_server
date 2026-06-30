---
id: TASK-490.12.1
title: 'Sync v2 M2: Protocol models and capabilities'
status: Done
labels:
- sync
- sync-v2
- m2
- attachments
priority: medium
parent_task_id: TASK-490.12
documentation:
- Docs/Design/Sync_V2_M2_Restore_Completeness_and_Blobs.md
- Docs/superpowers/plans/2026-05-23-sync-v2-m2-restore-completeness-blobs-implementation-plan.md
assignee:
- '@Codex'
modified_files:
- tldw_Server_API/app/api/v1/schemas/sync_v2_models.py
- tldw_Server_API/app/core/Sync/v2/models.py
- tldw_Server_API/app/core/Sync/v2/service.py
- tldw_Server_API/tests/Sync/test_sync_v2_models.py
- tldw_Server_API/tests/Sync/test_sync_v2_service.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Define the M2 public API/core model contract for blob upload sessions, download manifests, quota details, blob availability, restore completeness, and capability advertisement.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 M2 schema tests cover upload sessions, chunks, download manifests, quota/status, and restore completeness fields.
- [x] #2 Sync v2 capabilities can advertise resumable upload/download, checksum, and quota support while remaining disabled by default until later stages.
- [x] #3 M2 blob transfer no longer requires client_private_v1 as the default encryption policy; server_trusted_v1 remains the M2 default.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-23-sync-v2-m2-restore-completeness-blobs-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added M2 API schemas for blob upload creation, upload session status, chunk upload results, upload completion, download manifests, and restore completeness.
- Added M2 core type/dataclass scaffolding for blob availability, upload sessions, download chunks, and restore completeness details.
- Extended Sync v2 capabilities/settings with opt-in M2 blob transfer details and quota fields while preserving the default M1 `{"supported": false}` blob capability shape.
- Kept `server_trusted_v1` as the default blob encryption policy and did not introduce `client_private_v1` as a required M2 default.
- Verification:
  - RED: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_models.py tldw_Server_API/tests/Sync/test_sync_v2_service.py::test_capabilities_can_advertise_m2_resumable_blob_transfer -v` failed during collection with expected missing `SyncBlobChunkUploadResponse`.
  - GREEN: same targeted command passed with `30 passed, 6 warnings`.
  - Broader: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_models.py tldw_Server_API/tests/Sync/test_sync_v2_service.py -v` passed with `89 passed, 6 warnings`.
  - Expanded Sync v2 endpoint/profile/restore/security subset passed with `122 passed, 6 warnings`.
  - Full Sync v2 unit glob `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_*.py -v` passed with `293 passed, 6 warnings`.
  - Ruff passed on touched code/tests.
  - Bandit touched production scope wrote `/tmp/bandit_sync_v2_m2_protocol_models.json` with 0 results.
  - `git diff --check` passed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented Sync v2 M2 protocol model and capability scaffolding. Added public API schemas for resumable blob upload sessions, chunk upload responses, upload completion, download manifests, and restore completeness. Added matching core dataclass/type scaffolding plus service capability settings so M2 blob transfer and quota details can be advertised only when supports_attachments is enabled. Default M1 capabilities remain metadata-only with server_trusted_v1 as the only encryption policy.
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
