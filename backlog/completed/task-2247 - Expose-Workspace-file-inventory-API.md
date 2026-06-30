---
id: TASK-2247
title: Expose Workspace file inventory API
status: Done
priority: high
references:
- TASK-2246
documentation:
- Docs/superpowers/specs/2026-06-03-workspace-file-inventory-jobs-design.md
- Docs/superpowers/plans/2026-06-03-workspace-file-inventory-jobs-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute Task 6 from the Workspace file inventory Jobs implementation plan. Add API schemas and Workspace routes for enqueueing file inventory scans, reading inventory status, and listing redacted project-root-relative file inventory items.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Failing API tests are written first for scan/status/items routes, no-root and root-version conflicts, Jobs unavailable mapping, current-scan force semantics, active scan reuse, cursor pagination, relative-path-only items, and bounded/redacted diagnostics.
- [x] #2 Pydantic schemas are added for scan request, job status, counts, diagnostics, status response, item response, and items response while preserving the existing WorkspaceFileInventory summary shape.
- [x] #3 Workspaces endpoints expose POST /api/v1/workspaces/{workspace_id}/file-inventory/scan, GET /status, and GET /items with thin endpoint logic and correct 404/409/422/503 mapping.
- [x] #4 API responses do not expose absolute root paths, file contents, or raw exception strings.
- [x] #5 Focused API tests pass and Backlog records verification/completion state.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Red API tests were written first in `tldw_Server_API/tests/Workspaces/test_workspace_file_inventory_api.py`. The initial run failed with six 404s for the missing `file-inventory` routes. A follow-up red assertion confirmed raw Jobs `error_message` text could leak through the status response before the file-inventory job projection was sanitized.

Added Workspace file-inventory schemas for scan requests, job status, counts, diagnostics, status responses, item responses, and items pages while leaving the existing `WorkspaceFileInventory` summary shape in place.

Added `POST /api/v1/workspaces/{workspace_id}/file-inventory/scan`, `GET /api/v1/workspaces/{workspace_id}/file-inventory/status`, and `GET /api/v1/workspaces/{workspace_id}/file-inventory/items` to `workspaces.py`. The scan route handles no-root conflicts, expected root version mismatches, Jobs unavailable/enqueue failures, force=false current-status no-op behavior, and force=true enqueue behavior.

Extended `CharactersRAGDB.get_workspace_file_inventory_status()` to return `root_snapshot_token`, and extended `list_workspace_file_inventory_items()` with optional server-side `entry_kind` filtering so pagination is applied after all filters.

File-inventory job status suppresses raw `error_message`; bounded/redacted scan diagnostics remain the public failure detail channel.

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Exposed the Workspace file inventory API slice. Verification: initial red API run failed with `6 failed` 404s for missing routes; raw-error regression failed before sanitization and passed after suppressing file-inventory job `error_message`; focused API tests passed `6 passed, 7 warnings`; adjacent Workspace API/DB/job/worker/rate-limit suite passed `97 passed, 8 warnings`; compileall passed for touched modules/tests; Bandit JSON at `/tmp/bandit_workspace_file_inventory_api.json` reported zero findings.

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
