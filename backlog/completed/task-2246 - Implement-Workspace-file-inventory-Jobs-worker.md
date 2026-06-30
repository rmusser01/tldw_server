---
id: TASK-2246
title: Implement Workspace file inventory Jobs worker
status: Done
priority: high
references:
- TASK-2245
documentation:
- Docs/superpowers/specs/2026-06-03-workspace-file-inventory-jobs-design.md
- Docs/superpowers/plans/2026-06-03-workspace-file-inventory-jobs-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute Task 5 from the Workspace file inventory Jobs implementation plan. Add the Workspace file inventory Jobs enqueue helper, root scan resolution helper, and WorkerSDK worker that runs metadata scanning and persists scan/items state.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Failing Jobs helper tests are written first for payload redaction, scan-row-before-enqueue, scan-id idempotency key, Jobs unavailable handling, and active scan reuse only when a job id exists.
- [x] #2 Failing worker tests are written first for unsupported/malformed jobs, root version mismatch, host-local scan success, sandbox fail-closed behavior, root resolution failures, enqueue failure state, and worker result shape.
- [x] #3 Enqueue helper creates or reuses scan rows, creates Jobs rows without absolute root paths, attaches job ids idempotently, and marks enqueue failures as failed scans.
- [x] #4 Worker validates payloads, resolves roots without trusting job absolute paths, runs metadata scanning off-thread, persists items/counts/diagnostics, and completes scans as current/partial/failed.
- [x] #5 Focused Jobs helper and worker tests pass and Backlog records verification/completion state.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Red tests were written first in `tldw_Server_API/tests/Workspaces/test_workspace_file_inventory_jobs.py` and `tldw_Server_API/tests/Workspaces/test_workspace_file_inventory_worker.py`; the initial red run failed with `ModuleNotFoundError: No module named 'tldw_Server_API.app.core.Workspaces.file_inventory_jobs'` as recorded in `/tmp/workspace_file_inventory_jobs_worker_red.out`.

Implemented `file_inventory_jobs.py` with an identifier-only Jobs payload, scan-row-before-enqueue behavior, scan-id idempotency key, active scan reuse through the DB scan state, and failed scan marking when Jobs enqueue fails.

Extended `root_binding_service.py` with inventory-root resolution for host-local roots and fail-closed sandbox mounts, without trusting absolute paths from job payloads.

Added `workspace_file_inventory_jobs_worker.py` with payload validation, per-user DB loading, root version checks, `asyncio.to_thread` metadata scanning, item/count/diagnostic persistence, and current/partial/failed scan completion.

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the Workspace file inventory Jobs enqueue helper and WorkerSDK worker slice. Verification: initial red run captured the missing module failure; corrected focused/adjacent workspace pytest command passed `97 passed, 7 warnings in 25.44s`; compileall passed for touched modules and tests; Bandit JSON at `/tmp/bandit_workspace_file_inventory_jobs_worker.json` reported zero findings. One earlier broader pytest command used a nonexistent filename (`test_workspace_project_roots_service.py`) and failed as a command typo; it was corrected with `test_workspace_project_roots_db.py` before closeout.
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
