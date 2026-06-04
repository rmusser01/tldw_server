---
id: TASK-2248
title: Surface Workspace file inventory context and startup worker
status: Done
priority: high
references:
- TASK-2247
documentation:
- Docs/superpowers/specs/2026-06-03-workspace-file-inventory-jobs-design.md
- Docs/superpowers/plans/2026-06-03-workspace-file-inventory-jobs-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute Task 7 from the Workspace file inventory Jobs implementation plan. Project file inventory into Workspace context/capability responses, register the file inventory Jobs worker in primary job startup, and document the metadata-only inventory boundary.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Failing integration tests are written first for nested file inventory summary, scan_files/view_file_inventory/index_file_content action gates, and startup worker enabled/disabled behavior.
- [x] #2 Workspace context/capability projection includes file_inventory_state, nested file_inventory summary, scan_files, view_file_inventory, and unchanged disabled index_file_content semantics.
- [x] #3 Primary Jobs startup exposes workspace_file_inventory_jobs_stop_event/task handles and registers the worker behind WORKSPACE_FILE_INVENTORY_JOBS_WORKER_ENABLED for declarative and legacy startup paths.
- [x] #4 Workspace README documents metadata-only scanning, path redaction, no automatic file indexing, worker env flags, and the follow-up explicit indexing slice.
- [x] #5 Focused context/startup/API tests pass and Backlog records verification/completion state.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented the Task 7 context/startup slice. The red run failed as expected with missing nested `file_inventory`, missing `scan_files`/`view_file_inventory`, and missing `workspace_file_inventory_jobs_task` startup registration: `13 failed, 94 passed`. The implementation adds nested inventory projection, persisted inventory status enrichment for context/capability responses, conservative file inventory action gates, public schema serialization, primary Jobs worker handles/spec/startup path, and README boundary documentation.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed Workspace file inventory context/startup integration. Verification: focused green suite passed `107 passed, 6 warnings`; expanded adjacent suite passed `113 passed, 7 warnings`; compileall passed for touched Python modules/tests; `git diff --check` passed; Bandit JSON at `/tmp/bandit_workspace_file_inventory_context_startup.json` reported zero results, zero errors, and zero skipped files.
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
