---
id: TASK-2243
title: Implement Workspace file inventory DB persistence
status: Done
priority: high
references:
- TASK-2242
documentation:
- Docs/superpowers/specs/2026-06-03-workspace-file-inventory-jobs-design.md
- Docs/superpowers/plans/2026-06-03-workspace-file-inventory-jobs-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute Task 2 from the Workspace file inventory Jobs implementation plan. Add CharactersRAGDB schema and methods for file inventory scans/items/status/listing using TDD, including stale projection, full/partial coverage behavior, enqueue failure handling, and cleanup expectations.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Failing DB tests are written first for scan/item persistence, stale status, pagination, cleanup, and enqueue-failure reuse behavior.
- [x] #2 CharactersRAGDB creates file inventory scan/item tables with expected indexes and cascade/cleanup behavior.
- [x] #3 DB methods cover scan begin/job attach/enqueue failure/scanning/completion/item replacement/status/item listing.
- [x] #4 SQLite and backend-abstraction uniqueness/error behavior is handled consistently with Workspace DB patterns where this slice touches it.
- [x] #5 Focused DB tests pass and Backlog records verification/completion state.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added `workspace_file_inventory_scans` and `workspace_file_inventory_items` schema creation for SQLite and PostgreSQL final schema-healing paths.
- Added `CharactersRAGDB` methods for scan begin, job attach, enqueue failure, scanning transition, completion, item projection replacement, status projection, and item pagination.
- Kept root `version` stable for inventory-owned root-state projection updates so scans do not stale themselves; stale projection compares the scan root version against root-binding version changes.
- Added status fallback for failed `root_version_mismatch` attempts so the last completed scan remains visible as `stale` until a new scan completes.
- Red run: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_file_inventory_db.py -q` failed with missing inventory tables and missing DB methods.
- Additional red regressions:
  - `test_root_version_mismatch_failure_keeps_previous_completed_scan_stale` failed with `state == failed` before the fallback.
  - `test_item_listing_rejects_invalid_cursor_as_input_error` failed before cursor decode errors were mapped to `InputError`.
- Verification:
  - `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_file_inventory_db.py -q` -> 12 passed.
  - `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_core_models.py tldw_Server_API/tests/Workspaces/test_workspace_file_inventory_models.py tldw_Server_API/tests/Workspaces/test_workspace_project_roots_db.py tldw_Server_API/tests/Workspaces/test_workspace_file_inventory_db.py -q` -> 51 passed.
  - `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Workspaces/test_workspaces_api.py -q` -> 65 passed.
  - `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m compileall -q tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py tldw_Server_API/tests/Workspaces/test_workspace_file_inventory_db.py` -> passed.
  - `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m bandit -q -r tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py -f json -o /tmp/bandit_task_2243.json` -> 0 findings; existing skipped `nosec` warnings in the large DB file remain.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented Workspace file inventory DB persistence. Workspace Core now has durable scan/item tables, scan lifecycle methods, bounded status projection with stale handling, item replacement semantics for full and partial scans, relative-path pagination, and cleanup coverage through existing root/workspace delete paths.
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
