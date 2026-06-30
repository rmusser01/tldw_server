---
id: TASK-2242
title: Implement Workspace file inventory model contracts
status: Done
priority: high
references:
- TASK-2241
documentation:
- Docs/superpowers/specs/2026-06-03-workspace-file-inventory-jobs-design.md
- Docs/superpowers/plans/2026-06-03-workspace-file-inventory-jobs-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute Task 1 from the Workspace file inventory Jobs implementation plan. Add file inventory model helpers for durable/projected states, counts, diagnostics redaction, and opaque relative-path cursors using TDD.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Failing tests are written first for inventory states, counts, diagnostics, and cursor helpers.
- [x] #2 New file inventory model helpers pass focused tests.
- [x] #3 Helpers remain free of FastAPI, DB, and filesystem side effects.
- [x] #4 Backlog task records verification and completion state.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added `file_inventory_models.py` with durable/projected inventory state normalization, JSON-safe count normalization, bounded diagnostics with absolute path redaction, and opaque relative-path cursor helpers.
- Kept the module standalone: no FastAPI imports, DB imports, filesystem reads, or logging side effects.
- Red run: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_file_inventory_models.py -q --tb=short --disable-warnings` failed with `ModuleNotFoundError: No module named 'tldw_Server_API.app.core.Workspaces.file_inventory_models'`.
- Added a second red regression for `root_relative_only=False` after self-review found an absolute-path diagnostic footgun; it failed until diagnostics always used `redact_inventory_path_hint`.
- Verification:
  - `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_file_inventory_models.py -q` -> 14 passed.
  - `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_core_models.py tldw_Server_API/tests/Workspaces/test_workspace_file_inventory_models.py -q` -> 18 passed.
  - `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m compileall -q tldw_Server_API/app/core/Workspaces/file_inventory_models.py tldw_Server_API/tests/Workspaces/test_workspace_file_inventory_models.py` -> passed.
  - `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m bandit -q -r tldw_Server_API/app/core/Workspaces/file_inventory_models.py -f json -o /tmp/bandit_task_2242.json` -> 0 findings.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the Workspace file inventory model contract helpers and tests. The helpers provide fail-closed state normalization, stable inventory count keys, bounded/redacted diagnostics, and opaque cursor encoding/decoding for root-relative inventory pagination.
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
