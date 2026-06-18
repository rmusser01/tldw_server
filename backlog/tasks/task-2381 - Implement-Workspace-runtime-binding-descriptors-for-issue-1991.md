---
id: TASK-2381
title: Implement Workspace runtime binding descriptors for issue 1991
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-18 01:20'
labels: []
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1991'
  - 'https://github.com/rmusser01/tldw_server/issues/1984'
  - Docs/Design/Workspace_Container_Contract_2026_06.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Backend-first Workspace Phase 2 slice for #1991: durable runtime binding descriptors with secret-safe metadata handling, focused API routes, tests, and documentation updates.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Runtime binding descriptor literals and secret-safe metadata rules are defined in the Workspace core layer.
- [x] #2 Durable Workspace runtime binding persistence supports upsert/list/get/archive for a workspace.
- [x] #3 Public runtime binding responses expose path hints and redaction reports but not raw secrets or private path payloads.
- [x] #4 Workspace runtime binding API routes are available under /api/v1/workspaces/{workspace_id}/runtime-bindings.
- [x] #5 Tests cover metadata redaction/rejection, DB CRUD/archive, API happy path and fail-closed errors.
- [x] #6 Docs/plan/backlog references are updated and verification results are recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-18-workspace-runtime-bindings.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implementation plan: Docs/superpowers/plans/2026-06-18-workspace-runtime-bindings.md

Implemented runtime binding descriptor vocabulary and secret-safe normalizer, durable ChaChaNotes workspace_runtime_bindings persistence, Workspaces API schemas/routes, focused tests, and README documentation.

Verification:
- source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_runtime_bindings.py tldw_Server_API/tests/Workspaces/test_workspace_runtime_bindings_api.py tldw_Server_API/tests/Workspaces/test_workspace_project_roots_db.py tldw_Server_API/tests/Workspaces/test_workspaces_api.py -q -> 110 passed, 6 warnings
- python -m bandit -r touched backend files -f json -o /tmp/bandit_workspace_runtime_bindings.json -> 0 results, 0 errors
- git diff --check -> clean
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented Workspace runtime binding descriptors for #1991: centralized runtime binding vocabulary, secret-safe metadata/path redaction, durable ChaChaNotes workspace_runtime_bindings persistence, Workspaces API list/upsert/get/archive routes, focused unit/API coverage, and README documentation.

Verification:
- source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_runtime_bindings.py tldw_Server_API/tests/Workspaces/test_workspace_runtime_bindings_api.py tldw_Server_API/tests/Workspaces/test_workspace_project_roots_db.py tldw_Server_API/tests/Workspaces/test_workspaces_api.py -q -> 110 passed, 6 warnings
- python -m bandit -r touched backend files -f json -o /tmp/bandit_workspace_runtime_bindings.json -> 0 results, 0 errors
- git diff --check -> clean

Known skips/blockers: no known blockers; full repository test suite not run for this focused slice.
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
