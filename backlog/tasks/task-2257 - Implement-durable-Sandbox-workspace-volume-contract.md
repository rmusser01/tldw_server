---
id: TASK-2257
title: Implement durable Sandbox workspace-volume contract
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-04 06:07'
labels:
  - workspaces
  - sandbox
  - project-workspace
dependencies: []
documentation:
  - >-
    Docs/superpowers/specs/2026-06-04-canonical-workspaces-manager-project-creation-design.md
  - >-
    Docs/superpowers/plans/2026-06-04-canonical-workspaces-manager-project-creation.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Task 3 from the canonical Workspaces manager plan: add the Sandbox-owned durable workspace-volume contract and resolver behavior needed for future Project Workspace sandbox-managed roots. This task must not add the Workspace-owned provision-and-attach API endpoints; those are Task 4.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Sandbox model/store/service support durable workspace-bound volumes with idempotent provision lookup, state updates, and per-workspace listing.
- [x] #2 Workspace volume service validates workspace/user ownership, handles idempotency conflicts, and returns conservative states when no durable runtime mount is available.
- [x] #3 Workspace root binding service supports provisioning and cleanup_pending states while failing closed for unavailable, failed, not_configured, and cleanup_pending under strict sandbox validation.
- [x] #4 Diagnostics persisted or returned by the volume service are bounded and redacted; raw host paths/secrets are not exposed beyond allowed mount hints.
- [x] #5 Focused Sandbox and Workspace root binding tests are written red-first and pass after implementation.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Task 3 from Docs/superpowers/plans/2026-06-04-canonical-workspaces-manager-project-creation.md: Durable Sandbox Workspace-Volume Contract.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Red test evidence (2026-06-03):
- `python -m pytest tldw_Server_API/tests/sandbox/test_workspace_volumes.py -q` fails during collection with `ImportError: cannot import name 'WorkspaceVolumeState'`.
- `python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_root_binding_service.py -q` fails for `test_sandbox_resolver_accepts_new_non_strict_volume_states[provisioning]`, `[cleanup_pending]`, and strict `cleanup_pending` because the root binding service still rejects the new states as invalid.

Green/verification evidence (2026-06-03/04):
- `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/sandbox/test_workspace_volumes.py tldw_Server_API/tests/Workspaces/test_workspace_root_binding_service.py -v` -> 37 passed, 3 warnings.
- `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/Sandbox/models.py tldw_Server_API/app/core/Sandbox/store.py tldw_Server_API/app/core/Sandbox/workspace_volumes.py tldw_Server_API/app/core/Workspaces/root_binding_service.py -f json -o /tmp/bandit_task_2257.json` -> 0 findings in JSON results.
- `git diff --check` -> passed.
- `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m compileall -q tldw_Server_API/app/core/Sandbox/models.py tldw_Server_API/app/core/Sandbox/store.py tldw_Server_API/app/core/Sandbox/workspace_volumes.py tldw_Server_API/app/core/Workspaces/root_binding_service.py` -> passed.

Controller hardening review (2026-06-03): added red-first coverage for direct InMemory/SQLite workspace-volume store writes bypassing diagnostics and mount-path sanitization. Red evidence: `python -m pytest tldw_Server_API/tests/sandbox/test_workspace_volumes.py::test_store_direct_writes_bound_and_redact_workspace_volume_diagnostics -v` failed because a raw `/Users/...` mount path and secret-bearing diagnostics were persisted. Green evidence after fix: the same focused test passed; full focused slice `python -m pytest tldw_Server_API/tests/sandbox/test_workspace_volumes.py tldw_Server_API/tests/Workspaces/test_workspace_root_binding_service.py -v` passed with 38 tests and 3 warnings. Fresh controller checks: Bandit on touched backend modules wrote `/tmp/bandit_task_2257_controller.json` with exit 0, compileall exited 0, and `git diff --check` exited 0.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented Task 3 only: added Sandbox-owned durable Workspace volume models, store persistence for in-memory/SQLite/Postgres stores, a conservative `SandboxWorkspaceVolumeService`, diagnostic redaction/bounding, safe mount hint resolution, and Workspace root binding support for `provisioning` and `cleanup_pending` states. Added focused Sandbox and root binding tests covering idempotency, ownership validation, conservative no-runtime behavior, ready mount hints, diagnostic redaction, and strict/non-strict root binding state handling.
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
