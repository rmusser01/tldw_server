---
id: TASK-2258
title: Implement Workspace-owned sandbox root provisioning command
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-04 06:21'
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
Implement Task 4 from the canonical Workspaces manager plan: add Workspace-owned operation/idempotency records plus the sandbox root provision-and-attach command and API endpoints. Sandbox remains owner of durable volume mechanics; Workspace owns the product command, operation envelope, and primary root attach.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Workspace operation persistence supports idempotent creation, lookup, update, active listing, expiry cleanup, conflict detection, and redacted/bounded diagnostics.
- [x] #2 Workspace sandbox root provisioning service uses Workspace operation idempotency, calls SandboxWorkspaceVolumeService for durable volumes, attaches roots only through attach_primary_workspace_root, and preserves/upgrades workspace_profile=project.
- [x] #3 API exposes POST /api/v1/workspaces/{workspace_id}/roots/primary/sandbox-volume and GET /api/v1/workspaces/{workspace_id}/operations/{operation_id} with 400/409/202/200 behavior matching the plan.
- [x] #4 Workspace context includes active operations from the operation projection.
- [x] #5 Focused DB, service, and API tests are written red-first and pass after implementation.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Task 4 from Docs/superpowers/plans/2026-06-04-canonical-workspaces-manager-project-creation.md: Workspace-Owned Sandbox Root Provision-And-Attach Command.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Red test evidence (2026-06-03): focused Task 4 pytest failed during collection because `tldw_Server_API.app.core.Workspaces.operations` and `tldw_Server_API.app.core.Workspaces.sandbox_root_provisioning` did not exist yet. Green evidence: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_operations.py tldw_Server_API/tests/Workspaces/test_workspace_sandbox_root_provisioning.py tldw_Server_API/tests/Workspaces/test_workspaces_api.py -v` -> 80 passed, 6 warnings. Verification: compileall on touched Python modules exited 0; `git diff --check` exited 0. Bandit: broad plan scope wrote `/tmp/bandit_workspaces_project_root.json` and exited 1 due 119 pre-existing Sandbox subprocess/assert findings outside Task 4 files; touched-file Bandit wrote `/tmp/bandit_task_2258_touched.json` and exited 0 with 0 results.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented Task 4: added Workspace operation persistence/idempotency records, operation redaction/fingerprinting helpers, Workspace-owned sandbox root provisioning service, provision/poll API endpoints, sandbox volume service dependency override support, and active-operation projection in workspace context. Added red-first DB, service, and API tests covering idempotency retries/conflicts, expired operation cleanup preserving roots, conservative sandbox not_configured behavior, pollable status, and context active operations.
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
