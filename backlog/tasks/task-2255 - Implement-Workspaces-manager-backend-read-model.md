---
id: TASK-2255
title: Implement Workspaces manager backend read model
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-04 05:05'
labels:
  - workspaces
  - backend
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
Implement Task 1 from the canonical Workspaces manager plan: backend manager-facing read model helpers, context projection fields, operation response schema, and focused backend tests. This task must not implement durable operation persistence or sandbox provisioning; those are later slices.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Workspace Core exposes shared attention-state helpers for manager projections.
- [x] #2 Workspace context responses include file_inventory.available, attention_state, and active_operations with safe defaults.
- [x] #3 Workspace operation response/status schemas exist for later operation polling work.
- [x] #4 Focused Workspaces backend tests cover research/project/archived/unavailable/working states and pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Task 1 from Docs/superpowers/plans/2026-06-04-canonical-workspaces-manager-project-creation.md: Backend Canonical Read Model And Operation Envelope.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Verification 2026-06-04: focused backend suite passed with 107 tests: python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_core_models.py tldw_Server_API/tests/Workspaces/test_workspace_core_context.py tldw_Server_API/tests/Workspaces/test_workspaces_api.py -v. Bandit touched backend scope exited 0 and wrote /tmp/bandit_task_2255.json with no findings. git diff --check exited 0. Worker agent was closed after repeated timeouts; coordinator integrated and corrected the visible patch before verification.

Review fix 2026-06-04: addressed code-quality finding by moving file inventory availability into a shared Workspace Core helper and using it for /roots, /capabilities, and /context projections. Added cross-contract API assertion and model helper coverage. Re-verified focused suite: 108 passed, 6 warnings. Re-ran Bandit touched backend scope: 0 findings in /tmp/bandit_task_2255.json. git diff --check exited 0.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the Task 1 backend Workspaces manager read model. Added shared Workspace attention-state helpers, Sandbox-to-Workspace projection mapping, shared file-inventory availability rules, Workspace operation response schemas, and context response fields for attention_state and active_operations. The /roots, /capabilities, and /context projections now use consistent file_inventory.available semantics for root readiness. Verification: focused Workspaces backend suite passed with 108 tests and 6 warnings; Bandit touched backend scope reported 0 findings in /tmp/bandit_task_2255.json; git diff --check exited 0. Spec review approved with no findings. Code-quality review found one /roots availability consistency issue, which was fixed and re-reviewed as approved. No known blockers.
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
