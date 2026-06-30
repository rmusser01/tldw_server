---
id: TASK-2233
title: Plan Workspace Core contract implementation
status: Done
priority: high
documentation:
- Docs/superpowers/specs/2026-06-03-canonical-workspace-core-project-model-design.md
- Docs/superpowers/plans/2026-06-03-workspace-core-contract-implementation-plan.md
modified_files:
- Docs/superpowers/plans/2026-06-03-workspace-core-contract-implementation-plan.md
- backlog/tasks/task-2233 - Plan-Workspace-Core-contract-implementation.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Write the implementation plan for the first canonical Workspace Core contract slice based on TASK-2232. Scope the plan to additive backend contracts for persisted workspace_profile, root/capability state schemas, read-only context resolution, fail-closed capability defaults, and tests. Do not implement runtime code in this task.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Plan file exists under Docs/superpowers/plans with the required superpowers implementation-plan header.
- [x] #2 Plan maps the first implementation slice into small, sequential, test-first tasks with exact files and commands.
- [x] #3 Plan preserves the design constraints: one canonical workspace_id, persisted workspace_profile, computed capability states, one primary root, host_local and sandbox_volume root schemas, explicit file-content indexing policy, and fail-closed runtime context semantics.
- [x] #4 Plan identifies parallelizable work boundaries for backend models/context, DB persistence, API schemas/endpoints, and focused tests without requiring UI or full Sandbox implementation in the first slice.
- [x] #5 Task records verification and skips for the docs-only planning change.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created the Workspace Core contract implementation plan for the first backend slice after TASK-2232. The plan is saved at Docs/superpowers/plans/2026-06-03-workspace-core-contract-implementation-plan.md and scopes the work to persisted workspace_profile, Workspace-owned primary root persistence for host_local and sandbox_volume backends, read-only context/capability projection, additive API schema fields, a read-only roots endpoint, fail-closed runtime semantics, focused pytest coverage, compile smoke, Bandit, and clear follow-up slices for root attach, file inventory Jobs, indexing, MCP cleanup, ACP/harness consumption, UI, and previews. Verification: git diff --check and targeted rg checks. Bandit skipped because this task only adds documentation/backlog planning files.
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
