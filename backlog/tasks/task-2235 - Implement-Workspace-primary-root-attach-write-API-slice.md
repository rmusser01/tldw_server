---
id: TASK-2235
title: Implement Workspace primary root attach/write API slice
status: In Progress
priority: high
documentation:
- Docs/superpowers/specs/2026-06-03-canonical-workspace-core-project-model-design.md
- Docs/superpowers/plans/2026-06-03-workspace-core-contract-implementation-plan.md
- Docs/superpowers/specs/2026-06-03-workspace-primary-root-attach-api-design.md
- Docs/superpowers/plans/2026-06-03-workspace-primary-root-attach-api-implementation-plan.md
modified_files:
- Docs/superpowers/specs/2026-06-03-workspace-primary-root-attach-api-design.md
- Docs/superpowers/plans/2026-06-03-workspace-primary-root-attach-api-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add the next Workspace Core slice for attaching/replacing a Workspace-owned primary project root through a reusable service and thin API endpoints. Scope includes host_local validation, sandbox_volume wrapper contract validation, one-primary-root semantics, read contract compatibility, tests, docs, and verification. Exclude secondary roots, file inventory scans, Git operations, MCP trust mutation, and full Sandbox volume lifecycle.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Workspace primary-root attach/write API is designed and planned against the canonical Workspace Core model.
- [ ] #2 Reusable Workspace Core root-binding service owns validation and DB persistence orchestration instead of adding endpoint-only logic.
- [ ] #3 API supports attaching/replacing one primary `host_local` root with allowlist, traversal, and symlink escape validation.
- [ ] #4 API supports attaching/replacing one primary `sandbox_volume` root through a bounded Workspace-owned wrapper contract without implementing full Sandbox volume lifecycle.
- [ ] #5 Existing read-only roots/context/capability contracts remain compatible and do not expose local absolute paths.
- [ ] #6 Focused backend tests, compile smoke, diff hygiene, and Bandit touched-scope verification are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Design spec drafted for the Workspace primary root attach/write API slice at `Docs/superpowers/specs/2026-06-03-workspace-primary-root-attach-api-design.md`. The spec selects a reusable Workspace Core root-binding service plus thin `PUT /api/v1/workspaces/{workspace_id}/roots/primary` endpoint, with host-local allowlist/path validation, sandbox-volume wrapper validation, one-primary-root replacement semantics, redacted read responses, and explicit follow-up boundaries.
Spec review follow-up applied: hardened `root_id` omission/idempotency semantics, added field bounds for `root_id` and `display_name`, required DB-transactional `expected_workspace_version` enforcement, clarified same-binding operational-state repair, made default Sandbox resolver behavior fail-closed, and added config docs/test coverage requirements for Workspace project-root allowlists.
Implementation plan written at `Docs/superpowers/plans/2026-06-03-workspace-primary-root-attach-api-implementation-plan.md`. The plan decomposes the slice into config allowlists, DB write semantics, root-binding service, fail-closed sandbox capabilities, API schema/endpoint, read-contract regression tests, and final verification/Backlog closeout. Self-review tightened the plan so DB conflicts from the attach write are wrapped into stable Workspace-root `{code,message}` API payloads.
Task 4 slice implemented: Workspace Core context now fails closed for `sandbox_volume` project roots until `sandbox_mount_state` is `ready` or compatibility alias `mounted`; focused context tests, diff hygiene, and Bandit touched-scope verification were run.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
