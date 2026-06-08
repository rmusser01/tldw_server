---
id: TASK-2316
title: Implement Workspace cross-resource membership foundation
status: In Progress
labels:
- workspaces
- project-workspace
- membership
- implementation
priority: high
references:
- https://github.com/rmusser01/tldw_server/issues/1990
- https://github.com/rmusser01/tldw_server/issues/1984
- Docs/superpowers/specs/2026-06-07-workspace-cross-resource-membership-design.md
documentation:
- Docs/superpowers/specs/2026-06-07-workspace-cross-resource-membership-design.md
- Docs/superpowers/plans/2026-06-07-workspace-cross-resource-membership-implementation-plan.md
modified_files:
- tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py
- tldw_Server_API/tests/ChaChaNotesDB/test_workspace_resource_memberships_db.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the approved first server-backed Workspace cross-resource membership slice. Scope includes ChaChaNotes persistence, fail-closed resource adapters, Workspace Core service, API schemas/endpoints, explicit backfill helper, context summary, tests, docs, and verification. Preserve the boundary that generic membership is association, not ownership transfer, global filtering, or MCP permission/path trust.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 ChaChaNotes persists `workspace_resource_memberships` with SQLite/PostgreSQL schema support, idempotent create, conflict handling, soft-delete, restore, deterministic workspace listing, and reverse resource lookup.
- [ ] #2 Workspace membership models, schemas, adapters, service, and API endpoints implement the approved first slice for `workspace_note`, `media`, `workspace_source`, `workspace_artifact`, and `chat`.
- [ ] #3 Backfill helper is explicit and idempotent; Workspace context exposes compact membership totals without making membership a global Library/Notes/search filter.
- [ ] #4 MCP permission preview/path admission remains driven by MCP policy/root bindings, not generic membership.
- [ ] #5 Focused tests and Bandit verification are recorded; known skips or unrelated failures are documented.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-07-workspace-cross-resource-membership-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Execution started with subagent-driven Task 1: persistence and DB contract.
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
