---
id: TASK-579
title: Plan MCP Unified Stage 4M gateway external registry management implementation
status: Done
assignee: []
created_date: ''
updated_date: 2026-05-31 19:52
labels:
- mcp-unified
- stage-4m
- planning
- standalone
dependencies: []
documentation:
- Docs/superpowers/specs/2026-05-31-mcp-unified-stage4m-gateway-external-registry-management-design.md
- Docs/superpowers/plans/2026-05-31-mcp-unified-stage4m-gateway-external-registry-management-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create the implementation plan for MCP Unified Stage 4M gateway external registry management. The plan should translate the reviewed design into concrete manager, storage, bootstrap/config, FastAPI, CLI, test, validation, and review steps while preserving package boundaries and deferring real external process lifecycle.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Plan identifies exact package/test files for storage, manager, config/bootstrap, FastAPI, CLI, and validation work.
- [x] #2 Plan decomposes implementation into bite-sized TDD tasks with red/green commands and commit checkpoints.
- [x] #3 Plan preserves package boundaries, SQLAlchemy-only SQLite access, async store behavior, and deferred lifecycle scope from the reviewed design.
- [x] #4 Plan records focused validation commands, Bandit scope, diff checks, and known non-code review limitations.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created a dedicated implementation plan at Docs/superpowers/plans/2026-05-31-mcp-unified-stage4m-gateway-external-registry-management-implementation-plan.md.

The plan was reviewed against current package APIs and tightened for worktree venv activation plus explicit credential-grant fail-closed behavior.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implementation plan created and linked. It covers storage protocol/SQLite atomic create, GatewayExternalRegistryManager semantics, config/bootstrap wiring, FastAPI routes, CLI commands, integration validation, Bandit scope, and implementation commit checkpoints.

No runtime code was changed in this task. Bandit is documented as not applicable to this docs-only planning slice; the implementation plan requires Bandit for the touched package scope.
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
