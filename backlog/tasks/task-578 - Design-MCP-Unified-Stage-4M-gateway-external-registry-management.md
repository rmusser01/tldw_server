---
id: TASK-578
title: Design MCP Unified Stage 4M gateway external registry management
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-31 19:34'
labels:
  - mcp-unified
  - stage-4m
  - design
  - standalone
dependencies: []
documentation:
  - >-
    Docs/superpowers/specs/2026-05-31-mcp-unified-stage4m-gateway-external-registry-management-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Write and review the Stage 4M design spec for standalone gateway external-server registry management. Scope the next slice after Stage 4L around manager-owned CLI/FastAPI list/show/create/patch/disable/delete workflows for external server definitions, with SQLite persistence, audit posture, deterministic errors, package boundaries, and explicit deferral of real upstream process spawning, credential secret handling, install/update flows, and UI changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Stage 4M spec defines manager-owned external registry management scope for standalone gateway CLI and FastAPI.
- [x] #2 Spec documents external server definition lifecycle contracts: list, show, create, patch, disable/enable, and delete.
- [x] #3 Spec preserves package boundaries and reuses existing ExternalRegistryStore, AuditStore, and SQLite primitives without tldw_Server_API imports.
- [x] #4 Spec includes deterministic error/reason-code mappings, audit posture, validation rules, concurrency expectations, and focused verification strategy.
- [x] #5 Spec explicitly defers real upstream process spawning, credential secret handling, install/update flows, full lifecycle manager refresh, and UI changes.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Self-review pass completed. Tightened the draft around typed registry-store requirements, persistent-store atomic create, server-id slug validation for virtual tool names, websocket URL validation, persistent CLI store requirements, and no lifecycle side effects from registry mutations. Subagent review was not dispatched because current tool policy requires explicit user authorization for subagent delegation.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created the Stage 4M gateway external registry management design spec at Docs/superpowers/specs/2026-05-31-mcp-unified-stage4m-gateway-external-registry-management-design.md. The spec defines a manager-first CLI/FastAPI scope for external server definition list/show/create/patch/delete, persistent SQLite-backed registry behavior, deterministic errors, audit posture, server-id and transport validation, concurrency/runtime-state boundaries, and focused implementation verification. Bandit skipped because this task only adds documentation and Backlog tracking; git diff --check and marker scans were clean.
<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

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
