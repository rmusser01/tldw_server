---
id: TASK-368.1
title: Implement llama.cpp admin config facade
status: In Progress
assignee: []
created_date: '2026-05-15 03:42'
updated_date: '2026-05-15 03:50'
labels:
  - implementation
  - backend
  - llamacpp
dependencies: []
documentation:
  - Docs/superpowers/specs/2026-05-15-llamacpp-server-management-webui-design.md
  - >-
    Docs/superpowers/plans/2026-05-15-llamacpp-server-management-webui-implementation-plan.md
parent_task_id: TASK-368
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the first backend slice from the implementation plan: typed llama.cpp admin config endpoints, saved-vs-active runtime state, restart-required semantics, environment override reporting, comment-preserving config writes, and binary validation. Do not implement inventory, provider wiring, hardware, logs, or frontend changes in this task.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 GET and PUT /api/v1/llamacpp/config expose saved config, active config, restart-required reasons, warnings, and env override state.
- [ ] #2 Config updates use the existing comment-preserving setup config writer and refresh config caches.
- [ ] #3 POST /api/v1/llamacpp/validate reports binary validation results without starting a server.
- [ ] #4 Focused backend tests for the config facade and existing management API pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Started subagent-driven implementation for the backend config facade slice in worktree .worktrees/codex-llamacpp-webui-management on branch codex/llamacpp-webui-management. Baseline focused tests passed: test_llamacpp_management_api.py and test_llamacpp_handler.py, 37 passed with existing Loguru shutdown noise after pytest exit.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
