---
id: TASK-368.2
title: Implement llama.cpp model inventory resolver
status: To Do
assignee: []
created_date: '2026-05-15 03:42'
labels:
  - implementation
  - backend
  - llamacpp
dependencies:
  - TASK-368.1
documentation:
  - Docs/superpowers/specs/2026-05-15-llamacpp-server-management-webui-design.md
  - >-
    Docs/superpowers/plans/2026-05-15-llamacpp-server-management-webui-implementation-plan.md
parent_task_id: TASK-368
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the inventory/start-by-model backend slice from the implementation plan. Add safe recursive GGUF inventory, registered local model paths, stable model IDs, and a handler path-start helper while preserving the existing filename-based start_server endpoint.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 GET /api/v1/llamacpp/inventory returns bounded recursive GGUF inventory with stable model IDs and warnings.
- [ ] #2 POST /api/v1/llamacpp/models/register-path persists explicit local GGUF paths safely through allowed config keys.
- [ ] #3 POST /api/v1/llamacpp/start-by-model resolves model_id to a validated path and starts through the managed handler.
- [ ] #4 Existing /api/v1/llamacpp/start_server filename behavior and path hardening continue to pass tests.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
