---
id: TASK-368.3
title: Implement llama.cpp provider wiring and diagnostics
status: In Progress
assignee: []
created_date: '2026-05-15 03:43'
updated_date: '2026-05-15 06:03'
labels:
  - implementation
  - backend
  - llamacpp
dependencies:
  - TASK-368.2
documentation:
  - Docs/superpowers/specs/2026-05-15-llamacpp-server-management-webui-design.md
  - >-
    Docs/superpowers/plans/2026-05-15-llamacpp-server-management-webui-implementation-plan.md
parent_task_id: TASK-368
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the provider diagnostics backend slice from the implementation plan. Add explicit use-in-chat provider wiring, bounded managed log tailing, best-effort hardware snapshot, and permission coverage. Do not change frontend behavior in this task.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 POST /api/v1/llamacpp/use-in-chat explicitly updates only the llama.cpp provider endpoint after a managed server is running.
- [ ] #2 GET /api/v1/llamacpp/logs/tail returns bounded managed logs and cannot read arbitrary paths.
- [ ] #3 GET /api/v1/llamacpp/hardware returns best-effort RAM/CPU/GPU data with structured warnings and no hard dependency on NVIDIA hardware.
- [ ] #4 New endpoints retain admin-only permission coverage and focused backend tests pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Started subagent-driven implementation for the provider wiring, hardware snapshot, and safe log tail backend slice after TASK-368.2 passed spec and code-quality review. Scope is limited to Task 3 backend files and tests from the implementation plan.
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
