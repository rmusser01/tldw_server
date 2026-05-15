---
id: TASK-368.5
title: Implement llama.cpp guided admin UI
status: To Do
assignee: []
created_date: '2026-05-15 03:45'
updated_date: '2026-05-15 03:45'
labels:
  - implementation
  - frontend
  - llamacpp
dependencies:
  - TASK-368.4
documentation:
  - Docs/superpowers/specs/2026-05-15-llamacpp-server-management-webui-design.md
  - >-
    Docs/superpowers/plans/2026-05-15-llamacpp-server-management-webui-implementation-plan.md
parent_task_id: TASK-368
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the guided WebUI slice from the implementation plan. Reshape the llama.cpp admin page into readiness inventory and launch panels using the new client methods while preserving existing advanced launch controls and admin guard behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The page renders readiness state with saved active and restart required messaging.
- [ ] #2 The page renders model inventory and starts models by stable model ID.
- [ ] #3 Hardware warnings are shown without disabling start solely because hardware data is unknown or risky.
- [ ] #4 The chat wiring action appears after a running managed server is available and is never called automatically.
- [ ] #5 Focused frontend component tests pass.
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
