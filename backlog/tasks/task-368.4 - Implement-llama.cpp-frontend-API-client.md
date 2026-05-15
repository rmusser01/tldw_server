---
id: TASK-368.4
title: Implement llama.cpp frontend API client
status: To Do
assignee: []
created_date: '2026-05-15 03:44'
labels:
  - implementation
  - frontend
  - llamacpp
dependencies:
  - TASK-368.3
documentation:
  - Docs/superpowers/specs/2026-05-15-llamacpp-server-management-webui-design.md
  - >-
    Docs/superpowers/plans/2026-05-15-llamacpp-server-management-webui-implementation-plan.md
parent_task_id: TASK-368
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the frontend API client and type slice from the implementation plan. Add TypeScript admin types and facade client methods for config validation inventory registration start-by-model use-in-chat log tail and hardware snapshot. Do not reshape the page UI in this task.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Shared TypeScript types exist for the llama.cpp admin facade contracts.
- [ ] #2 TldwApiClient and the models audio domain client expose the new llama.cpp admin facade methods consistently.
- [ ] #3 Client ownership metadata is updated for the new methods.
- [ ] #4 Existing llama.cpp admin page tests still pass before page reshape.
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
