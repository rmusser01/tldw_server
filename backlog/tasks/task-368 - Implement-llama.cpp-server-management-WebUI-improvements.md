---
id: TASK-368
title: Implement llama.cpp server management WebUI improvements
status: In Progress
assignee: []
created_date: '2026-05-15 03:41'
labels:
  - implementation
  - llamacpp
  - webui
  - self-hosted
dependencies:
  - TASK-365
documentation:
  - Docs/superpowers/specs/2026-05-15-llamacpp-server-management-webui-design.md
  - >-
    Docs/superpowers/plans/2026-05-15-llamacpp-server-management-webui-implementation-plan.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the approved single-server llama.cpp server management WebUI flow from the design and implementation plan. The feature should let self-hosted admins configure and validate llama.cpp, inspect safe GGUF inventory, start a selected model by stable model ID, view warnings-first hardware guidance, explicitly wire the running managed server into Chat, and inspect bounded managed logs. Keep V1 to one managed server and preserve backend safety boundaries.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 All implementation-plan task slices are completed or explicitly documented as blocked.
- [ ] #2 The final feature preserves V1 constraints: one managed server, no downloads/uploads, explicit provider wiring, warnings not hard blocking, and backend-owned safety.
- [ ] #3 Focused backend and frontend tests pass, with any environment-limited E2E checks documented.
- [ ] #4 Bandit is run on touched backend scope and new actionable findings are fixed or documented.
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
