---
id: TASK-368.4
title: Implement llama.cpp frontend API client
status: In Progress
assignee: []
created_date: '2026-05-15 03:44'
updated_date: '2026-05-15 06:51'
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

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Started Task 4 frontend API client/types slice after TASK-368.3 was finalized and committed at a10bd4376. Scope is limited to `apps/packages/ui/src/types/llamacpp-admin.ts`, `apps/packages/ui/src/services/tldw/TldwApiClient.ts`, `apps/packages/ui/src/services/tldw/domains/models-audio.ts`, and `apps/packages/ui/src/services/tldw/client-ownership.ts`; no page reshape in this task.
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
