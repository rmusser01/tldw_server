---
id: TASK-368
title: Implement llama.cpp server management WebUI improvements
status: Done
assignee: []
created_date: '2026-05-15 03:41'
updated_date: '2026-05-15 15:55'
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
- [x] #1 All implementation-plan task slices are completed or explicitly documented as blocked.
- [x] #2 The final feature preserves V1 constraints: one managed server, no downloads/uploads, explicit provider wiring, warnings not hard blocking, and backend-owned safety.
- [x] #3 Focused backend and frontend tests pass, with any environment-limited E2E checks documented.
- [x] #4 Bandit is run on touched backend scope and new actionable findings are fixed or documented.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Completed all six llama.cpp server-management WebUI subtasks. Final implementation includes backend admin config facade, stable model inventory and resolver, provider wiring/log/hardware diagnostics, frontend client facade, guided admin UI, documentation, and E2E smoke coverage. V1 constraints were preserved: one managed server, no model downloads/uploads, start by backend-owned stable model_id, explicit Use this in Chat provider wiring, hardware warnings as advisory, and backend-owned path/permission safety.
Final verification: backend focused pytest passed 123 tests with 6 warnings; frontend focused Vitest passed 16 tests; tier-4 admin llama.cpp Playwright smoke passed 5 tests with mocked backend responses; Bandit /tmp/bandit_llamacpp_admin.json reported no findings; git diff --check passed.
Known skips/limitations: frontend tsc is still blocked by unrelated baseline errors in EmbeddingsModelSelectionConfig.tsx, persona-visuals.ts, and lib/api/vnPlay.ts. Broader Admin Vitest directory still has an unrelated ServerAdminPage.media-budget test failure. Playwright required an approved rerun because sandboxed dev-server bind to port 8080 failed with EPERM; approved rerun passed. Backend focused pytest emitted post-success Loguru cleanup warnings after the pass.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the approved self-hosted llama.cpp management WebUI improvement set end to end. Admins can inspect readiness and restart-required state, browse safe GGUF inventory, start models by stable model ID, keep hardware warnings advisory, explicitly wire a running managed server into Chat, and rely on bounded log/provider diagnostics. Docs, frontend smoke coverage, focused backend/frontend tests, Bandit, and final task tracking are complete with unrelated baseline limitations documented.
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
