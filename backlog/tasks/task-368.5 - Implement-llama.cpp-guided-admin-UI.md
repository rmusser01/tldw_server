---
id: TASK-368.5
title: Implement llama.cpp guided admin UI
status: Done
assignee: []
created_date: '2026-05-15 03:45'
updated_date: '2026-05-15 15:34'
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
- [x] #1 The page renders readiness state with saved active and restart required messaging.
- [x] #2 The page renders model inventory and starts models by stable model ID.
- [x] #3 Hardware warnings are shown without disabling start solely because hardware data is unknown or risky.
- [x] #4 The chat wiring action appears after a running managed server is available and is never called automatically.
- [x] #5 Focused frontend component tests pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Started Task 5 guided admin UI slice after TASK-368.4 was finalized and committed at cdb0578d4. Scope is the llama.cpp admin page/panel components and focused tests from the implementation plan. Preserve existing advanced launch controls, admin guard behavior, preset import/export, and warnings-first hardware behavior.

Implemented guided llama.cpp admin UI in commits 18c02cfee and 5432f53a: readiness, inventory, and launch panels; stable inventory model_id start flow; explicit Use this in Chat action; local GGUF path registration; advisory hardware warnings.
Verification: focused Vitest suite passed for LlamacppAdminPage, LlamacppReadinessPanel, LlamacppInventoryPanel, and LlamacppLaunchPanel: 14 tests passed. git diff --check passed. Follow-up spec/code reviews reported no remaining Critical or Important findings.
Known skips/limitations: frontend tsc still fails on unrelated baseline files EmbeddingsModelSelectionConfig.tsx, persona-visuals.ts, and lib/api/vnPlay.ts. Broader Admin test directory still has an unrelated ServerAdminPage.media-budget test failure. Bandit is not applicable for this frontend-only slice. Docs/E2E updates are deferred to TASK-368.6.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the guided llama.cpp admin UI. The page now separates readiness, inventory, and launch concerns, starts selected GGUF models by stable model_id, preserves advanced launch controls, keeps hardware probe failures advisory, keeps failed registration input for retry, and exposes chat-provider wiring only as an explicit post-running action. Focused UI tests pass; known unrelated frontend baseline failures are documented in notes.
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
