---
id: TASK-45.53
title: Migrate KnowledgeQA SetupDiagnostics labels to design-system registry
status: Done
labels:
- design-system
- webui
- product-state
- knowledge-qa
parent_task_id: TASK-45
references:
- apps/packages/ui/src/components/Option/KnowledgeQA/SetupDiagnostics.tsx
- apps/packages/ui/src/design-system/states.ts
- apps/packages/ui/scripts/verify-design-system-product-state.mjs
documentation:
- Docs/Design/tldw_web_design_system_contract.md
- Docs/Design/tldw_web_design_system_inventory.md
modified_files:
- apps/packages/ui/src/components/Option/KnowledgeQA/SetupDiagnostics.tsx
- apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/SetupDiagnostics.design-system.test.tsx
- backlog/tasks/task-45.53 - Migrate-KnowledgeQA-SetupDiagnostics-labels-to-design-system-registry.md
- backlog/tasks/task-45.44 - Track-remaining-tldw-design-system-migration-and-governance.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue the tldw_server WebUI design-system migration by routing the remaining KnowledgeQA SetupDiagnostics canonical Ready and Blocked labels through the shared design-system state registry instead of local string literals. Preserve existing diagnostics rendering and tests while removing the current product-state guard blockers for apps/packages/ui/src/components/Option/KnowledgeQA/SetupDiagnostics.tsx.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 SetupDiagnostics Ready label comes from the design-system state registry.
- [x] #2 SetupDiagnostics Blocked label comes from the design-system state registry.
- [x] #3 Focused KnowledgeQA diagnostics coverage preserves existing rendered labels.
- [x] #4 Direct product-state guard scan over SetupDiagnostics reports zero findings.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Routed KnowledgeQA SetupDiagnostics complete and blocked diagnostic status labels through the shared design-system state registry by using READY_STATE_LABEL and BLOCKED_STATE_LABEL. Added focused design-system regression coverage that mocks the registry labels and verifies complete and blocked diagnostic rows render those registry labels. Verification: focused SetupDiagnostics design-system test failed red before the implementation and now passes 2/2; adjacent KnowledgeQA.connection test passes 17/17; direct product-state guard scan over SetupDiagnostics reports no issues; git diff --check passes. Full verify:design-system-state no longer reports KnowledgeQA and now fails only on the remaining Onboarding Retrying and ACP readiness Setup required blockers plus allowed legacy baseline inventory. UI TypeScript still fails on unrelated Notes/background/voice-cloning debt with no touched-file diagnostics. Bandit is not applicable because this slice touched frontend TypeScript/TSX and Backlog markdown only.
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
