---
id: TASK-45.53
title: Migrate KnowledgeQA SetupDiagnostics labels to design-system registry
status: To Do
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
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue the tldw_server WebUI design-system migration by routing the remaining KnowledgeQA SetupDiagnostics canonical Ready and Blocked labels through the shared design-system state registry instead of local string literals. Preserve existing diagnostics rendering and tests while removing the current product-state guard blockers for apps/packages/ui/src/components/Option/KnowledgeQA/SetupDiagnostics.tsx.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 SetupDiagnostics Ready label comes from the design-system state registry.
- [ ] #2 SetupDiagnostics Blocked label comes from the design-system state registry.
- [ ] #3 Focused KnowledgeQA diagnostics coverage preserves existing rendered labels.
- [ ] #4 Direct product-state guard scan over SetupDiagnostics reports zero findings.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
