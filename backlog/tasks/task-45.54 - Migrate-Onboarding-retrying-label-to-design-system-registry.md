---
id: TASK-45.54
title: Migrate Onboarding retrying label to design-system registry
status: To Do
labels:
- design-system
- webui
- product-state
- onboarding
parent_task_id: TASK-45
references:
- apps/packages/ui/src/components/Option/Onboarding/steps/FirstChatStep.tsx
- apps/packages/ui/src/design-system/states.ts
- apps/packages/ui/scripts/verify-design-system-product-state.mjs
documentation:
- Docs/Design/tldw_web_design_system_contract.md
- Docs/Design/tldw_web_design_system_inventory.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue the tldw_server WebUI design-system migration by routing the remaining Onboarding FirstChatStep canonical Retrying label through the shared design-system state registry instead of a local string literal. Preserve existing first-chat retry behavior and tests while removing the current product-state guard blocker for apps/packages/ui/src/components/Option/Onboarding/steps/FirstChatStep.tsx.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 FirstChatStep Retrying label comes from the design-system state registry.
- [ ] #2 Existing first-chat retry behavior and copy are preserved.
- [ ] #3 Focused Onboarding coverage preserves the retrying state label.
- [ ] #4 Direct product-state guard scan over FirstChatStep reports zero findings.
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
