---
id: TASK-45.44.11.2
title: Migrate Persona AssistantVoiceCard wake warning alert to design-system Alert
status: Done
labels:
- design-system
- webui
- persona
- product-state
parent_task_id: TASK-45.44.11
references:
- https://github.com/rmusser01/tldw_server/issues/1668
modified_files:
- apps/packages/ui/src/components/PersonaGarden/AssistantVoiceCard.tsx
- apps/packages/ui/src/components/PersonaGarden/__tests__/LiveSessionPanel.test.tsx
- apps/packages/ui/src/components/Common/QuickIngest/ReviewStep.tsx
- apps/packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.integration.test.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remove the remaining Persona Garden AssistantVoiceCard AntD Alert product-state baseline entry by migrating the wake-warning message to the shared design-system Alert primitive while preserving live voice and wake behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 AssistantVoiceCard wake-warning UI uses the shared design-system Alert primitive instead of AntD Alert.
- [x] #2 Focused Persona Garden coverage verifies the wake warning renders through the design-system Alert contract.
- [x] #3 The AssistantVoiceCard AntD Alert entry is removed from the design-system product-state baseline.
- [x] #4 `bun run verify:design-system-state` passes from `apps/packages/ui`.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Replaced the `AssistantVoiceCard` wake-warning AntD `Alert` usage with `Alert as DesignSystemAlert` from `@/components/ui/primitives`.
- Added a focused regression test asserting the wake-warning text is rendered inside an element with `data-ds-component="Alert"`.
- Removed the `AssistantVoiceCard.tsx:Alert` entry from `apps/packages/ui/scripts/design-system-product-state-baseline.json`.
- Review follow-up: rebased onto current `dev`, migrated the newly inherited QuickIngest review-step offline warning to the design-system Alert primitive, and added coverage for that review-path alert contract.
- `bun run verify:design-system-state` passes from `apps/packages/ui`; the report still lists existing allowed/stale baseline inventory, but no blocked product-state findings remain.
- Bandit skipped because this is a TypeScript UI-only change.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated the Persona Garden AssistantVoiceCard wake-warning UI from AntD Alert to the shared design-system Alert primitive, added focused coverage for the DS contract, removed the corresponding product-state baseline exception, and addressed the PR review by bringing the current QuickIngest review-step offline warning onto the same design-system Alert path so the product-state verifier passes.
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
