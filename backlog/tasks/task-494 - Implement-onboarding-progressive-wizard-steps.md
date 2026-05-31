---
id: TASK-494
title: Implement onboarding progressive wizard steps
status: To Do
references:
- TASK-489
- Docs/superpowers/plans/2026-05-31-first-time-solo-user-onboarding-implementation-plan.md
documentation:
- Docs/superpowers/plans/2026-05-31-first-time-solo-user-onboarding-implementation-plan.md#task-8-provider-ingest-audio-advanced-and-first-chat-wizard-steps
modified_files:
- apps/packages/ui/src/components/Option/Onboarding/UnifiedSetupWizard.tsx
- apps/packages/ui/src/components/Option/Onboarding/steps/ProviderSetupStep.tsx
- apps/packages/ui/src/components/Option/Onboarding/steps/IngestDefaultsStep.tsx
- apps/packages/ui/src/components/Option/Onboarding/steps/AudioSetupStep.tsx
- apps/packages/ui/src/components/Option/Onboarding/steps/OptionalAdvancedStep.tsx
- apps/packages/ui/src/components/Option/Onboarding/steps/FirstChatStep.tsx
- apps/packages/ui/src/components/Option/Onboarding/FirstSourceMilestonePrompt.tsx
- apps/packages/ui/src/components/Option/Onboarding/__tests__/ProviderSetupStep.test.tsx
- apps/packages/ui/src/components/Option/Onboarding/__tests__/FirstChatStep.test.tsx
- apps/packages/ui/src/components/Option/Onboarding/__tests__/FirstSourceMilestonePrompt.test.tsx
- apps/packages/ui/src/routes/option-index.tsx
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Task 8 frontend slice from the unified onboarding plan. Add provider, ingest defaults, audio defaults, optional advanced, first chat, and first-source milestone UI steps and wire them into the unified wizard.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Wizard supports multiple provider saves and one default provider/model
- [ ] #2 First chat step only completes after backend success and displays model response
- [ ] #3 Completed onboarding shows post-onboarding first-source milestone without blocking navigation
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
