---
id: TASK-493
title: Implement onboarding frontend setup client and shell
status: To Do
references:
- TASK-489
- Docs/superpowers/plans/2026-05-31-first-time-solo-user-onboarding-implementation-plan.md
documentation:
- Docs/superpowers/plans/2026-05-31-first-time-solo-user-onboarding-implementation-plan.md#task-6-frontend-setup-api-domain-and-hook
- Docs/superpowers/plans/2026-05-31-first-time-solo-user-onboarding-implementation-plan.md#task-7-focused-webui-setup-shell-and-wizard-skeleton
modified_files:
- apps/packages/ui/src/types/setup-onboarding.ts
- apps/packages/ui/src/services/tldw/domains/setup-onboarding.ts
- apps/packages/ui/src/services/tldw/domains/index.ts
- apps/packages/ui/src/services/tldw/TldwApiClient.ts
- apps/packages/ui/src/services/tldw/openapi-guard.ts
- apps/packages/ui/src/hooks/useSetupOnboarding.ts
- apps/packages/ui/src/components/Option/Onboarding/UnifiedSetupWizard.tsx
- apps/packages/ui/src/components/Option/Onboarding/steps/SetupPathStep.tsx
- apps/packages/ui/src/components/Option/Onboarding/steps/PrivacySecurityStep.tsx
- apps/packages/ui/src/components/Option/Onboarding/steps/MultiUserExitPanel.tsx
- apps/packages/ui/src/routes/option-index.tsx
- apps/packages/ui/src/routes/option-setup.tsx
- apps/packages/ui/src/services/tldw/__tests__/setup-onboarding.test.ts
- apps/packages/ui/src/hooks/__tests__/useSetupOnboarding.test.tsx
- apps/packages/ui/src/components/Option/Onboarding/__tests__/UnifiedSetupWizard.test.tsx
- apps/packages/ui/src/routes/__tests__/option-index.unified-setup.test.tsx
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Task 6-7 frontend slice from the unified onboarding plan. Add setup onboarding API domain/hook and replace first-run routing with a focused setup shell, setup path step, privacy/security step, and multi-user exit.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Frontend client exposes first-run state, metadata, provider, settings, first-chat, complete, and skip APIs
- [ ] #2 Focused first-run shell hides normal app navigation while setup is required
- [ ] #3 Privacy/security step uses backend metadata and requires acknowledgement before provider setup
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
