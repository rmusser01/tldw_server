---
id: TASK-493
title: Implement onboarding frontend setup client and shell
status: Done
references:
- TASK-489
- Docs/superpowers/plans/2026-05-31-first-time-solo-user-onboarding-implementation-plan.md
documentation:
- Docs/superpowers/plans/2026-05-31-first-time-solo-user-onboarding-implementation-plan.md#task-6-frontend-setup-api-domain-and-hook
- Docs/superpowers/plans/2026-05-31-first-time-solo-user-onboarding-implementation-plan.md#task-7-focused-webui-setup-shell-and-wizard-skeleton
modified_files:
- Docs/superpowers/plans/2026-05-31-first-time-solo-user-onboarding-implementation-plan.md
- apps/packages/ui/src/types/setup-onboarding.ts
- apps/packages/ui/src/services/tldw/domains/setup-onboarding.ts
- apps/packages/ui/src/services/tldw/domains/index.ts
- apps/packages/ui/src/services/tldw/domains/media.ts
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
- apps/packages/ui/src/components/Option/Onboarding/__tests__/MultiUserExitPanel.test.tsx
- apps/packages/ui/src/routes/__tests__/option-index.unified-setup.test.tsx
- apps/packages/ui/src/components/Option/Onboarding/__tests__/OnboardingConnectForm.design-system.test.tsx
- apps/packages/ui/src/routes/__tests__/core-route-identity.test.tsx
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Task 6-7 frontend slice from the unified onboarding plan. Add setup onboarding API domain/hook and replace first-run routing with a focused setup shell, setup path step, privacy/security step, and multi-user exit.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Frontend client exposes first-run state, metadata, provider, settings, first-chat, complete, and skip APIs
- [x] #2 Focused first-run shell hides normal app navigation while setup is required
- [x] #3 Privacy/security step uses backend metadata and requires acknowledgement before provider setup
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented the Task 6-7 frontend slice with a setup onboarding API domain, hook, focused WebUI setup shell, shared `/setup` recovery wizard, setup-path step, privacy/security acknowledgement step, and multi-user exit guidance.

Setup onboarding API calls use the unauthenticated setup surface so first-run users are not blocked before auth is configured. Wizard step transitions now wait for backend state persistence and surface retryable errors instead of advancing on failed saves.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Frontend setup domain and hook now expose first-run state, metadata, provider catalog/save/validate, ingest/audio/advanced defaults, first-chat verification, completion, and skip APIs. The WebUI home route is backend-authoritative for first-run gating and renders a focused setup shell while setup is required; /setup reuses the same wizard as the operator recovery surface. Privacy/security uses backend metadata and requires acknowledgement before provider setup. Verification: targeted Vitest suite passed (8 files, 28 tests); bun run verify:openapi passed for 270 ClientPath entries with the existing 10 reviewed exceptions; git diff --check passed. Bandit skipped because this slice touched frontend/docs/task files only, no backend Python production code. Known blocker: bunx tsc --noEmit --pretty false still fails on existing repo-wide TypeScript baseline errors outside this slice.
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
