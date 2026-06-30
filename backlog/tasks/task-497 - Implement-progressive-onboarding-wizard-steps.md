---
id: TASK-497
title: Implement progressive onboarding wizard steps
status: Done
references:
- TASK-489
- Docs/superpowers/plans/2026-05-31-first-time-solo-user-onboarding-implementation-plan.md
documentation:
- Docs/superpowers/plans/2026-05-31-first-time-solo-user-onboarding-implementation-plan.md#task-8-provider-ingest-audio-advanced-and-first-chat-wizard-steps
modified_files:
- apps/packages/ui/src/components/Option/Onboarding/steps/ProviderSetupStep.tsx
- apps/packages/ui/src/components/Option/Onboarding/steps/IngestDefaultsStep.tsx
- apps/packages/ui/src/components/Option/Onboarding/steps/AudioSetupStep.tsx
- apps/packages/ui/src/components/Option/Onboarding/steps/OptionalAdvancedStep.tsx
- apps/packages/ui/src/components/Option/Onboarding/steps/FirstChatStep.tsx
- apps/packages/ui/src/components/Option/Onboarding/FirstSourceMilestonePrompt.tsx
- apps/packages/ui/src/components/Option/Onboarding/UnifiedSetupWizard.tsx
- apps/packages/ui/src/components/Option/Onboarding/__tests__/UnifiedSetupWizard.test.tsx
- apps/packages/ui/src/components/Option/Onboarding/__tests__/ProviderSetupStep.test.tsx
- apps/packages/ui/src/components/Option/Onboarding/__tests__/FirstChatStep.test.tsx
- apps/packages/ui/src/components/Option/Onboarding/__tests__/FirstSourceMilestonePrompt.test.tsx
- apps/packages/ui/src/routes/option-index.tsx
- apps/packages/ui/src/routes/__tests__/option-index.unified-setup.test.tsx
- apps/packages/ui/src/hooks/useSetupOnboarding.ts
- apps/packages/ui/src/services/tldw/domains/setup-onboarding.ts
- apps/packages/ui/src/services/tldw/__tests__/setup-onboarding.test.ts
- apps/packages/ui/src/services/tldw/openapi-guard.ts
- apps/packages/ui/src/types/setup-onboarding.ts
- tldw_Server_API/app/api/v1/endpoints/setup.py
- tldw_Server_API/tests/integration/test_unified_first_run_setup_api.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the progressive first-run WebUI wizard steps after the setup shell: provider configuration, ingest defaults, audio/STT/TTS defaults, optional advanced choices, first-chat completion, and the first-source post-onboarding milestone.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Provider setup supports hosted API keys and local OpenAI-compatible endpoints, persists provider/model setup through backend setup APIs, and never redisplays raw saved secrets.
- [x] Wizard order follows setup path, privacy/security, providers, ingest defaults, audio/STT/TTS, optional advanced, and first chat, with backend progress persisted after each successful step.
- [x] First-run completion requires a successful backend first-chat verification before calling setup completion.
- [x] After backend setup completion, the normal app shell offers an add-your-first-source milestone using frontend-local dismiss state only.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Implemented focused provider, ingest, audio, optional advanced, and first-chat steps.
- Added setup audio recommendations client/hook support for `/api/v1/setup/audio/recommendations`.
- Wired the completed-state first-source milestone prompt to Quick Ingest with frontend-local dismiss persistence.
- Persisted the selected first-chat model in public sanitized backend first-run state so wizard resume remains backend-authoritative.
- Verification: `bunx vitest run src/components/Option/Onboarding/__tests__/UnifiedSetupWizard.test.tsx src/components/Option/Onboarding/__tests__/ProviderSetupStep.test.tsx src/components/Option/Onboarding/__tests__/FirstChatStep.test.tsx src/components/Option/Onboarding/__tests__/FirstSourceMilestonePrompt.test.tsx src/services/tldw/__tests__/setup-onboarding.test.ts src/hooks/__tests__/useSetupOnboarding.test.tsx src/routes/__tests__/option-index.unified-setup.test.tsx src/routes/__tests__/core-route-identity.test.tsx` passed, 27 tests.
- Verification: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/integration/test_unified_first_run_setup_api.py -k first_run_state` passed, 18 tests.
- Verification: `bun run verify:openapi` passed.
- Verification: `git diff --check` passed.
- Known baseline: `bunx tsc --noEmit --pretty false` exits 2 from unrelated existing TS errors; filtered log has no onboarding/setup touched-file hits.
- Bandit: `source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/api/v1/endpoints/setup.py -f json -o /tmp/bandit_task497.json` passed with zero findings.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Progressive first-run wizard steps are implemented and wired into the solo setup shell. Provider/model resume is backend-authoritative through public sanitized first-run state. Focused component, hook, service, route, and backend integration tests pass. OpenAPI client path verification and Bandit pass. TypeScript remains blocked by existing baseline errors outside this onboarding slice; the filtered tsc log has no touched-file hits.
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
