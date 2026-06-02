---
id: TASK-514
title: Implement guided onboarding diagnostics and recovery
status: Done
labels:
- onboarding
- webui
- uat
- setup
priority: high
references:
- TASK-504
- TASK-506
- TASK-513
- Docs/superpowers/plans/2026-06-02-guided-onboarding-diagnostics-recovery-implementation-plan.md
modified_files:
- Docs/superpowers/plans/2026-06-02-guided-onboarding-diagnostics-recovery-implementation-plan.md
- apps/tldw-frontend/e2e/onboarding-uat/recovery.spec.ts
- apps/tldw-frontend/e2e/onboarding-uat/helpers.ts
- apps/tldw-frontend/e2e/onboarding-uat/scenarios.ts
- apps/tldw-frontend/e2e/onboarding-uat/mock-openai/configs/model-unavailable.json
- apps/tldw-frontend/scripts/__tests__/onboarding-uat-runner.test.ts
- apps/packages/ui/src/components/Option/Onboarding/onboarding-diagnostics.ts
- apps/packages/ui/src/components/Option/Onboarding/__tests__/onboarding-diagnostics.test.ts
- apps/packages/ui/src/components/Option/Onboarding/OnboardingConnectForm.tsx
- apps/packages/ui/src/components/Option/Onboarding/__tests__/OnboardingConnectForm.connection-ui.test.ts
- apps/packages/ui/src/components/Option/Onboarding/validation.ts
- apps/packages/ui/src/components/Option/Onboarding/__tests__/validation.test.ts
- apps/packages/ui/src/components/Option/Playground/PlaygroundChatErrorBanner.tsx
- apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundChatErrorBanner.test.tsx
- apps/packages/ui/src/components/Option/Playground/PlaygroundForm.tsx
- apps/packages/ui/src/hooks/chat-modes/chatModePipeline.ts
- apps/packages/ui/src/hooks/chat-modes/__tests__/chatModePipeline.dynamic-ui.test.ts
- apps/packages/ui/src/services/tldw/TldwAuth.ts
- apps/packages/ui/src/services/__tests__/tldw-auth.api-key-validation.test.ts
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement PR2 of the solo onboarding V2 roadmap: add real UAT failure/recovery scenarios, safe setup diagnostics, first-chat recovery actions, readiness overlays, and verification for first-time solo onboarding recovery.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-02-guided-onboarding-diagnostics-recovery-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented PR2 guided onboarding diagnostics/recovery.

Summary:
- Added real recovery UAT scenarios for setup endpoint recovery, transient provider retry, and model unavailable recovery using the existing isolated backend/WebUI/mock_openai_server harness.
- Added safe setup diagnostic mapping and a focused setup diagnostic panel with stable test ids/actions.
- Fixed setup connection validation to test the candidate server URL directly and avoid stale success state races.
- Added readiness diagnostic mapping for restart/config/network/download/install/RAG-storage/audio categories, with optional RAG/audio lanes deferrable before first chat unless selected.
- Added first-chat composer recovery actions: retry, edit provider, switch provider/model, diagnostics, and safe dismiss behavior.
- Fixed empty completed provider streams so they become recoverable assistant errors instead of blank successful assistant messages.
- Fixed failed chat submit handling so failed-but-handled chat results do not clear the new recovery banner or mark onboarding first chat complete.

Verification:
- Passed: bunx vitest run scripts/__tests__/onboarding-uat-runner.test.ts ../packages/ui/src/components/Option/Onboarding/__tests__/OnboardingConnectForm.connection-ui.test.ts ../packages/ui/src/components/Option/Onboarding/__tests__/OnboardingConnectForm.design-system.test.tsx ../packages/ui/src/components/Option/Onboarding/__tests__/onboarding-diagnostics.test.ts ../packages/ui/src/components/Option/Onboarding/__tests__/validation.test.ts ../packages/ui/src/services/__tests__/tldw-auth.api-key-validation.test.ts ../packages/ui/src/components/Option/Playground/__tests__/PlaygroundChatErrorBanner.test.tsx ../packages/ui/src/hooks/chat-modes/__tests__/chatModePipeline.dynamic-ui.test.ts (8 files, 56 tests).
- Passed: bun run e2e:onboarding:uat -- --scenario setup-endpoint-recovery --viewport desktop --mock-config hosted-success.json. Latest artifact summary: apps/tldw-frontend/test-results/onboarding-uat/2026-06-02T18-47-05-900Z-65vaxv/summary.json.
- Passed: bun run e2e:onboarding:uat -- --scenario provider-retry-recovery --viewport desktop --mock-config chat-fail-once.json. Later UAT runs cleaned this artifact directory.
- Passed: bun run e2e:onboarding:uat -- --scenario model-unavailable-recovery --viewport desktop --mock-config model-unavailable.json. Later UAT runs cleaned this artifact directory.
- Passed: bun run lint -- ...touched files... from apps/tldw-frontend, exited 0 with existing warnings; package UI paths were outside lint base and ignored by that config.
- Passed: git diff --check.
- Failed baseline: ./apps/tldw-frontend/node_modules/.bin/tsc -p apps/packages/ui/tsconfig.json --noEmit fails on existing package-wide baseline type errors across unrelated tests/modules; one untouched onboarding design-system test also appears in that baseline.
- Bandit not run because no Python files changed.

Known follow-up:
- The readiness diagnostic mapper is in place, but no broader readiness overlay was wired because OnboardingConnectForm does not currently receive a backend readiness issue payload beyond connection/setup checks. UI wiring should follow once that authoritative setup payload is exposed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented guided onboarding diagnostics and recovery for PR2. The first-run setup shell now has safe categorized diagnostics, candidate URL validation, stale-success protection, and recovery actions. First-chat failures now surface inline retry/edit/switch recovery, failed chat submissions no longer mark onboarding complete, and empty provider streams are converted into recoverable assistant errors. Recovery UAT coverage now exercises setup endpoint recovery, transient provider retry, and model-unavailable recovery through the isolated backend/WebUI/mock OpenAI harness.
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
