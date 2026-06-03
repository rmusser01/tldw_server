---
id: TASK-601
title: Implement solo onboarding local model guided alternative V2
status: In Progress
references:
- https://github.com/rmusser01/tldw_server/pull/2227
documentation:
- Docs/superpowers/specs/2026-06-02-solo-onboarding-v2-roadmap-design.md
modified_files:
- Docs/superpowers/plans/2026-06-03-solo-onboarding-local-model-v2-plan.md
- apps/packages/ui/src/components/Option/Onboarding/__tests__/ProviderSetupStep.test.tsx
- apps/packages/ui/src/components/Option/Onboarding/steps/ProviderSetupStep.tsx
- apps/tldw-frontend/scripts/__tests__/onboarding-uat-runner.test.ts
- tldw_Server_API/app/core/Setup/provider_validation.py
- tldw_Server_API/tests/Setup/test_setup_provider_validation.py
- backlog/tasks/task-601 - Implement-solo-onboarding-local-model-guided-alternative-V2.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement PR4 from the solo onboarding V2 roadmap. Start by extending the onboarding UAT harness with local OpenAI-compatible/Ollama scenarios, then improve the WebUI local setup flow with endpoint guidance, non-generative validation, model discovery when supported, manual model entry fallback, and inline recovery actions. Preserve hosted/local as peer setup choices while keeping local runtime installation out of app ownership.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-03-solo-onboarding-local-model-v2-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Stage 2 complete. Added backend manual-model fallback for reachable local OpenAI-compatible endpoints when model discovery is unavailable. Red run before implementation: 2 new provider validation tests failed as expected. Green verification: source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Setup/test_setup_provider_validation.py -q -> 28 passed, 6 warnings.

Stage 3 complete. Added frontend local endpoint guidance, manual model-discovery fallback copy/actions, and endpoint-unreachable retry/edit/switch recovery controls. Red run before implementation: 2 new ProviderSetupStep tests failed as expected. Green verification: bunx vitest run ../packages/ui/src/components/Option/Onboarding/__tests__/ProviderSetupStep.test.tsx --reporter=dot -> 21 passed.
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
