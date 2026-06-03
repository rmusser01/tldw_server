---
id: TASK-605
title: Implement solo onboarding local model guided alternative V2
status: In Progress
references:
- https://github.com/rmusser01/tldw_server/pull/2227
- https://github.com/rmusser01/tldw_server/pull/2236
documentation:
- Docs/superpowers/specs/2026-06-02-solo-onboarding-v2-roadmap-design.md
modified_files:
- Docs/superpowers/plans/2026-06-03-solo-onboarding-local-model-v2-plan.md
- apps/packages/ui/src/components/Option/Onboarding/__tests__/ProviderSetupStep.test.tsx
- apps/packages/ui/src/components/Option/Onboarding/steps/ProviderSetupStep.tsx
- apps/tldw-frontend/e2e/onboarding-uat/helpers.ts
- apps/tldw-frontend/e2e/onboarding-uat/mock-openai/configs/local-model-unavailable.json
- apps/tldw-frontend/e2e/onboarding-uat/mock-openai/configs/local-models-unavailable.json
- apps/tldw-frontend/e2e/onboarding-uat/playwright.config.ts
- apps/tldw-frontend/e2e/onboarding-uat/recovery.spec.ts
- apps/tldw-frontend/e2e/onboarding-uat/scenarios.ts
- apps/tldw-frontend/e2e/onboarding-uat/setup-happy-path.spec.ts
- apps/tldw-frontend/scripts/__tests__/onboarding-uat-runner.test.ts
- apps/tldw-frontend/scripts/onboarding-uat/run.mjs
- tldw_Server_API/app/core/Setup/provider_validation.py
- tldw_Server_API/tests/Setup/test_setup_provider_validation.py
- backlog/tasks/task-605 - Implement-solo-onboarding-local-model-guided-alternative-V2.md
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

Stage 4 complete. Added explicit local-provider UAT scenarios for discovered models, manual model fallback, local model unavailable recovery, and local-to-hosted switch recovery. Added mock OpenAI configs for /models unavailable and selected local model unavailable. Runner now resolves the registered scenario mock config when --mock-config is omitted and writes Playwright artifacts outside the runner artifact root. Red run before implementation: static runner tests failed on missing local configs/scenarios; later red checks caught scenario-only mock config mismatch, unsafe cleanup output overlap, and full-catalog switch choosing Anthropic before OpenAI. Green verification: bunx vitest run scripts/__tests__/onboarding-uat-runner.test.ts --reporter=dot -> 27 passed; bunx vitest run ../packages/ui/src/components/Option/Onboarding/__tests__/ProviderSetupStep.test.tsx --reporter=dot -> 22 passed. Real UAT: local-openai-manual-model-first-chat desktop passed; local-openai-discovered-model-first-chat desktop passed; local-openai-discovered-model-first-chat mobile passed; local-openai-model-unavailable-recovery desktop passed; local-to-hosted-switch-state-isolated desktop passed; setup-endpoint-recovery desktop passed. Note: do not run first-run desktop+mobile in one --viewport all execution against one runtime profile because the first viewport completes first-run setup for the second viewport.

Stage 5 verification complete. Focused verification: bunx vitest run scripts/__tests__/onboarding-uat-runner.test.ts --reporter=dot -> 27 passed; bunx vitest run ../packages/ui/src/components/Option/Onboarding/__tests__/ProviderSetupStep.test.tsx --reporter=dot -> 22 passed; source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Setup/test_setup_provider_validation.py -q -> 28 passed, 6 warnings; Bandit touched setup scope -> 0 findings in /tmp/bandit_onboarding_local_model_v2.json; git diff --check -> clean. Draft PR opened at https://github.com/rmusser01/tldw_server/pull/2236. Worktree status has two unrelated untracked watchlist template files left untouched.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented guided local model onboarding V2 across backend validation, WebUI provider setup, and repeatable UAT harness coverage. Local OpenAI-compatible endpoints now support discovered models and a backend-authoritative manual model fallback when /models is unavailable, while endpoint failures show inline recovery. The UAT runner now maps scenario IDs to their mock configs, keeps Playwright output separate from runner artifacts, and covers manual local setup, discovered local setup on desktop/mobile, local model unavailable recovery, endpoint recovery, and local-to-hosted switching.
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
