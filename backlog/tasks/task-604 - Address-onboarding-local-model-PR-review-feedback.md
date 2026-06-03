---
id: TASK-604
title: Address onboarding local model PR review feedback
status: Done
assignee: []
created_date: ''
updated_date: 2026-06-03 04:01
labels:
- onboarding
- review-feedback
- local-models
dependencies: []
priority: medium
modified_files:
- Docs/superpowers/plans/2026-06-03-solo-onboarding-local-model-v2-plan.md
- apps/packages/ui/src/components/Option/Onboarding/__tests__/ProviderSetupStep.test.tsx
- apps/packages/ui/src/components/Option/Onboarding/steps/ProviderSetupStep.tsx
- backlog/tasks/task-604 - Address-onboarding-local-model-PR-review-feedback.md
- backlog/tasks/task-605 - Implement-solo-onboarding-local-model-guided-alternative-V2.md
- tldw_Server_API/app/core/Setup/provider_validation.py
- tldw_Server_API/tests/Setup/test_setup_provider_validation.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rebase PR #2236 on latest dev and resolve review feedback for first-time solo onboarding local model setup, including OpenAI-compatible local endpoint validation parity and manual model fallback recovery behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 PR branch rebased onto latest origin/dev
- [x] #2 Review feedback from Gemini/Qodo/CodeRabbit is evaluated and either fixed or documented
- [x] #3 OpenAI-compatible local model validation probes the same versioned URL shape used by runtime chat adapters
- [x] #4 Manual model fallback recovery works when the local provider is not currently the first-chat default
- [x] #5 Targeted backend/frontend tests, diff check, and Bandit on touched backend code are recorded
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
PR #2236 rebased cleanly onto origin/dev at 61124888aa. Evaluated review feedback: CodeRabbit review was skipped/no actionable findings; Gemini requested replacing DOM-based field focus with React refs; Qodo reported two correctness bugs. Fixed the backend validator so OpenAI-compatible local model discovery uses runtime-compatible /v1/models normalization when the configured base URL omits /v1. Fixed the frontend manual-model fallback recovery so a selected non-default local provider is promoted to first-chat default and the model input is focused through React refs instead of document.getElementById. Added regression coverage for both issues. Rebase also exposed a duplicate Backlog id because origin/dev now contains a different TASK-601; renamed the onboarding task record and plan references from TASK-601 to TASK-605.

Verification: source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Setup/test_setup_provider_validation.py -q -> 29 passed, 6 warnings; bunx vitest run ../packages/ui/src/components/Option/Onboarding/__tests__/ProviderSetupStep.test.tsx --reporter=dot -> 23 passed; bunx vitest run scripts/__tests__/onboarding-uat-runner.test.ts --reporter=dot -> 27 passed; bun run e2e:onboarding:uat -- --scenario local-openai-manual-model-first-chat --viewport desktop --no-preserve-artifacts -> passed after rerun with local port-binding escalation; python -m bandit -r tldw_Server_API/app/core/Setup/provider_validation.py -f json -o /tmp/bandit_onboarding_local_model_v2_review.json -> 0 findings; git diff --check -> passed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed PR #2236 review feedback after rebasing onto latest dev. Local OpenAI-compatible validation now probes the runtime-compatible /v1/models URL when needed, manual model recovery works for non-default local providers through React refs, and the rebase-created Backlog task id collision was resolved by renaming the onboarding task to TASK-605.
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
