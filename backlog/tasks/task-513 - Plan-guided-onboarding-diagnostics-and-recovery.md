---
id: TASK-513
title: Plan guided onboarding diagnostics and recovery
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
- Docs/superpowers/specs/2026-06-02-solo-onboarding-v2-roadmap-design.md
documentation:
- Docs/superpowers/plans/2026-06-02-guided-onboarding-diagnostics-recovery-implementation-plan.md
modified_files:
- Docs/superpowers/plans/2026-06-02-guided-onboarding-diagnostics-recovery-implementation-plan.md
- apps/tldw-frontend/e2e/onboarding-uat/recovery.spec.ts
- apps/packages/ui/src/components/Option/Onboarding/onboarding-diagnostics.ts
- apps/packages/ui/src/components/Option/Onboarding/OnboardingConnectForm.tsx
- apps/packages/ui/src/components/Option/Playground/PlaygroundChatErrorBanner.tsx
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create the implementation plan for PR2 of the solo onboarding V2 roadmap: harness-first blocked/degraded setup scenarios followed by inline diagnostics and recovery actions in the focused setup shell.
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
Plan created and reviewed. Acceptance targets: PR2-only scope; harness-first recovery scenarios using mock_openai_server; setup/readiness/chat categories mapped to safe inline recovery actions; PR3 starter questions and PR4 local-provider discovery kept out of scope; verification includes targeted Vitest, onboarding UAT happy-path desktop/mobile, recovery UAT, and Bandit skip if no Python code changes.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created and reviewed the PR2 guided onboarding diagnostics/recovery implementation plan at Docs/superpowers/plans/2026-06-02-guided-onboarding-diagnostics-recovery-implementation-plan.md. The plan is scoped to recovery diagnostics only, starts with real UAT failure scenarios using mock_openai_server, preserves PR3/PR4 scope boundaries, and lists targeted Vitest/UAT/security verification. No Python code was changed, so Bandit is not applicable for this planning-only task.
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
