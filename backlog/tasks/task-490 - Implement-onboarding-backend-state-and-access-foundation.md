---
id: TASK-490
title: Implement onboarding backend state and access foundation
status: In Progress
assignee: []
created_date: ''
updated_date: '2026-05-31 06:15'
labels: []
dependencies: []
references:
  - TASK-489
  - >-
    Docs/superpowers/plans/2026-05-31-first-time-solo-user-onboarding-implementation-plan.md
documentation:
  - >-
    Docs/superpowers/plans/2026-05-31-first-time-solo-user-onboarding-implementation-plan.md#task-1-backend-first-run-state-store
  - >-
    Docs/superpowers/plans/2026-05-31-first-time-solo-user-onboarding-implementation-plan.md#task-2-setup-access-boundary-and-first-run-state-endpoints
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Task 1-2 slice from the unified onboarding plan. Add durable first-run state, required acknowledgement semantics, setup metadata, setup write access gating, and first-run state/skip endpoints.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 First-run state store persists state and enforces first chat plus acknowledged required steps before completion
- [ ] #2 First-run metadata endpoint returns auth/setup-path/origin diagnostics without secrets
- [ ] #3 First-run write endpoints are blocked when setup is disabled or already completed
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Started Task 1 subagent-driven slice: backend first-run state store and setup schemas. Baseline before implementation: tldw_Server_API/tests/Setup/test_setup_deps_remote_admin.py 8 passed; tldw_Server_API/tests/Config/test_config_providers_endpoints.py 33 passed; apps/packages/ui OnboardingConnectForm design-system Vitest 4 passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->
<!-- SECTION:FINAL_SUMMARY:END -->

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
