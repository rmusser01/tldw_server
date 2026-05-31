---
id: TASK-490
title: Implement onboarding backend state and access foundation
status: In Progress
assignee: []
created_date: ''
updated_date: 2026-05-31 06:15
labels: []
dependencies: []
references:
- TASK-489
- Docs/superpowers/plans/2026-05-31-first-time-solo-user-onboarding-implementation-plan.md
documentation:
- Docs/superpowers/plans/2026-05-31-first-time-solo-user-onboarding-implementation-plan.md#task-1-backend-first-run-state-store
- Docs/superpowers/plans/2026-05-31-first-time-solo-user-onboarding-implementation-plan.md#task-2-setup-access-boundary-and-first-run-state-endpoints
modified_files:
- tldw_Server_API/app/core/Setup/first_run_state.py
- tldw_Server_API/app/core/Setup/first_run_models.py
- tldw_Server_API/app/api/v1/schemas/setup_schemas.py
- tldw_Server_API/tests/Setup/test_first_run_state.py
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
Task 1 final code-quality fix slice: first-run step data is now recursively redacted before persistence for secret-looking keys, required non-secret setup fields remain intact, stale lock files with old metadata or dead POSIX owners are recovered before mutation, and the unlocked save path is private. TASK-490 remains In Progress because Task 2 endpoint/access-boundary work is intentionally not implemented. Red phase captured 2 expected failures in test_first_run_state.py before production changes: raw secrets persisted and stale_lock_seconds was unsupported. Verification after fix: test_first_run_state.py - 17 passed; test_setup_manager_masking.py - 1 passed; Ruff touched-file check passed; Bandit touched production-file scan reported 0 findings; git diff --check passed.
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
