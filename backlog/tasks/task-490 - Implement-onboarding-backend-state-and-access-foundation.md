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
- tldw_Server_API/app/api/v1/API_Deps/setup_deps.py
- tldw_Server_API/app/api/v1/endpoints/setup.py
- tldw_Server_API/app/api/v1/schemas/setup_schemas.py
- tldw_Server_API/tests/Setup/test_first_run_state.py
- tldw_Server_API/tests/Setup/test_setup_deps_remote_admin.py
- tldw_Server_API/tests/integration/test_unified_first_run_setup_api.py
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

Task 2 slice implemented first-run state/metadata/skip endpoints plus setup write access boundary checks. Red phase captured expected failures before production changes: remote disabled write detail lacked "localhost", `/api/v1/setup/first-run/state` returned 404, and `/api/v1/setup/first-run/skip` was missing; the initial red command timed out during TestClient lifespan shutdown after those failures, so the new integration tests were adjusted to use the existing setup-test pattern of `TestClient(app)` without a context manager.

Task 2 verification: `python -m pytest tldw_Server_API/tests/Setup/test_setup_deps_remote_admin.py tldw_Server_API/tests/integration/test_unified_first_run_setup_api.py -v` passed 15 tests; `python -m pytest tldw_Server_API/tests/Setup/test_first_run_state.py tldw_Server_API/tests/Setup/test_setup_manager_masking.py -q` passed 20 tests; Ruff touched-file check passed; Bandit touched production-file scan reported 0 findings; `git diff --check` passed.

Task 2 review-fix slice: hardened `_require_first_run_write_access` so completed setup returns `409 setup_already_completed` even when first-time setup is disabled, and so terminal first-run state files reject writes from the shared guard with `state_skipped` or `state_blocked` before endpoint mutation. First-run metadata now classifies browser access from `X-Forwarded-For` when present and no longer treats local API host alone as proof of local browser access. Red phase captured 4 expected failures before production changes: forwarded remote metadata classified local, disabled completed setup returned 404, and skipped/blocked terminal state tests reached the endpoint mutation path. Verification after fix: first-run setup API slice passed 19 tests; first-run state/masking slice passed 20 tests; Ruff touched-file check passed; Bandit touched production-file scan reported 0 findings; `git diff --check` passed.

Task 2 spec re-review fix: reordered `_require_first_run_write_access` so explicit completion still returns `409 setup_already_completed`, disabled incomplete setup returns `404 setup_disabled`, and only enabled legacy/inconsistent `needs_setup=False` statuses return the completed conflict. Red phase captured the new disabled-incomplete regression failing with `409` instead of `404`; verification after fix passed the first-run setup API slice, first-run state/masking slice, Ruff, Bandit, and `git diff --check`.

Task 2 coverage-only spec re-review fix: added direct integration coverage for enabled but inconsistent `needs_setup=False` setup status returning `409 setup_already_completed`. The branch already existed, so no red phase was expected; the focused test passed immediately, and requested verification passed.

Task 2 final review fix: first-run metadata now only honors `X-Forwarded-For` when proxy trust is enabled and the immediate setup client is loopback, and it validates the forwarded value as an IP before classification. Red phase captured spoofed `X-Forwarded-For: 127.0.0.1` from a remote immediate client reporting local bundled auth; verification after fix passed the required pytest suites, Ruff, Bandit, and `git diff --check`.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Task 1 delivered the durable first-run state store and setup schema re-exports, including secret redaction for private_key, access_key, and auth_key normalized variants. Task 2 delivered setup access-boundary coverage, first-run state/metadata/update/skip endpoints, setup-completed write rejection, terminal first-run state write rejection, and first-run metadata for auth/setup path/connection guidance. TASK-490 remains In Progress because later review gates still run here.
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
