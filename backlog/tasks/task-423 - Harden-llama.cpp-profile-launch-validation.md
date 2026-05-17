---
id: TASK-423
title: Harden llama.cpp profile launch validation
status: Done
labels:
- llamacpp
- local-llm
- backend
priority: medium
documentation:
- Docs/superpowers/plans/2026-05-17-llamacpp-managed-runtime-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Task 2 from the llama.cpp managed runtime closeout plan: fail closed for unsafe or internally conflicting launch definitions while preserving advisory warnings for hardware/resource risks.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Add or verify tests covering duplicate explicit host/port conflicts, wildcard host conflicts, and disabled-profile non-conflicts.
- [x] #2 Add or verify tests covering vision mmproj requirements and server_args/mmproj_model_id consistency.
- [x] #3 Add or verify tests covering reserved structured raw-arg rejection.
- [x] #4 Add or verify tests covering allowlist validation for path-like launch args.
- [x] #5 Centralize or reuse backend validation before profile persistence and launch.
- [x] #6 Run focused backend tests, diff checks, and Bandit on touched Python paths before PR.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Existing profile-store/profile-capability/process-runner coverage already covered duplicate explicit ports, wildcard conflicts, disabled duplicate ports, and launch-time mmproj/path validation.
- Added supervisor persistence-time regression coverage for vision profiles without mmproj, conflicting `mmproj_model_id`/`server_args["mmproj"]` selections, path-like server args outside allowlist, reserved model arg overrides even when unvalidated args are allowed, and invalid server_args updates preserving the stored profile.
- Shared server-arg validation now runs before supervisor create/update/default persistence and before launch.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Centralized llama.cpp profile launch validation before supervisor persistence and start by reusing launch asset resolution plus shared server_arg validation. The shared validation now rejects reserved structured args, unsupported args when unvalidated args are disabled, denylisted secret flags, invalid formatter values, and path-bearing args outside allowed paths. Focused backend tests pass; production Bandit scope passes. Full touched-scope Bandit including pytest files only reports the repository-standard B101 assert warnings in tests.
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
