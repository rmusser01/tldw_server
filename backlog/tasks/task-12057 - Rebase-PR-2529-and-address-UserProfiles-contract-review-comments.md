---
id: TASK-12057
title: Rebase PR 2529 and address UserProfiles contract review comments
status: In Progress
labels:
  - pr-review
  - userprofiles
priority: High
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rebase PR #2529 on latest dev, inspect all active PR review comments, implement verified fixes, run focused verification, push the updated branch, and resolve addressed review threads.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 PR #2529 branch is rebased onto latest origin/dev without unresolved conflicts.
- [x] #2 All active review comments are inventoried, technically verified, and either fixed or answered with rationale.
- [x] #3 Focused tests, compile checks, shard coverage or relevant CI guard checks, and Bandit on touched production code are run as applicable.
- [ ] #4 Updated branch is pushed and review threads are replied to/resolved.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Rebased `codex/userprofiles-contract-refactor` onto `origin/dev` cleanly.
- Active review threads covered stale transaction version reads, v2 structured errors, endpoint rate limiting, dependency injection, and audit logging.
- Added regression coverage for transaction-scoped version reads and v2 endpoint review feedback before production changes.
- Implemented `db_conn` propagation for `ProfileCommandService.get_profile_version()` reads, v2 `ProfileCommandService` dependency injection, `check_rate_limit` dependency, structured v2 error details, and sanitized audit suppression logging.
- Verification so far:
  - `python -m pytest tldw_Server_API/tests/UserProfile/test_profile_command_service.py::test_command_service_successful_write_returns_executor_result_and_version tldw_Server_API/tests/UserProfile/test_user_profile_v2_contract.py::test_v2_profile_update_route_has_rate_limit_dependency tldw_Server_API/tests/UserProfile/test_user_profile_v2_contract.py::test_v2_profile_update_uses_injected_command_service_and_structured_errors tldw_Server_API/tests/UserProfile/test_user_profile_v2_contract.py::test_v2_profile_update_logs_audit_failure_without_failing -q` -> 4 passed.
  - `python -m pytest tldw_Server_API/tests/UserProfile/test_profile_command_service.py tldw_Server_API/tests/UserProfile/test_user_profile_v2_contract.py tldw_Server_API/tests/UserProfile/test_user_profile_updates.py tldw_Server_API/tests/UserProfile/test_admin_profiles_service_update.py -q` -> 37 passed.
  - `python -m pytest tldw_Server_API/tests/UserProfile tldw_Server_API/tests/AuthNZ/unit/test_user_profile_command_backend_selection.py -q` -> 119 passed.
  - `python -m compileall tldw_Server_API/app/api/v2/endpoints/user_profiles.py tldw_Server_API/app/core/UserProfiles/command_service.py` -> passed.
  - `python -m bandit -r tldw_Server_API/app/api/v2/endpoints/user_profiles.py tldw_Server_API/app/core/UserProfiles/command_service.py -f json -o /tmp/bandit_pr2529.json` -> 0 findings.
  - `python Helper_Scripts/ci/check_shard_coverage.py --ci-file .github/workflows/ci.yml` -> passed.
  - `git diff --check` -> passed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Backlog task and implementation plan are updated with verification evidence.
- [ ] #2 Code changes are committed with a clear message.
- [x] #3 No unrelated dirty files are staged or modified.
- [ ] #4 PR branch is pushed to GitHub and remaining check status is reported.
<!-- DOD:END -->
