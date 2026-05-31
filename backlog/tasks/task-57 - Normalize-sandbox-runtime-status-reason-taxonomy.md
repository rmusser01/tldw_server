---
id: TASK-57
title: Normalize sandbox runtime status reason taxonomy
status: Done
assignee: []
created_date: '2026-05-05 02:45'
updated_date: '2026-05-05 03:09'
labels:
  - sandbox
  - macos
  - runtime-taxonomy
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add a narrow Phase 3 sandbox slice that centralizes known runtime status and error message aliases into the existing run status taxonomy so clients can rely on stable status_reason_code values without runtime-specific string guessing. Preserve raw runner messages for diagnostics and avoid broad runner behavior changes in this PR.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Known policy failure messages across sandbox runtimes normalize to policy_failed while preserving raw messages.
- [x] #2 Known timeout and runtime-unavailable messages normalize through central taxonomy helpers without broadening genuinely unknown failures beyond runtime_error.
- [x] #3 Focused tests cover added taxonomy aliases and retain runtime_error fallback coverage for unknown failures.
- [x] #4 Sandbox runtime capability documentation is updated to reflect the Phase 3 taxonomy pass and remaining limitations.
- [x] #5 Regression test covers policy_failed precedence when a failed run message also contains runtime_unavailable.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
PR #1284 review follow-up: verify current PR head and taxonomy ordering, add only a focused regression test for policy_failed precedence over runtime_unavailable if the guard is missing, avoid production-code changes unless the test exposes a real ordering bug, then run focused pytest and git diff checks. Bandit is not needed for a tests-only edit; record that skip reason.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Focused RED check failed before implementation because vz_linux_policy_failed and vz_macos_policy_failed normalized to runtime_unavailable.

Verification: /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/sandbox/test_run_status_reason_codes.py tldw_Server_API/tests/sandbox/test_runtime_inventory_contract.py -q passed with 24 tests.

Verification: /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r tldw_Server_API/app/core/Sandbox/run_status_taxonomy.py -f json -o /tmp/bandit_sandbox_runtime_taxonomy.json reported zero findings.

Verification: git diff --check passed.

Review finding verification: current PR head 2067246782934a4980d00892e726985b8d2f3582 matches the local sandbox-runtime-taxonomy worktree. normalize_run_status_reason currently checks _is_policy_failed_message before _is_runtime_unavailable_message, so behavior is already correct; the still-valid issue is the missing explicit regression test for mixed policy_failed/runtime_unavailable messages.

Verification: /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/sandbox/test_run_status_reason_codes.py -q passed with 8 tests.

Verification: /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/sandbox/test_run_status_reason_codes.py tldw_Server_API/tests/sandbox/test_runtime_inventory_contract.py -q passed with 25 tests.

Verification: git diff --check passed after the review follow-up edit.

Bandit: skipped for this follow-up because it only adds pytest assertions in a sandbox test file and makes no production-code changes; the PR's production taxonomy file already had zero Bandit findings in the prior verification.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Centralized known sandbox policy failure aliases in the run status taxonomy so VZ, Lima, seatbelt, and worktree policy admission failures normalize to policy_failed while raw messages remain unchanged. Added focused tests for policy/runtime-unavailable aliases, kept unknown failures falling through to runtime_error, and updated the sandbox runtime inventory to record the first Phase 3 taxonomy pass plus remaining structured error metadata work.

Review follow-up on PR #1284 added an explicit regression test asserting policy_failed wins when a failed run message also contains runtime_unavailable, including mixed-case wording. Current production ordering was verified as already correct, so no runtime taxonomy code changed.
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
