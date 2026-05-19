---
id: TASK-78
title: Add shared sandbox artifact and log cap contracts
status: Done
assignee: []
created_date: '2026-05-05 17:12'
updated_date: '2026-05-05 17:46'
labels:
  - sandbox
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Phase 4 reliability slice: enforce shared artifact quota and log-cap reporting contracts across applicable sandbox runtimes without changing runtime availability semantics.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Seatbelt and worktree runners apply shared artifact byte caps and report aggregate counters.
- [x] #2 Log-cap truncation is visible in resource_usage and maps to limits_applied.
- [x] #3 Focused sandbox tests and touched-scope security checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing tests for host-local artifact caps and log-cap status signals. 2. Reuse shared limit helpers in applicable runners. 3. Add minimal stream-hub truncation reporting and taxonomy/audit coverage. 4. Run focused tests, compile, lint, Bandit, and diff checks.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Verification: focused sandbox suite passed with 60 tests: test_sandbox_limits.py, test_streams_hub_lifecycle.py, test_seatbelt_runner.py, test_worktree_runner.py, test_run_status_reason_codes.py, and test_sandbox_run_limit_audit.py. py_compile passed for touched Sandbox app files. Ruff passed for touched app/test files. Bandit JSON at /tmp/bandit_sandbox_artifact_log_cap_contracts.json reported 105 existing low-severity subprocess findings in runner files; diff-line check found 0 findings on added lines. git diff --check passed. Known blocker: test_docker_runner_fake.py -q still stalls after the first TestClient case and was terminated; this matches the existing app-lifespan/TestClient hang pattern rather than the runner-only artifact/log cap changes.

Opened PR #1319: https://github.com/rmusser01/tldw_server/pull/1319

PR review fix pass: added docstrings/type annotations requested by Qodo; applied exclusions before artifact quota accounting; preserved artifact counter schema on collection errors; moved stream truncation publish/metrics outside the hub lock; used one Lima log cap for replay and accounting; cleared artifact counters for canceled seatbelt/worktree runs. Left the VZ Linux helper-unavailable/protocol-mismatch fallback suggestion unchanged because existing tests intentionally preserve session control rows when host truth is unavailable or untrusted. Verification: targeted red tests failed before fixes and passed after fixes; affected sandbox suite passed 88 tests; py_compile passed; Ruff passed; Bandit JSON at /tmp/bandit_sandbox_artifact_log_cap_review_fixes.json reported existing low-severity findings only with 0 findings on added lines; git diff --check passed.

Additional VZ review follow-up: accepted the session-reuse fallback finding after explicit re-review. VZ Linux now treats helper status-probe unavailable/protocol failures and absent status replies as unhealthy candidate VMs, clears stale session control, and attempts fresh VM provisioning. Regression tests were updated from the older fail/preserve-control expectation and now cover unavailable, protocol mismatch, absent status, and unhealthy status fallback paths.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added shared sandbox artifact/log cap contract plumbing. Seatbelt and worktree now use shared artifact byte caps and report aggregate counters; Docker, Firecracker, Lima, and VZ Linux paths reuse the shared helpers where applicable. Stream log truncation is tracked as resource_usage and included in audit/status limit signals. Added focused regression coverage for artifact caps, log truncation, and limit taxonomy/audit metadata.

PR: https://github.com/rmusser01/tldw_server/pull/1319

Review fixes addressed Qodo/CodeRabbit findings for documentation, typing, artifact exclusion/error counters, stream lock side effects, Lima log cap consistency, and canceled-run artifact accounting.
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
