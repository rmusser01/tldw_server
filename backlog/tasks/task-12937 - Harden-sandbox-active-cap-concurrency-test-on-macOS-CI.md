---
id: TASK-12937
title: Harden sandbox active-cap concurrency test on macOS CI
status: Done
labels:
- ci
- sandbox
- tests
priority: High
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Run 28994210037 job 86048012729 failed in test_global_active_cap_enforced_across_service_instances because the fake runner start event was never observed on macOS Python 3.12 after the broader platform-sandbox-state-store shard order. Prepare a minimal deterministic test harness hardening patch and keep it unpushed until all monitored CI tests complete.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Patched tldw_Server_API/tests/sandbox/test_execution_concurrency_cap.py to remove scheduler-dependent test flake from active-cap assertions. Verification: targeted concurrency file passed (4 passed); broad local platform-sandbox-state-store shard showed this test passing but had local-only async plugin failures unrelated to the CI failure; git diff --check passed; Bandit on touched test file reported only pre-existing low-severity B101 assert warnings and no new helper findings. Host python3.12 lacks pytest, so Python 3.12 local verification was unavailable. Branch remains unpushed per user instruction until the monitored main CI run completes.
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
