---
id: TASK-13369
title: TEST_MODE env disables webhook SSRF validation on delivery
status: Done
assignee: []
created_date: '2026-09-23 23:13'
updated_date: '2026-09-23 23:13'
labels:
  - bug
  - security
  - evaluations
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Evaluations webhook_manager._deliver_webhook set skip_dns = is_test_mode() or PYTEST_CURRENT_TEST, so a deployment carrying TEST_MODE delivered webhooks to user URLs without the DNS/SSRF target check. Found during TASK-13296.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Delivery validates the target unless running under pytest
- [x] #2 Regression test with TEST_MODE set and no pytest runtime
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed in e156820018: skip_dns = is_explicit_pytest_runtime(). test_delivery_keeps_ssrf_check_under_test_mode_env red on HEAD, green now; Evaluations suite 18 failed before and after (pre-existing).
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Webhook delivery always validates the target outside an active pytest run.
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
