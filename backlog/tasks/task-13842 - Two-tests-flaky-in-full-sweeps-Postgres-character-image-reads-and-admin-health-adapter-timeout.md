---
id: TASK-13842
title: 'Two tests flaky in full sweeps: Postgres character image reads and admin health
  adapter timeout'
status: To Do
labels:
- tests
- flaky
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Seen while sweeping the TASK-13560 change (PR #3247), each failing in one run and passing in another with no relevant change: tests/Character_Chat_NEW/integration/test_character_chat_images.py::test_character_image_reads_leave_event_loop_and_release_operation[postgres-*] (also flaky on unchanged dev per that PR's plan review), and tests/Admin/test_admin_llm_provider_test_runtime.py::test_admin_health_blocking_adapter_timeout_bounds_capacity_and_recovers (failed on the dev base sweep, passed with the change). Both are timing-sensitive.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each test is reproduced as flaky (or shown stable) with a repeated run, and the cause is recorded
- [ ] #2 A flaky test is made deterministic without weakening what it checks
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
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
