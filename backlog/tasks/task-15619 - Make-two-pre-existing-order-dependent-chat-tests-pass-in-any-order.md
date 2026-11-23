---
id: TASK-15619
title: Make two pre-existing order-dependent chat tests pass in any order
status: To Do
labels:
- tests
- chat
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Two pre-existing failures surfaced in the TASK-15615 sweep (PR #3555); both fail on dev without that PR. (1) tldw_Server_API/tests/Chat/integration/test_messages_provider_override_store_boundary.py: test_messages_dispatch_uses_one_atomic_override_snapshot and test_messages_policy_and_credentials_share_one_override_snapshot[messages] and [count_tokens] each FAILED and ERRORed in the 23-file sweep run with -n 2 (the file set built in Task 0 Step 4 of Docs/superpowers/plans/2026-11-22-llm-policy-review-minors.md). The file alone passes, 10 passed with -p no:xdist and 10 passed with -n 2 (dev cee4433eb2, 2026-11-22), so another file in the sweep leaves state that breaks them; which file is not known. (2) tldw_Server_API/tests/Chat/integration/test_document_generation_endpoints.py::test_document_factory_starts_without_default_executor_and_drains_cancel fails when run alone (3 of 3 runs with -p no:xdist: 'AssertionError: Timed out waiting for thread event' at line 231, the 1.0 s wait for factory_entered) but passes inside its whole file (64 passed). The test saturates the loop's default executor and then expects the factory to start within 1 s, so the likely cause is a cold-start step on the way to the factory that runs on the default executor or takes over 1 s cold; earlier tests in the file warm it. If that step runs on the default executor, it is the production behaviour the test exists to catch: the first document generation after start-up would wait on a saturated default executor.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The three messages boundary tests pass in the Task 0 Step 4 sweep set run with -n 2 and still pass when their file runs alone; the notes name the file that left the breaking state and what the state was
- [ ] #2 The document-generation drain test passes when run alone and inside its file, and still proves the factory starts without the default executor and that cancel drains in factory_exit, capacity_release, mark_used, runtime_close order
- [ ] #3 If the cold-start step runs on the default executor, it is moved off it; a longer timeout alone does not close this task
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
