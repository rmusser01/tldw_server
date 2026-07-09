---
id: TASK-12935
title: Fix Windows Notes auto-title CI ChaCha warm-up race
status: Done
assignee: []
created_date: ''
updated_date: '2026-07-09 16:50'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Main CI run 28994210037 job 86048011704 failed on windows-latest/Python 3.12 product-notes-persona. Failure: tldw_Server_API/tests/Notes/test_auto_title_integration.py::test_create_note_with_auto_title returned 500 {"detail":"Could not initialize character & notes database for user"}. Log shows single-user startup ChaChaNotes warm-up timed out during schema initialization, then first Notes request raced a second initialization against the same DB. Prepare local unpushed fix on codex/fix-main-guardian-notify-ts; do not push until all main CI tests complete per user instruction.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Startup ChaChaNotes warm-up does not schedule in TEST_MODE
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause: on Windows CI, single-user startup scheduled a best-effort ChaChaNotes warm-up while TEST_MODE was active. Schema initialization exceeded the 30s warm-up timeout, leaving the worker thread initializing the DB while the first Notes request attempted another initialization. Fix: skip speculative ChaCha warm-up in TEST_MODE while preserving normal single-user warm-up behavior outside tests.

Documentation update not needed; this is a test-startup behavior fix. Known hold: branch remains intentionally unpushed until the monitored main CI run completes, per user instruction.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Prepared unpushed local commit for main CI product-notes-persona failure. Added a regression test for TEST_MODE warm-up skipping and changed startup_chacha_warmup to skip ChaChaNotes warm-up when core testing.is_test_mode() is true. Verification: startup warm-up unit test file passed; Notes auto-title integration file passed under TEST_MODE; git diff --check passed; Bandit on startup_chacha_warmup.py reported 0 findings.
<!-- SECTION:FINAL_SUMMARY:END -->

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
