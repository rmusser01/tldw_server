---
id: TASK-2404
title: Verify and fix Character_Chat review findings
status: Done
assignee: []
created_date: '2026-06-23 18:10'
updated_date: '2026-06-23 18:52'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Verify the static review findings for tldw_Server_API/app/core/Character_Chat and address the validated issues with focused tests and security verification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each review finding is verified against current code and documented in task notes; validated Character_Chat issues are fixed with focused regression tests; touched scope tests pass; Bandit runs on touched Character_Chat paths
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Red-test verification: targeted pytest collected 8 tests; 7 failed for intended old behavior and 1 parser case passed because embedded-array fallback already handled that input. Failures covered memory prompt contract, world-book timeout compile/match path, V3 validation, core message cap, bulk regex validation, and workflow adapter signature drift. The run hung during cleanup and was interrupted after failure details were captured.

Implementation completed. Fixed memory extraction JSON-object contract, world-book regex runtime timeout usage, V3 card validation bounds, core post_message message-cap enforcement, bulk dictionary explicit-regex validation, workflow character_chat message adapter signature drift, and stale facade module line counts. Verification: red tests observed first; post-fix pytest 6 targeted Character_Chat tests passed; workflow adapter regression passed; direct ChatDictionaryService bulk regex check raised re.error; py_compile passed for touched prod/test files; git diff --check passed for touched paths. Bandit ran on touched production files and reported three pre-existing LOW findings in unchanged chat_dictionary.py lines 348, 3148, and 3173; no new Bandit finding is in the changed hunks. Limitation: tldw_Server_API/tests/Character_Chat/test_chat_dictionary_legacy.py::TestChatDictionaryService::test_bulk_add_entries_rejects_invalid_explicit_regex timed out during global app fixture setup before the test body, so direct service verification was used for that case.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Verified and addressed the validated Character_Chat review findings with focused regression coverage and security verification notes. Known verification limitation: the legacy chat dictionary pytest path times out in this local app setup before running the test body; direct service verification passed for the same behavior.
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
