---
id: TASK-2404
title: Verify and fix Character_Chat review findings
status: Done
assignee: []
created_date: 2026-06-23 18:10
updated_date: 2026-06-24 01:47
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
Rebased PR #2449 follow-up onto latest origin/dev and addressed all validated review comments with focused regression coverage. Remaining Bandit output is limited to previously documented low-severity unchanged-line findings in chat_dictionary.py.
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

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Reopened for PR #2449 follow-up: rebase onto latest origin/dev and address Gemini/Qodo review comments around message-limit semantics, memory parsing fallback, adapter DB offloading/error context, regex timeout exception handling, max_tokens coercion, and message-limit race mitigation.
PR #2449 follow-up completed after rebasing onto latest origin/dev. Addressed validated Gemini/Qodo comments: user-message-only core limit enforcement, in-process per-conversation serialization for user message count+insert, explicit memory parser fallback for embedded JSON objects, adapter preflight limit check before LLM calls, sync DB offloading via asyncio.to_thread, contextual write errors, defensive regex timeout exception handling, and max_tokens coercion. Verification: py_compile passed for touched files; focused pytest passed 14 tests; direct ChatDictionaryService invalid explicit regex check passed; git diff --check passed; Bandit on touched production paths reported only the known LOW unchanged-line chat_dictionary.py findings at lines 348, 3148, and 3173.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
