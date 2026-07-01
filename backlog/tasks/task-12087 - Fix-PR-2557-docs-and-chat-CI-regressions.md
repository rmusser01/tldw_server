---
id: TASK-12087
title: Fix PR 2557 docs and chat CI regressions
status: Done
assignee: []
created_date: '2026-07-01 14:41'
updated_date: '2026-07-01 14:47'
labels:
  - ci
  - pr-2557
  - tests
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/2557'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address the current PR #2557 CI failures: README release metadata contract wording and Chat_NEW persona_id alias integration expectations after alias removal.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 README release metadata contract tests pass for version 0.1.33
- [x] #2 Chat_NEW persona_id alias integration tests match current alias-removal behavior
- [x] #3 Targeted failing CI tests pass locally
- [x] #4 Changes are committed and pushed to PR #2557
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Verified README release metadata contract with: /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Docs/test_release_docs_contract.py -q (11 passed).
- Verified Chat_NEW persona alias integration coverage with: /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Chat_NEW/integration/test_chat_persona_exemplars_integration.py -q (15 passed).
- Ran targeted repro tests before the full-file checks: README metadata test (1 passed) and four persona alias boundary tests (4 passed).
- Formatting/security checks: git diff --check passed; pre-commit on touched files passed; Bandit on the touched Python test file passed with B101 skipped for pytest asserts.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed the two still-valid PR #2557 CI regressions: README release metadata wording now satisfies the 0.1.33 docs contract, and Chat_NEW persona_id alias integration tests now pin pre-removal behavior to 2026-06-30 while preserving the explicit removal-date rejection test for 2026-07-01. Validated with the targeted repro tests, the full release docs contract file, the full chat persona exemplars integration file, git diff --check, pre-commit on touched files, and Bandit on the touched Python test file with B101 skipped for pytest asserts. Known skip/blocker note: unrelated pre-existing generated Docs/Published and watchlist-template worktree changes were intentionally left unstaged.
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
