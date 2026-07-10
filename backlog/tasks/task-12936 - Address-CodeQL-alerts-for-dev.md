---
id: TASK-12936
title: Address CodeQL alerts for dev
status: Done
assignee: []
created_date: '2026-07-09 06:59'
updated_date: '2026-07-09 07:40'
labels:
  - security
  - codeql
  - dev
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Investigate current CodeQL alerts and implement minimal fixes for the alert classes before opening a dev-targeted follow-up PR.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Current CodeQL alert classes are investigated and addressed in code where practical.
- [x] #2 Regression tests cover changed parser/path/storage behavior.
- [x] #3 Verification and Bandit status are recorded before PR handoff.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-07-09: Implemented CodeQL hardening for current alert classes. Replaced vulnerable parser regexes in flashcard structured import, Skills frontmatter parsing, and moderation policy regex scanning with deterministic scanners. Replaced workflow list dynamic ORDER BY construction with a literal validated ordering map. Switched CSRF user binding to hmac.digest and suppressed public header/cookie-name Bandit false positives. Removed exception-detail leaks from setup, prompts, OCR, embeddings, and ACP endpoint responses. Removed frontend/runtime persistence of single-user API keys in local/session storage by using runtime-only key injection for smoke tests and extension bootstrap. Hardened path construction/validation across email mock output, chatbook export work dirs, file-artifact temp exports, RAG checkpoint/regression/personalization stores, research artifact storage, sandbox workspace/snapshot handling, filesystem storage, unified audit fallback queue, audio provider input conversion, visual identity asset storage, and skills service file operations. Added/updated regression tests for parser behavior, policy literal scanning, sandbox snapshot traversal rejection, and research artifact logical-name vs hashed-storage behavior.

Verification: py_compile passed for touched Python; git diff --check passed; focused parser/security tests passed: 54 passed; visual identity storage tests passed: 31 passed; broader focused backend batch passed: 130 passed; frontend vitest runtime/auth storage tests passed: 37 passed; frontend typecheck passed. Bandit app-scope report /tmp/bandit_codeql_main_alerts_app.json has no high/medium findings; remaining 6 LOW findings are pre-existing subprocess warnings in Audio_Transcription_Lib.py outside the changed path-validation logic.

2026-07-09: Reopened to address PR #2696 review comments from Gemini/Qodo: moderation scanner escape handling, async filesystem offloads, sandbox snapshot/workspace legacy fallback, and ACP runner-probe logging.

2026-07-09: Addressed PR #2696 review comments. Removed the redundant moderation regex backslash skip and added coverage for literal-backslash/live-quantifier handling. Offloaded reviewed async-path filesystem setup to `asyncio.to_thread`, logged ACP runner probe failures server-side, preserved legacy raw sandbox snapshot/workspace lookup while keeping new writes on hashed paths, and added regression coverage for those compatibility paths.

Review verification: focused tests passed (`87 passed, 2 warnings`); py_compile passed for touched Python files; git diff --check passed; Bandit touched-app report `/tmp/bandit_pr2696_review_comments.json` has zero findings.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed CodeQL alert classes with targeted parser, SQL ordering, path-safety, stack-trace redaction, CSRF binding, and frontend runtime-key-storage hardening. Followed up on PR #2696 review comments with moderation scanner correction, async filesystem offloads, ACP probe logging, and legacy sandbox snapshot/workspace compatibility. Local verification passed for the focused backend/frontend checks and the review-comment focused backend batch; Bandit reports for touched app scopes have no new findings. No user-facing documentation changes were needed.
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
