---
id: TASK-12936
title: Address post-merge CodeQL alerts on main
status: Done
assignee: []
created_date: '2026-07-09 06:59'
updated_date: '2026-07-09 07:00'
labels:
  - security
  - codeql
  - main
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Investigate current CodeQL alerts on main after PR 2692 merge and implement minimal fixes for the alert classes before opening a main-targeted follow-up PR.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Current CodeQL alert classes on main are investigated and addressed in code where practical.
- [x] #2 Regression tests cover changed parser/path/storage behavior.
- [x] #3 Verification and Bandit status are recorded before PR handoff.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-07-09: Implemented CodeQL hardening for current main alert classes. Replaced vulnerable parser regexes in flashcard structured import, Skills frontmatter parsing, and moderation policy regex scanning with deterministic scanners. Replaced workflow list dynamic ORDER BY construction with a literal validated ordering map. Switched CSRF user binding to hmac.digest and suppressed public header/cookie-name Bandit false positives. Removed exception-detail leaks from setup, prompts, OCR, embeddings, and ACP endpoint responses. Removed frontend/runtime persistence of single-user API keys in local/session storage by using runtime-only key injection for smoke tests and extension bootstrap. Hardened path construction/validation across email mock output, chatbook export work dirs, file-artifact temp exports, RAG checkpoint/regression/personalization stores, research artifact storage, sandbox workspace/snapshot handling, filesystem storage, unified audit fallback queue, audio provider input conversion, visual identity asset storage, and skills service file operations. Added/updated regression tests for parser behavior, policy literal scanning, sandbox snapshot traversal rejection, and research artifact logical-name vs hashed-storage behavior.

Verification: py_compile passed for touched Python; git diff --check passed; focused parser/security tests passed: 54 passed; visual identity storage tests passed: 31 passed; broader focused backend batch passed: 130 passed; frontend vitest runtime/auth storage tests passed: 37 passed; frontend typecheck passed. Bandit app-scope report /tmp/bandit_codeql_main_alerts_app.json has no high/medium findings; remaining 6 LOW findings are pre-existing subprocess warnings in Audio_Transcription_Lib.py outside the changed path-validation logic.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed the post-merge main CodeQL alert classes with targeted parser, SQL ordering, path-safety, stack-trace redaction, CSRF binding, and frontend runtime-key-storage hardening. Added regression coverage for parser and storage behaviors. Local verification passed for focused backend/frontend checks; app-scope Bandit has only pre-existing low-severity audio subprocess warnings outside the changed logic. No user-facing documentation changes were needed.
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
