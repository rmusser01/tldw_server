---
id: TASK-12090
title: Rebase PR 2568 and address review comments
status: Done
labels:
- review
- pr-2568
references:
- https://github.com/rmusser01/tldw_server/pull/2568
modified_files:
- apps/tldw-frontend/hooks/useConfig.tsx
- apps/tldw-frontend/hooks/__tests__/useConfig.networking.test.tsx
- backlog/tasks/task-12088 - Address-PR-2567-review-follow-ups-on-dev.md
- backlog/tasks/task-12089 - Address-current-main-CodeQL-alerts-in-PR-2568.md
- backlog/tasks/task-12090 - Rebase-PR-2568-and-address-review-comments.md
- tldw_Server_API/app/api/v1/endpoints/chatbooks.py
- tldw_Server_API/app/core/DB_Management/jobs_sql_fragments.py
- tldw_Server_API/app/core/Web_Scraping/WebSearch_APIs.py
- tldw_Server_API/tests/Jobs/test_jobs_event_filter_sql.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rebase PR #2568 on the latest dev branch and address current PR review comments/issues from CodeRabbit, Cubic, Gemini, checks, and inline review threads.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 PR #2568 is rebased on the latest origin/dev or confirmed already current.
- [x] #2 Current unresolved CodeRabbit, Cubic, and Gemini review comments are addressed in code, tests, or task metadata.
- [x] #3 Targeted frontend/backend tests, Python compile checks, diff check, and Bandit touched-scope verification are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
git rebase origin/dev reported the branch was already up to date. Review follow-ups addressed stale frontend credential persistence cleanup, JSON-shaped config assertions, Jobs SQL backend normalization and error messages, ambiguous WebSearch logging, the chatbooks CodeQL suppression function name, and backlog traceability comments.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
PR #2568 was already current with origin/dev. Addressed all current unresolved review comments found via GitHub review threads. Verification passed: git diff --check; `python -m pytest -q tldw_Server_API/tests/Jobs/test_jobs_event_filter_sql.py tldw_Server_API/tests/Web_Scraping/test_phase3_3_sanitizers.py` (32 passed); `bunx vitest run hooks/__tests__/useConfig.networking.test.tsx --reporter=dot` (10 passed); `python -m py_compile` for touched Python files; Bandit touched backend scope reported only known low-severity WebSearch baseline findings B311/B101/B311, and filtered Bandit with `-s B101,B311` returned 0 findings.
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
